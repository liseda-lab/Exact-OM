import time
from datetime import timedelta
from collections import deque
import warnings
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from typing import List, Optional, Tuple

from matcha_dl.core.contracts.trainer import EntityMapping
from matcha_dl.impl.trainer.mlp import MLPTrainer
from matcha_dl.core.entities.dataset import DatasetMask
from matcha_dl.utils.collate import DataCollator
from transformers import get_linear_schedule_with_warmup

class EncoderTrainer(MLPTrainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dataset.tokenizer = self.model.tokenizer
        self.collator = DataCollator(self.model.tokenizer)
        self._train_time_window = deque(maxlen=20)
        self._infer_time_window = deque(maxlen=20)
        self._best_val_loss = float('inf')

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = True,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = False,
        val_every: int = 1,
        save_interval: Optional[int] = None,
        log_every: int = 1,
        warmup_percent: float = 0.1,
        **kwargs
    ):
        if self.skip_training:
            self.log("Checkpoint exists: skipping training …", level="info")
            return

        # Enable cuDNN autotune for fixed-size inputs
        torch.backends.cudnn.benchmark = True
        warnings.filterwarnings("ignore", category=UserWarning)

        
        writer = SummaryWriter(self.logs_dir)
        scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
        self.dataset.default_kind = DatasetMask.train

        # Prepare DataLoader once with optimal settings
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else None,
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=self.collator
        )
        total_batches = len(dataloader)

        num_train_steps   = (epochs * total_batches) // gradient_accumulation_steps
        num_warmup_steps  = int(warmup_percent * num_train_steps)  # 10% warmup
        scheduler = get_linear_schedule_with_warmup(
            self._optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )

        early_stopping = False
        while self._epoch <= epochs and not early_stopping:
            self.model.train()
            self._optimizer.zero_grad()

            for step, (batch_prompts, target) in enumerate(dataloader, start=1):
                # GPU-accurate timing start
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                t0 = time.perf_counter()

                # forward/backward with non-blocking transfers
                batch_prompts = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch_prompts.items()
                }
                target = target.to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=mixed_precision):
                    logits = self.model(
                        batch_prompts["input_ids"],
                        batch_prompts["attention_mask"]
                    )
                    loss = self._loss(logits, target) / gradient_accumulation_steps

                if mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                global_step = step + self._epoch * total_batches
                writer.add_scalar("Loss/train", loss.item(), global_step)

                if (step % gradient_accumulation_steps == 0) or (step == total_batches):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    writer.add_scalar("GradNorm", grad_norm, global_step)

                    if mixed_precision:
                        scaler.step(self._optimizer)
                        scaler.update()
                    else:
                        self._optimizer.step()

                    scheduler.step()
                    self._optimizer.zero_grad()

                    current_lr = self._optimizer.param_groups[0]['lr']
                    writer.add_scalar("LR", current_lr, global_step)
                    if mixed_precision:
                        writer.add_scalar("AMP_Scale", scaler.get_scale(), global_step)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                batch_time = time.perf_counter() - t0
                self._train_time_window.append(batch_time)
                avg_time = sum(self._train_time_window) / len(self._train_time_window)
                throughput = batch_size / batch_time
                writer.add_scalar("Throughput", throughput, global_step)
                remaining = total_batches - step
                eta = timedelta(seconds=int(avg_time * remaining))

                if step % log_every == 0 or step == total_batches:
                    self.log(
                        f"Epoch {self._epoch}/{epochs} "
                        f"batch {step}/{total_batches} "
                        f"loss={loss.item():.4f} "
                        f"batch_time={batch_time:.2f}s ETA={eta}",
                        level="debug"
                    )

            # validation
            if val_every and self.dataset.validation_set is not None and (self._epoch % val_every) == 0:
                val_map, val_loss = self.predict(
                    kind=DatasetMask.validation,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    log_every=log_every
                )
                writer.add_scalar("Loss/validation", val_loss, self._epoch)
                self.log(f"Validation loss at epoch {self._epoch}: {val_loss:.4f}", level="debug")

                #checkpoint best
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self.save_checkpoint()
                    self.log("New best model saved.", level="info")

                if self.stopping and self.stopping(validation_loss=val_loss):
                    self.log(f"Early stopping at epoch {self._epoch}", level="info")
                    early_stopping = True

            # checkpointing
            if save_interval and (self._epoch % save_interval) == 0:
                self.save_checkpoint()

            self._epoch += 1

        writer.flush()
        writer.close()
        self.save_checkpoint()
        self.log(f"Training finished at epoch {self._epoch}", level="info")

    def predict(
        self,
        kind: DatasetMask = DatasetMask.inference,
        threshold: Optional[float] = None,
        cardinality: Optional[int] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        log_every: int = 1,
        mixed_precision: bool = False,
        plot_params: Optional[dict] = None,
        **kwargs
    ) -> Tuple[List[EntityMapping], float]:
        self.dataset.default_kind = kind

        # Prepare DataLoader for inference
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else None,
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=self.collator
        )
        total_batches = len(dataloader)

        self.model.eval()
        total_loss = 0.0
        num_examples = len(self.dataset)
        logits_all = None
        start_idx = 0

        self.log(f"Starting inference for {kind.name} set with {num_examples} examples", level="info")
        self.log(f"Total batches: {total_batches} of size"
                 f" {batch_size} with {num_workers} workers", level="debug")

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=mixed_precision):
            for step, (batch_prompts, target) in enumerate(dataloader, start=1):
                # timing start
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                t0 = time.perf_counter()

                # forward with non-blocking transfers
                batch_prompts = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch_prompts.items()
                }
                target = target.to(self.device, non_blocking=True)
                logits = self.model(
                    batch_prompts["input_ids"],
                    batch_prompts["attention_mask"]
                )
                loss = self._loss(logits, target)
                total_loss += loss.item() * batch_prompts["input_ids"].size(0)

                # accumulate logits
                if logits_all is None:
                    # allocate once on CPU
                    logits_shape = (num_examples,) + tuple(logits.shape[1:])
                    logits_all = torch.empty(logits_shape, dtype=logits.dtype)
                end_idx = start_idx + batch_prompts["input_ids"].size(0)
                logits_all[start_idx:end_idx] = logits.cpu()
                start_idx = end_idx

                # timing end + update window
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                batch_time = time.perf_counter() - t0
                self._infer_time_window.append(batch_time)
                avg_time = sum(self._infer_time_window) / len(self._infer_time_window)
                remaining = total_batches - step
                eta_str = str(timedelta(seconds=int(avg_time * remaining)))

                # log inference batch info at interval
                if step % log_every == 0 or step == total_batches:
                    self.log(
                        f"{kind.name} | batch {step}/{total_batches} | "
                        f"loss={loss.item():.4f} | "
                        f"batch_time={batch_time:.2f}s | ETA={eta_str}",
                        level="debug"
                    )

        # post-process
        df = self.dataset.dataframe.copy()
        df = df[df[kind] == True]
        df["Scores"] = logits_all.numpy()
        avg_loss = total_loss / num_examples

        if self.plot_params is not None and self.plot_params.get("enabled", False) and self.plot_params.get("plot_predictions", False):
            self.plot_score_distribution(
                df,
                kind=kind,
                **self.plot_params
            )
        
        return EntityMapping.read_table_mappings(
            df[["Src", "Tgt", "Scores"]],
            threshold=threshold,
            cardinality=cardinality
        ), avg_loss
