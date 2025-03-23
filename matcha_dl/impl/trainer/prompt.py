import torch
from typing import List, Tuple, Union, Optional
import re
import warnings
from statistics import mean
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from matcha_dl.core.contracts.trainer import EntityMapping, ITrainer
from matcha_dl.core.entities.dataset import DatasetMask

# 

class PromptTrainer(ITrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Set dataset tokenizer
        self.dataset.tokenizer(self.model.tokenizer)

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = True,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = False,
        val_every: Optional[int] = 1,
        save_interval: Optional[int] = 5,
        **kwargs
    ):
        if self.skip_training:
            self.log("Skipping training, because checkpoint already exists", level="info")
            return
        
        if self.dataset.example is not None and any(self.dataset.example):
            self.log("Skipping training loop because instruction training is active", level="info")

        warnings.filterwarnings("ignore", category=UserWarning)
        writer = SummaryWriter(self.logs_dir)
        early_stopping = False
        self.dataset.default_kind = DatasetMask.train
        scaler = torch.amp.GradScaler('cuda') if mixed_precision else None

        self.model.unexpected_response_count.reset()

        while self._epoch <= epochs and not early_stopping:
            self.model.train()
            _iter = 1
            self._optimizer.zero_grad()

            with tqdm(DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), unit="batch") as tepoch:
                for step, (batch_prompts, target) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {self._epoch}")

                    batch_prompts, target = (
                        batch_prompts.to(self.device),
                        target.unsqueeze(1).to(self.device),
                    )

                    with torch.amp.autocast('cuda', enabled=mixed_precision):
                        logits = self.model(batch_prompts["input_ids"], batch_prompts["attention_mask"])
                        loss = self._loss(logits, target) / gradient_accumulation_steps

                    writer.add_scalar("Loss/train", loss.item(), _iter)
                    loss.backward()

                    if (step + 1) % gradient_accumulation_steps == 0:
                        if mixed_precision:
                            scaler.step(self._optimizer)
                            scaler.update()
                        else:
                            self._optimizer.step()
                        self._optimizer.zero_grad()

                    tepoch.set_postfix(loss=loss.item())
                    _iter += 1

                if val_every and val_every > 0 and self.dataset.validation_set is not None:
                    if self._epoch % val_every == 0:
                        _, validation_loss = self.predict(kind=DatasetMask.validation, batch_size=batch_size, num_workers=num_workers)
                        writer.add_scalar("Loss/validation", validation_loss, self._epoch)
                        tepoch.set_postfix(validation_loss=validation_loss)
                        self.log(f"Validation loss at epoch {self._epoch} - {validation_loss}", level="info")
                        if self.stopping and self.stopping(validation_loss=validation_loss):
                            self.log(f"Early stopping at epoch {self._epoch}", level="info")
                            early_stopping = True
                            break

            if save_interval and save_interval > 0 and self._epoch % save_interval == 0:
                self.save_checkpoint()

            self._epoch += 1

        writer.flush()
        writer.close()
        self.save_checkpoint()

        self.log(f"Training finished at epoch {self._epoch}", level="info")
        self.log(f"Final validation loss - {validation_loss}", level="info")
        self.log(f"Final training loss - {loss.item()}", level="info")
        self.log(f"Number of unexpected responses from model during training: \n {self.model.unexpected_response_count}", level="debug")

    def predict(self, 
            kind: DatasetMask = DatasetMask.inference, 
            threshold: Optional[float] = None, 
            cardinality: Optional[int] = None, 
            batch_size: int = 32,
            num_workers: int = 0,
            **kwargs) -> Tuple[List[EntityMapping], float]:

        self.dataset.default_kind = kind
        
        # Create DataLoader with pin_memory optimization.
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        self.model.eval()
        self.model.unexpected_response_count.reset()
        total_loss = 0.0
        
        num_examples = len(self.dataset)
        logits_all = None
        start_idx = 0

        with torch.no_grad():
            for batch_prompts, target in dataloader:
                # Ensure each tensor in the batch is moved to the correct device.
                batch_prompts = {k: v.to(self.device) for k, v in batch_prompts.items()}
                # The batch size is provided by the tensor shape.
                current_batch_size = batch_prompts["input_ids"].size(0)
                target = target.unsqueeze(1).to(self.device)
                
                logits = self.model(batch_prompts["input_ids"], batch_prompts["attention_mask"])
                loss = self._loss(logits, target)
                total_loss += loss.item() * current_batch_size
                
                # Pre-allocate logits_all on the first iteration.
                if logits_all is None:
                    # Determine full shape: (num_examples, ...) where ... matches logits' extra dimensions.
                    logits_shape = (num_examples,) + logits.shape[1:]
                    logits_all = torch.empty(logits_shape, dtype=logits.dtype)
                
                end_idx = start_idx + current_batch_size
                logits_all[start_idx:end_idx] = logits.cpu()
                start_idx = end_idx

        self.log(f"Number of unexpected responses from model during inference: \n {self.model.unexpected_response_count}", level="debug")

        # Filter the dataframe based on the provided mask.
        df = self.dataset.dataframe.copy()
        df = df[df[kind] == True]
        
        # Assign scores from the pre-allocated tensor.
        df["Scores"] = logits_all.numpy()
        avg_loss = total_loss / num_examples
        return EntityMapping.read_table_mappings(df[["Src", "Tgt", "Scores"]], threshold=threshold, cardinality=cardinality), avg_loss