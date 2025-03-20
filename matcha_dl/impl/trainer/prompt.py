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
        
        if self.dataset.example:
            self.log("Skipping training loop because instruction training is active", level="info")

        warnings.filterwarnings("ignore", category=UserWarning)
        writer = SummaryWriter(self.logs_dir)
        early_stopping = False
        self.dataset.default_kind = "train"
        scaler = torch.amp.GradScaler('cuda') if mixed_precision else None

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
                        _, validation_loss = self.predict(kind='validation')
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

    def predict(self, 
                kind: DatasetMask = DatasetMask.inference, 
                threshold: Optional[float] = None, 
                cardinality: Optional[int] = None, 
                batch_size: int = 32, 
                **kwargs
        ) -> Tuple[List[EntityMapping], float]:

        df = self.dataset.dataframe.copy()
        df = df[df[kind] == True]
      
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        all_logits = []
        total_loss = 0.0

        with torch.no_grad():
            for batch_prompts, target in dataloader:
                batch_prompts, target = (
                    batch_prompts.to(self.device),
                    target.unsqueeze(1).to(self.device),
                )
                logits = self.model(batch_prompts["input_ids"], batch_prompts["attention_mask"])
                loss = self._loss(logits, target)
                total_loss += loss.item() * batch_prompts.size(0)
                all_logits.append(logits.cpu())

        df["Scores"] = torch.cat(all_logits).numpy()
        avg_loss = total_loss / len(self.dataset)
        return EntityMapping.read_table_mappings(df[["Src", "Tgt", "Scores"]], threshold=threshold, cardinality=cardinality), avg_loss