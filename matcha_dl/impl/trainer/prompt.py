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

from matcha_dl.core.contracts.trainer import EntityMapping
from matcha_dl.impl.trainer.mlp import MLPTrainer
from matcha_dl.core.entities.dataset import DatasetMask
from matcha_dl.utils.collate import DataCollator

# 

class PromptTrainer(MLPTrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Set dataset tokenizer
        self.dataset.tokenizer = self.model.tokenizer

    def train(
        self,
        **kwargs
    ):
        self.log(f"PromptTrainer only supports inference and does not support training.. skypping ...", level="debug")

    def predict(self, 
            kind: DatasetMask = DatasetMask.inference, 
            threshold: Optional[float] = None, 
            cardinality: Optional[int] = None, 
            batch_size: int = 32,
            num_workers: int = 0,
            **kwargs) -> Tuple[List[EntityMapping], float]:

        self.dataset.default_kind = kind
        
        collator = DataCollator(self.model.tokenizer)
        
        dataloader = DataLoader(self.dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=num_workers, 
                                pin_memory=True,
                                collate_fn=collator)
        
        self.model.eval()
        self.model.unexpected_response_count.reset()
        total_loss = 0.0
        
        num_examples = len(self.dataset)
        logits_all = None
        start_idx = 0

        with torch.no_grad():
            with tqdm(dataloader, unit="batch") as tepoch:
                for batch_prompts, target in tepoch:
                    tepoch.set_description(f"Running {self.dataset.default_kind}")
                    batch_prompts = {k: v.to(self.device) for k, v in batch_prompts.items()}
                    current_batch_size = batch_prompts["input_ids"].size(0)
                    target = target.to(self.device)
                    
                    logits = self.model(batch_prompts["input_ids"], batch_prompts["attention_mask"])
                    loss = self._loss(logits, target)
                    total_loss += loss.item() * current_batch_size
                    
                    if logits_all is None:
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