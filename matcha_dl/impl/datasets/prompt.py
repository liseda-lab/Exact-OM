from pathlib import Path
from typing import Optional

import random
import numpy as np
import pandas as pd

DataFrame = pd.DataFrame

from typing import List, Tuple

from matcha_dl.impl.datasets.tabular import TabularDataset
#from matcha_dl.core.entities.configs.prompt 

class PromptDataset(TabularDataset):

    def __init__(
                self,
                example: List[bool],
                task_context: List[bool],
                separator: List[str],
                label_type: List[str],
                label_cardinality: int,
                context_type: str,
                context_cardinality: int,
                context_semantics: str,
                likelihood: str,
                critic: bool,
                **kwargs
        ) -> None:

        super().__init__(**kwargs)

        self._example = example
        self._task_context = task_context
        self._separator = separator
        self._label_type = label_type
        self._label_cardinality = label_cardinality
        self._context_type = context_type
        self._context_cardinality = context_cardinality
        self._context_semantics = context_semantics
        self._likelihood = likelihood
        self._critic = critic
    
    @property
    def example(self) -> bool:
        return self._example
    
    @property
    def task_context(self) -> bool:
        return self._task_context
    
    @property
    def separator(self) -> str:
        return self._separator
    
    @property
    def label_type(self) -> str:
        return self._label_type
    
    @property
    def label_cardinality(self) -> int:
        return self._label_cardinality
    
    @property
    def context_type(self) -> str:
        return self._context_type
    
    @property
    def context_cardinality(self) -> int:
        return self._context_cardinality
    
    @property
    def context_semantics(self) -> str:
        return self._context_semantics
    
    @property
    def likelihood(self) -> str:
        return self._likelihood
    
    @property
    def critic(self) -> bool:
        return self._critic
    
    def process(self) -> "PromptDataset":

        if self.has_cache():
            self.load()
            return self

        return self


    def _generate_prompts(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Generates prompts for the dataset.
        """

        # If critic on
        # Features = [[1src2tgt1:Prompt1:str, Critic:str], [1src2tgt1:Prompt2:str, Critic: str]]
        # If critic off
        # Features = [[[1src2tgt1:Prompt1:str], [1src2tgt1:Prompt2:str]], ...]


        feats = []
        
        for _, row in dataset.iterrows():

            try:
                vector = self.matcha_features.get(row["Src"]).get(row["Tgt"])
            except AttributeError:
                self.log("Matcha features not loaded.", level="error", exc_info=True)
                raise ValueError("Scores for source {} and target {} not found.".format(row["Src"], row["Tgt"]))
            
            feats.append(vector)
            

        dataset["Features"] = feats

        return dataset



