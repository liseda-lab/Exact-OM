from ast import literal_eval
from pathlib import Path
from typing import Optional

import random
import numpy as np
import pandas as pd

import torch as th
from torch import Tensor

DataFrame = pd.DataFrame

from typing import List, Tuple

from exact.core.contracts.dataset import IDataset
from exact.core.entities.dataset import DatasetMask

from exact.core.entities.configs.dataset import PLotAgregationMethod


class TabularDataset(IDataset):

    def __init__(self, output_path: Path, **kwargs) -> None:
        super().__init__(output_path, **kwargs)

        self._df = None
        self._df_save_path = self.output_path / "dataset.csv"

    @property
    def dataframe(self) -> DataFrame:
        return self._df
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:
        return self.dataframe.__str__()

    def save(self) -> Path:

        if self.dataframe is None:
            self.log("Dataset is empty.", level="error")
            raise ValueError("Dataset is empty.")
        
        if self.has_cache():
            self.log("#Dataset already saved skyping...", level="debug")
            return self._df_save_path

        self.dataframe.to_csv(str(self._df_save_path), index=False)

        self.log(f"#Dataset saved to {self._df_save_path}", level="debug")

        return self._df_save_path
    
    def load(self):
        self._df = pd.read_csv(self._df_save_path, converters={"Features": literal_eval})

        self.log("#Loaded Cached Dataset...", level="info")

    def has_cache(self) -> bool:
        if self._cache_ok:
            return self._df_save_path.exists()
        return False

    def process(self) -> "TabularDataset":

        if self.has_cache():
            self.load()
            return self
        
        # Inference set

        self.log("Creating Inference set", level="debug")

        if self.candidates is None:
            self.log("Candidates not loaded.", level="error")
            raise ValueError("Candidates not loaded.")

        inference_set = self.candidates

        if self.full_reference is not None:
            # Update Labels based on full_reference
            self.log("#Updating Labels based on Full Reference...", level="debug")
            inference_set = inference_set.merge(self.full_reference, on=["Src", "Tgt"], how="left", suffixes=("", "_y"))
            inference_set["Label"] = inference_set["Label_y"].combine_first(inference_set["Label"])
            inference_set.drop(columns=["Label_y"], inplace=True)

            # Warn how many unique source-target pairs full reference has that are not in the candidates with percentage
            unique_pairs_inference = set(zip(inference_set["Src"], inference_set["Tgt"]))
            unique_pairs_full_ref = set(zip(self.full_reference["Src"], self.full_reference["Tgt"]))
            missing_pairs = unique_pairs_full_ref - unique_pairs_inference  

            if missing_pairs:
                missing_percentage = len(missing_pairs) / len(unique_pairs_full_ref) * 100
                self.log(f"#Warning: {len(missing_pairs)} unique source-target pairs from full reference are not in the candidates ({missing_percentage:.2f}%)", level="warning")

                # Warn how many unique source entities full reference has that are not in the candidates with percentage
                unique_src_inference = set(inference_set["Src"].unique())
                unique_src_full_ref = set(self.full_reference["Src"].unique())
                missing_src = unique_src_full_ref - unique_src_inference

                if missing_src:
                    missing_percentage = len(missing_src) / len(unique_src_full_ref) * 100
                    self.log(f"#Warning: {len(missing_src)} unique source entities from full reference are not in the candidates ({missing_percentage:.2f}%)", level="warning")

                # Warn how many unique target entities full reference has that are not in the candidates with percentage
                unique_tgt_inference = set(inference_set["Tgt"].unique())
                unique_tgt_full_ref = set(self.full_reference["Tgt"].unique())
                missing_tgt = unique_tgt_full_ref - unique_tgt_inference

                if missing_tgt:
                    missing_percentage = len(missing_tgt) / len(unique_tgt_full_ref) * 100
                    self.log(f"#Warning: {len(missing_tgt)} unique target entities from full reference are not in the candidates ({missing_percentage:.2f}%)", level="warning")

        # # # Filter candidates for mappings that are already in pre_filtered_mappings

        # pre_filter_candidates_mask = inference_set["Src"].isin(pre_filtered_mappings["Src"])
        # filtered_candidates = inference_set[pre_filter_candidates_mask]
        # inference_set = inference_set[~pre_filter_candidates_mask]


        inference_set = self.get_features(inference_set)

        self.log(f"#Inference Set: {len(inference_set)} samples", level="debug")

        inference_set = pd.concat([inference_set, pre_filtered_set], ignore_index=True)

        self.log("#Processing Done", level="debug")

        self._df = inference_set

        return self
            


