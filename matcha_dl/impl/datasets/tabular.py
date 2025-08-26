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

from matcha_dl.core.contracts.dataset import IDataset
from matcha_dl.core.entities.dataset import DatasetMask

from matcha_dl.core.entities.configs.dataset import PLotAgregationMethod


class TabularDataset(IDataset):

    def __init__(self, output_path: Path, **kwargs) -> None:
        super().__init__(output_path, **kwargs)

        self._df = None
        self._df_save_path = self.output_path / "dataset.csv"

    @property
    def dataframe(self) -> DataFrame:
        return self._df
    
    def __len__(self) -> int:
        if self.dataframe is None:
            return 0
        return len(self.dataframe[self.dataframe[self.default_kind]])
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:

        if idx >= len(self):
            self.log(f"Index {idx} out of bounds.", level="error")
            raise IndexError("Index out of bounds.")
        
        elif idx < 0:
            self.log(f"Index {idx} out of bounds.", level="error")
            raise IndexError("Index out of bounds.")
        
        return self.pre_process_x(self.x()[idx]), self.pre_process_y(self.y()[idx])
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:
        return self.dataframe.__str__()

    def x(self, kind: Optional[str] = None) -> np.ndarray:

        if kind is None:
            kind = self.default_kind

        if self.dataframe is None:
            self.log("Dataset is empty.", level="error")
            raise ValueError("Dataset is empty.")

        return self.dataframe[self.dataframe[kind]]["Features"].values

    def y(self, kind: Optional[str] = None) -> np.ndarray:

        if kind is None:
            kind = self.default_kind

        if self.dataframe is None:
            self.log("Dataset is empty.", level="error")
            raise ValueError("Dataset is empty.")

        return self.dataframe[self.dataframe[kind]]["Label"].values
    
    def pre_process_x(self, data: np.ndarray) -> Tensor:
        return th.tensor(data, dtype=th.float32)
    
    def pre_process_y(self, data: np.ndarray) -> Tensor:
        return th.tensor(data, dtype=th.float32)

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
    
    def get_features(self, dataset: DataFrame) -> DataFrame:
        return self._get_matcha_features(dataset)

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


        pre_filtered_set = pd.DataFrame()
        # pre-filter easy mappings
        if self.pre_filtering:
            self.log("#Pre-filtering Inference Set...", level="debug")

            self.log(f'#Getting Features...', level="debug")
            inference_set = self._get_matcha_features(inference_set)

            self.log(f'#Pre-filtering with threshold {self.pre_filtering_threshold}...', level="debug")

            # Filter easy mappings based on pre_filtering_threshold

            filter_mask = inference_set["Features"].apply(max) >= self.pre_filtering_threshold
            pre_filtered_mappings = inference_set[filter_mask]
            inference_set = inference_set[~filter_mask]

            # # Filter candidates for mappings that are already in pre_filtered_mappings

            pre_filter_candidates_mask = inference_set["Src"].isin(pre_filtered_mappings["Src"])
            filtered_candidates = inference_set[pre_filter_candidates_mask]
            inference_set = inference_set[~pre_filter_candidates_mask]

            # combine pre_filtered_mappings and filtered_candidates
            
            pre_filtered_set = pd.concat([pre_filtered_mappings, filtered_candidates], ignore_index=True)

            # assign pre-filtered label
            pre_filtered_set[DatasetMask.train] = False
            pre_filtered_set[DatasetMask.validation] = False
            pre_filtered_set[DatasetMask.inference] = False
            pre_filtered_set[DatasetMask.prefiltered] = True

            self.log(f"#Pre-filtered Inference Set: {len(pre_filtered_set)} samples", level="debug")

        # get features from matcha

        self.log("#Getting Features...", level="debug")
        inference_set = self.get_features(inference_set)

        # assign label
        inference_set[DatasetMask.train] = False
        inference_set[DatasetMask.validation] = False
        inference_set[DatasetMask.inference] = True
        inference_set[DatasetMask.prefiltered] = False

        self.log(f"#Inference Set: {len(inference_set)} samples", level="debug")

        inference_set = pd.concat([inference_set, pre_filtered_set], ignore_index=True)
        self.log("#Inference Set Created", level="debug")
        
        if self.reference is not None:

            if not self.in_context_training:

                if self.negatives is None:
                    self.log("Negatives not loaded.", level="error")
                    raise ValueError("Negatives not loaded.")

                # get training set
                self.log("Creating Training Set...", level="debug")

                # get positive samples from refs
                positive_set = self.reference

                # add negatives
                self.log("#Adding Negative Samples...", level="debug")
                negative_set = self.negatives

                # combine positive and negative samples
                training_set = pd.concat([positive_set, negative_set], ignore_index=True)

                # get features from matcha
                self.log("#Getting Features...", level="debug")
                training_set = self.get_features(training_set)

                self.log("#Shuffling Training Set...", level="debug")

                # Ensure reproducibility by using seeded np random state

                training_set = training_set.sample(frac=1, random_state=random.randint(0, 2**32 - 1)).reset_index(drop=True)

                if self.validation_set is not None and self.validation_set > 0:

                    self.log(f'#Splitting Training Set into Train and Validation Sets({self.validation_set*100}%)...', level="debug")

                    # split training set into train and validation sets
                
                    validation_set = training_set.sample(frac=self.validation_set, random_state=random.randint(0, 2**32 - 1)).reset_index(drop=True)

                    training_set = training_set.drop(validation_set.index).reset_index(drop=True)

                    # assign validation label
                    validation_set[DatasetMask.train] = False
                    validation_set[DatasetMask.validation] = True
                    validation_set[DatasetMask.inference] = False
                    validation_set[DatasetMask.prefiltered] = False

                # assign training label
                training_set[DatasetMask.train] = True
                training_set[DatasetMask.validation] = False
                training_set[DatasetMask.inference] = False
                training_set[DatasetMask.prefiltered] = False


                self.log("#Combining Training, Validation and Inference Sets...", level="debug")

                dataset = pd.concat([training_set, validation_set, inference_set], ignore_index=True)

            else:
                self.log("#In-Context Training Enabled: Skipping Training Set, using Inference Set...", level="debug")

                # if example is set to True, skip training set and use inference set
                dataset = inference_set

        else:

            self.log("#No Reference: Skipping Training Set, using Inference Set...", level="debug")

            # if reference is None, skip training set and use inference set
            dataset = inference_set

        self.log("#Processing Done", level="debug")

        self._df = dataset

        return self

    def plot_matcha_features(self) -> Path:
        
        aggregate_funcs = self.plot_params.get("aggregate_funcs", None)
        plot_prefiltered = self.plot_params.get("plot_prefiltered", False)

        other_params = dict(self.plot_params)    # shallow copy
        for k in ("aggregate_funcs","plot_prefiltered"):
            other_params.pop(k, None)

        base_path = super().plot_matcha_features()

        if plot_prefiltered and self.pre_filtering:
            self.log("Plotting Prefiltered Features...", level="debug")
            df_pref = self.dataframe[self.dataframe[DatasetMask.prefiltered]]

            if df_pref.empty:
                self.log("#No Prefiltered Features to plot", level="debug")
                return base_path

            # build a small map of {name:function} just like the base wrapper does
            methods = aggregate_funcs or [PLotAgregationMethod.mean, PLotAgregationMethod.max]
            agg_map = {
                m.value: (lambda df, m=m: getattr(df, m.value)(axis=1))
                for m in methods
            }

            # delegate to the same distribution-plotter
            return self.plot_matcha_distributions(
                data_dict={"prefiltered": df_pref},
                aggregate_funcs=agg_map,
                **other_params
            )

        return base_path
            


