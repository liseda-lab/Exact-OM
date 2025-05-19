from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from torch import Tensor

from matcha_dl.core.contracts.base import SelfRegisteringComponent, LoggingClass
from matcha_dl.core.entities.registry import ComponentType
from matcha_dl.core.entities.dataset import DatasetMask
from matcha_dl.utils.data import read_table
from matcha_dl.core.entities.configs.dataset import PLotAgregationMethod

from mowl.datasets import PathDataset as OWLDataset
from mowl.owlapi import OWLOntology
from org.semanticweb.HermiT import Reasoner
from org.semanticweb.owlapi.reasoner import InferenceType


# from jpype import java


DataFrame = pd.DataFrame


class IDataset(SelfRegisteringComponent, LoggingClass, Dataset):

    component_type = ComponentType.DATASET

    def __init__(self, 
                 output_path: Path, 
                 matchers: List[str],
                 validation_set: Optional[float] = 0.1,
                 pre_filtering: Optional[bool] = True,
                 pre_filtering_threshold: Optional[float] = 0.85,
                 example: Optional[List[bool]] = None,
                 num_workers: Optional[int] = None,
                 **kwargs
        ) -> None:


        self._output_path = output_path / "dataset"
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._matchers = matchers
        self._validation_set = validation_set
        self._num_workers = num_workers

        self._source = None
        self._target = None
        self._source_reasoner = None
        self._target_reasoner = None
        self._candidates = None
        self._reference = None
        self._negatives = None
        self._matcha_features = None

        self._pre_filtering = pre_filtering
        self._pre_filtering_threshold = pre_filtering_threshold

        self._default_kind = DatasetMask.train

        self._example = example
        if self.in_context_training:
            self._default_kind = DatasetMask.inference

        self._cache_ok = kwargs.get("cache_ok", True)

        LoggingClass.__init__(self, logger=kwargs.get("logger"))

    @property
    def default_kind(self) -> DatasetMask:
        return self._default_kind

    @default_kind.setter
    def default_kind(self, kind: DatasetMask):
        self._default_kind = kind

    @property
    def example(self) -> List[bool]:
        return self._example

    @property
    def in_context_training(self) -> bool:
        return self._example is not None and any(self._example)

    @property
    def matchers(self) -> List[str]:
        return self._matchers
    
    @property
    def validation_set(self) -> float:
        return self._validation_set
    
    @property
    def pre_filtering(self) -> bool:
        return self._pre_filtering
    
    @property
    def pre_filtering_threshold(self) -> float:
        return self._pre_filtering_threshold

    @property
    def source(self) -> OWLDataset:
        return self._source
    
    @property
    def target(self) -> OWLDataset:
        return self._target
    
    @property
    def source_reasoner(self) -> Reasoner:
        if self.source is None:
            self.log("Source ontology not loaded.", level="error")
            raise ValueError("Source ontology not loaded.")
        if self._source_reasoner is None:
            self._source_reasoner = Reasoner.ReasonerFactory().createReasoner(self.source.ontology)
            self._source_reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY, InferenceType.OBJECT_PROPERTY_HIERARCHY)
        return self._source_reasoner
    
    @property
    def target_reasoner(self) -> Reasoner:
        if self.target is None:
            self.log("Target ontology not loaded.", level="error")
            raise ValueError("Target ontology not loaded.")
        if self._target_reasoner is None:
            self._target_reasoner = Reasoner.ReasonerFactory().createReasoner(self.target.ontology)
            self._target_reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY, InferenceType.OBJECT_PROPERTY_HIERARCHY)
        return self._target_reasoner
    
    @property
    def candidates(self) -> DataFrame:
        return self._candidates
    
    @property
    def reference(self) -> DataFrame:
        return self._reference
    
    @property
    def negatives(self) -> DataFrame:
        return self._negatives
    
    @property
    def matcha_features(self) -> Dict[str, Dict[str, List[float]]]:
        return self._matcha_features
    
    @property
    def output_path(self) -> Path:
        return self._output_path
    
    @property
    def num_workers(self) -> Optional[int]:
        return self._num_workers
    
    def load_ontologies(self, source_path: Path, target_path: Path) -> None:
        
        self._source = OWLDataset(str(source_path))
        
        self.log("#Loaded Source...", level="debug")
        
        self._target = OWLDataset(str(target_path))
        
        self.log("#Loaded Target...", level="debug")

    def load_candidates(self, file_path: Path) -> None:

        if not file_path.exists():
            self.log(f"Candidates file not found at {file_path}", level="error")
            raise FileNotFoundError(f"Candidates file not found at {file_path}")
                                    
        self._candidates = read_table(file_path)
        self._candidates.columns = ["Src", "Tgt", "Label"]

        self.log("#Loaded Candidates...", level="debug")
              
    def load_reference(self, file_path: Path) -> None:
        
        self._reference = read_table(file_path)
        self._reference.columns = ["Src", "Tgt", "Label"]

        self.log("#Loaded Reference...", level="debug")

    def load_negatives(self, file_path: str) -> None:

        if not file_path.exists():
            self.log(f"Negatives file not found at {file_path}", level="error")
            raise FileNotFoundError(f"Negatives file not found at {file_path}")
        
        self._negatives = read_table(file_path)
        self._negatives.columns = ["Src", "Tgt", "Label"]

        self.log("#Loaded Negatives...", level="debug")

    
    def load_data(self, matcha_features_file: Path) -> None:

        if not matcha_features_file.exists():
            self.log(f"Matcha Features file not found at {matcha_features_file}", level="error")
            raise FileNotFoundError(f"Matcha Features file not found at {matcha_features_file}")

        df = read_table(matcha_features_file)
        df.columns = ["Src", "Tgt"] + self.matchers

        # self._matcha_features = {
        #     src_ent: {
        #         row["Tgt"]: row[self.matchers].values.tolist()
        #         for _, row in df[df["Src"] == src_ent].iterrows()
        #     }
        #     for src_ent in df["Src"].unique()
        # }

        df_grouped = df.groupby("Src")
        self._matcha_features = {
            src_ent: {
                row["Tgt"]: row[self.matchers].values.tolist()
                for _, row in group.iterrows()
            }
            for src_ent, group in df_grouped
        }

        self.log("#Loaded Matcha Features...", level="debug")

    def plot_matcha_features(
        self,
        plot_reference: bool = True,
        plot_negatives: bool = True,
        plot_candidates: bool = True,
        aggregate_funcs: Optional[List[PLotAgregationMethod]] = None,
        **kwargs
    ) -> Path:
        """
        Wrapper to select which datasets to plot (reference, negatives, candidates)
        and which aggregation methods to apply.
        """

        plot_dir=self.output_path / "matcha_features_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        data_dict: Dict[str, pd.DataFrame] = {}
        if plot_reference and getattr(self, 'reference', None) is not None:
            data_dict['reference'] = self.reference
        if plot_negatives and getattr(self, 'negatives', None) is not None:
            data_dict['negatives'] = self.negatives
        if plot_candidates and getattr(self, 'candidates', None) is not None:
            data_dict['candidates'] = self.candidates
        # build aggregate function map
        if aggregate_funcs is None:
            agg_list = [PLotAgregationMethod.mean, PLotAgregationMethod.max]
        else:
            agg_list = aggregate_funcs
        method_map: Dict[PLotAgregationMethod, Callable[[pd.DataFrame], pd.Series]] = {
            PLotAgregationMethod.mean: lambda df: df.mean(axis=1),
            PLotAgregationMethod.max: lambda df: df.max(axis=1),
            PLotAgregationMethod.min: lambda df: df.min(axis=1),
            PLotAgregationMethod.sum: lambda df: df.sum(axis=1),
            PLotAgregationMethod.median: lambda df: df.median(axis=1),
            PLotAgregationMethod.mode: lambda df: df.mode(axis=1)[0] if not df.mode(axis=1).empty else pd.Series([], dtype=float),
            PLotAgregationMethod.std: lambda df: df.std(axis=1),
            PLotAgregationMethod.var: lambda df: df.var(axis=1),
            PLotAgregationMethod.count: lambda df: df.count(axis=1),
        }
        agg_map: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
            method.value: method_map[method] for method in agg_list
        }
        return self.plot_matcha_distributions(
            data_dict=data_dict,
            aggregate_funcs=agg_map,
            plot_dir=plot_dir,
            **kwargs
        )
    
    def plot_matcha_distributions(
        self,
        data_dict: Dict[str, pd.DataFrame] = None,
        figsize: Tuple[int, int] = (8, 5),
        kde: bool = True,
        bins: int = 20,
        color: str = "blue",
        alpha: float = 0.6,
        dpi: int = 300,
        grid: bool = True,
        aggregate_funcs: Dict[str, Callable[[pd.DataFrame], pd.Series]] = None,
        plot_by_matcher: bool = False,
        plot_dir: Optional[Path] = None,
        **kwargs
    ) -> Path:
        """
        Generates and saves distribution plots for each feature in one or more datasets,
        plus optional aggregate distributions (e.g., mean, max).

        Parameters:
        -------------
        data_dict: dict of {name: DataFrame}
            Mapping of label to DataFrame to plot. If None, defaults to {'negatives': self.negatives}.
        figsize: tuple
            Size of each figure.
        kde: bool
            Whether to overlay a KDE.
        bins: int
            Number of histogram bins.
        color: str
            Color for the histograms.
        alpha: float
            Transparency level for the histograms.
        dpi: int
            Resolution for saved plots.
        grid: bool
            Whether to display grid lines.
        aggregate_funcs: dict of {label: function}
            Optional mapping from aggregate label (e.g. 'mean', 'max') to a function
            that takes feature_df and returns a pd.Series of aggregate scores.
        **kwargs:
            Additional keyword arguments passed to seaborn.histplot.

        Returns:
        --------
        Path to the directory where plots are saved.
        """
        # Default to negatives if no data_dict provided
        if data_dict is None:
            data_dict = {'negatives': self.negatives,
                         'candidates': self.candidates,
                         'reference': self.reference}

        # Default aggregates: mean and max
        if aggregate_funcs is None:
            aggregate_funcs = {
                'average': lambda df: df.mean(axis=1),
                'max': lambda df: df.max(axis=1),
            }

        for label, df in data_dict.items():
            if df is None:
                self.log(f"No data provided for '{label}'. Skipping.", level="warning")
                continue

            self.log(f"Plotting {label}...", level="debug")

            # Only compute features if not already present
            if 'Features' not in df.columns:
                df = self._get_matcha_features(df.copy())
            else:
                df = df.copy()

            feature_df = pd.DataFrame(df['Features'].tolist(), columns=self.matchers)

            if plot_by_matcher:

                # Plot each feature
                for feature in self.matchers:
                    plt.figure(figsize=figsize)
                    sns.histplot(
                        feature_df[feature],
                        kde=kde,
                        bins=bins,
                        color=color,
                        alpha=alpha,
                    )
                    plt.xlabel("Similarity Score")
                    plt.ylabel("Density")
                    plt.title(f"{label.capitalize()} - Distribution for {feature}")
                    plt.grid(grid)

                    out_path = plot_dir / f"{label}_{feature}_distribution.png"
                    plt.savefig(out_path.resolve(), dpi=dpi)
                    plt.close()

            if aggregate_funcs:

                # Plot aggregates
                for agg_label, func in aggregate_funcs.items():
                    series = func(feature_df)
                    plt.figure(figsize=figsize)
                    sns.histplot(
                        series,
                        kde=kde,
                        bins=bins,
                        color=color,
                        alpha=alpha,
                    )
                    plt.xlabel(f"{agg_label.capitalize()} Feature Score")
                    plt.ylabel("Density")
                    plt.title(f"{label.capitalize()} - {agg_label.capitalize()} Feature Distribution")
                    plt.grid(grid)

                    out_path = plot_dir / f"{label}_{agg_label}_feature_distribution.png"
                    plt.savefig(out_path.resolve(), dpi=dpi)
                    plt.close()

        self.log(f"Plots saved in: {plot_dir}", level="info")
        return plot_dir

    def _get_matcha_features(self, dataset: pd.DataFrame) -> pd.DataFrame:

        # check if dataset features are already set
        if "Features" in dataset.columns:
            self.log("Dataset already has features. Skipping feature extraction.", level="debug")
            return dataset

        feats = []
        
        for row in dataset.itertuples(index=False):

            try:
                vector = self.matcha_features[row.Src][row.Tgt]
            except AttributeError:
                self.log("Matcha features not loaded.", level="error", exc_info=True)
                raise ValueError("Scores for source {} and target {} not found.".format(row["Src"], row["Tgt"]))
            
            feats.append(vector)
            

        dataset["Features"] = feats

        return dataset

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def save(self) -> Path:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def has_cache(self) -> None:
        pass

    @abstractmethod
    def process(self) -> None:
        pass

    @abstractmethod
    def x(self, kind: Optional[str] = None) -> np.ndarray:
        pass

    @abstractmethod
    def y(self, kind: Optional[str] = None) -> np.ndarray:
        pass

