from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

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
                 sanity_check: Optional[bool] = True,
                 plot_params: Dict[str, Any] = {},
                 **kwargs
        ) -> None:


        self._output_path = output_path / "dataset"
        self._output_path.mkdir(parents=True, exist_ok=True)
        self.plot_dir= self.output_path / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self._matchers = matchers
        self._validation_set = validation_set
        self._num_workers = num_workers
        self._plot_params = plot_params

        self._source = None
        self._target = None
        self._source_reasoner = None
        self._target_reasoner = None
        self._candidates = None
        self._reference = None
        self._full_reference = None
        self._negatives = None
        self._matcha_features = None

        self._pre_filtering = pre_filtering
        self._pre_filtering_threshold = pre_filtering_threshold

        self._sanity_check = sanity_check

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
    def full_reference(self) -> DataFrame:
        return self._full_reference
    
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
    
    @property
    def sanity_check(self) -> bool:
        return self._sanity_check
    
    @property
    def plot_params(self) -> Dict[str, Any]:
        return self._plot_params

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

    def load_full_reference(self, file_path: Path) -> None:

        self._full_reference = read_table(file_path)
        self._full_reference.columns = ["Src", "Tgt", "Label"]
        self.log("#Loaded Full Reference...", level="debug")
              
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

    def plot_matcha_features(self) -> Path:
        """
        Wrapper to select which datasets to plot (reference, negatives, candidates)
        and which aggregation methods to apply.
        """
        plot_reference = self.plot_params.get("plot_reference", True)
        plot_negatives = self.plot_params.get("plot_negatives", True)
        plot_candidates = self.plot_params.get("plot_candidates", True)
        aggregate_funcs = self.plot_params.get("aggregate_funcs", None)

        other_params = dict(self.plot_params)    # shallow copy
        for k in ("plot_reference","plot_negatives","plot_candidates","aggregate_funcs"):
            other_params.pop(k, None)

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
            **other_params
        )
    
    def plot_matcha_distributions(
        self,
        data_dict: Dict[str, pd.DataFrame] = None,
        figsize: Tuple[int, int] = (8, 5),
        kde: bool = True,
        bins: int = 20,
        alpha: float = 0.6,
        dpi: int = 300,
        grid: bool = True,
        aggregate_funcs: Dict[str, Callable[[pd.DataFrame], pd.Series]] = None,
        plot_by_matcher: bool = False,
        plot_all_matchers: bool = False,
        all_alpha: Optional[float] = None,
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
        plot_by_matcher: bool
            Whether to plot each matcher separately.
        plot_all_matchers: bool
            Whether to plot all matchers in a single plot.
        all_alpha: float
            Transparency level for the combined plot of all matchers.
        **kwargs:
            Additional keyword arguments passed to seaborn.histplot.

        Returns:
        --------
        Path to the directory where plots are saved.
        """

        plot_dir = self.plot_dir / "matcha_features_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Defaults
        if data_dict is None:
            data_dict = {
                'negatives': self.negatives,
                'candidates': self.candidates,
                'reference': self.reference
            }
        if aggregate_funcs is None:
            aggregate_funcs = {
                'mean':   lambda df: df.mean(axis=1),
                'max':    lambda df: df.max(axis=1),
            }

        for label, df in data_dict.items():
            if df is None:
                self.log(f"No data for '{label}', skipping.", level="warning")
                continue

            self.log(f"Plotting {label}…", level="debug")

            # Copy df to avoid modifying original
            df = df.copy()

            df = df.drop(columns=["Features", "__feat_len"], errors="ignore")

            df = self._get_matcha_features(df)

            feature_df = pd.DataFrame(df['Features'].tolist(), columns=self.matchers)

            # -- Combined view of all matchers?
            if plot_all_matchers:
                alpha_all = all_alpha if all_alpha is not None else alpha * 0.4
                data_map = {m: feature_df[m] for m in self.matchers}
                self._plot_distributions_core(
                    data_map=data_map,
                    plot_dir=plot_dir,
                    prefix=f"{label}_all_matchers",
                    xlabel="Similarity Score",
                    title=f"{label.capitalize()} – All Matchers",
                    single_plot=True,
                    figsize=figsize,
                    kde=kde,
                    bins=bins,
                    alpha=alpha_all,
                    grid=grid,
                    dpi=dpi
                )
                # skip individual/aggregate if only combined desired
                continue

            # -- Individual matcher plots --
            if plot_by_matcher:
                for matcher in self.matchers:
                    self._plot_distributions_core(
                        data_map={matcher: feature_df[matcher]},
                        plot_dir=plot_dir,
                        prefix=f"{label}",
                        xlabel="Similarity Score",
                        title=f"{label.capitalize()}",
                        single_plot=False,
                        figsize=figsize,
                        kde=kde,
                        bins=bins,
                        alpha=alpha,
                        grid=grid,
                        dpi=dpi
                    )

            # -- Aggregate function plots --
            for agg_label, func in aggregate_funcs.items():
                series = func(feature_df)
                self._plot_distributions_core(
                    data_map={agg_label: series},
                    plot_dir=plot_dir,
                    prefix=f"{label}_{agg_label}",
                    xlabel=f"{agg_label.capitalize()} Feature Score",
                    title=f"{label.capitalize()} – {agg_label.capitalize()}",
                    single_plot=False,
                    figsize=figsize,
                    kde=kde,
                    bins=bins,
                    alpha=alpha,
                    grid=grid,
                    dpi=dpi
                )

        # Print original candidates to see 

        self.log(f"All plots saved in: {plot_dir}", level="debug")
        return plot_dir
    
    def _plot_distributions_core(
        self,
        data_map: Dict[str, pd.Series],
        plot_dir: Path,
        prefix: str,
        xlabel: str,
        title: str,
        single_plot: bool = False,
        figsize: Tuple[int, int] = (8, 5),
        kde: bool = True,
        bins: int = 20,
        alpha: float = 0.6,
        grid: bool = True,
        log_scale: bool = False,
        x_clip: Optional[float] = None, 
        dpi: int = 300,
        **kwargs
    ) -> None:
        """
        Core Seaborn-based plotting helper.
        - single_plot=True: overlays all series in one figure.
        - single_plot=False: one figure per series.
        """

        if single_plot:
            plt.figure(figsize=figsize)
            for label, series in data_map.items():
                sns.histplot(
                    series,
                    kde=kde,
                    bins=bins,
                    alpha=alpha,
                    label=label,
                    stat="probability",
                    common_norm=False,
                )
            if log_scale:
                plt.xscale('log')
            if x_clip is not None:
                high = pd.concat(list(data_map.values())).quantile(x_clip)
                plt.xlim(0, high)
            plt.xlabel(xlabel)
            plt.ylabel("Probability")
            plt.title(title)
            plt.legend()
            if grid:
                plt.grid(True)
            plt.tight_layout()
            out = plot_dir / f"{prefix}_combined_distribution.png"
            plt.savefig(out, dpi=dpi)
            plt.close()
            self.log(f"Saved combined plot: {out}", level="debug")
        else:
            for label, series in data_map.items():
                plt.figure(figsize=figsize)
                sns.histplot(
                    series,
                    kde=kde,
                    bins=bins,
                    alpha=alpha,
                    label=label,
                    stat="probability",
                    common_norm=False,
                )
                if log_scale:
                    plt.xscale('log')
                if x_clip is not None:
                    high = pd.concat(list(data_map.values())).quantile(x_clip)
                    plt.xlim(0, high)
                plt.xlabel(xlabel)
                plt.ylabel("Probability")
                plt.title(f"{title} – {label}")
                if grid:
                    plt.grid(True)
                plt.tight_layout()
                out = plot_dir / f"{prefix}_{label}_distribution.png"
                plt.savefig(out, dpi=dpi)
                plt.close()
                self.log(f"Saved plot for {label}: {out}", level="info")

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

