import random
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Tuple, TYPE_CHECKING, Union

import logging

import numpy as np
import pandas as pd
import torch as th
from torch import device as tdevice

import seaborn as sns
import matplotlib.pyplot as plt

from exact.core.contracts import SelfRegisteringComponent, LoggingClass
from exact.core.entities.registry import ComponentType
from exact.core.entities.mappings import EntityMapping
from exact.utils.mappings import fill_anchored_scores
from exact.utils.data import read_table
from exact.core.entities.dataset import DatasetMask

if TYPE_CHECKING:
    from exact.core.contracts.dataset import IDataset
    from exact.core.contracts.loss import ILoss
    from exact.core.contracts.model import IModel
    from exact.core.contracts.optimizer import IOptimizer
    from exact.core.contracts.stopper import IStopper

# TODO implement LR scheduler


class ITrainer(SelfRegisteringComponent, LoggingClass):

    component_type = ComponentType.TRAINER

    def __init__(
        self,
        dataset: 'IDataset',
        model: Type['IModel'],
        model_params: Optional[Dict[str, Any]] = {},
        device: tdevice = tdevice("cuda" if th.cuda.is_available() else "cpu"),
        output_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
        plot_params: Optional[Dict[str, Any]] = {},
        **kwargs,
    ):
        
        LoggingClass.__init__(self, logger=logger)

        # Load Args

        self._dataset = dataset
        self._device = device
        self._model = model(**model_params).to(self.device)
        self._plot_params = plot_params
        
        self._output_dir = output_dir

        # Create output directories

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.alignment_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    @property
    def device(self) -> th.device:
        return th.device(self._device if th.cuda.is_available() else "cpu")

    @property
    def dataset(self) -> 'IDataset':
        return self._dataset

    @property
    def model(self) -> 'IModel':
        return self._model

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def output_dir(self) -> Path:
        return self._output_dir
    
    @property
    def plot_dir(self) -> Path:
        return (self._output_dir / "plots").resolve()
    
    @property
    def plot_params(self) -> Dict[str, Any]:
        return self._plot_params

    @property
    def logs_dir(self) -> Path:
        return (self._output_dir / "training_logs").resolve()

    @property
    def alignment_dir(self) -> Path:
        return (self._output_dir / "alignment").resolve()


    @abstractmethod
    def predict(self, kind: DatasetMask = DatasetMask.inference, 
                threshold: Optional[float] = 0.7,
                **kwargs
    ) -> Tuple[List[EntityMapping], float]:
        
        pass

    def apply_prefilter(self, threshold: Optional[float] = None, cardinality: Optional[int] = None, **kwargs) -> List[EntityMapping]:
        """
        Apply prefiltering to the dataset. This method should be implemented in the derived class.
        """
        pass

    def save_alignment(self, 
                       preds: List[EntityMapping], 
                       candidates_one2many_path: Optional[Path] = None,
                       sub_dir: Optional[str] = None
                       ) -> None:
        
        if sub_dir is not None:
            alignment_dir = self.alignment_dir / sub_dir
            alignment_dir.mkdir(parents=True, exist_ok=True)

        else:
            alignment_dir = self.alignment_dir

        if candidates_one2many_path is not None:
            candidates_one2many = read_table(candidates_one2many_path)
            candidates_one2many.columns = ["Src", "Tgt", "Candidates"]
            return self._save_local_alignment(preds, candidates_one2many, alignment_dir)

        else:
            return self._save_global_alignment(preds, alignment_dir)

    def _save_global_alignment(self, preds: List[EntityMapping], save_dir: Optional[Path] = None):

        # Extract the mappings as tuples

        global_alignment = EntityMapping.as_tuples(preds, with_score=True)

        # Save the global alignment

        global_dir = str(save_dir) + f"/{'src2tgt.maps'}_global.tsv"

        pd.DataFrame(global_alignment, columns=["SrcEntity", "TgtEntity", "Score"]).to_csv(
            global_dir, sep="\t", index=False
        )

        return global_dir

    def _save_local_alignment(self, preds: List[EntityMapping], candidates_one2many: pd.DataFrame, save_dir: Optional[Path] = None):

        # candidates is now a 1-1 format for this the original candidates are required

        ranking_results = fill_anchored_scores(candidates_one2many.values, preds)

        local_dir = str(save_dir) + f"/{'src2tgt.maps'}_local.tsv"

        pd.DataFrame(ranking_results, columns=["SrcEntity", "TgtEntity", "TgtCandidates"]).to_csv(
            local_dir, sep="\t", index=False
        )

        return local_dir
    
    def plot_score_distribution(
        self,
        df: pd.DataFrame,
        kind: DatasetMask,
        figsize: Tuple[int, int] = (8, 6),
        kde: bool = False,
        bins: int = 30,
        all_alpha: float = 0.4,
        dpi: int = 300,
        grid: bool = True,
        log_scale: bool = False,
        x_clip: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Overlaid histogram/KDE of scores. If both labels {0,1} present, plots
        green=Positive, red=Negative. If only one label, plots a single
        blue “Distribution”.
        """
        labels = sorted(df["Label"].unique(), reverse=True)
        if len(labels) > 2:
            self.log(f"Expected ≤2 labels, found {labels}. Skipping plot.", level="warning")
            return

        fig, ax = plt.subplots(figsize=figsize)
        # decide how to name & color each series
        for lbl in labels:
            if len(labels) == 1:
                label_name = "Distribution"
                label_color = "blue"
                subset = df["Scores"]
            else:
                # exactly two labels case
                if lbl == 1:
                    label_name = "Positive"
                    label_color = "green"
                else:
                    label_name = "Negative"
                    label_color = "red"
                subset = df[df["Label"] == lbl]["Scores"]

            if subset.empty:
                self.log(f"Empty subset for label {lbl}: skipping.", level="warning")
                continue

            sns.histplot(
                subset,
                bins=bins,
                kde=kde,
                stat="probability",
                alpha=all_alpha,
                color=label_color,
                label=label_name,
                ax=ax,
                common_norm=False
            )

        # axis transforms
        if log_scale:
            ax.set_xscale("log")
        if x_clip is not None:
            cutoff = float(df["Scores"].quantile(x_clip))
            ax.set_xlim(0, cutoff)

        # labels & title
        ax.set_xlabel("Model score")
        ax.set_ylabel("Probability")
        ax.set_title(f"{kind.name.capitalize()} Score Distribution")

        if grid:
            ax.grid(True)
        ax.legend()
        plt.tight_layout()

        # save
        suffix = "_single" if len(labels) == 1 else ""
        out_file = self.plot_dir / f"{kind.name.lower()}_score_dist{suffix}.png"
        fig.savefig(out_file, dpi=dpi)
        plt.close(fig)

        self.log(f"Saved score distribution plot to {out_file}", level="debug")
