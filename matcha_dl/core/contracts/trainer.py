import random
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Tuple, TYPE_CHECKING

import logging

import numpy as np
import pandas as pd
import torch as th

from matcha_dl.core.contracts import SelfRegisteringComponent, LoggingClass
from matcha_dl.core.entities.registry import ComponentType
from matcha_dl.core.entities.mappings import EntityMapping
from matcha_dl.utils.mappings import fill_anchored_scores
from matcha_dl.utils.data import read_table

if TYPE_CHECKING:
    from matcha_dl.core.contracts.dataset import IDataset
    from matcha_dl.core.contracts.loss import ILoss
    from matcha_dl.core.contracts.model import IModel
    from matcha_dl.core.contracts.optimizer import IOptimizer
    from matcha_dl.core.contracts.stopper import IStopper


class ITrainer(SelfRegisteringComponent, LoggingClass):

    component_type = ComponentType.TRAINER

    def __init__(
        self,
        dataset: 'IDataset',
        model: Type['IModel'],
        loss: Type['ILoss'],
        optimizer: Type['IOptimizer'],
        loss_params: Optional[Dict[str, Any]] = {},
        optimizer_params: Optional[Dict[str, Any]] = {},
        model_params: Optional[Dict[str, Any]] = {},
        stopping: Optional[Type['IStopper']] = None,
        stopping_params: Optional[Dict[str, Any]] = {},
        device: Optional[int] = 0,
        output_dir: Optional[Path] = None,
        use_last_checkpoint: Optional[bool] = False,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        
        LoggingClass.__init__(self, logger=logger)

        # Load Args

        self._dataset = dataset
        self._device = device
        self._model = model(**model_params).to(self.device)
        self._optimizer = optimizer(self._model.parameters(), **optimizer_params)
        self._loss = loss(device=self.device, **loss_params)
        
        if stopping is not None:
            self._stopping = stopping(**stopping_params)
        else:
            self._stopping = None

        self._output_dir = output_dir

        self._epoch = 1

        # Load checkpoint if exists

        if use_last_checkpoint:
            if self.checkpoints_dir.is_dir() and any(self.checkpoints_dir.iterdir()):
                self.load_checkpoint()
                self.log(f"Loaded checkpoint {self._get_last_checkpoint()}")
            else:
                self.log(f"No checkpoints found in {self.checkpoints_dir}")

        # Create output directories

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.alignment_dir.mkdir(parents=True, exist_ok=True)

    @property
    def epoch(self) -> int:
        return self._epoch

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
    def optimizer(self) -> 'IOptimizer':
        return self._optimizer

    @property
    def loss(self) -> 'ILoss':
        return self._loss

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def stopping(self) -> Optional['IStopper']:
        return self._stopping

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def checkpoints_dir(self) -> Path:
        return (self._output_dir / "training_checkpoints").resolve()

    @property
    def logs_dir(self) -> Path:
        return (self._output_dir / "training_logs").resolve()

    @property
    def alignment_dir(self) -> Path:
        return (self._output_dir / "alignment").resolve()

    @property
    def checkpoints(self) -> List[str]:
        return [x.name for x in self.checkpoints_dir.glob("**/*") if x.is_file()]

    @abstractmethod
    def train(self, 
              epochs: Optional[int] = 100, 
              batch_size: Optional[int] = None,
              val_every: Optional[int] = 1, 
              save_interval: Optional[int] = 5, 
              **kwargs
    ) -> None:
        
        pass

    @abstractmethod
    def repair(self, **kwargs) -> None:
        pass

    @abstractmethod
    def predict(self, kind: str = "inference", 
                threshold: Optional[float] = 0.7,
                **kwargs
    ) -> Tuple[List[EntityMapping], float]:
        
        pass

    def save_alignment(self, preds: List[EntityMapping], candidates_one2many_path: Optional[Path] = None) -> None:

        if candidates_one2many_path is not None:
            candidates_one2many = read_table(candidates_one2many_path)
            candidates_one2many.columns = ["Src", "Tgt", "Candidates"]
            return self._save_local_alignment(preds, candidates_one2many)

        else:
            return self._save_global_alignment(preds)

    def _save_global_alignment(self, preds: List[EntityMapping], save_dir: Optional[Path] = None):

        # Get the best mapping for each unique source entity

        all_sources = {}
        for ent_map in preds:
            if ent_map.head not in all_sources or ent_map.score > all_sources[ent_map.head].score:
                all_sources[ent_map.head] = ent_map

        # Extract the mappings as tuples

        global_alignment = EntityMapping.as_tuples(list(all_sources.values()), with_score=True)

        # Save the global alignment

        global_dir = str(self.alignment_dir) + f"/{'src2tgt.maps'}_global.tsv"

        pd.DataFrame(global_alignment, columns=["SrcEntity", "TgtEntity", "Score"]).to_csv(
            global_dir, sep="\t", index=False
        )

        return global_dir

    def _save_local_alignment(self, preds: List[EntityMapping], candidates_one2many: pd.DataFrame):

        # candidates is now a 1-1 format for this the original candidates are required

        ranking_results = fill_anchored_scores(candidates_one2many.values, preds)

        local_dir = str(self.alignment_dir) + f"/{'src2tgt.maps'}_local.tsv"

        pd.DataFrame(ranking_results, columns=["SrcEntity", "TgtEntity", "TgtCandidates"]).to_csv(
            local_dir, sep="\t", index=False
        )

        return local_dir

    def load_checkpoint(self, checkpoint: Optional[str] = "last"):

        if checkpoint == "last":
            checkpoint = "{}.pt".format(self._get_last_checkpoint())
        
        else:
            if not (self.checkpoints_dir / checkpoint).exists():
                self.log(f"Checkpoint {checkpoint} not found")
                raise FileNotFoundError(f"Checkpoint {checkpoint} not found")

        checkpoint = th.load((self.checkpoints_dir / checkpoint).resolve(), weights_only=False)

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._epoch = checkpoint["epoch"]
        self._loss = checkpoint["loss"]

    def save_checkpoint(self) -> None:

        checkpoint = str(self._get_last_checkpoint() + 1)

        # Save torch model

        th.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            (self.checkpoints_dir / "{}.pt".format(checkpoint)).resolve(),
        )

        self.log(f"Saved checkpoint {checkpoint}", level="debug")


    def _get_last_checkpoint(self) -> int:
        try:
            res = int(sorted(self.checkpoints)[-1].split(".")[0])
        except IndexError:
            res = 0

        return res
