from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from torch import Tensor
from ast import literal_eval

from exact.core.contracts.base import SelfRegisteringComponent, LoggingClass
from exact.core.entities.registry import ComponentType
from exact.core.entities.dataset import DatasetMask
from exact.utils.data import read_table
from exact.core.entities.configs.dataset import PLotAgregationMethod

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
                 num_workers: Optional[int] = None,
                 sanity_check: Optional[bool] = True,
                 **kwargs
        ) -> None:


        self._output_path = output_path / "dataset"
        self._output_path.mkdir(parents=True, exist_ok=True)
        self.plot_dir= self.output_path / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self._num_workers = num_workers

        self._source = None
        self._target = None
        self._source_reasoner = None
        self._target_reasoner = None
        self._candidates = None
        self._reference = None
        self._full_reference = None
        self._negatives = None
        self._matcha_features = None

        self._sanity_check = sanity_check

        self._cache_ok = kwargs.get("cache_ok", True)

        LoggingClass.__init__(self, logger=kwargs.get("logger"))

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

        def get_cands(df: pd.DataFrame) -> pd.DataFrame:

            return pd.DataFrame([
                    [source, cand, 0]
                    for source, _, target_cands in df.values
                    for cand in literal_eval(target_cands)
                ], columns=["Src", "Tgt", "Score"])

        # Load One2Many candidates file
        candidates = read_table(str(file_path))
        candidates.columns = ["Src", "Tgt", "Candidates"]

        self.log("#Loaded Candidates Path...", level="debug")

        # Get One2One candidates df
        candidates = get_cands(candidates)

        self.log("#Loaded Candidates...", level="debug")

    def load_reference(self, file_path: Path) -> None:

        self._reference = read_table(file_path)
        self._reference.columns = ["Src", "Tgt", "Label"]
        self.log("#Loaded Reference...", level="debug")

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

