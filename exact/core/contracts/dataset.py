"""Dataset interface.

Concrete ontology loading, candidate generation, caching, and processing live in
``exact.impl.datasets.base``.  A small filtering template remains here for
backward-compatible lightweight dataset subclasses.
"""

from abc import abstractmethod
from typing import Any, Optional, Sequence, Tuple

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset

from exact.core.contracts.base import LoggingClass, SelfRegisteringComponent
from exact.core.entities.registry import ComponentType

DataFrame = pd.DataFrame


class IDataset(SelfRegisteringComponent, LoggingClass, Dataset):
    """Interface implemented by alignment datasets."""

    component_type = ComponentType.DATASET

    def __init__(self, *, logger=None, **_: Any) -> None:
        LoggingClass.__init__(self, logger=logger)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    def source_ignored_alignment_classes(self) -> set[str]:
        return self._ignored_alignment_class_iris("src")

    @property
    def target_ignored_alignment_classes(self) -> set[str]:
        return self._ignored_alignment_class_iris("tgt")

    def _ignored_alignment_class_iris(self, side: str) -> set[str]:
        if not getattr(self, "_filter_ignored_alignment_classes", False):
            return set()
        side_key = "src" if side in {"src", "source"} else "tgt"
        cached = (
            getattr(self, "_source_ignored_alignment_classes", None)
            if side_key == "src"
            else getattr(self, "_target_ignored_alignment_classes", None)
        )
        if cached is not None:
            return set(cached)
        source = (
            getattr(self, "source", None) if side_key == "src" else getattr(self, "target", None)
        )
        ignored = set(source.excluded_from_alignment()) if source is not None else set()
        if side_key == "src":
            self._source_ignored_alignment_classes = ignored
        else:
            self._target_ignored_alignment_classes = ignored
        return ignored

    def _filter_ignored_iris(self, iris: Sequence[Any], side: str) -> list[Any]:
        ignored = self._ignored_alignment_class_iris(side)
        return [iri for iri in iris if str(iri) not in ignored]

    def _filter_candidates_ignored_classes(self, df: Optional[DataFrame]) -> Optional[DataFrame]:
        if df is None or df.empty or not getattr(self, "_filter_ignored_alignment_classes", False):
            return df
        if not {"Src", "Tgt"}.issubset(df.columns):
            return df
        keep = ~(
            df["Src"].astype(str).isin(self.source_ignored_alignment_classes)
            | df["Tgt"].astype(str).isin(self.target_ignored_alignment_classes)
        )
        return df.loc[keep].reset_index(drop=True)

    def _filter_mappings_ignored_classes(
        self, df: Optional[DataFrame], label: str
    ) -> Optional[DataFrame]:
        del label
        return self._filter_candidates_ignored_classes(df)
