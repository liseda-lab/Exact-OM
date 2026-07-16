from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Tuple, Union

if TYPE_CHECKING:
    from exact.core.entities.mappings import EntityMapping, ReferenceMapping


@dataclass
class EvaluationData:
    prediction_mappings: Optional[List["EntityMapping"]] = None
    reference_mappings: Optional[List["ReferenceMapping"]] = None
    reference_and_candidates: Optional[List[Tuple["ReferenceMapping", List["EntityMapping"]]]] = (
        None
    )
    null_reference_mappings: Optional[List["ReferenceMapping"]] = None
    K: List[int] = field(default_factory=lambda: [1, 5, 10])


@dataclass(frozen=True)
class EvaluationRequest:
    """Backend-neutral inputs for one evaluation invocation."""

    alignment: Union[
        List[Tuple["ReferenceMapping", List["EntityMapping"]]], List["EntityMapping"], Path
    ]
    full_reference: Union[List["ReferenceMapping"], Path, None] = None
    train_reference: Union[List["ReferenceMapping"], Path, None] = None
    reference_candidates: Optional[Path] = None
    source: Any = None
    target: Any = None
    k: Tuple[int, ...] = (1, 5, 10)
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendEvaluation:
    """Metrics and capability metadata returned by one evaluator backend."""

    metrics: Mapping[str, Optional[float]]
    skipped: Mapping[str, str] = field(default_factory=dict)
    version: Optional[str] = None


class MetricNames(Enum):
    PRECISION = "P"
    RECALL = "R"
    F1 = "F1"
    HITS_AT_K = "Hits@{k}"
    MRR = "MRR"
