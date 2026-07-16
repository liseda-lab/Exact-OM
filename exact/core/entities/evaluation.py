from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

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


class MetricNames(Enum):
    PRECISION = "P"
    RECALL = "R"
    F1 = "F1"
    HITS_AT_K = "Hits@{k}"
    MRR = "MRR"
