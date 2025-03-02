
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from matcha_dl.core.entities.mappings import EntityMapping, ReferenceMapping


@dataclass
class EvaluationData:
    prediction_mappings: Optional[List['EntityMapping']] = None
    reference_mappings: Optional[List['ReferenceMapping']] = None
    reference_and_candidates: Optional[List[Tuple['ReferenceMapping', List['EntityMapping']]]] = None
    null_reference_mappings: Optional[List['ReferenceMapping']] = None
    K: Optional[List[int]] = None

class MetricNames(Enum):
    PRECISION = "P"
    RECALL = "R"
    F1 = "F1"
    HITS_AT_K = "Hits@{k}"
    MRR = "MRR"