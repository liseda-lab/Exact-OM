
from enum import Enum

class ComponentType(Enum):
    """Enum for valid component types."""
    MODEL = "model"
    DATASET = "dataset"
    TRAINER = "trainer"
    LOSS = "loss"
    OPTIMIZER = "optimizer"
    STOPPER = "stopper"
    METRIC = "metric"