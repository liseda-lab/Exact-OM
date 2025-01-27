
from .registry import ComponentRegistry
from .config import ConfigModel

from matcha_dl.core.contracts.metric import IMetric
from matcha_dl.core.contracts.trainer import ITrainer
from matcha_dl.core.contracts.model import IModel
from matcha_dl.core.contracts.loss import ILoss
from matcha_dl.core.contracts.stopper import IStopper
from matcha_dl.core.entities.datasets import IDataset

# Registering the validators

ComponentRegistry.register_validator("metric", IMetric)
ComponentRegistry.register_validator("trainer", ITrainer)
ComponentRegistry.register_validator("model", IModel)
ComponentRegistry.register_validator("loss", ILoss)
ComponentRegistry.register_validator("stopper", IStopper)
ComponentRegistry.register_validator("dataset", IDataset)

# Registering the dependencies

ComponentRegistry.register_dependency("MlpClassifier", {"dataset": "TabularDataset", "trainer": "MLPTrainer"})