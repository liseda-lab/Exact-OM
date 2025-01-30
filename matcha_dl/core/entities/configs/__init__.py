
from matcha_dl.core.contracts.dataset import IDataset
from matcha_dl.core.contracts.loss import ILoss
from matcha_dl.core.contracts.metric import IMetric
from matcha_dl.core.contracts.model import IModel
from matcha_dl.core.contracts.optimizer import IOptimizer
from matcha_dl.core.contracts.stopper import IStopper
from matcha_dl.core.contracts.trainer import ITrainer

from .config import ConfigModel
from .registry import ComponentRegistry, ComponentType

# Registering the validators

ComponentRegistry.register_validator(ComponentType.METRIC, IMetric)
ComponentRegistry.register_validator(ComponentType.TRAINER, ITrainer)
ComponentRegistry.register_validator(ComponentType.MODEL, IModel)
ComponentRegistry.register_validator(ComponentType.LOSS, ILoss)
ComponentRegistry.register_validator(ComponentType.STOPPER, IStopper)
ComponentRegistry.register_validator(ComponentType.DATASET, IDataset)
ComponentRegistry.register_validator(ComponentType.OPTIMIZER, IOptimizer)






