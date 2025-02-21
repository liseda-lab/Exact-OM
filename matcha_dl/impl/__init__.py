
from matcha_dl.core.entities.registry import ComponentRegistry, ComponentType

from . import datasets, losses, models, optimizers, stoppers, trainer, metrics

# Registering the dependencies

ComponentRegistry.register_dependency("MlpClassifier", {ComponentType.DATASET: "TabularDataset", ComponentType.TRAINER: "MLPTrainer"})
