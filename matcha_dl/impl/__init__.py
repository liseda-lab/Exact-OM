
from matcha_dl.core.entities.configs import ComponentRegistry, ComponentType

from . import datasets, losses, metrics, models, optimizers, stoppers, trainer

# Registering the dependencies

ComponentRegistry.register_dependency("MlpClassifier", {ComponentType.DATASET: "TabularDataset", ComponentType.TRAINER: "MLPTrainer"})
