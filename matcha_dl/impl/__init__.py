
from matcha_dl.core.entities.registry import ComponentRegistry, ComponentType

from . import datasets, losses, models, optimizers, stoppers, trainer, metrics

# Registering the dependencies

ComponentRegistry.register_dependency("MlpClassifier", {ComponentType.DATASET: "TabularDataset", ComponentType.TRAINER: "MLPTrainer"})
ComponentRegistry.register_dependency("PromptClassifier", {ComponentType.DATASET: "PromptDataset", ComponentType.TRAINER: "PromptTrainer"})
ComponentRegistry.register_dependency("EncoderClassifier", {ComponentType.DATASET: "ContextTabularDataset", ComponentType.TRAINER: "EncoderTrainer"})
