
from exact.core.entities.registry import ComponentRegistry, ComponentType

from . import datasets, trainer, metrics, models

# Registering the dependencies

ComponentRegistry.register_dependency("SemanticScorer", {ComponentType.DATASET: "ContextDataset", ComponentType.TRAINER: "SemanticAlignmentRunner"})
ComponentRegistry.register_dependency("PairAdaptiveSemanticScorer", {ComponentType.DATASET: "PairAdaptiveContextDataset", ComponentType.TRAINER: "SemanticAlignmentRunner"})
ComponentRegistry.register_dependency("SecondPassReranker", {ComponentType.DATASET: "ContextDataset", ComponentType.TRAINER: "SemanticAlignmentRunner"})
