from exact.core.entities.registry import ComponentRegistry, ComponentType

from . import datasets, metrics, models, trainer

# Registering the dependencies

ComponentRegistry.register_dependency(
    "SemanticScorer",
    {ComponentType.DATASET: "ContextDataset", ComponentType.TRAINER: "SemanticAlignmentRunner"},
)
ComponentRegistry.register_dependency(
    "PairAdaptiveSemanticScorer",
    {
        ComponentType.DATASET: "PairAdaptiveContextDataset",
        ComponentType.TRAINER: "SemanticAlignmentRunner",
    },
)
ComponentRegistry.register_dependency(
    "CandidateSetSelector",
    {
        ComponentType.DATASET: "PairAdaptiveContextDataset",
        ComponentType.TRAINER: "SemanticAlignmentRunner",
    },
)
ComponentRegistry.register_dependency(
    "SecondPassReranker",
    {
        ComponentType.DATASET: "PairAdaptiveContextDataset",
        ComponentType.TRAINER: "SemanticAlignmentRunner",
    },
)
