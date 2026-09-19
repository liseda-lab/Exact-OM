"""Opt-in matcher-independent repair; solver and neural dependencies load lazily."""

from .records import (
    BudgetsV2,
    ObjectiveV2,
    PolicyV2,
    RepairInputV2,
    RepairResultV2,
    ReplacementCandidateV2,
    RevisionObjectV2,
    make_objective,
)

__all__ = [
    "BudgetsV2",
    "ObjectiveV2",
    "PolicyV2",
    "RepairInputV2",
    "RepairResultV2",
    "ReplacementCandidateV2",
    "RevisionObjectV2",
    "make_objective",
    "repair",
    "prepare_repair",
    "repair_alignment",
    "bounded_freeze_neural_round",
]


def repair(*args, **kwargs):
    """Run bounded finite-pool repair without invoking matching."""
    from .kernel import repair as run

    return run(*args, **kwargs)


def __getattr__(name):
    """Load preparation/integration adapters only when explicitly requested."""
    if name == "bounded_freeze_neural_round":
        from .pipeline import bounded_freeze_neural_round

        return bounded_freeze_neural_round
    if name in {"prepare_repair", "repair_alignment"}:
        from . import api

        return getattr(api, name)
    raise AttributeError(name)
