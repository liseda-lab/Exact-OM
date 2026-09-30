"""Opt-in matcher-independent repair; solver and neural dependencies load lazily."""

from .records import (
    BudgetsV2,
    GenerationReportV3,
    ObjectiveV2,
    ObjectiveV3,
    PolicyV2,
    PolicyV3,
    ProposalRecordV3,
    RepairInputV2,
    RepairInputV3,
    RepairResultV2,
    RepairResultV3,
    ReplacementCandidateV2,
    ReplacementCandidateV3,
    RevisionObjectV2,
    RevisionObjectV3,
    SearchLedgerV3,
    make_objective,
    promote_input_v3,
)

__all__ = [
    "BudgetsV2",
    "ProposalRecordV3",
    "GenerationReportV3",
    "RepairInputV3",
    "RepairResultV3",
    "PolicyV3",
    "ObjectiveV3",
    "ReplacementCandidateV3",
    "RevisionObjectV3",
    "SearchLedgerV3",
    "promote_input_v3",
    "repair_neural_round",
    "bounded_freeze_checkpoint",
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
    if name == "bounded_freeze_checkpoint":
        from .pipeline import bounded_freeze_checkpoint

        return bounded_freeze_checkpoint
    if name == "repair_neural_round":
        from .pipeline import repair_neural_round

        return repair_neural_round
    if name == "bounded_freeze_neural_round":
        from .pipeline import bounded_freeze_neural_round

        return bounded_freeze_neural_round
    if name in {"prepare_repair", "repair_alignment"}:
        from . import api

        return getattr(api, name)
    raise AttributeError(name)
