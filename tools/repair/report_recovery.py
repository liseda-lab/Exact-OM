"""Finalize a completed, pinned training run after its report transport failed."""

from exact.repair.records import canonical_hash

# The coherent-endpoint trainer completed optimization before returning an
# oversized report. No other predecessor or changed repair dependency is admitted.
PREDECESSOR = "ef7f9cbded0e8bf9429889dec4f8455ef919de901758027eb5fd3fdf67e3f3ea"
UNCHANGED_DEPENDENCIES = "96a4a7f0858ba9428cbeb082fa993cbb6aca7b00cb36bb1bf60611dfb61713a1"


def completed_report_recovery(saved, options, warm_start_hash, dependencies):
    """Validate completed training without changing weights, selection, or costs."""
    expected = canonical_hash((options, warm_start_hash, PREDECESSOR))
    epochs = options["epochs"]
    history = saved.get("history", [])
    selected = next((row for row in history if row.get("epoch") == saved.get("best_epoch")), {})
    if not (
        dependencies == UNCHANGED_DEPENDENCIES
        and options.get("revision") == "v3"
        and saved.get("schema") == "exact-repair/training-state/v3"
        and saved.get("recovery_revision") == "exact-phase-resume/v3.1"
        and saved.get("identity") == expected
        and saved.get("next_epoch") == epochs
        and saved.get("next_offset") == 0
        and saved.get("phase") == "acquisition"
        and not saved.get("pending_acquisition")
        and not saved.get("development_progress")
        and [row.get("epoch") for row in history] == list(range(1, epochs + 1))
        and saved.get("best_state")
        and saved.get("best_criterion") is not None
        and selected.get("selection_criterion") is not None
        and tuple(selected["selection_criterion"]) == tuple(saved["best_criterion"])
    ):
        raise ValueError("Report recovery dependencies or completed selection are incompatible")
    return dict(
        migration="completed-report-transport/v1",
        source_identity=expected,
        source_implementation=PREDECESSOR,
        selected_epoch=saved["best_epoch"],
        reused="completed model/optimizer/RNG, acquisition evidence and development selection",
        invalidated="none",
        additional_optimization_epochs=0,
        budgets_reset=False,
    )
