"""Exercise the optional N0 diagnostic without changing the actual E07 selection."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from exact.experiments import harness
from exact.experiments.fitting_recipes import fitting_arms
from exact.experiments.nil_listwise import DIAGNOSTIC_ID, FIXED_JUDGE
from exact.experiments.staged_selection import (
    PrerequisiteUnavailable,
    materialize_selected_judge,
)
from tests.label_policy_test import declaration
from tests.staged_selection_test import bound, complete, selected  # noqa: F401


@pytest.fixture
def diagnostic(tmp_path, bound):  # noqa: F811
    base, suite = bound
    judge = {
        "enabled": True,
        "decision": {"mode": "listwise", "evidence": "structured_packet"},
        "gate": {"mode": "source_top_fraction", "quantile_fraction": 1.0},
        "fusion_weight": "source_first",
    }
    manifests = [
        complete(
            tmp_path,
            "E07",
            "facts_binary",
            harness.deep_merge(
                base,
                {
                    "llm": {
                        "experiment": {
                            **judge,
                            "decision": {"mode": "binary", "evidence": "structured_packet"},
                            "fusion_weight": "beta_u",
                        }
                    }
                },
            ),
        ),
        complete(
            tmp_path,
            "E07",
            "facts_listwise",
            harness.deep_merge(base, {"llm": {"experiment": judge}}),
        ),
    ]
    arms = [
        deepcopy(arm)
        for arm in fitting_arms()["E04"]
        if arm["id"] in {"nil_heuristic", "listwise_none"}
    ]
    arms[0]["role"] = "baseline"
    source = declaration(tmp_path, DIAGNOSTIC_ID, arms, base)
    source.config.selection.decisions = []
    source.config.arms[1].role = "diagnostic"
    source.config.arms[1].deployable = False
    source.config.screen.tasks[0].id = "N0-global_alignment"
    source.config.frozen_constants.update(
        selected_judge={"producer": "E07"},
        fixed_listwise_diagnostic=deepcopy(FIXED_JUDGE),
        evaluation_diagnostics={"N0-global_alignment": {"label_semantics": "benchmark_pool"}},
    )
    return source, suite, manifests, selected("E07", "facts_binary")


def test_fixed_diagnostic_preserves_winner_and_control_without_selecting(tmp_path, diagnostic):
    source, suite, manifests, selections = diagnostic
    previous_selection = deepcopy(selections)
    previous_control = deepcopy(source.config.arms[0].overlay)
    result = materialize_selected_judge(*diagnostic)
    binding = result.config.frozen_constants["staged_selection_binding"]["path"]
    record = json.loads(Path(binding).read_text())
    assert record["actual_E07_winner"]["arm"] == "facts_binary"
    assert record["fixed_comparator"]["arm"] == "facts_listwise"
    assert record["selection_eligible"] is False and record["training"] == []
    assert result.config.arms[0].overlay == previous_control
    assert result.config.arms[1].overlay["llm"]["experiment"]["decision"]["mode"] == "listwise"
    assert result.config.arms[1].overlay["llm"]["experiment"]["fusion_weight"] == "source_first"
    assert selections == previous_selection
    assert materialize_selected_judge(*diagnostic).raw_hash() == result.raw_hash()
    rows = [
        {
            "experiment_id": DIAGNOSTIC_ID,
            "arm_id": name,
            "task_id": "N0-global_alignment",
            "seed": 17,
            "status": "complete",
            "F1": value,
        }
        for name, value in (("nil_heuristic", 0.5), ("listwise_none", 0.9))
    ]
    report = harness.select_experiment(result, rows)
    assert report["status"] == "complete" and report["selection_eligible"] is False
    assert report["decisions"] == [] and report["combined_selected_overlay"] == {}
    assert (
        harness.selected_experiment_overlays(
            {"experiments": {DIAGNOSTIC_ID: report}}, [DIAGNOSTIC_ID]
        )
        == {}
    )


@pytest.mark.parametrize(
    "change",
    [
        "winner",
        "comparator_missing",
        "changed_config",
        "scope",
        "selectable",
        "labels",
        "pool",
        "mode",
    ],
)
def test_fixed_diagnostic_rejects_unapproved_or_unbound_changes(diagnostic, change):
    source, _, manifests, selections = diagnostic
    if change == "winner":
        selections["E07"]["decisions"][0]["selected_arm"] = "facts_listwise"
    elif change == "comparator_missing":
        manifests.pop()
    elif change == "changed_config":
        path = (
            Path(manifests[1]["fingerprint_payload"]["output_dir"]) / "_inputs/resolved.config.yaml"
        )
        path.write_text(path.read_text() + "\nnew_field: true\n")
    elif change == "scope":
        source.config.experiment_id = "E04-listwise"
    elif change == "selectable":
        source.config.arms[1].deployable = True
    elif change == "labels":
        source.config.frozen_constants["evaluation_diagnostics"]["N0-global_alignment"][
            "label_semantics"
        ] = "natural"
    elif change == "pool":
        source.config.arms[1].overlay["matching"]["threshold"] = 0.123
    elif change == "mode":
        source.config.screen.tasks[0].id = "D0-global_alignment"
    with pytest.raises(ValueError):
        materialize_selected_judge(*diagnostic)


def test_original_conditional_comparison_still_rejects_binary_winner(diagnostic):
    source, _, _, _ = diagnostic
    source.config.experiment_id = "E04-listwise"
    del source.config.frozen_constants["fixed_listwise_diagnostic"]
    with pytest.raises(PrerequisiteUnavailable, match="winner is binary"):
        materialize_selected_judge(*diagnostic)
