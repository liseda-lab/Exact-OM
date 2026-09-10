"""Run-stage artifact binding and final endpoint inference retain frozen semantics."""

import json
from dataclasses import replace

import pytest
import yaml

from exact.experiments import harness, label_policy
from exact.experiments.paper_metrics import SourceConfusion, SourceEvaluation
from tests.campaign_v2_test import _fake_model_run, _ready_suite


def test_stage_binds_artifacts_before_execution_and_current_result_validation(
    tmp_path, monkeypatch
):
    suite = _ready_suite(tmp_path)
    original = suite.sources[0]
    original.config.frozen_constants["label_policy_followup"] = {"producer": "E22"}
    calls = []

    def materialize(source, current, manifests):
        calls.append((source, current, manifests))
        arms = [
            arm.model_copy(
                update={
                    "overlay": harness.deep_merge(arm.overlay, {"matching": {"threshold": 0.5}})
                }
            )
            for arm in source.config.arms
        ]
        config = source.config.model_copy(update={"arms": arms})
        path = source.path.parent / "resolved-policy.yaml"
        path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
        return replace(source, config=config, path=path)

    monkeypatch.setattr(label_policy, "materialize_followup", materialize)
    monkeypatch.setattr(harness, "_run_subprocess", _fake_model_run)
    monkeypatch.setattr(harness, "build_dataset_inventory", lambda *args, **kwargs: None)
    selected = harness.run_stage(
        suite,
        stage="screen",
        output_root=tmp_path / "results",
        jobs=1,
        resume=False,
        workdir=tmp_path,
    )
    assert len(calls) == 1
    assert json.loads(selected.read_text())["experiments"]["E06"]["status"] == "selected"
    root = tmp_path / "results" / suite.suite_id / "screen"
    assert len(json.loads((root / "current-result-set.json").read_text())["cells"]) == 2
    progress = json.loads((root / "progress.json").read_text())
    assert progress["declarations"]["E06"]["path"].endswith("resolved-policy.yaml")
    with pytest.raises(ValueError, match="artifact-dependent plan"):
        harness._materialize_campaign_evidence(original, suite, [], {}, {}, plan_only=True)


def test_global_primary_bootstrap_excludes_local_tasks_and_requires_global_pairs(
    tmp_path, monkeypatch
):
    suite = _ready_suite(tmp_path)
    source = suite.sources[0]
    global_task = source.config.screen.tasks[0]
    global_task.overlay["data"]["execution_mode"] = "global_alignment"
    local_task = global_task.model_copy(deep=True, update={"id": "local"})
    local_task.overlay["data"]["execution_mode"] = "local_ranking"
    source.config.screen.tasks.append(local_task)
    records = []
    for arm in source.config.arms:
        for task in (global_task, local_task):
            records.append(
                {
                    "experiment_id": "E06",
                    "arm_id": arm.id,
                    "task_id": task.id,
                    "seed": 17,
                    "status": "complete",
                    "reference_completeness": "complete",
                    "candidate_pool_fingerprint": "same",
                }
            )

    def evaluate(record, cache):
        assert record["task_id"] == global_task.id
        count = (
            SourceConfusion(1, 0, 0) if record["arm_id"] == "replay" else SourceConfusion(0, 1, 1)
        )
        return SourceEvaluation("reference", record["arm_id"], {"s": count})

    monkeypatch.setattr(harness, "_overall_evaluation", evaluate)
    rows = harness._paired_bootstrap_rows(suite, records, stage="screen", resamples=100, seed=17)
    assert rows[0]["status"] == "complete"
    assert rows[0]["n_tasks"] == 1 and rows[0]["delta"] == 1
    missing = [
        row
        for row in records
        if not (row["arm_id"] == "replay" and row["task_id"] == global_task.id)
    ]
    assert (
        harness._paired_bootstrap_rows(suite, missing, stage="screen", resamples=100, seed=17)[0][
            "reason_code"
        ]
        == "unequal_or_missing_cells"
    )


def test_final_gain_claim_uses_interval_multiplicity_and_power_status(tmp_path):
    suite = _ready_suite(tmp_path)
    suite.sources[0].config.design.practical_effect = 0.003
    base = {
        "experiment_id": "E06",
        "endpoint_scope": "overall",
        "status": "complete",
        "inference_status": "confirmatory",
        "precision_claim_status": "eligible",
        "ci_low": 0.004,
        "p_value_adjusted": 0.04,
    }
    rows = [
        base.copy(),
        {**base, "ci_low": 0.001},
        {**base, "p_value_adjusted": 0.06},
        {**base, "inference_status": "descriptive"},
    ]
    harness._final_gain_claims(rows, suite, stage="confirm")
    assert [row["primary_gain_claim"] for row in rows] == [
        "supported",
        "not_supported",
        "not_supported",
        "inconclusive",
    ]


def test_inapplicable_nil_metrics_are_absent_from_selection_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "extract_evaluation_metrics", lambda _: {"F1": 0.8})
    path = tmp_path / "diagnostics/nil_metrics.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluation_only": True,
                "metrics": {
                    "nil_aware": {"F1": 0.5},
                    "natural_nil": {"F1": 0},
                    "non_nil_MRR": 0.9,
                    "applicability": {"nil_aware": True, "natural_nil": False, "non_nil_MRR": True},
                },
            }
        )
    )
    assert harness.cell_metrics(tmp_path) == {
        "F1": 0.8,
        "nil.nil_aware.F1": 0.5,
        "nil.non_nil_MRR": 0.9,
    }
