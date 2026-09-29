"""Complete matched E24 admission and immutable scientific-boundary regressions."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import queue_e24 as queue


def cells():
    from tests.e08_queue_test import cell

    values = []
    for case in queue.CASES:
        for arm in queue.ARMS:
            item = cell(arm)
            item.task_id = case + "-global_alignment"
            c = item.resolved_config
            c["selector"]["runtime_enabled"] = False
            c["matching"]["calibration"] = {"threshold_mode": "fixed"}
            c["matching"]["channels"] = {
                "diff": {
                    "controlled_perturbations": True,
                    "formulation": {"diff_off": "off"}.get(arm, arm),
                }
            }
            values.append(item)
    return values


def measurements(tmp_path):
    values = []
    for key in queue.KEYS:
        case = key.split("--")[0]
        output = tmp_path / key
        (output / "dataset").mkdir(parents=True)
        (output / "dataset/candidate_pool_sample_manifest.json").write_text(
            json.dumps(
                {
                    "gold_free_summary": {"candidate_pairs": 5900},
                    "per_kind": {"class": {"pool_sha256": case}},
                }
            )
        )
        values.append(
            dict(
                name=key,
                case_id=case,
                output_dir=str(output),
                status="passed",
                execution_status="complete",
                prefix=False,
                processed_pairs=5900,
                dataset_rows=5900,
                new_usage={"attempts": 0},
                wall_seconds=200 if case == "D1" else 100,
                worker_measurement={"peak_rss_bytes": 1, "peak_cuda_reserved_bytes": 1},
            )
        )
    return values


def test_both_case_populations_and_all_treatments_are_required():
    values = cells()
    before = copy.deepcopy(values)
    queue.validate_cells(values)
    assert values == before
    values.pop()
    with pytest.raises(ValueError, match="eight"):
        queue.validate_cells(values)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "case",
        "private",
        "selector",
        "gate",
        "formulation",
        "threshold",
        "supervision",
        "rationales",
    ],
)
def test_scientific_and_role_guards(change):
    values = cells()
    c = values[-1].resolved_config
    if change == "duplicate":
        values[-1] = values[0]
    elif change == "case":
        values[-1].task_id = "H0-global_alignment"
    elif change == "private":
        c["data"]["refs"]["test"] = "sealed.tsv"
    elif change == "selector":
        c["selector"]["runtime_enabled"] = True
    elif change == "gate":
        c["llm"]["experiment"]["gate"]["mode"] = "analytic"
    elif change == "formulation":
        c["matching"]["channels"]["diff"]["formulation"] = "asymmetric"
    elif change == "threshold":
        c["matching"]["threshold"] = 0.8
    elif change == "supervision":
        c["supervision"]["components"]["accept"] = "supervised"
    else:
        values[-1].generate_rationales = True
    with pytest.raises(ValueError):
        queue.validate_cells(values)


def test_matching_is_within_case_not_across_distinct_cases(tmp_path):
    values = measurements(tmp_path)
    pools = queue.matched_measurements(values)
    assert set(pools) == {"D0", "D1"}
    path = Path(values[-1]["output_dir"]) / "dataset/candidate_pool_sample_manifest.json"
    doc = json.loads(path.read_text())
    doc["per_kind"]["class"]["pool_sha256"] = "changed"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="populations"):
        queue.matched_measurements(values)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "wrong_case", "partial", "prefix", "failed", "charged"]
)
def test_incomplete_measurements_cannot_admit_family(tmp_path, change):
    values = measurements(tmp_path)
    if change == "missing":
        values.pop()
    elif change == "duplicate":
        values[-1] = values[0]
    elif change == "wrong_case":
        values[-1]["case_id"] = "D1"
    elif change == "partial":
        values[-1]["processed_pairs"] -= 1
    elif change == "prefix":
        values[-1]["prefix"] = True
    elif change == "failed":
        values[-1]["execution_status"] = "failed"
    else:
        values[-1]["new_usage"]["attempts"] = 1
    with pytest.raises(ValueError):
        queue.matched_measurements(values)


def test_forecast_counts_setup_once_per_cell_and_inventory_once_per_case(tmp_path):
    from tools.resume_cached_batch import load_helpers

    rows = measurements(tmp_path)
    h = load_helpers(queue.DATA / "e24-qualification-01")
    receipt = tmp_path / "measurement.json"
    receipt.write_text("{}")
    proof = queue.family_forecast(rows, receipt, h)
    assert proof["comparison_seconds"] == (4 * 200 + 4 * 100) * 1.5
    assert proof["inventory_seconds"] == (200 + 100) * 1.5
    assert proof["estimate"]["seconds_per_unit"] == 0
    assert proof["whole_family_seconds"] == 2250


def test_ready_lock_only_changes_operational_readiness():
    from tests.e08_queue_test import scientific_lock

    original = scientific_lock()
    target = original["steps"][1]
    target.update(id="E24", case="D1", additional_cases=["D0"])
    target["arms"] = [dict(id=a, overlay={"marker": a}) for a in queue.ARMS]
    target["readiness"] = {
        a: dict(screen={"status": "blocked_input_resolution"}, confirm={"status": "planned"})
        for a in queue.ARMS
    }
    before = copy.deepcopy(original)
    result = queue.ready_lock(original, {"measured": True}, "commit")
    assert original == before
    for key in ("arms", "case", "additional_cases", "selection", "design", "inherits"):
        assert result["steps"][1][key] == before["steps"][1][key]


def test_live_old_numeric_step_prevents_dispatch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        queue.subprocess, "check_output", lambda *a, **k: "14372.0\n14372.20\n14372.18\n"
    )
    with pytest.raises(ValueError, match="remains live"):
        queue.no_previous_owner(tmp_path, "14372.20")


def test_live_descendant_prevents_dispatch_even_if_slurm_owner_is_gone(monkeypatch, tmp_path):
    import psutil

    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **k: "14372.0\n14372.20\n")
    monkeypatch.setattr(
        queue, "read", lambda _: {"runs": [{"status_path": "/old/run/status.json"}]}
    )
    proc = SimpleNamespace(
        pid=123, info={"name": "python", "cmdline": ["python", "/old/run/worker.py"]}
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *a: [proc])
    with pytest.raises(ValueError, match="descendant"):
        queue.no_previous_owner(tmp_path, "14372.20")


def test_cached_probe_cannot_import_other_case_receipt(tmp_path):
    from tools import qualify_cached_family as probe

    root = tmp_path / "queue"
    directory = root / "qualification" / "D1--normalised"
    directory.mkdir(parents=True)
    supervisor = tmp_path / "hourly-supervisor-01"
    supervisor.mkdir()
    (supervisor / "registry.json").write_text('{"pause_paths": []}')
    (root / "reuse-plan.json").write_text("{}")
    config = root / "config.yaml"
    config.write_text(
        'data: {refs: {valid: development.tsv}}\nllm: {experiment: {gate: {mode: "off"}}}\n'
    )
    (directory / "measurement.json").write_text(json.dumps({"bindings": [], "case_id": "D0"}))
    with pytest.raises(ValueError, match="case changed"):
        probe.run_probe(
            code_root=tmp_path,
            script=root / "continue.py",
            config=config,
            name="D1--normalised",
            directory=directory,
            shared=tmp_path / "unused",
            campaign=None,
            forecast_seconds=1,
            case_id="D1",
        )


def test_plan_has_four_treatment_rows_but_materialization_requires_eight_cells(
    monkeypatch, tmp_path
):
    from exact.experiments import campaign, harness

    monkeypatch.setattr(
        campaign,
        "campaign_plan",
        lambda *a, **k: {
            "budget_errors": [],
            "rows": [
                dict(step="E24", arm=arm, status="screen_ready", issues=[]) for arm in queue.ARMS
            ],
        },
    )
    source = SimpleNamespace(config=SimpleNamespace(experiment_id="E24", depends_on=[]))
    monkeypatch.setattr(
        campaign, "materialize_campaign", lambda *a, **k: SimpleNamespace(sources=[source])
    )
    monkeypatch.setattr(campaign, "validate_comparison_cells", lambda *a: None)
    monkeypatch.setattr(harness, "inherited_selection_overlay", lambda *a: {})
    materialized = cells()
    monkeypatch.setattr(harness, "build_cells", lambda *a, **k: materialized)
    assert len(queue.cells_for(None, tmp_path, {})) == 8
    materialized.pop()
    with pytest.raises(ValueError, match="eight"):
        queue.cells_for(None, tmp_path, {})
