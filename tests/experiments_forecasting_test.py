"""Admission forecasts must not extrapolate ontology loading by pair count."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from exact.experiments.forecasting import derive_e09_forecast


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _phase(root, name, dataset, alignment, inference=None, postprocess=0, pairs=6000):
    complete = inference is not None
    output = root / name
    state = "complete" if complete else "interrupted"
    total = dataset + alignment + postprocess + 5
    stages = [
        {"stage": "Dataset", "seconds": dataset, "cache_status": "fresh"},
        {"stage": "Alignment", "seconds": alignment, "cache_status": "fresh"},
        {"stage": "Total", "seconds": total, "cache_status": "fresh"},
    ]
    if complete:
        stages.extend(
            [
                {
                    "stage": "Alignment.Inference",
                    "seconds": inference,
                    "cache_status": "fresh",
                    "work_done": pairs,
                    "work_total": pairs,
                    "unit": "examples",
                },
                {"stage": "Postprocess", "seconds": postprocess, "cache_status": "fresh"},
                {"stage": "Postprocess.Evaluation", "seconds": 0.5, "cache_status": "fresh"},
            ]
        )
    ledger = {
        "schema_version": 1,
        "sessions": [
            {
                "command": "align",
                "run_id": name,
                "dataset_signature": "dataset",
                "config_fingerprint": name,
                "ended_at": "done" if complete else None,
                "stages": stages,
            }
        ],
    }
    worker = {
        "wall_seconds": total + 5,
        "peak_rss_bytes": 2 * 1024**3,
        "peak_cuda_reserved_bytes": 1024**3,
        "return_code": 0 if complete else 130,
        "scorer_encoded_texts": 0,
        "scorer_encoder_batches": 0,
    }
    phase = {
        "id": name,
        "status": "passed",
        "execution_status": state,
        "new_worker_calls": 1,
        "processed_pairs": pairs,
        "dataset_rows": pairs if complete else 6000,
        "wall_seconds": total + 15,
        "output_dir": str(output),
        "worker_measurement": worker,
        "worker_measurement_binding": _write(output / "validation-worker.json", worker),
        "manifest": _write(
            output / "experiment_manifest.json", {"status": state, "timing_ledger": ledger}
        ),
        "operational_timing_binding": _write(output / "timings.json", ledger),
        "new_usage": {
            "unknown": 0,
            "unpriced_attempts": 0,
            "attempts": 3,
            "billable_tokens": 101,
            "reported_cost_usd": 0.25,
        },
    }
    if not complete:
        phase["interrupted_binding"] = _write(
            output / "interrupted.json",
            {
                "completed_pairs": pairs,
                "status": "interrupted",
            },
        )
    return phase


@pytest.fixture
def receipts(tmp_path):
    common = {
        "status": "passed",
        "source_cap": 300,
        "generate_rationales": False,
        "no_private_test_references": True,
        "runtime": {"python": "frozen"},
    }
    phases = [
        _phase(tmp_path, "cold300", 2000, 1000, 980, 10),
        _phase(tmp_path, "warm300", 1900, 900, 880, 10),
        _phase(tmp_path, "ancestor-prefix", 1800, 300, pairs=512),
        _phase(tmp_path, "sibling-prefix", 1800, 320, pairs=512),
        {"id": "completed-replay", "status": "passed", "new_worker_calls": 0},
    ]
    control = _phase(tmp_path, "D0-current-control", 500, 1200, 1170, 15, pairs=5842)
    return (
        {**common, "phases": phases},
        {**common, "measurement": control, "worker_measurement": control["worker_measurement"]},
    )


def _revise_timing(phase, edit, *, manifest_too=True):
    path = Path(phase["operational_timing_binding"]["path"])
    ledger = json.loads(path.read_text())
    edit(ledger)
    phase["operational_timing_binding"] = _write(path, ledger)
    if manifest_too:
        path = Path(phase["manifest"]["path"])
        manifest = json.loads(path.read_text())
        manifest["timing_ledger"] = ledger
        phase["manifest"] = _write(path, manifest)


def test_separates_setup_scoring_and_real_output_cost(receipts):
    estimate, reason = derive_e09_forecast(*receipts)
    assert estimate["preparation_seconds"] == 4 * 2020
    assert estimate["seconds_per_unit"] == 320 / 512
    assert estimate["evaluation_seconds"] == pytest.approx(4 * 45 * 6000 / 5842)
    assert estimate["cold_seconds"] == estimate["fitting_seconds"] == 0
    assert estimate["safety_factor"] == 1.5
    assert estimate["peak_ram_gb"] == 3
    assert estimate["peak_vram_gb"] == 1.5
    assert (estimate["requests"], estimate["tokens"], estimate["projected_usd"]) == (18, 606, 1.5)
    expected = 1.5 * (4 * 2020 + 24000 * 320 / 512 + 4 * 45 * 6000 / 5842)
    assert reason["projected_seconds"] == pytest.approx(expected)
    assert reason["legacy_projected_seconds"] > reason["projected_seconds"]


def test_large_setup_is_charged_once_per_arm_not_per_pair(receipts):
    before, reason = derive_e09_forecast(*receipts)
    cold = receipts[0]["phases"][0]
    extra = 10000

    def add_setup(ledger):
        for row in ledger["sessions"][0]["stages"]:
            if row["stage"] in ("Dataset", "Total"):
                row["seconds"] += extra

    _revise_timing(cold, add_setup)
    cold["wall_seconds"] += extra
    cold["worker_measurement"]["wall_seconds"] += extra
    cold["worker_measurement_binding"] = _write(
        Path(cold["worker_measurement_binding"]["path"]), cold["worker_measurement"]
    )
    after, revised = derive_e09_forecast(*receipts)
    assert after["seconds_per_unit"] == before["seconds_per_unit"]
    assert revised["projected_seconds"] - reason["projected_seconds"] == pytest.approx(
        4 * 1.5 * extra
    )


def test_interrupted_missing_inference_uses_positive_enclosing_bound(receipts):
    _, reason = derive_e09_forecast(*receipts)
    row = reason["phase_breakdown"]["sibling-prefix"]
    assert row["scoring_seconds"] == 320
    assert "upper bound" in row["scoring_basis"]
    assert row["output_seconds_at_6000_pairs"] is None


def test_output_binding_supports_original_complete_receipts(receipts):
    cold = receipts[0]["phases"][0]
    cold["output_bindings"] = [cold.pop("operational_timing_binding")]
    derive_e09_forecast(*receipts)


@pytest.mark.parametrize(
    "artifact",
    ["manifest", "operational_timing_binding", "worker_measurement_binding", "interrupted_binding"],
)
def test_tampered_bound_files_rejected(receipts, artifact):
    phase = receipts[0]["phases"][2]
    path = Path(phase[artifact]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        derive_e09_forecast(*receipts)


def test_worker_snapshot_cannot_override_bound_memory(receipts):
    receipts[0]["phases"][0]["worker_measurement"]["peak_rss_bytes"] = 1
    with pytest.raises(ValueError, match="Worker measurement"):
        derive_e09_forecast(*receipts)


@pytest.mark.parametrize(
    "stage",
    [
        "Dataset",
        "Alignment",
        "Total",
        "Alignment.Inference",
        "Postprocess",
        "Postprocess.Evaluation",
    ],
)
def test_missing_timings_fail_closed(receipts, stage):
    cold = receipts[0]["phases"][0]
    _revise_timing(
        cold,
        lambda ledger: ledger["sessions"][0].update(
            stages=[r for r in ledger["sessions"][0]["stages"] if r["stage"] != stage]
        ),
    )
    with pytest.raises(ValueError):
        derive_e09_forecast(*receipts)


def test_prefix_without_alignment_never_becomes_free(receipts):
    phase = receipts[0]["phases"][2]
    _revise_timing(
        phase,
        lambda ledger: ledger["sessions"][0].update(
            stages=[r for r in ledger["sessions"][0]["stages"] if r["stage"] != "Alignment"]
        ),
    )
    with pytest.raises(ValueError, match="Alignment"):
        derive_e09_forecast(*receipts)


def test_foreign_or_stale_timing_not_substituted(receipts):
    cold, warm = receipts[0]["phases"][:2]
    cold["operational_timing_binding"] = warm["operational_timing_binding"]
    with pytest.raises(ValueError, match="path does not match"):
        derive_e09_forecast(*receipts)


def test_rebound_timing_must_match_manifest(receipts):
    cold = receipts[0]["phases"][0]
    _revise_timing(
        cold, lambda ledger: ledger["sessions"][0].update(run_id="foreign"), manifest_too=False
    )
    with pytest.raises(ValueError, match="differs from bound manifest"):
        derive_e09_forecast(*receipts)


def test_prefix_count_must_match_durable_interruption(receipts):
    receipts[0]["phases"][2]["processed_pairs"] = 6000
    with pytest.raises(ValueError, match="Durable prefix count"):
        derive_e09_forecast(*receipts)


@pytest.mark.parametrize(
    "mutation", ["sessions", "duplicate", "negative", "nonfinite", "overlap", "reused"]
)
def test_invalid_or_ambiguous_stage_accounting_rejected(receipts, mutation):
    def edit(ledger):
        rows = ledger["sessions"][0]["stages"]
        if mutation == "sessions":
            ledger["sessions"].append(copy.deepcopy(ledger["sessions"][0]))
        elif mutation == "duplicate":
            rows.append(copy.deepcopy(rows[0]))
        elif mutation == "reused":
            next(r for r in rows if r["stage"] == "Alignment.Inference")[
                "cache_status"
            ] = "cache_hit"
        else:
            rows[0]["seconds"] = {"negative": -1, "nonfinite": float("nan"), "overlap": 100000}[
                mutation
            ]

    _revise_timing(receipts[0]["phases"][0], edit)
    with pytest.raises(ValueError):
        derive_e09_forecast(*receipts)


def test_quality_fields_do_not_affect_estimate(receipts):
    expected = derive_e09_forecast(*receipts)
    receipts[0]["quality"] = {"private_test": "must never inspect"}
    receipts[1]["measurement"]["f1"] = float("nan")
    assert derive_e09_forecast(*receipts) == expected


def test_module_loads_independently_of_numerical_package(receipts):
    path = Path(__file__).resolve().parents[1] / "exact/experiments/forecasting.py"
    spec = importlib.util.spec_from_file_location("pinned_e09_forecast", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.derive_e09_forecast(*receipts) == derive_e09_forecast(*receipts)


def test_missing_binding_cannot_fall_back_to_unbound_timing(receipts):
    del receipts[0]["phases"][0]["operational_timing_binding"]
    with pytest.raises(ValueError, match="Missing or conflicting timing binding"):
        derive_e09_forecast(*receipts)


def test_prefix_reuse_cannot_supply_marginal_rate(receipts):
    def edit(ledger):
        for row in ledger["sessions"][0]["stages"]:
            if row["stage"] == "Alignment":
                row["cache_status"] = "cache_hit"

    _revise_timing(receipts[0]["phases"][2], edit)
    with pytest.raises(ValueError, match="Fresh measured Alignment"):
        derive_e09_forecast(*receipts)


def test_duplicate_control_identity_cannot_replace_treatment(receipts):
    receipts[1]["measurement"]["id"] = "ancestor-prefix"
    with pytest.raises(ValueError, match="Distinct declared D0"):
        derive_e09_forecast(*receipts)
