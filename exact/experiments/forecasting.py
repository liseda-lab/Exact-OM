"""Bound E09 runtime from authenticated production qualification stage timings.

This module deliberately uses only the standard library: operational launchers
can load a pinned copy without changing their frozen numerical-code import path.
It reads timing/provenance fields only, never alignment quality or label values.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

_PHASES = ("cold300", "warm300", "ancestor-prefix", "sibling-prefix")
_ARMS, _PAIRS, _SAFETY = 4, 6000, 1.5


def _number(value, label, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(f"Invalid or missing {label}: {value!r}")
    return value


def _path(value, root):
    if not isinstance(value, str) or not value:
        raise ValueError("Missing evidence path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _read(binding, root, expected):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError(f"Missing exact path/SHA256 binding for {expected}")
    path = _path(binding["path"], root)
    if path != expected:
        raise ValueError(f"Evidence path does not match phase: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError(f"Evidence SHA256 mismatch: {path}")
    return json.loads(raw)


def _stages(phase, root):
    """Require one authenticated session; retries cannot silently lose costs."""
    name = phase["id"]
    output = _path(phase.get("output_dir"), root)
    manifest = _read(phase.get("manifest"), root, output / "experiment_manifest.json")
    worker = _read(phase.get("worker_measurement_binding"), root, output / "validation-worker.json")
    if worker != phase.get("worker_measurement"):
        raise ValueError(f"Worker measurement differs from bound evidence: {name}")
    if manifest.get("status") != phase["execution_status"]:
        raise ValueError(f"Manifest execution status mismatch: {name}")
    timing_path = output / "timings.json"
    bindings = [
        b for b in phase.get("output_bindings", []) if _path(b.get("path"), root) == timing_path
    ]
    if phase.get("operational_timing_binding") is not None:
        bindings.append(phase["operational_timing_binding"])
    if not bindings or any(b != bindings[0] for b in bindings):
        raise ValueError(f"Missing or conflicting timing binding: {name}")
    ledger = _read(bindings[0], root, timing_path)
    if ledger != manifest.get("timing_ledger"):
        raise ValueError(f"Timing ledger differs from bound manifest: {name}")
    sessions = ledger.get("sessions", [])
    if ledger.get("schema_version") != 1 or len(sessions) != 1:
        raise ValueError(f"Exactly one measured timing session required: {name}")
    session = sessions[0]
    if session.get("command") != "align" or not all(
        session.get(k) for k in ("run_id", "config_fingerprint", "dataset_signature")
    ):
        raise ValueError(f"Missing timing session identity: {name}")
    rows = session.get("stages", [])
    stages = {row["stage"]: row for row in rows}
    if len(stages) != len(rows):
        raise ValueError(f"Duplicate timing stages: {name}")
    for stage, row in stages.items():
        _number(row.get("seconds"), name + " " + stage)
    values = {
        key: _number(stages.get(key, {}).get("seconds"), name + " " + key, positive=True)
        for key in ("Dataset", "Alignment", "Total")
    }
    alignment = values["Alignment"]
    if stages["Alignment"].get("cache_status") != "fresh":
        raise ValueError(f"Fresh measured Alignment stage required: {name}")
    pairs = phase["processed_pairs"]
    complete = phase["execution_status"] == "complete"
    if complete:
        inference = stages.get("Alignment.Inference", {})
        if (
            inference.get("cache_status") != "fresh"
            or inference.get("work_done") != pairs
            or inference.get("work_total") != phase.get("dataset_rows")
            or inference.get("unit") != "examples"
            or pairs != phase.get("dataset_rows")
            or not session.get("ended_at")
        ):
            raise ValueError(f"Incomplete or reused inference timing: {name}")
        scoring = _number(inference.get("seconds"), name + " inference", positive=True)
        postprocess = _number(
            stages.get("Postprocess", {}).get("seconds"), name + " outputs", positive=True
        )
        evaluation = _number(
            stages.get("Postprocess.Evaluation", {}).get("seconds"), name + " evaluation"
        )
        if scoring > alignment or evaluation > postprocess:
            raise ValueError(f"Nested timing exceeds parent stage: {name}")
        basis = "Alignment.Inference"
    else:
        interrupted = _read(phase.get("interrupted_binding"), root, output / "interrupted.json")
        if interrupted != {"completed_pairs": pairs, "status": "interrupted"}:
            raise ValueError(f"Durable prefix count differs from bound evidence: {name}")
        # Interrupted runs need not flush Alignment.Inference or scorer counters.
        # The enclosing Alignment interval is an explicit upper bound, not zero.
        if "Postprocess" in stages:
            raise ValueError(f"Unexpected completed output stage for prefix: {name}")
        scoring, postprocess, evaluation = alignment, 0.0, 0.0
        basis = "Alignment (inclusive upper bound for interrupted prefix)"
    wall = _number(phase.get("wall_seconds"), name + " phase wall", positive=True)
    worker_wall = _number(worker.get("wall_seconds"), name + " worker wall", positive=True)
    if (
        values["Dataset"] + alignment + postprocess > values["Total"] + 1e-6
        or values["Total"] > worker_wall + 1e-6
        or worker_wall > wall + 1e-6
        or worker.get("return_code") != (0 if complete else 130)
    ):
        raise ValueError(f"Inconsistent stage/worker/phase wall accounting: {name}")
    return {
        "processed_pairs": pairs,
        "phase_wall_seconds": wall,
        "dataset_seconds": values["Dataset"],
        "setup_seconds": wall - alignment - postprocess,
        "scoring_seconds": scoring,
        "seconds_per_pair": scoring / pairs,
        "scoring_basis": basis,
        "output_seconds": alignment - scoring + postprocess,
        "evaluation_seconds": evaluation,
        "output_seconds_at_6000_pairs": (
            (alignment - scoring + postprocess) * _PAIRS / pairs if complete else None
        ),
        "timing_binding": bindings[0],
        "manifest_binding": phase["manifest"],
        "worker_binding": phase["worker_measurement_binding"],
        "timing_run_id": session["run_id"],
    }


def derive_e09_forecast(qualification, d0, *, evidence_root=None):
    """Return WorkEstimate fields and auditable reasoning for four 6,000-pair arms.

    Callers must authenticate the enclosing qualification/control receipts and
    retain the existing runtime/configuration admission checks. Every timing,
    worker, manifest and prefix-count file used here is independently hash-bound.
    Missing or inconsistent stage evidence fails closed; incomplete prefixes use
    their measured enclosing Alignment duration as a conservative scoring bound.
    """
    root = Path(evidence_root or ".").resolve()
    for receipt in (qualification, d0):
        if (
            receipt.get("status") != "passed"
            or receipt.get("source_cap") != 300
            or receipt.get("generate_rationales") is not False
            or receipt.get("no_private_test_references") is not True
        ):
            raise ValueError("Passed unchanged 300-source qualification/control required")
    if qualification.get("runtime") != d0.get("runtime"):
        raise ValueError("Qualification/control runtime mismatch")
    raw_phases = qualification.get("phases", [])
    phases = {p["id"]: p for p in raw_phases}
    if len(raw_phases) != 5 or set(phases) != {*_PHASES, "completed-replay"}:
        raise ValueError("Complete declared production qualification phases required")
    replay = phases["completed-replay"]
    if replay.get("status") != "passed" or replay.get("new_worker_calls") != 0:
        raise ValueError("Completed CAS replay must execute no new scoring workers")
    control = d0["measurement"]
    if control.get("id") != "D0-current-control":
        raise ValueError("Distinct declared D0 control measurement required")
    if d0.get("worker_measurement") != control.get("worker_measurement"):
        raise ValueError("D0 worker measurement mismatch")
    measurements, breakdown, old_rates = [], {}, {}
    for phase in [*(phases[name] for name in _PHASES), control]:
        name = phase["id"]
        expected = "interrupted" if name.endswith("-prefix") else "complete"
        if phase.get("status") != "passed" or phase.get("execution_status") != expected:
            raise ValueError(f"Unexpected qualification execution state: {name}")
        pairs = _number(phase.get("processed_pairs"), name + " committed pairs", positive=True)
        if int(pairs) != pairs or pairs > _PAIRS or phase.get("new_worker_calls") != 1:
            raise ValueError(f"Invalid pair count or worker count: {name}")
        breakdown[name] = _stages(phase, root)
        measurements.append(phase["worker_measurement"])
        if name in _PHASES:
            old_rates[name] = phase["wall_seconds"] / pairs
    setup = max(row["setup_seconds"] for row in breakdown.values())
    rate = max(row["seconds_per_pair"] for row in breakdown.values())
    outputs = max(row["output_seconds_at_6000_pairs"] or 0 for row in breakdown.values())
    peak_ram = max(_number(m.get("peak_rss_bytes"), "RAM", positive=True) for m in measurements)
    peak_vram = max(
        _number(m.get("peak_cuda_reserved_bytes"), "VRAM", positive=True) for m in measurements
    )
    usage = phases["cold300"].get("new_usage", {})
    if usage.get("unknown") != 0 or usage.get("unpriced_attempts") != 0:
        raise ValueError("Cold template token and price accounting must be resolved")
    requests = _number(usage.get("attempts"), "cold template requests")
    tokens = _number(usage.get("billable_tokens"), "cold template tokens")
    if int(requests) != requests or int(tokens) != tokens:
        raise ValueError("Request/token counts must be integers")
    usd = _number(usage.get("reported_cost_usd"), "cold template USD")
    estimate = {
        "cold_seconds": 0,
        "units": _ARMS * _PAIRS,
        "seconds_per_unit": rate,
        "preparation_seconds": _ARMS * setup,
        "fitting_seconds": 0,
        "evaluation_seconds": _ARMS * outputs,
        "safety_factor": _SAFETY,
        "peak_ram_gb": _SAFETY * peak_ram / 1024**3,
        "peak_vram_gb": _SAFETY * peak_vram / 1024**3,
        "requests": math.ceil(_ARMS * _SAFETY * requests),
        "tokens": math.ceil(_ARMS * _SAFETY * tokens),
        "projected_usd": _ARMS * _SAFETY * usd,
    }
    legacy_seconds = _SAFETY * (
        _ARMS * phases["cold300"]["wall_seconds"]
        + _ARMS * _PAIRS * max(old_rates.values())
        + _ARMS * control["wall_seconds"]
    )
    seconds = _SAFETY * _ARMS * (setup + _PAIRS * rate + outputs)
    return estimate, {
        "method": "Maximum measured setup once per arm, 24,000 pairs at the slowest measured scoring rate, and maximum completed output/evaluation cost scaled to 6,000 pairs per arm; all multiplied by 1.5.",
        "phase_breakdown": breakdown,
        "maximum_planned_pairs_per_arm": _PAIRS,
        "components_before_safety_seconds": {
            "setup": _ARMS * setup,
            "scoring": _ARMS * _PAIRS * rate,
            "outputs_and_evaluation": _ARMS * outputs,
        },
        "projected_seconds": seconds,
        "projected_hours": seconds / 3600,
        "legacy_projected_seconds": legacy_seconds,
        "legacy_projected_hours": legacy_seconds / 3600,
        "phase_wall_seconds_per_committed_pair": old_rates,
        "observed_peak_rss_bytes": peak_ram,
        "observed_peak_cuda_reserved_bytes": peak_vram,
        "limits": [
            "Extrapolation, not a completed four-arm measurement; every arm receives the slowest observed treatment-prefix rate.",
            "Setup includes cold preparation, native loading and all unallocated phase overhead once per arm; verified cache reuse is not assumed to remove loading.",
            "Prefix Alignment is an inclusive scoring upper bound; missing inference records or zero scorer counters never imply free scoring.",
            "Output allowance includes fitting and Alignment tail plus Postprocess; actual D0 evaluation time is included, not the entire D0 run.",
            "Future allowances do not charge historical probe execution again. Memory peaks and cold template requests/tokens/USD keep the original 1.5 and 4*1.5 multipliers.",
            "Decision LLM and rationales remain off; original hosted template verbalization remains enabled. No quality fields or labels inform this forecast.",
        ],
    }
