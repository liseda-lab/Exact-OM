"""Adopt verified G0 probes without replacing their original identities or costs."""

from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from exact.core.entities.configs.yaml_io import load_yaml_mapping
from exact.experiments.harness import hash_payload
from exact.experiments.recovery import ArtifactStore
from exact.llm.ledger import RequestLedger
from exact.ontology.versions import ontology_execution_identity
from exact.utils.fitted_artifacts import freeze_json
from exact.utils.provenance import sha256_file

DATASET_FILES = tuple(
    f"dataset/{name}"
    for name in (
        "dataset.csv",
        "dataset.meta.json",
        "candidate_pool_manifest.json",
        "candidate_pool_sample_manifest.json",
    )
)
USAGE_KEYS = (
    "attempts",
    "prompt_tokens",
    "completion_tokens",
    "reported_cost_usd",
    "unknown",
    "unpriced_attempts",
)


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric {label}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid {label}")
    return float(value)


def _verified_extraction(saved_run: Path) -> tuple[ArtifactStore, dict]:
    runtime = json.loads((saved_run / "recovery-runtime.json").read_text())
    store = ArtifactStore(saved_run.parent)
    payload = store.verify(runtime["identity"]["artifact_id"])
    if payload["identity"] != runtime["identity"] or payload["identity"]["stage"] != "extraction":
        raise ValueError("Saved extraction identity differs from its runtime record")
    required = {
        *DATASET_FILES,
        "alignment/maps_global.tsv",
        "source_decisions.json",
        "timings.json",
    }
    if not required.issubset(payload["outputs"]):
        raise ValueError("Saved extraction lacks required durable outputs")
    for name, item in payload["outputs"].items():
        path = saved_run / name
        if not path.resolve().is_relative_to(saved_run.resolve()):
            raise ValueError("Saved output escapes its run directory")
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"Saved output differs from verified extraction: {name}")
    return store, payload


def _configuration(saved_run: Path) -> dict:
    config = dict(load_yaml_mapping(saved_run / "_inputs/resolved.config.yaml"))
    if (
        config.get("run", {}).get("source_cap") != 64
        or config.get("data", {}).get("execution_mode") != "global_alignment"
        or config.get("data", {}).get("refs") != {}
        or config.get("data", {}).get("train_candidates") is not None
        or config.get("dataset", {}).get("verbalization_mode") != "deterministic"
    ):
        raise ValueError("Only the unchanged label-free cold64 configuration can be adopted")
    return config


def _verify_usage(previous: Path, expected: dict) -> None:
    """Check historical charges without opening a writable ledger or exposing prompts."""
    if not isinstance(expected, dict) or set(expected) != set(USAGE_KEYS):
        raise ValueError("Saved cumulative hosted usage is missing")
    path = previous / "shared/openrouter/requests.sqlite3"
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT a.state,a.usage FROM attempts a JOIN requests r USING(request_id)"
        ).fetchall()
    actual: dict[str, int | float] = dict.fromkeys(USAGE_KEYS, 0)
    for state, raw in rows:
        usage = json.loads(raw or "{}")
        actual["attempts"] += 1
        actual["unknown"] += state in {"unknown", "sent"}
        for field in ("prompt_tokens", "completion_tokens"):
            actual[field] += int(usage.get(field) or 0)
        if usage.get("cost") is None:
            actual["unpriced_attempts"] += 1
        else:
            actual["reported_cost_usd"] += float(usage["cost"])
    if any(
        (
            not math.isclose(actual[key], expected[key], rel_tol=0, abs_tol=1e-12)
            if key == "reported_cost_usd"
            else actual[key] != expected[key]
        )
        for key in USAGE_KEYS
    ):
        raise ValueError("Saved request ledger differs from cumulative reported usage")


def _copy_shared(
    previous: Path, output: Path, expected_usage: dict, *, reuse_hosted: bool = True
) -> dict:
    _verify_usage(previous, expected_usage)
    destination = output / "shared"
    if destination.exists():
        raise ValueError("Resume destination already has shared caches; refusing to reset usage")
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".cold-adoption-", dir=output))
    copied = {}
    try:
        relatives = ["embeddings/vectors.sqlite3"]
        if reuse_hosted:
            relatives.append("openrouter/requests.sqlite3")
        for relative in relatives:
            source_path = previous / "shared" / relative
            if not source_path.is_file():
                raise ValueError(f"Missing saved shared cache: {relative}")
            target_path = temporary / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(source_path.resolve().as_uri() + "?mode=ro", uri=True) as source:
                with sqlite3.connect(target_path) as target:
                    source.backup(target)
                    if target.execute("PRAGMA quick_check").fetchone() != ("ok",):
                        raise ValueError(f"Invalid shared SQLite cache: {relative}")
        roles = RequestLedger(temporary / "openrouter").summary()["roles"]
        usage = {key: sum(item[key] for item in roles.values()) for key in USAGE_KEYS}
        if usage != (expected_usage if reuse_hosted else dict.fromkeys(USAGE_KEYS, 0)):
            raise ValueError("Copied request ledger has unexpected cumulative usage")
        for path in temporary.rglob("*.sqlite3"):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            copied[path.relative_to(temporary).as_posix()] = sha256_file(path)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return copied


def adopt_cold_probe(
    previous: Path,
    output: Path,
    *,
    campaign_sha256: str,
    limits: dict,
    expected_config: dict,
    materialize: bool = True,
) -> tuple[dict, float]:
    """Return the original cold measurement and cumulative prior elapsed time.

    A preparation-only call verifies evidence but never writes to either root.
    Adoption retains the original implementation identity; it is not new inference.
    """
    previous, output = previous.resolve(), output.resolve()
    if output == previous or output.is_relative_to(previous) or previous.is_relative_to(output):
        raise ValueError("Resume output must be separate from the saved validation root")
    report = json.loads((previous / "report.json").read_text())
    if (
        report.get("status") != "failed"
        or report.get("reason") != "ValueError: A completed artifact must have durable outputs"
        or report.get("stages") != []
    ):
        raise ValueError("Only the saved cold64 durable-output publication failure is supported")
    if report.get("campaign_sha256") != campaign_sha256 or report.get("limits") != limits:
        raise ValueError("Resume campaign and all limits must equal the saved validation plan")
    saved_run = previous / "cold64/run"
    config = _configuration(saved_run)
    if hash_payload(config) != hash_payload(expected_config):
        raise ValueError("Resume configuration differs from the saved cold probe")
    manifest_path = saved_run / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("experiment_id") != "G0"
        or manifest.get("source_cap") != 64
        or manifest.get("resolved_config_hash") != hash_payload(config)
        or manifest.get("experiment_config_hash") != campaign_sha256
    ):
        raise ValueError("Original experiment manifest does not bind this cold configuration")
    worker_path = saved_run / "validation-worker.json"
    worker = json.loads(worker_path.read_text())
    if type(worker.get("return_code")) is not int or worker["return_code"] != 0:
        raise ValueError("Saved cold worker did not finish successfully")
    wall = _positive(worker.get("wall_seconds"), "original cold duration")
    elapsed = _positive(report.get("elapsed_seconds"), "prior cumulative elapsed")
    if elapsed < wall:
        raise ValueError("Prior cumulative elapsed cannot be smaller than the cold worker duration")
    store, extraction = _verified_extraction(saved_run)
    usage = report.get("hosted_usage")
    if not isinstance(usage, dict) or set(usage) != set(USAGE_KEYS):
        raise ValueError("Saved cumulative hosted usage is missing")
    row = {
        "id": "cold64",
        "status": "complete",
        "imported": True,
        "wall_seconds": wall,
        "output_dir": str(saved_run),
        "new_worker_calls": 0,
        "new_usage": {key: 0 for key in USAGE_KEYS},
        "source_cap": 64,
        "hosted": False,
        "evaluate": False,
        "manifest": str(manifest_path),
        "worker_measurement": worker,
        "recovery": {
            "artifact_id": extraction["identity"]["artifact_id"],
            "parents": extraction["identity"]["parents"],
            "store_root": str(store.root),
            "original_implementation": extraction["identity"]["implementation"],
            "original_identity": extraction["identity"],
        },
        "adoption_evidence": {
            "previous_report": str(previous / "report.json"),
            "previous_report_sha256": sha256_file(previous / "report.json"),
            "original_manifest_sha256": sha256_file(manifest_path),
            "original_worker_sha256": sha256_file(worker_path),
            "verified_outputs": {
                name: item["sha256"] for name, item in extraction["outputs"].items()
            },
            "prior_elapsed_seconds": elapsed,
            "prior_hosted_usage": usage,
        },
    }
    if materialize:
        row["adoption_evidence"]["shared_backups"] = _copy_shared(previous, output, usage)
        freeze_json(output / "cold64.adoption.json", row)
    return row, elapsed


def _completed_configuration(saved_run: Path, name: str, expected: dict) -> dict:
    config = dict(load_yaml_mapping(saved_run / "_inputs/resolved.config.yaml"))
    data = config.get("data", {})
    fit = name == "fit64"
    scorers = [
        item
        for item in config.get("pipeline", [])
        if item.get("name") == "PairAdaptiveSemanticScorer"
    ]
    if (
        config.get("run", {}).get("source_cap") != 64
        or config.get("run", {}).get("seed") != 17
        or data.get("execution_mode") != "global_alignment"
        or data.get("reference_role") is not None
        or set(data.get("refs", {})) != ({"train"} if fit else set())
        or bool(data.get("train_candidates")) != fit
        or config.get("dataset", {}).get("verbalization_mode") != "deterministic"
        or not scorers
        or any(
            item.get("params", {}).get("use_llm") is not False
            or item.get("params", {}).get("generate_llm_rationales") is not False
            for item in scorers
        )
    ):
        raise ValueError(
            "Only completed offline 64-source probes without rationales can be adopted"
        )
    normalized = deepcopy(expected)
    if fit:
        for parent, field in (("data", "train_candidates"), ("refs", "train")):
            old = data if parent == "data" else data["refs"]
            new = (
                normalized.get("data", {})
                if parent == "data"
                else normalized.get("data", {}).get("refs", {})
            )
            if not old.get(field) or not new.get(field):
                raise ValueError("Fitting probe training binding is missing")
            if sha256_file(Path(old[field])) != sha256_file(Path(new[field])):
                raise ValueError("Fitting probe training bytes changed")
            new[field] = old[field]
    if hash_payload(config) != hash_payload(normalized):
        raise ValueError("Completed probe configuration differs from the requested probe")
    return config


def adopt_completed_probes(
    previous: Path,
    output: Path,
    *,
    campaign_sha256: str,
    expected_configs: dict[str, dict],
    materialize: bool = True,
) -> tuple[dict[str, dict], float]:
    """Adopt cold/warm/fit measurements from a budget-blocked G0 attempt.

    Hosted evidence is excluded: the new no-rationale probe needs a cold hosted
    ledger. Historical charges remain explicit; only embeddings are copied.
    Changed resource limits are the caller's separately recorded amendment.
    """
    names = ("cold64", "warm64", "fit64")
    if set(expected_configs) != set(names):
        raise ValueError("Exactly cold64, warm64 and fit64 expected configurations are required")
    previous, output = previous.resolve(), output.resolve()
    if output == previous or output.is_relative_to(previous) or previous.is_relative_to(output):
        raise ValueError("Resume output must be separate from the saved validation root")
    report_path = previous / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "blocked_budget" or report.get("campaign_sha256") != campaign_sha256:
        raise ValueError("Only the same campaign's budget-blocked completed probes can be adopted")
    elapsed = _positive(report.get("elapsed_seconds"), "prior cumulative elapsed")
    usage = report.get("hosted_usage")
    _verify_usage(previous, usage)
    if usage["unknown"] or usage["unpriced_attempts"]:
        raise ValueError("Historical unresolved hosted charges require explicit reconciliation")
    rows = {}
    for name in names:
        matches = [row for row in report.get("stages", []) if row.get("id") == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one completed {name} measurement")
        row = matches[0]
        saved_run = previous / name / "run"
        measurement_path = previous / f"{name}.measurement.json"
        if (
            row != json.loads(measurement_path.read_text())
            or row.get("status") != "complete"
            or row.get("hosted") is not False
            or row.get("evaluate") is not False
            or row.get("source_cap") != 64
            or row.get("new_worker_calls") != 1
            or row.get("new_usage") != dict.fromkeys(USAGE_KEYS, 0)
            or Path(row.get("output_dir", "")).resolve() != saved_run
        ):
            raise ValueError(f"Saved {name} does not bind a complete offline worker measurement")
        config = _completed_configuration(saved_run, name, expected_configs[name])
        manifest_path = saved_run / "experiment_manifest.json"
        if Path(row.get("manifest", "")).resolve() != manifest_path:
            raise ValueError("Completed probe measurement names a different manifest")
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status") != "complete"
            or manifest.get("experiment_id") != "G0"
            or manifest.get("source_cap") != 64
            or manifest.get("resolved_config_hash") != hash_payload(config)
            or manifest.get("experiment_config_hash") != campaign_sha256
        ):
            raise ValueError("Original experiment manifest does not bind the completed probe")
        worker_path = saved_run / "validation-worker.json"
        worker = json.loads(worker_path.read_text())
        if (
            worker != row.get("worker_measurement")
            or type(worker.get("return_code")) is not int
            or worker["return_code"] != 0
        ):
            raise ValueError("Saved probe worker did not finish successfully")
        if name == "warm64" and (
            type(worker.get("dataset_cache_hits")) is not int
            or worker["dataset_cache_hits"] < 1
            or worker.get("scorer_encoded_texts") != 0
        ):
            raise ValueError("Warm probe lacks measured dataset and encoder cache reuse")
        if (
            not elapsed
            >= _positive(row.get("wall_seconds"), "original probe duration")
            >= _positive(worker.get("wall_seconds"), "original worker duration")
        ):
            raise ValueError("Saved probe durations are inconsistent")
        store, extraction = _verified_extraction(saved_run)
        identity = extraction["identity"]
        artifact = identity["artifact_id"]
        if any(
            item.get("recovery", {}).get("artifacts", {}).get("extraction") != artifact
            for item in (row, manifest)
        ):
            raise ValueError("Completed probe recovery identity differs from its saved extraction")
        current_native = ontology_execution_identity(
            str(config.get("dataset", {}).get("reasoner", "asserted"))
        )
        if (
            not all(current_native.values())
            or identity.get("dependencies", {}).get("ontology_artifacts") != current_native
        ):
            raise ValueError(
                "Completed probe native implementation differs from the installed stack"
            )
        if name == "fit64" and not any(
            path.startswith("fitting/") and path.endswith("/training_units.json")
            for path in extraction["outputs"]
        ):
            raise ValueError("Completed fitting probe lacks verified training-unit artifacts")
        adopted = deepcopy(row)
        adopted.update(imported=True, new_worker_calls=0, new_usage=dict.fromkeys(USAGE_KEYS, 0))
        adopted["recovery"].update(
            original_identity=identity,
            original_implementation=identity["implementation"],
            store_root=str(store.root),
        )
        adopted["adoption_evidence"] = {
            "previous_report": str(report_path),
            "previous_report_sha256": sha256_file(report_path),
            "original_measurement_sha256": sha256_file(measurement_path),
            "original_manifest_sha256": sha256_file(manifest_path),
            "original_worker_sha256": sha256_file(worker_path),
            "verified_outputs": {
                path: item["sha256"] for path, item in extraction["outputs"].items()
            },
            "prior_elapsed_seconds": elapsed,
            "prior_hosted_usage": usage,
            "prior_limits": report.get("limits"),
            "current_ontology_artifacts": current_native,
            "hosted_cache_reused": False,
        }
        rows[name] = adopted
    if sum(row["wall_seconds"] for row in rows.values()) > elapsed:
        raise ValueError("Completed probe times exceed prior cumulative elapsed")
    if materialize:
        backups = _copy_shared(previous, output, usage, reuse_hosted=False)
        for name, row in rows.items():
            row["adoption_evidence"]["shared_backups"] = backups
            freeze_json(output / f"{name}.adoption.json", row)
    return rows, elapsed


def seed_warm_dataset(saved_run: Path, worker_output: Path) -> dict[str, str]:
    """Seed only verified dataset files, after recovery preparation, for fresh scoring."""
    saved_run, worker_output = saved_run.resolve(), worker_output.resolve()
    if saved_run == worker_output or worker_output.is_relative_to(saved_run):
        raise ValueError("Warm worker must use a distinct output directory")
    old = _configuration(saved_run)
    if hash_payload(old) != hash_payload(_configuration(worker_output)):
        raise ValueError("Warm worker configuration differs from the saved cold probe")
    store, extraction = _verified_extraction(saved_run)
    copied = {}
    for name in DATASET_FILES:
        item = extraction["outputs"][name]
        target = worker_output / name
        if not target.resolve().is_relative_to(worker_output):
            raise ValueError("Warm dataset path escapes worker output")
        if target.exists() and sha256_file(target) != item["sha256"]:
            raise ValueError(f"Warm dataset already contains different bytes: {name}")
        copied[name] = item["sha256"]
    for name in DATASET_FILES:
        target = worker_output / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            shutil.copyfile(store.root / extraction["outputs"][name]["path"], temporary)
            temporary.replace(target)
    return copied
