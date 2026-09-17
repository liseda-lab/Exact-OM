"""Adopt verified G0 probes without replacing their original identities or costs."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from exact.core.entities.configs.yaml_io import load_yaml_mapping
from exact.experiments.harness import hash_payload
from exact.experiments.recovery import ArtifactStore
from exact.experiments.runtime import _code_identity
from exact.llm.ledger import RequestLedger
from exact.ontology.versions import ontology_execution_identity
from exact.utils.fitted_artifacts import freeze_json
from exact.utils.provenance import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
AUDIT_FILE = "exact/impl/trainer/audit_io.py"

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


def _ledger_usage(previous: Path) -> dict:
    """Read charges without opening a writable ledger or exposing prompts."""
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
    return actual


def _verify_usage(previous: Path, expected: dict) -> None:
    if not isinstance(expected, dict) or set(expected) != set(USAGE_KEYS):
        raise ValueError("Saved cumulative hosted usage is missing")
    actual = _ledger_usage(previous)
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
        expected_copy = expected_usage if reuse_hosted else dict.fromkeys(USAGE_KEYS, 0)
        if any(
            not math.isclose(usage[key], expected_copy[key], rel_tol=0, abs_tol=1e-12)
            for key in USAGE_KEYS
        ):
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


def _source_at_revision(revision: str, relative: str) -> bytes:
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Repair requires the original full Git revision")
    return subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=REPOSITORY)


def _audit_repair_compatibility(original: dict, current: dict, revision: str) -> None:
    """Prove every extraction file except the output-writer body is unchanged."""
    old, new = original["files"], current["files"]
    changed = {name for name in old.keys() | new.keys() if old.get(name) != new.get(name)}
    if changed != {AUDIT_FILE}:
        raise ValueError(f"Output repair has unrelated implementation changes: {sorted(changed)}")
    previous = _source_at_revision(revision, AUDIT_FILE)
    if hashlib.sha256(previous).hexdigest() != old[AUDIT_FILE]:
        raise ValueError("Original audit source does not match checkpoint implementation")

    def outside_writer(source: bytes) -> str:
        tree = ast.parse(source)
        methods = [
            method
            for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == "AuditIOMixin"
            for method in item.body
            if isinstance(method, ast.FunctionDef) and method.name == "_write_source_decisions"
        ]
        if len(methods) != 1:
            raise ValueError("Expected exactly one audit writer method")
        methods[0].body = [ast.Pass()]
        return ast.dump(tree, include_attributes=False)

    if outside_writer(previous) != outside_writer((REPOSITORY / AUDIT_FILE).read_bytes()):
        raise ValueError("Output repair changes code outside the audit writer body")


def _hosted_configuration(saved_run: Path, expected: dict, *, cap: int) -> dict:
    config = dict(load_yaml_mapping(saved_run / "_inputs/resolved.config.yaml"))
    normalized = deepcopy(expected)
    for parent, field in (("data", "train_candidates"), ("refs", "train")):
        old = config.get("data", {})
        new = normalized.get("data", {})
        if parent == "refs":
            old, new = old.get("refs", {}), new.get("refs", {})
        if old.get(field) != new.get(field):
            if not old.get(field) or not new.get(field):
                raise ValueError("Hosted probe training binding changed")
            if sha256_file(Path(old[field])) != sha256_file(Path(new[field])):
                raise ValueError("Hosted probe training bytes changed")
            new[field] = old[field]
    scorers = [
        item
        for item in config.get("pipeline", [])
        if item.get("name") == "PairAdaptiveSemanticScorer"
    ]
    if (
        config.get("run", {}).get("source_cap") != cap
        or config.get("run", {}).get("seed") != 17
        or config.get("data", {}).get("execution_mode") != "global_alignment"
        or not scorers
        or any(
            item.get("params", {}).get("use_llm") is not True
            or item.get("params", {}).get("generate_llm_rationales") is not False
            for item in scorers
        )
        or hash_payload(config) != hash_payload(normalized)
    ):
        raise ValueError("Hosted probe configuration or rationale policy changed")
    return config


def _repair_checkpoint(record: dict) -> tuple[ArtifactStore, dict]:
    store = ArtifactStore(Path(record["source_root"]))
    if (
        sha256_file(store.root / "run/experiment_manifest.json")
        != record["original_manifest_sha256"]
    ):
        raise ValueError("Original repair run manifest changed")
    path = (
        store.directory / "checkpoints" / record["artifact_id"] / f"{record['sequence']:08d}.json"
    )
    if sha256_file(path) != record["checkpoint_sha256"]:
        raise ValueError("Original repair checkpoint manifest changed")
    checkpoint = store.latest_checkpoint(record["artifact_id"])
    if checkpoint is None or checkpoint["sequence"] != record["sequence"]:
        raise ValueError("Declared repair checkpoint or its bytes are unavailable")
    cursor = checkpoint["cursor"]
    if (
        not isinstance(cursor.get("dataset_rows"), int)
        or cursor["dataset_rows"] <= 0
        or cursor.get("next_pair") != cursor["dataset_rows"]
        or len(checkpoint["completed_ids"]) != cursor["dataset_rows"]
    ):
        raise ValueError("Output repair requires completed pair inference")
    additional = [
        name
        for name in checkpoint["outputs"]
        if name.startswith("checkpoints/inference_additional_models_") and name.endswith(".json")
    ]
    if len(additional) != 1:
        raise ValueError("Output repair requires one completed selector checkpoint")
    item = checkpoint["outputs"][additional[0]]
    selector = json.loads((store.root / item["path"]).read_text())
    if (
        selector.get("complete") is not True
        or selector.get("candidate_records_count") != cursor["dataset_rows"]
    ):
        raise ValueError("Output repair selector checkpoint is incomplete")
    if not {*DATASET_FILES, additional[0] + "l.zst"}.issubset(checkpoint["outputs"]):
        raise ValueError("Output repair lacks its frozen dataset or selector rows")
    return store, checkpoint


def seed_repaired_checkpoint(metadata: dict, worker_output: Path) -> dict:
    """Restore original checkpoint bytes after prepare, before continuation detection.

    The caller declares extraction/evaluation repair through the ordinary repair plan.
    This does not publish a completed artifact or rename the original checkpoint ID.
    """
    record = metadata["checkpoint_repair"]
    store, checkpoint = _repair_checkpoint(record)
    runtime_path = worker_output / "recovery-runtime.json"
    runtime = json.loads(runtime_path.read_text())
    original, current = checkpoint["identity"], runtime["identity"]
    omit = {"artifact_id", "implementation"}
    if {k: v for k, v in original.items() if k not in omit} != {
        k: v for k, v in current.items() if k not in omit
    }:
        raise ValueError("Repaired extraction inputs, parameters or dependencies changed")
    if current["implementation"] != _code_identity(REPOSITORY, evaluation=False):
        raise ValueError("Current repair runtime identity differs from executable source")
    _audit_repair_compatibility(
        original["implementation"], current["implementation"], record["original_revision"]
    )
    allowed = {"checkpoints", "dataset", "fitting", "explanations"}
    if any(
        Path(name).parts[0] not in allowed and name not in {"timings.json", "source_decisions.json"}
        for name in checkpoint["outputs"]
    ):
        raise ValueError("Repair checkpoint contains outputs outside inference state")
    restored = store.restore_checkpoint(checkpoint, worker_output)
    evidence = {
        "original_identity": original,
        "current_identity": current,
        "original_checkpoint": record,
        "restored_outputs": {name: checkpoint["outputs"][name]["sha256"] for name in restored},
        "scope": "completed inference retained; output writer and evaluation rerun",
        "new_inference_measurement": False,
    }
    freeze_json(worker_output / "checkpoint-repair.json", evidence)
    return evidence


def adopt_failed_validation(
    previous: Path,
    output: Path,
    *,
    campaign_sha256: str,
    expected_configs: dict[str, dict],
    materialize: bool = True,
) -> tuple[dict[str, dict], float, dict]:
    """Adopt G0's unchanged probes and declare the completed-inference output repair."""
    previous, output = previous.resolve(), output.resolve()
    if output == previous or output.is_relative_to(previous) or previous.is_relative_to(output):
        raise ValueError("Resume output must be separate from the saved validation root")
    if set(expected_configs) != {"cold64", "warm64", "fit64", "hosted20", "global300"}:
        raise ValueError("Failed validation adoption requires exactly five expected configurations")
    report_path = previous / "report.json"
    report = json.loads(report_path.read_text())
    if (
        report.get("status") != "failed"
        or report.get("campaign_sha256") != campaign_sha256
        or not str(report.get("reason", "")).startswith("RuntimeError: global300:")
    ):
        raise ValueError("Only this campaign's global300 output failure can be repaired")
    elapsed = _positive(report.get("elapsed_seconds"), "prior cumulative elapsed")
    origin = Path(report["resume"]["from"]).resolve()
    origin_report_path = origin / "report.json"
    origin_report = json.loads(origin_report_path.read_text())
    if (
        report["resume"].get("previous_elapsed_seconds") != origin_report["elapsed_seconds"]
        or report["resume"].get("previous_hosted_usage") != origin_report["hosted_usage"]
    ):
        raise ValueError("Imported historical charges or elapsed evidence changed")
    inherited, _ = adopt_completed_probes(
        origin,
        output,
        campaign_sha256=campaign_sha256,
        expected_configs={name: expected_configs[name] for name in ("cold64", "warm64", "fit64")},
        materialize=False,
    )
    ledger_usage = _ledger_usage(previous)
    usage = report["hosted_usage"]
    expected_usage = {
        key: ledger_usage[key] + origin_report["hosted_usage"][key] for key in USAGE_KEYS
    }
    if (
        any(
            not math.isclose(usage[key], expected_usage[key], rel_tol=0, abs_tol=1e-12)
            for key in USAGE_KEYS
        )
        or usage["unknown"]
        or usage["unpriced_attempts"]
    ):
        raise ValueError("Cumulative hosted charges differ from current and historical ledgers")
    rows = {}
    for name in expected_configs:
        matches = [row for row in report["stages"] if row.get("id") == name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one saved {name} measurement")
        row = matches[0]
        measurement_path = previous / f"{name}.measurement.json"
        if row != json.loads(measurement_path.read_text()):
            raise ValueError("Saved report and stage measurement differ")
        if name in inherited:
            evidence = row.get("adoption_evidence", {})
            if evidence.get("previous_report_sha256") != sha256_file(origin_report_path) or {
                k: v for k, v in row.items() if k != "adoption_evidence"
            } != {k: v for k, v in inherited[name].items() if k != "adoption_evidence"}:
                raise ValueError("Imported completed measurement or original evidence changed")
        else:
            run = previous / name / "run"
            config = _hosted_configuration(
                run, expected_configs[name], cap=20 if name == "hosted20" else 300
            )
            manifest = json.loads((run / "experiment_manifest.json").read_text())
            worker = json.loads((run / "validation-worker.json").read_text())
            complete = name == "hosted20"
            if (
                Path(row["output_dir"]).resolve() != run
                or Path(row["manifest"]).resolve() != run / "experiment_manifest.json"
                or row["worker_measurement"] != worker
                or worker.get("return_code") != (0 if complete else 1)
                or row.get("status") != ("complete" if complete else "failed")
                or manifest.get("status") != row["status"]
                or manifest.get("experiment_config_hash") != campaign_sha256
                or manifest.get("resolved_config_hash") != hash_payload(config)
                or not elapsed
                >= _positive(row["wall_seconds"], "probe wall")
                >= _positive(worker["wall_seconds"], "worker wall")
            ):
                raise ValueError("Hosted stage does not bind the original worker and configuration")
            if complete:
                _, artifact = _verified_extraction(run)
                if artifact["identity"]["dependencies"].get(
                    "ontology_artifacts"
                ) != ontology_execution_identity(
                    str(config.get("dataset", {}).get("reasoner", "asserted"))
                ):
                    raise ValueError("Hosted probe native implementation changed")
            else:
                stderr = (run / "experiment.stderr.log").read_text()
                if (
                    "audit_io.py" not in stderr
                    or "InvalidIndexError: Reindexing only valid with uniquely valued Index objects"
                    not in stderr
                ):
                    raise ValueError("Saved failure is not the audited output-column collision")
                identity = json.loads((run / "recovery-runtime.json").read_text())["identity"]
                store = ArtifactStore(run.parent)
                checkpoint = store.latest_checkpoint(identity["artifact_id"])
                if checkpoint is None or checkpoint["identity"] != identity:
                    raise ValueError("Failed stage has no verified matching inference checkpoint")
                checkpoint_path = (
                    store.directory
                    / "checkpoints"
                    / identity["artifact_id"]
                    / f"{checkpoint['sequence']:08d}.json"
                )
                record = {
                    "source_root": str(store.root),
                    "artifact_id": identity["artifact_id"],
                    "sequence": checkpoint["sequence"],
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "original_revision": manifest["git"]["commit"],
                    "original_manifest_sha256": sha256_file(run / "experiment_manifest.json"),
                    "original_wall_seconds": row["wall_seconds"],
                }
                _repair_checkpoint(record)
                _audit_repair_compatibility(
                    identity["implementation"],
                    _code_identity(REPOSITORY, evaluation=False),
                    record["original_revision"],
                )
                repair = {
                    "resume_from": str(store.root),
                    "checkpoint_repair": record,
                    "repair_record": {
                        "kind": "g0_completed_inference_output_repair",
                        "affected_stages": ["extraction"],
                        "scientific_choices_unchanged": True,
                        "reporting_labels_exposed": True,
                        "exposure_scope": "development evaluator smoke; no private/final labels",
                        "original_checkpoint": record,
                    },
                }
                continue
        adopted = deepcopy(row)
        adopted.update(imported=True, new_worker_calls=0, new_usage=dict.fromkeys(USAGE_KEYS, 0))
        adopted["measurement_usage"] = row.get("measurement_usage", row["new_usage"])
        adopted["adoption_evidence"] = {
            **row.get("adoption_evidence", {}),
            "previous_report": str(report_path),
            "previous_report_sha256": sha256_file(report_path),
            "original_measurement_sha256": sha256_file(measurement_path),
            "prior_elapsed_seconds": elapsed,
            "prior_hosted_usage": usage,
            "prior_ledger_usage": ledger_usage,
            "prior_limits": report.get("limits"),
            "hosted_cache_reused": True,
        }
        rows[name] = adopted
    measured_usage = {
        key: sum(row["new_usage"][key] for row in report["stages"]) for key in USAGE_KEYS
    }
    if any(
        not math.isclose(measured_usage[key], ledger_usage[key], rel_tol=0, abs_tol=1e-12)
        for key in USAGE_KEYS
    ):
        raise ValueError("Stage request measurements differ from copied ledger usage")
    incremental_wall = sum(
        row["wall_seconds"] for row in report["stages"] if row.get("new_worker_calls") == 1
    )
    if elapsed < origin_report["elapsed_seconds"] + incremental_wall:
        raise ValueError("Prior elapsed time omits completed or failed worker charges")
    if materialize:
        backups = _copy_shared(previous, output, ledger_usage, reuse_hosted=True)
        for name, row in rows.items():
            row["adoption_evidence"]["shared_backups"] = backups
            freeze_json(output / f"{name}.adoption.json", row)
        freeze_json(output / "global300.repair.json", repair)
    return rows, elapsed, repair
