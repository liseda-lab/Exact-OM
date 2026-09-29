#!/usr/bin/env python3
"""Migrate verified E19 evidence after the neutral-validation-only repair."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest()}


def verify(record):
    if bind(record["path"]) != record:
        raise ValueError("Recovery binding changed: " + record["path"])
    return Path(record["path"])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError("Immutable migration record changed: " + str(path))
        return
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(text)
    os.replace(temporary, path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def changed_identity(identity, old_code, new_code, parents=None):
    """Re-key a proved compatible artifact, retaining every scientific field."""
    from exact.experiments.recovery import stage_identity

    value = copy.deepcopy(identity)
    if value["stage"] == "extraction":
        if value["implementation"] != old_code:
            raise ValueError("Original artifact source is not the reviewed E19 source")
        value["implementation"] = new_code
    if parents is not None:
        value["parents"] = parents
    value.pop("artifact_id")
    value.pop("schema_version")
    return stage_identity(**value)


def outputs(store, manifest):
    return {name: store.root / item["path"] for name, item in manifest["outputs"].items()}


def migrate_control(parent, root, preflight, old_code, new_code, once, account):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.recovery import ArtifactStore

    name = "D0_E03--analytic_shipped"
    origin = parent / "qualification" / name
    target = root / "qualification" / name
    receipt = target / "measurement.json"
    if receipt.exists():
        return
    row = read(origin / "measurement.json")
    cell = SimpleNamespace(
        seed=17,
        source_cap=300,
        generate_rationales=False,
        split_role="development",
        resolved_config=load_yaml_mapping(verify(preflight["configs"][name])),
    )
    old_store, ids, manifest = once.verify_measurement(row, cell, account)
    store = ArtifactStore(target)
    artifacts = {}
    measured_alias = None
    for stage in ("inputs", "extraction", "evaluation"):
        original = old_store.verify(ids[stage])
        parents = None
        if stage == "extraction":
            parents = [artifacts["inputs"]]
        elif stage == "evaluation":
            parents = [artifacts["extraction"]]
        identity = changed_identity(original["identity"], old_code, new_code, parents)
        payloads = outputs(old_store, original)
        measure_name = "stats/execution_measurement.json"
        if stage == "extraction" and measure_name in payloads:
            measured_alias = read(payloads[measure_name])
            if measured_alias.get("artifact_id") != ids["extraction"]:
                raise ValueError("Original control execution measurement identity differs")
            measured_alias.update(
                artifact_id=identity["artifact_id"],
                origin_artifact_id=ids["extraction"],
                compatibility_migration=bind(root / "repair.json"),
            )
            payloads[measure_name] = (
                json.dumps(measured_alias, sort_keys=True, indent=2) + "\n"
            ).encode()
        store.publish(identity, payloads)
        artifacts[stage] = identity["artifact_id"]
    out = target / "run"
    old_out = Path(row["output_dir"])
    for stage in ("inputs", "extraction", "evaluation"):
        store.restore(artifacts[stage], out)
    for item in row["bindings"]:
        source = verify(item)
        relative = source.relative_to(old_out)
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.as_posix() != "stats/execution_measurement.json":
            shutil.copyfile(source, destination)
    if measured_alias is not None:
        manifest["execution_measurement"] = measured_alias
    origin_binding = bind(old_out / "experiment_manifest.json")
    manifest["recovery"]["artifacts"] = artifacts
    manifest["recovery"]["compatibility_migration"] = {
        "origin_manifest": origin_binding,
        "origin_artifacts": ids,
        "proof": bind(root / "repair.json"),
        "reason": "Only fitted-fusion precondition validation changed; analytic_shipped does not call it.",
        "new_numerical_execution": False,
    }
    runtime = read(out / "recovery-runtime.json")
    runtime.update(root=str(target), identity=store.verify(artifacts["extraction"])["identity"])
    # These are newly generated migration receipts; the original attempt is untouched.
    (out / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (out / "recovery-runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    attempt_id = manifest["recovery"]["attempt_id"]
    attempt = old_store.root / "attempts" / attempt_id / "attempt.json"
    write(target / "attempts" / attempt_id / "attempt.json", read(attempt))
    row.update(
        output_dir=str(out),
        bindings=[bind(out / Path(item["path"]).relative_to(old_out)) for item in row["bindings"]],
        compatibility_migration=manifest["recovery"]["compatibility_migration"],
    )
    once.verify_measurement(row, cell, account)
    write(receipt, row)


def migrate_prefit(parent, root, old_code, new_code):
    """Seal only raw training inputs/evidence, never a fitted artifact or prediction."""
    from exact.experiments.recovery import ArtifactStore

    name = "D0_E03--analytic_fitted"
    origin = parent / "qualification" / name
    target = root / "qualification" / name
    old_runtime = read(origin / "run/recovery-runtime.json")
    identity = changed_identity(old_runtime["identity"], old_code, new_code)
    store = ArtifactStore(target)
    if store.latest_checkpoint(identity["artifact_id"]) is not None:
        return
    for parent_id in identity["parents"]:
        store.import_artifact(origin, parent_id)
    fitting = origin / "run/fitting"
    files = {}
    for path in fitting.rglob("*"):
        if not path.is_file():
            continue
        if (
            path.suffix != ".json"
            or path.name in {"fusion.json", "selector.json", "graph.json"}
            or ".folds" in str(path)
        ):
            raise ValueError("Unexpected fitted artifact in failed prefit evidence: " + str(path))
        files[path.relative_to(origin / "run").as_posix()] = path
    tables = [p for p in files.values() if p.name == "training_scores.json"]
    if len(tables) != 1:
        raise ValueError("One complete raw training table is required")
    # Config, original source and source-group memberships are already immutable
    # bindings. Hash all consumed raw evidence; no old fitted output is accepted.
    records = {name: bind(path) for name, path in files.items()}
    population_path = origin / "run/dataset/candidate_pool_sample_manifest.json"
    population = read(population_path)
    files["dataset/candidate_pool_sample_manifest.json"] = population_path
    checkpoint = store.checkpoint(
        identity,
        completed_ids=[],
        cursor={"next_pair": 0, "dataset_rows": population["gold_free_summary"]["candidate_pairs"]},
        outputs=files,
        state={
            "boundary": "verified_raw_training_only",
            "inference_pairs_complete": 0,
            "repair": bind(root / "repair.json"),
        },
    )
    write(
        root / "prefit-import.json",
        {
            "original_runtime": bind(origin / "run/recovery-runtime.json"),
            "original_identity": old_runtime["identity"],
            "target_identity": identity,
            "raw_evidence": records,
            "checkpoint": checkpoint,
            "fitted_artifacts_imported": False,
        },
    )


def register_recovery_lineage(recipe, prepared):
    """Link the failed owner only after dispatcher registration of this live step."""
    supervisor = Path(recipe["supervisor"])
    own_step = "14372." + os.environ["SLURM_STEP_ID"]
    for _ in range(45):
        with (supervisor / "registry.json.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            registry = prepared.controls(supervisor, Path(recipe["root"]))
            current = next((r for r in registry["runs"] if r["id"] == recipe["id"]), None)
            if current is not None:
                if (
                    current["step_id"] != own_step
                    or current.get("dispatch_nonce") != recipe["dispatch_nonce"]
                ):
                    raise ValueError("Recovery registration does not match this live owner")
                parent = next(r for r in registry["runs"] if r["id"] == recipe["parent_run_id"])
                if parent.get("superseded_by") not in (None, recipe["id"]):
                    raise ValueError("The failed E19 owner already has another replacement")
                parent.update(
                    enabled=False,
                    superseded_by=recipe["id"],
                    reason="Precision-validation repair; verified control and raw training checkpoints retained, all original costs carried forward.",
                )
                parent.pop("pending_recovery", None)
                registry["remaining_work_status"] = "pending"
                prepared.write(supervisor / "registry.json", registry)
                return registry
        time.sleep(1)
    raise RuntimeError("Dispatcher has not verified this step; retain reservation and receipts")


def current_supervisor_step(recipe):
    """Resolve the current supervisor after a redeploy, never a stale prepared step."""
    path = Path(recipe["supervisor"]) / "supervisor-step-id"
    step = path.read_text().strip()
    allocation, separator, number = step.partition(".")
    if allocation != "14372" or separator != "." or not number.isdigit():
        raise ValueError("Supervisor receipt must name a numeric retained-allocation step")
    return step


def run(recipe_path):
    recipe = read(recipe_path)
    root, parent, code = map(Path, (recipe["root"], recipe["parent_root"], recipe["code_root"]))
    sys.path.insert(0, str(code))
    os.environ.update(read(verify(recipe["environment"])))
    operation = Path(recipe["operational_code"])
    prepared = module("e19_recovery_prepared", operation / "tools/prepared_batch.py")
    registry = prepared.controls(Path(recipe["supervisor"]), root)
    once = module("e19_recovery_once", operation / "tools/measured_once.py")
    resume = module("e19_recovery_resume", operation / "tools/resume_e19_once.py")
    args = SimpleNamespace(
        root=root,
        code_root=code,
        supervisor_step=current_supervisor_step(recipe),
        accounting_policy=verify(recipe["accounting_policy"]),
        accounting_policy_sha256=recipe["accounting_policy"]["sha256"],
    )
    resume.check_owner(args)
    registry = register_recovery_lineage(recipe, prepared)
    for item in recipe["bindings"]:
        verify(item)
    from exact.experiments.runtime import _code_identity
    from tools.resume_cached_batch import load_helpers

    h = load_helpers(root)
    original = read(verify(recipe["original_preflight"]))
    old_code = original["source"]["extraction_code"]
    actual_old = _code_identity(Path(recipe["original_source"]), evaluation=False)
    new_code = _code_identity(code, evaluation=False)
    if old_code != actual_old:
        raise ValueError("Original frozen numerical source changed")
    changed = {
        name
        for name in old_code["files"].keys() | new_code["files"].keys()
        if old_code["files"].get(name) != new_code["files"].get(name)
    }
    expected = set(recipe["changed_numerical_files"])
    if changed != expected:
        raise ValueError("Repair scope differs from its reviewed dependency proof")
    for name, record in recipe["changed_numerical_files"].items():
        if new_code["files"][name] != record["sha256"]:
            raise ValueError("Repaired numerical bytes changed")
    source = h.validate_source(code)
    if source["native_code_sha256"] != original["source"]["native_code_sha256"]:
        raise ValueError("Native runtime changed")
    # Verify every original fixed input before constructing the new attempt record.
    for key in ("budget_import", "parent_completion", "prospective_campaign", "source_selection"):
        verify(original[key])
    for item in [
        *original["pilot_cost_basis"],
        *original["records"],
        *original["operational_files"],
        *original["configs"].values(),
    ]:
        verify(item)
    started = time.time()
    current = root / "qualification/runtime/exact-om-focused-v2"
    if not (current / "budget.json").exists():
        latest, account = prepared.latest_account(registry)
        # The previous E19 charges must all be included, regardless of which
        # prepared branch used the serial spending lane in the meantime.
        old_account = read(parent / "qualification/runtime/exact-om-focused-v2/budget.json")
        if any(account["work"].get(k) != v for k, v in old_account["work"].items()):
            raise ValueError("Latest ledger lost original E19 costs")
        prepared.copy_account(latest, account, current)
    else:
        account = read(current / "budget.json")
    preflight = copy.deepcopy(original)
    preflight["source"] = source
    preflight["repair_parent"] = recipe["original_preflight"]
    preflight["repair"] = bind(root / "repair.json")
    write(root / "preflight.json", preflight)
    prepared.status(
        root, "migrating_verified_checkpoints", cumulative_budget=str(current / "budget.json")
    )
    import exact.experiments.budget as budget

    policy = module("e19_recovery_accounting_policy", args.accounting_policy)
    budget.BudgetLedger = policy.BudgetLedger
    ledger = policy.BudgetLedger(current / "budget.json", account["limits"])
    work_id = "preparation/E19/precision-migration/" + os.environ["SLURM_STEP_ID"]
    ledger.admit(work_id, group="reserve", seconds=0, requests=0, tokens=0)
    if "14372" in account.get("allocations", {}):
        ledger.record_allocation(
            "14372", start=account["allocations"]["14372"]["start"], end=time.time()
        )
    outcome = "failed"
    try:
        migrate_control(parent, root, preflight, old_code, new_code, once, account)
        migrate_prefit(parent, root, old_code, new_code)
        outcome = "complete"
    finally:
        ledger.finish(
            work_id,
            start=started,
            end=time.time(),
            status=outcome,
            requests=0,
            tokens=0,
            actual_usd=0,
        )
    account = read(current / "budget.json")
    # Reuse the existing coordinator and its exact cached-only guards. Add the
    # original failed work IDs to cumulative per-arm timing without double charging.
    original_loader = resume.module_file

    def loader(name, path):
        result = original_loader(name, path)
        if name == "run_once_qualification":
            original_probe = result.run_probe

            def probe(**kwargs):
                if kwargs["name"] == "D0_E03--analytic_fitted":
                    kwargs["previous_work_ids"] = [
                        key
                        for key in account["work"]
                        if key.startswith(
                            "qualification/" + parent.name + "/" + kwargs["name"] + "/"
                        )
                    ]
                row = original_probe(**kwargs)
                if kwargs["name"] == "D0_E03--analytic_fitted":
                    ids = [
                        key
                        for key, value in account["work"].items()
                        if key.startswith(
                            "qualification/" + parent.name + "/" + kwargs["name"] + "/"
                        )
                    ]
                    if any(
                        account["work"][key]["status"] not in {"failed", "interrupted"}
                        for key in ids
                    ):
                        raise ValueError("Historical fitted attempts are not closed")
                    current_ids = row.get("budget_work_ids", [row["budget_work_id"]])
                    missing = [key for key in ids if key not in current_ids]
                    if missing:
                        row["budget_work_ids"] = [*missing, *current_ids]
                        row["wall_seconds"] += sum(
                            account["work"][key]["seconds"] for key in missing
                        )
                        row["timing_scope"] = (
                            "Cumulative original failed and repaired attempts; each charged once"
                        )
                        row["checkpoint_continued"] = True
                        prepared.write(kwargs["directory"] / "measurement.json", row)
                return row

            result.run_probe = probe
        return result

    resume.module_file = loader
    return resume.run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    recipe = read(args.recipe)
    root = Path(recipe["root"])
    with (root / "continuation.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return run(args.recipe)
        except BaseException as error:
            # Root-local evidence only. Never alter the original failed attempt.

            state = {
                "status": "blocked",
                "at": time.time(),
                "error": str(error),
                "cumulative_budget": str(
                    next(
                        (
                            p
                            for p in (
                                root / "E19/runtime/exact-om-focused-v2/budget.json",
                                root / "qualification/runtime/exact-om-focused-v2/budget.json",
                            )
                            if p.exists()
                        ),
                        root / "qualification/runtime/exact-om-focused-v2/budget.json",
                    )
                ),
            }
            (root / "status.json").write_text(json.dumps(state, indent=2) + "\n")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
