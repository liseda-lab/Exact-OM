#!/usr/bin/env python3
"""Prepare immutable launch recipes now; bind verified upstream results at dispatch."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    from exact.utils.provenance import sha256_file

    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def verified(value):
    path = Path(value["path"])
    if binding(path) != value:
        raise ValueError("Prepared binding changed: " + str(path))
    return path


def write(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if immutable:
        if path.exists():
            if path.read_text() != text:
                raise ValueError("Immutable prepared artifact changed: " + str(path))
        else:
            with path.open("x") as stream:
                stream.write(text)
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def status(root, state, **values):
    write(root / "status.json", {"status": state, "at": time.time(), **values})


def controls(supervisor, root):
    registry = read(supervisor / "registry.json")
    paths = [supervisor / "PAUSE", supervisor / "STOP", root / "STOP", root / "PAUSE"]
    paths += [Path(p) for p in registry.get("pause_paths", [])]
    if any(p.exists() for p in paths):
        raise RuntimeError("An intentional experiment pause/STOP is present")
    return registry


def resolve_run(registry, identifier):
    runs = {row["id"]: row for row in registry["runs"]}
    visited = set()
    while identifier in runs:
        if identifier in visited:
            raise ValueError("Cyclic run replacement")
        visited.add(identifier)
        run = runs[identifier]
        if not run.get("superseded_by"):
            return run
        identifier = run["superseded_by"]
    raise ValueError("Required run has no completed submission: " + identifier)


def completed_run(run):
    complete = read(run["completion_path"])
    if complete.get("status") != "complete" or complete.get("exit_code", 0) != 0:
        raise ValueError("Required comparison is incomplete: " + run["id"])
    if complete.get("selection"):
        verified(complete["selection"])
    for item in complete.get("manifests", []):
        if read(verified(item)).get("status") != "complete":
            raise ValueError("Incomplete upstream cell")
    return complete


def source_campaign(complete):
    if complete.get("campaign"):
        return verified(complete["campaign"])
    runtime = verified(complete["selection"]).parent.parent
    path = runtime.parent.parent / "campaign.lock.yaml"
    if not path.is_file():
        raise ValueError("Completion lacks an executable source campaign")
    return path


def latest_account(registry):
    """Choose the cumulative lineage, rejecting forks that lose closed charges."""
    choices = []
    for row in registry["runs"]:
        if not row.get("enabled", True):
            continue
        path = Path(row["status_path"])
        if not path.exists():
            continue
        report = read(path)
        account = report.get("cumulative_budget")
        if account and Path(account).is_file():
            state = read(account)
            choices.append((len(state["work"]), Path(account), state))
    if not choices:
        raise ValueError("No authoritative cumulative account")
    _, path, state = max(choices, key=lambda item: (item[0], item[1].stat().st_mtime_ns))
    for _, _, previous in choices:
        if state["limits"] != previous["limits"]:
            raise ValueError("Divergent cumulative accounting limits")
        for allocation, observation in previous.get("allocations", {}).items():
            current = state.get("allocations", {}).get(allocation, {})
            if (
                current.get("start") != observation["start"]
                or current.get("end", 0) < observation["end"]
            ):
                raise ValueError("Lost cumulative allocation accounting")
        for key, value in previous["work"].items():
            if (value["status"] != "reserved" or key == "historical/G0") and state["work"].get(
                key
            ) != value:
                raise ValueError(
                    "Divergent cumulative accounting; reconcile before dispatch: " + key
                )
    pending = [
        key
        for key, value in state["work"].items()
        if value["status"] == "reserved" and key != "historical/G0"
    ]
    if pending:
        raise ValueError("Earlier owner has unclosed accounting: " + repr(pending))
    return path, state


def historical_selection(lock, step_id, campaign, runtime, directory):
    """Bind original scientific outputs; do not impersonate new predictions."""
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.campaign import CampaignLock, external_selection_result

    previous = load_yaml_mapping(campaign)
    step = copy.deepcopy(next(item for item in previous["steps"] if item["id"] == step_id))
    if step.get("external_selection") or step.get("external_acceptance"):
        return
    cells = sorted((runtime / "screen/runs" / step_id).glob("*/*/seed-*/experiment_manifest.json"))
    expected = (
        len(step["arms"])
        * len(step["execution_modes"])
        * len(step["seeds"])
        * (1 + len(step.get("additional_cases", [])))
    )
    if len(cells) != expected or any(read(p).get("status") != "complete" for p in cells):
        raise ValueError("Historical result does not contain every cell: " + step_id)
    record = {
        "schema_version": 1,
        "kind": "exact_om_prior_selection",
        "campaign": binding(campaign),
        "selection": binding(runtime / "screen/selection.json"),
        "result_set": binding(runtime / "screen/current-result-set.json"),
        "cells": [binding(p) for p in cells],
        "evidence": [
            binding(runtime / "screen" / name)
            for name in ("design.json", "progress.json", "metrics.json", "paired_bootstrap.json")
        ],
    }
    target = directory / (step_id + ".json")
    write(target, record, immutable=True)
    step.update(external_selection=binding(target), estimate=None)
    for readiness in step["readiness"].values():
        readiness["screen"].update(
            status="complete",
            missing=[],
            reason="Verified original completed scientific comparison",
        )
    for case_id in [step["case"], *step.get("additional_cases", [])]:
        case = previous["cases"][case_id]
        if case_id in lock["cases"] and lock["cases"][case_id] != case:
            raise ValueError("Historical case bindings disagree: " + case_id)
        lock["cases"][case_id] = copy.deepcopy(case)
    index = next(i for i, value in enumerate(lock["steps"]) if value["id"] == step_id)
    lock["steps"][index] = step
    parsed = CampaignLock.model_validate(lock)
    external_selection_result(parsed, parsed.steps[index], directory)


def prepare_lock(recipe, root, registry, *, completed=True):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping

    lock = copy.deepcopy(load_yaml_mapping(verified(recipe["base_campaign"])))
    group = read(verified(recipe["group"]))
    name = recipe["scientific_step"]
    row = next(value for value in group["rows"] if value["step"] == recipe["group_step"])
    target = copy.deepcopy(row["declaration"])
    if name != recipe["group_step"]:
        target = copy.deepcopy(next(value for value in lock["steps"] if value["id"] == name))
    for case_id in [target["case"], *target.get("additional_cases", [])]:
        original_id = "D0" if case_id == "D0_E03" else case_id
        prepared = group["cases"][original_id]
        retained = lock["cases"][original_id]
        for field in (
            "source",
            "target",
            "source_universe",
            "kind",
            "role",
            "overlay",
            "negative_policy",
            "evaluation_source_labels",
            "evaluation_candidate_labels",
            "evaluation_label_semantics",
        ):
            if prepared.get(field) != retained.get(field):
                raise ValueError(
                    "Prepared case differs from retained case: " + case_id + "/" + field
                )
        for field in ("references", "local_references", "candidates", "frozen_global_candidates"):
            for role in ("train", "valid"):
                if prepared.get(field, {}).get(role) != retained.get(field, {}).get(role):
                    raise ValueError(
                        "Prepared development binding differs: "
                        + case_id
                        + "/"
                        + field
                        + "/"
                        + role
                    )
    # Preserve the established bounded training population without changing D0 validation.
    if target["case"] in {"D0", "D0_E03"} and "D0_E03" in lock["cases"]:
        original, bounded = lock["cases"]["D0"], lock["cases"]["D0_E03"]
        for field in ("source", "target", "source_universe", "role", "kind"):
            if original[field] != bounded[field]:
                raise ValueError("Bounded training alias changes development population")
        if original["references"]["valid"] != bounded["references"]["valid"]:
            raise ValueError("Bounded training alias changes development labels")
        target["case"] = "D0_E03"
    index = next(i for i, value in enumerate(lock["steps"]) if value["id"] == name)
    lock["steps"][index] = target
    if lock.get("generate_rationales") is not False:
        raise ValueError("Experimental rationale generation must remain disabled")
    lock["blueprint"]["path"] = str(verified(recipe["blueprint"]))
    if completed:
        for identifier in recipe["depends_on"]:
            completed_run(resolve_run(registry, identifier))
    known = {
        step["id"]
        for step in lock["steps"]
        if step.get("external_selection") or step.get("external_acceptance")
    }
    for run in registry["runs"]:
        if not run.get("enabled", True) or not Path(run["completion_path"]).is_file():
            continue
        # Failed independent branches remain registered for recovery. They do not
        # provide importable evidence and must not block an unrelated ready cell.
        if read(run["completion_path"]).get("status") != "complete":
            continue
        receipt = completed_run(run)
        if not receipt.get("selection"):
            continue
        selection_path = verified(receipt["selection"])
        experiments = read(selection_path).get("experiments", {})
        needed = (
            {
                key
                for key, result in experiments.items()
                if result.get("status") in {"selected", "screened_out"}
            }
            - known
            - {name}
        )
        needed &= {step["id"] for step in lock["steps"]}
        if not needed:
            continue
        previous_path = source_campaign(receipt)
        previous = load_yaml_mapping(previous_path)
        runtime = selection_path.parent.parent
        for prior in previous["steps"]:
            if (
                prior["id"] not in needed
                or prior.get("external_selection")
                or prior.get("external_acceptance")
            ):
                continue
            if (runtime / "screen/runs" / prior["id"]).is_dir() and prior["phase"] in {
                "initial",
                "expansion",
                "late",
                "sentinel",
            }:
                historical_selection(lock, prior["id"], previous_path, runtime, root / "history")
                known.add(prior["id"])
    for step in lock["steps"]:
        if step["id"] == name:
            step["estimate"] = None
            for states in step["readiness"].values():
                states["screen"].update(
                    status="screen_ready",
                    missing=[],
                    reason="Prepared complete comparison; verified dependencies bound before dispatch; runtime forecast advisory",
                    inspected_commit=recipe["commit"],
                    tests=recipe["checks"],
                    implemented_paths=["exact/experiments/campaign.py", "tools/prepared_batch.py"],
                )
        elif not step.get("external_selection") and not step.get("external_acceptance"):
            step["estimate"] = None
            for states in step["readiness"].values():
                states["screen"].update(
                    status="blocked_input_resolution",
                    missing=["separately queued"],
                    reason="Separately queued comparison; never execute implicitly",
                )
    return lock


def copy_account(source, state, destination, *, request_ledger=None):
    destination.mkdir(parents=True, exist_ok=True)
    budget = destination / "budget.json"
    if budget.exists() and read(budget) != state:
        raise ValueError("Immutable prepared artifact changed: " + str(budget))
    old = (
        verified(request_ledger)
        if request_ledger is not None
        else source.parent / "openrouter/requests.sqlite3"
    )
    new = destination / "openrouter/requests.sqlite3"
    if not new.exists():
        new.parent.mkdir(parents=True, exist_ok=True)
        temporary = new.with_suffix(".pending")
        with sqlite3.connect(old.as_uri() + "?mode=ro", uri=True) as before, sqlite3.connect(
            temporary
        ) as after:
            before.backup(after)
            if after.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("Hosted ledger copy failed integrity verification")
        os.replace(temporary, new)
    # Publish the account only after its request history is durable. A failed
    # cache transfer must not become the newest authoritative budget.
    write(budget, state, immutable=True)
    # Large embeddings/native preparation stay in the shared durable cache.
    write(
        destination / "account-import.json",
        {"source": binding(source), "request_ledger": str(old)},
        immutable=True,
    )


def hosted_caps(state, ledger):
    from tools.qualify_cached_family import usage

    actual = usage(ledger)
    limits = state["limits"]
    result = {}
    for plural, wire in (("requests", "attempts"), ("tokens", "billable_tokens")):
        used = sum(row[plural] for row in state["work"].values())
        remaining = limits[plural + "_cap"] - limits.get("final_" + plural + "_reserved", 0) - used
        if remaining < 0:
            raise ValueError("Protected hosted spending exhausted")
        result["EXACT_OPENROUTER_" + plural[:-1].upper() + "_CAP"] = str(actual[wire] + remaining)
    return {**result, "EXACT_OPENROUTER_RETRY_UNKNOWN": "0"}


def run_recipe(path):
    from types import SimpleNamespace

    from exact.core.entities.configs.yaml_io import dump_yaml_document
    from exact.experiments.campaign import campaign_plan
    from tools.experiment_resources import guarded_execute
    from tools.resume_e19_once import check_owner

    recipe = read(path)
    root, code, supervisor = map(Path, (recipe["root"], recipe["code_root"], recipe["supervisor"]))
    with (root / "worker.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = controls(supervisor, root)
        check_owner(
            SimpleNamespace(
                root=root, supervisor_step=(supervisor / "supervisor-step-id").read_text().strip()
            )
        )
        if (
            os.environ.get("SLURM_JOB_ID") != recipe["allocation"]
            or not os.environ.get("SLURM_STEP_ID", "").isdigit()
        ):
            raise ValueError("A retained numeric Slurm step is required")
        identity = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=code, text=True
        ).strip()
        if (
            identity != recipe["commit"]
            or subprocess.check_output(
                ["git", "diff", "--name-only", "HEAD"], cwd=code, text=True
            ).strip()
        ):
            raise ValueError("Frozen execution source changed")
        previous_completion = root / "completion.json"
        if previous_completion.exists():
            previous = read(previous_completion)
            if previous.get("status") == "complete":
                raise ValueError("Completed comparison must never be relaunched")
            write(root / ("failed-" + previous["step_id"] + ".json"), previous, immutable=True)
            previous_completion.unlink()
        os.environ.update(read(verified(recipe["environment"])))
        os.environ["EXACT_EVIDENCE_PREFETCH"] = "0"
        os.environ["OPENROUTER_API_KEY"] = (
            (Path(recipe["repository"]) / "api_key").read_text().strip()
        )
        runtime = root / "runtime" / recipe["campaign_id"]
        status(root, "binding_dependencies", cumulative_budget=str(runtime / "budget.json"))
        campaign = root / "campaign.lock.yaml"
        launch = root / "launch.json"
        started = time.time()
        if launch.exists():
            retained = read(launch)
            if retained["recipe"] != binding(path) or verified(retained["campaign"]) != campaign:
                raise ValueError("Resume must retain the original recipe and campaign")
            for identifier in recipe["depends_on"]:
                completed_run(resolve_run(registry, identifier))
            parent, state = latest_account(registry)
            if parent != runtime / "budget.json":
                raise ValueError(
                    "Resume account is no longer authoritative; reconcile before retry"
                )
        else:
            lock = prepare_lock(recipe, root, registry)
            text = dump_yaml_document(lock)
            if campaign.exists() and campaign.read_text() != text:
                raise ValueError("Existing campaign differs; use an explicit repair record")
            if not campaign.exists():
                campaign.write_text(text)
            parent, state = latest_account(registry)
            if parent == runtime / "budget.json":
                raise ValueError("Incomplete launch owns an imported account; inspect before retry")
            copy_account(parent, state, runtime)
            write(
                launch,
                {
                    "campaign": binding(campaign),
                    "recipe": binding(path),
                    "source_commit": identity,
                    "parent_budget": binding(parent),
                },
                immutable=True,
            )
        os.environ.update(hosted_caps(state, runtime / "openrouter"))
        plan = campaign_plan(campaign, stage="screen")
        relevant = [row for row in plan["rows"] if row["step"] == recipe["scientific_step"]]
        if plan["budget_errors"] or not relevant or any(row["issues"] for row in relevant):
            raise ValueError("Prepared complete comparison failed admission: " + repr(relevant))
        from exact.experiments.budget import BudgetLedger

        ledger = BudgetLedger(runtime / "budget.json", state["limits"])
        setup = "prepared-dispatch/" + recipe["scientific_step"] + "/" + os.environ["SLURM_STEP_ID"]
        ledger.admit(setup, group="reserve", seconds=0, forecast_known=False)
        ledger.finish(
            setup,
            start=started,
            end=time.time(),
            status="complete",
            requests=0,
            tokens=0,
            actual_usd=0,
        )
        for allocation, observation in state.get("allocations", {}).items():
            if allocation == recipe["allocation"]:
                ledger.record_allocation(allocation, start=observation["start"], end=time.time())
        status(
            root,
            "running",
            scientific_step=recipe["scientific_step"],
            cumulative_budget=str(runtime / "budget.json"),
        )
        guarded_execute(campaign, root, code, check_pause=lambda: controls(supervisor, root))
        selection = runtime / "screen/selection.json"
        result = read(selection)["experiments"][recipe["scientific_step"]]
        if result["status"] not in {"selected", "screened_out", "complete"}:
            raise ValueError("Comparison did not reach a complete scientific decision")
        manifests = sorted(
            (runtime / "screen/runs" / recipe["scientific_step"]).glob(
                "*/*/seed-*/experiment_manifest.json"
            )
        )
        from exact.core.entities.configs.yaml_io import load_yaml_mapping

        declared = next(
            s for s in load_yaml_mapping(campaign)["steps"] if s["id"] == recipe["scientific_step"]
        )
        expected = {
            (arm["id"], case + "-" + mode, mode, seed)
            for arm in declared["arms"]
            for case in [declared["case"], *declared.get("additional_cases", [])]
            for mode in declared["execution_modes"]
            for seed in declared["seeds"]
        }
        reports = [read(p) for p in manifests]
        observed = {(r["arm_id"], r["task_id"], r["execution_mode"], r["seed"]) for r in reports}
        if (
            len(reports) != len(expected)
            or observed != expected
            or any(
                r.get("status") != "complete"
                or r.get("return_code") != 0
                or r.get("extraction_complete") is not True
                or r.get("generate_rationales") is not False
                for r in reports
            )
        ):
            raise ValueError("Incomplete comparison cells")
        write(
            root / "completion.json",
            {
                "status": "complete",
                "exit_code": 0,
                "campaign": binding(campaign),
                "selection": binding(selection),
                "manifests": [binding(p) for p in manifests],
                "cumulative_budget": binding(runtime / "budget.json"),
                "step_id": os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                "dispatch_nonce": recipe["dispatch_nonce"],
                "generate_rationales": False,
            },
            immutable=True,
        )
        status(root, "complete", cumulative_budget=str(runtime / "budget.json"))


def prepare(root, batch, *, base_campaign, code, supervisor, environment_path, checks):
    """Only static preparation. No ontology loading, numerical work or hosted calls."""
    import uuid

    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.campaign import CampaignLock

    root.mkdir(parents=True, exist_ok=True)
    group = read(batch["binding_path"])
    name = batch["id"].removesuffix("-run-once-followup")
    rows = [row for row in group["rows"] if row["step"] == batch["scientific_step"]]
    if len(rows) != 1:
        raise ValueError("Prepared group does not identify exactly one declaration")
    row = rows[0]
    if row["declaration"]["phase"] not in {"initial", "expansion", "late", "sentinel"}:
        raise ValueError("Freeze/final work requires a separately bound final selection")
    if name == "E13-enrichment":
        raise ValueError(
            "Native enrichment must be materialized and its paired arm inputs bound first"
        )
    if batch.get("needs_user"):
        raise ValueError("A scientific/input decision is still required")
    base = load_yaml_mapping(base_campaign)
    if name not in {step["id"] for step in base["steps"]}:
        raise ValueError("Comparison alias is not declared in the retained campaign")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    recipe = {
        "schema_version": 1,
        "root": str(root.resolve()),
        "scientific_step": name,
        "group_step": batch["scientific_step"],
        "group": binding(batch["binding_path"]),
        "base_campaign": binding(base_campaign),
        "blueprint": binding(Path(base_campaign).parent / base["blueprint"]["path"]),
        "depends_on": batch["depends_on"],
        "supervisor": str(supervisor.resolve()),
        "code_root": str(code.resolve()),
        "commit": commit,
        "checks": checks,
        "repository": read(supervisor / "policy.json")["repository"],
        "environment": binding(environment_path),
        "allocation": read(supervisor / "policy.json")["allocation"],
        "campaign_id": base["campaign_id"],
        "dispatch_nonce": uuid.uuid4().hex,
    }
    preview = prepare_lock(
        recipe, root / "preview", read(supervisor / "registry.json"), completed=False
    )
    CampaignLock.model_validate(preview)
    recipe_path = root / "recipe.json"
    write(recipe_path, recipe, immutable=True)
    python = Path(recipe["repository"]) / ".venv/bin/python"
    worker = root / "worker-entry.sh"
    step_path = root / "step.json"
    worker.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(recipe['repository'])}
[[ "${{SLURM_JOB_ID:-}}" = {shlex.quote(recipe['allocation'])} && "${{SLURM_STEP_ID:-}}" =~ ^[0-9]+$ ]]
finish() {{
    result=$?
    printf "%s\\n" "$result" > {shlex.quote(str(root / 'exit-code'))}
    {shlex.quote(str(python))} {shlex.quote(str(code / 'tools/prepared_batch.py'))} --recipe {shlex.quote(str(recipe_path))} --record-exit "$result" || true
    exit "$result"
}}
trap finish EXIT
printf '{{"step_id":"%s.%s","dispatch_nonce":"{recipe['dispatch_nonce']}"}}\\n' "$SLURM_JOB_ID" "$SLURM_STEP_ID" > {shlex.quote(str(step_path))}.pending
mv {shlex.quote(str(step_path))}.pending {shlex.quote(str(step_path))}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH={shlex.quote(str(code))}
{shlex.quote(str(python))} -u {shlex.quote(str(code / 'tools/prepared_batch.py'))} --recipe {shlex.quote(str(recipe_path))}
"""
    )
    descriptor = {
        "nonce": recipe["dispatch_nonce"],
        "tmux_socket": f"/tmp/tmux-{os.getuid()}/default",
        "argv": [
            shutil.which("srun") or "/usr/bin/srun",
            "--jobid=" + recipe["allocation"],
            "--overlap",
            "--immediate=15",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=6",
            "--cpu-bind=none",
            "--gres=gpu:rtx4090:1",
            "--time=0",
            "/bin/bash",
            str(worker.resolve()),
        ],
        "bindings": [
            binding(worker),
            binding(recipe_path),
            binding(code / "tools/prepared_batch.py"),
            recipe["group"],
            recipe["environment"],
        ],
        "pause_paths": [str(root / "STOP"), str(root / "runtime" / recipe["campaign_id"] / "STOP")],
        "step_path": str(step_path.resolve()),
        "launcher_log": str((root / "launcher.log").resolve()),
        "run": {
            "id": batch["id"],
            "status_path": str((root / "status.json").resolve()),
            "exit_path": str((root / "exit-code").resolve()),
            "completion_path": str((root / "completion.json").resolve()),
        },
    }
    write(root / "launch-descriptor.json", descriptor, immutable=True)
    status(root, "prepared_waiting_dependencies", dependencies=batch["depends_on"])
    return descriptor


def record_exit(path, result):
    """Retain a short failure even when it exits between dispatcher polls."""
    recipe = read(path)
    root = Path(recipe["root"])
    with (root / "worker.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        completion = root / "completion.json"
        step_id = recipe["allocation"] + "." + os.environ.get("SLURM_STEP_ID", "")
        if completion.exists():
            previous = read(completion)
            if previous.get("status") == "complete" or previous.get("step_id") == step_id:
                return
            write(root / ("failed-" + previous["step_id"] + ".json"), previous, immutable=True)
        write(
            completion,
            {
                "status": "failed",
                "exit_code": result,
                "step_id": step_id,
                "dispatch_nonce": recipe["dispatch_nonce"],
            },
        )
        report = root / "status.json"
        if not report.exists() or read(report).get("status") != "failed":
            status(
                root,
                "failed",
                error="Worker exited without a complete comparison",
                exit_code=result,
                cumulative_budget=str(root / "runtime" / recipe["campaign_id"] / "budget.json"),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--record-exit", type=int)
    args = parser.parse_args()
    if args.record_exit is not None:
        record_exit(args.recipe, args.record_exit)
        return
    try:
        run_recipe(args.recipe)
    except BaseException as exc:
        recipe = read(args.recipe)
        root = Path(recipe["root"])
        complete = root / "completion.json"
        if isinstance(exc, BlockingIOError) or (
            complete.exists() and read(complete).get("status") == "complete"
        ):
            raise
        status(
            Path(recipe["root"]),
            "failed",
            error=type(exc).__name__ + ": " + str(exc),
            cumulative_budget=str(
                Path(recipe["root"]) / "runtime" / recipe["campaign_id"] / "budget.json"
            ),
        )
        record_exit(args.recipe, 130 if isinstance(exc, KeyboardInterrupt) else 1)
        raise


if __name__ == "__main__":
    main()
