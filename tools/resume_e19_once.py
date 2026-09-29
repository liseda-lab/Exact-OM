#!/usr/bin/env python3
"""Continue the retained E19 numerical source and publish each measured arm once."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def module_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_owner(args):
    """Allow only the retained shell, the explicitly bound supervisor and this step."""
    prefix = "14372."
    if (
        not args.supervisor_step.startswith(prefix)
        or not args.supervisor_step[len(prefix) :].isdigit()
    ):
        raise ValueError("Supervisor must be a numeric step in allocation 14372")
    own = prefix + os.environ.get("SLURM_STEP_ID", "")
    steps = subprocess.check_output(
        ["squeue", "--steps", "-h", "-j", "14372", "-o", "%i"], text=True
    ).splitlines()
    unexpected = set(map(str.strip, steps)) - {"14372.0", "14372.extern", own, args.supervisor_step}
    if unexpected:
        raise ValueError("Another numerical worker remains live: " + repr(sorted(unexpected)))
    import psutil

    process = psutil.Process()
    own_tree = {process.pid, *(parent.pid for parent in process.parents())}
    root = str(args.root.resolve())
    for item in psutil.process_iter(["pid", "name", "cmdline"]):
        if (
            item.pid not in own_tree
            and (item.info["name"] or "").startswith(("python", "srun"))
            and any(
                arg == root or arg.startswith(root + "/") for arg in (item.info["cmdline"] or [])
            )
        ):
            raise ValueError("A previous E19 owner/descendant remains: " + str(item.pid))


def run(args):
    # Only accounting and orchestration are amended. The exact/ modules and workers
    # continue to execute from the unchanged, originally measured checkout.
    sys.path.insert(0, str(args.code_root))
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.campaign import CampaignLock, execute_campaign
    from tools.resume_cached_batch import account_inventory, load_helpers, read

    queue = module_file("retained_e19_queue", args.code_root / "tools/queue_e19.py")
    qualification = module_file(
        "run_once_qualification", Path(__file__).with_name("qualify_cached_family.py")
    )
    once = module_file("run_once_promotion", Path(__file__).with_name("measured_once.py"))
    h = load_helpers(args.root)
    if (
        os.environ.get("SLURM_JOB_ID") != "14372"
        or not os.environ.get("SLURM_STEP_ID", "").isdigit()
    ):
        raise ValueError("Use a detached numeric step inside the retained allocation 14372")
    h.check_pause()
    check_owner(args)
    receipt = read(args.root / "preflight.json")
    if (
        receipt["status"] != "preflight_passed_not_admitted"
        or h.validate_source(args.code_root) != receipt["source"]
    ):
        raise ValueError("Retained source differs from its original measured preflight")
    for key in ("budget_import", "parent_completion", "prospective_campaign", "source_selection"):
        h.verify(receipt[key])
    for item in [
        *receipt["pilot_cost_basis"],
        *receipt["records"],
        *receipt["operational_files"],
        *receipt["configs"].values(),
    ]:
        h.verify(item)
    if (args.root / "completion.json").exists():
        completed = read(args.root / "completion.json")
        if completed.get("status") != "complete":
            raise ValueError("Existing completion receipt is not successful")
        for item in [
            completed["selection"],
            completed["cumulative_budget"],
            *completed["manifests"],
        ]:
            h.verify(item)
        print("E19 already complete; no work repeated", flush=True)
        return 0
    lock = load_yaml_mapping(h.verify(receipt["prospective_campaign"]))
    current = args.root / "qualification/runtime/exact-om-focused-v2"
    wave = args.root / "E19"
    destination = wave / "runtime/exact-om-focused-v2"
    if destination.exists():
        if not (wave / "run-once-forecast.json").exists():
            raise ValueError(
                "An earlier duplicated scientific runtime exists; reconcile it before migration"
            )
        current = destination
    queue.validate_budget(read(current / "budget.json"), lock)
    if args.accounting_policy:
        if qualification.binding(args.accounting_policy)["sha256"] != args.accounting_policy_sha256:
            raise ValueError("Reviewed accounting policy bytes changed")
        import exact.experiments.budget as budget

        policy = module_file("reviewed_run_once_budget", args.accounting_policy)
        budget.BudgetLedger = policy.BudgetLedger
    amendment = {
        "policy": "Each measured E19 execution supplies the corresponding scientific cell; no second execution",
        "original_preflight": qualification.binding(args.root / "preflight.json"),
        "numerical_source": receipt["source"],
        "operational_files": [
            qualification.binding(Path(__file__)),
            qualification.binding(Path(qualification.__file__)),
            qualification.binding(Path(once.__file__)),
        ],
        "accounting_policy": (
            qualification.binding(args.accounting_policy) if args.accounting_policy else None
        ),
        "costs_retained": True,
        "scientific_choices_unchanged": True,
    }
    amendment_id = hashlib.sha256(json.dumps(amendment, sort_keys=True).encode()).hexdigest()
    amendment_path = args.root / "run-once-amendments" / (amendment_id + ".json")
    h.freeze(amendment_path, amendment)
    retained = read(queue.DATA / "d1-production-probe-01/launch.json")
    os.environ.update(retained["environment"])
    os.environ["PYTHONPATH"] = str(args.code_root)
    os.environ["OPENROUTER_API_KEY"] = (queue.PROJECT / "api_key").read_text().strip()
    cached = qualification.usage(current / "openrouter")
    rows = []
    for key in queue.KEYS:
        h.check_pause()
        h.status("running_E19_once", arm=key, cumulative_budget=str(current / "budget.json"))
        rows.append(
            qualification.run_probe(
                code_root=args.code_root,
                script=args.root / "continue.py",
                config=h.verify(receipt["configs"][key]),
                name=key,
                directory=args.root / "qualification" / key,
                shared=current,
                campaign=CampaignLock.model_validate(lock),
                forecast_seconds=receipt["pilot_allowance_seconds_per_arm"][key],
                case_id=key.split("--")[0],
                resume=True,
            )
        )
    h.freeze(
        args.root / "fitting-evidence.json",
        queue.fitted_evidence(rows, read(args.root / "training.json")),
    )
    pools = queue.matched_measurements(rows)
    if qualification.usage(current / "openrouter") != cached:
        raise ValueError("Cached scientific execution incurred incremental hosted usage")
    measurement = wave / "measured-once.json"
    h.freeze(
        measurement,
        {
            "status": "complete_scientific_executions",
            "rows": rows,
            "candidate_population": pools,
            "method": "Original complete executions become scientific cells; original and interrupted costs retained",
        },
    )
    proof = once.reused_comparison_forecast(queue.family_forecast(rows, measurement, h))
    h.freeze(wave / "run-once-forecast.json", proof)
    campaign = wave / "campaign.lock.yaml"
    ready = queue.ready_lock(lock, proof["estimate"], receipt["source"]["commit"])
    for states in next(step for step in ready["steps"] if step["id"] == queue.FAMILY)[
        "readiness"
    ].values():
        states["screen"][
            "reason"
        ] = "Complete matched measured executions are imported as scientific cells by exact numerical identity; no duplicate worker."
    h.freeze(campaign, ready)
    h.check_pause()
    if not destination.exists():
        h.copy_state(current, destination)
    current = destination
    h.status(
        "finalizing_E19_once",
        cumulative_budget=str(current / "budget.json"),
        campaign=str(campaign),
    )
    once.promote_measured_cells(
        campaign,
        current,
        rows,
        queue.FAMILY,
        read(h.verify(receipt["source_selection"])),
        args.code_root,
    )
    os.environ.update(
        EXACT_OPENROUTER_REQUEST_CAP=str(cached["attempts"]),
        EXACT_OPENROUTER_TOKEN_CAP=str(cached["billable_tokens"]),
        EXACT_OPENROUTER_RETRY_UNKNOWN="0",
    )
    launch = {
        "campaign": qualification.binding(campaign),
        "source": receipt["source"],
        "budget_path": str(current / "budget.json"),
        "step_id": "14372." + os.environ["SLURM_STEP_ID"],
        "run_once_amendment": qualification.binding(amendment_path),
    }
    h.freeze(wave / "launches" / (launch["step_id"] + ".json"), launch)
    if not (wave / "launch.json").exists():
        h.freeze(wave / "launch.json", launch)
    execute_campaign(
        campaign,
        stage="screen",
        output_root=wave / "runtime",
        workdir=args.code_root,
        jobs=1,
        reuse_plan_only=True,
    )
    h.check_pause()
    with account_inventory(
        h,
        current,
        queue.FAMILY + "/once/" + os.environ["SLURM_STEP_ID"],
        "channels",
        proof["inventory_seconds"],
    ):
        with once.require_reuse_only():
            h.guarded_execute(campaign, wave, args.code_root)
    if qualification.usage(current / "openrouter") != cached:
        raise ValueError("Result finalization incurred hosted usage")
    manifests = sorted(
        (current / "screen/runs/E19").glob("*/D*-global_alignment/seed-17/experiment_manifest.json")
    )
    if len(manifests) != len(queue.ARMS) or any(
        read(path).get("status") != "complete" for path in manifests
    ):
        raise ValueError("Completion requires all three verified E19 cells")
    h.freeze(
        args.root / "completion.json",
        {
            "status": "complete",
            "exit_code": 0,
            "cells": len(manifests),
            "selection": qualification.binding(current / "screen/selection.json"),
            "manifests": [qualification.binding(path) for path in manifests],
            "cumulative_budget": qualification.binding(current / "budget.json"),
            "source": receipt["source"],
            "cached_usage_unchanged": True,
            "generate_rationales": False,
            "measured_results": qualification.binding(current / "measured-results.json"),
            "run_once_amendment": qualification.binding(amendment_path),
        },
    )
    h.status("complete", cells=len(manifests), cumulative_budget=str(current / "budget.json"))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--supervisor-step", default="14372.28")
    parser.add_argument("--accounting-policy", type=Path)
    parser.add_argument("--accounting-policy-sha256")
    args = parser.parse_args(argv)
    if bool(args.accounting_policy) != bool(args.accounting_policy_sha256):
        parser.error("Bind both the reviewed accounting policy path and its SHA256")
    with (args.root / "continuation.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return run(args)
        except BaseException as exc:
            sys.path.insert(0, str(args.code_root))
            from tools.resume_cached_batch import load_helpers

            h = load_helpers(args.root)
            latest = next(
                (
                    path
                    for path in (
                        args.root / "E19/runtime/exact-om-focused-v2/budget.json",
                        args.root / "qualification/runtime/exact-om-focused-v2/budget.json",
                    )
                    if path.exists()
                ),
                None,
            )
            h.status(
                "blocked",
                cumulative_budget=str(latest),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
