#!/usr/bin/env python3
"""Execute one reviewed corrected cell in its registered retained Slurm step.

Preparation never creates the required rollout admission. This adapter deliberately
does not publish queue rows, change allowances, fit heads or evaluate labels.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from exact.experiments.mixed_scale import binding, prepare_successor_bundle, read_binding, verified


def admission_environment(admission, policy):
    """Use the supervisor's actual hard-pause amendment and canonical E17 scope."""
    from exact.experiments.runtime import hosted_scope_environment
    from exact.utils.hosted_spending import load_spending_policy, spending_policy_environment

    spending = load_spending_policy(policy.get("hosted_spending_policy"))
    if (spending is None or spending["policy"]["mode"] != "hard_pause"
            or admission.get("spending_policy") != policy.get("hosted_spending_policy")):
        raise ValueError("Corrected worker requires the unchanged supervisor hard-pause spending policy")
    environment = read_binding(admission["environment"])
    if (not isinstance(environment, dict) or any(not isinstance(v, str) for v in environment.values())
            or environment.get("EXACT_PAIR_CONTEXT_BATCHING", "0") != "0"):
        raise ValueError("Invalid reviewed environment or unqualified numerical batching")
    environment.update(spending_policy_environment(spending))
    environment.update(hosted_scope_environment(spending["policy"]["campaign_id"], "E17", policy=spending))
    environment["EXACT_EVIDENCE_PREFETCH"] = "0"
    environment["EXACT_PAIR_CONTEXT_BATCHING"] = "0"
    return environment, spending


def run(descriptor_path, admission_path):
    from exact.core.entities.configs.config import ConfigModel
    from exact.experiments.budget import BudgetLedger
    from exact.utils.hosted_spending import record_spending_policy
    from tools.prepared_batch import controls, copy_account, latest_account, status, write
    from tools.qualify_cached_family import usage
    from tools.resume_e19_once import check_owner

    started = time.time()
    descriptor, admission = json.loads(descriptor_path.read_text()), json.loads(admission_path.read_text())
    if (descriptor.get("kind") != "corrected_cell_worker"
            or admission.get("kind") != "corrected_worker_rollout_admission"
            or admission.get("rollout_authorized") is not True
            or admission.get("descriptor") != binding(descriptor_path)):
        raise ValueError("A reviewed descriptor-specific rollout admission is required")
    supervisor, root, code = (Path(admission[field]).resolve() for field in ("supervisor", "root", "code_root"))
    policy = read_binding(admission["supervisor_policy"])
    if verified(admission["supervisor_policy"]) != supervisor / "policy.json":
        raise ValueError("Admission does not bind the active supervisor policy")
    if (os.environ.get("SLURM_JOB_ID") != str(policy["allocation"])
            or not os.environ.get("SLURM_STEP_ID", "").isdigit()):
        raise ValueError("Corrected worker requires a retained numeric Slurm step")
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
            != descriptor["source_revision"] or subprocess.check_output(
                ["git", "diff", "--name-only", "HEAD"], cwd=code, text=True).strip()
            or Path(__file__).resolve().parent.parent != code):
        raise ValueError("Corrected worker must execute the reviewed clean frozen source")
    environment, spending = admission_environment(admission, policy)
    registered = None
    for _ in range(30):
        registry = controls(supervisor, root)
        registered = next((row for row in registry["runs"] if row["id"] == admission["run_id"]), None)
        if registered is not None:
            break
        time.sleep(1)  # Dispatch registers the worker after receiving its numeric step receipt.
    if registered is None or Path(registered["status_path"]) != root / "status.json":
        raise ValueError("Worker has not been registered in the live supervisor")
    step = os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"]
    receipt = json.loads((root / "step.json").read_text())
    if (registered.get("step_id") != step or receipt.get("step_id") != step
            or registered.get("dispatch_nonce") != receipt.get("dispatch_nonce")
            or not registered.get("dispatch_nonce")):
        raise ValueError("Worker Slurm step/dispatch nonce differs from its registered receipt")
    check_owner(SimpleNamespace(root=root, supervisor_step=(supervisor / "supervisor-step-id").read_text().strip()))
    # Re-run all scientific binding checks immediately before any model or paid call.
    rebuilt = prepare_successor_bundle(root / "validation", registry=supervisor / "registry.json",
        public_inputs=verified(descriptor["public_inputs"]), selection_freeze=descriptor["selection_freeze"],
        deployments={descriptor["cell"]["id"]: descriptor["inference"]}, source_revision=descriptor["source_revision"])
    row = next(row for row in rebuilt["logical_to_physical"] if row["id"] == descriptor["cell"]["id"])
    if descriptor != read_binding(row["worker"]):
        raise ValueError("Worker differs from the reconstructed corrected scientific descriptor")
    manifest = read_binding(descriptor["inference"])
    if descriptor["runs"] != manifest["runs"]:
        raise ValueError("Worker commands changed their frozen inference runs")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "launch.json").exists():
            raise ValueError("Existing attempt requires explicit recovery; never silently repeat uncertain paid requests")
        os.environ.update(environment)
        parent, state = latest_account(registry)
        runtime = root / "runtime"
        copy_account(parent, state, runtime)
        record_spending_policy(root, spending)
        write(root / "launch.json", {"descriptor": binding(descriptor_path), "admission": binding(admission_path),
            "parent_budget": binding(parent), "spending_policy": spending}, immutable=True)
        status(root, "running", cumulative_budget=str(runtime / "budget.json"))
        ledger = BudgetLedger(runtime / "budget.json", state["limits"])
        before = usage(runtime / "openrouter")
        forecast = admission["forecast"]
        work = descriptor["cell"]["id"] + "/" + os.environ["SLURM_STEP_ID"]
        ledger.admit(work, group="final", seconds=forecast["seconds"], requests=forecast["requests"],
            tokens=forecast["tokens"], projected_usd=forecast["usd"], hosted_usage_baseline=before)
        os.environ.update(EXACT_OPENROUTER_LEDGER_DIR=str(runtime / "openrouter"),
            EXACT_EXPERIMENT_MODE="1", EXACT_EXPERIMENT_STOP_FILE=str(runtime / "STOP"),
            EXACT_EXPERIMENT_ROLE="valid" if descriptor["cell"]["section"] in {"bounded", "components"} else "test")
        if descriptor["executor"] != "pinned_published_matcher" and not os.environ.get("OPENROUTER_API_KEY"):
            os.environ["OPENROUTER_API_KEY"] = (Path(policy["repository"]) / "api_key").read_text().strip()
        outcome = "failed"
        done = threading.Event()
        def monitor():
            from tools.experiment_resources import memory_limit_bytes, process_tree_rss_bytes
            while not done.wait(1):
                try:
                    controls(supervisor, root)
                    if process_tree_rss_bytes() > memory_limit_bytes():
                        raise RuntimeError("Cooperative worker memory guard")
                except (OSError, ValueError, RuntimeError) as error:
                    (runtime / "STOP").touch()
                    write(root / "intervention.json", {"reason": str(error), "action": "checkpoint_and_stop"})
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        try:
            for index, entry in enumerate(manifest["runs"]):
                controls(supervisor, root)
                config_path = verified(entry["config"])
                config = ConfigModel.load_config(config_path)
                output = Path(entry["run_dir"])
                output.mkdir(parents=True, exist_ok=True)
                runtime_path = output / "recovery-runtime.json"
                from exact.experiments.corrected_worker import runtime_record
                write(runtime_path, runtime_record(descriptor, entry, config, runtime=runtime, code=code))
                os.environ["EXACT_EXPERIMENT_RUNTIME"] = str(runtime_path)
                if descriptor["executor"] == "pinned_published_matcher":
                    from exact.experiments.published_matcher import run_cell
                    cell = SimpleNamespace(published_matcher=manifest["published_matcher"], source_cap=None,
                        seed=17, recovery={"root": str(runtime)}, config_hash=entry["config"]["sha256"],
                        output_dir=Path(entry["run_dir"]), resolved_config=config.model_dump(mode="json", by_alias=True))
                    code_value, _, _ = run_cell(cell, evaluate=False)
                else:
                    command = [sys.executable, "-m", "exact.delivery.cli.main", "-o", entry["run_dir"],
                        "-y", str(config_path), "-s", str(verified(manifest["source"])),
                        "-t", str(verified(manifest["target"]))]
                    log = root / f"run-{index}.log"
                    with log.open("ab") as stream:
                        process = subprocess.Popen(command, cwd=code, stdout=stream, stderr=subprocess.STDOUT)
                        while process.poll() is None:
                            time.sleep(1)
                        code_value = process.returncode
                if code_value:
                    raise RuntimeError(f"Corrected inference failed with exit {code_value}")
            outcome = "complete"
        finally:
            done.set()
            monitor_thread.join()
            after = usage(runtime / "openrouter")
            delta = {key: after[key] - before[key] for key in ("attempts", "billable_tokens", "reported_cost_usd")}
            cost = None if after["unpriced_attempts"] > before["unpriced_attempts"] else delta["reported_cost_usd"]
            ledger.finish(work, start=started, end=time.time(), status=outcome,
                requests=delta["attempts"], tokens=delta["billable_tokens"], actual_usd=cost)
            observation = state.get("allocations", {}).get(os.environ["SLURM_JOB_ID"])
            if observation:
                ledger.record_allocation(os.environ["SLURM_JOB_ID"], start=observation["start"], end=time.time())
            result = {"status": outcome, "descriptor": binding(descriptor_path), "cumulative_budget": str(ledger.path),
                "step_id": os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
                "scientific_acceptance": "pending_output_coverage_and_reporting"}
            write(root / "completion.json", result, immutable=True)
            status(root, outcome, cumulative_budget=str(ledger.path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--admission", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.descriptor, args.admission), sort_keys=True))


if __name__ == "__main__":
    main()
