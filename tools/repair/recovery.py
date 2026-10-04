"""Bind a queued replacement to its real worker receipt under the dispatcher lock."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from exact.experiments.dispatch import _read, _register, _step, _validate, _write
from exact.experiments.science_health import inspect_science


def link_recovery(directory, run_id, *, steps, step_id):
    """Idempotently link only a nonce-bound live replacement; never launch work."""
    directory = Path(directory)
    with (directory / "registry.json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = _read(directory / "registry.json")
        runs = {row["id"]: row for row in registry["runs"]}
        batch = next((b for b in registry["pending_batches"] if b["id"] == run_id), None)
        run = runs.get(run_id)
        if batch is not None:
            launch = _validate(batch, step_id.split(".")[0])
            if _step(launch, step_id.split(".")[0], steps) != step_id:
                raise ValueError("Recovery has no matching live step receipt")
            candidate = launch["run"]
        elif run is not None and run["step_id"] == step_id and step_id in steps:
            candidate = run
            receipt = _read(Path(run["status_path"]).with_name("step.json"))
            if receipt != dict(step_id=step_id, dispatch_nonce=run["dispatch_nonce"]):
                raise ValueError("Registered recovery nonce does not match its receipt")
        else:
            raise ValueError("Recovery is neither queued nor registered for this step")
        prior = runs[candidate["recovery_of"]]
        if prior["step_id"] in steps or step_id not in steps:
            raise ValueError("Prior worker is live or replacement disappeared")
        if candidate["logical_id"] != prior["logical_id"]:
            raise ValueError("Recovery changed logical job")
        continuation = candidate.get("recovery_kind") == "operational_continuation"
        if not continuation and not 1 <= candidate["repair_attempt"] <= candidate["max_repairs"] <= 2:
            raise ValueError("Recovery repair limit exceeded")
        receipt = Path(prior["completion_path"])
        if hashlib.sha256(receipt.read_bytes()).hexdigest() != candidate["recovery_completion_sha256"]:
            raise ValueError("Failed predecessor evidence changed")
        completion = _read(receipt)
        if continuation:
            from tools.repair.expanded_corpus import bound
            from tools.repair.scaling_endpoint_continuation import CAUSE, SCHEMA

            plan = bound(candidate["continuation_plan"])
            if (candidate.get("repair_attempt") != 0 or candidate.get("max_repairs") != 2
                    or type(candidate.get("continuation_attempt")) is not int
                    or candidate["continuation_attempt"] < 1
                    or plan.get("schema") != SCHEMA
                    or plan.get("recovery_kind") != "operational_continuation"
                    or plan.get("prior_costs_reset") is not False
                    or plan.get("scientific_budgets_changed") is not False
                    or plan.get("completion") != dict(path=str(receipt), sha256=candidate["recovery_completion_sha256"])
                    or plan.get("original_step") != prior["step_id"]
                    or completion.get("status") != "failed" or completion.get("error") != CAUSE):
                raise ValueError("Operational continuation requires bound timeout evidence and unchanged scientific limits")
        if completion.get("status") != "failed":
            # Successful orchestration can contain failed native rows. This
            # exception requires explicit, byte-bound scanner evidence; a model
            # claim or an ordinary completed result never authorizes replay.
            expected = candidate.get("recovery_scientific_failure")
            scientific = inspect_science(prior, completion)
            if (completion.get("status") != "complete" or not isinstance(expected, dict)
                    or scientific["errors"] or expected not in scientific["failures"]):
                raise ValueError("Completed predecessor has no matching qualified scientific failure")
        if prior.get("superseded_by") not in {None, run_id}:
            raise ValueError("Another replacement already owns the predecessor")
        if prior.get("superseded_by") is None and prior.get("pending_recovery") != run_id:
            raise ValueError("Replacement was not queued for this predecessor")
        if batch is not None:
            _register(registry, batch, step_id)
        prior.update(enabled=False, superseded_by=run_id)
        prior.pop("pending_recovery", None)
        _write(directory / "registry.json", registry)
        return dict(status="linked", run_id=run_id, recovery_of=prior["id"], step_id=step_id)


def main():
    directory, run_id = sys.argv[1:]
    allocation = os.environ["SLURM_JOB_ID"]
    step_id = allocation + "." + os.environ["SLURM_STEP_ID"]
    result = subprocess.run(
        ["squeue", "--steps", "--noheader", "--jobs", allocation, "--format=%i"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if "error:" in result.stderr.lower():
        raise RuntimeError(result.stderr)
    steps = dict.fromkeys(result.stdout.split(), "RUNNING")
    print(json.dumps(link_recovery(directory, run_id, steps=steps, step_id=step_id)))


if __name__ == "__main__":
    main()
