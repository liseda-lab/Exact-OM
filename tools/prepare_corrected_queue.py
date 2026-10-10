#!/usr/bin/env python3
"""Export reviewed corrected workers as an offline supervisor queue proposal.

Never writes registry.json. Rows remain disabled until an explicitly authorized
publisher rechecks this proposal's registry and dispatch-state snapshots under
registry.json.lock. Existing supervisor polling needs no restart.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import uuid

from exact.experiments.mixed_scale import binding, read_binding, verified
from exact.utils.fitted_artifacts import freeze_json


def prepare(bundle_path, destination, *, admissions=None):
    bundle = json.loads(Path(bundle_path).read_text())
    registry_path = verified(bundle["registry_snapshot"])
    registry = json.loads(registry_path.read_text())
    old = bundle["pending_rows_to_replace"]
    old_ids = {row["id"] for row in old}
    if (not old_ids or any(row["id"] in old_ids for row in registry.get("runs", []))
            or [row for row in registry.get("pending_batches", []) if row["id"] in old_ids] != old):
        raise ValueError("Historical pending rows changed or already started")
    dispatch_path = registry_path.parent / "dispatch-state.json"
    dispatch_snapshot = binding(dispatch_path) if dispatch_path.exists() else None
    dispatch = json.loads(dispatch_path.read_text()) if dispatch_path.exists() else {}
    if old_ids & set(dispatch):
        raise ValueError("Historical dispatch is started or uncertain; reconcile before replacement")
    rows, admissions = [], admissions or {}
    if set(admissions) - {row["id"] for row in bundle["logical_to_physical"]}:
        raise ValueError("Admission names an undeclared corrected cell")
    parents = sorted({name for row in old for name in row.get("depends_on", [])} - old_ids)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for cell in bundle["logical_to_physical"]:
        identifier = "corrected-" + cell["id"].replace("/", "--")
        row = {"id": identifier, "enabled": False, "needs_user": True, "depends_on": parents,
               "logical_cell": cell["id"], "resources": {"cpus": 0, "gpus": 0, "memory_mb": 0}}
        if "worker" in cell:
            row["worker"] = cell["worker"]
        if cell["id"] in admissions:
            if "worker" not in cell:
                raise ValueError("Cannot prepare launch for an unbound scientific cell")
            admission = read_binding(admissions[cell["id"]])
            from tools.run_corrected_cell import admission_environment
            policy = read_binding(admission["supervisor_policy"])
            if (admission.get("descriptor") != cell["worker"]
                    or admission.get("kind") != "corrected_worker_rollout_admission"
                    or admission.get("rollout_authorized") is not True
                    or admission.get("run_id") != identifier
                    or admission.get("commit") != bundle["source_revision"]
                    or admission.get("scientific_step") != "E17"):
                raise ValueError("Launch requires a reviewed cell-specific corrected rollout admission")
            _, spending = admission_environment(admission, policy)
            root, code = Path(admission["root"]).resolve(), Path(admission["code_root"]).resolve()
            root.mkdir(parents=True, exist_ok=True)
            python = Path(admission["python"]).resolve()
            nonce = uuid.uuid4().hex
            worker, step = root / "worker-entry.sh", root / "step.json"
            content = ("#!/usr/bin/env bash\nset -euo pipefail\n" +
                f"cd {shlex.quote(str(code))}\n" +
                f"trap 'result=$?; printf \"%s\\n\" \"$result\" > {shlex.quote(str(root / 'exit-code'))}' EXIT\n" +
                f"printf '{{\"step_id\":\"%s.%s\",\"dispatch_nonce\":\"{nonce}\"}}\\n' \"$SLURM_JOB_ID\" \"$SLURM_STEP_ID\" > {shlex.quote(str(step))}\n" +
                f"{shlex.quote(str(python))} -m tools.run_corrected_cell --descriptor {shlex.quote(cell['worker']['path'])} --admission {shlex.quote(admissions[cell['id']]['path'])}\n")
            with worker.open("x") as stream:
                stream.write(content)
            resources = admission["resources"]
            launch = {"nonce": nonce, "tmux_socket": f"/tmp/tmux-{os.getuid()}/default",
                "argv": ["/usr/bin/srun", "--jobid=" + str(policy["allocation"]), "--overlap", "--immediate=15",
                    "--nodes=1", "--ntasks=1", "--cpus-per-task=" + str(resources["cpus"]),
                    "--mem=" + str(resources["memory_mb"]), "--gres=" + admission["gres"], "--time=0",
                    "/bin/bash", str(worker)],
                "bindings": [binding(worker), cell["worker"], admissions[cell["id"]], binding(code / "tools/run_corrected_cell.py")],
                "pause_paths": [str(root / "STOP"), str(root / "runtime/STOP")],
                "step_path": str(step), "launcher_log": str(root / "launcher.log"),
                "run": {"id": identifier, "status_path": str(root / "status.json"),
                    "exit_path": str(root / "exit-code"), "completion_path": str(root / "completion.json"),
                    "hosted_scope": {"campaign_id": spending["policy"]["campaign_id"], "experiment_id": "E17"}}}
            from tools.storage_guard import guard_launch as storage_guard
            from tools.hosted_prompt_guard import guard_launch as hosted_guard
            launch = storage_guard(launch, policy, registry_path.parent)
            launch = hosted_guard(launch, policy, recipe_path=verified(admissions[cell["id"]]),
                                  receipt_path=root / "hosted-prompt-guard.json")
            row.update(launch=launch, resources=resources)
            for key in ("gpu_devices", "resource_profile"):
                if key in admission:
                    row[key] = admission[key]
            from exact.experiments.dispatch import _validate, _validate_resource_binding
            _validate(row, str(policy["allocation"]))
            _validate_resource_binding(registry, row)
        rows.append(row)
        parents = [identifier]  # One heavy worker; complete every predecessor before the next.
    if binding(registry_path) != bundle["registry_snapshot"]:
        raise ValueError("Registry changed during offline queue preparation")
    if (binding(dispatch_path) if dispatch_path.exists() else None) != dispatch_snapshot:
        raise ValueError("Dispatch status changed during offline queue preparation")
    return freeze_json(destination / "queue-proposal.json", {
        "kind": "offline_corrected_queue_proposal", "bundle": binding(bundle_path),
        "registry_snapshot": bundle["registry_snapshot"],
        "dispatch_snapshot": dispatch_snapshot,
        "replace_only_pending_ids": sorted(old_ids), "proposed_pending_batches": rows,
        "launchable": False, "live_queue_changed": False, "registry_lock": str(registry_path) + ".lock",
        "remaining_rollout_actions": ["complete_all_scientific_and_operational_gates", "explicit_rollout_authorization",
            "recheck_registry_and_dispatch_snapshots_under_lock", "enable_reviewed_rows_and_replace_only_unstarted_E17_rows"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--admissions", type=Path, help="Optional reviewed cell ID to admission binding JSON map")
    args = parser.parse_args()
    print(json.dumps(prepare(args.bundle, args.output,
        admissions=json.loads(args.admissions.read_text()) if args.admissions else None), sort_keys=True))


if __name__ == "__main__":
    main()
