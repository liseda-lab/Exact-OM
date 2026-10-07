"""Deterministic handoff of reviewed Slurm launch descriptors; no model invocation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from exact.experiments.supervision import _registry, inspect_runs, pending_batches


def _read(path):
    return json.loads(path.read_text())


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _validate(batch, allocation):
    launch = batch["launch"]
    argv = launch["argv"]
    if (
        not isinstance(argv, list) or len(argv) < 4
        or not all(isinstance(arg, str) and "\x00" not in arg for arg in argv)
        or argv[0] != "/usr/bin/srun" or argv[-2] != "/bin/bash"
        or not Path(argv[-1]).is_absolute()
    ):
        raise ValueError("Launch must use absolute srun and a bound bash worker, without inline shell")
    jobids = []
    for index, arg in enumerate(argv[1:-2], 1):
        if arg == "--jobid":
            jobids.append(argv[index + 1])
        elif arg.startswith("--jobid="):
            jobids.append(arg.split("=", 1)[1])
        if arg in {"--pty", "--multi-prog", "--test-only"}:
            raise ValueError("Interactive/multi-program/test-only dispatch is forbidden")
    if jobids != [allocation]:
        raise ValueError("Launch must name only the retained allocation")
    if launch["run"]["id"] != batch["id"]:
        raise ValueError("Pending and registered run IDs must match")
    if not isinstance(launch.get("nonce"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{12,128}", launch["nonce"]):
        raise ValueError("Launch requires a unique dispatch nonce")
    paths = [launch["step_path"], launch["launcher_log"], launch["tmux_socket"]]
    paths += [launch["run"][field] for field in ("status_path", "exit_path", "completion_path")]
    if any(not isinstance(path, str) or not Path(path).is_absolute() for path in paths):
        raise ValueError("Launch receipt paths must be absolute")
    verified = set()
    for item in launch["bindings"]:
        path = Path(item["path"])
        if not path.is_absolute() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Reviewed launch binding changed: " + str(path))
        verified.add(str(path))
    if argv[-1] not in verified:
        raise ValueError("Worker script is not a reviewed launch binding")
    return launch


def _tmux_server(launch):
    """Use only an existing server in this allocation's extern cgroup.

    setsid detaches a terminal, not a Slurm cgroup. The server must own srun so
    replacing the supervisor step cannot signal the scientific launcher's group.
    """
    command = ["/usr/bin/tmux", "-N", "-S", launch["tmux_socket"]]
    result = subprocess.run(command + ["display-message", "-p", "#{pid}"],
                            capture_output=True, text=True, timeout=10)
    if result.returncode or not result.stdout.strip().isdigit():
        raise ValueError("Existing detached tmux server unavailable; never create one inside supervisor")
    pid = int(result.stdout.strip())
    process = Path("/proc") / str(pid)
    if process.stat().st_uid != os.getuid():
        raise ValueError("Detached tmux server has another owner")
    own = Path("/proc/self/cgroup").read_text()
    server = (process / "cgroup").read_text()
    own_group = re.search(r"^0::(.+)/step_(?:[0-9]+|extern)/", own, re.MULTILINE)
    server_group = re.search(r"^0::(.+)/step_extern/", server, re.MULTILINE)
    if not own_group or not server_group or own_group.group(1) != server_group.group(1):
        raise ValueError("Detached tmux server must be in this retained allocation's extern cgroup")
    return command, pid


def _detached_wrapper(launch):
    """Persist exact reviewed argv and an independent launcher exit receipt."""
    log = Path(launch["launcher_log"])
    wrapper = log.with_name(log.name + ".dispatch.sh")
    exit_path = log.with_name(log.name + ".exit")
    quote = shlex.quote
    script = (
        "#!/bin/bash\n"
        + shlex.join(launch["argv"]) + " </dev/null >>" + quote(str(log)) + " 2>&1\n"
        + "result=$?\nprintf '%s\\n' \"$result\" >" + quote(str(exit_path) + ".tmp") + "\n"
        + "mv " + quote(str(exit_path) + ".tmp") + " " + quote(str(exit_path)) + "\n"
        + 'exit "$result"\n'
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    if wrapper.exists() and wrapper.read_text() != script:
        raise ValueError("Detached launcher wrapper changed; inspect prior owner")
    if exit_path.exists():
        raise ValueError("Fresh detached launcher already has an exit receipt")
    with wrapper.open("w") as stream:
        stream.write(script)
        stream.flush()
        os.fsync(stream.fileno())
    return wrapper, exit_path


def _step(launch, allocation, steps):
    path = Path(launch["step_path"])
    if not path.exists():
        return None
    receipt = _read(path)
    step = receipt.get("step_id", "")
    if receipt.get("dispatch_nonce") != launch["nonce"] or not re.fullmatch(re.escape(allocation) + r"\.\d+", step):
        raise ValueError("Step receipt does not match its retained allocation and dispatch nonce")
    if step in steps:
        return step
    # A short job can finish between ticks. Require matching terminal evidence,
    # rather than accepting old exit/status files from an unrelated execution.
    complete = Path(launch["run"]["completion_path"])
    exit_path = Path(launch["run"]["exit_path"])
    if complete.exists() and exit_path.exists():
        evidence = _read(complete)
        if (evidence.get("step_id") == step and evidence.get("dispatch_nonce") == launch["nonce"]
                and evidence.get("status") in {"complete", "failed", "interrupted"}
                and int(exit_path.read_text().strip()) == evidence.get("exit_code")):
            return step
    return None


def _recovery_predecessor(registry, batch):
    parents = [run for run in registry["runs"] if run.get("pending_recovery") == batch["id"]]
    if len(parents) > 1:
        raise ValueError("Queued recovery must have a single predecessor")
    parent = parents[0] if parents else None
    declared = batch.get("recovery_for")
    if declared is not None and (parent is None or declared != parent["id"]):
        raise ValueError("Queued recovery_for must match its predecessor's pending_recovery")
    if parent is not None and parent.get("superseded_by") is not None:
        raise ValueError("Queued recovery predecessor already has a registered successor")
    return parent


def _register(registry, batch, step):
    if any(run["id"] == batch["id"] for run in registry["runs"]):
        raise ValueError("Pending launch duplicates a registered run ID")
    parent = _recovery_predecessor(registry, batch)
    runs = [{**run} for run in registry["runs"]]
    if parent is not None:
        predecessor = next(run for run in runs if run["id"] == parent["id"])
        predecessor.update(enabled=False, superseded_by=batch["id"], retain_accounting=True)
        predecessor.pop("pending_recovery")
    runs.append({
        **batch["launch"]["run"], "step_id": step, "enabled": True,
        "depends_on": batch.get("depends_on", []), "resources": batch.get("resources", {}),
        "dispatch_nonce": batch["launch"]["nonce"],
    })
    # Validate the entire transition before mutating even the in-memory registry.
    # The caller publishes registration and lineage together under the same lock.
    _registry(runs)
    registry["runs"] = runs
    registry["pending_batches"] = [row for row in registry["pending_batches"] if row["id"] != batch["id"]]
    for path in batch["launch"].get("pause_paths", []):
        if path not in registry.setdefault("pause_paths", []):
            registry["pause_paths"].append(path)


def dispatch_ready(directory, allocation, steps, *, supervisor_step=None, validate_launch=None):
    """Reserve, start or reconcile at most one prepared batch under the registry lock.

    An uncertain prior spawn is never repeated. Its receipt can be recovered after
    controller restart; absent proof it becomes a normal bounded repair incident.
    """
    directory = Path(directory)
    if not re.fullmatch(r"\d+", allocation) or not any(key.startswith(allocation + ".") for key in steps):
        raise ValueError("Retained allocation unavailable; dispatch cannot create allocations")
    path = directory / "registry.json"
    with (directory / "registry.json.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "registry_busy"}
        registry = _read(path)
        if any(Path(p).exists() for p in [directory / "PAUSE", directory / "STOP", *registry.get("pause_paths", [])]):
            return {"status": "paused"}
        state_path = directory / "dispatch-state.json"
        state = _read(state_path) if state_path.exists() else {}
        batches = {row["id"]: row for row in registry.get("pending_batches", []) if row.get("launch")}
        now = time.time()
        # Resolve reservations first. A controller crash between registry/state
        # writes is harmless: registration is authoritative and is never repeated.
        for name, record in state.items():
            if record["status"] in {"registered", "resolved"}:
                continue
            existing = next((run for run in registry["runs"] if run["id"] == name), None)
            if existing:
                if existing.get("dispatch_nonce") == record.get("nonce"):
                    record.update(status="registered", step_id=existing["step_id"])
                else:
                    record.update(status="failed", error="Registered run conflicts with dispatch reservation")
                _write(state_path, state)
                continue
            if name not in batches:
                record.update(status="failed", error="Reserved batch removed before verified registration")
                _write(state_path, state)
                continue
            batch = batches[name]
            try:
                if _identity(batch) != record["descriptor_sha256"]:
                    raise ValueError("Reserved launch descriptor changed; inspect prior owner")
                step = _step(batch["launch"], allocation, steps)
                if step:
                    _register(registry, batch, step)
                    _write(path, registry)
                    record.update(status="registered", step_id=step)
                    _write(state_path, state)
                    return {"status": "registered", "batch_id": name, "step_id": step}
                launcher_exit = Path(record["launcher_exit_path"]) if record.get("launcher_exit_path") else None
                if launcher_exit and launcher_exit.exists():
                    record["launcher_exit_code"] = int(launcher_exit.read_text().strip())
                    raise ValueError("Launcher exited without a verified step/terminal receipt")
                if now - record["reserved_epoch"] > 90:
                    raise ValueError("Launch has no verified receipt after 90s; inspect before any retry")
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                record.update(status="failed", error=str(exc))
                _write(state_path, state)
        # An unresolved launch owns its requested resources even before Slurm
        # reports the step. Do not admit another worker into the same capacity.
        observation = inspect_runs(registry["runs"], step_states=steps)
        reserved = {}
        for record in state.values():
            if record["status"] in {"reserved", "starting", "failed"} and record.get("may_have_started", True):
                for key, value in record.get("resources", {}).items():
                    reserved[key] = reserved.get(key, 0) + value
        capacity = {key: value - reserved.get(key, 0) for key, value in registry.get("capacity", {}).items()}
        ready = {incident["batch_id"] for incident in pending_batches({**registry, "capacity": capacity}, observation)}
        blocked = None
        for name, batch in batches.items():
            if name not in ready or name in state:
                continue
            if batch.get("resources", {}).get("gpus", 0):
                known = {run["step_id"] for run in registry["runs"]}
                exempt = {allocation + ".0", allocation + ".extern", allocation + ".batch"}
                if supervisor_step:
                    exempt.add(allocation + "." + str(supervisor_step))
                if set(steps) - known - exempt:
                    blocked = {"status": "unregistered_steps_present", "steps": sorted(set(steps) - known - exempt)}
                    continue
                if any(run["step_id"] in steps and run.get("resources", {}).get("gpus", 1)
                       for run in registry["runs"]):
                    blocked = {"status": "gpu_busy"}
                    continue
            record = {
                "status": "reserved", "reserved_epoch": now,
                "descriptor_sha256": _identity(batch),
                "resources": batch.get("resources", {}),
                "nonce": batch["launch"].get("nonce"),
                "may_have_started": False,
            }
            state[name] = record
            _write(state_path, state)  # Durable reservation precedes any process creation.
            try:
                _recovery_predecessor(registry, batch)
                launch = _validate(batch, allocation)
                if validate_launch is not None:
                    validate_launch(launch)
                if Path(launch["step_path"]).exists():
                    raise ValueError("Fresh dispatch already has a step receipt; inspect prior owner")
                command, server_pid = _tmux_server(launch)
                wrapper, launcher_exit = _detached_wrapper(launch)
                session = "exact-dispatch-" + launch["nonce"]
                record.update(
                    may_have_started=True, detached_session=session, tmux_server_pid=server_pid,
                    launcher_exit_path=str(launcher_exit), wrapper_path=str(wrapper),
                    wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                )
                _write(state_path, state)
                result = subprocess.run(
                    command + ["new-session", "-d", "-s", session, "/bin/bash", str(wrapper)],
                    stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
                )
                if result.returncode:
                    raise ValueError("Detached launch failed: " + result.stderr.strip())
                record.update(status="starting")
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                record.update(status="failed", error=str(exc))
            _write(state_path, state)
            return {"status": record["status"], "batch_id": name}
        return blocked or {"status": "no_ready_launch"}


def pending_recoveries(registry, observation, allocation, steps, state):
    """Recognize verified queued recovery chains without claiming science is ready."""
    runs = {row["id"]: row for row in registry["runs"]}
    batches = {row["id"]: row for row in registry.get("pending_batches", [])}
    findings = {row["run_id"]: row for row in observation["findings"]}

    def replacement(run, seen):
        identifier = run.get("pending_recovery")
        return (
            isinstance(identifier, str) and identifier in batches and identifier not in runs
            and run["step_id"] not in steps and viable(identifier, seen)
        )

    def viable(identifier, seen):
        if not isinstance(identifier, str) or identifier in seen:
            return False
        seen = seen | {identifier}
        if identifier in runs:
            run = runs[identifier]
            successor = run.get("superseded_by")
            if successor:
                return successor in runs and viable(successor, seen)
            status = findings.get(identifier, {}).get("status")
            if status in {"healthy", "waiting", "complete"}:
                return True
            return status in {"needs_attention", "failed"} and replacement(run, seen)
        batch = batches.get(identifier)
        if (not batch or not batch.get("enabled", True) or batch.get("needs_user")
                or not batch.get("launch")):
            return False
        launch = _validate(batch, allocation)
        resources = batch["resources"]
        if not isinstance(resources, dict) or not resources or any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not 0 <= value <= registry["capacity"].get(key, 0)
            for key, value in resources.items()
        ):
            return False
        record = state.get(identifier)
        if record is not None and (
            record.get("status") not in {"reserved", "starting"}
            or record.get("nonce") != launch["nonce"]
            or record.get("descriptor_sha256") != _identity(batch)
            or record.get("resources") != resources
        ):
            return False
        parents = batch.get("depends_on", [])
        return isinstance(parents, list) and all(viable(parent, seen) for parent in parents)

    waiting = {}
    for name, run in runs.items():
        try:
            if replacement(run, {name}):
                waiting[name] = run["pending_recovery"]
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            # Broken/cyclic chains leave the original failure actionable.
            continue
    return waiting


def dispatch_incidents(directory):
    path = Path(directory) / "dispatch-state.json"
    state = _read(path) if path.exists() else {}
    incidents = [{
        "id": _identity(["dispatch", name, row["descriptor_sha256"]])[:24],
        "kind": "dispatch_failed", "run_ids": [name], "batch_id": name,
        "reason": row.get("error", "Prepared launch requires inspection"),
        "dispatch_state": str(path),
    } for name, row in state.items() if row["status"] == "failed"]
    status_path = Path(directory) / "dispatch-status.json"
    status = _read(status_path) if status_path.exists() else {}
    if status.get("status") in {"dispatch_error", "unregistered_steps_present"}:
        detail = status.get("error") or status.get("steps")
        incidents.append({
            "id": _identity(["dispatch_controller", detail])[:24],
            "kind": "dispatch_blocked", "run_ids": [],
            "reason": "Prepared handoff requires inspection: " + str(detail),
            "dispatch_state": str(status_path),
        })
    return incidents
