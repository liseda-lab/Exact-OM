#!/usr/bin/env python3
"""Guard a detached worker's storage without changing its frozen numerical code."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

GIB = 1024**3


def tree_bytes(root):
    """Conservative allocated/apparent bytes, counting hard-linked inodes once."""
    seen, total = set(), 0

    def inaccessible(error):
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=inaccessible):
        for name in [".", *dirs, *files]:
            try:
                info = (Path(directory) / name).lstat()
            except FileNotFoundError:
                continue  # Atomic checkpoint replacement during the walk.
            identity = info.st_dev, info.st_ino
            if identity not in seen:
                seen.add(identity)
                total += max(info.st_size, info.st_blocks * 512)
    return total


def check_storage(root, *, min_free, max_used, growth_reserve, used=None):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Storage usage root does not exist: {root}")
    free = shutil.disk_usage(root).free
    if free < min_free:
        raise ValueError(f"Storage free reserve: {free} bytes available; {min_free} required")
    if used is None:
        used = tree_bytes(root)
    if used + growth_reserve > max_used:
        raise ValueError(
            f"Storage usage ceiling: {used} bytes plus {growth_reserve} growth reserve "
            f"exceeds {max_used}: {root}"
        )
    return used


def pause(paths, reason):
    for path in paths:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x") as stream:
                stream.write("Storage safety guard: " + reason + "\n")
        except FileExistsError:
            pass  # Never overwrite an existing user's pause or STOP reason.


def stop_owned_worker(process, *, grace_seconds=30):
    """Interrupt the scientific CLI first so its accounting parent can settle."""
    try:
        owner = psutil.Process(process.pid)
        descendants = owner.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for child in descendants:
        try:
            if "exact.delivery.cli.main" in child.cmdline():
                child.send_signal(signal.SIGINT)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    # Emergency escalation affects only descendants of the wrapper's own Popen.
    # Individual PID/create-time checked psutil handles also cover new sessions.
    try:
        owned = [*reversed(owner.children(recursive=True)), owner]
    except psutil.NoSuchProcess:
        owned = descendants
    for child in owned:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    process.wait()


def _diagnostic(message):
    try:
        print(message, file=sys.stderr, flush=True)
    except OSError:
        pass  # The launcher's log may share the exhausted/read-only filesystem.


def deduplicate_completed(helper, batch_root):
    """Optional maintenance after the scientific worker has settled and exited."""
    if helper is None:
        return
    try:
        subprocess.run(
            [
                sys.executable,
                str(helper),
                str(batch_root),
                "--apply",
                "--receipt",
                str(Path(batch_root) / "storage-dedup.json"),
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        # Originals remain valid; an optional reclamation failure is not a failed run.
        _diagnostic("Completed fitting deduplication skipped: " + str(error))


def publish_step_receipt(path, nonce):
    """Identify this real Slurm guard before a potentially slow admission scan."""
    if path is None and nonce is None:
        return
    if (
        path is None
        or not Path(path).is_absolute()
        or not isinstance(nonce, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{12,128}", nonce)
    ):
        raise ValueError("Step receipt requires an absolute path and valid dispatch nonce")
    job, step = os.getenv("SLURM_JOB_ID", ""), os.getenv("SLURM_STEP_ID", "")
    if not re.fullmatch(r"[0-9]+", job) or not re.fullmatch(r"[0-9]+", step):
        raise ValueError("Step receipt requires a numeric Slurm job and step")
    cgroup = Path("/proc/self/cgroup").read_text()
    if not re.search(r"(?:^|/)step_" + re.escape(step) + r"(?:/|$)", cgroup, re.MULTILINE):
        raise ValueError("Step receipt requires the matching Slurm step cgroup")
    path = Path(path)
    payload = {"step_id": job + "." + step, "dispatch_nonce": nonce}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".step-receipt-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or json.loads(path.read_text()) != payload:
                raise ValueError("Existing Slurm step receipt conflicts with this dispatch")
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run(
    command,
    *,
    root,
    pause_paths,
    min_free=100 * GIB,
    max_used=600 * GIB,
    growth_reserve=16 * GIB,
    interval=5,
    scan_interval=60,
    dedup_helper=None,
    completed_run_root=None,
    step_path=None,
    dispatch_nonce=None,
):
    if not command or min(min_free, max_used, growth_reserve, interval, scan_interval) < 0:
        raise ValueError("A worker command and nonnegative storage limits are required")
    if not interval or not scan_interval or not max_used:
        raise ValueError("Storage ceiling and polling intervals must be positive")
    if (dedup_helper is None) != (completed_run_root is None):
        raise ValueError("Completed fitting deduplication requires helper and batch root")
    process = None
    try:
        publish_step_receipt(step_path, dispatch_nonce)
        used = check_storage(
            root, min_free=min_free, max_used=max_used, growth_reserve=growth_reserve
        )
        env = {**os.environ, "EXACT_STORAGE_MIN_FREE_BYTES": str(min_free)}
        process = subprocess.Popen(command, env=env, start_new_session=True)
        scanned = time.monotonic()
        while True:
            try:
                result = process.wait(timeout=interval)
                if result == 0:
                    deduplicate_completed(dedup_helper, completed_run_root)
                return result
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                rescan = now - scanned >= scan_interval
                used = check_storage(
                    root,
                    min_free=min_free,
                    max_used=max_used,
                    growth_reserve=growth_reserve,
                    used=None if rescan else used,
                )
                if rescan:
                    scanned = now
    except (OSError, ValueError) as error:
        try:
            _diagnostic("STORAGE GUARD paused: " + str(error))
            try:
                pause(pause_paths, str(error))
            except OSError as pause_error:
                _diagnostic("Could not persist storage pause: " + str(pause_error))
        finally:
            if process is not None:
                stop_owned_worker(process)
        return 75
    except BaseException:
        if process is not None:
            stop_owned_worker(process)
        raise


def _binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _policy_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def validate_policy(config):
    """A reviewed immutable guard program and explicit filesystem boundaries."""
    if not isinstance(config, dict):
        raise ValueError("Storage guard policy must be a mapping")
    for key in ("python", "usage_root"):
        if not isinstance(config.get(key), str) or not Path(config[key]).is_absolute():
            raise ValueError("Storage guard policy requires absolute " + key)
    for key in ("min_free_bytes", "max_used_bytes", "growth_reserve_bytes"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError("Storage guard policy requires positive " + key)
    if config["growth_reserve_bytes"] >= config["max_used_bytes"]:
        raise ValueError("Storage growth reserve must be below usage ceiling")
    source = config.get("source", {})
    if (
        not isinstance(source.get("path"), str)
        or not Path(source["path"]).is_absolute()
        or _binding(source["path"]) != source
    ):
        raise ValueError("Storage guard source binding changed")
    if "completed_fitting_dedup" in config:
        retention = config["completed_fitting_dedup"]
        if not isinstance(retention, dict) or not isinstance(retention.get("source"), dict):
            raise ValueError("Completed fitting deduplication requires a source binding")
        helper = retention["source"]
        if (
            not isinstance(helper.get("path"), str)
            or Path(helper["path"]).parent != Path(source["path"]).parent
            or _binding(helper["path"]) != helper
        ):
            raise ValueError("Completed fitting deduplication source binding changed")


def _wrapper(
    config, supervisor, worker, stop_path, completion_path=None, step_path=None, dispatch_nonce=None
):
    command = [
        config["python"],
        "-u",
        config["source"]["path"],
        "--usage-root",
        config["usage_root"],
        "--pause-path",
        str(Path(supervisor).resolve() / "PAUSE"),
        "--pause-path",
        stop_path,
    ]
    for option in ("min_free_bytes", "max_used_bytes", "growth_reserve_bytes"):
        command.extend(["--" + option.replace("_", "-"), str(config[option])])
    if "completed_fitting_dedup" in config:
        if (
            not isinstance(completion_path, str)
            or not Path(completion_path).is_absolute()
            or Path(completion_path).name != "completion.json"
        ):
            raise ValueError("Completed fitting deduplication requires bound completion.json")
        command.extend(
            [
                "--dedup-helper",
                config["completed_fitting_dedup"]["source"]["path"],
                "--completed-run-root",
                str(Path(completion_path).parent),
            ]
        )
    if step_path is not None or dispatch_nonce is not None:
        if (
            not isinstance(step_path, str)
            or not Path(step_path).is_absolute()
            or not isinstance(dispatch_nonce, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{12,128}", dispatch_nonce)
        ):
            raise ValueError("Storage guard requires bound Slurm receipt metadata")
        command.extend(["--step-path", step_path, "--dispatch-nonce", dispatch_nonce])
    command.extend(["--", "/bin/bash", worker])
    return "#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n"


def guard_launch(launch, policy, supervisor):
    """Wrap a not-yet-dispatched descriptor; preserve all original worker bindings."""
    config = policy.get("storage_guard")
    if config is None:
        return launch
    validate_policy(config)
    if launch.get("storage_guard"):
        validate_launch(launch, policy, supervisor)
        return launch
    launch = copy.deepcopy(launch)
    worker = launch["argv"][-1]
    original = _binding(worker)
    if original not in launch["bindings"]:
        raise ValueError("Storage guard cannot wrap an unbound or changed original worker")
    stops = launch.get("pause_paths", [])
    if not stops or not all(Path(path).is_absolute() for path in stops):
        raise ValueError("Storage guard requires a registered absolute runtime STOP path")
    stop_path = stops[-1]
    wrapper = Path(worker).with_name(
        Path(worker).stem + ".storage-guard-" + _policy_hash(config)[:12] + ".sh"
    )
    content = _wrapper(
        config,
        supervisor,
        worker,
        stop_path,
        launch.get("run", {}).get("completion_path"),
        launch.get("step_path"),
        launch.get("nonce"),
    )
    if wrapper.exists():
        if wrapper.read_text() != content:
            raise ValueError("Immutable storage guard wrapper changed: " + str(wrapper))
    else:
        with wrapper.open("x") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    launch["argv"][-1] = str(wrapper)
    for binding in (_binding(wrapper), config["source"]):
        if binding not in launch["bindings"]:
            launch["bindings"].append(binding)
    if "completed_fitting_dedup" in config:
        launch["bindings"].append(config["completed_fitting_dedup"]["source"])
    launch["storage_guard"] = {
        "worker": original,
        "stop_path": stop_path,
        "policy_sha256": _policy_hash(config),
    }
    validate_launch(launch, policy, supervisor)
    return launch


def validate_launch(launch, policy, supervisor):
    """Fail closed under the dispatch lock if a future descriptor bypasses the guard."""
    config = policy.get("storage_guard")
    if config is None:
        return
    validate_policy(config)
    record = launch.get("storage_guard")
    if not isinstance(record, dict) or record.get("policy_sha256") != _policy_hash(config):
        raise ValueError("Prepared launch lacks the required reviewed storage guard policy")
    original = record.get("worker", {})
    if (
        not isinstance(original.get("path"), str)
        or original not in launch["bindings"]
        or _binding(original["path"]) != original
        or config["source"] not in launch["bindings"]
    ):
        raise ValueError("Storage guard source or original worker lacks a valid binding")
    if (
        "completed_fitting_dedup" in config
        and config["completed_fitting_dedup"]["source"] not in launch["bindings"]
    ):
        raise ValueError("Completed fitting deduplication helper lacks a valid binding")
    stop = record.get("stop_path")
    if (
        not isinstance(stop, str)
        or not Path(stop).is_absolute()
        or stop not in launch.get("pause_paths", [])
    ):
        raise ValueError("Storage guard runtime STOP is not a registered control")
    wrapper = Path(launch["argv"][-1])
    if _binding(wrapper) not in launch["bindings"] or wrapper.read_text() != _wrapper(
        config,
        supervisor,
        original["path"],
        stop,
        launch.get("run", {}).get("completion_path"),
        launch.get("step_path"),
        launch.get("nonce"),
    ):
        raise ValueError("Prepared worker does not execute the required storage guard")


def pause_incident(paths):
    """An emergency storage pause needs attention even while ordinary pauses are silent."""
    # Prefer the global marker so duplicate runtime STOP files share one episode.
    for path in sorted(map(Path, paths), key=lambda item: (item.name != "PAUSE", str(item))):
        try:
            with path.open() as stream:
                reason = stream.read(8192).strip()
                stat = os.fstat(stream.fileno())
        except FileNotFoundError:
            continue
        if reason.startswith("Storage safety guard: "):
            episode = [reason, str(path.resolve()), stat.st_ino, stat.st_mtime_ns]
            return {
                "id": hashlib.sha256(json.dumps(episode).encode()).hexdigest()[:24],
                "kind": "storage_safety",
                "run_ids": [],
                "reason": reason,
            }
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usage-root", type=Path, required=True)
    parser.add_argument("--pause-path", type=Path, action="append", required=True)
    parser.add_argument("--min-free-bytes", type=int, default=100 * GIB)
    parser.add_argument("--max-used-bytes", type=int, default=600 * GIB)
    parser.add_argument("--growth-reserve-bytes", type=int, default=16 * GIB)
    parser.add_argument("--dedup-helper", type=Path)
    parser.add_argument("--completed-run-root", type=Path)
    parser.add_argument("--step-path", type=Path)
    parser.add_argument("--dispatch-nonce")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Storage guard received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    raise SystemExit(
        run(
            command,
            root=args.usage_root,
            pause_paths=args.pause_path,
            min_free=args.min_free_bytes,
            max_used=args.max_used_bytes,
            growth_reserve=args.growth_reserve_bytes,
            dedup_helper=args.dedup_helper,
            completed_run_root=args.completed_run_root,
            step_path=args.step_path,
            dispatch_nonce=args.dispatch_nonce,
        )
    )


if __name__ == "__main__":
    main()
