"""Allocation-aware cooperative guards for experiment workers and finalization."""

from __future__ import annotations

import os
import threading
from pathlib import Path


def memory_limit_bytes():
    """Respect the actual allocation/cgroup with headroom for the host."""
    import psutil

    caps = [psutil.virtual_memory().total]
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or os.cpu_count() or 1
    for field, multiplier in (("SLURM_MEM_PER_NODE", 1), ("SLURM_MEM_PER_CPU", cpus)):
        value = os.environ.get(field, "")
        if value.isdigit() and int(value) > 0:
            caps.append(int(value) * multiplier * 1024**2)
    try:
        group = next(
            line.split(":", 2)[2]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        directory = Path("/sys/fs/cgroup") / group.lstrip("/")
        while directory.is_relative_to("/sys/fs/cgroup"):
            path = directory / "memory.max"
            value = path.read_text().strip() if path.is_file() else "max"
            if value.isdigit():
                caps.append(int(value))
            directory = directory.parent
    except (OSError, StopIteration):
        pass
    capacity = min(caps)
    return max(1, int(capacity - max(2 * 1024**3, capacity * 0.10)))


def process_tree_rss_bytes():
    import psutil

    process = psutil.Process()
    total = 0
    for item in [process, *process.children(recursive=True)]:
        try:
            total += item.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def guarded_execute(campaign, wave, code_root, *, check_pause=None):
    """Finalize with allocation headroom, cooperative STOP and frozen request caps."""
    from exact.experiments import harness
    from exact.experiments.campaign import execute_campaign

    check_pause = check_pause or (lambda: None)
    check_pause()
    runtime = Path(wave) / "runtime/exact-om-focused-v2"
    stop = runtime / "STOP"
    if stop.exists():
        raise ValueError("Existing STOP prevents campaign finalization: " + str(stop))
    limit = memory_limit_bytes()
    failures = []
    done = threading.Event()

    def check():
        reason = None
        try:
            check_pause()
            rss = process_tree_rss_bytes()
            if rss > limit:
                reason = f"Cooperative process-tree RSS guard: {rss} bytes exceeds {limit} bytes"
        except (OSError, ValueError, RuntimeError) as error:
            reason = str(error)
        if reason:
            failures.append(reason)
            stop.parent.mkdir(parents=True, exist_ok=True)
            try:
                with stop.open("x") as stream:
                    stream.write(reason + "\n")
            except FileExistsError:
                pass
            return False
        return True

    if not check():
        raise ValueError("Campaign finalization stopped: " + failures[-1])

    def monitor():
        while not done.wait(2):
            if not check():
                return

    original = harness._run_subprocess
    frozen_caps = {
        key: os.environ[key]
        for key in (
            "EXACT_OPENROUTER_REQUEST_CAP",
            "EXACT_OPENROUTER_TOKEN_CAP",
            "EXACT_OPENROUTER_RETRY_UNKNOWN",
        )
    }

    def capped(command, *, cwd, stdout_path, stderr_path, env=None):
        return original(
            command,
            cwd=cwd,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            env={**(env or {}), **frozen_caps},
        )

    harness._run_subprocess = capped
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        result = execute_campaign(
            campaign,
            stage="screen",
            output_root=Path(wave) / "runtime",
            workdir=code_root,
            jobs=1,
            resume=True,
        )
    finally:
        harness._run_subprocess = original
        done.set()
        thread.join()
    if result or failures or stop.exists():
        raise ValueError("Campaign did not complete or has a retained STOP: " + str(failures))
