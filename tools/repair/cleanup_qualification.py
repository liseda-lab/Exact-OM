"""Charged CUDA lifecycle qualification; no model selection or scientific rows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.repair.shared_release import immutable


def child(ready):
    import torch

    torch.set_num_threads(1)
    value = torch.zeros((1024, 1024), device="cuda")
    torch.cuda.synchronize()
    immutable(Path(ready), dict(pid=os.getpid(), bytes=value.numel() * value.element_size()))
    time.sleep(60)


def qualify(output):
    from exact.repair import workers
    from tools.repair.primary_runtime import DEV_GPU, owned_device

    owned_device(DEV_GPU)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    deadline = float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"])
    rows = []
    for index, grace in enumerate((0.2, 2.0, 2.0)):
        if time.time() + 20 >= deadline:
            raise TimeoutError("No lifecycle qualification balance")
        ready = output / f"ready-{index}.json"
        process = subprocess.Popen(
            [sys.executable, "-m", "tools.repair.cleanup_qualification", "child", str(ready)],
            start_new_session=True,
        )
        pids = [process.pid]
        try:
            until = min(deadline - 3, time.time() + 15)
            while not ready.exists() and process.poll() is None and time.time() < until:
                time.sleep(0.02)
            if not ready.exists():
                raise RuntimeError("CUDA child failed to initialize")
            pids = workers._tree_pids(process.pid)
            started = time.monotonic()
            cleaned = workers._stop(process.pid, True, grace)
            elapsed = time.monotonic() - started
            # Reconcile only these already-killed owned PIDs, including the old
            # short-grace diagnostic. Never leave a failed diagnostic alive.
            retired = False
            while time.monotonic() - started < 2.0:
                states = {}
                for pid in pids:
                    try:
                        states[str(pid)] = (
                            Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                        )
                    except FileNotFoundError:
                        pass
                retired = all(state in {"Z", "X"} for state in states.values())
                if retired:
                    break
                time.sleep(0.01)
            row = dict(
                grace_seconds=grace,
                cleanup_complete=cleaned,
                cleanup_seconds=elapsed,
                retired_within_seconds=time.monotonic() - started,
                all_owned_processes_retired=retired,
                owned_pids=pids,
            )
            immutable(output / f"stop-{index}.json", row)
            rows.append(row)
            if not retired or (grace == 2.0 and not cleaned):
                raise RuntimeError("CUDA cleanup qualification failed")
        finally:
            workers._stop(process.pid, True, 2.0)
            workers._STOP_EXIT_CODES.pop(process.pid, None)
            process.poll()
    report = dict(
        status="complete",
        gpu_uuid=DEV_GPU,
        rows=rows,
        hosted_calls=0,
        optimizer_updates=0,
        scientific_rows=0,
    )
    immutable(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("child", "qualify"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (child if args.mode == "child" else qualify)(args.output)


if __name__ == "__main__":
    main()
