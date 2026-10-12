"""Own one remote E07 replay through a real, locally supervised Slurm step."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import socket
import subprocess
from pathlib import Path

from tools.prepared_batch import binding, controls, read, verified, write
from tools.storage_guard import publish_step_receipt


def load(path):
    plan = read(path)
    for item in plan["bindings"]:
        verified(item)
    controls(Path(plan["supervisor"]), Path(plan["output"]))
    return plan


def remote_environment(environment):
    """Start a separate t2 allocation; never inherit the retained job or stale keys."""
    return {
        k: v
        for k, v in environment.items()
        if not k.startswith(("SLURM_", "EXACT_"))
        and k not in {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "CUDA_VISIBLE_DEVICES"}
    }


def coordinate(path):
    plan = load(path)
    root = Path(plan["dispatch_root"])
    with (root / "owner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 73  # Another owner retains its own status and exit receipts.
        if os.getenv("SLURM_JOB_ID") != plan["coordinator_allocation"]:
            raise ValueError("Coordinator is outside its retained allocation")
        publish_step_receipt(root / "step.json", plan["nonce"])
        owner = read(root / "step.json")
        if (root / "remote-step.json").exists() or (root / "dispatch-claim.json").exists():
            raise ValueError("Remote dispatch already has an owner; inspect before resubmitting")
        process, code = None, 1
        try:
            write(root / "status.json", {"status": "running", **owner})
            # Persist ownership before spawning: a crash before remote startup is
            # an uncertain dispatch, not permission to create another allocation.
            publish_step_receipt(root / "dispatch-claim.json", plan["nonce"])
            process = subprocess.Popen(
                plan["remote_argv"],
                cwd=plan["code_root"],
                env=remote_environment(os.environ),
            )
            code = process.wait()
            remote = (
                read(root / "remote-step.json") if (root / "remote-step.json").exists() else None
            )
            result = Path(plan["output"]) / "completion.json"
            if code != 0 or remote is None or not result.exists():
                raise RuntimeError(f"Remote Luna worker exited {code}; inspect remote receipts")
            if (
                remote.get("dispatch_nonce") != plan["nonce"]
                or re.fullmatch(r"[0-9]+\.[0-9]+", str(remote.get("step_id"))) is None
                or remote["step_id"].split(".")[0] == plan["coordinator_allocation"]
            ):
                raise ValueError("Remote receipt does not belong to this independent dispatch")
            receipt = read(result)
            if receipt.get("status") != "complete":
                raise ValueError("Remote worker lacks a completed comparison")
            report = read(verified(receipt["report"]))
            verified(report["manifest"])
            verified(report["predictions"])
            if report.get("status") != "complete":
                raise ValueError("Remote comparison report is incomplete")
            write(
                root / "completion.json",
                {
                    "status": "complete",
                    "exit_code": 0,
                    **owner,
                    "remote_step": remote,
                    "result": binding(result),
                },
            )
            write(root / "status.json", {"status": "complete", **owner, "remote_step": remote})
        except BaseException as exc:
            code = code or 1
            error = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            write(root / "status.json", {"status": "failed", **owner, "error": error})
            write(
                root / "completion.json",
                {"status": "failed", "exit_code": code, **owner, "error": error},
            )
            raise
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            temporary = root / "exit-code.partial"
            temporary.write_text(str(code) + "\n")
            temporary.replace(root / "exit-code")
        return code


def worker(path):
    plan = load(path)
    if socket.gethostname().split(".")[0] != "liseda-t2":
        raise ValueError("The Luna worker must run on liseda-t2")
    root = Path(plan["dispatch_root"])
    publish_step_receipt(root / "remote-step.json", plan["nonce"])
    for key in list(os.environ):
        if key.startswith("EXACT_") or key in {"OPENROUTER_API_KEY", "OPENAI_API_KEY"}:
            os.environ.pop(key)
    os.environ.update(read(verified(plan["environment"])))
    # Deliberately read the current credential; it is never copied into a receipt.
    os.environ["OPENROUTER_API_KEY"] = Path(plan["api_key_path"]).read_text().strip()
    from tools.compare_luna_e07 import run

    try:
        return 0 if run(plan["output"]) is not None else 75
    except BaseException as exc:
        write(root / "remote-error.json", {"type": type(exc).__name__, "message": str(exc)[:1000]})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("coordinate", "worker"))
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(globals()[args.command](args.plan))


if __name__ == "__main__":
    main()
