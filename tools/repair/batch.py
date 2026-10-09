"""Freeze, launch and receipt small XR-2 batches; monitoring uses the existing supervisor.

Commands are argv arrays, never shell fragments. Scientific outputs live under
stable logical job IDs; each replacement has separate logs and Slurm receipts.
The shared ledger charges failed attempts and outstanding reservations as well
as successes. Its worker-seconds cap is conservative when jobs run concurrently.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import shlex
import uuid
import os
import signal
import subprocess
import tarfile
import time
from pathlib import Path

from exact.repair.api import write_artifact


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextlib.contextmanager
def locked(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _budget_identity(job):
    for field in ("seconds", "slice_seconds", "deadline_epoch"):
        val = job.get(field)
        if val is not None and (
            isinstance(val, bool)
            or not isinstance(val, (int, float))
            or not math.isfinite(val)
            or val <= 0
        ):
            raise ValueError("Job budget must be finite and positive: " + field)
    cleanup = job.get("cleanup_seconds", 0)
    if (
        isinstance(cleanup, bool)
        or not isinstance(cleanup, (int, float))
        or not math.isfinite(cleanup)
        or cleanup < 0
    ):
        raise ValueError("Cleanup allowance must be finite and nonnegative")
    stages = job.get("budget_stages", [])
    if not isinstance(stages, (list, tuple)) or any(
        not isinstance(x, str) or not x for x in stages
    ):
        raise ValueError("Budget stages must be explicit stable names")
    return {
        "logical_id": job.get("logical_id", job["id"]),
        "seconds": job["seconds"],
        "budget_stages": sorted(job.get("budget_stages", [])),
        "deadline_epoch": job.get("deadline_epoch"),
    }


def _stage_remaining(value, job, now):
    remaining = float("inf")
    stages = job.get("budget_stages", [])
    if len(set(stages)) != len(stages):
        raise ValueError("A stage may appear only once in an attempt")
    for name in stages:
        limits = value.get("stage_limits", {})[name]
        if (
            not isinstance(limits, dict)
            or not limits
            or set(limits)
            - {
                "elapsed_seconds",
                "worker_seconds",
                "gpu_seconds",
                "allocated_cpu_seconds",
                "deadline_epoch",
            }
        ):
            raise ValueError("Unknown or empty cumulative stage contract: " + name)
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in limits.values()
        ):
            raise ValueError("Stage limits must be finite positive numbers")
        state = value.setdefault("stages", {}).setdefault(
            name,
            {
                "started_epoch": now,
                "limits": dict(limits),
            },
        )
        if state["limits"] != limits:
            raise ValueError("A started stage's cumulative limits cannot change")
        rows = [r for r in value["attempts"].values() if name in r.get("budget_stages", [])]
        duration = lambda row: row.get("elapsed_seconds", row["reserved_seconds"])
        if "elapsed_seconds" in limits:
            remaining = min(remaining, state["started_epoch"] + limits["elapsed_seconds"] - now)
        if "deadline_epoch" in limits:
            remaining = min(remaining, limits["deadline_epoch"] - now)
        for field, resource in (
            ("worker_seconds", None),
            ("gpu_seconds", "gpus"),
            ("allocated_cpu_seconds", "cpus"),
        ):
            multiplier = 1 if resource is None else job["resources"][resource]
            if field in limits and multiplier:
                used = sum(
                    duration(r) * (1 if resource is None else r["resources"][resource])
                    for r in rows
                )
                remaining = min(remaining, (limits[field] - used) / multiplier)
    return remaining


def charge(
    ledger,
    job,
    attempt,
    *,
    elapsed=None,
    cpu_seconds=None,
    peak_rss_mb=None,
    claim=False,
    worker_started_epoch=None,
):
    """Reserve under one lock, preserving stage clocks and every failed/replacement cost.

    The optional claim adopts a dispatch reservation once, including time queued
    after admission. Existing callers still reject duplicate reservations.
    """
    path = Path(ledger)
    with locked(path.with_suffix(".lock")):
        value, now, key = read(path), time.time(), str(attempt)
        identity = _budget_identity(job)
        logical = identity["logical_id"]
        prior_reservation = value["attempts"].get(key)
        if elapsed is None and claim and prior_reservation is not None:
            if (
                prior_reservation.get("budget_identity") != identity
                or prior_reservation.get("resources") != job["resources"]
                or prior_reservation.get("gpu_devices") != job.get("gpu_devices")
                or prior_reservation.get("status") != "reserved"
                or prior_reservation.get("worker_claimed_epoch") is not None
            ):
                raise ValueError("Launch reservation is changed, settled or already claimed")
            prior_reservation["worker_claimed_epoch"] = now
            prior_reservation["worker_started_epoch"] = (
                now if worker_started_epoch is None else worker_started_epoch
            )
            write_artifact(path, value)
            queue = max(
                0, prior_reservation["worker_started_epoch"] - prior_reservation["started_epoch"]
            )
            return max(0, prior_reservation["reserved_seconds"] - queue)
        if elapsed is None:
            if key in value["attempts"]:
                raise ValueError("Attempt already charged")
            prior_identity = value.setdefault("logical_budgets", {}).setdefault(logical, identity)
            if prior_identity != identity:
                raise ValueError(
                    "Replacement changed cumulative logical-job budget or stage identity"
                )
            rows = list(value["attempts"].values())
            total = sum(r.get("elapsed_seconds", r["reserved_seconds"]) for r in rows)
            prior = sum(
                r.get("elapsed_seconds", r["reserved_seconds"])
                for r in rows
                if r["logical_id"] == logical
            )
            remaining = min(
                job["seconds"] - prior,
                job.get("slice_seconds", float("inf")),
                _stage_remaining(value, job, now),
            )
            if value["limit_worker_seconds"] is not None:
                remaining = min(value["limit_worker_seconds"] - total, remaining)
            if job.get("deadline_epoch") is not None:
                remaining = min(remaining, job["deadline_epoch"] - now)
            if not math.isfinite(remaining) or remaining <= 0:
                raise TimeoutError("Cumulative campaign/job/stage budget or deadline exhausted")
            value["attempts"][key] = {
                "logical_id": logical,
                "budget_identity": identity,
                "reserved_seconds": remaining,
                "resources": job["resources"],
                "gpu_devices": job.get("gpu_devices"),
                "budget_stages": identity["budget_stages"],
                "status": "reserved",
                "started_epoch": now,
                **(
                    {
                        "worker_claimed_epoch": now,
                        "worker_started_epoch": (
                            now if worker_started_epoch is None else worker_started_epoch
                        ),
                    }
                    if claim
                    else {}
                ),
            }
        else:
            row = value["attempts"][key]
            if row["logical_id"] != logical:
                raise ValueError("Settlement changed logical work identity")
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Measured elapsed cost must be finite and nonnegative")
            # Reconciliation may increase measured cost; it never erases already
            # settled work. Queue time is measured only for a claimed reservation.
            elapsed += max(
                0, row.get("worker_started_epoch", row["started_epoch"]) - row["started_epoch"]
            )
            if row.get("status") == "settled" and elapsed < row["elapsed_seconds"]:
                raise ValueError("Settlement cannot reduce previously accumulated cost")
            if row.get("status") == "settled" and (cpu_seconds or 0) < (
                row.get("cpu_seconds") or 0
            ):
                raise ValueError("Settlement cannot reduce measured CPU cost")
            row.update(
                elapsed_seconds=elapsed,
                cpu_seconds=cpu_seconds,
                peak_rss_mb=peak_rss_mb,
                status="settled",
                finished_epoch=now,
            )
        rows = list(value["attempts"].values())
        duration = lambda row: row.get("elapsed_seconds", row["reserved_seconds"])
        value["cumulative"] = {
            "worker_seconds": sum(duration(r) for r in rows),
            "allocated_cpu_seconds": sum(duration(r) * r["resources"]["cpus"] for r in rows),
            "allocated_gpu_seconds": sum(duration(r) * r["resources"]["gpus"] for r in rows),
            "allocated_memory_mb_seconds": sum(
                duration(r) * r["resources"]["memory_mb"] for r in rows
            ),
            "measured_cpu_seconds": sum(r.get("cpu_seconds") or 0 for r in rows),
            "external_api_cost_usd": value.get("cumulative", {}).get("external_api_cost_usd", 0),
        }
        for name, state in value.get("stages", {}).items():
            members = [r for r in rows if name in r.get("budget_stages", [])]
            state["cumulative"] = {
                "elapsed_seconds": max(0, now - state["started_epoch"]),
                "worker_seconds": sum(duration(r) for r in members),
                "gpu_seconds": sum(duration(r) * r["resources"]["gpus"] for r in members),
                "allocated_cpu_seconds": sum(duration(r) * r["resources"]["cpus"] for r in members),
            }
        write_artifact(path, value)
        return value["attempts"][key]["reserved_seconds"]


def reserve_launch(batch_path, job_id, attempt):
    """Called only after the dispatcher validates the frozen manifest binding."""
    path = Path(batch_path).resolve()
    if sha(path) != path.with_name("batch.sha256").read_text().strip():
        raise ValueError("Frozen batch manifest changed before reservation")
    batch = read(path)
    job = next(row for row in batch["jobs"] if row["id"] == job_id)
    return charge(batch["ledger"], job, attempt)


def source_snapshot(repo, commit, directory):
    """Commit-addressed shared exports avoid copying a checkout for every batch."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / commit
    with locked(directory / (commit + ".lock")):
        if not destination.exists():
            temporary = directory / (commit + ".pending-" + uuid.uuid4().hex)
            temporary.mkdir()
            archive, code = temporary / "code.tar", temporary / "code"
            subprocess.run(
                ["git", "archive", "--format=tar", "--output", str(archive), commit],
                cwd=repo,
                check=True,
            )
            with tarfile.open(archive) as stream:
                stream.extractall(code, filter="data")
            files = {
                str(p.relative_to(temporary)): sha(p) for p in temporary.rglob("*") if p.is_file()
            }
            write_artifact(temporary / "source.json", {"commit": commit, "files": files})
            temporary.rename(destination)
        manifest = read(destination / "source.json")
        if manifest["commit"] != commit:
            raise ValueError("Shared source snapshot has another commit")
        for relative, digest in manifest["files"].items():
            source = (destination / relative).resolve()
            if not source.is_relative_to(destination) or sha(source) != digest:
                raise ValueError("Shared source snapshot changed: " + relative)
    return destination / "code", destination / "source.json"


def verify(batch):
    if batch.get("source_manifest"):
        reference = batch["source_manifest"]
        if sha(reference["path"]) != reference["sha256"]:
            raise ValueError("Shared source manifest changed")
        source = Path(reference["path"]).parent
        for relative, digest in read(reference["path"])["files"].items():
            child = (source / relative).resolve()
            if not child.is_relative_to(source) or sha(child) != digest:
                raise ValueError("Shared source snapshot changed: " + relative)
    for path, digest in batch["frozen_files"].items():
        if sha(path) != digest:
            raise ValueError("Frozen batch file changed: " + path)
    actual = subprocess.check_output(
        [batch["python"], "-m", "pip", "list", "--format=json"], text=True
    )
    if json.loads(actual) != batch["packages"]:
        raise ValueError("Installed environment changed since the batch was frozen")
    from exact.repair.study import runtime_manifest

    pinned = read(batch.get("runtime", Path(batch["code"]).parent / "runtime.json"))
    actual_runtime = runtime_manifest()
    for key in ("dependencies", "ontology_implementations", "code_hashes"):
        if actual_runtime[key] != pinned[key]:
            raise ValueError("Installed repair implementation changed: " + key)


def freeze(specification, destination):
    """Export a committed source tree and resolved configs without local data/secrets."""
    from tools.repair.prepare import load_protocol

    spec, root = read(specification), Path(destination).resolve()
    repo = Path(spec["repository"])
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=repo, text=True)
    if dirty.strip():
        raise ValueError("Commit tracked source changes before freezing a batch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    root.mkdir(parents=True, exist_ok=False)
    source_manifest = None
    if spec.get("source_store"):
        code, manifest = source_snapshot(repo, commit, spec["source_store"])
        source_manifest = {"path": str(manifest), "sha256": sha(manifest)}
    else:
        archive = root / "code.tar"
        subprocess.run(
            ["git", "archive", "--format=tar", "--output", str(archive), commit],
            cwd=repo,
            check=True,
        )
        code = root / "code"
        with tarfile.open(archive) as stream:
            stream.extractall(code, filter="data")
    protocol = load_protocol(repo / spec["protocol_source"])
    write_artifact(root / "protocol.json", protocol)
    # Resolve source/native identities exactly as the worker will: with the
    # declared interpreter in the committed export. An external preparation
    # script may itself have imported a different editable installation.
    runtime = json.loads(
        subprocess.check_output(
            [
                spec["python"],
                "-c",
                "import json; from exact.repair.study import runtime_manifest; "
                "print(json.dumps(runtime_manifest()))",
            ],
            cwd=code,
            text=True,
        )
    )
    write_artifact(root / "runtime.json", runtime)
    packages = json.loads(
        subprocess.check_output([spec["python"], "-m", "pip", "list", "--format=json"], text=True)
    )
    paths = [root / "protocol.json", root / "runtime.json"]
    if source_manifest is None:
        paths += [p for p in code.rglob("*") if p.is_file()] + [archive]
    # Bind external immutable inputs too; runtime outputs/ledgers must stay outside
    # this list. Resolve relative inputs against the specified source checkout.
    paths += [(repo / item).resolve() for item in spec.get("input_files", [])]
    batch = {
        **spec,
        "commit": commit,
        "code": str(code),
        "protocol": str(root / "protocol.json"),
        "runtime": str(root / "runtime.json"),
        **({"source_manifest": source_manifest} if source_manifest else {}),
        "packages": packages,
        "frozen_files": {str(p): sha(p) for p in paths},
    }
    write_artifact(root / "batch.json", batch)
    (root / "batch.sha256").write_text(sha(root / "batch.json") + "\n")
    return root / "batch.json"


def checked_batch(path):
    path = Path(path).resolve()
    if sha(path) != path.with_name("batch.sha256").read_text().strip():
        raise ValueError("Frozen batch manifest changed")
    batch = read(path)
    verify(batch)
    return batch


def descendants(pid):
    pending, found = [pid], set()
    while pending:
        item = pending.pop()
        if item in found:
            continue
        found.add(item)
        try:
            for task in Path(f"/proc/{item}/task").iterdir():
                pending.extend(map(int, (task / "children").read_text().split()))
        except OSError:
            pass
    return found


def record_outputs(work, attempt):
    """Retain atomic-write debris as metadata, never as published output bytes."""
    from tools.repair.scaling_payload_recovery import payload_manifest

    published, unpublished = payload_manifest(work)
    outputs = {str(Path(item["path"]).relative_to(work)): item["sha256"] for item in published}
    write_artifact(
        Path(attempt) / "outputs-unpublished.json",
        {
            "schema": "exact-repair/unpublished-output-inventory/v1",
            "files": unpublished,
            "original_files_modified": False,
        },
    )
    write_artifact(Path(attempt) / "outputs.json", outputs)
    return outputs


def run(path, job_id, attempt):
    import resource

    from exact.repair.workers import _resident_tree_bytes

    # Write startup/step identity even if later integrity validation fails.
    raw = read(path)
    job = next(job for job in raw["jobs"] if job["id"] == job_id)
    attempt = Path(attempt).resolve()
    status = dict(
        status="running",
        job_id=job_id,
        pid=os.getpid(),
        step_id=os.environ["SLURM_JOB_ID"] + "." + os.environ["SLURM_STEP_ID"],
        started_epoch=time.time(),
        batch=str(Path(path).resolve()),
    )
    write_artifact(attempt / "status.json", status)
    work = Path(raw["campaign"]) / "work" / job_id
    work.mkdir(parents=True, exist_ok=True)
    started, peak, process, stopped, charged = time.monotonic(), 0.0, None, False, False

    def stop(_sig, _frame):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    error, code = None, 0
    try:
        batch = checked_batch(path)
        seconds = charge(
            batch["ledger"], job, attempt, claim=True, worker_started_epoch=status["started_epoch"]
        )
        cleanup = job.get("cleanup_seconds", 2 if job.get("budget_stages") else 0)
        seconds = max(0, seconds - cleanup)
        charged = True
        values = dict(
            python=batch["python"],
            code=batch["code"],
            protocol=batch["protocol"],
            work=str(work),
            campaign=batch["campaign"],
        )
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"OPENAI_API_KEY", "OPENROUTER_API_KEY", "CODEX_API_KEY"}
            and not k.startswith("EXACT_")
        }
        env.update(
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONHASHSEED="0",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1",
        )
        if job.get("budget_stages") or job.get("deadline_epoch") is not None:
            env["EXACT_REPAIR_DEADLINE_EPOCH"] = str(
                time.time() + max(0, seconds - (time.monotonic() - started))
            )
            env["EXACT_REPAIR_STAGE_LEDGER"] = str(batch["ledger"])
        for index, arguments in enumerate(job["commands"]):
            if time.monotonic() - started >= seconds:
                raise TimeoutError("Cumulative worker budget exhausted before command launch")
            argv = [value.format(**values) for value in arguments]
            write_artifact(attempt / f"command-{index}.json", {"argv": argv, "cwd": batch["code"]})
            with (attempt / f"command-{index}.log").open("w") as log:
                process = subprocess.Popen(
                    argv,
                    cwd=batch["code"],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    peak = max(peak, _resident_tree_bytes(process.pid) / 1024**2)
                    write_artifact(
                        attempt / "status.json",
                        {
                            **status,
                            "elapsed_seconds": elapsed,
                            "command_index": index,
                            "peak_rss_mb": peak,
                        },
                    )
                    if (Path(batch["campaign"]) / "STOP").exists() or (work / "STOP").exists():
                        stopped = True
                    if stopped or elapsed >= seconds or peak > job["resources"]["memory_mb"]:
                        raise TimeoutError(
                            "worker interrupted"
                            if stopped
                            else "worker time/memory budget exhausted"
                        )
                    time.sleep(1)
                if process.returncode:
                    tail = (
                        (attempt / f"command-{index}.log")
                        .read_text(errors="replace")
                        .splitlines()[-1:]
                    )
                    raise RuntimeError(
                        f"command {index} exited {process.returncode}: " + " ".join(tail)[:1000]
                    )
                process = None
        record_outputs(work, attempt)
    except Exception as exc:
        error, code = {"type": type(exc).__name__, "message": str(exc)}, 1
    finally:
        if process is not None and process.poll() is None:
            # Bounded reasoner workers create their own process groups; collect
            # descendants before killing the parent, and never signal an allocation.
            for pid in sorted(descendants(process.pid), reverse=True):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
        elapsed = time.monotonic() - started
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = usage.ru_utime + usage.ru_stime
        output_error = None
        if code != 0:
            try:
                record_outputs(work, attempt)
            except Exception as exc:
                # Preserve the original failure and make unbound partial evidence explicit.
                output_error = type(exc).__name__ + ": " + str(exc)
        if charged:
            charge(raw["ledger"], job, attempt, elapsed=elapsed, cpu_seconds=cpu, peak_rss_mb=peak)
        final = {
            **status,
            "status": "complete" if code == 0 else "failed",
            "exit_code": code,
            "error": error,
            "elapsed_seconds": elapsed,
            "cpu_seconds": cpu,
            "peak_rss_mb": peak,
            "finished_epoch": time.time(),
            "work": str(work),
            **({"output_publication_error": output_error} if output_error else {}),
        }
        write_artifact(attempt / "status.json", final)
        write_artifact(attempt / "completion.json", final)
        (attempt / "exit_code").write_text(str(code) + "\n")
    return code


def prepare_dispatch(
    batch_path, job_id, attempt, *, tmux_socket, run_id=None, depends_on=(), nonce=None
):
    """Prepare a reviewed descriptor; the deterministic supervisor owns submission."""
    path = Path(batch_path).resolve()
    batch = checked_batch(path)
    job = next(row for row in batch["jobs"] if row["id"] == job_id)
    attempt = Path(attempt).resolve()
    attempt.mkdir(parents=True, exist_ok=False)
    nonce = nonce or uuid.uuid4().hex
    run_id = run_id or job_id
    resources = {key: job["resources"][key] for key in ("cpus", "gpus", "memory_mb")}
    worker = attempt / "worker.sh"
    argv = [
        batch["python"],
        "-m",
        "tools.repair.campaign_handoff",
        "worker",
        str(path),
        job_id,
        str(attempt),
        nonce,
    ]
    worker.write_text(
        "#!/bin/bash\nset -eu\ncd "
        + shlex.quote(batch["code"])
        + "\nexec "
        + shlex.join(argv)
        + "\n"
    )
    launch = [
        "/usr/bin/srun",
        "--jobid=" + batch["allocation"],
        "--overlap",
        "--exact",
        "-N1",
        "-n1",
        "--cpus-per-task=" + str(resources["cpus"]),
        "--mem=" + str(resources["memory_mb"]),
        "--gres=" + job["resources"]["gres"],
        "--cpu-bind=none",
        "--unbuffered",
        "--job-name=" + run_id,
        "--chdir=" + batch["code"],
        "/bin/bash",
        str(worker),
    ]
    record = {
        "id": run_id,
        "depends_on": list(depends_on),
        "resources": resources,
        **{
            key: job[key]
            for key in (
                "gpu_devices",
                "resource_profile",
                "priority",
                "stage",
                "deadline_epoch",
                "deadline_policy",
                "not_before_epoch",
            )
            if key in job
        },
        "budget_reservation": {"batch": str(path), "job_id": job_id, "attempt": str(attempt)},
        "launch": {
            "argv": launch,
            "nonce": nonce,
            "tmux_socket": str(Path(tmux_socket).resolve()),
            "bindings": [{"path": str(item), "sha256": sha(item)} for item in (worker, path)],
            "step_path": str(attempt / "step.json"),
            "launcher_log": str(attempt / "launcher.log"),
            "run": {
                "id": run_id,
                "logical_id": job.get("logical_id", job_id),
                "status_path": str(attempt / "status.json"),
                "completion_path": str(attempt / "completion.json"),
                "exit_path": str(attempt / "exit_code"),
                "budget_path": batch["ledger"],
            },
        },
    }
    write_artifact(attempt / "descriptor.json", record)
    return record


def submit(path, supervisor):
    """Detach srun steps inside the retained allocation and register their lineage."""
    batch = checked_batch(path)
    supervisor = Path(supervisor).resolve()
    job_info = subprocess.check_output(
        ["scontrol", "show", "job", batch["allocation"], "-o"], text=True
    )
    if "JobState=RUNNING" not in job_info or "BatchFlag=0" not in job_info:
        raise ValueError("The retained interactive allocation must still be running")
    totals = {
        key: sum(job["resources"][key] for job in batch["jobs"])
        for key in ("cpus", "gpus", "memory_mb")
    }
    if any(totals[key] > batch["capacity"][key] for key in totals):
        raise ValueError("Batch exceeds capacity reserved for experiments")
    with locked(supervisor / "registry-update.lock"):
        registry = read(supervisor / "registry.json")
        stops = [supervisor / "STOP", supervisor / "PAUSE", Path(batch["campaign"]) / "STOP"]
        stops.extend(Path(item) for item in registry.get("pause_paths", []))
        if any(path.exists() for path in stops):
            raise RuntimeError("Submission paused by a STOP/PAUSE file")
        live = subprocess.check_output(
            ["squeue", "--steps", "-h", "-j", batch["allocation"], "-o", "%i"], text=True
        ).split()
        used = {
            key: sum(
                r.get("resources", {}).get(key, 0)
                for r in registry["runs"]
                if r.get("enabled", True) and r["step_id"] in live
            )
            for key in totals
        }
        if any(used[key] + totals[key] > batch["capacity"][key] for key in totals):
            raise ValueError("Insufficient capacity alongside the live independent work")
        from exact.experiments.supervision import gpu_conflict, validate_admission

        owners = [r for r in registry["runs"] if r["step_id"] in live]
        for job in batch["jobs"]:
            if "gpu_devices" in job or "resource_profile" in job:
                admission = {**job, "resources": {key: job["resources"][key] for key in totals}}
                validate_admission(registry, admission)
                if gpu_conflict(admission, owners):
                    raise ValueError("A requested GPU already has a live or scheduled owner")
                owners.append(admission)
        for job in batch["jobs"]:
            prior = [
                r
                for r in registry["runs"]
                if r.get("logical_id") == job.get("logical_id", job["id"])
                and not r.get("superseded_by")
            ]
            if prior:
                if prior[0]["step_id"] in live:
                    raise ValueError("Logical work already has a live Slurm step")
                if (
                    Path(prior[0]["completion_path"]).exists()
                    and read(prior[0]["completion_path"]).get("status") == "complete"
                ):
                    raise ValueError("Completed logical work cannot be resubmitted")
            attempts = Path(batch["campaign"]) / "attempts" / job["id"]
            attempts.mkdir(parents=True, exist_ok=True)
            attempt = attempts / f"{len(list(attempts.iterdir())) + 1:03d}"
            attempt.mkdir()
            resources = job["resources"]
            command = [
                "srun",
                "--jobid=" + batch["allocation"],
                "--overlap",
                "--exact",
                "-N1",
                "-n1",
                "--cpus-per-task=" + str(resources["cpus"]),
                "--mem=" + str(resources["memory_mb"]),
                "--gres=" + resources["gres"],
                "--cpu-bind=none",
                "--unbuffered",
                "--job-name=" + job["id"],
                "--chdir=" + batch["code"],
                batch["python"],
                "-m",
                "tools.repair.batch",
                "run",
                str(Path(path).resolve()),
                job["id"],
                str(attempt),
            ]
            write_artifact(attempt / "launch.json", {"argv": command})
            with (attempt / "launcher.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            deadline = time.monotonic() + 30
            while (
                not (attempt / "status.json").exists()
                and process.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.2)
            if not (attempt / "status.json").exists():
                # Do not silently retry an ambiguous launch; preserve its PID/log.
                write_artifact(
                    attempt / "submission-error.json",
                    {"srun_pid": process.pid, "returncode": process.poll()},
                )
                raise RuntimeError("Slurm launch has no startup receipt; inspect " + str(attempt))
            receipt = read(attempt / "status.json")
            name = job["id"] + "-" + attempt.name
            if prior:
                prior[0].update(enabled=False, superseded_by=name)
            registry["runs"].append(
                dict(
                    id=name,
                    logical_id=job.get("logical_id", job["id"]),
                    step_id=receipt["step_id"],
                    status_path=str(attempt / "status.json"),
                    completion_path=str(attempt / "completion.json"),
                    exit_path=str(attempt / "exit_code"),
                    depends_on=[],
                    enabled=True,
                    resources={key: resources[key] for key in totals},
                    **{
                        key: job[key]
                        for key in (
                            "gpu_devices",
                            "resource_profile",
                            "deadline_epoch",
                            "budget_stages",
                            "priority",
                        )
                        if key in job
                    },
                )
            )
            write_artifact(supervisor / "registry.json", registry)
        if "pending_batches" in registry:
            registry["pending_batches"] = [
                item for item in registry["pending_batches"] if item["id"] != batch["id"]
            ]
            write_artifact(supervisor / "registry.json", registry)
        write_artifact(
            Path(path).parent / "submission.json",
            {
                "allocation": batch["allocation"],
                "submitted_epoch": time.time(),
                "runs": registry["runs"],
            },
        )
    return registry


def qualify_gpus(output):
    """Exercise actual HGT/R-GCN forward/backward paths on both allocated GPUs."""
    import torch

    from exact.repair.graph import build_observable_graph
    from exact.repair.model import RepairModel
    from tools.repair.corpus import generate_corpus

    if torch.cuda.device_count() != 2:
        raise RuntimeError("GPU qualification requires both allocated GPUs")
    torch.set_num_threads(1)
    case = generate_corpus(parents_per_family=1, siblings_per_parent=1, families=("range",))[0]
    graph = build_observable_graph(case.problem.objects, fixed_axioms=case.problem.fixed_axioms)
    rows = []
    for device in range(2):
        for encoder in ("hgt", "rgcn"):
            torch.manual_seed(13)
            model = RepairModel(
                graph.metadata, encoder=encoder, hidden_dim=8, heads=2, layers=1
            ).to(f"cuda:{device}")
            unary, _ = model.score_inventory(case.problem.objects, model.encode(graph))
            loss = torch.cat(unary).square().mean()
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(p).all().item() for p in gradients):
                raise RuntimeError("Nonfinite or missing GPU gradients")
            rows.append(
                dict(
                    device=device,
                    name=torch.cuda.get_device_name(device),
                    encoder=encoder,
                    loss=loss.item(),
                )
            )
    write_artifact(
        output,
        {"status": "complete", "scope": "component conformance, not model quality", "rows": rows},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("specification", type=Path)
    freezing.add_argument("destination", type=Path)
    submission = commands.add_parser("submit")
    submission.add_argument("batch", type=Path)
    submission.add_argument("supervisor", type=Path)
    worker = commands.add_parser("run")
    worker.add_argument("batch", type=Path)
    worker.add_argument("job_id")
    worker.add_argument("attempt", type=Path)
    reservation = commands.add_parser("reserve-launch")
    reservation.add_argument("batch", type=Path)
    reservation.add_argument("job_id")
    reservation.add_argument("attempt", type=Path)
    gpu = commands.add_parser("qualify-gpus")
    gpu.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.specification, args.destination))
    elif args.command == "submit":
        print(json.dumps(submit(args.batch, args.supervisor)))
    elif args.command == "qualify-gpus":
        qualify_gpus(args.output)
    elif args.command == "reserve-launch":
        print(
            json.dumps(
                {"reserved_worker_seconds": reserve_launch(args.batch, args.job_id, args.attempt)}
            )
        )
    else:
        return run(args.batch, args.job_id, args.attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
