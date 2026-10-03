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


def charge(ledger, job, attempt, *, elapsed=None, cpu_seconds=None, peak_rss_mb=None):
    """Reserve before launch; never clear previous costs to make a retry eligible."""
    path = Path(ledger)
    with locked(path.with_suffix(".lock")):
        value = read(path)
        key = str(attempt)
        if elapsed is None:
            if key in value["attempts"]:
                raise ValueError("Attempt already charged")
            total = sum(
                row.get("elapsed_seconds", row["reserved_seconds"])
                for row in value["attempts"].values()
            )
            prior = sum(
                row.get("elapsed_seconds", row["reserved_seconds"])
                for row in value["attempts"].values()
                if row["logical_id"] == job["id"]
            )
            remaining = job["seconds"] - prior
            # Explicit null removes only the campaign-wide ceiling. A worker
            # still has a finite, cumulatively charged logical-job allowance.
            if value["limit_worker_seconds"] is not None:
                remaining = min(value["limit_worker_seconds"] - total, remaining)
            if remaining <= 0:
                raise TimeoutError("Cumulative campaign/job budget exhausted")
            value["attempts"][key] = {
                "logical_id": job["id"],
                "reserved_seconds": remaining,
                "resources": job["resources"],
                "status": "reserved",
                "started_epoch": time.time(),
            }
        else:
            value["attempts"][key].update(
                elapsed_seconds=elapsed,
                cpu_seconds=cpu_seconds,
                peak_rss_mb=peak_rss_mb,
                status="settled",
                finished_epoch=time.time(),
            )
        rows = list(value["attempts"].values())
        value["cumulative"] = {
            "worker_seconds": sum(r.get("elapsed_seconds", r["reserved_seconds"]) for r in rows),
            "allocated_cpu_seconds": sum(
                r.get("elapsed_seconds", r["reserved_seconds"]) * r["resources"]["cpus"]
                for r in rows
            ),
            "allocated_gpu_seconds": sum(
                r.get("elapsed_seconds", r["reserved_seconds"]) * r["resources"]["gpus"]
                for r in rows
            ),
            "allocated_memory_mb_seconds": sum(
                r.get("elapsed_seconds", r["reserved_seconds"]) * r["resources"]["memory_mb"]
                for r in rows
            ),
            "measured_cpu_seconds": sum(r.get("cpu_seconds") or 0 for r in rows),
            "external_api_cost_usd": 0,
        }
        write_artifact(path, value)
        return value["attempts"][key]["reserved_seconds"]


def verify(batch):
    for path, digest in batch["frozen_files"].items():
        if sha(path) != digest:
            raise ValueError("Frozen batch file changed: " + path)
    actual = subprocess.check_output(
        [batch["python"], "-m", "pip", "list", "--format=json"], text=True
    )
    if json.loads(actual) != batch["packages"]:
        raise ValueError("Installed environment changed since the batch was frozen")
    from exact.repair.study import runtime_manifest

    pinned = read(Path(batch["code"]).parent / "runtime.json")
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
    archive = root / "code.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "--output", str(archive), commit], cwd=repo, check=True
    )
    code = root / "code"
    with tarfile.open(archive) as stream:
        stream.extractall(code, filter="data")
    protocol = load_protocol(repo / spec["protocol_source"])
    write_artifact(root / "protocol.json", protocol)
    # Runtime source/native dependency hashes qualify the same installed optional stack.
    from exact.repair.study import runtime_manifest

    write_artifact(root / "runtime.json", runtime_manifest())
    packages = json.loads(
        subprocess.check_output([spec["python"], "-m", "pip", "list", "--format=json"], text=True)
    )
    paths = [p for p in code.rglob("*") if p.is_file()] + [
        root / "protocol.json",
        root / "runtime.json",
        archive,
    ]
    # Bind external immutable inputs too; runtime outputs/ledgers must stay outside
    # this list. Resolve relative inputs against the specified source checkout.
    paths += [(repo / item).resolve() for item in spec.get("input_files", [])]
    batch = {
        **spec,
        "commit": commit,
        "code": str(code),
        "protocol": str(root / "protocol.json"),
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
        seconds = charge(batch["ledger"], job, attempt)
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
            PYTHONUNBUFFERED="1", PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
        )
        for index, arguments in enumerate(job["commands"]):
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
        outputs = {
            str(p.relative_to(work)): sha(p)
            for p in work.rglob("*")
            if p.is_file() and p.suffix != ".lock"
        }
        write_artifact(attempt / "outputs.json", outputs)
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
        }
        write_artifact(attempt / "status.json", final)
        write_artifact(attempt / "completion.json", final)
        (attempt / "exit_code").write_text(str(code) + "\n")
    return code


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
        for job in batch["jobs"]:
            prior = [
                r
                for r in registry["runs"]
                if r.get("logical_id") == job["id"] and not r.get("superseded_by")
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
                    logical_id=job["id"],
                    step_id=receipt["step_id"],
                    status_path=str(attempt / "status.json"),
                    completion_path=str(attempt / "completion.json"),
                    exit_path=str(attempt / "exit_code"),
                    depends_on=[],
                    enabled=True,
                    resources={key: resources[key] for key in totals},
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
    gpu = commands.add_parser("qualify-gpus")
    gpu.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.specification, args.destination))
    elif args.command == "submit":
        print(json.dumps(submit(args.batch, args.supervisor)))
    elif args.command == "qualify-gpus":
        qualify_gpus(args.output)
    else:
        return run(args.batch, args.job_id, args.attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
