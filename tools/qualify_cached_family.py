"""Execute measured, resumable cached cells once for their scientific comparison."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def binding(path):
    from exact.utils.provenance import sha256_file

    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


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


def preserve_training_progress(directory, identity):
    """Seal complete training shards before recovery archives a stopped attempt."""
    from exact.experiments.recovery import ArtifactStore
    from exact.experiments.runtime import _files

    directory = Path(directory)
    store = ArtifactStore(directory)
    previous = store.latest_checkpoint(identity["artifact_id"])
    if previous is not None:
        return previous
    output = directory / "run"
    runtime = json.loads((output / "recovery-runtime.json").read_text())
    if runtime["identity"] != identity:
        raise ValueError("Stopped training numerical identity changed")
    paths = _files(output, ("fitting", "dataset"))
    if not any(name.startswith("fitting/") for name in paths):
        return None
    for path in paths.values():
        if path.suffix == ".json":
            json.loads(path.read_text())
    population = json.loads((output / "dataset/candidate_pool_sample_manifest.json").read_text())
    return store.checkpoint(
        identity,
        completed_ids=[],
        cursor={"next_pair": 0, "dataset_rows": population["gold_free_summary"]["candidate_pairs"]},
        outputs=paths,
        state={"boundary": "stopped_training_shards", "inference_pairs_complete": 0},
    )


def usage(directory):
    from tools.run_experiment_validation import ledger_totals

    values = ledger_totals(directory)
    tokens = 0
    with sqlite3.connect(
        (Path(directory) / "requests.sqlite3").as_uri() + "?mode=ro", uri=True
    ) as db:
        for raw, reserved in db.execute(
            "SELECT a.usage,r.tokens FROM attempts a LEFT JOIN reservations r USING(request_id,number)"
        ):
            row = json.loads(raw or "{}")
            known = int(row.get("prompt_tokens") or 0) + int(row.get("completion_tokens") or 0)
            tokens += (
                known
                if "prompt_tokens" in row and "completion_tokens" in row
                else max(known, int(reserved or 0))
            )
    return {**values, "billable_tokens": tokens}


def control_paths(script, directory):
    """Re-read the registry so a later supervisor/user pause takes effect."""
    root = Path(script).resolve().parent
    supervisor = root.parent / "hourly-supervisor-01"
    registry = json.loads((supervisor / "registry.json").read_text())
    return [
        supervisor / "PAUSE",
        supervisor / "STOP",
        root / "PAUSE",
        root / "STOP",
        Path(directory) / "STOP",
        *[Path(p) for p in registry["pause_paths"]],
        *root.rglob("STOP"),
    ]


def check_controls(script, directory):
    present = [str(p) for p in control_paths(script, directory) if p.exists()]
    if present:
        raise RuntimeError("Intentional pause/STOP prevents qualification: " + str(present))


def preserve_stop(path, reason):
    """Retain a user's existing stop text, including when forwarding signals."""
    try:
        with Path(path).open("x") as stream:
            stream.write(reason + "\n")
    except FileExistsError:
        pass


def recovery_metadata(script, directory, *, prefix=False):
    """Use only this probe's artifacts; other arms never provide score checkpoints."""
    root = Path(script).resolve().parent
    plan = root / "reuse-plan.json"
    if not plan.is_file():
        raise ValueError("Record the family dependency-based reuse plan before qualification")
    # Validate the record before any worker or accounting mutation.
    if not isinstance(json.loads(plan.read_text()), dict):
        raise ValueError("Qualification reuse plan must be an object")
    return {
        "root": str(Path(directory).resolve()),
        "repair_record": str(plan),
        "stage": "screen",
        "evaluation_enabled": not prefix,
        "stop_after_checkpoint": prefix,
    }


def cached_worker_env(env, shared, initial, code_root, stop):
    """Override looser inherited/subprocess caps at the final worker boundary."""
    worker_env = {
        **os.environ,
        **(env or {}),
        "PYTHONPATH": str(code_root),
        "EXACT_OPENROUTER_LEDGER_DIR": str(shared / "openrouter"),
        "EXACT_OPENROUTER_REQUEST_CAP": str(initial["attempts"]),
        "EXACT_OPENROUTER_TOKEN_CAP": str(initial["billable_tokens"]),
        "EXACT_OPENROUTER_RETRY_UNKNOWN": "0",
        "EXACT_EXPERIMENT_STOP_FILE": str(stop),
    }
    cache_root = Path(worker_env.get("EXACT_EXPERIMENT_SHARED_CACHE_ROOT", str(shared)))
    if "EXACT_EXPERIMENT_SHARED_CACHE_ROOT" in worker_env:
        worker_env.setdefault("EXACT_EMBEDDING_CACHE_DIR", str(cache_root / "embeddings"))
        worker_env.setdefault("EXACT_NUMERICAL_CACHE_ROOT", str(cache_root))
    else:
        worker_env["EXACT_EMBEDDING_CACHE_DIR"] = str(shared / "embeddings")
    scope = Path(worker_env["EXACT_DATASET_CACHE_DIR"]).name
    worker_env["EXACT_DATASET_CACHE_DIR"] = str(cache_root / "datasets" / scope)
    return worker_env


@contextmanager
def accounted_probe(shared, work_id, forecast_seconds):
    """Account every admitted attempt, including setup and worker failures."""
    from exact.experiments.budget import BudgetLedger

    path = shared / "budget.json"
    state = json.loads(path.read_text())
    initial = usage(shared / "openrouter")
    ledger = BudgetLedger(path, state["limits"])
    ledger.admit(work_id, group="reserve", seconds=forecast_seconds, requests=0, tokens=0)
    started, outcome = time.time(), {"status": "failed"}
    try:
        yield initial, outcome, started
    finally:
        after = usage(shared / "openrouter")
        delta = {k: after[k] - initial[k] for k in initial}
        if any(delta.values()):
            outcome["status"] = "failed"
        ledger.finish(
            work_id,
            start=started,
            end=time.time(),
            status=outcome["status"],
            requests=delta["attempts"],
            tokens=delta["billable_tokens"],
            actual_usd=(
                None
                if delta["unknown"] or delta["unpriced_attempts"]
                else delta["reported_cost_usd"]
            ),
        )
        if any(delta.values()):
            raise ValueError("Cached-only qualification produced incremental hosted usage")


def validate_probe_config(model):
    if set(model.get("data", {}).get("refs", {})) - {"train", "valid"}:
        raise ValueError("Qualification may bind training/development references only")
    if model.get("llm", {}).get("experiment", {}).get("gate", {}).get("mode") != "off":
        raise ValueError("Qualification requires the declared decision gate off")


def validate_full_measurement(checkpoint, worker, calls, *, prefix=False):
    cursor = checkpoint.get("cursor", {}) if checkpoint else {}
    processed, total = cursor.get("next_pair", 0), cursor.get("dataset_rows", 0)
    if not 0 < processed <= total <= 6000:
        raise ValueError("Qualification has no valid durable pair checkpoint")
    if not prefix and (processed != total or worker.get("return_code") != 0 or calls != 1):
        raise ValueError("Full qualification needs every pair and one successful measured worker")


def run_worker(args):
    from exact.llm.routing import OpenRouterClient
    from tools.run_experiment_validation import worker

    original = OpenRouterClient._generation

    def templates_only(self, *positional, **keywords):
        if keywords.get("role") != "verbaliser":
            raise RuntimeError("Next-batch qualification allows verbalization templates only")
        return original(self, *positional, **keywords)

    OpenRouterClient._generation = templates_only
    try:
        return worker(args.worker, args.output, not args.prefix, False)
    finally:
        OpenRouterClient._generation = original


def run_probe(
    *,
    code_root,
    script,
    config,
    name,
    directory,
    shared,
    campaign,
    forecast_seconds,
    prefix=False,
    case_id="D0",
    resume=False,
):
    """One matched development-case probe; use prefix=False for family admission."""
    import psutil

    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments import harness
    from exact.experiments.campaign import _case_task
    from exact.experiments.recovery import ArtifactStore
    from exact.experiments.schema import ArmConfig, ResourceConfig, TaskConfig

    script, directory, shared, code_root = map(Path, (script, directory, shared, code_root))
    if case_id not in {"D0", "D1", "D0_E03"}:
        raise ValueError("Qualification requires a declared development case")
    if case_id == "D0_E03":
        original, bounded = campaign.cases["D0"], campaign.cases[case_id]
        for field in ("source", "target", "source_universe", "kind", "role", "task"):
            if getattr(original, field) != getattr(bounded, field):
                raise ValueError("Bounded training alias changed the D0 development population")
        if (
            bounded.role != "development"
            or bounded.references["valid"] != original.references["valid"]
        ):
            raise ValueError("Bounded training alias changed D0 development references")
    check_controls(script, directory)
    metadata = recovery_metadata(script, directory, prefix=prefix)
    model = dict(load_yaml_mapping(config))
    validate_probe_config(model)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "measurement.json"
    if receipt.exists():
        row = json.loads(receipt.read_text())
        for item in row["bindings"]:
            if binding(item["path"]) != item:
                raise ValueError("Retained qualification evidence changed")
        if row.get("case_id", "D0") != case_id:
            raise ValueError("Qualification case changed")
        if row.get("config") != binding(config):
            raise ValueError("Qualification configuration changed")
        accounting = json.loads((shared / "budget.json").read_text())["work"].get(
            row["budget_work_id"], {}
        )
        if accounting.get("status") != "complete":
            raise ValueError("Qualification receipt has no completed accounting entry")
        if (
            row.get("prefix") != prefix
            or row.get("status") != "passed"
            or row.get("execution_status") != ("interrupted" if prefix else "complete")
            or any(row.get("new_usage", {}).values())
        ):
            raise ValueError("Retained qualification does not match its declared complete scope")
        validate_full_measurement(
            {"cursor": {"next_pair": row["processed_pairs"], "dataset_rows": row["dataset_rows"]}},
            row["worker_measurement"],
            row["worker_calls"],
            prefix=prefix,
        )
        return row
    continuing = (directory / "run").exists() or (directory / "recovery").exists()
    if continuing and not resume:
        raise ValueError(
            "Unfinished qualification needs recorded recovery; resumed time is not a whole-arm measurement"
        )
    previous_work = []
    if continuing:
        state = json.loads((shared / "budget.json").read_text())
        prefix_id = "qualification/" + script.parent.name + "/" + name + "/"
        previous_work = [key for key in state["work"] if key.startswith(prefix_id)]
        if not previous_work or any(
            state["work"][key]["status"] not in {"failed", "interrupted"} for key in previous_work
        ):
            raise ValueError("Checkpoint continuation requires closed original accounting")
    work_id = "qualification/" + script.parent.name + "/" + name + "/" + os.environ["SLURM_STEP_ID"]
    expected_config = binding(config)
    stop = directory / "STOP"
    with accounted_probe(shared, work_id, forecast_seconds) as (initial, outcome, started):
        suite = harness.LoadedSuite(
            "next-batch-qualification",
            campaign.baseline_id,
            (),
            None,
            binding(script)["sha256"],
            None,
            None,
            {},
            model_lock=campaign.model_lock.verify(script.parent),
            model_lock_hash=campaign.model_lock.sha256,
            model_lock_payload=dict(load_yaml_mapping(campaign.model_lock.verify(script.parent))),
        )
        from exact.core.entities.configs.config import ConfigModel

        task = TaskConfig.model_validate(
            _case_task(campaign.cases[case_id], case_id, "global_alignment", "valid", script.parent)
        )
        label, modes = harness.resolve_supervision(
            ConfigModel.model_validate(model),
            task=task,
            arm=ArmConfig(id=name, role="baseline", overlay={}),
        )
        cell = harness.RunCell(
            "next-batch-qualification",
            "QUAL-" + case_id,
            "screen",
            name,
            "baseline",
            case_id + "-global_alignment",
            "development",
            "valid",
            "known_incomplete",
            17,
            300,
            ResourceConfig(kind="gpu", device="0"),
            directory / "run",
            model,
            harness.hash_payload(model),
            binding(script)["sha256"],
            binding(config)["sha256"],
            None,
            label,
            modes,
            campaign.cases[case_id].negative_policy,
            recovery=metadata,
            generate_rationales=False,
        )
        from dataclasses import replace

        planned_cell = replace(cell, recovery={**cell.recovery, "reuse_plan_only": True})
        check_controls(script, directory)
        harness.execute_cell(planned_cell, suite, workdir=code_root, resume=True)
        if continuing:
            from exact.experiments.runtime import CellRecovery

            expected = CellRecovery(
                planned_cell,
                harness._provenance_payload(planned_cell, suite, workdir=code_root),
                code_root,
            )
            preserve_training_progress(directory, expected.identities["extraction"])
        calls = []
        rss_limit = memory_limit_bytes()

        def launch(command, *, cwd, stdout_path, stderr_path, env=None):
            wrapper = load_yaml_mapping(Path(command[-1]))["job"]
            check_controls(script, directory)
            worker_env = cached_worker_env(env, shared, initial, code_root, stop)
            args = [
                sys.executable,
                "-u",
                str(script),
                "--code-root",
                str(code_root),
                "--worker",
                wrapper["config_file"],
                "--output",
                wrapper["output_dir"],
            ]
            if prefix:
                args.append("--prefix")
            peak, begin, forwarded = 0, time.monotonic(), False
            with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
                child = subprocess.Popen(
                    args,
                    cwd=code_root,
                    env=worker_env,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
                calls.append(child.pid)
                try:
                    while child.poll() is None:
                        try:
                            process = psutil.Process(child.pid)
                            peak = max(
                                peak,
                                sum(
                                    p.memory_info().rss
                                    for p in [process, *process.children(recursive=True)]
                                    if p.is_running()
                                ),
                            )
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                        if peak > rss_limit:
                            preserve_stop(
                                stop, f"Cooperative process-tree RSS guard ({rss_limit} bytes)"
                            )
                        try:
                            check_controls(script, directory)
                        except (RuntimeError, OSError, ValueError, KeyError) as error:
                            preserve_stop(stop, str(error))
                        if stop.exists() and not forwarded:
                            try:
                                os.killpg(child.pid, signal.SIGINT)
                            except ProcessLookupError:
                                pass
                            forwarded = True
                        time.sleep(2)
                finally:
                    if child.poll() is None:
                        try:
                            os.killpg(child.pid, signal.SIGINT)
                        except ProcessLookupError:
                            pass
                        child.wait()
            return child.returncode, time.monotonic() - begin, peak // 1024

        original = harness._run_subprocess
        original_measurement = harness._execution_measurement

        def measured_attempts(cell, recovery, **values):
            measurement = original_measurement(cell, recovery, **values)
            if continuing and values["complete"] and "extraction" not in recovery.reuse:
                measurement.update(
                    status="measured",
                    reason="Cumulative interrupted execution; includes restart overhead",
                    wall_seconds=values["elapsed"]
                    + sum(state["work"][key]["seconds"] for key in previous_work),
                    measurement_scope="cumulative_attempts",
                    accounting_work_ids=[*previous_work, work_id],
                )
                harness._atomic_json(
                    cell.output_dir / "stats/execution_measurement.json", measurement
                )
            return measurement

        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

        def requested_stop(signum, _frame):
            preserve_stop(stop, f"Cooperative signal {signum}")

        for sig in old_handlers:
            signal.signal(sig, requested_stop)
        harness._run_subprocess = launch
        harness._execution_measurement = measured_attempts
        try:
            check_controls(script, directory)
            result = harness.execute_cell(cell, suite, workdir=code_root, resume=True)
            expected = "interrupted" if prefix else "complete"
            if result["status"] != expected or stop.exists():
                raise ValueError(
                    f"Qualification {name} did not reach its declared {expected} boundary"
                )
            runtime = json.loads((cell.output_dir / "recovery-runtime.json").read_text())
            checkpoint = ArtifactStore(directory).latest_checkpoint(
                runtime["identity"]["artifact_id"]
            )
            worker_path = cell.output_dir / "validation-worker.json"
            worker = json.loads(worker_path.read_text())
            validate_full_measurement(checkpoint, worker, len(calls), prefix=prefix)
            after = usage(shared / "openrouter")
            delta = {k: after[k] - initial[k] for k in initial}
            if delta["attempts"] or delta["unknown"] or delta["unpriced_attempts"]:
                raise ValueError("Qualification unexpectedly needed new hosted work")
            paths = [
                cell.manifest_path,
                worker_path,
                cell.output_dir / "timings.json",
                cell.output_dir / "recovery-runtime.json",
            ]
            if not prefix:
                ArtifactStore(directory).verify(runtime["identity"]["artifact_id"])
                paths += [
                    cell.output_dir / "alignment/paper.maps_global.tsv",
                    cell.output_dir / "source_decisions.json",
                    cell.output_dir / "dataset/candidate_pool_sample_manifest.json",
                ]
            row = {
                "status": "passed",
                "name": name,
                "case_id": case_id,
                "execution_status": result["status"],
                "prefix": prefix,
                "processed_pairs": checkpoint["cursor"]["next_pair"],
                "dataset_rows": checkpoint["cursor"]["dataset_rows"],
                "wall_seconds": time.time() - started,
                "worker_measurement": worker,
                "new_usage": delta,
                "worker_calls": len(calls),
                "output_dir": str(cell.output_dir),
                "bindings": [binding(p) for p in paths],
                "source_cap": 300,
                "seed": 17,
                "generate_rationales": False,
                "no_private_test_references": True,
                "budget_work_id": work_id,
                "config": expected_config,
            }
            if continuing:
                row["budget_work_ids"] = [*previous_work, work_id]
                row["wall_seconds"] += sum(state["work"][key]["seconds"] for key in previous_work)
                row["timing_scope"] = (
                    "Sum of recorded execution attempts; includes interruption overhead"
                )
                row["checkpoint_continued"] = True
            outcome["status"] = "complete"
        finally:
            harness._run_subprocess = original
            harness._execution_measurement = original_measurement
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)

    harness._atomic_json(receipt, row)
    return row
