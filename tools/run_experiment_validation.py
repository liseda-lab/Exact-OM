#!/usr/bin/env python3
"""Bounded G0 operational validation. Never launches or admits an experiment campaign.

Run inside tmux/nohup; status.json, per-run logs, budget-plan.json and report.json
are the monitoring interface. Only D0 training/development inputs are accessible.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    from exact.experiments.harness import _atomic_json

    _atomic_json(Path(path), value)


def node_info(output):
    import psutil

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()
    packages = {}
    for name in ("exact-om", "torch", "transformers", "pyowl-core", "pyowl2vec-star-projector"):
        packages[name] = importlib.metadata.version(name)
    disk = psutil.disk_usage(output)
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "available_ram_bytes": psutil.virtual_memory().available,
        "gpu": gpu,
        "disk_free_bytes": disk.free,
        "packages": packages,
    }


def bounded_training(case, root, output, limit=64):
    """Bound feature extraction, rather than applying a label budget after extraction."""

    from exact.utils.data import read_table
    from exact.utils.provenance import sha256_file

    pool = read_table(case.candidates["train"].verify(root))
    reference = read_table(case.references["train"].verify(root))
    source = "Src" if "Src" in pool else "SrcEntity"
    groups = sorted(
        set(pool[source].astype(str)),
        key=lambda item: hashlib.sha256(f"17\0{item}".encode()).hexdigest(),
    )[:limit]
    if len(groups) != limit:
        raise ValueError(f"Need {limit} independent training groups for this declared probe")
    selected = pool.loc[pool[source].astype(str).isin(groups)].copy()
    if "confirmed_label" not in selected:
        raise ValueError("Bounded fitting requires explicit benchmark-confirmed training labels")
    destination = Path(output) / "inputs"
    destination.mkdir(parents=True, exist_ok=True)
    pool_path, reference_path = (
        destination / "train.candidates.tsv",
        destination / "train.reference.tsv",
    )
    for path, frame in (
        (pool_path, selected),
        (reference_path, reference.loc[reference.iloc[:, 0].astype(str).isin(groups)]),
    ):
        text = frame.to_csv(index=False, sep="\t")
        if path.exists() and path.read_text() != text:
            raise ValueError(f"Immutable validation training population changed: {path}")
        path.write_text(text)
    reporting = set(case.source_universe.verify(root).read_text().splitlines())
    if reporting & set(groups):
        raise ValueError("Validation train/development source overlap")
    return {
        "pool": str(pool_path),
        "reference": str(reference_path),
        "sources": groups,
        "source_groups": len(groups),
        "pairs": len(selected),
        "pool_sha256": sha256_file(pool_path),
        "reference_sha256": sha256_file(reference_path),
    }


def validation_config(
    base, case, root, training, *, mode, cap, hosted, fit, evaluate, generate_rationales=False
):
    from exact.core.entities.configs.config import ConfigModel
    from exact.experiments.campaign import _case_task
    from exact.experiments.harness import deep_merge
    from exact.experiments.rationale_policy import apply_rationale_policy

    task = _case_task(case, "D0", mode, "valid", root)
    config = deep_merge(base, task["overlay"])
    config["run"].update(seed=17, source_cap=cap, experiment_audit=True)
    config["inference"].update(checkpoint_every=1, log_every=1, num_workers=0)
    config["dataset"]["num_workers"] = 0
    config["data"].update(track=None, task=None, descriptor=None)
    config["data"]["reference_role"] = "valid" if evaluate else None
    config["data"]["refs"] = {"valid": config["data"]["refs"]["valid"]} if evaluate else {}
    config["data"]["train_candidates"] = training["pool"] if fit else None
    if fit:
        config["data"]["refs"]["train"] = training["reference"]
    if not hosted:
        config["dataset"]["verbalization_mode"] = "deterministic"
        config["llm"]["verbaliser"]["model"] = None
        for model in config["pipeline"]:
            if model["name"] == "PairAdaptiveSemanticScorer":
                model["params"].update(use_llm=False, generate_llm_rationales=False)
    config = ConfigModel.from_mapping(config, warn_v1=False).model_dump(mode="json", by_alias=True)
    config = apply_rationale_policy(config, generate_rationales=hosted and generate_rationales)
    return config, task


def ledger_totals(path):
    from exact.llm.ledger import RequestLedger

    roles = RequestLedger(path).summary()["roles"]
    keys = (
        "attempts",
        "prompt_tokens",
        "completion_tokens",
        "reported_cost_usd",
        "unknown",
        "unpriced_attempts",
    )
    return {key: sum(item[key] for item in roles.values()) for key in keys}


def forecast(rows, *, elapsed, max_seconds, requests, tokens, limits):
    """Conservative baseline-only bound; no claim that unmeasured families are admitted."""
    by_id = {row["id"]: row for row in rows}
    cold, warm, fit = (by_id[key]["wall_seconds"] for key in ("cold64", "warm64", "fit64"))
    hosted = by_id.get("hosted20")
    hosted_usage = hosted.get("measurement_usage", hosted["new_usage"]) if hosted else None
    # Production, independent warm recovery check, and local production: 900 source-visits.
    remaining_seconds = 1.5 * (cold * 4 + 900 * warm / 64 + 3 * fit)
    remaining_requests = remaining_tokens = 0
    if hosted:
        remaining_seconds += 1.5 * hosted["wall_seconds"] * 600 / 20
        remaining_requests = int(1.5 * hosted_usage["attempts"] * 600 / 20)
        remaining_tokens = int(
            1.5 * (hosted_usage["prompt_tokens"] + hosted_usage["completion_tokens"]) * 600 / 20
        )
    fits = (
        (max_seconds is None or elapsed + remaining_seconds <= max_seconds)
        and requests + remaining_requests <= limits["requests"]
        and tokens + remaining_tokens <= limits["tokens"]
    )
    return {
        "safety_factor": 1.5,
        "elapsed_seconds": elapsed,
        "remaining_seconds": remaining_seconds,
        "remaining_requests": remaining_requests,
        "remaining_tokens": remaining_tokens,
        "fits_limits": fits,
        "limits": limits,
        "hosted_measured": hosted is not None,
        "scope": "D0 baseline operational validation only; no campaign admission or treatment forecast",
        "conservative_policy": "Include cold reloads and repeated scoring; do not assume cross-mode hosted cache overlap.",
    }


class ValidationBudgetExceeded(RuntimeError):
    pass


def cold_budget_bound(cold, *, elapsed, limits):
    """Reject an impossible continuation without spending hosted or fitting work."""
    seconds = 1.5 * 4 * cold["wall_seconds"]
    return {
        "safety_factor": 1.5,
        "elapsed_seconds": elapsed,
        "remaining_seconds_lower_bound": seconds,
        "fits_limits": (
            limits["soft_seconds"] is None or elapsed + seconds <= limits["soft_seconds"]
        ),
        "limits": limits,
        "unmeasured": ["warm64", "hosted20", "fit64", "production300"],
        "scope": "Cold reload term of the existing G0 forecast; omitted terms are nonnegative",
    }


def worker(config_path, output, evaluate, require_dataset_cache=False):
    import torch

    from exact.core.actions.alignment import run_alignment
    from exact.impl.datasets.base import BaseAlignmentDataset
    from exact.impl.models.scorer_common import ScorerCommonMixin
    from exact.llm.routing import OpenRouterClient

    counts = {"scorer_encoder_batches": 0, "scorer_encoded_texts": 0}
    original_encode = ScorerCommonMixin._encode_texts
    original_init = OpenRouterClient.__init__
    original_has_cache = BaseAlignmentDataset.has_cache

    def has_cache(self):
        cached = original_has_cache(self)
        if not cached:
            raise ValueError("Warm dataset cache is incompatible; refusing a silent cold rebuild")
        counts["dataset_cache_hits"] = counts.get("dataset_cache_hits", 0) + 1
        return True

    def encode(self, tokenizer, model, texts, max_len):
        counts["scorer_encoder_batches"] += 1
        counts["scorer_encoded_texts"] += len(texts)
        return original_encode(self, tokenizer, model, texts, max_len)

    def client_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.max_retries = 0
        self.retry_unknown_requests = False

    ScorerCommonMixin._encode_texts = encode
    OpenRouterClient.__init__ = client_init
    if require_dataset_cache:
        BaseAlignmentDataset.has_cache = has_cache
    started = time.monotonic()
    code = 0
    try:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
        run_alignment(
            output_dir_path=output, configs_file_path=config_path, run_eval=evaluate, device=0
        )
    except KeyboardInterrupt:
        code = 130
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        write_json(
            output / "validation-worker.json",
            {
                **counts,
                "return_code": code,
                "wall_seconds": time.monotonic() - started,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "peak_cuda_allocated_bytes": (
                    torch.cuda.max_memory_allocated(0) if torch.cuda.is_initialized() else None
                ),
                "peak_cuda_reserved_bytes": (
                    torch.cuda.max_memory_reserved(0) if torch.cuda.is_initialized() else None
                ),
            },
        )
    return code


def forward_stop(process, stop, forwarded):
    """Interrupt even ontology loading, which precedes the first trainer STOP poll."""
    if stop.exists() and not forwarded:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        return True
    return forwarded


def execute(args):
    import psutil

    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments import harness
    from exact.experiments.campaign import (
        load_campaign,
        openrouter_only,
        validate_baseline,
    )
    from exact.experiments.harness import LoadedSuite, RunCell
    from exact.experiments.schema import ResourceConfig
    from exact.utils.provenance import sha256_file

    if args.api_key_file is not None:
        key = args.api_key_file.expanduser().read_text().strip()
        if not key:
            raise ValueError("API-key file is empty")
        os.environ["OPENROUTER_API_KEY"] = key
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError(
            "Validation output already has a report; retain it and choose a new directory"
        )
    lock, _ = load_campaign(args.campaign)
    lock_root = args.campaign.resolve().parent
    validate_baseline(lock, lock_root)
    case = lock.cases["D0"]
    if case.role != "development":
        raise ValueError("Validation accepts only the declared D0 development case")
    for binding in (
        case.source,
        case.target,
        case.source_universe,
        case.references.get("valid"),
        case.candidates.get("valid"),
    ):
        if binding is None:
            raise ValueError("D0 full ontology/source/local-pool binding is missing")
        binding.verify(lock_root)
    base_path = lock.base_config if lock.base_config.is_absolute() else lock_root / lock.base_config
    base = openrouter_only(load_yaml_mapping(base_path), lock.openrouter_profile)
    training = bounded_training(case, lock_root, output)
    limits = {
        "requests": args.requests_cap,
        "tokens": args.tokens_cap,
        "seconds": args.max_seconds,
        "soft_seconds": args.max_seconds - 1800 if args.max_seconds is not None else None,
        "ram_gb": args.ram_gb,
    }
    info = {
        "schema_version": 1,
        "campaign": str(args.campaign.resolve()),
        "campaign_sha256": sha256_file(args.campaign),
        "baseline": lock.baseline_id,
        "training": training,
        "limits": limits,
        "node": node_info(output),
        "source": case.source.model_dump(mode="json"),
        "target": case.target.model_dump(mode="json"),
        "development_only": True,
        "changes_campaign_readiness": False,
        "generate_rationales": getattr(args, "generate_rationales", False),
        "production_scope": "300 development groups; frozen matching configuration, bounded64-group training population",
        "status": "prepared",
        "stages": [],
    }
    # Downtime is not charged, but a continuation retains all previous active time.
    started = time.monotonic()
    prior_elapsed = 0.0
    imported_cold = None
    imported_probes = {}
    prior_usage = {}
    global_repair = None
    if args.resume_from:
        from tools.validation_resume import adopt_cold_probe

        expected, _ = validation_config(
            base,
            case,
            lock_root,
            training,
            mode="global_alignment",
            cap=64,
            hosted=False,
            fit=False,
            evaluate=False,
        )
        expected = harness._bind_model_lock_revisions(
            expected,
            dict(load_yaml_mapping(lock.model_lock.verify(lock_root))),
            require_complete=True,
            hosted_only=True,
        )
        imported_cold, prior_elapsed = adopt_cold_probe(
            args.resume_from.resolve(),
            output,
            campaign_sha256=info["campaign_sha256"],
            limits=limits,
            expected_config=expected,
            materialize=args.execute,
        )
        info["resume"] = {
            "from": str(args.resume_from.resolve()),
            "previous_elapsed_seconds": prior_elapsed,
            "scope": "Adopt verified cold64; budget-gate remaining work before a fresh warm64 worker",
        }
        info["stages"].append(imported_cold)
        write_json(output / "cold64.measurement.json", imported_cold)
    elif getattr(args, "resume_probes_from", None) or getattr(args, "resume_failed_from", None):
        from tools.validation_resume import (
            adopt_completed_probes,
            adopt_failed_validation,
        )

        failed_from = getattr(args, "resume_failed_from", None)
        previous = failed_from or args.resume_probes_from
        expected_configs = {}
        names = ("cold64", "warm64", "fit64")
        if failed_from:
            names += ("hosted20", "global300")
        for name in names:
            expected, _ = validation_config(
                base,
                case,
                lock_root,
                training,
                mode="global_alignment",
                cap=300 if name == "global300" else 20 if name == "hosted20" else 64,
                hosted=name in {"hosted20", "global300"},
                fit=name in {"fit64", "global300"},
                evaluate=name == "global300",
                generate_rationales=getattr(args, "generate_rationales", False),
            )
            expected_configs[name] = harness._bind_model_lock_revisions(
                expected,
                dict(load_yaml_mapping(lock.model_lock.verify(lock_root))),
                require_complete=True,
                hosted_only=True,
            )
        options = dict(
            campaign_sha256=info["campaign_sha256"],
            expected_configs=expected_configs,
            materialize=args.execute,
        )
        if failed_from:
            imported_probes, prior_elapsed, global_repair = adopt_failed_validation(
                previous.resolve(), output, **options
            )
        else:
            imported_probes, prior_elapsed = adopt_completed_probes(
                previous.resolve(), output, **options
            )
        imported_cold = imported_probes["cold64"]
        evidence = imported_cold["adoption_evidence"]
        previous_usage = evidence["prior_hosted_usage"]
        copied_usage = evidence.get("prior_ledger_usage", {})
        prior_usage = {
            key: value - copied_usage.get(key, 0) for key, value in previous_usage.items()
        }
        if any(value < 0 for value in prior_usage.values()):
            raise ValueError("Copied hosted usage exceeds the retained cumulative charges")
        if (
            prior_usage["attempts"] >= args.requests_cap
            or prior_usage["prompt_tokens"] + prior_usage["completion_tokens"] >= args.tokens_cap
        ):
            raise ValueError("Hosted allowance must exceed retained prior usage")
        info["resume"] = {
            "from": str(previous.resolve()),
            "previous_elapsed_seconds": prior_elapsed,
            "previous_hosted_usage": previous_usage,
            "copied_hosted_usage": copied_usage,
            "scope": (
                "Reuse completed probes and restore the verified output-repair checkpoint"
                if failed_from
                else "Reuse verified local probes; remeasure hosted20 with fresh responses"
            ),
            "global_repair": global_repair,
        }
        for name, row in imported_probes.items():
            info["stages"].append(row)
            write_json(output / f"{name}.measurement.json", row)

    def elapsed_seconds():
        return prior_elapsed + time.monotonic() - started

    write_json(output / "plan.json", info)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "plan": str(output / "plan.json"),
                    "run": "Repeat this command with --execute, preferably inside tmux.",
                }
            )
        )
        return 0
    shared = output / "shared"

    def usage_totals():
        current_usage = ledger_totals(shared / "openrouter")
        return {key: value + prior_usage.get(key, 0) for key, value in current_usage.items()}

    stop = output / "STOP"
    current = {"evaluate": False, "phase": "starting", "worker_calls": 0, "warm_from": None}
    from exact.experiments.runtime import CellRecovery

    original_run = harness._run_subprocess
    original_prepare = CellRecovery.prepare
    suite = LoadedSuite(
        "g0-validation",
        lock.baseline_id,
        (),
        None,
        "g0-validation",
        None,
        None,
        {},
        model_lock_payload=dict(load_yaml_mapping(lock.model_lock.verify(lock_root))),
    )
    info.update(status="running", pid=os.getpid(), started_at=time.time())

    def stopped(signum, frame):
        stop.write_text(f"signal {signum}\n")

    signal.signal(signal.SIGINT, stopped)
    signal.signal(signal.SIGTERM, stopped)

    def run(command, *, cwd, stdout_path, stderr_path, env=None):
        wrapper = load_yaml_mapping(Path(command[-1]))["job"]
        worker_output = Path(wrapper["output_dir"])
        if current["warm_from"] is not None:
            from tools.validation_resume import seed_warm_dataset

            seed_warm_dataset(Path(current["warm_from"]), worker_output)
        worker_env = {
            **os.environ,
            **(env or {}),
            "EXACT_OPENROUTER_LEDGER_DIR": str(shared / "openrouter"),
            "EXACT_EMBEDDING_CACHE_DIR": str(shared / "embeddings"),
            "EXACT_OPENROUTER_REQUEST_CAP": str(args.requests_cap - prior_usage.get("attempts", 0)),
            "EXACT_OPENROUTER_TOKEN_CAP": str(
                args.tokens_cap
                - prior_usage.get("prompt_tokens", 0)
                - prior_usage.get("completion_tokens", 0)
            ),
            "EXACT_OPENROUTER_RETRY_UNKNOWN": "0",
            "EXACT_EXPERIMENT_STOP_FILE": str(stop),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        if args.scratch_root:
            args.scratch_root.mkdir(parents=True, exist_ok=True)
            worker_env["TMPDIR"] = str(args.scratch_root.resolve())
        launch = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            wrapper["config_file"],
            "--worker-output",
            str(worker_output),
        ]
        if current["evaluate"]:
            launch.append("--evaluate")
        if current["warm_from"] is not None:
            launch.append("--require-dataset-cache")
        current["worker_calls"] += 1
        wall_start = time.monotonic()
        peak = 0
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                launch,
                cwd=ROOT,
                env=worker_env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            stop_forwarded = False
            while process.poll() is None:
                elapsed = elapsed_seconds()
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(
                        item.memory_info().rss
                        for item in [parent, *parent.children(recursive=True)]
                        if item.is_running()
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    rss = 0
                peak = max(peak, rss)
                if limits["soft_seconds"] is not None and elapsed >= limits["soft_seconds"]:
                    stop.write_text("foundation soft deadline\n")
                if rss > args.ram_gb * 1024**3:
                    stop.write_text("RAM limit exceeded\n")
                stop_forwarded = forward_stop(process, stop, stop_forwarded)
                if args.max_seconds is not None and elapsed >= args.max_seconds:
                    os.killpg(process.pid, signal.SIGKILL)
                write_json(
                    output / "status.json",
                    {
                        "status": "stopping" if stop.exists() else "running",
                        "phase": current["phase"],
                        "pid": os.getpid(),
                        "worker_pid": process.pid,
                        "elapsed_seconds": elapsed,
                        "peak_rss_bytes": peak,
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                    },
                )
                time.sleep(2)
            code = process.wait()
        return code, time.monotonic() - wall_start, peak // 1024

    def prepare(recovery):
        original_prepare(recovery)
        if recovery.metadata.get("checkpoint_repair"):
            from tools.validation_resume import seed_repaired_checkpoint

            seed_repaired_checkpoint(recovery.metadata, recovery.cell.output_dir)

    harness._run_subprocess = run
    CellRecovery.prepare = prepare
    cells = {}

    def stage(
        name,
        *,
        cap,
        hosted,
        fit=False,
        evaluate=False,
        mode="global_alignment",
        resume_from=None,
        interrupt=False,
        warm_from=None,
        repair_metadata=None,
    ):
        if stop.exists():
            raise InterruptedError("Validation STOP exists; completed artifacts retained")
        if limits["soft_seconds"] is not None and elapsed_seconds() >= limits["soft_seconds"]:
            raise ValidationBudgetExceeded("The cumulative validation time reached its soft limit")
        current.update(evaluate=evaluate, phase=name, warm_from=warm_from)
        config, task = validation_config(
            base,
            case,
            lock_root,
            training,
            mode=mode,
            cap=cap,
            hosted=hosted,
            fit=fit,
            evaluate=evaluate,
            generate_rationales=getattr(args, "generate_rationales", False),
        )
        config = harness._bind_model_lock_revisions(
            config, suite.model_lock_payload, require_complete=True, hosted_only=True
        )
        stage_root = output / name
        metadata = {
            "root": str(stage_root),
            "stage": "screen",
            "stop_after_checkpoint": interrupt,
            "evaluation_enabled": evaluate,
        }
        if resume_from:
            metadata["resume_from"] = str(cells[resume_from].output_dir.parent)
        if repair_metadata:
            metadata.update(repair_metadata)
        cell = RunCell(
            "g0-validation",
            "E00" if evaluate else "G0",
            "screen",
            "production",
            "baseline",
            f"D0-{mode}",
            "development",
            "valid",
            case.reference_completeness,
            17,
            cap,
            ResourceConfig(kind="gpu", device="0"),
            stage_root / "run",
            config,
            harness.hash_payload(config),
            info["campaign_sha256"],
            info["campaign_sha256"],
            None,
            "in_pair_supervised" if fit else "target_label_free",
            {},
            "confirmed_negatives",
            recovery=metadata,
            generate_rationales=getattr(args, "generate_rationales", False),
        )
        before, calls_before, wall = (
            usage_totals(),
            current["worker_calls"],
            time.monotonic(),
        )
        result = harness.execute_cell(cell, suite, workdir=ROOT, resume=False)
        after = usage_totals()
        row = {
            "id": name,
            "status": result["status"],
            "wall_seconds": time.monotonic() - wall,
            "output_dir": str(cell.output_dir),
            "new_worker_calls": current["worker_calls"] - calls_before,
            "new_usage": {key: after[key] - before[key] for key in before},
            "source_cap": cap,
            "hosted": hosted,
            "evaluate": evaluate,
            "manifest": str(cell.manifest_path),
            "recovery": result.get("recovery"),
            "peak_memory_kb": result.get("peak_memory_kb"),
            "timing_ledger": result.get("timing_ledger"),
        }
        measurement = cell.output_dir / "validation-worker.json"
        if measurement.exists():
            row["worker_measurement"] = json.loads(measurement.read_text())
        info["stages"].append(row)
        cells[name] = cell
        write_json(
            output / "status.json",
            {
                "status": "running",
                "phase": name,
                "completed_stages": info["stages"],
                "elapsed_seconds": elapsed_seconds(),
            },
        )
        write_json(output / f"{name}.measurement.json", row)
        expected = "interrupted" if interrupt else "complete"
        if result["status"] != expected:
            raise RuntimeError(f"{name}: {result.get('failure', result['status'])}")
        return row

    try:
        cold = imported_cold or stage("cold64", cap=64, hosted=False)
        bound = cold_budget_bound(cold, elapsed=elapsed_seconds(), limits=limits)
        if not bound["fits_limits"]:
            write_json(output / "budget-plan.json", bound)
            raise ValidationBudgetExceeded(
                "The measured cold-reload term alone exceeds the remaining validation budget; "
                "warm, hosted, fitting and production300 stages were not started."
            )
        if "warm64" not in imported_probes:
            stage("warm64", cap=64, hosted=False, warm_from=cold["output_dir"])
        if not args.skip_hosted and "hosted20" not in imported_probes:
            stage("hosted20", cap=20, hosted=True)
        fitted = imported_probes.get("fit64") or stage("fit64", cap=64, hosted=False, fit=True)
        fits = list(Path(fitted["output_dir"]).glob("fitting/**/training_units.json"))
        if not fits:
            raise RuntimeError("Small current-selector fit produced no training-unit artifact")
        totals = usage_totals()
        bound = forecast(
            info["stages"],
            elapsed=elapsed_seconds(),
            max_seconds=limits["soft_seconds"],
            requests=totals["attempts"],
            tokens=totals["prompt_tokens"] + totals["completion_tokens"],
            limits=limits,
        )
        write_json(output / "budget-plan.json", bound)
        if args.skip_hosted:
            info["status"] = "blocked_auth"
            info["reason"] = (
                "Local resource/fit checks completed; actual production300 held until hosted authentication and throughput succeed."
            )
        elif not bound["fits_limits"]:
            info["status"] = "blocked_budget"
            info["reason"] = (
                "Measured conservative production-validation forecast exceeds the remaining foundation limits."
            )
        else:
            stage(
                "global300",
                cap=300,
                hosted=True,
                fit=True,
                evaluate=True,
                repair_metadata=global_repair,
            )
            stage("global300-stop", cap=300, hosted=True, fit=True, evaluate=True, interrupt=True)
            stage(
                "global300-resume",
                cap=300,
                hosted=True,
                fit=True,
                evaluate=True,
                resume_from="global300-stop",
            )
            replay = stage(
                "global300-replay",
                cap=300,
                hosted=True,
                fit=True,
                evaluate=True,
                resume_from="global300-resume",
            )
            expected = (cells["global300"].output_dir / "alignment/maps_global.tsv").read_bytes()
            for name in ("global300-resume", "global300-replay"):
                if (cells[name].output_dir / "alignment/maps_global.tsv").read_bytes() != expected:
                    raise ValueError("Global interruption/relocation replay changed mappings")
            if replay["new_worker_calls"] or replay["new_usage"]["attempts"]:
                raise ValueError("Completed global replay performed new model work")
            stage("local300", cap=300, hosted=True, fit=True, evaluate=True, mode="local_ranking")
            replay = stage(
                "local300-replay",
                cap=300,
                hosted=True,
                fit=True,
                evaluate=True,
                mode="local_ranking",
                resume_from="local300",
            )
            if (cells["local300-replay"].output_dir / "alignment/maps_local.tsv").read_bytes() != (
                cells["local300"].output_dir / "alignment/maps_local.tsv"
            ).read_bytes():
                raise ValueError("Local relocation replay changed mappings")
            if replay["new_worker_calls"] or replay["new_usage"]["attempts"]:
                raise ValueError("Completed local replay performed new model work")
            info["status"] = "passed"
    except ValidationBudgetExceeded as exc:
        info.update(status="blocked_budget", reason=str(exc))
    except InterruptedError as exc:
        info.update(status="interrupted", reason=str(exc))
    except Exception as exc:
        info.update(
            status="interrupted" if stop.exists() else "failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
    finally:
        harness._run_subprocess = original_run
        CellRecovery.prepare = original_prepare
        info.update(
            ended_at=time.time(),
            elapsed_seconds=elapsed_seconds(),
            hosted_usage=usage_totals(),
            g0_admission="not_granted; review measurements, omitted scenarios and family-specific forecasts",
        )
        write_json(output / "report.json", info)
        write_json(output / "status.json", info)
    return 0 if info["status"] == "passed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument(
        "--resume-from",
        type=Path,
        help="Adopt a verified cold64 completed before the empty-evaluation bookkeeping failure",
    )
    recovery.add_argument(
        "--resume-probes-from",
        type=Path,
        help="Reuse verified cold64/warm64/fit64 from a budget-blocked G0; remeasure hosted20",
    )
    recovery.add_argument(
        "--resume-failed-from",
        type=Path,
        help="Recover verified completed probes and global300 after the source-decision writer failure",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Read the key into worker environment only; never write its value to artifacts",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-hosted", action="store_true")
    parser.add_argument(
        "--generate-rationales",
        action="store_true",
        help="Explicitly opt into narrative explanations",
    )
    wall_limit = parser.add_mutually_exclusive_group()
    wall_limit.add_argument("--max-seconds", type=int, default=43200)
    wall_limit.add_argument(
        "--no-time-limit",
        dest="max_seconds",
        action="store_const",
        const=None,
        help="Disable wall-time admission and deadlines; retain STOP, RAM and hosted limits",
    )
    parser.add_argument("--ram-gb", type=float, default=56)
    parser.add_argument("--requests-cap", type=int, default=2000)
    parser.add_argument("--tokens-cap", type=int, default=3200000)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--evaluate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--require-dataset-cache", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, args.worker_output, args.evaluate, args.require_dataset_cache)
    if not args.campaign or not args.output_root:
        parser.error("--campaign and --output-root are required")
    if (
        (args.max_seconds is not None and not 1800 < args.max_seconds <= 43200)
        or not 0 < args.ram_gb <= 60
        or args.requests_cap <= 0
        or args.tokens_cap <= 0
    ):
        parser.error(
            "Use a wall limit within (30min,12h] or --no-time-limit, RAM within (0,60] GiB, and positive hosted caps"
        )
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
