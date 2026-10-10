#!/usr/bin/env python3
"""Prepare finite, unpaid scoring checks for the existing supervisor; never publish.

The request binds reference/candidate source manifests, an environment JSON file,
and H0/H1/H2 config/workload/template files. Each case gets a separate numeric
Slurm step after the successful G4 recovery lineage. Scientific promotion and
full fitted-recipe verification remain separate from these binary fixtures.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from tools.prepared_batch import binding, completed_run, read, resolve_run, verified, write


def verify_source(ref):
    source = read(verified(ref))
    root = Path(source["root"])
    if not root.is_absolute() or not re.fullmatch(r"[a-f0-9]{40}", source["revision"]):
        raise ValueError("A frozen absolute source root and revision are required")
    files = source["files"]
    required = {"tools/benchmark_scoring_throughput.py", "exact/impl/models/pair_adaptive_scorer.py"}
    if not required <= set(files):
        raise ValueError("Frozen source inventory omits required scoring code")
    actual = {str(path.relative_to(root)) for folder in ("exact", "tools")
              for path in (root / folder).rglob("*.py")}
    if actual != set(files):
        raise ValueError("Frozen source module inventory changed")
    for relative, item in files.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts or verified(item) != root / relative:
            raise ValueError("Frozen source file escapes its declared root")
    return source


def freeze_source(root, revision, output):
    """Bind an already reviewed checkout/archive, including every Python module."""
    root = Path(root).resolve()
    files = {str(path.relative_to(root)): binding(path)
             for folder in ("exact", "tools") for path in sorted((root / folder).rglob("*.py"))}
    marker = root / "source-revision.txt"
    actual = (marker.read_text().strip() if marker.exists() else
              subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    if actual != revision:
        raise ValueError("Declared source revision differs from the reviewed source")
    write(output, {"kind": "throughput_source_inventory", "root": str(root),
                   "revision": revision, "files": files}, immutable=True)
    result = binding(output)
    verify_source(result)
    return result


def _prerequisite(recipe):
    registry = read(Path(recipe["supervisor"]) / "registry.json")
    run = resolve_run(registry, recipe["g4_dependency"])
    complete = completed_run(run)
    if (complete.get("step_id") != run["step_id"]
            or complete.get("dispatch_nonce") != run.get("dispatch_nonce")
            or int(Path(run["exit_path"]).read_text().strip()) != 0):
        raise ValueError("G4 completion ownership or terminal exit differs")
    return {"run_id": run["id"], "completion": binding(run["completion_path"])}


def _environment(recipe):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("EXACT_") and key not in {"OPENROUTER_API_KEY", "OPENAI_API_KEY"}}
    configured = read(verified(recipe["environment"]))
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in configured.items()):
        raise ValueError("Verification environment must contain string values")
    env.update(configured)
    root = Path(recipe["root"])
    temporary = root / "temporary"
    temporary.mkdir(exist_ok=True)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OPENROUTER_API_KEY="",
               OPENAI_API_KEY="", EXACT_PAIR_CONTEXT_BATCHING="0", EXACT_EVIDENCE_PREFETCH="0",
               EXACT_EXPERIMENT_ROLE="throughput_fixture", TMPDIR=str(temporary),
               EXACT_HOSTED_CACHE_ONLY="1", EXACT_OPENROUTER_LEDGER_DIR=str(root / "fixture-ledger"),
               EXACT_DATASET_CACHE_DIR=str(root / "prepared-cache"), EXACT_DATASET_CACHE_LOCAL_DIR="",
               PYTHONPYCACHEPREFIX=str(root / "pycache"), PYTHONDONTWRITEBYTECODE="1")
    return env


def _command(argv, *, root, env, phase, nonce, step):
    """A heartbeat reports liveness only; a quiet scorer is never declared failed."""
    with (root / (phase + ".log")).open("x") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
        while process.poll() is None:
            write(root / "status.json", {"status": "running", "phase": phase,
                  "updated_at": time.time(), "step_id": step, "dispatch_nonce": nonce,
                  "pid": process.pid, "paid_calls": 0})
            time.sleep(5)
        if process.returncode:
            raise RuntimeError(f"{phase} exited {process.returncode}; see {log.name}")


def _measurement(path, recipe, source, *, full):
    report = read(path)
    if (report.get("namespace") != "qualification_only" or report.get("paid_calls") != 0
            or report.get("fixture_outputs_promotable") is not False
            or report.get("source_revision") != source["revision"]
            or report.get("config") != recipe["config"]
            or report.get("workload") != recipe["workload"]
            or report.get("templates") != recipe["templates"]
            or str(report.get("slurm_job")) != recipe["allocation"]
            or str(report.get("slurm_step")) != os.getenv("SLURM_STEP_ID")
            or recipe["expected_gpu"] not in report.get("gpu", "")):
        raise ValueError("Measurement identity, GPU or unpaid fixture contract differs")
    precision = report.get("precision")
    if not isinstance(precision, dict) or any(type(precision.get(key)) is not bool
            for key in ("fp16", "cuda_matmul_allow_tf32")):
        raise ValueError("Measurement lacks observed precision settings")
    cold = report["cold"]
    chunks = cold["chunks"]
    count = sum(row["distinct_computed_pairs"] for row in chunks)
    seconds = sum(row["elapsed_seconds"] for row in chunks)
    if (count <= 0 or not math.isfinite(seconds) or seconds <= 0
            or count != cold["distinct_computed_pairs"]
            or not math.isclose(seconds, cold["elapsed_seconds"], rel_tol=1e-12)
            or not math.isclose(count / seconds, cold["pairs_per_second"], rel_tol=1e-12)):
        raise ValueError("Measurement has an empty or inconsistent cold denominator")
    if full and (count < 5000 or len(chunks) < 3 or seconds < 30
                 or cold.get("minimum_measurement_contract_met") is not True):
        raise ValueError("Full verification lacks 5000 unique cold pairs, three chunks or 30 seconds")
    if full and cold.get("throughput_target_status") != ("met" if count / seconds >= 200 else "missed"):
        raise ValueError("Throughput target status differs from the measured cold rate")
    if any(report.get(mode, {}).get("namespace") != "qualification_only" for mode in ("warm", "replay")):
        raise ValueError("Measurement lacks separate warm/replay receipts")
    return report


def run(recipe_path):
    recipe = read(recipe_path)
    root = Path(recipe["root"])
    step = os.getenv("SLURM_JOB_ID", "") + "." + os.getenv("SLURM_STEP_ID", "")
    nonce = recipe["nonce"]
    if any((root / name).exists() for name in ("completion.json", "reference", "candidate")):
        raise ValueError("Existing verification outputs require an inspected fresh recovery")
    code, result = 1, {}
    try:
        if not re.fullmatch(re.escape(recipe["allocation"]) + r"\.\d+", step):
            raise ValueError("Verification must run inside its reviewed numeric Slurm step")
        write(root / "step.json", {"step_id": step, "dispatch_nonce": nonce})
        for item in recipe["bindings"]:
            verified(item)
        prerequisite = _prerequisite(recipe)
        sources = {name: verify_source(recipe[name]) for name in ("reference", "candidate")}
        env = _environment(recipe)
        for name, source in sources.items():
            argv = [recipe["python"], "-u", str(Path(source["root"]) / "tools/benchmark_scoring_throughput.py"),
                    "run", "--config", recipe["config"]["path"], "--workload", recipe["workload"]["path"],
                    "--templates", recipe["templates"]["path"], "--output", str(root / name),
                    "--chunk-pairs", str(recipe["chunk_pairs"])]
            if name == "reference":
                argv += ["--max-chunks", str(recipe["reference_chunks"])]
            _command(argv, root=root, env=env, phase=name, nonce=nonce, step=step)
        reports = {name: _measurement(root / name / "measurement.json", recipe, source, full=name == "candidate")
                   for name, source in sources.items()}
        compare = Path(sources["candidate"]["root"]) / "tools/compare_scoring_parity.py"
        _command([recipe["python"], str(compare), "--baseline", str(root / "reference"),
                  "--candidate", str(root / "candidate"), "--output", str(root / "parity.json")],
                 root=root, env=env, phase="parity", nonce=nonce, step=step)
        parity = read(root / "parity.json")
        if (parity.get("status") != "passed" or parity.get("compared_rows", 0) <= 0
                or parity.get("compared_floats", 0) <= 0 or parity.get("query_boundaries_equal") is not True
                or parity.get("numeric_mismatch_count") != 0 or parity.get("discrete_mismatch_count") != 0):
            raise ValueError("Evidence parity failed or lacks nonempty verified comparisons")
        if reports["reference"].get("precision") != reports["candidate"].get("precision"):
            raise ValueError("Reference and candidate precision differ")
        result = {"prerequisite": prerequisite, "parity": binding(root / "parity.json"),
                  "measurements": {name: binding(root / name / "measurement.json") for name in sources},
                  "cold_pairs_per_second": reports["candidate"]["cold"]["pairs_per_second"],
                  "throughput_target_status": reports["candidate"]["cold"]["throughput_target_status"],
                  "scientific_promotion": False, "fitted_recipe_parity": "pending"}
        code = 0
    except Exception as exc:
        result = {"error": {"type": type(exc).__name__, "message": str(exc)}}
    finally:
        receipt = {"status": "complete" if code == 0 else "failed", "exit_code": code,
                   "step_id": step, "dispatch_nonce": nonce, "paid_calls": 0,
                   "recipe": binding(recipe_path), "updated_at": time.time(), **result}
        write(root / "status.json", receipt)
        write(root / "completion.json", receipt)
        (root / "exit-code").write_text(str(code) + "\n")
    return code


def prepare(request_path, destination):
    request = read(request_path)
    supervisor = Path(request["supervisor"]).resolve()
    registry_path, policy_path = supervisor / "registry.json", supervisor / "policy.json"
    registry_ref, policy_ref = binding(registry_path), binding(policy_path)
    registry, policy = read(registry_path), read(policy_path)
    dispatch_path = supervisor / "dispatch-state.json"
    dispatch_ref = binding(dispatch_path) if dispatch_path.exists() else None
    for name in ("reference", "candidate"):
        verify_source(request[name])
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    usage_root = Path(policy["storage_guard"]["usage_root"]).resolve()
    if not destination.is_relative_to(usage_root):
        raise ValueError("Verification outputs must be inside the approved storage guard root")
    identifiers = [case["id"] for case in request["cases"]]
    if not identifiers or len(set(identifiers)) != len(identifiers) or any(name not in {"H0", "H1", "H2"} for name in identifiers):
        raise ValueError("Only unique declared H0/H1/H2 checks are supported")
    rows = []
    parents = [request.get("g4_dependency", "G4-run-once-followup")]
    resolve_run(registry, parents[0])
    resources = request["resources"]
    if (resources.get("gpus") != 1 or resources.get("cpus", 0) <= 0
            or resources.get("memory_gib", 0) <= 0
            or any(value > registry["capacity"].get(key, 0) for key, value in resources.items())
            or not request.get("expected_gpu") or not Path(request["python"]).is_absolute()
            or request.get("reference_chunks", 1) < 1 or request.get("chunk_pairs", 512) < 1):
        raise ValueError("Verification resources, GPU, Python or chunk boundaries are invalid")
    for case in request["cases"]:
        root = destination / case["id"]
        root.mkdir()
        nonce = uuid.uuid4().hex
        identifier = destination.name + "-" + case["id"]
        inputs = [request[name] for name in ("reference", "candidate", "environment")]
        inputs += [case[name] for name in ("config", "workload", "templates")]
        for item in inputs:
            verified(item)
        workload = read(verified(case["workload"]))
        if (workload.get("pair") != case["id"] or workload.get("namespace") != "qualification_only"
                or workload.get("reference_labels_used") is not False):
            raise ValueError("Verification requires the matching label-free qualification workload")
        recipe = {**case, "root": str(root), "nonce": nonce, "allocation": str(policy["allocation"]),
                  "supervisor": str(supervisor), "g4_dependency": request.get("g4_dependency", "G4-run-once-followup"),
                  "python": request["python"], "reference": request["reference"], "candidate": request["candidate"],
                  "environment": request["environment"], "expected_gpu": request["expected_gpu"],
                  "chunk_pairs": request.get("chunk_pairs", 512), "reference_chunks": request.get("reference_chunks", 1),
                  "bindings": inputs, "paid_calls": 0, "network_forbidden": True}
        recipe_path = root / "recipe.json"
        write(recipe_path, recipe, immutable=True)
        worker_tool = Path(read(verified(request["candidate"]))["root"]) / "tools/prepare_throughput_checks.py"
        worker = root / "worker-entry.sh"
        command = [request["python"], str(worker_tool), "run", "--recipe", str(recipe_path)]
        worker.write_text("#!/usr/bin/env bash\nset -uo pipefail\n" + shlex.join(command) + "\n" +
                          "result=$?\nif [ ! -e " + shlex.quote(str(root / "exit-code")) + " ]; then\n" +
                          "  printf '%s\\n' \"$result\" > " + shlex.quote(str(root / "exit-code")) +
                          "\nfi\nexit \"$result\"\n")
        launch = {"nonce": nonce, "tmux_socket": request.get("tmux_socket", f"/tmp/tmux-{os.getuid()}/default"),
                  "argv": ["/usr/bin/srun", "--jobid=" + str(policy["allocation"]), "--overlap", "--immediate=15",
                           "--nodes=1", "--ntasks=1", "--cpus-per-task=" + str(resources["cpus"]),
                           "--mem=" + str(int(resources["memory_gib"] * 1024)),
                           "--cpu-bind=none", "--gres=" + request["gres"], "--time=0", "/bin/bash", str(worker)],
                  "bindings": [binding(worker), binding(worker_tool), binding(recipe_path), *inputs],
                  "pause_paths": [str(root / "STOP")], "step_path": str(root / "step.json"),
                  "launcher_log": str(root / "launcher.log"),
                  "run": {"id": identifier, "status_path": str(root / "status.json"),
                          "completion_path": str(root / "completion.json"), "exit_path": str(root / "exit-code")}}
        from tools.storage_guard import guard_launch as storage_guard
        from tools.hosted_prompt_guard import guard_launch as hosted_guard
        launch = storage_guard(launch, policy, supervisor)
        launch = hosted_guard(launch, policy, recipe_path=recipe_path,
                              receipt_path=root / "hosted-prompt-guard.json", mode="no_new_hosted_calls",
                              evidence=[request["reference"], request["candidate"], binding(worker_tool)])
        row = {"id": identifier, "enabled": False, "needs_user": False, "depends_on": parents,
               "resources": resources, "priority": 100, "launch": launch,
               "scope": "Unpaid binary-fixture GPU parity and cold/warm/replay measurement; no scientific promotion."}
        from exact.experiments.dispatch import _validate, _validate_resource_binding
        _validate(row, str(policy["allocation"]))
        _validate_resource_binding(registry, row)
        rows.append(row)
    if binding(registry_path) != registry_ref or binding(policy_path) != policy_ref or (
            binding(dispatch_path) if dispatch_path.exists() else None) != dispatch_ref:
        raise ValueError("Supervisor snapshot changed during preparation")
    output = destination / "queue-proposal.json"
    write(output, {"kind": "offline_throughput_checks", "registry_snapshot": registry_ref,
                   "policy_snapshot": policy_ref, "dispatch_snapshot": dispatch_ref,
                   "request": binding(request_path), "proposed_pending_batches": rows,
                   "live_queue_changed": False, "launchable": False,
                   "remaining_checks": ["G4 recovery and corrected selection/fitting freeze",
                       "Actual fitted selected-recipe GPU parity, grouped/exemplar fixtures and full mapping parity",
                       "All-root storage, hosted and whole-program cost/runtime forecast",
                       "Explicit corrected-campaign rollout admission"]}, immutable=True)
    return binding(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--request", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    worker = sub.add_parser("run")
    worker.add_argument("--recipe", type=Path, required=True)
    freeze = sub.add_parser("freeze-source")
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--revision", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        return run(args.recipe)
    result = (prepare(args.request, args.output) if args.command == "prepare" else
              freeze_source(args.root, args.revision, args.output))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
