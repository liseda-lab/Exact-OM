"""Finite GPU qualifications cannot silently become scientific or successful work."""
import json
from pathlib import Path
import sys
import subprocess

import pytest

from tools import hosted_prompt_guard, storage_guard
from tools import prepare_throughput_checks as checks
from exact.experiments.supervision import inspect_runs, pending_batches


def save(path, value):
    checks.write(path, value)
    return checks.binding(path)


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "source"
    for relative in ("tools/benchmark_scoring_throughput.py", "tools/compare_scoring_parity.py",
                     "tools/prepare_throughput_checks.py", "exact/impl/models/pair_adaptive_scorer.py"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture only\n")
    (source / "source-revision.txt").write_text("a" * 40)
    frozen = checks.freeze_source(source, "a" * 40, tmp_path / "source.json")
    supervisor = tmp_path / "supervisor"
    supervisor.mkdir()
    complete = save(tmp_path / "g4-complete.json", {"status": "complete", "exit_code": 0,
                    "step_id": "123.7", "dispatch_nonce": "g4-owned-nonce"})
    exit_path = tmp_path / "g4-exit"
    exit_path.write_text("0\n")
    registry = {"capacity": {"cpus": 6, "gpus": 1, "memory_gib": 112}, "pending_batches": [],
                "runs": [{"id": "G4", "step_id": "123.7", "dispatch_nonce": "g4-owned-nonce",
                          "completion_path": complete["path"], "exit_path": str(exit_path)}]}
    save(supervisor / "registry.json", registry)
    save(supervisor / "dispatch-state.json", {})
    prompt = save(tmp_path / "prompt.json", {"schema_version": 1, "max_input_tokens": 8192,
                  "max_input_bytes": 32768, "max_output_tokens": 1024, "chat_overhead_tokens": 128})
    policy = {"allocation": "123", "node": "fixture-node", "storage_guard": {
        "python": sys.executable, "usage_root": str(tmp_path), "min_free_bytes": 1,
        "max_used_bytes": 10**9, "growth_reserve_bytes": 100,
        "source": checks.binding(Path(storage_guard.__file__))},
        "hosted_prompt_guard": {"schema_version": 1, "prompt_policy": prompt,
            "allowed_source_sha256": {name: ["a" * 64] for name in hosted_prompt_guard.REQUIRED_SOURCE_FILES}}}
    save(supervisor / "policy.json", policy)
    request = {"supervisor": str(supervisor), "python": sys.executable, "reference": frozen,
               "candidate": frozen, "environment": save(tmp_path / "env.json", {}),
               "expected_gpu": "RTX 4090", "gres": "gpu:rtx4090:1", "g4_dependency": "G4",
               "resources": {"cpus": 6, "gpus": 1, "memory_gib": 112}, "cases": [
                   {"id": name, "workload": save(tmp_path / f"{name}-workload.json", {
                       "namespace": "qualification_only", "pair": name, "reference_labels_used": False}),
                    "config": save(tmp_path / f"{name}.json", {}),
                    "templates": save(tmp_path / f"{name}-templates.json", {})}
                   for name in ("H0", "H1", "H2")]}
    request_path = tmp_path / "request.json"
    save(request_path, request)
    proposal_ref = checks.prepare(request_path, tmp_path / "checks")
    proposal = checks.read(proposal_ref["path"])
    return tmp_path, request, registry, policy, proposal


def test_prepare_preserves_live_registry_and_produces_guarded_disabled_tasks(prepared):
    _, request, registry, policy, proposal = prepared
    supervisor = Path(request["supervisor"])
    assert checks.read(supervisor / "registry.json") == registry
    assert proposal["live_queue_changed"] is False
    assert proposal["launchable"] is False
    assert len(proposal["proposed_pending_batches"]) == 3
    for row in proposal["proposed_pending_batches"]:
        assert row["enabled"] is False
        assert row["depends_on"] == ["G4"]  # Independent failures do not strand another case.
        launch = row["launch"]
        assert "--mem=114688" in launch["argv"]
        assert "--jobid=123" in launch["argv"]
        storage_guard.validate_launch(launch, policy, supervisor)
        hosted_prompt_guard.validate_launch(launch, policy)
        receipt = checks.read(launch["hosted_prompt_guard"]["path"])
        assert receipt["mode"] == "no_new_hosted_calls"
        assert len(receipt["evidence"]) == 3


def test_pending_check_follows_successful_g4_recovery_and_waits_for_gpu(prepared):
    _, _, registry, _, proposal = prepared
    row = proposal["proposed_pending_batches"][0]
    row["enabled"] = True
    registry["pending_batches"] = [row]
    predecessor = registry["runs"][0]
    predecessor.update(enabled=False, superseded_by="G4-recovery")
    registry["runs"].append({**predecessor, "id": "G4-recovery", "enabled": True,
                             "step_id": "123.8", "resources": row["resources"]})
    registry["runs"][-1].pop("superseded_by")
    active = inspect_runs(registry["runs"], step_states={"123.8": "RUNNING"})
    assert not pending_batches(registry, active)
    done = inspect_runs(registry["runs"], step_states={})
    assert [item["batch_id"] for item in pending_batches(registry, done)] == [row["id"]]


def _recipe(prepared):
    root = prepared[0] / "checks/H0"
    return root / "recipe.json", checks.read(root / "recipe.json")


def _fake_measurement(recipe, *, full):
    count, seconds = (6000, 300.0) if full else (200, 10.0)
    chunks = [{"distinct_computed_pairs": count // (3 if full else 1),
               "elapsed_seconds": seconds / (3 if full else 1)} for _ in range(3 if full else 1)]
    return {"namespace": "qualification_only", "paid_calls": 0, "fixture_outputs_promotable": False,
            "source_revision": "a" * 40, "gpu": "NVIDIA GeForce RTX 4090",
            "slurm_job": "123", "slurm_step": "99",
            "config": recipe["config"], "workload": recipe["workload"], "templates": recipe["templates"],
            "precision": {"fp16": True, "cuda_matmul_allow_tf32": False},
            "cold": {"chunks": chunks, "elapsed_seconds": seconds, "distinct_computed_pairs": count,
                     "pairs_per_second": count / seconds, "minimum_measurement_contract_met": full,
                     "throughput_target_status": "missed"},
            "warm": {"namespace": "qualification_only"}, "replay": {"namespace": "qualification_only"}}


def _fake_commands(monkeypatch, *, failed_parity=False, invalid_count=False):
    observed = []
    def command(argv, *, root, env, phase, nonce, step):
        observed.append((phase, env))
        recipe = checks.read(root / "recipe.json")
        if phase == "parity":
            save(root / "parity.json", {"status": "failed" if failed_parity else "passed",
                "compared_rows": 200, "compared_floats": 1500, "query_boundaries_equal": True,
                "numeric_mismatch_count": int(failed_parity), "discrete_mismatch_count": 0})
        else:
            output = root / phase
            output.mkdir()
            report = _fake_measurement(recipe, full=phase == "candidate")
            if invalid_count and phase == "candidate":
                report["cold"]["distinct_computed_pairs"] += 1
            save(output / "measurement.json", report)
    monkeypatch.setattr(checks, "_command", command)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "99")
    return observed


def test_completed_measurement_reports_target_miss_without_scientific_promotion(prepared, monkeypatch):
    path, recipe = _recipe(prepared)
    observed = _fake_commands(monkeypatch)
    monkeypatch.setenv("EXACT_OPENROUTER_LEDGER_DIR", "/scientific-ledger")
    monkeypatch.setenv("EXACT_EXPERIMENT_RUNTIME", "/scientific-runtime")
    assert checks.run(path) == 0
    root = Path(recipe["root"])
    receipt = checks.read(root / "completion.json")
    assert receipt["status"] == "complete" and receipt["throughput_target_status"] == "missed"
    assert receipt["scientific_promotion"] is False and receipt["fitted_recipe_parity"] == "pending"
    assert receipt["dispatch_nonce"] == recipe["nonce"] and receipt["step_id"] == "123.99"
    assert [phase for phase, _ in observed] == ["reference", "candidate", "parity"]
    for _, env in observed:
        assert "EXACT_EXPERIMENT_RUNTIME" not in env
        assert env["EXACT_OPENROUTER_LEDGER_DIR"] == str(root / "fixture-ledger")
        assert env["EXACT_HOSTED_CACHE_ONLY"] == "1" and env["EXACT_PAIR_CONTEXT_BATCHING"] == "0"
        assert env["OPENROUTER_API_KEY"] == "" and env["HF_HUB_OFFLINE"] == "1"
    old = (root / "completion.json").read_bytes()
    with pytest.raises(ValueError, match="Existing verification"):
        checks.run(path)
    assert (root / "completion.json").read_bytes() == old


@pytest.mark.parametrize("bad", ["parity", "denominator", "source", "prerequisite"])
def test_untrustworthy_results_fail_with_owned_receipt(prepared, monkeypatch, bad):
    path, recipe = _recipe(prepared)
    observed = _fake_commands(monkeypatch, failed_parity=bad == "parity", invalid_count=bad == "denominator")
    if bad == "source":
        source = checks.read(recipe["candidate"]["path"])
        Path(source["root"], "exact/impl/models/pair_adaptive_scorer.py").write_text("# changed\n")
    if bad == "prerequisite":
        save(prepared[0] / "g4-complete.json", {"status": "failed", "exit_code": 1})
    assert checks.run(path) == 1
    root = Path(recipe["root"])
    complete = checks.read(root / "completion.json")
    assert complete["status"] == "failed" and complete["exit_code"] == 1
    assert complete["step_id"] == "123.99" and complete["dispatch_nonce"] == recipe["nonce"]
    assert (root / "exit-code").read_text() == "1\n"
    if bad in {"source", "prerequisite"}:
        assert observed == []


def test_unbound_added_source_is_rejected(prepared):
    source = checks.read(prepared[1]["candidate"]["path"])
    (Path(source["root"]) / "tools/unreviewed.py").write_text("# unreviewed\n")
    with pytest.raises(ValueError, match="inventory changed"):
        checks.verify_source(prepared[1]["candidate"])


def test_wrapper_preserves_prior_terminal_exit_on_duplicate_invocation(prepared):
    path, recipe = _recipe(prepared)
    root = Path(recipe["root"])
    (root / "exit-code").write_text("0\n")
    source = checks.read(recipe["candidate"]["path"])
    # Simulate the duplicate-output guard rejecting a second invocation before
    # writing anything; no real worker, GPU, Slurm or network runs in this test.
    (Path(source["root"]) / "tools/prepare_throughput_checks.py").write_text("raise SystemExit(7)\n")
    result = subprocess.run(["/bin/bash", str(root / "worker-entry.sh")], check=False)
    assert result.returncode == 7
    assert (root / "exit-code").read_text() == "0\n"
