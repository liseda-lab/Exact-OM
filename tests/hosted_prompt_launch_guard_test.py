"""Prospective workers cannot bypass reviewed prompt admission through stale source."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from exact.experiments import dispatch
from tools import hosted_prompt_guard as guard
from tools import prepared_batch, supervise_experiments


def save(path, value):
    path.write_text(json.dumps(value))
    return guard.binding(path)


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "source"
    sources = {}
    for name in guard.REQUIRED_SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# reviewed implementation " + name)
        sources[name] = [guard.binding(path)["sha256"]]
    prompt = save(tmp_path / "prompt.json", {
        "schema_version": 1, "max_input_tokens": 8192, "max_input_bytes": 32768,
        "max_output_tokens": 1024, "chat_overhead_tokens": 128,
    })
    policy = {"hosted_prompt_guard": {
        "schema_version": 1, "prompt_policy": prompt, "allowed_source_sha256": sources,
    }}
    environment = save(tmp_path / "environment.json", {
        "EXACT_EXPERIMENT_MODE": "1", "EXACT_LLM_PROMPT_POLICY_PATH": prompt["path"],
        "EXACT_LLM_PROMPT_POLICY_SHA256": prompt["sha256"],
    })
    recipe = tmp_path / "recipe.json"
    save(recipe, {"code_root": str(source), "commit": "a" * 40, "environment": environment})
    worker = tmp_path / "worker.sh"
    worker.write_text("#!/bin/bash\nexit 0\n")
    launch = {
        "argv": ["/usr/bin/srun", "--jobid=14372", "/bin/bash", str(worker)],
        "nonce": "prompt-policy-test-nonce", "tmux_socket": str(tmp_path / "tmux.sock"),
        "step_path": str(tmp_path / "step.json"), "launcher_log": str(tmp_path / "launcher.log"),
        "bindings": [guard.binding(worker), guard.binding(recipe)],
        "run": {"id": "next", "status_path": str(tmp_path / "status.json"),
                "exit_path": str(tmp_path / "exit"), "completion_path": str(tmp_path / "complete.json")},
    }
    return launch, policy, recipe, tmp_path / "guard.json"


def attach(prepared, **kwargs):
    launch, policy, recipe, receipt = prepared
    return guard.guard_launch(launch, policy, recipe_path=recipe, receipt_path=receipt, **kwargs)


def test_guarded_receipt_binds_worker_source_policy_and_environment(prepared):
    launch = attach(prepared)
    assert launch["argv"] == prepared[0]["argv"]
    assert "hosted_prompt_guard" not in prepared[0]
    guard.validate_launch(launch, prepared[1])
    assert attach(prepared) == launch
    old_source = Path(json.loads(prepared[2].read_text())["code_root"]) / "exact/llm/routing.py"
    old_source.write_text("# stale worker without prompt admission")
    with pytest.raises(ValueError, match="binding"):
        guard.validate_launch(launch, prepared[1])


def test_unapproved_source_cannot_certify_itself_with_new_receipt(prepared):
    recipe = json.loads(prepared[2].read_text())
    (Path(recipe["code_root"]) / "exact/llm/routing.py").write_text("# unreviewed")
    with pytest.raises(ValueError, match="unapproved"):
        attach(prepared)


@pytest.mark.parametrize("key", [
    "EXACT_LLM_PROMPT_POLICY_PATH", "EXACT_LLM_PROMPT_POLICY_SHA256", "EXACT_EXPERIMENT_MODE",
])
def test_missing_worker_policy_binding_fails_even_when_recipe_is_rehashed(prepared, key):
    launch, _, recipe_path, _ = prepared
    recipe = json.loads(recipe_path.read_text())
    environment_path = Path(recipe["environment"]["path"])
    environment = json.loads(environment_path.read_text())
    environment.pop(key)
    recipe["environment"] = save(environment_path, environment)
    old = guard.binding(recipe_path)
    launch["bindings"].remove(old)
    launch["bindings"].append(save(recipe_path, recipe))
    with pytest.raises(ValueError, match="environment lacks"):
        attach(prepared)


def test_no_new_calls_requires_bound_evidence_and_preserves_worker(prepared, tmp_path):
    evidence = save(tmp_path / "offline.json", {"hosted_calls": False, "reviewed": "native admission"})
    launch = attach(prepared, mode="no_new_hosted_calls", evidence=[evidence])
    assert launch["argv"] == prepared[0]["argv"]
    assert launch["bindings"][:2] == prepared[0]["bindings"]
    guard.validate_launch(launch, prepared[1])
    Path(evidence["path"]).write_text("changed")
    with pytest.raises(ValueError, match="binding"):
        guard.validate_launch(launch, prepared[1])


def test_no_new_calls_cannot_be_inferred_from_run_name(prepared):
    prepared[0]["run"]["id"] = "E14-native-no-hosted-calls"
    with pytest.raises(ValueError, match="evidence"):
        attach(prepared, mode="no_new_hosted_calls")
    with pytest.raises(ValueError, match="receipt"):
        guard.validate_launch(prepared[0], prepared[1])


def test_source_environment_or_nonce_cannot_be_transplanted(prepared):
    launch = attach(prepared)
    changed = copy.deepcopy(launch)
    changed["nonce"] += "-changed"
    with pytest.raises(ValueError, match="does not match"):
        guard.validate_launch(changed, prepared[1])
    with pytest.raises(ValueError, match="Immutable"):
        attach(prepared, mode="no_new_hosted_calls", evidence=[guard.binding(prepared[2])])


def test_missing_guard_fails_before_any_launch_and_is_not_retried(prepared, tmp_path, monkeypatch):
    launch, policy, _, _ = prepared
    save(tmp_path / "registry.json", {
        "runs": [], "pending_batches": [{"id": "next", "launch": launch, "resources": {}}],
        "capacity": {},
    })
    monkeypatch.setattr(dispatch, "_tmux_server", lambda _: pytest.fail("No launch allowed"))
    def validate(value):
        guard.validate_launch(value, policy)
    steps = {"14372.0": "RUNNING"}
    first = dispatch.dispatch_ready(tmp_path, "14372", steps, validate_launch=validate)
    assert first == {"status": "failed", "batch_id": "next"}
    state = json.loads((tmp_path / "dispatch-state.json").read_text())["next"]
    assert state["may_have_started"] is False
    assert "receipt" in state["error"]
    second = dispatch.dispatch_ready(tmp_path, "14372", steps, validate_launch=validate)
    assert second["status"] == "no_ready_launch"


def test_supervisor_preserves_storage_check_before_prompt_check(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(supervise_experiments.storage_guard, "validate_launch", lambda *args: calls.append("storage"))
    monkeypatch.setattr(supervise_experiments.hosted_prompt_guard, "validate_launch", lambda *args: calls.append("prompt"))
    supervise_experiments.validate_prepared_launch({}, {}, tmp_path)
    assert calls == ["storage", "prompt"]


def test_new_preparation_attaches_receipt_after_storage_wrapper(prepared, tmp_path, monkeypatch):
    _, policy, recipe_path, _ = prepared
    recipe = json.loads(recipe_path.read_text())
    root = tmp_path / "prepared"
    root.mkdir()
    supervisor = tmp_path / "supervisor"
    supervisor.mkdir()
    save(supervisor / "policy.json", policy)
    code = Path(recipe["code_root"])
    (code / "tools").mkdir()
    (code / "tools/prepared_batch.py").write_text("# reviewed worker")
    recipe.update(repository=str(tmp_path), allocation="14372", campaign_id="future",
                  dispatch_nonce="new-guarded-preparation", group=guard.binding(recipe_path))
    wrapper = tmp_path / "storage-wrapper.sh"
    wrapper.write_text("#!/bin/bash\nexit 0\n")

    def storage(launch, *_):
        launch["argv"][-1] = str(wrapper)
        launch["bindings"].append(guard.binding(wrapper))
        return launch

    monkeypatch.setattr(supervise_experiments.storage_guard, "guard_launch", storage)
    launch = prepared_batch.prepare_launch(recipe, root, code, supervisor, {"id": "future", "depends_on": []})
    guard.validate_launch(launch, policy)
    receipt = json.loads(Path(launch["hosted_prompt_guard"]["path"]).read_text())
    assert receipt["worker"] == guard.binding(wrapper)
    assert json.loads((root / "launch-descriptor.json").read_text()) == launch
