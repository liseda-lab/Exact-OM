"""Bootstrap preparation remains inert and binds retained-allocation ownership."""

import json
import sys
from pathlib import Path

import pytest

from tools.repair import supervisor_bootstrap as bootstrap


def test_policy_and_commands_preserve_venv_interpreter_symlink(tmp_path):
    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(sys.executable).resolve())
    config, instructions = tmp_path / "config.toml", tmp_path / "instructions.md"
    config.write_text('model="gpt-6-astra"\n')
    instructions.write_text("Preserve the authorized allocation and virtual environment.")
    target = tmp_path / "supervisor"
    launch = bootstrap.prepare(
        target,
        allocation="14451",
        node="liseda-05",
        code=tmp_path,
        repository=tmp_path,
        python=interpreter,
        instructions=instructions,
        codex="/home/user/.local/bin/codex",
        codex_config=config,
    )
    policy = json.loads((target / "policy.json").read_text())
    assert policy["python"] == str(interpreter) != str(interpreter.resolve())
    assert launch["argv"][-2] == str(interpreter)
    assert policy["notifications"]["command"][0] == str(interpreter)
    assert "os.execv(" + repr(str(interpreter)) in (target / "entry.py").read_text()


def test_successor_reuses_live_state_and_launch_owner_without_copy_or_mutation(
    tmp_path, monkeypatch
):
    state = tmp_path / "state"
    state.mkdir()
    retained = {
        name: '{"preserved":true}\n'
        for name in ("registry.json", "state.json", "dispatch-state.json", "policy.json")
    }
    for name, content in retained.items():
        (state / name).write_text(content)
    config, instructions = tmp_path / "config.toml", tmp_path / "instructions.md"
    config.write_text('model="gpt-6-astra"\n')
    instructions.write_text("Preserve workers and budget reservations.")
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *a, **k: pytest.fail("Preparation cannot start processes"),
    )
    target = tmp_path / "successor"
    launch = bootstrap.prepare(
        target,
        allocation="14451",
        node="liseda-05",
        code=tmp_path,
        repository=tmp_path,
        python=sys.executable,
        instructions=instructions,
        codex="/home/user/.local/bin/codex",
        codex_config=config,
        state_directory=state,
    )
    assert launch["tmux_socket"] == str(state / "tmux.sock")
    policy = json.loads((target / "policy.json").read_text())
    assert policy["state_directory"] == str(state)
    assert str(state) in policy["notifications"]["command"]
    entry = (target / "entry.py").read_text()
    assert "--policy" in entry and str(target / "policy.json") in entry
    assert "--directory" in entry and str(state) in entry
    assert {name: (state / name).read_text() for name in retained} == retained
    (state / "STOP").touch()
    monkeypatch.setenv("SLURM_JOB_ID", "14451")
    with pytest.raises(ValueError, match="STOP/PAUSE"):
        bootstrap.start(target)


def test_preparation_pins_login_model_schedule_and_owned_launch_without_starting(
    tmp_path, monkeypatch
):
    config, instructions = tmp_path / "config.toml", tmp_path / "instructions.md"
    config.write_text('model="gpt-6-astra"\n')
    instructions.write_text("Only the authorized campaign. Preserve interactive access.")
    monkeypatch.setattr(
        bootstrap.subprocess, "run", lambda *a, **k: pytest.fail("prepare cannot launch")
    )
    target = tmp_path / "supervisor"
    launch = bootstrap.prepare(
        target,
        allocation="14451",
        node="liseda-05",
        code=tmp_path,
        repository=tmp_path,
        python="/usr/bin/python3",
        instructions=instructions,
        codex="/home/user/.local/bin/codex",
        codex_config=config,
    )
    policy = json.loads((target / "policy.json").read_text())
    assert policy["interval_seconds"] == 300 and policy["max_agent_runs_per_day"] is None
    assert policy["model"] == "gpt-6-astra" and policy["model_reasoning_effort"] == "xhigh"
    assert policy["authentication"] == "chatgpt" and policy["max_attempts_per_incident"] == 2
    assert "--jobid=14451" in launch["argv"] and "--gres=none" in launch["argv"]
    assert policy["notifications"]["recipient"] == "pgcotovio@gmail.com"
    assert not (target / "startup.json").exists()
    for binding in launch["bindings"]:
        assert bootstrap.sha(binding["path"]) == binding["sha256"]
    compile((target / "entry.py").read_text(), "entry.py", "exec")
    with pytest.raises(ValueError, match="already exists"):
        bootstrap.prepare(
            target,
            allocation="14451",
            node="liseda-05",
            code=tmp_path,
            repository=tmp_path,
            python="/usr/bin/python3",
            instructions=instructions,
            codex="/home/user/.local/bin/codex",
            codex_config=config,
        )


@pytest.mark.parametrize("block", ["STOP", "PAUSE", "allocation"])
def test_bootstrap_cannot_clear_stops_or_start_outside_ownership(tmp_path, monkeypatch, block):
    (tmp_path / "policy.json").write_text(json.dumps({"allocation": "14451"}))
    (tmp_path / "launch.json").write_text("{}")
    if block != "allocation":
        (tmp_path / block).touch()
    monkeypatch.setenv("SLURM_JOB_ID", "other")
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(ValueError):
        bootstrap.start(tmp_path)
    if block != "allocation":
        assert (tmp_path / block).exists()
