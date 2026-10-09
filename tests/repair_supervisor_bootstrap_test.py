"""Bootstrap preparation remains inert and binds retained-allocation ownership."""

import json
from pathlib import Path

import pytest

from tools.repair import supervisor_bootstrap as bootstrap


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
