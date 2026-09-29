"""Safety regressions for matched family qualification before campaign admission."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from tools import qualify_cached_family as qualification


def controls(tmp_path):
    root = tmp_path / "next-family"
    root.mkdir()
    supervisor = tmp_path / "hourly-supervisor-01"
    supervisor.mkdir()
    (supervisor / "registry.json").write_text(json.dumps({"pause_paths": []}))
    script, probe = root / "continue.py", root / "qualification" / "current"
    probe.mkdir(parents=True)
    return root, supervisor, script, probe


@pytest.mark.parametrize(
    "control", ["supervisor_pause", "supervisor_stop", "root", "nested", "registered"]
)
def test_new_pause_is_observed_without_restarting_probe(tmp_path, control):
    root, supervisor, script, probe = controls(tmp_path)
    qualification.check_controls(script, probe)
    if control == "supervisor_pause":
        path = supervisor / "PAUSE"
    elif control == "supervisor_stop":
        path = supervisor / "STOP"
    elif control == "root":
        path = root / "STOP"
    elif control == "nested":
        path = probe / "STOP"
    else:
        path = tmp_path / "other-family" / "STOP"
        path.parent.mkdir()
        (supervisor / "registry.json").write_text(json.dumps({"pause_paths": [str(path)]}))
    path.write_text("user requested pause\n")
    with pytest.raises(RuntimeError, match="Intentional pause/STOP"):
        qualification.check_controls(script, probe)
    assert path.read_text() == "user requested pause\n"


def test_signals_and_resource_guards_preserve_user_stop(tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("intentional scientific interruption\n")
    qualification.preserve_stop(stop, "memory guard")
    assert stop.read_text() == "intentional scientific interruption\n"
    stop.unlink()
    qualification.preserve_stop(stop, "memory guard")
    assert stop.read_text() == "memory guard\n"


def test_fresh_probe_reuse_plan_does_not_import_other_arms(tmp_path):
    root, _, script, probe = controls(tmp_path)
    with pytest.raises(ValueError, match="dependency-based reuse plan"):
        qualification.recovery_metadata(script, probe)
    plan = {"affected_stages": ["evidence", "pair_scores", "extraction"]}
    (root / "reuse-plan.json").write_text(json.dumps(plan))
    result = qualification.recovery_metadata(script, probe)
    assert "resume_from" not in result
    assert result["root"] == str(probe.resolve())
    assert result["repair_record"] == str(root / "reuse-plan.json")
    assert result["evaluation_enabled"] is True
    assert result["stop_after_checkpoint"] is False
    assert json.loads((root / "reuse-plan.json").read_text()) == plan


def test_worker_boundary_cannot_inherit_new_request_allowance(tmp_path, monkeypatch):
    monkeypatch.setenv("EXACT_OPENROUTER_REQUEST_CAP", "50000")
    monkeypatch.setenv("EXACT_OPENROUTER_TOKEN_CAP", "32000000")
    monkeypatch.setenv("EXACT_OPENROUTER_RETRY_UNKNOWN", "1")
    env = {
        "EXACT_OPENROUTER_REQUEST_CAP": "99999",
        "EXACT_OPENROUTER_TOKEN_CAP": "99999999",
        "EXACT_OPENROUTER_RETRY_UNKNOWN": "1",
        "EXACT_DATASET_CACHE_DIR": "/old/datasets/matched-config-hash",
    }
    shared = tmp_path / "shared"
    result = qualification.cached_worker_env(
        env,
        shared,
        {"attempts": 4147, "billable_tokens": 2460315},
        tmp_path / "frozen-code",
        tmp_path / "STOP",
    )
    assert result["EXACT_OPENROUTER_REQUEST_CAP"] == "4147"
    assert result["EXACT_OPENROUTER_TOKEN_CAP"] == "2460315"
    assert result["EXACT_OPENROUTER_RETRY_UNKNOWN"] == "0"
    assert result["EXACT_OPENROUTER_LEDGER_DIR"] == str(shared / "openrouter")
    assert result["EXACT_DATASET_CACHE_DIR"] == str(shared / "datasets/matched-config-hash")
    assert env["EXACT_OPENROUTER_REQUEST_CAP"] == "99999"


def state():
    return {
        "schema_version": 2,
        "limits": {
            "envelopes_hours": {"reserve": 60, "foundation": 12},
            "node_hours_cap": 336,
            "requests_cap": 50000,
            "tokens_cap": 32000000,
            "final_requests_reserved": 5000,
            "final_tokens_reserved": 8000000,
        },
        "work": {
            "failed-unknown": {
                "status": "failed",
                "group": "reserve",
                "start": 1,
                "end": 2,
                "seconds": 1,
                "requests": 28,
                "tokens": 700,
                "actual_usd": None,
                "projected_usd": 0,
            },
        },
        "intervals": [[1, 2]],
    }


def usage():
    return {
        "attempts": 4147,
        "billable_tokens": 2460315,
        "unknown": 0,
        "unpriced_attempts": 28,
        "reported_cost_usd": 0.4,
    }


@pytest.mark.parametrize("result", ["success", "setup_failure", "worker_failure"])
def test_admitted_probe_closes_failed_accounting_and_keeps_history(tmp_path, monkeypatch, result):
    original = state()
    (tmp_path / "budget.json").write_text(json.dumps(original))
    monkeypatch.setattr(qualification, "usage", lambda _: usage())
    times = iter([100, 140])
    monkeypatch.setattr(qualification, "time", SimpleNamespace(time=lambda: next(times)))

    def run():
        with qualification.accounted_probe(tmp_path, "qualification/new/current/17", 100) as (
            _,
            outcome,
            _,
        ):
            if result != "success":
                raise ValueError(result)
            outcome["status"] = "complete"

    if result == "success":
        run()
    else:
        with pytest.raises(ValueError, match=result):
            run()
    saved = json.loads((tmp_path / "budget.json").read_text())
    assert saved["limits"] == original["limits"]
    assert saved["work"]["failed-unknown"] == original["work"]["failed-unknown"]
    row = saved["work"]["qualification/new/current/17"]
    assert row["status"] == ("complete" if result == "success" else "failed")
    assert row["seconds"] == 40
    assert row["requests"] == row["tokens"] == 0


def test_incremental_uncertain_charge_is_retained_and_probe_fails(tmp_path, monkeypatch):
    original = state()
    (tmp_path / "budget.json").write_text(json.dumps(original))
    before = usage()
    after = {
        **before,
        "attempts": before["attempts"] + 1,
        "billable_tokens": before["billable_tokens"] + 700,
        "unknown": 1,
        "unpriced_attempts": before["unpriced_attempts"] + 1,
    }
    totals = iter([before, after])
    monkeypatch.setattr(qualification, "usage", lambda _: next(totals))
    with pytest.raises(ValueError, match="incremental hosted"):
        with qualification.accounted_probe(tmp_path, "qualification/unknown", 100) as (
            _,
            outcome,
            _,
        ):
            outcome["status"] = "complete"
    saved = json.loads((tmp_path / "budget.json").read_text())
    row = saved["work"]["qualification/unknown"]
    assert row["status"] == "failed"
    assert row["requests"] == 1 and row["tokens"] == 700
    assert row["actual_usd"] is None
    assert saved["work"]["failed-unknown"] == original["work"]["failed-unknown"]


def test_private_references_and_enabled_judgments_fail_before_probe():
    base = {
        "data": {"refs": {"train": "training", "valid": "development"}},
        "llm": {"experiment": {"gate": {"mode": "off"}}},
    }
    qualification.validate_probe_config(base)
    for role in ("test", "private", "final"):
        model = copy.deepcopy(base)
        model["data"]["refs"][role] = "not-allowed"
        with pytest.raises(ValueError, match="training/development"):
            qualification.validate_probe_config(model)
    model = copy.deepcopy(base)
    model["llm"]["experiment"]["gate"]["mode"] = "all"
    with pytest.raises(ValueError, match="decision gate off"):
        qualification.validate_probe_config(model)


@pytest.mark.parametrize(
    "processed,total,code,calls",
    [
        (512, 5842, 0, 1),
        (5842, 5842, 1, 1),
        (5842, 5842, 0, 0),
        (0, 5842, 0, 1),
        (5843, 5842, 0, 1),
    ],
)
def test_partial_or_cached_work_cannot_be_a_whole_arm_measurement(processed, total, code, calls):
    with pytest.raises(ValueError):
        qualification.validate_full_measurement(
            {"cursor": {"next_pair": processed, "dataset_rows": total}},
            {"return_code": code},
            calls,
        )
    qualification.validate_full_measurement(
        {"cursor": {"next_pair": 5842, "dataset_rows": 5842}},
        {"return_code": 0},
        1,
    )


def test_optional_shared_cache_root_preserves_current_request_account(tmp_path, monkeypatch):
    monkeypatch.delenv("EXACT_EMBEDDING_CACHE_DIR", raising=False)
    monkeypatch.delenv("EXACT_NUMERICAL_CACHE_ROOT", raising=False)
    shared = tmp_path / "current-runtime"
    common = tmp_path / "common-cache"
    env = {
        "EXACT_DATASET_CACHE_DIR": "/old/datasets/reference-scope",
        "EXACT_EXPERIMENT_SHARED_CACHE_ROOT": str(common),
    }
    result = qualification.cached_worker_env(
        env,
        shared,
        {"attempts": 100, "billable_tokens": 1000},
        tmp_path / "code",
        tmp_path / "STOP",
    )
    assert result["EXACT_EMBEDDING_CACHE_DIR"] == str(common / "embeddings")
    assert result["EXACT_NUMERICAL_CACHE_ROOT"] == str(common)
    assert result["EXACT_DATASET_CACHE_DIR"] == str(common / "datasets/reference-scope")
    assert result["EXACT_OPENROUTER_LEDGER_DIR"] == str(shared / "openrouter")
    assert result["EXACT_OPENROUTER_REQUEST_CAP"] == "100"
    assert result["EXACT_OPENROUTER_TOKEN_CAP"] == "1000"
