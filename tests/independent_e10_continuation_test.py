"""Regression tests for independent admission without changing E01 or E10 science."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from tools.continue_independent_e10 import (
    E01_REASON,
    EXPECTED_ARMS,
    check_previous_owner,
    independent_lock,
    latest_budget,
    require_numeric_step,
    require_only_e10,
    validate_budget,
)
from tools.resume_cached_batch import RECONCILIATION_ID


def saved_lock():
    ready = {"screen": {"status": "blocked_input_resolution", "reason": "unstarted"}}
    return {
        "scientific_binding": {"seed": 17, "threshold": 0.7, "source_cap": 300},
        "steps": [
            {
                "id": "E01",
                "estimate": {"seconds": 10},
                "external_selection": None,
                "arms": [{"id": "greedy"}],
                "readiness": {"greedy": copy.deepcopy(ready)},
                "policy_paths": ["matching.extraction"],
            },
            {
                "id": "E10-analytic",
                "estimate": None,
                "arms": [
                    {"id": name, "overlay": {"gamma": 2, "tau": 0.5}}
                    for name in sorted(EXPECTED_ARMS)
                ],
                "requires": ["E00", "pool_freeze", "E26", "E05_initial"],
                "inherits": ["E26", "E05_initial"],
                "policy_paths": ["matching.fusion.gamma", "matching.fusion.tau"],
                "readiness": {name: copy.deepcopy(ready) for name in EXPECTED_ARMS},
            },
        ],
    }


def test_independent_family_preserves_all_scientific_fields_and_parent():
    parent = saved_lock()
    before = copy.deepcopy(parent)
    result = independent_lock(parent, {"seconds": 12}, "frozen")
    assert parent == before
    assert result["scientific_binding"] == parent["scientific_binding"]
    for original, revised in zip(parent["steps"], result["steps"]):
        assert {k: v for k, v in original.items() if k not in {"estimate", "readiness"}} == {
            k: v for k, v in revised.items() if k not in {"estimate", "readiness"}
        }
    assert result["steps"][0]["readiness"]["greedy"]["screen"]["reason"] == E01_REASON
    assert result["steps"][0]["estimate"] is None
    assert result["steps"][0]["external_selection"] is None
    assert all(
        state["screen"]["status"] == "screen_ready"
        for state in result["steps"][1]["readiness"].values()
    )


@pytest.mark.parametrize(
    "change", ["arms", "requires", "inherits", "policy_paths", "duplicate_arm"]
)
def test_changed_comparison_or_dependency_is_rejected(change):
    parent = saved_lock()
    step = parent["steps"][1]
    if change == "arms":
        step["arms"].pop()
    elif change == "duplicate_arm":
        step["arms"].append(copy.deepcopy(step["arms"][0]))
    else:
        step[change].append("E01")
    with pytest.raises(ValueError, match="comparison or dependencies"):
        independent_lock(parent, {}, "frozen")


def test_failed_e01_cannot_be_imported_as_complete():
    parent = saved_lock()
    parent["steps"][0]["external_selection"] = {"path": "fabricated"}
    with pytest.raises(ValueError, match="completed selection"):
        independent_lock(parent, {}, "frozen")


def budget():
    return {
        "limits": {
            "requests_cap": 50000,
            "tokens_cap": 32000000,
            "final_requests_reserved": 5000,
            "final_tokens_reserved": 8000000,
        },
        "work": {
            "historical/G0": {"status": "reserved", "seconds": 43200},
            RECONCILIATION_ID: {"status": "failed", "seconds": 2366.3},
            "failed-unknown": {"status": "failed", "requests": 1, "tokens": 700},
        },
    }


def test_budget_validation_retains_failed_costs_and_protected_allowance():
    state = budget()
    original = copy.deepcopy(state)
    validate_budget(state, state["limits"])
    assert state == original
    with pytest.raises(ValueError, match="limits differ"):
        validate_budget(state, {**state["limits"], "final_requests_reserved": 0})
    with pytest.raises(ValueError, match="limits differ"):
        validate_budget(state, {**state["limits"], "requests_cap": 100000})


@pytest.mark.parametrize("change", ["missing_failed_setup", "reserved_owner"])
def test_unreconciled_or_live_accounting_is_rejected(change):
    state = budget()
    if change == "missing_failed_setup":
        del state["work"][RECONCILIATION_ID]
    else:
        state["work"]["active-owner"] = {"status": "reserved"}
    with pytest.raises(ValueError):
        validate_budget(state, state["limits"])


@pytest.mark.parametrize(
    "job,step", [("14372", "extern"), ("14372", "batch"), ("14373", "12"), ("14372", "")]
)
def test_only_authorized_numeric_allocation_steps_are_accepted(monkeypatch, job, step):
    monkeypatch.setenv("SLURM_JOB_ID", job)
    monkeypatch.setenv("SLURM_STEP_ID", step)
    with pytest.raises(ValueError, match="numeric step"):
        require_numeric_step()
    monkeypatch.setenv("SLURM_JOB_ID", "14372")
    monkeypatch.setenv("SLURM_STEP_ID", "12")
    require_numeric_step()


def test_only_independent_family_can_be_runnable():
    plan = {
        "budget_errors": [],
        "rows": [
            {"step": "E10-analytic", "status": "screen_ready", "issues": []},
            {"step": "E01", "status": "blocked_input_resolution", "issues": []},
        ],
    }
    require_only_e10(plan)
    plan["rows"][1]["status"] = "screen_ready"
    with pytest.raises(ValueError, match="independent E10"):
        require_only_e10(plan)


def test_latest_budget_is_the_failed_current_attempt(tmp_path):
    assert latest_budget(tmp_path / "next-batch-recovery-04") == (
        tmp_path / "next-batch-recovery-03/E01/runtime/exact-om-focused-v2/budget.json"
    )


@pytest.mark.parametrize("owner", ["step", "orphan"])
def test_previous_live_step_or_orphan_worker_blocks_recovery(tmp_path, monkeypatch, owner):
    import psutil

    import tools.continue_independent_e10 as continuation

    root = tmp_path / "next-batch-recovery-04"
    monkeypatch.setattr(
        continuation.subprocess,
        "check_output",
        lambda *a, **k: "14372.11\n" if owner == "step" else "14372.0\n14372.8\n",
    )
    command = [
        "python",
        str(tmp_path / "runtime/next-batch-repair-code-03/tools/resume_cached_batch.py"),
    ]
    process = SimpleNamespace(info={"pid": -1, "name": "python", "cmdline": command})
    monkeypatch.setattr(psutil, "process_iter", lambda *a: [process] if owner == "orphan" else [])
    with pytest.raises(ValueError, match="alive"):
        check_previous_owner(root)
