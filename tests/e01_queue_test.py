"""CPU-only checks for the amended E01 queue and cumulative ownership handoff."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from tools import queue_e01_replay as queue


def saved_lock():
    arms = [
        {
            "id": name,
            "role": "baseline" if name == "greedy" else "candidate",
            "overlay": {
                "matching": {
                    "threshold": 0.7,
                    "entity_kinds": ["class"],
                    "fusion": {"gamma": 2.0},
                    "cardinality": 1,
                    "target_cardinality": 1,
                    "extraction": {"mode": name, "max_component_nodes": 500},
                },
                "selector": {"runtime_enabled": False},
                "llm": {"experiment": {"enabled": True, "gate": {"mode": "off"}}},
            },
            "stages": ["screen", "confirm"],
        }
        for name in queue.ARMS[1:]
    ]
    readiness = {
        arm["id"]: {
            "screen": {"status": "screen_ready", "reason": "old inspected evidence"},
            "confirm": {"status": "implementing", "reason": "unqualified"},
        }
        for arm in arms
    }
    return {
        "blueprint": {"path": "old.yaml", "sha256": "old"},
        "source_cap": 300,
        "seeds": [17],
        "reference_role": "valid",
        "eligible_entities": {"kind": "class", "imports": True},
        "steps": [
            {
                "id": "E01",
                "arms": arms,
                "external_selection": None,
                "estimate": {"cold_seconds": 10},
                "readiness": readiness,
                "policy_paths": ["matching.extraction"],
                "selection": {
                    "decisions": [
                        {
                            "baseline": "greedy",
                            "candidates": list(queue.ARMS[2:-1]),
                            "required_controls": ["greedy"],
                            "metric": "F1",
                            "min_delta": 0.003,
                            "guards": [{"metric": "candidate_recall", "min_delta": -0.005}],
                            "tie_breaks": [
                                {"kind": "metric", "metric": "wall_seconds", "direction": "min"},
                                {"kind": "arm_order", "order": list(queue.ARMS[2:-1])},
                            ],
                        }
                    ]
                },
                "design": {"primary_comparison": "E01", "assumptions": ["public development"]},
            },
            {
                "id": "E10-analytic",
                "estimate": {"cold_seconds": 20},
                "arms": [{"id": "analytic", "overlay": {"gamma": 2.0, "tau": 0.5}}],
                "readiness": {"analytic": copy.deepcopy(readiness["greedy"])},
            },
            {"id": "E09", "external_selection": {"path": "valid-selection.json"}},
        ],
    }


def test_amendment_preserves_parent_scores_scope_and_selection_guards():
    original = saved_lock()
    retained = copy.deepcopy(original)
    blueprint = {"path": "amended.yaml", "sha256": "new"}
    estimate = {"cold_seconds": 12, "safety_factor": 1.5}
    revised = queue.amend_lock(original, blueprint, estimate, "tested-commit")
    assert original == retained
    assert {k: v for k, v in revised.items() if k not in {"steps", "blueprint"}} == {
        k: v for k, v in original.items() if k not in {"steps", "blueprint"}
    }
    assert revised["blueprint"] == blueprint
    step = revised["steps"][0]
    assert [arm["id"] for arm in step["arms"]] == list(queue.ARMS)
    for arm in step["arms"]:
        unrestricted = arm["id"] == queue.ARMS[0]
        prior = next(
            item
            for item in original["steps"][0]["arms"]
            if item["id"] == ("greedy" if unrestricted else arm["id"])
        )
        matching = arm["overlay"]["matching"]
        assert matching["threshold"] == 0.7
        assert matching["entity_kinds"] == ["class"]
        assert matching["fusion"] == prior["overlay"]["matching"]["fusion"]
        assert matching["cardinality"] is None if unrestricted else matching["cardinality"] == 1
        assert (
            matching["target_cardinality"] is None
            if unrestricted
            else matching["target_cardinality"] == 1
        )
        assert matching["extraction"] == {
            "mode": "threshold" if unrestricted else arm["id"],
            "max_component_nodes": 500,
            "anchor_conflict_policy": "compete",
        }
        for key in ("selector", "llm"):
            assert arm["overlay"][key] == prior["overlay"][key]
        assert (
            step["readiness"][arm["id"]]["confirm"]
            == original["steps"][0]["readiness"]["greedy"]["confirm"]
        )
    decision = step["selection"]["decisions"][0]
    assert decision["baseline"] == queue.ARMS[0]
    assert decision["candidates"] == list(queue.ARMS[1:-1])
    assert decision["required_controls"] == list(queue.ARMS[:2])
    for key in ("metric", "min_delta", "guards"):
        assert decision[key] == original["steps"][0]["selection"]["decisions"][0][key]
    assert not step["arms"][-1]["deployable"]
    assert step["arms"][-1]["role"] == "diagnostic"
    assert step["policy_paths"] == [
        "matching.extraction",
        "matching.cardinality",
        "matching.target_cardinality",
    ]
    assert step["estimate"] == estimate
    assert revised["steps"][1]["arms"] == original["steps"][1]["arms"]
    assert revised["steps"][1]["estimate"] is None
    assert revised["steps"][2] == original["steps"][2]


@pytest.mark.parametrize("change", ["missing_arm", "duplicate_arm", "completed_selection"])
def test_amendment_refuses_another_design_or_completed_e01(change):
    saved = saved_lock()
    if change == "missing_arm":
        saved["steps"][0]["arms"].pop()
    elif change == "duplicate_arm":
        saved["steps"][0]["arms"].append(copy.deepcopy(saved["steps"][0]["arms"][0]))
    else:
        saved["steps"][0]["external_selection"] = {"path": "completed.json"}
    with pytest.raises(ValueError, match="five-arm E01"):
        queue.amend_lock(saved, {}, {}, "new")


def test_successor_follows_all_registered_recoveries_without_mutating_registry():
    registry = {
        "runs": [
            {"id": "initial", "enabled": False, "superseded_by": "repair-one"},
            {"id": "repair-one", "enabled": False, "superseded_by": "repair-two"},
            {"id": "repair-two", "step_id": "14372.99", "enabled": True},
        ]
    }
    retained = copy.deepcopy(registry)
    assert queue.successor(registry, "initial") == registry["runs"][-1]
    assert registry == retained


@pytest.mark.parametrize(
    "runs",
    [
        [{"id": "initial", "superseded_by": "initial"}],
        [{"id": "initial", "superseded_by": "missing"}],
        [{"id": "initial", "enabled": False}],
    ],
)
def test_successor_refuses_cycles_missing_replacements_and_disabled_owners(runs):
    with pytest.raises(ValueError):
        queue.successor({"runs": runs}, "initial")


def test_closed_budget_preserves_failed_unknown_and_historical_reservation():
    state = {
        "limits": {"requests_cap": 50000, "final_requests_reserved": 5000},
        "work": {
            "historical/G0": {"status": "reserved", "seconds": 43200},
            "E01/failed": {"status": "failed", "requests": 3, "tokens": 777},
            "E10/complete": {"status": "complete", "requests": 0, "tokens": 0},
            "request/unknown": {"status": "unknown", "requests": 1, "tokens": 5},
        },
    }
    retained = copy.deepcopy(state)
    queue.closed_budget(state, state["limits"])
    assert state == retained
    with pytest.raises(ValueError, match="limits"):
        queue.closed_budget(state, {"requests_cap": 20000, "final_requests_reserved": 5000})
    state["work"]["E10/live"] = {"status": "reserved", "seconds": 600}
    with pytest.raises(ValueError, match="active owners.*E10/live"):
        queue.closed_budget(state, state["limits"])


def test_wait_follows_replacement_and_never_returns_a_live_owner(tmp_path, monkeypatch):
    import psutil

    supervisor = tmp_path / "hourly-supervisor-01"
    supervisor.mkdir()
    registry_path = supervisor / "registry.json"
    statuses = []
    roots = [tmp_path / "old", tmp_path / "replacement"]
    rows = []
    for index, root in enumerate(roots):
        root.mkdir()
        (root / "completion.json").write_text(json.dumps({"status": "complete", "exit_code": 0}))
        (root / "status.json").write_text(json.dumps({"status": "complete"}))
        (root / "launcher-exit-code").write_text("0\n")
        rows.append(
            {
                "id": queue.UPSTREAM if index == 0 else "recovery-new",
                "step_id": "14372.12" if index == 0 else "14372.99",
                "completion_path": str(root / "completion.json"),
                "status_path": str(root / "status.json"),
                "exit_path": str(root / "launcher-exit-code"),
            }
        )
    registry_path.write_text(json.dumps({"runs": [rows[0]]}))
    ticks = []

    def next_tick(seconds):
        ticks.append(seconds)
        registry_path.write_text(
            json.dumps(
                {
                    "runs": [
                        {**rows[0], "enabled": False, "superseded_by": rows[1]["id"]},
                        rows[1],
                    ]
                }
            )
        )

    monkeypatch.setattr(queue, "live_steps", lambda: {"14372.12"})
    monkeypatch.setattr(queue.time, "sleep", next_tick)
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [])
    helpers = SimpleNamespace(
        DATA=tmp_path,
        check_pause=lambda: None,
        status=lambda status, **fields: statuses.append((status, fields)),
    )
    owner, completion = queue.wait_for_upstream(helpers)
    assert owner["id"] == "recovery-new"
    assert completion["status"] == "complete"
    assert ticks == [30]
    assert statuses[0][0] == "waiting_dependency"


def test_amendment_preserves_inherited_threshold_without_creating_a_scoring_override():
    from exact.experiments.harness import deep_merge

    parent = saved_lock()
    for arm in parent["steps"][0]["arms"]:
        arm["overlay"]["matching"].pop("threshold")
    revised = queue.amend_lock(parent, {}, {}, "tested")
    for arm in revised["steps"][0]["arms"]:
        assert "threshold" not in arm["overlay"]["matching"]
        resolved = deep_merge({"matching": {"threshold": 0.7}}, arm["overlay"])
        assert resolved["matching"]["threshold"] == 0.7


def test_wait_allows_launcher_exit_receipt_to_arrive_after_step_disappears(tmp_path, monkeypatch):
    import psutil

    directory = tmp_path / "hourly-supervisor-01"
    directory.mkdir()
    completion = tmp_path / "completion.json"
    completion.write_text(json.dumps({"status": "complete", "exit_code": 0}))
    exit_path = tmp_path / "launcher-exit-code"
    row = {
        "id": queue.UPSTREAM,
        "step_id": "14372.12",
        "completion_path": str(completion),
        "exit_path": str(exit_path),
        "status_path": str(tmp_path / "absent-status.json"),
    }
    (directory / "registry.json").write_text(json.dumps({"runs": [row]}))
    ticks = []

    def next_tick(seconds):
        ticks.append(seconds)
        exit_path.write_text("0\n")

    monkeypatch.setattr(queue, "live_steps", lambda: set())
    monkeypatch.setattr(queue.time, "sleep", next_tick)
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: [])
    helpers = SimpleNamespace(DATA=tmp_path, check_pause=lambda: None, status=lambda *a, **k: None)
    owner, _ = queue.wait_for_upstream(helpers)
    assert owner == row
    assert ticks == [1]
