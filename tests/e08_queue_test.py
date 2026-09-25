"""Admission and scientific-boundary checks for the matched E08 queue."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import queue_e08 as queue


def cell(arm):
    return SimpleNamespace(
        arm_id=arm,
        generate_rationales=False,
        split_role="development",
        seed=17,
        source_cap=300,
        resolved_config={
            "data": {"refs": {"train": "train.tsv", "valid": "valid.tsv"}},
            "llm": {"experiment": {"enabled": True, "gate": {"mode": "off"}}},
            "matching": {
                "extraction": {"mode": "greedy"},
                "threshold": 0.7,
                "cardinality": 1,
                "target_cardinality": 1,
            },
            "supervision": {
                "mode": "label_free",
                "components": {name: "label_free" for name in ("retrieval", "accept", "llm")},
            },
            "selector": {"enabled": False, "runtime_enabled": None},
        },
    )


def measurements(tmp_path):
    rows = []
    for arm in queue.ARMS:
        output = tmp_path / arm
        (output / "dataset").mkdir(parents=True)
        manifest = {
            "gold_free_summary": {"candidate_pairs": 6000, "source_entities": 300},
            "per_kind": {"class": {"pool_sha256": "a" * 64, "candidate_pairs": 6000}},
        }
        (output / "dataset/candidate_pool_sample_manifest.json").write_text(json.dumps(manifest))
        rows.append(
            {
                "name": arm,
                "status": "passed",
                "execution_status": "complete",
                "prefix": False,
                "new_usage": {"attempts": 0, "billable_tokens": 0, "unknown": 0},
                "processed_pairs": 5842,
                "dataset_rows": 5842,
                "output_dir": str(output),
            }
        )
    return rows


def test_all_declared_cells_pass_without_modifying_their_configuration():
    cells = [cell(arm) for arm in queue.ARMS]
    before = copy.deepcopy(cells)
    queue.validate_cells(cells)
    assert cells == before


@pytest.mark.parametrize("shape", ["missing", "duplicate", "extra"])
def test_cell_validation_requires_exact_complete_arm_set(shape):
    cells = [cell(arm) for arm in queue.ARMS]
    if shape == "missing":
        cells.pop()
    elif shape == "duplicate":
        cells[-1].arm_id = cells[0].arm_id
    else:
        cells.append(cell("signed_identifiers"))
    with pytest.raises(ValueError, match="three E08 arms"):
        queue.validate_cells(cells)


@pytest.mark.parametrize(
    "field,value",
    [("generate_rationales", True), ("split_role", "reporting"), ("seed", 29), ("source_cap", 200)],
)
def test_cell_role_and_frozen_population_guards(field, value):
    cells = [cell(arm) for arm in queue.ARMS]
    setattr(cells[1], field, value)
    with pytest.raises(ValueError, match="policy changed"):
        queue.validate_cells(cells)


@pytest.mark.parametrize("role", ["test", "private", "final"])
def test_private_reference_cannot_enter_qualification(role):
    cells = [cell(arm) for arm in queue.ARMS]
    cells[1].resolved_config["data"]["refs"][role] = "sealed.tsv"
    with pytest.raises(ValueError, match="policy changed"):
        queue.validate_cells(cells)


@pytest.mark.parametrize("change", ["hosted_gate", "extraction", "supervision", "component"])
def test_optimization_or_hosted_role_changes_fail(change):
    cells = [cell(arm) for arm in queue.ARMS]
    config = cells[1].resolved_config
    if change == "hosted_gate":
        config["llm"]["experiment"]["gate"]["mode"] = "analytic"
    elif change == "extraction":
        config["matching"]["extraction"]["mode"] = "mutual_best"
    elif change == "supervision":
        config["supervision"]["mode"] = "supervised"
    else:
        config["supervision"]["components"]["accept"] = "supervised"
    with pytest.raises(ValueError, match="policy changed"):
        queue.validate_cells(cells)


def test_measurement_population_is_content_bound(tmp_path):
    rows = measurements(tmp_path)
    before = copy.deepcopy(rows)
    assert queue.matched_measurements(rows)["per_kind"]["class"]["pool_sha256"] == "a" * 64
    assert rows == before
    path = Path(rows[1]["output_dir"]) / "dataset/candidate_pool_sample_manifest.json"
    value = json.loads(path.read_text())
    value["per_kind"]["class"]["pool_sha256"] = "b" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="candidate populations"):
        queue.matched_measurements(rows)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "failed", "interrupted", "prefix", "partial", "charged"]
)
def test_incomplete_or_charged_measurements_cannot_admit_a_comparison(tmp_path, change):
    rows = measurements(tmp_path)
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1]["name"] = rows[0]["name"]
    elif change == "failed":
        rows[0]["status"] = "failed"
    elif change == "interrupted":
        rows[0]["execution_status"] = "interrupted"
    elif change == "prefix":
        rows[0]["prefix"] = True
    elif change == "partial":
        rows[0]["processed_pairs"] -= 1
    else:
        rows[0]["new_usage"]["unknown"] = 1
    with pytest.raises(ValueError):
        queue.matched_measurements(rows)


def scientific_lock():
    return {
        "campaign_id": "preserved",
        "steps": [
            {"id": "E01", "external_selection": {"sha256": "prior"}},
            {
                "id": "E08",
                "inherits": ["E05_initial"],
                "arms": [{"id": arm, "overlay": {"marker": arm}} for arm in queue.ARMS],
                "readiness": {
                    arm: {
                        "screen": {"status": "blocked_input_resolution"},
                        "confirm": {"status": "planned"},
                    }
                    for arm in queue.ARMS
                },
                "estimate": None,
                "selection": {"min_delta": 0.003, "guards": ["candidate_recall"]},
                "design": {"scope": "development", "negative_policy": "confirmed_only"},
            },
        ],
    }


def test_ready_lock_preserves_scientific_declaration_and_other_steps():
    original = scientific_lock()
    before = copy.deepcopy(original)
    estimate = {"measured": True}
    result = queue.ready_lock(original, estimate, "frozen-commit")
    assert original == before
    assert result["steps"][0] == original["steps"][0]
    step = result["steps"][1]
    for name in ("arms", "selection", "design", "inherits"):
        assert step[name] == before["steps"][1][name]
    assert step["estimate"] == estimate
    for arm in queue.ARMS:
        assert step["readiness"][arm]["confirm"] == {"status": "planned"}
        assert step["readiness"][arm]["screen"]["inspected_commit"] == "frozen-commit"


@pytest.mark.parametrize("change", ["arm", "inherited_policy"])
def test_ready_lock_rejects_changed_family_membership(change):
    lock = scientific_lock()
    if change == "arm":
        lock["steps"][1]["arms"].pop()
    else:
        lock["steps"][1]["inherits"].append("E01")
    with pytest.raises(ValueError, match="inherited policy"):
        queue.ready_lock(lock, {}, "commit")


def budget_state():
    return {
        "limits": {
            "envelopes_hours": {"reserve": 60, "channels": 40},
            "node_hours_cap": 336,
            "requests_cap": 50000,
            "tokens_cap": 32000000,
            "final_requests_reserved": 5000,
            "final_tokens_reserved": 8000000,
        },
        "work": {
            "historical/G0": {"status": "reserved", "tokens": 100},
            "failed-uncertain": {"status": "failed", "requests": 28, "tokens": 700},
            "last-E01": {"status": "complete", "seconds": 40},
        },
    }


def test_budget_validation_preserves_cumulative_failed_and_uncertain_costs(monkeypatch):
    state = budget_state()
    original = copy.deepcopy(state)
    monkeypatch.setattr(queue, "campaign_limits", lambda _: original["limits"])
    queue.validate_budget(state, {})
    assert state == original


@pytest.mark.parametrize(
    "limit",
    [
        "requests_cap",
        "tokens_cap",
        "node_hours_cap",
        "final_requests_reserved",
        "final_tokens_reserved",
    ],
)
def test_changed_caps_or_protected_allowances_fail(monkeypatch, limit):
    state = budget_state()
    original = copy.deepcopy(state)
    monkeypatch.setattr(queue, "campaign_limits", lambda _: original["limits"])
    state["limits"][limit] += 1
    with pytest.raises(ValueError, match="protected final allowance"):
        queue.validate_budget(state, {})


def test_unclosed_previous_owner_reservation_is_not_reconciled(monkeypatch):
    state = budget_state()
    state["work"]["prior/worker"] = {"status": "reserved", "seconds": 400}
    original = copy.deepcopy(state)
    monkeypatch.setattr(queue, "campaign_limits", lambda _: original["limits"])
    with pytest.raises(ValueError, match="unclosed accounting"):
        queue.validate_budget(state, {})
    assert state == original


@pytest.mark.parametrize("stale_cache_budget", [False, True])
def test_missing_matched_probe_stops_before_comparison_admission(
    tmp_path, monkeypatch, stale_cache_budget
):
    """Exercise orchestration up to its gate with no real worker or credential read."""
    from exact.core.entities.configs import yaml_io
    from exact.experiments import campaign

    root = tmp_path / "queue"
    root.mkdir()
    fake_project = tmp_path / "synthetic-project"
    fake_project.mkdir()
    (fake_project / "api_key").write_text("test-only-not-a-credential")
    monkeypatch.setattr(queue, "PROJECT", fake_project)
    monkeypatch.setenv("SLURM_JOB_ID", "14372")
    monkeypatch.setenv("SLURM_STEP_ID", "19")
    monkeypatch.setattr(queue, "no_previous_owner", lambda: None)
    monkeypatch.setattr(queue, "validate_budget", lambda *_: None)
    monkeypatch.setattr(queue, "usage", lambda _: {"attempts": 0, "billable_tokens": 0})
    monkeypatch.setattr(yaml_io, "load_yaml_mapping", lambda _: {})
    monkeypatch.setattr(
        campaign, "CampaignLock", SimpleNamespace(model_validate=lambda value: value)
    )
    identity = {"commit": "frozen"}
    latest_budget = tmp_path / "latest-budget.json"
    latest_budget.write_text(json.dumps(budget_state()))
    monkeypatch.setattr(queue, "BUDGET", latest_budget)
    cache_source = root / "cache-source"
    cache_source.mkdir()
    cache_budget = latest_budget
    if stale_cache_budget:
        cache_budget = tmp_path / "stale-budget.json"
        older = budget_state()
        older["work"].pop("last-E01")
        cache_budget.write_text(json.dumps(older))
    (cache_source / "budget.json").symlink_to(cache_budget)
    receipt = {
        "status": "preflight_passed_not_admitted",
        "source": identity,
        "budget_import": queue.binding(latest_budget),
        "parent_completion": {},
        "prospective_campaign": {},
        "source_selection": {},
        "pilot_cost_basis": {},
        "records": [],
        "operational_files": [],
        "configs": {arm: {"path": arm} for arm in queue.ARMS},
        "pilot_allowance_seconds_per_arm": 100,
    }
    monkeypatch.setattr(
        queue,
        "read",
        lambda path: (
            receipt
            if Path(path).name == "preflight.json"
            else (
                {"bindings": []} if Path(path).name == "cache-sources.json" else {"environment": {}}
            )
        ),
    )
    h = SimpleNamespace(
        check_pause=lambda: None,
        validate_source=lambda *_: identity,
        verify=lambda item: item.get("path", "placeholder"),
        status=lambda *_, **__: None,
        copy_state=lambda *_: None,
    )
    calls = []

    def probe(**kwargs):
        calls.append(kwargs["name"])
        if kwargs["name"] == "unified_bank":
            raise ValueError("missing matched treatment")
        return {"name": kwargs["name"]}

    def forbidden(*_, **__):
        pytest.fail("Scientific comparison must not be admitted or launched without all probes")

    monkeypatch.setattr(queue, "run_probe", probe)
    monkeypatch.setattr(queue, "check_admission", forbidden)
    monkeypatch.setattr(queue, "comparison_forecast", forbidden)
    monkeypatch.setattr(campaign, "execute_campaign", forbidden)
    error = "stale budget" if stale_cache_budget else "missing matched treatment"
    with pytest.raises(ValueError, match=error):
        queue.run(SimpleNamespace(root=root, code_root=tmp_path / "code"), h)
    assert calls == ([] if stale_cache_budget else ["current", "unified_bank"])
