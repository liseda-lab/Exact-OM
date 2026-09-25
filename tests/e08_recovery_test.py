"""Bounded E08 admission recovery preserves scientific and accounting identities."""

from __future__ import annotations

import copy
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import recover_e08 as recovery
from tools.qualify_cached_family import binding


def helpers():
    def verify(item):
        if binding(item["path"]) != item:
            raise ValueError("Bound evidence changed")
        return Path(item["path"])

    return SimpleNamespace(verify=verify)


def evidence(tmp_path):
    rows, configs, state = [], {}, {"work": {}}
    for arm in recovery.queue.ARMS:
        config = tmp_path / (arm + ".yaml")
        config.write_text("frozen: " + arm + "\n")
        output = tmp_path / (arm + ".json")
        output.write_text('{"complete": true}\n')
        configs[arm] = binding(config)
        rows.append(
            {
                "name": arm,
                "status": "passed",
                "execution_status": "complete",
                "prefix": False,
                "seed": 17,
                "source_cap": 300,
                "generate_rationales": False,
                "no_private_test_references": True,
                "new_usage": {
                    "attempts": 0,
                    "billable_tokens": 0,
                    "unknown": 0,
                    "unpriced_attempts": 0,
                },
                "config": configs[arm],
                "bindings": [binding(output)],
                "processed_pairs": 5842,
                "dataset_rows": 5842,
                "worker_calls": 1,
                "worker_measurement": {"return_code": 0},
                "wall_seconds": 1200,
                "budget_work_id": "qualification/" + arm,
            }
        )
        state["work"]["qualification/" + arm] = {
            "status": "complete",
            "group": "reserve",
            "requests": 0,
            "tokens": 0,
            "seconds": 1201,
        }
    state["work"]["failed-interrupted"] = {"status": "failed", "requests": 8, "tokens": 90}
    state["work"]["unknown-charge"] = {"status": "failed", "requests": 1, "tokens": 1000}
    return rows, configs, state


def test_reuses_only_complete_receipts_without_mutating_science_or_cumulative_budget(tmp_path):
    rows, configs, state = evidence(tmp_path)
    before = copy.deepcopy((rows, configs, state))
    recovery.validate_receipts(rows, state, configs, helpers())
    assert (rows, configs, state) == before


@pytest.mark.parametrize(
    "change",
    [
        "corruption",
        "configuration",
        "rationales",
        "private",
        "seed",
        "population",
        "charged",
        "unknown",
        "missing_usage",
        "partial",
        "missing_arm",
        "missing_account",
        "unfinished_account",
        "short_account",
    ],
)
def test_incompatible_retained_measurements_never_admit(tmp_path, change):
    rows, configs, state = evidence(tmp_path)
    first = rows[0]
    if change == "corruption":
        Path(first["bindings"][0]["path"]).write_text("corrupted")
    elif change == "configuration":
        first["config"] = configs[rows[1]["name"]]
    elif change == "rationales":
        first["generate_rationales"] = True
    elif change == "private":
        first["no_private_test_references"] = False
    elif change == "seed":
        first["seed"] = 29
    elif change == "population":
        first["source_cap"] = 301
    elif change == "charged":
        first["new_usage"]["attempts"] = 1
    elif change == "unknown":
        first["new_usage"]["unknown"] = 1
    elif change == "missing_usage":
        del first["new_usage"]["unknown"]
    elif change == "partial":
        first["processed_pairs"] -= 1
    elif change == "missing_arm":
        rows.pop()
    elif change == "missing_account":
        del state["work"][first["budget_work_id"]]
    elif change == "unfinished_account":
        state["work"][first["budget_work_id"]]["status"] = "reserved"
    else:
        state["work"][first["budget_work_id"]]["seconds"] = 1000
    with pytest.raises(ValueError):
        recovery.validate_receipts(rows, state, configs, helpers())


BEFORE = """class BaseAlignmentDataset:
    def restrict_sources(self):
        pool_frame = self._candidates
        if pool_frame is None and self._df is not None and "cand_sim" in self._df.columns:
            pool_frame = self._df[self._df["cand_sim"].notna()].reset_index(drop=True)
        if pool_frame is not None:
            self._refresh_candidate_pool_manifest(frame=pool_frame)
        return self._df
"""
AFTER = BEFORE.replace(
    '        if pool_frame is None and self._df is not None and "cand_sim" in self._df.columns:\n'
    '            pool_frame = self._df[self._df["cand_sim"].notna()].reset_index(drop=True)',
    textwrap.indent(recovery.METADATA_BRANCH, "        "),
)


def test_exact_metadata_branch_repair_has_proven_unchanged_numerical_ast():
    recovery.validate_metadata_patch(BEFORE, AFTER)


@pytest.mark.parametrize("change", ["numerical", "selection", "extra", "unchanged"])
def test_metadata_repair_cannot_license_other_source_changes(change):
    after = AFTER
    if change == "numerical":
        after = after.replace("return self._df", "return self._candidates")
    elif change == "selection":
        after = after.replace(".notna()", ".isna()")
    elif change == "extra":
        after += "\ndef unrelated():\n    return 1\n"
    else:
        after = BEFORE
    with pytest.raises(ValueError, match="metadata-only"):
        recovery.validate_metadata_patch(BEFORE, after)


def test_zero_incremental_caps_preserve_uncertain_billed_usage():
    cached = {"attempts": 4670, "billable_tokens": 2794797, "unknown": 8}
    original = dict(cached)
    assert recovery.cached_caps(cached) == {
        "EXACT_OPENROUTER_REQUEST_CAP": "4670",
        "EXACT_OPENROUTER_TOKEN_CAP": "2794797",
        "EXACT_OPENROUTER_RETRY_UNKNOWN": "0",
    }
    assert cached == original


def budget_fixture(tmp_path, monkeypatch):
    root = tmp_path / "old"
    root.mkdir()
    budget = root / "budget.json"
    state = {
        "limits": {"protected": 10},
        "work": {
            recovery.RECONCILIATION_ID: {
                "status": "failed",
                "seconds": recovery.FAILED_SETUP_SECONDS,
            },
            "failed-interrupted": {"status": "failed", "requests": 12, "tokens": 1000},
        },
    }
    budget.write_text(json.dumps(state))
    (root / "status.json").write_text(json.dumps({"cumulative_budget": str(budget)}))
    monkeypatch.setattr(recovery, "BUDGET", budget)
    monkeypatch.setattr(recovery, "OLD_ROOT", root)
    monkeypatch.setattr(recovery.queue, "campaign_limits", lambda lock: state["limits"])
    return root, budget, state


def test_latest_budget_is_read_without_resetting_or_reconciling_rows(tmp_path, monkeypatch):
    _, budget, expected = budget_fixture(tmp_path, monkeypatch)
    before = budget.read_bytes()
    assert recovery.validate_latest_budget(binding(budget), {}, helpers()) == expected
    assert budget.read_bytes() == before


@pytest.mark.parametrize(
    "change", ["stale_path", "stale_bytes", "new_owner", "reservation", "inventory"]
)
def test_stale_or_incomplete_budget_is_rejected(tmp_path, monkeypatch, change):
    root, budget, state = budget_fixture(tmp_path, monkeypatch)
    item = binding(budget)
    if change == "stale_path":
        previous = root / "earlier.json"
        previous.write_text(json.dumps(state))
        item = binding(previous)
    elif change == "stale_bytes":
        budget.write_text(json.dumps({**state, "later_charge": 8}))
    elif change == "new_owner":
        (root / "status.json").write_text(
            json.dumps({"cumulative_budget": str(root / "newer.json")})
        )
    elif change == "reservation":
        state["work"]["unfinished"] = {"status": "reserved"}
        budget.write_text(json.dumps(state))
        item = binding(budget)
    else:
        state["work"][recovery.RECONCILIATION_ID]["seconds"] = 0
        budget.write_text(json.dumps(state))
        item = binding(budget)
    with pytest.raises(ValueError):
        recovery.validate_latest_budget(item, {}, helpers())


@pytest.mark.parametrize("hidden_change", ["tracked_evaluator", "untracked_evaluator"])
def test_evaluation_changes_cannot_hide_outside_prediction_fingerprint(
    tmp_path, monkeypatch, hidden_change
):
    old = {
        "commit": "historical",
        "native_code_sha256": {"native": "fixed"},
        "extraction_code": {"files": {recovery.METADATA_SOURCE: "old"}},
    }
    current = {
        "native_code_sha256": {"native": "fixed"},
        "extraction_code": {
            "files": {recovery.METADATA_SOURCE: "new", recovery.AUDIT_SOURCE: "new"}
        },
    }

    def git(command, **kwargs):
        if command[1] == "diff":
            return "exact/evaluation.py\n" if hidden_change == "tracked_evaluator" else ""
        if command[1] == "ls-files":
            return "exact/evaluation.py\n" if hidden_change == "untracked_evaluator" else ""
        raise AssertionError("Numerical patch validation must not begin")

    monkeypatch.setattr(recovery.subprocess, "check_output", git)
    with pytest.raises(ValueError, match="evaluation source changed"):
        recovery.verify_source_impact(tmp_path, old, current, {"old_source_commit": "historical"})
