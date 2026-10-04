"""An operational timeout must not renew an interrupted scientific allowance."""
from pathlib import Path

import pytest

from tools.repair import scaling_endpoint_continuation as continuation
from tools.repair.batch import read, sha
from tools.repair.expanded_corpus import binding
from tools.repair.expanded_profile import checkpoint


def fixture(tmp_path):
    original = tmp_path / "qualification"
    row = original / "rows/prefix.json"
    checkpoint(row, "prefix", status="timeout", detail="stage deadline exhausted",
               cleanup_complete=True, payloads=[], result=None)
    guard = original / "inflight/interrupted.json"
    checkpoint(guard, "interrupted", case_index=3, stage="semantic_circuit", started_epoch=20)
    complete = tmp_path / "completion.json"
    complete.write_text('{"finished_epoch": 55}')
    plan = dict(original=str(original), output=str(tmp_path / "revision"),
                original_step="14408.295", rows=[binding(row)], guard=binding(guard),
                completion=binding(complete), interrupted_payloads=[], unpublished_payloads=[],
                interrupted_case_id="coherent-case")
    owner = dict(step_id="14408.295", surviving_processes=0, cgroup_absent=True, signals_sent=0)
    return plan, read(complete), read(guard), owner


def test_revision_preserves_evidence_and_interrupted_unknown_without_replay(tmp_path):
    plan, complete, guard, owner = fixture(tmp_path)
    original = Path(plan["original"])
    before = {str(p): sha(p) for p in original.rglob("*.json")}
    continuation.seed_revision(plan, "key", complete, guard, owner)
    revision = Path(plan["output"])
    assert sha(revision / "rows/prefix.json") == plan["rows"][0]["sha256"]
    saved = read(revision / "rows/interrupted.json")
    assert saved["status"] == "unknown_after_operational_timeout"
    assert saved["result"] is saved["pool"] is None
    assert saved["resources"] == {}  # No invented row CPU or memory telemetry.
    assert saved["elapsed_seconds"] == 35 and saved["additional_scientific_seconds"] == 0
    assert saved["original_guard"] == plan["guard"]
    assert {str(p): sha(p) for p in original.rglob("*.json")} == before
    assert not (revision / "inflight/interrupted.json").exists()
    digest = sha(revision / "rows/interrupted.json")
    continuation.seed_revision(plan, "key", complete, guard, dict(owner, checked_epoch=100))
    assert sha(revision / "rows/interrupted.json") == digest


@pytest.mark.parametrize("field,value", [("surviving_processes", 1), ("cgroup_absent", False),
                                         ("signals_sent", 1), ("step_id", "14408.0")])
def test_unresolved_ownership_prevents_any_revision(tmp_path, field, value):
    plan, complete, guard, owner = fixture(tmp_path)
    with pytest.raises(ValueError, match="owner cleanup"):
        continuation.seed_revision(plan, "key", complete, guard, dict(owner, **{field: value}))
    assert not Path(plan["output"]).exists()


@pytest.mark.parametrize("which", ["original", "prefix", "unknown"])
def test_changed_evidence_is_rejected_without_rewriting(tmp_path, which):
    plan, complete, guard, owner = fixture(tmp_path)
    continuation.seed_revision(plan, "key", complete, guard, owner)
    path = (Path(plan["rows"][0]["path"]) if which == "original" else
            Path(plan["output"]) / "rows" / ("prefix.json" if which == "prefix" else "interrupted.json"))
    path.write_text('{}')
    with pytest.raises(ValueError):
        continuation.seed_revision(plan, "key", complete, guard, owner)
    assert path.read_text() == '{}'
