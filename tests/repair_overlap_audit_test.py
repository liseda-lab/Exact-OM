"""Integrity and missingness regressions for final overlap accounting."""

import json

import pytest

from exact.repair.records import canonical_hash
from tools.repair import overlap_audit as audit


def save(path, value):
    path.write_text(json.dumps(value))
    return audit.binding(path)


def test_mutated_payload_is_rejected(tmp_path):
    ref = save(tmp_path / "payload.json", {"logical_status": "UNKNOWN"})
    (tmp_path / "payload.json").write_text('{"logical_status":"VERIFIED_FEASIBLE"}')
    with pytest.raises(ValueError, match="Changed evidence"):
        audit.Evidence().read(ref)


def test_checkpoint_cannot_change_dependencies_or_content(tmp_path):
    value = dict(identity="source-a", result="unknown")
    ref = save(tmp_path / "row.json", dict(value, content_hash=canonical_hash(value)))
    with pytest.raises(ValueError, match="dependencies"):
        audit.Evidence().checkpoint(ref, "source-b")
    value["result"] = "verified"
    ref = save(tmp_path / "row.json", dict(value, content_hash="stale"))
    with pytest.raises(ValueError, match="content"):
        audit.Evidence().checkpoint(ref, "source-a")


def test_receipt_cannot_use_another_dispatch_nonce(tmp_path):
    run = dict(step_id="14408.9", dispatch_nonce="expected")
    complete = save(tmp_path / "completion.json", dict(step_id="14408.9", dispatch_nonce="other"))
    step = save(tmp_path / "step.json", run)
    with pytest.raises(ValueError, match="nonce"):
        audit.receipt(audit.Evidence(), dict(run=run, completion=complete, step=step), {})


def test_process_completion_does_not_make_a_pair_available():
    rows = [dict(arm_id="model", control=control, condition=condition, status="complete",
                 logical_status="UNKNOWN", semantic_benefit=None, elapsed_seconds=1,
                 row_id=control + condition)
            for control in ("corrupted", "coherent") for condition in ("nonshared", "shared")]
    pairs = [dict(arm_id="model", control=control, available=False,
                  shared_minus_nonshared_benefit=None, shared_minus_nonshared_seconds=0,
                  rows={c: control + c for c in ("nonshared", "shared")})
             for control in ("corrupted", "coherent")]
    schedule = dict(arms=[dict(id="model")])
    assert audit.validate_pairs(schedule, dict(pairs=pairs), rows) == pairs
    pairs[0]["available"] = True
    with pytest.raises(ValueError, match="scientific availability"):
        audit.validate_pairs(schedule, dict(pairs=pairs), rows)


@pytest.mark.parametrize("corruption", ["missing", "duplicate"])
def test_evaluation_cannot_drop_or_duplicate_scheduled_rows(tmp_path, corruption):
    rows = [dict(id=str(i), case_index=0, arm_id="model") for i in range(36)]
    schedule = dict(rows=rows, cases=[dict(condition="shared", control="corrupted")])
    effective = save(tmp_path / "schedule.json", dict(schedule, witness_report={}))
    source = tmp_path / "tools/repair/fresh_evaluation.py"
    identity = canonical_hash((effective["sha256"], "source-hash", {}))
    refs = []
    for row in rows:
        value = dict(identity=canonical_hash((identity, row)), row=row, cleanup_complete=True,
                     status="timeout", elapsed_seconds=300, resources={})
        refs.append(save(tmp_path / (row["id"] + ".json"),
                         dict(value, content_hash=canonical_hash(value))))
    selected = refs[:-1] if corruption == "missing" else refs[:-1] + [refs[0]]
    value = dict(identity=identity, status="complete", runtime={}, schedule=effective, rows=selected)
    ref = save(tmp_path / "evaluation.json", dict(value, content_hash=canonical_hash(value)))
    report = dict(evaluation_reports=[ref], witness_report={})
    sources = {ref["path"]: dict(runtime={}, batch=dict(code=str(tmp_path),
                                                       frozen_files={str(source): "source-hash"}))}
    with pytest.raises(ValueError, match="denominator|Duplicate"):
        audit.validate_rows(audit.Evidence(), schedule, report, sources)
