"""Unpublished write recovery retains outcomes and refuses spent-row replay."""

import copy
from pathlib import Path

import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.representative_capacity import recovery_rows
from tools.repair.scaling_payload_recovery import payload_manifest


def fixture(tmp_path):
    rows = [dict(id=str(i), case_seconds=45, adapter=dict(pool_dependency=None)) for i in range(4)]
    prior = dict(rows=rows, source_commit="science", dispatch_nonce="old", models={"a": "fixed"})
    write_artifact(tmp_path / "manifest.json", prior)
    work = tmp_path / "work"
    write_artifact(work / "identity.json", dict(identity=canonical_hash(prior)))
    for index, status in enumerate(("complete", "timeout")):
        write_artifact(
            work / "rows" / str(index) / "completion.json",
            dict(
                identity=canonical_hash((canonical_hash(prior), rows[index])),
                status=status,
                result=dict(logical_status="UNKNOWN") if status == "complete" else None,
                cleanup_complete=True,
                charged_seconds=31 + index,
                artifacts=[],
            ),
        )
    write_artifact(
        work / "rows/2/started.json",
        dict(
            identity=canonical_hash((canonical_hash(prior), rows[2])),
            cost_identity="2",
            started_epoch=100,
            deadline_epoch=140,
            reserved_seconds=45,
        ),
    )
    payload = work / "rows/2/payload"
    payload.mkdir()
    (payload / ".selection.json.abcdefgh").touch(mode=0)
    published, unpublished = payload_manifest(work)
    write_artifact(
        tmp_path / "outputs.json",
        {str(Path(r["path"]).relative_to(work)): r["sha256"] for r in published},
    )
    write_artifact(tmp_path / "unpublished.json", dict(files=unpublished))
    write_artifact(
        tmp_path / "batch.json",
        dict(jobs=[dict(id="failed", commands=[[str(tmp_path / "manifest.json")]])]),
    )
    write_artifact(
        tmp_path / "completion.json",
        dict(
            status="failed",
            exit_code=1,
            step_id="14451.132",
            dispatch_nonce="old",
            batch=str(tmp_path / "batch.json"),
            job_id="failed",
            work=str(work),
            finished_epoch=150,
            error=dict(message="command 0 exited 1: PermissionError: " + unpublished[0]["path"]),
        ),
    )
    write_artifact(tmp_path / "step.json", dict(step_id="14451.132", dispatch_nonce="old"))
    write_artifact(
        tmp_path / "retirement.json",
        dict(
            retired_step="14451.132",
            live_steps=["14451.0"],
            observed_epoch=151,
            all_owned_processes_retired=True,
        ),
    )
    recovery = dict(
        schema="exact-repair/capacity-payload-recovery/v1",
        manifest=binding(tmp_path / "manifest.json"),
        failed_index=2,
        retirement=binding(tmp_path / "retirement.json"),
        temporary=unpublished[0],
        unpublished_inventory=binding(tmp_path / "unpublished.json"),
        budgets_reset=False,
        history_reset=False,
        failed_case_replayed=False,
        receipt=dict(
            **{
                key: binding(tmp_path / (key + ".json"))
                for key in ("batch", "completion", "step", "outputs")
            },
            expected_status="failed",
            step_id="14451.132",
            dispatch_nonce="old",
        ),
    )
    write_artifact(tmp_path / "recovery.json", recovery)
    return dict(prior, dispatch_nonce="new", technical_recovery=binding(tmp_path / "recovery.json"))


def test_retains_scientific_unknowns_and_full_interrupted_charge(tmp_path):
    manifest = fixture(tmp_path)
    retained = recovery_rows(manifest)
    assert list(retained) == ["0", "1", "2"]
    assert retained["0"]["result"] == dict(logical_status="UNKNOWN")
    assert retained["1"]["status"] == "timeout"
    assert retained["1"]["charged_seconds"] == 32
    assert retained["2"]["status"] == "prior_implementation_failure"
    assert retained["2"]["result"] is None
    assert retained["2"]["charged_seconds"] == retained["2"]["prior_charged_seconds"] == 45
    assert not (tmp_path / "work/rows/2/completion.json").exists()


@pytest.mark.parametrize(
    "damage", ["budget", "model", "nonce", "live", "artifact", "spent_tail", "history", "guard"]
)
def test_rejects_unbound_changes_or_replay(tmp_path, damage):
    manifest = fixture(tmp_path)
    recovery = read(tmp_path / "recovery.json")
    if damage == "budget":
        manifest["rows"][0]["case_seconds"] += 1
    elif damage == "model":
        manifest["models"]["a"] = "changed"
    elif damage == "nonce":
        recovery["receipt"]["dispatch_nonce"] = "wrong"
    elif damage == "live":
        value = read(tmp_path / "retirement.json")
        value["live_steps"].append("14451.132")
        write_artifact(tmp_path / "retirement.json", value)
        recovery["retirement"] = binding(tmp_path / "retirement.json")
    elif damage == "artifact":
        (tmp_path / "work/rows/0/completion.json").write_text("{}")
    elif damage == "spent_tail":
        (tmp_path / "work/rows/3").mkdir()
    elif damage == "history":
        recovery["history_reset"] = True
    else:
        value = read(tmp_path / "work/rows/2/started.json")
        value["reserved_seconds"] = 90
        write_artifact(tmp_path / "work/rows/2/started.json", value)
    write_artifact(tmp_path / "recovery.json", recovery)
    manifest["technical_recovery"] = binding(tmp_path / "recovery.json")
    with pytest.raises((ValueError, RuntimeError)):
        recovery_rows(manifest)
