"""A narrow supervision-key migration cannot bless another scientific change."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from tools.recover_e22_budget import (
    migrate_execution_measurement,
    verify_identity_transition,
)


def identities():
    old = {
        "artifact_id": "old",
        "stage": "extraction",
        "parameters": {"supervision": {"rerank": "supervised"}, "threshold": 0.5},
        "implementation": {"files": {"runtime.py": "old"}},
        "parents": ["inputs"],
        "seed": 17,
    }
    new = copy.deepcopy(old)
    new.update(artifact_id="new", implementation={"files": {"runtime.py": "new"}})
    new["parameters"].update(
        supervision={"label_budget": 25}, resolved_supervision={"rerank": "supervised"}
    )
    return old, new


def test_exact_identity_key_migration():
    old, new = identities()
    verify_identity_transition(
        old, new, old_implementation=old["implementation"], stage="extraction"
    )


@pytest.mark.parametrize("change", ["threshold", "seed", "parents", "mode", "old_code"])
def test_migration_rejects_other_changes(change):
    old, new = identities()
    implementation = copy.deepcopy(old["implementation"])
    if change == "threshold":
        new["parameters"]["threshold"] = 0.6
    elif change == "seed":
        new["seed"] = 29
    elif change == "parents":
        new["parents"] = ["changed-inputs"]
    elif change == "mode":
        new["parameters"]["resolved_supervision"] = {"rerank": "label_free"}
    else:
        implementation = {"files": {"runtime.py": "unverified"}}
    with pytest.raises(ValueError, match="differs beyond"):
        verify_identity_transition(old, new, old_implementation=implementation, stage="extraction")


@pytest.mark.parametrize("fail", [False, True])
def test_migration_charges_success_and_failure_once(tmp_path, monkeypatch, fail):
    from tools import finalize_prepared_selection
    from tools import recover_e22_budget as repair

    (tmp_path / "budget.json").write_text('{"limits": {}}')
    ledger = []

    class Account:
        def admit(self, work_id, **kw):
            ledger.append(("admit", work_id, kw))

        def finish(self, work_id, **kw):
            ledger.append(("finish", work_id, kw))

    monkeypatch.setattr("exact.experiments.budget.BudgetLedger", lambda *a: Account())
    monkeypatch.setattr(
        finalize_prepared_selection,
        "charge_failed_finalization",
        lambda *a: ledger.append(("prior",)),
    )

    def execute(*a, **kw):
        if fail:
            raise ValueError("damaged artifact")
        return {"verified": True}

    monkeypatch.setattr(repair, "_import_verified_controls", execute)
    if fail:
        with pytest.raises(ValueError, match="damaged artifact"):
            repair.import_verified_controls({}, None, tmp_path, None)
    else:
        assert repair.import_verified_controls({}, None, tmp_path, None) == {"verified": True}
    assert [item[0] for item in ledger] == ["prior", "admit", "finish"]
    assert ledger[1][1] == ledger[2][1]
    assert ledger[2][2]["status"] == ("failed" if fail else "complete")
    assert ledger[2][2]["requests"] == ledger[2][2]["tokens"] == 0


@pytest.mark.parametrize("arm", ["budget_25", "label_free"])
def test_migrated_control_restores_original_execution_measurement(tmp_path, arm):
    from exact.experiments.harness import _execution_measurement
    from exact.experiments.recovery import ArtifactStore, stage_identity

    options = {
        "stage": "extraction",
        "inputs": {"source": "a" * 64},
        "role": "development",
        "entity_kind": "class",
        "implementation": {"sha256": "b" * 64},
        "dependencies": {},
        "seed": 17,
    }
    original = stage_identity(**options, parameters={"arm": arm, "key": "original"})
    migrated = stage_identity(**options, parameters={"arm": arm, "key": "corrected"})
    measurement = {
        "schema_version": 1,
        "status": "measured",
        "artifact_id": original["artifact_id"],
        "origin_attempt": "original-independent-execution",
        "reason": None,
        "wall_seconds": 123.5,
        "peak_memory_kb": 700684,
    }
    raw = (json.dumps(measurement) + "\n").encode()
    name = "stats/execution_measurement.json"
    predictions = b"source,target,score\ns:1,t:2,0.75\n"
    source = ArtifactStore(tmp_path / "original")
    saved = source.publish(original, {name: raw, "alignment/predictions.csv": predictions})
    before = source._manifest_path(original["artifact_id"]).read_bytes()
    output = saved["outputs"][name]
    rebinding = migrate_execution_measurement(
        source._blob(output["sha256"]).read_bytes(),
        source_artifact_id=original["artifact_id"],
        target_artifact_id=migrated["artifact_id"],
        source_sha256=output["sha256"],
    )
    cell = SimpleNamespace(output_dir=tmp_path / "restored")
    recovery = SimpleNamespace(identities={"extraction": migrated}, reuse={"extraction"})
    historical = ArtifactStore(tmp_path / "broken-import")
    historical.publish(migrated, {name: raw, "alignment/predictions.csv": predictions})
    historical.restore(migrated["artifact_id"], cell.output_dir)
    with pytest.raises(ValueError, match="does not match restored predictions"):
        _execution_measurement(
            cell, recovery, elapsed=0, peak_kb=None, complete=True, continued=False
        )
    with pytest.raises(ValueError, match="immutable artifact"):
        historical.publish(migrated, {name: rebinding, "alignment/predictions.csv": predictions})

    destination = ArtifactStore(tmp_path / "fresh-import")
    destination.publish(migrated, {name: rebinding, "alignment/predictions.csv": predictions})
    destination.restore(migrated["artifact_id"], cell.output_dir)
    restored = _execution_measurement(
        cell, recovery, elapsed=0, peak_kb=None, complete=True, continued=False
    )
    proof = restored.pop("identity_migration")
    assert proof == {
        "reason": "E22-supervision-key-v1",
        "source_artifact_id": original["artifact_id"],
        "source_measurement_sha256": output["sha256"],
    }
    assert restored == {**measurement, "artifact_id": migrated["artifact_id"]}
    assert (cell.output_dir / "alignment/predictions.csv").read_bytes() == predictions
    assert source._manifest_path(original["artifact_id"]).read_bytes() == before
    assert source._blob(output["sha256"]).read_bytes() == raw
    assert historical.verify(migrated["artifact_id"])["outputs"][name] == output


@pytest.mark.parametrize("damage", ["checksum", "schema", "artifact", "migration", "shape"])
def test_measurement_migration_rejects_unverified_source(damage):
    measurement = {"schema_version": 1, "artifact_id": "a" * 64}
    if damage == "schema":
        measurement["schema_version"] = 2
    elif damage == "artifact":
        measurement["artifact_id"] = "c" * 64
    elif damage == "migration":
        measurement["identity_migration"] = {}
    elif damage == "shape":
        measurement = []
    raw = json.dumps(measurement).encode()
    digest = "f" * 64 if damage == "checksum" else hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="Original execution measurement"):
        migrate_execution_measurement(
            raw,
            source_artifact_id="a" * 64,
            target_artifact_id="b" * 64,
            source_sha256=digest,
        )
