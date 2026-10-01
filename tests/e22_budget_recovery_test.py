"""A narrow supervision-key migration cannot bless another scientific change."""

import copy

import pytest

from tools.recover_e22_budget import verify_identity_transition


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
