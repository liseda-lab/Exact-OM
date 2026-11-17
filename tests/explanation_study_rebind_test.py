import json

import pytest

from exact_inspect.study import StudyStore
from exact_inspect.study.rebind import rebind_unstarted


def fixture(tmp_path):
    store = StudyStore("sqlite:///" + str(tmp_path / "test.sqlite"), allow_test_sqlite=True)
    payload = {"cases": ["unchanged"], "protocol_versions": {"forms": "old", "software": "same"}}
    with store.transaction() as db:
        for revision in ("old", "new"):
            body = {**payload, "study_revision": revision}
            db.execute(
                "INSERT INTO studies(revision,payload,frozen_hash) VALUES(?,?,?)",
                (revision, json.dumps(body), revision),
            )
        state = json.dumps({"stage": "welcome", "revision": 0, "consent": None, "assignment": None})
        for sid, test, revoked in [("active", 0, 0), ("test", 1, 0), ("revoked", 0, 1)]:
            db.execute(
                "INSERT INTO sessions(id,study_revision,generation,invite_digest,revoked,test,revision,state) VALUES(?,?,?,?,?,?,?,?)",
                (sid, "old", 1, "secret-digest-" + sid, revoked, test, 0, state),
            )
    return store


def test_preserves_links_state_and_nonparticipant_sessions(tmp_path):
    store = fixture(tmp_path)
    assert rebind_unstarted(store, "old", "new", ["active"]) == 1
    with store.transaction() as db:
        rows = {r["id"]: dict(r) for r in db.execute("SELECT * FROM sessions")}
    assert rows["active"]["study_revision"] == "new"
    assert rows["active"]["invite_digest"] == "secret-digest-active"
    assert rows["active"]["generation"] == 1 and rows["active"]["revision"] == 0
    assert rows["test"]["study_revision"] == rows["revoked"]["study_revision"] == "old"


@pytest.mark.parametrize("change", ["started", "new_invitation", "changed_cases"])
def test_aborts_without_modifying_any_session(tmp_path, change):
    store = fixture(tmp_path)
    with store.transaction() as db:
        if change == "started":
            db.execute("UPDATE sessions SET revision=1 WHERE id=?", ("active",))
        elif change == "new_invitation":
            db.execute("UPDATE sessions SET test=0 WHERE id=?", ("test",))
        else:
            db.execute(
                "UPDATE studies SET payload=? WHERE revision=?",
                (json.dumps({"cases": ["changed"], "protocol_versions": {}}), "new"),
            )
    with pytest.raises(ValueError):
        rebind_unstarted(store, "old", "new", ["active"])
    with store.transaction() as db:
        assert {r["study_revision"] for r in db.execute("SELECT study_revision FROM sessions")} == {
            "old"
        }
