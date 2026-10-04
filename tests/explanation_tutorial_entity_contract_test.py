"""Explicit v2 tutorial action identities at the real HTTP validation boundary."""

import copy
import hashlib
import json
import os
from typing import Annotated
from uuid import uuid4

import pytest
from pydantic import Field
from fastapi.testclient import TestClient

from exact_inspect.study.api import create_study_app
from exact_inspect.study.models import EntityRef, Publish
from exact_inspect.study.store import StudyStore
from exact_inspect.study.tutorial import entity_key, walk_entities
from exact_inspect.study.v2_models import StudyStateV2, TutorialAction, TutorialProgressMutation
from tests.explanation_corrective_http_test import corrected, mutate, reach_tutorial
from tests.explanation_study_v2_test import actions_for
from tests.explanation_study_v2_fixtures import publication_v2
from tests.explanation_study_test import ADMIN, ORIGIN, SECRET, invite


def test_missing_action_kind_cannot_complete_parent_navigation(corrected):
    client, store, publication = corrected
    state = reach_tutorial(client, publication)
    action = next(
        a
        for a in actions_for(publication.definition, store.assets_dir)
        if a["action"] == "navigate_parent"
    )
    action = copy.deepcopy(action)
    action["entity"].pop("kind")
    result = mutate(
        client,
        "PUT",
        "tutorial/progress",
        tutorial_version=state["tutorial"]["version"],
        actions=[action],
    )
    assert result.status_code == 422
    assert any(e["loc"][-1] == "kind" for e in result.json()["detail"])
    assert client.get("/api/v1/study/state").json() == state


def test_explicit_types_invalid_types_and_atomic_rejections(corrected):
    client, store, publication = corrected
    state = reach_tutorial(client, publication)
    actions = actions_for(publication.definition, store.assets_dir)
    parent = next(a for a in actions if a["action"] == "navigate_parent")
    resource = json.loads((store.assets_dir / "practice-explanations.json").read_text())
    prop = next(e for e in walk_entities(resource) if e.get("kind") == "object_property")
    search = next(a for a in actions if a["action"] == "search_entity")
    for action in [parent, {**search, "entity": prop}]:
        response = mutate(
            client,
            "PUT",
            "tutorial/progress",
            tutorial_version=state["tutorial"]["version"],
            actions=[action],
        )
        assert response.status_code == 200, response.text
        assert (
            action["requirement_id"]
            in response.json()["tutorial_progress"]["completed_requirements"]
        )
    state = client.get("/api/v1/study/state").json()

    def ledger():
        with store.transaction() as db:
            return {
                table: db.execute(
                    f"SELECT COUNT(*) AS n FROM {table} WHERE session_id=?",
                    (state["session_id"],),
                ).fetchone()["n"]
                for table in ("history", "mutations")
            }

    before = ledger()
    invalid = [
        {**parent["entity"], "kind": None},
        {**parent["entity"], "kind": "property"},
        {**parent["entity"], "kind": "individual"},
        {**parent["entity"], "kind": "data_property"},
        {**parent["entity"], "kind": "object_property"},
        {**prop, "kind": "class"},
        {**parent["entity"], "ontology_version_id": "another-ontology"},
    ]
    for entity in invalid:
        response = mutate(
            client,
            "PUT",
            "tutorial/progress",
            tutorial_version=state["tutorial"]["version"],
            help_opened=True,
            actions=[{**parent, "entity": entity}],
        )
        assert response.status_code == 422, response.text
        assert client.get("/api/v1/study/state").json() == state
        assert ledger() == before
    # All four recorded kinds are legal; permission still requires an exact frozen identity.
    for kind in ["class", "object_property", "data_property", "individual"]:
        assert (
            TutorialAction.model_validate({**search, "entity": {**prop, "kind": kind}}).entity.kind
            == kind
        )


def test_legacy_entity_default_and_frozen_v2_bytes_remain_compatible(corrected):
    client, store, publication = corrected
    legacy = {"ontology_version_id": "legacy", "iri": "urn:legacy:Class"}
    assert EntityRef.model_validate(legacy).kind == "class"
    raw = publication.model_dump(mode="json")
    raw["definition"]["cases"][0]["source"].pop("kind")
    reparsed = Publish.model_validate(raw)
    assert reparsed.model_dump(mode="json") == publication.model_dump(mode="json")
    assert store.publish(reparsed) == store.publish(publication)
    state = reach_tutorial(client, publication)
    action = next(
        a
        for a in actions_for(publication.definition, store.assets_dir)
        if a["action"] == "navigate_parent"
    )
    body = dict(
        idempotency_key=uuid4().hex,
        expected_revision=state["revision"],
        tutorial_version=state["tutorial"]["version"],
        actions=[action],
    )
    path = "/api/v1/study/tutorial/progress"
    implicit = copy.deepcopy(body)
    implicit["actions"][0]["entity"].pop("kind")

    # Reconstruct the old request models to store a genuinely implicit historical
    # receipt. The new HTTP validator must reject its original request, while the
    # explicit resend still has the exact old canonical hash and replays the receipt.
    class LegacyAction(TutorialAction):
        entity: EntityRef | None = None

    class LegacyProgress(TutorialProgressMutation):
        actions: Annotated[list[LegacyAction], Field(max_length=100)] = Field(default_factory=list)

    with store.transaction() as db:
        generation = db.execute(
            "SELECT generation FROM sessions WHERE id=?", (state["session_id"],)
        ).fetchone()["generation"]
    old_request = LegacyProgress.model_validate(implicit)
    old_receipt = store.mutate(state["session_id"], generation, "tutorial_progress", old_request)
    restored = StudyStore(store.database_url, store.assets_dir, allow_test_sqlite=True)
    assert (
        restored.state(state["session_id"], generation)["tutorial_progress"]
        == old_receipt["tutorial_progress"]
    )
    assert client.put(path, json=implicit).status_code == 422
    replay = client.put(path, json=body)
    assert replay.status_code == 200
    assert replay.json() == StudyStateV2.model_validate(old_receipt).model_dump(
        mode="json", exclude_unset=True
    )
    assert (
        restored.mutate(
            state["session_id"],
            generation,
            "tutorial_progress",
            TutorialProgressMutation.model_validate(body),
        )
        == old_receipt
    )


def test_entity_key_does_not_infer_action_evidence():
    for kind in (None, "property"):
        with pytest.raises(ValueError, match="explicit recorded kind"):
            entity_key({"ontology_version_id": "frozen", "iri": "urn:entity", "kind": kind})
    with pytest.raises(ValueError, match="explicit recorded kind"):
        entity_key({"ontology_version_id": "frozen", "iri": "urn:entity"})


def test_explicit_action_preserves_legacy_frozen_resource_defaults(tmp_path):
    publication = publication_v2(tmp_path)
    path = tmp_path / "practice-explanations.json"
    resource = json.loads(path.read_bytes())
    parent = copy.deepcopy(resource["hierarchy"][0]["parent"])
    # Historically admitted resource entities and hierarchy endpoints could omit
    # class kinds. Keep their exact bytes after publication and normalize on read.
    removed = 0
    for entity in walk_entities(resource):
        if entity.get("kind") == "class":
            entity.pop("kind")
            removed += 1
    assert removed > 0
    frozen_bytes = json.dumps(resource).encode()
    path.write_bytes(frozen_bytes)
    asset = next(a for a in publication.definition.assets if a.asset_id == "practice-explanations")
    asset.sha256 = hashlib.sha256(frozen_bytes).hexdigest()
    asset.size_bytes = len(frozen_bytes)
    publication = Publish.model_validate(publication.model_dump(mode="json"))
    frozen_publication = publication.model_dump_json()
    app = create_study_app(
        os.environ.get("EXACT_STUDY_TEST_DATABASE_URL", f"sqlite:///{tmp_path / 'legacy.sqlite'}"),
        SECRET,
        ADMIN,
        ORIGIN,
        tmp_path,
        allow_test_sqlite=True,
    )
    store = app.state.study_store
    receipt = store.publish(publication)
    with TestClient(app, base_url=ORIGIN, headers={"Origin": ORIGIN}) as client:
        state = client.post(
            "/api/v1/study/session", json={"secret": invite(store, publication)}
        ).json()
        client.headers["X-Study-Session"] = state["session_id"]
        state = reach_tutorial(client, publication)
        response = mutate(
            client,
            "PUT",
            "tutorial/progress",
            tutorial_version=state["tutorial"]["version"],
            actions=[
                {"requirement_id": "context.parent", "action": "navigate_parent", "entity": parent}
            ],
        )
        assert response.status_code == 200, response.text
        assert "context.parent" in response.json()["tutorial_progress"]["completed_requirements"]
    assert path.read_bytes() == frozen_bytes
    assert publication.model_dump_json() == frozen_publication
    assert store.publish(publication) == receipt
