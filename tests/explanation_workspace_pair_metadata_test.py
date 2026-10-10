"""The shared graph comparison contract remains bound to the current study pair."""

import copy
import json
import sqlite3

import pytest

from exact_inspect.contracts import DomainError, file_hash
from exact_inspect.evidence_metadata import evidence_completeness
from exact_inspect.study.workspace import (
    WorkspaceScope,
    build_workspace_scope,
    validate_publication_workspaces,
)
from tests.explanation_workspace_evidence_test import EvidenceStore, prepared_evidence
from tests.explanation_study_workspace_test import client_for


def inputs(tmp_path):
    context, case, policy, rows, label = prepared_evidence(tmp_path)
    source = rows[0]
    target = copy.deepcopy(source)
    target.update(
        evidence_id="target-feature", side="target", entity=case["candidates"][0]["entity"]
    )
    target["fact_ids"] = [context.facts(target["entity"], category="labels")["items"][0]["fact_id"]]
    rows = [source, target]
    bridge = dict(
        bridge_id="recorded-comparison",
        level="feature_correspondence",
        channel="attributes",
        source=case["source"],
        target=target["entity"],
        source_evidence_ids=[source["evidence_id"]],
        target_evidence_ids=[target["evidence_id"]],
        source_fact_ids=source["fact_ids"],
        target_fact_ids=target["fact_ids"],
        artifact_refs=["sha256:" + "a" * 64],
    )
    metadata = dict(
        source=case["source"],
        target=target["entity"],
        evidence_metadata=dict(
            status="selected",
            counts=evidence_completeness(rows),
            bridge_status="recorded",
            bridges=[bridge],
        ),
    )
    return context, case, policy, rows, copy.deepcopy(metadata)


def prepare(tmp_path, *, modern=True, change=None):
    context, case, policy, rows, metadata = inputs(tmp_path)
    if change:
        change(metadata)
    scope = build_workspace_scope(
        tmp_path,
        scope_id="comparison-scope",
        kind="case",
        case=case,
        contexts={context.ontology_version_id: context},
        policy=policy,
        evidence={"candidate-a": rows},
        pair_metadata={"candidate-a": metadata} if modern else None,
        components=["context", "evidence"],
    )
    study = dict(
        contract_version="exact-study/1.0",
        study_revision="metadata-revision",
        cases=[case],
        policy_hash=policy.policy_hash[7:],
        visibility_policy=policy.model_dump(mode="json"),
        workspace_scopes=[scope.model_dump(mode="json")],
    )
    return EvidenceStore(tmp_path, study), scope


def test_actual_scoped_http_metadata_and_authorization(tmp_path):
    store, scope = prepare(tmp_path)
    validate_publication_workspaces(store, store.study, {})
    frozen = file_hash(tmp_path / scope.index_path)
    with client_for(store) as client:
        prefix = "/api/v1/study/workspace/comparison-scope"
        assert (
            client.get(prefix + "/capabilities").json()["evidence_metadata_contract"]
            == "exact-evidence-metadata/1"
        )
        response = client.get(prefix + "/evidence-metadata", params={"candidate_id": "candidate-a"})
        assert response.status_code == 200, response.text
        value = response.json()
        assert value["source"]["iri"] == "urn:A" and value["target"]["iri"] == "urn:B"
        assert value["evidence_metadata"]["counts"]["selected"] == 2
        assert value["evidence_metadata"]["bridges"][0]["level"] == "feature_correspondence"
        assert "PRIVATE" not in response.text
        assert (
            client.get(prefix + "/evidence", params={"candidate_id": "candidate-a"}).status_code
            == 200
        )
        assert (
            client.get(prefix + "/evidence-metadata", params={"candidate_id": "alien"}).status_code
            == 403
        )
        store.current["condition"] = "ontology_baseline"
        assert (
            client.get(
                prefix + "/evidence-metadata", params={"candidate_id": "candidate-a"}
            ).status_code
            == 403
        )
        store.current["condition"] = "explanation"
        store.state["stage"] = "paused"
        assert (
            client.get(
                prefix + "/evidence-metadata", params={"candidate_id": "candidate-a"}
            ).status_code
            == 403
        )
        store.state["stage"] = "case"
        assert (
            client.get(
                prefix + "/evidence-metadata", params={"candidate_id": "candidate-a"}
            ).status_code
            == 200
        )
    assert file_hash(tmp_path / scope.index_path) == frozen


def test_legacy_index_and_nested_scope_serialization_remain_unchanged(tmp_path):
    store, scope = prepare(tmp_path, modern=False)
    frozen = scope.model_dump(mode="json")
    assert "index_version" not in frozen
    assert WorkspaceScope.model_validate(frozen).model_dump(mode="json") == frozen
    validate_publication_workspaces(store, store.study, {})
    with client_for(store) as client:
        prefix = "/api/v1/study/workspace/comparison-scope"
        assert client.get(prefix + "/capabilities").json()["evidence_metadata_contract"] is None
        assert (
            client.get(
                prefix + "/evidence-metadata", params={"candidate_id": "candidate-a"}
            ).status_code
            == 404
        )
        assert (
            client.get(prefix + "/evidence", params={"candidate_id": "candidate-a"}).status_code
            == 200
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["target"].update(kind="object_property"),
        lambda m: m["evidence_metadata"]["bridges"][0].update(target_evidence_ids=["alien"]),
        lambda m: m["evidence_metadata"]["bridges"][0].update(source_fact_ids=["alien"]),
        lambda m: m["evidence_metadata"]["bridges"][0].update(channel="unrecorded-channel"),
    ],
)
def test_cross_pair_and_unlinked_comparisons_fail_admission(tmp_path, change):
    with pytest.raises((ValueError, DomainError)) as error:
        prepare(tmp_path, change=change)
    assert "comparison" in str(error.value).lower() or "evidence" in str(error.value).lower()
    assert not list((tmp_path / "workspaces").rglob("workspace.sqlite"))


def test_publication_rechecks_tampered_metadata_before_serving(tmp_path):
    store, scope = prepare(tmp_path)
    database = tmp_path / scope.index_path
    with sqlite3.connect(database) as db:
        value = json.loads(db.execute("SELECT payload FROM pair_metadata").fetchone()[0])
        value["target"]["iri"] = "urn:alien"
        db.execute("UPDATE pair_metadata SET payload=?", (json.dumps(value),))
    scope = scope.model_copy(update={"index_sha256": file_hash(database)[7:]})
    store.study["workspace_scopes"] = [scope.model_dump(mode="json")]
    with pytest.raises(ValueError, match="active typed pair"):
        validate_publication_workspaces(store, store.study, {})
