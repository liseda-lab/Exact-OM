"""Scoped selected-evidence visibility over new and historical immutable indexes."""

import copy
import json
import sqlite3

import pyowl_core as core
import pytest

from exact_inspect.context import build_context_package
from exact_inspect.contracts import VisibilityPolicy, file_hash
from exact_inspect.models import SelectedEvidence
from exact_inspect.study.store import StudyStore
from exact_inspect.study.workspace import build_workspace_scope, validate_publication_workspaces
from tests.explanation_study_workspace_test import Store, client_for


class EvidenceStore(Store):
    _check_safe_json = staticmethod(StudyStore._check_safe_json)


def prepared_evidence(tmp_path):
    context = build_context_package(
        core.load_snapshot(b"""Ontology(
                Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>))
                AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "Source")
                AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:B> "Target")
                AnnotationAssertion(<http://purl.obolibrary.org/obo/IAO_0000115> <urn:A> "PRIVATE HIDDEN DEFINITION")
            )"""),
        tmp_path / "original",
    )
    version = context.ontology_version_id
    source = {"ontology_version_id": version, "iri": "urn:A", "kind": "class"}
    target = {**source, "iri": "urn:B"}
    case = {
        "case_id": "case-a",
        "source": source,
        "candidates": [{"candidate_id": "candidate-a", "entity": target}],
        "explanation_refs": [],
    }
    policy = VisibilityPolicy(ontology_ids=(version,), categories=("labels",))
    label = context.facts(source, category="labels")["items"][0]["fact_id"]
    hidden = context.facts(source, category="definitions")["items"][0]["fact_id"]
    rows = []
    for number, (resolution, fid) in enumerate(
        [("resolved", label), ("resolved", hidden), ("unresolved", None), ("not_exported", None)]
    ):
        row = {
            "evidence_id": f"selected-{number}",
            "channel": "attributes",
            "side": "source",
            "role": "attribute",
            "entity": source,
            "feature_id": f"feature-{number}" if fid else None,
            "fact_ids": [fid] if fid else [],
            "source_axiom_refs": ["PRIVATE RAW REFERENCE"],
            "axiom_origins": [{"raw": "PRIVATE ORIGIN"}],
            "semantic_terms": {"value": "PRIVATE COPIED VALUE"},
            "historical_item_alias": "PRIVATE LEGACY ALIAS",
            "display": {"text": "PRIVATE DISPLAY"},
            "values": {"weight": 0.5},
            "interpretation": "projected",
            "provenance_status": "recorded",
            "status": "available" if fid else "not_exported",
            "origin_resolution": resolution,
            "graph_projection": "unsupported",
            "reason": None if fid else "PRIVATE MISSING REASON",
        }
        SelectedEvidence.model_validate(row)
        rows.append(row)
    return context, case, policy, rows, label


def build_scope(tmp_path, context, case, policy, rows):
    return build_workspace_scope(
        tmp_path,
        scope_id="evidence-scope",
        kind="case",
        case=case,
        contexts={context.ontology_version_id: context},
        policy=policy,
        evidence={"candidate-a": rows},
        components=["context", "evidence"],
    )


@pytest.mark.parametrize("historical_raw_payload", [False, True])
def test_scoped_evidence_keeps_all_rows_but_redacts_raw_projections(
    tmp_path, historical_raw_payload
):
    context, case, policy, rows, label = prepared_evidence(tmp_path)
    scope = build_scope(tmp_path, context, case, policy, rows)
    database = tmp_path / scope.index_path
    with sqlite3.connect(database) as db:
        saved = [json.loads(r[0]) for r in db.execute("SELECT payload FROM evidence ORDER BY id")]
        assert "PRIVATE" not in json.dumps(saved)
        assert len(saved) == 4
        assert saved[0]["fact_ids"] == [label]
        assert saved[1]["origin_resolution"] == "withheld"
        if historical_raw_payload:
            # Reconstruct pre-correction immutable bytes and bind the publication to
            # them before any request. Serving must sanitize without rewriting.
            db.executemany(
                "UPDATE evidence SET payload=? WHERE id=?",
                [(json.dumps(row), row["evidence_id"]) for row in rows],
            )
    scope = scope.model_copy(update={"index_sha256": file_hash(database).removeprefix("sha256:")})
    study = {
        "contract_version": "exact-study/1.0",
        "study_revision": "evidence-revision",
        "cases": [case],
        "policy_hash": policy.policy_hash.removeprefix("sha256:"),
        "visibility_policy": policy.model_dump(mode="json"),
        "workspace_scopes": [scope.model_dump(mode="json")],
    }
    store = EvidenceStore(tmp_path, study)
    frozen_hash = file_hash(database)
    validate_publication_workspaces(store, study, {})
    with client_for(store) as client:
        url = "/api/v1/study/workspace/evidence-scope/evidence"
        params = {"candidate_id": "candidate-a", "limit": 1}
        items = []
        cursor = None
        for _ in range(4):
            response = client.get(url, params={**params, **({"cursor": cursor} if cursor else {})})
            assert response.status_code == 200, response.text
            assert "PRIVATE" not in response.text
            page = response.json()
            assert page["total_count"] == 4 and page["returned_count"] == 1
            items.extend(page["items"])
            cursor = page["next_cursor"]
        assert cursor is None
        assert [item["evidence_id"] for item in items] == [row["evidence_id"] for row in rows]
        assert [item["origin_resolution"] for item in items] == [
            "resolved",
            "withheld",
            "unresolved",
            "not_exported",
        ]
        assert items[0]["fact_ids"] == [label]
        assert all(item["fact_ids"] == [] for item in items[1:])
        assert all(item["graph_projection"] == "unsupported" for item in items)
        first_cursor = client.get(url, params=params).json()["next_cursor"]
        store.current["presentation_id"] = "next-presentation"
        assert client.get(url, params={**params, "cursor": first_cursor}).status_code == 409
        store.current["condition"] = "baseline"
        assert client.get(url, params=params).status_code == 403
    assert file_hash(database) == frozen_hash


def test_scoped_sanitization_keeps_typed_subject_admission(tmp_path, monkeypatch):
    context, case, policy, rows, _ = prepared_evidence(tmp_path)
    other_fact = context.facts(case["candidates"][0]["entity"], category="labels")["items"][0]
    wrong_subject = copy.deepcopy(rows[0])
    wrong_subject["fact_ids"] = [other_fact["fact_id"]]
    scope = build_scope(tmp_path, context, case, policy, [wrong_subject])
    with sqlite3.connect(tmp_path / scope.index_path) as db:
        saved = json.loads(db.execute("SELECT payload FROM evidence").fetchone()[0])
    assert saved["origin_resolution"] == "unresolved" and saved["fact_ids"] == []
    # The admission guard remains strict for any link surviving the filter.
    monkeypatch.setattr(
        "exact_inspect.evidence_metadata.filter_evidence_items", lambda items, *a, **k: items
    )
    from exact_inspect.study.workspace import _admit_evidence

    with pytest.raises(ValueError, match="outside the admitted typed subject"):
        _admit_evidence(
            wrong_subject,
            case["candidates"][0],
            case,
            policy,
            {context.ontology_version_id: context},
        )
