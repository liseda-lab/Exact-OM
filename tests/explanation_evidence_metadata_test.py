"""Recorded exact-rule effects and conservation of selected-evidence identities."""

import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from exact.runs import ExplanationStore
from exact.runs.decisions import observe_dataset
from exact_inspect.artifacts import publish_bundle, validate_bundle
from exact_inspect.context import OntologyContext
from exact_inspect.contracts import DomainError, VisibilityPolicy
from exact_inspect.decisions import (
    DecisionStore,
    _event,
    _missing_event,
    import_run,
    selected_evidence,
)
from exact_inspect.evidence_metadata import (
    PairEvidenceMetadata,
    evidence_completeness,
    exact_match_basis,
    filter_evidence_items,
    filter_pair_metadata,
    pair_evidence_metadata,
    recorded_bridges,
)
from exact_inspect.models import GenerationManifest, SelectedEvidence
from exact_inspect.service import create_prepared_app
from tests.explanation_preparation_test import prepared  # noqa: F401

SOURCE = {"ontology_version_id": "ov", "iri": "urn:shared", "kind": "class"}
TARGET = {"ontology_version_id": "ov", "iri": "urn:shared", "kind": "individual"}


def features():
    record = {
        "src_iri": SOURCE["iri"],
        "tgt_iri": TARGET["iri"],
        "src_kind": "class",
        "tgt_kind": "individual",
        "attributes": {
            "source": [
                {"property_iri": "urn:p", "value": "visible", "graph_projection": "unsupported"},
                {"property_iri": "urn:p", "value": "unresolved"},
                {"text": "private legacy text", "item_id": "legacy"},
            ],
            "target": [{"property_iri": "urn:p", "value": "hidden"}],
        },
    }
    items = selected_evidence(
        record,
        SOURCE,
        TARGET,
        resolver=lambda item, entity: {
            "fact_ids": [] if item.get("value") == "unresolved" else [item["value"]]
        },
    )
    return record, items


def test_actual_producer_effects_remain_distinct_and_never_infer_exact_basis():
    raw = [
        {"Src": "urn:s", "Tgt": "urn:exact", "SrcKind": "class", "TgtKind": "class"},
        {"Src": "urn:s", "Tgt": "urn:removed", "SrcKind": "class", "TgtKind": "class"},
        {"Src": "urn:other", "Tgt": "urn:scored", "SrcKind": "class", "TgtKind": "class"},
    ]
    dataset = SimpleNamespace(
        _candidates=pd.DataFrame(raw),
        dataframe=pd.DataFrame([{**raw[0], "prefiltered": True}, {**raw[2], "prefiltered": False}]),
        candidate_pool_manifest={"origin": "provided"},
        filter_exact_matches=True,
        exact_matches=None,
        drop_exact_match_sources=True,
    )
    observe_dataset(dataset)
    events = [
        next(e for e in value["candidate_decision"]["events"] if e["stage"] == "prefilter")
        for value in dataset._candidate_stage_observations.values()
    ]
    assert {e["reason_code"] for e in events} == {
        "exact_match_protected",
        "exact_source_prefiltered",
        "prefilter_retained",
    }
    adapted = {e["reason_code"]: _event(e, {}, []) for e in events}
    assert adapted["exact_match_protected"]["exact_basis"]["status"] == "not_recorded"
    assert adapted["exact_source_prefiltered"]["outcome"] == "rejected"
    assert adapted["exact_source_prefiltered"]["exact_basis"]["status"] == "not_recorded"
    assert adapted["prefilter_retained"]["exact_basis"]["status"] == "not_applicable"
    dataset.filter_exact_matches = False
    observe_dataset(dataset)
    disabled = next(
        e
        for v in dataset._candidate_stage_observations.values()
        for e in v["candidate_decision"]["events"]
        if e["stage"] == "prefilter"
    )
    assert exact_match_basis(disabled)["status"] == "not_applicable"
    assert _missing_event("prefilter")["exact_basis"]["status"] == "not_recorded"
    # Identical text and scores never promote a historic record to a saved rule.
    assert (
        exact_match_basis(
            {
                "stage": "prefilter",
                "status": "completed",
                "reason_code": "exact_match_protected",
                "values": {"S_final": 1.0},
            }
        )["predicate"]
        is None
    )


def test_saved_basis_requires_precise_inputs_and_typed_pair_scope():
    saved = {
        "status": "recorded",
        "predicate": "candidate_token_key_equality",
        "normalization_rule": "candidate_token_key/v1",
        "comparison_key": "alpha beta",
        "source": {"entity": SOURCE, "value": "Alpha beta", "fact_ids": ["s-label"]},
        "target": {"entity": TARGET, "value": "beta Alpha", "fact_ids": ["t-label"]},
        "artifact_refs": ["sha256:producer"],
        "reason_code": "saved_exact_rule_inputs",
    }
    event = {
        "stage": "prefilter",
        "reason_code": "exact_match_protected",
        "status": "completed",
        "exact_basis": saved,
    }
    assert exact_match_basis(event, source=SOURCE, target=TARGET)["comparison_key"] == "alpha beta"
    with pytest.raises(DomainError):
        exact_match_basis(event, source=TARGET, target=SOURCE)
    malformed = copy.deepcopy(event)
    del malformed["exact_basis"]["source"]
    with pytest.raises(DomainError):
        exact_match_basis(malformed)


def test_counts_preserve_source_only_unresolved_withheld_and_unsupported_list_links():
    record, items = features()
    for item in items:
        SelectedEvidence.model_validate(item)
    metadata = pair_evidence_metadata(record, items, SOURCE, TARGET, ["artifact"])
    assert metadata["counts"] == {
        "selected": 4,
        "resolved": 2,
        "unresolved": 1,
        "not_exported": 1,
        "withheld": 0,
        "graph_supported": 0,
        "graph_unsupported": 1,
        "graph_not_recorded": 3,
    }
    assert metadata["bridge_status"] == "not_recorded" and metadata["bridges"] == []

    class Context:
        def axiom(self, fid, policy):
            if fid == "hidden":
                raise DomainError("withheld", "Hidden original", 404)
            return {"id": fid}

    safe = filter_evidence_items(items, {"ov": Context()}, VisibilityPolicy())
    assert len(safe) == 4
    for item in safe:
        SelectedEvidence.model_validate(item)
    counts = filter_pair_metadata(metadata, safe)["counts"]
    assert counts["resolved"] == 1 and counts["withheld"] == 1
    assert counts["unresolved"] == 1 and counts["not_exported"] == 1
    assert safe[0]["graph_projection"] == "unsupported" and safe[0]["fact_ids"] == ["visible"]
    assert "private legacy text" not in json.dumps(safe)
    assert safe[-1]["fact_ids"] == [] and safe[-1]["feature_id"] is None
    assert PairEvidenceMetadata.model_validate(filter_pair_metadata(metadata, safe))


def test_no_selected_unexported_and_request_failure_are_not_one_empty_state():
    assert (
        pair_evidence_metadata({"attributes": {}}, [], SOURCE, TARGET, [])["status"]
        == "none_selected"
    )
    assert pair_evidence_metadata({}, [], SOURCE, TARGET, [])["status"] == "not_exported"
    assert (
        pair_evidence_metadata({"evidence_status": "failed"}, [], SOURCE, TARGET, [])["status"]
        == "request_failed"
    )
    _, items = features()

    class FailedContext:
        def axiom(self, fid, policy):
            raise DomainError("artifact_conflict", "Modified prepared facts", 409)

    with pytest.raises(DomainError) as error:
        filter_evidence_items(items, {"ov": FailedContext()}, VisibilityPolicy())
    assert error.value.status_code == 409


def test_only_explicit_recorded_bridges_can_connect_both_sides_and_punned_entities():
    record, items = features()
    assert recorded_bridges(record, items, SOURCE, TARGET, ["artifact"]) == []
    bridge = {
        "level": "channel_comparison",
        "channel": "attributes",
        "source": SOURCE,
        "target": TARGET,
    }
    record["evidence_bridges"] = [bridge]
    assert (
        recorded_bridges(record, items, SOURCE, TARGET, ["artifact"])[0]["level"]
        == "channel_comparison"
    )
    bridge.update(
        level="feature_correspondence",
        source_evidence_ids=[items[0]["evidence_id"]],
        target_evidence_ids=[items[-1]["evidence_id"]],
        source_fact_ids=["visible"],
        target_fact_ids=["hidden"],
    )
    explicit = recorded_bridges(record, items, SOURCE, TARGET, ["artifact"])[0]
    assert explicit["source"]["iri"] == explicit["target"]["iri"]
    assert explicit["source"]["kind"] != explicit["target"]["kind"]
    bridge["target_evidence_ids"] = [items[0]["evidence_id"]]
    with pytest.raises(DomainError):
        recorded_bridges(record, items, SOURCE, TARGET, ["artifact"])


def test_real_http_lists_all_selected_statuses_with_scoped_original_links(
    prepared, tmp_path
):  # noqa: F811
    _, package, _, _ = prepared
    manifest = validate_bundle(package)
    oid, locator = next(iter(manifest.ontologies.items()))
    context = OntologyContext(package.parent / locator)
    entity = {"ontology_version_id": oid, "iri": "urn:A", "kind": "class"}
    fact = context.facts(entity, category="definitions")["items"][0]
    record = {
        "src_iri": "urn:A",
        "tgt_iri": "urn:B",
        "src_kind": "class",
        "tgt_kind": "class",
        "attributes": {
            "source": [
                {
                    "property_iri": fact["predicate_iri"],
                    "value": fact["value"]["lexical_form"],
                    "datatype": fact["value"]["datatype"],
                    "language": fact["value"]["language"],
                    "entity_iri": "urn:A",
                },
                {"property_iri": "urn:missing", "value": "no original"},
                {"text": "PRIVATE UNMAPPED TEXT"},
            ]
        },
    }
    run = tmp_path / "metadata-run"
    ExplanationStore.create(run).append([record])
    import_run(
        run,
        package.parent / "decisions",
        source_ontology_version_id=oid,
        target_ontology_version_id=oid,
        run_id="metadata",
        evidence_resolver=context.resolve_feature,
    )
    metadata = manifest.model_dump(exclude={"package_id", "artifacts"})
    metadata["runs"] = {"metadata": "decisions"}
    package = publish_bundle(package.parent, **metadata)
    pair_id = DecisionStore(package.parent / "decisions").candidates("urn:A")["items"][0]["pair_id"]
    with TestClient(create_prepared_app(package)) as client:
        response = client.get(
            "/api/v1/runs/metadata/pair-evidence", params={"pair_id": pair_id, "limit": 1}
        )
        assert response.status_code == 200, response.text
        assert response.json()["total_count"] == 3
        items = response.json()["items"]
        while response.json()["next_cursor"]:
            response = client.get(
                "/api/v1/runs/metadata/pair-evidence",
                params={"pair_id": pair_id, "limit": 1, "cursor": response.json()["next_cursor"]},
            )
            assert response.status_code == 200
            items.extend(response.json()["items"])
        assert {item["origin_resolution"] for item in items} == {
            "resolved",
            "unresolved",
            "not_exported",
        }
        assert "PRIVATE UNMAPPED TEXT" not in json.dumps(items)
        trace = client.get("/api/v1/runs/metadata/pair", params={"pair_id": pair_id}).json()
        assert trace["evidence_metadata"]["counts"]["selected"] == 3
        assert trace["evidence_metadata"]["bridge_status"] == "not_recorded"


def test_additive_description_contract_preserves_legacy_generation_manifest_hash():
    original = {
        "ontology_context_hashes": ["context"],
        "visibility_policy_hash": "policy",
        "packet_hash": "packet",
        "prompt_hash": "prompt",
        "output_schema": "schema",
        "requested_model": "saved-model",
        "returned_model": "saved-model",
        "provider": None,
        "parameters_hash": "parameters",
        "language": "en",
        "response_hash": "response",
        "status": "validated",
    }
    assert GenerationManifest.model_validate(original).model_dump(mode="json") == original
    extended = {**original, "description_contract": "exact-description/2"}
    assert GenerationManifest.model_validate(extended).model_dump(mode="json") == extended


def test_origin_missingness_does_not_erase_recorded_projection_limitation():
    _, items = features()
    items[1]["graph_projection"] = "unsupported"
    safe = filter_evidence_items(items, {}, VisibilityPolicy())
    assert safe[1]["origin_resolution"] == "unresolved"
    assert safe[1]["graph_projection"] == "unsupported"
    assert evidence_completeness(safe)["graph_unsupported"] == 2
