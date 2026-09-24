"""Typed read contracts preserve real native fact links and reject false semantics."""

import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from exact.runs import ExplanationStore, RunLayout
from exact_inspect.artifacts import publish_bundle, validate_bundle
from exact_inspect.client import InspectClient
from exact_inspect.context import OntologyContext
from exact_inspect.contracts import EntityRef
from exact_inspect.decisions import DecisionStore, import_run
from exact_inspect.models import AxiomResponse, Candidate, SelectedEvidence
from exact_inspect.service import create_prepared_app
from tests.explanation_preparation_test import prepared  # noqa: F401,F811


@pytest.fixture
def read_package(prepared, tmp_path):  # noqa: F811
    _, package, _, _ = prepared
    manifest = validate_bundle(package)
    ontology_id, locator = next(iter(manifest.ontologies.items()))
    context = OntologyContext(package.parent / locator)
    entity = EntityRef(ontology_version_id=ontology_id, iri="urn:A", kind="class")
    fact = context.facts(entity, category="definitions")["items"][0]
    axiom = context.axiom(fact["axiom_ref"])
    run = tmp_path / "original-run"
    ExplanationStore.create(run).append(
        [
            {
                "src_iri": "urn:A",
                "tgt_iri": "urn:B",
                "src_kind": "class",
                "tgt_kind": "class",
                "confidences": {"S_final": 0.9},
                "attributes": {
                    "source": [
                        {
                            "property_iri": fact["predicate_iri"],
                            "entity_iri": "urn:A",
                            "value": fact["value"]["lexical_form"],
                            "datatype": fact["value"]["datatype"],
                            "language": fact["value"]["language"],
                            "support": 0.75,
                            "source_axiom_refs": [axiom["original_axiom_digest"]],
                            "interpretation": "asserted",
                        }
                    ]
                },
            }
        ]
    )
    RunLayout.open(run).mapping_path("global").write_text(
        "SrcEntity\tTgtEntity\tScore\tRelation\tSrcKind\tTgtKind\nurn:A\turn:B\t0.9\t=\tclass\tclass\n"
    )
    import_run(
        run,
        package.parent / "decisions",
        source_ontology_version_id=ontology_id,
        target_ontology_version_id=ontology_id,
        run_id="fixture-run",
        evidence_resolver=context.resolve_feature,
    )
    metadata = manifest.model_dump(exclude={"package_id", "artifacts"})
    metadata["runs"] = {"fixture-run": "decisions"}
    package = publish_bundle(package.parent, **metadata)
    with TestClient(create_prepared_app(package)) as api:
        yield api, DecisionStore(package.parent / "decisions"), context, entity, manifest


def test_real_shaped_models_validate_identity_links_and_reject_semantic_lies(read_package):
    _, store, context, entity, _ = read_package
    candidate = store.candidates(entity.iri)["items"][0]
    validated = Candidate.model_validate(candidate)
    assert "values" not in validated.model_dump(mode="json")
    assert validated.saved_alignment_member is True
    assert validated.scores[0].meaning
    bad = copy.deepcopy(candidate)
    bad["scores"][0]["name"] = "candidate_joint_rank"
    with pytest.raises(ValidationError, match="Ordinal"):
        Candidate.model_validate(bad)
    bad = copy.deepcopy(candidate)
    bad["membership_provenance"]["artifact_hash"] = None
    with pytest.raises(ValidationError, match="membership"):
        Candidate.model_validate(bad)
    evidence = store.evidence(candidate["pair_id"])[0]
    validated_evidence = SelectedEvidence.model_validate(evidence)
    assert validated_evidence.entity.ontology_version_id == context.ontology_version_id
    original = context.axiom(validated_evidence.fact_ids[0])
    assert original["original_axiom_digest"] in validated_evidence.source_axiom_refs
    assert AxiomResponse.model_validate(original).ast.type == "AnnotationAssertion"
    with pytest.raises(ValidationError, match="original-fact"):
        SelectedEvidence.model_validate({**evidence, "fact_ids": []})
    with pytest.raises(ValidationError, match="same stored assertion"):
        AxiomResponse.model_validate({**original, "axiom_id": "different"})
    with pytest.raises(ValidationError, match="availability"):
        AxiomResponse.model_validate({**original, "original_syntax": None})


def test_typed_fixture_client_covers_fact_hierarchy_candidate_evidence_and_text_routes(
    read_package,
):
    api, _, _, entity, manifest = read_package

    def request(request):
        return api.request(request.method, request.url.path, params=request.url.params)

    client = InspectClient("http://testserver", transport=httpx.MockTransport(request))
    try:
        facts = client.facts(entity, category="definitions")
        assert facts.items[0].value.lexical_form == "Definition A"
        assert client.hierarchy(entity).items[0]["parent"]["iri"] == "urn:B"
        candidates = client.candidates("fixture-run", entity.iri)
        candidate = candidates.items[0]
        assert candidate.saved_alignment_member is True
        evidence = client.evidence("fixture-run", candidate.pair_id)
        assert evidence.items[0].fact_ids == [facts.items[0].fact_id]
        assert client.axiom(entity.ontology_version_id, facts.items[0].axiom_ref).original_syntax
        explanation = client.explanation(next(iter(manifest.explanations)))
        assert explanation.grounding_status == "validated"
        wire = api.get("/api/v1/runs/fixture-run/candidates", params={"source": entity.iri}).json()
        assert "values" not in wire["items"][0]
        assert "ground_truth" not in json.dumps(wire)
        schema = api.get("/openapi.json").json()
        for route, expected in (
            ("/api/v1/runs/{run_id}/candidates", "Candidate"),
            ("/api/v1/runs/{run_id}/pair-evidence", "SelectedEvidence"),
            ("/api/v1/axioms/{axiom_id}", "AxiomResponse"),
        ):
            response = schema["paths"][route]["get"]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]
            assert expected in response["$ref"]
        assert "values" not in schema["components"]["schemas"]["Candidate"]["properties"]
    finally:
        client.close()


def test_invalid_prepared_candidate_has_safe_contract_error(read_package, monkeypatch):
    api, _, _, entity, _ = read_package
    store = api.app.state.service.run("fixture-run")
    page = store.candidates(entity.iri)
    page["items"][0]["membership_provenance"]["artifact_hash"] = None
    monkeypatch.setattr(store, "candidates", lambda *args, **kwargs: page)
    response = api.get("/api/v1/runs/fixture-run/candidates", params={"source": entity.iri})
    assert response.status_code == 409
    assert response.json()["code"] == "invalid_prepared_resource"
    assert "artifact_hash" not in response.text
