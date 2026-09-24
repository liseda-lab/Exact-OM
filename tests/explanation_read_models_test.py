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


def test_saved_pair_reads_without_matching_or_preparation_dependencies(read_package):
    import subprocess
    import sys

    _, store, _, _, _ = read_package
    package = store.directory.parent / "package.json"
    code = r"""
import importlib.abc
import sys
from pathlib import Path
class ServingOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"exact", "pandas", "torch", "pyowl_core", "transformers", "sentence_transformers", "openai"}:
            raise AssertionError("Read route imported an offline dependency: " + fullname)
sys.meta_path.insert(0, ServingOnly())
from fastapi.testclient import TestClient
from exact_inspect.service import create_prepared_app
with TestClient(create_prepared_app(Path(sys.argv[1]))) as client:
    sources = client.get("/api/v1/runs/fixture-run/sources")
    assert sources.status_code == 200, sources.text
    candidates = client.get("/api/v1/runs/fixture-run/candidates", params={"source":"urn:A"})
    assert candidates.status_code == 200, candidates.text
    pair = candidates.json()["items"][0]["pair_id"]
    for route in ("pair", "pair-evidence"):
        response = client.get("/api/v1/runs/fixture-run/" + route, params={"pair_id":pair})
        assert response.status_code == 200, response.text
    assert not any(name in sys.modules for name in ("exact", "pandas", "torch", "pyowl_core", "openai"))
"""
    subprocess.run([sys.executable, "-c", code, str(package)], check=True)


def test_ontology_listing_reads_verified_metadata_without_constructing_indexes(
    prepared, monkeypatch  # noqa: F811
):
    _, package, _, _ = prepared
    bundle = validate_bundle(package)
    ontology_id, locator = next(iter(bundle.ontologies.items()))
    nested = package.parent / locator / "manifest.json"
    metadata = json.loads(nested.read_bytes())
    metadata["source_derivation"] = {
        "status": "declared_derivative",
        "receipt_hash": "sha256:" + "a" * 64,
        "original_source_sha256": "sha256:" + "b" * 64,
        "path": "/private/source.owl",
        "private_notes": "not public",
    }
    metadata["capabilities"]["private_path"] = "/private/parser-cache"
    nested.write_text(json.dumps(metadata))
    package = publish_bundle(
        package.parent, **bundle.model_dump(exclude={"package_id", "artifacts"})
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Ontology listing must not construct an index reader")

    monkeypatch.setattr("exact_inspect.context.OntologyContext", forbidden)
    monkeypatch.setattr("exact_inspect.decisions.DecisionStore", forbidden)
    app = create_prepared_app(package)
    with TestClient(app) as client:
        for _ in range(2):
            response = client.get("/api/v1/ontologies")
            assert response.status_code == 200, response.text
            item = response.json()["items"][0]
            assert item["ontology_version_id"] == ontology_id
            assert item["name"] == metadata["name"]
            assert item["scope"] == metadata["scope"]
            assert item["capabilities"]["typed_expressions"] is True
            assert item["capabilities"]["reasoner_inferred"]["status"] == "not_run"
            assert item["completeness"] == metadata["completeness"]
            assert item["source_root_sha256"] == metadata["identity"]["root_sha256"]
            assert item["source_derivation"] == {
                key: metadata["source_derivation"][key]
                for key in ("status", "receipt_hash", "original_source_sha256")
            }
            assert "/private/" not in response.text and "parser" not in response.text
            assert "options" not in response.text and "private_notes" not in response.text
        assert app.state.service._contexts == app.state.service._runs == {}
        # Cached summaries still verify the nested manifest's immutable file identity.
        nested.write_text(nested.read_text() + " ")
        changed = client.get("/api/v1/ontologies")
        assert changed.status_code == 409
        assert changed.json()["code"] == "corrupt_artifact"


def test_ontology_listing_enforces_the_shared_response_byte_budget(
    prepared, monkeypatch  # noqa: F811
):
    _, package, _, _ = prepared
    app = create_prepared_app(package)
    monkeypatch.setattr(
        app.state.service,
        "ontology_metadata",
        lambda key: {"ontology_version_id": key, "name": "x" * (2 * 1024**2)},
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/ontologies")
    assert response.status_code == 413
    assert response.json()["code"] == "response_too_large"
