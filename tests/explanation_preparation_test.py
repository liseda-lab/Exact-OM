"""End-to-end preparation, offline API and selective repair over real native tiny OWL inputs."""

import json
import shutil
import sqlite3
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from exact_inspect.artifacts import atomic_json
from exact_inspect.contracts import DomainError, file_hash
from exact_inspect.generation import ExplanationJobs
from exact_inspect.preparation import (
    ExecutionLock,
    InputBinding,
    OntologyBinding,
    Preparation,
)
from exact_inspect.service import create_prepared_app
from exact_inspect.verification import verify_backend


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = tmp_path / "source.ofn"
    source.write_text(
        """Ontology(<urn:fixture>
Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>))
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "A")
AnnotationAssertion(<http://purl.obolibrary.org/obo/IAO_0000115> <urn:A> "Definition A")
AnnotationAssertion(<http://www.geneontology.org/formats/oboInOwl#hasDbXref> <urn:A> "forbidden-target")
SubClassOf(<urn:A> <urn:B>)
)"""
    )
    binding = OntologyBinding(
        name="tiny",
        root=InputBinding(path=str(source), sha256=file_hash(source), size=source.stat().st_size),
    )
    lock = ExecutionLock(
        design_revision="synthetic-native/1",
        ontologies=[binding],
        runtime={"pyowl-core": "0.2.1"},
        profile={"name": "fixture", "model": "fixture/model"},
        entities=[
            {"id": "a", "ontology": "tiny", "iri": "urn:A"},
            {"id": "b", "ontology": "tiny", "iri": "urn:B"},
        ],
        pairs=[{"source": "a", "target": "b"}],
    )
    calls = []

    def provider(self, messages, profile):
        packet = json.loads(messages[1]["content"])
        calls.append(packet)
        facts = [f for f in packet["facts"] if (f.get("value") or {}).get("term_type") == "literal"]
        claims = [
            {"text": f["value"]["lexical_form"], "fact_ids": [f["fact_id"]], "category": "key_fact"}
            for f in facts[:2]
        ]
        content = {"claims": claims, "limitations": [], "relation": None}
        return {
            "model": profile.model,
            "provider": "fixture",
            "choices": [{"message": {"content": json.dumps(content)}}],
        }

    monkeypatch.setattr(ExplanationJobs, "_provider", provider)
    output = tmp_path / "prepared"
    report = Preparation(lock, output, input_root=tmp_path).run()
    package = output / report["outputs"]["portable-export"] / "package.json"
    return lock, package, output, calls


def test_prepare_profiles_comparison_portable_reads_and_no_runtime_work(prepared, tmp_path):
    lock, package, output, calls = prepared
    assert len(calls) == 3
    assert all("forbidden-target" not in json.dumps(p) for p in calls)
    client = TestClient(create_prepared_app(package))
    ontology = client.get("/api/v1/ontologies").json()["items"][0]["ontology_version_id"]
    params = {"ontology_version_id": ontology, "iri": "urn:A", "kind": "class"}
    summary = client.get("/api/v1/entity-context", params=params)
    assert summary.status_code == 200, summary.text
    assert summary.json()["definitions"]["items"][0]["value"]["lexical_form"] == "Definition A"
    assert "forbidden-target" not in summary.text
    page = client.get(
        "/api/v1/entities", params={"ontology_version_id": ontology, "limit": 1}
    ).json()
    assert page["next_cursor"]
    stale = client.get(
        "/api/v1/entities",
        params={"ontology_version_id": ontology, "term": "A", "cursor": page["next_cursor"]},
    )
    assert stale.status_code == 409
    assert (
        client.get(
            "/api/v1/entities", params={"ontology_version_id": ontology, "limit": 101}
        ).status_code
        == 422
    )
    report = verify_backend(package, tmp_path / "verification")
    assert report["checks"]["bounded_reads"] == "passed"
    assert len(calls) == 3
    copied = tmp_path / "portable"
    shutil.copytree(package.parent, copied)
    shutil.rmtree(output)
    code = """import sys
from pathlib import Path
from fastapi.testclient import TestClient
from exact_inspect.service import create_prepared_app
c=TestClient(create_prepared_app(Path(sys.argv[1])))
o=c.get('/api/v1/ontologies').json()['items'][0]['ontology_version_id']
r=c.get('/api/v1/entity-context',params={'ontology_version_id':o,'iri':'urn:A'})
assert r.status_code == 200, r.text
assert not any(m in sys.modules for m in ['torch','pyowl_core','transformers','sentence_transformers'])
"""
    subprocess.run([sys.executable, "-c", code, str(copied / "package.json")], check=True)


def test_outer_manifest_prevents_inner_manifest_and_database_tampering(prepared):
    _, package, _, _ = prepared
    manifest = json.loads(package.read_bytes())
    ontology, locator = next(iter(manifest["ontologies"].items()))
    root = package.parent / locator
    db = root / "context.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute("DELETE FROM terms")
    inner = json.loads((root / "manifest.json").read_bytes())
    inner["artifacts"]["context.sqlite"] = file_hash(db)
    atomic_json(root / "manifest.json", inner)
    # Even if an attacker updates inner integrity metadata, outer content binding fails.
    with pytest.raises(DomainError):
        client = TestClient(create_prepared_app(package))
        client.get("/api/v1/entities", params={"ontology_version_id": ontology}).raise_for_status()


def test_policy_and_prompt_repairs_reuse_verified_upstream_stages(prepared, tmp_path):
    lock, _, old, calls = prepared
    changed = lock.model_copy(update={"profile": {"name": "fixture", "model": "fixture/stronger"}})
    repaired = tmp_path / "repaired"
    result = Preparation(changed, repaired, input_root=tmp_path, resume_from=old).run()
    before = json.loads((old / "preparation.json").read_bytes())
    assert result["outputs"]["context-index"] == before["outputs"]["context-index"]
    assert result["outputs"]["profiles"] != before["outputs"]["profiles"]
    assert result["outputs"]["portable-export"] != before["outputs"]["portable-export"]
    assert len(calls) == 6


def test_demo_requires_explicit_development_admission(prepared):
    _, package, _, _ = prepared
    with pytest.raises(ValueError, match="development"):
        create_prepared_app(package, profile="public_demo")


def test_study_adapter_preserves_originals_and_rejects_forged_comparisons(prepared):
    from exact_inspect.artifacts import validate_bundle
    from exact_inspect.context import OntologyContext
    from exact_inspect.generation import FactPacket, comparison_templates
    from exact_inspect.study.builder import build_explanation_resource
    from exact_inspect.study.resources import ExplanationResource

    lock, package, output, _ = prepared
    manifest = validate_bundle(package)
    contexts = {
        identity: OntologyContext(package.parent / locator)
        for identity, locator in manifest.ontologies.items()
    }
    packets, results = [], []
    for request in (output / "generations").glob("*/request.json"):
        packet = FactPacket.model_validate(json.loads(request.read_bytes())["packet"])
        result = json.loads((request.parent / "explanation.json").read_bytes())
        if packet.task == "pair_comparison":
            result["claims"] = [claim.model_dump() for claim in comparison_templates(packet)]
        packets.append(packet)
        results.append(result)
    entities = next(p.entities for p in packets if p.task == "pair_comparison")
    resource = build_explanation_resource(
        contexts, entities, policy=lock.policy, packets=packets, explanations=results
    )
    assert resource.pair_comparison
    assert resource.hierarchy
    assert "forbidden-target" not in resource.model_dump_json()
    assert any(f.value.term_type == "expression_ref" and f.value.ast for f in resource.facts)
    assert any(f.value.lexical_form == "Definition A" for f in resource.facts)
    assert ExplanationResource.model_validate_json(resource.model_dump_json()) == resource
    forged = resource.model_dump()
    forged["pair_comparison"][0]["text"] = "The source and target are equivalent."
    with pytest.raises(ValueError, match="grounded semantic template"):
        ExplanationResource.model_validate(forged)


def test_bind_lock_pins_bytes_without_parsing_and_rejects_changed_input(tmp_path):
    from exact_inspect.cli import main
    from exact_inspect.preparation import read_lock

    source = tmp_path / "source.ofn"
    source.write_text("Ontology()")
    template = tmp_path / "bindings.json"
    template.write_text(
        json.dumps(
            {
                "design_revision": "binding-test/1",
                "ontologies": [{"name": "tiny", "root": {"path": "source.ofn"}}],
            }
        )
    )
    lock_path = tmp_path / "lock.json"
    assert main(["bind-lock", "--template", str(template), "--output", str(lock_path)]) == 0
    lock = read_lock(lock_path)
    assert lock.ontologies[0].root.sha256 == file_hash(source)
    assert lock.runtime_hashes["pyowl-core"].startswith("sha256:")
    assert not list(tmp_path.glob("**/context.sqlite"))
    source.write_text("Ontology(Declaration(Class(<urn:changed>)))")
    with pytest.raises(ValueError, match="immutable"):
        main(["bind-lock", "--template", str(template), "--output", str(lock_path)])


def test_verified_context_reuse_never_reparses_and_checks_binding(prepared, tmp_path, monkeypatch):
    import pyowl_core

    from exact_inspect.artifacts import validate_bundle

    lock, package, preparation_root, _ = prepared
    validate_bundle(package)
    report = json.loads((preparation_root / "preparation.json").read_bytes())
    context_manifest = (
        preparation_root / report["outputs"]["context-index"] / "ontology-0" / "manifest.json"
    )
    reusable = InputBinding(
        path=str(context_manifest),
        sha256=file_hash(context_manifest),
        size=context_manifest.stat().st_size,
    )
    bound = lock.model_copy(
        update={
            "ontologies": [lock.ontologies[0].model_copy(update={"prepared_context": reusable})]
        }
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A verified prepared context must not invoke an ontology parser")

    monkeypatch.setattr(pyowl_core, "load_snapshot", forbidden)
    report = Preparation(bound, tmp_path / "reuse-context", input_root=tmp_path).run(
        ["context-index"]
    )
    assert "context-index" in report["outputs"]
    wrong_scope = bound.model_copy(
        update={"ontologies": [bound.ontologies[0].model_copy(update={"scope": "closure"})]}
    )
    with pytest.raises(DomainError, match="differs from execution inputs"):
        Preparation(wrong_scope, tmp_path / "wrong-context", input_root=tmp_path).run(
            ["context-index"]
        )


def test_prepared_api_rejects_wrong_cursor_key_types_and_streams_filtered_axioms(prepared):
    from exact_inspect.contracts import encode_cursor

    _, package, _, _ = prepared
    client = TestClient(create_prepared_app(package))
    page = client.get("/api/v1/ontologies").json()
    for key in (["x"], 1, False, []):
        response = client.get(
            "/api/v1/ontologies", params={"cursor": encode_cursor(page["scope"], key)}
        )
        assert response.status_code == 422
        assert response.json()["code"] == "invalid_cursor"
    ontology = page["items"][0]["ontology_version_id"]
    fact = client.get(
        "/api/v1/entity-facts",
        params={
            "ontology_version_id": ontology,
            "iri": "urn:A",
            "kind": "class",
            "category": "definitions",
        },
    ).json()["items"][0]
    response = client.get(
        "/api/v1/axioms/" + fact["axiom_ref"] + "/stream", params={"ontology_version_id": ontology}
    )
    assert response.status_code == 200
    assert response.json()["axiom_id"] == fact["axiom_ref"]
    assert "forbidden-target" not in response.text
