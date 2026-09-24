"""Physical portable-policy boundaries, including SQLite pages and canonical originals."""

from __future__ import annotations

import base64
import json

import pyowl_core as core
import pytest

from exact_inspect.context import build_context_package
from exact_inspect.context_export import export_policy_context
from exact_inspect.context_semantics import decode_payload
from exact_inspect.contracts import DomainError, EntityRef, VisibilityPolicy, file_hash

OWL = b"""Ontology(<urn:policy-fixture>
Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>))
AnnotationAssertion(Annotation(<http://www.geneontology.org/formats/oboInOwl#hasDbXref> "forbidden-nested-label") <http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "Visible label")
AnnotationAssertion(Annotation(<http://www.w3.org/2000/01/rdf-schema#comment> "forbidden-nested-comment") <http://purl.obolibrary.org/obo/IAO_0000115> <urn:A> "Visible definition")
AnnotationAssertion(<http://www.geneontology.org/formats/oboInOwl#hasDbXref> <urn:A> "forbidden-target")
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#comment> <urn:A> "forbidden-comment")
SubClassOf(Annotation(<http://www.geneontology.org/formats/oboInOwl#hasDbXref> "forbidden-nested-logical") <urn:A> <urn:B>)
)"""


@pytest.fixture
def context(tmp_path):
    snapshot = core.load_snapshot(
        OWL,
        options=core.LoadOptions(imports=core.ImportPolicy.IGNORE, preserve_source_map=True),
    )
    original = build_context_package(snapshot, tmp_path / "original")
    policy = VisibilityPolicy(
        policy_id="portable-test",
        categories=tuple(
            category for category in VisibilityPolicy().categories if category != "comments"
        ),
    )
    return original, policy


def test_portable_context_removes_forbidden_rows_qualifiers_and_original_bytes(context, tmp_path):
    original, policy = context
    original_hash = file_hash(original.database)
    filtered = export_policy_context(original, tmp_path / "portable", policy)
    assert file_hash(original.database) == original_hash
    assert filtered.ontology_version_id == original.ontology_version_id
    assert file_hash(filtered.database) != original_hash
    assert b"forbidden-" not in filtered.database.read_bytes()
    assert b"forbidden-" not in (filtered.path / "manifest.json").read_bytes()
    assert {path.name for path in filtered.path.iterdir()} == {
        "context.sqlite",
        "manifest.json",
    }
    with filtered._connection() as connection:
        rows = list(connection.execute("SELECT * FROM axioms"))
        assert len(rows) == 5
        assert {row["category"] for row in rows} == {
            "axioms",
            "labels",
            "definitions",
            "hierarchy",
        }
        for row in rows:
            payload = decode_payload(row["payload"])
            assert "forbidden-" not in json.dumps(payload)
            assert "forbidden-" not in row["fact"]
            assert row["payload_bytes"] == len(
                json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            )
            if payload.get("original_syntax"):
                assert b"forbidden-" not in base64.b64decode(payload["original_syntax"])
            else:
                assert payload["original_availability"] == "not_exported"
        assert connection.execute("PRAGMA freelist_count").fetchone()[0] == 0
        assert all(
            row[0] not in {"xrefs", "comments"}
            for row in connection.execute("SELECT DISTINCT category FROM terms")
        )
    entity = EntityRef(ontology_version_id=filtered.ontology_version_id, iri="urn:A", kind="class")
    summary = filtered.entity_context(entity, policy=policy)
    assert summary["preferred_label"]["value"] == "Visible label"
    assert summary["definitions"]["items"][0]["value"]["lexical_form"] == "Visible definition"
    assert summary["completeness"]["extraction"] == "complete_for_policy"
    assert filtered.hierarchy(entity, policy=policy)["items"][0]["parent"]["iri"] == "urn:B"
    assert filtered.manifest["completeness"]["axiom_count"] == 5
    assert sum(filtered.manifest["completeness"]["category_counts"].values()) == 5
    assert filtered.manifest["policy_filter"]["policy_hash"] == policy.policy_hash
    assert export_policy_context(original, filtered.path, policy).manifest == filtered.manifest


def test_portable_context_cannot_be_relabelled_as_broader_policy(context, tmp_path):
    original, policy = context
    filtered = export_policy_context(original, tmp_path / "portable", policy)
    broader = VisibilityPolicy(categories=(*policy.categories, "xrefs"), allow_mapping_xrefs=True)
    with pytest.raises(DomainError, match="restore"):
        export_policy_context(filtered, tmp_path / "broader", broader)
    with pytest.raises(DomainError, match="different inputs"):
        export_policy_context(original, filtered.path, broader)
    excluded = VisibilityPolicy(ontology_ids=("another-ontology",))
    with pytest.raises(DomainError, match="outside"):
        export_policy_context(original, tmp_path / "excluded", excluded)


def test_original_fact_ids_survive_physical_filtering(context, tmp_path):
    original, policy = context
    filtered = export_policy_context(original, tmp_path / "portable", policy)
    entity = EntityRef(ontology_version_id=original.ontology_version_id, iri="urn:A", kind="class")
    original_facts = original.facts(entity, category="definitions", policy=policy)
    copied_facts = filtered.facts(entity, category="definitions", policy=policy)
    assert copied_facts["items"] == original_facts["items"]
    assert copied_facts["scope"]["context_revision"] != original_facts["scope"]["context_revision"]
    for fact in copied_facts["items"]:
        detail = filtered.axiom(fact["axiom_ref"], policy=policy)
        assert detail["original_axiom_digest"]
        assert detail["origins"]
        assert "forbidden-" not in json.dumps(detail)
