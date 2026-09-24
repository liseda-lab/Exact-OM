"""Behavioral context tests use the installed native parser, never a second OWL parser."""

import base64
import json
import shutil
import sqlite3
import subprocess
import sys

import pyowl_core as core
import pytest

from exact_inspect.context import (
    OntologyContext,
    build_context_package,
    expression_text,
    prepare_context,
    typed_node,
)
from exact_inspect.contracts import (
    DomainError,
    EntityRef,
    Page,
    VisibilityPolicy,
    file_hash,
)

OWL = b"""Ontology(<urn:test> Import(<urn:unresolved>)
Declaration(Class(<urn:A>)) Declaration(NamedIndividual(<urn:A>))
Declaration(Class(<urn:duplicate>)) Declaration(Class(<urn:sparse>))
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "Same label"@en)
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:duplicate> "Same label"@en)
AnnotationAssertion(Annotation(<urn:citation> "PMID:1") <http://purl.obolibrary.org/obo/IAO_0000115> <urn:A> "Exact definition"@en)
AnnotationAssertion(<http://www.geneontology.org/formats/oboInOwl#hasExactSynonym> <urn:A> "Exact synonym")
AnnotationAssertion(<urn:looks_like_definition> <urn:A> "Not a definition")
AnnotationAssertion(<http://www.geneontology.org/formats/oboInOwl#hasDbXref> <urn:A> "hidden-target")
SubClassOf(<urn:A> <urn:B>) SubClassOf(<urn:A> <urn:C>)
SubClassOf(<urn:B> <urn:D>) SubClassOf(<urn:C> <urn:D>) SubClassOf(<urn:D> <urn:A>)
EquivalentClasses(<urn:A> ObjectIntersectionOf(<urn:E> ObjectSomeValuesFrom(ObjectInverseOf(<urn:p>) ObjectComplementOf(<urn:F>))))
SubClassOf(<urn:A> ObjectAllValuesFrom(<urn:p> ObjectUnionOf(<urn:F> <urn:G>)))
SubClassOf(<urn:A> ObjectMinCardinality(2 <urn:p> <urn:F>))
SubClassOf(<urn:A> ObjectMaxCardinality(4 <urn:p> <urn:F>))
SubClassOf(<urn:A> ObjectExactCardinality(3 <urn:p> <urn:F>))
SubClassOf(<urn:A> ObjectHasValue(<urn:p> <urn:A>))
ObjectPropertyDomain(<urn:p> ObjectUnionOf(<urn:A> <urn:B>))
ObjectPropertyRange(<urn:p> <urn:C>) InverseObjectProperties(<urn:p> <urn:inverse>)
TransitiveObjectProperty(<urn:p>)
SubObjectPropertyOf(ObjectPropertyChain(<urn:p> <urn:p>) <urn:p>)
DataPropertyDomain(<urn:data> <urn:A>) DataPropertyRange(<urn:data> <http://www.w3.org/2001/XMLSchema#integer>)
ClassAssertion(<urn:C> <urn:A>)
ObjectPropertyAssertion(<urn:p> <urn:A> <urn:other>)
NegativeObjectPropertyAssertion(<urn:p> <urn:other> <urn:A>)
DataPropertyAssertion(<urn:data> <urn:A> "001"^^<http://www.w3.org/2001/XMLSchema#integer>)
)"""


@pytest.fixture
def context(tmp_path):
    snapshot = core.load_snapshot(
        OWL,
        options=core.LoadOptions(
            backend=core.BackendPreference.NATIVE,
            imports=core.ImportPolicy.IGNORE,
            preserve_source_map=True,
        ),
    )
    return build_context_package(snapshot, tmp_path / "context"), snapshot


def ref(context, iri="urn:A", kind="class"):
    return EntityRef(ontology_version_id=context.ontology_version_id, iri=iri, kind=kind)


def test_punning_duplicate_labels_exact_annotations_and_missingness(context):
    index, _ = context
    page = index.search(term="same", language="en")
    Page.model_validate(page)
    assert {(row["entity"]["iri"], row["entity"]["kind"]) for row in page["items"]} == {
        ("urn:A", "class"),
        ("urn:A", "individual"),
        ("urn:duplicate", "class"),
    }
    rich = index.entity_context(ref(index), language="fr")
    assert rich["preferred_label"]["language_fallback"] is True
    definition = rich["categories"]["definitions"]["items"][0]
    assert definition["value"]["lexical_form"] == "Exact definition"
    assert definition["value"]["language"] == "en"
    assert definition["qualifiers"][0]["value"]["lexical_form"] == "PMID:1"
    assert rich["categories"]["definitions"]["total_count"] == 1
    assert (
        rich["categories"]["annotations"]["items"][0]["value"]["lexical_form"] == "Not a definition"
    )
    assert rich["categories"]["synonyms"]["items"][0]["synonym_scope"] == "exact"
    sparse = index.entity_context(ref(index, "urn:sparse"))
    assert sparse["categories"]["definitions"]["status"] == "absent_in_scope"
    assert sparse["completeness"]["imports_complete"] is False
    assert sparse["alignment_eligible"] is False
    assert index.resolve_predicate("urn:imported-predicate")["status"] == "unresolved_import"


def test_lossless_original_identity_all_constructors_and_relocation(context, tmp_path):
    index, snapshot = context
    expected = {
        "sha256:" + core.structural_digest(axiom).hex(): core.canonical_bytes(axiom)
        for axiom in snapshot.iter_axioms()
    }
    with sqlite3.connect(index.database) as connection:
        ids = [row[0] for row in connection.execute("SELECT id FROM axioms")]
    actual = {
        index.axiom(identifier)["original_axiom_digest"]: base64.b64decode(
            index.axiom(identifier)["original_syntax"]
        )
        for identifier in ids
    }
    assert actual == expected
    for identifier in ids:
        axiom = index.axiom(identifier)
        assert axiom["origins"][0]["source_sha256"].startswith("sha256:")
        assert axiom["origins"][0]["span"] is not None
    copied = tmp_path / "relocated"
    shutil.copytree(index.path, copied)
    shutil.rmtree(index.path)
    reopened = OntologyContext(copied)
    assert reopened.ontology_version_id == index.ontology_version_id
    assert (
        reopened.facts(ref(reopened), category="definitions")["items"][0]["value"]["lexical_form"]
        == "Exact definition"
    )
    assert reopened.manifest["portable"]["browsing"] == ["manifest.json", "context.sqlite"]


def test_asserted_structural_hierarchy_cycles_and_edges_between_existing_nodes(context):
    index, _ = context
    asserted = index.hierarchy(ref(index))
    assert {edge["parent"]["iri"] for edge in asserted["items"]} == {"urn:B", "urn:C"}
    structural = index.hierarchy(ref(index), basis="structural_navigation")
    assert {edge["parent"]["iri"] for edge in structural["items"]} == {"urn:B", "urn:C", "urn:E"}
    inferred = next(edge for edge in structural["items"] if edge["parent"]["iri"] == "urn:E")
    assert inferred["interpretation"]["rule"] == "equivalent_conjunct"
    assert inferred["interpretation"]["premises"] == [inferred["axiom_id"]]
    ancestors = index.hierarchy(ref(index), direction="ancestors")
    assert len(ancestors["nodes"]) == 4
    assert len(ancestors["items"]) == 5  # Diamond's second incoming edge and cycle both survive.
    limited = index.hierarchy(ref(index), direction="ancestors", limit=2)
    assert limited["node_budget_reached"] and limited["status"] == "partial"
    assert index.hierarchy(ref(index), basis="reasoner_inferred")["status"] == "not_run"


def test_property_individual_profiles_and_semantic_literals(context):
    index, _ = context
    profile = index.entity_context(ref(index, "urn:p", "object_property"))
    assert profile["categories"]["domains"]["total_count"] == 1
    domain = index.axiom(profile["categories"]["domains"]["items"][0]["id"])
    assert domain["ast"]["domain"]["type"] == "ObjectUnionOf"
    assert profile["categories"]["characteristics"]["total_count"] >= 3
    individual = index.entity_context(ref(index, kind="individual"))
    assert individual["categories"]["types"]["total_count"] == 1
    assertions = individual["categories"]["assertions"]["items"]
    typed = [index.axiom(item["id"])["ast"] for item in assertions]
    data = next(item for item in typed if item["type"] == "DataPropertyAssertion")
    assert data["value"]["lexical_form"] == "001"
    assert data["value"]["datatype"]["iri"]["value"].endswith("#integer")
    assert len(assertions) == 3
    assert sum("outgoing" in item["assertion_directions"] for item in assertions) == 2
    assert sum("incoming" in item["assertion_directions"] for item in assertions) == 1
    assert individual["categories"]["types"]["items"][0]["reference_role"] == "asserted_type"


def test_keysets_policy_bounds_and_corruption(context):
    index, _ = context
    first = index.search(limit=2)
    second = index.search(limit=2, cursor=first["next_cursor"])
    assert (
        len({json.dumps(row["entity"], sort_keys=True) for row in first["items"] + second["items"]})
        == 4
    )
    with pytest.raises(DomainError) as error:
        index.search(term="same", cursor=first["next_cursor"])
    assert error.value.status_code == 409
    restricted = VisibilityPolicy(categories=("definitions",))
    assert index.search(term="same", policy=restricted)["total_count"] == 0
    assert index.search(term="hidden-target")["total_count"] == 0
    assert index.facts(ref(index), category="xrefs", policy=restricted)["total_count"] is None
    xref = index.facts(ref(index), category="xrefs")["items"][0]
    with pytest.raises(KeyError):
        index.axiom(xref["id"], policy=restricted)
    with pytest.raises(DomainError):
        index.search(cursor=first["next_cursor"], policy=restricted)
    with pytest.raises(ValueError):
        index.search(limit=101)
    with pytest.raises(KeyError):
        index.entity_context(ref(index, kind="data_property"))
    with index.database.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        OntologyContext(index.path)


def test_closure_origins_and_scope_identity(tmp_path):
    root = b"Ontology(<urn:r> Import(<urn:i>) SubClassOf(<urn:A> <urn:B>))"
    imported = b"Ontology(<urn:i> SubClassOf(<urn:A> <urn:B>) SubClassOf(<urn:B> <urn:C>))"
    snapshot = core.load_snapshot(
        root,
        options=core.LoadOptions(backend=core.BackendPreference.NATIVE, preserve_source_map=True),
        resolver=core.MappingResolver({"urn:i": imported}),
    )
    index = build_context_package(
        snapshot, tmp_path / "closure", scope="closure", matcher_scope="root"
    )
    limited = build_context_package(snapshot, tmp_path / "root", scope="root")
    assert index.ontology_version_id != limited.ontology_version_id
    assert index.manifest["context_extension"] is True
    edge = index.hierarchy(ref(index))["items"][0]
    assert len(index.axiom(edge["axiom_id"])["origins"]) == 2
    assert index.hierarchy(ref(index, "urn:B"))["total_count"] == 1
    assert limited.hierarchy(ref(limited, "urn:B"))["total_count"] == 0


def test_preparation_verifies_pins_and_strict_local_imports(tmp_path):
    source = tmp_path / "source.owl"
    source.write_bytes(b"Ontology(<urn:r> Import(<urn:i>) Declaration(Class(<urn:A>)))")
    imported = tmp_path / "import.owl"
    imported.write_bytes(b"Ontology(<urn:i> SubClassOf(<urn:A> <urn:B>))")
    with pytest.raises(ValueError, match="hash"):
        prepare_context(source, tmp_path / "bad", expected_hash="0" * 64)
    index = prepare_context(
        source,
        tmp_path / "ok",
        scope="closure",
        expected_hash=file_hash(source),
        imports={"urn:i": {"path": str(imported), "sha256": file_hash(imported)}},
    )
    assert index.manifest["completeness"]["imports_complete"]
    assert index.hierarchy(ref(index))["items"][0]["parent"]["iri"] == "urn:B"


def test_expression_templates_preserve_modality_and_unknowns():
    p = core.ObjectProperty(core.IRI("urn:may-have"))
    cls = core.Class(core.IRI("urn:A"))
    expressions = [
        core.ObjectSomeValuesFrom(p, cls),
        core.ObjectAllValuesFrom(p, cls),
        core.ObjectHasValue(p, core.NamedIndividual(core.IRI("urn:i"))),
        core.ObjectMinCardinality(2, p, cls),
        core.ObjectMaxCardinality(3, p, cls),
        core.ObjectExactCardinality(2, p, cls),
        core.ObjectInverseOf(p),
        core.ObjectComplementOf(cls),
        core.ObjectIntersectionOf(core.CanonicalSet([cls, core.Class(core.IRI("urn:B"))])),
    ]
    for expression in expressions:
        result = expression_text(typed_node(expression))
        assert result["status"] == "available"
    assert "only" in expression_text(typed_node(expressions[1]))["text"]
    assert expression_text({"type": "FutureConstructor", "data": [1, 2]})["status"] == "unsupported"


def test_nested_mapping_qualifiers_do_not_escape_policy(tmp_path):
    snapshot = core.load_snapshot(
        b"""Ontology(Declaration(Class(<urn:A>))
      AnnotationAssertion(Annotation(<http://www.geneontology.org/formats/oboInOwl#hasDbXref> "secret-answer")
        <http://purl.obolibrary.org/obo/IAO_0000115> <urn:A> "Visible definition"))"""
    )
    index = build_context_package(snapshot, tmp_path / "nested")
    policy = VisibilityPolicy(categories=("definitions",))
    page = index.facts(ref(index), category="definitions", policy=policy)
    assert page["items"][0]["qualifiers"] == []
    assert "secret-answer" not in json.dumps(page)
    detail = index.axiom(page["items"][0]["id"], policy=policy)
    assert "secret-answer" not in json.dumps(detail)
    assert detail["original_syntax"] is None
    assert detail["original_availability"] == "not_exported"


def test_imported_predicate_label_and_prepared_generation_packet(tmp_path):
    from exact_inspect.generation import entity_packet

    snapshot = core.load_snapshot(
        b"""Ontology(<urn:r> Import(<urn:i>) Declaration(Class(<urn:A>))
            AnnotationAssertion(<http://purl.obolibrary.org/obo/IAO_0000115> <urn:A> "Definition"))""",
        resolver=core.MappingResolver(
            {
                "urn:i": b"""Ontology(<urn:i>
            Declaration(AnnotationProperty(<urn:metadata>))
            AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:metadata> "Imported metadata"@en))"""
            }
        ),
    )
    index = build_context_package(snapshot, tmp_path / "imported", scope="closure")
    assert index.resolve_predicate("urn:metadata")["value"] == "Imported metadata"
    entity = ref(index)
    facts = index.facts(entity, category="definitions")["items"]
    packet = entity_packet(
        entity,
        facts,
        context_hash=index.manifest["artifacts"]["context.sqlite"],
        policy=VisibilityPolicy(),
    )
    assert packet.facts[0]["value"]["term_type"] == "literal"
    assert packet.facts[0]["predicate_iri"] == "http://purl.obolibrary.org/obo/IAO_0000115"


def test_request_paths_never_parse_and_verified_index_cannot_change(context, monkeypatch):
    index, _ = context

    def forbidden(*args, **kwargs):
        raise AssertionError("request attempted to parse ontology")

    monkeypatch.setattr(core, "load_snapshot", forbidden)
    index.entity_context(ref(index))
    index.search(term="Same")
    index.hierarchy(ref(index))
    reopened = OntologyContext(index.path)
    with index.database.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="changed"):
        reopened.search()


def test_resolve_exact_original_axioms_and_literal_terms(context):
    index, _ = context
    fact = index.facts(ref(index), category="definitions")["items"][0]
    original = index.axiom(fact["id"])
    by_identity = index.resolve_feature(
        {"source_axiom_refs": [original["original_axiom_digest"]]}, ref(index)
    )
    assert by_identity["fact_ids"] == [fact["id"]]
    by_terms = index.resolve_feature(
        {
            "property_iri": fact["predicate_iri"],
            "value": fact["value"]["lexical_form"],
            "datatype": fact["value"]["datatype"],
            "language": fact["value"]["language"],
        },
        ref(index),
    )
    assert by_terms["fact_ids"] == [fact["id"]]
    # Text alone, or wrong literal datatype, cannot invent asserted provenance.
    assert (
        index.resolve_feature({"value": "Exact definition"}, ref(index))["status"] == "not_exported"
    )


def test_checkpoint_resume_copies_committed_chunks_and_preserves_coverage(tmp_path):
    snapshot = core.load_snapshot(OWL, options=core.LoadOptions(imports=core.ImportPolicy.IGNORE))
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls == 7

    original = tmp_path / "original"
    with pytest.raises(InterruptedError):
        build_context_package(snapshot, original, checkpoint_interval=2, stop_requested=stop)
    assert not original.exists()
    staging = tmp_path / ".original.building"
    with sqlite3.connect(staging / "context.sqlite") as connection:
        assert (
            connection.execute("SELECT value FROM metadata WHERE key='axiom_count'").fetchone()[0]
            == "6"
        )
    relocated = tmp_path / "other" / "resumed"
    relocated.parent.mkdir()
    shutil.copytree(staging, relocated.parent / ".resumed.building")
    shutil.rmtree(staging)
    index = build_context_package(snapshot, relocated, checkpoint_interval=2)
    assert index.manifest["runtime"]["resumed_axiom_count"] == 6
    assert index.manifest["completeness"]["axiom_count"] == len(list(snapshot.iter_axioms()))
    assert (
        index.facts(ref(index), category="definitions")["items"][0]["value"]["lexical_form"]
        == "Exact definition"
    )
    with pytest.raises(ValueError, match="coverage"):
        build_context_package(snapshot, relocated, alignment_eligible={("urn:A", "class")})


def test_large_literals_are_bounded_but_original_axiom_stream_is_lossless(tmp_path):
    value = "large" * 500000
    snapshot = core.load_snapshot(
        (
            'Ontology(Declaration(Class(<urn:A>)) AnnotationAssertion(<http://purl.obolibrary.org/obo/IAO_0000115> <urn:A> "'
            + value
            + '"))'
        ).encode()
    )
    index = build_context_package(snapshot, tmp_path / "large")
    context = index.entity_context(ref(index))
    assert len(json.dumps(context).encode()) < 2 * 1024 * 1024
    page = context["definitions"]
    assert page["status"] == "partial"
    identifier = page["items"][0]["id"]
    with pytest.raises(DomainError) as error:
        index.axiom(identifier)
    assert error.value.status_code == 413
    original = json.loads(b"".join(index.axiom_chunks(identifier)))
    assert original["value"]["lexical_form"] == value
    assert base64.b64decode(original["original_syntax"]) in {
        core.canonical_bytes(axiom) for axiom in snapshot.iter_axioms()
    }


def test_policy_ontology_export_preserves_logic_and_filters_nested_annotations(context, tmp_path):
    from exact_inspect.context_resources import (
        export_ontology_resource,
        validate_ontology_resource,
    )

    index, original = context
    policy = VisibilityPolicy()
    path = tmp_path / "resource.ofn"
    receipt_path = export_ontology_resource(index, path, policy)
    receipt = validate_ontology_resource(
        path, receipt_path, policy_hash=policy.policy_hash, ontology_ids=[index.ontology_version_id]
    )
    assert receipt["logical_axioms_preserved"]
    assert "hidden-target" not in path.read_text()
    reloaded = core.load_snapshot(path)
    assert not reloaded.import_manifest.edges

    def logical(snapshot):
        return {
            core.canonical_bytes(axiom)
            for axiom in snapshot.iter_axioms()
            if type(axiom).__name__ not in {"AnnotationAssertion"}
        }

    assert logical(reloaded) == logical(original)
    path.write_text(path.read_text() + "tampered")
    with pytest.raises(ValueError, match="bytes"):
        validate_ontology_resource(path, receipt_path)


@pytest.mark.parametrize("use_workspace", [False, True])
def test_abrupt_writer_exit_releases_lock_and_rolls_back_only_inflight_chunk(
    tmp_path, use_workspace
):
    destination = tmp_path / "crashed"
    program = r"""
import os, sys
import pyowl_core as core
from exact_inspect.context import build_context_package
snapshot = core.load_snapshot(b"Ontology(" + b" ".join(("Declaration(Class(<urn:c%d>))" % i).encode() for i in range(10)) + b")")
count = 0
def crash():
    global count
    count += 1
    if count == 6:
        os._exit(7)
    return False
build_context_package(snapshot, sys.argv[1], checkpoint_interval=2, durable_interval=4,
    work_directory=sys.argv[2] or None, stop_requested=crash)
"""
    workspace = str(tmp_path / "scratch") if use_workspace else ""
    result = subprocess.run(
        [sys.executable, "-c", program, str(destination), workspace], check=False
    )
    assert result.returncode == 7
    assert not destination.exists()
    snapshot = core.load_snapshot(
        b"Ontology("
        + b" ".join(("Declaration(Class(<urn:c%d>))" % i).encode() for i in range(10))
        + b")"
    )
    index = build_context_package(
        snapshot,
        destination,
        checkpoint_interval=2,
        durable_interval=4,
        work_directory=workspace or None,
    )
    assert index.manifest["runtime"]["resumed_axiom_count"] == 4
    assert index.manifest["completeness"]["axiom_count"] == 10
    assert index.search()["total_count"] == 10


def test_authoritative_fact_rejects_entity_misattribution_and_opaque_ncit_maps(tmp_path):
    owl = b"""Ontology(Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>))
      AnnotationAssertion(Annotation(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P378> "NCI")
       Annotation(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P395> "hidden-code")
       <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P97> <urn:A> "Visible")
      AnnotationAssertion(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P207> <urn:A> "hidden-cui")
      AnnotationAssertion(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P375> <urn:A> "hidden-mapping"))"""
    index = build_context_package(core.load_snapshot(owl), tmp_path / "opaque")
    policy = VisibilityPolicy(categories=(*VisibilityPolicy().categories, "annotations"))
    page = index.facts(ref(index), category="definitions", policy=policy)
    fact = page["items"][0]
    assert index.fact(fact["id"], ref(index), policy=policy) == fact
    assert fact["qualifier_categories"] == [
        {
            "predicate_iri": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P378",
            "category": "definition_citations",
            "role": "source",
        }
    ]
    with pytest.raises(KeyError):
        index.fact(fact["id"], ref(index, "urn:B"), policy=policy)
    assert "hidden-" not in json.dumps(index.entity_context(ref(index), policy=policy))
    from exact_inspect.context_resources import export_ontology_resource

    resource = tmp_path / "opaque.ofn"
    export_ontology_resource(index, resource, policy)
    assert '"NCI"' in resource.read_text()
    assert "hidden-" not in resource.read_text()


def test_structural_hierarchy_respects_source_axiom_category_policy(context):
    index, _ = context
    policy = VisibilityPolicy(categories=("hierarchy",))
    page = index.hierarchy(ref(index), basis="structural_navigation", policy=policy)
    assert {edge["parent"]["iri"] for edge in page["items"]} == {"urn:B", "urn:C"}
    ancestors = index.hierarchy(
        ref(index), direction="ancestors", basis="structural_navigation", policy=policy
    )
    assert "urn:E" not in {item["iri"] for item in ancestors["nodes"]}


def test_compressed_axiom_detail_streams_losslessly_and_enforces_decoded_budget(context, tmp_path):
    from exact_inspect.context_semantics import (
        decode_payload,
        decoded_chunks,
        encode_payload,
    )

    index, _ = context
    fact = index.facts(ref(index), category="definitions")["items"][0]
    expected = index.axiom(fact["id"])
    copied = tmp_path / "compressed"
    shutil.copytree(index.path, copied)
    with sqlite3.connect(copied / "context.sqlite") as connection:
        connection.execute(
            "UPDATE axioms SET payload=? WHERE id=?", (encode_payload(expected), fact["id"])
        )
    manifest = json.loads((copied / "manifest.json").read_text())
    manifest["artifacts"]["context.sqlite"] = file_hash(copied / "context.sqlite")
    (copied / "manifest.json").write_text(json.dumps(manifest))
    reopened = OntologyContext(copied)
    assert reopened.axiom(fact["id"]) == expected
    assert json.loads(b"".join(reopened.axiom_chunks(fact["id"], chunk_bytes=7))) == expected
    huge = {"value": "abc" * 100000}
    encoded = encode_payload(huge)
    with pytest.raises(ValueError, match="budget"):
        decode_payload(encoded, max_bytes=100)
    blocks = list(decoded_chunks([encoded], compressed=True, chunk_bytes=1024))
    assert max(map(len, blocks)) <= 1024
    assert json.loads(b"".join(blocks)) == huge
    with pytest.raises(ValueError, match="compressed"):
        decode_payload(encoded[:-1])


def test_neighbor_feature_origins_and_projector_subclass_alias(context):
    index, _ = context
    edge = index.hierarchy(ref(index, "urn:B"))["items"][0]
    original = index.axiom(edge["axiom_id"])
    resolved = index.resolve_feature(
        {"source_axiom_refs": [original["original_axiom_digest"]]}, ref(index, "urn:sparse")
    )
    assert resolved["fact_ids"] == [edge["axiom_id"]]
    alias = index.resolve_feature(
        {"subject_iri": "urn:B", "object_iri": "urn:D", "rel_iri": "http://subclassof"}, ref(index)
    )
    assert alias["fact_ids"] == [edge["axiom_id"]]


def test_context_rejects_metadata_view_before_query(context):
    index, _ = context
    with sqlite3.connect(index.database) as connection:
        connection.execute("DROP TABLE metadata")
        connection.execute(
            "CREATE VIEW metadata AS SELECT 'ontology_version_id' AS key, "
            "randomblob(1000000000) AS value"
        )
    manifest = json.loads((index.path / "manifest.json").read_text())
    manifest["artifacts"]["context.sqlite"] = file_hash(index.database)
    (index.path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DomainError, match="schema is not supported") as error:
        OntologyContext(index.path)
    assert error.value.envelope.code == "unsafe_sqlite_schema"


def test_local_resume_recovers_hot_journal_before_copy(tmp_path):
    snapshot = core.load_snapshot(OWL, options=core.LoadOptions(imports=core.ImportPolicy.IGNORE))
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls == 7

    destination = tmp_path / "resumed"
    with pytest.raises(InterruptedError):
        build_context_package(snapshot, destination, checkpoint_interval=2, stop_requested=stop)
    database = tmp_path / ".resumed.building" / "context.sqlite"
    script = """
import os, sqlite3, sys
connection = sqlite3.connect(sys.argv[1])
connection.execute('PRAGMA cache_size=1')
connection.execute("UPDATE metadata SET value='999' WHERE key='axiom_count'")
connection.executemany("INSERT INTO entities VALUES (?, 'class', 0)",
                       [('urn:uncommitted:' + str(i) + 'x' * 2048,) for i in range(100)])
os._exit(7)
"""
    assert subprocess.run([sys.executable, "-c", script, str(database)]).returncode == 7
    assert database.with_name(database.name + "-journal").exists()
    index = build_context_package(
        snapshot, destination, checkpoint_interval=2, work_directory=tmp_path / "local"
    )
    assert index.manifest["runtime"]["resumed_axiom_count"] == 6
    assert index.search(term="uncommitted")["total_count"] == 0
    assert index.manifest["completeness"]["axiom_count"] == len(list(snapshot.iter_axioms()))
