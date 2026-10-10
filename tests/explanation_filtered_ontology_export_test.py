"""Native OWL export must also accept physically filtered structural originals."""

import copy
import json

import pyowl_core as core
import pytest

from exact_inspect.context import build_context_package
from exact_inspect.context_export import export_policy_context
from exact_inspect.context_resources import (
    _native_admitted_axiom,
    export_ontology_resource,
    validate_ontology_resource,
)
from exact_inspect.context_semantics import decode_payload, typed_node
from exact_inspect.contracts import VisibilityPolicy, canonical_json, file_hash
from tests.explanation_context_test import OWL as COMPLEX_OWL

OWL = b"""Ontology(
 Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>))
 Declaration(ObjectProperty(<urn:p>)) Declaration(DataProperty(<urn:d>))
 Declaration(NamedIndividual(<urn:i>))
 AnnotationAssertion(
  Annotation(Annotation(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P395> "HIDDEN-NESTED-MAPPING")
   <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P378> "NCI")
  Annotation(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P395> "HIDDEN-MAPPING")
  <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P97> <urn:A> "Exact definition"@en)
 AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "Alpha"@en)
 AnnotationAssertion(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P207> <urn:A> "HIDDEN-CUI")
 SubClassOf(Annotation(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P395> "HIDDEN-LOGICAL-QUALIFIER")
  <urn:A> ObjectSomeValuesFrom(ObjectInverseOf(<urn:p>) ObjectIntersectionOf(<urn:A> <urn:B>)))
 DataPropertyAssertion(Annotation(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P395> "HIDDEN-DATA-QUALIFIER")
  <urn:d> <urn:i> "001"^^<http://www.w3.org/2001/XMLSchema#integer>)
)"""


def test_export_physically_filtered_context_preserves_exact_visible_axioms_and_counts(tmp_path):
    original = build_context_package(core.load_snapshot(OWL), tmp_path / "original")
    policy = VisibilityPolicy(categories=(*VisibilityPolicy().categories, "annotations"))
    filtered = export_policy_context(original, tmp_path / "filtered", policy)
    with filtered._connection() as db:
        payloads = [decode_payload(row[0]) for row in db.execute("SELECT payload FROM axioms")]
    redacted = [p for p in payloads if p["original_syntax"] is None]
    assert {p["ast"]["type"] for p in redacted} == {
        "AnnotationAssertion",
        "SubClassOf",
        "DataPropertyAssertion",
    }
    assert "HIDDEN" not in json.dumps(payloads)
    before = {name: file_hash(filtered.path / name) for name in ("manifest.json", "context.sqlite")}
    target = tmp_path / "filtered.ofn"
    receipt_path = export_ontology_resource(filtered, target, policy)
    receipt = validate_ontology_resource(
        target,
        receipt_path,
        policy_hash=policy.policy_hash,
        ontology_ids=[filtered.ontology_version_id],
    )
    parsed = core.load_snapshot(target)
    observed = {canonical_json(typed_node(axiom)) for axiom in parsed.iter_axioms()}
    assert observed == {canonical_json(p["ast"]) for p in payloads}
    assert sum(receipt["retained_categories"].values()) == len(payloads) == len(observed)
    assert receipt["logical_axioms_preserved"] is True
    assert not receipt["removed_categories"]
    assert not parsed.import_manifest.edges
    text = target.read_text()
    assert "HIDDEN" not in text and '"NCI"' in text and '"Exact definition"@en' in text
    assert '"001"^^<http://www.w3.org/2001/XMLSchema#integer>' in text
    assert {name: file_hash(filtered.path / name) for name in before} == before
    frozen_export = file_hash(target)
    assert export_ontology_resource(filtered, target, policy) == receipt_path
    assert file_hash(target) == frozen_export


def test_public_native_roundtrip_covers_complex_asserted_structures():
    original = core.load_snapshot(
        COMPLEX_OWL, options=core.LoadOptions(imports=core.ImportPolicy.IGNORE)
    )
    for axiom in original.iter_axioms():
        restored = _native_admitted_axiom(typed_node(axiom), core)
        assert core.canonical_bytes(restored) == core.canonical_bytes(axiom)


@pytest.mark.parametrize(
    "corruption", ["unknown_constructor", "extra_field", "missing_field", "wrong_kind"]
)
def test_structural_recovery_rejects_unadmitted_changes(corruption):
    axiom = next(core.load_snapshot(b"Ontology(Declaration(Class(<urn:A>)))").iter_axioms())
    ast = copy.deepcopy(typed_node(axiom))
    if corruption == "unknown_constructor":
        ast["type"] = "UnknownClassAssertion"
    elif corruption == "extra_field":
        ast["invented"] = "unreviewed original"
    elif corruption == "missing_field":
        del ast["entity"]["kind"]
    else:
        ast["entity"]["kind"] = "individual"
    with pytest.raises(ValueError):
        _native_admitted_axiom(ast, core)
