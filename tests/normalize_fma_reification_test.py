import hashlib
import json

import pytest
from lxml import etree
from pyowl_core import (
    BackendPreference,
    LoadOptions,
    SubClassOf,
    UnsupportedSyntaxError,
    load_snapshot,
)

from tools import normalize_fma_reification as normalizer

RDF = normalizer.RDF[1:-1]
OWL = normalizer.OWL[1:-1]
RDFS = normalizer.RDFS[1:-1]
DC = "http://purl.org/dc/elements/1.1/"
RESTRICTION = '<owl:Restriction><owl:onProperty rdf:resource="urn:part"/><owl:someValuesFrom rdf:resource="urn:Whole"/></owl:Restriction>'
ASSERTION = f"<rdfs:subClassOf>{RESTRICTION}</rdfs:subClassOf>"


def document(assertion=ASSERTION, target=None):
    target = (
        f"<owl:annotatedTarget>{RESTRICTION}</owl:annotatedTarget>" if target is None else target
    )
    return f"""<rdf:RDF xmlns:rdf="{RDF}" xmlns:owl="{OWL}" xmlns:rdfs="{RDFS}" xmlns:dc="{DC}">
<!-- Preserve the ontology, annotations, literal whitespace, and source order. -->
<owl:Ontology rdf:about="urn:test"/>
<owl:AnnotationProperty rdf:about="{DC}contributor"/>
<owl:AnnotationProperty rdf:about="{DC}description"/>
<owl:ObjectProperty rdf:about="urn:part"/>
<owl:Class rdf:about="urn:Whole"/>
<owl:Class rdf:about="urn:Part"><rdfs:label xml:lang="en">  Part  </rdfs:label>{assertion}</owl:Class>
<owl:Axiom><dc:contributor rdf:resource="urn:author"/>
<owl:annotatedSource rdf:resource="urn:Part"/>
<rdfs:comment xml:lang="en">  leading and trailing  </rdfs:comment>
<owl:annotatedProperty rdf:resource="{RDFS}subClassOf"/>{target}
<dc:description rdf:datatype="http://www.w3.org/2001/XMLSchema#string">\n  KEEP\tthis whitespace \n</dc:description>
</owl:Axiom></rdf:RDF>""".encode()


def test_native_reification_is_corrected_without_losing_axiom_annotations():
    original = document()
    options = LoadOptions(backend=BackendPreference.NATIVE)
    with pytest.raises(UnsupportedSyntaxError, match="main triple is absent"):
        load_snapshot(original, document_iri="urn:test", options=options)
    normalized, receipt = normalizer.normalize_bytes(original, expected_changes=1)
    snapshot = load_snapshot(normalized, document_iri="urn:test", options=options)
    assert snapshot.report.backend == "native" and snapshot.is_complete
    axioms = list(snapshot.iter_axioms(SubClassOf))
    assert len(axioms) == 1
    axiom = axioms[0]
    assert axiom.sub_class.iri.value == "urn:Part"
    assert axiom.super_class.property.iri.value == "urn:part"
    assert axiom.super_class.filler.iri.value == "urn:Whole"
    annotations = {a.property.iri.value: a.value for a in axiom.annotations}
    assert annotations[DC + "contributor"].value == "urn:author"
    assert annotations[RDFS + "comment"].lexical_form == "  leading and trailing  "
    assert annotations[RDFS + "comment"].language == "en"
    assert annotations[DC + "description"].lexical_form == "\n  KEEP\tthis whitespace \n"
    assert (
        annotations[DC + "description"].datatype.iri.value
        == "http://www.w3.org/2001/XMLSchema#string"
    )
    assert receipt["before_after_audit"]["equal"]
    assert receipt["counts"]["retained_annotations_on_changed_axioms"] == 3


def test_already_valid_input_is_byte_identical_and_changes_are_deterministic():
    first, _ = normalizer.normalize_bytes(document(), expected_changes=1)
    again, _ = normalizer.normalize_bytes(document(), expected_changes=1)
    assert first == again
    unchanged, record = normalizer.normalize_bytes(first, expected_changes=0)
    assert unchanged == first
    assert record["counts"]["rewired_annotation_targets"] == 0


@pytest.mark.parametrize("assertion,count", [("", 0), (ASSERTION + ASSERTION, 2)])
def test_missing_or_ambiguous_asserted_restriction_fails(assertion, count):
    with pytest.raises(ValueError, match=f"found {count}"):
        normalizer.normalize_bytes(document(assertion=assertion), expected_changes=1)


def test_unsupported_literal_restriction_does_not_silently_normalize_values():
    target = f'<owl:annotatedTarget>{RESTRICTION.replace("owl:someValuesFrom", "owl:hasValue")}</owl:annotatedTarget>'
    with pytest.raises(ValueError, match="Only named-property someValuesFrom"):
        normalizer.normalize_bytes(document(target=target), expected_changes=1)


def test_explicitly_identified_duplicate_is_not_merged():
    target = (
        "<owl:annotatedTarget>"
        + RESTRICTION.replace("<owl:Restriction>", '<owl:Restriction rdf:nodeID="other">')
        + "</owl:annotatedTarget>"
    )
    with pytest.raises(ValueError, match="explicitly identified"):
        normalizer.normalize_bytes(document(target=target), expected_changes=1)


def test_annotation_order_text_and_attributes_are_exactly_preserved():
    original = document()
    normalized, _ = normalizer.normalize_bytes(original, expected_changes=1)
    before, after = (etree.fromstring(value) for value in (original, normalized))
    before_axiom = before.find(normalizer.OWL + "Axiom")
    after_axiom = after.find(normalizer.OWL + "Axiom")
    assert [node.tag for node in before_axiom] == [node.tag for node in after_axiom]
    for left, right in zip(before_axiom, after_axiom):
        if left.tag != normalizer.OWL + "annotatedTarget":
            assert etree.tostring(left, method="c14n") == etree.tostring(right, method="c14n")
    before_label = before.find(".//" + normalizer.RDFS + "label")
    after_label = after.find(".//" + normalizer.RDFS + "label")
    assert before_label.text == after_label.text == "  Part  "


def test_independent_serialized_audit_detects_unrelated_annotation_changes(monkeypatch):
    serialize = normalizer.etree.tostring

    def damaged(*args, **kwargs):
        return serialize(*args, **kwargs).replace(b"leading and trailing", b"wrong annotation")

    monkeypatch.setattr(normalizer.etree, "tostring", damaged)
    with pytest.raises(ValueError, match="changed document content"):
        normalizer.normalize_bytes(document(), expected_changes=1)


def test_existing_blank_node_identifier_is_reused_without_removing_it():
    original = document(
        assertion=ASSERTION.replace("<owl:Restriction>", '<owl:Restriction rdf:nodeID="existing">')
    )
    normalized, record = normalizer.normalize_bytes(original, expected_changes=1)
    root = etree.fromstring(normalized)
    assert (
        root.find(".//" + normalizer.OWL + "annotatedTarget").get(normalizer.RDF + "nodeID")
        == "existing"
    )
    assert record["counts"]["introduced_node_ids"] == 0
    assert record["before_after_audit"]["equal"]


def test_frozen_input_and_change_count_are_required_and_outputs_are_immutable(tmp_path):
    source, output, receipt = (
        tmp_path / name for name in ("original.owl", "fixed.owl", "receipt.json")
    )
    original = document()
    source.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    with pytest.raises(ValueError, match="checksum"):
        normalizer.prepare(source, output, receipt, expected_sha256="0" * 64, expected_changes=1)
    with pytest.raises(ValueError, match="Expected 90"):
        normalizer.prepare(source, output, receipt, expected_sha256=digest)
    assert not output.exists() and not receipt.exists()
    record = normalizer.prepare(source, output, receipt, expected_sha256=digest, expected_changes=1)
    assert record["status"] == "passed" and record["native_validation"] == "pending"
    assert record["normalized"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(receipt.read_text())["original"]["sha256"] == digest
    assert source.read_bytes() == original
    with pytest.raises(ValueError, match="outputs cannot exist"):
        normalizer.prepare(source, output, receipt, expected_sha256=digest, expected_changes=1)


def test_dtd_and_changed_restriction_child_order_fail_closed():
    with pytest.raises(ValueError, match="without a DTD"):
        normalizer.normalize_bytes(b"<!DOCTYPE rdf:RDF []>" + document(), expected_changes=1)
    reversed_target = '<owl:annotatedTarget><owl:Restriction><owl:someValuesFrom rdf:resource="urn:Whole"/><owl:onProperty rdf:resource="urn:part"/></owl:Restriction></owl:annotatedTarget>'
    with pytest.raises(ValueError, match="found 0"):
        normalizer.normalize_bytes(document(target=reversed_target), expected_changes=1)


def intersection_document(members=2, literal="Partition 2"):
    # Actual FMA attributed_part pattern (fma15810 at source line257519), with
    # the same two-, three- and four-member shapes observed in all11complex cases.
    inner = '<owl:Restriction><owl:onProperty rdf:resource="urn:related_part"/><owl:someValuesFrom rdf:resource="urn:Whole"/></owl:Restriction>'
    declarations = '<owl:ObjectProperty rdf:about="urn:related_part"/>'
    for index, value in enumerate(
        [literal, "Regional", "Position relative to corpus callosum"][: members - 1]
    ):
        declarations += f'<owl:DatatypeProperty rdf:about="urn:partition{index}"/>'
        inner += f'<owl:Restriction><owl:onProperty rdf:resource="urn:partition{index}"/><owl:hasValue>{value}</owl:hasValue></owl:Restriction>'
    nested = (
        '<owl:Restriction><owl:onProperty rdf:resource="urn:part"/><owl:someValuesFrom><owl:Class><owl:intersectionOf rdf:parseType="Collection">'
        + inner
        + "</owl:intersectionOf></owl:Class></owl:someValuesFrom></owl:Restriction>"
    )
    original = document(
        assertion=f"<rdfs:subClassOf>{nested}</rdfs:subClassOf>",
        target=f"<owl:annotatedTarget>{nested}</owl:annotatedTarget>",
    )
    return original.replace(
        b'<owl:Class rdf:about="urn:Whole"/>',
        declarations.encode() + b'<owl:Class rdf:about="urn:Whole"/>',
    )


@pytest.mark.parametrize("members", [2, 3, 4])
def test_actual_fma_intersection_literal_pattern_preserves_native_axiom(members):
    from pyowl_core import DataHasValue, ObjectIntersectionOf

    original = intersection_document(members, literal="  Partition 2  ")
    normalized, receipt = normalizer.normalize_bytes(original, expected_changes=1)
    snapshot = load_snapshot(
        normalized, document_iri="urn:test", options=LoadOptions(backend=BackendPreference.NATIVE)
    )
    assert snapshot.report.backend == "native" and snapshot.is_complete
    (axiom,) = snapshot.iter_axioms(SubClassOf)
    filler = axiom.super_class.filler
    assert isinstance(filler, ObjectIntersectionOf)
    assert len(filler.operands) == members
    literals = [
        item.value.lexical_form for item in filler.operands if isinstance(item, DataHasValue)
    ]
    assert "  Partition 2  " in literals
    assert len(axiom.annotations) == 3
    assert receipt["before_after_audit"]["equal"]


@pytest.mark.parametrize("alteration", ["whitespace", "language", "datatype"])
def test_nested_literal_differences_cannot_match_an_asserted_restriction(alteration):
    original = intersection_document()
    if alteration == "whitespace":
        original = original.replace(b"<owl:hasValue>Partition 2", b"<owl:hasValue> Partition 2", 1)
    elif alteration == "language":
        original = original.replace(
            b'<owl:Class rdf:about="urn:Part">', b'<owl:Class rdf:about="urn:Part" xml:lang="en">'
        )
    else:
        original = original.replace(
            b"<owl:hasValue>",
            b'<owl:hasValue rdf:datatype="http://www.w3.org/2001/XMLSchema#string">',
            1,
        )
    with pytest.raises(ValueError, match="found 0"):
        normalizer.normalize_bytes(original, expected_changes=1)
