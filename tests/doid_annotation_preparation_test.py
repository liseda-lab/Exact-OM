"""Annotation-only normalization keeps the strict RDF mapping boundary intact."""

import hashlib
from pathlib import Path

import pyowl_core
import pytest

from tools.prepare_doid_annotations import (
    declaration_derivative,
    prepare,
    verify_authorities,
)

FIXTURE = b"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
 xmlns:owl="http://www.w3.org/2002/07/owl#"
 xmlns:obo="http://purl.obolibrary.org/obo/"
 xmlns:oio="http://www.geneontology.org/formats/oboInOwl#">
 <owl:Ontology rdf:about="urn:doid:annotation-test"/>
 <owl:Class rdf:about="urn:doid:A">
  <obo:OBI_9991118 xml:lang="en">disease &amp; syndrome</obo:OBI_9991118>
  <oio:created_by rdf:datatype="http://www.w3.org/2001/XMLSchema#string">curator</oio:created_by>
 </owl:Class>
</rdf:RDF>
"""


def test_derivative_retains_original_bytes_and_strict_mapping():
    with pytest.raises(pyowl_core.UnsupportedSyntaxError) as exc:
        pyowl_core.load_snapshot(FIXTURE, document_iri="urn:doid:annotation-test")
    assert exc.value.rdf_mapping_report.dropped_triples == 2
    derivative, offset, insertion, counts = declaration_derivative(FIXTURE)
    assert derivative[:offset] + derivative[offset + len(insertion) :] == FIXTURE
    assert list(counts.values()) == [1, 1]
    snapshot = pyowl_core.load_snapshot(
        derivative,
        document_iri="urn:doid:annotation-test",
        options=pyowl_core.LoadOptions(allow_partial_rdf_mapping=False),
    )
    report = snapshot.root.rdf_mapping_report
    assert report.conformant and report.dropped_triples == 0
    assert report.total_triples == exc.value.rdf_mapping_report.total_triples + 2


@pytest.mark.parametrize(
    "original",
    [
        FIXTURE.replace(b"<obo:OBI_9991118", b"<obo:other").replace(
            b"</obo:OBI_9991118>", b"</obo:other>"
        ),
        FIXTURE.replace(
            b"</rdf:RDF>",
            b'<owl:ObjectProperty rdf:about="http://purl.obolibrary.org/obo/OBI_9991118"/>'
            b"</rdf:RDF>",
        ),
        FIXTURE.replace(
            b'<oio:created_by rdf:datatype="http://www.w3.org/2001/XMLSchema#string">curator</oio:created_by>',
            b'<oio:created_by rdf:resource="urn:curator"/>',
        ),
    ],
)
def test_unexpected_or_conflicting_input_fails(original):
    with pytest.raises(ValueError):
        declaration_derivative(original)


def test_upstream_hash_mismatch_fails(tmp_path):
    (tmp_path / "obi-edit.owl").write_bytes(FIXTURE)
    with pytest.raises(ValueError, match="Upstream declaration hash"):
        verify_authorities(tmp_path)


def test_preparation_is_immutable_and_bound_to_original(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.prepare_doid_annotations.verify_authorities", lambda _path: None)
    source = tmp_path / "input.owl"
    source.write_bytes(FIXTURE)
    expected = hashlib.sha256(FIXTURE).hexdigest()
    output = tmp_path / "normalized"
    payload = prepare(source, output, tmp_path, expected)
    assert source.read_bytes() == Path(payload["original_copy"]["path"]).read_bytes() == FIXTURE
    assert payload["original_reconstructed_sha256"] == expected
    assert prepare(source, output, tmp_path, expected) == payload
    with pytest.raises(ValueError, match="Original ontology hash"):
        prepare(source, output, tmp_path, "0" * 64)
    Path(payload["derivative"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing artifact differs"):
        prepare(source, output, tmp_path, expected)


def test_same_closure_ext_declarations_preserve_reified_annotation():
    from tools.prepare_doid_annotations import (
        EXT_IRIS,
        OWL,
        RDF,
        ext_declaration_derivative,
    )

    original = f"""<rdf:RDF xmlns:rdf="{RDF}" xmlns:owl="{OWL}"
      xmlns:obo="http://purl.obolibrary.org/obo/"
      xmlns:oio="http://www.geneontology.org/formats/oboInOwl#">
      <owl:Ontology rdf:about="urn:test:ext"/>
      <owl:AnnotationProperty rdf:about="urn:test:p">
        <obo:IAO_0000115>definition</obo:IAO_0000115>
      </owl:AnnotationProperty>
      <owl:Axiom>
        <owl:annotatedSource rdf:resource="urn:test:p"/>
        <owl:annotatedProperty rdf:resource="{EXT_IRIS[0]}"/>
        <owl:annotatedTarget>definition</owl:annotatedTarget>
        <oio:hasDbXref>reference</oio:hasDbXref>
        <oio:id>local-id</oio:id>
      </owl:Axiom>
    </rdf:RDF>""".encode()
    evidence = (
        f'<rdf:RDF xmlns:rdf="{RDF}" xmlns:owl="{OWL}">'
        + "".join(f'<owl:AnnotationProperty rdf:about="{iri}"/>' for iri in EXT_IRIS)
        + "</rdf:RDF>"
    ).encode()
    with pytest.raises(pyowl_core.UnsupportedSyntaxError):
        pyowl_core.parse_document(original)
    derivative, insertion = ext_declaration_derivative(original, evidence)
    assert derivative.replace(insertion, b"", 1) == original
    document = pyowl_core.parse_document(derivative)
    assert document.rdf_mapping_report.dropped_triples == 0
    with pytest.raises(ValueError, match="Missing unique"):
        ext_declaration_derivative(
            original, evidence.replace(b"AnnotationProperty", b"ObjectProperty")
        )
    conflict = evidence.replace(
        b"</rdf:RDF>", f'<owl:ObjectProperty rdf:about="{EXT_IRIS[0]}"/></rdf:RDF>'.encode()
    )
    with pytest.raises(ValueError, match="conflicting"):
        ext_declaration_derivative(original, conflict)
