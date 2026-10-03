"""E14 preparation changes only the explicitly reviewed metadata nodes."""

import json

import pytest
from lxml import etree

from tools import prepare_bridge_ontology as preparation

IRI = preparation.NCIT + "textArea"
COUNTS = {
    iri: {"declaration": int(iri == IRI), "annotation_range": int(iri == IRI)}
    for iri in preparation.EXPECTED
}


def document(content, *, base='xml:base="https://example.org/ontology"'):
    return (
        f'<rdf:RDF xmlns:rdf="{preparation.NS["rdf"]}" '
        f'xmlns:rdfs="{preparation.NS["rdfs"]}" xmlns:owl="{preparation.NS["owl"]}" '
        f'xmlns:n="{preparation.NCIT}" {base}>{content}</rdf:RDF>'
    )


def fixture(tmp_path, extra="", *, base='xml:base="https://example.org/ontology"'):
    source = tmp_path / "source.owl"
    source.write_text(
        document(
            f'<rdfs:Datatype rdf:about="{IRI}"/>'
            '<owl:AnnotationProperty rdf:about="#description">'
            "<rdfs:label>Description</rdfs:label>"
            f'<rdfs:range rdf:resource="{IRI}"/></owl:AnnotationProperty>'
            '<owl:Class rdf:about="#Parent"/>'
            '<owl:Class rdf:about="#Child"><rdfs:subClassOf rdf:resource="#Parent"/>'
            '<rdfs:label xml:lang="en">Child</rdfs:label></owl:Class>'
            '<rdfs:Datatype rdf:about="#enum"><owl:equivalentClass><rdfs:Datatype>'
            '<owl:oneOf rdf:parseType="Collection"><rdf:Description rdf:about="#value"/>'
            "</owl:oneOf></rdfs:Datatype></owl:equivalentClass></rdfs:Datatype>" + extra,
            base=base,
        )
    )
    target = tmp_path / "target.owl"
    target.write_text(document('<owl:Class rdf:about="urn:other"/>'))
    return {"source": preparation.bind(source), "target": preparation.bind(target)}


def prepare(inputs, tmp_path, imports=None):
    return preparation.prepare(inputs, imports or {}, tmp_path / "derived", expected_counts=COUNTS)


def test_preserves_every_other_xml_node_and_original_bytes(tmp_path):
    inputs = fixture(tmp_path)
    receipt = prepare(inputs, tmp_path)
    derived, imports = preparation.verify_preparation(receipt, inputs, {})
    assert imports == {}
    assert derived["target"] == inputs["target"]
    assert derived["source"] != inputs["source"]
    for binding in inputs.values():
        assert preparation.bind(binding["path"]) == binding
    actual = preparation._parse(derived["source"]["path"])
    expected = preparation._parse(inputs["source"]["path"])
    for node in expected.xpath(
        "//rdfs:Datatype[@rdf:about=$iri] | //rdfs:range[@rdf:resource=$iri]",
        namespaces=preparation.NS,
        iri=IRI,
    ):
        node.getparent().remove(node)
    assert etree.tostring(actual, method="c14n") == etree.tostring(expected, method="c14n")
    manifest = json.loads(preparation._verified(receipt).read_text())
    assert manifest["removed_counts"] == COUNTS
    assert manifest["logical_content_preserved"] is True
    assert prepare(inputs, tmp_path) == receipt
    with pytest.raises(ValueError, match="count contract changed"):
        preparation.prepare(inputs, {}, tmp_path / "derived")


@pytest.mark.parametrize(
    "extra",
    [
        f'<owl:Class rdf:about="urn:c"><rdfs:label rdf:datatype="{IRI}">text</rdfs:label></owl:Class>',
        f'<owl:DatatypeProperty rdf:about="urn:p"><rdfs:range rdf:resource="{IRI}"/></owl:DatatypeProperty>',
        f'<rdf:Description rdf:about="{IRI}"><rdf:type rdf:resource="{preparation.NS["rdfs"]}Datatype"/></rdf:Description>',
        '<n:textArea rdf:about="urn:c"/>',
        f'<rdf:Description rdf:about="urn:c" rdf:type="{IRI}"/>',
        '<owl:Class rdf:about="urn:c" n:textArea="bad"/>',
    ],
)
def test_rejects_logical_literal_alternative_rdf_and_qname_uses(tmp_path, extra):
    inputs = fixture(tmp_path, extra)
    with pytest.raises(ValueError, match="Unsupported datatype"):
        prepare(inputs, tmp_path)
    assert not (tmp_path / "derived/preparation.json").exists()


def test_rejects_nonempty_datatype_declaration_instead_of_losing_metadata(tmp_path):
    inputs = fixture(tmp_path)
    source = tmp_path / "source.owl"
    source.write_text(
        source.read_text().replace(
            f'<rdfs:Datatype rdf:about="{IRI}"/>',
            f'<rdfs:Datatype rdf:about="{IRI}"><rdfs:label>Keep me</rdfs:label></rdfs:Datatype>',
        )
    )
    inputs["source"] = preparation.bind(source)
    with pytest.raises(ValueError, match="non-metadata use"):
        prepare(inputs, tmp_path)


def test_requires_full_pinned_import_closure_and_checks_logical_uses_there(tmp_path):
    inputs = fixture(
        tmp_path,
        '<owl:Ontology rdf:about=""><owl:imports rdf:resource="import.owl"/></owl:Ontology>',
    )
    imported = tmp_path / "import.owl"
    imported.write_text(
        document(
            f'<owl:Class rdf:about="urn:c"><rdfs:label rdf:datatype="{IRI}">x</rdfs:label></owl:Class>'
        )
    )
    with pytest.raises(ValueError, match="not pinned"):
        prepare(inputs, tmp_path)
    with pytest.raises(ValueError, match="non-metadata use"):
        prepare(inputs, tmp_path, {"https://example.org/import.owl": preparation.bind(imported)})


def test_rejects_metadata_property_used_as_a_data_property_in_an_import(tmp_path):
    inputs = fixture(tmp_path)
    imported = tmp_path / "extra.owl"
    imported.write_text(document('<owl:DatatypeProperty rdf:about="#description"/>'))
    with pytest.raises(ValueError, match="logical property use"):
        prepare(inputs, tmp_path, {"urn:extra": preparation.bind(imported)})


def test_relative_datatype_reference_is_audited(tmp_path):
    inputs = fixture(
        tmp_path,
        '<owl:Class rdf:about="urn:c"><rdfs:label xml:base="'
        + preparation.NCIT[:-1]
        + '" rdf:datatype="#textArea">bad</rdfs:label></owl:Class>',
    )
    with pytest.raises(ValueError, match="non-metadata use"):
        prepare(inputs, tmp_path)


def test_preserves_missing_and_nested_base_after_moving_copy(tmp_path):
    extra = '<owl:Class xml:base="nested/" rdf:about="Child"><rdfs:subClassOf rdf:resource="Parent"/></owl:Class>'
    inputs = fixture(tmp_path, extra, base="")
    receipt = prepare(inputs, tmp_path)
    derived, _ = preparation.verify_preparation(receipt, inputs, {})
    original_tree = preparation._parse(inputs["source"]["path"])
    derived_tree = preparation._parse(derived["source"]["path"])

    def iri_inventory(tree):
        return [
            preparation._iri(attribute)
            for attribute in tree.xpath(
                "//owl:Class/@rdf:about | //rdfs:subClassOf/@rdf:resource",
                namespaces=preparation.NS,
            )
        ]

    assert iri_inventory(original_tree) == iri_inventory(derived_tree)
    assert derived_tree.getroot().get(preparation.XML_BASE) == (tmp_path / "source.owl").as_uri()


def test_counts_and_mutations_cannot_be_silently_reused(tmp_path):
    inputs = fixture(tmp_path)
    with pytest.raises(ValueError, match="counts differ"):
        preparation.prepare(inputs, {}, tmp_path / "derived")
    receipt = prepare(inputs, tmp_path)
    manifest = json.loads(preparation._verified(receipt).read_text())
    derived = preparation._verified(manifest["derived_inputs"]["source"])
    derived.write_text(derived.read_text() + "\n")
    with pytest.raises(ValueError, match="input changed"):
        preparation.verify_preparation(receipt, inputs, {})


def test_changed_input_during_native_preparation_never_publishes(tmp_path, monkeypatch):
    inputs = fixture(tmp_path)
    original = preparation._canonical_digest

    def changed(tree):
        result = original(tree)
        with (tmp_path / "source.owl").open("a") as stream:
            stream.write("\n")
        return result

    monkeypatch.setattr(preparation, "_canonical_digest", changed)
    with pytest.raises(ValueError, match="input changed"):
        prepare(inputs, tmp_path)
    assert not (tmp_path / "derived/preparation.json").exists()


def test_dtd_and_conflicting_bindings_fail_closed(tmp_path):
    inputs = fixture(tmp_path)
    with pytest.raises(ValueError, match="conflicting bindings"):
        prepare(inputs, tmp_path, {"urn:alias": {**inputs["source"], "sha256": "0" * 64}})
    source = tmp_path / "source.owl"
    source.write_text('<!DOCTYPE rdf:RDF [<!ENTITY unsafe "anything">]>' + source.read_text())
    inputs["source"] = preparation.bind(source)
    with pytest.raises(ValueError, match="without a DTD"):
        prepare(inputs, tmp_path)


@pytest.mark.parametrize(
    "typing",
    [
        '<rdf:type rdf:resource="#DatatypeProperty"/>',
        '<rdf:type rdf:resource="#ObjectProperty"/>',
        '<rdf:type rdf:resource="#FunctionalProperty"/>',
    ],
)
def test_resolves_relative_property_types_in_imports(tmp_path, typing):
    inputs = fixture(tmp_path)
    imported = tmp_path / "property.owl"
    imported.write_text(
        document(
            '<rdf:Description rdf:about="https://example.org/ontology#description">'
            + typing
            + "</rdf:Description>",
            base='xml:base="http://www.w3.org/2002/07/owl"',
        )
    )
    with pytest.raises(ValueError, match="logical property use"):
        prepare(inputs, tmp_path, {"urn:property": preparation.bind(imported)})


def test_rejects_rdf_type_attribute_property_punning(tmp_path):
    inputs = fixture(
        tmp_path,
        '<rdf:Description rdf:about="#description" rdf:type="'
        + preparation.NS["owl"]
        + 'DatatypeProperty"/>',
    )
    with pytest.raises(ValueError, match="logical property use"):
        prepare(inputs, tmp_path)


def test_incomplete_receipt_is_not_published(tmp_path, monkeypatch):
    inputs = fixture(tmp_path)

    def crash(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(preparation.os, "link", crash)
    with pytest.raises(OSError, match="publication failure"):
        prepare(inputs, tmp_path)
    assert not (tmp_path / "derived/preparation.json").exists()
    assert not list((tmp_path / "derived").glob(".preparation*.tmp"))


def test_native_scan_preserves_reference_locations_among_nested_nodes_and_comments(tmp_path):
    source = tmp_path / "nested.owl"
    source.write_text(
        document(
            '<!-- preceding sibling -->\n<owl:Class rdf:about="urn:outer">'
            '<unqualified xmlns=""><!-- nested comment -->text<child rdf:about="urn:innocent"/>'
            '<child xml:base="' + preparation.NCIT[:-1] + '" rdf:type="#textArea"/>'
            "</unqualified></owl:Class>"
        )
    )
    tree = preparation._parse(source)
    references = preparation._candidate_attributes(tree, preparation.EXPECTED)
    assert len(references) == 1
    assert references[0].getparent().tag == "child"
    assert preparation._iri(references[0]) == IRI
    with pytest.raises(ValueError, match="non-metadata use"):
        preparation._audit(tree)


@pytest.mark.parametrize(
    "attribute",
    [
        'n:textArea="anything"',
        f'rdf:type="{IRI}"',
        'owl:imports="urn:unknown"',
    ],
)
def test_native_scan_includes_document_root_attributes(tmp_path, attribute):
    source = tmp_path / "root.owl"
    source.write_text(document("", base=attribute))
    with pytest.raises(ValueError, match="Unsupported datatype|element-form imports"):
        preparation._audit(preparation._parse(source))


def test_builtin_comment_annotation_range_does_not_need_explicit_declaration(tmp_path):
    inputs = fixture(tmp_path)
    source = tmp_path / "source.owl"
    source.write_text(
        source.read_text()
        .replace(
            '<owl:AnnotationProperty rdf:about="#description">',
            '<rdf:Description rdf:about="' + preparation.NS["rdfs"] + 'comment">',
        )
        .replace("</owl:AnnotationProperty>", "</rdf:Description>")
    )
    inputs["source"] = preparation.bind(source)
    receipt = prepare(inputs, tmp_path)
    derived, _ = preparation.verify_preparation(receipt, inputs, {})
    tree = preparation._parse(derived["source"]["path"])
    assert (
        tree.xpath("string(//rdf:Description/rdfs:label)", namespaces=preparation.NS)
        == "Description"
    )
    assert not tree.xpath("//rdf:Description/rdfs:range", namespaces=preparation.NS)


def test_undeclared_custom_property_range_remains_rejected(tmp_path):
    inputs = fixture(tmp_path)
    source = tmp_path / "source.owl"
    source.write_text(source.read_text().replace("owl:AnnotationProperty", "rdf:Description"))
    inputs["source"] = preparation.bind(source)
    with pytest.raises(ValueError, match="non-metadata use"):
        prepare(inputs, tmp_path)
