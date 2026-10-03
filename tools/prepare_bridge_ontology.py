"""Prepare E14's metadata-only reasoning copy using native libxml2 operations.

This is a pinned experiment transformation, not a general OWL normalizer. Original
ontologies remain authoritative; strict native reasoner admission is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urljoin

from lxml import etree

from exact.utils.provenance import sha256_file

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
}
XML_BASE = "{http://www.w3.org/XML/1998/namespace}base"
NCIT = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"
EXPECTED = {
    NCIT + "textArea": {"declaration": 1, "annotation_range": 6},
    NCIT + "user-system": {"declaration": 1, "annotation_range": 1},
    NCIT + "date-time-system": {"declaration": 1, "annotation_range": 1},
}


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _verified(binding):
    path = Path(binding["path"]).resolve()
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"E14 preparation input changed: {path}")
    return path


def _implementation():
    return {
        "sha256": sha256_file(Path(__file__)),
        "lxml": list(etree.LXML_VERSION),
        "libxml2": list(etree.LIBXML_VERSION),
        "policy": "e14_ncit_metadata_only_v1",
    }


def _parse(path):
    parser = etree.XMLParser(
        resolve_entities=False, load_dtd=False, no_network=True, huge_tree=True
    )
    tree = etree.parse(str(path), parser, base_url=Path(path).resolve().as_uri())
    if tree.docinfo.doctype or tree.getroot().tag != "{" + NS["rdf"] + "}RDF":
        raise ValueError("E14 preparation requires RDF/XML without a DTD")
    return tree


def _resolved(node, value):
    return urljoin(node.base or node.getroottree().docinfo.URL, value)


def _iri(attribute):
    parent = attribute.getparent()
    value = str(attribute)
    if attribute.attrname == "{" + NS["rdf"] + "}ID":
        value = "#" + value
    return _resolved(parent, value)


def _candidate_attributes(tree, iris, context=None):
    # Native XPath scans the document; Python sees only potential matches and
    # relative references (the pinned release uses absolute RDF identifiers).
    fragments = sorted({iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1] for iri in iris})
    terms = [f"contains(., '{fragment}')" for fragment in fragments]
    predicate = "not(contains(., ':')) or " + " or ".join(terms)
    if context:
        predicate = f"({predicate}) and ({context})"
    return tree.xpath(
        f"//@rdf:about[{predicate}] | //@rdf:resource[{predicate}] | "
        f"//@rdf:datatype[{predicate}] | //@rdf:ID[{predicate}] | //@rdf:type[{predicate}]",
        namespaces=NS,
    )


def _audit(tree):
    root = tree.getroot()
    nodes, records, properties = [], [], set()
    for attribute in _candidate_attributes(tree, EXPECTED):
        iri = _iri(attribute)
        if iri not in EXPECTED:
            continue
        node = attribute.getparent()
        empty = not len(node) and not (node.text or "").strip() and len(node.attrib) == 1
        if (
            node.tag == "{" + NS["rdfs"] + "}Datatype"
            and attribute.attrname == "{" + NS["rdf"] + "}about"
            and node.getparent() is root
            and empty
        ):
            kind = "declaration"
        elif (
            node.tag == "{" + NS["rdfs"] + "}range"
            and attribute.attrname == "{" + NS["rdf"] + "}resource"
            and node.getparent().tag == "{" + NS["owl"] + "}AnnotationProperty"
            and node.getparent().getparent() is root
            and set(node.getparent().attrib) == {"{" + NS["rdf"] + "}about"}
            and empty
        ):
            kind = "annotation_range"
            properties.add(
                _resolved(node.getparent(), node.getparent().get("{" + NS["rdf"] + "}about"))
            )
        else:
            raise ValueError(
                f"Unsupported datatype has non-metadata use at line {node.sourceline}: {iri}"
            )
        nodes.append(node)
        records.append({"datatype": iri, "kind": kind, "line": node.sourceline})
    for iri in EXPECTED:
        # Expanded XML names can encode RDF predicates/types without RDF IRI attributes.
        if tree.xpath(
            "boolean(//*[concat(namespace-uri(), local-name())=$iri] | "
            "//@*[concat(namespace-uri(), local-name())=$iri])",
            iri=iri,
        ):
            raise ValueError(f"Unsupported datatype appears as an XML name: {iri}")
    imported = set()
    for node in tree.xpath("//owl:imports", namespaces=NS):
        if (
            set(node.attrib) != {"{" + NS["rdf"] + "}resource"}
            or len(node)
            or (node.text or "").strip()
        ):
            raise ValueError("E14 preparation requires explicit resource-valued imports")
        imported.add(_resolved(node, node.get("{" + NS["rdf"] + "}resource")))
    # Do not silently miss imports encoded as generic RDF property attributes.
    if tree.xpath("boolean(//@owl:imports)", namespaces=NS):
        raise ValueError("E14 preparation requires element-form imports")
    return nodes, records, properties, imported


def _check_property_types(tree, properties):
    if not properties:
        return
    logical_types = {
        NS["owl"] + name
        for name in (
            "DatatypeProperty",
            "ObjectProperty",
            "FunctionalProperty",
            "InverseFunctionalProperty",
            "TransitiveProperty",
            "SymmetricProperty",
            "AsymmetricProperty",
            "ReflexiveProperty",
            "IrreflexiveProperty",
        )
    } | {NS["rdf"] + "Property"}
    logical_predicates = {
        NS["owl"] + name for name in ("onProperty", "inverseOf", "equivalentProperty")
    }
    type_names = [
        "owl:" + iri.removeprefix(NS["owl"]) for iri in logical_types if iri.startswith(NS["owl"])
    ] + ["rdf:Property"]
    context = " or ".join(
        "parent::" + name
        for name in type_names + ["owl:onProperty", "owl:inverseOf", "owl:equivalentProperty"]
    )
    context += " or parent::*[rdf:type or @rdf:type]"
    for attribute in _candidate_attributes(tree, properties, context):
        if _iri(attribute) not in properties:
            continue
        node = attribute.getparent()
        expanded = etree.QName(node).namespace + etree.QName(node).localname
        types = {
            _iri(value)
            for value in node.xpath("./@rdf:type | ./rdf:type/@rdf:resource", namespaces=NS)
        }
        if expanded in logical_types | logical_predicates or types & logical_types:
            raise ValueError("Metadata range property also has a logical property use")


class _Digest:
    def __init__(self):
        self.digest = hashlib.sha256()

    def write(self, chunk):
        self.digest.update(chunk)


def _canonical_digest(tree):
    sink = _Digest()
    tree.write_c14n(sink, with_comments=True)
    return sink.digest.hexdigest()


def _counts(records):
    counts = {iri: {"declaration": 0, "annotation_range": 0} for iri in EXPECTED}
    for record in records:
        counts[record["datatype"]][record["kind"]] += 1
    return counts


def prepare(inputs, imports, output, *, expected_counts=None):
    """Return a bound immutable manifest for the complete pinned original closure."""
    if set(inputs) != {"source", "target"}:
        raise ValueError("E14 preparation requires exactly source and target")
    implementation = _implementation()
    expected_counts = EXPECTED if expected_counts is None else expected_counts
    if set(expected_counts) != set(EXPECTED) or any(
        set(counts) != {"declaration", "annotation_range"}
        or any(type(value) is not int or value < 0 for value in counts.values())
        for counts in expected_counts.values()
    ):
        raise ValueError("E14 preparation requires an explicit metadata count contract")
    output = Path(output).resolve()
    manifest_path = output / "preparation.json"
    if manifest_path.exists():
        receipt = bind(manifest_path)
        verify_preparation(receipt, inputs, imports)
        if json.loads(manifest_path.read_text())["expected_counts"] != expected_counts:
            raise ValueError("E14 metadata count contract changed")
        return receipt
    if output.exists() and any(output.iterdir()):
        raise ValueError("E14 preparation output is nonempty without a completed manifest")
    originals = {}
    for item in [*inputs.values(), *imports.values()]:
        canonical = str(Path(item["path"]).resolve())
        if item["path"] != canonical:
            raise ValueError("E14 bindings require canonical absolute paths")
        if canonical in originals and originals[canonical] != item:
            raise ValueError("E14 has conflicting bindings for one document")
        originals[canonical] = item
    reports, all_records, properties = {}, [], set()
    for binding in originals.values():
        path = _verified(binding)
        tree = _parse(path)
        audit_nodes, records, declared, required = _audit(tree)
        if required - imports.keys():
            raise ValueError(f"E14 imports are not pinned: {sorted(required - imports.keys())}")
        reports[str(path)] = {"removed": records, "imports": sorted(required)}
        all_records.extend(records)
        properties.update(declared)
        del audit_nodes, tree
    if _counts(all_records) != expected_counts:
        raise ValueError("E14 metadata counts differ from the reviewed NCIT inventory")
    output.mkdir(parents=True, exist_ok=True)
    derived = {}
    for binding in originals.values():
        path = _verified(binding)
        tree = _parse(path)
        _check_property_types(tree, properties)
        nodes, records, _, _ = _audit(tree)
        report = reports[str(path)]
        if records != report["removed"]:
            raise ValueError("E14 metadata audit changed during preparation")
        if not nodes:
            derived[binding["path"]] = binding
            del tree
            continue
        root = tree.getroot()
        original_base = _resolved(root, "")
        # Materialize the original document base, so moving the copy cannot change IRIs.
        root.set(XML_BASE, original_base)
        for node in nodes:
            # Keep inter-element text, including whitespace, in its original order.
            previous = node.getprevious()
            if node.tail:
                if previous is None:
                    node.getparent().text = (node.getparent().text or "") + node.tail
                else:
                    previous.tail = (previous.tail or "") + node.tail
            node.getparent().remove(node)
        del node
        canonical = _canonical_digest(tree)
        destination = output / (binding["sha256"] + ".reasoning.owl")
        temporary = destination.with_suffix(".tmp")
        tree.write(str(temporary), encoding="UTF-8", xml_declaration=True)
        del nodes, root, tree
        check = _parse(temporary)
        if _audit(check)[0] or _canonical_digest(check) != canonical:
            raise ValueError("E14 derived document did not preserve the audited XML content")
        del check
        os.replace(temporary, destination)
        derived[binding["path"]] = bind(destination)
        report.update({"canonical_sha256": canonical, "document_base": original_base})
    # Publishing is conditional on unchanged original bytes after all native work.
    for binding in originals.values():
        _verified(binding)
    if _implementation() != implementation:
        raise ValueError("E14 preparation implementation changed while running")
    manifest = {
        "schema_version": 1,
        "kind": "e14_metadata_only_reasoning_preparation",
        "implementation": implementation,
        "original_inputs": inputs,
        "original_imports": imports,
        "derived_inputs": {name: derived[value["path"]] for name, value in inputs.items()},
        "derived_imports": {name: derived[value["path"]] for name, value in imports.items()},
        "excluded_datatypes": sorted(EXPECTED),
        "removed_counts": _counts(all_records),
        "expected_counts": expected_counts,
        "documents": reports,
        "logical_content_preserved": True,
        "strict_reasoner_admission_required": True,
    }
    temporary_manifest = output / f".preparation.{os.getpid()}.tmp"
    try:
        with temporary_manifest.open("x") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Link publishes a complete receipt atomically and never replaces another receipt.
        os.link(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return bind(manifest_path)


def verify_preparation(manifest_binding, original_inputs, original_imports):
    """Check hashes/identity without reparsing ontologies; admission remains required."""
    manifest = json.loads(_verified(manifest_binding).read_text())
    expected = {
        "schema_version": 1,
        "kind": "e14_metadata_only_reasoning_preparation",
        "implementation": _implementation(),
        "original_inputs": original_inputs,
        "original_imports": original_imports,
        "excluded_datatypes": sorted(EXPECTED),
        "logical_content_preserved": True,
        "strict_reasoner_admission_required": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("E14 preparation identity or original bindings changed")
    records = [
        record for document in manifest["documents"].values() for record in document["removed"]
    ]
    if (
        manifest["removed_counts"] != manifest["expected_counts"]
        or _counts(records) != manifest["removed_counts"]
    ):
        raise ValueError("E14 preparation removal evidence changed")
    for label, originals in (("inputs", original_inputs), ("imports", original_imports)):
        derived = manifest["derived_" + label]
        if set(derived) != set(originals):
            raise ValueError("E14 preparation closure changed")
        for binding in [*originals.values(), *derived.values()]:
            _verified(binding)
    return manifest["derived_inputs"], manifest["derived_imports"]
