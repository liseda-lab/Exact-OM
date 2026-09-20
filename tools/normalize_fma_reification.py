#!/usr/bin/env python3
"""Reconnect duplicated FMA restriction annotations; never add or discard OWL axioms."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from lxml import etree

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
OWL = "{http://www.w3.org/2002/07/owl#}"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
METADATA = {OWL + name for name in ("annotatedSource", "annotatedProperty", "annotatedTarget")}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse(data: bytes):
    tree = etree.parse(
        io.BytesIO(data),
        etree.XMLParser(
            resolve_entities=False,
            no_network=True,
            remove_blank_text=False,
            huge_tree=True,
        ),
    )
    if tree.docinfo.doctype or tree.getroot().tag != RDF + "RDF":
        raise ValueError("Expected RDF/XML without a DTD")
    return tree


def _one(element, tag):
    rows = element.findall(tag)
    if len(rows) != 1:
        raise ValueError(f"Expected one {tag} in axiom annotation")
    return rows[0]


def _resource(element):
    value = element.get(RDF + "resource", "")
    if (
        set(element.attrib) != {RDF + "resource"}
        or len(element)
        or element.text not in (None, "")
        or not urlsplit(value).scheme
    ):
        raise ValueError("Only a simple absolute rdf:resource is supported here")
    return value


def _literal(element):
    language_key = "{http://www.w3.org/XML/1998/namespace}lang"
    datatype = element.get(RDF + "datatype")
    if (
        len(element)
        or set(element.attrib) - {RDF + "datatype", language_key}
        or (
            datatype is not None
            and (not urlsplit(datatype).scheme or language_key in element.attrib)
        )
    ):
        raise ValueError("Unsupported hasValue literal")
    # Inherited XML language affects plain literals, including whitespace-only ones.
    language = None
    if datatype is None:
        node = element
        while node is not None:
            if language_key in node.attrib:
                language = node.get(language_key)
                break
            node = node.getparent()
    return ("literal", tuple(sorted(element.attrib.items())), element.text or "", language)


def _intersection(element):
    if element.attrib or len(element) != 1 or (element.text or "").strip():
        raise ValueError("Unsupported anonymous someValuesFrom filler")
    cls = element[0]
    if (
        cls.tag != OWL + "Class"
        or cls.attrib
        or len(cls) != 1
        or (cls.text or "").strip()
        or (cls.tail or "").strip()
    ):
        raise ValueError("Expected anonymous owl:Class intersection filler")
    collection = cls[0]
    if (
        collection.tag != OWL + "intersectionOf"
        or dict(collection.attrib) != {RDF + "parseType": "Collection"}
        or not 2 <= len(collection) <= 4
        or (collection.text or "").strip()
        or (collection.tail or "").strip()
    ):
        raise ValueError("Expected the FMA two-to-four-member restriction intersection")
    members = []
    for member in collection:
        if (member.tail or "").strip():
            raise ValueError("Unexpected intersection text")
        members.append(_restriction(member, nested=True))
    return ("intersection", tuple(members))


def _restriction(element, *, nested=False):
    """The two observed FMA shapes, with ordered members and exact literal values."""
    allowed_attributes = set() if nested else {RDF + "nodeID"}
    if (
        element.tag != OWL + "Restriction"
        or set(element.attrib) - allowed_attributes
        or (element.text or "").strip()
        or len(element) != 2
    ):
        raise ValueError("Unsupported restriction shape")
    tags = {child.tag for child in element}
    supported = [{OWL + "onProperty", OWL + "someValuesFrom"}]
    if nested:
        supported.append({OWL + "onProperty", OWL + "hasValue"})
    if tags not in supported:
        raise ValueError("Only named-property someValuesFrom and nested FMA hasValue are supported")
    rows = []
    for child in element:
        if (child.tail or "").strip():
            raise ValueError("Unexpected text inside restriction")
        if child.tag == OWL + "hasValue":
            value = _literal(child)
        elif child.tag == OWL + "someValuesFrom" and len(child):
            if nested:
                raise ValueError("Nested anonymous fillers beyond the FMA pattern are unsupported")
            value = _intersection(child)
        else:
            value = _resource(child)
        rows.append((child.tag, value))
    return tuple(rows)


def _plan(tree):
    root = tree.getroot()
    classes = {}
    used_ids = set(root.xpath("//@rdf:nodeID", namespaces={"rdf": RDF[1:-1]}))
    for node in root.findall(OWL + "Class"):
        classes.setdefault(node.get(RDF + "about"), []).append(node)
    changes, assigned, introduced = [], {}, []
    for index, axiom in enumerate(root):
        if axiom.tag != OWL + "Axiom":
            continue
        if _resource(_one(axiom, OWL + "annotatedProperty")) != SUBCLASS:
            continue
        source = _resource(_one(axiom, OWL + "annotatedSource"))
        target = _one(axiom, OWL + "annotatedTarget")
        if not len(target):  # Already uses an explicit IRI/blank-node reference.
            if set(target.attrib) not in ({RDF + "resource"}, {RDF + "nodeID"}):
                raise ValueError("Unsupported subclass annotation target")
            continue
        if target.attrib or len(target) != 1 or (target.text or "").strip():
            raise ValueError("Unsupported inline subclass annotation target")
        duplicate = target[0]
        key = _restriction(duplicate)
        if (duplicate.tail or "").strip():
            raise ValueError("Unexpected annotation-target text")
        candidates = []
        for node in classes.get(source, []):
            for assertion in node.findall(RDFS + "subClassOf"):
                if assertion.attrib or len(assertion) != 1:
                    continue
                try:
                    same = _restriction(assertion[0]) == key
                except ValueError:
                    same = False
                if same:
                    candidates.append(assertion[0])
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one asserted restriction for {source}; found {len(candidates)}"
            )
        existing = candidates[0]
        old_id = existing.get(RDF + "nodeID")
        duplicate_id = duplicate.get(RDF + "nodeID")
        if duplicate_id:
            if duplicate_id == old_id:
                continue  # Repeated description of the same RDF blank node is already valid.
            raise ValueError("Cannot merge an explicitly identified annotation restriction")
        if existing not in assigned:
            identifier = old_id or "exactFmaAxiom_" + _sha(json.dumps([source, key]).encode())[:24]
            if not old_id:
                if identifier in used_ids:
                    raise ValueError("Generated restriction nodeID collides with existing input")
                used_ids.add(identifier)
                introduced.append(identifier)
            assigned[existing] = identifier
        changes.append(
            {
                "axiom_index": index,
                "source": source,
                "node_id": assigned[existing],
                "restriction": [list(row) for row in key],
                "annotation_count": sum(child.tag not in METADATA for child in axiom),
            }
        )
    return changes, assigned, introduced


def _audit(tree, changes, introduced):
    """Compare the whole XML document after resolving only the intended rewired targets."""
    root = tree.getroot()
    nodes = {}
    for node in root.iter():
        identifier = node.get(RDF + "nodeID")
        if identifier and node.tag == OWL + "Restriction":
            nodes.setdefault(identifier, []).append(node)
    swaps, removed = [], []
    try:
        for change in changes:
            axiom = root[change["axiom_index"]]
            target = _one(axiom, OWL + "annotatedTarget")
            if len(target):
                restriction = target[0]
            else:
                matches = nodes.get(target.get(RDF + "nodeID"), [])
                if len(matches) != 1:
                    raise ValueError("Rewired target does not resolve to one asserted restriction")
                restriction = matches[0]
                assertion = restriction.getparent()
                owner = assertion.getparent()
                if (
                    target.get(RDF + "nodeID") != change["node_id"]
                    or assertion.tag != RDFS + "subClassOf"
                    or owner.tag != OWL + "Class"
                    or owner.get(RDF + "about") != change["source"]
                ):
                    raise ValueError(
                        "Rewired target is not its declared source's asserted restriction"
                    )
            # Render the exact named-property restriction value in an audit-only node.
            # Neither annotation literals nor other document content are normalized.
            replacement = etree.Element(OWL + "annotatedTarget", nsmap=target.nsmap)
            replacement.text = json.dumps(_restriction(restriction), ensure_ascii=False)
            replacement.tail = target.tail
            axiom.replace(target, replacement)
            swaps.append((axiom, target, replacement))
        for identifier in introduced:
            for node in nodes.get(identifier, []):
                removed.append((node, identifier))
                del node.attrib[RDF + "nodeID"]
        digest = hashlib.sha256()

        class Sink:
            def write(self, data):
                digest.update(data)

        tree.write_c14n(Sink(), with_comments=True)
        return digest.hexdigest()
    finally:
        for node, identifier in removed:
            node.set(RDF + "nodeID", identifier)
        for parent, original, replacement in reversed(swaps):
            parent.replace(replacement, original)


def normalize_bytes(original: bytes, *, expected_changes: int):
    tree = _parse(original)
    changes, assigned, introduced = _plan(tree)
    if len(changes) != expected_changes:
        raise ValueError(f"Expected {expected_changes} rewiring changes; found {len(changes)}")
    before = _audit(tree, changes, introduced)
    for restriction, identifier in assigned.items():
        restriction.set(RDF + "nodeID", identifier)
    for change in changes:
        target = _one(tree.getroot()[change["axiom_index"]], OWL + "annotatedTarget")
        target.remove(target[0])
        target.text = None
        target.set(RDF + "nodeID", change["node_id"])
    normalized = (
        etree.tostring(tree, encoding="UTF-8", xml_declaration=True) if changes else original
    )
    after = _audit(_parse(normalized), changes, introduced)
    if before != after:
        raise ValueError(
            "Rewiring changed document content outside the permitted identity correction"
        )
    return normalized, {
        "counts": {
            "rewired_annotation_targets": len(changes),
            "affected_classes": len({row["source"] for row in changes}),
            "introduced_node_ids": len(introduced),
            "retained_annotations_on_changed_axioms": sum(
                row["annotation_count"] for row in changes
            ),
        },
        "changes": changes,
        "before_after_audit": {
            "method": "Full XML C14N with comments, after resolving changed targets to exact ordered restriction values and ignoring only introduced nodeIDs and their structural-wrapper indentation",
            "before_sha256": before,
            "after_sha256": after,
            "equal": True,
            "annotation_literal_text_language_datatype_preserved": True,
            "output_reparsed_independently": True,
        },
    }


def prepare(
    source: Path, output: Path, receipt: Path, *, expected_sha256: str, expected_changes: int = 90
):
    paths = [path.resolve() for path in (source, output, receipt)]
    if len(set(paths)) != 3 or output.exists() or receipt.exists():
        raise ValueError(
            "Source, new derivative and new receipt must be distinct; outputs cannot exist"
        )
    original = source.read_bytes()
    if _sha(original) != expected_sha256:
        raise ValueError("Original FMA checksum differs from its frozen binding")
    normalized, audit = normalize_bytes(original, expected_changes=expected_changes)
    result = {
        "schema_version": 1,
        "kind": "fma_reification_identity_normalization",
        "status": "passed",
        "scope": "XML structural-preservation audit only; full native validation remains pending",
        "original": {"path": str(paths[0]), "sha256": _sha(original)},
        "normalized": {"path": str(paths[1]), "sha256": _sha(normalized)},
        "normalizer": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha(Path(__file__).read_bytes()),
        },
        "xml_backend": {"lxml": etree.LXML_VERSION, "libxml2": etree.LIBXML_VERSION},
        "native_validation": "pending",
        "alignment_references_used": False,
        **audit,
    }
    for path, data in (
        (output, normalized),
        (receipt, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-changes", type=int, default=90)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source,
                args.output,
                args.receipt,
                expected_sha256=args.expected_sha256,
                expected_changes=args.expected_changes,
            ),
            indent=2,
        )
    )
