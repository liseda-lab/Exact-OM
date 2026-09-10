#!/usr/bin/env python3
"""Add two upstream-declared DOID annotations without rewriting original bytes."""

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from exact.utils.fitted_artifacts import freeze_json

OWL = "http://www.w3.org/2002/07/owl#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
EXT_IRIS = (
    "http://purl.obolibrary.org/obo/IAO_0000115",
    "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
    "http://www.geneontology.org/formats/oboInOwl#id",
)
AUTHORITIES = (
    {
        "iri": "http://purl.obolibrary.org/obo/OBI_9991118",
        "file": "obi-edit.owl",
        "url": "https://raw.githubusercontent.com/obi-ontology/obi/"
        "b29b902bc7f2a7b154d96185b3ae2865e9a9de04/src/ontology/obi-edit.owl",
        "sha256": "7f0ab45e156bf2f9aff153b2a3cb1d31a7d73835439a906029526cb2ab53265b",
    },
    {
        "iri": "http://www.geneontology.org/formats/oboInOwl#created_by",
        "file": "omo-full.owl",
        "url": "https://raw.githubusercontent.com/information-artifact-ontology/"
        "ontology-metadata/82052f5c32483949c845dc86e10d0a18bf3a754b/omo-full.owl",
        "sha256": "ad312195a6c98828ed5bbdc313261498ae5e809305a95629c3925c3699e72ccf",
    },
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_authorities(evidence_dir: Path) -> None:
    for authority in AUTHORITIES:
        data = (evidence_dir / authority["file"]).read_bytes()
        if _sha(data) != authority["sha256"]:
            raise ValueError(f"Upstream declaration hash mismatch: {authority['file']}")
        declarations = [
            node
            for node in ET.fromstring(data).iter(f"{{{OWL}}}AnnotationProperty")
            if node.get(f"{{{RDF}}}about") == authority["iri"]
        ]
        if len(declarations) != 1:
            raise ValueError("Pinned upstream document lacks a unique annotation declaration")


def declaration_derivative(original: bytes) -> tuple[bytes, int, bytes, dict[str, int]]:
    """Insert declarations only; retain lexical values, blank nodes and source order."""
    root = ET.fromstring(original)
    if root.tag != f"{{{RDF}}}RDF" or original.count(b"</rdf:RDF>") != 1:
        raise ValueError("Expected a DOID RDF/XML document with one rdf:RDF closing tag")
    counts = {}
    for authority in AUTHORITIES:
        iri = authority["iri"]
        if any(node.get(f"{{{RDF}}}about") == iri for node in root):
            raise ValueError(f"Property already has a top-level description: {iri}")
        namespace, local = iri.rsplit("#" if "#" in iri else "/", 1)
        namespace += "#" if "#" in iri else "/"
        uses = list(root.iter(f"{{{namespace}}}{local}"))
        if not uses or any(len(node) or node.get(f"{{{RDF}}}resource") for node in uses):
            raise ValueError(f"Expected existing literal annotation uses: {iri}")
        counts[iri] = len(uses)
    insertion = "".join(
        f'    <owl:AnnotationProperty xmlns:owl="{OWL}" xmlns:rdf="{RDF}" '
        f'rdf:about="{authority["iri"]}"/>\n'
        for authority in AUTHORITIES
    ).encode()
    offset = original.index(b"</rdf:RDF>")
    return original[:offset] + insertion + original[offset:], offset, insertion, counts


def _write_immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing artifact differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(source: Path, output_dir: Path, evidence_dir: Path, expected_sha256: str) -> dict:
    original = source.read_bytes()
    if _sha(original) != expected_sha256:
        raise ValueError("Original ontology hash mismatch")
    verify_authorities(evidence_dir)
    derivative, offset, insertion, counts = declaration_derivative(original)
    original_path = output_dir / "original.owl"
    derivative_path = output_dir / "doid.annotation-declarations.owl"
    if source.resolve() in {original_path.resolve(), derivative_path.resolve()}:
        raise ValueError("Source must remain outside derivative output paths")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_immutable(original_path, original)
    _write_immutable(derivative_path, derivative)
    payload = {
        "schema": 1,
        "kind": "explicit_annotation_declaration_derivative",
        "source": {"path": str(source.resolve()), "sha256": _sha(original)},
        "original_copy": {"path": str(original_path.resolve()), "sha256": _sha(original)},
        "derivative": {"path": str(derivative_path.resolve()), "sha256": _sha(derivative)},
        "upstream_authorities": list(AUTHORITIES),
        "literal_uses": counts,
        "insertion_offset": offset,
        "insertion_bytes": len(insertion),
        "insertion_sha256": _sha(insertion),
        "inserted_declarations": 2,
        "deleted_source_bytes": 0,
        "original_reconstructed_sha256": _sha(
            derivative[:offset] + derivative[offset + len(insertion) :]
        ),
        "uses_alignment_labels": False,
        "validation": "Not implied by normalization; strict loading must pass separately.",
    }
    freeze_json(output_dir / "normalization.json", payload)
    return payload


def ext_declaration_derivative(original: bytes, evidence: bytes) -> tuple[bytes, bytes]:
    """Localize three declarations already asserted by the same pinned OWL closure."""
    root, declared_root = ET.fromstring(original), ET.fromstring(evidence)
    if root.tag != f"{{{RDF}}}RDF" or original.count(b"</rdf:RDF>") != 1:
        raise ValueError("Expected one ext.owl RDF/XML root")
    for iri in EXT_IRIS:
        declarations = [
            node
            for node in declared_root.iter(f"{{{OWL}}}AnnotationProperty")
            if node.get(f"{{{RDF}}}about") == iri
        ]
        if len(declarations) != 1:
            raise ValueError(f"Missing unique same-closure annotation declaration: {iri}")
        for tree in (root, declared_root):
            for node in tree:
                if node.get(f"{{{RDF}}}about") != iri:
                    continue
                if tree is root or node.tag in {
                    f"{{{OWL}}}ObjectProperty",
                    f"{{{OWL}}}DatatypeProperty",
                }:
                    raise ValueError(f"Existing or conflicting ext property declaration: {iri}")
                if any(
                    child.tag == f"{{{RDF}}}type"
                    and child.get(f"{{{RDF}}}resource")
                    in {OWL + "ObjectProperty", OWL + "DatatypeProperty"}
                    for child in node
                ):
                    raise ValueError(f"Conflicting same-closure property type: {iri}")
    insertion = "".join(
        f'    <owl:AnnotationProperty xmlns:owl="{OWL}" xmlns:rdf="{RDF}" rdf:about="{iri}"/>\n'
        for iri in EXT_IRIS
    ).encode()
    offset = original.index(b"</rdf:RDF>")
    return original[:offset] + insertion + original[offset:], insertion


def prepare_ext_annotations(output_dir: Path) -> dict:
    """Preserve ext.owl and freeze a resolver map selecting its additive derivative."""
    original_path = output_dir / "ext.owl"
    original = original_path.read_bytes()
    evidence_path = output_dir / "original.owl"
    evidence = evidence_path.read_bytes()
    if _sha(original) != "7584b8971dc49b98819c00335e4ce1421682951a80ca4e2ef7f8c743221c9147":
        raise ValueError("ext.owl differs from the pinned 2026-05-30 release")
    if _sha(evidence) != "611355c445537fcf4bae2c519f1b3598af5a8fea793274316e35525b7d05e945":
        raise ValueError("Declaration evidence differs from the pinned DOID root")
    derivative, insertion = ext_declaration_derivative(original, evidence)
    derivative_path = output_dir / "ext.annotation-declarations.owl"
    _write_immutable(derivative_path, derivative)
    offset = original.index(b"</rdf:RDF>")
    payload = {
        "schema": 1,
        "kind": "same_closure_annotation_declaration_localization",
        "source": {"path": str(original_path.resolve()), "sha256": _sha(original)},
        "evidence": {"path": str(evidence_path.resolve()), "sha256": _sha(evidence)},
        "derivative": {"path": str(derivative_path.resolve()), "sha256": _sha(derivative)},
        "inserted_annotation_declarations": list(EXT_IRIS),
        "insertion_offset": offset,
        "insertion_bytes": len(insertion),
        "original_reconstructed_sha256": _sha(
            derivative[:offset] + derivative[offset + len(insertion) :]
        ),
        "deleted_source_bytes": 0,
        "new_declarations_in_complete_closure": 0,
        "validation": "Not implied by normalization; strict closure loading must pass separately.",
    }
    freeze_json(output_dir / "ext.normalization.json", payload)
    mapping = json.loads((output_dir / "import-map.json").read_text())
    import_iri = "http://purl.obolibrary.org/obo/doid/obo/ext.owl"
    if mapping[import_iri]["sha256"] != _sha(original):
        raise ValueError("Import map differs from original ext.owl")
    mapping[import_iri] = payload["derivative"]
    freeze_json(output_dir / "import-map.normalized.json", mapping)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--complete-ext-closure", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(prepare(args.source, args.output_dir, args.evidence_dir, args.expected_sha256))
    )

    if args.complete_ext_closure:
        print(json.dumps(prepare_ext_annotations(args.output_dir)))


if __name__ == "__main__":
    main()
