#!/usr/bin/env python3
"""Capture deterministic ontology-backend snapshots.

The default ``exact`` backend records the installed Java-free implementation.
``--backend mowl`` is a release-evidence mode for the historical backend; run it
from an isolated Python 3.10 environment containing ``mowl-borg==1.0.1`` and a
JDK.  All mOWL imports are lazy so this tool remains usable in a normal Exact
installation, where Java is deliberately absent.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import zstandard

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PART_OF = "http://purl.obolibrary.org/obo/BFO_0000050"
HAS_PART = "http://purl.obolibrary.org/obo/BFO_0000051"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
OWL_DEPRECATED = "http://www.w3.org/2002/07/owl#deprecated"
USE_IN_ALIGNMENT = "http://oaei.ontologymatching.org/bio-ml/ann/use_in_alignment"
RDF_PLAIN_LITERAL = "http://www.w3.org/1999/02/22-rdf-syntax-ns#PlainLiteral"


def _capture_exact(path: Path) -> dict[str, Any]:
    from exact.core.entities.kinds import EntityKind
    from exact.ontology import load_ontology

    source = load_ontology(path)
    classes = list(source.entities(EntityKind.CLASS))
    properties = sorted(
        set(source.entities(EntityKind.OBJECT_PROPERTY))
        | set(source.entities(EntityKind.DATA_PROPERTY))
    )
    families = {"is_a": (), "part_of": (PART_OF,), "has_part": (HAS_PART,)}
    return {
        "schema_version": 1,
        "origin_name": path.name,
        "classes": classes,
        "labels": {iri: source.labels(iri) or [source.short_form(iri)] for iri in classes},
        "annotations": {
            iri: [asdict(value) for value in source.annotations(iri)] for iri in classes
        },
        "attributes": {iri: [asdict(value) for value in source.attributes(iri)] for iri in classes},
        "hierarchy": {iri: source.hierarchy_bundle(iri, families) for iri in classes},
        "projection": {
            "taxonomy": [edge.astuple() for edge in source.projection_edges(method="taxonomy")],
            "owl2vecstar": [
                edge.astuple() for edge in source.projection_edges(method="owl2vecstar")
            ],
            "owl2vecstar_literals": [
                edge.astuple()
                for edge in source.projection_edges(method="owl2vecstar", include_literals=True)
            ],
        },
        "excluded_from_alignment": sorted(source.excluded_from_alignment()),
        "property_domains": {prop: source.property_domains(prop) for prop in properties},
        "property_ranges": {prop: source.property_ranges(prop) for prop in properties},
    }


def _java_items(value: Any) -> list[Any]:
    """Convert a Java collection/stream-like value into a Python list."""

    if value is None:
        return []
    try:
        return list(value)
    except Exception:  # Java collection proxies expose generated exception types.
        iterator = value.iterator()
        result: list[Any] = []
        while iterator.hasNext():
            result.append(iterator.next())
        return result


def _named_class_iri(expression: Any) -> str | None:
    if expression is None:
        return None
    try:
        if expression.isOWLClass():
            return str(expression.asOWLClass().getIRI().toString())
    except Exception:  # JPype maps OWL API failures to generated exception classes.
        pass
    try:
        return str(expression.asOWLClass().getIRI().toString())
    except Exception:
        return None


def _property_iri(expression: Any) -> str | None:
    if expression is None:
        return None
    try:
        return str(expression.getNamedProperty().getIRI().toString())
    except Exception:
        pass
    try:
        return str(expression.asOWLObjectProperty().getIRI().toString())
    except Exception:
        return None


def _restriction_targets(expression: Any) -> list[tuple[str, str]]:
    """Extract the relation families observed by the legacy dataset walker."""

    result: list[tuple[str, str]] = []
    expression_type = ""
    try:
        expression_type = str(expression.getClassExpressionType().toString())
    except Exception:
        pass
    if "Intersection" in expression_type or hasattr(expression, "getOperands"):
        try:
            operands = _java_items(expression.getOperandsAsList())
        except Exception:
            operands = _java_items(expression.getOperands())
        for operand in operands:
            result.extend(_restriction_targets(operand))
    if hasattr(expression, "getProperty") and hasattr(expression, "getFiller"):
        relation = _property_iri(expression.getProperty())
        target = _named_class_iri(expression.getFiller())
        if relation == PART_OF and target:
            result.append(("part_of", target))
        elif relation == HAS_PART and target:
            result.append(("has_part", target))
    return result


def _axiom_restriction_targets(ontology: Any, owl_class: Any) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for axiom in _java_items(ontology.getSubClassAxiomsForSubClass(owl_class)):
        result.extend(_restriction_targets(axiom.getSuperClass()))
    for axiom in _java_items(ontology.getEquivalentClassesAxioms(owl_class)):
        try:
            expressions = _java_items(axiom.getClassExpressionsAsList())
        except Exception:
            expressions = _java_items(axiom.getClassExpressions())
        for expression in expressions:
            if _named_class_iri(expression) == str(owl_class.getIRI().toString()):
                continue
            result.extend(_restriction_targets(expression))
    return sorted(set(result))


def _annotation_value(annotation: Any) -> dict[str, Any]:
    property_iri = str(annotation.getProperty().getIRI().toString())
    value = annotation.getValue()
    if value.isLiteral():
        literal = value.asLiteral().get()
        lang = str(literal.getLang()) if literal.hasLang() else None
        datatype: str | None = str(literal.getDatatype().getIRI().toString())
        if lang or datatype == RDF_PLAIN_LITERAL:
            datatype = None
        return {
            "property_iri": property_iri,
            "value": str(literal.getLiteral()),
            "lang": lang,
            "datatype": datatype,
            "is_literal": True,
        }
    try:
        rendered = str(value.asIRI().get().toString())
    except Exception:
        rendered = str(value)
    return {
        "property_iri": property_iri,
        "value": rendered,
        "lang": None,
        "datatype": None,
        "is_literal": False,
    }


def _annotation_sort_key(value: dict[str, Any]) -> tuple[Any, ...]:
    return (
        value["property_iri"],
        value["value"],
        value["lang"] or "",
        value["datatype"] or "",
        value["is_literal"],
    )


def _short_form(iri: str) -> str:
    parsed = urlsplit(iri)
    if parsed.fragment:
        return unquote(parsed.fragment)
    path = parsed.path.rstrip("/")
    if path:
        return unquote(path.rsplit("/", 1)[-1])
    return iri.rsplit(":", 1)[-1]


def _expression_iri(expression: Any) -> str:
    named = _named_class_iri(expression)
    if named is not None:
        return named
    try:
        return str(expression.asOWLDatatype().getIRI().toString())
    except Exception:
        return str(expression)


def _projection(projector: Any, ontology: Any) -> list[tuple[str, str, str]]:
    return sorted(
        {(str(edge.src), str(edge.rel), str(edge.dst)) for edge in projector.project(ontology)}
    )


def _capture_mowl(path: Path, *, jvm_heap: str) -> dict[str, Any]:
    try:
        import jpype
        import mowl
    except ImportError as exc:  # pragma: no cover - release-evidence environment only
        raise RuntimeError(
            "mOWL baseline capture needs Python 3.10, a JDK, and "
            "mowl-borg==1.0.1 in an isolated environment"
        ) from exc

    if not jpype.isJVMStarted():
        mowl.init_jvm(jvm_heap)
    from mowl.datasets import PathDataset
    from mowl.projection import OWL2VecStarProjector, TaxonomyProjector
    from org.semanticweb.elk.owlapi import ElkReasonerFactory
    from org.semanticweb.owlapi.model import IRI
    from org.semanticweb.owlapi.search import EntitySearcher

    ontology = PathDataset(str(path)).ontology
    factory = ontology.getOWLOntologyManager().getOWLDataFactory()
    reasoner = ElkReasonerFactory().createReasoner(ontology)
    classes = sorted(
        str(owl_class.getIRI().toString())
        for owl_class in _java_items(ontology.getClassesInSignature())
    )
    labels: dict[str, list[str]] = {}
    annotations: dict[str, list[dict[str, Any]]] = {}
    attributes: dict[str, list[dict[str, Any]]] = {}
    hierarchy: dict[str, dict[str, list[str]]] = {}
    excluded: set[str] = set()
    for iri in classes:
        owl_class = factory.getOWLClass(IRI.create(iri))
        values = sorted(
            (
                _annotation_value(annotation)
                for annotation in _java_items(EntitySearcher.getAnnotations(owl_class, ontology))
            ),
            key=_annotation_sort_key,
        )
        annotations[iri] = values
        labels[iri] = sorted(
            value["value"]
            for value in values
            if value["property_iri"] == RDFS_LABEL and value["is_literal"]
        ) or [_short_form(iri)]
        attributes[iri] = [
            value for value in values if value["property_iri"] != RDFS_LABEL and value["is_literal"]
        ]
        for value in values:
            lexical = value["value"].strip().lower()
            if value["property_iri"] == USE_IN_ALIGNMENT and lexical in {"false", "0"}:
                excluded.add(iri)
            elif value["property_iri"] == OWL_DEPRECATED and lexical in {"true", "1"}:
                excluded.add(iri)
        restrictions = _axiom_restriction_targets(ontology, owl_class)
        hierarchy[iri] = {
            "is_a": sorted(
                str(parent.getIRI().toString())
                for parent in reasoner.getSuperClasses(owl_class, True).getFlattened()
                if not parent.isOWLThing() and not parent.isOWLNothing()
            ),
            "part_of": [target for family, target in restrictions if family == "part_of"],
            "has_part": [target for family, target in restrictions if family == "has_part"],
        }

    object_properties = {
        str(prop.getIRI().toString())
        for prop in _java_items(ontology.getObjectPropertiesInSignature())
    }
    data_properties = {
        str(prop.getIRI().toString())
        for prop in _java_items(ontology.getDataPropertiesInSignature())
    }
    property_domains: dict[str, list[str]] = {}
    property_ranges: dict[str, list[str]] = {}
    for prop_iri in sorted(object_properties | data_properties):
        domains: list[str] = []
        ranges: list[str] = []
        if prop_iri in object_properties:
            prop = factory.getOWLObjectProperty(IRI.create(prop_iri))
            domains.extend(
                _expression_iri(axiom.getDomain())
                for axiom in _java_items(ontology.getObjectPropertyDomainAxioms(prop))
            )
            ranges.extend(
                _expression_iri(axiom.getRange())
                for axiom in _java_items(ontology.getObjectPropertyRangeAxioms(prop))
            )
        if prop_iri in data_properties:
            prop = factory.getOWLDataProperty(IRI.create(prop_iri))
            domains.extend(
                _expression_iri(axiom.getDomain())
                for axiom in _java_items(ontology.getDataPropertyDomainAxioms(prop))
            )
            ranges.extend(
                _expression_iri(axiom.getRange())
                for axiom in _java_items(ontology.getDataPropertyRangeAxioms(prop))
            )
        property_domains[prop_iri] = sorted(set(domains))
        property_ranges[prop_iri] = sorted(set(ranges))

    return {
        "schema_version": 1,
        "origin_name": path.name,
        "classes": classes,
        "labels": labels,
        "annotations": annotations,
        "attributes": attributes,
        "hierarchy": hierarchy,
        "projection": {
            "taxonomy": _projection(TaxonomyProjector(), ontology),
            "owl2vecstar": _projection(OWL2VecStarProjector(), ontology),
            "owl2vecstar_literals": _projection(
                OWL2VecStarProjector(include_literals=True), ontology
            ),
        },
        "excluded_from_alignment": sorted(excluded),
        "property_domains": property_domains,
        "property_ranges": property_ranges,
    }


def capture(
    path: Path,
    *,
    backend: str = "exact",
    jvm_heap: str = "4g",
) -> dict[str, Any]:
    """Capture ``path`` through the selected ontology backend."""

    if backend == "exact":
        return _capture_exact(path)
    if backend == "mowl":
        return _capture_mowl(path, jvm_heap=jvm_heap)
    raise ValueError(f"Unknown backend: {backend!r}")


def write_snapshot(
    path: Path,
    output_dir: Path,
    *,
    backend: str = "exact",
    jvm_heap: str = "4g",
) -> Path:
    payload = json.dumps(
        capture(path, backend=backend, jvm_heap=jvm_heap),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}.backend.json.zst"
    output_path.write_bytes(zstandard.ZstdCompressor(level=10).compress(payload))
    return output_path


def expand_inputs(paths: list[Path]) -> list[Path]:
    ontologies: list[Path] = []
    for path in paths:
        if path.is_dir():
            discovered = sorted(path.glob("*.owl"))
            if not discovered:
                raise FileNotFoundError(f"No .owl ontologies found in dataset directory {path}")
            ontologies.extend(discovered)
        else:
            ontologies.append(path)
    return ontologies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ontologies", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("exp/baselines"))
    parser.add_argument("--backend", choices=("exact", "mowl"), default="exact")
    parser.add_argument("--jvm-heap", default="4g")
    args = parser.parse_args()
    for ontology in expand_inputs(args.ontologies):
        print(
            write_snapshot(
                ontology,
                args.output_dir,
                backend=args.backend,
                jvm_heap=args.jvm_heap,
            )
        )


if __name__ == "__main__":
    main()
