"""Typed OWL context values, exact annotation adapters, and conservative render templates."""

from __future__ import annotations

import dataclasses
import enum
import json
import zlib
from collections.abc import Iterable, Mapping
from typing import Any

CONTEXT_SCHEMA = "exact-context/2"
KINDS = {"class", "object_property", "data_property", "individual"}
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OBO = "http://purl.obolibrary.org/obo/"
OBO_IN_OWL = "http://www.geneontology.org/formats/oboInOwl#"
NCIT = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"
FMA = "http://purl.org/sig/ont/fma/"
# Exact predicates, deliberately no suffix/substring heuristics.
ANNOTATION_REGISTRY: dict[str, tuple[str, str | None]] = {
    RDFS + "label": ("labels", "preferred"),
    RDFS + "comment": ("comments", None),
    NCIT + "P108": ("labels", "preferred"),
    NCIT + "P97": ("definitions", None),
    NCIT + "P325": ("alternate_definitions", None),
    NCIT + "P90": ("synonyms", "full"),
    NCIT + "P378": ("definition_citations", "source"),
    NCIT + "P381": ("definition_citations", "attribution"),
    NCIT + "P383": ("term_metadata", "term_type"),
    NCIT + "P384": ("term_metadata", "term_source"),
    NCIT + "P386": ("term_metadata", "subsource"),
    NCIT + "P391": ("term_metadata", "source_date"),
    OBO + "IAO_0000115": ("definitions", None),
    OBO_IN_OWL + "hasDefinition": ("definitions", None),
    OBO_IN_OWL + "hasDbXref": ("xrefs", None),
    FMA + "definition": ("definitions", None),
    FMA + "synonym": ("synonyms", "related"),
    "http://purl.org/sig/ont/fma.owl#definition": ("definitions", None),
    "http://purl.org/sig/ont/fma.owl#synonym": ("synonyms", "related"),
}
# Opaque codes are mapped from the pinned NCIT AnnotationProperty declarations.
for _name in (
    "P100",
    "P102",
    "P175",
    "P207",
    "P208",
    "P210",
    "P211",
    "P215",
    "P216",
    "P315",
    "P319",
    "P320",
    "P321",
    "P329",
    "P330",
    "P331",
    "P332",
    "P334",
    "P354",
    "P362",
    "P367",
    "P368",
    "P369",
    "P385",
    "P387",
    "P399",
    "P400",
    "P93",
):
    ANNOTATION_REGISTRY[NCIT + _name] = ("xrefs", None)
for _name in ("P375", "P392", "P393", "P394", "P395", "P396", "P397"):
    ANNOTATION_REGISTRY[NCIT + _name] = ("mappings", None)
for _name, _scope in (
    ("hasExactSynonym", "exact"),
    ("hasBroadSynonym", "broad"),
    ("hasNarrowSynonym", "narrow"),
    ("hasRelatedSynonym", "related"),
):
    ANNOTATION_REGISTRY[OBO_IN_OWL + _name] = ("synonyms", _scope)
for _name in ("exactMatch", "closeMatch", "broadMatch", "narrowMatch", "relatedMatch"):
    ANNOTATION_REGISTRY["http://www.w3.org/2004/02/skos/core#" + _name] = ("mappings", None)
CATEGORIES = (
    "labels",
    "definitions",
    "alternate_definitions",
    "definition_citations",
    "term_metadata",
    "synonyms",
    "comments",
    "xrefs",
    "mappings",
    "annotations",
    "hierarchy",
    "restrictions",
    "types",
    "assertions",
    "domains",
    "ranges",
    "characteristics",
    "axioms",
    "usage",
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_PAYLOAD_MAGIC = b"EXACTCTXZ1\0"


def encode_payload(payload: dict[str, Any]) -> bytes:
    """Losslessly compress rich axiom detail; fact/search rows stay independently readable."""
    return _PAYLOAD_MAGIC + zlib.compress(_json(payload).encode("utf-8"))


def decode_payload(value: str | bytes, *, max_bytes: int | None = None) -> dict[str, Any]:
    """Read plain or versioned compressed detail with a decoded-byte request budget."""
    if isinstance(value, str):
        decoded = value.encode("utf-8")
    else:
        if not value.startswith(_PAYLOAD_MAGIC):
            raise ValueError("Unsupported context axiom codec")
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(
            value[len(_PAYLOAD_MAGIC) :], max_bytes + 1 if max_bytes is not None else 0
        )
        if max_bytes is not None and len(decoded) > max_bytes:
            raise ValueError("Decoded axiom exceeds its byte budget")
        if not decoder.eof or decoder.unused_data:
            raise ValueError("Invalid compressed context axiom")
    if max_bytes is not None and len(decoded) > max_bytes:
        raise ValueError("Decoded axiom exceeds its byte budget")
    payload = json.loads(decoded)
    if not isinstance(payload, dict):
        raise ValueError("Context axiom detail must be an object")
    return payload


def decoded_chunks(chunks: Iterable[bytes], *, compressed: bool, chunk_bytes: int):
    """Incrementally decode exact local artifacts without buffering large expressions."""
    if not compressed:
        yield from chunks
        return
    decoder = zlib.decompressobj()
    header = b""
    header_done = False
    for chunk in chunks:
        if not header_done:
            header += chunk
            if len(header) < len(_PAYLOAD_MAGIC):
                continue
            if not header.startswith(_PAYLOAD_MAGIC):
                raise ValueError("Unsupported context axiom codec")
            chunk, header = header[len(_PAYLOAD_MAGIC) :], b""
            header_done = True
        while chunk:
            block = decoder.decompress(chunk, chunk_bytes)
            if block:
                yield block
            chunk = decoder.unconsumed_tail
    if not header_done or not decoder.eof or decoder.unused_data:
        raise ValueError("Invalid compressed context axiom")


def typed_node(value: Any) -> Any:
    """Encode the public typed structural model without losing any constructor field."""
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__name__,
            **{
                field.name: typed_node(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, bytes):
        return {"encoding": "hex", "value": value.hex()}
    if isinstance(value, Mapping):
        return {str(key): typed_node(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return [typed_node(item) for item in value]
    except TypeError as error:
        raise TypeError(f"Unsupported pyowl structural value: {type(value).__name__}") from error


def _iri(node: Any) -> str | None:
    if not isinstance(node, dict):
        return None
    if node.get("type") == "IRI":
        return str(node["value"])
    return _iri(node.get("iri"))


def _kind(node: Mapping[str, Any]) -> str | None:
    kind = node.get("kind")
    return "individual" if kind == "named_individual" else kind


def _entities(node: Any):
    if isinstance(node, dict):
        kind = _kind(node)
        if kind in KINDS and (iri := _iri(node)):
            yield iri, kind
        for value in node.values():
            yield from _entities(value)
    elif isinstance(node, list):
        for value in node:
            yield from _entities(value)


def expression_text(node: Mapping[str, Any]) -> dict[str, str]:
    """Render conservative constructor templates, retaining structured fallback syntax."""
    kind = node.get("type", "unknown")
    iri = _iri(node)
    if iri:
        return {"status": "available", "text": f"<{iri}>"}
    if kind == "Literal":
        suffix = (
            ("@" + node["language"])
            if node.get("language")
            else ("^^<" + str(_iri(node.get("datatype"))) + ">")
        )
        return {"status": "available", "text": json.dumps(node["lexical_form"]) + suffix}

    def rendered(item: Mapping[str, Any]) -> str:
        return expression_text(item)["text"]

    if kind in {"ObjectIntersectionOf", "ObjectUnionOf", "DataIntersectionOf", "DataUnionOf"}:
        join = " and " if "Intersection" in kind else " or "
        text = "(" + join.join(rendered(item) for item in node["operands"]) + ")"
    elif kind in {"ObjectComplementOf", "DataComplementOf"}:
        text = "not (" + rendered(node.get("operand", node.get("data_range"))) + ")"
    elif kind == "ObjectInverseOf":
        text = "inverse of " + rendered(node["property"])
    elif kind.endswith("SomeValuesFrom") or kind.endswith("AllValuesFrom"):
        quantifier = "some" if "Some" in kind else "only"
        prop = node.get("property", node.get("properties"))
        text = f"{rendered(prop) if isinstance(prop, dict) else _json(prop)} {quantifier} {rendered(node['filler'])}"
    elif kind.endswith("HasValue"):
        text = rendered(node["property"]) + " has value " + rendered(node["value"])
    elif kind.endswith("Cardinality"):
        quantifier = "at least" if "Min" in kind else "at most" if "Max" in kind else "exactly"
        text = f"{rendered(node['property'])} {quantifier} {node['cardinality']}"
        if node.get("filler"):
            text += " of " + rendered(node["filler"])
    else:
        return {"status": "unsupported", "text": _json(node)}
    return {"status": "available", "text": text}


def _category(ast: dict[str, Any]) -> str:
    kind = ast["type"]
    if kind == "AnnotationAssertion":
        return ANNOTATION_REGISTRY.get(_iri(ast["property"]) or "", ("annotations", None))[0]
    if kind in {"SubClassOf", "EquivalentClasses", "DisjointUnion"}:
        terms = ast.get("expressions", ast.get("class_expressions", ast.get("operands", [])))
        if kind == "SubClassOf":
            terms = [ast["sub_class"], ast["super_class"]]
        return "restrictions" if any(term.get("type") != "Class" for term in terms) else "hierarchy"
    if kind == "ClassAssertion":
        return "types"
    if "PropertyAssertion" in kind or kind in {"SameIndividual", "DifferentIndividuals"}:
        return "assertions"
    if kind.endswith("Domain"):
        return "domains"
    if kind.endswith("Range"):
        return "ranges"
    if "Propert" in kind and kind != "Declaration":
        return "characteristics"
    return "axioms"


def _ref(version: str, iri: str, kind: str) -> dict[str, str]:
    return {"ontology_version_id": version, "iri": iri, "kind": kind}


def _ref_dict(entity: Any) -> dict[str, Any]:
    return entity.model_dump(mode="json") if hasattr(entity, "model_dump") else dict(entity)


def _allowed(policy: Any, version: str, category: str) -> bool:
    if policy is None:
        return True
    return bool(
        policy.allows_fact(
            {
                "ontology_version_id": version,
                "category": category,
                "subject": _ref(version, "urn:policy-check", "class"),
            },
            category=category,
        )
    )


def _edge_candidates(ast: dict[str, Any]):
    """Yield named adjacency and explicit conservative structural consequences."""
    kind = ast["type"]
    if kind in {"SubClassOf", "SubObjectPropertyOf", "SubDataPropertyOf"}:
        child = ast.get("sub_class", ast.get("sub_property"))
        parent = ast.get("super_class", ast.get("super_property"))
        if not isinstance(child, dict) or not isinstance(parent, dict):
            return
        if _iri(child) and _iri(parent) and _kind(child) in KINDS and _kind(child) == _kind(parent):
            yield child, parent, "literal_asserted", "subclass", None
            yield child, parent, "structural_navigation", "subclass", "named_subsumption"
        elif _iri(child) and parent.get("type") == "ObjectIntersectionOf":
            for operand in parent["operands"]:
                if operand.get("type") == "Class":
                    yield child, operand, "structural_navigation", "subclass", "subclass_conjunct"
    if kind in {"EquivalentClasses", "EquivalentObjectProperties", "EquivalentDataProperties"}:
        operands = ast.get(
            "expressions",
            ast.get("class_expressions", ast.get("properties", ast.get("operands", []))),
        )
        named = [item for item in operands if _iri(item) and _kind(item) in KINDS]
        for child in named:
            for parent in named:
                if child != parent:
                    yield child, parent, "literal_asserted", "equivalent", None
                    yield child, parent, "structural_navigation", "equivalent", "named_equivalence"
            for expression in operands:
                if expression.get("type") == "ObjectIntersectionOf":
                    for parent in expression["operands"]:
                        if parent.get("type") == "Class":
                            yield child, parent, "structural_navigation", "subclass", "equivalent_conjunct"


def _visible_node(node: Any, policy: Any, version: str) -> Any:
    """Apply mapping visibility to nested annotation qualifiers before rendering/export."""
    if isinstance(node, list):
        return [
            _visible_node(item, policy, version)
            for item in node
            if not (
                isinstance(item, dict)
                and item.get("type") == "Annotation"
                and not _allowed(
                    policy,
                    version,
                    ANNOTATION_REGISTRY.get(
                        _iri(item.get("property")) or "", ("annotations", None)
                    )[0],
                )
            )
        ]
    if isinstance(node, dict):
        return {key: _visible_node(value, policy, version) for key, value in node.items()}
    return node


def _fact_view(payload: dict[str, Any], subject: dict[str, Any], policy: Any) -> dict[str, Any]:
    """Project stored rich axioms to the shared Fact primitive without dropping detail links."""
    value = payload.get("value")
    if isinstance(value, dict) and value.get("type") == "Literal":
        term = {
            "term_type": "literal",
            "lexical_form": value["lexical_form"],
            "datatype": _iri(value.get("datatype")),
            "language": value.get("language"),
        }
    elif isinstance(value, dict) and _iri(value):
        term = {"term_type": "iri", "iri": _iri(value)}
    else:
        term = {"term_type": "expression_ref", "expression_id": payload["axiom_id"]}
    ast = payload.get("ast", {})
    source_iri = _iri(ast.get("source"))
    target_iri = _iri(ast.get("target"))
    directions = []
    if source_iri == subject["iri"] and subject["kind"] == "individual":
        directions.append("outgoing")
    if target_iri == subject["iri"] and subject["kind"] == "individual":
        directions.append("incoming")
    origins = []
    for origin in payload["origins"]:
        span = origin.get("span")
        origins.append(
            {
                "document_sha256": origin["source_sha256"],
                "axiom_id": payload["original_axiom_digest"],
                "source_span": (
                    {"start": span["byte_start"], "end": span["byte_end"], "unit": "byte"}
                    if span
                    and span.get("byte_start") is not None
                    and span.get("byte_end") is not None
                    else None
                ),
                "provenance_status": "recorded",
            }
        )
    return {
        "id": payload["id"],
        "fact_id": payload["id"],
        "subject": subject,
        "predicate_iri": payload.get("predicate"),
        "value": term,
        "axiom_ref": payload["axiom_id"],
        "axiom_id": payload["axiom_id"],
        "interpretation": payload["interpretation"]["kind"],
        "origins": origins,
        "premise_ids": [],
        "derivation_id": None,
        "category": payload["category"],
        "qualifiers": _visible_node(
            payload.get("qualifiers", []), policy, subject["ontology_version_id"]
        ),
        "qualifier_categories": [
            {
                "predicate_iri": _iri(item.get("property")),
                "category": ANNOTATION_REGISTRY.get(
                    _iri(item.get("property")) or "", ("annotations", None)
                )[0],
                "role": ANNOTATION_REGISTRY.get(
                    _iri(item.get("property")) or "", ("annotations", None)
                )[1],
            }
            for item in _visible_node(
                payload.get("qualifiers", []), policy, subject["ontology_version_id"]
            )
        ],
        "synonym_scope": payload.get("synonym_scope"),
        "assertion_directions": directions,
        "reference_role": (
            "asserted_type"
            if ast.get("type") == "ClassAssertion"
            and _iri(ast.get("individual")) == subject["iri"]
            and subject["kind"] == "individual"
            else "entity_reference"
        ),
        "availability": "available",
    }
