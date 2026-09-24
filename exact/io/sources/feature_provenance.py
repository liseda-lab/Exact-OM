"""Exact-term provenance for the supported named and existential feature projection.

The original axiom hash and syntax are retained independently of feature grouping.
Unknown projector rules stay unavailable; labels are never used for reconciliation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pyowl_core as owl

RDFS = "http://www.w3.org/2000/01/rdf-schema#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def _iri(node):
    if isinstance(node, owl.IRI):
        return node.value
    return getattr(getattr(node, "iri", None), "value", None)


class FeatureProvenance:
    def __init__(self, source):
        self.edges = defaultdict(list)
        self.attributes = defaultdict(list)
        snapshot = getattr(source, "owl_snapshot", None)
        if not callable(snapshot):
            return
        self._loaded_subjects = set()
        self._indexed = None
        ontology = snapshot()
        if callable(getattr(ontology, "view", None)):
            self._indexed = (
                ontology.view(
                    owl.AxiomTypeIndex, include_origins=False, require_native_pipeline=True
                ),
                ontology.view(
                    owl.AnnotationAssertionIndex,
                    include_origins=False,
                    include_nested=False,
                    require_native_pipeline=True,
                ),
            )
        else:
            # Small protocol fixtures may supply a stream without the optional native index.
            for axiom in ontology.iter_axioms():
                self._record_axiom(axiom)

    def _ensure_subject(self, subject, kind):
        if (
            not subject
            or not getattr(self, "_indexed", None)
            or (subject, kind) in self._loaded_subjects
        ):
            return
        self._loaded_subjects.add((subject, kind))
        factory = {
            "class": owl.Class,
            "object_property": owl.ObjectProperty,
            "data_property": owl.DataProperty,
            "individual": owl.NamedIndividual,
        }[kind]
        entity = factory(owl.IRI(subject))
        axioms, annotations = self._indexed
        for axiom in annotations.assertions(owl.IRI(subject)):
            self._record_axiom(axiom)
        for axiom_type in (
            owl.SubClassOf,
            owl.EquivalentClasses,
            owl.SubObjectPropertyOf,
            owl.SubDataPropertyOf,
            owl.ClassAssertion,
            owl.ObjectPropertyDomain,
            owl.ObjectPropertyRange,
            owl.DataPropertyDomain,
            owl.DataPropertyRange,
        ):
            for axiom in axioms.iter(axiom_type, referencing=entity):
                self._record_axiom(axiom)

    def _record_axiom(self, axiom):
        reference = {
            "axiom_id": "sha256:" + owl.structural_hexdigest(axiom),
            "original_syntax": repr(axiom),
            "syntax_format": "pyowl_core_repr",
        }
        if isinstance(axiom, owl.AnnotationAssertion) and isinstance(axiom.value, owl.Literal):
            value = axiom.value
            datatype = _iri(value.datatype)
            facade_datatype = (
                None
                if value.language is not None
                or datatype
                in {
                    "http://www.w3.org/2001/XMLSchema#string",
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#PlainLiteral",
                }
                else datatype
            )
            reference["literal_term"] = {
                "lexical_form": value.lexical_form,
                "datatype": datatype,
                "language": value.language,
            }
            key = (
                _iri(axiom.subject),
                _iri(axiom.property),
                value.lexical_form,
                facade_datatype,
                value.language,
            )
            self.attributes[key].append(reference)
        elif isinstance(axiom, owl.SubClassOf) and isinstance(axiom.sub_class, owl.Class):
            self._class_edges(_iri(axiom.sub_class), axiom.super_class, reference, equivalent=False)
        elif isinstance(axiom, owl.EquivalentClasses):
            for owner in axiom.expressions:
                if isinstance(owner, owl.Class):
                    for expression in axiom.expressions:
                        if expression != owner:
                            self._class_edges(_iri(owner), expression, reference, equivalent=True)
        elif type(axiom).__name__ in ("SubObjectPropertyOf", "SubDataPropertyOf"):
            self._edge(
                _iri(axiom.sub_property),
                RDFS + "subPropertyOf",
                _iri(axiom.super_property),
                reference,
                "asserted",
            )
        elif isinstance(axiom, owl.ClassAssertion) and isinstance(
            axiom.class_expression, owl.Class
        ):
            self._edge(
                _iri(axiom.individual),
                RDF_TYPE,
                _iri(axiom.class_expression),
                reference,
                "asserted",
            )
        elif type(axiom).__name__ in (
            "ObjectPropertyDomain",
            "DataPropertyDomain",
            "ObjectPropertyRange",
            "DataPropertyRange",
        ):
            role = "domain" if hasattr(axiom, "domain") else "range"
            self._edge(
                _iri(axiom.property),
                RDFS + role,
                _iri(getattr(axiom, role)),
                reference,
                "asserted",
            )

    def _edge(self, subject, predicate, obj, reference, interpretation, rule=None):
        if subject and predicate and obj:
            self.edges[subject, predicate, obj].append(
                {**reference, "interpretation": interpretation, "rule": rule}
            )

    def _class_edges(self, subject, expression, reference, *, equivalent):
        if isinstance(expression, owl.Class):
            self._edge(
                subject,
                RDFS + "subClassOf",
                _iri(expression),
                reference,
                "structurally_derived" if equivalent else "asserted",
                "equivalent_named_operand" if equivalent else None,
            )
        elif isinstance(expression, owl.ObjectIntersectionOf):
            for operand in expression.operands:
                self._class_edges(subject, operand, reference, equivalent=True)
        # This is a projection of an existential occurrence, not a binary OWL assertion.
        for node in owl.walk(expression):
            if isinstance(node, owl.ObjectSomeValuesFrom):
                for filler in owl.walk(node.filler):
                    if isinstance(filler, owl.Class):
                        self._edge(
                            subject,
                            _iri(node.property),
                            _iri(filler),
                            reference,
                            "projected",
                            "existential_named_filler_projection",
                        )

    def enrich(self, item: dict[str, Any], *, kind: str, family=None, predicates=()):
        self._ensure_subject(item.get("entity_iri", item.get("subject_iri")), kind)
        result = dict(item)
        predicate = item.get("rel_iri")
        if not predicate and family == "is_a":
            predicate = (
                RDF_TYPE
                if kind == "individual"
                else RDFS + "subPropertyOf" if kind.endswith("_property") else RDFS + "subClassOf"
            )
        if predicate in ("domain", "range"):
            result["role"] = "property_" + predicate
            predicate = RDFS + predicate
        references = []
        if item.get("prop_iri") or item.get("property_iri"):
            key = (
                item.get("entity_iri"),
                item.get("prop_iri", item.get("property_iri")),
                item.get("literal_lexical_form", item.get("value")),
                item.get("datatype"),
                item.get("language"),
            )
            references = self.attributes.get(key, [])
        else:
            for pred in [predicate] if predicate else predicates:
                references.extend(
                    self.edges.get((item.get("subject_iri"), pred, item.get("object_iri")), [])
                )
            if predicate:
                result["rel_iri"] = predicate
        references = list({row["axiom_id"]: row for row in references}.values())
        # Type closure needs its complete multi-axiom proof; never borrow direct type status.
        if item.get("type_closure"):
            references = []
        literal_terms = [row["literal_term"] for row in references if row.get("literal_term")]
        if literal_terms:
            result["literal_terms"] = literal_terms
        result["grouped_semantic_features"] = [
            self.enrich(feature, kind=kind, family=family, predicates=predicates)
            for feature in item.get("grouped_semantic_features", [])
        ]
        result["entity_kind"] = kind
        result["source_axiom_refs"] = sorted(row["axiom_id"] for row in references)
        result["axiom_origins"] = references
        result["provenance_status"] = "available" if references else "provenance_unavailable"
        result["interpretation"] = (
            references[0].get("interpretation", "asserted")
            if references
            else "structurally_derived" if family else "projected"
        )
        result["derivation"] = {
            "provider": "exact",
            "version": "feature-provenance/1",
            "rules": sorted({row["rule"] for row in references if row.get("rule")}),
            "premises": result["source_axiom_refs"],
            "premises_unavailable": not bool(references),
        }
        return result
