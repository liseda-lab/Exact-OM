"""Participant-safe prepared explanation envelopes, independent of study answer keys.

These are frozen presentation records produced from the shared ontology engine;
structured OWL values retain the public constructor shape without a serving parser.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, StrictInt, model_validator

from ..context_semantics import ANNOTATION_REGISTRY, _category, _iri, _visible_node
from ..contracts import Score, VisibilityPolicy
from .models import EntityRef, Identifier, StrictModel
from .owl_ast import validate_annotation, validate_axiom


class ByteSourceSpan(StrictModel):
    """Half-open original-document byte interval, never arbitrary provenance metadata."""

    start: Annotated[StrictInt, Field(ge=0)]
    end: Annotated[StrictInt, Field(ge=0)]
    unit: Literal["byte"] = "byte"

    @model_validator(mode="after")
    def ordered(self):
        if self.end < self.start:
            raise ValueError("Source span end precedes its start")
        return self


class FactOrigin(StrictModel):
    document_sha256: Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
    axiom_id: Annotated[str | None, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
    source_span: ByteSourceSpan | None
    provenance_status: Literal[
        "recorded", "occurrence_unavailable", "unavailable_source", "not_exported"
    ]


class OriginalValue(StrictModel):
    """A complete literal, exact IRI, or faithful original OWL expression."""

    term_type: Literal["literal", "iri", "expression_ref"]
    expression_id: Identifier | None = None
    ast: dict[str, Any] | None = None
    lexical_form: str | None = None
    datatype: str | None = None
    language: str | None = None
    iri: str | None = None
    # Retain old null fields while refusing unadmitted raw syntax bytes.
    syntax: None = None
    syntax_format: None = None

    @model_validator(mode="after")
    def exact_value(self):
        if self.term_type != "expression_ref" and (self.expression_id or self.ast is not None):
            raise ValueError("Structured expressions cannot masquerade as literal/IRI values")
        if self.term_type == "expression_ref":
            if (
                not self.expression_id
                or not self.ast
                or any(
                    value is not None
                    for value in (
                        self.lexical_form,
                        self.iri,
                        self.datatype,
                        self.language,
                    )
                )
            ):
                raise ValueError("Expressions require a faithful typed AST and expression identity")
            validate_axiom(self.ast)
        elif self.term_type == "literal":
            if self.lexical_form is None or self.iri is not None:
                raise ValueError("Literal values require lexical form and no IRI/expression")
        elif self.term_type == "iri":
            if not self.iri or any(
                value is not None
                for value in (
                    self.lexical_form,
                    self.datatype,
                    self.language,
                )
            ):
                raise ValueError("IRI values require exactly one IRI")
        return self


class OriginalFact(StrictModel):
    fact_id: Identifier
    subject: EntityRef
    predicate_iri: str | None
    category: Literal[
        "labels",
        "definitions",
        "alternate_definitions",
        "annotations",
        "synonyms",
        "comments",
        "hierarchy",
        "parents",
        "children",
        "restrictions",
        "usage",
        "axioms",
        "types",
        "assertions",
        "domains",
        "ranges",
        "characteristics",
        "equivalences",
        "mappings",
        "xrefs",
        "mapping_xrefs",
    ]
    value: OriginalValue
    qualifiers: list[dict[str, Any]] = Field(default_factory=list)
    axiom_ref: Identifier
    document_sha256: Annotated[str | None, Field(pattern=r"^[a-f0-9]{64}$")] = None
    origins: list[FactOrigin] = Field(default_factory=list)
    interpretation: Literal["asserted", "structurally_derived", "reasoner_inferred", "projected"]
    premises: list[Identifier] = Field(default_factory=list)
    derivation: str | None = None

    @model_validator(mode="after")
    def structural_category(self):
        for qualifier in self.qualifiers:
            validate_annotation(qualifier)
        ast = self.value.ast
        if ast is not None and self.qualifiers:
            raise ValueError("Expression qualifiers must remain in their original axiom AST")
        if ast is not None:
            if self.value.expression_id != self.axiom_ref or _category(ast) != self.category:
                raise ValueError("OWL expression category or axiom identity differs from its fact")
            predicate = _iri(ast.get("property")) if ast["type"] == "AnnotationAssertion" else None
            if predicate != self.predicate_iri:
                raise ValueError("OWL expression predicate differs from its fact")
        else:
            if not self.predicate_iri:
                raise ValueError("Original annotation values require their exact predicate")
            category = ANNOTATION_REGISTRY.get(self.predicate_iri, ("annotations", None))[0]
            if category != self.category:
                raise ValueError("Literal or IRI annotation category differs from its predicate")
        if (
            self.document_sha256
            and self.origins
            and not any(
                origin.document_sha256 == "sha256:" + self.document_sha256
                for origin in self.origins
            )
        ):
            raise ValueError("Original document digest differs from recorded provenance")
        return self


class GroundedClaim(StrictModel):
    claim_id: Identifier
    text: Annotated[str, Field(min_length=1, max_length=4000)]
    fact_ids: Annotated[list[Identifier], Field(min_length=1, max_length=20)]
    interpretation: Literal["generated"] = "generated"
    grounding: Literal["exact_extract", "semantic_template"]
    category: Literal[
        "meaning",
        "scope",
        "key_fact",
        "agreement",
        "difference",
        "explicit_incompatibility",
        "unknown",
        "review_question",
    ] = "key_fact"
    scoped_entities: list[EntityRef] = Field(default_factory=list, max_length=2)
    packet_fact_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    reviewer_receipt: Identifier | None = None
    generation_manifest_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class HierarchyEdge(StrictModel):
    child: EntityRef
    parent: EntityRef
    basis: Literal["literal_asserted", "structural_navigation", "reasoner_inferred"]
    fact_ids: Annotated[list[Identifier], Field(min_length=1)]
    derivation: str | None = None


class EvidenceLink(StrictModel):
    evidence_id: Identifier
    candidate_id: Identifier
    fact_ids: list[Identifier]
    channel: Identifier
    role: Literal["source", "target", "comparison"]
    interpretation: Literal["projected", "matcher_comparison"]
    scores: list[Score] = Field(default_factory=list)
    saved_value: float | None = None
    value_meaning: str | None = None
    status: Literal["available", "not_exported", "unresolved"]


class ExplanationResource(StrictModel):
    """Strict allowlist for study panels; arbitrary scorer dictionaries cannot enter."""

    artifact_type: Literal["study_explanation"] = "study_explanation"
    contract_version: Literal["exact-study/1.0"] = "exact-study/1.0"
    policy_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    entities: Annotated[list[EntityRef], Field(min_length=1)]
    facts: list[OriginalFact]
    entity_profiles: list[GroundedClaim] = Field(default_factory=list)
    pair_comparison: list[GroundedClaim] = Field(default_factory=list)
    hierarchy: list[HierarchyEdge] = Field(default_factory=list)
    evidence: list[EvidenceLink] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    capabilities: dict[
        Literal["context", "profiles", "hierarchy", "evidence", "comparison"],
        Literal["available", "partial", "not_exported", "not_requested", "unsupported"],
    ]

    @model_validator(mode="after")
    def grounded_scope(self):
        facts = {fact.fact_id: fact for fact in self.facts}
        if len(facts) != len(self.facts):
            raise ValueError("Duplicate fact identity")
        identities = {(e.ontology_version_id, e.iri, e.kind) for e in self.entities}
        if any(
            (f.subject.ontology_version_id, f.subject.iri, f.subject.kind) not in identities
            for f in self.facts
        ):
            raise ValueError("Fact subject is outside the prepared entity universe")
        for claim in self.entity_profiles + self.pair_comparison:
            if not set(claim.fact_ids) <= set(facts):
                raise ValueError("Claim references an unavailable fact")
            if claim.grounding == "exact_extract" and claim.text not in {
                facts[fid].value.lexical_form for fid in claim.fact_ids
            }:
                raise ValueError("An exact extract must equal a cited original literal")
            if claim.grounding == "semantic_template":
                from ..contracts import EntityRef as CoreEntity
                from ..generation import FactPacket, comparison_templates

                if len(claim.scoped_entities) != 2 or any(
                    (e.ontology_version_id, e.iri, e.kind) not in identities
                    for e in claim.scoped_entities
                ):
                    raise ValueError("Comparison requires its ordered, permitted entity pair")
                pair = [CoreEntity.model_validate(e.model_dump()) for e in claim.scoped_entities]
                if not set(claim.packet_fact_ids) <= set(facts):
                    raise ValueError("Comparison packet refers to unavailable facts")
                selected = [facts[fid].model_dump() for fid in claim.packet_fact_ids]
                packet = FactPacket(
                    task="pair_comparison",
                    entities=pair,
                    context_hashes=[],
                    policy_hash=self.policy_hash,
                    facts=selected,
                    missingness=[],
                    selection={},
                )
                if not any(
                    claim.text == template.text
                    and claim.category == template.category
                    and set(claim.fact_ids) == set(template.fact_ids)
                    for template in comparison_templates(packet)
                ):
                    raise ValueError("Comparison does not match a grounded semantic template")
        for edge in self.hierarchy:
            if not set(edge.fact_ids) <= set(facts):
                raise ValueError("Hierarchy references unavailable facts")
            if edge.basis != "literal_asserted" and not edge.derivation:
                raise ValueError("Derived hierarchy requires an explicit rule/provider")
        for evidence in self.evidence:
            if not set(evidence.fact_ids) <= set(facts):
                raise ValueError("Evidence references unavailable facts")
            if evidence.saved_value is not None and not evidence.value_meaning:
                raise ValueError("Saved evidence numbers need their original meaning")
        return self


def validate_explanation_resource(content, study):
    """Apply the frozen visibility policy before admitting a participant artifact."""
    resource = ExplanationResource.model_validate_json(content)
    if resource.policy_hash != study["policy_hash"]:
        raise ValueError("Explanation resource policy differs from study policy")
    policy = study.get("visibility_policy")
    if policy:
        policy_model = VisibilityPolicy.model_validate(policy)
        allowed_ontologies = set(policy["ontology_ids"])
        if allowed_ontologies and any(
            e.ontology_version_id not in allowed_ontologies for e in resource.entities
        ):
            raise ValueError("Explanation entity is outside the allowed ontology universe")
        for fact in resource.facts:
            if (
                _visible_node(fact.qualifiers, policy_model, fact.subject.ontology_version_id)
                != fact.qualifiers
            ):
                raise ValueError("Original annotation qualifier is prohibited by the study policy")
            if fact.category not in policy["categories"] or (
                fact.category in {"xrefs", "mappings", "mapping_xrefs"}
                and not policy["allow_mapping_xrefs"]
            ):
                raise ValueError("Explanation fact is prohibited by the study policy")
            if (
                fact.value.ast is not None
                and _visible_node(fact.value.ast, policy_model, fact.subject.ontology_version_id)
                != fact.value.ast
            ):
                raise ValueError("Nested OWL annotation is prohibited by the study policy")
    return resource
