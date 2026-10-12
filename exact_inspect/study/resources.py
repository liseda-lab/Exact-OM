"""Participant-safe prepared explanation envelopes, independent of study answer keys.

These are frozen presentation records produced from the shared ontology engine;
structured OWL values retain the public constructor shape without a serving parser.
"""

from __future__ import annotations

from ..postmigration_contracts import DescriptionPresentation

from collections.abc import Iterator
from typing import Annotated, Any, Literal

from pydantic import Field, StrictInt, model_validator

from ..context_semantics import ANNOTATION_REGISTRY, _category, _iri, _visible_node
from ..contracts import Score, VisibilityPolicy, canonical_hash, canonical_json
from ..description_review import (
    Digest,
    IndependentReviewSupport,
    validate_receipt_claims,
)
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
    grounding: Literal[
        "exact_extract", "semantic_template", "typed_description", "independent_review"
    ]
    preparation_kind: Literal["grounded_description", "fact_extract", "factual_fallback"] | None = (
        None
    )
    support_status: (
        Literal["typed_template_support", "exact_support", "fallback", "independently_reviewed"]
        | None
    ) = None
    support_reasons: list[str] = Field(default_factory=list)
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
    packet_fact_subjects: list[EntityRef] = Field(default_factory=list, max_length=100)
    presentation: DescriptionPresentation | None = None
    packet_render_revision: Literal["typed-readable/3"] | None = None
    packet_reference_label_ids: list[Identifier] = Field(default_factory=list, max_length=60)
    reviewer_receipt: Identifier | None = None
    generation_manifest_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class HierarchyEdge(StrictModel):
    child: EntityRef
    parent: EntityRef
    basis: Literal["literal_asserted", "structural_navigation", "reasoner_inferred"]
    fact_ids: Annotated[list[Identifier], Field(min_length=1)]
    derivation: str | None = None


class EvidenceLink(StrictModel):
    origin_resolution: Literal["resolved", "unresolved", "not_exported", "withheld"] | None = None
    graph_projection: Literal["supported", "unsupported", "not_recorded"] | None = None
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


REFERENCED_LABEL_LIMIT = 100
REFERENCED_LABEL_BYTES = 64 * 1024


def referenced_terms(facts: list[OriginalFact]) -> Iterator[tuple[str, str]]:
    """Yield same-ontology IRIs explicitly present in admitted original facts.

    Structural expressions precede adjacency so property/filler labels are not
    crowded out by child classes. Literal text is never parsed as an IRI.
    Callers stop after their bounded selection or required reference set is found.
    """
    for fact in sorted(facts, key=lambda item: item.category == "hierarchy"):
        version = fact.subject.ontology_version_id
        for iri in (fact.predicate_iri, fact.value.iri, fact.value.datatype):
            if iri:
                yield version, iri
        pending: list[Any] = [fact.value.ast, *fact.qualifiers]
        while pending:
            node = pending.pop()
            if isinstance(node, dict):
                if node.get("type") == "IRI":
                    yield version, node["value"]
                pending.extend(node.values())
            elif isinstance(node, list):
                pending.extend(node)


def validate_claim_identities(claims):
    """IDs are client index keys; one ID cannot silently erase another scoped claim."""
    seen = {}
    for claim in claims:
        value = claim.model_dump(mode="json")
        previous = seen.setdefault(claim.claim_id, value)
        if previous != value:
            raise ValueError("Conflicting claim identity for different content or scope")


from ..evidence_metadata import PairEvidenceMetadata


class FrozenPairMetadata(StrictModel):
    source: EntityRef
    target: EntityRef
    evidence_metadata: PairEvidenceMetadata


class ExplanationResource(StrictModel):
    """Strict allowlist for study panels; arbitrary scorer dictionaries cannot enter."""

    artifact_type: Literal["study_explanation"] = "study_explanation"
    contract_version: Literal["exact-study/1.0"] = "exact-study/1.0"
    policy_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    entities: Annotated[list[EntityRef], Field(min_length=1)]
    facts: list[OriginalFact]
    referenced_labels: list[OriginalFact] = Field(
        default_factory=list, max_length=REFERENCED_LABEL_LIMIT
    )
    referenced_labels_truncated: bool = False
    entity_profiles: list[GroundedClaim] = Field(default_factory=list)
    pair_comparison: list[GroundedClaim] = Field(default_factory=list)
    independent_reviews: dict[Digest, IndependentReviewSupport] = Field(
        default_factory=dict, max_length=100
    )
    hierarchy: list[HierarchyEdge] = Field(default_factory=list)
    evidence: list[EvidenceLink] = Field(default_factory=list)
    pair_metadata: dict[Identifier, FrozenPairMetadata] = Field(default_factory=dict, max_length=5)
    limitations: list[str] = Field(default_factory=list)
    capabilities: dict[
        Literal["context", "profiles", "hierarchy", "evidence", "comparison"],
        Literal["available", "partial", "not_exported", "not_requested", "unsupported"],
    ]

    @model_validator(mode="after")
    def grounded_scope(self):
        validate_claim_identities(self.entity_profiles + self.pair_comparison)
        facts: dict[str, list[OriginalFact]] = {}
        contextual_facts = {}
        identities = {(e.ontology_version_id, e.iri, e.kind) for e in self.entities}
        for fact in self.facts:
            subject = (fact.subject.ontology_version_id, fact.subject.iri, fact.subject.kind)
            if subject not in identities:
                raise ValueError("Fact subject is outside the prepared entity universe")
            key = (fact.fact_id, subject)
            if key in contextual_facts:
                raise ValueError("Duplicate fact identity for the same subject")
            copies = facts.setdefault(fact.fact_id, [])
            if copies and (
                copies[0].subject.ontology_version_id != fact.subject.ontology_version_id
                or copies[0].model_dump(exclude={"subject"}) != fact.model_dump(exclude={"subject"})
            ):
                raise ValueError("Shared original fact copies disagree")
            copies.append(fact)
            contextual_facts[key] = fact
        required_labels = {
            (label.subject.ontology_version_id, label.subject.iri)
            for label in self.referenced_labels
        }
        if required_labels:
            for reference in referenced_terms(self.facts):
                required_labels.discard(reference)
                if not required_labels:
                    break
        if required_labels:
            raise ValueError("Label term is outside the admitted original references")
        label_keys = set()
        for label in self.referenced_labels:
            key = (
                label.fact_id,
                label.subject.ontology_version_id,
                label.subject.iri,
                label.subject.kind,
            )
            if key in label_keys:
                raise ValueError("Duplicate referenced label")
            label_keys.add(key)
            if label.category != "labels" or label.value.term_type != "literal":
                raise ValueError("Referenced labels must be original literal label facts")
        if (
            len(canonical_json([label.model_dump(mode="json") for label in self.referenced_labels]))
            > REFERENCED_LABEL_BYTES
        ):
            raise ValueError("Referenced labels exceed the prepared byte budget")
        for label in self.referenced_labels:
            if label.fact_id not in facts:
                facts[label.fact_id] = [label]
            subject = (label.subject.ontology_version_id, label.subject.iri, label.subject.kind)
            contextual_facts[(label.fact_id, subject)] = label
        for review_hash, support in self.independent_reviews.items():
            # Recheck the full nested envelope even when the caller supplied a
            # previously constructed model with mutable JSON descendants.
            support = IndependentReviewSupport.model_validate(support.model_dump(mode="json"))
            if (
                review_hash != canonical_hash(support.receipt)
                or support.packet.policy_hash != "sha256:" + self.policy_hash
            ):
                raise ValueError("Independent review is outside the frozen packet or policy")
            reviewed = [
                claim
                for claim in self.entity_profiles + self.pair_comparison
                if claim.reviewer_receipt == review_hash
            ]
            validate_receipt_claims(
                support.receipt, support.packet, [claim.model_dump() for claim in reviewed]
            )
            from ..readable_descriptions import label_facts

            for raw in [*support.packet.facts, *label_facts(support.packet)]:
                subject = EntityRef.model_validate(raw["subject"])
                key = (raw["fact_id"], (subject.ontology_version_id, subject.iri, subject.kind))
                original = contextual_facts.get(key)
                if original is None or any(
                    raw.get(field) != getattr(original, field)
                    for field in (
                        "fact_id",
                        "predicate_iri",
                        "category",
                        "axiom_ref",
                        "interpretation",
                    )
                ):
                    raise ValueError("Reviewed packet original differs from the admitted facts")
                if OriginalValue.model_validate(raw["value"]) != original.value:
                    raise ValueError("Reviewed packet value differs from its original")
            for claim in reviewed:
                if (
                    claim.grounding != "independent_review"
                    or claim.support_status != "independently_reviewed"
                    or claim.preparation_kind != "grounded_description"
                    or claim.generation_manifest_sha256 != support.generation_manifest_sha256
                    or [e.model_dump() for e in claim.scoped_entities]
                    != [e.model_dump() for e in support.packet.entities]
                ):
                    raise ValueError("Reviewed claim scope differs from its support record")
        for claim in self.entity_profiles + self.pair_comparison:
            if (
                claim.grounding == "independent_review"
                and claim.reviewer_receipt not in self.independent_reviews
            ):
                raise ValueError("Independent claim review receipt is unavailable")
            if not set(claim.fact_ids) <= set(facts):
                raise ValueError("Claim references an unavailable fact")
            scope = {(e.ontology_version_id, e.iri, e.kind) for e in claim.scoped_entities}
            if not scope <= identities:
                raise ValueError("Claim is outside the prepared entity universe")
            if claim.grounding == "exact_extract" and claim.text not in {
                fact.value.lexical_form
                for fid in claim.fact_ids
                for fact in facts[fid]
                if not scope
                or (fact.subject.ontology_version_id, fact.subject.iri, fact.subject.kind) in scope
            }:
                raise ValueError("An exact extract must equal a cited original literal")
            if claim.grounding in {"semantic_template", "typed_description"}:
                from ..contracts import EntityRef as CoreEntity
                from ..descriptions import DESCRIPTION_VERSION, description_templates
                from ..generation import FactPacket, comparison_templates

                if claim.grounding == "semantic_template" and len(claim.scoped_entities) != 2:
                    raise ValueError("Comparison requires its ordered, permitted entity pair")
                if not 1 <= len(claim.scoped_entities) <= 2:
                    raise ValueError("Description requires its permitted typed entity scope")
                pair = [CoreEntity.model_validate(e.model_dump()) for e in claim.scoped_entities]
                if not set(claim.packet_fact_ids) <= set(facts):
                    raise ValueError("Comparison packet refers to unavailable facts")
                selected = []
                if claim.packet_fact_subjects and len(claim.packet_fact_subjects) != len(
                    claim.packet_fact_ids
                ):
                    raise ValueError(
                        "Comparison packet fact subjects must align with fact identities"
                    )
                for index, fid in enumerate(claim.packet_fact_ids):
                    if claim.packet_fact_subjects:
                        subject = claim.packet_fact_subjects[index]
                        key = (fid, (subject.ontology_version_id, subject.iri, subject.kind))
                        if key not in contextual_facts:
                            raise ValueError(
                                "Comparison packet subject has no admitted original fact"
                            )
                        selected.append(contextual_facts[key].model_dump())
                    else:
                        # Legacy packets remain valid when each original has one
                        # unambiguous contextual copy in this comparison's scope.
                        copies = [
                            fact
                            for fact in facts[fid]
                            if (
                                fact.subject.ontology_version_id,
                                fact.subject.iri,
                                fact.subject.kind,
                            )
                            in scope
                        ]
                        if len(copies) != 1:
                            raise ValueError(
                                "Legacy comparison packet has an ambiguous fact subject"
                            )
                        selected.append(copies[0].model_dump())
                packet = FactPacket(
                    task="pair_comparison" if len(pair) == 2 else "entity_profile",
                    entities=pair,
                    context_hashes=[],
                    policy_hash=self.policy_hash,
                    facts=selected,
                    missingness=[],
                    selection=(
                        {
                            "description_contract": DESCRIPTION_VERSION,
                            **(
                                {
                                    "render_revision": claim.packet_render_revision,
                                    "referenced_labels": [
                                        f.model_dump(mode="json")
                                        for f in self.referenced_labels
                                        if f.fact_id in claim.packet_reference_label_ids
                                    ],
                                }
                                if claim.packet_render_revision
                                else {}
                            ),
                        }
                        if claim.grounding == "typed_description"
                        else {}
                    ),
                )
                if not any(
                    claim.text == template.text
                    and claim.category == template.category
                    and set(claim.fact_ids) == set(template.fact_ids)
                    for template in comparison_templates(packet) + description_templates(packet)
                ):
                    raise ValueError("Comparison does not match a grounded semantic template")
        from ..readable_descriptions import presentation

        presentation_groups = {}
        for claim in self.entity_profiles + self.pair_comparison:
            if not set(claim.packet_reference_label_ids) <= {
                f.fact_id for f in self.referenced_labels
            }:
                raise ValueError("Description packet references unavailable labels")
            if claim.presentation is not None:
                key = (
                    claim.generation_manifest_sha256,
                    canonical_hash([e.model_dump(mode="json") for e in claim.scoped_entities]),
                )
                presentation_groups.setdefault(key, []).append(claim)
        for claims in presentation_groups.values():
            first = claims[0]
            support = self.independent_reviews.get(first.reviewer_receipt)
            if support:
                presentation_facts = support.packet.facts
            elif first.packet_fact_subjects:
                presentation_facts = [
                    contextual_facts[(fid, (e.ontology_version_id, e.iri, e.kind))].model_dump(
                        mode="json"
                    )
                    for fid, e in zip(
                        first.packet_fact_ids, first.packet_fact_subjects, strict=True
                    )
                ]
            else:
                presentation_facts = [
                    f.model_dump(mode="json")
                    for f in self.facts
                    if f.subject in first.scoped_entities
                ]
            expected = presentation(
                {
                    "task": (
                        "entity_profile" if len(first.scoped_entities) == 1 else "pair_comparison"
                    ),
                    "entities": [e.model_dump(mode="json") for e in first.scoped_entities],
                    "claims": [c.model_dump(mode="json") for c in claims],
                    "support_status": first.support_status,
                    "preparation_kind": first.preparation_kind,
                    "independent_review": (
                        support.receipt.model_dump(mode="json") if support else None
                    ),
                    "support_packet": {
                        "facts": presentation_facts,
                        "selection": {"render_revision": first.packet_render_revision},
                    },
                }
            )
            if any(c.presentation.model_dump(mode="json") != expected for c in claims):
                raise ValueError(
                    "Description presentation differs from its cited scope and support"
                )
        for candidate_id, pair in self.pair_metadata.items():
            if any(
                (e.ontology_version_id, e.iri, e.kind) not in identities
                for e in [pair.source, pair.target]
            ):
                raise ValueError("Frozen comparison is outside the permitted pair")
            rows = {e.evidence_id: e for e in self.evidence if e.candidate_id == candidate_id}
            if pair.evidence_metadata.counts.selected != len(rows):
                raise ValueError("Frozen comparison inventory differs from selected evidence")
            for bridge in pair.evidence_metadata.bridges:
                if (
                    bridge.source.model_dump() != pair.source.model_dump()
                    or bridge.target.model_dump() != pair.target.model_dump()
                ):
                    raise ValueError("Frozen bridge differs from its typed pair")
                for side in ["source", "target"]:
                    for eid in getattr(bridge, side + "_evidence_ids"):
                        if eid not in rows or rows[eid].role != side:
                            raise ValueError("Frozen bridge refers to another selected feature")
                    permitted_facts = {
                        fid for e in rows.values() if e.role == side for fid in e.fact_ids
                    }
                    if not set(getattr(bridge, side + "_fact_ids")) <= permitted_facts:
                        raise ValueError("Frozen bridge cites unavailable original facts")
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
        for fact in [*resource.facts, *resource.referenced_labels]:
            if not policy_model.allows_ontology(fact.subject.ontology_version_id):
                raise ValueError("Original fact is outside the allowed ontology universe")
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
