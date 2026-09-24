"""Prepare strict study panels from the shared, already indexed ontology engine.

This adapter never loads a matcher, parses an ontology, or dispatches a model.
Answer keys, production verdicts and unrestricted scorer dictionaries are not inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..context import OntologyContext
from ..contracts import EntityRef, Score, VisibilityPolicy, canonical_hash
from ..generation import ExplanationOutput, FactPacket, comparison_templates, grounding
from .models import EntityRef as StudyEntity
from .resources import (
    EvidenceLink,
    ExplanationResource,
    GroundedClaim,
    HierarchyEdge,
    OriginalFact,
    OriginalValue,
)


def build_explanation_resource(
    contexts: Mapping[str, OntologyContext],
    entities: list[EntityRef],
    *,
    policy: VisibilityPolicy,
    packets: Iterable[FactPacket] = (),
    explanations: Iterable[dict[str, Any]] = (),
    evidence: Mapping[str, list[dict[str, Any]]] | None = None,
) -> ExplanationResource:
    """Freeze a bounded original-context view and independently validated generated text.

    Packets are the saved generation request packets, never reconstructed from a
    candidate score. Their complete facts are retained for citation verification.
    Hierarchy remains explicitly bounded; distant nodes are navigation targets.
    """
    facts: dict[str, OriginalFact] = {}
    edges: dict[str, HierarchyEdge] = {}
    limitations: list[str] = []
    permitted = {canonical_hash(entity) for entity in entities}
    if len(permitted) != len(entities):
        raise ValueError("Duplicate study entity")

    def admit(raw: dict[str, Any]) -> None:
        entity = EntityRef.model_validate(raw["subject"])
        if canonical_hash(entity) not in permitted or not policy.allows_fact(raw):
            raise ValueError("Fact is outside the frozen study policy or entity universe")
        context = contexts[entity.ontology_version_id]
        original = context.axiom(raw["axiom_ref"], policy=policy)
        authoritative = context.fact(raw["axiom_ref"], entity, policy=policy)
        for field in ("fact_id", "subject", "predicate_iri", "value", "interpretation", "category"):
            if raw.get(field) != authoritative.get(field):
                raise ValueError("Generation packet fact differs from the original ontology")
        value = dict(
            raw.get("value") or {"term_type": "expression_ref", "expression_id": raw["axiom_ref"]}
        )
        if value["term_type"] == "expression_ref":
            value["ast"] = original["ast"]
        origins = raw.get("origins", [])
        document = next((o["document_sha256"] for o in origins if o.get("document_sha256")), None)
        fact = OriginalFact(
            fact_id=raw["fact_id"],
            subject=StudyEntity.model_validate(entity.model_dump()),
            predicate_iri=raw.get("predicate_iri"),
            category=raw["category"],
            value=OriginalValue.model_validate(value),
            qualifiers=(
                original["ast"].get("annotations", [])
                if value["term_type"] in {"literal", "iri"}
                else []
            ),
            axiom_ref=raw["axiom_ref"],
            document_sha256=document.removeprefix("sha256:") if document else None,
            origins=origins,
            interpretation=raw["interpretation"],
            premises=raw.get("premise_ids", []),
            derivation=raw.get("derivation_id"),
        )
        # One original axiom can occur in the context of several referenced terms.
        # Keep one authoritative original record instead of inventing new axiom IDs.
        facts.setdefault(fact.fact_id, fact)

    packet_map = {}
    for packet in packets:
        if packet.policy_hash != policy.policy_hash or any(
            canonical_hash(e) not in permitted for e in packet.entities
        ):
            raise ValueError("Generation packet is outside the frozen study scope")
        packet_map[canonical_hash(packet)] = packet
        for raw in packet.facts:
            admit(raw)

    for entity in entities:
        if not policy.allows_ontology(entity.ontology_version_id):
            raise ValueError("Study ontology is outside the visibility policy")
        context = contexts[entity.ontology_version_id]
        for category in ("labels", "definitions", "synonyms", "hierarchy", "restrictions"):
            page = context.facts(entity, category=category, limit=20, policy=policy)
            for fact in page["items"]:
                admit(fact)
            if page["truncated"]:
                limitations.append(
                    f"Original {category} for {entity.iri} are bounded to the prepared page."
                )
        page = context.hierarchy(entity, policy=policy, limit=20)
        for edge in page["items"]:
            if edge["axiom_id"] in facts:
                edges[edge["id"]] = HierarchyEdge(
                    child=edge["child"],
                    parent=edge["parent"],
                    basis=page["basis"],
                    fact_ids=[edge["axiom_id"]],
                )
        if page["truncated"]:
            limitations.append(f"Named parents for {entity.iri} are bounded to the prepared page.")

    evidence_links = []
    for candidate_id, rows in (evidence or {}).items():
        for row in rows:
            entity = EntityRef.model_validate(row["entity"])
            if canonical_hash(entity) not in permitted:
                raise ValueError("Evidence belongs to another entity")
            context = contexts[entity.ontology_version_id]
            resolved = []
            for fact_id in row.get("fact_ids", []):
                try:
                    original_fact = context.fact(fact_id, entity, policy=policy)
                except KeyError:
                    continue
                admit(original_fact)
                resolved.append(fact_id)
            if row.get("fact_ids") and not resolved:
                continue
            evidence_links.append(
                EvidenceLink(
                    evidence_id=row["evidence_id"],
                    candidate_id=candidate_id,
                    fact_ids=resolved,
                    channel=row["channel"],
                    role=row["side"],
                    interpretation="projected",
                    status="available" if resolved else "not_exported",
                    scores=[
                        Score(
                            name=name,
                            stage="pair_scoring",
                            value=value,
                            meaning="Original saved evidence "
                            + name
                            + "; calibration not established",
                        )
                        for name, value in row.get("values", {}).items()
                        if name
                        in {"support", "importance", "weight", "edge_ic", "unsupported_mass"}
                    ],
                )
            )

    profiles: list[GroundedClaim] = []
    comparisons: list[GroundedClaim] = []
    for result in explanations:
        manifest = result["manifest"]
        packet = packet_map[manifest["packet_hash"]]
        if (
            manifest["visibility_policy_hash"] != policy.policy_hash
            or result["entities"] != [e.model_dump() for e in packet.entities]
            or result["grounding_status"] != "validated"
        ):
            raise ValueError("Generated artifact does not match its permitted input packet")
        output = ExplanationOutput(claims=result["claims"])
        if grounding(output, packet)[0] != "validated":
            raise ValueError("Generated claims do not have verified original support")
        templates = comparison_templates(packet)
        target = profiles if packet.task == "entity_profile" else comparisons
        for claim in output.claims:
            if not claim.fact_ids:
                if claim.text in packet.missingness:
                    limitations.append(claim.text)
                    continue
                raise ValueError("Study factual claims require support")
            is_template = any(
                claim.text == t.text
                and claim.category == t.category
                and set(claim.fact_ids) == set(t.fact_ids)
                for t in templates
            )
            target.append(
                GroundedClaim(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    fact_ids=claim.fact_ids,
                    category=claim.category,
                    grounding="semantic_template" if is_template else "exact_extract",
                    scoped_entities=(
                        [StudyEntity.model_validate(e.model_dump()) for e in packet.entities]
                        if is_template
                        else []
                    ),
                    packet_fact_ids=[f["fact_id"] for f in packet.facts] if is_template else [],
                    generation_manifest_sha256=canonical_hash(manifest).removeprefix("sha256:"),
                )
            )
        limitations.extend(packet.missingness)
        if manifest["status"] != "validated":
            limitations.append(
                "Generated interpretation is unavailable; original fact excerpts are shown."
            )
    return ExplanationResource(
        policy_hash=policy.policy_hash.removeprefix("sha256:"),
        entities=[StudyEntity.model_validate(e.model_dump()) for e in entities],
        facts=list(facts.values()),
        entity_profiles=profiles,
        pair_comparison=comparisons,
        hierarchy=list(edges.values()),
        evidence=evidence_links,
        limitations=list(dict.fromkeys(limitations)),
        capabilities={
            "context": "partial",
            "profiles": "available" if profiles else "not_requested",
            "hierarchy": "partial",
            "evidence": "available" if evidence_links else "not_exported",
            "comparison": "available" if comparisons else "not_requested",
        },
    )
