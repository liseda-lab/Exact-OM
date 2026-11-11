"""Lightweight offline description preparation shared with final-study tooling."""

from exact_inspect.contracts import canonical_hash
from exact_inspect.descriptions import (
    DESCRIPTION_VERSION,
    description_templates,
    prepare_description_packet,
)
from exact_inspect.generation import (
    comparison_templates,
    entity_packet,
    factual_fallback,
)


def packet_for(entity, contexts, policy):
    context = contexts[entity.ontology_version_id]
    facts = {}
    counts = {}
    for category, limit in [
        ("definitions", 8),
        ("alternate_definitions", 4),
        ("hierarchy", 12),
        ("restrictions", 12),
        ("labels", 3),
        ("synonyms", 6),
        ("annotations", 6),
    ]:
        page = context.facts(entity, category=category, limit=limit, policy=policy)
        counts[category] = len(page["items"])
        facts.update({f["fact_id"]: f for f in page["items"]})
    missingness = (
        []
        if counts["definitions"]
        else ["No original definition is available in the admitted context."]
    )
    packet = entity_packet(
        entity,
        list(facts.values()),
        context_hash=context.manifest["artifacts"]["context.sqlite"],
        policy=policy,
        missingness=missingness,
    )
    packet = prepare_description_packet(packet, context, policy)
    return (
        packet.model_copy(update={"selection": {**packet.selection, "description_prose": True}}),
        counts,
    )


def literal_label(entity, contexts, policy):
    rows = contexts[entity.ontology_version_id].labels([entity.iri], policy=policy)
    for row in rows:
        if row["entity"] == entity.model_dump() and row["preferred_label"]["status"] == "available":
            return row["preferred_label"]["value"][:512]
    return entity.iri[:512]


def deterministic_result(packet):
    templates = description_templates(packet)
    if packet.task == "pair_comparison":
        templates = comparison_templates(packet)[:3]
    output = factual_fallback(packet) if not templates else None
    claims = templates[:4] if templates else output.claims
    key = canonical_hash(
        {
            "contract": DESCRIPTION_VERSION,
            "packet": packet.model_dump(mode="json"),
            "renderer": "typed-owl/1",
        }
    )
    return dict(
        artifact_type="generated_explanation",
        contract_version="exact-explain/1.0",
        explanation_id=key,
        task=packet.task,
        entities=[e.model_dump() for e in packet.entities],
        claims=[
            c.model_copy(
                update={
                    "claim_id": canonical_hash(
                        {"generation": key, "claim": c.model_dump(mode="json")}
                    )
                }
            ).model_dump()
            for c in claims
        ],
        grounding_status="validated",
        limitations=packet.missingness,
        fixture_provenance="Real ontology facts; deterministic typed rendering, no model request.",
        preparation_kind="grounded_description" if templates else "factual_fallback",
        support_status="typed_template_support" if templates else "fallback",
        support_reasons=[] if templates else ["No supported non-label description available."],
        support_packet=packet.model_dump(mode="json"),
        manifest=dict(
            ontology_context_hashes=packet.context_hashes,
            visibility_policy_hash=packet.policy_hash,
            packet_hash=canonical_hash(packet),
            prompt_hash=canonical_hash("typed-owl/1"),
            output_schema="exact-explain-generation/1",
            requested_model="none-typed-renderer",
            returned_model=None,
            provider=None,
            parameters_hash=canonical_hash({}),
            language=packet.language,
            response_hash=None,
            status="validated" if templates else "failed",
            description_contract=DESCRIPTION_VERSION,
        ),
    )
