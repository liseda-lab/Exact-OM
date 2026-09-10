"""Bounded ontology-only evidence acquisition for a frozen comparative pool."""

import hashlib
import json
from dataclasses import replace

from exact.llm.routing import extract_chat_text

from .listwise_llm import listwise_labels


def ontology_packet(dataset, iri, side, *, max_facts=24):
    """Expose facts/provenance only; no reference or scorer feature is accessible."""
    kind = dataset.entity_kind_for(iri, side, warn_unknown=False)
    features = dataset.get_entity_features(iri, side, kind)
    items = list(features.get("attributes", [])) + list(features.get("object_triples", []))
    for family in sorted(features.get("hierarchy", {})):
        items.extend(features["hierarchy"][family])
    allowed = {
        "triple",
        "text",
        "value",
        "prop_iri",
        "datatype",
        "language",
        "subject_iri",
        "predicate_iri",
        "object_iri",
        "entity_iri",
        "type_closure",
        "origin",
        "provenance",
    }
    unique = {}
    for item in items:
        fact = {key: value for key, value in item.items() if key in allowed}
        if not fact:
            continue
        identity = hashlib.sha256(
            json.dumps([side, iri, fact], sort_keys=True).encode()
        ).hexdigest()
        unique[identity] = {"id": identity, "side": side, "entity_iri": iri, "fact": fact}
    return {
        "entity_iri": iri,
        "side": side,
        "facts": [unique[key] for key in sorted(unique)[:max_facts]],
        "omitted_facts": max(0, len(unique) - max_facts),
        "empty_state": "unobserved" if not unique else None,
    }


def acquire_plan_evidence(model, plan, profile):
    """One hosted request may select at most two existing entity evidence packets."""
    call = plan.calls[0]
    choices = {"S": (plan.source_iri, "src")}
    choices.update(
        {
            letter: (target, "tgt")
            for letter, target in zip(
                listwise_labels(len(call.candidate_ids))[:-1], call.candidate_ids
            )
        }
    )
    payload = model._llm_router.hosted.chat_completion(
        profile=profile,
        role="evidence_acquisition",
        messages=[
            {
                "role": "system",
                "content": "Choose zero, one, or two ontology evidence packets to clarify equivalence. Return only a JSON array of allowed entity keys. Do not answer the alignment question yet.",
            },
            {
                "role": "user",
                "content": call.prompt["user"]
                + "\nAllowed packet keys (S is source): "
                + json.dumps(choices, sort_keys=True),
            },
        ],
        max_tokens=32,
        temperature=0.0,
        top_p=1.0,
        seed=call.seed % (2**31 - 1),
    )
    record = {
        "response_id": payload.get("id"),
        "model": payload.get("model"),
        "provider": payload.get("provider"),
        "usage": payload.get("usage", {}),
        "raw_selection": extract_chat_text(payload),
        "packets": [],
    }
    try:
        selected = json.loads(record["raw_selection"])
        if (
            not isinstance(selected, list)
            or len(selected) > 2
            or len(set(selected)) != len(selected)
            or any(key not in choices for key in selected)
        ):
            raise ValueError("invalid packet selection")
    except (ValueError, TypeError):
        record["fallback"] = "original_evidence"
        return plan, record
    packets = [ontology_packet(model._attached_dataset, *choices[key]) for key in selected]
    record["packets"] = packets
    addition = "\nAdditional ontology evidence (absence is unobserved):\n" + json.dumps(
        packets, sort_keys=True
    )
    calls = tuple(
        replace(item, prompt={**item.prompt, "user": item.prompt["user"] + addition})
        for item in plan.calls
    )
    return replace(plan, calls=calls), record
