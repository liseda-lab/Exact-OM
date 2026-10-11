"""Conservative readable templates with bound labels and explicit display semantics."""

from __future__ import annotations
import copy
from .contracts import canonical_hash, canonical_json

RENDER_REVISION = "typed-readable/3"
PRESENTATION_CONTRACT = "exact-description-presentation/1"
KINDS = {
    "Class": "class",
    "ObjectProperty": "object_property",
    "DataProperty": "data_property",
    "Datatype": "datatype",
    "NamedIndividual": "individual",
}


def references(node):
    if isinstance(node, dict):
        iri = (node.get("iri") or {}).get("value") if isinstance(node.get("iri"), dict) else None
        if node.get("type") in KINDS and iri:
            yield iri, KINDS[node["type"]]
        for value in node.values():
            yield from references(value)
    elif isinstance(node, list):
        for value in node:
            yield from references(value)


def label_facts(packet):
    if packet.selection.get("render_revision") != RENDER_REVISION:
        return []
    labels = packet.selection.get("referenced_labels", [])
    allowed = {
        (f["subject"]["ontology_version_id"], iri, kind)
        for f in packet.facts
        for iri, kind in references((f.get("value") or {}).get("ast", {}))
    }
    if not isinstance(labels, list) or len(labels) > 60 or len(canonical_json(labels)) > 48 * 1024:
        raise ValueError("Referenced labels exceed packet bounds")
    seen = set()
    for f in labels:
        e = f["subject"]
        key = (e["ontology_version_id"], e["iri"], e["kind"])
        if (
            key not in allowed
            or key in seen
            or f.get("category") != "labels"
            or f["value"]["term_type"] != "literal"
        ):
            raise ValueError("Label must name one exact typed reference in an admitted axiom")
        seen.add(key)
    return labels


def enrich(packet, contexts, policy):
    """Add only bounded original labels for exact typed references; no new medical facts."""
    labels, used = [], 0
    refs = sorted(
        {
            (f["subject"]["ontology_version_id"], iri, kind)
            for f in packet.facts
            for iri, kind in references((f.get("value") or {}).get("ast", {}))
        }
    )
    for version, iri, kind in refs:
        if len(labels) >= 60:
            break
        context = contexts[version]
        if len(iri) > 2048:
            continue
        ref = {"ontology_version_id": version, "iri": iri, "kind": kind}
        label = context._label(ref, packet.language, policy)
        if label["status"] != "available":
            continue
        fact = context.fact(label["fact_id"], ref, policy=policy)
        allowed = {
            "fact_id",
            "subject",
            "predicate_iri",
            "value",
            "axiom_ref",
            "interpretation",
            "origins",
            "premise_ids",
            "derivation_id",
            "category",
        }
        fact = {k: v for k, v in fact.items() if k in allowed}
        size = len(canonical_json(fact))
        if used + size > 48 * 1024:
            continue
        used += size
        labels.append(fact)
    result = packet.model_copy(
        update={
            "selection": {
                **packet.selection,
                "render_revision": RENDER_REVISION,
                "referenced_labels": labels,
            }
        }
    )
    label_facts(result)
    return result


def excerpt(text, limit=360):
    if len(text) <= limit:
        return text
    prefix = text[: limit + 1]
    if " " not in prefix:
        return None
    return prefix.rsplit(" ", 1)[0].rstrip() + "…"


def profile_templates(packet):
    from .generation import Claim
    from .descriptions import render_expression, _iri

    labels = {
        (f["subject"]["ontology_version_id"], f["subject"]["iri"], f["subject"]["kind"]): f
        for f in label_facts(packet)
    }
    out = []
    for fact in packet.facts:
        value = fact.get("value") or {}
        text = None
        ids = [fact["fact_id"]]
        category = "scope"
        if fact.get("category") == "definitions" and value.get("term_type") == "literal":
            text = excerpt(value["lexical_form"], 1200)
            category = "meaning"
        ast = value.get("ast") or {}
        phrase = None
        if (
            fact["subject"]["kind"] == "class"
            and ast.get("type") == "SubClassOf"
            and _iri(ast.get("sub_class")) == fact["subject"]["iri"]
        ):
            expression = ast.get("super_class")
            phrase = render_expression(expression)
            if phrase:
                for iri, kind in references(expression):
                    label = labels.get((fact["subject"]["ontology_version_id"], iri, kind))
                    if label is None:
                        phrase = None
                        break
                    phrase = phrase.replace(
                        "<" + iri + ">", "“" + label["value"]["lexical_form"] + "”"
                    )
                    ids.append(label["fact_id"])
            if phrase:
                text = "Every instance has this necessary condition: " + phrase + "."
        elif fact["subject"]["kind"] == "class" and ast.get("type") == "EquivalentClasses":
            expr = ast.get("expressions", [])
            if any(_iri(x) == fact["subject"]["iri"] for x in expr):
                parts = []
                for x in expr:
                    if _iri(x) == fact["subject"]["iri"]:
                        continue
                    part = render_expression(x)
                    if part:
                        for iri, kind in references(x):
                            label = labels.get((fact["subject"]["ontology_version_id"], iri, kind))
                            if label is None:
                                part = None
                                break
                            part = part.replace(
                                "<" + iri + ">", "“" + label["value"]["lexical_form"] + "”"
                            )
                            ids.append(label["fact_id"])
                    parts.append(part)
                if parts and all(parts):
                    text = (
                        "The ontology defines this class as equivalent to: "
                        + "; ".join(parts)
                        + "."
                    )
        if text and len(text) <= 4000 and len(set(ids)) <= 20:
            out.append(Claim(text=text, fact_ids=list(dict.fromkeys(ids)), category=category))
    return sorted(out, key=lambda c: (c.category != "meaning", c.fact_ids))


def comparison_templates(packet):
    from .generation import Claim

    if packet.task != "pair_comparison":
        return []
    out = []
    for side, entity in zip(["Source", "Target"], packet.entities):
        fact = next(
            (
                f
                for f in packet.facts
                if f["subject"] == entity.model_dump()
                and f.get("category") == "definitions"
                and (f.get("value") or {}).get("term_type") == "literal"
            ),
            None,
        )
        if fact:
            text = excerpt(fact["value"]["lexical_form"])
            if text:
                out.append(
                    Claim(
                        text=side + " definition: " + text,
                        fact_ids=[fact["fact_id"]],
                        category="key_fact",
                    )
                )
    return out


def presentation(result):
    packet = result.get("support_packet") or {}
    facts = {}
    for f in packet.get("facts", []):
        facts.setdefault(f["fact_id"], []).append(f)
    roles = {
        canonical_hash(e): role for e, role in zip(result.get("entities", []), ["source", "target"])
    }
    covered = set()
    claim_roles = {}
    for c in result.get("claims", []):
        sides = sorted(
            {
                roles.get(canonical_hash(f["subject"]))
                for fid in c["fact_ids"]
                for f in facts.get(fid, [])
            }
            - {None}
        )
        covered.update(sides)
        claim_roles[c["claim_id"]] = sides
    if result.get("task") == "entity_profile":
        kind = (
            "reviewed_profile"
            if result.get("support_status") == "independently_reviewed"
            else (
                "necessary_conditions"
                if any(c.get("category") == "scope" for c in result.get("claims", []))
                and result.get("preparation_kind") != "factual_fallback"
                else "available_facts"
            )
        )
    elif covered == {"source", "target"}:
        kind = (
            "reviewed_comparison"
            if result.get("support_status") == "independently_reviewed"
            and any(len(v) == 2 for v in claim_roles.values())
            else "paired_context"
        )
    elif covered:
        kind = "source_only_profile" if covered == {"source"} else "target_only_profile"
    else:
        kind = "unavailable"
    return {
        "contract_version": PRESENTATION_CONTRACT,
        "kind": kind,
        "covered_sides": sorted(covered),
        "claim_sides": claim_roles,
        "relation_asserted": False,
        "scope_note": "Descriptions and excerpts do not determine mapping equivalence or incompatibility.",
        "render_revision": packet.get("selection", {}).get("render_revision"),
        "review_receipt_sha256": (
            canonical_hash(result["independent_review"])
            if result.get("independent_review")
            else None
        ),
    }


def revise(previous, contexts, policy):
    """Keep independent approvals byte-for-byte; version changed deterministic outputs."""
    if previous.get("support_status") == "independently_reviewed":
        return previous
    from .generation import FactPacket, grounding, ExplanationOutput
    from .descriptions import selected_support, validate_prepared_support

    packet = enrich(FactPacket.model_validate(previous["support_packet"]), contexts, policy)
    claims = (
        comparison_templates(packet)
        if packet.task == "pair_comparison"
        else profile_templates(packet)
    )[:2]
    meaningful = bool(claims)
    if not claims:
        from .generation import Claim

        facts = [
            f
            for f in packet.facts
            if f.get("category") == "labels"
            and (f.get("value") or {}).get("term_type") == "literal"
        ]
        claims = [
            Claim(text=f["value"]["lexical_form"], fact_ids=[f["fact_id"]], category="key_fact")
            for f in facts[:1]
        ]
    assert grounding(ExplanationOutput(claims=claims), packet)[0] == "validated"
    result = copy.deepcopy(previous)
    result.pop("independent_review", None)
    result["claims"] = [
        c.model_copy(
            update={
                "claim_id": canonical_hash(
                    {
                        "renderer": RENDER_REVISION,
                        "packet": canonical_hash(packet),
                        "claim": c.model_dump(),
                    }
                )
            }
        ).model_dump()
        for c in claims
    ]
    result["support_packet"] = packet.model_dump(mode="json")
    status = "validated" if meaningful else "failed"
    result["manifest"].update(
        packet_hash=canonical_hash(packet),
        prompt_hash=canonical_hash(RENDER_REVISION),
        requested_model="deterministic:" + RENDER_REVISION,
        returned_model=None,
        provider=None,
        response_hash=None,
        parameters_hash=canonical_hash({"render_revision": RENDER_REVISION}),
        status=status,
    )
    result["manifest"].pop("description_review_hash", None)
    result["preparation_kind"], result["support_status"] = selected_support(
        result["claims"], packet, status
    )
    result["support_reasons"] = (
        []
        if meaningful
        else ["No readable definition or supported named necessary condition; terminology only."]
    )
    result["fixture_provenance"] = (
        "Deterministic rendering of admitted original facts; no provider generation or semantic approval."
    )
    result["explanation_id"] = canonical_hash(
        {
            "predecessor": previous["explanation_id"],
            "packet": packet.model_dump(mode="json"),
            "claims": result["claims"],
            "renderer": RENDER_REVISION,
        }
    )
    validate_prepared_support(result, contexts, policy)
    return result
