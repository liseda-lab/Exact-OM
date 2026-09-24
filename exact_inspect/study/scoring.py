"""Researcher-only scoring with explicit positive/negative denominators."""

from __future__ import annotations


def score_response(
    case_kind, acceptable_candidate_ids, response, initial_order=(), *, origin="natural"
):
    """Score one first submission; a draft/absence remains missing."""
    if not response or response.get("workflow_state") != "submitted":
        return {"missing": True, "rr": None}
    ranked = response["ranked_candidate_ids"]
    acceptable = set(acceptable_candidate_ids)
    kind = response["response_type"]
    result = {"missing": False, "rr": None, "response_depth": len(ranked)}
    if case_kind == "answer_present":
        rr = next((1 / i for i, cid in enumerate(ranked, 1) if cid in acceptable), 0.0)
        system_rr = (
            next((1 / i for i, cid in enumerate(initial_order, 1) if cid in acceptable), 0.0)
            if initial_order
            else None
        )
        result.update(
            rr=rr,
            system_rr=system_rr,
            delta_rr=rr - system_rr if system_rr is not None else None,
            hit_at_1=int(bool(ranked) and ranked[0] in acceptable),
            acceptable_omitted=int(not set(ranked) & acceptable),
            uncertain=int(kind == "insufficient_evidence"),
            explicit_none=int(kind == "none_of_these"),
        )
        if origin == "natural" and initial_order:
            system_correct = initial_order[0] in acceptable
            participant_correct = bool(ranked) and ranked[0] in acceptable
            result.update(
                correct_to_wrong=int(system_correct and not participant_correct),
                wrong_to_correct=int(not system_correct and participant_correct),
            )
    elif case_kind == "answer_absent":
        result.update(
            correct_none=int(kind == "none_of_these"),
            false_endorsement=int(bool(ranked)),
            uncertain=int(kind == "insufficient_evidence"),
        )
    else:
        result["excluded_unresolved"] = True
    return result


def summarize(scores):
    """Keep missingness and negative-case outcomes out of positive MRR."""
    positive = [r for r in scores if r.get("rr") is not None]
    negative = [r for r in scores if "correct_none" in r]

    def mean(rows, field):
        observed = [row[field] for row in rows if row.get(field) is not None]
        return sum(observed) / len(observed) if observed else None

    return {
        "missing": sum(r["missing"] for r in scores),
        "excluded_unresolved": sum(bool(r.get("excluded_unresolved")) for r in scores),
        "system_mrr": mean(positive, "system_rr"),
        "mean_delta_rr": mean(positive, "delta_rr"),
        "hit_at_1": mean(positive, "hit_at_1"),
        "acceptable_omission_rate": mean(positive, "acceptable_omitted"),
        "positive_uncertain_rate": mean(positive, "uncertain"),
        "positive_explicit_none_rate": mean(positive, "explicit_none"),
        "mean_positive_response_depth": mean(positive, "response_depth"),
        "natural_positive_change_denominator": sum("correct_to_wrong" in r for r in positive),
        "correct_to_wrong": sum(r.get("correct_to_wrong", 0) for r in positive),
        "wrong_to_correct": sum(r.get("wrong_to_correct", 0) for r in positive),
        "assigned": len(scores),
        "submitted": sum(not r["missing"] for r in scores),
        "positive_submitted": len(positive),
        "negative_submitted": len(negative),
        "mrr": sum(r["rr"] for r in positive) / len(positive) if positive else None,
        "correct_none_rate": (
            sum(r["correct_none"] for r in negative) / len(negative) if negative else None
        ),
        "false_endorsement_rate": (
            sum(r["false_endorsement"] for r in negative) / len(negative) if negative else None
        ),
        "negative_uncertain_rate": (
            sum(r["uncertain"] for r in negative) / len(negative) if negative else None
        ),
    }
