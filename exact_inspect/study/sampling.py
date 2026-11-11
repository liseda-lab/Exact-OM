"""Private rank-stratified allocation without altering any candidate rank or score."""

from __future__ import annotations


def validate_pool(cases, sampling):
    strata = sampling.strata
    if sorted(s.reference_rank for s in strata) != [1, 2, 3, 4, 5]:
        raise ValueError("Sampling requires exactly the five reference-rank strata")
    ids = [cid for stratum in strata for cid in stratum.case_ids]
    if (
        not 10 <= len(ids) <= 40
        or len(set(ids)) != len(ids)
        or set(ids) != {c.case_id for c in cases}
    ):
        raise ValueError("Sampling requires ten to forty cases, each in exactly one stratum")
    if len({(c.source.ontology_version_id, c.source.iri, c.source.kind) for c in cases}) != len(
        cases
    ):
        raise ValueError("Sampling requires distinct sources")
    if len({c.transfer_group for c in cases}) != len(cases):
        raise ValueError("Sampling pool cannot repeat near-duplicate source groups")


def validate_reference_strata(study, keys):
    # Development review keeps unresolved keys and correctness metrics disabled.
    # Its private selector receipt supplies provisional reference strata for review.
    if study.get("purpose") == "development_review":
        return
    for stratum in study["sampling"]["strata"]:
        for cid in stratum["case_ids"]:
            key = keys[cid]
            accepted = key["acceptable_candidate_ids"]
            if (
                key["case_kind"] != "answer_present"
                or key["origin"] != "natural"
                or len(accepted) != 1
                or key["original_production_ranks"].get(accepted[0]) != stratum["reference_rank"]
            ):
                raise ValueError(
                    "Sampling stratum must match one natural reference answer's original rank"
                )


def sample_blocks(sampling, sequence, rng):
    """Sample two distinct sources per rank, one for each condition; caller freezes it."""
    conditions = ["explanation", "ontology_baseline"]
    if sequence % 2:
        conditions.reverse()
    by_condition = {condition: [] for condition in conditions}
    for stratum in sorted(sampling["strata"], key=lambda item: item["reference_rank"]):
        chosen = rng.sample(sorted(stratum["case_ids"]), 2)
        for condition, cid in zip(conditions, chosen):
            by_condition[condition].append(cid)
    return {
        "schedule_id": "rank-balanced/1/" + conditions[0] + "-first",
        "blocks": [{"condition": c, "case_ids": by_condition[c]} for c in conditions],
    }


def consultation_requested(study, condition):
    scope = study.get("consultation_scope", "each_case")
    return scope == "each_case" or (scope == "baseline_cases" and condition == "ontology_baseline")
