"""No-call E25 diagnostics over frozen, safely labeled source interventions."""

from __future__ import annotations

import math

from .fitting import fingerprint, safe_training_labels

NONE = "__NONE__"


def _inventory(population, reference_pairs, negative_label_policy):
    if not {"Src", "Tgt", "S_base", "U"}.issubset(population):
        raise ValueError("Oracle replay requires frozen Src/Tgt/S_base/U evidence")
    if any(
        not math.isfinite(float(value))
        for row in population[["S_base", "U"]].values
        for value in row
    ):
        raise ValueError("Oracle population scores/uncertainties must be finite")
    if population.duplicated(["Src", "Tgt"]).any():
        raise ValueError("Oracle population contains duplicate candidate pairs")
    labeled, reference = safe_training_labels(
        population, reference_pairs, {"negative_label_policy": negative_label_policy}
    )
    known_pairs = set(zip(labeled.Src.astype(str), labeled.Tgt.astype(str)))
    groups, excluded = {}, []
    for source, group in population.groupby("Src", sort=True):
        source = str(source)
        pairs = set(zip(group.Src.astype(str), group.Tgt.astype(str)))
        if not pairs <= known_pairs or not any(pair[0] == source for pair in reference):
            excluded.append({"source": source, "reason": "incomplete_source_labels"})
        else:
            groups[source] = group
    return groups, reference, excluded


def _winner(scores, threshold):
    if not scores:
        return None
    target = min(scores, key=lambda value: (-scores[value], value))
    return target if scores[target] >= threshold else None


def _outcome(
    group, probabilities, choice, reference, *, threshold, beta, fusion_weight, constant_weight
):
    source = str(group.iloc[0].Src)
    baseline = {str(row.Tgt): float(row.S_base) for row in group.itertuples()}
    if any(not math.isfinite(value) for value in baseline.values()):
        raise ValueError("Nonfinite frozen oracle scores")
    before = _winner(baseline, threshold)
    if fusion_weight == "source_first":
        if choice is None or choice not in {NONE, *probabilities}:
            raise ValueError("Source-first replay needs a cached canonical choice")
        after = choice if choice != NONE and baseline[choice] >= threshold else None
    else:
        mixed = dict(baseline)
        for row in group.itertuples():
            target = str(row.Tgt)
            if target in probabilities:
                weight = (
                    constant_weight
                    if fusion_weight == "constant"
                    else max(0.0, min(1.0, beta * float(row.U)))
                )
                mixed[target] = (1 - weight) * baseline[target] + weight * probabilities[target]
        after = _winner(mixed, threshold)
    before_correct = (source, before) in reference
    after_correct = (source, after) in reference
    return {
        "source": source,
        "baseline_choice": before,
        "intervention_choice": after,
        "baseline_correct": before_correct,
        "intervention_correct": after_correct,
        "net_correction": int(after_correct) - int(before_correct),
    }


def _payload(
    population,
    selected,
    decision_probs,
    choices,
    outcomes,
    excluded,
    *,
    mode,
    protocol,
    budget,
    negative_label_policy,
    threshold,
    beta,
    fusion_weight,
    constant_weight,
    teacher_binding,
):
    if fusion_weight not in {"beta_u", "constant", "source_first"}:
        raise ValueError("Oracle replay requires a frozen supported integration rule")
    population_rows = (
        population[["Src", "Tgt", "S_base", "U"]]
        .sort_values(["Src", "Tgt"], kind="mergesort")
        .to_dict("records")
    )
    return {
        "schema_version": 1,
        "kind": "llm_oracle_replay",
        "mode": mode,
        "protocol": protocol,
        "selected_sources": selected,
        "decision_probs": decision_probs,
        "source_choices": choices,
        "no_llm_invocations": True,
        "deployable": False,
        "teacher_binding": teacher_binding,
        "counterfactuals": outcomes,
        "excluded_sources": excluded,
        "budget": int(budget),
        "budget_kind": "maximum_source_interventions",
        "selected_count": len(selected),
        "selection_unit": "source",
        "population_rows": population_rows,
        "population_sha256": fingerprint(population_rows),
        **(
            {"dataset_signature": teacher_binding["dataset_signature"]}
            if teacher_binding and teacher_binding.get("dataset_signature")
            else {}
        ),
        "negative_label_policy": negative_label_policy,
        "fixed_fusion": {
            "threshold": threshold,
            "beta": beta,
            "fusion_weight": fusion_weight,
            "constant_weight": constant_weight,
        },
        "interpretation": "source_independent_fixed_acceptance_diagnostic",
    }


def observed_response_oracle(
    population,
    observed_records,
    reference_pairs,
    *,
    budget,
    negative_label_policy,
    threshold=0.5,
    beta=0.8,
    fusion_weight="beta_u",
    constant_weight=0.5,
    teacher_binding=None,
):
    """Select at most budget cached interventions by observed correction minus harm.

    Exact ties use canonical source IDs. Missing/invalid responses are ineligible;
    no probability or response is synthesized for an unobserved candidate.
    """
    if not 0 <= budget <= 200:
        raise ValueError("Oracle intervention budget must be between zero and 200 sources")
    groups, reference, excluded = _inventory(population, reference_pairs, negative_label_policy)
    records = {}
    for record in observed_records:
        source = str(record["source"])
        if source in records:
            raise ValueError("Oracle replay contains duplicate observed source decisions")
        records[source] = record
    probabilities, choices, outcomes = {}, {}, []
    for source, group in groups.items():
        record = records.get(source)
        if not record or not record.get("valid"):
            excluded.append({"source": source, "reason": "no_valid_observed_response"})
            continue
        values = {
            str(target): float(value) for target, value in record["pair_probabilities"].items()
        }
        if (
            not values
            or not set(values) <= set(group.Tgt.astype(str))
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values())
        ):
            raise ValueError(
                "Observed oracle response is incompatible with the frozen candidate population"
            )
        probabilities[source], choices[source] = values, record.get("choice")
        outcomes.append(
            _outcome(
                group,
                values,
                record.get("choice"),
                reference,
                threshold=threshold,
                beta=beta,
                fusion_weight=fusion_weight,
                constant_weight=constant_weight,
            )
        )
    ordered = sorted(outcomes, key=lambda row: (-row["net_correction"], row["source"]))
    # Under a maximum intervention budget, a harmful call is never compulsory.
    selected = [row["source"] for row in ordered if row["net_correction"] >= 0][:budget]
    return _payload(
        population,
        selected,
        {source: probabilities[source] for source in selected},
        {source: choices[source] for source in selected},
        outcomes,
        excluded,
        mode="oracle_replay",
        protocol="observed_cached",
        budget=budget,
        negative_label_policy=negative_label_policy,
        threshold=threshold,
        beta=beta,
        fusion_weight=fusion_weight,
        constant_weight=constant_weight,
        teacher_binding=teacher_binding,
    )


def perfect_intervention(
    population,
    reference_pairs,
    selected_sources,
    *,
    displayed_candidates,
    negative_label_policy,
    threshold=0.5,
    beta=0.8,
    fusion_weight="beta_u",
    constant_weight=0.5,
    teacher_binding=None,
):
    """Perfect probabilities on exactly the observed oracle's selected sources/support."""
    groups, reference, excluded = _inventory(population, reference_pairs, negative_label_policy)
    selected = list(selected_sources)
    if (
        len(selected) > 200
        or len(set(selected)) != len(selected)
        or any(source not in groups for source in selected)
    ):
        raise ValueError("Perfect intervention requires the same safely labeled source population")
    probabilities, choices, outcomes = {}, {}, []
    for source in selected:
        group = groups[source]
        targets = set(displayed_candidates[source])
        if not targets or not targets <= set(group.Tgt.astype(str)):
            raise ValueError("Perfect intervention candidate support differs from observed support")
        probabilities[source] = {
            target: float((source, target) in reference) for target in sorted(targets)
        }
        positives = {
            str(row.Tgt): float(row.S_base)
            for row in group.itertuples()
            if str(row.Tgt) in targets and (source, str(row.Tgt)) in reference
        }
        choices[source] = _winner(positives, threshold) or NONE
        outcomes.append(
            _outcome(
                group,
                probabilities[source],
                choices[source],
                reference,
                threshold=threshold,
                beta=beta,
                fusion_weight=fusion_weight,
                constant_weight=constant_weight,
            )
        )
    return _payload(
        population,
        selected,
        probabilities,
        choices,
        outcomes,
        excluded,
        mode="oracle_perfect",
        protocol="perfect_fixed_sources",
        budget=len(selected),
        negative_label_policy=negative_label_policy,
        threshold=threshold,
        beta=beta,
        fusion_weight=fusion_weight,
        constant_weight=constant_weight,
        teacher_binding=teacher_binding,
    )
