"""No-call E25 diagnostics with explicit verified-label or benchmark outcomes."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .fitting import fingerprint, safe_training_labels

NONE = "__NONE__"


def _inventory(
    population,
    reference_pairs,
    negative_label_policy,
    nil_sources=(),
    *,
    outcome_semantics="verified_labels",
    ignored_pairs=(),
):
    if outcome_semantics not in {"verified_labels", "benchmark_reference"}:
        raise ValueError("Unknown oracle outcome semantics")
    if tuple(ignored_pairs):
        raise ValueError("Oracle replay does not support ignored-pair scoring semantics")
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
    if population[["Src", "Tgt"]].isna().any().any() or any(
        not str(value) or str(value) == NONE
        for row in population[["Src", "Tgt"]].values
        for value in row
    ):
        raise ValueError("Oracle population requires real nonempty source/candidate identities")
    reference_pairs = {(str(source), str(target)) for source, target in reference_pairs}
    nil_sources = set(map(str, nil_sources))
    if nil_sources & {source for source, _ in reference_pairs}:
        raise ValueError("Explicit NIL sources conflict with positive references")
    population = population.copy()
    if outcome_semantics == "benchmark_reference":
        # This branch never supplies invented labels to the training-label helper.
        # No recorded positive means no credited emitted pair, not ontology-wide NIL.
        groups = {str(source): group for source, group in population.groupby("Src", sort=True)}
        reference_sources = {source for source, _ in reference_pairs}
        reference = reference_pairs | {
            (source, NONE) for source in groups if source not in reference_sources
        }
        return groups, reference, []
    if negative_label_policy == "confirmed_negatives" and nil_sources:
        population.loc[population.Src.astype(str).isin(nil_sources), "confirmed_label"] = 0
    # Reserved NONE is a source label, never a displayed candidate or synthesized answer.
    labeled, reference = safe_training_labels(
        population,
        reference_pairs | {(source, NONE) for source in nil_sources},
        {"negative_label_policy": negative_label_policy},
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
    group,
    probabilities,
    choice,
    reference,
    *,
    threshold,
    beta,
    fusion_weight,
    constant_weight,
    outcome_semantics="verified_labels",
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
    before_correct = (source, before if before is not None else NONE) in reference
    after_correct = (source, after if after is not None else NONE) in reference
    result = {
        "source": source,
        "baseline_choice": before,
        "intervention_choice": after,
    }
    if outcome_semantics == "benchmark_reference":
        result.update(
            baseline_reference_match=before_correct,
            intervention_reference_match=after_correct,
            net_reference_gain=int(after_correct) - int(before_correct),
            reference_abstention=(source, NONE) in reference,
        )
    else:
        result.update(
            baseline_correct=before_correct,
            intervention_correct=after_correct,
            net_correction=int(after_correct) - int(before_correct),
        )
    return result


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
    outcome_semantics="verified_labels",
):
    if fusion_weight not in {"beta_u", "constant", "source_first"}:
        raise ValueError("Oracle replay requires a frozen supported integration rule")
    population_rows = (
        population[["Src", "Tgt", "S_base", "U"]]
        .sort_values(["Src", "Tgt"], kind="mergesort")
        .to_dict("records")
    )
    payload = {
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
    if outcome_semantics == "benchmark_reference":
        payload.update(
            outcome_semantics=outcome_semantics,
            no_training_use=True,
            training_use_permitted=False,
            global_f1_optimality_claim=False,
            probability_semantics=(
                "synthetic_reference_membership"
                if mode == "oracle_perfect"
                else "observed_cached_model_probability"
            ),
            reference_abstention_semantics="no_recorded_positive_not_natural_nil",
            scoring_scope="source_local_fixed_threshold_reference_outcome",
            ignored_reference_pairs=[],
            ignored_pair_policy="reject_nonempty",
        )
    return payload


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
    nil_sources=(),
    outcome_semantics="verified_labels",
    ignored_pairs=(),
):
    """Select at most budget cached interventions by observed correction minus harm.

    Exact ties use canonical source IDs. Missing/invalid responses are ineligible;
    no probability or response is synthesized for an unobserved candidate.
    """
    if not 0 <= budget <= 200:
        raise ValueError("Oracle intervention budget must be between zero and 200 sources")
    groups, reference, excluded = _inventory(
        population,
        reference_pairs,
        negative_label_policy,
        nil_sources,
        outcome_semantics=outcome_semantics,
        ignored_pairs=ignored_pairs,
    )
    records = {}
    for record in observed_records:
        source = str(record["source"])
        if outcome_semantics == "benchmark_reference" and source not in groups:
            raise ValueError("Observed source is outside the frozen candidate population")
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
                outcome_semantics=outcome_semantics,
            )
        )
    gain = "net_reference_gain" if outcome_semantics == "benchmark_reference" else "net_correction"
    ordered = sorted(outcomes, key=lambda row: (-row[gain], row["source"]))
    # Under a maximum intervention budget, a harmful call is never compulsory.
    selected = [row["source"] for row in ordered if row[gain] >= 0][:budget]
    return _payload(
        population,
        selected,
        {source: probabilities[source] for source in selected},
        {source: choices[source] for source in selected},
        outcomes,
        excluded,
        mode="oracle_replay",
        protocol=(
            "benchmark_reference_observed_cached"
            if outcome_semantics == "benchmark_reference"
            else "observed_cached"
        ),
        budget=budget,
        negative_label_policy=negative_label_policy,
        threshold=threshold,
        beta=beta,
        fusion_weight=fusion_weight,
        constant_weight=constant_weight,
        teacher_binding=teacher_binding,
        outcome_semantics=outcome_semantics,
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
    nil_sources=(),
    outcome_semantics="verified_labels",
    ignored_pairs=(),
):
    """Perfect probabilities on exactly the observed oracle's selected sources/support."""
    groups, reference, excluded = _inventory(
        population,
        reference_pairs,
        negative_label_policy,
        nil_sources,
        outcome_semantics=outcome_semantics,
        ignored_pairs=ignored_pairs,
    )
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
        if source not in displayed_candidates:
            raise ValueError("Perfect intervention lacks observed candidate support")
        if outcome_semantics == "benchmark_reference":
            observed = displayed_candidates[source]
            if not isinstance(observed, Mapping) or any(
                not math.isfinite(float(value)) or not 0 <= float(value) <= 1
                for value in observed.values()
            ):
                raise ValueError("Benchmark perfect support requires observed finite probabilities")
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
                outcome_semantics=outcome_semantics,
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
        protocol=(
            "benchmark_reference_perfect_fixed_sources"
            if outcome_semantics == "benchmark_reference"
            else "perfect_fixed_sources"
        ),
        budget=len(selected),
        negative_label_policy=negative_label_policy,
        threshold=threshold,
        beta=beta,
        fusion_weight=fusion_weight,
        constant_weight=constant_weight,
        teacher_binding=teacher_binding,
        outcome_semantics=outcome_semantics,
    )
