"""Nested source-label budgets and a conservative frozen count policy."""

from __future__ import annotations

import random
from .fitting import fingerprint, freeze_json


def select_label_budget(frame, reference_pairs, *, budget, seed, selection="passive"):
    reference = {(str(source), str(target)) for source, target in reference_pairs}
    positive_sources = {
        str(row.Src) for row in frame.itertuples() if (str(row.Src), str(row.Tgt)) in reference
    }
    order = sorted(positive_sources)
    random.Random(seed).shuffle(order)
    if selection == "uncertainty":
        uncertainty = {}
        for source, group in frame.groupby("Src", sort=False):
            scores = sorted(group.S_final.astype(float), reverse=True)
            gap = scores[0] - scores[1] if len(scores) > 1 else scores[0]
            uncertainty[str(source)] = 1.0 - min(1.0, max(0.0, gap))
        order.sort(key=lambda source: (-uncertainty[source], source))
    elif selection != "passive":
        raise ValueError("Label selection must be passive or uncertainty")
    count = len(order) if budget is None else min(int(budget), len(order))
    selected = set(order[:count])
    selected_reference = (
        reference if budget is None else {pair for pair in reference if pair[0] in selected}
    )
    selected_frame = frame[frame.Src.astype(str).isin(selected)]
    kind_counts = {}
    for row in selected_frame.itertuples():
        kind_counts.setdefault(str(getattr(row, "SrcKind", "class")), set()).add(str(row.Src))
    manifest = {
        "schema_version": 1,
        "kind": "label_budget",
        "seed": int(seed),
        "selection": selection,
        "requested_groups": budget,
        "available_groups": len(order),
        "effective_groups": count,
        "selected_sources": order[:count],
        "nested_order_sha256": fingerprint(order),
        "pool_sha256": fingerprint(frame[["Src", "Tgt"]].values.tolist()),
        "reference_sha256": fingerprint(sorted(reference)),
        "count_definition": "positive_candidate_source_groups",
        "kind_counts": {kind: len(sources) for kind, sources in kind_counts.items()},
        "component_units": {
            "rerank": count,
            "accept": len(
                {source for source, _ in selected_reference} & set(frame.Src.astype(str))
            ),
            "calibration": len(
                {source for source, _ in selected_reference} & set(frame.Src.astype(str))
            ),
            "fusion": count,
            "llm": count,
        },
        "component_definitions": {
            "rerank": "positive_candidate_source_groups",
            "accept": "annotated_candidate_source_groups",
            "calibration": "annotated_candidate_source_groups",
            "fusion": "positive_candidate_source_groups",
            "llm": "positive_candidate_source_groups",
        },
        "pair_rows": len(selected_frame),
        "known_positive_pairs": len(selected_reference),
        "all_available": budget is None or count < int(budget),
    }
    return selected_reference, manifest


def fit_count_policy(records, path, *, component, binding, practical_effect=0.0):
    """Freeze a crossover only when a paired interval supports the gain.

    Three noisy curve points can legitimately leave a label-free fixed policy.
    Inputs are permitted development paired comparisons, never final outcomes.
    """
    if any(record.get("role") != "development" for record in records):
        raise ValueError("Count policy selection requires development comparisons")
    supported = [
        record
        for record in records
        if float(record["gain_ci_low"]) > practical_effect and int(record["effective_groups"]) > 0
    ]
    minimum = min((int(record["effective_groups"]) for record in supported), default=None)
    payload = {
        "schema_version": 1,
        "kind": "supervision_count_policy",
        "binding": binding,
        "count_definition": "positive_candidate_source_groups",
        "components": {
            component: {
                "minimum_groups": minimum,
                "crossover": "supported" if minimum is not None else "inconclusive",
            }
        },
        "fallback": "label_free",
        "comparisons": records,
        "practical_effect": practical_effect,
    }
    return freeze_json(path, payload)
