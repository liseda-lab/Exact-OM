"""Development-only stage attribution from frozen final source traces.

The six ceilings have different, explicit reachable sets. They are conditional
upper bounds over the supplied positive reference, not simulated future model
runs and not evidence that an unlabelled prediction is false.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Iterable, Mapping

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from exact.experiments.paper_metrics import normalize_relation

Pair = tuple[str, str]
TypedPair = tuple[str, str, str]
_CANONICAL_RELATIONS = {
    "equivalence": "=",
    "source_subsumed_by_target": "<",
    "source_subsumes_target": ">",
}
_DEVELOPMENT = {
    "development",
    "dev",
    "validation",
    "valid",
    "diagnostic",
    "oracle",
    "research_development",
}
_POLICIES = {
    "complete_reference",
    "confirmed_negatives",
    "positive_unlabelled",
    "unknown",
    "not_applicable",
}


def _score(row: Mapping[str, Any]) -> float:
    for name in ("P_rank", "S_select", "S_pair_final", "S_base", "S_final"):
        value = row.get(name)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return 0.0


def _winner(candidates: list[dict[str, Any]]) -> str | None:
    for row in candidates:
        trace = row.get("selector_explanation") or {}
        if isinstance(trace, str):
            trace = json.loads(trace)
        if trace.get("source_decision_target") is not None:
            return str(trace["source_decision_target"])
    if not candidates:
        return None
    return str(min(candidates, key=lambda row: (-_score(row), str(row["target"])))["target"])


def _cardinality_ceiling(
    pairs: set[Pair], source_cap: int | None, target_cap: int | None
) -> set[Pair]:
    """Exact sparse maximum-cardinality bipartite flow, without dense assignment caps."""
    if not pairs:
        return set()
    if any(value is not None and value < 1 for value in (source_cap, target_cap)):
        raise ValueError("Oracle cardinalities must be positive or unbounded")
    sources = {value: index + 1 for index, value in enumerate(sorted({s for s, _ in pairs}))}
    targets = {
        value: index + len(sources) + 1 for index, value in enumerate(sorted({t for _, t in pairs}))
    }
    sink = len(sources) + len(targets) + 1
    edges = [(0, index, source_cap or len(pairs)) for index in sources.values()]
    edges += [(index, sink, target_cap or len(pairs)) for index in targets.values()]
    edges += [(sources[source], targets[target], 1) for source, target in sorted(pairs)]
    graph = csr_matrix(
        (
            [capacity for _, _, capacity in edges],
            ([start for start, _, _ in edges], [end for _, end, _ in edges]),
        ),
        shape=(sink + 1, sink + 1),
        dtype="int64",
    )
    flow = maximum_flow(graph, 0, sink).flow
    return {
        (source, target) for source, target in pairs if flow[sources[source], targets[target]] > 0
    }


def attribute_source_errors(
    trace: Mapping[str, Any],
    reference_rows: Iterable[Mapping[str, Any]],
    *,
    reference_role: str,
    negative_label_policy: str,
    confirmed_negatives: Iterable[Pair] = (),
    nil_sources: Iterable[str] = (),
) -> dict[str, Any]:
    """Attribute known errors; no model calls, fitted state changes, or inferred negatives.

    References are positive rows with SrcEntity/TgtEntity (or Src/Tgt), optionally
    Relation using canonical symbols or recognized report aliases (including <=/>=).
    Complete-reference policy explicitly declares closed-world
    negatives within this source universe; other policies require explicit negative
    pairs or NIL source labels to count a false positive.
    """
    role = str(reference_role).strip().lower()
    if role not in _DEVELOPMENT:
        raise ValueError(
            "E00 attribution and oracle ceilings require a development/diagnostic reference role"
        )
    if negative_label_policy not in _POLICIES:
        raise ValueError("Unknown negative-label policy for E00 attribution")
    if (
        trace.get("schema_version") != 2
        or trace.get("stage") != "after_cardinality_and_relation_typing"
    ):
        raise ValueError("E00 requires the final schema-2 source decision trace")
    if trace.get("source_universe_status") != "declared":
        raise ValueError("E00 requires an explicit frozen source universe")
    universe = set(map(str, trace["source_universe"]))
    source_rows = {str(row["Src"]): row for row in trace["records"]}
    if set(source_rows) != universe or len(source_rows) != len(trace["records"]):
        raise ValueError("Final source trace must contain every eligible source exactly once")
    gold: set[TypedPair] = set()
    excluded = 0
    for row in reference_rows:
        source, target = row.get("SrcEntity", row.get("Src")), row.get("TgtEntity", row.get("Tgt"))
        relation = _CANONICAL_RELATIONS.get(normalize_relation(str(row.get("Relation", "="))))
        if source is None or target is None or relation not in {"=", "<", ">"}:
            raise ValueError("E00 reference rows require source, target, and a canonical relation")
        if str(source) not in universe:
            excluded += 1
            continue
        gold.add((str(source), str(target), relation))
    gold_pairs = {(source, target) for source, target, _ in gold}
    negative_pairs = {(str(source), str(target)) for source, target in confirmed_negatives}
    known_nil = set(map(str, nil_sources)) & universe
    if negative_pairs & gold_pairs or known_nil & {source for source, _ in gold_pairs}:
        raise ValueError("Contradictory positive and negative/NIL reference labels")
    if negative_label_policy == "complete_reference":
        known_nil |= universe - {source for source, _ in gold_pairs}

    def negative(pair: Pair) -> bool:
        return pair not in gold_pairs and (
            negative_label_policy == "complete_reference"
            or pair in negative_pairs
            or pair[0] in known_nil
        )

    def metrics(predicted: set[Pair], positives: set[Pair] | None = None) -> dict[str, Any]:
        positives = gold_pairs if positives is None else positives
        tp = len(predicted & positives)
        fp = sum(negative(pair) for pair in predicted)
        unknown = len(predicted) - tp - fp
        fn = len(positives - predicted)
        return {
            "tp": tp,
            "fp_confirmed": fp,
            "fn_known_positive": fn,
            "unassessed_predictions": unknown,
            "precision": None if unknown else tp / (tp + fp) if tp + fp else 0.0,
            "recall_known_positive": tp / len(positives) if positives else None,
            "f1": None if unknown else 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }

    candidates_all: set[Pair] = set()
    winners: set[Pair] = set()
    threshold_pairs: set[Pair] = set()
    pre_typing: set[Pair] = set()
    emitted: set[Pair] = set()
    emitted_typed: set[TypedPair] = set()
    anchor_sources: set[str] = set()
    records: list[dict[str, Any]] = []
    for source in sorted(universe):
        row = source_rows[source]
        candidates = row.get("candidates", [])
        available = {(source, str(item["target"])) for item in candidates}
        if len(available) != len(candidates):
            raise ValueError("E00 candidate source trace must contain distinct target concepts")
        candidates_all |= available
        winner = _winner(candidates)
        winner_pair = (source, winner) if winner is not None else None
        if winner_pair is not None:
            winners.add(winner_pair)
        local_gold = {pair for pair in gold_pairs if pair[0] == source}
        selected = {(source, str(target)) for target in row.get("pre_typing_targets", [])}
        observed = {(source, str(target)) for target in row.get("emitted_targets", [])}
        if not selected <= available or not observed <= selected:
            raise ValueError(
                "E00 selected/emitted pairs must belong to the frozen candidate universe"
            )
        pre_typing |= selected
        emitted |= observed
        protected = {
            (source, str(item["target"])) for item in candidates if item.get("protected_exact")
        }
        if any(negative(pair) for pair in protected):
            anchor_sources.add(source)
        eligible = {
            (source, str(item["target"])) for item in candidates if item.get("threshold_positive")
        }
        threshold_pairs |= eligible
        typed = {
            (source, str(item["target"]), str(item.get("relation", "=")))
            for item in candidates
            if item.get("emitted")
        }
        emitted_typed |= typed
        flags = {
            "candidate_loss": bool(local_gold - available),
            "in_pool_ranking": bool(local_gold & available) and winner_pair not in local_gold,
            "acceptance_nil": (
                winner_pair in local_gold
                and winner_pair not in selected
                and winner_pair not in eligible
            )
            or (source in known_nil and bool(observed)),
            "exact_anchors": any(negative(pair) for pair in protected),
            "collisions": any(
                (source, str(item["target"])) in local_gold
                and item.get("reason") == "cardinality_or_extraction"
                for item in candidates
            ),
            "typing": any(
                pair in selected
                and not any(
                    (pair[0], pair[1], relation) in typed
                    for relation in ("=", "<", ">")
                    if (pair[0], pair[1], relation) in gold
                )
                for pair in local_gold
            ),
        }
        records.append(
            {
                "Src": source,
                "known_nil": source in known_nil,
                "reference_status": (
                    "positive" if local_gold else "nil" if source in known_nil else "unassessed"
                ),
                "ranked_target": winner,
                "flags": flags,
                "missing_positive_targets": sorted(target for _, target in local_gold - available),
                "unassessed_emitted_targets": sorted(
                    target
                    for pair_source, target in observed
                    if (pair_source, target) not in gold_pairs
                    and not negative((pair_source, target))
                ),
                "metrics": metrics(observed, local_gold),
            }
        )

    policy = trace.get("policy", {})
    collision_gold = _cardinality_ceiling(
        threshold_pairs & gold_pairs,
        policy.get("source_cardinality"),
        policy.get("target_cardinality"),
    )
    anchor_corrected = {pair for pair in emitted if pair[0] not in anchor_sources}
    anchor_corrected |= {pair for pair in gold_pairs if pair[0] in anchor_sources}
    supports = {
        "retrieval": (
            gold_pairs,
            "Ideal candidate availability; downstream ranking, acceptance and typing assumed perfect.",
        ),
        "ranking": (
            gold_pairs & candidates_all,
            "Perfect ranking within the frozen pool; acceptance/cardinality/typing relaxed.",
        ),
        "acceptance_nil": (
            gold_pairs & winners,
            "Perfect accept/abstain with the frozen ranked target per source; typing/cardinality relaxed.",
        ),
        "exact_anchors": (
            anchor_corrected,
            "Correct only hard anchors with confirmed wrong endpoints; other emitted decisions frozen; released competitors not rescored.",
        ),
        "collisions": (
            collision_gold,
            "Gold-only assignment within threshold-eligible pairs; both configured cardinality constraints retained; hard anchors relaxed.",
        ),
        "typing": (
            gold_pairs & pre_typing,
            "Perfect canonical relation for already selected pairs; includes typer abstentions, never changes the selected pair universe.",
        ),
    }
    ceilings: dict[str, dict[str, Any]] = {
        name: {
            "definition": definition,
            "metrics": metrics(pairs),
            "reachable_pairs": [list(pair) for pair in sorted(pairs)],
        }
        for name, (pairs, definition) in supports.items()
    }
    ceilings["typing"]["correct_typed_pairs"] = [
        list(item) for item in sorted(gold) if item[:2] in pre_typing
    ]

    def typed_metrics(predicted: set[TypedPair]) -> dict[str, Any]:
        tp = len(predicted & gold)
        fp = sum(
            item not in gold and (item[:2] in gold_pairs or negative(item[:2]))
            for item in predicted
        )
        unknown = len(predicted) - tp - fp
        fn = len(gold - predicted)
        return {
            "tp": tp,
            "fp_confirmed": fp,
            "fn_known_positive": fn,
            "unassessed_predictions": unknown,
            "recall_known_positive": tp / len(gold) if gold else None,
            "f1": None if unknown else 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }

    ceilings["typing"]["typed_metrics"] = typed_metrics(
        {item for item in gold if item[:2] in pre_typing}
    )
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "deployable": False,
        "reference_role": role,
        "negative_label_policy": negative_label_policy,
        "source_universe": sorted(universe),
        "outside_universe_reference_rows": excluded,
        "metric_scope": "supplied_known_positive_reference",
        "flags_can_overlap": True,
        "observed": metrics(emitted),
        "observed_correct_typed_pairs": len(emitted_typed & gold),
        "observed_typed": typed_metrics(emitted_typed),
        "error_source_counts": dict(
            Counter(name for row in records for name, active in row["flags"].items() if active)
        ),
        "oracle_ceilings": ceilings,
        "records": records,
    }
