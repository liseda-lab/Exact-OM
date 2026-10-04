"""E24 known-pair direction diagnostic; gold never selects a scoring hypothesis."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import torch

RELATIONS = ("<", ">")
TOLERANCE = 1e-6


def score_directions(scorer: Any, source: list, target: list) -> dict[str, Any]:
    """Reuse one support matrix for both hypotheses and an actual reversed replay."""
    if not scorer.diff_enabled or not scorer.use_context:
        raise ValueError("E24 requires the enabled difference channel and context evidence")
    if scorer.use_llm or scorer.generate_llm_rationales:
        raise ValueError("E24 diagnostic requires hosted decisions and rationales disabled")
    matrix = scorer._object_support_matrix(source, target)
    if tuple(matrix.shape) != (len(source), len(target)):
        raise ValueError("support matrix shape does not match evidence")
    if not torch.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():
        raise ValueError("support matrix must contain finite probabilities")
    previous = dict(scorer.diff_config)
    scores: dict[str, dict[str, float]] = {}
    try:
        for name, left, right, support in (
            ("forward", source, target, matrix),
            ("reversed", target, source, matrix.T),
        ):
            scores[name] = {}
            for relation in RELATIONS:
                scorer.diff_config = {
                    **previous,
                    "enabled": True,
                    "formulation": "asymmetric",
                    "relation_interpretation": relation,
                }
                value = scorer._score_difference_channel(left, right, support, verbalize=False)
                score = float(value["score"])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("difference score must be a finite probability")
                scores[name][relation] = score
    finally:
        scorer.diff_config = previous
    supported = all(
        any(float(item.get("score", 0)) > 0 for item in items) for items in (source, target)
    ) and bool(matrix.numel() and matrix.max().item() > 0)

    def decision(values):
        delta = values["<"] - values[">"]
        return None if not supported or abs(delta) <= TOLERANCE else ("<" if delta > 0 else ">")

    predicted = decision(scores["forward"])
    reversed_predicted = decision(scores["reversed"])
    reversal_error = max(
        abs(scores["forward"][relation] - scores["reversed"][other])
        for relation, other in (("<", ">"), (">", "<"))
    )
    expected_reverse = {"<": ">", ">": "<", None: None}[predicted]
    return {
        "scores": scores,
        "decision": predicted,
        "status": "unsupported" if not supported else ("tie" if predicted is None else "decided"),
        "source_facts": len(source),
        "target_facts": len(target),
        "reversed_decision": reversed_predicted,
        "reversal_max_error": reversal_error,
        "reversal_pass": reversal_error <= TOLERANCE and reversed_predicted == expected_reverse,
        "symmetric_control_score": sum(scores["forward"].values()) / 2,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report abstention-aware direction metrics separately from equality diagnostics."""
    if any(row["relation"] not in {"=", "<", ">"} for row in rows):
        raise ValueError("diagnostic references must use =, <, >")
    directional = [row for row in rows if row["relation"] in RELATIONS]

    def metrics(control=False):
        per_class = {}
        for relation in RELATIONS:
            group = [row for row in directional if row["relation"] == relation]
            covered = [row for row in group if row["decision"] is not None]
            correct = sum(("<" if control else row["decision"]) == relation for row in covered)
            per_class[relation] = {
                "pairs": len(group),
                "covered": len(covered),
                "correct": correct,
                "coverage": len(covered) / len(group) if group else None,
                "recall_including_abstentions": correct / len(group) if group else None,
                "covered_accuracy": correct / len(covered) if covered else None,
            }
        values = list(per_class.values())

        def macro(key):
            return (
                sum(item[key] for item in values) / 2
                if all(item[key] is not None for item in values)
                else None
            )

        return {
            "per_class": per_class,
            "balanced_accuracy_including_abstentions": macro("recall_including_abstentions"),
            "balanced_accuracy_covered": macro("covered_accuracy"),
            "balanced_coverage": macro("coverage"),
            "coverage": (
                sum(row["decision"] is not None for row in directional) / len(directional)
                if directional
                else None
            ),
        }

    equality = [row for row in rows if row["relation"] == "="]
    supported_equality = [row for row in equality if row["status"] != "unsupported"]
    gaps = [
        abs(row["scores"]["forward"]["<"] - row["scores"]["forward"][">"])
        for row in supported_equality
    ]
    return {
        "pairs": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "directional": metrics(),
        "matched_fixed_less_control": {
            "rule": "always < on exactly the diagnostic's decided pairs",
            **metrics(True),
        },
        "symmetric_control": {
            "rule": "equal directional scores, always abstain",
            "coverage": 0.0 if directional else None,
        },
        "equality": {
            "pairs": len(equality),
            "supported": len(supported_equality),
            "direction_decisions": sum(row["decision"] is not None for row in equality),
            "mean_absolute_direction_gap_supported": sum(gaps) / len(gaps) if gaps else None,
            "max_absolute_direction_gap_supported": max(gaps) if gaps else None,
        },
        "reversal": {
            "pairs": len(rows),
            "failures": sum(not row["reversal_pass"] for row in rows),
            "maximum_error": max((row["reversal_max_error"] for row in rows), default=0.0),
        },
    }
