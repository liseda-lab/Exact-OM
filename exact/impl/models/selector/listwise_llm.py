"""Pure prompt and probability helpers for listwise LLM decisions."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple


def listwise_labels(candidate_count: int) -> Tuple[str, ...]:
    """Return the constrained candidate alphabet followed by NIL label ``Z``."""

    count = int(candidate_count)
    if count < 1 or count > 25:
        raise ValueError("listwise candidate_count must be between 1 and 25")
    return tuple(chr(ord("A") + index) for index in range(count)) + ("Z",)


def build_listwise_decision_prompt(
    source_label: str,
    source_summary: str,
    candidate_briefs: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, int | None]]:
    """Build an E07 one-token prompt and its auditable letter-to-index map."""

    labels = listwise_labels(len(candidate_briefs))
    real_labels = labels[:-1]
    options = "\n\n".join(
        f"{label}. {str(brief).strip()}" for label, brief in zip(real_labels, candidate_briefs)
    )
    alphabet = ", ".join(labels)
    prompt = {
        "system": "You compare ontology candidates and return exactly one allowed token.",
        "user": (
            f"Return exactly one token from {{{alphabet}}}.\n"
            "Choose the candidate equivalent to the source, or Z when none is equivalent.\n\n"
            f"Source: {str(source_label).strip()}\n"
            f"Source summary: {str(source_summary).strip()}\n\n"
            f"Candidates:\n{options}"
        ),
    }
    index_by_label: Dict[str, int | None] = {
        label: index for index, label in enumerate(real_labels)
    }
    index_by_label["Z"] = None
    return prompt, index_by_label


def categorical_probabilities_from_logprobs(
    label_logprobs: Mapping[str, float],
    candidate_count: int,
) -> Dict[str, float]:
    """Normalize first-token log probabilities over ``A..,Z`` deterministically."""

    labels = listwise_labels(candidate_count)
    missing = [label for label in labels if label not in label_logprobs]
    if missing:
        raise ValueError(f"Missing listwise label logprobs: {', '.join(missing)}")
    values = [float(label_logprobs[label]) for label in labels]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("listwise label logprobs must be finite")
    maximum = max(values)
    masses = [math.exp(value - maximum) for value in values]
    denominator = sum(masses)
    return {label: float(mass / denominator) for label, mass in zip(labels, masses)}


def transform_listwise_probabilities(
    categorical: Mapping[str, float],
    candidate_count: int,
    mode: str,
    *,
    eps: float = 1.0e-12,
) -> Dict[str, float]:
    """Apply the four pre-registered E07 pair-evidence transformations."""

    labels = listwise_labels(candidate_count)
    missing = [label for label in labels if label not in categorical]
    if missing:
        raise ValueError(f"Missing listwise probabilities: {', '.join(missing)}")
    probabilities = {label: float(categorical[label]) for label in labels}
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities.values()):
        raise ValueError("listwise probabilities must be finite and non-negative")
    total = sum(probabilities.values())
    if total <= float(eps):
        raise ValueError("listwise probabilities must have positive mass")
    probabilities = {label: value / total for label, value in probabilities.items()}
    p_none = probabilities["Z"]
    maximum_real = max(probabilities[label] for label in labels[:-1])
    normalized_mode = str(mode).strip().lower()
    transformed: Dict[str, float] = {}
    for label in labels[:-1]:
        probability = probabilities[label]
        if normalized_mode == "raw_joint":
            value = probability
        elif normalized_mode == "conditional_real":
            value = probability / max(1.0 - p_none, float(eps))
        elif normalized_mode == "pairwise_vs_none":
            value = probability / max(probability + p_none, float(eps))
        elif normalized_mode == "max_normalized":
            value = probability / max(maximum_real, float(eps))
        else:
            raise ValueError(
                "listwise probability mode must be raw_joint, conditional_real, "
                "pairwise_vs_none, or max_normalized"
            )
        transformed[label] = float(min(1.0, max(0.0, value)))
    return transformed


__all__ = [
    "build_listwise_decision_prompt",
    "categorical_probabilities_from_logprobs",
    "listwise_labels",
    "transform_listwise_probabilities",
]
