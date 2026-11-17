"""Pure prompt and probability helpers for listwise LLM decisions."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

LISTWISE_NONE_KEY = "__NONE__"


@dataclass(frozen=True)
class ListwiseCandidate:
    target_iri: str
    score: float
    brief: str


@dataclass(frozen=True)
class ListwiseCall:
    source_iri: str
    permutation_index: int
    sample_index: int
    seed: int
    temperature: float
    candidate_ids: Tuple[str, ...]
    prompt: Mapping[str, str]


@dataclass(frozen=True)
class ListwiseCallPlan:
    source_iri: str
    mode: str
    selected_candidate_ids: Tuple[str, ...]
    overflow_candidate_ids: Tuple[str, ...]
    calls: Tuple[ListwiseCall, ...]
    candidate_scores: Tuple[float, ...] = ()


@dataclass(frozen=True)
class ListwiseAggregate:
    categorical: Mapping[str, float]
    pair_probabilities: Mapping[str, float]
    p_none: float
    call_count: int
    probability_mode: str
    max_normalized_diagnostic: bool


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


def _derived_seed(*parts: object) -> int:
    digest = hashlib.sha256("\x00".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _normalize_candidates(
    candidates: Sequence[Mapping[str, Any] | ListwiseCandidate]
) -> List[ListwiseCandidate]:
    normalized: List[ListwiseCandidate] = []
    for candidate in candidates:
        if isinstance(candidate, ListwiseCandidate):
            item = candidate
        elif isinstance(candidate, Mapping):
            try:
                item = ListwiseCandidate(
                    target_iri=str(candidate["target_iri"]),
                    score=float(candidate["score"]),
                    brief=str(candidate["brief"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "each listwise candidate requires target_iri, finite score, and brief"
                ) from exc
        else:
            raise ValueError("listwise candidates must be mappings or ListwiseCandidate values")
        if not item.target_iri:
            raise ValueError("listwise target_iri must not be empty")
        if not math.isfinite(item.score):
            raise ValueError("listwise candidate scores must be finite")
        normalized.append(item)
    ids = [candidate.target_iri for candidate in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("listwise candidate target IRIs must be unique per source")
    return normalized


def build_listwise_call_plan(
    *,
    source_iri: str,
    source_label: str,
    source_summary: str,
    candidates: Sequence[Mapping[str, Any] | ListwiseCandidate],
    mode: str,
    request_seed: int,
    max_candidates: int = 5,
    permutations: int = 1,
) -> ListwiseCallPlan:
    """Freeze candidates before ordering; primary E07 uses one deterministic call.

    Candidate truncation happens before permutation. ``listwise`` makes one
    deterministic call per permutation; the primary uses one permutation.
    ``listwise_sc`` makes three samples at temperature 0.7 over two permutations.
    """

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"listwise", "listwise_sc"}:
        raise ValueError("listwise call-plan mode must be 'listwise' or 'listwise_sc'")
    cap = int(max_candidates)
    if cap < 1 or cap > 5:
        raise ValueError("E07 listwise max_candidates must be between 1 and 5")
    if permutations not in {1, 2}:
        raise ValueError("listwise permutations must be 1 or 2")
    source_id = str(source_iri)
    if not source_id:
        raise ValueError("source_iri must not be empty")
    ordered = sorted(
        _normalize_candidates(candidates),
        key=lambda candidate: (-candidate.score, candidate.target_iri),
    )
    if not ordered:
        raise ValueError("empty candidate pools require the no-candidate fallback")
    selected = ordered[:cap]
    overflow = ordered[cap:]

    first_order = list(selected)
    random.Random(_derived_seed(source_id, int(request_seed), "permutation", 0)).shuffle(
        first_order
    )
    permutation_orders = [first_order]
    if permutations == 2 or normalized_mode == "listwise_sc":
        permutation_orders.append(list(reversed(first_order)))
    samples_per_permutation = 3 if normalized_mode == "listwise_sc" else 1
    temperature = 0.7 if normalized_mode == "listwise_sc" else 0.0
    calls: List[ListwiseCall] = []
    for permutation_index, order in enumerate(permutation_orders):
        briefs = [candidate.brief for candidate in order]
        prompt, _ = build_listwise_decision_prompt(source_label, source_summary, briefs)
        for sample_index in range(samples_per_permutation):
            calls.append(
                ListwiseCall(
                    source_iri=source_id,
                    permutation_index=permutation_index,
                    sample_index=sample_index,
                    seed=_derived_seed(
                        source_id,
                        int(request_seed),
                        "permutation",
                        permutation_index,
                        "sample",
                        sample_index,
                    ),
                    temperature=temperature,
                    candidate_ids=tuple(candidate.target_iri for candidate in order),
                    prompt=dict(prompt),
                )
            )
    return ListwiseCallPlan(
        source_iri=source_id,
        mode=normalized_mode,
        selected_candidate_ids=tuple(candidate.target_iri for candidate in selected),
        overflow_candidate_ids=tuple(candidate.target_iri for candidate in overflow),
        calls=tuple(calls),
        candidate_scores=tuple(candidate.score for candidate in selected),
    )


def aggregate_listwise_call_probabilities(
    plan: ListwiseCallPlan,
    call_probabilities: Sequence[Mapping[str, float]],
    *,
    probability_mode: str,
    eps: float = 1.0e-12,
) -> ListwiseAggregate:
    """Map calls back to target IRIs and average their raw categorical probabilities."""

    if len(call_probabilities) != len(plan.calls):
        raise ValueError(
            f"listwise aggregate expected {len(plan.calls)} calls, got {len(call_probabilities)}"
        )
    outcomes = (*plan.selected_candidate_ids, LISTWISE_NONE_KEY)
    probability_sums = {outcome: 0.0 for outcome in outcomes}

    for call, raw in zip(plan.calls, call_probabilities):
        labels = listwise_labels(len(call.candidate_ids))
        missing = [label for label in labels if label not in raw]
        if missing:
            raise ValueError(f"Missing listwise probabilities: {', '.join(missing)}")
        values = {label: float(raw[label]) for label in labels}
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("listwise call probabilities must be finite and non-negative")
        total = sum(values.values())
        if total <= float(eps):
            raise ValueError("listwise call probabilities must have positive mass")
        values = {label: value / total for label, value in values.items()}
        mapped = {
            target_iri: values[label] for label, target_iri in zip(labels[:-1], call.candidate_ids)
        }
        mapped[LISTWISE_NONE_KEY] = values["Z"]
        for outcome, value in mapped.items():
            probability_sums[outcome] += value

    call_count = len(plan.calls)
    categorical = {outcome: probability_sums[outcome] / call_count for outcome in outcomes}
    total = sum(categorical.values())
    categorical = {outcome: value / total for outcome, value in categorical.items()}
    p_none = categorical[LISTWISE_NONE_KEY]
    maximum_real = max(categorical[target] for target in plan.selected_candidate_ids)
    normalized_mode = str(probability_mode).strip().lower()
    pair_probabilities: Dict[str, float] = {}
    for target in plan.selected_candidate_ids:
        probability = categorical[target]
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
        pair_probabilities[target] = float(min(1.0, max(0.0, value)))
    return ListwiseAggregate(
        categorical=categorical,
        pair_probabilities=pair_probabilities,
        p_none=float(p_none),
        call_count=call_count,
        probability_mode=normalized_mode,
        max_normalized_diagnostic=normalized_mode == "max_normalized",
    )


__all__ = [
    "LISTWISE_NONE_KEY",
    "ListwiseAggregate",
    "ListwiseCall",
    "ListwiseCallPlan",
    "ListwiseCandidate",
    "aggregate_listwise_call_probabilities",
    "build_listwise_decision_prompt",
    "build_listwise_call_plan",
    "categorical_probabilities_from_logprobs",
    "listwise_labels",
    "transform_listwise_probabilities",
]
