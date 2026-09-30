"""Typed symbolic supervision, bounded labels and full-repair learning utilities.

Only this evaluator-side module sees desired/unwanted queries. Graph construction
accepts public observations independently; neural dependencies are imported lazily.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, cast

import pyowl_core as owl


class TeacherOracle(Protocol):
    """Three-valued queries over one frozen repaired theory."""

    def consistent(self) -> bool | None:
        """Decide consistency, or return unknown."""
        ...

    def entails(self, axiom: Any) -> bool | None:
        """Decide a typed consequence, or return unknown."""
        ...

    def satisfiable(self, expression: Any) -> bool | None:
        """Decide class-expression satisfiability, or return unknown."""
        ...


@dataclass(frozen=True)
class TeacherProbe:
    """A declared desired/unwanted consequence with typed non-vacuity conditions."""

    probe_id: str
    axiom: Any
    family: str
    desired: bool = True
    nonvacuity: tuple[Any, ...] | None = None

    def conditions(self) -> tuple[Any, ...]:
        """Derive typed conditions; never treat a disjoint intersection as nonempty."""
        if self.nonvacuity is not None:
            return self.nonvacuity
        if isinstance(self.axiom, owl.SubClassOf):
            if self.axiom.super_class == owl.OWL_NOTHING and isinstance(
                self.axiom.sub_class, owl.ObjectIntersectionOf
            ):
                return tuple(self.axiom.sub_class.operands)
            return (self.axiom.sub_class,)
        if isinstance(self.axiom, owl.DisjointClasses):
            return tuple(self.axiom.expressions)
        if isinstance(self.axiom, owl.EquivalentClasses):
            return tuple(self.axiom.expressions)
        if isinstance(self.axiom, (owl.ObjectPropertyDomain, owl.ObjectPropertyRange)):
            return (
                owl.ObjectSomeValuesFrom(
                    self.axiom.property, owl.Class(owl.IRI("http://www.w3.org/2002/07/owl#Thing"))
                ),
            )
        if isinstance(self.axiom, owl.SubObjectPropertyOf):
            if isinstance(self.axiom.sub_property, owl.ObjectProperty):
                return (
                    owl.ObjectSomeValuesFrom(
                        self.axiom.sub_property,
                        owl.Class(owl.IRI("http://www.w3.org/2002/07/owl#Thing")),
                    ),
                )
            raise ValueError("Property-chain probes require explicit typed non-vacuity")
        if isinstance(
            self.axiom,
            (
                owl.ClassAssertion,
                owl.ObjectPropertyAssertion,
                owl.DataPropertyAssertion,
                owl.SameIndividual,
                owl.DifferentIndividuals,
            ),
        ):
            # Named individuals denote domain elements in a consistent OWL theory.
            return ()
        raise ValueError("This probe type requires explicit typed non-vacuity conditions")


@dataclass(frozen=True)
class ProbeOutcome:
    """Keep raw unwanted entailment distinct from desired restoration credit."""

    probe_id: str
    family: str
    desired: bool
    entailed: bool | None
    nonvacuous: bool | None
    credit: bool | None


@dataclass(frozen=True)
class TeacherResult:
    """A complete semantic vector, or an explicitly incomplete unknown result."""

    consistent: bool | None
    outcomes: tuple[ProbeOutcome, ...]
    benefit: float | None

    @property
    def complete(self) -> bool:
        """Whether a complete scalar is justified by all required outcomes."""
        return self.consistent is True and self.benefit is not None


def _all_known(values: Iterable[bool | None]) -> bool | None:
    values = tuple(values)
    if any(value is None for value in values):
        return None
    return all(values)


def evaluate_teacher(
    oracle: TeacherOracle,
    probes: Sequence[TeacherProbe],
    *,
    desired_weights: Mapping[str, float] | None = None,
    unwanted_weights: Mapping[str, float] | None = None,
) -> TeacherResult:
    """Score frozen semantic families only after consistency and complete query labels."""
    if len({probe.probe_id for probe in probes}) != len(probes):
        raise ValueError("Probe IDs must be unique")
    weights = (desired_weights or {}, unwanted_weights or {})
    if any(not math.isfinite(v) or v < 0 for row in weights for v in row.values()):
        raise ValueError("Teacher family weights must be finite and nonnegative")
    consistency = oracle.consistent()
    if consistency is not True:
        return TeacherResult(consistency, (), None)
    outcomes = []
    satisfiability_cache: dict[str, bool | None] = {}
    for probe in probes:
        entailed = oracle.entails(probe.axiom)
        conditions = probe.conditions() if probe.desired else ()
        values = []
        for expression in conditions:
            key = owl.structural_hexdigest(expression)
            if key not in satisfiability_cache:
                satisfiability_cache[key] = oracle.satisfiable(expression)
            values.append(satisfiability_cache[key])
        nonvacuous = _all_known(values)
        # Retain unknown raw labels: a known-false conjunction must not mask coverage.
        credit = (
            (None if entailed is None or nonvacuous is None else entailed and nonvacuous)
            if probe.desired
            else entailed
        )
        outcomes.append(
            ProbeOutcome(probe.probe_id, probe.family, probe.desired, entailed, nonvacuous, credit)
        )
    if any(outcome.credit is None for outcome in outcomes):
        return TeacherResult(True, tuple(outcomes), None)
    families: dict[tuple[bool, str], list[bool]] = {}
    for outcome in outcomes:
        families.setdefault((outcome.desired, outcome.family), []).append(bool(outcome.credit))
    benefit = sum(
        (1.0 if desired else -1.0)
        * weights[0 if desired else 1].get(family, 1.0)
        * sum(values)
        / len(values)
        for (desired, family), values in families.items()
    )
    return TeacherResult(True, tuple(outcomes), benefit)


@dataclass(frozen=True)
class RepairLabel:
    """One whole-repair label; unknown policy or query decisions stay masked."""

    assignment: tuple[int, ...]
    feasible: bool | None
    benefit: float | None
    cost: float
    semantic_vector: tuple[ProbeOutcome, ...] = ()

    def __post_init__(self) -> None:
        if self.feasible is not None and type(self.feasible) is not bool:
            raise ValueError("Feasibility must be true, false, or unknown")
        if not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError("Teacher cost must be finite and nonnegative")
        if self.benefit is not None and (
            not math.isfinite(self.benefit) or self.feasible is not True
        ):
            raise ValueError("Only verified feasible repairs have finite semantic benefit")

    @property
    def usable(self) -> bool:
        """Whether this row may supervise complete-repair benefit."""
        return self.feasible is True and self.benefit is not None


@dataclass(frozen=True)
class TeacherCache:
    """Frozen finite-universe labels with completeness and dependency identity."""

    candidate_counts: tuple[int, ...]
    labels: tuple[RepairLabel, ...]
    complete: bool
    stop_reason: str
    hashes: tuple[tuple[str, str], ...]
    elapsed_seconds: float
    schema: str = "exact-repair/teacher-cache/v2"

    def __post_init__(self) -> None:
        if self.schema not in {"exact-repair/teacher-cache/v2", "exact-repair/teacher-cache/v3"}:
            raise ValueError("Unknown teacher-cache schema")
        if any(type(count) is not int or count < 1 for count in self.candidate_counts):
            raise ValueError("Teacher cache inventories must be nonempty")
        seen = set()
        for label in self.labels:
            if (
                len(label.assignment) != len(self.candidate_counts)
                or any(
                    type(choice) is not int or not 0 <= choice < count
                    for choice, count in zip(label.assignment, self.candidate_counts)
                )
                or label.assignment in seen
            ):
                raise ValueError("Teacher cache labels must identify distinct valid assignments")
            seen.add(label.assignment)
        if self.complete and (
            len(seen) != math.prod(self.candidate_counts)
            or any(label.feasible is not False and not label.usable for label in self.labels)
        ):
            raise ValueError("A complete teacher cache requires every assignment and query decided")

    @property
    def coverage(self) -> dict[str, int]:
        """Return denominators including unknown and unvisited assignments."""
        return {
            "requested": math.prod(self.candidate_counts),
            "visited": len(self.labels),
            "usable": sum(label.usable for label in self.labels),
            "unknown_policy": sum(label.feasible is None for label in self.labels),
            "unknown_queries": sum(
                label.feasible is True and label.benefit is None for label in self.labels
            ),
        }


def enumerate_teacher(
    candidate_counts: Sequence[int],
    label_assignment: Callable[[tuple[int, ...]], RepairLabel],
    *,
    hashes: Mapping[str, str],
    max_assignments: int = 256,
    deadline_seconds: float = 10.0,
) -> TeacherCache:
    """Bound whole-case enumeration without assuming conflict-component independence.

    Callbacks must enforce their own reasoner-call deadline. This outer deadline
    prevents starting another label once the case budget is exhausted.
    """
    required = {"input", "patch", "policy", "query", "inventory", "backend"}
    if not required <= hashes.keys() or any(not hashes[key] for key in required):
        raise ValueError("Teacher cache requires input/patch/policy/query/inventory/backend hashes")
    if any(type(n) is not int or n < 1 for n in candidate_counts):
        raise ValueError("Every object must have a nonempty frozen candidate inventory")
    if max_assignments < 0 or deadline_seconds <= 0 or not math.isfinite(deadline_seconds):
        raise ValueError("Invalid teacher enumeration budget")
    counts = tuple(candidate_counts)
    start = time.monotonic()
    labels: list[RepairLabel] = []
    reason = "complete"
    for assignment in itertools.product(*(range(n) for n in counts)):
        if len(labels) >= max_assignments:
            reason = "assignment_cap"
            break
        if time.monotonic() - start >= deadline_seconds:
            reason = "deadline"
            break
        label = label_assignment(assignment)
        if label.assignment != assignment:
            raise ValueError("Teacher callback returned a different assignment")
        if not math.isfinite(label.cost) or label.cost < 0:
            raise ValueError("Teacher costs must be finite and nonnegative")
        if label.benefit is not None and not math.isfinite(label.benefit):
            raise ValueError("Teacher benefit must be finite")
        if label.feasible is not True and label.benefit is not None:
            raise ValueError("Infeasible/unknown assignments cannot earn semantic benefit")
        labels.append(label)
    decided = all(label.feasible is False or label.usable for label in labels)
    complete = len(labels) == math.prod(counts) and decided
    if reason == "complete" and not decided:
        reason = "unknown_labels"
    return TeacherCache(
        counts,
        tuple(labels),
        complete,
        reason,
        tuple(sorted(hashes.items())),
        time.monotonic() - start,
    )


def teacher_marginals(
    cache: TeacherCache, temperature: float = 1.0
) -> tuple[tuple[float, ...], ...]:
    """Normalise full-repair utility only for an exhaustive decided feasible universe."""
    if not cache.complete:
        raise ValueError("Exact teacher marginals require a complete finite cache")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    feasible = [label for label in cache.labels if label.usable]
    if not feasible:
        raise ValueError("The teacher universe has no verified feasible assignment")
    utilities = [(cast(float, label.benefit) - label.cost) / temperature for label in feasible]
    maximum = max(utilities)
    weights = [math.exp(value - maximum) for value in utilities]
    normalizer = sum(weights)
    marginals = [[0.0] * n for n in cache.candidate_counts]
    for label, weight in zip(feasible, weights):
        for index, choice in enumerate(label.assignment):
            marginals[index][choice] += weight / normalizer
    return tuple(tuple(row) for row in marginals)


def feasible_anchor(labels: Sequence[RepairLabel]) -> int | None:
    """Prefer actual feasible, complete lowest-cost labels, with stable assignment ties."""
    usable = [index for index, label in enumerate(labels) if label.usable]
    return min(usable, key=lambda i: (labels[i].cost, labels[i].assignment)) if usable else None


def benefit_losses(
    predictions: Any,
    labels: Sequence[RepairLabel],
    *,
    beta: float = 1.0,
    rank_temperature: float = 1.0,
    max_pairs: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Learn anchored full-repair differences and unequal-benefit pair rankings."""
    import torch.nn.functional as functional

    if predictions.ndim != 1 or len(predictions) != len(labels):
        raise ValueError("Provide one semantic prediction per whole-repair label")
    if beta <= 0 or rank_temperature <= 0 or max_pairs < 0:
        raise ValueError("Loss scales must be positive and pair count nonnegative")
    anchor = feasible_anchor(labels)
    zero = predictions.sum() * 0.0
    if anchor is None:
        return {"value": zero, "rank": zero, "usable": 0, "pairs": 0, "anchor": None}
    usable = [index for index, label in enumerate(labels) if label.usable]
    target = predictions.new_tensor([labels[index].benefit for index in usable])
    target_anchor = cast(float, labels[anchor].benefit)
    value = functional.smooth_l1_loss(
        predictions[usable] - predictions[anchor], target - target_anchor, beta=beta
    )
    # Reservoir sampling bounds memory even for a large labelled finite universe.
    sampled: list[tuple[int, int]] = []
    rng, seen = random.Random(seed), 0
    for first, second in itertools.combinations(usable, 2):
        if labels[first].benefit == labels[second].benefit:
            continue
        seen += 1
        if len(sampled) < max_pairs:
            sampled.append((first, second))
        elif max_pairs:
            replace = rng.randrange(seen)
            if replace < max_pairs:
                sampled[replace] = (first, second)
    rank = zero
    if sampled:
        first_indices, second_indices = zip(*sampled)
        signs = predictions.new_tensor(
            [
                1.0 if cast(float, labels[i].benefit) > cast(float, labels[j].benefit) else -1.0
                for i, j in sampled
            ]
        )
        rank = functional.softplus(
            -signs
            * (predictions[list(first_indices)] - predictions[list(second_indices)])
            / rank_temperature
        ).mean()
    return {
        "value": value,
        "rank": rank,
        "usable": len(usable),
        "pairs": len(sampled),
        "anchor": anchor,
    }


def proposal_loss(
    log_probabilities: Sequence[Any], cache: TeacherCache, temperature: float = 1.0
) -> Any:
    """Cross-entropy against exact teacher marginals, including caller's circuit log Z."""
    import torch

    marginals = teacher_marginals(cache, temperature)
    if len(log_probabilities) != len(marginals):
        raise ValueError("One normalised candidate log-probability vector is required per object")
    terms = []
    for probabilities, target in zip(log_probabilities, marginals):
        if probabilities.ndim != 1 or len(probabilities) != len(target):
            raise ValueError("Proposal candidate vectors must match the frozen teacher inventory")
        if not torch.isclose(
            torch.logsumexp(probabilities, 0), probabilities.new_tensor(0.0), atol=1e-5
        ):
            raise ValueError("Candidate probabilities must include the exact normaliser")
        weights = probabilities.new_tensor(target)
        positive = weights > 0
        terms.append(-(weights[positive] * probabilities[positive]).sum())
    if not terms:
        raise ValueError("Proposal supervision requires at least one editable object")
    return torch.stack(terms).sum()


def grouped_split(
    parent_families: Mapping[str, str],
    *,
    seed: int = 13,
    train_fraction: float = 0.7,
    development_fraction: float = 0.15,
    heldout_families: Iterable[str] = (),
) -> dict[str, str]:
    """Assign clean structural parents before any renamed/corrupted siblings exist."""
    if train_fraction < 0 or development_fraction < 0 or train_fraction + development_fraction > 1:
        raise ValueError("Split fractions must be nonnegative and sum to at most one")
    heldout = frozenset(heldout_families)
    result = {}
    for parent, family in sorted(parent_families.items()):
        if family in heldout:
            result[parent] = "test"
            continue
        digest = hashlib.sha256(f"{seed}\0{parent}".encode()).digest()
        unit = int.from_bytes(digest[:8], "big") / 2**64
        result[parent] = (
            "train"
            if unit < train_fraction
            else "development" if unit < train_fraction + development_fraction else "test"
        )
    return result


def fit_cost_preferences(
    benefit_differences: Any,
    feature_differences: Any,
    preferred: Any,
    initial_weights: Any,
    *,
    steps: int = 100,
    learning_rate: float = 0.05,
    regularization: float = 0.1,
) -> Any:
    """Fit nonnegative profile weights from labelled complete-repair pair choices only."""
    import torch
    import torch.nn.functional as functional

    if len(preferred) == 0:
        raise ValueError("Preference fitting requires labelled choices")
    if feature_differences.shape != (len(preferred), len(initial_weights)):
        raise ValueError("Preference feature dimensions do not match")
    if (initial_weights < 0).any() or regularization < 0 or steps < 0 or learning_rate <= 0:
        raise ValueError("Invalid nonnegative preference profile or fitting budget")
    initial = initial_weights.detach().clone()
    raw = torch.nn.Parameter(torch.log(torch.expm1(initial.clamp_min(1e-4))))
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    benefit = benefit_differences.detach()
    features, targets = feature_differences.detach(), preferred.detach()
    for _ in range(steps):
        optimizer.zero_grad()
        weights = functional.softplus(raw)
        loss = functional.binary_cross_entropy_with_logits(benefit - features @ weights, targets)
        loss = loss + regularization * (weights - initial).square().mean()
        loss.backward()
        optimizer.step()
    return functional.softplus(raw).detach()


class OwlTeacherOracle:
    """Batch all frozen soft probes through one shared-snapshot verifier session."""

    def __init__(self, verifier: Any, snapshot: Any, probes: Sequence[TeacherProbe]) -> None:
        conditions = tuple(
            {condition for probe in probes if probe.desired for condition in probe.conditions()}
        )
        report = verifier.check_theory(
            snapshot, (), required=tuple(probe.axiom for probe in probes), activated=conditions
        )
        self._outcomes = {
            (item.kind, item.query_id): item.verdict if item.complete else None
            for item in report.obligations
        }
        self._supported = report.support.input_supported and report.support.complete_imports

    @staticmethod
    def _key(value: Any) -> str:
        return hashlib.sha256(value.canonical_bytes()).hexdigest()

    def consistent(self) -> bool | None:
        """Return the actual consistency obligation, never a soft-query aggregate."""
        return self._outcomes.get(("consistency", "consistency")) if self._supported else None

    def entails(self, axiom: Any) -> bool | None:
        """Read the batch's typed entailment result."""
        return self._outcomes.get(("required_entailment", self._key(axiom)))

    def satisfiable(self, expression: Any) -> bool | None:
        """Read the batch's whole-expression non-vacuity result."""
        return self._outcomes.get(("active_satisfiability", self._key(expression)))


def interaction_loss(
    predictions: Any, labels: Sequence[RepairLabel], *, max_quartets: int = 64, beta: float = 1.0
) -> dict[str, Any]:
    """Feasible, complete counterfactual differences on identical backgrounds."""
    import torch
    import torch.nn.functional as functional

    if len(predictions) != len(labels) or max_quartets < 0:
        raise ValueError("Invalid quartet predictions or budget")
    lookup = {label.assignment: i for i, label in enumerate(labels) if label.usable}
    seen: set[tuple[int, int, int, int]] = set()
    contrasts: list[Any] = []
    targets: list[float] = []
    for base, i00 in sorted(lookup.items()):
        if len(contrasts) >= max_quartets:
            break
        for i, j in itertools.combinations(range(len(base)), 2):
            # Only the two changed positions differ; no invented infeasible benefit.
            for other, i11 in sorted(lookup.items()):
                if not (base[i] < other[i] and base[j] < other[j]):
                    continue
                if any(base[k] != other[k] for k in range(len(base)) if k not in {i, j}):
                    continue
                a10, a01 = list(base), list(base)
                a10[i], a01[j] = other[i], other[j]
                if tuple(a10) not in lookup or tuple(a01) not in lookup:
                    continue
                indices = i00, lookup[tuple(a10)], lookup[tuple(a01)], i11
                if indices in seen or len(contrasts) >= max_quartets:
                    continue
                seen.add(indices)
                b00, b10, b01, b11 = indices
                contrasts.append(
                    predictions[b11] - predictions[b10] - predictions[b01] + predictions[b00]
                )
                targets.append(
                    cast(float, labels[b11].benefit)
                    - cast(float, labels[b10].benefit)
                    - cast(float, labels[b01].benefit)
                    + cast(float, labels[b00].benefit)
                )
    loss = (
        functional.smooth_l1_loss(
            torch.stack(contrasts), predictions.new_tensor(targets), beta=beta
        )
        if contrasts
        else predictions.sum() * 0
    )
    return {"loss": loss, "eligible": len(contrasts), "targets": tuple(targets)}


def risk_loss(logits: Any, labels: Sequence[RepairLabel]) -> dict[str, Any]:
    """Only decided whole-policy labels supervise failure probability."""
    import torch.nn.functional as functional

    if logits.ndim != 1 or len(logits) != len(labels):
        raise ValueError("Risk requires one logit per complete assignment")
    known = [i for i, label in enumerate(labels) if label.feasible is not None]
    target = logits.new_tensor([float(labels[i].feasible is False) for i in known])
    return {
        "loss": (
            functional.binary_cross_entropy_with_logits(logits[known], target)
            if known
            else logits.sum() * 0
        ),
        "eligible": len(known),
        "unknown": len(labels) - len(known),
    }


def sample_conditioned_marginals(
    cache: TeacherCache, temperature: float = 1.0
) -> tuple[tuple[float, ...], ...]:
    """Empirical unique-plan utility distribution, never an exhaustive teacher claim."""
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Temperature must be positive")
    usable = [label for label in cache.labels if label.usable]
    if not usable:
        raise ValueError("Sample contains no eligible feasible semantic targets")
    utilities = [(cast(float, label.benefit) - label.cost) / temperature for label in usable]
    weights = [math.exp(value - max(utilities)) for value in utilities]
    total = sum(weights)
    rows = [[0.0] * size for size in cache.candidate_counts]
    for label, weight in zip(usable, weights):
        for i, choice in enumerate(label.assignment):
            rows[i][choice] += weight / total
    return tuple(tuple(row) for row in rows)


def covered_proposal_loss(
    log_probabilities: Any,
    target: Sequence[float],
    *,
    target_kind: str,
    reachable_subset: bool = False,
    elementary: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Account unreachable target mass before any explicitly projected objective."""
    import torch

    if target_kind not in {"exact", "sample_conditioned"}:
        raise ValueError("Declare exact or sample_conditioned target provenance")
    weights = log_probabilities.new_tensor(target)
    if weights.shape != log_probabilities.shape or not torch.isclose(
        weights.sum(), weights.new_tensor(1.0)
    ):
        raise ValueError("Target must be a normalized inventory distribution")
    positive = weights > 0
    missing = positive & ~torch.isfinite(log_probabilities)
    missing_mass = float(weights[missing].sum())
    selected = positive & ~missing
    status = "supervised"
    if missing_mass and not reachable_subset:
        status, selected = "unreachable_target_excluded", torch.zeros_like(positive)
    elif missing_mass:
        status = "reachable_subset_projected"
        weights = weights / weights[selected].sum() if selected.any() else weights
    loss = -(weights[selected] * log_probabilities[selected]).sum()
    if not selected.any():
        loss = torch.where(torch.isfinite(log_probabilities), log_probabilities, 0.0).sum() * 0
    partitions = {}
    if elementary is not None:
        if len(elementary) != len(target):
            raise ValueError("Target partition must match candidate inventory")
        for name, flag in (("elementary", True), ("complex", False)):
            mask = weights.new_tensor([value == flag for value in elementary], dtype=torch.bool)
            partitions[name] = {
                "target_mass": float(log_probabilities.new_tensor(target)[mask].sum()),
                "missing_mass": float(log_probabilities.new_tensor(target)[mask & missing].sum()),
            }
    return {
        "loss": loss,
        "status": status,
        "target_kind": target_kind,
        "missing_target_mass": missing_mass,
        "partitions": partitions,
        "renormalized": bool(missing_mass and reachable_subset),
        "eligible": int(selected.any()),
    }


@dataclass(frozen=True)
class SampledRepairRound:
    """A bounded supervised acquisition ledger; every scheduled row is retained."""

    case_id: str
    parent_group_id: str
    split: str
    inventory_hash: str
    model_hash: str
    sampler_hash: str
    round_id: str
    cache: TeacherCache
    selections: tuple[tuple[tuple[int, ...], str, float | None], ...]
    requested: int
    duplicates: int
    stop_reason: str
    assignment_maps: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    schema: str = "exact-repair/sampled-round/v3"


def collect_sampled_repairs(
    candidate_counts: Sequence[int],
    label_assignment: Callable,
    *,
    case_id: str,
    parent_group_id: str,
    split: str,
    hashes: Mapping[str, str],
    model_hash: str,
    round_id: str,
    max_assignments: int,
    deadline_seconds: float,
    seed: int = 0,
    proposed: Sequence[tuple[tuple[int, ...], str]] = (),
    object_candidate_ids: Sequence[tuple[str, Sequence[str]]] = (),
    exploration_fraction: float = 0.5,
    counterfactual_attempts: int | None = None,
    quartet_attempts: int = 0,
) -> SampledRepairRound:
    """Freeze sampler, acquire independent controls plus declared model alternatives.

    First draws come from uniform controls and single-object counterfactuals;
    optimizer/learned proposals have unknown propensity. Training splits only.
    No rejection-resampling removes unknown or infeasible strata.
    """
    from .records import canonical_hash

    if split != "train":
        raise ValueError("Active supervised acquisition is training-only")
    counts = tuple(candidate_counts)
    if (
        any(type(n) is not int or n < 1 for n in counts)
        or max_assignments < 0
        or deadline_seconds <= 0
    ):
        raise ValueError("Invalid sample inventory/budgets")
    if object_candidate_ids and (
        len(object_candidate_ids) != len(counts)
        or len({row[0] for row in object_candidate_ids}) != len(counts)
        or any(
            len(ids) != count or len(set(ids)) != count
            for (_, ids), count in zip(object_candidate_ids, counts)
        )
    ):
        raise ValueError("Candidate identity map does not match sample inventory")
    rng = random.Random(seed)
    scheduled = list(proposed)
    if not 0 <= exploration_fraction <= 1 or quartet_attempts < 0:
        raise ValueError("Invalid sampling stratum budget")
    random_count = int(max_assignments * exploration_fraction)
    random_rows = [
        (tuple(rng.randrange(n) for n in counts), "uniform_control") for _ in range(random_count)
    ]
    base = proposed[0][0] if proposed else tuple(0 for _ in counts)
    counterfactuals = []
    for i, n in enumerate(counts):
        for choice in range(n):
            if choice != base[i]:
                changed = list(base)
                changed[i] = choice
                counterfactuals.append((tuple(changed), "counterfactual"))
    if counterfactual_attempts is not None:
        counterfactuals = counterfactuals[:counterfactual_attempts]
    quartets: list[tuple[tuple[int, ...], str]] = []
    for i, j in itertools.combinations(range(len(counts)), 2):
        if len(quartets) >= quartet_attempts:
            break
        if counts[i] > 1 and counts[j] > 1:
            for left, right in ((0, 0), (1, 0), (0, 1), (1, 1)):
                row = list(base)
                row[i], row[j] = left, right
                quartets.append((tuple(row), "counterfactual_quartet"))
    scheduled = random_rows + scheduled + quartets[:quartet_attempts] + counterfactuals
    scheduled = scheduled[:max_assignments]
    started, labels, selections, seen, duplicates = time.monotonic(), [], [], set(), 0
    reason = "scheduled_complete"
    for assignment, stratum in scheduled:
        if len(assignment) != len(counts) or any(
            type(c) is not int or not 0 <= c < n for c, n in zip(assignment, counts)
        ):
            raise ValueError("Sampler supplied invalid assignment")
        if assignment in seen:
            duplicates += 1
            continue
        if time.monotonic() - started >= deadline_seconds:
            reason = "deadline"
            break
        seen.add(assignment)
        label = label_assignment(assignment)
        if label.assignment != assignment:
            raise ValueError("Verifier labeled a different assignment")
        labels.append(label)
        selections.append(
            (assignment, stratum, 1.0 / math.prod(counts) if stratum == "uniform_control" else None)
        )
    cache = TeacherCache(
        counts,
        tuple(labels),
        False,
        "sample_conditioned:" + reason,
        tuple(sorted(hashes.items())),
        time.monotonic() - started,
        "exact-repair/teacher-cache/v3",
    )
    maps = []
    for assignment, _, _ in selections:
        mapping = tuple(
            sorted(
                (name, ids[choice]) for (name, ids), choice in zip(object_candidate_ids, assignment)
            )
        )
        if mapping:
            maps.append((canonical_hash((case_id, hashes["inventory"], mapping)), mapping))
    return SampledRepairRound(
        case_id,
        parent_group_id,
        split,
        hashes["inventory"],
        model_hash,
        canonical_hash((counts, max_assignments, seed, proposed)),
        round_id,
        cache,
        tuple(selections),
        len(scheduled),
        duplicates,
        reason,
        tuple(maps),
    )


def generated_checkpoint_criterion(
    reports: Sequence[Mapping[str, Any]],
    *,
    minimum_coverage: float = 1.0,
    uncertainty_z: float = 1.96,
    fallback: str = "stop",
) -> tuple | None:
    """Coverage gate, external quality lower confidence bound, then verifier effort.

    The schedule denominator includes failed generation and missing labels. Cache
    regret and the model's own value never enter selection.
    """
    if (
        not reports
        or not 0 <= minimum_coverage <= 1
        or uncertainty_z < 0
        or fallback not in {"stop", "exploratory"}
    ):
        raise ValueError("Invalid generated checkpoint selection declaration")
    known = [
        r["decoded"]
        for r in reports
        if r.get("decoded", {}).get("status") == "verified"
        and r["decoded"].get("selected_utility") is not None
    ]
    coverage = len(known) / len(reports)
    if coverage < minimum_coverage and fallback == "stop":
        return None
    if not known:
        return None
    groups: dict[str, list[float]] = {}
    for index, report in enumerate(reports):
        row = report.get("decoded", {})
        if row.get("status") == "verified" and row.get("selected_utility") is not None:
            groups.setdefault(report.get("parent_group_id", str(index)), []).append(
                float(row["selected_utility"])
            )
    values = [sum(group) / len(group) for group in groups.values()]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    lower_bound = mean - uncertainty_z * math.sqrt(variance / len(values))
    effort = sum(float(row.get("checks", 0)) for row in known) / len(known)
    return (-coverage, -lower_bound, effort)


def fidelity_comparison_losses(
    prediction_a: Any,
    prediction_b: Any,
    comparison: Any,
    *,
    temperature: float = 1.0,
    beta: float = 1.0,
) -> dict[str, Any]:
    """Validated weak semantic labels supervise complete plans; abstentions mask all."""
    import torch.nn.functional as functional

    if temperature <= 0 or beta <= 0:
        raise ValueError("Positive fidelity loss scales required")
    zero = (prediction_a + prediction_b) * 0
    if not comparison.global_target_eligible:
        return {"value": zero, "rank": zero, "tie": zero, "eligible": 0, "provenance": "llm_weak"}
    difference = prediction_a - prediction_b
    target = difference.new_tensor(comparison.overall_score_a - comparison.overall_score_b)
    value = functional.smooth_l1_loss(difference, target, beta=beta)
    ranking = (
        functional.softplus((-1 if comparison.decision == "A" else 1) * difference / temperature)
        if comparison.decision in {"A", "B"}
        else zero
    )
    tie = (
        functional.smooth_l1_loss(difference, difference.new_zeros(()), beta=beta)
        if comparison.decision == "tie"
        else zero
    )
    return {"value": value, "rank": ranking, "tie": tie, "eligible": 1, "provenance": "llm_weak"}


def conditional_proposal_loss(
    cache: TeacherCache, log_probability: Callable, *, temperature: float = 1.0
) -> dict[str, Any]:
    """Ordered complete-plan likelihood with frozen prefix per circuit invocation.

    The callable consumes (object index, earlier choices, current choice). It
    must return a normalized circuit log probability with immutable logits.
    """
    import torch

    if temperature <= 0:
        raise ValueError("Conditional proposal temperature must be positive")
    labels = [row for row in cache.labels if row.usable]
    if not labels:
        raise ValueError("No usable conditional proposal target")
    utilities = [(cast(float, row.benefit) - row.cost) / temperature for row in labels]
    weights = [math.exp(value - max(utilities)) for value in utilities]
    total = sum(weights)
    terms, missing_mass = [], 0.0
    for row, weight in zip(labels, weights):
        probabilities = [
            log_probability(i, row.assignment[:i], choice)
            for i, choice in enumerate(row.assignment)
        ]
        log_joint = torch.stack(probabilities).sum()
        if not bool(torch.isfinite(log_joint)):
            missing_mass += weight / total
        terms.append((weight / total, log_joint))
    if missing_mass:
        loss = sum(torch.where(torch.isfinite(value), value, 0.0) * 0 for _, value in terms)
    else:
        loss = -sum(weight * value for weight, value in terms)
    return {
        "loss": loss,
        "missing_target_mass": missing_mass,
        "status": "unreachable_target_excluded" if missing_mass else "supervised",
        "target_kind": (
            "exact_conditional_joint" if cache.complete else "sample_conditioned_conditional_joint"
        ),
        "eligible": 0 if missing_mass else len(labels),
        "order": "canonical_inventory_object_order",
    }
