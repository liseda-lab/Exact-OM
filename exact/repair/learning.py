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

    def __post_init__(self) -> None:
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
