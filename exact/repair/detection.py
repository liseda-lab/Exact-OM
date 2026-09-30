"""Sound, incomplete restriction-aware rejection with asserted proof supports.

Positive Horn completion keeps each existential witness separate. Universals and
ranges constrain every matching witness; distinct existentials never identify
successors. Recursion/work bounds cause misses, never successful acceptance.
No normalization symbols or inferred axioms escape into the asserted input.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pyowl_core as owl

from .records import PolicyV2, ProofSupportV3, canonical_hash

RULE_VERSION = "repair-horn/v3"


def _supported(expression: Any) -> bool:
    if isinstance(expression, owl.Class):
        return True
    if isinstance(expression, owl.ObjectIntersectionOf):
        return all(_supported(x) for x in expression.operands)
    if isinstance(expression, (owl.ObjectSomeValuesFrom, owl.ObjectAllValuesFrom)):
        return isinstance(expression.property, owl.ObjectProperty) and _supported(expression.filler)
    return False


def _rules(axioms: tuple[Any, ...]) -> tuple[list[tuple[Any, Any, frozenset]], tuple[Any, ...]]:
    rules: list[tuple[Any, Any, frozenset]] = []
    omitted: list[Any] = []
    for axiom in axioms:
        pairs = []
        if isinstance(axiom, owl.SubClassOf):
            pairs = [(axiom.sub_class, axiom.super_class)]
        elif isinstance(axiom, owl.EquivalentClasses):
            pairs = [(a, b) for a in axiom.expressions for b in axiom.expressions if a != b]
        elif isinstance(axiom, owl.DisjointClasses):
            pairs = [
                (owl.ObjectIntersectionOf(owl.CanonicalSet((a, b))), owl.OWL_NOTHING)
                for a, b in combinations(axiom.expressions, 2)
            ]
        elif isinstance(axiom, owl.ObjectPropertyDomain) and isinstance(
            axiom.property, owl.ObjectProperty
        ):
            pairs = [(owl.ObjectSomeValuesFrom(axiom.property, owl.OWL_THING), axiom.domain)]
        elif isinstance(axiom, owl.ObjectPropertyRange) and isinstance(
            axiom.property, owl.ObjectProperty
        ):
            pairs = [(owl.OWL_THING, owl.ObjectAllValuesFrom(axiom.property, axiom.range))]
        elif isinstance(axiom, owl.Declaration):
            continue
        if not pairs or any(not _supported(a) or not _supported(b) for a, b in pairs):
            omitted.append(axiom)
        else:
            rules.extend((a, b, frozenset((axiom,))) for a, b in pairs)
    return rules, tuple(omitted)


class _Completion:
    def __init__(self, axioms: tuple[Any, ...], max_contexts: int, max_depth: int):
        self.rules, self.omitted = _rules(axioms)
        self.max_contexts, self.max_depth = max_contexts, max_depth
        self.contexts = 0
        self.cache: dict[Any, dict[Any, frozenset]] = {}
        self.running: set[Any] = set()
        self.truncated = False

    def complete(self, seeds: dict[Any, frozenset], depth: int = 0) -> dict[Any, frozenset]:
        key = tuple(
            sorted(((x, s) for x, s in seeds.items()), key=lambda pair: canonical_hash(pair))
        )
        if key in self.cache:
            return self.cache[key]
        if key in self.running or self.contexts >= self.max_contexts or depth > self.max_depth:
            self.truncated = True
            return {**seeds, owl.OWL_THING: frozenset()}
        self.contexts += 1
        self.running.add(key)
        facts = {**seeds, owl.OWL_THING: frozenset()}

        def put(expression: Any, support: frozenset) -> bool:
            if expression not in facts:
                facts[expression] = support
                return True
            return False

        def entails(expression: Any, current: dict[Any, frozenset]) -> frozenset | None:
            if expression in current:
                return current[expression]
            if isinstance(expression, owl.ObjectIntersectionOf):
                supports = [entails(x, current) for x in expression.operands]
                if all(x is not None for x in supports):
                    return frozenset().union(*(x for x in supports if x is not None))
            if isinstance(expression, owl.ObjectSomeValuesFrom):
                for some, support in list(current.items()):
                    if (
                        isinstance(some, owl.ObjectSomeValuesFrom)
                        and some.property == expression.property
                    ):
                        witness = successor(some, support, current)
                        result = entails(expression.filler, witness)
                        if result is not None:
                            return support | result
            return None

        def successor(
            some: Any, support: frozenset, current: dict[Any, frozenset]
        ) -> dict[Any, frozenset]:
            initial = {some.filler: support}
            for universal, premises in list(current.items()):
                if (
                    isinstance(universal, owl.ObjectAllValuesFrom)
                    and universal.property == some.property
                ):
                    initial[universal.filler] = (
                        initial.get(universal.filler, frozenset()) | premises | support
                    )
            return self.complete(initial, depth + 1)

        changed = True
        while changed and owl.OWL_NOTHING not in facts:
            changed = False
            for expression, support in list(facts.items()):
                if isinstance(expression, owl.ObjectIntersectionOf):
                    for operand in expression.operands:
                        changed |= put(operand, support)
            for lhs, rhs, premise in self.rules:
                rule_support = entails(lhs, facts)
                if rule_support is not None:
                    changed |= put(rhs, rule_support | premise)
            for expression, support in list(facts.items()):
                if isinstance(expression, owl.ObjectSomeValuesFrom):
                    witness = successor(expression, support, facts)
                    if owl.OWL_NOTHING in witness:
                        changed |= put(owl.OWL_NOTHING, support | witness[owl.OWL_NOTHING])
        self.running.remove(key)
        self.cache[key] = facts
        return facts


def detect_violations(
    axioms: tuple[Any, ...],
    active: tuple[Any, ...],
    policy: PolicyV2,
    *,
    max_contexts: int = 512,
    max_depth: int = 16,
) -> tuple[ProofSupportV3, ...]:
    """Export sufficient monotone violations; an empty tuple has no positive meaning."""
    if max_contexts < 1 or max_depth < 0:
        raise ValueError("detector bounds must be positive contexts/nonnegative depth")
    axioms = tuple(sorted(set(axioms), key=canonical_hash))
    active = tuple(sorted(set(active), key=canonical_hash))
    theory_hash = canonical_hash((axioms, active))
    completion = _Completion(axioms, max_contexts, max_depth)
    proofs = []
    classes = tuple(
        owl.Class(owl.IRI(c)) if isinstance(c, str) else c for c in policy.monitored_classes
    )
    exceptions = {owl.Class(owl.IRI(c)) if isinstance(c, str) else c for c in policy.exceptions}
    queries = [("consistency", owl.OWL_THING, None)]
    queries += [
        ("class_satisfiability", c, None)
        for c in classes
        if c not in exceptions and c != owl.OWL_NOTHING
    ]
    queries += [("active_satisfiability", expression, expression) for expression in active]
    for kind, query, activation in queries:
        if not _supported(query):
            continue
        facts = completion.complete({query: frozenset()})
        if owl.OWL_NOTHING in facts:
            proofs.append(
                ProofSupportV3(
                    theory_hash,
                    policy.content_hash,
                    kind,
                    query,
                    tuple(sorted(facts[owl.OWL_NOTHING], key=canonical_hash)),
                    activation,
                )
            )
    for query in policy.prohibited:
        if isinstance(query, owl.SubClassOf) and _supported(query.sub_class):
            facts = completion.complete({query.sub_class: frozenset()})
            if query.super_class in facts or owl.OWL_NOTHING in facts:
                support = (
                    facts[query.super_class]
                    if query.super_class in facts
                    else facts[owl.OWL_NOTHING]
                )
                proofs.append(
                    ProofSupportV3(
                        theory_hash,
                        policy.content_hash,
                        "prohibited_entailment",
                        query,
                        tuple(sorted(support, key=canonical_hash)),
                    )
                )
    return tuple(proofs)


def validate_proof(
    proof: ProofSupportV3, axioms: tuple[Any, ...], active: tuple[Any, ...], policy: PolicyV2
) -> bool:
    """Recheck rule evidence and exact scope before allowing a permanent presence cut."""
    canonical_axioms = tuple(sorted(set(axioms), key=canonical_hash))
    canonical_active = tuple(sorted(set(active), key=canonical_hash))
    if (
        not isinstance(proof, ProofSupportV3)
        or not proof.complete
        or proof.rule_version != RULE_VERSION
        or proof.policy_hash != policy.content_hash
        or proof.theory_hash != canonical_hash((canonical_axioms, canonical_active))
        or not set(proof.asserted_support) <= set(axioms)
    ):
        return False
    # Re-run on asserted premises only. Full-policy scope and activation must match.
    derived = detect_violations(proof.asserted_support, canonical_active, policy)
    return any(
        p.kind == proof.kind
        and p.query == proof.query
        and p.activation_expression == proof.activation_expression
        for p in derived
    )
