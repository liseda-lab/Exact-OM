"""Design conformance only. Run: python -m unittest discover -s reference -v."""

from dataclasses import replace
from itertools import product
import random
import unittest

from kernel_model import (Cut, PresenceCut, State, Verdict, assignments, axiom_origins,
                          rejects_presence, base_states, literals,
                          materialize, repair, replay_optimality, replay_safety,
                          score, weighted_encoding)


def binary(weights):
    return tuple((State(frozenset(), 0), State(frozenset({str(i)}), w))
                 for i, w in enumerate(weights))


def truth_optimum(states, oracle, governed=lambda x: True, interactions=None):
    safe = [x for x in assignments(states) if governed(x)
            and oracle(*materialize(states, x)) is Verdict.SAFE]
    return safe, max((score(states, x, interactions) for x in safe), default=None)


class KernelTests(unittest.TestCase):
    def assert_exact(self, states, oracle, **kwargs):
        governed = kwargs.get("governed", lambda x: True)
        result = repair(states, oracle, **kwargs)
        safe, optimum = truth_optimum(states, oracle, governed, kwargs.get("interactions"))
        if safe:
            self.assertEqual(result.status, "OPTIMAL_IN_POOL")
            self.assertIn(result.assignment, safe)
            self.assertEqual(result.value, optimum)
            self.assertEqual(result.gap, 0)
            self.assertTrue(replay_optimality(states, oracle, result, governed, kwargs.get("interactions")))
        else:
            self.assertEqual(result.status, "NO_FEASIBLE_IN_POOL")
        for cut in result.cuts:
            self.assertTrue(all(not cut.literals <= literals(x) for x in safe))
        self.assertLessEqual(len(result.cuts), len(list(assignments(states))))
        return result

    def test_every_monotone_three_decision_safety_predicate(self):
        subsets = [frozenset(str(i) for i in range(3) if bits & (1 << i))
                   for bits in range(8)]
        predicates = 0
        for mask in range(1 << 7):
            unsafe = {subsets[i + 1] for i in range(7) if mask & (1 << i)}
            if any(a <= b and b not in unsafe for a in unsafe for b in subsets):
                continue
            predicates += 1
            oracle = lambda a, tags: Verdict.UNSAFE if a in unsafe else Verdict.SAFE
            for weights in product((-2, 0, 3), repeat=3):
                self.assert_exact(binary(weights), oracle)
                self.assert_exact(binary(weights), oracle, active_progress=True)
        self.assertEqual(predicates, 19)

    def test_missing_explanations_still_reaches_optimum(self):
        states = binary((5, 9, 5))
        oracle = lambda a, tags: (Verdict.UNSAFE if ({"0", "1"} <= a or {"1", "2"} <= a)
                                 else Verdict.SAFE)
        result = self.assert_exact(states, oracle)
        self.assertEqual(result.assignment, (1, 0, 1))
        self.assertGreater(len(result.cuts), 0)
        self.assertTrue(all(c.kind == "assignment" for c in result.cuts))

    def test_invalid_support_is_discarded(self):
        oracle = lambda a, tags: Verdict.UNSAFE if {"0", "1"} <= a else Verdict.SAFE
        result = self.assert_exact(binary((7, 3)), oracle,
                                   propose_support=lambda x: [(0, 1)])
        self.assertTrue(all(c.kind == "assignment" for c in result.cuts))

    def test_unknown_support_retains_progress_cut(self):
        def oracle(a, tags):
            if a == frozenset({"0"}):
                return Verdict.UNKNOWN
            return Verdict.UNSAFE if {"0", "1"} <= a else Verdict.SAFE
        result = repair(binary((3, 7)), oracle, propose_support=lambda x: [(0, 1)])
        self.assertEqual(result.status, "OPTIMAL_IN_POOL")
        self.assertEqual(result.assignment, (0, 1))
        self.assertEqual(result.cuts[0].kind, "assignment")

    def test_unknown_candidate_never_generates_cut(self):
        oracle = lambda a, tags: Verdict.UNKNOWN if a else Verdict.SAFE
        result = repair(binary((2, 4)), oracle)
        self.assertEqual(result.status, "INCUMBENT_WITH_GAP")
        self.assertEqual(result.assignment, (0, 0))
        self.assertEqual(result.cuts, ())
        self.assertEqual(result.upper_bound, 6)

    def test_baseline_unknown_has_no_fallback(self):
        result = repair(binary((2,)), lambda a, tags: Verdict.UNKNOWN)
        self.assertEqual(result.status, "UNRESOLVED")
        self.assertIsNone(result.assignment)
        self.assertIsNone(result.gap)

    def test_no_feasible_assignment_requires_exhaustion(self):
        result = repair(binary((2,)), lambda a, tags: Verdict.UNSAFE)
        self.assertEqual(result.status, "NO_FEASIBLE_IN_POOL")
        self.assertGreater(result.solves, 0)

    def test_budget_retains_trivial_upper_bound(self):
        result = repair(binary((2, 4)), lambda a, tags: Verdict.SAFE, max_solves=0, initial_assignment=(0, 0))
        self.assertEqual(result.status, "INCUMBENT_WITH_GAP")
        self.assertEqual(result.value, 0)
        self.assertEqual(result.upper_bound, 6)

    def test_no_check_budget_never_assumes_safe_baseline(self):
        result = repair(binary((2,)), lambda a, tags: Verdict.SAFE, max_checks=0)
        self.assertEqual(result.status, "UNRESOLVED")
        self.assertEqual(result.checks, 0)

    def test_frozen_baseline_relative_named_coherence(self):
        # Tiny named-subclass/disjointness fragment; not a general OWL reasoner.
        core = {("Old", "L"), ("Old", "R"), ("X", "L")}
        nodes = {"Old", "L", "R", "X"}
        def unsat(extra):
            reach = {n: {n} for n in nodes}
            for a, b in core | extra:
                reach[a].add(b)
            changed = True
            while changed:
                before = {n: frozenset(v) for n, v in reach.items()}
                for n in nodes:
                    reach[n] |= set().union(*(reach[p] for p in tuple(reach[n])))
                changed = any(before[n] != reach[n] for n in nodes)
            return {n for n in nodes if {"L", "R"} <= reach[n]}
        baseline = unsat(set())
        self.assertEqual(baseline, {"Old"})
        def oracle(a, tags):
            extra = {tuple(edge.split(">")) for edge in a}
            return Verdict.UNSAFE if unsat(extra) - baseline else Verdict.SAFE
        states = (base_states("X", "R", "forward"),)
        result = self.assert_exact(states, oracle, propose_support=lambda x: [(0, 1)])
        self.assertEqual(result.assignment, (0,))
        self.assertEqual(baseline, {"Old"})

    def test_conditional_policy_requires_activation(self):
        # Antecedent is already impossible in the core. Only selecting its active_expression is prohibited.
        states = ((State(frozenset(), 0), State(frozenset(), 10, frozenset({"active_expression"}))),)
        oracle = lambda a, tags: Verdict.UNSAFE if "active_expression" in tags else Verdict.SAFE
        result = self.assert_exact(states, oracle, propose_support=lambda x: [(0, 1)])
        self.assertEqual(result.assignment, (0,))
        self.assertEqual(result.cuts[0].literals, frozenset({(0, 1)}))

    def test_duplicate_axioms_do_not_create_fake_incompatibility(self):
        states = ((State(frozenset(), 0), State(frozenset({"A>B"}), 4)),
                  (State(frozenset(), 0), State(frozenset({"A>B"}), 7)))
        result = self.assert_exact(states, lambda a, tags: Verdict.SAFE)
        self.assertEqual(result.assignment, (1, 1))
        self.assertEqual(result.value, 11)  # explicit additive edit utility
        self.assertEqual(materialize(states, result.assignment)[0], frozenset({"A>B"}))

    def test_equal_axiom_states_with_different_utility_keep_best(self):
        states = ((State(frozenset(), 0), State(frozenset({"a"}), 2),
                   State(frozenset({"a"}), 8)),)
        result = self.assert_exact(states, lambda a, tags: Verdict.SAFE)
        self.assertEqual(result.value, 8)

    def test_directional_inventory_is_weakening_only(self):
        self.assertEqual(len(base_states("A", "B", "eq")), 4)
        self.assertEqual([s.axioms for s in base_states("A", "B", "forward")],
                         [frozenset(), frozenset({"A>B"})])
        self.assertEqual([s.axioms for s in base_states("A", "B", "backward")],
                         [frozenset(), frozenset({"B>A"})])

    def test_weighted_encoding_all_boolean_valuations(self):
        states = ((State(frozenset(), -9), State(frozenset({"a"}), 3)),
                  (State(frozenset(), 2**60), State(frozenset({"b"}), 2**60 + 1),
                   State(frozenset({"c"}), -(2**60))))
        variables, hard, soft, cap = weighted_encoding(states)
        def satisfies(clause, true):
            return any((lit > 0) == (abs(lit) in true) for lit in clause)
        hard_models = 0
        for bits in product((False, True), repeat=len(variables)):
            true = {i + 1 for i, on in enumerate(bits) if on}
            if not all(satisfies(c, true) for c in hard):
                continue
            hard_models += 1
            x = tuple(next(h for h in range(len(g)) if variables[i, h] in true)
                      for i, g in enumerate(states))
            cost = sum(w for c, w in soft if not satisfies(c, true))
            self.assertEqual(cost, cap - score(states, x))
        self.assertEqual(hard_models, 6)

    def test_governed_infeasibility_and_no_unusable_fallback(self):
        states = binary((5,))
        oracle = lambda a, tags: Verdict.UNSAFE if a else Verdict.SAFE
        result = self.assert_exact(states, oracle, governed=lambda x: x == (1,))
        self.assertEqual(result.status, "NO_FEASIBLE_IN_POOL")
        unknown = repair(states, oracle, governed=lambda x: x == (1,), max_solves=0)
        self.assertEqual(unknown.status, "UNRESOLVED")
        self.assertIsNone(unknown.assignment)

    def test_nonmonotone_policy_uses_only_complete_assignment_cuts(self):
        # A larger set can cure this synthetic violation; support lifting is not licensed.
        states = binary((5, -1))
        oracle = lambda a, tags: Verdict.UNSAFE if a == frozenset({"0"}) else Verdict.SAFE
        result = self.assert_exact(states, oracle, monotone=False,
                                   propose_support=lambda x: [(0, 1)])
        self.assertEqual(result.assignment, (1, 1))
        self.assertTrue(all(c.kind == "assignment" for c in result.cuts))

    def test_safety_only_replay_cannot_certify_optimality(self):
        states = binary((5, 3))
        oracle = lambda a, tags: Verdict.SAFE
        result = repair(states, oracle)
        forged = replace(result, assignment=(1, 0), value=5, upper_bound=5)
        self.assertTrue(replay_safety(states, oracle, forged))
        self.assertFalse(replay_optimality(states, oracle, forged))
        bad_cut = Cut(frozenset({(1, 1)}), "monotone_support")
        self.assertFalse(replay_optimality(states, oracle, replace(forged, cuts=(bad_cut,))))

    def test_random_multistate_instances(self):
        rng = random.Random(20260917)
        for _ in range(120):
            states = tuple(tuple([State(frozenset(), rng.randint(-4, 4))] +
                                 [State(frozenset({f"{i}:{h}"}), rng.randint(-8, 12))
                                  for h in range(1, rng.randint(2, 4))]) for i in range(3))
            conflicts = []
            for _ in range(rng.randrange(5)):
                objects = rng.sample(range(3), rng.randint(1, 3))
                conflicts.append(frozenset(f"{i}:{rng.randrange(1, len(states[i]))}" for i in objects))
            oracle = lambda a, tags: Verdict.UNSAFE if any(c <= a for c in conflicts) else Verdict.SAFE
            self.assert_exact(states, oracle)

    def test_empty_inventory_and_invalid_numbers(self):
        self.assert_exact((), lambda a, tags: Verdict.SAFE)
        for invalid in (True, 1.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                repair(((State(frozenset(), invalid),),), lambda a, tags: Verdict.SAFE)

class XR2RegressionTests(unittest.TestCase):
    assert_exact = KernelTests.assert_exact
    """Cases for the general finite-inventory contract."""

    def test_no_empty_alternative_is_required(self):
        states = ((State(frozenset({"a"}), 5), State(frozenset({"b"}), 2)),)
        oracle = lambda a, tags: Verdict.UNSAFE if "a" in a else Verdict.SAFE
        r = self.assert_exact(states, oracle, initial_assignment=(0,))
        self.assertEqual(r.assignment, (1,))

    def test_failed_initial_positive_query_does_not_stop_search(self):
        oracle = lambda a, tags: Verdict.SAFE if "0" in a else Verdict.UNSAFE
        r = self.assert_exact(binary((3,)), oracle, initial_assignment=(0,), monotone=False)
        self.assertEqual(r.assignment, (1,))

    def test_unknown_initial_assignment_does_not_stop_search(self):
        oracle = lambda a, tags: Verdict.SAFE if "0" in a else Verdict.UNKNOWN
        r = repair(binary((3,)), oracle, initial_assignment=(0,))
        self.assertEqual(r.status, "OPTIMAL_IN_POOL")
        self.assertEqual(r.pending, ((0,),))
        self.assertEqual(r.upper_bound, 3)

    def test_pending_high_value_survives_lower_master_bound(self):
        states = ((State(frozenset({"unknown"}), 10),
                   State(frozenset({"bad"}), 9),
                   State(frozenset({"ok"}), 7)),)
        def oracle(a, tags):
            return Verdict.UNKNOWN if "unknown" in a else (
                Verdict.UNSAFE if "bad" in a else Verdict.SAFE)
        r = repair(states, oracle)
        self.assertEqual((r.status, r.value, r.upper_bound), ("INCUMBENT_WITH_GAP", 7, 10))
        self.assertEqual(r.pending, ((0,),))
        self.assertEqual(len(r.cuts), 1)

    def test_support_probe_does_not_fill_missing_objects_with_keep(self):
        states = ((State(frozenset({"x"}), 9), State(frozenset({"y"}), 1)),
                  (State(frozenset({"clash"}), 8), State(frozenset({"ok"}), 2)))
        oracle = lambda a, tags: Verdict.UNSAFE if "clash" in a else Verdict.SAFE
        r = self.assert_exact(states, oracle, propose_support=lambda x: {(0, x[0])})
        self.assertTrue(all(c.kind == "assignment" for c in r.cuts))

    def test_empty_fixed_conflict_and_empty_inventory(self):
        r = repair((), lambda a, tags: Verdict.UNSAFE)
        self.assertEqual(r.status, "NO_FEASIBLE_IN_POOL")
        self.assertEqual(r.cuts[0].literals, frozenset())

    def test_signed_pair_factors_change_the_optimum(self):
        states = binary((9, 9))
        pairs = {(0, 1, 1, 1): -10}
        r = self.assert_exact(states, lambda a, tags: Verdict.SAFE, interactions=pairs)
        self.assertEqual(r.value, 9)
        self.assertNotEqual(r.assignment, (1, 1))

    def test_pair_weighted_encoding_all_boolean_valuations(self):
        states = binary((-3, 4))
        pairs = {(0, 1, 1, 1): -8, (0, 0, 1, 1): 6, (0, 1, 1, 0): 2}
        variables, hard, soft, cap = weighted_encoding(states, pairs)
        satisfies = lambda clause, true: any((v > 0) == (abs(v) in true) for v in clause)
        models = 0
        for bits in product((False, True), repeat=len(variables)):
            true = {i + 1 for i, on in enumerate(bits) if on}
            if not all(satisfies(c, true) for c in hard):
                continue
            x = tuple(next(h for h in range(len(g)) if variables[i, h] in true)
                      for i, g in enumerate(states))
            self.assertEqual(cap - sum(w for c, w in soft if not satisfies(c, true)),
                             score(states, x, pairs))
            models += 1
        self.assertEqual(models, 4)

    def test_presence_uses_mapping_and_ontology_emitters(self):
        states = ((State(frozenset(), 0), State(frozenset({"alpha"}), 3)),
                  (State(frozenset({"alpha"}), 2), State(frozenset(), 1)),
                  (State(frozenset(), 0), State(frozenset({"beta"}), 4)))
        self.assertEqual(axiom_origins(states, "alpha"), frozenset({(0, 1), (1, 0)}))
        cut = PresenceCut(frozenset({"alpha", "beta"}))
        self.assertTrue(rejects_presence(states, (0, 0, 1), cut))
        self.assertTrue(rejects_presence(states, (1, 1, 1), cut))
        self.assertFalse(rejects_presence(states, (0, 1, 1), cut))
        self.assertTrue(rejects_presence(states, (0, 1, 1), cut, frozenset({"alpha"})))
        conditional = PresenceCut(cut.support, frozenset({(0, 1)}))
        self.assertFalse(rejects_presence(states, (0, 0, 1), conditional))
        self.assertTrue(rejects_presence(states, (1, 1, 1), conditional))

    def test_random_arbitrary_policies_and_pair_factors(self):
        rng = random.Random(20260919)
        for _ in range(120):
            states = tuple(tuple(State(frozenset({f"{i}:{h}"}), rng.randint(-7, 7))
                                 for h in range(3)) for i in range(3))
            all_x = list(assignments(states))
            safe_axioms = {materialize(states, x)[0] for x in all_x if rng.random() < .3}
            oracle = lambda a, tags: Verdict.SAFE if a in safe_axioms else Verdict.UNSAFE
            pairs = {(0, rng.randrange(3), 2, rng.randrange(3)): rng.randint(-10, 10)}
            self.assert_exact(states, oracle, interactions=pairs, monotone=False)

    def test_invalid_pair_indices_and_values(self):
        for pairs in ({(0, 1, 0, 1): 2}, {(1, 1, 0, 1): 2},
                      {(0, 4, 1, 1): 2}, {(0, 1, 1, 1): True}):
            with self.assertRaises(ValueError):
                weighted_encoding(binary((1, 2)), pairs)


if __name__ == "__main__":
    unittest.main()
