"""Finite XR-2 truth model; no production OWL, SAT, circuit or wall-time implementation."""

from dataclasses import dataclass
from enum import Enum
from itertools import combinations, product
from typing import Callable


class Verdict(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class State:
    axioms: frozenset[str]
    utility: int
    activation: frozenset[str] = frozenset()


Inventory = tuple[tuple[State, ...], ...]
Assignment = tuple[int, ...]
Literal = tuple[int, int]
Oracle = Callable[[frozenset[str], frozenset[str]], Verdict]


@dataclass(frozen=True)
class Cut:
    literals: frozenset[Literal]
    kind: str  # assignment or monotone_support; empty cut excludes every assignment


@dataclass(frozen=True)
class PresenceCut:
    """Exclude joint axiom presence only when all activation literals hold."""
    support: frozenset[str]
    activation: frozenset[Literal] = frozenset()


@dataclass(frozen=True)
class Result:
    status: str
    assignment: Assignment | None
    value: int | None
    upper_bound: int | None
    cuts: tuple[Cut, ...]
    pending: tuple[Assignment, ...]
    checks: int
    solves: int
    reason: str

    @property
    def gap(self):
        return None if self.value is None else self.upper_bound - self.value


def literals(x):
    return frozenset(enumerate(x))


def validate_inventory(states, interactions):
    for group in states:
        if not group:
            raise ValueError("each revision object needs at least one alternative")
        if any(type(s.utility) is not int for s in group):
            raise ValueError("utilities must be integers, not booleans or floats")
    for key, value in interactions.items():
        if (type(key) is not tuple or len(key) != 4
                or any(type(k) is not int for k in key)):
            raise ValueError("pair key must be four integer indices")
        i, a, j, b = key
        if not (0 <= i < j < len(states) and 0 <= a < len(states[i])
                and 0 <= b < len(states[j]) and type(value) is int):
            raise ValueError("pair factors require ordered distinct valid objects and integer values")


def score(states, x, interactions=None):
    return sum(states[i][h].utility for i, h in enumerate(x)) + sum(
        value for (i, a, j, b), value in (interactions or {}).items()
        if x[i] == a and x[j] == b)


def union_states(selected):
    chosen = list(selected)
    return (frozenset().union(*(s.axioms for s in chosen)),
            frozenset().union(*(s.activation for s in chosen)))


def materialize(states, x):
    return union_states(states[i][h] for i, h in enumerate(x))


def assignments(states):
    return product(*(range(len(group)) for group in states))


def base_states(source, target, relation):
    """Compilation fixture only; production uses shared-core OWL values."""
    f, b = f"{source}>{target}", f"{target}>{source}"
    alternatives = {"eq": ((), (f,), (b,), (f, b)),
                    "forward": ((), (f,)), "backward": ((), (b,))}
    return tuple(State(frozenset(a), len(a)) for a in alternatives[relation])


def axiom_origins(states, axiom):
    """All candidate emitters, regardless of mapping/ontology provenance."""
    return frozenset((i, h) for i, group in enumerate(states)
                     for h, state in enumerate(group) if axiom in state.axioms)


def rejects_presence(states, x, cut, fixed=frozenset()):
    present = lambda axiom: axiom in fixed or bool(axiom_origins(states, axiom) & literals(x))
    return cut.activation <= literals(x) and all(present(a) for a in cut.support)


def weighted_encoding(states, interactions=None):
    """WCNF-shaped clauses; cap minus violated soft weight equals the objective."""
    interactions = interactions or {}
    validate_inventory(states, interactions)
    variables, hard, soft, cap = {}, [], [], 0
    for i, group in enumerate(states):
        row = []
        maximum = max(s.utility for s in group)
        cap += maximum
        for h, state in enumerate(group):
            v = len(variables) + 1
            variables[i, h] = v
            row.append(v)
            if maximum > state.utility:
                soft.append(((-v,), maximum - state.utility))
        hard.append(tuple(row))
        hard.extend((-a, -b) for a, b in combinations(row, 2))
    for key, value in sorted(interactions.items()):
        i, a, j, b = key
        y = len(variables) + 1
        variables[key] = y
        left, right = variables[i, a], variables[j, b]
        hard.extend(((-y, left), (-y, right), (y, -left, -right)))
        if value > 0:
            soft.append(((y,), value))
            cap += value
        elif value < 0:
            soft.append(((-y,), -value))
    return variables, hard, soft, cap


def repair(states: Inventory, oracle: Oracle, *, governed=lambda x: True,
           max_solves=None, max_checks=None, propose_support=None, monotone=True,
           active_progress=False, initial_assignment=None, interactions=None):
    """Exhaustive master. UNKNOWN is pending, not a cut; counts are finite-call budgets.

    Support lifting requires monotone UNSAFE in both asserted axioms and activated
    obligations. The caller must disable it for policies such as positive entailment.
    The oracle includes the fixed background; supplied axioms are revised assertions.
    """
    interactions = interactions or {}
    validate_inventory(states, interactions)
    for budget in (max_solves, max_checks):
        if budget is not None and (type(budget) is not int or budget < 0):
            raise ValueError("budgets must be nonnegative integer call counts")
    if active_progress and not monotone:
        raise ValueError("active-state cuts require a monotone policy")
    utility = lambda x: score(states, x, interactions)
    cap = sum(max(s.utility for s in group) for group in states)
    cap += sum(max(0, v) for v in interactions.values())
    upper, incumbent, cuts, pending, checks, solves = cap, None, [], {}, 0, 0

    def check(axioms, activation):
        nonlocal checks
        if max_checks is not None and checks >= max_checks:
            return Verdict.UNKNOWN
        checks += 1
        verdict = oracle(axioms, activation)
        if not isinstance(verdict, Verdict):
            raise ValueError("invalid oracle result")
        return verdict

    def finish(reason, forced=None):
        value = None if incumbent is None else utility(incumbent)
        status = forced or ("UNRESOLVED" if incumbent is None else
                            "OPTIMAL_IN_POOL" if upper == value else "INCUMBENT_WITH_GAP")
        return Result(status, incumbent, value, upper, tuple(cuts), tuple(pending),
                      checks, solves, reason)

    if initial_assignment is not None:
        initial_assignment = tuple(initial_assignment)
        if (len(initial_assignment) != len(states) or any(
                type(h) is not int or not 0 <= h < len(states[i])
                for i, h in enumerate(initial_assignment))):
            raise ValueError("invalid initial assignment")
        if governed(initial_assignment):
            v = check(*materialize(states, initial_assignment))
            if v is Verdict.SAFE:
                incumbent = initial_assignment
            elif v is Verdict.UNKNOWN:
                pending[initial_assignment] = utility(initial_assignment)
            else:
                cuts.append(Cut(literals(initial_assignment), "assignment"))

    while True:
        if max_solves is not None and solves >= max_solves:
            return finish("master budget")
        remaining = (x for x in assignments(states) if governed(x) and x not in pending
                     and all(not c.literals <= literals(x) for c in cuts))
        x = max(remaining, key=utility, default=None)
        solves += 1
        bounds = list(pending.values())
        if x is not None:
            bounds.append(utility(x))
        if incumbent is not None:
            bounds.append(utility(incumbent))
        if not bounds:
            upper = None
            return finish("logical exhaustion", "NO_FEASIBLE_IN_POOL")
        upper = min(upper, max(bounds))
        if incumbent is not None:
            assert upper >= utility(incumbent)
            if upper == utility(incumbent):
                return finish("incumbent meets bound including pending alternatives")
            if x is None or utility(x) <= utility(incumbent):
                return finish("higher-valued pending alternatives")
        if x is None:
            return finish("only pending alternatives remain")
        if max_checks is not None and checks >= max_checks:
            return finish("verification budget")
        verdict = check(*materialize(states, x))
        if verdict is Verdict.SAFE:
            incumbent = x
            return finish("best unblocked assignment verified")
        if verdict is Verdict.UNKNOWN:
            pending[x] = utility(x)
            continue

        cut = Cut(literals(x), "assignment")
        if active_progress:
            active = frozenset((i, h) for i, h in enumerate(x)
                               if states[i][h].axioms or states[i][h].activation)
            cut = Cut(active, "monotone_support")
        if monotone and propose_support is not None:
            proposed = frozenset(propose_support(x))
            if proposed <= literals(x):
                probe = union_states(states[i][h] for i, h in proposed)
                if check(*probe) is Verdict.UNSAFE:
                    cut = Cut(proposed, "monotone_support")
        assert cut.literals <= literals(x) and cut not in cuts
        if incumbent is not None:
            assert not cut.literals <= literals(incumbent)
        cuts.append(cut)
        pending = {p: u for p, u in pending.items() if not cut.literals <= literals(p)}


def replay_safety(states, oracle, result, governed=lambda x: True):
    return (result.assignment is not None and governed(result.assignment)
            and oracle(*materialize(states, result.assignment)) is Verdict.SAFE)


def replay_optimality(states, oracle, result, governed=lambda x: True, interactions=None):
    """Exhaustive cut/bound replay. A safety check alone does not verify optimality."""
    if not replay_safety(states, oracle, result, governed):
        return False
    all_x = [x for x in assignments(states) if governed(x)]
    for cut in result.cuts:
        excluded = [x for x in all_x if cut.literals <= literals(x)]
        if not excluded or any(oracle(*materialize(states, x)) is not Verdict.UNSAFE
                               for x in excluded):
            return False
    remaining = [x for x in all_x if all(not c.literals <= literals(x) for c in result.cuts)]
    utility = lambda x: score(states, x, interactions)
    value = utility(result.assignment)
    return (result.status == "OPTIMAL_IN_POOL" and result.value == value == result.upper_bound
            and max(map(utility, remaining), default=None) == value)
