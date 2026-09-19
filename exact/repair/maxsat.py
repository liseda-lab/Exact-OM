"""Exact signed integer finite-pool optimisation through public PySAT RC2 APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .records import ObjectiveV2, RepairInputV2


@dataclass(frozen=True)
class PresenceCut:
    """Externally proved monotone support, conditional on all listed activations."""

    support: tuple[Any, ...]
    activation: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class MasterResult:
    """Only a normally completed RC2 solve provides an optimum or emptiness proof."""

    assignment: tuple[int, ...] | None
    upper_bound: int | None
    status: str
    solver: str


def solve_master(
    objective: ObjectiveV2,
    exclusions: tuple[tuple[int, ...], ...] = (),
    *,
    problem: RepairInputV2 | None = None,
    presence_cuts: tuple[PresenceCut, ...] = (),
) -> MasterResult:
    """Solve one frozen master; supervisor interruption returns no bound.

    Sequential-counter cardinalities avoid quadratic one-hot clauses. Logical
    no-goods and temporary scheduling blocks have the same SAT shape but remain
    separate ledgers in the kernel and are never confused during replay.
    """
    try:
        import pysat
        from pysat.card import CardEnc, EncType
        from pysat.examples.rc2 import RC2
        from pysat.formula import WCNF, IDPool
    except ImportError as error:
        raise RuntimeError("repair requires the optional 'repair' dependencies") from error

    pool = IDPool()
    formula = WCNF()
    variables: dict[tuple[int, int], int] = {}
    for i, row in enumerate(objective.unary):
        lits = [pool.id((i, a)) for a in range(len(row))]
        variables.update({(i, a): lit for a, lit in enumerate(lits)})
        formula.extend(
            CardEnc.equals(lits, bound=1, vpool=pool, encoding=EncType.seqcounter).clauses
        )
        maximum = max(row)
        for lit, value in zip(lits, row):
            if maximum > value:
                formula.append([-lit], weight=maximum - value)
    for i, a, j, b, value in objective.pairs:
        if value == 0:
            continue
        y = pool.id(("pair", i, a, j, b))
        left, right = variables[i, a], variables[j, b]
        formula.extend([[-y, left], [-y, right], [y, -left, -right]])
        formula.append([y if value > 0 else -y], weight=abs(value))
    for assignment in exclusions:
        objective.score(assignment)  # validate before producing a clause
        formula.append([-variables[i, a] for i, a in enumerate(assignment)])
    if presence_cuts:
        if problem is None:
            raise ValueError("presence cuts require the exact asserted problem")
        if len(problem.objects) != len(objective.unary) or any(
            len(obj.candidates) != len(row) for obj, row in zip(problem.objects, objective.unary)
        ):
            raise ValueError("presence cuts require the same frozen objective inventory")
        if any((i, a) not in variables for cut in presence_cuts for i, a in cut.activation):
            raise ValueError("presence cut activation is outside the frozen inventory")
        fixed = frozenset(problem.fixed_axioms)
        presence: dict[Any, int] = {}
        for cut in presence_cuts:
            clause = [-variables[i, a] for i, a in cut.activation]
            for axiom in cut.support:
                if axiom in fixed:
                    continue
                if axiom not in presence:
                    p = pool.id(("presence", len(presence)))
                    presence[axiom] = p
                    emitters = [
                        variables[i, a]
                        for i, obj in enumerate(problem.objects)
                        for a, c in enumerate(obj.candidates)
                        if axiom in c.axioms
                    ]
                    formula.append([-p, *emitters])
                    formula.extend([[-v, p] for v in emitters])
                clause.append(-presence[axiom])
            formula.append(clause)
    with RC2(formula, solver="g3") as solver:
        model = solver.compute()
        if model is None:
            return MasterResult(None, None, "empty", f"python-sat/{pysat.__version__}:RC2:g3")
        positive = {v for v in model if v > 0}
        assignment = tuple(
            next(a for a in range(len(row)) if variables[i, a] in positive)
            for i, row in enumerate(objective.unary)
        )
        value = objective.score(assignment)
        # The encoding's constant offset is an independent result sanity check.
        if value != objective.upper_cap - solver.cost:
            raise RuntimeError("RC2 objective does not match the frozen integer objective")
        return MasterResult(assignment, value, "optimal", f"python-sat/{pysat.__version__}:RC2:g3")
