"""Existing SDD compiler and differentiable, conditioned Bernoulli mixtures.

The same differentiable evaluator supports direct typed-slot grammars and the
bounded-enumeration reference arm. Compilation costs are explicit; no claim is
made that arbitrary finite grammars always admit small circuits. PySDD and torch
are imported only when compilation or evaluation is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from math import isfinite
from time import perf_counter
from typing import Any, Iterable, Sequence, cast

from .candidates import deduplicate_candidates, expression_size, expression_tree
from .records import ReplacementCandidateV2, canonical_hash


class EmptyProposalSpace(ValueError):
    """The constrained event has zero probability; no distribution exists."""


@dataclass(frozen=True)
class ProposalEncoding:
    """One designated Boolean encoding per complete canonical replacement."""

    fields: tuple[tuple[str, tuple[str, ...]], ...]
    assignments: tuple[tuple[bool, ...], ...]
    candidate_ids: tuple[str, ...]
    constraint_identity: str = ""

    def __post_init__(self) -> None:
        width = sum(len(values) for _, values in self.fields)
        if width < 1 or len({name for name, _ in self.fields}) != len(self.fields):
            raise ValueError("encoding requires distinct nonempty categorical fields")
        if any(not values or len(set(values)) != len(values) for _, values in self.fields):
            raise ValueError("field values must be nonempty and distinct")
        if len(self.assignments) != len(self.candidate_ids):
            raise ValueError("each encoding must identify its candidate")
        if len(set(self.assignments)) != len(self.assignments) or len(
            set(self.candidate_ids)
        ) != len(self.candidate_ids):
            raise ValueError("one designated encoding is required per candidate")
        for assignment in self.assignments:
            if len(assignment) != width or any(type(v) is not bool for v in assignment):
                raise ValueError(
                    "encoding assignments must be Boolean vectors of the declared width"
                )
            offset = 0
            for _, values in self.fields:
                if sum(assignment[offset : offset + len(values)]) != 1:
                    raise ValueError("categorical fields must be one-hot")
                offset += len(values)

    @property
    def variable_count(self) -> int:
        """Return the Boolean variable count, including explicit unused markers."""
        return sum(len(values) for _, values in self.fields)

    @property
    def content_hash(self) -> str:
        """Include every language/menu/dependency constraint in the cache identity."""
        return cast(str, canonical_hash(self))


def encode_candidates(
    candidates: Iterable[ReplacementCandidateV2],
    *,
    max_depth: int = 2,
    max_constructors: int = 2,
    constraint_identity: str = "",
) -> ProposalEncoding:
    """Encode the full finite menu, including deterministic elementary alternatives.

    The bundle field records endpoint and template roles that cannot be inferred
    from an unordered set of expression roots. Slot fields explicitly encode
    constructor/class/property choices, with one unique inactive value. This is
    a bounded-enumeration arm, not an unenumerated grammar-compiler claim.
    """
    import pyowl_core as owl

    if any(type(v) is not int or v < 0 for v in (max_depth, max_constructors)):
        raise ValueError("grammar bounds must be nonnegative integers")
    if max_depth > 8:
        raise ValueError("binary slot encoding is limited to depth eight")
    pool = deduplicate_candidates(candidates)
    if len({candidate.object_id for candidate in pool}) > 1:
        raise ValueError("a proposal circuit applies to one revision object")
    elementary = {"keep", "delete", "retain_subsumption", "replace_endpoint"}
    roots: list[tuple[tuple, ...]] = []
    templates = []
    directions = []
    classes: set[str] = set()
    properties: set[str] = set()

    def vocabulary(tree: tuple) -> None:
        if tree[0] == "class":
            classes.add(tree[1])
        elif tree[0] == "exists":
            properties.add(tree[1])
            vocabulary(tree[2])
        else:
            vocabulary(tree[1])
            vocabulary(tree[2])

    for candidate in pool:
        template = next(
            (tag for tag in sorted(elementary) if tag in candidate.action_tags),
            candidate.action_tags[0],
        )
        templates.append(template)
        retained = []
        expressions = set(candidate.active_expressions)
        for axiom in candidate.axioms:
            if isinstance(axiom, owl.SubClassOf):
                if isinstance(axiom.sub_class, owl.Class) and isinstance(
                    axiom.super_class, owl.Class
                ):
                    retained.append(owl.structural_hexdigest(axiom))
                if template not in elementary:
                    expressions.update((axiom.sub_class, axiom.super_class))
            elif template not in elementary and isinstance(axiom, owl.ObjectPropertyDomain):
                expressions.add(axiom.domain)
            elif template not in elementary and isinstance(axiom, owl.ObjectPropertyRange):
                expressions.add(axiom.range)
        directions.append(tuple(sorted(retained)))
        if template in elementary:
            expressions.clear()
        trees = []
        for expression in sorted(expressions, key=owl.canonical_bytes):
            depth, count = expression_size(expression)
            if depth > max_depth or count > max_constructors:
                raise ValueError("candidate expression exceeds canonical grammar bounds")
            tree = expression_tree(expression)
            trees.append(tree)
            vocabulary(tree)
        roots.append(tuple(trees))
    max_roots = max(map(len, roots), default=0)
    slots_per_root = 2 ** (max_depth + 1) - 1
    retained_directions = sorted({d for row in directions for d in row})
    fields: list[tuple[str, tuple[str, ...]]] = [
        ("bundle", tuple(c.candidate_id for c in pool) or ("unused",)),
        ("template", tuple(sorted(set(templates))) or ("unused",)),
    ]
    fields.extend((f"retained:{direction}", ("false", "true")) for direction in retained_directions)
    for root in range(max_roots):
        for slot in range(slots_per_root):
            prefix = f"expression:{root}:{slot}"
            fields.extend(
                [
                    (f"{prefix}:constructor", ("unused", "class", "and", "exists")),
                    (f"{prefix}:class", ("unused", *sorted(classes))),
                    (f"{prefix}:property", ("unused", *sorted(properties))),
                ]
            )
    assignments = []
    for candidate, template, candidate_trees, candidate_directions in zip(
        pool, templates, roots, directions
    ):
        choices = {name: "unused" for name, _ in fields}
        choices.update(bundle=candidate.candidate_id, template=template)
        for direction in retained_directions:
            choices[f"retained:{direction}"] = (
                "true" if direction in candidate_directions else "false"
            )

        def fill(root: int, slot: int, tree: tuple) -> None:
            prefix = f"expression:{root}:{slot}"
            choices[f"{prefix}:constructor"] = tree[0]
            if tree[0] == "class":
                choices[f"{prefix}:class"] = tree[1]
            elif tree[0] == "exists":
                choices[f"{prefix}:property"] = tree[1]
                fill(root, 2 * slot + 1, tree[2])
            else:
                fill(root, 2 * slot + 1, tree[1])
                fill(root, 2 * slot + 2, tree[2])

        for root, tree in enumerate(candidate_trees):
            fill(root, 0, tree)
        assignments.append(
            tuple(choices[name] == value for name, values in fields for value in values)
        )
    identity = canonical_hash((constraint_identity, max_depth, max_constructors))
    return ProposalEncoding(
        tuple(fields), tuple(assignments), tuple(c.candidate_id for c in pool), identity
    )


@dataclass(frozen=True)
class CompiledCircuit:
    """Compiled root plus native manager or immutable transported compiler artifact."""

    encoding: Any
    manager: Any
    root: Any
    cache_key: str
    compiler_version: str
    compilation_seconds: float
    node_count: int
    telemetry: tuple[tuple[str, Any], ...] = ()


def compile_encoding(
    encoding: ProposalEncoding,
    *,
    variable_order: Sequence[int] | None = None,
    max_nodes: int | None = None,
    max_seconds: float | None = None,
    vtree_type: str = "balanced",
    collect: bool = True,
    max_live_nodes: int | None = None,
    max_reachable_nodes: int | None = None,
    max_elements: int | None = None,
) -> CompiledCircuit:
    """Compile finite assignments with explicit native ownership and resource limits."""
    order = (
        tuple(range(1, encoding.variable_count + 1))
        if variable_order is None
        else tuple(variable_order)
    )
    if sorted(order) != list(range(1, encoding.variable_count + 1)):
        raise ValueError("variable order must be a permutation of all Boolean variables")
    if vtree_type not in {"right", "balanced"}:
        raise ValueError("finite ProposalEncoding vtree_type must be right or balanced")
    if type(collect) is not bool:
        raise ValueError("compiler collect must be Boolean")
    if any(
        value is not None and (type(value) is not int or value < 1)
        for value in (max_nodes, max_live_nodes, max_reachable_nodes, max_elements)
    ):
        raise ValueError("circuit structural limits must be positive integers")
    if max_seconds is not None and (not isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("compiler wall limit must be finite and positive")
    try:
        compiler_version = version("pysdd")
    except ModuleNotFoundError as exc:
        raise ImportError("repair proposals require the optional pysdd dependency") from exc
    return _compile(
        encoding,
        order,
        compiler_version,
        max_nodes,
        max_seconds,
        vtree_type,
        collect,
        max_live_nodes,
        max_reachable_nodes,
        max_elements,
    )


@lru_cache(maxsize=32)
def _compile(
    encoding: ProposalEncoding,
    order: tuple[int, ...],
    compiler_version: str,
    max_nodes: int | None,
    max_seconds: float | None,
    vtree_type: str,
    collect: bool,
    max_live_nodes: int | None,
    max_reachable_nodes: int | None,
    max_elements: int | None,
) -> CompiledCircuit:
    try:
        from pysdd.sdd import SddManager, Vtree
    except ImportError as exc:
        raise ImportError("repair proposals require the optional pysdd dependency") from exc
    from .grammar import CircuitBudgetExceeded

    started = perf_counter()
    manager = SddManager(
        vtree=Vtree(
            var_count=encoding.variable_count, var_order=list(order), vtree_type=vtree_type
        ),
        auto_gc_and_minimize=False,
    )
    checks = peak_allocated = 0

    def check() -> None:
        nonlocal checks, peak_allocated
        checks += 1
        peak_allocated = max(peak_allocated, manager.count())
        if collect and (
            checks % 128 == 0 or (max_nodes is not None and manager.count() > max_nodes)
        ):
            manager.garbage_collect()
        if max_nodes is not None and manager.count() > max_nodes:
            raise CircuitBudgetExceeded("finite compilation exceeds its allocated node limit")
        if max_live_nodes is not None and manager.live_count() > max_live_nodes:
            raise CircuitBudgetExceeded("finite compilation exceeds its live node limit")
        if max_elements is not None and manager.live_size() > max_elements:
            raise CircuitBudgetExceeded("finite compilation exceeds its live element limit")
        if max_seconds is not None and perf_counter() - started > max_seconds:
            raise CircuitBudgetExceeded("finite compilation exceeds its wall time limit")

    root = manager.false()
    root.ref()
    check()
    for assignment in encoding.assignments:
        term = manager.true()
        term.ref()
        for variable, value in enumerate(assignment, 1):
            replacement = term & manager.literal(variable if value else -variable)
            replacement.ref()
            term.deref()
            term = replacement
            check()
        replacement = root | term
        replacement.ref()
        root.deref()
        term.deref()
        root = replacement
        check()
    if collect:
        manager.garbage_collect()
    check()
    if max_reachable_nodes is not None and root.count() > max_reachable_nodes:
        raise CircuitBudgetExceeded("finite compilation exceeds its reachable node limit")
    if max_elements is not None and root.size() > max_elements:
        raise CircuitBudgetExceeded("finite compilation exceeds its reachable element limit")
    key = canonical_hash(
        (
            encoding.content_hash,
            order,
            "pysdd",
            compiler_version,
            "finite-resource-contract/review-2",
            vtree_type,
            collect,
        )
    )
    return CompiledCircuit(
        encoding,
        manager,
        root,
        key,
        compiler_version,
        perf_counter() - started,
        root.count(),
        (
            ("vtree", vtree_type),
            ("collect", collect),
            ("peak_allocated", peak_allocated),
            ("resource_contract", "compiler-resource-contract/review-2"),
        ),
    )


@dataclass(frozen=True)
class ProposalSample:
    """A reproducible constrained draw, including its mixture component."""

    candidate_id: str
    assignment: tuple[bool, ...]
    component: int
    log_probability: float


class ConditionedMixture:
    """Stable differentiable WMC and exact ancestral sampling on a compiled SDD."""

    circuit: CompiledCircuit | FactoredCircuit

    def __init__(self, circuit: CompiledCircuit, literal_logits: Any, component_logits: Any = None):
        import torch
        import torch.nn.functional as functional

        if not isinstance(literal_logits, torch.Tensor) or not literal_logits.is_floating_point():
            raise TypeError("literal_logits must be a floating-point tensor")
        if literal_logits.ndim != 2 or literal_logits.shape[1] != circuit.encoding.variable_count:
            raise ValueError("literal logits must have shape (components, Boolean variables)")
        count = literal_logits.shape[0]
        if count < 1 or torch.isnan(literal_logits).any():
            raise ValueError("mixture requires non-NaN logits and at least one component")
        if component_logits is None:
            component_logits = literal_logits.new_zeros(count)
        if (
            not isinstance(component_logits, torch.Tensor)
            or component_logits.shape != (count,)
            or component_logits.device != literal_logits.device
            or not torch.isfinite(component_logits).all()
        ):
            raise ValueError("component logits must be a finite vector on the literal-logit device")
        self.circuit = circuit
        self.literal_logits = literal_logits
        self.log_positive = functional.logsigmoid(literal_logits)
        self.log_negative = functional.logsigmoid(-literal_logits)
        self.log_mixture = torch.log_softmax(component_logits, dim=0)
        self._values: dict[int, Any] = {}
        self.component_log_normalizers = self._evaluate(circuit.root)
        self.log_normalizer = torch.logsumexp(
            self.log_mixture + self.component_log_normalizers, dim=0
        )
        if torch.isneginf(self.log_normalizer):
            raise EmptyProposalSpace("the constrained proposal space has zero probability")
        if not torch.isfinite(self.log_normalizer):
            raise FloatingPointError("nonfinite constrained proposal normalizer")
        self.component_posterior = torch.softmax(
            self.log_mixture + self.component_log_normalizers, dim=0
        )
        self._candidate_by_assignment = dict(
            zip(
                getattr(circuit.encoding, "assignments", ()),
                getattr(circuit.encoding, "candidate_ids", ()),
            )
        )
        self._grammar = hasattr(circuit.encoding, "decode")

    def accepts(self, assignment: Sequence[bool]) -> bool:
        """Evaluate a complete assignment without allocating new compiler nodes."""
        key = tuple(assignment)
        if len(key) != self.circuit.encoding.variable_count or any(
            type(v) is not bool for v in key
        ):
            return False
        if not self._grammar:
            return key in self._candidate_by_assignment
        assert isinstance(self.circuit, CompiledCircuit)
        values: dict[int, bool] = {}
        pending = [(self.circuit.root, False)]
        while pending:
            node, expanded = pending.pop()
            if node.id in values:
                continue
            if node.is_false():
                values[node.id] = False
            elif node.is_true():
                values[node.id] = True
            elif node.is_literal():
                values[node.id] = key[abs(node.literal) - 1] == (node.literal > 0)
            elif expanded:
                values[node.id] = any(values[p.id] and values[s.id] for p, s in node.elements())
            else:
                pending.append((node, True))
                pending.extend(
                    (child, False)
                    for pair in node.elements()
                    for child in pair
                    if child.id not in values
                )
        return values[self.circuit.root.id]

    def candidate(self, assignment: Sequence[bool]) -> ReplacementCandidateV2:
        """Materialize one directly generated replacement after validating its slots."""
        if not self._grammar:
            raise TypeError("bounded-enumeration circuits do not own candidate values")
        if not self.accepts(assignment):
            raise ValueError("assignment is outside the constrained grammar")
        return cast(ReplacementCandidateV2, self.circuit.encoding.decode(assignment))

    def candidate_log_probability(self, candidate: ReplacementCandidateV2) -> Any:
        """Sum probability of every encoding of an identical bundle and activation."""
        import torch

        if self._grammar:
            assignments = self.circuit.encoding.candidate_assignments(candidate)
        else:
            assignments = tuple(
                a
                for a, identifier in self._candidate_by_assignment.items()
                if identifier == candidate.candidate_id
            )
        terms = [self.log_probability(a) for a in assignments if self.accepts(a)]
        if not terms:
            return self.literal_logits.new_tensor(-float("inf"))
        return torch.logsumexp(torch.stack(terms), dim=0)

    def _evaluate(self, node: Any) -> Any:
        import torch

        pending = [(node, False)]
        while pending:
            current, expanded = pending.pop()
            if current.id in self._values:
                continue
            if current.is_false():
                value = self.literal_logits.new_full((self.literal_logits.shape[0],), -float("inf"))
            elif current.is_true():
                # Missing scopes are smoothed implicitly: p(z)+p(not z)=1.
                value = self.literal_logits.new_zeros(self.literal_logits.shape[0])
            elif current.is_literal():
                literal = current.literal
                value = (self.log_positive if literal > 0 else self.log_negative)[
                    :, abs(literal) - 1
                ]
            elif expanded:
                terms = [self._values[p.id] + self._values[s.id] for p, s in current.elements()]
                value = torch.logsumexp(torch.stack(terms), dim=0)
            else:
                pending.append((current, True))
                pending.extend(
                    (child, False)
                    for pair in current.elements()
                    for child in pair
                    if child.id not in self._values
                )
                continue
            self._values[current.id] = value
        return self._values[node.id]

    def log_probability(self, assignment: Sequence[bool]) -> Any:
        """Compute exact mixture likelihood including the differentiable log normalizer."""
        import torch

        key = tuple(assignment)
        if len(key) != self.circuit.encoding.variable_count or any(
            type(v) is not bool for v in key
        ):
            raise ValueError("proposal assignment must be a Boolean vector of the circuit width")
        if not self.accepts(key):
            return self.literal_logits.new_tensor(-float("inf"))
        selected = torch.tensor(key, device=self.literal_logits.device, dtype=torch.bool)
        terms = torch.where(selected, self.log_positive, self.log_negative).sum(dim=1)
        return torch.logsumexp(self.log_mixture + terms, dim=0) - self.log_normalizer

    def sample(self, count: int = 1, *, seed: int = 0) -> tuple[ProposalSample, ...]:
        """Draw posterior components then weighted SDD branches with a local RNG."""
        import torch

        if type(count) is not int or count < 0:
            raise ValueError("sample count must be a nonnegative integer")
        generator = torch.Generator(device=self.literal_logits.device).manual_seed(seed)
        assert isinstance(self.circuit, CompiledCircuit)
        samples = []
        candidate_masses: dict[str, float] = {}
        with torch.no_grad():
            for _ in range(count):
                component = int(
                    torch.multinomial(self.component_posterior, 1, generator=generator).item()
                )
                assignment: dict[int, bool] = {}

                pending = [self.circuit.root]
                while pending:
                    node = pending.pop()
                    if node.is_false():
                        raise RuntimeError("a zero-probability SDD branch was selected")
                    if node.is_literal():
                        assignment[abs(node.literal) - 1] = node.literal > 0
                    elif not node.is_true():
                        elements = node.elements()
                        weights = torch.stack(
                            [
                                self._values[p.id][component] + self._values[s.id][component]
                                for p, s in elements
                            ]
                        )
                        selected = int(
                            torch.multinomial(
                                torch.softmax(weights, dim=0), 1, generator=generator
                            ).item()
                        )
                        pending.extend(reversed(elements[selected]))

                for variable in range(self.circuit.encoding.variable_count):
                    if variable not in assignment:
                        probability = torch.sigmoid(self.literal_logits[component, variable])
                        assignment[variable] = bool(
                            torch.rand((), device=probability.device, generator=generator)
                            < probability
                        )
                key = tuple(assignment[i] for i in range(self.circuit.encoding.variable_count))
                if not self.accepts(key):
                    raise RuntimeError(
                        "compiled circuit sampled an assignment outside its language"
                    )
                candidate = self.candidate(key) if self._grammar else None
                candidate_id = (
                    candidate.candidate_id
                    if candidate is not None
                    else self._candidate_by_assignment[key]
                )
                if candidate_id not in candidate_masses:
                    candidate_masses[candidate_id] = float(
                        (
                            self.candidate_log_probability(candidate)
                            if candidate is not None
                            else self.log_probability(key)
                        ).item()
                    )
                samples.append(
                    ProposalSample(candidate_id, key, component, candidate_masses[candidate_id])
                )
        return tuple(samples)


@dataclass(frozen=True)
class CompiledFamily:
    """One disjoint template with an explicit map to the reference Boolean space."""

    name: str
    circuit: CompiledCircuit | None
    indices: tuple[int, ...]
    fixed: tuple[tuple[int, bool], ...]
    status: str = "resolved"
    detail: str = ""
    failure_telemetry: tuple = ()


@dataclass(frozen=True)
class FactoredCircuit:
    encoding: Any
    families: tuple[CompiledFamily, ...]
    cache_key: str
    compilation_seconds: float
    compiler_version: str = "family-reference-v3"

    @property
    def node_count(self) -> int:
        return sum(f.circuit.node_count for f in self.families if f.circuit is not None)

    @property
    def complete(self) -> bool:
        return all(f.status in {"resolved", "empty_language"} for f in self.families)

    @property
    def telemetry(self) -> tuple:
        return tuple(
            (f.name, f.status, f.detail, f.circuit.telemetry if f.circuit else f.failure_telemetry)
            for f in self.families
        )


class FactoredConditionedMixture(ConditionedMixture):
    """CG-019 exactly: choose component/family using fixed-bit mass times WMC.

    The full reference logits remain the likelihood parameterization. Failed
    families explicitly define a reduced support; their missing mass is unknown.
    """

    def __init__(self, circuit: FactoredCircuit, literal_logits: Any, component_logits: Any = None):
        import torch
        import torch.nn.functional as functional

        if literal_logits.ndim != 2 or literal_logits.shape[1] != circuit.encoding.variable_count:
            raise ValueError("literal logits must have shape (components, reference variables)")
        if not literal_logits.is_floating_point() or torch.isnan(literal_logits).any():
            raise ValueError("literal logits must be floating point without NaN")
        count = literal_logits.shape[0]
        component_logits = (
            literal_logits.new_zeros(count) if component_logits is None else component_logits
        )
        if (
            count < 1
            or component_logits.shape != (count,)
            or component_logits.device != literal_logits.device
            or not torch.isfinite(component_logits).all()
        ):
            raise ValueError("invalid family mixture components")
        self.circuit = circuit
        self.literal_logits = literal_logits
        self.log_positive = functional.logsigmoid(literal_logits)
        self.log_negative = functional.logsigmoid(-literal_logits)
        self.log_mixture = torch.log_softmax(component_logits, 0)
        self._grammar = True
        self._candidate_by_assignment = {}
        self._families = []
        masses = []
        for family in circuit.families:
            if family.status != "resolved":
                continue
            fixed = literal_logits.new_zeros(count)
            for index, value in family.fixed:
                fixed = fixed + (self.log_positive if value else self.log_negative)[:, index]
            local = None
            if family.circuit is not None:
                try:
                    local = ConditionedMixture(
                        family.circuit, literal_logits[:, family.indices], component_logits
                    )
                except EmptyProposalSpace:
                    continue
                fixed = fixed + local.component_log_normalizers
            self._families.append((family, local))
            masses.append(fixed)
        if not masses:
            raise EmptyProposalSpace("no completed family has nonzero mass")
        self.family_component_log_normalizers = torch.stack(masses)
        self.component_log_normalizers = torch.logsumexp(self.family_component_log_normalizers, 0)
        self.log_normalizer = torch.logsumexp(self.log_mixture + self.component_log_normalizers, 0)
        if torch.isneginf(self.log_normalizer):
            raise EmptyProposalSpace("completed families have zero probability")
        if not torch.isfinite(self.log_normalizer):
            raise FloatingPointError("nonfinite family proposal normalizer")
        self.component_posterior = torch.softmax(
            self.log_mixture + self.component_log_normalizers, 0
        )
        self.family_component_posterior = torch.softmax(
            (self.family_component_log_normalizers + self.log_mixture).flatten(), 0
        ).reshape(len(masses), count)

    def accepts(self, assignment: Sequence[bool]) -> bool:
        bits = tuple(assignment)
        if len(bits) != self.circuit.encoding.variable_count or any(
            type(b) is not bool for b in bits
        ):
            return False
        for family, local in self._families:
            if all(bits[index] == value for index, value in family.fixed):
                if local is None or local.accepts(tuple(bits[i] for i in family.indices)):
                    return True
        return False

    def sample(self, count: int = 1, *, seed: int = 0) -> tuple[ProposalSample, ...]:
        import torch

        if type(count) is not int or count < 0:
            raise ValueError("sample count must be a nonnegative integer")
        generator = torch.Generator(device=self.literal_logits.device).manual_seed(seed)
        result = []
        cached = {}
        with torch.no_grad():
            for _ in range(count):
                index = int(
                    torch.multinomial(
                        self.family_component_posterior.flatten(), 1, generator=generator
                    )
                )
                family_index, component = divmod(index, self.literal_logits.shape[0])
                family, local = self._families[family_index]
                bits = dict(family.fixed)
                if local is not None:
                    assert family.circuit is not None
                    assignment = {}
                    pending = [family.circuit.root]
                    while pending:
                        node = pending.pop()
                        if node.is_false():
                            raise RuntimeError("sampled zero-mass family branch")
                        if node.is_literal():
                            assignment[abs(node.literal) - 1] = node.literal > 0
                        elif not node.is_true():
                            elements = node.elements()
                            weights = torch.stack(
                                [
                                    local._values[p.id][component] + local._values[s.id][component]
                                    for p, s in elements
                                ]
                            )
                            selected = int(
                                torch.multinomial(torch.softmax(weights, 0), 1, generator=generator)
                            )
                            pending.extend(reversed(elements[selected]))
                    for local_index, full_index in enumerate(family.indices):
                        if local_index not in assignment:
                            assignment[local_index] = bool(
                                torch.rand(
                                    (), device=self.literal_logits.device, generator=generator
                                )
                                < self.literal_logits[component, full_index].sigmoid()
                            )
                        bits[full_index] = assignment[local_index]
                full = tuple(bits[i] for i in range(self.circuit.encoding.variable_count))
                if not self.accepts(full) or not self.circuit.encoding.accepts(full):
                    raise RuntimeError("family binding produced an invalid full encoding")
                candidate = self.candidate(full)
                if candidate.candidate_id not in cached:
                    cached[candidate.candidate_id] = float(
                        self.candidate_log_probability(candidate)
                    )
                result.append(
                    ProposalSample(
                        candidate.candidate_id, full, component, cached[candidate.candidate_id]
                    )
                )
        return tuple(result)
