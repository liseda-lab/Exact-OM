"""Compact finite expression grammars compiled by the public PySDD API.

Only menus, templates and typed tree slots are represented. No expression or
replacement universe is enumerated. Several template derivations can denote the
same emitted bundle; ``candidate_assignments`` enumerates its finite inverse
images, so callers sum their probabilities rather than discard probability mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Any, Iterable, Sequence, cast

import pyowl_core as owl

from .candidates import (
    MAPPING_ACTIONS,
    ONTOLOGY_ACTIONS,
    canonical_expression,
    deduplicate_candidates,
    expression_order_key,
    expression_size,
    expression_tree,
    intersection,
    make_candidate,
    mapping_candidates,
    normalise_axioms,
    ontology_candidates,
    replacement_cost_features,
)
from .records import ReplacementCandidateV2, RevisionObjectV2, canonical_hash


class CircuitBudgetExceeded(RuntimeError):
    """The compiler exceeded an explicit resource limit without relaxing K."""


@dataclass(frozen=True)
class GrammarTemplate:
    """A fixed replacement or a single bounded expression argument template."""

    name: str
    action: str
    index: int = -1
    retained: str = "none"
    fixed: ReplacementCandidateV2 | None = None
    class_menu: tuple[str, ...] | None = None
    property_menu: tuple[str, ...] | None = None


@dataclass(frozen=True)
class GrammarEncoding:
    """Finite slot language with no materialized list of expression assignments."""

    revision: RevisionObjectV2
    classes: tuple[Any, ...]
    properties: tuple[Any, ...]
    templates: tuple[GrammarTemplate, ...]
    max_depth: int
    max_constructors: int
    constraint_identity: str = ""

    @property
    def slot_count(self) -> int:
        return int(2 ** (min(self.max_depth, self.max_constructors) + 1) - 1)

    @property
    def fields(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        fields = [("template", tuple(t.name for t in self.templates))]
        fields.extend(
            (f"retained:{owl.structural_hexdigest(a)}", ("false", "true"))
            for a in normalise_axioms(self.revision.original_axioms)
        )
        classes = tuple(str(c.iri.value) for c in self.classes)
        properties = tuple(str(p.iri.value) for p in self.properties)
        for slot in range(self.slot_count):
            prefix = f"expression:0:{slot}"
            fields.extend(
                (
                    (f"{prefix}:constructor", ("unused", "class", "and", "exists")),
                    (f"{prefix}:class", ("unused", *classes)),
                    (f"{prefix}:property", ("unused", *properties)),
                )
            )
        return tuple(fields)

    @property
    def variable_count(self) -> int:
        return sum(len(categories) for _, categories in self.fields)

    @property
    def content_hash(self) -> str:
        return str(canonical_hash(self))

    @property
    def elementary_candidates(self) -> tuple[ReplacementCandidateV2, ...]:
        return deduplicate_candidates(t.fixed for t in self.templates if t.fixed is not None)

    def representatives(self) -> tuple[ReplacementCandidateV2, ...]:
        """Deterministic family coverage in O(templates × observed classes).

        Try named leaves and the fixed original subclasses. This supplies controls
        and a bounded seed per applicable template without enumerating expressions.
        """
        arguments = list(self.classes)
        for axiom in normalise_axioms(self.revision.original_axioms):
            if isinstance(axiom, owl.SubClassOf):
                try:
                    arguments.append(canonical_expression(axiom.sub_class))
                except ValueError:
                    pass
        candidates = list(self.elementary_candidates)
        for template in self.templates:
            if template.fixed is not None:
                continue
            for expression in arguments:
                try:
                    self.assignment(template, expression)
                except ValueError:
                    continue
                symbols = tuple(owl.walk(expression))
                if template.class_menu is not None and any(
                    isinstance(n, owl.Class) and str(n.iri.value) not in template.class_menu
                    for n in symbols
                ):
                    continue
                if template.property_menu is not None and any(
                    isinstance(n, owl.ObjectProperty)
                    and str(n.iri.value) not in template.property_menu
                    for n in symbols
                ):
                    continue
                candidate = self.emit(template, expression)
                if all(
                    expression_size(e)[0] <= self.max_depth
                    and expression_size(e)[1] <= self.max_constructors
                    for e in (expression, *candidate.active_expressions)
                ):
                    candidates.append(candidate)
                    break
        return deduplicate_candidates(candidates)

    def retained_choices(self, template: GrammarTemplate) -> dict[str, str]:
        """Explicit retained-direction selectors determined by each complete template."""
        return {
            f"retained:{owl.structural_hexdigest(axiom)}": (
                "true"
                if (
                    axiom in template.fixed.axioms
                    if template.fixed is not None
                    else template.retained == "all"
                    or (template.retained == "other" and index != template.index)
                )
                else "false"
            )
            for index, axiom in enumerate(normalise_axioms(self.revision.original_axioms))
        }

    def choices(self, assignment: Sequence[bool]) -> dict[str, str]:
        if len(assignment) != self.variable_count or any(type(v) is not bool for v in assignment):
            raise ValueError("proposal assignment must be a Boolean vector of the circuit width")
        selected = {}
        offset = 0
        for name, categories in self.fields:
            row = assignment[offset : offset + len(categories)]
            if sum(row) != 1:
                raise ValueError("grammar fields must be one-hot")
            selected[name] = categories[row.index(True)]
            offset += len(categories)
        return selected

    def assignment(self, template: GrammarTemplate, expression: Any = None) -> tuple[bool, ...]:
        choices = {name: "unused" for name, _ in self.fields}
        choices["template"] = template.name
        choices.update(self.retained_choices(template))

        def fill(slot: int, tree: tuple) -> None:
            if slot >= self.slot_count:
                raise ValueError("expression exceeds grammar depth")
            prefix = f"expression:0:{slot}"
            choices[f"{prefix}:constructor"] = tree[0]
            if tree[0] == "class":
                choices[f"{prefix}:class"] = tree[1]
            elif tree[0] == "exists":
                choices[f"{prefix}:property"] = tree[1]
                fill(2 * slot + 1, tree[2])
            else:
                fill(2 * slot + 1, tree[1])
                fill(2 * slot + 2, tree[2])

        if expression is not None:
            fill(0, expression_tree(expression))
        for name, categories in self.fields:
            if choices[name] not in categories:
                raise ValueError("expression uses a symbol outside its observed finite menu")
        return tuple(choices[name] == value for name, values in self.fields for value in values)

    def _argument(self, choices: dict[str, str]) -> Any:
        def expression(slot: int) -> Any:
            prefix = f"expression:0:{slot}"
            constructor = choices[f"{prefix}:constructor"]
            if constructor == "class":
                if choices[f"{prefix}:class"] == "unused":
                    raise ValueError("active class slot requires a class choice")
                return owl.Class(owl.IRI(choices[f"{prefix}:class"]))
            if constructor == "exists":
                if choices[f"{prefix}:property"] == "unused":
                    raise ValueError("active existential slot requires a property choice")
                role = owl.ObjectProperty(owl.IRI(choices[f"{prefix}:property"]))
                return owl.ObjectSomeValuesFrom(role, expression(2 * slot + 1))
            if constructor == "and":
                return intersection(expression(2 * slot + 1), expression(2 * slot + 2))
            raise ValueError("active template requires an expression")

        return expression(0)

    def accepts(self, assignment: Sequence[bool]) -> bool:
        """Validate canonical typed slots independently of compilation or enumeration."""
        try:
            choices = self.choices(assignment)
            template = next(t for t in self.templates if t.name == choices["template"])
            if template.fixed is not None:
                return tuple(assignment) == self.assignment(template)
            expression = self._argument(choices)
            if tuple(assignment) != self.assignment(template, expression):
                return False
            for node in owl.walk(expression):
                if (
                    isinstance(node, owl.Class)
                    and template.class_menu is not None
                    and str(node.iri.value) not in template.class_menu
                ):
                    return False
                if (
                    isinstance(node, owl.ObjectProperty)
                    and template.property_menu is not None
                    and str(node.iri.value) not in template.property_menu
                ):
                    return False
            candidate = self.emit(template, expression)
            return all(
                expression_size(e)[0] <= self.max_depth
                and expression_size(e)[1] <= self.max_constructors
                for e in (expression, *candidate.active_expressions)
            )
        except (ValueError, TypeError, KeyError, StopIteration):
            return False

    def decode(self, assignment: Sequence[bool]) -> ReplacementCandidateV2:
        """Decode a validated assignment; use ``accepts`` for untrusted slot vectors."""
        choices = self.choices(assignment)
        template = next(t for t in self.templates if t.name == choices["template"])
        return (
            template.fixed
            if template.fixed is not None
            else self.emit(template, self._argument(choices))
        )

    def emit(self, template: GrammarTemplate, expression: Any) -> ReplacementCandidateV2:
        if template.fixed is not None:
            return template.fixed
        expression = canonical_expression(expression)
        originals = normalise_axioms(self.revision.original_axioms)
        axiom = originals[template.index]
        left, right = axiom.sub_class, axiom.super_class
        retained = tuple(a for i, a in enumerate(originals) if i != template.index)
        active: tuple[Any, ...] = ()
        emitted: tuple[Any, ...]
        if template.action in {
            "specialise_subclass",
            "specialise_ontology_subclass",
            "composite",
            "complex_equivalence",
        }:
            specialised = intersection(left, expression)
            emitted = (owl.SubClassOf(specialised, right, axiom.annotations),)
            active = (specialised,)
            if template.retained in {"other", "all"}:
                emitted += retained
            if template.action in {"composite", "complex_equivalence"}:
                emitted += (owl.SubClassOf(right, expression),)
        else:
            emitted = (owl.SubClassOf(right, expression),)
            emitted += originals if template.retained == "all" else retained
        costs = replacement_cost_features(
            originals,
            emitted,
            kind=self.revision.kind,
            authorship=self.revision.authorship,
            active_expressions=active,
        )
        provenance = [("grammar_template", template.name)]
        if self.revision.kind == "ontology_axiom":
            provenance.extend(
                (("occurrence_id", self.revision.occurrence_id), ("source", self.revision.source))
            )
        else:
            provenance.extend(
                (("subclass", str(left.iri.value)), ("superclass", str(right.iri.value)))
            )
        return make_candidate(
            self.revision.object_id,
            emitted,
            (template.action,),
            active_expressions=active,
            provenance=provenance,
            cost_features=costs,
        )

    def candidate_assignments(
        self, candidate: ReplacementCandidateV2
    ) -> tuple[tuple[bool, ...], ...]:
        """Find inverse encodings from the candidate syntax, without grammar enumeration.

        Every argument occurs as an emitted superclass or inside an activated
        specialised subclass. Intersection absorption can remove original left
        operands; all corresponding subsets must therefore be restored. Mapping
        left endpoints are named. Ontology intersections use their finite operand
        subsets, bounded by the candidate's own syntax rather than vocabulary size.
        """
        from itertools import combinations

        expressions = set()
        for root in (*candidate.axioms, *candidate.active_expressions):
            for node in owl.walk(root):
                if isinstance(
                    node, (owl.Class, owl.ObjectIntersectionOf, owl.ObjectSomeValuesFrom)
                ):
                    try:
                        expressions.add(canonical_expression(node))
                    except ValueError:
                        pass
        original_lefts = [
            a.sub_class
            for a in normalise_axioms(self.revision.original_axioms)
            if isinstance(a, owl.SubClassOf)
        ]
        for active in candidate.active_expressions:
            operands = (
                set(active.operands) if isinstance(active, owl.ObjectIntersectionOf) else {active}
            )
            for left in original_lefts:
                removed = (
                    set(left.operands) if isinstance(left, owl.ObjectIntersectionOf) else {left}
                )
                if not removed <= operands:
                    continue
                mandatory = operands - removed
                optional = sorted(removed, key=owl.canonical_bytes)
                for size in range(len(optional) + 1):
                    for restored in combinations(optional, size):
                        choices = mandatory | set(restored)
                        if choices:
                            expressions.add(intersection(*choices))
        result = set()
        for template in self.templates:
            if template.fixed is not None:
                if template.fixed.candidate_id == candidate.candidate_id:
                    result.add(self.assignment(template))
                continue
            for expression in expressions:
                if self.emit(template, expression).candidate_id == candidate.candidate_id:
                    try:
                        result.add(self.assignment(template, expression))
                    except ValueError:
                        pass
        return tuple(sorted(bits for bits in result if self.accepts(bits)))


def mapping_grammar(
    revision: RevisionObjectV2,
    classes: Iterable[Any],
    properties: Iterable[Any] = (),
    *,
    max_depth: int = 2,
    max_constructors: int = 2,
    enabled_actions: Iterable[str] | None = None,
    fixed_axioms: Iterable[Any] = (),
    constraint_identity: str = "",
    source_classes: Iterable[Any] | None = None,
    target_classes: Iterable[Any] | None = None,
    source_properties: Iterable[Any] | None = None,
    target_properties: Iterable[Any] | None = None,
) -> GrammarEncoding:
    """Build a mapping or ontology grammar from observed symbols and finite controls.

    Bounds cover both the generated argument E and each canonical activated
    intersection after the template adds its fixed left endpoint. Existing endpoint alternatives and justified ontology edits remain
    finite template branches. They do not require enumerating any expression.
    """
    if any(type(v) is not int or v < 0 for v in (max_depth, max_constructors)) or max_depth > 8:
        raise ValueError("grammar bounds must be nonnegative integers with depth at most eight")
    classes, properties = tuple(classes), tuple(properties)
    if any(not isinstance(c, owl.Class) for c in classes) or any(
        not isinstance(p, owl.ObjectProperty) for p in properties
    ):
        raise TypeError("grammar menus require named shared-core classes and object properties")
    classes = tuple(sorted(set(classes), key=lambda c: str(c.iri.value)))
    properties = tuple(sorted(set(properties), key=lambda p: str(p.iri.value)))
    side_menus = {
        "source": (
            None if source_classes is None else tuple(str(c.iri.value) for c in source_classes),
            (
                None
                if source_properties is None
                else tuple(str(p.iri.value) for p in source_properties)
            ),
        ),
        "target": (
            None if target_classes is None else tuple(str(c.iri.value) for c in target_classes),
            (
                None
                if target_properties is None
                else tuple(str(p.iri.value) for p in target_properties)
            ),
        ),
    }
    allowed = MAPPING_ACTIONS if revision.kind == "mapping" else ONTOLOGY_ACTIONS
    actions = allowed if enabled_actions is None else frozenset(enabled_actions) | {"keep"}
    if actions - allowed:
        raise ValueError("unknown replacement action")
    originals = normalise_axioms(revision.original_axioms)
    controls = list(revision.candidates)
    if revision.kind == "mapping":
        controls.append(
            make_candidate(revision.object_id, originals, ("keep",), cost_features=(("edit", 0.0),))
        )
        if revision.eligible and not revision.locked:
            if "delete" in actions:
                controls.append(
                    make_candidate(
                        revision.object_id,
                        (),
                        ("delete",),
                        cost_features=replacement_cost_features(originals, ()),
                    )
                )
            if originals and all(
                isinstance(a, owl.SubClassOf)
                and isinstance(a.sub_class, owl.Class)
                and isinstance(a.super_class, owl.Class)
                for a in originals
            ):
                first = originals[0]
                if len(originals) == 1 or (
                    len(originals) == 2
                    and originals[1].sub_class == first.super_class
                    and originals[1].super_class == first.sub_class
                ):
                    controls.extend(
                        mapping_candidates(
                            revision.object_id,
                            first.sub_class,
                            first.super_class,
                            "=" if len(originals) == 2 else "<",
                            enabled_actions=actions,
                        )
                    )
    if revision.kind == "ontology_axiom":
        controls.extend(
            ontology_candidates(revision, fixed_axioms=fixed_axioms, enabled_actions=actions)
        )
    # Fixed syntax-changing ontology controls are finite. Expression specialisation
    # and complex mapping candidates are represented by slots, never a full pool.
    fixed_actions = {"keep", "delete", "retain_subsumption", "replace_endpoint"} | (
        set(ONTOLOGY_ACTIONS) - {"specialise_ontology_subclass"}
    )
    controls = [
        c for c in deduplicate_candidates(controls) if set(c.action_tags) & actions & fixed_actions
    ]
    templates = [
        GrammarTemplate(f"fixed:{c.candidate_id}", c.action_tags[0], fixed=c) for c in controls
    ]
    if revision.eligible and not revision.locked and classes:
        for index, axiom in enumerate(originals):
            if not isinstance(axiom, owl.SubClassOf):
                continue
            try:
                canonical_expression(axiom.sub_class)
            except ValueError:
                continue
            if revision.kind == "ontology_axiom":
                candidates = [("specialise_ontology_subclass", "other")]
            elif isinstance(axiom.sub_class, owl.Class) and isinstance(
                axiom.super_class, owl.Class
            ):
                candidates = [
                    ("specialise_subclass", "none"),
                    ("add_necessary_condition", "all"),
                    ("composite", "other"),
                ]
                if len(originals) > 1:
                    candidates += [
                        ("specialise_subclass", "other"),
                        ("add_necessary_condition", "other"),
                        ("complex_equivalence", "other"),
                    ]
            else:
                continue
            for action, retained in candidates:
                if action in actions:
                    side = ""
                    if revision.kind == "ontology_axiom":
                        side = revision.source.split(":", 1)[0]
                    elif (
                        axiom.sub_class == revision.source_entity
                        or str(cast(owl.Class, axiom.sub_class).iri.value) == revision.source_entity
                    ):
                        side = "source"
                    elif (
                        axiom.sub_class == revision.target_entity
                        or str(cast(owl.Class, axiom.sub_class).iri.value) == revision.target_entity
                    ):
                        side = "target"
                    class_menu, property_menu = side_menus.get(side, (None, None))
                    templates.append(
                        GrammarTemplate(
                            f"{action}:{index}:{retained}",
                            action,
                            index,
                            retained,
                            class_menu=class_menu,
                            property_menu=property_menu,
                        )
                    )
    if not templates:
        raise ValueError("a grammar requires at least its unchanged control")
    return GrammarEncoding(
        revision,
        classes,
        properties,
        tuple(templates),
        max_depth,
        max_constructors,
        constraint_identity,
    )


def compile_grammar(
    encoding: GrammarEncoding,
    *,
    variable_order: Sequence[int] | None = None,
    max_nodes: int = 100000,
    max_seconds: float | None = None,
) -> Any:
    """Compile typed-slot constraints with PySDD, without enumerating expressions.

    The cooperative clock/node checks run between public compiler operations.
    A hard wall deadline must additionally supervise the whole proposal worker,
    because an individual native compiler operation cannot be interrupted here.
    """
    from importlib.metadata import version

    if (
        type(max_nodes) is not int
        or max_nodes < 1
        or (max_seconds is not None and max_seconds <= 0)
    ):
        raise ValueError("compiler resource limits must be positive")
    order = (
        tuple(range(1, encoding.variable_count + 1))
        if variable_order is None
        else tuple(variable_order)
    )
    if sorted(order) != list(range(1, encoding.variable_count + 1)):
        raise ValueError("variable order must be a permutation of all Boolean variables")
    return _compile_grammar(encoding, order, max_nodes, max_seconds, version("pysdd"))


@lru_cache(maxsize=16)
def _compile_grammar(
    encoding: GrammarEncoding,
    order: tuple[int, ...],
    max_nodes: int,
    max_seconds: float | None,
    compiler_version: str,
) -> Any:
    from pysdd.sdd import SddManager, Vtree

    from .circuit import CompiledCircuit

    started = perf_counter()
    manager = SddManager(
        vtree=Vtree(var_count=encoding.variable_count, var_order=list(order), vtree_type="right"),
        auto_gc_and_minimize=False,
    )
    true, false = manager.true(), manager.false()
    literals = {}
    offset = 1
    for name, categories in encoding.fields:
        literals[name] = {value: manager.literal(offset + i) for i, value in enumerate(categories)}
        offset += len(categories)

    def check() -> None:
        if manager.count() > max_nodes:
            raise CircuitBudgetExceeded(
                f"grammar compilation exceeds {max_nodes} allocated SDD nodes"
            )
        if max_seconds is not None and perf_counter() - started > max_seconds:
            raise CircuitBudgetExceeded("grammar compilation exceeds its wall time limit")

    def disjunction(values: Iterable[Any]) -> Any:
        result = false
        for value in values:
            result = result | value
            check()
        return result

    def conjunction(values: Iterable[Any]) -> Any:
        result = true
        for value in values:
            result = result & value
            check()
        return result

    def field(slot: int, kind: str, value: str) -> Any:
        if slot >= encoding.slot_count:
            return true if value == "unused" else false
        return literals[f"expression:0:{slot}:{kind}"].get(value, false)

    constraints = []
    for category_literals in literals.values():
        zero, one = true, false
        for literal in category_literals.values():
            one = (one & ~literal) | (zero & literal)
            zero = zero & ~literal
            check()
        constraints.append(one)

    @lru_cache(maxsize=None)
    def category_compare(
        left: int, right: int, kind: str, values: tuple[str, ...]
    ) -> tuple[Any, Any]:
        equal, less, greater_choices = false, false, false
        for value in reversed(values):
            a, b = field(left, kind, value), field(right, kind, value)
            equal = equal | (a & b)
            less = less | (a & greater_choices)
            greater_choices = greater_choices | b
            check()
        return less, equal

    @lru_cache(maxsize=None)
    def compare(left: int, right: int) -> tuple[Any, Any]:
        # Canonical structural order: class, intersection, existential. Intersections
        # are right-associated sorted sequences of non-intersection operands.
        if left >= encoding.slot_count or right >= encoding.slot_count:
            return false, (
                true if left >= encoding.slot_count and right >= encoding.slot_count else false
            )
        less, equal = false, false
        kinds = ("class", "and", "exists")
        for i, kind in enumerate(kinds):
            a, b = field(left, "constructor", kind), field(right, "constructor", kind)
            less = less | (a & disjunction(field(right, "constructor", k) for k in kinds[i + 1 :]))
            if kind == "class":
                child_less, child_equal = category_compare(
                    left, right, "class", tuple(str(c.iri.value) for c in encoding.classes)
                )
            elif kind == "exists":
                role_less, role_equal = category_compare(
                    left, right, "property", tuple(str(p.iri.value) for p in encoding.properties)
                )
                filler_less, filler_equal = compare(2 * left + 1, 2 * right + 1)
                child_less, child_equal = (
                    role_less | (role_equal & filler_less),
                    role_equal & filler_equal,
                )
            else:
                a_less, a_equal = compare(2 * left + 1, 2 * right + 1)
                b_less, b_equal = compare(2 * left + 2, 2 * right + 2)
                child_less, child_equal = a_less | (a_equal & b_less), a_equal & b_equal
            less = less | (a & b & child_less)
            equal = equal | (a & b & child_equal)
            check()
        return less, equal

    for slot in range(encoding.slot_count):
        unused, leaf, both, exists = (
            field(slot, "constructor", k) for k in ("unused", "class", "and", "exists")
        )
        left, right = 2 * slot + 1, 2 * slot + 2
        left_unused, right_unused = field(left, "constructor", "unused"), field(
            right, "constructor", "unused"
        )
        constraints.extend(
            (
                ~leaf | ~field(slot, "class", "unused"),
                leaf | field(slot, "class", "unused"),
                ~exists | ~field(slot, "property", "unused"),
                exists | field(slot, "property", "unused"),
            )
        )
        constraints.append(~(unused | leaf) | (left_unused & right_unused))
        constraints.append(~exists | (~left_unused & right_unused))
        constraints.append(
            ~both | (~left_unused & ~right_unused & ~field(left, "constructor", "and"))
        )
        direct_less, _ = compare(left, right)
        sequence_less, _ = compare(left, 2 * right + 1)
        ordered = (field(right, "constructor", "and") & sequence_less) | (
            ~field(right, "constructor", "and") & direct_less
        )
        constraints.append(~both | ordered)

    @lru_cache(maxsize=None)
    def fits(slot: int, depth: int, budget: int) -> Any:
        if depth < 0 or budget < 0 or slot >= encoding.slot_count:
            return false
        result = field(slot, "constructor", "class")
        if depth and budget:
            result = result | (
                field(slot, "constructor", "exists") & fits(2 * slot + 1, depth - 1, budget - 1)
            )
            result = result | (
                field(slot, "constructor", "and")
                & disjunction(
                    fits(2 * slot + 1, depth - 1, n) & fits(2 * slot + 2, depth - 1, budget - 1 - n)
                    for n in range(budget)
                )
            )
        check()
        return result

    @lru_cache(maxsize=None)
    def compare_constant(slot: int, tree: tuple) -> tuple[Any, Any]:
        if slot >= encoding.slot_count:
            return false, false
        kinds = ("class", "and", "exists")
        kind = tree[0]
        less = disjunction(field(slot, "constructor", k) for k in kinds[: kinds.index(kind)])
        if kind == "class":
            child_less = disjunction(
                field(slot, "class", str(c.iri.value))
                for c in encoding.classes
                if str(c.iri.value) < tree[1]
            )
            child_equal = field(slot, "class", tree[1])
        elif kind == "exists":
            role_less = disjunction(
                field(slot, "property", str(p.iri.value))
                for p in encoding.properties
                if str(p.iri.value) < tree[1]
            )
            role_equal = field(slot, "property", tree[1])
            filler_less, filler_equal = compare_constant(2 * slot + 1, tree[2])
            child_less, child_equal = (
                role_less | (role_equal & filler_less),
                role_equal & filler_equal,
            )
        else:
            left_less, left_equal = compare_constant(2 * slot + 1, tree[1])
            right_less, right_equal = compare_constant(2 * slot + 2, tree[2])
            child_less, child_equal = (
                left_less | (left_equal & right_less),
                left_equal & right_equal,
            )
        return (
            less | (field(slot, "constructor", kind) & child_less),
            field(slot, "constructor", kind) & child_equal,
        )

    def specialised_bounds(left: Any) -> Any:
        fixed = tuple(
            sorted(
                left.operands if isinstance(left, owl.ObjectIntersectionOf) else (left,),
                key=expression_order_key,
            )
        )
        fixed_sizes = tuple(expression_size(a) for a in fixed)

        @lru_cache(maxsize=None)
        def merge(slot: int | None, index: int, depth: int, budget: int) -> Any:
            if depth < 0 or budget < 0:
                return false
            if slot is None:
                if index == len(fixed):
                    return false
                d, n = expression_size(intersection(*fixed[index:]))
                return true if d <= depth and n <= budget else false
            if index == len(fixed):
                return fits(slot, depth, budget)

            def emit_variable(atom: int, tail: int | None, next_index: int) -> Any:
                if tail is None and next_index == len(fixed):
                    return fits(atom, depth, budget)
                return disjunction(
                    fits(atom, depth - 1, n) & merge(tail, next_index, depth - 1, budget - 1 - n)
                    for n in range(budget)
                )

            def emit_fixed(tail: int | None, next_index: int) -> Any:
                d, n = fixed_sizes[index]
                if tail is None and next_index == len(fixed):
                    return true if d <= depth and n <= budget else false
                if d > depth - 1:
                    return false
                return merge(tail, next_index, depth - 1, budget - 1 - n)

            result = false
            for is_and in (False, True):
                atom = 2 * slot + 1 if is_and else slot
                tail = 2 * slot + 2 if is_and else None
                selected = field(slot, "constructor", "and")
                selected = selected if is_and else ~selected
                less, equal = compare_constant(atom, expression_tree(fixed[index]))
                result = result | (
                    selected
                    & (
                        (less & emit_variable(atom, tail, index))
                        | (equal & emit_fixed(tail, index + 1))
                        | (~less & ~equal & emit_fixed(slot, index + 1))
                    )
                )
                check()
            return result

        return merge(0, 0, encoding.max_depth, encoding.max_constructors)

    originals = normalise_axioms(encoding.revision.original_axioms)
    for template in encoding.templates:
        inactive = field(0, "constructor", "unused")
        valid = (
            inactive
            if template.fixed is not None
            else ~inactive & fits(0, encoding.max_depth, encoding.max_constructors)
        )
        if template.fixed is None and template.action in {
            "specialise_subclass",
            "specialise_ontology_subclass",
            "composite",
            "complex_equivalence",
        }:
            valid = valid & specialised_bounds(originals[template.index].sub_class)
        if template.fixed is None:
            for slot in range(encoding.slot_count):
                if template.class_menu is not None:
                    valid = valid & disjunction(
                        field(slot, "class", value) for value in ("unused", *template.class_menu)
                    )
                if template.property_menu is not None:
                    valid = valid & disjunction(
                        field(slot, "property", value)
                        for value in ("unused", *template.property_menu)
                    )
        valid = valid & conjunction(
            literals[name][value] for name, value in encoding.retained_choices(template).items()
        )
        constraints.append(~literals["template"][template.name] | valid)
    root = conjunction(constraints)
    root.ref()
    manager.garbage_collect()
    key = canonical_hash(
        (encoding.content_hash, order, "pysdd", compiler_version, "typed-slots-v1")
    )
    return CompiledCircuit(
        encoding, manager, root, str(key), compiler_version, perf_counter() - started, root.count()
    )
