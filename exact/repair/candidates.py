"""Finite replacement templates over the shared OWL model.

Generation is syntactic. Only asserted, fixed subclass premises justify the
simple generalisations here; editable assertions never prune the grammar.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
from itertools import combinations
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

import pyowl_core as owl

from .records import ReplacementCandidateV2, RevisionObjectV2

MAPPING_ACTIONS = frozenset(
    {
        "keep",
        "delete",
        "retain_subsumption",
        "replace_endpoint",
        "specialise_subclass",
        "add_necessary_condition",
        "complex_equivalence",
        "composite",
    }
)
ONTOLOGY_ACTIONS = frozenset(
    {
        "keep",
        "delete",
        "remove_disjointness",
        "remove_superclass_conjunct",
        "specialise_ontology_subclass",
        "generalise_superclass",
        "generalise_domain",
        "generalise_range",
        "generalise_existential_filler",
    }
)


@lru_cache(maxsize=65536)
def _key(value: Any) -> bytes:
    return cast(bytes, owl.canonical_bytes(value))


def canonical_expression(expression: Any) -> Any:
    """Flatten/sort/deduplicate intersections without inventing OWL equivalences."""
    if isinstance(expression, owl.Class):
        return expression
    if isinstance(expression, owl.ObjectSomeValuesFrom):
        if not isinstance(expression.property, owl.ObjectProperty):
            raise ValueError("the proposal grammar permits named object properties only")
        return owl.ObjectSomeValuesFrom(
            expression.property, canonical_expression(expression.filler)
        )
    if isinstance(expression, owl.ObjectIntersectionOf):
        operands: list[Any] = []
        for operand in expression.operands:
            child = canonical_expression(operand)
            operands.extend(
                child.operands if isinstance(child, owl.ObjectIntersectionOf) else [child]
            )
        unique = sorted(set(operands), key=_key)
        if len(unique) == 1:
            return unique[0]
        return owl.ObjectIntersectionOf(owl.CanonicalSet(unique))
    raise ValueError("expression is outside the named-class/intersection/existential grammar")


def intersection(*expressions: Any) -> Any:
    """Construct the unique shared-core value for an associative intersection."""
    unique = set(expressions)
    if not unique:
        raise ValueError("an intersection requires an operand")
    if len(unique) == 1:
        return canonical_expression(next(iter(unique)))
    return canonical_expression(owl.ObjectIntersectionOf(owl.CanonicalSet(unique)))


def expression_order_key(expression: Any) -> tuple:
    """Stable structural order usable by both finite enumeration and slot constraints."""
    expression = canonical_expression(expression)
    if isinstance(expression, owl.Class):
        return (0, str(expression.iri.value))
    if isinstance(expression, owl.ObjectSomeValuesFrom):
        return (
            2,
            str(cast(owl.ObjectProperty, expression.property).iri.value),
            expression_order_key(expression.filler),
        )
    operands = sorted(expression_order_key(e) for e in expression.operands)
    result = operands[-1]
    for operand in reversed(operands[:-1]):
        result = (1, operand, result)
    return result


def expression_tree(expression: Any) -> tuple:
    """Return the fixed right-associated binary encoding of a canonical expression."""
    expression = canonical_expression(expression)
    if isinstance(expression, owl.Class):
        return ("class", str(expression.iri.value))
    if isinstance(expression, owl.ObjectSomeValuesFrom):
        return (
            "exists",
            str(cast(owl.ObjectProperty, expression.property).iri.value),
            expression_tree(expression.filler),
        )
    operands = sorted(expression.operands, key=expression_order_key)
    children = [expression_tree(item) for item in operands]
    result = children[-1]
    for child in reversed(children[:-1]):
        result = ("and", child, result)
    return result


def expression_size(expression: Any) -> tuple[int, int]:
    """Return (root-zero maximum depth, constructor count) on the canonical tree."""

    def size(tree: tuple) -> tuple[int, int]:
        if tree[0] == "class":
            return 0, 0
        children = [size(tree[2])] if tree[0] == "exists" else [size(t) for t in tree[1:]]
        return 1 + max(d for d, _ in children), 1 + sum(n for _, n in children)

    return size(expression_tree(expression))


def finite_expression_menu(
    classes: Iterable[Any],
    properties: Iterable[Any] = (),
    *,
    max_depth: int = 2,
    max_constructors: int = 2,
    max_expressions: int = 10000,
) -> tuple[Any, ...]:
    """Enumerate a finite grammar; fail explicitly rather than silently truncate it."""
    if any(type(v) is not int or v < 0 for v in (max_depth, max_constructors)):
        raise ValueError("grammar bounds must be nonnegative integers")
    if type(max_expressions) is not int or max_expressions < 1:
        raise ValueError("max_expressions must be a positive integer")
    leaves, roles = set(classes), set(properties)
    if any(not isinstance(c, owl.Class) for c in leaves):
        raise TypeError("class menu must contain shared-core named classes")
    if any(not isinstance(r, owl.ObjectProperty) for r in roles):
        raise TypeError("property menu must contain shared-core named object properties")
    if len(leaves) > max_expressions:
        raise ValueError("finite grammar exceeds max_expressions")
    levels: list[set[Any]] = [leaves] + [set() for _ in range(max_constructors)]
    total = len(leaves)
    for constructors in range(1, max_constructors + 1):

        def add(expression: Any) -> None:
            nonlocal total
            depth, count = expression_size(expression)
            if count != constructors or depth > max_depth or expression in levels[count]:
                return
            levels[count].add(expression)
            total += 1
            if total > max_expressions:
                raise ValueError("finite grammar exceeds max_expressions")

        for role in sorted(roles, key=_key):
            for filler in sorted(levels[constructors - 1], key=_key):
                add(owl.ObjectSomeValuesFrom(role, filler))
        for left_size in range(constructors):
            right_size = constructors - left_size - 1
            for left in sorted(levels[left_size], key=_key):
                for right in sorted(levels[right_size], key=_key):
                    if _key(left) < _key(right):
                        add(intersection(left, right))
    return tuple(sorted(set().union(*levels), key=_key))


def normalise_axioms(axioms: Iterable[Any]) -> tuple[Any, ...]:
    """Expand class equivalence to its complete set of directed inclusions."""
    result: set[Any] = set()
    for axiom in axioms:
        if isinstance(axiom, owl.EquivalentClasses):
            result.update(
                owl.SubClassOf(a, b, axiom.annotations)
                for a in axiom.expressions
                for b in axiom.expressions
                if a != b
            )
        else:
            result.add(axiom)
    return tuple(sorted(result, key=_key))


def _expression_cost(axioms: Iterable[Any]) -> float:
    """Count canonical grammar constructors in the complete emitted replacement."""
    return float(
        sum(
            (
                len(node.operands) - 1
                if isinstance(node, owl.ObjectIntersectionOf)
                else int(isinstance(node, owl.ObjectSomeValuesFrom))
            )
            for axiom in axioms
            for node in owl.walk(axiom)
        )
    )


def replacement_cost_features(
    original_axioms: Iterable[Any],
    emitted_axioms: Iterable[Any],
    *,
    kind: str = "mapping",
    authorship: str = "unknown",
    active_expressions: Iterable[Any] = (),
) -> tuple[tuple[str, float], ...]:
    """Derive protocol costs from complete emitted syntax, independent of aliases.

    Removed directions count original mapping inclusions absent from the bundle.
    Constructor increments count introduced canonical constructor occurrences;
    existing constructors retained by an edit are not charged as newly created.
    """
    from collections import Counter

    if kind not in {"mapping", "ontology_axiom"}:
        raise ValueError("cost features require a mapping or ontology-axiom object")
    original, emitted = normalise_axioms(original_axioms), normalise_axioms(emitted_axioms)
    edited = original != emitted or bool(tuple(active_expressions))
    mapping_edit = kind == "mapping" and edited
    ontology_edit = kind == "ontology_axiom" and edited

    def constructors(axioms: tuple[Any, ...]) -> Counter[bytes]:
        result: Counter[bytes] = Counter()
        for axiom in axioms:
            for node in owl.walk(axiom):
                count = (
                    len(node.operands) - 1
                    if isinstance(node, owl.ObjectIntersectionOf)
                    else int(isinstance(node, owl.ObjectSomeValuesFrom))
                )
                if count:
                    result[_key(node)] += count
        return result

    introduced = constructors(emitted) - constructors(original)
    directed = (owl.SubClassOf, owl.SubObjectPropertyOf, owl.SubDataPropertyOf)
    removed = (
        sum(isinstance(a, directed) and a not in emitted for a in original) if mapping_edit else 0
    )
    original_classes = [a for a in original if isinstance(a, owl.SubClassOf)]
    emitted_classes = [a for a in emitted if isinstance(a, owl.SubClassOf)]

    def operands(expression: Any) -> set[Any]:
        return (
            set(expression.operands)
            if isinstance(expression, owl.ObjectIntersectionOf)
            else {expression}
        )

    specialised = {
        a
        for a in emitted_classes
        if a not in original
        and any(
            a.super_class == old.super_class
            and a.sub_class != old.sub_class
            and isinstance(a.sub_class, owl.ObjectIntersectionOf)
            and operands(old.sub_class) < operands(a.sub_class)
            for old in original_classes
        )
    }

    def endpoint_pair(axioms: list[Any]) -> set[Any] | None:
        if not axioms or any(
            not isinstance(e, owl.Class) for a in axioms for e in (a.sub_class, a.super_class)
        ):
            return None
        if len(axioms) == 1:
            return {axioms[0].sub_class, axioms[0].super_class}
        if (
            len(axioms) == 2
            and axioms[0].sub_class == axioms[1].super_class
            and axioms[0].super_class == axioms[1].sub_class
        ):
            return {axioms[0].sub_class, axioms[0].super_class}
        return None

    before, after = endpoint_pair(original_classes), endpoint_pair(emitted_classes)
    endpoint_change = bool(
        mapping_edit
        and len(original_classes) == len(emitted_classes)
        and before is not None
        and after is not None
        and before != after
    )
    necessary = (
        sum(a not in original and a not in specialised for a in emitted_classes)
        if mapping_edit and not endpoint_change
        else 0
    )
    values = {
        "edit": float(edited),
        "delete": float(edited and not emitted),
        "mapping_deletion": float(mapping_edit and not emitted),
        "relation_change": float(mapping_edit),
        "removed_direction": float(removed),
        "endpoint_change": float(endpoint_change),
        "subclass_specialisation": float(len(specialised)),
        "necessary_condition": float(necessary),
        "expression_size": _expression_cost(emitted) if edited else 0.0,
        "new_constructor": float(sum(introduced.values())) if edited else 0.0,
        "ontology_edit": float(ontology_edit),
        "human_ontology_edit": float(ontology_edit and authorship == "human"),
        "human_authored_ontology_edit": float(ontology_edit and authorship == "human"),
    }
    return tuple(sorted(values.items()))


def make_candidate(
    object_id: str,
    axioms: Iterable[Any],
    action_tags: Iterable[str],
    *,
    active_expressions: Iterable[Any] = (),
    provenance: Iterable[tuple[str, str]] = (),
    cost_features: Iterable[tuple[str, float]] = (),
) -> ReplacementCandidateV2:
    """Assign identity from the complete replacement and its activated obligations."""
    emitted = normalise_axioms(axioms)
    active = tuple(sorted({canonical_expression(e) for e in active_expressions}, key=_key))
    digest = sha256()
    for chunk in [
        object_id.encode(),
        *[_key(a) for a in emitted],
        b"activation",
        *[_key(e) for e in active],
    ]:
        digest.update(len(chunk).to_bytes(8, "big"))
        digest.update(chunk)
    return ReplacementCandidateV2(
        object_id=object_id,
        candidate_id=digest.hexdigest(),
        axioms=emitted,
        action_tags=tuple(sorted(set(action_tags))),
        active_expressions=active,
        cost_features=tuple(sorted(cost_features)),
        provenance=tuple(sorted(set(provenance))),
    )


def deduplicate_candidates(
    candidates: Iterable[ReplacementCandidateV2],
) -> tuple[ReplacementCandidateV2, ...]:
    """Merge action/provenance aliases while preserving distinct activations."""
    unique: dict[str, ReplacementCandidateV2] = {}
    for candidate in candidates:
        normalized = make_candidate(
            candidate.object_id,
            candidate.axioms,
            candidate.action_tags,
            active_expressions=candidate.active_expressions,
            provenance=candidate.provenance,
            cost_features=candidate.cost_features,
        )
        previous = unique.get(normalized.candidate_id)
        if previous is not None:
            # Costs describe resulting edits, not the number of derivations.
            chosen = previous if "keep" in previous.action_tags else normalized
            normalized = replace(
                chosen,
                action_tags=tuple(sorted(set(previous.action_tags) | set(normalized.action_tags))),
                provenance=tuple(sorted(set(previous.provenance) | set(normalized.provenance))),
            )
        unique[normalized.candidate_id] = normalized
    return tuple(
        sorted(unique.values(), key=lambda c: ("keep" not in c.action_tags, c.candidate_id))
    )


def mapping_candidates(
    object_id: str,
    source: Any,
    target: Any,
    relation: str = "=",
    *,
    entity_kind: str = "class",
    eligible: bool = True,
    locked: bool = False,
    expressions: Iterable[Any] = (),
    endpoint_alternatives: Iterable[tuple[str, Any]] = (),
    enabled_actions: Iterable[str] = MAPPING_ACTIONS,
) -> tuple[ReplacementCandidateV2, ...]:
    """Build complete mirrored replacements; directional inputs never gain a reverse control."""
    actions = frozenset(enabled_actions) | {"keep"}
    if actions - MAPPING_ACTIONS:
        raise ValueError("unknown mapping action")
    if relation not in {"=", "<", ">"}:
        raise ValueError("mapping relation must be '=', '<', or '>'")
    kinds = {
        "class": owl.Class,
        "object_property": owl.ObjectProperty,
        "data_property": owl.DataProperty,
        "individual": owl.NamedIndividual,
    }
    if entity_kind not in kinds or not all(
        isinstance(e, kinds[entity_kind]) for e in (source, target)
    ):
        raise TypeError("mapping endpoints must match their declared shared-core entity kind")

    def originals(left: Any, right: Any) -> tuple[Any, ...]:
        if entity_kind == "individual":
            if relation != "=":
                raise ValueError("individual mappings support equality only")
            return (owl.SameIndividual(owl.CanonicalSet((left, right))),)
        constructor: Callable[..., Any]
        if entity_kind == "class":
            constructor = owl.SubClassOf
        elif entity_kind == "object_property":
            constructor = owl.SubObjectPropertyOf
        else:
            constructor = owl.SubDataPropertyOf
        return tuple(
            constructor(a, b)
            for a, b, selected in ((left, right, relation != ">"), (right, left, relation != "<"))
            if selected
        )

    original = originals(source, target)
    result = [make_candidate(object_id, original, ["keep"], cost_features=[("edit", 0.0)])]
    if not eligible or locked:
        return tuple(result)

    def emit(axioms: Iterable[Any], tag: str, active: Iterable[Any] = (), **roles: str) -> None:
        if tag not in actions:
            return
        values = tuple(axioms)
        active = tuple(active)
        cost = replacement_cost_features(original, values, active_expressions=active)
        result.append(
            make_candidate(
                object_id,
                values,
                [tag],
                active_expressions=active,
                cost_features=cost,
                provenance=roles.items(),
            )
        )

    emit((), "delete")
    if entity_kind != "class":
        return deduplicate_candidates(result)
    if relation == "=":
        for axiom in original:
            emit(
                (axiom,),
                "retain_subsumption",
                subclass=str(axiom.sub_class.iri.value),
                superclass=str(axiom.super_class.iri.value),
            )
    for side, replacement in endpoint_alternatives:
        if side not in {"source", "target"} or not isinstance(replacement, owl.Class):
            raise ValueError("endpoint alternatives must be ('source'|'target', named class)")
        emit(
            originals(
                replacement if side == "source" else source,
                replacement if side == "target" else target,
            ),
            "replace_endpoint",
            endpoint_side=side,
        )
    expressions = tuple(sorted({canonical_expression(e) for e in expressions}, key=_key))
    for axiom in original:
        left, right = axiom.sub_class, axiom.super_class
        retained = tuple(a for a in original if a != axiom)
        roles = {"subclass": str(left.iri.value), "superclass": str(right.iri.value)}
        for expression in expressions:
            specialised = intersection(left, expression)
            inclusion = owl.SubClassOf(specialised, right)
            necessary = owl.SubClassOf(right, expression)
            emit((inclusion,), "specialise_subclass", (specialised,), **roles)
            if retained:
                emit((inclusion, *retained), "specialise_subclass", (specialised,), **roles)
                emit((*retained, necessary), "add_necessary_condition", **roles)
            emit((*original, necessary), "add_necessary_condition", **roles)
            if relation == "=":
                emit(
                    (inclusion, *retained, necessary),
                    "complex_equivalence",
                    (specialised,),
                    **roles,
                )
            # Independent necessary conditions can be combined with specialisation
            # on a directional original without inventing the missing direction.
            emit((inclusion, *retained, necessary), "composite", (specialised,), **roles)
    return deduplicate_candidates(result)


def ontology_candidates(
    revision: RevisionObjectV2,
    *,
    expressions: Iterable[Any] = (),
    fixed_axioms: Iterable[Any] = (),
    enabled_actions: Iterable[str] = ONTOLOGY_ACTIONS,
) -> tuple[ReplacementCandidateV2, ...]:
    """Replace one eligible occurrence, preserving all unedited assertions in its object.

    Generalisation premises are deliberately limited to direct fixed subclass
    assertions. Broader inferred premises require separately verified evidence.
    """
    if revision.kind != "ontology_axiom":
        raise ValueError("ontology candidate generation requires an ontology_axiom object")
    actions = frozenset(enabled_actions) | {"keep"}
    if actions - ONTOLOGY_ACTIONS:
        raise ValueError("unknown ontology action")
    originals = normalise_axioms(revision.original_axioms)
    result = [
        make_candidate(revision.object_id, originals, ["keep"], cost_features=[("edit", 0.0)])
    ]
    if not revision.eligible or revision.locked:
        return tuple(result)
    expressions = tuple(sorted({canonical_expression(e) for e in expressions}, key=_key))
    fixed = normalise_axioms(fixed_axioms)
    generalisations: dict[Any, set[Any]] = {}
    for axiom in fixed:
        if isinstance(axiom, owl.SubClassOf):
            generalisations.setdefault(axiom.sub_class, set()).add(axiom.super_class)

    def emit(
        index: int | None,
        replacements: Iterable[Any],
        tag: str,
        active: Iterable[Any] = (),
        premise: Any = None,
    ) -> None:
        if tag not in actions:
            return
        retained = [] if index is None else [a for i, a in enumerate(originals) if i != index]
        provenance = [("occurrence_id", revision.occurrence_id), ("source", revision.source)]
        if premise is not None:
            provenance.append(("fixed_premise", owl.structural_hexdigest(premise)))
        emitted = [*retained, *replacements]
        active = tuple(active)
        result.append(
            make_candidate(
                revision.object_id,
                emitted,
                [tag],
                active_expressions=active,
                provenance=provenance,
                cost_features=replacement_cost_features(
                    originals,
                    emitted,
                    kind="ontology_axiom",
                    authorship=revision.authorship,
                    active_expressions=active,
                ),
            )
        )

    emit(None, (), "delete")
    for index, axiom in enumerate(originals):
        annotations = axiom.annotations
        if isinstance(axiom, owl.DisjointClasses):
            pairs = list(combinations(sorted(axiom.expressions, key=_key), 2))
            for removed in pairs:
                emit(
                    index,
                    (
                        owl.DisjointClasses(owl.CanonicalSet(pair), annotations)
                        for pair in pairs
                        if pair != removed
                    ),
                    "remove_disjointness",
                )
        if isinstance(axiom, owl.SubClassOf):
            left, right = axiom.sub_class, axiom.super_class
            if (
                isinstance(left, owl.ObjectIntersectionOf)
                and len(left.operands) == 2
                and right == owl.OWL_NOTHING
            ):
                emit(index, (), "remove_disjointness")
            if isinstance(right, owl.ObjectIntersectionOf):
                for removed_conjunct in right.operands:
                    new_right = intersection(*(e for e in right.operands if e != removed_conjunct))
                    emit(
                        index,
                        [owl.SubClassOf(left, new_right, annotations)],
                        "remove_superclass_conjunct",
                    )
            for expression in expressions:
                try:
                    new_left = intersection(left, expression)
                except ValueError:
                    # Existing out-of-grammar OWL remains available to keep/delete
                    # and verification; it is not silently rewritten by a template.
                    continue
                emit(
                    index,
                    [owl.SubClassOf(new_left, right, annotations)],
                    "specialise_ontology_subclass",
                    [new_left],
                )
            for broader in sorted(generalisations.get(right, ()), key=_key):
                premise = next(
                    a
                    for a in fixed
                    if isinstance(a, owl.SubClassOf)
                    and a.sub_class == right
                    and a.super_class == broader
                )
                emit(
                    index,
                    [owl.SubClassOf(left, broader, annotations)],
                    "generalise_superclass",
                    premise=premise,
                )
            if isinstance(right, owl.ObjectSomeValuesFrom):
                for broader in sorted(generalisations.get(right.filler, ()), key=_key):
                    premise = next(
                        a
                        for a in fixed
                        if isinstance(a, owl.SubClassOf)
                        and a.sub_class == right.filler
                        and a.super_class == broader
                    )
                    replacement = owl.ObjectSomeValuesFrom(right.property, broader)
                    emit(
                        index,
                        [owl.SubClassOf(left, replacement, annotations)],
                        "generalise_existential_filler",
                        premise=premise,
                    )
        if isinstance(axiom, (owl.ObjectPropertyDomain, owl.ObjectPropertyRange)):
            is_domain = isinstance(axiom, owl.ObjectPropertyDomain)
            narrow = axiom.domain if isinstance(axiom, owl.ObjectPropertyDomain) else axiom.range
            for broader in sorted(generalisations.get(narrow, ()), key=_key):
                premise = next(
                    a
                    for a in fixed
                    if isinstance(a, owl.SubClassOf)
                    and a.sub_class == narrow
                    and a.super_class == broader
                )
                emit(
                    index,
                    [type(axiom)(axiom.property, broader, annotations)],
                    "generalise_domain" if is_domain else "generalise_range",
                    premise=premise,
                )
    return deduplicate_candidates(result)


def budget_candidates(
    candidates: Sequence[ReplacementCandidateV2],
    cap: int,
    *,
    scores: Mapping[str, float] | None = None,
    enabled_actions: Iterable[str] | None = None,
) -> tuple[ReplacementCandidateV2, ...]:
    """Freeze a ranked pool after retaining all controls and one available family member."""
    import math

    if type(cap) is not int or cap < 0:
        raise ValueError("candidate cap must be a nonnegative integer")
    unique = deduplicate_candidates(candidates)
    if len({c.object_id for c in unique}) > 1:
        raise ValueError("candidate budgeting applies to one revision object")
    scores = scores or {}
    if any(not math.isfinite(v) for v in scores.values()):
        raise ValueError("candidate ranks must be finite")
    ranked = sorted(unique, key=lambda c: (-scores.get(c.candidate_id, 0.0), c.candidate_id))
    mandatory = {
        c.candidate_id
        for c in unique
        if {"keep", "delete", "retain_subsumption", "replace_endpoint"} & set(c.action_tags)
    }
    families = (
        set(enabled_actions)
        if enabled_actions is not None
        else {t for c in unique for t in c.action_tags}
    )
    for family in sorted(families):
        members = [c for c in ranked if family in c.action_tags]
        if members and not any(c.candidate_id in mandatory for c in members):
            mandatory.add(members[0].candidate_id)
    if len(mandatory) > cap:
        raise ValueError(
            f"invalid candidate budget: cap {cap} cannot retain {len(mandatory)} mandatory entries"
        )
    chosen = set(mandatory)
    for candidate in ranked:
        if len(chosen) >= cap:
            break
        chosen.add(candidate.candidate_id)
    return tuple(c for c in unique if c.candidate_id in chosen)
