"""Bounded, observable vocabulary retrieval, independent of the matching runtime.

The index uses asserted signatures and annotation text, typed matcher alternatives,
bounded structural walks, and pre-decision diagnosis supports. Retrieval suggests
symbols; it never treats text or an editable assertion as a logical constraint.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Any

import pyowl_core as owl

from .graph import GraphExplanation, structural_id
from .records import FrozenMapping, RepairInputV2, canonical_hash


@dataclass(frozen=True)
class RetrievalConfig:
    """Finite per-side menus and deterministic work limits, fixed before retrieval."""

    classes_per_side: int = 8
    properties_per_side: int = 4
    endpoints_per_side: int = 2
    neighborhood_hops: int = 2
    neighborhood_axioms: int = 128
    lexical_postings: int = 4096
    evidence_nodes: int = 100000

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in asdict(self).values()):
            raise ValueError("retrieval budgets must be nonnegative integers")
        if self.classes_per_side < 1 or self.evidence_nodes < 1:
            raise ValueError("class and evidence budgets must be positive")


@dataclass(frozen=True)
class ObjectMenus:
    """Typed, side-specific observed symbols for one finite replacement grammar."""

    object_id: str
    source_classes: tuple[Any, ...] = ()
    target_classes: tuple[Any, ...] = ()
    source_properties: tuple[Any, ...] = ()
    target_properties: tuple[Any, ...] = ()
    endpoint_alternatives: tuple[tuple[str, Any], ...] = ()

    @property
    def classes(self) -> tuple[Any, ...]:
        return tuple(sorted(set(self.source_classes + self.target_classes), key=_entity_key))

    @property
    def properties(self) -> tuple[Any, ...]:
        return tuple(sorted(set(self.source_properties + self.target_properties), key=_entity_key))


@dataclass(frozen=True)
class RetrievalResult:
    """Immutable menus, graph descriptions, and auditable retrieval omissions."""

    menus: tuple[ObjectMenus, ...]
    symbols: tuple[Any, ...]
    descriptions: tuple[tuple[str, FrozenMapping], ...]
    explanations: tuple[GraphExplanation, ...]
    provenance: FrozenMapping

    @cached_property
    def menu_index(self) -> FrozenMapping:
        return FrozenMapping({menu.object_id: index for index, menu in enumerate(self.menus)})

    def for_object(self, object_id: str) -> ObjectMenus:
        return self.menus[int(self.menu_index[object_id])]

    def graph_evidence(self, evidence: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> dict:
        """Attach descriptions directly to the retrieved graph entity nodes."""
        result = dict(evidence)
        # Retrieval bookkeeping is provenance, not a learned feature channel.
        result.pop("retrieval", None)
        for node_id, description in self.descriptions:
            previous = result.get(node_id)
            result[node_id] = (
                {"description": description, "observed": previous}
                if previous is not None
                else {"description": description}
            )
        return result

    def capture(self) -> FrozenMapping:
        """Serialize the exact selected menus without carrying a graph/model runtime."""
        menus = []
        for menu in self.menus:
            row: dict[str, Any] = {"object_id": menu.object_id}
            for name in (
                "source_classes",
                "target_classes",
                "source_properties",
                "target_properties",
            ):
                row[name] = [str(entity.iri.value) for entity in getattr(menu, name)]
            row["endpoint_alternatives"] = [
                [side, str(entity.iri.value)] for side, entity in menu.endpoint_alternatives
            ]
            menus.append(row)
        return FrozenMapping(
            {
                "menus": menus,
                "descriptions": dict(self.descriptions),
                "explanations": [
                    {
                        "explanation_id": item.explanation_id,
                        "support_object_ids": item.support_object_ids,
                        "support_axioms": [
                            owl.canonical_bytes(ax).hex() for ax in item.support_axioms
                        ],
                        "witness": (
                            owl.canonical_bytes(item.witness).hex()
                            if item.witness is not None
                            else None
                        ),
                    }
                    for item in self.explanations
                ],
                "provenance": self.provenance,
            }
        )


def _entity_key(entity: Any) -> tuple[str, str]:
    return str(entity.kind.value), str(entity.iri.value)


def _tokens(text: str) -> frozenset[str]:
    # Camel-case splitting makes an unlabelled IRI a useful, explicitly weak cue.
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return frozenset(re.findall(r"[^\W_]+", text.lower(), re.UNICODE))


def _local_name(iri: str) -> str:
    return re.split(r"[/#:]", iri)[-1]


def _input_identity(problem: RepairInputV2, evidence: Mapping[str, Any]) -> str:
    return canonical_hash(
        (
            problem.source_axioms,
            problem.target_axioms,
            problem.fixed_axioms,
            tuple(
                (o.object_id, o.original_axioms, o.source_entity, o.target_entity)
                for o in problem.objects
            ),
            evidence,
        )
    )


def _restore_capture(captured: Mapping[str, Any]) -> RetrievalResult:
    menus = []
    symbols: set[Any] = set()
    for row in captured["menus"]:
        values = {}
        for name in ("source_classes", "target_classes", "source_properties", "target_properties"):
            constructor = owl.Class if name.endswith("classes") else owl.ObjectProperty
            values[name] = tuple(constructor(owl.IRI(iri)) for iri in row[name])
            symbols.update(values[name])
        menus.append(
            ObjectMenus(
                row["object_id"],
                **values,
                endpoint_alternatives=tuple(
                    (side, owl.Class(owl.IRI(iri))) for side, iri in row["endpoint_alternatives"]
                ),
            )
        )
    explanations = tuple(
        GraphExplanation(
            row["explanation_id"],
            tuple(row["support_object_ids"]),
            tuple(owl.decode_canonical(bytes.fromhex(value)) for value in row["support_axioms"]),
            owl.decode_canonical(bytes.fromhex(row["witness"])) if row["witness"] else None,
        )
        for row in captured["explanations"]
    )
    return RetrievalResult(
        tuple(menus),
        tuple(sorted(symbols, key=_entity_key)),
        tuple(
            (key, FrozenMapping(value)) for key, value in sorted(captured["descriptions"].items())
        ),
        explanations,
        FrozenMapping(captured["provenance"]),
    )


def retrieve_vocabulary(
    problem: RepairInputV2,
    *,
    config: RetrievalConfig | None = None,
    explanations: Iterable[GraphExplanation] = (),
) -> RetrievalResult:
    """Retrieve observed classes/properties; no matching, learning, or OWL calls.

    Matching rows are indexed once by endpoint. Structural expansion visits
    axiom hyperedges, avoiding quadratic entity cliques. Lexical postings and
    neighborhoods have explicit work caps, and every truncation is recorded.
    A captured configuration is reused by default so preparation and proposal
    cannot silently disagree on menu budgets.
    """
    from .api import observable_evidence

    explanations = tuple(explanations)
    evidence, excluded = observable_evidence(dict(problem.evidence))
    captured = evidence.pop("retrieval", None)
    if config is None and isinstance(captured, Mapping):
        config = RetrievalConfig(**captured.get("provenance", {}).get("config", {}))
    config = config or RetrievalConfig()
    input_identity = _input_identity(problem, evidence)
    if isinstance(captured, Mapping) and not explanations:
        provenance = captured.get("provenance", {})
        if (
            provenance.get("implementation") == "observable-retrieval/v1"
            and provenance.get("input_identity") == input_identity
            and provenance.get("config") == asdict(config)
        ):
            return _restore_capture(captured)
    omissions: list[dict[str, Any]] = []
    if excluded:
        omissions.append({"channel": "private_evidence", "paths": excluded})
    sides: dict[str, set[Any]] = {"source": set(), "target": set()}
    incident: dict[Any, list[int]] = defaultdict(list)
    axiom_terms: list[tuple[Any, ...]] = []
    annotations: dict[str, list[dict[str, str]]] = defaultdict(list)
    axiom_lookup: dict[str, Any] = {}
    partitioned = set(problem.source_axioms) | set(problem.target_axioms)
    unpartitioned = set(problem.fixed_axioms) - partitioned
    unpartitioned.update(
        axiom
        for obj in problem.objects
        for axiom in obj.original_axioms
        if axiom not in partitioned and obj.source_entity is None and obj.target_entity is None
    )
    unknown_symbols: set[Any] = set()
    known_symbols: set[Any] = set()
    for side, axioms in (
        ("source", problem.source_axioms),
        ("target", problem.target_axioms),
        ("unknown", tuple(unpartitioned)),
    ):
        for axiom in sorted(axioms, key=owl.structural_hexdigest):
            signature_terms = tuple(sorted(owl.signature(axiom), key=_entity_key))
            if side == "unknown":
                unknown_symbols.update(signature_terms)
                for members in sides.values():
                    members.update(signature_terms)
            else:
                known_symbols.update(signature_terms)
                sides[side].update(signature_terms)
            index = len(axiom_terms)
            axiom_terms.append(signature_terms)
            for entity in signature_terms:
                incident[entity].append(index)
            axiom_lookup[canonical_hash(axiom)] = axiom
            axiom_lookup[owl.structural_hexdigest(axiom)] = axiom
            if isinstance(axiom, owl.AnnotationAssertion) and isinstance(axiom.value, owl.Literal):
                subject = axiom.subject
                if isinstance(subject, owl.IRI):
                    annotations[str(subject.value)].append(
                        {
                            "property": str(axiom.property.iri.value),
                            "text": axiom.value.lexical_form,
                        }
                    )
    object_index = {obj.object_id: obj for obj in problem.objects}
    object_seeds = {
        obj.object_id: {entity for axiom in obj.original_axioms for entity in owl.signature(axiom)}
        for obj in problem.objects
    }
    entity_objects: dict[Any, set[str]] = defaultdict(set)
    for obj in problem.objects:
        for entity in object_seeds[obj.object_id]:
            entity_objects[entity].add(obj.object_id)
        for side in sides:
            endpoint = getattr(obj, f"{side}_entity", None)
            if endpoint is not None:
                sides[side].add(endpoint)
        for axiom in obj.original_axioms:
            axiom_lookup[canonical_hash(axiom)] = axiom
            axiom_lookup[owl.structural_hexdigest(axiom)] = axiom
    entities = sides["source"] | sides["target"]
    lookup = {_entity_key(entity): entity for entity in entities}
    text_tokens: dict[Any, frozenset[str]] = {}
    postings: dict[str, list[Any]] = defaultdict(list)
    for entity in sorted(entities, key=_entity_key):
        text = " ".join(row["text"] for row in annotations.get(str(entity.iri.value), ()))
        annotation_tokens = _tokens(text + " " + _local_name(str(entity.iri.value)))
        text_tokens[entity] = annotation_tokens
        for token in annotation_tokens:
            postings[token].append(entity)

    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    diagnostics = list(explanations)
    queue: deque[tuple[str, Any, str | None, str | None]] = deque([("", evidence, None, None)])
    visited = 0
    skipped_evidence_nodes = 0
    while queue and visited < config.evidence_nodes:
        path, value, inherited_source, inherited_target = queue.popleft()
        visited += 1
        if isinstance(value, Mapping):
            left = next(
                (
                    value[k]
                    for k in ("SrcEntity", "Src", "src_iri", "source_iri", "source")
                    if k in value
                ),
                None,
            )
            right = next(
                (
                    value[k]
                    for k in ("TgtEntity", "Tgt", "tgt_iri", "target_iri", "target")
                    if k in value
                ),
                None,
            )
            explicit_endpoint = isinstance(left, str) or isinstance(right, str)
            left = left if isinstance(left, str) else inherited_source
            right = right if isinstance(right, str) else inherited_target
            if explicit_endpoint and isinstance(left, str) and isinstance(right, str):
                row = {**value, "_source": left, "_target": right}
                by_source[left].append(row)
                by_target[right].append(row)
            if "support_object_ids" in value or "support_axiom_hashes" in value:
                support_ids = tuple(str(item) for item in value.get("support_object_ids", ()))
                hashes = tuple(str(item) for item in value.get("support_axiom_hashes", ()))
                if not value.get("available_before_decision", True):
                    omissions.append(
                        {"channel": "diagnosis", "path": path, "reason": "post_decision"}
                    )
                elif any(item not in object_index for item in support_ids) or any(
                    item not in axiom_lookup for item in hashes
                ):
                    omissions.append(
                        {"channel": "diagnosis", "path": path, "reason": "incomplete_support"}
                    )
                else:
                    witness = value.get("witness_iri")
                    diagnostics.append(
                        GraphExplanation(
                            str(
                                value.get(
                                    "explanation_id", canonical_hash((path, support_ids, hashes))
                                )
                            ),
                            support_ids,
                            tuple(axiom_lookup[item] for item in hashes),
                            owl.Class(owl.IRI(witness)) if isinstance(witness, str) else None,
                        )
                    )
            items = sorted(
                (key, child)
                for key, child in value.items()
                if key not in {"omitted", "evidence_omissions"}
            )
            remaining = max(0, config.evidence_nodes - visited - len(queue))
            queue.extend((f"{path}.{key}", child, left, right) for key, child in items[:remaining])
            skipped_evidence_nodes += max(0, len(items) - remaining)
        elif isinstance(value, (tuple, list)):
            remaining = max(0, config.evidence_nodes - visited - len(queue))
            queue.extend(
                (f"{path}[{i}]", child, inherited_source, inherited_target)
                for i, child in enumerate(value[:remaining])
            )
            skipped_evidence_nodes += max(0, len(value) - remaining)
    if queue or skipped_evidence_nodes:
        omissions.append(
            {
                "channel": "evidence_scan",
                "unvisited_branches": len(queue) + skipped_evidence_nodes,
                "limit": config.evidence_nodes,
            }
        )

    diagnostic_terms: dict[str, set[Any]] = defaultdict(set)
    unique_diagnostics: dict[str, GraphExplanation] = {}
    for explanation in diagnostics:
        if not explanation.available_before_decision:
            raise ValueError("post-decision explanations cannot guide retrieval")
        if any(item not in object_index for item in explanation.support_object_ids):
            raise ValueError("diagnosis contains an unknown repair object")
        if any(canonical_hash(axiom) not in axiom_lookup for axiom in explanation.support_axioms):
            raise ValueError("diagnosis support must contain observed asserted axioms")
        previous = unique_diagnostics.get(explanation.explanation_id)
        if previous is not None and previous != explanation:
            raise ValueError("diagnosis IDs must identify unique complete supports")
        unique_diagnostics[explanation.explanation_id] = explanation
        support_terms: set[Any] = set()
        for axiom in explanation.support_axioms:
            support_terms.update(owl.signature(axiom))
        for object_id in explanation.support_object_ids:
            support_terms.update(object_seeds[object_id])
        if explanation.witness is not None:
            support_terms.update(owl.signature(explanation.witness))
        affected = set(explanation.support_object_ids)
        for entity in support_terms:
            affected.update(entity_objects.get(entity, ()))
        for object_id in affected:
            diagnostic_terms[object_id].update(support_terms)

    menus = []
    selected_symbols: set[Any] = set()
    for obj in problem.objects:
        seeds = object_seeds[obj.object_id]
        scores: dict[Any, float] = defaultdict(float)

        def add(entity: Any, score: float) -> None:
            if entity in entities:
                scores[entity] += score

        for entity in seeds:
            add(entity, 20.0)
        for entity in diagnostic_terms[obj.object_id]:
            add(entity, 8.0)
        visited_axioms: set[int] = set()
        frontier = seeds
        seen_terms = set(seeds)
        for hop in range(config.neighborhood_hops):
            next_frontier: set[Any] = set()
            remaining = max(0, config.neighborhood_axioms - len(visited_axioms))
            indexes = []
            last_index = -1
            for index in heapq.merge(*(incident.get(entity, ()) for entity in frontier)):
                if index == last_index or index in visited_axioms:
                    continue
                last_index = index
                indexes.append(index)
                if len(indexes) > remaining:
                    break
            if len(indexes) > remaining:
                omissions.append(
                    {
                        "object_id": obj.object_id,
                        "channel": "neighborhood",
                        "truncated": True,
                        "omitted_axioms_lower_bound": 1,
                    }
                )
            for index in indexes[:remaining]:
                visited_axioms.add(index)
                for entity in axiom_terms[index]:
                    if entity not in seen_terms:
                        add(entity, 2.0 / (hop + 1))
                        next_frontier.add(entity)
            seen_terms.update(next_frontier)
            frontier = next_frontier
            if not frontier or len(visited_axioms) >= config.neighborhood_axioms:
                break
        query = set().union(*(text_tokens.get(entity, frozenset()) for entity in seeds))
        lexical_work = 0
        for token in sorted(query, key=lambda term: (len(postings.get(term, ())), term)):
            matches = postings.get(token, ())
            remaining = max(0, config.lexical_postings - lexical_work)
            if len(matches) > remaining:
                omissions.append(
                    {
                        "object_id": obj.object_id,
                        "channel": "lexical",
                        "token": token,
                        "omitted_postings": len(matches) - remaining,
                    }
                )
            for entity in matches[:remaining]:
                add(entity, 1.0 / math.sqrt(max(1, len(text_tokens[entity]))))
            lexical_work += min(len(matches), remaining)
        for side, endpoint, match_index, replacement_key in (
            ("target", obj.source_entity, by_source, "_target"),
            ("source", obj.target_entity, by_target, "_source"),
        ):
            if endpoint is None:
                continue
            for match_row in match_index.get(str(endpoint.iri.value), ()):
                match_kind = str(
                    match_row.get(
                        "Kind", match_row.get("SrcKind", match_row.get("kind", endpoint.kind.value))
                    )
                )
                target_kind = str(match_row.get("TgtKind", match_kind))
                if match_kind != endpoint.kind.value or target_kind != match_kind:
                    continue
                replacement = lookup.get((match_kind, match_row[replacement_key]))
                if replacement is None or replacement not in sides[side]:
                    omissions.append(
                        {
                            "object_id": obj.object_id,
                            "channel": "matching",
                            "iri": match_row[replacement_key],
                            "reason": "outside_typed_side_signature",
                        }
                    )
                    continue
                score = match_row.get(
                    "Score", match_row.get("score", match_row.get("confidence", 0.0))
                )
                score = float(score) if isinstance(score, (int, float)) else 0.0
                add(replacement, 6.0 + score / (1.0 + abs(score)))
        chosen: dict[str, tuple[Any, ...]] = {}
        alternatives: list[tuple[str, Any]] = []
        for side in sides:
            for kind, suffix, limit in (
                (owl.Class, "classes", config.classes_per_side),
                (owl.ObjectProperty, "properties", config.properties_per_side),
            ):
                ranked = sorted(
                    (
                        entity
                        for entity in scores
                        if entity in sides[side]
                        and isinstance(entity, kind)
                        and entity not in (owl.OWL_THING, owl.OWL_NOTHING)
                    ),
                    key=lambda entity: (-scores[entity], _entity_key(entity)),
                )
                # Mapped endpoints remain available even when other candidates
                # receive many independent observed evidence contributions.
                endpoint = getattr(obj, f"{side}_entity", None)
                if isinstance(endpoint, kind) and endpoint in ranked:
                    ranked.remove(endpoint)
                    ranked.insert(0, endpoint)
                chosen[f"{side}_{suffix}"] = tuple(ranked[:limit])
                if len(ranked) > limit:
                    omissions.append(
                        {
                            "object_id": obj.object_id,
                            "channel": f"{side}_{suffix}",
                            "omitted": [str(entity.iri.value) for entity in ranked[limit:]],
                        }
                    )
                selected_symbols.update(ranked[:limit])
            endpoint = getattr(obj, f"{side}_entity", None)
            if isinstance(endpoint, owl.Class) and obj.eligible and not obj.locked:
                alternatives.extend(
                    (side, entity) for entity in chosen[f"{side}_classes"] if entity != endpoint
                )
                alternatives = [pair for pair in alternatives if pair[0] != side] + [
                    pair for pair in alternatives if pair[0] == side
                ][: config.endpoints_per_side]
        menus.append(
            ObjectMenus(obj.object_id, **chosen, endpoint_alternatives=tuple(alternatives))
        )
    unknown_only = unknown_symbols - known_symbols
    descriptions = []
    for entity in sorted(selected_symbols, key=_entity_key):
        descriptions.append(
            (
                structural_id(entity),
                FrozenMapping(
                    {
                        "iri": str(entity.iri.value),
                        "kind": str(entity.kind.value),
                        "local_name": _local_name(str(entity.iri.value)),
                        "annotations": sorted(
                            annotations.get(str(entity.iri.value), ()),
                            key=lambda row: (row["property"], row["text"]),
                        ),
                        "sides": (
                            ["unknown"]
                            if entity in unknown_only
                            else sorted(side for side in sides if entity in sides[side])
                        ),
                    }
                ),
            )
        )
    provenance = FrozenMapping(
        {
            "implementation": "observable-retrieval/v1",
            "input_identity": input_identity,
            "config": asdict(config),
            "omissions": omissions,
            "examined_evidence_nodes": visited,
            "diagnosis_ids": sorted(unique_diagnostics),
            "unpartitioned_symbols": [
                list(_entity_key(entity)) for entity in sorted(unknown_only, key=_entity_key)
            ],
        }
    )
    return RetrievalResult(
        tuple(menus),
        tuple(sorted(selected_symbols, key=_entity_key)),
        tuple(descriptions),
        tuple(unique_diagnostics[key] for key in sorted(unique_diagnostics)),
        provenance,
    )
