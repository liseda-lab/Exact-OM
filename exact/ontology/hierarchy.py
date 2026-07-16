"""Fast asserted-hierarchy indexes with equivalence-cycle normalization."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

OWL_THING = "http://www.w3.org/2002/07/owl#Thing"
OWL_NOTHING = "http://www.w3.org/2002/07/owl#Nothing"


class HierarchyIndex:
    """An immutable, integer-indexed child-to-parent directed graph.

    Strongly connected components are collapsed.  Queries made for any member
    therefore see the same external parents and children, while equivalent peers
    and self edges are not reported.
    """

    def __init__(
        self,
        entities: Iterable[str],
        parent_edges: Iterable[tuple[str, str]],
        *,
        filter_owl_bounds: bool = True,
    ) -> None:
        excluded = {OWL_THING, OWL_NOTHING} if filter_owl_bounds else set()
        normalized_edges = {
            (str(child), str(parent))
            for child, parent in parent_edges
            if child != parent and child not in excluded and parent not in excluded
        }
        names = {str(entity) for entity in entities if str(entity) not in excluded}
        for child, parent in normalized_edges:
            names.add(child)
            names.add(parent)

        self._iris = tuple(sorted(names))
        self._ids = {iri: index for index, iri in enumerate(self._iris)}
        count = len(self._iris)
        parents: list[set[int]] = [set() for _ in range(count)]
        children: list[set[int]] = [set() for _ in range(count)]
        for child, parent in normalized_edges:
            child_id, parent_id = self._ids[child], self._ids[parent]
            parents[child_id].add(parent_id)
            children[parent_id].add(child_id)

        components, component_for = self._strong_components(parents, children)
        self._components = tuple(tuple(sorted(component)) for component in components)
        self._component_for = tuple(component_for)

        component_parents: list[set[int]] = [set() for _ in self._components]
        component_children: list[set[int]] = [set() for _ in self._components]
        for child_id, node_parents in enumerate(parents):
            child_component = component_for[child_id]
            for parent_id in node_parents:
                parent_component = component_for[parent_id]
                if child_component == parent_component:
                    continue
                component_parents[child_component].add(parent_component)
                component_children[parent_component].add(child_component)
        self._parents = tuple(tuple(sorted(values)) for values in component_parents)
        self._children = tuple(tuple(sorted(values)) for values in component_children)

    @staticmethod
    def _strong_components(
        parents: list[set[int]], children: list[set[int]]
    ) -> tuple[list[list[int]], list[int]]:
        """Iterative Kosaraju traversal (safe for SNOMED-depth graphs)."""

        count = len(parents)
        visited = bytearray(count)
        order: list[int] = []
        for root in range(count):
            if visited[root]:
                continue
            visited[root] = 1
            stack: list[tuple[int, bool]] = [(root, False)]
            while stack:
                node, expanded = stack.pop()
                if expanded:
                    order.append(node)
                    continue
                stack.append((node, True))
                for adjacent in parents[node]:
                    if not visited[adjacent]:
                        visited[adjacent] = 1
                        stack.append((adjacent, False))

        component_for = [-1] * count
        components: list[list[int]] = []
        for root in reversed(order):
            if component_for[root] >= 0:
                continue
            component_id = len(components)
            component: list[int] = []
            component_for[root] = component_id
            component_stack = [root]
            while component_stack:
                node = component_stack.pop()
                component.append(node)
                for adjacent in children[node]:
                    if component_for[adjacent] < 0:
                        component_for[adjacent] = component_id
                        component_stack.append(adjacent)
            components.append(component)
        return components, component_for

    def _component(self, iri: str) -> int | None:
        entity_id = self._ids.get(str(iri))
        return None if entity_id is None else self._component_for[entity_id]

    def _expand_components(self, component_ids: Iterable[int]) -> list[str]:
        return sorted(
            self._iris[entity_id]
            for component_id in component_ids
            for entity_id in self._components[component_id]
        )

    def direct_parents(self, iri: str) -> list[str]:
        component = self._component(iri)
        return [] if component is None else self._expand_components(self._parents[component])

    def direct_children(self, iri: str) -> list[str]:
        component = self._component(iri)
        return [] if component is None else self._expand_components(self._children[component])

    @lru_cache(maxsize=None)
    def _component_ancestors(self, component: int) -> frozenset[int]:
        seen: set[int] = set()
        stack = list(self._parents[component])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self._parents[current])
        return frozenset(seen)

    @lru_cache(maxsize=None)
    def _component_descendants(self, component: int) -> frozenset[int]:
        seen: set[int] = set()
        stack = list(self._children[component])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self._children[current])
        return frozenset(seen)

    def ancestors(self, iri: str) -> set[str]:
        component = self._component(iri)
        if component is None:
            return set()
        return set(self._expand_components(self._component_ancestors(component)))

    def descendants(self, iri: str) -> set[str]:
        component = self._component(iri)
        if component is None:
            return set()
        return set(self._expand_components(self._component_descendants(component)))

    def equivalent_entities(self, iri: str) -> frozenset[str]:
        component = self._component(iri)
        if component is None:
            return frozenset()
        entity_id = self._ids[str(iri)]
        return frozenset(
            self._iris[item] for item in self._components[component] if item != entity_id
        )
