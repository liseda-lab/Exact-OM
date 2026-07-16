"""Reasoning protocol and plugin resolution."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Protocol, cast, runtime_checkable

from exact.ontology.store import OwlOntologySource


@runtime_checkable
class ReasonerProtocol(Protocol):
    def direct_parents(self, iri: str) -> list[str]: ...

    def direct_children(self, iri: str) -> list[str]: ...

    def ancestors(self, iri: str) -> set[str]: ...

    def descendants(self, iri: str) -> set[str]: ...


class AssertedHierarchyReasoner:
    """The built-in reasoner backed by the normalized asserted hierarchy."""

    def __init__(self, store: OwlOntologySource) -> None:
        self.store = store

    def direct_parents(self, iri: str) -> list[str]:
        return self.store.hierarchy.direct_parents(iri)

    def direct_children(self, iri: str) -> list[str]:
        return self.store.hierarchy.direct_children(iri)

    def ancestors(self, iri: str) -> set[str]:
        return self.store.hierarchy.ancestors(iri)

    def descendants(self, iri: str) -> set[str]:
        return self.store.hierarchy.descendants(iri)


def _reasoner_entry_points():
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return list(discovered.select(group="exact.reasoners"))
    return list(discovered.get("exact.reasoners", ()))  # pragma: no cover - Python <3.10.


def load_reasoner(name: str, store: OwlOntologySource) -> ReasonerProtocol:
    """Load the asserted reasoner or a factory from ``exact.reasoners``."""

    if name == "asserted":
        return AssertedHierarchyReasoner(store)

    plugins = {plugin.name: plugin for plugin in _reasoner_entry_points()}
    if name not in plugins:
        installed = ", ".join(sorted(plugins)) or "none"
        raise ValueError(
            f"Unknown reasoner {name!r}. Built in: asserted. Installed plugins: {installed}."
        )

    factory = plugins[name].load()
    if hasattr(factory, "create"):
        reasoner = factory.create(store)
    elif callable(factory):
        reasoner = factory(store)
    else:
        reasoner = factory
    required = ("direct_parents", "direct_children", "ancestors", "descendants")
    missing = [method for method in required if not callable(getattr(reasoner, method, None))]
    if missing:
        raise TypeError(f"Reasoner plugin {name!r} does not implement: {', '.join(missing)}")
    return cast(ReasonerProtocol, reasoner)
