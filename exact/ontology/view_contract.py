"""Side-effect-free validation of the public :mod:`pyowl_core` view protocol."""

from __future__ import annotations

import inspect
from typing import TypeGuard, cast

import pyowl_core
from pyowl_core import OntologyView

_MISSING = object()
_VIEW_ATTRIBUTES = (
    "capabilities",
    "iter_axioms",
    "iter_extensions",
    "contains",
    "ontology_annotations",
    "signature",
    "view",
    "origin_index",
    "is_complete",
    "structural_fingerprint",
    "logical_fingerprint",
    "signature_fingerprint",
    "report",
)


def is_ontology_view(value: object) -> TypeGuard[OntologyView]:
    """Recognize the frozen public protocol without evaluating lazy properties.

    ``runtime_checkable`` protocols accessed descriptors on Python 3.10 and 3.11.
    Exact therefore mirrors the public ``OntologyView`` member ledger with static
    inspection before asking core to validate compatibility.
    """

    return all(
        inspect.getattr_static(value, name, _MISSING) is not _MISSING for name in _VIEW_ATTRIBUTES
    )


def retain_ontology_view(value: object) -> OntologyView:
    """Validate a view through core and require exact identity preservation."""

    if not is_ontology_view(value):
        raise TypeError("snapshot must implement pyowl_core.OntologyView")
    retained = pyowl_core.coerce_snapshot(cast(pyowl_core.OntologyInput, value))
    if retained is not value:  # pragma: no cover - core's view contract requires identity.
        raise RuntimeError("pyowl_core.coerce_snapshot did not preserve OntologyView identity")
    return retained


__all__ = ["is_ontology_view", "retain_ontology_view"]
