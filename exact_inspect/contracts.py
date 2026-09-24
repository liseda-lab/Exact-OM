"""Versioned, CPU-only wire primitives and semantic identities for prepared inspection."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "exact-explain/1.0"
EntityKind = Literal["class", "object_property", "data_property", "individual"]
Availability = Literal[
    "available",
    "absent_in_scope",
    "not_exported",
    "unavailable_source",
    "unresolved_import",
    "unsupported",
    "filtered",
    "partial",
    "failed",
    "not_requested",
    "not_run",
]


def canonical_json(value: Any) -> bytes:
    """Encode semantic data without platform-dependent whitespace or nonfinite values."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    """Hash canonical semantic content; locators and display labels are caller-excluded."""
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path: Path) -> str:
    """Hash every byte, rejecting partial reads through the consuming size binding."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


class WireModel(BaseModel):
    """Reject unknown input fields and accidental mutation of public identities."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class EntityRef(WireModel):
    """An ontology-scoped typed entity; punning and duplicate labels remain distinct."""

    ontology_version_id: str = Field(min_length=1)
    iri: str = Field(min_length=1)
    kind: EntityKind


class Scope(WireModel):
    """The complete semantic scope to which a collection cursor is bound."""

    ontology_version_id: str
    context_revision: str
    basis: str
    visibility_policy_hash: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    filter_id: str


T = TypeVar("T")


class Page(WireModel, Generic[T]):
    """A bounded collection with explicit missingness and policy-sensitive counts."""

    items: list[T]
    returned_count: int = Field(ge=0)
    total_count: int | None = Field(default=None, ge=0)
    next_cursor: str | None = None
    truncated: bool = False
    scope: Scope
    status: Availability = "available"
    reason: str | None = None

    @model_validator(mode="after")
    def check_counts(self) -> Page[T]:
        """Prevent contradictory completeness and count claims."""
        if self.returned_count != len(self.items):
            raise ValueError("returned_count does not match items")
        if self.total_count is not None and self.total_count < self.returned_count:
            raise ValueError("total_count is smaller than returned_count")
        if self.next_cursor and not self.truncated:
            raise ValueError("A continuation cursor requires truncated=true")
        return self


class ErrorEnvelope(WireModel):
    """Stable safe errors; internal paths and raw provider errors are excluded."""

    code: str
    message: str
    retryable: bool = False
    context_id: str | None = None


class DomainError(ValueError):
    """A public contract failure suitable for HTTP and CLI translation."""

    def __init__(self, code: str, message: str, status_code: int = 422, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.envelope = ErrorEnvelope(code=code, message=message, retryable=retryable)


class VisibilityPolicy(WireModel):
    """Immutable allowlist applied before search, counting, packet construction and caching."""

    policy_id: str = "exploration-v1"
    ontology_ids: tuple[str, ...] = ()
    categories: tuple[str, ...] = (
        "labels",
        "definitions",
        "definition_citations",
        "term_metadata",
        "synonyms",
        "comments",
        "hierarchy",
        "parents",
        "children",
        "restrictions",
        "usage",
        "axioms",
        "types",
        "assertions",
        "domains",
        "ranges",
        "characteristics",
        "equivalences",
    )
    allow_mapping_xrefs: bool = False

    @property
    def policy_hash(self) -> str:
        """Bind every derived artifact to the full effective allowlist."""
        return canonical_hash(self)

    def allows_ontology(self, ontology_id: str) -> bool:
        """Check the permitted ontology universe independently from candidate membership."""
        return not self.ontology_ids or ontology_id in self.ontology_ids

    def allows_fact(self, fact: dict[str, Any], category: str | None = None) -> bool:
        """Check explicit category metadata; unknown categories fail closed."""
        subject = fact.get("subject", {})
        if isinstance(subject, BaseModel):
            subject = subject.model_dump()
        if not self.allows_ontology(subject.get("ontology_version_id", "")):
            return False
        category = category or fact.get("category")
        if category in {"mappings", "xrefs", "mapping_xrefs"}:
            return self.allow_mapping_xrefs and category in self.categories
        return category in self.categories


def encode_cursor(scope: Any, last_key: Any) -> str:
    """Create a keyset locator bound to snapshot, query, order and visibility policy."""
    return (
        base64.urlsafe_b64encode(
            canonical_json(
                {
                    "scope": canonical_hash(scope),
                    "last": last_key,
                }
            )
        )
        .decode()
        .rstrip("=")
    )


def decode_cursor(cursor: str | None, scope: Any) -> Any:
    """Reject malformed or stale cursors instead of mixing collection revisions."""
    if cursor is None:
        return None
    if len(cursor) > 8192:
        raise DomainError("invalid_cursor", "Invalid cursor")
    try:
        value = json.loads(
            base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        )
        if not isinstance(value, dict) or set(value) != {"scope", "last"}:
            raise ValueError("shape")
    except (ValueError, UnicodeError) as exc:
        raise DomainError("invalid_cursor", "Invalid cursor") from exc
    last = value["last"]
    if not isinstance(last, (str, int, float, list)) or (
        isinstance(last, list) and any(not isinstance(v, (str, int, float)) for v in last)
    ):
        raise DomainError("invalid_cursor", "Invalid cursor key")
    if value["scope"] != canonical_hash(scope):
        raise DomainError(
            "stale_cursor", "Cursor belongs to another query or package revision", 409
        )
    return value["last"]


class Score(WireModel):
    """A recorded value with stage semantics, never an implied calibrated probability."""

    name: str
    stage: str
    value: float
    meaning: str
    range: dict[str, float] | None = None
    calibration_status: Literal["not_established", "validated", "not_applicable"] = (
        "not_established"
    )
    calibration_artifact: str | None = None

    @model_validator(mode="after")
    def check_calibration(self) -> Score:
        """Require evidence for calibration and finite numeric values."""
        if not math.isfinite(self.value):
            raise ValueError("Score must be finite")
        if self.calibration_status == "validated" and not self.calibration_artifact:
            raise ValueError("Validated calibration requires an artifact")
        return self
