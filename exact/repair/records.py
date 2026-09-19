"""Immutable, versioned records for the opt-in Exact-Repair research pipeline."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from bisect import bisect_left
from collections.abc import Iterator, Mapping
from decimal import ROUND_HALF_EVEN, Decimal
from functools import cached_property
from typing import Any, ClassVar

SCHEMA = "exact-repair/records/v2"


class FrozenMapping(Mapping[str, Any]):
    """Pickleable immutable storage for nested matcher evidence."""

    __slots__ = ("_items",)
    _items: tuple[tuple[str, Any], ...]

    def __init__(self, values: Mapping[str, Any]) -> None:
        if any(not isinstance(key, str) for key in values):
            raise TypeError("evidence mapping keys must be strings")
        object.__setattr__(
            self, "_items", tuple(sorted((key, _freeze(value)) for key, value in values.items()))
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("frozen evidence cannot be mutated")

    def __getitem__(self, key: str) -> Any:
        position = bisect_left(self._items, key, key=lambda item: item[0])
        if position < len(self._items) and self._items[position][0] == key:
            return self._items[position][1]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return type(self), (dict(self._items),)


def _freeze(value: Any) -> Any:
    if (
        dataclasses.is_dataclass(value)
        and not isinstance(value, Record)
        and not type(value).__module__.startswith("pyowl_core")
    ):
        raise TypeError("external evidence dataclasses must be converted to plain mappings")
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _encode(value: Any) -> Any:
    """Encode shared OWL values using the core's authoritative canonical format."""
    if type(value).__module__.startswith("pyowl_core"):
        from pyowl_core import canonical_bytes

        return {"$owl": canonical_bytes(value).hex()}
    if dataclasses.is_dataclass(value):
        return {
            "$record": type(value).__name__,
            **{f.name: _encode(getattr(value, f.name)) for f in dataclasses.fields(value)},
        }
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_encode(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_encode(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("record numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported record value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return deterministic JSON, rejecting nonfinite evidence and coefficients."""
    return json.dumps(_encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    """Hash semantic record content independently of Python object identity."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class Record:
    """Canonical v2 serialization shared by all repair records."""

    schema_version: ClassVar[str] = SCHEMA

    def __post_init__(self) -> None:
        """Freeze supplied containers so hashing and worker inputs cannot drift."""
        if not dataclasses.is_dataclass(self):
            raise TypeError("repair records must be dataclasses")
        for field in dataclasses.fields(self):
            object.__setattr__(self, field.name, _freeze(getattr(self, field.name)))

    @cached_property
    def content_hash(self) -> str:
        """Return this record's content identity."""
        return canonical_hash((self.schema_version, self))

    def to_dict(self) -> dict[str, Any]:
        """Return an integrity-protected JSON-compatible envelope."""
        return {"schema": self.schema_version, "hash": self.content_hash, "record": _encode(self)}


@dataclasses.dataclass(frozen=True)
class ReplacementCandidateV2(Record):
    """A complete replacement, including conditional expression obligations."""

    object_id: str
    candidate_id: str
    axioms: tuple[Any, ...]
    action_tags: tuple[str, ...] = ("keep",)
    active_expressions: tuple[Any, ...] = ()
    cost_features: tuple[tuple[str, float], ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.object_id or not self.candidate_id:
            raise ValueError("candidate and object IDs must be nonempty")
        if any(not math.isfinite(v) or v < 0 for _, v in self.cost_features):
            raise ValueError("structural costs must be finite and nonnegative")


@dataclasses.dataclass(frozen=True)
class RevisionObjectV2(Record):
    """One mapping or explicitly selected ontology axiom occurrence."""

    object_id: str
    kind: str
    original_axioms: tuple[Any, ...]
    candidates: tuple[ReplacementCandidateV2, ...]
    eligible: bool = True
    locked: bool = False
    occurrence_id: str = ""
    source: str = ""
    authorship: str = "unknown"
    source_entity: Any | None = None
    target_entity: Any | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind not in {"mapping", "ontology_axiom"}:
            raise ValueError("unknown revision object kind")
        if self.authorship not in {"human", "generated", "unknown"}:
            raise ValueError("unknown authorship")
        if self.kind == "ontology_axiom" and not self.occurrence_id:
            raise ValueError("ontology edits require an exact occurrence ID")
        if not self.candidates:
            raise ValueError("each object needs at least one candidate")
        ids = [c.candidate_id for c in self.candidates]
        if len(ids) != len(set(ids)) or any(c.object_id != self.object_id for c in self.candidates):
            raise ValueError("candidate IDs must be unique and belong to their object")
        original = frozenset(self.original_axioms)
        if (self.locked or not self.eligible) and any(
            frozenset(c.axioms) != original or c.active_expressions for c in self.candidates
        ):
            raise ValueError("ineligible or locked objects allow only unchanged replacements")


@dataclasses.dataclass(frozen=True)
class PolicyV2(Record):
    """Frozen full-signature policy; exceptions carry source-only proof identities."""

    monitored_classes: tuple[Any, ...] = ()
    required: tuple[Any, ...] = ()
    prohibited: tuple[Any, ...] = ()
    exceptions: tuple[Any, ...] = ()
    exception_evidence: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.exceptions and len(self.exception_evidence) != len(self.exceptions):
            raise ValueError("every source exception requires frozen proof evidence")
        if not set(self.exceptions) <= set(self.monitored_classes):
            raise ValueError("exceptions must belong to the monitored signature")


@dataclasses.dataclass(frozen=True)
class BudgetsV2(Record):
    """Finite supervisor and solver/verification-call budgets, in seconds."""

    total_seconds: float = 60.0
    solver_seconds: float = 10.0
    verification_seconds: float = 10.0
    max_solves: int = 100
    max_checks: int = 100
    retries: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        for value in (self.total_seconds, self.solver_seconds, self.verification_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("time budgets must be finite and positive")
        for value in (self.max_solves, self.max_checks, self.retries):
            if type(value) is not int or value < 0:
                raise ValueError("call budgets must be nonnegative integers")


@dataclasses.dataclass(frozen=True)
class RepairInputV2(Record):
    """Matcher-independent asserted input with deployment-only soft evidence."""

    fixed_axioms: tuple[Any, ...]
    objects: tuple[RevisionObjectV2, ...]
    policy: PolicyV2
    source_identity: str = ""
    target_identity: str = ""
    matcher_identity: str = "external"
    evidence: tuple[tuple[str, Any], ...] = ()
    source_axioms: tuple[Any, ...] = ()
    target_axioms: tuple[Any, ...] = ()
    budgets: BudgetsV2 = dataclasses.field(default_factory=BudgetsV2)
    candidate_coverage: str = "bounded_enumerated"
    source_documents: tuple[tuple[str, str, str], ...] = ()
    target_documents: tuple[tuple[str, str, str], ...] = ()
    model_status: str = "untrained; explicit frozen objective"
    graph_identity: str = ""
    proposal_provenance: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        ids = [o.object_id for o in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("revision object IDs must be unique")
        canonical_json(self.evidence)


@dataclasses.dataclass(frozen=True)
class ObjectiveV2(Record):
    """Frozen integer objective; learned benefit and preference provenance are separate."""

    unary: tuple[tuple[int, ...], ...]
    pairs: tuple[tuple[int, int, int, int, int], ...] = ()
    scale: int = 1000
    benefit: tuple[tuple[float, ...], ...] = ()
    costs: tuple[tuple[float, ...], ...] = ()
    profile: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.scale) is not int or self.scale <= 0:
            raise ValueError("quantisation scale must be a positive integer")
        if any(not row or any(type(w) is not int for w in row) for row in self.unary):
            raise ValueError("utilities must be nonempty rows of integers")
        seen = set()
        for i, a, j, b, w in self.pairs:
            if any(type(v) is not int for v in (i, a, j, b, w)) or not (
                0 <= i < j < len(self.unary)
                and 0 <= a < len(self.unary[i])
                and 0 <= b < len(self.unary[j])
            ):
                raise ValueError("pair factors require ordered distinct objects and valid indices")
            if (i, a, j, b) in seen:
                raise ValueError("duplicate pair coefficient")
            seen.add((i, a, j, b))
        if any(v < 0 or not math.isfinite(v) for _, v in self.profile):
            raise ValueError("preference costs must be finite and nonnegative")

    def score(self, assignment: tuple[int, ...]) -> int:
        """Evaluate the exported integer objective exactly."""
        if len(assignment) != len(self.unary) or any(
            type(a) is not int or not 0 <= a < len(row) for row, a in zip(self.unary, assignment)
        ):
            raise ValueError("invalid assignment")
        return sum(row[a] for row, a in zip(self.unary, assignment)) + sum(
            w for i, a, j, b, w in self.pairs if assignment[i] == a and assignment[j] == b
        )

    @property
    def upper_cap(self) -> int:
        """Return a valid conservative upper bound without invoking a solver."""
        return sum(max(row) for row in self.unary) + sum(max(0, p[4]) for p in self.pairs)


def quantize(value: float | Decimal, scale: int) -> int:
    """Round a combined coefficient once, using decimal half-even rounding."""
    number = Decimal(str(value))
    if not number.is_finite() or type(scale) is not int or scale <= 0:
        raise ValueError("finite coefficient and positive integer scale required")
    return int((number * scale).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def make_objective(
    objects: tuple[RevisionObjectV2, ...],
    benefits: tuple[tuple[float, ...], ...] | None = None,
    *,
    profile: tuple[tuple[str, float], ...] = (),
    pairs: tuple[tuple[int, int, int, int, float], ...] = (),
    scale: int = 1000,
) -> ObjectiveV2:
    """Subtract explicit edit preferences once, then freeze and quantise."""
    costs_by_name = dict(profile)
    if len(costs_by_name) != len(profile) or any(
        not math.isfinite(v) or v < 0 for v in costs_by_name.values()
    ):
        raise ValueError("cost profile requires unique nonnegative finite weights")
    benefits = (
        benefits
        if benefits is not None
        else tuple(tuple(0.0 for _ in obj.candidates) for obj in objects)
    )
    if len(benefits) != len(objects) or any(
        len(row) != len(obj.candidates) for row, obj in zip(benefits, objects)
    ):
        raise ValueError("benefit shape does not match the candidate inventory")
    costs = tuple(
        tuple(sum(costs_by_name.get(k, 0) * v for k, v in c.cost_features) for c in obj.candidates)
        for obj in objects
    )
    unary = tuple(
        tuple(quantize(Decimal(str(b)) - Decimal(str(c)), scale) for b, c in zip(br, cr))
        for br, cr in zip(benefits, costs)
    )
    return ObjectiveV2(
        unary,
        tuple((i, a, j, b, quantize(w, scale)) for i, a, j, b, w in pairs),
        scale,
        benefits,
        costs,
        profile,
    )


@dataclasses.dataclass(frozen=True)
class ObligationV2(Record):
    """One policy query's verdict, support and completeness."""

    name: str
    verdict: str
    complete: bool
    detail: str = ""


@dataclasses.dataclass(frozen=True)
class VerificationReportV2(Record):
    """Evidence bound to the exact assignment, asserted theory and policy."""

    assignment_hash: str
    theory_hash: str
    policy_hash: str
    verdict: str
    scope: str
    obligations: tuple[ObligationV2, ...]
    backend: str = ""
    detail: str = ""
    support: tuple[tuple[str, Any], ...] = ()

    @property
    def authorizes(self) -> bool:
        """Only complete supported positive evidence authorises an incumbent."""
        return (
            self.verdict == "VERIFIED_FEASIBLE"
            and self.scope in {"full_owl", "complete_supported_fragment"}
            and bool(self.obligations)
            and all(q.complete and q.verdict == "pass" for q in self.obligations)
        )


@dataclasses.dataclass(frozen=True)
class PendingAssignmentV2(Record):
    """Unknown candidates retain their objective contribution to the bound."""

    assignment: tuple[int, ...]
    value: int
    report: VerificationReportV2
    attempts: int = 1


@dataclasses.dataclass(frozen=True)
class RepairResultV2(Record):
    """Factored safety, search and coverage outcome with replay evidence."""

    input_hash: str
    objective_hash: str
    logical_status: str
    search_status: str
    verification_scope: str
    candidate_coverage: str
    assignment: tuple[int, ...] | None
    selected: tuple[ReplacementCandidateV2, ...]
    alignment: tuple[Any, ...] | None
    ontology_patch: tuple[tuple[str, tuple[Any, ...], tuple[Any, ...]], ...]
    lower_bound: int | None
    upper_bound: int | None
    verification: VerificationReportV2 | None
    pending: tuple[PendingAssignmentV2, ...] = ()
    exclusions: tuple[tuple[tuple[int, ...], VerificationReportV2], ...] = ()
    failures: tuple[str, ...] = ()
    solves: int = 0
    checks: int = 0
    baseline: tuple[tuple[str, VerificationReportV2], ...] = ()
    model_status: str = "untrained; explicit frozen objective"
    elapsed_seconds: float = 0.0
    stage_seconds: tuple[tuple[str, float], ...] = ()

    @property
    def gap(self) -> int | None:
        """Return the finite-pool integer gap only when an incumbent exists."""
        if self.lower_bound is None or self.upper_bound is None:
            return None
        return self.upper_bound - self.lower_bound


def read_record(payload: dict[str, Any]) -> Record:
    """Read v2 only; reject legacy records and corrupted artifacts explicitly."""
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported repair record schema; explicit v2 migration required")
    registry = {cls.__name__: cls for cls in Record.__subclasses__()}

    def decode(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(decode(v) for v in value)
        if isinstance(value, dict):
            if "$owl" in value:
                from pyowl_core import decode_canonical

                return decode_canonical(bytes.fromhex(value["$owl"]))
            if "$record" in value:
                name = value["$record"]
                if name not in registry:
                    raise ValueError("unknown repair record")
                return registry[name](**{k: decode(v) for k, v in value.items() if k != "$record"})
            return {k: decode(v) for k, v in value.items()}
        return value

    result = decode(payload["record"])
    if not isinstance(result, Record) or result.content_hash != payload.get("hash"):
        raise ValueError("repair artifact content hash mismatch")
    return result
