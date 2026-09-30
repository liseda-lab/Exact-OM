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
SCHEMA_V3 = "exact-repair/records/v3"


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
    memory_mb: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.memory_mb is not None and (
            not math.isfinite(self.memory_mb) or self.memory_mb <= 0
        ):
            raise ValueError("memory_mb must be positive and finite")
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


def candidate_cost(
    obj: RevisionObjectV2,
    candidate: ReplacementCandidateV2,
    profile: tuple[tuple[str, float], ...],
) -> float:
    """Apply declared preferences to structural features, including XR-2 public names.

    An alias is a name for the same coefficient, not a second cost. Supplying both
    spellings is rejected, preventing accidental double subtraction.
    """
    aliases = {"mapping_deletion": "delete", "human_ontology_edit": "human_authored_ontology_edit"}
    weights = {}
    for name, weight in profile:
        key = aliases.get(name, name)
        if key in weights or not math.isfinite(weight) or weight < 0:
            raise ValueError("cost profile requires unique nonnegative finite weights")
        weights[key] = weight
    features: dict[str, float] = {}
    for name, value in candidate.cost_features:
        if obj.kind != "mapping" and name == "mapping_deletion":
            continue
        key = aliases.get(name, name)
        if key in features and features[key] != value:
            raise ValueError(f"conflicting structural cost aliases for {key}")
        features[key] = value
    # The public mapping-deletion preference does not also charge ontology edits.
    if obj.kind != "mapping" and any(name == "mapping_deletion" for name, _ in profile):
        weights.pop("delete", None)
    return sum(weight * features.get(name, 0.0) for name, weight in weights.items())


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
    costs = tuple(tuple(candidate_cost(obj, c, profile) for c in obj.candidates) for obj in objects)
    unary = tuple(
        tuple(quantize(Decimal(str(b)) - Decimal(str(c)), scale) for b, c in zip(br, cr))
        for br, cr in zip(benefits, costs)
    )
    record_type = (
        ObjectiveV3 if any(isinstance(obj, RevisionObjectV3) for obj in objects) else ObjectiveV2
    )
    extra = (
        {"pool_hash": canonical_hash(objects), "raw_pairs": pairs}
        if record_type is ObjectiveV3
        else {}
    )
    return record_type(
        unary,
        tuple((i, a, j, b, quantize(w, scale)) for i, a, j, b, w in pairs),
        scale,
        benefits,
        costs,
        profile,
        **extra,
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
    ontology_patch: tuple[tuple[str, tuple[Any, ...], tuple[Any, ...]], ...] | None
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
    first_verified_seconds: float | None = None

    @property
    def gap(self) -> int | None:
        """Return the finite-pool integer gap only when an incumbent exists."""
        if self.lower_bound is None or self.upper_bound is None:
            return None
        return self.upper_bound - self.lower_bound


def read_record(payload: dict[str, Any]) -> Record:
    """Read versioned records without silently promoting historical evidence."""
    if payload.get("schema") not in {SCHEMA, SCHEMA_V3}:
        raise ValueError("unsupported repair record schema; explicit migration required")
    if set(payload) != {"schema", "hash", "record"}:
        raise ValueError("invalid repair record envelope")
    registry = {}
    pending = list(Record.__subclasses__())
    while pending:
        cls = pending.pop()
        registry[cls.__name__] = cls
        pending.extend(cls.__subclasses__())

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
    if (
        not isinstance(result, Record)
        or result.schema_version != payload["schema"]
        or result.content_hash != payload.get("hash")
    ):
        raise ValueError("repair artifact content hash mismatch")
    return result


def public_classes(problem: RepairInputV2) -> tuple[Any, ...]:
    """Full public vocabulary, including deleted originals and unselected alternatives."""
    import pyowl_core as owl

    nodes = [
        *problem.fixed_axioms,
        *problem.source_axioms,
        *problem.target_axioms,
        *problem.policy.required,
        *problem.policy.prohibited,
    ]
    for obj in problem.objects:
        nodes.extend(obj.original_axioms)
        nodes.extend(e for e in (obj.source_entity, obj.target_entity) if e is not None)
        for candidate in obj.candidates:
            nodes.extend(candidate.axioms)
            nodes.extend(candidate.active_expressions)
    return tuple(
        sorted(
            {
                entity
                for node in nodes
                for entity in owl.signature(node)
                if isinstance(entity, owl.Class) and entity != owl.OWL_NOTHING
            },
            key=canonical_hash,
        )
    )


def freeze_public_policy(problem: RepairInputV2) -> PolicyV2:
    """Expand the monitored signature when a new inventory starts an epoch."""
    import pyowl_core as owl

    classes = {
        owl.Class(owl.IRI(c)) if isinstance(c, str) else c for c in problem.policy.monitored_classes
    }
    return dataclasses.replace(
        problem.policy,
        exceptions=tuple(
            owl.Class(owl.IRI(c)) if isinstance(c, str) else c for c in problem.policy.exceptions
        ),
        monitored_classes=tuple(sorted(classes | set(public_classes(problem)), key=canonical_hash)),
    )


@dataclasses.dataclass(frozen=True)
class ReplacementCandidateV3(ReplacementCandidateV2):
    """V3 complete bundle; semantic identity remains independent of neural scores."""

    schema_version: ClassVar[str] = SCHEMA_V3
    generation_identity: str = "elementary"

    def __post_init__(self) -> None:
        super().__post_init__()
        import pyowl_core as owl

        if any(not isinstance(a, owl.AxiomNode) for a in self.axioms):
            raise TypeError("replacement axioms must use the shared OWL representation")
        if any(not isinstance(e, owl.ClassExpression) for e in self.active_expressions):
            raise TypeError("active obligations require class expressions")


@dataclasses.dataclass(frozen=True)
class RevisionObjectV3(RevisionObjectV2):
    """An occurrence and its explicitly permitted action families."""

    schema_version: ClassVar[str] = SCHEMA_V3
    allowed_families: tuple[str, ...] = ()
    occurrence_origins: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class PolicyV3(PolicyV2):
    """Full public coherence with explicit immutable exception proof references."""

    schema_version: ClassVar[str] = SCHEMA_V3
    exception_proof_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        import pyowl_core as owl

        if any(not isinstance(c, owl.Class) for c in self.monitored_classes):
            raise TypeError("v3 monitored public classes require typed shared OWL values")
        if any(not isinstance(q, owl.AxiomNode) for q in (*self.required, *self.prohibited)):
            raise TypeError("policy entailment queries require shared OWL axioms")


@dataclasses.dataclass(frozen=True)
class RepairInputV3(RepairInputV2):
    """New execution input; validates public vocabulary on every construction/read."""

    schema_version: ClassVar[str] = SCHEMA_V3
    generation_universe: str = "elementary-plus-bounded-complex"
    migration_source_hash: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if set(freeze_public_policy(self).monitored_classes) != set(self.policy.monitored_classes):
            raise ValueError("policy omits public classes from original/candidate/query signature")


@dataclasses.dataclass(frozen=True)
class ObjectiveV3(ObjectiveV2):
    """An objective explicitly bound to its pool, model and calibrated target basis."""

    schema_version: ClassVar[str] = SCHEMA_V3
    pool_hash: str = ""
    pair_selection_hash: str = ""
    model_hash: str = ""
    target_basis: str = "symbolic-semantic-vector/v3"
    calibration_hash: str = ""
    raw_pairs: tuple[tuple[int, int, int, int, float], ...] = ()
    epoch_hash: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("benefit", "costs"):
            rows = getattr(self, name)
            if rows and (
                len(rows) != len(self.unary)
                or any(len(row) != len(u) for row, u in zip(rows, self.unary))
            ):
                raise ValueError(f"{name} shape differs from the integer objective")
            if any(
                not math.isfinite(v) or (name == "costs" and v < 0) for row in rows for v in row
            ):
                raise ValueError(f"invalid {name} coefficient")
        if (
            self.raw_pairs
            and tuple((i, a, j, b, quantize(w, self.scale)) for i, a, j, b, w in self.raw_pairs)
            != self.pairs
        ):
            raise ValueError("raw pair coefficients do not match the integer objective")


@dataclasses.dataclass(frozen=True)
class ProofSupportV3(Record):
    """Sufficient asserted premises; parent validates the qualified derivation again."""

    schema_version: ClassVar[str] = SCHEMA_V3
    theory_hash: str
    policy_hash: str
    kind: str
    query: Any
    asserted_support: tuple[Any, ...]
    activation_expression: Any | None = None
    rule_version: str = "repair-horn/v3"
    complete: bool = True
    origin_ids: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class VerificationEventV3(Record):
    """Completed, hash-bound query event, independent of live solver variables."""

    schema_version: ClassVar[str] = SCHEMA_V3
    assignment_hash: str
    theory_hash: str
    policy_hash: str
    sequence: int
    obligation: ObligationV2
    backend: str
    capability_hash: str
    proof: ProofSupportV3 | None = None
    capability: tuple[tuple[str, Any], ...] = ()
    receipt_index: int = -1


@dataclasses.dataclass(frozen=True)
class VerificationReportV3(VerificationReportV2):
    """Final coverage plus acknowledged events; partial passes never authorize."""

    schema_version: ClassVar[str] = SCHEMA_V3
    expected_obligations: tuple[str, ...] = ()
    events: tuple[VerificationEventV3, ...] = ()
    proofs: tuple[ProofSupportV3, ...] = ()

    @property
    def authorizes(self) -> bool:
        return (
            super().authorizes
            and bool(self.expected_obligations)
            and len(self.obligations) == len(self.expected_obligations)
            and {q.name for q in self.obligations} == set(self.expected_obligations)
            and len(set(self.expected_obligations)) == len(self.expected_obligations)
        )


@dataclasses.dataclass(frozen=True)
class DeferredAssignmentV3(Record):
    """An untested enumerated plan still contributes its exact value to the bound."""

    schema_version: ClassVar[str] = SCHEMA_V3
    assignment: tuple[int, ...]
    value: int
    risk: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.risk):
            raise ValueError("risk must be finite")


@dataclasses.dataclass(frozen=True)
class SearchLedgerV3(Record):
    """Parent-owned restart state with separate proof and scheduling exclusions."""

    schema_version: ClassVar[str] = SCHEMA_V3
    input_hash: str
    objective_hash: str
    policy_hash: str
    logical_exclusions: tuple[tuple[tuple[int, ...], VerificationReportV2], ...] = ()
    proofs: tuple[ProofSupportV3, ...] = ()
    deferred: tuple[DeferredAssignmentV3, ...] = ()
    pending: tuple[PendingAssignmentV2, ...] = ()
    feasible: tuple[tuple[tuple[int, ...], VerificationReportV2], ...] = ()
    work_upper_bound: int | None = None
    upper_bound: int | None = None
    work_empty: bool = False
    revision: int = 0
    solves: int = 0
    checks: int = 0
    elapsed_seconds: float = 0.0
    shortlist_size: int = 1
    utility_window: int = 0
    risk_identity: str = "none"
    failures: tuple[str, ...] = ()
    baseline: tuple[tuple[str, VerificationReportV2], ...] = ()
    events: tuple[VerificationEventV3, ...] = ()
    shortlist_seconds: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.shortlist_seconds is not None and (
            type(self.shortlist_seconds) not in (int, float)
            or not math.isfinite(self.shortlist_seconds)
            or self.shortlist_seconds <= 0
        ):
            raise ValueError("invalid ledger shortlist deadline")
        for value in (self.revision, self.solves, self.checks, self.utility_window):
            if type(value) is not int or value < 0:
                raise ValueError("ledger counters must be nonnegative integers")
        if type(self.shortlist_size) is not int or self.shortlist_size < 1:
            raise ValueError("invalid ledger shortlist size")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("ledger elapsed time must be finite and nonnegative")
        assignments = (
            [p.assignment for p in self.deferred]
            + [p.assignment for p in self.pending]
            + [a for a, _ in self.feasible]
            + [a for a, _ in self.logical_exclusions]
        )
        if len(set(assignments)) != len(assignments):
            raise ValueError("ledger frontier regions overlap")
        if any(p.attempts < 1 for p in self.pending):
            raise ValueError("pending attempts must be positive")
        retained = [p.value for p in self.pending] + [p.value for p in self.deferred]
        if retained and (self.upper_bound is None or self.upper_bound < max(retained)):
            raise ValueError("ledger bound does not cover unresolved plans")


@dataclasses.dataclass(frozen=True)
class BaselineReportV3(Record):
    """One shared four-theory diagnosis charged once before matched arm selection."""

    schema_version: ClassVar[str] = SCHEMA_V3
    identity: str
    reports: tuple[tuple[str, VerificationReportV2], ...]
    elapsed_seconds: float
    checks: int
    failures: tuple[str, ...] = ()
    accounting: str = "common-preparation"
    exception_checks: tuple[ObligationV2, ...] = ()


@dataclasses.dataclass(frozen=True)
class RepairResultV3(RepairResultV2):
    """V3 frontier and proof evidence alongside the exact authorizing plan."""

    schema_version: ClassVar[str] = SCHEMA_V3
    ontology_patch: tuple[tuple[str, tuple[Any, ...], tuple[Any, ...]], ...] | None
    ledger: SearchLedgerV3 | None = None
    generation_status: str = "COMPLETE_DECLARED_ENUMERATION"
    optimization_bypassed: bool = False
    resource_counters: tuple[tuple[str, float], ...] = ()


def promote_input_v3(problem: RepairInputV2) -> RepairInputV3:
    """Explicitly start a new execution epoch; preserve the old artifact identity."""

    def fields(record: Any) -> dict[str, Any]:
        return {f.name: getattr(record, f.name) for f in dataclasses.fields(record)}

    objects = tuple(
        (
            RevisionObjectV3(
                **{
                    **fields(obj),
                    "candidates": tuple(
                        (
                            candidate
                            if isinstance(candidate, ReplacementCandidateV3)
                            else ReplacementCandidateV3(**fields(candidate))
                        )
                        for candidate in obj.candidates
                    ),
                }
            )
            if not isinstance(obj, RevisionObjectV3)
            else obj
        )
        for obj in problem.objects
    )
    policy = freeze_public_policy(problem)
    if not isinstance(policy, PolicyV3):
        policy = PolicyV3(**fields(policy))
    values = {**fields(problem), "objects": objects, "policy": policy}
    if not isinstance(problem, RepairInputV3):
        values["migration_source_hash"] = problem.content_hash
    return RepairInputV3(**values)


def replace_inventory(
    problem: RepairInputV2, objects: tuple[RevisionObjectV2, ...], **changes: Any
) -> RepairInputV2:
    """Atomically update a pool and its public policy, including validated v3 inputs."""
    values = {f.name: getattr(problem, f.name) for f in dataclasses.fields(RepairInputV2)}
    temporary = RepairInputV2(**{**values, "objects": objects})
    if isinstance(problem, RepairInputV3):
        objects = tuple(
            dataclasses.replace(
                obj,
                candidates=tuple(
                    (
                        c
                        if isinstance(c, ReplacementCandidateV3)
                        else ReplacementCandidateV3(
                            **{f.name: getattr(c, f.name) for f in dataclasses.fields(c)}
                        )
                    )
                    for c in obj.candidates
                ),
            )
            for obj in objects
        )
    return dataclasses.replace(
        problem, objects=objects, policy=freeze_public_policy(temporary), **changes
    )


class _PayloadRecordV3(Record):
    """A typed record with the historical flat report accessors, without Mapping coercion."""

    schema_version: ClassVar[str] = SCHEMA_V3
    payload: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return iter(self.payload.items())

    def __len__(self) -> int:
        return len(self.payload)

    def __contains__(self, key: str) -> bool:
        return key in self.payload

    def keys(self) -> Any:
        return self.payload.keys()

    def items(self) -> Any:
        return self.payload.items()

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def _validate_fields(self, required: set[str], allowed: set[str]) -> None:
        if not isinstance(self.payload, FrozenMapping):
            raise TypeError("typed report payload must be a mapping")
        keys = set(self.payload)
        if required - keys or keys - allowed:
            raise ValueError(
                f"invalid report fields: missing={sorted(required-keys)}, unknown={sorted(keys-allowed)}"
            )
        canonical_json(self.payload)

    def _hash_field(self, name: str, *, nullable: bool = False) -> None:
        value = self.payload.get(name)
        if nullable and value is None:
            return
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"{name} must be a canonical SHA256 identity")


@dataclasses.dataclass(frozen=True)
class ProposalRecordV3(_PayloadRecordV3):
    """One complete sampled encoding/bundle and its explicitly scoped probability."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        super().__post_init__()
        names = {
            "schema",
            "object_id",
            "candidate_id",
            "candidate",
            "assignment",
            "component",
            "log_probability",
            "grammar_hash",
            "circuit_hash",
            "retrieval_hash",
            "context_hash",
            "seed",
            "family",
            "probability_semantics",
            "distribution_scope",
        }
        self._validate_fields(names, names)
        if self["schema"] != "exact-repair/proposal/v3":
            raise ValueError("unsupported proposal payload schema")
        candidate = self["candidate"]
        if (
            not isinstance(candidate, ReplacementCandidateV3)
            or candidate.object_id != self["object_id"]
            or candidate.candidate_id != self["candidate_id"]
        ):
            raise ValueError("proposal bundle/object identity mismatch")
        for name in ("candidate_id", "grammar_hash", "retrieval_hash", "context_hash"):
            self._hash_field(name)
        self._hash_field("circuit_hash", nullable=True)
        if not isinstance(self["family"], str) or not self["family"]:
            raise ValueError("proposal must identify its complete template family")
        if not self["assignment"] or any(type(bit) is not bool for bit in self["assignment"]):
            raise ValueError("proposal assignment must be a nonempty Boolean vector")
        if any(type(self[name]) is not int or self[name] < 0 for name in ("component", "seed")):
            raise ValueError("proposal component and seed must be nonnegative integers")
        value = self["log_probability"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value > 1e-6
        ):
            raise ValueError("proposal log probability must be finite and at most zero")
        if self["probability_semantics"] not in {
            "conditioned_bundle_probability",
            "unconditioned_bundle_mass",
        }:
            raise ValueError("unknown proposal probability semantics")
        if self["distribution_scope"] not in {
            "declared_language",
            "completed_families_only_unknown_omitted_mass",
        }:
            raise ValueError("unknown proposal distribution scope")


@dataclasses.dataclass(frozen=True)
class GenerationReportV3(_PayloadRecordV3):
    """Strict generation receipt, retaining typed draws and incomplete-family evidence."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        super().__post_init__()
        counts = {
            "seed",
            "mixtures",
            "max_depth",
            "max_constructors",
            "attempted_draws",
            "enumerated_candidates",
            "valid_draws",
            "rejected_draws",
            "unique_draws",
            "duplicate_draws",
            "retained",
            "mandatory",
            "contextual_checks",
            "circuit_nodes",
            "boolean_variables",
            "max_nodes_per_family",
        }
        required = counts | {
            "schema",
            "object_id",
            "generation_status",
            "grammar_hash",
            "retrieval_hash",
            "graph_hash",
            "model_hash",
            "pair_selection_hash",
            "samples",
            "family_statuses",
            "distribution_scope",
            "probability_semantics",
        }
        optional = {
            "representative_max_checks",
            "preparation_identity",
            "support_admission",
            "support_omissions",
            "generation_identity",
            "language_hash",
            "protected_family_reports",
            "final_candidate_removals",
            "omitted_generation_symbols",
            "enabled_actions",
            "declared_circuit_limits",
            "preserved_candidate_ids",
            "removed_present",
            "proposal_context_hash",
            "proposal_context_order",
            "proposal_context_object_ids",
            "compiler_telemetry",
            "arm",
            "contextual_proof_hashes",
            "contextual_truncated",
            "contextual_scope",
            "circuit_hash",
            "compilation_seconds",
            "circuit_build_seconds",
            "compilation_cache_hit",
            "proposal_setup_seconds",
            "sampling_seconds",
            "proposal_seconds",
            "context_proof_seconds",
            "graph_seconds",
            "model_seconds",
            "omitted_graph_nodes",
            "omitted_supports",
            "pair_selection_omissions",
            "expansion_schedule_hash",
            "expansion_index",
            "previous_pool_hash",
        }
        self._validate_fields(required, required | optional)
        if "representative_max_checks" in self and (
            type(self["representative_max_checks"]) is not int
            or self["representative_max_checks"] < 0
        ):
            raise ValueError("representative traversal cap must be a nonnegative integer")
        for name in ("generation_identity", "language_hash"):
            if name in self:
                self._hash_field(name)
        for family in self.get("protected_family_reports", ()):
            if family.get("status") not in {
                "retained",
                "empty_language",
                "search_exhausted",
                "compile_timeout",
                "compile_node_limit",
                "compile_memory_limit",
                "worker_error",
                "removed_by_intervention",
                "candidate_cap",
            }:
                raise ValueError("invalid protected-family coverage status")
        if self["schema"] != "exact-repair/generation-report/v3":
            raise ValueError("unsupported generation payload schema")
        if not isinstance(self["object_id"], str) or not self["object_id"]:
            raise ValueError("generation report requires an object identity")
        if self["generation_status"] not in {
            "COMPLETE_DECLARED_ENUMERATION",
            "SAMPLED",
            "PARTIAL_RESOURCE_LIMIT",
            "INVALID_LANGUAGE",
            "ERROR",
        }:
            raise ValueError("invalid generation status")
        for name in counts | ({"expansion_index"} if "expansion_index" in self else set()):
            if type(self[name]) is not int or self[name] < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "grammar_hash",
            "retrieval_hash",
            "graph_hash",
            "model_hash",
            "pair_selection_hash",
        ):
            self._hash_field(name)
        for name in (
            "circuit_hash",
            "proposal_context_hash",
            "expansion_schedule_hash",
            "previous_pool_hash",
        ):
            if name in self:
                self._hash_field(name, nullable=name in {"circuit_hash", "previous_pool_hash"})
        for name, value in self.items():
            if name.endswith("_seconds") and (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative duration")
        if self["distribution_scope"] not in {
            "declared_language",
            "completed_families_only_unknown_omitted_mass",
        }:
            raise ValueError("invalid generation distribution scope")
        if self["probability_semantics"] not in {
            "conditioned_bundle_probability",
            "unconditioned_bundle_mass",
            "not_applicable",
        }:
            raise ValueError("invalid generation probability semantics")
        family_states = {}
        allowed_family_states = {
            "resolved",
            "empty_language",
            "compile_timeout",
            "compile_node_limit",
            "compile_memory_limit",
            "worker_error",
            "not_applicable",
            "retrieval_empty",
            "retrieval_truncated",
            "proof_unavailable",
            "weakening_unproved",
            "zero_probability",
            "numeric_failure",
            "artifact_invalid",
            "binding_invalid",
            "invalid_budget",
            "input_invalid",
            "cancelled",
            "partial",
        }
        for row in self["family_statuses"]:
            if (
                len(row) != 3
                or not isinstance(row[0], str)
                or not row[0]
                or row[1] not in allowed_family_states
                or not isinstance(row[2], str)
                or row[0] in family_states
            ):
                raise ValueError("invalid or duplicate family status record")
            family_states[row[0]] = row[1]
        if self["distribution_scope"] == "completed_families_only_unknown_omitted_mass" and self[
            "generation_status"
        ] not in {"PARTIAL_RESOURCE_LIMIT", "ERROR"}:
            raise ValueError("incomplete family mass cannot authorize complete generation status")
        samples = self["samples"]
        if (
            len(samples) != self["valid_draws"]
            or self["valid_draws"] > self["attempted_draws"]
            or self["unique_draws"] > self["valid_draws"]
            or self["duplicate_draws"] != self["valid_draws"] - self["unique_draws"]
            or self["retained"] < self["mandatory"]
        ):
            raise ValueError("inconsistent generation count accounting")
        for sample in samples:
            if (
                not isinstance(sample, ProposalRecordV3)
                or sample["object_id"] != self["object_id"]
                or len(sample["assignment"]) != self["boolean_variables"]
                or sample["grammar_hash"] != self["grammar_hash"]
                or sample["retrieval_hash"] != self["retrieval_hash"]
                or sample["distribution_scope"] != self["distribution_scope"]
                or sample["probability_semantics"] != self["probability_semantics"]
                or sample["seed"] != self["seed"]
                or sample["component"] >= self["mixtures"]
                or sample["circuit_hash"] != self.get("circuit_hash")
                or sample["context_hash"] != self.get("proposal_context_hash")
                or (family_states and family_states.get(sample["family"]) != "resolved")
            ):
                raise ValueError("sample does not bind to its generation receipt")
        if len({sample["candidate_id"] for sample in samples}) != self["unique_draws"]:
            raise ValueError("unique draw count differs from sampled canonical bundles")


@dataclasses.dataclass(frozen=True)
class CompactVerificationReportV3(VerificationReportV3):
    """Bounded final coverage receipt; full query events stay in a durable journal."""

    coverage_hash: str = ""
    coverage_count: int = 0
    passed_count: int = 0
    failed_count: int = 0
    unknown_count: int = 0
    exact_coverage: bool = False
    transport_identity: str = "durable-event-stream/r1"

    def __post_init__(self):
        super().__post_init__()
        counts = (self.coverage_count, self.passed_count, self.failed_count, self.unknown_count)
        if any(type(x) is not int or x < 0 for x in counts) or sum(counts[1:]) != counts[0]:
            raise ValueError("invalid compact coverage counts")
        if len(self.coverage_hash) != 64:
            raise ValueError("compact coverage requires a canonical query identity")

    @property
    def authorizes(self) -> bool:
        return (
            self.verdict == "VERIFIED_FEASIBLE"
            and self.scope in {"full_owl", "complete_supported_fragment"}
            and self.exact_coverage
            and self.coverage_count > 0
            and self.passed_count == self.coverage_count
            and all(q.complete and q.verdict == "pass" for q in self.obligations)
        )


def compact_verification_report(report: Any) -> Any:
    """Validate and reduce complete coverage without a final event-sized message."""
    if (
        not isinstance(report, VerificationReportV3)
        or isinstance(report, CompactVerificationReportV3)
        or max(len(report.obligations), len(report.events)) <= 128
    ):
        return report
    names = tuple(q.name for q in report.obligations)
    expected = tuple(report.expected_obligations)
    passed = sum(q.complete and q.verdict == "pass" for q in report.obligations)
    failed = sum(q.complete and q.verdict == "fail" for q in report.obligations)
    values = {f.name: getattr(report, f.name) for f in dataclasses.fields(VerificationReportV3)}
    bounded_support: list[tuple[str, Any]] = []
    for key, value in report.support:
        if isinstance(value, (tuple, list)) and len(value) > 128:
            bounded_support.extend(
                ((key + "_hash", canonical_hash(value)), (key + "_count", len(value)))
            )
        else:
            bounded_support.append((key, value))
    values.update(
        support=tuple(bounded_support),
        proofs=report.proofs[:1],
        obligations=tuple(q for q in report.obligations if q.verdict != "pass")[:32],
        expected_obligations=(),
        events=tuple(e for e in report.events if e.obligation.verdict == "fail")[:1],
        coverage_hash=canonical_hash(tuple(sorted(expected))),
        coverage_count=len(names),
        passed_count=passed,
        failed_count=failed,
        unknown_count=len(names) - passed - failed,
        exact_coverage=(
            len(set(names)) == len(names) == len(expected) == len(set(expected))
            and set(names) == set(expected)
        ),
    )
    return CompactVerificationReportV3(**values)


def compose_verification_report(
    report: VerificationReportV3,
    base_expected: tuple[str, ...],
    additions: tuple[ObligationV2, ...],
    *,
    source_exception_proof_hashes: tuple[str, ...],
) -> VerificationReportV3:
    """Extend exact coverage without promoting incomplete base evidence.

    The caller supplies independently reconstructed base queries and qualified
    source proof identities. Compact reports retain counts and the union hash;
    they do not expand their bounded inline evidence into a full query list.
    """
    extra_names = tuple(q.name for q in additions)
    expected = base_expected + extra_names
    unique = len(set(expected)) == len(expected)
    support = dict(report.support)
    if "coverage_composition" in support:
        identity_matches = (
            support["coverage_composition"] == "coverage-composition/v1"
            and tuple(support.get("source_exception_proof_hashes", ()))
            == source_exception_proof_hashes
            and support.get("source_exception_checks_hash") == canonical_hash(additions)
        )
        coverage_matches = (
            report.exact_coverage
            and report.coverage_count == len(expected)
            and report.coverage_hash == canonical_hash(tuple(sorted(expected)))
            if isinstance(report, CompactVerificationReportV3)
            else (
                len(report.expected_obligations) == len(expected)
                and set(report.expected_obligations) == set(expected)
                and tuple(q for q in report.obligations if q.name in extra_names) == additions
            )
        )
        if identity_matches and unique and coverage_matches:
            return report
        return dataclasses.replace(report, verdict="UNKNOWN", scope="partial_detection")
    support.update(
        coverage_composition="coverage-composition/v1",
        source_exception_proof_hashes=source_exception_proof_hashes,
        source_exception_checks_hash=canonical_hash(additions),
    )
    if isinstance(report, CompactVerificationReportV3):
        exact = (
            report.exact_coverage
            and report.coverage_count == len(base_expected)
            and report.coverage_hash == canonical_hash(tuple(sorted(base_expected)))
            and unique
        )
        passed = sum(q.complete and q.verdict == "pass" for q in additions)
        failed = sum(q.complete and q.verdict == "fail" for q in additions)
        return dataclasses.replace(
            report,
            obligations=tuple(q for q in additions + report.obligations if q.verdict != "pass")[
                :32
            ],
            expected_obligations=(),
            coverage_hash=canonical_hash(tuple(sorted(expected))),
            coverage_count=report.coverage_count + len(additions),
            passed_count=report.passed_count + passed,
            failed_count=report.failed_count + failed,
            unknown_count=report.unknown_count + len(additions) - passed - failed,
            exact_coverage=exact,
            support=tuple(sorted(support.items())),
        )
    exact = (
        len(report.expected_obligations) == len(base_expected)
        and set(report.expected_obligations) == set(base_expected)
        and unique
    )
    return dataclasses.replace(
        report,
        obligations=additions + report.obligations,
        expected_obligations=extra_names + report.expected_obligations,
        verdict=report.verdict if exact else "UNKNOWN",
        scope=report.scope if exact else "partial_detection",
        support=tuple(sorted(support.items())),
    )


@dataclasses.dataclass(frozen=True)
class RecoverySearchLedgerV3(SearchLedgerV3):
    """Corrective recovery identity retaining exception proofs and event journals."""

    recovery_identity: str = "verification-recovery/r2"
    baseline_artifact: BaselineReportV3 | None = None
    baseline_qualification_hash: str = ""
    event_journals: tuple[tuple[tuple[int, ...], str], ...] = ()
    persistence_reservation_seconds: float = 0.0


@dataclasses.dataclass(frozen=True)
class SourceExceptionProofV3(Record):
    """Complete source-only receipt resolving immutable premises by input hash."""

    schema_version: ClassVar[str] = SCHEMA_V3
    side: str
    asserted_source_hash: str
    documents: tuple[tuple[str, str, str], ...]
    policy_hash: str
    exceptions: tuple[Any, ...]
    theory_hash: str
    qualification_hash: str
    report: Mapping[str, Any]
    evidence_revision: str = "source-exception-evidence/v3.1"

    def __post_init__(self):
        super().__post_init__()
        if (
            self.side not in {"source", "target"}
            or self.evidence_revision != "source-exception-evidence/v3.1"
        ):
            raise ValueError("Invalid source-exception evidence identity")
        if any(
            len(value) != 64
            for value in (
                self.asserted_source_hash,
                self.policy_hash,
                self.theory_hash,
                self.qualification_hash,
            )
        ):
            raise ValueError("Source-exception evidence requires canonical dependency hashes")


@dataclasses.dataclass(frozen=True)
class QualifiedBaselineReportV3(BaselineReportV3):
    """Corrective baseline artifact; historical v3 bare checks stay readable."""

    exception_proofs: tuple[SourceExceptionProofV3, ...] = ()
    qualification_hash: str = ""
    evidence_revision: str = "source-exception-evidence/v3.1"
