"""Strict, versioned study request and frozen publication contracts."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts import VisibilityPolicy

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:/-]+$")]
Condition = Literal["explanation", "ontology_baseline"]
ResponseType = Literal["ranked_candidates", "none_of_these", "insufficient_evidence"]


class StrictModel(BaseModel):
    """Reject unknown fields instead of accepting hidden ownership overrides."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EntityRef(StrictModel):
    ontology_version_id: Identifier
    iri: Annotated[str, Field(min_length=1, max_length=2048)]
    kind: Literal["class", "object_property", "data_property", "individual"] = "class"


class Candidate(StrictModel):
    candidate_id: Identifier
    entity: EntityRef
    label: Annotated[str, Field(min_length=1, max_length=512)]
    score: float
    score_meaning: Annotated[str, Field(min_length=1, max_length=512)]
    display_position: Annotated[int, Field(ge=1, le=5)]


class Asset(StrictModel):
    asset_id: Identifier
    path: Annotated[str, Field(min_length=1, max_length=512)]
    sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    size_bytes: Annotated[int, Field(ge=0)]
    kind: Literal["ontology", "explanation"]
    admission_receipt_path: str | None = None
    admission_receipt_sha256: Annotated[str | None, Field(pattern=r"^[a-f0-9]{64}$")] = None
    media_type: Literal[
        "application/json",
        "application/rdf+xml",
        "text/turtle",
        "text/plain",
        "application/owl+xml",
        "application/owl-functional",
    ]
    policy_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]

    @model_validator(mode="after")
    def validate_admission(self):
        for locator in (self.path, self.admission_receipt_path):
            if locator is None:
                continue
            path = PurePosixPath(locator)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in locator
                or path.as_posix() in {".", "study-definition.json"}
            ):
                raise ValueError(
                    "Asset paths must be safe relative paths outside reserved publication files"
                )
        if self.kind == "ontology":
            if not self.admission_receipt_path or not self.admission_receipt_sha256:
                raise ValueError("Ontology resources require a verified filtering receipt")
            if self.media_type == "application/json":
                raise ValueError("Ontology downloads must use an ontology serialization")
        elif self.media_type != "application/json":
            raise ValueError("Explanations require the strict JSON resource contract")
        return self


class FrozenCase(StrictModel):
    case_id: Identifier
    source: EntityRef
    source_label: Annotated[str, Field(min_length=1, max_length=512)]
    transfer_group: Identifier
    package_version: Identifier
    candidates: Annotated[list[Candidate], Field(min_length=5, max_length=5)]
    ontology_resource_ids: Annotated[list[Identifier], Field(min_length=2)]
    explanation_refs: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_candidates(self):
        if len({x.candidate_id for x in self.candidates}) != 5:
            raise ValueError("A case requires five unique candidate IDs")
        if (
            len(
                {
                    (x.entity.ontology_version_id, x.entity.iri, x.entity.kind)
                    for x in self.candidates
                }
            )
            != 5
        ):
            raise ValueError("A case requires five unique candidate identities")
        if [x.display_position for x in self.candidates] != list(range(1, 6)):
            raise ValueError("Initial order must have contiguous positions 1 to 5")
        return self


class ConstructionProvenance(StrictModel):
    """Private unchanged retrieval set and a precise, reproducible exclusion rule."""

    original_candidate_ids: Annotated[list[Identifier], Field(min_length=5, max_length=5)]
    original_scores: dict[str, float]
    original_ranks: dict[str, Annotated[int, Field(ge=1)]]
    replacement_rule: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def complete_original_set(self):
        ids = set(self.original_candidate_ids)
        if len(ids) != 5 or ids != set(self.original_scores) or ids != set(self.original_ranks):
            raise ValueError("Constructed cases require the complete original candidate set")
        if [self.original_ranks[cid] for cid in self.original_candidate_ids] != list(range(1, 6)):
            raise ValueError("Original retrieved candidates must preserve ranks 1 to 5")
        return self


class CaseKey(StrictModel):
    case_id: Identifier
    case_kind: Literal["answer_present", "answer_absent", "unresolved"]
    acceptable_candidate_ids: list[Identifier]
    adjudication_version: Identifier
    criterion: Annotated[str, Field(min_length=1)]
    evidence: Annotated[list[str], Field(min_length=1)]
    origin: Literal["natural", "constructed", "unresolved"]
    original_production_ranks: dict[str, Annotated[int, Field(ge=1)]]
    construction_provenance: ConstructionProvenance | None = None


class Block(StrictModel):
    condition: Condition
    case_ids: Annotated[list[Identifier], Field(min_length=1)]


class Schedule(StrictModel):
    schedule_id: Identifier
    blocks: Annotated[list[Block], Field(min_length=2, max_length=2)]


class StudyDefinition(StrictModel):
    contract_version: Literal["exact-study/1.0"] = "exact-study/1.0"
    study_revision: Identifier
    software_version: Identifier
    information_version: Identifier
    information_text: Annotated[str, Field(min_length=1)]
    consent_text: Annotated[str, Field(min_length=1)]
    instructions: Annotated[str, Field(min_length=1)]
    setup_instructions: Annotated[str, Field(min_length=1)]
    tutorial_steps: Annotated[list[str], Field(min_length=4)]
    form_version: Identifier = "exact-study-forms/1"
    closes_at: datetime
    synthetic: bool = True
    policy_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    visibility_policy: VisibilityPolicy
    analysis_plan: Annotated[str, Field(min_length=1)]
    launch_approvals: Annotated[list[str], Field(min_length=0)] = Field(default_factory=list)
    cases: Annotated[list[FrozenCase], Field(min_length=2)]
    assets: list[Asset] = Field(default_factory=list)
    schedules: Annotated[list[Schedule], Field(min_length=4, max_length=64)]
    components: list[
        Literal[
            "original_context",
            "entity_description",
            "hierarchy",
            "evidence_table",
            "evidence_graph",
            "pair_comparison",
        ]
    ] = Field(
        default=[
            "original_context",
            "entity_description",
            "hierarchy",
            "evidence_table",
            "evidence_graph",
            "pair_comparison",
        ]
    )

    @model_validator(mode="after")
    def validate_design(self):
        if self.closes_at.tzinfo is None:
            raise ValueError("Study closing time must include a timezone")
        if (
            self.visibility_policy
            and self.policy_hash != self.visibility_policy.policy_hash.removeprefix("sha256:")
        ):
            raise ValueError("Study policy hash must match its immutable visibility policy")
        if not self.synthetic and not self.launch_approvals:
            raise ValueError("Live publication requires owner-supplied launch approvals")
        if len(self.components) != len(set(self.components)):
            raise ValueError("Duplicate explanation component")
        ids = {c.case_id for c in self.cases}
        if len(ids) != len(self.cases) or len({s.schedule_id for s in self.schedules}) != len(
            self.schedules
        ):
            raise ValueError("Duplicate case or schedule ID")
        asset_map = {a.asset_id: a for a in self.assets}
        if len(asset_map) != len(self.assets):
            raise ValueError("Duplicate asset ID")
        paths = [
            PurePosixPath(locator).as_posix()
            for asset in self.assets
            for locator in (asset.path, asset.admission_receipt_path)
            if locator is not None
        ]
        if len(paths) != len(set(paths)):
            raise ValueError("Asset and admission receipt paths must be unique")
        if any(a.policy_hash != self.policy_hash for a in self.assets):
            raise ValueError("All resources must share the frozen information policy")
        groups = {c.case_id: c.transfer_group for c in self.cases}
        conditions = {cid: set() for cid in ids}
        signatures = set()
        for schedule in self.schedules:
            flat = [cid for block in schedule.blocks for cid in block.case_ids]
            if len(flat) != len(ids) or set(flat) != ids:
                raise ValueError("Every schedule must show each case exactly once")
            if len(set(groups[cid] for cid in flat)) != len(flat):
                raise ValueError("Near-duplicate source groups may appear only once")
            if {b.condition for b in schedule.blocks} != {"explanation", "ontology_baseline"}:
                raise ValueError("A schedule must contain both condition blocks")
            signatures.add(tuple((b.condition, tuple(sorted(b.case_ids))) for b in schedule.blocks))
            for block in schedule.blocks:
                for cid in block.case_ids:
                    conditions[cid].add(block.condition)
        if len(signatures) != len(self.schedules) or any(len(v) != 2 for v in conditions.values()):
            raise ValueError("Schedules must cross complementary forms and block orders")
        for signature in signatures:
            if tuple(reversed(signature)) not in signatures:
                raise ValueError("Every form needs both block orders")
        for case in self.cases:
            if not case.explanation_refs:
                raise ValueError("Explanation cases require prepared explanation resources")
            if len(case.ontology_resource_ids) != len(set(case.ontology_resource_ids)) or len(
                case.explanation_refs
            ) != len(set(case.explanation_refs)):
                raise ValueError("Duplicate case resource reference")
            if any(
                not self.visibility_policy.allows_ontology(entity.ontology_version_id)
                for entity in [case.source, *(c.entity for c in case.candidates)]
            ):
                raise ValueError("Case entity is outside the frozen ontology universe")
            for aid in case.ontology_resource_ids:
                if aid not in asset_map or asset_map[aid].kind != "ontology":
                    raise ValueError("Unknown ontology resource")
            for aid in case.explanation_refs:
                if aid not in asset_map or asset_map[aid].kind != "explanation":
                    raise ValueError("Unknown explanation resource")
        return self


class Publish(StrictModel):
    definition: StudyDefinition
    case_keys: list[CaseKey]


class Mutation(StrictModel):
    idempotency_key: Identifier
    expected_revision: Annotated[int, Field(ge=0)]


class Consent(Mutation):
    information_version: Identifier
    accepted: bool


class Setup(Mutation):
    protege_installed: bool
    source_opened: bool
    target_opened: bool
    practice_source_located: bool
    practice_definition_parents_inspected: bool
    protege_version: Annotated[str | None, Field(max_length=80)] = None
    completed_tutorial_steps: list[int] = Field(default_factory=list)


class Questionnaire(Mutation):
    form_version: Identifier
    answers: dict
    submitted: bool = False


class Ranking(Mutation):
    presentation_id: Identifier
    response_type: ResponseType | None = None
    ranked_candidate_ids: Annotated[list[Identifier], Field(max_length=5)] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_ranking(self):
        ids = self.ranked_candidate_ids
        if len(set(ids)) != len(ids):
            raise ValueError("Candidate ranks must be unique")
        if self.response_type == "ranked_candidates" and not ids:
            raise ValueError("A ranked response requires at least one candidate")
        if self.response_type != "ranked_candidates" and ids:
            raise ValueError("None, insufficient information, and unanswered require an empty list")
        return self


class Consultation(Mutation):
    consulted_external_ontologies: bool
    methods: list[Literal["protege", "other_editor", "plain_files", "other_resource"]] = Field(
        default_factory=list
    )
    other_editor: Annotated[str | None, Field(max_length=160)] = None
    other_resource: Annotated[str | None, Field(max_length=160)] = None

    @model_validator(mode="after")
    def validate_methods(self):
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("Duplicate consultation method")
        if bool(self.methods) != self.consulted_external_ontologies:
            raise ValueError("Yes requires a method; No requires no methods")
        if (self.other_editor and "other_editor" not in self.methods) or (
            self.other_resource and "other_resource" not in self.methods
        ):
            raise ValueError("Optional text requires its corresponding method")
        return self


class Resume(Mutation):
    gap_activity: Literal["external_work", "break", "unknown"] = "unknown"


class Event(StrictModel):
    event_id: Identifier
    page_instance_id: Identifier
    sequence: Annotated[int, Field(ge=0)]
    case_id: Identifier
    presentation_id: Identifier
    type: Literal[
        "case_ready",
        "candidate_inspected",
        "rank_add",
        "rank_remove",
        "rank_move",
        "keep_initial_order",
        "response_type_change",
        "hierarchy_expand",
        "hierarchy_collapse",
        "definition_open",
        "axiom_open",
        "evidence_open",
        "table_open",
        "graph_open",
        "comparison_open",
        "graph_zoom",
        "graph_fit",
        "external_resource_link",
        "pause",
        "resume",
        "visibility",
        "submit",
        "revision",
    ]
    component_id: Identifier | None = None
    element_id: Identifier | None = None
    client_monotonic_ms: Annotated[float, Field(ge=0)]
    build_version: Identifier
    visibility: Literal["visible", "hidden"] | None = None
    loading_ms: Annotated[float | None, Field(ge=0)] = None

    @model_validator(mode="after")
    def no_external_urls(self):
        for value in (
            self.event_id,
            self.page_instance_id,
            self.component_id,
            self.element_id,
            self.build_version,
        ):
            if value and "://" in value:
                raise ValueError("Telemetry accepts component identifiers, not visited URLs")
        return self


class EventBatch(StrictModel):
    events: Annotated[list[Event], Field(min_length=1, max_length=100)]


class TimingSegment(StrictModel):
    segment_id: Identifier
    page_instance_id: Identifier
    case_id: Identifier | None = None
    presentation_id: Identifier | None = None
    stage: Literal["setup", "background", "practice", "case", "consultation", "final"]
    monotonic_start_ms: Annotated[float, Field(ge=0)]
    monotonic_end_ms: Annotated[float, Field(ge=0)]

    @model_validator(mode="after")
    def bounded_interval(self):
        if not 0 <= self.monotonic_end_ms - self.monotonic_start_ms <= 300_000:
            raise ValueError("Timing segments must cover at most five minutes")
        return self


class Exchange(StrictModel):
    secret: Annotated[str, Field(min_length=32, max_length=128)]


class Invitations(StrictModel):
    count: Annotated[int, Field(ge=1, le=1000)] = 1
    test: bool = True


class RankingResponse(StrictModel):
    """Acknowledged answer identity and workflow are independent of semantic choice."""

    artifact_type: Literal["ranking_response"] = "ranking_response"
    contract_version: Literal["exact-study/1.0"] = "exact-study/1.0"
    study_revision: Identifier
    session_id: Identifier
    case_id: Identifier
    presentation_id: Identifier
    response_type: ResponseType | None
    ranked_candidate_ids: list[Identifier]
    revision: Annotated[int, Field(ge=0)]
    workflow_state: Literal["draft", "submitted"]
    saved_at: datetime
    submitted_at: datetime | None


class StudyCase(StrictModel):
    """Condition-filtered current case; no answer-key or researcher-only fields."""

    artifact_type: Literal["study_case"] = "study_case"
    contract_version: Literal["exact-study/1.0"] = "exact-study/1.0"
    study_revision: Identifier
    case_id: Identifier
    presentation_id: Identifier
    condition: Condition
    source: EntityRef
    source_label: str
    candidates: Annotated[list[Candidate], Field(min_length=5, max_length=5)]
    ontology_resource_ids: list[Identifier]
    package_version: Identifier
    explanation_refs: list[Identifier]


class PublicAsset(StrictModel):
    """Verified download metadata without private filesystem or receipt paths."""

    asset_id: Identifier
    sha256: str
    size_bytes: Annotated[int, Field(ge=0)]
    kind: Literal["ontology"]
    media_type: str
    policy_hash: str


class ConsentReceipt(StrictModel):
    information_version: Identifier
    accepted: bool
    acknowledged_at: datetime


class SetupReceipt(StrictModel):
    protege_installed: bool
    source_opened: bool
    target_opened: bool
    practice_source_located: bool
    practice_definition_parents_inspected: bool
    protege_version: str | None
    completed_tutorial_steps: list[int]
    verification: Literal["self_reported_task_confirmed"]
    saved_at: datetime


class QuestionnaireResponse(StrictModel):
    form_version: Identifier
    answers: dict
    submitted: bool
    answer_states: dict[str, Literal["answered", "skipped", "not_answered"]]
    saved_at: datetime


class ConsultationReceipt(StrictModel):
    consulted_external_ontologies: bool
    methods: list[Literal["protege", "other_editor", "plain_files", "other_resource"]]
    other_editor: str | None
    other_resource: str | None
    saved_at: datetime


class StudyState(StrictModel):
    """Complete acknowledged participant state for a separate first-party frontend."""

    artifact_type: Literal["study_state"] = "study_state"
    contract_version: Literal["exact-study/1.0"] = "exact-study/1.0"
    study_revision: Identifier
    session_id: Identifier
    revision: Annotated[int, Field(ge=0)]
    stage: Literal[
        "welcome",
        "setup",
        "background",
        "practice",
        "case",
        "consultation",
        "final",
        "paused",
        "closed",
        "completed",
    ]
    assignment_id: Identifier | None
    current_case_id: Identifier | None
    current_presentation_id: Identifier | None
    completed_cases: Annotated[int, Field(ge=0)]
    assigned_case_count: Annotated[int, Field(ge=0)]
    ranking: RankingResponse | None
    consultation: ConsultationReceipt | None
    consent: ConsentReceipt | None
    setup: SetupReceipt | None
    questionnaires: dict[str, QuestionnaireResponse]
    information_version: Identifier
    information_text: str
    consent_text: str
    instructions: str
    setup_instructions: str
    tutorial_steps: list[str]
    forms: dict
    ontology_resources: list[PublicAsset]
    synthetic: bool
    gap_recovery: str


class EventAcknowledgement(StrictModel):
    acknowledged_event_ids: list[Identifier]
    sequence_gaps: list[dict[str, str | int]]


class TimingAcknowledgement(StrictModel):
    acknowledged_segment_id: Identifier
