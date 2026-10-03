"""Corrected study/2 contracts, kept separate from immutable legacy wire models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .models import (
    Asset,
    Candidate,
    EntityRef,
    Identifier,
    Mutation,
    PublicAsset,
    RankingResponse,
    ResponseType,
    StrictModel,
    StudyCase,
    StudyDefinition,
    StudyState,
)
from .workspace import WorkspaceScope

Method = Literal[
    "protege",
    "other_editor",
    "plain_files",
    "other_resource",
    "queries_scripts",
    "reasoner",
    "other_method",
]
RequirementAction = Literal[
    "inspect_other_candidate",
    "return_to_candidate",
    "search_entity",
    "navigate_parent",
    "navigate_child",
    "return_to_compared",
    "open_citation",
    "open_original_axiom",
    "locate_in_evidence_list",
    "inspect_graph_or_list",
    "change_graph_view",
    "rank_with_details_open",
    "add_rank",
    "move_rank",
    "remove_rank",
    "undo_rank",
    "keep_initial_order",
    "check_partial_ranking",
    "choose_none",
    "choose_insufficient",
    "copy_iri",
    "locate_downloads",
    "report_multiple_methods",
    "report_no_methods",
]
ResourceScope = Literal["supplied_only", "different_or_additional", "unsure"]


class ResourceMetadata(StrictModel):
    title: Annotated[str, Field(min_length=1, max_length=512)]
    role: Literal["source", "target", "both"]
    ontology_version_id: Identifier
    version_label: Annotated[str, Field(min_length=1, max_length=160)]
    license_note: Annotated[str, Field(min_length=1, max_length=2000)]
    information_notice: Annotated[str, Field(min_length=1, max_length=4000)]


class AssetV2(Asset):
    # Explanation resources have no download role; ontology resources require all metadata.
    title: Annotated[str | None, Field(min_length=1, max_length=512)] = None
    role: Literal["source", "target", "both"] | None = None
    ontology_version_id: Identifier | None = None
    version_label: Annotated[str | None, Field(min_length=1, max_length=160)] = None
    license_note: Annotated[str | None, Field(min_length=1, max_length=2000)] = None
    information_notice: Annotated[str | None, Field(min_length=1, max_length=4000)] = None

    @model_validator(mode="after")
    def require_metadata(self):
        if self.kind == "ontology":
            ResourceMetadata.model_validate(
                {k: getattr(self, k) for k in ResourceMetadata.model_fields}
            )
        return self


class PublicAssetV2(PublicAsset, ResourceMetadata):
    pass


class SetupV2(Mutation):
    setup_version: Identifier
    instructions_acknowledged: bool
    external_inspection_optional_understood: bool
    resource_access: Literal["not_checked", "available", "needs_help"]
    familiar_methods: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=80)]], Field(max_length=12)
    ] = Field(default_factory=list)
    submitted: bool = False

    @model_validator(mode="after")
    def ready_on_submit(self):
        if self.submitted and not (
            self.instructions_acknowledged
            and self.external_inspection_optional_understood
            and self.resource_access == "available"
        ):
            raise ValueError("Submit requires both acknowledgements and available resources")
        return self


class SetupReceiptV2(StrictModel):
    setup_version: Identifier
    instructions_acknowledged: bool
    external_inspection_optional_understood: bool
    resource_access: Literal["not_checked", "available", "needs_help"]
    familiar_methods: list[str]
    submitted: bool
    saved_at: str


class LessonRequirement(StrictModel):
    requirement_id: Identifier
    action: RequirementAction
    label: Annotated[str, Field(min_length=1, max_length=512)]
    alternatives: Annotated[list[RequirementAction], Field(max_length=4)] = Field(
        default_factory=list
    )


class TutorialLesson(StrictModel):
    lesson_id: Identifier
    title: Annotated[str, Field(min_length=1, max_length=512)]
    steps: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=2000)]],
        Field(min_length=1, max_length=20),
    ]
    view: Literal["explanation", "baseline"]
    requirements: Annotated[list[LessonRequirement], Field(min_length=1, max_length=30)]
    optional: bool = False


class AssessmentOption(StrictModel):
    code: Identifier
    label: Annotated[str, Field(min_length=1, max_length=2000)]


class AssessmentRow(StrictModel):
    row_id: Identifier
    label: Annotated[str, Field(min_length=1, max_length=2000)]


class AssessmentPartB(StrictModel):
    prompt: Annotated[str, Field(min_length=1, max_length=2000)]
    options: Annotated[list[AssessmentOption], Field(min_length=2, max_length=12)]


class AssessmentResponse(StrictModel):
    choice: Identifier | None = None
    choices: Annotated[list[Identifier] | None, Field(max_length=12)] = None
    matches: Annotated[dict[Identifier, Identifier] | None, Field(max_length=12)] = None
    part_b: Identifier | None = None


class AssessmentItem(StrictModel):
    question_id: Identifier
    title: Annotated[str, Field(min_length=1, max_length=512)]
    prompt: Annotated[str, Field(min_length=1, max_length=4000)]
    kind: Literal["single", "multiple", "match", "match_and_single"]
    options: Annotated[list[AssessmentOption] | None, Field(max_length=12)] = None
    rows: Annotated[list[AssessmentRow] | None, Field(max_length=12)] = None
    row_options: Annotated[list[AssessmentOption] | None, Field(max_length=12)] = None
    part_b: AssessmentPartB | None = None
    lesson_id: Identifier


class AssessmentRule(StrictModel):
    response: AssessmentResponse
    correct_feedback: Annotated[str, Field(min_length=1, max_length=4000)]
    incorrect_feedback: Annotated[str, Field(min_length=1, max_length=4000)]


class TutorialCase(StrictModel):
    case_id: Identifier = "practice-case"
    source: EntityRef
    source_label: Annotated[str, Field(min_length=1, max_length=512)]
    candidates: Annotated[list[Candidate], Field(min_length=5, max_length=5)]
    explanation_refs: Annotated[list[Identifier], Field(min_length=1, max_length=10)]
    ontology_resources: Annotated[list[PublicAssetV2], Field(min_length=2, max_length=10)]


class TutorialPublic(StrictModel):
    tutorial_id: Identifier
    version: Identifier
    hash: Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
    synthetic: Literal[True] = True
    intro: Annotated[str, Field(min_length=1, max_length=8000)]
    lessons: Annotated[list[TutorialLesson], Field(min_length=6, max_length=20)]
    assessment: Annotated[list[AssessmentItem], Field(min_length=5, max_length=5)]
    case: TutorialCase


class TutorialDefinition(TutorialPublic):
    hash: str = ""
    practice_id: Identifier = "practice-case"
    transfer_group: Identifier
    assessment_version: Identifier
    compatible_builds: Annotated[list[Identifier], Field(min_length=1, max_length=20)]
    completion_policy: Literal["all_required_actions_and_all_items_pass"] = (
        "all_required_actions_and_all_items_pass"
    )
    grading: dict[Identifier, AssessmentRule]

    @model_validator(mode="after")
    def complete_definition(self):
        from .tutorial import tutorial_hash, validate_definition

        if not self.hash:
            self.hash = tutorial_hash(self.model_dump(mode="json"))
        validate_definition(self.model_dump(mode="json"))
        return self


class TutorialAction(StrictModel):
    requirement_id: Identifier
    action: RequirementAction
    candidate_id: Identifier | None = None
    entity: EntityRef | None = None
    fact_id: Identifier | None = None
    ranked_candidate_ids: Annotated[list[Identifier] | None, Field(max_length=5)] = None
    response_type: ResponseType | None = None
    methods: Annotated[list[Method] | None, Field(max_length=7)] = None
    asset_ids: Annotated[list[Identifier] | None, Field(max_length=10)] = None


class PracticeAnswer(StrictModel):
    key: Identifier
    response_type: ResponseType | None = None
    ranked_candidate_ids: Annotated[list[Identifier], Field(max_length=5)] = Field(
        default_factory=list
    )


class AssessmentDraft(StrictModel):
    question_id: Identifier
    response: AssessmentResponse


class LessonPosition(StrictModel):
    view: Literal["lesson"]
    lesson_id: Identifier
    question_id: None


class AssessmentPosition(StrictModel):
    view: Literal["assessment"]
    lesson_id: None
    question_id: Identifier | None


TutorialPosition = Annotated[LessonPosition | AssessmentPosition, Field(discriminator="view")]


class TutorialProgressMutation(Mutation):
    tutorial_version: Identifier
    position: TutorialPosition | None = None
    current_lesson_id: Identifier | None = None
    lesson_id: Identifier | None = None
    completed_requirements: Annotated[list[Identifier], Field(max_length=100)] = Field(
        default_factory=list
    )
    actions: Annotated[list[TutorialAction], Field(max_length=100)] = Field(default_factory=list)
    practice: PracticeAnswer | None = None
    assessment_draft: AssessmentDraft | None = None
    help_opened: bool = False

    @model_validator(mode="after")
    def explicit_position(self):
        if "position" in self.model_fields_set:
            if self.position is None:
                raise ValueError("Position must be a complete object, not null")
            if "current_lesson_id" in self.model_fields_set:
                if self.current_lesson_id != self.position.lesson_id:
                    raise ValueError("Legacy lesson and position disagree")
        return self


class TutorialAssessment(Mutation):
    tutorial_version: Identifier
    question_id: Identifier
    attempt_id: Identifier
    response: AssessmentResponse


class TutorialComplete(Mutation):
    tutorial_version: Identifier


class AttemptReceipt(StrictModel):
    attempt_id: Identifier
    question_id: Identifier
    response: AssessmentResponse
    correct: bool
    feedback: str
    revisit_lesson_id: Identifier | None
    submitted_at: str


class TutorialReceipt(StrictModel):
    tutorial_version: Identifier
    current_lesson_id: Identifier | None
    # Only pre-extension immutable mutation receipts may omit position. Fresh
    # state always projects a normalized complete object; omit this default on replay.
    position: TutorialPosition | None = None
    completed_requirements: list[Identifier]
    practice: dict
    assessment_drafts: dict[Identifier, AssessmentResponse]
    attempts: list[AttemptReceipt]
    passed_items: list[Identifier]
    outstanding: list[Identifier]
    help_opened: int
    completed_at: str | None


class ConsultationDraftV2(Mutation):
    presentation_id: Identifier
    form_version: Identifier
    consulted_external_ontologies: bool | None = None
    methods: Annotated[list[Method], Field(max_length=7)] = Field(default_factory=list)
    other_editor: Annotated[str | None, Field(max_length=160)] = None
    other_resource: Annotated[str | None, Field(max_length=160)] = None
    other_method: Annotated[str | None, Field(max_length=160)] = None
    resource_scope: ResourceScope | None = None

    @model_validator(mode="after")
    def coherent_draft(self):
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("Duplicate consultation method")
        if self.consulted_external_ontologies is not True and (
            self.methods
            or any(getattr(self, k) for k in ("other_editor", "other_resource", "other_method"))
            or self.resource_scope is not None
        ):
            raise ValueError("No or unanswered requires empty methods, names and resource scope")
        for name in ("other_editor", "other_resource", "other_method"):
            if getattr(self, name) and name not in self.methods:
                raise ValueError("Optional name requires its corresponding method")
        return self


class ConsultationV2(ConsultationDraftV2):
    @model_validator(mode="after")
    def complete_answer(self):
        if self.consulted_external_ontologies is None or (
            self.consulted_external_ontologies and not self.methods
        ):
            raise ValueError("Final consultation requires Yes with methods or No with none")
        return self


class ConsultationReceiptV2(StrictModel):
    presentation_id: Identifier
    form_version: Identifier
    consulted_external_ontologies: bool | None
    methods: list[Method]
    other_editor: str | None
    other_resource: str | None
    other_method: str | None
    resource_scope: ResourceScope | None
    saved_at: str
    case_id: Identifier | None = None
    submitted_at: str | None = None


class ProtocolVersions(StrictModel):
    setup: Identifier = "setup/2"
    tutorial: Identifier
    assessment: Identifier
    forms: Identifier = "exact-study-forms/2"
    information: Identifier
    resource_policy: Identifier
    software: Identifier
    export: Literal["exact-study-analysis/2", "exact-study-analysis/3"] = "exact-study-analysis/2"


class StudyDefinitionV2(StudyDefinition):
    contract_version: Literal["exact-study/2.0"] = "exact-study/2.0"
    form_version: Identifier = "exact-study-forms/2"
    tutorial_steps: list[str] = Field(default_factory=list)
    assets: list[AssetV2]
    tutorial: TutorialDefinition
    protocol_versions: ProtocolVersions
    # Validated by the workspace admission facade before publication freezes.
    workspace_scopes: Annotated[list[WorkspaceScope], Field(min_length=1)]
    resource_scope_analysis_rule: Annotated[str, Field(min_length=1, max_length=4000)]

    @model_validator(mode="after")
    def corrected_contract(self):
        p = self.protocol_versions
        if (
            p.tutorial != self.tutorial.version
            or p.assessment != self.tutorial.assessment_version
            or p.forms != self.form_version
            or p.information != self.information_version
            or p.software != self.software_version
            or p.resource_policy != self.visibility_policy.policy_id
        ):
            raise ValueError("Protocol versions must bind the actual frozen definitions")
        if self.software_version not in self.tutorial.compatible_builds:
            raise ValueError("Tutorial does not support the frozen software build")
        from .tutorial import validate_disjoint_definition

        validate_disjoint_definition(self.model_dump(mode="json"))
        return self


class RankingResponseV2(RankingResponse):
    contract_version: Literal["exact-study/2.0"] = "exact-study/2.0"


class WorkspaceDescriptor(StrictModel):
    scope_id: Identifier


class StudyCaseV2(StudyCase):
    contract_version: Literal["exact-study/2.0"] = "exact-study/2.0"
    workspace: WorkspaceDescriptor | None = None


class QuestionDefinitionV2(StrictModel):
    id: Identifier
    label: str
    options: dict[str, str] | None
    multiple: bool
    required: bool
    show_if: dict | None
    matrix: dict[str, str] | None
    option_order: list[Identifier]
    row_order: list[Identifier]

    @model_validator(mode="after")
    def exact_orders(self):
        for mapping, order in ((self.options, self.option_order), (self.matrix, self.row_order)):
            if len(order) != len(set(order)) or set(order) != set(mapping or {}):
                raise ValueError("Question orders must exactly match their option and row keys")
        return self


class FormsV2(StrictModel):
    version: Identifier
    background: list[QuestionDefinitionV2]
    consultation: list[QuestionDefinitionV2]
    final: list[QuestionDefinitionV2]
    experience_note: str


class StudyStateV2(StudyState):
    contract_version: Literal["exact-study/2.0"] = "exact-study/2.0"
    stage: Literal[
        "welcome",
        "setup",
        "background",
        "practice",
        "tutorial",
        "case",
        "consultation",
        "final",
        "paused",
        "closed",
        "completed",
    ]
    setup: SetupReceiptV2 | None
    consultation: ConsultationReceiptV2 | None
    ranking: RankingResponseV2 | None
    ontology_resources: list[PublicAssetV2]
    forms: FormsV2
    tutorial: TutorialPublic
    tutorial_progress: TutorialReceipt
    consultation_draft: ConsultationReceiptV2 | None
    previous_consultation: ConsultationReceiptV2 | None
    protocol_versions: ProtocolVersions
    telemetry: dict = Field(default_factory=dict)
