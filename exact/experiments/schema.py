"""Strict schemas for the lean paper experiment suite."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from exact.core.entities.configs.strict import StrictConfigModel
from exact.core.entities.configs.yaml_io import load_yaml_mapping

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SCREEN_REFERENCE_ROLES = {
    "dev",
    "development",
    "diagnostic",
    "valid",
    "validation",
}
_REPORTING_REFERENCE_ROLES = {"full", "reporting", "test"}


def _identifier(value: str, label: str) -> str:
    normalized = str(value).strip()
    if not normalized or not _IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(
            f"{label} must be a non-empty path-safe identifier containing only "
            "letters, digits, '.', '_', or '-'"
        )
    return normalized


def _reference_role(value: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError("reference_role must be non-empty")
    return normalized


class ImplementationConfig(StrictConfigModel):
    status: Literal["ready", "deferred", "blocked"] = "ready"
    reason: Optional[str] = None

    @model_validator(mode="after")
    def require_reason(self) -> "ImplementationConfig":
        if self.status != "ready" and not str(self.reason or "").strip():
            raise ValueError(f"implementation status {self.status!r} requires a reason")
        return self


class BaselineManifest(StrictConfigModel):
    """Immutable identity of the production baseline used by paper experiments."""

    schema_version: Literal[1] = 1
    baseline_id: str
    parent: str
    exact_om_version: str
    pyowlcore_version: str
    config: Path
    config_sha256: str
    source_commit: str
    source_tree_sha256: str
    source_tree_files: int = Field(ge=0)
    status: Literal["frozen_configuration"]
    experiment_flags: Literal["disabled"]
    note: str

    @field_validator("baseline_id", "parent")
    @classmethod
    def validate_ids(cls, value: str) -> str:
        return _identifier(value, "baseline id")

    @field_validator("exact_om_version", "pyowlcore_version", "note")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("baseline version and note fields must be non-empty")
        return normalized

    @field_validator("config_sha256", "source_tree_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if not _SHA256_RE.fullmatch(normalized):
            raise ValueError("baseline SHA-256 fields must contain 64 lowercase hex characters")
        return normalized

    @field_validator("source_commit")
    @classmethod
    def validate_source_commit(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if not _GIT_COMMIT_RE.fullmatch(normalized):
            raise ValueError("baseline source_commit must contain 40 lowercase hex characters")
        return normalized


class ResourceConfig(StrictConfigModel):
    kind: Literal["cpu", "gpu", "llm"] = "cpu"
    device: Optional[str] = None
    profile: Optional[str] = None
    concurrency: int = Field(1, ge=1)

    def serialization_key(self) -> str:
        if self.kind == "cpu":
            return "cpu"
        identity = self.device if self.kind == "gpu" else self.profile
        return f"{self.kind}:{identity or 'default'}"


class TaskConfig(StrictConfigModel):
    id: str
    split_role: Literal["development", "diagnostic", "reporting"]
    track: Optional[str] = None
    task: Optional[str] = None
    reference_role: str
    overlay: Dict[str, Any] = Field(default_factory=dict)
    source_cap: Optional[int] = Field(None, ge=1)
    reference_completeness: Literal["complete", "known_incomplete", "unknown"] = "unknown"
    capabilities: List[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _identifier(value, "task id")

    @field_validator("reference_role")
    @classmethod
    def normalize_reference_role(cls, value: str) -> str:
        return _reference_role(value)

    @model_validator(mode="after")
    def require_input_selector(self) -> "TaskConfig":
        data = self.overlay.get("data") if isinstance(self.overlay, dict) else None
        if not self.track and not isinstance(data, dict):
            raise ValueError(f"task {self.id!r} must declare track/task or a data overlay")
        return self


class StageConfig(StrictConfigModel):
    tasks: List[TaskConfig]
    seeds: List[int]
    source_cap: Optional[int] = Field(None, ge=1)

    @model_validator(mode="after")
    def validate_stage(self) -> "StageConfig":
        if not self.tasks:
            raise ValueError("a stage must declare at least one task")
        if not self.seeds:
            raise ValueError("a stage must declare at least one seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("stage seeds must be unique")
        task_ids = [task.id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("stage task identifiers must be unique")
        return self


StageName = Literal["screen", "confirm"]


def _default_arm_stages() -> List[StageName]:
    return ["screen", "confirm"]


class ArmConfig(StrictConfigModel):
    id: str
    role: Literal["baseline", "candidate", "control", "diagnostic", "oracle"]
    overlay: Dict[str, Any] = Field(default_factory=dict)
    stages: List[StageName] = Field(default_factory=_default_arm_stages)
    resource: Optional[ResourceConfig] = None
    required_control: bool = False
    deployable: bool = True
    supervision_label: Optional[
        Literal[
            "target_label_free",
            "in_pair_supervised",
            "cross_pair_transfer",
            "oracle_diagnostic",
        ]
    ] = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _identifier(value, "arm id")

    @model_validator(mode="after")
    def oracle_is_not_deployable(self) -> "ArmConfig":
        if self.role == "oracle" and self.deployable:
            raise ValueError(f"oracle arm {self.id!r} must set deployable: false")
        if len(set(self.stages)) != len(self.stages):
            raise ValueError(f"arm {self.id!r} stages must be unique")
        return self


class SelectionDecisionConfig(StrictConfigModel):
    id: str
    baseline: str
    candidates: List[str]
    metric: str
    direction: Literal["max", "min"] = "max"
    min_delta: float = 0.0
    required_controls: List[str] = Field(default_factory=list)
    allow_screened_out: bool = True

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _identifier(value, "selection decision id")

    @field_validator("baseline")
    @classmethod
    def validate_baseline(cls, value: str) -> str:
        return _identifier(value, "selection baseline id")

    @field_validator("candidates", "required_controls")
    @classmethod
    def validate_arm_references(cls, values: List[str]) -> List[str]:
        return [_identifier(value, "selection arm id") for value in values]

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("selection metric must be non-empty")
        return normalized

    @model_validator(mode="after")
    def nonempty_candidates(self) -> "SelectionDecisionConfig":
        if not self.candidates:
            raise ValueError(f"selection decision {self.id!r} has no candidates")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError(f"selection decision {self.id!r} candidates must be unique")
        if len(set(self.required_controls)) != len(self.required_controls):
            raise ValueError(f"selection decision {self.id!r} required controls must be unique")
        if self.baseline in self.candidates:
            raise ValueError("a selection baseline cannot also be a candidate")
        return self


class SelectionConfig(StrictConfigModel):
    decisions: List[SelectionDecisionConfig]

    @model_validator(mode="after")
    def validate_decisions(self) -> "SelectionConfig":
        if not self.decisions:
            raise ValueError("selection must declare at least one decision")
        decision_ids = [decision.id for decision in self.decisions]
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("selection decision identifiers must be unique")
        return self


class DesignConfig(StrictConfigModel):
    primary_comparison: str
    primary_endpoint: str
    independent_unit: str
    power_status: Literal["powered", "underpowered", "descriptive"]
    assumptions: List[str]
    required_slices: List[str] = Field(default_factory=list)
    regression_bound: Optional[float] = None
    non_inferiority_margin: Optional[float] = None
    cost_bound: Optional[float] = None
    multiplicity: Literal["none", "holm"] = "none"
    reporting_exclusions: List[str] = Field(default_factory=list)

    @field_validator("primary_comparison", "primary_endpoint", "independent_unit")
    @classmethod
    def validate_nonempty_declaration(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("frozen design declarations must be non-empty")
        return normalized

    @field_validator("assumptions")
    @classmethod
    def validate_assumptions(cls, values: List[str]) -> List[str]:
        normalized = [str(value).strip() for value in values]
        if not normalized or any(not value for value in normalized):
            raise ValueError("frozen design assumptions must be non-empty")
        return normalized


class ComponentConfig(StrictConfigModel):
    id: str
    overlay: Dict[str, Any]
    source_experiment: Optional[str] = None
    claim: bool = True

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _identifier(value, "component id")

    @field_validator("source_experiment")
    @classmethod
    def validate_source_experiment(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _identifier(value, "component source experiment id")


class InteractionConfig(StrictConfigModel):
    id: str
    left: str
    right: str
    justification: str
    confirm: bool = True

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _identifier(value, "interaction id")

    @field_validator("left", "right")
    @classmethod
    def validate_component_reference(cls, value: str) -> str:
        return _identifier(value, "interaction component id")

    @field_validator("justification")
    @classmethod
    def validate_justification(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("interaction justification must be non-empty")
        return normalized

    @model_validator(mode="after")
    def endpoints_are_distinct(self) -> "InteractionConfig":
        if self.left == self.right:
            raise ValueError(f"interaction {self.id!r} endpoints must be distinct")
        return self


class CompositionConfig(StrictConfigModel):
    rolling_overlay: Dict[str, Any] = Field(default_factory=dict)
    components: List[ComponentConfig]
    interactions: List[InteractionConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_composition(self) -> "CompositionConfig":
        if not self.components:
            raise ValueError("E17 composition must declare at least one component")
        component_ids = [component.id for component in self.components]
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("E17 component identifiers must be unique")
        claimed_sources: list[str] = []
        for component in self.components:
            if component.claim and component.source_experiment is None:
                raise ValueError(
                    f"claimed E17 component {component.id!r} must declare " "source_experiment"
                )
            if component.claim and component.source_experiment is not None:
                claimed_sources.append(component.source_experiment)
        if len(set(claimed_sources)) != len(claimed_sources):
            raise ValueError("claimed E17 components must have unique source experiments")
        interaction_ids = [interaction.id for interaction in self.interactions]
        if len(set(interaction_ids)) != len(interaction_ids):
            raise ValueError("E17 interaction identifiers must be unique")
        known = set(component_ids)
        interaction_pairs: set[frozenset[str]] = set()
        for interaction in self.interactions:
            missing = sorted({interaction.left, interaction.right}.difference(known))
            if missing:
                raise ValueError(
                    f"E17 interaction {interaction.id!r} references unknown components {missing}"
                )
            pair = frozenset((interaction.left, interaction.right))
            if pair in interaction_pairs:
                raise ValueError("E17 interactions must use unique unordered component pairs")
            interaction_pairs.add(pair)
        return self


class ExperimentConfig(StrictConfigModel):
    schema_version: Literal[1] = 1
    experiment_id: str
    title: str
    implementation: ImplementationConfig = Field(default=ImplementationConfig.model_validate({}))
    base_config: Path
    baseline_id: str = "R_0"
    depends_on: List[str] = Field(default_factory=list)
    resource: ResourceConfig = Field(default=ResourceConfig.model_validate({}))
    screen: StageConfig
    confirm: StageConfig
    arms: List[ArmConfig]
    selection: SelectionConfig
    design: DesignConfig
    composition: Optional[CompositionConfig] = None
    negative_label_policy: str = "not_applicable"

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        return _identifier(value, "experiment id")

    @field_validator("depends_on")
    @classmethod
    def validate_dependencies(cls, values: List[str]) -> List[str]:
        normalized = [_identifier(value, "dependency experiment id") for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("experiment dependencies must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_matrix(self) -> "ExperimentConfig":
        if not self.arms:
            raise ValueError(f"{self.experiment_id}: must declare at least one arm")
        arm_ids = [arm.id for arm in self.arms]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError(f"{self.experiment_id}: arm identifiers must be unique")
        known = set(arm_ids)
        if self.experiment_id == "E17" and self.composition is not None:
            component_ids = [component.id for component in self.composition.components]
            for component in self.composition.components:
                if (
                    component.source_experiment is not None
                    and component.source_experiment not in self.depends_on
                ):
                    raise ValueError(
                        f"E17 component {component.id!r} source experiment "
                        f"{component.source_experiment!r} is not a declared dependency"
                    )
            known.update({"rolling", "stack_all"})
            known.update(f"stack_minus_{item}" for item in component_ids)
            for interaction in self.composition.interactions:
                known.update(
                    f"interaction_{interaction.id}_{cell}" for cell in ("00", "10", "01", "11")
                )
        for decision in self.selection.decisions:
            referenced = {decision.baseline, *decision.candidates, *decision.required_controls}
            missing = sorted(referenced.difference(known))
            if missing:
                raise ValueError(
                    f"{self.experiment_id}: selection {decision.id!r} references unknown arms {missing}"
                )
        bad_screen = [
            task.id
            for task in self.screen.tasks
            if task.split_role == "reporting" or task.reference_role not in _SCREEN_REFERENCE_ROLES
        ]
        if bad_screen:
            raise ValueError(
                f"{self.experiment_id}: screen cannot contain reporting references {bad_screen}"
            )
        bad_confirm = [
            task.id
            for task in self.confirm.tasks
            if task.split_role != "reporting"
            or task.reference_role not in _REPORTING_REFERENCE_ROLES
        ]
        if bad_confirm:
            raise ValueError(
                f"{self.experiment_id}: confirm tasks must use reporting references: {bad_confirm}"
            )
        if len(self.confirm.seeds) < 3:
            raise ValueError(f"{self.experiment_id}: confirm requires at least three paired seeds")
        if self.confirm.source_cap is not None or any(
            task.source_cap is not None for task in self.confirm.tasks
        ):
            raise ValueError(f"{self.experiment_id}: confirm cannot cap reporting sources")
        if self.experiment_id == "E17" and self.composition is None:
            raise ValueError("E17 requires a bounded composition declaration")
        if self.experiment_id == "E17" and self.composition is not None:
            if not any(
                decision.baseline == "rolling" and decision.candidates == ["stack_all"]
                for decision in self.selection.decisions
            ):
                raise ValueError(
                    "E17 primary selection must compare only stack_all against "
                    "the rolling baseline"
                )
            declared = {arm.id: arm for arm in self.arms}
            mandatory: dict[str, tuple[str, bool]] = {
                "rolling": ("baseline", False),
                "stack_all": ("candidate", False),
            }
            mandatory.update(
                {
                    f"stack_minus_{component.id}": ("control", True)
                    for component in self.composition.components
                    if component.claim
                }
            )
            for interaction in self.composition.interactions:
                if not interaction.confirm:
                    continue
                mandatory.update(
                    {
                        f"interaction_{interaction.id}_{cell}": ("control", True)
                        for cell in ("00", "10", "01", "11")
                    }
                )
            for arm_id, (role, required_control) in mandatory.items():
                template = declared.get(arm_id)
                if template is None:
                    continue
                missing_stages = sorted({"screen", "confirm"}.difference(template.stages))
                if missing_stages:
                    raise ValueError(
                        f"E17 mandatory arm {arm_id!r} must include screen and "
                        f"confirm (missing {missing_stages})"
                    )
                if template.role != role:
                    raise ValueError(f"E17 mandatory arm {arm_id!r} must have role {role!r}")
                if required_control and not template.required_control:
                    raise ValueError(
                        f"E17 mandatory control {arm_id!r} must set required_control: true"
                    )
            for component in self.composition.components:
                if component.claim:
                    continue
                arm_id = f"stack_minus_{component.id}"
                template = declared.get(arm_id)
                if template is not None and "confirm" in template.stages:
                    raise ValueError(f"E17 unclaimed leaveout {arm_id!r} cannot include confirm")
            for interaction in self.composition.interactions:
                if interaction.confirm:
                    continue
                for cell in ("00", "10", "01", "11"):
                    arm_id = f"interaction_{interaction.id}_{cell}"
                    template = declared.get(arm_id)
                    if template is not None and "confirm" in template.stages:
                        raise ValueError(
                            f"E17 exploratory interaction arm {arm_id!r} cannot " "include confirm"
                        )
        # One selection decision freezes one surviving candidate regardless of
        # how many development candidates or diagnostic controls it names.
        confirmatory_comparisons = len(self.selection.decisions)
        if self.experiment_id == "E17" and self.composition is not None:
            confirmatory_comparisons += sum(
                component.claim for component in self.composition.components
            )
            confirmatory_comparisons += sum(
                interaction.confirm for interaction in self.composition.interactions
            )
        if confirmatory_comparisons > 1 and self.design.multiplicity == "none":
            raise ValueError(
                f"{self.experiment_id}: multiple confirmatory comparisons "
                "require a multiplicity rule"
            )
        return self


class SuiteEntry(StrictConfigModel):
    id: str
    config: Path
    depends_on: List[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _identifier(value, "suite experiment id")

    @field_validator("depends_on")
    @classmethod
    def validate_dependencies(cls, values: List[str]) -> List[str]:
        normalized = [_identifier(value, "suite dependency id") for value in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError("suite dependencies must be unique")
        return normalized


class SuiteConfig(StrictConfigModel):
    schema_version: Literal[1] = 1
    suite_id: str
    baseline_id: str
    baseline_manifest: Path
    dataset_lock: Optional[Path] = None
    experiments: List[SuiteEntry]

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str) -> str:
        return _identifier(value, "suite id")

    @model_validator(mode="after")
    def validate_graph(self) -> "SuiteConfig":
        ids = [entry.id for entry in self.experiments]
        if len(ids) != len(set(ids)):
            raise ValueError("suite experiment identifiers must be unique")
        known = set(ids)
        for entry in self.experiments:
            missing = sorted(set(entry.depends_on).difference(known))
            if missing:
                raise ValueError(f"{entry.id}: unknown dependencies {missing}")
            if entry.id in entry.depends_on:
                raise ValueError(f"{entry.id}: experiment cannot depend on itself")
        _topological_ids(self.experiments)
        return self


def _topological_ids(entries: List[SuiteEntry]) -> List[str]:
    order_index = {entry.id: index for index, entry in enumerate(entries)}
    pending = {entry.id: set(entry.depends_on) for entry in entries}
    result: List[str] = []
    while pending:
        ready = sorted(
            (item for item, dependencies in pending.items() if not dependencies),
            key=lambda item: order_index[item],
        )
        if not ready:
            cycle = ", ".join(sorted(pending))
            raise ValueError(f"suite dependency graph contains a cycle involving: {cycle}")
        for item in ready:
            result.append(item)
            pending.pop(item)
            for dependencies in pending.values():
                dependencies.discard(item)
    return result


def suite_order(suite: SuiteConfig) -> List[str]:
    return _topological_ids(suite.experiments)


def load_experiment(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml_mapping(Path(path)))


def load_suite(path: Path) -> SuiteConfig:
    return SuiteConfig.model_validate(load_yaml_mapping(Path(path)))


def load_baseline(path: Path) -> BaselineManifest:
    return BaselineManifest.model_validate(load_yaml_mapping(Path(path)))


__all__ = [
    "ArmConfig",
    "BaselineManifest",
    "CompositionConfig",
    "DesignConfig",
    "ExperimentConfig",
    "ResourceConfig",
    "SelectionDecisionConfig",
    "StageConfig",
    "SuiteConfig",
    "TaskConfig",
    "load_baseline",
    "load_experiment",
    "load_suite",
    "suite_order",
]
