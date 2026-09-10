"""Bounded v2 campaign declarations and planning for the existing experiment runner.

The design blueprint describes research intent. A lock adds materialized cases,
strict treatment configurations, readiness evidence and measured work estimates.
Planning never loads ontology models or reads reporting labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Optional, cast

from pydantic import Field, field_validator, model_validator

from exact.core.entities.configs.strict import StrictConfigModel
from exact.core.entities.configs.yaml_io import load_yaml_mapping
from exact.experiments.schema import (
    ArmConfig,
    DesignConfig,
    SelectionConfig,
    _identifier,
)
from exact.utils.provenance import sha256_file, sha256_path

ReadinessStatus = Literal[
    "planned",
    "implementing",
    "fixture_ready",
    "screen_ready",
    "confirm_ready",
    "complete",
    "screened_out",
    "inapplicable",
    "blocked_input_resolution",
    "deferred_budget",
    "interrupted",
    "failed",
    "superseded",
]
TERMINAL = {"complete", "screened_out", "inapplicable", "deferred_budget"}
ENVELOPES = (
    "foundation",
    "retrieval",
    "channels",
    "decisions",
    "llm",
    "extensions",
    "sentinels",
    "final",
    "reserve",
)


def digest(value: Any) -> str:
    """Hash canonical numerical or scientific content, independently of its location."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class InputBinding(StrictConfigModel):
    """A materialized file; its path is a locator and its digest is the identity."""

    path: Path
    sha256: str

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("input sha256 must be 64 lowercase hexadecimal characters")
        return value

    def verify(self, root: Path) -> Path:
        path = self.path if self.path.is_absolute() else root / self.path
        if not path.exists():
            raise FileNotFoundError(f"unresolved materialized input: {path}")
        if sha256_path(path) != self.sha256:
            raise ValueError(f"materialized input changed: {path}")
        return path.resolve()


class Readiness(StrictConfigModel):
    """Evidence for one arm at one stage, independent of other arms' readiness."""

    status: ReadinessStatus = "planned"
    reason: str
    implemented_paths: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    inspected_commit: Optional[str] = None
    missing: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_required(self) -> "Readiness":
        if not self.reason.strip():
            raise ValueError("readiness needs a concrete reason")
        if self.status in {"fixture_ready", "screen_ready", "confirm_ready", "complete"}:
            if not self.implemented_paths or not self.tests or not self.inspected_commit:
                raise ValueError("ready paths require code, tests, and an inspected commit")
        return self


class CaseBinding(StrictConfigModel):
    """Case selection made from capabilities, before inspecting quality outcomes."""

    task: str
    kind: Literal["class", "object_property", "data_property", "individual"] = "class"
    role: Literal["development", "reporting"]
    source: Optional[InputBinding] = None
    target: Optional[InputBinding] = None
    source_universe: Optional[InputBinding] = None
    references: dict[str, InputBinding] = Field(default_factory=dict)
    local_references: dict[str, InputBinding] = Field(default_factory=dict)
    candidates: dict[str, InputBinding] = Field(default_factory=dict)
    frozen_global_candidates: dict[str, InputBinding] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    reference_completeness: Literal["complete", "known_incomplete", "unknown"] = "unknown"
    negative_policy: Literal["complete_reference", "confirmed_only", "positive_unlabelled"] = (
        "positive_unlabelled"
    )
    selection_reason: str
    original_reference_role: Optional[str] = None
    transformation: Optional[str] = None
    heldout_case: Optional[str] = None
    overlay: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_roles(self) -> "CaseBinding":
        allowed = (
            {"train", "valid", "internal_check"}
            if self.role == "development"
            else {"train", "test"}
        )
        if (
            set(self.references)
            | set(self.local_references)
            | set(self.candidates)
            | set(self.frozen_global_candidates)
        ) - allowed:
            raise ValueError(f"{self.task}: reference/pool role violates {self.role} access")
        if (
            self.negative_policy == "complete_reference"
            and self.reference_completeness != "complete"
        ):
            raise ValueError("ordinary negatives require explicit complete-reference semantics")
        if (
            self.original_reference_role == "test"
            and self.role == "development"
            and not self.transformation
        ):
            raise ValueError(
                "repurposed test data requires a recorded research-split transformation"
            )
        return self


class WorkEstimate(StrictConfigModel):
    """Measured cold/warm work, including retries and a frozen safety factor."""

    cold_seconds: float = Field(ge=0)
    units: int = Field(ge=0)
    seconds_per_unit: float = Field(ge=0)
    preparation_seconds: float = Field(0, ge=0)
    fitting_seconds: float = Field(0, ge=0)
    evaluation_seconds: float = Field(0, ge=0)
    safety_factor: float = Field(1.5, ge=1.5)
    peak_ram_gb: float = Field(gt=0)
    peak_vram_gb: float = Field(0, ge=0)
    requests: int = Field(0, ge=0)
    tokens: int = Field(0, ge=0)
    projected_usd: float = Field(0, ge=0)
    measurement_artifact: InputBinding

    @field_validator(
        "cold_seconds",
        "seconds_per_unit",
        "preparation_seconds",
        "fitting_seconds",
        "evaluation_seconds",
        "safety_factor",
        "peak_ram_gb",
        "peak_vram_gb",
        "projected_usd",
    )
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("resource estimates must be finite")
        return value

    def seconds(self) -> float:
        return self.safety_factor * (
            self.cold_seconds
            + self.units * self.seconds_per_unit
            + self.preparation_seconds
            + self.fitting_seconds
            + self.evaluation_seconds
        )


class CampaignStep(StrictConfigModel):
    """One named sequential comparison, with no implicit Cartesian expansion."""

    id: str
    family: str
    case: str
    additional_cases: list[str] = Field(default_factory=list)
    phase: Literal["initial", "expansion", "sentinel", "freeze", "final", "late"] = "initial"
    budget_group: Literal[
        "foundation",
        "retrieval",
        "channels",
        "decisions",
        "llm",
        "extensions",
        "sentinels",
        "final",
    ]
    requires: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    inherits: list[str] = Field(default_factory=list)
    arms: list[ArmConfig]
    arm_inputs: dict[str, dict[Literal["source", "target"], InputBinding]] = Field(
        default_factory=dict
    )
    policy_paths: list[str] = Field(default_factory=list)
    readiness: dict[str, dict[Literal["screen", "confirm"], Readiness]]
    source_cap: Optional[int] = Field(300, ge=1)
    training_source_cap: int = Field(2000, ge=1)
    seeds: list[int] = Field(default_factory=lambda: [17])
    execution_modes: list[Literal["global_alignment", "local_ranking"]] = Field(
        default=["global_alignment"]
    )
    estimate: Optional[WorkEstimate] = None
    selection: SelectionConfig
    design: DesignConfig

    @field_validator("id", "family", "case")
    @classmethod
    def identifier(cls, value: str) -> str:
        return _identifier(value, "campaign identifier")

    @model_validator(mode="after")
    def bounded(self) -> "CampaignStep":
        ids = [arm.id for arm in self.arms]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("step arms must be nonempty and unique")
        if set(self.arm_inputs) - set(ids):
            raise ValueError("arm input views reference an undeclared arm")
        for bindings in self.arm_inputs.values():
            if self.family != "E13" or set(bindings) != {"source", "target"}:
                raise ValueError("E13 input views must bind both source and target")
        for arm in self.arms:
            if arm.overlay.get("data"):
                raise ValueError("arm data overrides require explicit immutable arm_inputs")
        for path in self.policy_paths:
            parts = path.split(".")
            if parts[0] not in {"matching", "selector", "candidates", "llm", "supervision"} or any(
                not part or not part.replace("_", "").isalnum() for part in parts
            ):
                raise ValueError(f"invalid component policy path: {path}")
        if set(self.readiness) != set(ids):
            raise ValueError("readiness must enumerate every arm exactly once")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("step seeds must be nonempty and unique")
        if not self.execution_modes or len(set(self.execution_modes)) != len(self.execution_modes):
            raise ValueError("execution modes must be nonempty and unique")
        if self.phase == "initial" and (
            self.source_cap is None or self.source_cap > 300 or self.seeds != [17]
        ):
            raise ValueError("initial screens use at most 300 sources and seed 17")
        if self.phase in {"expansion", "sentinel", "freeze"} and (
            self.source_cap is None or self.source_cap > 1000
        ):
            raise ValueError("development expansions use at most 1000 sources")
        if self.phase == "final" and self.source_cap is not None:
            raise ValueError("final reporting cannot cap sources")
        if self.training_source_cap > 2000 and self.phase == "initial":
            raise ValueError("initial training uses at most 2000 source groups")
        if set(self.inherits) - set(self.requires):
            raise ValueError("inherited policies must also be declared dependencies")
        known = set(ids)
        for decision in self.selection.decisions:
            if {decision.baseline, *decision.candidates, *decision.required_controls} - known:
                raise ValueError("selection references an undeclared treatment or control")
            if any(
                arm.id in decision.candidates and (not arm.deployable or arm.role == "oracle")
                for arm in self.arms
            ):
                raise ValueError("oracle/diagnostic arms cannot be selected")
        return self


class CampaignLock(StrictConfigModel):
    """Executable v2 bindings; scientific source blueprint remains immutable."""

    schema_version: Literal[2] = 2
    kind: Literal["exact_om_campaign_lock"] = "exact_om_campaign_lock"
    campaign_id: str
    blueprint: InputBinding
    profile: Literal["core_14d", "extended_21d"] = "core_14d"
    base_config: Path
    baseline_id: str = "R_v2"
    historical_parent: Literal["R_0"] = "R_0"
    model_lock: Optional[InputBinding] = None
    cases: dict[str, CaseBinding]
    steps: list[CampaignStep]
    openrouter_profile: str = "openrouter_gpt4o_mini"
    max_cpu_workers: int = Field(2, ge=1, le=8)
    heavy_gpu_workers: Literal[1] = 1
    openrouter_concurrency: int = Field(4, ge=1, le=8)
    final_requests_reserved: int = Field(0, ge=0)
    final_tokens_reserved: int = Field(0, ge=0)
    final_selection: Optional[InputBinding] = None
    freeze_step: Optional[str] = None
    composition_sources: list[str] = Field(default_factory=list)
    baseline_manifest: Optional[InputBinding] = None

    @field_validator("campaign_id", "baseline_id")
    @classmethod
    def identifier(cls, value: str) -> str:
        return _identifier(value, "campaign identifier")

    @model_validator(mode="after")
    def valid_cases(self) -> "CampaignLock":
        ids = [step.id for step in self.steps]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("campaign step IDs must be nonempty and unique")
        for step in self.steps:
            if step.case not in self.cases:
                raise ValueError(f"{step.id}: unknown case {step.case}")
            expected_role = "reporting" if step.phase == "final" else "development"
            for case_id in [step.case, *step.additional_cases]:
                if case_id not in self.cases or self.cases[case_id].role != expected_role:
                    raise ValueError(
                        f"{step.id}: {expected_role} step requires a matching case role: {case_id}"
                    )
            if step.arm_inputs and (
                step.additional_cases
                or "normalized_evidence" not in self.cases[step.case].capabilities
            ):
                raise ValueError(
                    "E13 arm views require one case with normalized_evidence capability"
                )
            if len({step.case, *step.additional_cases}) != 1 + len(step.additional_cases):
                raise ValueError(f"{step.id}: duplicate case")
        for name, case in self.cases.items():
            if case.heldout_case:
                heldout = self.cases.get(case.heldout_case)
                if heldout is None or heldout.role != "reporting" or case.heldout_case == name:
                    raise ValueError(f"{name}: distinct held-out case must be bound prospectively")
        if self.freeze_step is not None:
            frozen = next((step for step in self.steps if step.id == self.freeze_step), None)
            if frozen is None or frozen.phase != "freeze":
                raise ValueError("freeze_step must name a development composition comparison")
        if set(self.composition_sources) - set(ids):
            raise ValueError("composition source must name a declared development comparison")
        dependency_order(self.steps)
        return self


def dependency_order(steps: list[CampaignStep]) -> list[str]:
    """Resolve named output ports and steps; reject unknown ports and cycles."""
    producers: dict[str, str] = {}
    known = {step.id for step in steps}
    for step in steps:
        for port in step.produces:
            if port in producers or port in known:
                raise ValueError(f"duplicate/ambiguous dependency port: {port}")
            producers[port] = step.id
    dependencies = {}
    for step in steps:
        missing = set(step.requires) - known - set(producers)
        if missing:
            raise ValueError(f"{step.id}: unknown dependency ports {sorted(missing)}")
        dependencies[step.id] = {producers.get(item, item) for item in step.requires}
    order: list[str] = []
    by_id = {step.id: step for step in steps}
    while dependencies:
        available = sorted(
            (name for name, parents in dependencies.items() if not parents),
            key=lambda name: (ENVELOPES.index(by_id[name].budget_group), name),
        )
        if not available:
            raise ValueError(f"campaign dependency cycle: {sorted(dependencies)}")
        for name in available:
            order.append(name)
            del dependencies[name]
        for parents in dependencies.values():
            parents.difference_update(available)
    return order


def node_profile(path: Path) -> dict[str, Any]:
    """Inspect resources without allocating a GPU or importing a model backend."""
    memory: dict[str, int] = {}
    status = Path("/proc/meminfo")
    if status.is_file():
        for line in status.read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                memory[key] = int(value.split()[0]) * 1024
    gpu = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        for line in result.stdout.splitlines():
            name, vram, driver = (part.strip() for part in line.split(","))
            gpu.append({"name": name, "vram_gb": int(vram) / 1024, "driver": driver})
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        ),
        "memory_bytes": memory,
        "scratch_free_bytes": shutil.disk_usage(path).free,
        "gpus": gpu,
        "heavy_gpu_workers": 1,
    }


def validate_baseline(lock: CampaignLock, root: Path) -> Any:
    """Bind R_v2 to its frozen configuration and unchanged historical R_0 parent."""
    from exact.experiments.schema import load_baseline
    from exact.experiments.harness import _assert_experiment_flags_disabled

    if lock.baseline_manifest is None:
        raise ValueError("R_v2 baseline manifest with historical parent R_0 is missing")
    path = lock.baseline_manifest.verify(root)
    baseline = load_baseline(path)
    if baseline.baseline_id != lock.baseline_id or baseline.parent != lock.historical_parent:
        raise ValueError(
            "campaign baseline identity or historical parent differs from its frozen manifest"
        )
    base = lock.base_config if lock.base_config.is_absolute() else root / lock.base_config
    frozen = baseline.config if baseline.config.is_absolute() else path.parent / baseline.config
    if sha256_file(base) != baseline.config_sha256 or sha256_file(frozen) != baseline.config_sha256:
        raise ValueError("campaign baseline configuration differs from its frozen bytes")
    _assert_experiment_flags_disabled(frozen)
    return baseline


def load_campaign(path: Path) -> tuple[CampaignLock, dict[str, Any]]:
    """Read and validate a lock against the unchanged normative blueprint."""
    lock = CampaignLock.model_validate(load_yaml_mapping(path))
    blueprint_path = lock.blueprint.verify(path.resolve().parent)
    design = load_yaml_mapping(blueprint_path)
    if design.get("design_schema_version") != 2 or design.get("kind") != "exact_om_campaign_design":
        raise ValueError("campaign blueprint is not a v2 scientific design")
    families = {item["id"]: item for item in design["experiments"]}
    seen: dict[str, int] = {}
    for step in lock.steps:
        if step.family not in families:
            raise ValueError(f"unknown experiment family: {step.family}")
        if step.phase == "initial":
            seen[step.family] = seen.get(step.family, 0) + len(step.arms)
    for family, count in seen.items():
        if count > families[family]["initial_treatment_cap"]:
            raise ValueError(f"{family}: {count} initial treatments exceed its frozen cap")
    if set(families) - {step.family for step in lock.steps}:
        raise ValueError(
            "campaign must account for every E00–E26 family, including unresolved paths"
        )
    return lock, dict(design)


def campaign_plan(path: Path, *, stage: str, verify_inputs: bool = True) -> dict[str, Any]:
    """Return a bounded work/readiness inventory without opening reporting labels."""
    if stage not in {"screen", "confirm"}:
        raise ValueError("public stages are screen and confirm")
    lock, blueprint = load_campaign(path)
    root = path.resolve().parent
    try:
        validate_baseline(lock, root)
        baseline_error = None
    except (OSError, ValueError) as exc:
        baseline_error = str(exc)
    limits = blueprint["profiles"][lock.profile]
    envelopes = dict(blueprint["budget_envelopes_core_hours"])
    if lock.profile == "extended_21d":
        for key, extra in blueprint["budget_envelopes_extended_extra_hours"].items():
            envelopes[key] += extra
    rows, blockers = [], []
    seconds = dict.fromkeys(ENVELOPES, 0.0)
    requests = tokens = 0
    lookup = {step.id: step for step in lock.steps}
    for step_id in dependency_order(lock.steps):
        step = lookup[step_id]
        if (step.phase == "final") != (stage == "confirm"):
            continue
        input_errors = []
        for case_id in [step.case, *step.additional_cases]:
            case = lock.cases[case_id]
            for name in ("source", "target", "source_universe"):
                binding = getattr(case, name)
                if binding is None:
                    input_errors.append(f"{case_id}: {name} binding missing")
                elif verify_inputs and stage == "screen":
                    try:
                        binding.verify(root)
                    except (OSError, ValueError) as exc:
                        input_errors.append(str(exc))
            role = "test" if stage == "confirm" else "valid"
            if any(
                role
                not in (
                    {**case.references, **case.local_references}
                    if mode == "local_ranking"
                    else case.references
                )
                for mode in step.execution_modes
            ):
                input_errors.append(f"{case_id}: {role} reference binding missing")
            if "local_ranking" in step.execution_modes and role not in case.candidates:
                input_errors.append(f"{case_id}: local ranking candidate binding missing")
            # Confirmation is validated after G4; planning never opens final gold.
            for binding in (
                *case.references.values(),
                *case.local_references.values(),
                *case.candidates.values(),
                *case.frozen_global_candidates.values(),
            ):
                if verify_inputs and stage == "screen":
                    try:
                        binding.verify(root)
                    except (OSError, ValueError) as exc:
                        input_errors.append(str(exc))
        if step.estimate:
            seconds[step.budget_group] += step.estimate.seconds()
            requests += step.estimate.requests
            tokens += step.estimate.tokens
        for arm in step.arms:
            readiness = step.readiness[arm.id].get(cast(Literal["screen", "confirm"], stage))
            status = readiness.status if readiness else "planned"
            issues = list(input_errors)
            if verify_inputs and stage == "screen":
                for binding in step.arm_inputs.get(arm.id, {}).values():
                    try:
                        binding.verify(root)
                    except (OSError, ValueError) as exc:
                        issues.append(str(exc))
            if baseline_error:
                issues.append(baseline_error)
            if status not in {"screen_ready", "confirm_ready", "complete"}:
                issues.append(readiness.reason if readiness else f"{stage} readiness missing")
            if step.estimate is None and status not in TERMINAL:
                issues.append("measured cold/warm resource forecast missing")
            if stage == "confirm" and lock.final_selection is None:
                issues.append("G4 frozen final selection missing")
            rows.append(
                {
                    "step": step.id,
                    "family": step.family,
                    "arm": arm.id,
                    "case": step.case,
                    "phase": step.phase,
                    "status": status,
                    "execution_modes": step.execution_modes,
                    "seeds": step.seeds,
                    "source_cap": step.source_cap,
                    "requires": step.requires,
                    "budget_group": step.budget_group,
                    "issues": sorted(set(issues)),
                }
            )
    for group, amount in seconds.items():
        if amount > envelopes[group] * 3600:
            blockers.append(f"{group} forecast exceeds {envelopes[group]} hours")
    if (
        requests + (lock.final_requests_reserved if stage == "screen" else 0)
        > limits["llm_request_planning_cap"]
    ):
        blockers.append("request forecast consumes protected final allocation")
    if (
        tokens + (lock.final_tokens_reserved if stage == "screen" else 0)
        > limits["llm_token_planning_cap"]
    ):
        blockers.append("token forecast consumes protected final allocation")
    return {
        "schema_version": 2,
        "campaign_id": lock.campaign_id,
        "stage": stage,
        "profile": lock.profile,
        "blueprint_sha256": lock.blueprint.sha256,
        "order": dependency_order(lock.steps),
        "rows": rows,
        "budget_errors": blockers,
        "forecast_hours": {key: value / 3600 for key, value in seconds.items()},
        "envelopes_hours": envelopes,
        "requests": requests,
        "tokens": tokens,
        "final_hours_reserved": limits["final_hours_reserved"],
        "recovery_reserve_hours": limits["recovery_reserve_hours"],
        "resource_policy": {
            "heavy_gpu_workers": 1,
            "cpu_workers": lock.max_cpu_workers,
            "openrouter_concurrency": lock.openrouter_concurrency,
        },
    }


def openrouter_only(mapping: Mapping[str, Any], profile: str) -> dict[str, Any]:
    """Bind every generative role to one hosted profile, with no local fallback."""
    from exact.core.entities.configs.config import ConfigModel
    from exact.experiments.harness import deep_merge

    config = ConfigModel.from_mapping(mapping, warn_v1=False)
    selected = config.llm.profiles.get(profile)
    if selected is None or selected.backend != "openrouter":
        raise ValueError("campaign generative profile must use OpenRouter")
    routing = config.llm.routing.model_dump()
    for key in routing:
        routing[key] = None if "fallback" in key else profile
    hosted = selected.model_dump(mode="json")
    hosted["provider"] = {**hosted.get("provider", {}), "allow_fallbacks": False}
    return deep_merge(
        config.model_dump(mode="json", by_alias=True),
        {"llm": {"routing": routing, "profiles": {profile: hosted}}},
    )


def _case_task(case: CaseBinding, case_id: str, mode: str, role: str, root: Path) -> dict[str, Any]:
    from exact.experiments.harness import deep_merge

    def locate(binding: Optional[InputBinding]) -> Optional[str]:
        if binding is None:
            return None
        return str((binding.path if binding.path.is_absolute() else root / binding.path).resolve())

    pools = case.candidates if mode == "local_ranking" else case.frozen_global_candidates
    references = dict(case.references)
    if mode == "local_ranking" and role in case.local_references:
        references[role] = case.local_references[role]
    data: dict[str, Any] = {
        "source": locate(case.source),
        "source_universe": locate(case.source_universe),
        "target": locate(case.target),
        "refs": {name: locate(binding) for name, binding in references.items()},
        "reference_role": role,
        "execution_mode": mode,
        "candidates": locate(pools.get(role)),
        "candidate_source": "track" if role in pools else "generated",
        "candidate_provenance": (
            ("benchmark_supplied" if mode == "local_ranking" else "frozen_generated")
            if role in pools
            else "generated"
        ),
    }
    if "train" in case.candidates:
        data["train_candidates"] = locate(case.candidates["train"])
    return {
        "id": f"{case_id}-{mode}",
        "split_role": case.role,
        "reference_role": role,
        "reference_completeness": case.reference_completeness,
        "capabilities": case.capabilities,
        "overlay": deep_merge(
            case.overlay,
            {
                "data": data,
                "matching": {"entity_kinds": [case.kind]},
                "supervision": {
                    "negative_label_policy": {
                        "complete_reference": "complete_reference",
                        "confirmed_only": "confirmed_negatives",
                        "positive_unlabelled": "unknown",
                    }[case.negative_policy]
                },
            },
        ),
    }


def materialize_campaign(path: Path, directory: Path, *, stage: str) -> Any:
    """Write strict declarations and load them into the existing harness.

    Incomplete arms are retained in the plan and never silently enabled. Each
    comparison executes only when all its treatments and controls are ready.
    """
    from exact.core.entities.configs.config import ConfigModel
    from exact.core.entities.configs.yaml_io import dump_yaml_document
    from exact.experiments.harness import (
        ExperimentSource,
        LoadedSuite,
        _write_json_once,
    )
    from exact.experiments.schema import ExperimentConfig

    lock, blueprint = load_campaign(path)
    root = path.resolve().parent
    base_path = lock.base_config if lock.base_config.is_absolute() else root / lock.base_config
    base = openrouter_only(load_yaml_mapping(base_path), lock.openrouter_profile)
    base = ConfigModel.from_mapping(base, warn_v1=False).model_dump(mode="json", by_alias=True)
    base["run"]["experiment_audit"] = True
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    resolved_base = directory / "base.config.yaml"

    def write_once(target: Path, value: Any) -> None:
        content = dump_yaml_document(value)
        if target.exists():
            if target.read_text() != content:
                raise ValueError(f"immutable campaign declaration changed: {target}")
        else:
            with target.open("x", encoding="utf-8") as stream:
                stream.write(content)

    write_once(resolved_base, base)
    plan = campaign_plan(path, stage=stage)
    final = (
        json.loads(lock.final_selection.verify(root).read_text())
        if stage == "confirm" and lock.final_selection
        else None
    )
    _write_json_once(
        directory / f"{stage}.plan.{digest(plan)[:16]}.json", plan, label="campaign plan"
    )
    sources = []
    by_id = {step.id: step for step in lock.steps}
    producers = {port: step.id for step in lock.steps for port in step.produces}
    for step_id in dependency_order(lock.steps):
        step = by_id[step_id]
        if (step.phase == "final") != (stage == "confirm"):
            continue
        case = lock.cases[step.case]
        rows = [row for row in plan["rows"] if row["step"] == step.id]
        problems = sorted({issue for row in rows for issue in row["issues"]})
        tasks = [
            _case_task(
                lock.cases[case_id], case_id, mode, "test" if stage == "confirm" else "valid", root
            )
            for case_id in [step.case, *step.additional_cases]
            for mode in step.execution_modes
        ]
        # The unused stage has no opposite-role file paths. The stage-specific
        # suite cannot expose final labels to a screen worker.
        screen_tasks = (
            tasks
            if stage == "screen"
            else [
                {
                    **task,
                    "split_role": "development",
                    "reference_role": "valid",
                    "overlay": {"data": {}},
                }
                for task in tasks
            ]
        )
        confirm_tasks = (
            tasks
            if stage == "confirm"
            else [
                {
                    **task,
                    "split_role": "reporting",
                    "reference_role": "test",
                    "overlay": {"data": {}},
                }
                for task in tasks
            ]
        )
        arms = [arm.model_dump(mode="json") for arm in step.arms]
        for arm in arms:
            if arm["id"] in step.arm_inputs:
                arm["overlay"]["data"] = {
                    name: str(binding.verify(root))
                    for name, binding in step.arm_inputs[arm["id"]].items()
                }
        declaration: dict[str, Any] = {
            "schema_version": 2,
            "experiment_id": step.id,
            "title": step.design.primary_comparison,
            "depends_on": [producers.get(item, item) for item in step.inherits],
            "base_config": str(resolved_base),
            "baseline_id": lock.baseline_id,
            "implementation": (
                {
                    "status": "blocked",
                    "reason_code": "campaign_not_ready",
                    "reason": "; ".join(problems),
                }
                if problems
                else {"status": "ready"}
            ),
            "resource": {"kind": "gpu", "device": "0", "concurrency": 1},
            "screen": {
                "tasks": screen_tasks,
                "seeds": step.seeds if stage == "screen" else [17],
                "source_cap": step.source_cap if stage == "screen" else 300,
            },
            "confirm": {
                "tasks": confirm_tasks,
                "seeds": step.seeds if stage == "confirm" else [17, 29, 43],
            },
            "arms": arms,
            "selection": step.selection.model_dump(mode="json"),
            "design": step.design.model_dump(mode="json"),
            "negative_label_policy": case.negative_policy,
            "frozen_constants": {
                "campaign_v2": {
                    "family": step.family,
                    "arm_inputs": {
                        arm: {
                            name: binding.model_dump(mode="json")
                            for name, binding in bindings.items()
                        }
                        for arm, bindings in step.arm_inputs.items()
                    },
                    "policy_paths": step.policy_paths,
                    "phase": step.phase,
                    "requires": [producers.get(item, item) for item in step.requires],
                    "produces": step.produces,
                    "budget_group": step.budget_group,
                    "estimate": step.estimate.model_dump(mode="json") if step.estimate else None,
                    "training_source_cap": step.training_source_cap,
                    "readiness": {
                        arm: (
                            statuses[cast(Literal["screen", "confirm"], stage)].status
                            if cast(Literal["screen", "confirm"], stage) in statuses
                            else "planned"
                        )
                        for arm, statuses in step.readiness.items()
                    },
                    "historical_parent": lock.historical_parent,
                    "blueprint_sha256": lock.blueprint.sha256,
                }
            },
        }
        if final is not None:
            frozen_arms = final["experiments"][step.id]["arms"]
            declaration["arms"] = [
                {**arm, "overlay": frozen_arms[arm["id"]]} for arm in declaration["arms"]
            ]
        config = ExperimentConfig.model_validate(declaration)
        config_path = directory / f"{step.id}.{digest(config.model_dump(mode="json"))[:16]}.yaml"
        write_once(config_path, config.model_dump(mode="json"))
        sources.append(ExperimentSource(config, config_path))
    model_payload = (
        dict(load_yaml_mapping(lock.model_lock.verify(root))) if lock.model_lock else None
    )
    return LoadedSuite(
        suite_id=lock.campaign_id,
        baseline_id=lock.baseline_id,
        sources=tuple(sources),
        suite_path=path.resolve(),
        suite_hash=campaign_identity(lock, root),
        dataset_lock=None,
        dataset_lock_hash=None,
        specification={
            "path": str(lock.blueprint.path),
            "sha256": lock.blueprint.sha256,
            "files": 1,
            "algorithm": "sha256-file-v1",
        },
        model_lock=lock.model_lock.path if lock.model_lock else None,
        model_lock_hash=lock.model_lock.sha256 if lock.model_lock else None,
        model_lock_payload=model_payload,
        baseline_manifest=lock.baseline_manifest.verify(root) if lock.baseline_manifest else None,
        baseline_manifest_hash=lock.baseline_manifest.sha256 if lock.baseline_manifest else None,
        baseline=validate_baseline(lock, root) if lock.baseline_manifest else None,
    )


def execute_campaign(
    path: Path,
    *,
    stage: str,
    output_root: Path,
    workdir: Path,
    jobs: int = 1,
    resume: bool = False,
    resume_from: Optional[Path] = None,
    repair_record: Optional[Path] = None,
    reuse_plan_only: bool = False,
    stop_after_checkpoint: bool = False,
) -> int:
    """Admit a v2 stage and use the shared runner with recovery-aware provenance."""
    from dataclasses import replace

    from exact.experiments.harness import run_stage

    lock, blueprint = load_campaign(path)
    plan = campaign_plan(path, stage=stage)
    if plan["budget_errors"]:
        raise ValueError("; ".join(plan["budget_errors"]))
    root = path.resolve().parent
    validate_baseline(lock, root)
    if lock.model_lock is None:
        raise ValueError("execution requires a materialized model/tokenizer revision lock")
    models = load_yaml_mapping(lock.model_lock.verify(root))
    if models.get("status") != "complete":
        raise ValueError("execution requires complete model/tokenizer bindings")
    ready = {row["step"] for row in plan["rows"] if not row["issues"]}
    incomplete = {row["step"] for row in plan["rows"] if row["issues"]}
    ready.difference_update(incomplete)
    if not ready and not reuse_plan_only:
        raise ValueError("no complete comparison is ready; inspect --dry-run for per-arm reasons")
    if stage == "confirm":
        if incomplete:
            raise ValueError(f"frozen final panel is not fully ready: {sorted(incomplete)}")
        if lock.final_selection is None:
            raise ValueError("confirmation requires the immutable G4 final selection")
        selection_path = lock.final_selection.verify(root)
    else:
        selection_path = None
    directory = output_root / lock.campaign_id / "declarations" / stage
    suite = materialize_campaign(path, directory, stage=stage)
    continuation = [
        sys.executable,
        str(workdir / "tools/run_experiment.py"),
        "--campaign",
        str(path.resolve()),
        "--stage",
        stage,
        "--output-root",
        str(output_root.resolve()),
        "--resume",
    ]
    if repair_record:
        continuation.extend(["--repair-record", str(repair_record.resolve())])
    metadata: dict[str, Any] = {
        "lock_path": str(path.resolve()),
        "continuation_command": shlex.join(continuation),
        "root": str((output_root / lock.campaign_id).resolve()),
        "resume_from": str(resume_from.resolve()) if resume_from else None,
        "repair_record": str(repair_record.resolve()) if repair_record else None,
        "stop_after_checkpoint": stop_after_checkpoint,
        "reuse_plan_only": reuse_plan_only,
        "stage": stage,
        "selection_record": str(selection_path) if selection_path else None,
        "budget_limits": {
            "envelopes_hours": plan["envelopes_hours"],
            "node_hours_cap": blueprint["profiles"][lock.profile]["node_hours_cap"],
            "requests_cap": blueprint["profiles"][lock.profile]["llm_request_planning_cap"],
            "tokens_cap": blueprint["profiles"][lock.profile]["llm_token_planning_cap"],
            "final_requests_reserved": lock.final_requests_reserved,
            "final_tokens_reserved": lock.final_tokens_reserved,
        },
        "allowed_steps": sorted(ready),
        "profile": lock.profile,
        "cpu_workers": min(jobs, lock.max_cpu_workers),
        "openrouter_concurrency": lock.openrouter_concurrency,
    }
    suite = replace(suite, campaign=metadata)
    stop_context = (
        nullcontext()
        if reuse_plan_only
        else cooperative_signals(Path(metadata["root"]), resume=resume or resume_from is not None)
    )
    with stop_context:
        selection_result = run_stage(
            suite,
            stage=stage,
            output_root=output_root,
            jobs=min(jobs, lock.max_cpu_workers + 1),
            resume=resume or resume_from is not None,
            workdir=workdir,
            selection_record_path=selection_path,
            dry_run=False,
        )
        if stage == "screen" and lock.freeze_step and selection_result and not reuse_plan_only:
            selected = json.loads(selection_result.read_text())
            if selected.get("experiments", {}).get(lock.freeze_step, {}).get("status") in {
                "selected",
                "screened_out",
            }:
                freeze_final_selection(
                    path, selection_result, Path(metadata["root"]) / "G4.selection.json"
                )
    return 0


def load_progress(suite: Any, stage: str) -> dict[str, Any]:
    """Read only signed development progress; cell artifacts are revalidated on resume."""
    root = Path(suite.campaign.get("resume_from") or suite.campaign["root"])
    path = root / stage / "progress.json"
    if not path.is_file():
        return {}
    record = json.loads(path.read_text())
    claimed = record.pop("progress_hash", None)
    if claimed != digest(record) or record.get("suite_hash") != suite.suite_hash:
        raise ValueError("partial campaign progress has changed or belongs to another design")
    if record.get("stage") != stage:
        raise ValueError("partial campaign progress has the wrong stage")
    experiments = record.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError("partial campaign experiments must be an object")
    return dict(experiments)


def write_progress(
    suite: Any, stage: str, experiments: Mapping[str, Any], manifests: Any, *, status: str
) -> None:
    """Retain immutable progress snapshots and a replaceable continuation pointer."""
    from exact.experiments.harness import _atomic_json, _write_json_once

    root = Path(suite.campaign["root"]) / stage
    record = {
        "schema_version": 2,
        "suite_hash": suite.suite_hash,
        "stage": stage,
        "status": status,
        "experiments": dict(experiments),
        "declarations": {
            source.config.experiment_id: {"path": str(source.path), "sha256": source.raw_hash()}
            for source in suite.sources
        },
        "cells": [
            {
                key: item.get(key)
                for key in ("experiment_id", "arm_id", "task_id", "seed", "status", "recovery")
            }
            for item in manifests
        ],
    }
    record["progress_hash"] = digest(record)
    _write_json_once(
        root / "progress" / f"{record['progress_hash']}.json", record, label="campaign progress"
    )
    _atomic_json(root / "progress.json", record)


def dependency_blockers(source: Any, selections: Mapping[str, Any]) -> list[str]:
    """A null result publishes its baseline policy; unfinished comparisons publish nothing."""
    spec = source.config.frozen_constants.get("campaign_v2", {})
    return [
        name
        for name in spec.get("requires", [])
        if selections.get(name, {}).get("status")
        not in {
            "selected",
            "complete",
            "screened_out",
            "inapplicable",
            "deferred_budget",
            "blocked_input_resolution",
        }
    ]


def run_comparison(cells: Any, suite: Any, source: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Reserve complete comparisons and retain actual cumulative costs on every exit."""
    import time
    import uuid

    from exact.experiments.budget import BudgetLedger
    from exact.experiments.harness import run_cells
    from exact.llm.ledger import RequestLedger

    metadata = suite.campaign
    if metadata.get("reuse_plan_only"):
        return run_cells(cells, suite, **kwargs)
    spec = source.config.frozen_constants["campaign_v2"]
    estimate = WorkEstimate.model_validate(spec["estimate"])
    root = Path(metadata["root"])
    account = BudgetLedger(root / "budget.json", metadata["budget_limits"])
    requests = RequestLedger(root / "openrouter")

    def usage() -> dict[str, float]:
        roles = requests.summary()["roles"].values()
        return {
            field: sum(role[field] for role in roles)
            for field in (
                "attempts",
                "prompt_tokens",
                "completion_tokens",
                "reported_cost_usd",
                "unpriced_attempts",
                "unknown",
            )
        }

    work_id = f"{metadata['stage']}/{source.config.experiment_id}/{uuid.uuid4().hex}"
    account.admit(
        work_id,
        group=spec["budget_group"],
        seconds=estimate.seconds(),
        requests=estimate.requests,
        tokens=estimate.tokens,
        projected_usd=estimate.projected_usd,
    )
    before, start, status = usage(), time.time(), "failed"
    try:
        manifests = run_cells(cells, suite, **kwargs)
        states = {item.get("status") for item in manifests}
        status = (
            "complete"
            if states == {"complete"}
            else "interrupted" if "interrupted" in states else "failed"
        )
        return manifests
    finally:
        after = usage()
        delta = {key: after[key] - before[key] for key in before}
        # Unknown delivery retains the reserved token exposure instead of becoming zero cost.
        tokens = int(delta["prompt_tokens"] + delta["completion_tokens"])
        if delta["unknown"]:
            tokens = max(tokens, estimate.tokens)
        account.finish(
            work_id,
            start=start,
            end=time.time(),
            status=status,
            requests=int(delta["attempts"]),
            tokens=tokens,
            actual_usd=None if delta["unpriced_attempts"] else delta["reported_cost_usd"],
        )


@contextmanager
def cooperative_signals(root: Path, *, resume: bool = False) -> Iterator[None]:
    """Stop scheduling on SIGINT/TERM; workers finish their next durable boundary."""
    import signal
    import threading
    import time

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("campaign execution must install signal handlers on the main thread")
    root.mkdir(parents=True, exist_ok=True)
    stop = root / "STOP"
    if stop.exists():
        if not resume:
            raise ValueError("campaign has a retained STOP request; use --resume to continue")
        stop.rename(root / f"STOP.resumed-{time.time_ns()}")

    def request_stop(signum: int, frame: Any) -> None:
        if not stop.exists():
            with stop.open("x") as stream:
                stream.write(f"signal={signum} time={time.time()}\n")
                stream.flush()
                os.fsync(stream.fileno())

    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        for signum in previous:
            signal.signal(signum, request_stop)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def campaign_identity(lock: CampaignLock, root: Path) -> str:
    """Scientific design identity excludes readiness updates and movable file locators."""
    value = lock.model_dump(mode="json", exclude={"final_selection"})
    for step in value["steps"]:
        step.pop("readiness")
    base = lock.base_config if lock.base_config.is_absolute() else root / lock.base_config
    value["base_config"] = {"sha256": sha256_file(base)}

    def content(item: Any) -> Any:
        if isinstance(item, dict):
            if set(item) == {"path", "sha256"}:
                return {"sha256": item["sha256"]}
            return {key: content(val) for key, val in item.items()}
        if isinstance(item, list):
            return [content(val) for val in item]
        return item

    return digest(content(value))


def compose_development(source: Any, lock: CampaignLock, selections: Mapping[str, Any]) -> Any:
    """Resolve the declared core composition while retaining its unchanged baseline arm."""
    from dataclasses import replace

    from exact.experiments.harness import deep_merge, selected_experiment_overlays

    if source.config.experiment_id != lock.freeze_step:
        return source
    components = selected_experiment_overlays({"experiments": selections}, lock.composition_sources)
    core: dict[str, Any] = {}
    for overlay in components.values():
        core = deep_merge(core, overlay)
    arms = []
    for arm in source.config.arms:
        overlay = deep_merge(core, arm.overlay) if arm.id in {"core", "optional"} else arm.overlay
        arms.append(arm.model_copy(update={"overlay": overlay}))
    from exact.core.entities.configs.yaml_io import dump_yaml_document

    config = source.config.model_copy(update={"arms": arms, "depends_on": []})
    payload = config.model_dump(mode="json")
    path = source.path.parent / f"{config.experiment_id}.resolved.{digest(payload)[:16]}.yaml"
    content = dump_yaml_document(payload)
    if path.exists() and path.read_text() != content:
        raise ValueError(f"resolved composition changed: {path}")
    if not path.exists():
        with path.open("x") as stream:
            stream.write(content)
    return replace(source, config=config, path=path)


def freeze_final_selection(path: Path, selection_path: Path, destination: Path) -> dict[str, Any]:
    """Mechanically freeze the declared final panel from completed development evidence."""
    from exact.experiments.harness import (
        _write_json_once,
        deep_merge,
        selected_experiment_overlays,
    )

    lock, blueprint = load_campaign(path)
    root = path.resolve().parent
    selected = json.loads(selection_path.read_text())
    claimed = selected.pop("selection_hash", None)
    if claimed != digest(selected) or selected.get("stage") != "screen":
        raise ValueError("G4 requires a verified development selection record")
    if selected.get("suite_hash") != campaign_identity(lock, root):
        raise ValueError("G4 evidence belongs to a different scientific design")
    experiments = selected.get("experiments", {})
    gate = experiments.get(lock.freeze_step)
    if not gate or gate.get("status") not in {"selected", "screened_out"}:
        raise ValueError("G4 composition/sentinel comparison has not completed")
    for step in lock.steps:
        if step.phase == "final":
            continue
        status = experiments.get(step.id, {}).get("status")
        declared_terminal = all(
            step.readiness[arm.id].get("screen")
            and step.readiness[arm.id]["screen"].status
            in {"inapplicable", "deferred_budget", "blocked_input_resolution"}
            for arm in step.arms
        )
        if (
            status
            not in {
                "selected",
                "complete",
                "screened_out",
                "inapplicable",
                "deferred_budget",
                "blocked_input_resolution",
            }
            and not declared_terminal
        ):
            raise ValueError(
                f"G4 waits for an empirical result or explicit input/capability/budget disposition: {step.id}"
            )
    gate_step = next(step for step in lock.steps if step.id == lock.freeze_step)
    if len([gate_step.case, *gate_step.additional_cases]) < 2:
        raise ValueError("G4 requires the primary development case and a declared sentinel")
    components = selected_experiment_overlays(selected, lock.composition_sources)
    main = gate.get("combined_selected_overlay") or {}
    composed_components: dict[str, Any] = {}
    for overlay in components.values():
        composed_components = deep_merge(composed_components, overlay)

    def amendments(selected: Mapping[str, Any], inherited: Mapping[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in selected.items():
            prior = inherited.get(key)
            if isinstance(value, Mapping) and isinstance(prior, Mapping):
                change = amendments(value, prior)
                if change:
                    result[key] = change
            elif key not in inherited or value != prior:
                result[key] = value
        return result

    gate_amendments = amendments(main, composed_components)
    panel: dict[str, Any] = {}
    count = 0
    for step in lock.steps:
        if step.phase != "final":
            continue
        if step.seeds != [17, 29, 43] or step.source_cap is not None:
            raise ValueError("final recipes require full populations and paired seeds 17,29,43")
        arm_ids = {arm.id for arm in step.arms}
        if not {"baseline", "stack_all"}.issubset(arm_ids):
            raise ValueError("every final case requires baseline and stack_all")
        if (
            main.get("supervision", {}).get("mode") not in {None, "label_free"}
            or any(
                value != "label_free"
                for value in main.get("supervision", {}).get("components", {}).values()
            )
        ) and "label_free" not in arm_ids:
            raise ValueError("supervised final stack requires its label-free control")
        overlays = {}
        for arm in step.arms:
            if arm.role == "oracle" or not arm.deployable or "confirm" not in arm.stages:
                raise ValueError("final panel contains a development-only arm")
            composed = dict(main)
            if arm.id == "baseline":
                composed = {}
            elif arm.id.startswith("stack_minus_"):
                component = arm.id.removeprefix("stack_minus_")
                if component not in components or not main:
                    raise ValueError("attribution must remove an actually selected component")
                composed = {}
                for name, overlay in components.items():
                    if name != component:
                        composed = deep_merge(composed, overlay)
                composed = deep_merge(composed, gate_amendments)
            overlays[arm.id] = deep_merge(composed, arm.overlay)
        cases = [step.case, *step.additional_cases]
        for case_id in cases:
            case = lock.cases[case_id]
            if case.source_universe is None or "test" not in case.references:
                raise ValueError(
                    "final populations and reference identities must be bound before G4"
                )
        panel[step.id] = {
            "status": "selected",
            "arms": overlays,
            "cases": cases,
            "design": step.design.model_dump(mode="json"),
            "seeds": step.seeds,
            "execution_modes": step.execution_modes,
            "population_hashes": {
                case_id: {
                    "source_universe": cast(
                        InputBinding, lock.cases[case_id].source_universe
                    ).sha256,
                    "reporting_reference": lock.cases[case_id].references["test"].sha256,
                    "local_reference": (
                        lock.cases[case_id].local_references.get("test")
                        or lock.cases[case_id].references["test"]
                    ).sha256,
                }
                for case_id in cases
            },
        }
        count += len(overlays) * len(cases) * len(step.execution_modes)
    if not panel or count > blueprint["profiles"][lock.profile]["final_arm_task_design_cap"]:
        raise ValueError("final panel is empty or exceeds the frozen arm/task cap")
    record = {
        "schema_version": 2,
        "kind": "exact_om_final_selection",
        "stage": "screen",
        "suite_id": lock.campaign_id,
        "suite_hash": campaign_identity(lock, root),
        "development_selection_sha256": sha256_file(selection_path),
        "development_selection_hash": claimed,
        "gate": lock.freeze_step,
        "component_overlays": components,
        "experiments": panel,
        "final_arm_task_count": count,
        "no_final_outcomes_consumed": True,
    }
    record["selection_hash"] = digest(record)
    _write_json_once(destination, record, label="G4 final selection")
    return record


def validate_final_selection(record: Mapping[str, Any], suite: Any) -> dict[str, Any]:
    """Validate immutable final recipes against this campaign before opening test labels."""
    value = dict(record)
    claimed = value.pop("selection_hash", None)
    if (
        claimed != digest(value)
        or value.get("kind") != "exact_om_final_selection"
        or value.get("no_final_outcomes_consumed") is not True
    ):
        raise ValueError("invalid G4 selection identity")
    if value.get("suite_hash") != suite.suite_hash or value.get("suite_id") != suite.suite_id:
        raise ValueError("final selection belongs to a different campaign design")
    expected = {source.config.experiment_id for source in suite.sources}
    if set(value.get("experiments", {})) != expected:
        raise ValueError("final selection recipe set changed")
    lock, _ = load_campaign(Path(suite.suite_path))
    steps = {step.id: step for step in lock.steps}
    root = Path(suite.suite_path).resolve().parent
    count = 0
    for source in suite.sources:
        config = source.config
        step = steps[config.experiment_id]
        panel = value["experiments"][config.experiment_id]
        actual = {arm.id: arm.overlay for arm in config.arms}
        if panel["arms"] != actual:
            raise ValueError("final treatment changed after G4")
        if (
            config.confirm.source_cap is not None
            or config.confirm.seeds != [17, 29, 43]
            or panel["seeds"] != config.confirm.seeds
        ):
            raise ValueError("final seeds or full-population contract changed after G4")
        cases = [step.case, *step.additional_cases]
        if (
            panel["cases"] != cases
            or panel["execution_modes"] != step.execution_modes
            or panel["design"] != config.design.model_dump(mode="json")
        ):
            raise ValueError("final case/mode/design panel changed after G4")
        expected_tasks = [
            _case_task(lock.cases[case], case, mode, "test", root)
            for case in cases
            for mode in step.execution_modes
        ]
        if len(config.confirm.tasks) != len(expected_tasks):
            raise ValueError("final task panel changed after G4")
        for task, expected_task in zip(config.confirm.tasks, expected_tasks):
            if (
                task.id != expected_task["id"]
                or task.split_role != "reporting"
                or task.reference_role != "test"
                or task.source_cap is not None
                or task.overlay != expected_task["overlay"]
            ):
                raise ValueError("final task population or reporting input changed after G4")
        populations = {
            case: {
                "source_universe": cast(InputBinding, lock.cases[case].source_universe).sha256,
                "reporting_reference": lock.cases[case].references["test"].sha256,
                "local_reference": (
                    lock.cases[case].local_references.get("test")
                    or lock.cases[case].references["test"]
                ).sha256,
            }
            for case in cases
        }
        if panel.get("population_hashes") != populations:
            raise ValueError("final source population identity changed after G4")
        if any(
            arm.role == "oracle" or not arm.deployable or "confirm" not in arm.stages
            for arm in config.arms
        ):
            raise ValueError("final panel contains a development-only arm")
        count += len(actual) * len(expected_tasks)
    if count != value.get("final_arm_task_count"):
        raise ValueError("final arm/task count changed after G4")
    value["selection_hash"] = claimed
    return value
