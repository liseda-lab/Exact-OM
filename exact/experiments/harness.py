"""Execution, freezing, provenance, and aggregation for paper experiments."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

from exact.core.entities.configs.config import ConfigModel
from exact.core.entities.configs.yaml_io import dump_yaml_document, load_yaml_mapping
from exact.core.entities.kinds import EntityKind
from exact.experiments.paper_metrics import (
    RecomputedEvaluation,
    SourceEvaluation,
    e17_interaction_bootstrap,
    extract_evaluation_metrics,
    holm_adjust_p_values,
    normalize_relation,
    paired_global_f1_bootstrap,
    recompute_global_prf,
)
from exact.experiments.reporting import (
    enrich_inventory_from_manifests,
    inspect_dataset_task,
    metric_reports,
    stable_bootstrap_seed,
)
from exact.experiments.schema import (
    ArmConfig,
    BaselineManifest,
    ExperimentConfig,
    ResourceConfig,
    TaskConfig,
    load_baseline,
    load_experiment,
    load_suite,
    suite_order,
)
from exact.utils.data import read_table
from exact.utils.provenance import file_provenance, sha256_file

SCHEMA_VERSION = 1
MANIFEST_NAME = "experiment_manifest.json"
_RESERVED_PIPELINE_PARAMS = {
    "anchor_rescoring_config",
    "experiment_config",
    "fusion_config",
    "llm_experiment_config",
}
_COMPONENTS = (
    "retrieval",
    "fusion",
    "rerank",
    "llm",
    "accept",
    "calibration",
    "structure",
    "relation",
)
_SCREEN_REFERENCE_ROLES = {
    "dev",
    "development",
    "diagnostic",
    "valid",
    "validation",
}
_SCREEN_RETAINED_REFERENCE_ROLES = _SCREEN_REFERENCE_ROLES | {"train", "training"}
_LOCK_TRACK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json", by_alias=True))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive mapping merge; lists and scalars replace atomically."""

    merged: dict[str, Any] = {str(key): _jsonable(value) for key, value in base.items()}
    for key, value in overlay.items():
        key = str(key)
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = _jsonable(value)
    return merged


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write heterogeneous research rows with deterministic JSON cells."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: (
                            canonical_json(row.get(key))
                            if isinstance(row.get(key), (Mapping, list, tuple))
                            else row.get(key)
                        )
                        for key in fieldnames
                    }
                )
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(value: Path, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


@dataclass(frozen=True)
class ExperimentSource:
    config: ExperimentConfig
    path: Path

    @property
    def directory(self) -> Path:
        return self.path.parent

    @property
    def base_config_path(self) -> Path:
        return _resolve_path(self.config.base_config, self.directory)

    def raw_hash(self) -> str:
        return sha256_file(self.path)


@dataclass(frozen=True)
class LoadedSuite:
    suite_id: str
    baseline_id: str
    sources: tuple[ExperimentSource, ...]
    suite_path: Optional[Path]
    suite_hash: str
    dataset_lock: Optional[Path]
    dataset_lock_hash: Optional[str]
    baseline_manifest: Optional[Path] = None
    baseline_manifest_hash: Optional[str] = None
    baseline: Optional[BaselineManifest] = None

    @property
    def by_id(self) -> dict[str, ExperimentSource]:
        return {source.config.experiment_id: source for source in self.sources}


@dataclass(frozen=True)
class RunCell:
    suite_id: str
    experiment_id: str
    stage: str
    arm_id: str
    arm_role: str
    task_id: str
    split_role: str
    reference_role: str
    seed: int
    source_cap: Optional[int]
    resource: ResourceConfig
    output_dir: Path
    resolved_config: dict[str, Any]
    config_hash: str
    experiment_config_hash: str
    design_hash: str
    selection_hash: Optional[str]
    supervision_label: str
    resolved_supervision: dict[str, Any]
    negative_label_policy: str

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / MANIFEST_NAME

    @property
    def cell_id(self) -> str:
        return "/".join(
            (
                self.experiment_id,
                self.stage,
                self.arm_id,
                self.task_id,
                f"seed-{self.seed}",
            )
        )

    def dry_run_row(self) -> dict[str, Any]:
        return {
            "cell": self.cell_id,
            "resource": self.resource.serialization_key(),
            "output": str(self.output_dir),
            "config_hash": self.config_hash,
            "supervision": self.supervision_label,
            "source_cap": self.source_cap,
        }


def _baseline_record(suite: LoadedSuite) -> Optional[dict[str, Any]]:
    if suite.baseline_manifest is None:
        return None
    return {
        "path": str(suite.baseline_manifest),
        "sha256": suite.baseline_manifest_hash,
        "baseline_id": suite.baseline_id,
    }


def _repository_root(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    raise ValueError(f"cannot locate repository root for frozen baseline config {resolved}")


def _git_is_ancestor(workdir: Path, ancestor: str, descendant: str) -> bool:
    """Return whether ``ancestor`` is in ``descendant``'s local history."""

    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=workdir,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def _validate_baseline_manifest(
    path: Path,
    *,
    expected_id: str,
    sources: Sequence[ExperimentSource],
) -> BaselineManifest:
    baseline = load_baseline(path)
    if baseline.baseline_id != expected_id:
        raise ValueError(
            f"baseline manifest id {baseline.baseline_id!r} does not match {expected_id!r}"
        )
    config_path = _resolve_path(baseline.config, path.parent)
    if not config_path.is_file():
        raise FileNotFoundError(f"frozen baseline config does not exist: {config_path}")
    observed_config_hash = sha256_file(config_path)
    if observed_config_hash != baseline.config_sha256:
        raise ValueError(
            "frozen baseline config hash mismatch: "
            f"{observed_config_hash} != {baseline.config_sha256}"
        )
    for source in sources:
        if source.config.baseline_id != expected_id:
            raise ValueError(
                f"{source.config.experiment_id}: baseline id {source.config.baseline_id!r} "
                f"does not match suite baseline {expected_id!r}"
            )
        if source.base_config_path != config_path:
            raise ValueError(
                f"{source.config.experiment_id}: base config {source.base_config_path} does not "
                f"match frozen baseline config {config_path}"
            )
    package_names = {name.lower(): version for name, version in _package_versions().items()}
    observed_versions = {
        "exact-om": package_names.get("exact-om"),
        "pyowl-core": package_names.get("pyowl-core"),
    }
    expected_versions = {
        "exact-om": baseline.exact_om_version,
        "pyowl-core": baseline.pyowlcore_version,
    }
    for distribution, expected in expected_versions.items():
        observed = observed_versions[distribution]
        if observed != expected:
            raise ValueError(
                f"frozen baseline requires {distribution} {expected}, observed {observed!r}"
            )
    repository = _repository_root(config_path)
    git = _git_provenance(repository)
    observed_commit = git.get("commit")
    if observed_commit != baseline.source_commit and (
        not isinstance(observed_commit, str)
        or not _git_is_ancestor(repository, baseline.source_commit, observed_commit)
    ):
        raise ValueError(
            "frozen baseline source commit is not the current commit or an ancestor: "
            f"{baseline.source_commit!r} versus {observed_commit!r}"
        )
    tree = git.get("source_tree")
    if not isinstance(tree, Mapping):
        raise ValueError("cannot compute frozen baseline source-tree identity")
    if (
        tree.get("sha256") != baseline.source_tree_sha256
        or tree.get("files") != baseline.source_tree_files
    ):
        raise ValueError(
            "frozen baseline source-tree mismatch: "
            f"{tree!r} != {{'sha256': {baseline.source_tree_sha256!r}, "
            f"'files': {baseline.source_tree_files}}}"
        )
    return baseline


def _validate_dataset_lock(path: Path) -> dict[str, Any]:
    payload = dict(load_yaml_mapping(path))
    tracks = payload.get("tracks")
    if payload.get("schema_version") != 1 or not isinstance(tracks, Mapping) or not tracks:
        raise ValueError(f"invalid dataset lock schema at {path}")
    for track, raw_entry in tracks.items():
        if (
            not isinstance(track, str)
            or track != track.strip()
            or not _LOCK_TRACK_RE.fullmatch(track)
        ):
            raise ValueError(f"dataset lock has invalid track name {track!r}")
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"dataset lock track {track!r} must be a mapping")
        descriptor = raw_entry.get("descriptor")
        expected_hash = raw_entry.get("descriptor_sha256")
        roles = raw_entry.get("roles")
        if not descriptor or not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError(
                f"dataset lock track {track!r} must declare descriptor and descriptor_sha256"
            )
        descriptor_path = _resolve_path(Path(str(descriptor)), path.parent)
        if not descriptor_path.is_file():
            raise FileNotFoundError(
                f"dataset lock descriptor does not exist for {track!r}: {descriptor_path}"
            )
        observed_hash = sha256_file(descriptor_path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"dataset lock descriptor hash mismatch for {track!r}: "
                f"{observed_hash} != {expected_hash}"
            )
        descriptor_payload = load_yaml_mapping(descriptor_path)
        descriptor_name = descriptor_payload.get("name")
        if str(descriptor_name or "") != str(track):
            raise ValueError(
                f"dataset lock track {track!r} points to descriptor named " f"{descriptor_name!r}"
            )
        if "revision" in raw_entry:
            expected_revision = raw_entry.get("revision")
            upstream = descriptor_payload.get("upstream")
            observed_revision = upstream.get("revision") if isinstance(upstream, Mapping) else None
            if (
                expected_revision is None
                or observed_revision is None
                or str(observed_revision) != str(expected_revision)
            ):
                raise ValueError(
                    f"dataset lock revision mismatch for {track!r}: "
                    f"{observed_revision!r} != {expected_revision!r}"
                )
        if (
            not isinstance(roles, list)
            or not roles
            or any(not str(role).strip() for role in roles)
            or len({str(role) for role in roles}) != len(roles)
        ):
            raise ValueError(f"dataset lock track {track!r} has invalid split roles")
    return payload


def _validate_task_lock_roles(
    sources: Sequence[ExperimentSource],
    lock: Mapping[str, Any],
) -> None:
    tracks = lock.get("tracks")
    if not isinstance(tracks, Mapping):
        raise ValueError("dataset lock has no track declarations")
    for source in sources:
        for stage_name, stage in (
            ("screen", source.config.screen),
            ("confirm", source.config.confirm),
        ):
            for task in stage.tasks:
                if task.track is None:
                    continue
                entry = tracks.get(task.track)
                if not isinstance(entry, Mapping):
                    raise ValueError(
                        f"{source.config.experiment_id}/{stage_name}/{task.id}: "
                        f"track {task.track!r} is absent from the dataset lock"
                    )
                roles = {str(role) for role in entry.get("roles") or ()}
                if task.reference_role not in roles:
                    raise ValueError(
                        f"{source.config.experiment_id}/{stage_name}/{task.id}: reference role "
                        f"{task.reference_role!r} is not locked for track {task.track!r}"
                    )


def load_suite_or_experiment(
    *,
    suite_path: Optional[Path] = None,
    experiment_path: Optional[Path] = None,
) -> LoadedSuite:
    if (suite_path is None) == (experiment_path is None):
        raise ValueError("provide exactly one of suite_path or experiment_path")
    if experiment_path is not None:
        path = Path(experiment_path).expanduser().resolve()
        config = load_experiment(path)
        collection_root = (
            path.parent.parent if path.parent.name == config.experiment_id else path.parent
        )
        single_lock_path = collection_root / "datasets.lock.yaml"
        if not single_lock_path.is_file():
            raise FileNotFoundError(
                f"standard sibling dataset lock does not exist: {single_lock_path}"
            )
        single_lock = _validate_dataset_lock(single_lock_path)
        source = ExperimentSource(config, path)
        single_sources = (source,)
        _validate_task_lock_roles(single_sources, single_lock)
        baseline_path = collection_root / "baselines" / f"{config.baseline_id}.yaml"
        if not baseline_path.is_file():
            raise FileNotFoundError(
                f"standard sibling baseline manifest does not exist: {baseline_path}"
            )
        baseline = _validate_baseline_manifest(
            baseline_path,
            expected_id=config.baseline_id,
            sources=single_sources,
        )
        single_lock_hash = sha256_file(single_lock_path)
        baseline_hash = sha256_file(baseline_path)
        suite_hash = hash_payload(
            {
                "experiment_manifest_sha256": sha256_file(path),
                "baseline_manifest_sha256": baseline_hash,
                "dataset_lock_sha256": single_lock_hash,
            }
        )
        return LoadedSuite(
            suite_id=f"single-{config.experiment_id.lower()}",
            baseline_id=config.baseline_id,
            sources=single_sources,
            suite_path=None,
            suite_hash=suite_hash,
            dataset_lock=single_lock_path,
            dataset_lock_hash=single_lock_hash,
            baseline_manifest=baseline_path,
            baseline_manifest_hash=baseline_hash,
            baseline=baseline,
        )

    assert suite_path is not None
    path = Path(suite_path).expanduser().resolve()
    suite = load_suite(path)
    entries = {entry.id: entry for entry in suite.experiments}
    sources: list[ExperimentSource] = []
    for experiment_id in suite_order(suite):
        entry = entries[experiment_id]
        config_path = _resolve_path(entry.config, path.parent)
        config = load_experiment(config_path)
        if config.experiment_id != entry.id:
            raise ValueError(f"suite entry {entry.id} points to config for {config.experiment_id}")
        if config.baseline_id != suite.baseline_id:
            raise ValueError(
                f"{entry.id}: baseline id {config.baseline_id!r} disagrees with "
                f"suite baseline {suite.baseline_id!r}"
            )
        declared = set(config.depends_on)
        listed = set(entry.depends_on)
        if declared != listed:
            raise ValueError(
                f"{entry.id}: suite dependencies {sorted(listed)} disagree with "
                f"experiment dependencies {sorted(declared)}"
            )
        sources.append(ExperimentSource(config, config_path))
    lock_path = _resolve_path(suite.dataset_lock, path.parent) if suite.dataset_lock else None
    if lock_path is not None and not lock_path.is_file():
        raise FileNotFoundError(f"declared dataset lock does not exist: {lock_path}")
    lock = _validate_dataset_lock(lock_path) if lock_path is not None else None
    if lock is not None:
        _validate_task_lock_roles(sources, lock)
    lock_hash = sha256_file(lock_path) if lock_path is not None else None
    baseline_path = _resolve_path(suite.baseline_manifest, path.parent)
    if not baseline_path.is_file():
        raise FileNotFoundError(f"declared baseline manifest does not exist: {baseline_path}")
    baseline = _validate_baseline_manifest(
        baseline_path,
        expected_id=suite.baseline_id,
        sources=sources,
    )
    baseline_hash = sha256_file(baseline_path)
    suite_hash = hash_payload(
        {
            "suite_manifest_sha256": sha256_file(path),
            "baseline_manifest_sha256": baseline_hash,
            "dataset_lock_sha256": lock_hash,
        }
    )
    return LoadedSuite(
        suite_id=suite.suite_id,
        baseline_id=suite.baseline_id,
        sources=tuple(sources),
        suite_path=path,
        suite_hash=suite_hash,
        dataset_lock=lock_path,
        dataset_lock_hash=lock_hash,
        baseline_manifest=baseline_path,
        baseline_manifest_hash=baseline_hash,
        baseline=baseline,
    )


def _validate_overlay_surface(overlay: Mapping[str, Any], label: str) -> None:
    pipeline = overlay.get("pipeline")
    if pipeline is None:
        return
    if not isinstance(pipeline, list):
        raise ValueError(f"{label}: pipeline overlay must be a list")
    for entry in pipeline:
        if not isinstance(entry, Mapping):
            continue
        params = entry.get("params")
        if not isinstance(params, Mapping):
            continue
        reserved = sorted(set(params).intersection(_RESERVED_PIPELINE_PARAMS))
        legacy = sorted(set(params).intersection({"tau", "gamma", "beta", "tau_LLM"}))
        if reserved or legacy:
            names = reserved + legacy
            raise ValueError(
                f"{label}: experiment controls {names} must use matching.*, selector.*, "
                "llm.*, or candidates.*, not pipeline params"
            )


def _base_mapping(source: ExperimentSource) -> dict[str, Any]:
    return dict(load_yaml_mapping(source.base_config_path))


def _dataset_lock_roles(suite: LoadedSuite) -> dict[str, tuple[str, ...]]:
    if suite.dataset_lock is None:
        return {}
    payload = _validate_dataset_lock(suite.dataset_lock)
    tracks = payload.get("tracks")
    if not isinstance(tracks, Mapping):
        return {}
    return {
        str(track): tuple(str(role) for role in (entry.get("roles") or ()))
        for track, entry in tracks.items()
        if isinstance(entry, Mapping)
    }


def _inventory_config(source: ExperimentSource, task: TaskConfig, stage: str) -> ConfigModel:
    mapping = _merge_many(_base_mapping(source), task.overlay)
    data_overlay: dict[str, Any] = {"reference_role": task.reference_role}
    if task.track:
        data_overlay["track"] = task.track
    if task.task:
        data_overlay["task"] = task.task
    mapping = deep_merge(mapping, {"data": data_overlay, "run": {"source_cap": None}})
    if stage == "screen":
        mapping = _screen_safe_mapping(mapping, task=task)
    return ConfigModel.from_mapping(mapping, warn_v1=False)


def _write_inventory(
    suite: LoadedSuite,
    *,
    stage: str,
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    stage_root = Path(output_root).expanduser().resolve() / suite.suite_id / stage
    _atomic_json(
        stage_root / "dataset_inventory.json",
        {
            "schema_version": 1,
            "suite_id": suite.suite_id,
            "stage": stage,
            "suite_hash": suite.suite_hash,
            "baseline_manifest": _baseline_record(suite),
            "dataset_lock": (
                {"path": str(suite.dataset_lock), "sha256": suite.dataset_lock_hash}
                if suite.dataset_lock is not None
                else None
            ),
            "rows": list(rows),
        },
    )
    _atomic_csv_rows(stage_root / "dataset_inventory.csv", rows)


def build_dataset_inventory(
    suite: LoadedSuite,
    *,
    stage: str,
    output_root: Path,
    selection_record: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Inspect each executable paper task before any model or reporting run starts."""

    roles = _dataset_lock_roles(suite)
    cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    frozen = (selection_record or {}).get("experiments") or {}
    for source in suite.sources:
        config = source.config
        if config.implementation.status != "ready":
            continue
        if stage == "confirm":
            record = frozen.get(config.experiment_id)
            if not isinstance(record, Mapping) or record.get("status") != "selected":
                continue
        stage_config = config.screen if stage == "screen" else config.confirm
        for task in stage_config.tasks:
            resolved = _inventory_config(source, task, stage)
            resolved_mapping = resolved.model_dump(mode="json", by_alias=True)
            cache_key = hash_payload(
                {
                    "data": resolved_mapping.get("data"),
                    "io": resolved_mapping.get("io"),
                    "dataset": {
                        "hierarchical_relation_families": (
                            resolved_mapping.get("dataset") or {}
                        ).get("hierarchical_relation_families")
                    },
                    "entity_kinds": (resolved_mapping.get("matching") or {}).get("entity_kinds"),
                }
            )
            row = cache.get(cache_key)
            if row is None:
                row = inspect_dataset_task(
                    resolved,
                    experiment_id=config.experiment_id,
                    task_id=task.id,
                    stage=stage,
                    split_role=task.split_role,
                    reference_role=task.reference_role,
                    reference_completeness=task.reference_completeness,
                    capabilities=task.capabilities,
                    split_availability=roles.get(str(resolved.data.track), ()),
                )
                cache[cache_key] = dict(row)
            materialized = dict(_jsonable(row))
            materialized.update(
                {
                    "experiment_id": config.experiment_id,
                    "task_id": task.id,
                    "stage": stage,
                    "split_role": task.split_role,
                    "reference_role": task.reference_role,
                    "reference_completeness": task.reference_completeness,
                    "capabilities": sorted(set(task.capabilities)),
                    "split_availability": sorted(set(roles.get(str(resolved.data.track), ()))),
                }
            )
            rows.append(materialized)
    rows.sort(key=lambda row: (str(row["experiment_id"]), str(row["task_id"])))
    _write_inventory(suite, stage=stage, output_root=output_root, rows=rows)
    return rows


def _candidate_design_hash(mapping: Mapping[str, Any]) -> str:
    candidates = mapping.get("candidates") or {}
    fitted = {
        "encoder_finetune": candidates.get("encoder_finetune"),
        "cross_encoder": candidates.get("cross_encoder"),
    }
    return hash_payload({"candidates": candidates, "fitted_retrieval": fitted})


def _experiment_design_payload(
    source: ExperimentSource,
    *,
    baseline_manifest_hash: Optional[str] = None,
) -> dict[str, Any]:
    config = source.config
    return {
        "experiment_id": config.experiment_id,
        "baseline_id": config.baseline_id,
        "baseline_manifest_hash": baseline_manifest_hash,
        "selection": config.selection,
        "design": config.design,
        "reporting": config.confirm,
        "composition": config.composition,
        "arms": [
            {
                "id": arm.id,
                "role": arm.role,
                "stages": arm.stages,
                "required_control": arm.required_control,
                "deployable": arm.deployable,
                "overlay_hash": hash_payload(arm.overlay),
            }
            for arm in config.arms
        ],
    }


def experiment_design_hash(
    source: ExperimentSource,
    *,
    baseline_manifest_hash: Optional[str] = None,
) -> str:
    return hash_payload(
        _experiment_design_payload(
            source,
            baseline_manifest_hash=baseline_manifest_hash,
        )
    )


def _component_arms(
    config: ExperimentConfig,
    *,
    promoted_overlays: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> list[ArmConfig]:
    if config.experiment_id != "E17":
        return list(config.arms)
    assert config.composition is not None
    composition = config.composition
    components = {component.id: component for component in composition.components}
    effective_components: dict[str, dict[str, Any]] = {}
    for component in composition.components:
        promoted: Mapping[str, Any] = {}
        if component.source_experiment:
            if promoted_overlays is None or component.source_experiment not in promoted_overlays:
                raise ValueError(
                    f"E17 component {component.id!r} is missing the selected promotion "
                    f"from dependency {component.source_experiment!r}"
                )
            promoted = promoted_overlays[component.source_experiment]
        effective_components[component.id] = _merge_many(promoted, component.overlay)

    generated: dict[str, dict[str, Any]] = {
        "rolling": dict(composition.rolling_overlay),
        "stack_all": _merge_many(
            composition.rolling_overlay,
            *(effective_components[component.id] for component in composition.components),
        ),
    }
    for component in composition.components:
        generated[f"stack_minus_{component.id}"] = _merge_many(
            composition.rolling_overlay,
            *(
                effective_components[candidate.id]
                for candidate in composition.components
                if candidate.id != component.id
            ),
        )
    for interaction in composition.interactions:
        if interaction.left not in components or interaction.right not in components:
            raise ValueError(f"E17 interaction {interaction.id} references an undeclared component")
        generated[f"interaction_{interaction.id}_00"] = dict(composition.rolling_overlay)
        generated[f"interaction_{interaction.id}_10"] = _merge_many(
            composition.rolling_overlay, effective_components[interaction.left]
        )
        generated[f"interaction_{interaction.id}_01"] = _merge_many(
            composition.rolling_overlay, effective_components[interaction.right]
        )
        generated[f"interaction_{interaction.id}_11"] = _merge_many(
            composition.rolling_overlay,
            effective_components[interaction.left],
            effective_components[interaction.right],
        )

    arms: list[ArmConfig] = []
    declared = {arm.id: arm for arm in config.arms}
    for arm_id, overlay in generated.items():
        template = declared.get(arm_id)
        if template is None:
            role: Literal["baseline", "candidate", "control", "diagnostic", "oracle"] = (
                "baseline"
                if arm_id == "rolling"
                else "candidate" if arm_id == "stack_all" else "control"
            )
            stages: list[Literal["screen", "confirm"]] = ["screen", "confirm"]
            if arm_id.startswith("stack_minus_"):
                component_id = arm_id[len("stack_minus_") :]
                if not components[component_id].claim:
                    stages = ["screen"]
            if arm_id.startswith("interaction_"):
                interaction_id = arm_id[len("interaction_") : -3]
                match = next(item for item in composition.interactions if item.id == interaction_id)
                stages = ["screen", "confirm"] if match.confirm else ["screen"]
            template = ArmConfig(
                id=arm_id,
                role=role,
                overlay={},
                stages=stages,
                required_control=role == "control" and "confirm" in stages,
            )
        arms.append(template.model_copy(update={"overlay": overlay}))
    for arm in config.arms:
        if arm.id not in generated:
            arms.append(arm)
    return arms


def _merge_many(*mappings: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        merged = deep_merge(merged, mapping)
    return merged


def _training_available(task: TaskConfig, mapping: Mapping[str, Any]) -> bool:
    if "train_reference" in task.capabilities:
        return True
    data = mapping.get("data")
    return isinstance(data, Mapping) and bool((data.get("refs") or {}).get("train"))


def resolve_supervision(
    config: ConfigModel,
    *,
    task: TaskConfig,
    arm: ArmConfig,
) -> tuple[str, dict[str, Any]]:
    training = _training_available(task, config.model_dump(mode="python"))
    root_mode = str(config.supervision.mode)
    overrides = {str(key): str(value) for key, value in config.supervision.components.items()}
    resolved: dict[str, Any] = {}
    for component in _COMPONENTS:
        requested = overrides.get(component, root_mode)
        mode, reason = config.supervision.resolve_component(
            component,
            training_available=training,
        )
        resolved[component] = {
            "requested": requested,
            "resolved": mode,
            "reason": reason,
        }

    if arm.supervision_label is not None:
        label = arm.supervision_label
    elif arm.role == "oracle":
        label = "oracle_diagnostic"
    else:
        modes = {entry["resolved"] for entry in resolved.values()}
        label = "in_pair_supervised" if "supervised" in modes else "target_label_free"
    has_supervised = any(entry["resolved"] == "supervised" for entry in resolved.values())
    if label == "target_label_free" and has_supervised:
        raise ValueError("declared target_label_free arm resolves a supervised component")
    if label == "in_pair_supervised" and not has_supervised:
        raise ValueError("declared in_pair_supervised arm resolves no supervised component")
    return label, resolved


def _resolve_config(
    source: ExperimentSource,
    *,
    task: TaskConfig,
    arm: ArmConfig,
    stage: LiteralStage,
    seed: int,
    source_cap: Optional[int],
    inherited_overlay: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    for label, overlay in (
        ("inherited overlay", inherited_overlay),
        (f"task {task.id}", task.overlay),
        (f"arm {arm.id}", arm.overlay),
    ):
        _validate_overlay_surface(overlay, f"{source.config.experiment_id} {label}")
    mapping = _merge_many(_base_mapping(source), inherited_overlay, task.overlay, arm.overlay)
    data_overlay: dict[str, Any] = {"reference_role": task.reference_role}
    if task.track:
        data_overlay["track"] = task.track
    if task.task:
        data_overlay["task"] = task.task
    mapping = deep_merge(
        mapping,
        {
            "data": data_overlay,
            "run": {
                "seed": seed,
                "source_cap": source_cap,
                "experiment_audit": True,
            },
        },
    )
    if stage == "screen":
        mapping = _screen_safe_mapping(mapping, task=task)
    validated = ConfigModel.from_mapping(mapping, warn_v1=False)
    resolved = validated.model_dump(mode="json", by_alias=True)
    if stage == "screen":
        resolved_data = resolved.get("data")
        if not isinstance(resolved_data, Mapping):
            raise ValueError("screen resolved config has no data declaration")
        role = str(resolved_data.get("reference_role") or "").strip().lower()
        if role not in _SCREEN_REFERENCE_ROLES:
            raise ValueError(f"screen reference_role {role!r} is not a development/validation role")
        leaked = sorted(
            str(key)
            for key in (resolved_data.get("refs") or {})
            if str(key).strip().lower() not in _SCREEN_RETAINED_REFERENCE_ROLES
        )
        if leaked:
            raise ValueError(f"screen resolved config retained unsafe references: {leaked}")
    label, supervision = resolve_supervision(validated, task=task, arm=arm)
    return resolved, validated.fingerprint(), label, supervision


def _screen_safe_mapping(
    mapping: Mapping[str, Any],
    *,
    task: TaskConfig,
) -> dict[str, Any]:
    role = str(task.reference_role).strip().lower()
    if role not in _SCREEN_REFERENCE_ROLES:
        raise ValueError(
            f"screen reference_role {task.reference_role!r} is not a " "development/validation role"
        )
    cleaned = dict(_jsonable(mapping))
    data = dict(cleaned.get("data") or {})
    refs = data.get("refs")
    if isinstance(refs, Mapping):
        data["refs"] = {
            str(key): value
            for key, value in refs.items()
            if str(key).strip().lower() in _SCREEN_RETAINED_REFERENCE_ROLES
        }
    for legacy_key in ("full_reference", "reporting_reference", "test_reference"):
        data.pop(legacy_key, None)
    data["reference_role"] = role
    cleaned["data"] = data
    return cleaned


def _selected_arm_ids(
    source: ExperimentSource,
    *,
    stage: str,
    selection_record: Optional[Mapping[str, Any]],
    arms: Optional[Sequence[ArmConfig]] = None,
) -> set[str]:
    resolved_arms = list(arms) if arms is not None else _component_arms(source.config)
    if stage == "screen":
        return {arm.id for arm in resolved_arms if "screen" in arm.stages}
    if selection_record is None:
        raise ValueError("confirm requires a selection record")
    experiment = (selection_record.get("experiments") or {}).get(source.config.experiment_id)
    if not isinstance(experiment, Mapping):
        raise ValueError(f"selection record has no entry for {source.config.experiment_id}")
    status = experiment.get("status")
    if status == "screened_out":
        return set()
    if status != "selected":
        raise ValueError(
            f"{source.config.experiment_id}: selection status {status!r} is not confirmable"
        )
    selected = set()
    for decision in experiment.get("decisions") or []:
        if not isinstance(decision, Mapping):
            continue
        selected.add(str(decision["baseline"]))
        candidate = decision.get("selected_arm")
        if candidate:
            selected.add(str(candidate))
        selected.update(str(item) for item in decision.get("required_controls") or [])
    selected.update(arm.id for arm in resolved_arms if arm.required_control)
    unavailable = sorted(
        arm_id
        for arm_id in selected
        if arm_id in {arm.id for arm in resolved_arms}
        and "confirm" not in next(arm.stages for arm in resolved_arms if arm.id == arm_id)
    )
    if unavailable:
        raise ValueError(
            f"{source.config.experiment_id}: frozen confirm arms are not "
            f"confirm-eligible: {unavailable}"
        )
    return selected


def build_cells(
    suite: LoadedSuite,
    source: ExperimentSource,
    *,
    stage: LiteralStage,
    output_root: Path,
    selection_record: Optional[Mapping[str, Any]] = None,
    inherited_overlay: Optional[Mapping[str, Any]] = None,
    promoted_component_overlays: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> list[RunCell]:
    config = source.config
    if config.implementation.status != "ready":
        return []
    stage_config = config.screen if stage == "screen" else config.confirm
    arms = _component_arms(
        config,
        promoted_overlays=promoted_component_overlays,
    )
    selected_ids = _selected_arm_ids(
        source,
        stage=stage,
        selection_record=selection_record,
        arms=arms,
    )
    by_id = {arm.id: arm for arm in arms}
    missing = selected_ids.difference(by_id)
    if missing:
        raise ValueError(f"{config.experiment_id}: selected unknown arms {sorted(missing)}")

    design_hash = experiment_design_hash(
        source,
        baseline_manifest_hash=suite.baseline_manifest_hash,
    )
    selection_hash = (
        str(selection_record.get("selection_hash")) if selection_record is not None else None
    )
    cells: list[RunCell] = []
    for arm in arms:
        if arm.id not in selected_ids or stage not in arm.stages:
            continue
        for task in stage_config.tasks:
            source_cap = task.source_cap or stage_config.source_cap
            for seed in stage_config.seeds:
                resolved, config_hash, label, supervision = _resolve_config(
                    source,
                    task=task,
                    arm=arm,
                    stage=stage,
                    seed=seed,
                    source_cap=source_cap,
                    inherited_overlay=inherited_overlay or {},
                )
                resource_config = arm.resource or config.resource
                output_dir = (
                    Path(output_root).expanduser().resolve()
                    / suite.suite_id
                    / stage
                    / "runs"
                    / config.experiment_id
                    / arm.id
                    / task.id
                    / f"seed-{seed}"
                )
                cells.append(
                    RunCell(
                        suite_id=suite.suite_id,
                        experiment_id=config.experiment_id,
                        stage=stage,
                        arm_id=arm.id,
                        arm_role=arm.role,
                        task_id=task.id,
                        split_role=task.split_role,
                        reference_role=task.reference_role,
                        seed=seed,
                        source_cap=source_cap,
                        resource=resource_config,
                        output_dir=output_dir,
                        resolved_config=resolved,
                        config_hash=config_hash,
                        experiment_config_hash=source.raw_hash(),
                        design_hash=design_hash,
                        selection_hash=selection_hash,
                        supervision_label=label,
                        resolved_supervision=supervision,
                        negative_label_policy=config.negative_label_policy,
                    )
                )
    return cells


LiteralStage = str


_SOURCE_FINGERPRINT_SUFFIXES = {".json", ".py", ".yaml", ".yml"}


def _git_null_paths(workdir: Path, *args: str) -> Optional[list[str]]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=workdir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    ]


def _dirty_source_fingerprint(workdir: Path) -> Optional[dict[str, Any]]:
    """Hash changed runtime source without inspecting specs, data, or experiment outputs."""

    tracked = _git_null_paths(
        workdir,
        "diff",
        "--name-only",
        "-z",
        "HEAD",
        "--",
        "exact",
        "tools",
    )
    untracked = _git_null_paths(
        workdir,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        "exact",
        "tools",
    )
    if tracked is None or untracked is None:
        return None
    entries: list[dict[str, Any]] = []
    for raw in sorted(set(tracked).union(untracked)):
        relative = Path(raw)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] not in {"exact", "tools"}
            or relative.suffix.lower() not in _SOURCE_FINGERPRINT_SUFFIXES
            or relative.name.endswith((".orig", ".rej"))
        ):
            continue
        path = workdir / relative
        if path.is_symlink():
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return {
        "sha256": hash_payload(entries),
        "files": len(entries),
    }


def _source_tree_fingerprint(workdir: Path) -> Optional[dict[str, Any]]:
    """Hash the complete executable source tree independently of commit metadata."""

    tracked = _git_null_paths(
        workdir,
        "ls-files",
        "-z",
        "--",
        "exact",
        "tools",
    )
    untracked = _git_null_paths(
        workdir,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        "exact",
        "tools",
    )
    if tracked is None or untracked is None:
        return None
    entries: list[dict[str, Any]] = []
    for raw in sorted(set(tracked).union(untracked)):
        relative = Path(raw)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] not in {"exact", "tools"}
            or relative.suffix.lower() not in _SOURCE_FINGERPRINT_SUFFIXES
            or relative.name.endswith((".orig", ".rej"))
        ):
            continue
        path = workdir / relative
        if path.is_symlink():
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return {"sha256": hash_payload(entries), "files": len(entries)}


def _git_provenance(workdir: Path) -> dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=workdir,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
        "worktree_source": _dirty_source_fingerprint(workdir),
        "source_tree": _source_tree_fingerprint(workdir),
    }


_PACKAGE_CACHE: Optional[dict[str, str]] = None
_PACKAGE_LOCK = threading.Lock()


def _package_versions() -> dict[str, str]:
    global _PACKAGE_CACHE
    with _PACKAGE_LOCK:
        if _PACKAGE_CACHE is None:
            versions: dict[str, str] = {}
            for distribution in importlib.metadata.distributions():
                name = distribution.metadata.get("Name")
                if name:
                    versions[str(name)] = distribution.version
            _PACKAGE_CACHE = dict(sorted(versions.items(), key=lambda item: item[0].lower()))
        return dict(_PACKAGE_CACHE)


def _path_provenance(mapping: Mapping[str, Any]) -> dict[str, Any]:
    data = mapping.get("data")
    if not isinstance(data, Mapping):
        return {}
    root = Path(str(data.get("root") or ".")).expanduser().resolve()

    def resolved_file(value: Any, *, relative_to_root: bool, label: str) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = (root / path) if relative_to_root else path.resolve()
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"configured {label} does not exist: {path}")
        return path

    result: dict[str, Any] = {}
    for key in ("source", "target", "candidates"):
        value = data.get(key)
        if value:
            result[key] = file_provenance(
                resolved_file(value, relative_to_root=True, label=f"data.{key}")
            )
    descriptor = data.get("descriptor")
    if descriptor:
        result["descriptor"] = file_provenance(
            resolved_file(
                descriptor,
                relative_to_root=False,
                label="data.descriptor",
            )
        )
    refs = data.get("refs")
    if isinstance(refs, Mapping):
        result["references"] = {
            str(role): file_provenance(
                resolved_file(
                    value,
                    relative_to_root=True,
                    label=f"data.refs.{role}",
                )
            )
            for role, value in refs.items()
            if value
        }
    if data.get("track"):
        result["track"] = {
            "name": data.get("track"),
            "task": data.get("task"),
            "revision": data.get("revision"),
            "root": data.get("root"),
            "reference_role": data.get("reference_role"),
        }
    return result


def _artifact_provenance(value: Any, prefix: str = "") -> dict[str, Any]:
    found: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if (
                ("artifact" in str(key).lower() or prefix.lower().endswith("artifacts"))
                and isinstance(item, (str, os.PathLike))
                and item
            ):
                candidate = Path(str(item)).expanduser()
                if not candidate.is_file():
                    raise FileNotFoundError(
                        f"configured fitted artifact does not exist: {candidate}"
                    )
                found[path] = file_provenance(candidate)
            else:
                found.update(_artifact_provenance(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_artifact_provenance(item, f"{prefix}.{index}"))
    return found


def _model_identities(mapping: Mapping[str, Any]) -> dict[str, Any]:
    pipeline = []
    for entry in mapping.get("pipeline") or []:
        if not isinstance(entry, Mapping):
            continue
        params_value = entry.get("params")
        params: Mapping[str, Any] = params_value if isinstance(params_value, Mapping) else {}
        pipeline.append(
            {
                "name": entry.get("name"),
                "lexical_model": params.get("lexical_model_name"),
                "context_model": params.get("context_model_name"),
                "llm_model": params.get("llm_model_name"),
            }
        )
    llm = mapping.get("llm") if isinstance(mapping.get("llm"), Mapping) else {}
    profiles = llm.get("profiles") if isinstance(llm, Mapping) else {}
    safe_profiles: dict[str, dict[str, Any]] = {}
    if isinstance(profiles, Mapping):
        for name, profile in profiles.items():
            if not isinstance(profile, Mapping):
                continue
            safe_profile = {
                key: profile.get(key)
                for key in ("backend", "model", "tokenizer")
                if profile.get(key) is not None
            }
            endpoint = _safe_endpoint_identity(profile.get("api_base"))
            if endpoint is not None:
                safe_profile["api_base"] = endpoint
            safe_profiles[str(name)] = safe_profile
    candidates_value = mapping.get("candidates")
    candidates: Mapping[str, Any] = (
        candidates_value if isinstance(candidates_value, Mapping) else {}
    )
    return {
        "pipeline": pipeline,
        "candidate_retrieval": {
            "encoder": candidates.get("encoder") or candidates.get("lexical_encoder_name"),
        },
        "llm_profiles": safe_profiles,
    }


def _safe_endpoint_identity(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        parsed = urlsplit(str(value))
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if not parsed.scheme or not hostname:
        return None
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


_SENSITIVE_RUNTIME_KEYS = {
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "extra_headers",
    "headers",
    "password",
    "refresh_token",
    "secret",
    "token",
}
_SAFE_RUNTIME_STRINGS = {
    "backend",
    "effective_model",
    "fallback_state",
    "model",
    "model_name",
    "profile",
    "provider",
    "requested_model",
    "requested_profile",
    "scoring_mode",
    "status",
    "tokenizer",
}


def _safe_runtime_usage(value: Any, *, field: Optional[str] = None) -> Any:
    """Keep counters and public model identity while dropping runtime secrets."""

    normalized_field = str(field or "").strip().lower()
    if normalized_field in _SENSITIVE_RUNTIME_KEYS or normalized_field.endswith(
        ("_api_key", "_credential", "_password", "_secret")
    ):
        return None
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            cleaned = _safe_runtime_usage(item, field=str(key))
            if cleaned is not None:
                safe[str(key)] = cleaned
        return safe
    if isinstance(value, list):
        safe_items = [
            cleaned
            for item in value
            if (cleaned := _safe_runtime_usage(item, field=field)) is not None
        ]
        return safe_items
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if normalized_field in {"api_base", "endpoint", "url"}:
            return _safe_endpoint_identity(value)
        if normalized_field in _SAFE_RUNTIME_STRINGS:
            return value
        if normalized_field.endswith(("_hash", "_hashes")):
            normalized = value.strip().lower()
            if len(normalized) in {40, 64} and all(
                character in "0123456789abcdef" for character in normalized
            ):
                return normalized
    return None


def _provenance_payload(
    cell: RunCell,
    suite: LoadedSuite,
    *,
    workdir: Path,
) -> dict[str, Any]:
    inputs = _path_provenance(cell.resolved_config)
    artifacts = _artifact_provenance(cell.resolved_config)
    candidate_design_hash = _candidate_design_hash(cell.resolved_config)
    git = _git_provenance(workdir)
    packages = _package_versions()
    fingerprint_payload = {
        "code_commit": git.get("commit"),
        "worktree_source": git.get("worktree_source"),
        "package_versions_sha256": hash_payload(packages),
        "cell": cell.cell_id,
        "output_dir": str(cell.output_dir),
        "config_hash": cell.config_hash,
        "experiment_config_hash": cell.experiment_config_hash,
        "design_hash": cell.design_hash,
        "selection_hash": cell.selection_hash,
        "baseline_manifest_hash": suite.baseline_manifest_hash,
        "dataset_lock_hash": suite.dataset_lock_hash,
        "inputs": inputs,
        "candidate_pool_design_hash": candidate_design_hash,
        "artifacts": artifacts,
        "models": _model_identities(cell.resolved_config),
        "supervision": cell.resolved_supervision,
        "supervision_label": cell.supervision_label,
        "negative_label_policy": cell.negative_label_policy,
    }
    package_names = {name.lower(): version for name, version in packages.items()}
    return {
        "fingerprint": hash_payload(fingerprint_payload),
        "fingerprint_payload": fingerprint_payload,
        "git": git,
        "runtime_versions": {
            "exact_om": package_names.get("exact-om"),
            "pyowl_core": package_names.get("pyowl-core"),
        },
        "packages": packages,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required JSON file does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _prepare_cell(
    cell: RunCell,
    suite: LoadedSuite,
    *,
    workdir: Path,
    resume: bool,
) -> tuple[dict[str, Any], bool]:
    provenance = _provenance_payload(cell, suite, workdir=workdir)
    if cell.manifest_path.is_file():
        existing = _load_json(cell.manifest_path)
        old_fingerprint = existing.get("fingerprint")
        if old_fingerprint != provenance["fingerprint"]:
            raise ValueError(
                f"resume fingerprint mismatch for {cell.cell_id}: "
                f"{old_fingerprint} != {provenance['fingerprint']}"
            )
        if existing.get("status") == "complete":
            if not resume:
                raise FileExistsError(
                    f"completed output already exists for {cell.cell_id}; pass --resume to reuse"
                )
            _validate_reused_candidate_pool(cell, existing)
            return existing, True
        if not resume:
            raise FileExistsError(
                f"partial output exists for {cell.cell_id}; pass --resume to continue"
            )
    elif cell.output_dir.exists() and any(cell.output_dir.iterdir()):
        raise FileExistsError(
            f"non-empty output exists without a provenance manifest: {cell.output_dir}"
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": cell.suite_id,
        "experiment_id": cell.experiment_id,
        "stage": cell.stage,
        "arm_id": cell.arm_id,
        "arm_role": cell.arm_role,
        "task_id": cell.task_id,
        "split_role": cell.split_role,
        "reference_role": cell.reference_role,
        "seed": cell.seed,
        "source_cap": cell.source_cap,
        "resource": cell.resource,
        "resolved_config_hash": cell.config_hash,
        "experiment_config_hash": cell.experiment_config_hash,
        "selection_record_hash": cell.selection_hash,
        "design_declaration_hash": cell.design_hash,
        "baseline_manifest": _baseline_record(suite),
        "dataset_lock": (
            {"path": str(suite.dataset_lock), "sha256": suite.dataset_lock_hash}
            if suite.dataset_lock
            else None
        ),
        "supervision_label": cell.supervision_label,
        "resolved_supervision": cell.resolved_supervision,
        "negative_label_policy": cell.negative_label_policy,
        "status": "pending",
        **provenance,
    }
    return manifest, False


def _write_cell_inputs(cell: RunCell) -> tuple[Path, Path]:
    cell.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = cell.output_dir / "_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    config_path = inputs / "resolved.config.yaml"
    _atomic_text(config_path, dump_yaml_document(cell.resolved_config))
    data = dict(cell.resolved_config.get("data") or {})
    wrapper = {
        "data": data,
        "job": {
            "name": cell.cell_id.replace("/", "-"),
            "output_dir": str(cell.output_dir),
            "config_file": str(config_path),
            "run_eval": True,
            "save_logs": True,
        },
    }
    if cell.resource.kind == "gpu" and cell.resource.device is not None:
        wrapper["job"]["device"] = cell.resource.device
    wrapper_path = inputs / "job.yaml"
    _atomic_text(wrapper_path, dump_yaml_document(wrapper))
    return config_path, wrapper_path


def _run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float, Optional[int]]:
    started = time.monotonic()
    peak_kb: Optional[int] = None
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        status_path = Path("/proc") / str(process.pid) / "status"
        while process.poll() is None:
            try:
                for line in status_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith(("VmHWM:", "VmRSS:")):
                        current = int(line.split()[1])
                        peak_kb = current if peak_kb is None else max(peak_kb, current)
            except (FileNotFoundError, OSError, ValueError, IndexError):
                pass
            time.sleep(0.1)
        return_code = int(process.returncode or 0)
    return return_code, time.monotonic() - started, peak_kb


def _read_optional_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _post_run_provenance(cell: RunCell) -> dict[str, Any]:
    dataset_dir = cell.output_dir / "dataset"
    sampled_pool_path = dataset_dir / "candidate_pool_sample_manifest.json"
    pool_path = (
        sampled_pool_path
        if sampled_pool_path.is_file()
        else dataset_dir / "candidate_pool_manifest.json"
    )
    pool = _read_optional_json(pool_path)
    run_stats = _read_optional_json(cell.output_dir / "stats" / "run_stats.json") or {}
    timing = _read_optional_json(cell.output_dir / "timings.json")
    llm_usage = _safe_runtime_usage(run_stats.get("llm") or run_stats.get("llm_usage"))
    return {
        "candidate_pool": pool,
        "candidate_pool_fingerprint": (
            pool.get("fingerprint") if isinstance(pool, Mapping) else None
        ),
        "candidate_pool_manifest_provenance": (
            file_provenance(pool_path) if pool_path.is_file() else None
        ),
        "runtime_dataset_provenance": (run_stats.get("provenance") or {}).get("dataset"),
        "source_sampling": run_stats.get("source_sampling"),
        "timing_ledger": timing,
        "llm_usage": llm_usage,
    }


_RUNTIME_LLM_IDENTITY_KEYS = {
    "backend",
    "effective_model",
    "endpoint",
    "profile",
    "provider",
    "requested_model",
    "requested_profile",
    "tokenizer",
}


def _validate_paired_llm_identities(
    manifests: Sequence[Mapping[str, Any]],
) -> None:
    """Reject resolved model drift across confirmed paired arms."""

    groups: dict[tuple[str, str, int, str], dict[str, set[str]]] = {}
    for manifest in manifests:
        if manifest.get("status") != "complete":
            continue
        usage = manifest.get("llm_usage")
        identities = usage.get("backend_identities") if isinstance(usage, Mapping) else None
        if not isinstance(identities, Mapping):
            continue
        arm = str(manifest.get("arm_id"))
        for task, raw_entries in identities.items():
            if not isinstance(raw_entries, list):
                continue
            projected = [
                {
                    key: entry.get(key)
                    for key in sorted(_RUNTIME_LLM_IDENTITY_KEYS)
                    if isinstance(entry, Mapping) and entry.get(key) is not None
                }
                for entry in raw_entries
                if isinstance(entry, Mapping)
            ]
            projected.sort(key=canonical_json)
            if not projected:
                continue
            key = (
                str(manifest.get("experiment_id")),
                str(manifest.get("task_id")),
                int(manifest.get("seed") or 0),
                str(task),
            )
            groups.setdefault(key, {}).setdefault(canonical_json(projected), set()).add(arm)
    for key, identities in groups.items():
        compared_arms = set().union(*identities.values()) if identities else set()
        if len(compared_arms) > 1 and len(identities) > 1:
            experiment, task_id, seed, llm_task = key
            detail = {identity: sorted(arms) for identity, arms in sorted(identities.items())}
            raise ValueError(
                "resolved LLM identity changed within paired confirm arms for "
                f"{experiment}/{task_id}/seed-{seed}/{llm_task}: {detail}"
            )


def _validate_reused_candidate_pool(
    cell: RunCell,
    existing: Mapping[str, Any],
) -> None:
    expected_fingerprint = existing.get("candidate_pool_fingerprint")
    expected_file = existing.get("candidate_pool_manifest_provenance")
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise ValueError(
            f"completed cell {cell.cell_id} has no persisted candidate-pool fingerprint"
        )
    if not isinstance(expected_file, Mapping) or not expected_file.get("sha256"):
        raise ValueError(
            f"completed cell {cell.cell_id} has no persisted candidate-pool manifest hash"
        )
    current = _post_run_provenance(cell)
    if current.get("candidate_pool_fingerprint") != expected_fingerprint:
        raise ValueError(f"candidate-pool fingerprint changed for {cell.cell_id}")
    current_file = current.get("candidate_pool_manifest_provenance")
    if not isinstance(current_file, Mapping):
        raise ValueError(f"candidate-pool manifest is missing for {cell.cell_id}")
    for key in ("path", "sha256"):
        if current_file.get(key) != expected_file.get(key):
            raise ValueError(f"candidate-pool manifest {key} changed for {cell.cell_id}")
    pool = current.get("candidate_pool")
    if not isinstance(pool, Mapping):
        raise ValueError(f"candidate-pool manifest is invalid for {cell.cell_id}")
    inputs = pool.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"candidate-pool manifest has no input provenance for {cell.cell_id}")
    fingerprinted_inputs: dict[str, Mapping[str, Any]] = {}
    for name in ("source", "target"):
        configured = inputs.get(name)
        if (
            not isinstance(configured, Mapping)
            or not configured.get("path")
            or not configured.get("sha256")
        ):
            raise ValueError(
                f"candidate-pool manifest has no fingerprinted {name} input " f"for {cell.cell_id}"
            )
        fingerprinted_inputs[name] = configured
    for raw_name, configured in inputs.items():
        name = str(raw_name)
        if name in fingerprinted_inputs or not isinstance(configured, Mapping):
            continue
        if "path" not in configured and "sha256" not in configured:
            continue
        if not configured.get("path") or not configured.get("sha256"):
            raise ValueError(
                f"candidate-pool manifest has incomplete fingerprinted {name} "
                f"input for {cell.cell_id}"
            )
        fingerprinted_inputs[name] = configured
    for name, configured in fingerprinted_inputs.items():
        path = Path(str(configured["path"])).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"candidate-pool {name} input is missing for {cell.cell_id}: {path}")
        observed = file_provenance(path)
        for key in ("path", "sha256"):
            if observed.get(key) != configured.get(key):
                raise ValueError(f"candidate-pool {name} input {key} changed for {cell.cell_id}")


def execute_cell(
    cell: RunCell,
    suite: LoadedSuite,
    *,
    workdir: Path,
    resume: bool,
) -> dict[str, Any]:
    manifest, reused = _prepare_cell(cell, suite, workdir=workdir, resume=resume)
    if reused:
        return manifest
    _, wrapper_path = _write_cell_inputs(cell)
    manifest["status"] = "running"
    manifest["started_at"] = _utc_now()
    _atomic_json(cell.manifest_path, manifest)

    command = [
        sys.executable,
        str((workdir / "tools" / "run_exact_job.py").resolve()),
        "--run-config",
        str(wrapper_path),
    ]
    stdout_path = cell.output_dir / "experiment.stdout.log"
    stderr_path = cell.output_dir / "experiment.stderr.log"
    try:
        return_code, elapsed, peak_kb = _run_subprocess(
            command,
            cwd=workdir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "ended_at": _utc_now(),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _atomic_json(cell.manifest_path, manifest)
        return manifest

    post_run = _post_run_provenance(cell)
    manifest.update(
        {
            "status": "complete" if return_code == 0 else "failed",
            "ended_at": _utc_now(),
            "wall_seconds": elapsed,
            "peak_memory_kb": peak_kb,
            "return_code": return_code,
            **post_run,
        }
    )
    if return_code == 0 and (
        not manifest.get("candidate_pool_fingerprint")
        or not manifest.get("candidate_pool_manifest_provenance")
    ):
        manifest["status"] = "failed"
        manifest["failure"] = {
            "type": "MissingCandidatePoolManifest",
            "message": "successful run did not persist a fingerprinted candidate-pool manifest",
        }
    elif return_code != 0:
        manifest["failure"] = {
            "type": "SubprocessError",
            "message": f"run_exact_job exited with status {return_code}",
            "stderr": str(stderr_path),
        }
    _atomic_json(cell.manifest_path, manifest)
    return manifest


def run_cells(
    cells: Sequence[RunCell],
    suite: LoadedSuite,
    *,
    workdir: Path,
    jobs: int,
    resume: bool,
) -> list[dict[str, Any]]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    locks: dict[str, threading.BoundedSemaphore] = {}
    for cell in cells:
        if cell.resource.kind == "cpu":
            continue
        key = cell.resource.serialization_key()
        existing = locks.get(key)
        if existing is None:
            locks[key] = threading.BoundedSemaphore(cell.resource.concurrency)

    def run_one(cell: RunCell) -> dict[str, Any]:
        lock = locks.get(cell.resource.serialization_key())
        if lock is None:
            return execute_cell(cell, suite, workdir=workdir, resume=resume)
        with lock:
            return execute_cell(cell, suite, workdir=workdir, resume=resume)

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(run_one, cell): cell for cell in cells}
        for future in as_completed(futures):
            cell = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                completed.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "suite_id": cell.suite_id,
                        "experiment_id": cell.experiment_id,
                        "stage": cell.stage,
                        "arm_id": cell.arm_id,
                        "task_id": cell.task_id,
                        "seed": cell.seed,
                        "baseline_manifest": _baseline_record(suite),
                        "status": "failed",
                        "failure": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )
    return sorted(
        completed,
        key=lambda item: (
            str(item.get("experiment_id")),
            str(item.get("arm_id")),
            str(item.get("task_id")),
            int(item.get("seed", 0)),
        ),
    )


def _require_successful_cells(
    manifests: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> None:
    failed = [
        "/".join(
            (
                str(manifest.get("experiment_id")),
                str(manifest.get("arm_id")),
                str(manifest.get("task_id")),
                f"seed-{manifest.get('seed')}",
            )
        )
        for manifest in manifests
        if manifest.get("status") != "complete"
    ]
    if failed:
        raise ValueError(f"{stage} stage has failed or incomplete cells: {sorted(failed)}")
    missing_metrics: list[str] = []
    metric_errors: list[str] = []
    for manifest in manifests:
        output_dir_value = (manifest.get("fingerprint_payload") or {}).get("output_dir")
        metrics: dict[str, float] = {}
        if output_dir_value:
            try:
                metrics = cell_metrics(Path(str(output_dir_value)))
            except (OSError, TypeError, ValueError) as exc:
                metric_errors.append(
                    "/".join(
                        (
                            str(manifest.get("experiment_id")),
                            str(manifest.get("arm_id")),
                            str(manifest.get("task_id")),
                            f"seed-{manifest.get('seed')}",
                        )
                    )
                    + f": {type(exc).__name__}: {exc}"
                )
                continue
        if not metrics or not any(math.isfinite(value) for value in metrics.values()):
            missing_metrics.append(
                "/".join(
                    (
                        str(manifest.get("experiment_id")),
                        str(manifest.get("arm_id")),
                        str(manifest.get("task_id")),
                        f"seed-{manifest.get('seed')}",
                    )
                )
            )
    if metric_errors or missing_metrics:
        raise ValueError(
            f"{stage} stage complete cells have metric extraction errors "
            f"{sorted(metric_errors)} or no finite metrics {sorted(missing_metrics)}"
        )


def cell_metrics(output_dir: Path) -> dict[str, float]:
    """Read only the authoritative evaluator JSON metric payloads."""

    return extract_evaluation_metrics(Path(output_dir))


def _metric_value(metrics: Mapping[str, float], requested: str) -> Optional[float]:
    if requested in metrics:
        return float(metrics[requested])
    exact = [
        (key, float(value)) for key, value in metrics.items() if key.lower() == requested.lower()
    ]
    if len(exact) == 1:
        return exact[0][1]
    if len(exact) > 1:
        raise ValueError(
            f"metric {requested!r} is ambiguous across exact keys "
            f"{sorted(key for key, _ in exact)}"
        )
    suffixes = (f".{requested}", f"_{requested}", f"/{requested}")
    matches = [
        (key, float(value))
        for key, value in metrics.items()
        if any(key.lower().endswith(suffix.lower()) for suffix in suffixes)
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"metric {requested!r} is ambiguous across suffix matches "
            f"{sorted(key for key, _ in matches)}"
        )
    return matches[0][1]


def _declared_bootstrap_comparisons(
    suite: LoadedSuite,
) -> dict[str, list[tuple[str, str, str, bool]]]:
    comparisons: dict[str, list[tuple[str, str, str, bool]]] = {}
    for source in suite.sources:
        config = source.config
        if config.implementation.status != "ready":
            continue
        declared: list[tuple[str, str, str, bool]] = []
        for decision in config.selection.decisions:
            declared.extend(
                (decision.id, decision.baseline, candidate, True)
                for candidate in decision.candidates
            )
            declared.extend(
                (
                    f"{decision.id}__control__{control}",
                    decision.baseline,
                    control,
                    False,
                )
                for control in decision.required_controls
            )
        if config.experiment_id == "E17" and config.composition is not None:
            declared.extend(
                (
                    f"marginal_{component.id}",
                    f"stack_minus_{component.id}",
                    "stack_all",
                    True,
                )
                for component in config.composition.components
                if component.claim
            )
        comparisons[config.experiment_id] = declared
    return comparisons


def _declared_e17_interactions(
    suite: LoadedSuite,
) -> dict[str, list[tuple[str, str, str, str, str]]]:
    declared: dict[str, list[tuple[str, str, str, str, str]]] = {}
    for source in suite.sources:
        config = source.config
        if (
            config.experiment_id != "E17"
            or config.implementation.status != "ready"
            or config.composition is None
        ):
            continue
        declared[config.experiment_id] = [
            (
                interaction.id,
                f"interaction_{interaction.id}_00",
                f"interaction_{interaction.id}_10",
                f"interaction_{interaction.id}_01",
                f"interaction_{interaction.id}_11",
            )
            for interaction in config.composition.interactions
            if interaction.confirm
        ]
    return declared


def _stage_cell_keys(config: ExperimentConfig, stage: str) -> set[tuple[str, int]]:
    stage_config = config.screen if stage == "screen" else config.confirm
    return {(task.id, seed) for task in stage_config.tasks for seed in stage_config.seeds}


def _record_index(
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str, str, int], Mapping[str, Any]],
    set[tuple[str, str, str, int]],
]:
    index: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    duplicates: set[tuple[str, str, str, int]] = set()
    for record in records:
        key = (
            str(record.get("experiment_id")),
            str(record.get("arm_id")),
            str(record.get("task_id")),
            int(record.get("seed") or 0),
        )
        if key in index:
            duplicates.add(key)
        index[key] = record
    return index, duplicates


def _comparison_base(
    *,
    experiment_id: str,
    decision_id: str,
    baseline_arm: str,
    candidate_arm: str,
    confirmatory: bool,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "decision_id": decision_id,
        "comparison_id": f"{decision_id}:{baseline_arm}->{candidate_arm}",
        "baseline_arm": baseline_arm,
        "candidate_arm": candidate_arm,
        "task_id": "__task_macro__",
        "metric": "task_macro_global_F1",
        "endpoint_scope": "overall",
        "confirmatory": confirmatory,
        "multiplicity_family": experiment_id,
    }


def _unavailable_row(
    base: Mapping[str, Any],
    *,
    reason_code: str,
    reason: str,
    fatal: bool = False,
    **details: Any,
) -> dict[str, Any]:
    return {
        **base,
        "status": "unavailable",
        "reason_code": reason_code,
        "reason": reason,
        "fatal": fatal,
        **details,
    }


def _arm_cell_records(
    index: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
    duplicates: set[tuple[str, str, str, int]],
    *,
    experiment_id: str,
    arms: Sequence[str],
    expected_cells: set[tuple[str, int]],
) -> tuple[Optional[dict[str, dict[tuple[str, int], Mapping[str, Any]]]], dict[str, Any]]:
    by_arm: dict[str, dict[tuple[str, int], Mapping[str, Any]]] = {}
    cell_sets: dict[str, set[tuple[str, int]]] = {}
    invalid: list[str] = []
    duplicate_cells: list[str] = []
    for arm in arms:
        cells = {
            (task, seed): record
            for (experiment, record_arm, task, seed), record in index.items()
            if experiment == experiment_id and record_arm == arm
        }
        by_arm[arm] = cells
        cell_sets[arm] = set(cells)
        duplicate_cells.extend(
            f"{arm}/{task}/seed-{seed}"
            for experiment, duplicate_arm, task, seed in duplicates
            if experiment == experiment_id and duplicate_arm == arm
        )
        for (task, seed), record in cells.items():
            if record.get("status") != "complete":
                invalid.append(f"{arm}/{task}/seed-{seed}: status={record.get('status')}")
            elif record.get("metric_error"):
                invalid.append(
                    f"{arm}/{task}/seed-{seed}: metric_error={record.get('metric_error')}"
                )
    unequal = any(cells != expected_cells for cells in cell_sets.values())
    if duplicate_cells or invalid or unequal:
        return None, {
            "expected_cells": [f"{task}/seed-{seed}" for task, seed in sorted(expected_cells)],
            "observed_cells": {
                arm: [f"{task}/seed-{seed}" for task, seed in sorted(cells)]
                for arm, cells in sorted(cell_sets.items())
            },
            "duplicate_cells": sorted(duplicate_cells),
            "invalid_cells": sorted(invalid),
        }
    return by_arm, {}


def _candidate_pool_guard(
    experiment_id: str,
    cells_by_arm: Mapping[str, Mapping[tuple[str, int], Mapping[str, Any]]],
) -> tuple[str, dict[str, Any], Optional[str]]:
    fingerprints = {
        arm: {
            f"{task}/seed-{seed}": record.get("candidate_pool_fingerprint")
            for (task, seed), record in sorted(cells.items())
        }
        for arm, cells in sorted(cells_by_arm.items())
    }
    if experiment_id == "E17":
        return (
            "deferred_e17_candidate_pool_policy",
            fingerprints,
            "E17 candidate-pool equivalence groups are deferred",
        )
    tasks = sorted({task for cells in cells_by_arm.values() for task, _seed in cells})
    if experiment_id == "E05":
        for arm, cells in cells_by_arm.items():
            for task in tasks:
                values = {
                    record.get("candidate_pool_fingerprint")
                    for (cell_task, _seed), record in cells.items()
                    if cell_task == task
                }
                if None in values or "" in values:
                    return (
                        "missing_retrieval_treatment_fingerprint",
                        fingerprints,
                        f"E05 arm {arm!r} task {task!r} has a missing candidate-pool fingerprint",
                    )
                if len(values) != 1:
                    return (
                        "retrieval_treatment_seed_drift",
                        fingerprints,
                        f"E05 arm {arm!r} task {task!r} changed candidate pool across seeds",
                    )
        return "allowed_retrieval_treatment", fingerprints, None
    for task in tasks:
        values = {
            record.get("candidate_pool_fingerprint")
            for cells in cells_by_arm.values()
            for (cell_task, _seed), record in cells.items()
            if cell_task == task
        }
        if None in values or "" in values:
            return (
                "missing_fixed_retrieval_fingerprint",
                fingerprints,
                f"fixed-retrieval task {task!r} has a missing candidate-pool fingerprint",
            )
        if len(values) != 1:
            return (
                "candidate_pool_mismatch",
                fingerprints,
                f"fixed-retrieval task {task!r} used unequal candidate-pool fingerprints",
            )
    return "matched_fixed_retrieval", fingerprints, None


def _overall_evaluation(
    record: Mapping[str, Any],
    cache: dict[str, SourceEvaluation],
) -> SourceEvaluation:
    output_dir = str(record.get("output_dir") or "")
    if not output_dir:
        raise ValueError("complete cell is missing output_dir")
    cached = cache.get(output_dir)
    if cached is not None:
        return cached
    evaluation = recompute_global_prf(Path(output_dir)).overall
    cache[output_dir] = evaluation
    return evaluation


def _explicit_typed_slices(
    record: Mapping[str, Any],
) -> tuple[Optional[set[tuple[str, str]]], Optional[str]]:
    output_dir = Path(str(record.get("output_dir") or ""))
    report_path = output_dir / "evaluation" / "evaluation_results.json"
    payload = _read_optional_json(report_path)
    if not isinstance(payload, Mapping):
        return None, f"authoritative global evaluation report is unavailable: {report_path}"
    meta = payload.get("meta")
    refs = meta.get("refs") if isinstance(meta, Mapping) else None
    if not isinstance(refs, Mapping):
        return None, "global evaluation report has no meta.refs"
    artifacts = {
        "alignment": refs.get("alignment"),
        "full_reference": refs.get("full_reference"),
    }
    combinations: set[tuple[str, str]] = set()
    for label, provenance in artifacts.items():
        if not isinstance(provenance, Mapping) or not provenance.get("path"):
            return None, f"global evaluation report has no {label} provenance path"
        path = Path(str(provenance["path"])).expanduser()
        if not path.is_absolute():
            path = (report_path.parent / path).resolve()
        else:
            path = path.resolve()
        if label == "alignment" and not path.name.endswith("maps_global.tsv"):
            return None, f"evaluated alignment is not maps_global.tsv: {path.name}"
        if not path.is_file():
            return None, f"typed {label} artifact is missing: {path}"
        frame = read_table(path)
        columns = {str(column).strip().lower(): column for column in frame.columns}
        required = {"relation", "srckind", "tgtkind"}
        missing = sorted(required.difference(columns))
        if missing:
            return None, f"typed {label} artifact lacks explicit columns {missing}: {path}"
        for row in frame.to_dict(orient="records"):
            try:
                source_kind = EntityKind(str(row[columns["srckind"]]).strip().lower())
                target_kind = EntityKind(str(row[columns["tgtkind"]]).strip().lower())
                relation = normalize_relation(str(row[columns["relation"]]))
            except (TypeError, ValueError) as exc:
                return None, f"typed {label} artifact has an invalid dimension: {exc}"
            if source_kind == target_kind:
                combinations.add((source_kind.value, relation))
    return combinations, None


def _typed_slice_rows(
    *,
    base: Mapping[str, Any],
    baseline_arm: str,
    candidate_arm: str,
    cells_by_arm: Mapping[str, Mapping[tuple[str, int], Mapping[str, Any]]],
    pool_status: str,
    fingerprints: Mapping[str, Any],
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    capabilities: dict[str, Any] = {}
    requested: set[tuple[str, str]] = set()
    for arm, cells in sorted(cells_by_arm.items()):
        for (task, cell_seed), record in sorted(cells.items()):
            slices, reason = _explicit_typed_slices(record)
            cell_name = f"{arm}/{task}/seed-{cell_seed}"
            if slices is None:
                capabilities[cell_name] = {"available": False, "reason": reason}
            else:
                capabilities[cell_name] = {
                    "available": True,
                    "slices": [f"{kind}|{relation}" for kind, relation in sorted(slices)],
                }
                requested.update(slices)
    unavailable = [
        name for name, value in capabilities.items() if value.get("available") is not True
    ]
    typed_base = {
        **base,
        "comparison_id": f"{base['comparison_id']}:typed",
        "metric": "task_macro_typed_kind_relation_F1",
        "endpoint_scope": "typed_kind_relation",
        "confirmatory": False,
        "typed_slice_capabilities": capabilities,
    }
    if unavailable:
        return [
            _unavailable_row(
                typed_base,
                reason_code="typed_artifact_unavailable",
                reason="explicit enriched kind/relation artifacts are unavailable for one or more cells",
                unavailable_cells=sorted(unavailable),
            )
        ]
    if not requested:
        return [
            _unavailable_row(
                typed_base,
                reason_code="typed_slices_empty",
                reason="explicit enriched artifacts contain no within-kind relation slices",
            )
        ]

    ordered_slices = sorted(requested)
    recomputed: dict[str, RecomputedEvaluation] = {}
    try:
        for cells in cells_by_arm.values():
            for record in cells.values():
                output_dir = str(record.get("output_dir") or "")
                if not output_dir:
                    raise ValueError("complete cell is missing output_dir")
                if output_dir not in recomputed:
                    recomputed[output_dir] = recompute_global_prf(
                        Path(output_dir), slices=ordered_slices
                    )
    except Exception as exc:
        return [
            _unavailable_row(
                typed_base,
                reason_code="typed_artifact_verification_failed",
                reason=f"{type(exc).__name__}: {exc}",
            )
        ]

    rows: list[dict[str, Any]] = []
    for kind, relation in ordered_slices:
        name = f"{kind}|{relation}"
        slice_base = {
            **typed_base,
            "comparison_id": f"{base['comparison_id']}:{name}",
            "entity_kind": kind,
            "relation": relation,
        }
        try:
            baseline = {
                cell: recomputed[str(record["output_dir"])].slices[name]
                for cell, record in cells_by_arm[baseline_arm].items()
            }
            candidate = {
                cell: recomputed[str(record["output_dir"])].slices[name]
                for cell, record in cells_by_arm[candidate_arm].items()
            }
            result = paired_global_f1_bootstrap(
                baseline,
                candidate,
                resamples=resamples,
                seed=seed,
            )
        except Exception as exc:
            rows.append(
                _unavailable_row(
                    slice_base,
                    reason_code="typed_slice_inference_failed",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        rows.append(
            {
                **slice_base,
                "status": "complete",
                "fatal": False,
                "candidate_pool_guard": pool_status,
                "candidate_pool_fingerprints": fingerprints,
                **result.as_dict(),
            }
        )
    return rows


def _paired_bootstrap_rows(
    suite: LoadedSuite,
    records: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    index, duplicates = _record_index(records)
    configs = {source.config.experiment_id: source.config for source in suite.sources}
    cache: dict[str, SourceEvaluation] = {}
    rows: list[dict[str, Any]] = []
    for experiment_id, comparisons in sorted(_declared_bootstrap_comparisons(suite).items()):
        config = configs[experiment_id]
        expected_cells = _stage_cell_keys(config, stage)
        stage_seeds = config.screen.seeds if stage == "screen" else config.confirm.seeds
        typed_required = {"entity_kind", "relation"}.issubset(set(config.design.required_slices))
        for decision_id, baseline_arm, candidate_arm, confirmatory in comparisons:
            base = _comparison_base(
                experiment_id=experiment_id,
                decision_id=decision_id,
                baseline_arm=baseline_arm,
                candidate_arm=candidate_arm,
                confirmatory=confirmatory,
            )
            if stage == "confirm" and len(set(stage_seeds)) < 3:
                rows.append(
                    _unavailable_row(
                        base,
                        reason_code="insufficient_confirm_seeds",
                        reason="confirmatory inference requires at least three paired seeds",
                        paired_seeds=sorted(set(stage_seeds)),
                    )
                )
                continue
            cells_by_arm, cell_details = _arm_cell_records(
                index,
                duplicates,
                experiment_id=experiment_id,
                arms=(baseline_arm, candidate_arm),
                expected_cells=expected_cells,
            )
            if cells_by_arm is None:
                rows.append(
                    _unavailable_row(
                        base,
                        reason_code="unequal_or_missing_cells",
                        reason="paired arms do not have the exact declared task/seed cells",
                        **cell_details,
                    )
                )
                continue
            pool_status, fingerprints, pool_error = _candidate_pool_guard(
                experiment_id, cells_by_arm
            )
            if pool_error is not None:
                rows.append(
                    _unavailable_row(
                        base,
                        reason_code=pool_status,
                        reason=pool_error,
                        fatal=pool_status != "deferred_e17_candidate_pool_policy",
                        candidate_pool_guard=pool_status,
                        candidate_pool_fingerprints=fingerprints,
                    )
                )
                continue
            try:
                baseline = {
                    cell: _overall_evaluation(record, cache)
                    for cell, record in cells_by_arm[baseline_arm].items()
                }
                candidate = {
                    cell: _overall_evaluation(record, cache)
                    for cell, record in cells_by_arm[candidate_arm].items()
                }
                result = paired_global_f1_bootstrap(
                    baseline,
                    candidate,
                    resamples=resamples,
                    seed=seed,
                )
            except Exception as exc:
                rows.append(
                    _unavailable_row(
                        base,
                        reason_code="artifact_verification_failed",
                        reason=f"{type(exc).__name__}: {exc}",
                        fatal=stage == "confirm",
                        candidate_pool_guard=pool_status,
                        candidate_pool_fingerprints=fingerprints,
                    )
                )
                if typed_required:
                    rows.extend(
                        _typed_slice_rows(
                            base=base,
                            baseline_arm=baseline_arm,
                            candidate_arm=candidate_arm,
                            cells_by_arm=cells_by_arm,
                            pool_status=pool_status,
                            fingerprints=fingerprints,
                            resamples=resamples,
                            seed=seed,
                        )
                    )
                continue
            rows.append(
                {
                    **base,
                    "status": "complete",
                    "fatal": False,
                    "candidate_pool_guard": pool_status,
                    "candidate_pool_fingerprints": fingerprints,
                    **result.as_dict(),
                }
            )
            if typed_required:
                rows.extend(
                    _typed_slice_rows(
                        base=base,
                        baseline_arm=baseline_arm,
                        candidate_arm=candidate_arm,
                        cells_by_arm=cells_by_arm,
                        pool_status=pool_status,
                        fingerprints=fingerprints,
                        resamples=resamples,
                        seed=seed,
                    )
                )
    return rows


def _e17_interaction_rows(
    suite: LoadedSuite,
    records: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    index, duplicates = _record_index(records)
    configs = {source.config.experiment_id: source.config for source in suite.sources}
    cache: dict[str, SourceEvaluation] = {}
    rows: list[dict[str, Any]] = []
    for experiment_id, interactions in sorted(_declared_e17_interactions(suite).items()):
        config = configs[experiment_id]
        expected_cells = _stage_cell_keys(config, stage)
        stage_seeds = config.screen.seeds if stage == "screen" else config.confirm.seeds
        for interaction_id, arm_00, arm_10, arm_01, arm_11 in interactions:
            arms = (arm_00, arm_10, arm_01, arm_11)
            base = {
                "experiment_id": experiment_id,
                "decision_id": f"interaction_{interaction_id}",
                "comparison_id": f"interaction_{interaction_id}",
                "cells": list(arms),
                "task_id": "__task_macro__",
                "metric": "task_macro_global_F1_interaction_residual",
                "endpoint_scope": "overall",
                "confirmatory": True,
                "multiplicity_family": experiment_id,
            }
            if stage == "confirm" and len(set(stage_seeds)) < 3:
                rows.append(
                    _unavailable_row(
                        base,
                        reason_code="insufficient_confirm_seeds",
                        reason="confirmatory inference requires at least three paired seeds",
                        paired_seeds=sorted(set(stage_seeds)),
                    )
                )
                continue
            cells_by_arm, cell_details = _arm_cell_records(
                index,
                duplicates,
                experiment_id=experiment_id,
                arms=arms,
                expected_cells=expected_cells,
            )
            if cells_by_arm is None:
                rows.append(
                    _unavailable_row(
                        base,
                        reason_code="unequal_or_missing_cells",
                        reason="2x2 cells do not have the exact declared task/seed cells",
                        **cell_details,
                    )
                )
                continue
            pool_status, fingerprints, pool_error = _candidate_pool_guard(
                experiment_id, cells_by_arm
            )
            if pool_error is not None:
                rows.append(
                    _unavailable_row(
                        base,
                        reason_code=pool_status,
                        reason=pool_error,
                        fatal=pool_status != "deferred_e17_candidate_pool_policy",
                        candidate_pool_guard=pool_status,
                        candidate_pool_fingerprints=fingerprints,
                    )
                )
                continue
            try:
                evaluations = {
                    arm: {
                        cell: _overall_evaluation(record, cache)
                        for cell, record in cells_by_arm[arm].items()
                    }
                    for arm in arms
                }
                result = e17_interaction_bootstrap(
                    evaluations[arm_00],
                    evaluations[arm_10],
                    evaluations[arm_01],
                    evaluations[arm_11],
                    resamples=resamples,
                    seed=seed,
                )
            except Exception as exc:
                rows.append(
                    _unavailable_row(
                        base,
                        reason_code="artifact_verification_failed",
                        reason=f"{type(exc).__name__}: {exc}",
                        fatal=stage == "confirm",
                        candidate_pool_guard=pool_status,
                        candidate_pool_fingerprints=fingerprints,
                    )
                )
                continue
            rows.append(
                {
                    **base,
                    "status": "complete",
                    "fatal": False,
                    "candidate_pool_guard": pool_status,
                    "candidate_pool_fingerprints": fingerprints,
                    **result.as_dict(),
                }
            )
    return rows


def _apply_bootstrap_multiplicity(
    rows: list[dict[str, Any]],
    suite: LoadedSuite,
    *,
    stage: str,
) -> None:
    policies = {
        source.config.experiment_id: source.config.design.multiplicity for source in suite.sources
    }
    for row in rows:
        policy = policies.get(str(row.get("experiment_id")), "none")
        row["multiplicity"] = policy
        if row.get("status") == "complete" and policy == "none":
            row["p_value_adjusted"] = row.get("p_value")
    if stage != "confirm":
        return
    for experiment_id, policy in sorted(policies.items()):
        if policy != "holm":
            continue
        family = [
            row
            for row in rows
            if row.get("experiment_id") == experiment_id
            and row.get("status") == "complete"
            and row.get("confirmatory") is True
            and row.get("endpoint_scope") == "overall"
        ]
        raw = {str(row["comparison_id"]): float(row["p_value"]) for row in family}
        adjusted = holm_adjust_p_values(raw)
        for row in family:
            row["p_value_adjusted"] = adjusted[str(row["comparison_id"])]
            row["p_value_adjustment"] = "holm"
            row["multiplicity_family_size"] = len(family)


def _finalize_stage_reports(
    suite: LoadedSuite,
    *,
    stage: str,
    stage_root: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    long_rows, macro_rows = metric_reports(records)
    _atomic_json(
        stage_root / "metrics_long.json",
        {"schema_version": 1, "rows": long_rows},
    )
    _atomic_csv_rows(stage_root / "metrics_long.csv", long_rows)
    _atomic_json(
        stage_root / "metrics_macro.json",
        {"schema_version": 1, "rows": macro_rows},
    )
    _atomic_csv_rows(stage_root / "metrics_macro.csv", macro_rows)

    bootstrap_seed = stable_bootstrap_seed(suite.suite_id, stage)
    bootstrap_rows = _paired_bootstrap_rows(
        suite,
        records,
        stage=stage,
        resamples=10_000,
        seed=bootstrap_seed,
    )
    bootstrap_rows.extend(
        _e17_interaction_rows(
            suite,
            records,
            stage=stage,
            resamples=10_000,
            seed=bootstrap_seed,
        )
    )
    _apply_bootstrap_multiplicity(bootstrap_rows, suite, stage=stage)
    _atomic_json(
        stage_root / "paired_bootstrap.json",
        {
            "schema_version": 1,
            "resampling_unit": "source_entity",
            "default_resamples": 10_000,
            "rows": bootstrap_rows,
        },
    )
    _atomic_csv_rows(stage_root / "paired_bootstrap.csv", bootstrap_rows)
    fatal = [
        f"{row.get('experiment_id')}/{row.get('comparison_id')}: {row.get('reason')}"
        for row in bootstrap_rows
        if row.get("fatal") is True
    ]
    if fatal:
        raise ValueError(
            "paired inference failed after writing aggregate reports: " f"{sorted(fatal)}"
        )


def aggregate_stage(
    suite: LoadedSuite,
    *,
    stage: str,
    output_root: Path,
    manifests: Optional[Iterable[Mapping[str, Any]]] = None,
    finalize_reports: bool = True,
) -> list[dict[str, Any]]:
    stage_root = Path(output_root).expanduser().resolve() / suite.suite_id / stage
    records: list[dict[str, Any]] = []
    if manifests is None:
        paths = sorted((stage_root / "runs").glob(f"**/{MANIFEST_NAME}"))
        loaded: Iterable[Mapping[str, Any]] = (_load_json(path) for path in paths)
    else:
        loaded = manifests
    manifest_list = list(loaded)
    for manifest in manifest_list:
        output_dir_value = manifest.get("fingerprint_payload", {}).get("output_dir")
        if output_dir_value:
            output_dir = Path(str(output_dir_value))
        else:
            output_dir = (
                stage_root
                / "runs"
                / str(manifest.get("experiment_id"))
                / str(manifest.get("arm_id"))
                / str(manifest.get("task_id"))
                / f"seed-{manifest.get('seed')}"
            )
        metrics: dict[str, float] = {}
        metric_error: Optional[dict[str, str]] = None
        if manifest.get("status") == "complete":
            try:
                metrics = cell_metrics(output_dir)
            except (OSError, TypeError, ValueError) as exc:
                metric_error = {"type": type(exc).__name__, "message": str(exc)}
        row: dict[str, Any] = {
            "output_dir": str(output_dir),
            "experiment_id": manifest.get("experiment_id"),
            "stage": manifest.get("stage"),
            "arm_id": manifest.get("arm_id"),
            "task_id": manifest.get("task_id"),
            "seed": manifest.get("seed"),
            "status": manifest.get("status"),
            "supervision_label": manifest.get("supervision_label"),
            "config_hash": manifest.get("resolved_config_hash"),
            "candidate_pool_fingerprint": manifest.get("candidate_pool_fingerprint"),
            "llm_usage": manifest.get("llm_usage"),
            "wall_seconds": manifest.get("wall_seconds"),
            "peak_memory_kb": manifest.get("peak_memory_kb"),
            "failure": (manifest.get("failure") or {}).get("message"),
            "metric_error": metric_error,
            "metrics": metrics,
        }
        records.append(row)
    records.sort(
        key=lambda row: (
            str(row["experiment_id"]),
            str(row["arm_id"]),
            str(row["task_id"]),
            int(row["seed"] or 0),
        )
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(stage_root / "metrics.json", {"schema_version": 1, "rows": records})
    metric_keys = sorted(
        {key for record in records for key in (record.get("metrics") or {}).keys()}
    )
    fieldnames = [
        "output_dir",
        "experiment_id",
        "stage",
        "arm_id",
        "task_id",
        "seed",
        "status",
        "supervision_label",
        "config_hash",
        "candidate_pool_fingerprint",
        "llm_usage",
        "wall_seconds",
        "peak_memory_kb",
        "failure",
        "metric_error",
        *metric_keys,
    ]
    csv_path = stage_root / "metrics.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            flattened = {key: record.get(key) for key in fieldnames}
            for structured_field in ("llm_usage", "metric_error"):
                if isinstance(flattened.get(structured_field), Mapping):
                    flattened[structured_field] = canonical_json(flattened[structured_field])
            flattened.update(record.get("metrics") or {})
            writer.writerow(flattened)
    os.replace(temporary, csv_path)
    _write_per_source_index(stage_root, records)
    long_rows, macro_rows = metric_reports(records)
    _atomic_json(
        stage_root / "metrics_long.json",
        {"schema_version": 1, "rows": long_rows},
    )
    _atomic_csv_rows(stage_root / "metrics_long.csv", long_rows)
    _atomic_json(
        stage_root / "metrics_macro.json",
        {"schema_version": 1, "rows": macro_rows},
    )
    _atomic_csv_rows(stage_root / "metrics_macro.csv", macro_rows)

    inventory_path = stage_root / "dataset_inventory.json"
    if inventory_path.is_file():
        inventory_payload = _load_json(inventory_path)
        inventory_rows = inventory_payload.get("rows") or []
        if not isinstance(inventory_rows, list):
            raise ValueError(f"invalid dataset inventory rows at {inventory_path}")
        enriched = enrich_inventory_from_manifests(inventory_rows, manifest_list)
        _write_inventory(
            suite,
            stage=stage,
            output_root=output_root,
            rows=enriched,
        )
    if finalize_reports:
        _finalize_stage_reports(
            suite,
            stage=stage,
            stage_root=stage_root,
            records=records,
        )
    return records


def _write_per_source_index(stage_root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    entries: list[dict[str, Any]] = []
    for record in records:
        run_dir = (
            stage_root
            / "runs"
            / str(record.get("experiment_id"))
            / str(record.get("arm_id"))
            / str(record.get("task_id"))
            / f"seed-{record.get('seed')}"
        )
        candidates = [
            run_dir / "explanations" / "index.json",
            run_dir / "alignment" / "maps_local.tsv",
            run_dir / "alignment" / "maps_global.tsv",
        ]
        entries.append(
            {
                "experiment_id": record.get("experiment_id"),
                "arm_id": record.get("arm_id"),
                "task_id": record.get("task_id"),
                "seed": record.get("seed"),
                "status": record.get("status"),
                "artifacts": [
                    str(path.relative_to(stage_root)) for path in candidates if path.is_file()
                ],
            }
        )
    _atomic_json(
        stage_root / "per_source_outputs.json",
        {"schema_version": 1, "runs": entries},
    )


def _scores_by_arm(
    records: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    metric: str,
) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    failed: set[str] = set()
    missing: list[str] = []
    for record in records:
        if record.get("experiment_id") != experiment_id:
            continue
        arm = str(record.get("arm_id"))
        if record.get("status") != "complete":
            failed.add(arm)
            continue
        value = _metric_value(record.get("metrics") or {}, metric)
        if value is None:
            missing.append(f"{arm}/{record.get('task_id')}/seed-{record.get('seed')}")
            continue
        if not math.isfinite(value):
            raise ValueError(
                f"{experiment_id}: selection metric {metric!r} is non-finite for "
                f"{arm}/{record.get('task_id')}/seed-{record.get('seed')}"
            )
        values.setdefault(arm, []).append(value)
    if failed:
        raise ValueError(
            f"{experiment_id}: selection cannot ignore failed cells for arms {sorted(failed)}"
        )
    if missing:
        raise ValueError(
            f"{experiment_id}: complete cells are missing selection metric "
            f"{metric!r}: {sorted(missing)}"
        )
    return {arm: sum(entries) / len(entries) for arm, entries in values.items() if entries}


def select_experiment(
    source: ExperimentSource,
    records: Sequence[Mapping[str, Any]],
    *,
    promoted_component_overlays: Optional[Mapping[str, Mapping[str, Any]]] = None,
    baseline_manifest_hash: Optional[str] = None,
) -> dict[str, Any]:
    config = source.config
    resolved_arms = _component_arms(
        config,
        promoted_overlays=promoted_component_overlays,
    )
    arms = {arm.id: arm for arm in resolved_arms}
    decisions: list[dict[str, Any]] = []
    all_selected = True
    for rule in config.selection.decisions:
        scores = _scores_by_arm(records, experiment_id=config.experiment_id, metric=rule.metric)
        required = {rule.baseline, *rule.candidates}
        missing = sorted(required.difference(scores))
        if missing:
            raise ValueError(
                f"{config.experiment_id}/{rule.id}: missing selection metric "
                f"{rule.metric!r} for arms {missing}"
            )
        baseline_score = scores[rule.baseline]
        eligible: list[tuple[float, str, float]] = []
        for arm_id in rule.candidates:
            score = scores[arm_id]
            signed_delta = (
                score - baseline_score if rule.direction == "max" else baseline_score - score
            )
            if signed_delta >= rule.min_delta:
                ranking = score if rule.direction == "max" else -score
                eligible.append((ranking, arm_id, signed_delta))
        eligible.sort(key=lambda item: (-item[0], item[1]))
        selected = eligible[0] if eligible else None
        if selected is None and not rule.allow_screened_out:
            raise ValueError(
                f"{config.experiment_id}/{rule.id}: no candidate passed and screened_out is forbidden"
            )
        all_selected = all_selected and selected is not None
        selected_arm = selected[1] if selected else None
        decisions.append(
            {
                "id": rule.id,
                "baseline": rule.baseline,
                "baseline_score": baseline_score,
                "metric": rule.metric,
                "direction": rule.direction,
                "min_delta": rule.min_delta,
                "candidate_scores": {arm_id: scores[arm_id] for arm_id in sorted(rule.candidates)},
                "selected_arm": selected_arm,
                "selected_score": scores[selected_arm] if selected_arm else None,
                "signed_delta": selected[2] if selected else None,
                "selected_overlay": arms[selected_arm].overlay if selected_arm else None,
                "required_controls": list(rule.required_controls),
            }
        )
    base = ConfigModel.from_mapping(_base_mapping(source), warn_v1=False).model_dump(
        mode="json", by_alias=True
    )
    return {
        "experiment_config_hash": source.raw_hash(),
        "resolved_arms_hash": hash_payload({arm.id: arm.overlay for arm in resolved_arms}),
        "base_config_hash": hash_payload(base),
        "design_hash": experiment_design_hash(
            source,
            baseline_manifest_hash=baseline_manifest_hash,
        ),
        "candidate_pool_design_hash": _candidate_design_hash(base),
        "status": "selected" if all_selected else "screened_out",
        "decisions": decisions,
    }


def selected_experiment_overlays(
    selection_record: Mapping[str, Any],
    dependencies: Sequence[str],
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    experiments = selection_record.get("experiments") or {}
    for dependency in dependencies:
        record = experiments.get(dependency)
        if not isinstance(record, Mapping) or record.get("status") != "selected":
            continue
        overlay: dict[str, Any] = {}
        for decision in record.get("decisions") or []:
            if isinstance(decision, Mapping) and isinstance(
                decision.get("selected_overlay"), Mapping
            ):
                overlay = deep_merge(overlay, decision["selected_overlay"])
        selected[dependency] = overlay
    return selected


def inherited_selection_overlay(
    selection_record: Mapping[str, Any],
    dependencies: Sequence[str],
) -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    for dependency_overlay in selected_experiment_overlays(selection_record, dependencies).values():
        overlay = deep_merge(overlay, dependency_overlay)
    return overlay


def _design_record(
    suite: LoadedSuite,
    frozen_selection: Mapping[str, Any],
) -> dict[str, Any]:
    experiments = {
        source.config.experiment_id: {
            "design": _experiment_design_payload(
                source,
                baseline_manifest_hash=suite.baseline_manifest_hash,
            ),
            "design_hash": experiment_design_hash(
                source,
                baseline_manifest_hash=suite.baseline_manifest_hash,
            ),
        }
        for source in suite.sources
    }
    record = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "suite_hash": suite.suite_hash,
        "baseline_id": suite.baseline_id,
        "baseline_manifest_hash": suite.baseline_manifest_hash,
        "dataset_lock_hash": suite.dataset_lock_hash,
        "experiments": experiments,
        "frozen_selection": _jsonable(frozen_selection),
    }
    record["design_record_hash"] = hash_payload(record)
    return record


def write_selection_record(
    suite: LoadedSuite,
    experiments: Mapping[str, Any],
    *,
    output_root: Path,
) -> Path:
    suite_root = Path(output_root).expanduser().resolve() / suite.suite_id
    confirm_root = suite_root / "confirm"
    if confirm_root.exists() and any(path.is_file() for path in confirm_root.rglob("*")):
        raise FileExistsError(
            "cannot overwrite frozen selection/design after confirm artifacts exist"
        )
    stage_root = suite_root / "screen"
    design = _design_record(suite, experiments)
    _atomic_json(stage_root / "design.json", design)
    record: dict[str, Any] = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "stage": "screen",
        "suite_hash": suite.suite_hash,
        "baseline_id": suite.baseline_id,
        "baseline_manifest_hash": suite.baseline_manifest_hash,
        "dataset_lock_hash": suite.dataset_lock_hash,
        "design_record": "design.json",
        "design_record_hash": design["design_record_hash"],
        "experiments": _jsonable(experiments),
    }
    record["selection_hash"] = hash_payload(record)
    path = stage_root / "selection.json"
    _atomic_json(path, record)
    return path


def load_and_validate_selection(
    path: Path,
    suite: LoadedSuite,
) -> dict[str, Any]:
    record = _load_json(Path(path).expanduser().resolve())
    embedded_hash = record.pop("selection_hash", None)
    computed_hash = hash_payload(record)
    record["selection_hash"] = embedded_hash
    if not embedded_hash or embedded_hash != computed_hash:
        raise ValueError("selection record content does not match its frozen selection_hash")
    if record.get("stage") != "screen":
        raise ValueError("confirm requires a screen-stage selection record")
    if record.get("suite_id") != suite.suite_id:
        raise ValueError(
            f"selection suite {record.get('suite_id')!r} does not match {suite.suite_id!r}"
        )
    if record.get("suite_hash") != suite.suite_hash:
        raise ValueError("suite manifest changed after screening")
    if record.get("baseline_manifest_hash") != suite.baseline_manifest_hash:
        raise ValueError("baseline manifest changed after screening")
    if record.get("dataset_lock_hash") != suite.dataset_lock_hash:
        raise ValueError("dataset lock changed after screening")
    design_path = Path(path).expanduser().resolve().parent / str(
        record.get("design_record", "design.json")
    )
    design = _load_json(design_path)
    design_hash = design.pop("design_record_hash", None)
    computed_design_hash = hash_payload(design)
    design["design_record_hash"] = design_hash
    if design_hash != computed_design_hash or design_hash != record.get("design_record_hash"):
        raise ValueError("frozen design record changed after screening")

    experiments = record.get("experiments") or {}
    if design.get("frozen_selection") != experiments:
        raise ValueError("selection outcomes disagree with the immutable design record")
    for source in suite.sources:
        experiment_id = source.config.experiment_id
        frozen = experiments.get(experiment_id)
        if not isinstance(frozen, Mapping):
            raise ValueError(f"selection record is missing {experiment_id}")
        if frozen.get("experiment_config_hash") != source.raw_hash():
            raise ValueError(f"{experiment_id}: experiment config changed after screening")
        if frozen.get("design_hash") != experiment_design_hash(
            source,
            baseline_manifest_hash=suite.baseline_manifest_hash,
        ):
            raise ValueError(f"{experiment_id}: confirmatory design changed after screening")
        if frozen.get("status") in {"selected", "screened_out"}:
            promoted = (
                selected_experiment_overlays(record, source.config.depends_on)
                if experiment_id == "E17"
                else None
            )
            resolved_arms = _component_arms(
                source.config,
                promoted_overlays=promoted,
            )
            resolved_arms_hash = hash_payload({arm.id: arm.overlay for arm in resolved_arms})
            if frozen.get("resolved_arms_hash") != resolved_arms_hash:
                raise ValueError(
                    f"{experiment_id}: resolved arm composition changed after screening"
                )
        base = ConfigModel.from_mapping(_base_mapping(source), warn_v1=False).model_dump(
            mode="json", by_alias=True
        )
        if frozen.get("base_config_hash") != hash_payload(base):
            raise ValueError(f"{experiment_id}: base config changed after screening")
        if frozen.get("candidate_pool_design_hash") != _candidate_design_hash(base):
            raise ValueError(f"{experiment_id}: candidate-pool design changed after screening")
    return record


def print_dry_run(
    suite: LoadedSuite,
    *,
    stage: str,
    output_root: Path,
    selection_record: Optional[Mapping[str, Any]] = None,
) -> list[RunCell]:
    cells: list[RunCell] = []
    for source in suite.sources:
        config = source.config
        dependencies = ", ".join(config.depends_on) or "-"
        if config.implementation.status != "ready":
            print(
                f"DEFERRED\t{config.experiment_id}\tdepends={dependencies}\t"
                f"{config.implementation.status}: {config.implementation.reason}"
            )
            continue
        promoted_components = (
            selected_experiment_overlays(selection_record, config.depends_on)
            if config.experiment_id == "E17" and selection_record is not None
            else (
                {
                    component.source_experiment: {}
                    for component in config.composition.components
                    if component.source_experiment is not None
                }
                if config.experiment_id == "E17"
                and stage == "screen"
                and config.composition is not None
                else None
            )
        )
        inherited = (
            {}
            if config.experiment_id == "E17"
            else (
                inherited_selection_overlay(selection_record, config.depends_on)
                if selection_record is not None
                else {}
            )
        )
        experiment_cells = build_cells(
            suite,
            source,
            stage=stage,
            output_root=output_root,
            selection_record=selection_record,
            inherited_overlay=inherited,
            promoted_component_overlays=promoted_components,
        )
        if not experiment_cells and stage == "confirm":
            print(f"SCREENED_OUT\t{config.experiment_id}\tdepends={dependencies}")
        for cell in experiment_cells:
            row = cell.dry_run_row()
            print(
                "RUN\t{cell}\tresource={resource}\tsupervision={supervision}\t"
                "cap={source_cap}\tconfig={config_hash}\toutput={output}".format(**row)
            )
        cells.extend(experiment_cells)
    return cells


def run_stage(
    suite: LoadedSuite,
    *,
    stage: str,
    output_root: Path,
    jobs: int,
    resume: bool,
    workdir: Path,
    selection_record_path: Optional[Path] = None,
    dry_run: bool = False,
) -> Optional[Path]:
    if stage not in {"screen", "confirm"}:
        raise ValueError("stage must be screen or confirm")
    selection: Optional[dict[str, Any]] = None
    if stage == "confirm":
        if selection_record_path is None:
            raise ValueError("confirm requires --selection-record")
        selection = load_and_validate_selection(selection_record_path, suite)
    elif selection_record_path is not None:
        raise ValueError("--selection-record is only valid for confirm")

    if dry_run:
        print_dry_run(
            suite,
            stage=stage,
            output_root=output_root,
            selection_record=selection,
        )
        return None

    build_dataset_inventory(
        suite,
        stage=stage,
        output_root=output_root,
        selection_record=selection,
    )

    all_manifests: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    incremental_selection: dict[str, Any] = {
        "experiments": selections,
        "selection_hash": None,
    }
    for source in suite.sources:
        config = source.config
        if config.implementation.status != "ready":
            if stage == "screen":
                selections[config.experiment_id] = {
                    "experiment_config_hash": source.raw_hash(),
                    "base_config_hash": hash_payload(
                        ConfigModel.from_mapping(_base_mapping(source), warn_v1=False).model_dump(
                            mode="json", by_alias=True
                        )
                    ),
                    "design_hash": experiment_design_hash(
                        source,
                        baseline_manifest_hash=suite.baseline_manifest_hash,
                    ),
                    "candidate_pool_design_hash": _candidate_design_hash(
                        ConfigModel.from_mapping(_base_mapping(source), warn_v1=False).model_dump(
                            mode="json", by_alias=True
                        )
                    ),
                    "status": config.implementation.status,
                    "reason": config.implementation.reason,
                    "decisions": [],
                }
            continue
        source_selection = selection if stage == "confirm" else incremental_selection
        assert source_selection is not None
        promoted_components = (
            selected_experiment_overlays(source_selection, config.depends_on)
            if config.experiment_id == "E17"
            else None
        )
        inherited = (
            {}
            if config.experiment_id == "E17"
            else inherited_selection_overlay(source_selection, config.depends_on)
        )
        cells = build_cells(
            suite,
            source,
            stage=stage,
            output_root=output_root,
            selection_record=selection,
            inherited_overlay=inherited,
            promoted_component_overlays=promoted_components,
        )
        manifests = run_cells(
            cells,
            suite,
            workdir=workdir,
            jobs=jobs,
            resume=resume,
        )
        if stage == "confirm":
            _validate_paired_llm_identities(manifests)
        all_manifests.extend(manifests)
        records = aggregate_stage(
            suite,
            stage=stage,
            output_root=output_root,
            manifests=all_manifests,
            finalize_reports=False,
        )
        _require_successful_cells(manifests, stage=stage)
        if stage == "screen":
            selections[config.experiment_id] = select_experiment(
                source,
                records,
                promoted_component_overlays=promoted_components,
                baseline_manifest_hash=suite.baseline_manifest_hash,
            )

    aggregate_stage(
        suite,
        stage=stage,
        output_root=output_root,
        manifests=all_manifests,
        finalize_reports=True,
    )
    if stage == "screen":
        return write_selection_record(suite, selections, output_root=output_root)
    return None


__all__ = [
    "LoadedSuite",
    "RunCell",
    "aggregate_stage",
    "build_cells",
    "canonical_json",
    "deep_merge",
    "hash_payload",
    "inherited_selection_overlay",
    "load_and_validate_selection",
    "load_suite_or_experiment",
    "print_dry_run",
    "run_stage",
    "select_experiment",
    "write_selection_record",
]
