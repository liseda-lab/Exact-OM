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
from dataclasses import dataclass, replace
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
    SelectionDecisionConfig,
    SelectionGuardConfig,
    SelectionTieBreakConfig,
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


def specification_tree_identity(path: Path, *, relative_to: Path) -> dict[str, Any]:
    """Hash Markdown specs using the clarification's length-prefixed wire format."""

    root = Path(path).expanduser().resolve()
    base = Path(relative_to).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"specification tree does not exist: {root}")
    files = sorted(
        (candidate for candidate in root.rglob("*.md") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(base).as_posix(),
    )
    if not files:
        raise ValueError(f"specification tree contains no Markdown files: {root}")
    digest = hashlib.sha256()
    for candidate in files:
        try:
            relative = candidate.relative_to(base).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"specification file {candidate} is outside provenance base {base}"
            ) from exc
        path_bytes = relative.encode("utf-8")
        content = candidate.read_bytes()
        digest.update(len(path_bytes).to_bytes(8, "big", signed=False))
        digest.update(path_bytes)
        digest.update(len(content).to_bytes(8, "big", signed=False))
        digest.update(content)
    return {
        "path": str(root),
        "sha256": digest.hexdigest(),
        "files": len(files),
        "algorithm": "sha256-length-prefixed-v1",
    }


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
    specification: dict[str, Any]
    model_lock: Optional[Path] = None
    model_lock_hash: Optional[str] = None
    model_lock_payload: Optional[dict[str, Any]] = None
    confirmed_components_record: Optional[Path] = None
    confirmed_components_hash: Optional[str] = None
    confirmed_parent_selection_hash: Optional[str] = None
    confirmed_component_overlays: Optional[dict[str, dict[str, Any]]] = None
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
    reference_completeness: str
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


_FROZEN_EXPERIMENT_DEFAULTS: dict[tuple[str, ...], Any] = {
    ("matching", "extraction", "mode"): "greedy",
    ("matching", "anchor_rescoring", "mode"): "off",
    ("matching", "calibration", "mode"): "none",
    ("matching", "calibration", "threshold_mode"): "fixed",
    ("matching", "calibration", "artifact"): None,
    ("matching", "nil", "mode"): "off",
    ("matching", "fusion", "enabled"): False,
    ("matching", "channels", "strsim", "enabled"): False,
    ("matching", "channels", "attr", "enabled"): False,
    ("matching", "channels", "hier", "enabled"): False,
    ("matching", "channels", "diff", "enabled"): False,
    ("matching", "channels", "lex", "enabled"): False,
    ("matching", "channels", "property", "enabled"): False,
    ("matching", "channels", "instance", "enabled"): False,
    ("matching", "channels", "graph", "mode"): "off",
    ("matching", "relation_prediction"): "none",
    ("selector", "enabled"): False,
    ("selector", "runtime_enabled"): None,
    ("llm", "experiment", "enabled"): False,
    ("candidates", "adaptive_k", "enabled"): False,
    ("candidates", "encoder_finetune", "mode"): "off",
    ("candidates", "cross_encoder", "mode"): "off",
    ("candidates", "multi_view", "mode"): "labels",
    ("run", "experiment_audit"): False,
}


def _assert_experiment_flags_disabled(config_path: Path) -> None:
    mapping = ConfigModel.from_mapping(load_yaml_mapping(config_path), warn_v1=False).model_dump(
        mode="json", by_alias=True
    )
    violations: dict[str, tuple[Any, Any]] = {}
    for path, expected in _FROZEN_EXPERIMENT_DEFAULTS.items():
        value: Any = mapping
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if value != expected:
            violations[".".join(path)] = (expected, value)
    if violations:
        raise ValueError("frozen baseline has experimental switches enabled: " f"{violations}")


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
    _assert_experiment_flags_disabled(config_path)
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


def _validate_model_lock(path: Path) -> dict[str, Any]:
    payload = dict(load_yaml_mapping(path))
    models = payload.get("models")
    status = payload.get("status")
    if (
        payload.get("schema_version") != 1
        or status not in {"complete", "incomplete"}
        or not isinstance(models, Mapping)
        or not models
    ):
        raise ValueError(f"invalid model lock schema at {path}")
    requested_ids: dict[str, str] = {}
    for name, raw in models.items():
        if not str(name).strip() or not isinstance(raw, Mapping):
            raise ValueError(f"model lock has an invalid entry {name!r}")
        requested_id = str(raw.get("requested_id") or "").strip()
        if not requested_id:
            raise ValueError(f"model lock entry {name!r} has no requested_id")
        previous = requested_ids.setdefault(requested_id, str(name))
        if previous != str(name):
            raise ValueError(
                "model lock requested_id is ambiguous: "
                f"{requested_id!r} is declared by {previous!r} and {name!r}"
            )
        if status == "complete":
            revision = str(raw.get("resolved_revision") or "").strip().lower()
            artifact_hash = str(raw.get("artifact_sha256") or "").strip().lower()
            if not (
                re.fullmatch(r"[0-9a-f]{64}", artifact_hash)
                or re.fullmatch(r"[0-9a-f]{40}", revision)
            ):
                raise ValueError(
                    f"complete model lock entry {name!r} requires a 40-hex resolved "
                    "revision or 64-hex local artifact SHA-256"
                )
        elif not str(raw.get("reason") or "").strip():
            raise ValueError(f"incomplete model lock entry {name!r} requires a reason")
    return payload


def _configured_model_ids(mapping: Mapping[str, Any]) -> set[str]:
    """Return local/Hugging Face model IDs which must be revision locked."""

    required: set[str] = set()
    candidates = mapping.get("candidates")
    if isinstance(candidates, Mapping):
        encoder = candidates.get("encoder") or candidates.get("lexical_encoder_name")
        if encoder:
            required.add(str(encoder))
    for entry in mapping.get("pipeline") or ():
        if not isinstance(entry, Mapping):
            continue
        params = entry.get("params")
        if not isinstance(params, Mapping):
            continue
        for field in ("lexical_model_name", "context_model_name", "llm_model_name"):
            value = params.get(field)
            if value:
                required.add(str(value))
    llm = mapping.get("llm")
    profiles = llm.get("profiles") if isinstance(llm, Mapping) else None
    if isinstance(profiles, Mapping):
        for profile in profiles.values():
            if not isinstance(profile, Mapping):
                continue
            if str(profile.get("backend") or "").strip().lower() == "local_hf" and profile.get(
                "model"
            ):
                required.add(str(profile["model"]))
            if profile.get("tokenizer"):
                required.add(str(profile["tokenizer"]))
    return required


def _source_model_mappings(source: ExperimentSource) -> Iterable[Mapping[str, Any]]:
    """Yield every frozen overlay context which can introduce a runtime model."""

    base = _base_mapping(source)
    yield base
    tasks = tuple(source.config.screen.tasks) + tuple(source.config.confirm.tasks)
    for task in tasks:
        yield deep_merge(base, task.overlay)
    for arm in source.config.arms:
        arm_mapping = deep_merge(base, arm.overlay)
        yield arm_mapping
        for task in tasks:
            yield deep_merge(arm_mapping, task.overlay)
    composition = source.config.composition
    if composition is not None:
        yield deep_merge(base, composition.rolling_overlay)
        for component in composition.components:
            yield deep_merge(base, component.overlay)


def _validate_model_lock_bindings(
    sources: Sequence[ExperimentSource],
    lock: Mapping[str, Any],
) -> None:
    """Require the lock to name every model reachable from the frozen design."""

    required: set[str] = set()
    for source in sources:
        for mapping in _source_model_mappings(source):
            validated = ConfigModel.from_mapping(mapping, warn_v1=False).model_dump(
                mode="json", by_alias=True
            )
            required.update(_configured_model_ids(validated))
    models = lock.get("models")
    declared = {
        str(raw.get("requested_id"))
        for raw in (models or {}).values()
        if isinstance(raw, Mapping) and raw.get("requested_id")
    }
    missing = sorted(required.difference(declared))
    if missing:
        raise ValueError(f"model lock is missing configured runtime model identities: {missing}")


def _bind_model_lock_revisions(
    mapping: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    require_complete: bool,
) -> dict[str, Any]:
    """Inject immutable revisions into every local/HF loader configuration."""

    status = str(lock.get("status") or "")
    if status != "complete":
        if require_complete:
            raise ValueError("confirmation requires a complete immutable model lock")
        return dict(_jsonable(mapping))
    models = lock.get("models")
    by_id = {
        str(raw.get("requested_id")): raw
        for raw in (models or {}).values()
        if isinstance(raw, Mapping) and raw.get("requested_id")
    }
    bound = dict(_jsonable(mapping))

    def locked_revision(model_id: Any, *, context: str) -> Optional[str]:
        if not model_id:
            return None
        requested_id = str(model_id)
        entry = by_id.get(requested_id)
        if entry is None:
            raise ValueError(
                f"model lock is missing configured runtime model identity {requested_id!r} "
                f"for {context}"
            )
        revision = str(entry.get("resolved_revision") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(
                f"model lock entry for {requested_id!r} cannot bind {context}: "
                "a 40-hex resolved_revision is required by the runtime loader"
            )
        return revision

    def bind_field(
        container: dict[str, Any],
        *,
        model_field: str,
        revision_field: str,
        context: str,
    ) -> None:
        revision = locked_revision(container.get(model_field), context=context)
        if revision is None:
            return
        configured = container.get(revision_field)
        if configured is not None and str(configured).strip().lower() != revision:
            raise ValueError(
                f"configured {context} revision {configured!r} disagrees with "
                f"the immutable model lock revision {revision!r}"
            )
        container[revision_field] = revision

    candidates_value = bound.get("candidates")
    if isinstance(candidates_value, Mapping):
        candidates = dict(candidates_value)
        model_field = "encoder" if candidates.get("encoder") else "lexical_encoder_name"
        bind_field(
            candidates,
            model_field=model_field,
            revision_field="encoder_revision",
            context="candidate encoder",
        )
        bound["candidates"] = candidates

    pipeline: list[Any] = []
    for index, entry_value in enumerate(bound.get("pipeline") or ()):
        if not isinstance(entry_value, Mapping):
            pipeline.append(entry_value)
            continue
        entry = dict(entry_value)
        params_value = entry.get("params")
        if isinstance(params_value, Mapping):
            params = dict(params_value)
            for model_field, revision_field, label in (
                ("lexical_model_name", "lexical_model_revision", "lexical encoder"),
                ("context_model_name", "context_model_revision", "context encoder"),
                ("llm_model_name", "llm_model_revision", "pipeline local LLM"),
            ):
                bind_field(
                    params,
                    model_field=model_field,
                    revision_field=revision_field,
                    context=f"pipeline[{index}] {label}",
                )
            entry["params"] = params
        pipeline.append(entry)
    bound["pipeline"] = pipeline

    llm_value = bound.get("llm")
    if isinstance(llm_value, Mapping):
        llm = dict(llm_value)
        profiles_value = llm.get("profiles")
        if isinstance(profiles_value, Mapping):
            profiles: dict[str, Any] = {}
            for name, profile_value in profiles_value.items():
                if not isinstance(profile_value, Mapping):
                    profiles[str(name)] = profile_value
                    continue
                profile = dict(profile_value)
                if str(profile.get("backend") or "").strip().lower() == "local_hf":
                    bind_field(
                        profile,
                        model_field="model",
                        revision_field="revision",
                        context=f"local LLM profile {name!r}",
                    )
                bind_field(
                    profile,
                    model_field="tokenizer",
                    revision_field="tokenizer_revision",
                    context=f"LLM profile {name!r} tokenizer",
                )
                profiles[str(name)] = profile
            llm["profiles"] = profiles
        bound["llm"] = llm
    return bound


def _validate_task_lock_roles(
    sources: Sequence[ExperimentSource],
    lock: Mapping[str, Any],
) -> None:
    tracks = lock.get("tracks")
    if not isinstance(tracks, Mapping):
        raise ValueError("dataset lock has no track declarations")
    for source in sources:
        if source.config.implementation.status != "ready":
            continue
        for stage_name, stage in (
            ("screen", source.config.screen),
            ("confirm", source.config.confirm),
        ):
            for task in stage.tasks:
                if task.availability.status != "ready":
                    continue
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
                if "train_reference" in task.capabilities and "train" not in roles:
                    raise ValueError(
                        f"{source.config.experiment_id}/{stage_name}/{task.id}: declares "
                        f"train_reference capability but track {task.track!r} has no locked "
                        "train role"
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
        single_model_lock_path = collection_root / "models.lock.yaml"
        if not single_model_lock_path.is_file():
            raise FileNotFoundError(
                f"standard sibling model lock does not exist: {single_model_lock_path}"
            )
        single_model_lock = _validate_model_lock(single_model_lock_path)
        source = ExperimentSource(config, path)
        single_sources = (source,)
        _validate_model_lock_bindings(single_sources, single_model_lock)
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
        repository = _repository_root(source.base_config_path)
        specification = specification_tree_identity(
            repository / "specs" / "experiments",
            relative_to=repository,
        )
        single_lock_hash = sha256_file(single_lock_path)
        single_model_lock_hash = sha256_file(single_model_lock_path)
        baseline_hash = sha256_file(baseline_path)
        suite_hash = hash_payload(
            {
                "experiment_manifest_sha256": sha256_file(path),
                "baseline_manifest_sha256": baseline_hash,
                "dataset_lock_sha256": single_lock_hash,
                "model_lock_sha256": single_model_lock_hash,
                "specification_sha256": specification["sha256"],
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
            specification=specification,
            model_lock=single_model_lock_path,
            model_lock_hash=single_model_lock_hash,
            model_lock_payload=single_model_lock,
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
    model_lock_path = _resolve_path(suite.model_lock, path.parent)
    if not model_lock_path.is_file():
        raise FileNotFoundError(f"declared model lock does not exist: {model_lock_path}")
    model_lock = _validate_model_lock(model_lock_path)
    _validate_model_lock_bindings(sources, model_lock)
    model_lock_hash = sha256_file(model_lock_path)
    baseline_path = _resolve_path(suite.baseline_manifest, path.parent)
    if not baseline_path.is_file():
        raise FileNotFoundError(f"declared baseline manifest does not exist: {baseline_path}")
    baseline = _validate_baseline_manifest(
        baseline_path,
        expected_id=suite.baseline_id,
        sources=sources,
    )
    baseline_hash = sha256_file(baseline_path)
    repository = _repository_root(sources[0].base_config_path)
    specification_path = _resolve_path(suite.specification.path, path.parent)
    specification = specification_tree_identity(
        specification_path,
        relative_to=repository,
    )
    expected_specification = suite.specification.model_dump(mode="json")
    expected_specification["path"] = str(specification_path)
    mismatches = {
        key: (expected_specification.get(key), specification.get(key))
        for key in ("path", "sha256", "files", "algorithm")
        if expected_specification.get(key) != specification.get(key)
    }
    if mismatches:
        raise ValueError(
            "authoritative specification tree changed after the suite was frozen: " f"{mismatches}"
        )
    suite_hash = hash_payload(
        {
            "suite_manifest_sha256": sha256_file(path),
            "baseline_manifest_sha256": baseline_hash,
            "dataset_lock_sha256": lock_hash,
            "model_lock_sha256": model_lock_hash,
            "specification_sha256": specification["sha256"],
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
        specification=specification,
        model_lock=model_lock_path,
        model_lock_hash=model_lock_hash,
        model_lock_payload=model_lock,
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


def _stage_artifact_root(
    suite: LoadedSuite,
    *,
    stage: str,
    output_root: Path,
) -> Path:
    stage_root = Path(output_root).expanduser().resolve() / suite.suite_id / stage
    if suite.confirmed_components_hash is not None:
        return stage_root / "e17" / suite.confirmed_components_hash
    return stage_root


def _write_inventory(
    suite: LoadedSuite,
    *,
    stage: str,
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    stage_root = _stage_artifact_root(
        suite,
        stage=stage,
        output_root=output_root,
    )
    _atomic_json(
        stage_root / "dataset_inventory.json",
        {
            "schema_version": 1,
            "suite_id": suite.suite_id,
            "stage": stage,
            "suite_hash": suite.suite_hash,
            "baseline_manifest": _baseline_record(suite),
            "specification": suite.specification,
            "dataset_lock": (
                {"path": str(suite.dataset_lock), "sha256": suite.dataset_lock_hash}
                if suite.dataset_lock is not None
                else None
            ),
            "confirmed_components_hash": suite.confirmed_components_hash,
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
            if task.availability.status != "ready":
                rows.append(
                    {
                        "experiment_id": config.experiment_id,
                        "task_id": task.id,
                        "stage": stage,
                        "split_role": task.split_role,
                        "reference_role": task.reference_role,
                        "reference_completeness": task.reference_completeness,
                        "status": task.availability.status,
                        "reason_code": task.availability.reason_code,
                        "missing_capability": task.availability.missing_capability,
                        "expected_dataset": task.availability.expected_dataset,
                        "reason": task.availability.reason,
                    }
                )
                continue
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
        "frozen_constants": config.frozen_constants,
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
    if label == "cross_pair_transfer":
        raise NotImplementedError(
            "cross_pair_transfer requires immutable donor-fit and recipient-application "
            "orchestration; refusing to fit from the recipient task's training reference"
        )
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
            if task.availability.status != "ready":
                continue
            source_cap = task.source_cap or stage_config.source_cap
            for seed in stage_config.seeds:
                resolved, _unbound_hash, label, supervision = _resolve_config(
                    source,
                    task=task,
                    arm=arm,
                    stage=stage,
                    seed=seed,
                    source_cap=source_cap,
                    inherited_overlay=inherited_overlay or {},
                )
                resolved = _bind_model_lock_revisions(
                    resolved,
                    suite.model_lock_payload,
                    require_complete=stage == "confirm",
                )
                validated = ConfigModel.from_mapping(resolved, warn_v1=False)
                resolved = validated.model_dump(mode="json", by_alias=True)
                config_hash = validated.fingerprint()
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
                        reference_completeness=task.reference_completeness,
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
                "lexical_revision": params.get("lexical_model_revision"),
                "context_model": params.get("context_model_name"),
                "context_revision": params.get("context_model_revision"),
                "llm_model": params.get("llm_model_name"),
                "llm_revision": params.get("llm_model_revision"),
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
                for key in (
                    "backend",
                    "model",
                    "revision",
                    "tokenizer",
                    "tokenizer_revision",
                )
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
            "revision": candidates.get("encoder_revision"),
        },
        "llm_profiles": safe_profiles,
    }


def _llm_runtime_required(mapping: Mapping[str, Any]) -> bool:
    llm = mapping.get("llm")
    experiment = llm.get("experiment") if isinstance(llm, Mapping) else None
    if isinstance(experiment, Mapping) and experiment.get("enabled") is True:
        gate = experiment.get("gate")
        gate_mode = str((gate or {}).get("mode") or "analytic").strip().lower()
        if gate_mode in {"off", "oracle"}:
            return False
    for entry in mapping.get("pipeline") or ():
        if not isinstance(entry, Mapping):
            continue
        params = entry.get("params")
        if isinstance(params, Mapping) and params.get("use_llm") is True:
            return True
    return isinstance(experiment, Mapping) and experiment.get("enabled") is True


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
    "request_time",
    "requested_revision",
    "requested_model",
    "requested_profile",
    "resolved_revision",
    "scoring_mode",
    "status",
    "tokenizer",
    "tokenizer_revision",
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
        if normalized_field == "endpoint_identity":
            if "://" in value:
                return _safe_endpoint_identity(value)
            identity = value.split("?", 1)[0].split("#", 1)[0].strip()
            if (
                identity
                and "@" not in identity
                and len(identity) <= 512
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", identity)
            ):
                return identity
            return None
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
        "model_lock_hash": suite.model_lock_hash,
        "specification_sha256": suite.specification.get("sha256"),
        "confirmed_components_hash": (
            suite.confirmed_components_hash if cell.experiment_id == "E17" else None
        ),
        "inputs": inputs,
        "candidate_pool_design_hash": candidate_design_hash,
        "artifacts": artifacts,
        "models": _model_identities(cell.resolved_config),
        "llm_required": _llm_runtime_required(cell.resolved_config),
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
        "reference_completeness": cell.reference_completeness,
        "seed": cell.seed,
        "source_cap": cell.source_cap,
        "resource": cell.resource,
        "resolved_config_hash": cell.config_hash,
        "experiment_config_hash": cell.experiment_config_hash,
        "selection_record_hash": cell.selection_hash,
        "design_declaration_hash": cell.design_hash,
        "baseline_manifest": _baseline_record(suite),
        "specification": suite.specification,
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
        "candidate_recall": run_stats.get("candidate_recall"),
        "candidate_recall_after_exact": run_stats.get("candidate_recall_after_exact"),
        "candidate_recall_diagnostics": run_stats.get("candidate_recall_diagnostics"),
        "mean_pool_size": run_stats.get("mean_pool_size"),
        "gold_rank_p90": run_stats.get("gold_rank_p90"),
        "gold_rank_median": run_stats.get("gold_rank_median"),
        "coverage": run_stats.get("coverage"),
        "abstention_rate": run_stats.get("abstention_rate"),
        "metric_applicability": run_stats.get("metric_applicability"),
        "explanation_reconstruction": run_stats.get("explanation_reconstruction"),
        "selection_evidence": run_stats.get("selection_evidence"),
        "observed_execution": run_stats.get("observed_execution"),
        "timing_ledger": timing,
        "llm_usage": llm_usage,
    }


_RUNTIME_LLM_IDENTITY_KEYS = {
    "backend",
    "cache_hashes",
    "effective_model",
    "endpoint",
    "endpoint_identity",
    "profile",
    "provider",
    "prompt_hashes",
    "request_seed",
    "request_time",
    "requested_revision",
    "requested_model",
    "requested_profile",
    "resolved_revision",
    "tokenizer",
    "tokenizer_revision",
    "decoding",
}

_PAIRED_LLM_IDENTITY_KEYS = {
    "backend",
    "effective_model",
    "endpoint",
    "endpoint_identity",
    "profile",
    "provider",
    "requested_model",
    "requested_profile",
    "requested_revision",
    "resolved_revision",
    "tokenizer",
    "tokenizer_revision",
}

_REQUIRED_LLM_IDENTITY_FIELDS = {
    "provider",
    "requested_model",
    "requested_revision",
    "effective_model",
    "resolved_revision",
    "tokenizer",
    "tokenizer_revision",
    "prompt_hashes",
    "decoding",
    "request_seed",
    "cache_hashes",
    "request_time",
}


def _validate_complete_llm_identity(identity: Mapping[str, Any], *, cell: str) -> None:
    missing = sorted(
        field
        for field in _REQUIRED_LLM_IDENTITY_FIELDS
        if identity.get(field) in (None, "", [], {})
    )
    if not identity.get("endpoint") and not identity.get("endpoint_identity"):
        missing.append("endpoint_identity")
    if missing:
        raise ValueError(f"confirmatory LLM identity is incomplete for {cell}: {sorted(missing)}")
    tokenizer_revision = str(identity.get("tokenizer_revision") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision):
        raise ValueError(
            f"confirmatory LLM tokenizer revision is not an immutable 40-hex commit for "
            f"{cell}: {tokenizer_revision!r}"
        )
    mutable = {"latest", "main", "master", "head", "default", "stable"}
    resolved_revision = str(identity.get("resolved_revision")).strip().lower()
    if resolved_revision in mutable:
        raise ValueError(
            f"confirmatory LLM identity uses mutable resolved revision for {cell}: "
            f"{resolved_revision!r}"
        )
    try:
        datetime.fromisoformat(str(identity.get("request_time")).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"confirmatory LLM request_time is invalid for {cell}") from exc


def _validate_paired_llm_identities(
    manifests: Sequence[Mapping[str, Any]],
) -> None:
    """Reject resolved model drift across confirmed paired arms."""

    groups: dict[tuple[str, str, int, str], dict[str, set[str]]] = {}
    for manifest in manifests:
        if manifest.get("status") != "complete":
            continue
        fingerprint = manifest.get("fingerprint_payload")
        llm_required = (
            fingerprint.get("llm_required") is True if isinstance(fingerprint, Mapping) else False
        )
        usage = manifest.get("llm_usage")
        identities = usage.get("backend_identities") if isinstance(usage, Mapping) else None
        if not isinstance(identities, Mapping):
            if llm_required:
                raise ValueError(
                    "confirmatory LLM cell has no resolved backend identity: "
                    f"{manifest.get('experiment_id')}/{manifest.get('arm_id')}/"
                    f"{manifest.get('task_id')}/seed-{manifest.get('seed')}"
                )
            continue
        arm = str(manifest.get("arm_id"))
        observed = 0
        for task, raw_entries in identities.items():
            if not isinstance(raw_entries, list):
                continue
            for entry in raw_entries:
                if isinstance(entry, Mapping):
                    _validate_complete_llm_identity(
                        entry,
                        cell=(
                            f"{manifest.get('experiment_id')}/{arm}/"
                            f"{manifest.get('task_id')}/seed-{manifest.get('seed')}/{task}"
                        ),
                    )
                    observed += 1
            projected = [
                {
                    key: entry.get(key)
                    for key in sorted(_PAIRED_LLM_IDENTITY_KEYS)
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
        if llm_required and observed == 0:
            raise ValueError(
                "confirmatory LLM cell has no complete backend identity: "
                f"{manifest.get('experiment_id')}/{arm}/{manifest.get('task_id')}/"
                f"seed-{manifest.get('seed')}"
            )
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
                        "reference_completeness": cell.reference_completeness,
                        "baseline_manifest": _baseline_record(suite),
                        "specification": suite.specification,
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
    *,
    stage: str = "screen",
    observed_arms: Optional[Mapping[str, set[str]]] = None,
) -> dict[str, list[tuple[str, str, str, bool]]]:
    comparisons: dict[str, list[tuple[str, str, str, bool]]] = {}
    for source in suite.sources:
        config = source.config
        if config.implementation.status != "ready":
            continue
        declared: list[tuple[str, str, str, bool]] = []
        present = (observed_arms or {}).get(config.experiment_id, set())
        for decision in config.selection.decisions:
            declared.extend(
                (decision.id, decision.baseline, candidate, True)
                for candidate in decision.candidates
                if stage != "confirm" or candidate in present
            )
            declared.extend(
                (
                    f"{decision.id}__control__{control}",
                    decision.baseline,
                    control,
                    False,
                )
                for control in decision.required_controls
                if stage != "confirm" or control in present
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
    return {
        (task.id, seed)
        for task in stage_config.tasks
        if task.availability.status == "ready"
        for seed in stage_config.seeds
    }


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
    if experiment_id == "E20":
        missing = [
            f"{arm}/{task}/seed-{seed}"
            for arm, cells in sorted(cells_by_arm.items())
            for (task, seed), record in sorted(cells.items())
            if not record.get("candidate_pool_fingerprint")
        ]
        if missing:
            return (
                "missing_retrieval_treatment_fingerprint",
                fingerprints,
                f"E20 retrieval treatments have missing fingerprints: {missing}",
            )
        return "allowed_retrieval_treatment", fingerprints, None

    arms = set(cells_by_arm)
    e17_retrieval_treatment = experiment_id == "E17" and (
        arms == {"rolling", "stack_all"}
        or "stack_minus_retrieval" in arms
        or any(arm.startswith("interaction_retrieval_") for arm in arms)
    )
    if e17_retrieval_treatment:
        missing: list[str] = []
        unbound_changes: list[str] = []
        for task in tasks:
            seeds = sorted(
                {
                    seed
                    for cells in cells_by_arm.values()
                    for cell_task, seed in cells
                    if cell_task == task
                }
            )
            for seed in seeds:
                records = {arm: cells[(task, seed)] for arm, cells in cells_by_arm.items()}
                values = {
                    arm: record.get("candidate_pool_fingerprint") for arm, record in records.items()
                }
                if any(not value for value in values.values()):
                    missing.append(f"{task}/seed-{seed}")
                    continue
                if len(set(values.values())) > 1:
                    design_hashes = {
                        record.get("candidate_pool_design_hash") for record in records.values()
                    }
                    if None in design_hashes or "" in design_hashes or len(design_hashes) == 1:
                        unbound_changes.append(f"{task}/seed-{seed}")
        if missing:
            return (
                "missing_retrieval_treatment_fingerprint",
                fingerprints,
                f"E17 retrieval treatment cells have missing fingerprints: {missing}",
            )
        if unbound_changes:
            return (
                "unbound_retrieval_treatment_change",
                fingerprints,
                "E17 changed candidate pools without distinct candidate-design/refit provenance "
                f"for {unbound_changes}",
            )
        return "allowed_retrieval_treatment", fingerprints, None

    cell_keys = sorted({cell for cells in cells_by_arm.values() for cell in cells})
    for task, seed in cell_keys:
        values = {
            cells[(task, seed)].get("candidate_pool_fingerprint") for cells in cells_by_arm.values()
        }
        if None in values or "" in values:
            return (
                "missing_fixed_retrieval_fingerprint",
                fingerprints,
                f"fixed-retrieval cell {task!r}/seed-{seed} has a missing fingerprint",
            )
        if len(values) != 1:
            return (
                "candidate_pool_mismatch",
                fingerprints,
                f"fixed-retrieval cell {task!r}/seed-{seed} used unequal fingerprints",
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
    fatal_if_unavailable: bool,
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
                fatal=fatal_if_unavailable,
                unavailable_cells=sorted(unavailable),
            )
        ]
    if not requested:
        return [
            _unavailable_row(
                typed_base,
                reason_code="typed_slices_empty",
                reason="explicit enriched artifacts contain no within-kind relation slices",
                fatal=fatal_if_unavailable,
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
                fatal=fatal_if_unavailable,
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
                    fatal=fatal_if_unavailable,
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


def _declared_power_slice_rows(
    *,
    base: Mapping[str, Any],
    baseline_arm: str,
    candidate_arm: str,
    cells_by_arm: Mapping[str, Mapping[tuple[str, int], Mapping[str, Any]]],
    pool_status: str,
    fingerprints: Mapping[str, Any],
    resamples: int,
    seed: int,
    power_slices: Sequence[Any],
) -> list[dict[str, Any]]:
    """Report each frozen confirm task/kind/relation power declaration separately."""

    rows: list[dict[str, Any]] = []
    for declaration in power_slices:
        kind = str(declaration.entity_kind).strip().lower()
        relation = normalize_relation(str(declaration.relation))
        slice_name = f"{kind}|{relation}"
        slice_base = {
            **base,
            "comparison_id": (f"{base['comparison_id']}:{declaration.task}:{kind}:{relation}"),
            "task_id": declaration.task,
            "metric": "typed_kind_relation_F1",
            "endpoint_scope": "typed_kind_relation",
            "confirmatory": declaration.status == "powered",
            "entity_kind": kind,
            "relation": relation,
            "power_slice_id": declaration.id,
            "power_status": declaration.status,
            "hypothesized_effect": declaration.hypothesized_effect,
            "power_assumptions": list(declaration.assumptions),
            "inference_status": (
                "confirmatory"
                if declaration.status == "powered"
                else (
                    "inconclusive"
                    if declaration.status == "underpowered"
                    else (
                        "descriptive"
                        if declaration.status == "descriptive"
                        else "deferred_unavailable"
                    )
                )
            ),
        }
        if declaration.status == "deferred_unavailable":
            rows.append(
                _unavailable_row(
                    slice_base,
                    reason_code=str(declaration.reason_code),
                    reason="frozen task/kind/relation power slice is unavailable",
                    fatal=False,
                    missing_capability=declaration.missing_capability,
                    expected_dataset=declaration.expected_dataset,
                )
            )
            continue

        relevant = {
            arm: {cell: record for cell, record in cells.items() if cell[0] == declaration.task}
            for arm, cells in cells_by_arm.items()
        }
        capabilities: dict[str, Any] = {}
        for arm, cells in sorted(relevant.items()):
            for (task, cell_seed), record in sorted(cells.items()):
                slices, reason = _explicit_typed_slices(record)
                cell_name = f"{arm}/{task}/seed-{cell_seed}"
                if slices is None:
                    capabilities[cell_name] = {"available": False, "reason": reason}
                else:
                    capabilities[cell_name] = {
                        "available": (kind, relation) in slices,
                        "slices": [
                            f"{slice_kind}|{slice_relation}"
                            for slice_kind, slice_relation in sorted(slices)
                        ],
                    }
        unavailable = sorted(
            cell
            for cell, capability in capabilities.items()
            if capability.get("available") is not True
        )
        missing_arms = sorted(arm for arm in (baseline_arm, candidate_arm) if not relevant.get(arm))
        if missing_arms or unavailable:
            rows.append(
                _unavailable_row(
                    {**slice_base, "typed_slice_capabilities": capabilities},
                    reason_code="typed_artifact_unavailable",
                    reason=(
                        "the frozen task/kind/relation slice is absent from one or more "
                        "explicit enriched artifacts"
                    ),
                    fatal=True,
                    missing_arms=missing_arms,
                    unavailable_cells=unavailable,
                )
            )
            continue

        try:
            recomputed: dict[str, RecomputedEvaluation] = {}
            for cells in relevant.values():
                for record in cells.values():
                    output_dir = str(record.get("output_dir") or "")
                    if not output_dir:
                        raise ValueError("complete cell is missing output_dir")
                    if output_dir not in recomputed:
                        recomputed[output_dir] = recompute_global_prf(
                            Path(output_dir), slices=[(kind, relation)]
                        )
            baseline = {
                cell: recomputed[str(record["output_dir"])].slices[slice_name]
                for cell, record in relevant[baseline_arm].items()
            }
            candidate = {
                cell: recomputed[str(record["output_dir"])].slices[slice_name]
                for cell, record in relevant[candidate_arm].items()
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
                    fatal=True,
                )
            )
            continue
        rows.append(
            {
                **slice_base,
                "status": "complete",
                "fatal": False,
                "raw_confidence_interval": "paired_95_percent_unadjusted",
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
    observed_arms: dict[str, set[str]] = {}
    for record in records:
        observed_arms.setdefault(str(record.get("experiment_id")), set()).add(
            str(record.get("arm_id"))
        )
    declarations = _declared_bootstrap_comparisons(
        suite,
        stage=stage,
        observed_arms=observed_arms,
    )
    for experiment_id, comparisons in sorted(declarations.items()):
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
            base["inference_status"] = (
                "descriptive"
                if stage == "screen" or config.design.power_status == "descriptive"
                else (
                    "inconclusive"
                    if config.design.power_status == "underpowered"
                    else "confirmatory"
                )
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
                        fatal=stage == "confirm",
                        **cell_details,
                    )
                )
                continue
            completeness = sorted(
                {
                    str(record.get("reference_completeness") or "unknown")
                    for cells in cells_by_arm.values()
                    for record in cells.values()
                }
            )
            base["reference_completeness"] = completeness
            if completeness == ["complete"]:
                base["precision_claim_status"] = "eligible"
            else:
                base["precision_claim_status"] = "deferred_pending_adjudication"
                base["inference_status"] = "descriptive_reference_incomplete"
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
                if typed_required and stage == "confirm":
                    rows.extend(
                        _declared_power_slice_rows(
                            base=base,
                            baseline_arm=baseline_arm,
                            candidate_arm=candidate_arm,
                            cells_by_arm=cells_by_arm,
                            pool_status=pool_status,
                            fingerprints=fingerprints,
                            resamples=resamples,
                            seed=seed,
                            power_slices=config.design.power_slices,
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
            if typed_required and stage == "confirm":
                rows.extend(
                    _declared_power_slice_rows(
                        base=base,
                        baseline_arm=baseline_arm,
                        candidate_arm=candidate_arm,
                        cells_by_arm=cells_by_arm,
                        pool_status=pool_status,
                        fingerprints=fingerprints,
                        resamples=resamples,
                        seed=seed,
                        power_slices=config.design.power_slices,
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
        if row.get("status") == "complete":
            row.setdefault("raw_confidence_interval", "paired_95_percent_unadjusted")
        if row.get("status") == "complete" and policy == "none":
            row["p_value_adjusted"] = row.get("p_value")
            row["p_value_adjustment"] = "none"
    if stage != "confirm":
        return
    for experiment_id, policy in sorted(policies.items()):
        declared_family = sorted(
            {
                str(row["comparison_id"])
                for row in rows
                if row.get("experiment_id") == experiment_id
                and row.get("confirmatory") is True
                and row.get("endpoint_scope") == "overall"
            }
        )
        family_id = f"{experiment_id}:confirm:primary-v1"
        family_hash = hash_payload(declared_family)
        for row in rows:
            if (
                row.get("experiment_id") == experiment_id
                and row.get("confirmatory") is True
                and row.get("endpoint_scope") == "overall"
            ):
                row["multiplicity_family"] = family_id
                row["multiplicity_family_hash"] = family_hash
                row["multiplicity_family_members"] = declared_family
                row["multiplicity_family_size"] = len(declared_family)
        if policy != "holm":
            continue
        family = [
            row
            for row in rows
            if row.get("experiment_id") == experiment_id
            and row.get("confirmatory") is True
            and row.get("endpoint_scope") == "overall"
        ]
        incomplete = [row for row in family if row.get("status") != "complete"]
        if incomplete:
            for row in family:
                row["p_value_adjustment"] = "holm_unavailable_family_incomplete"
            continue
        raw = {str(row["comparison_id"]): float(row["p_value"]) for row in family}
        adjusted = holm_adjust_p_values(raw)
        order = sorted(raw, key=lambda comparison_id: (raw[comparison_id], comparison_id))
        rank = {comparison_id: index + 1 for index, comparison_id in enumerate(order)}
        for row in family:
            comparison_id = str(row["comparison_id"])
            row["p_value_adjusted"] = adjusted[comparison_id]
            row["p_value_adjustment"] = "holm"
            row["holm_adjustment_order"] = order
            row["holm_rank"] = rank[comparison_id]


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
    artifact_root = _stage_artifact_root(
        suite,
        stage=stage,
        output_root=output_root,
    )
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
            "reference_completeness": manifest.get("reference_completeness"),
            "config_hash": manifest.get("resolved_config_hash"),
            "candidate_pool_fingerprint": manifest.get("candidate_pool_fingerprint"),
            "candidate_pool_design_hash": (manifest.get("fingerprint_payload") or {}).get(
                "candidate_pool_design_hash"
            ),
            "fitted_artifacts": (manifest.get("fingerprint_payload") or {}).get("artifacts"),
            "llm_usage": manifest.get("llm_usage"),
            "llm_required": (manifest.get("fingerprint_payload") or {}).get("llm_required"),
            "candidate_recall": manifest.get("candidate_recall"),
            "candidate_recall_after_exact": manifest.get("candidate_recall_after_exact"),
            "candidate_recall_diagnostics": manifest.get("candidate_recall_diagnostics"),
            "mean_pool_size": manifest.get("mean_pool_size"),
            "gold_rank_p90": manifest.get("gold_rank_p90"),
            "gold_rank_median": manifest.get("gold_rank_median"),
            "coverage": manifest.get("coverage"),
            "abstention_rate": manifest.get("abstention_rate"),
            "metric_applicability": manifest.get("metric_applicability"),
            "explanation_reconstruction": manifest.get("explanation_reconstruction"),
            "selection_evidence": manifest.get("selection_evidence"),
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
    artifact_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(artifact_root / "metrics.json", {"schema_version": 1, "rows": records})
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
        "reference_completeness",
        "config_hash",
        "candidate_pool_fingerprint",
        "candidate_pool_design_hash",
        "fitted_artifacts",
        "llm_usage",
        "llm_required",
        "candidate_recall",
        "candidate_recall_after_exact",
        "candidate_recall_diagnostics",
        "mean_pool_size",
        "gold_rank_p90",
        "gold_rank_median",
        "coverage",
        "abstention_rate",
        "metric_applicability",
        "explanation_reconstruction",
        "selection_evidence",
        "wall_seconds",
        "peak_memory_kb",
        "failure",
        "metric_error",
        *metric_keys,
    ]
    csv_path = artifact_root / "metrics.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            flattened = {key: record.get(key) for key in fieldnames}
            for structured_field in (
                "fitted_artifacts",
                "llm_usage",
                "metric_error",
                "candidate_recall_diagnostics",
                "metric_applicability",
                "explanation_reconstruction",
                "selection_evidence",
            ):
                if isinstance(flattened.get(structured_field), Mapping):
                    flattened[structured_field] = canonical_json(flattened[structured_field])
            flattened.update(record.get("metrics") or {})
            writer.writerow(flattened)
    os.replace(temporary, csv_path)
    _write_per_source_index(stage_root, records, report_root=artifact_root)
    long_rows, macro_rows = metric_reports(records)
    _atomic_json(
        artifact_root / "metrics_long.json",
        {"schema_version": 1, "rows": long_rows},
    )
    _atomic_csv_rows(artifact_root / "metrics_long.csv", long_rows)
    _atomic_json(
        artifact_root / "metrics_macro.json",
        {"schema_version": 1, "rows": macro_rows},
    )
    _atomic_csv_rows(artifact_root / "metrics_macro.csv", macro_rows)

    inventory_path = artifact_root / "dataset_inventory.json"
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
            stage_root=artifact_root,
            records=records,
        )
    return records


def _write_per_source_index(
    stage_root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    report_root: Optional[Path] = None,
) -> None:
    destination = report_root or stage_root
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
            run_dir / "alignment" / "paper.maps_global.tsv",
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
        destination / "per_source_outputs.json",
        {"schema_version": 1, "runs": entries},
    )


_SELECTION_ENDPOINT_PATHS = {
    "llm.calls": ("llm_usage", "calls"),
    "llm.input_tokens": ("llm_usage", "input_tokens"),
    "llm.output_tokens": ("llm_usage", "output_tokens"),
    "llm.total_tokens": ("llm_usage", "total_tokens"),
}


def _selection_metric_value(record: Mapping[str, Any], requested: str) -> Optional[float]:
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping):
        value = _metric_value(metrics, requested)
        if value is not None:
            return value
    if requested in record:
        value = record[requested]
    else:
        path = _SELECTION_ENDPOINT_PATHS.get(requested, tuple(requested.split(".")))
        value = record
        for part in path:
            if not isinstance(value, Mapping) or part not in value:
                return None
            value = value[part]
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"selection metric {requested!r} cannot be boolean")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"selection metric {requested!r} must be numeric") from exc


def _metric_cells_by_arm(
    records: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    metric: str,
    arms: Optional[set[str]] = None,
) -> dict[str, dict[tuple[str, int], float]]:
    values: dict[str, dict[tuple[str, int], float]] = {}
    failed: set[str] = set()
    missing: list[str] = []
    for record in records:
        if record.get("experiment_id") != experiment_id:
            continue
        arm = str(record.get("arm_id"))
        if arms is not None and arm not in arms:
            continue
        if record.get("status") != "complete":
            failed.add(arm)
            continue
        task = str(record.get("task_id"))
        try:
            seed = int(record.get("seed"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{experiment_id}: selection record for {arm}/{task} has an invalid seed"
            ) from exc
        value = _selection_metric_value(record, metric)
        if value is None:
            missing.append(f"{arm}/{task}/seed-{seed}")
            continue
        if not math.isfinite(value):
            raise ValueError(
                f"{experiment_id}: selection metric {metric!r} is non-finite for "
                f"{arm}/{task}/seed-{seed}"
            )
        key = (task, seed)
        arm_values = values.setdefault(arm, {})
        if key in arm_values:
            raise ValueError(
                f"{experiment_id}: duplicate selection cell for {arm}/{task}/seed-{seed}"
            )
        arm_values[key] = value
    if failed:
        raise ValueError(
            f"{experiment_id}: selection cannot ignore failed cells for arms {sorted(failed)}"
        )
    if missing:
        raise ValueError(
            f"{experiment_id}: complete cells are missing selection metric "
            f"{metric!r}: {sorted(missing)}"
        )
    return values


def _scores_by_arm(
    records: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    metric: str,
    arms: Optional[set[str]] = None,
) -> dict[str, float]:
    values = _metric_cells_by_arm(
        records,
        experiment_id=experiment_id,
        metric=metric,
        arms=arms,
    )
    if arms is not None:
        missing_arms = sorted(arms.difference(values))
        if missing_arms:
            raise ValueError(
                f"{experiment_id}: missing selection metric {metric!r} for arms {missing_arms}"
            )
        cell_sets = {arm: set(values[arm]) for arm in sorted(arms)}
        expected = next(iter(cell_sets.values()), set())
        unequal = {arm: sorted(cells) for arm, cells in cell_sets.items() if cells != expected}
        if unequal:
            raise ValueError(
                f"{experiment_id}: selection metric {metric!r} has unequal task/seed cells: "
                f"{unequal}"
            )
    return {arm: sum(entries.values()) / len(entries) for arm, entries in values.items() if entries}


def _selection_ci_evidence(
    records: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for record in records:
        if record.get("experiment_id") != experiment_id:
            continue
        document = record.get("selection_evidence")
        if document is None:
            continue
        if not isinstance(document, Mapping) or document.get("schema_version") != 1:
            raise ValueError(f"{experiment_id}: selection_evidence must use schema_version 1")
        comparisons = document.get("comparisons")
        if not isinstance(comparisons, list):
            raise ValueError(f"{experiment_id}: selection_evidence comparisons must be a list")
        for raw in comparisons:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{experiment_id}: selection evidence rows must be mappings")
            decision_id = str(raw.get("decision_id") or "").strip()
            baseline = str(raw.get("baseline") or "").strip()
            candidate = str(raw.get("candidate") or "").strip()
            metric = str(raw.get("metric") or "").strip()
            direction = str(raw.get("direction") or "").strip()
            if not all((decision_id, baseline, candidate, metric)) or direction not in {
                "max",
                "min",
            }:
                raise ValueError(f"{experiment_id}: incomplete selection evidence identity")
            try:
                estimate = float(raw.get("estimate"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{experiment_id}: selection evidence estimate is invalid"
                ) from exc
            interval = raw.get("confidence_interval")
            if not isinstance(interval, Mapping):
                raise ValueError(
                    f"{experiment_id}: selection evidence requires confidence_interval"
                )
            try:
                confidence = float(interval.get("confidence"))
                lower = float(interval.get("lower"))
                upper = float(interval.get("upper"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{experiment_id}: selection evidence confidence interval is invalid"
                ) from exc
            if not all(math.isfinite(value) for value in (estimate, confidence, lower, upper)):
                raise ValueError(f"{experiment_id}: selection evidence must be finite")
            if not math.isclose(confidence, 0.95, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"{experiment_id}: selection evidence must be a paired 95% interval"
                )
            if interval.get("paired") is not True:
                raise ValueError(f"{experiment_id}: selection evidence must be paired")
            method = str(interval.get("method") or "").strip()
            independent_unit = str(interval.get("independent_unit") or "").strip()
            if not method or not independent_unit:
                raise ValueError(
                    f"{experiment_id}: selection CI requires method and independent_unit"
                )
            if lower > estimate or estimate > upper:
                raise ValueError(
                    f"{experiment_id}: selection CI does not contain its point estimate"
                )
            normalized = {
                "decision_id": decision_id,
                "baseline": baseline,
                "candidate": candidate,
                "metric": metric,
                "direction": direction,
                "estimate": estimate,
                "confidence_interval": {
                    "confidence": confidence,
                    "lower": lower,
                    "upper": upper,
                    "paired": True,
                    "method": method,
                    "independent_unit": independent_unit,
                },
            }
            key = (decision_id, baseline, candidate, metric, direction)
            existing = index.get(key)
            if existing is not None and canonical_json(existing) != canonical_json(normalized):
                raise ValueError(f"{experiment_id}: conflicting selection CI evidence for {key}")
            index[key] = normalized
    return index


def _scope_groups(
    values: Mapping[tuple[str, int], float],
    scope: str,
) -> dict[str, list[float]]:
    if scope == "aggregate":
        return {"aggregate": list(values.values())}
    if scope == "each_task":
        grouped: dict[str, list[float]] = {}
        for (task, _seed), value in sorted(values.items()):
            grouped.setdefault(task, []).append(value)
        return grouped
    return {f"{task}/seed-{seed}": [value] for (task, seed), value in sorted(values.items())}


def _threshold_pass(value: float, threshold: float, *, strict: bool) -> bool:
    return value > threshold if strict else value >= threshold


def _baseline_delta_evaluation(
    *,
    experiment_id: str,
    decision_id: str,
    baseline: str,
    candidate: str,
    metric: str,
    direction: str,
    scope: str,
    evidence: str,
    min_delta: Optional[float],
    min_relative_delta: Optional[float],
    strict: bool,
    cells: Mapping[str, Mapping[tuple[str, int], float]],
    ci_index: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_cells = cells[baseline]
    candidate_cells = cells[candidate]
    if set(baseline_cells) != set(candidate_cells):
        raise ValueError(
            f"{experiment_id}/{decision_id}: guard metric {metric!r} has unequal cells for "
            f"{baseline!r} and {candidate!r}"
        )
    base_groups = _scope_groups(baseline_cells, scope)
    candidate_groups = _scope_groups(candidate_cells, scope)
    ci: Optional[Mapping[str, Any]] = None
    if evidence == "paired_ci_lower":
        key = (decision_id, baseline, candidate, metric, direction)
        ci = ci_index.get(key)
        if ci is None:
            raise ValueError(
                f"{experiment_id}/{decision_id}: missing paired 95% CI evidence for "
                f"{baseline}->{candidate} metric {metric!r}"
            )
    evaluations: list[dict[str, Any]] = []
    passed = True
    for scope_id in sorted(base_groups):
        baseline_score = sum(base_groups[scope_id]) / len(base_groups[scope_id])
        candidate_score = sum(candidate_groups[scope_id]) / len(candidate_groups[scope_id])
        signed_delta = (
            candidate_score - baseline_score
            if direction == "max"
            else baseline_score - candidate_score
        )
        tested_delta = float(ci["confidence_interval"]["lower"]) if ci is not None else signed_delta
        delta_passed = (
            True if min_delta is None else _threshold_pass(tested_delta, min_delta, strict=strict)
        )
        relative_delta: Optional[float] = None
        relative_passed = True
        if min_relative_delta is not None:
            if baseline_score == 0.0:
                raise ValueError(
                    f"{experiment_id}/{decision_id}: relative selection guard {metric!r} "
                    f"has a zero baseline for {scope_id}"
                )
            relative_delta = signed_delta / abs(baseline_score)
            relative_passed = _threshold_pass(
                relative_delta,
                min_relative_delta,
                strict=strict,
            )
        item_passed = delta_passed and relative_passed
        passed = passed and item_passed
        evaluations.append(
            {
                "scope_id": scope_id,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "signed_delta": signed_delta,
                "tested_delta": tested_delta,
                "relative_delta": relative_delta,
                "passed": item_passed,
            }
        )
    return {
        "comparison": "baseline_delta",
        "baseline": baseline,
        "candidate": candidate,
        "metric": metric,
        "direction": direction,
        "scope": scope,
        "evidence": evidence,
        "min_delta": min_delta,
        "min_relative_delta": min_relative_delta,
        "strict": strict,
        "confidence_interval": ci,
        "evaluations": evaluations,
        "passed": passed,
    }


def _guard_evaluation(
    *,
    experiment_id: str,
    decision: SelectionDecisionConfig,
    guard: SelectionGuardConfig,
    candidate: str,
    records: Sequence[Mapping[str, Any]],
    ci_index: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = guard.baseline or decision.baseline
    required = {candidate} if guard.comparison == "absolute_threshold" else {baseline, candidate}
    cells = _metric_cells_by_arm(
        records,
        experiment_id=experiment_id,
        metric=guard.metric,
        arms=required,
    )
    missing = sorted(required.difference(cells))
    if missing:
        raise ValueError(
            f"{experiment_id}/{decision.id}: missing guard metric {guard.metric!r} "
            f"for arms {missing}"
        )
    if guard.comparison == "baseline_delta":
        result = _baseline_delta_evaluation(
            experiment_id=experiment_id,
            decision_id=decision.id,
            baseline=baseline,
            candidate=candidate,
            metric=guard.metric,
            direction=guard.direction,
            scope=guard.scope,
            evidence=guard.evidence,
            min_delta=guard.min_delta,
            min_relative_delta=guard.min_relative_delta,
            strict=guard.strict,
            cells=cells,
            ci_index=ci_index,
        )
    elif guard.comparison == "absolute_threshold":
        groups = _scope_groups(cells[candidate], guard.scope)
        evaluations: list[dict[str, Any]] = []
        passed = True
        assert guard.threshold is not None
        for scope_id in sorted(groups):
            value = sum(groups[scope_id]) / len(groups[scope_id])
            tested = value if guard.direction == "max" else -value
            threshold = guard.threshold if guard.direction == "max" else -guard.threshold
            item_passed = _threshold_pass(tested, threshold, strict=guard.strict)
            passed = passed and item_passed
            evaluations.append(
                {"scope_id": scope_id, "candidate_score": value, "passed": item_passed}
            )
        result = {
            "comparison": guard.comparison,
            "candidate": candidate,
            "metric": guard.metric,
            "direction": guard.direction,
            "scope": guard.scope,
            "threshold": guard.threshold,
            "strict": guard.strict,
            "evaluations": evaluations,
            "passed": passed,
        }
    else:
        baseline_cells = cells[baseline]
        candidate_cells = cells[candidate]
        if set(baseline_cells) != set(candidate_cells):
            raise ValueError(
                f"{experiment_id}/{decision.id}: matched guard {guard.metric!r} has unequal cells"
            )
        base_groups = _scope_groups(baseline_cells, guard.scope)
        candidate_groups = _scope_groups(candidate_cells, guard.scope)
        assert guard.match_tolerance is not None
        evaluations = []
        passed = True
        for scope_id in sorted(base_groups):
            baseline_score = sum(base_groups[scope_id]) / len(base_groups[scope_id])
            candidate_score = sum(candidate_groups[scope_id]) / len(candidate_groups[scope_id])
            absolute_tolerance = guard.match_tolerance.absolute
            relative_tolerance = guard.match_tolerance.relative * abs(baseline_score)
            tolerance = (
                max(absolute_tolerance, relative_tolerance)
                if guard.match_tolerance.combine == "max"
                else min(absolute_tolerance, relative_tolerance)
            )
            difference = abs(candidate_score - baseline_score)
            item_passed = difference <= tolerance
            passed = passed and item_passed
            evaluations.append(
                {
                    "scope_id": scope_id,
                    "baseline_score": baseline_score,
                    "candidate_score": candidate_score,
                    "absolute_difference": difference,
                    "allowed_difference": tolerance,
                    "passed": item_passed,
                }
            )
        result = {
            "comparison": guard.comparison,
            "baseline": baseline,
            "candidate": candidate,
            "metric": guard.metric,
            "scope": guard.scope,
            "match_tolerance": guard.match_tolerance.model_dump(mode="json"),
            "evaluations": evaluations,
            "passed": passed,
        }
    return {"id": guard.id, **result}


def _mapping_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"resolved arm overlay is missing numeric path {path!r}")
        current = current[part]
    return current


def _tie_break_scores(
    tie_break: SelectionTieBreakConfig,
    *,
    experiment_id: str,
    candidates: Sequence[str],
    baseline: str,
    records: Sequence[Mapping[str, Any]],
    arms: Mapping[str, ArmConfig],
    base: Mapping[str, Any],
) -> dict[str, float]:
    candidate_set = set(candidates)
    if tie_break.kind == "metric":
        assert tie_break.metric is not None
        return _scores_by_arm(
            records,
            experiment_id=experiment_id,
            metric=tie_break.metric,
            arms=candidate_set,
        )
    if tie_break.kind == "arm_order":
        return {arm_id: float(index) for index, arm_id in enumerate(tie_break.order)}
    baseline_mapping = deep_merge(base, arms[baseline].overlay)
    scores: dict[str, float] = {}
    for arm_id in candidates:
        candidate_mapping = deep_merge(base, arms[arm_id].overlay)
        distance = 0.0
        for path in tie_break.paths:
            try:
                baseline_value = float(_mapping_path(baseline_mapping, path))
                candidate_value = float(_mapping_path(candidate_mapping, path))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{experiment_id}: numeric overlay tie-break path {path!r} is not numeric "
                    f"for {baseline!r} and {arm_id!r}"
                ) from exc
            if not math.isfinite(baseline_value) or not math.isfinite(candidate_value):
                raise ValueError(
                    f"{experiment_id}: numeric overlay tie-break path {path!r} is non-finite"
                )
            distance += abs(candidate_value - baseline_value)
        scores[arm_id] = distance
    return scores


def _within_tolerance(value: float, best: float, tolerance: float) -> bool:
    difference = best - value
    if tolerance == 0.0:
        return difference == 0.0
    return difference < tolerance


def _lexicographic_candidate_ranking(
    eligible: Sequence[str],
    *,
    primary_scores: Mapping[str, float],
    primary_direction: str,
    primary_tolerance: float,
    tie_breaks: Sequence[SelectionTieBreakConfig],
    tie_scores: Sequence[Mapping[str, float]],
) -> list[str]:
    remaining = set(eligible)
    ranking: list[str] = []
    while remaining:
        normalized_primary = {
            arm: primary_scores[arm] if primary_direction == "max" else -primary_scores[arm]
            for arm in remaining
        }
        best = max(normalized_primary.values())
        finalists = {
            arm
            for arm, value in normalized_primary.items()
            if _within_tolerance(value, best, primary_tolerance)
        }
        for tie_break, scores in zip(tie_breaks, tie_scores):
            if len(finalists) <= 1:
                break
            normalized = {
                arm: scores[arm] if tie_break.direction == "max" else -scores[arm]
                for arm in finalists
            }
            tie_best = max(normalized.values())
            finalists = {
                arm
                for arm, value in normalized.items()
                if _within_tolerance(value, tie_best, tie_break.tolerance)
            }
        winner = min(finalists)
        ranking.append(winner)
        remaining.remove(winner)
    return ranking


def _mapping_delta(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, candidate_value in candidate.items():
        baseline_value = baseline.get(key)
        if isinstance(candidate_value, Mapping) and isinstance(baseline_value, Mapping):
            nested = _mapping_delta(baseline_value, candidate_value)
            if nested:
                delta[str(key)] = nested
        elif candidate_value != baseline_value:
            delta[str(key)] = candidate_value
    return delta


def _strict_overlay_merge(
    target: dict[str, Any],
    addition: Mapping[str, Any],
    *,
    path: tuple[str, ...] = (),
) -> dict[str, Any]:
    merged = dict(target)
    for raw_key, value in addition.items():
        key = str(raw_key)
        current_path = (*path, key)
        if key not in merged:
            merged[key] = value
            continue
        existing = merged[key]
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _strict_overlay_merge(dict(existing), value, path=current_path)
        elif existing != value:
            raise ValueError(
                "independent selection decisions produced conflicting overlays at "
                f"{'.'.join(current_path)}: {existing!r} != {value!r}"
            )
    return merged


def _e17_component_removal_guard(
    source: ExperimentSource,
    records: Sequence[Mapping[str, Any]],
    *,
    promoted_component_overlays: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    """Refuse to freeze an E17 stack whose development leaveout requires removal."""

    config = source.config
    if config.experiment_id != "E17":
        return None
    constants = config.frozen_constants.get("component_removal")
    if not isinstance(constants, Mapping):
        raise ValueError("E17 requires frozen_constants.component_removal")
    required_constants = {
        "macro_f1_improvement",
        "local_mrr_improvement",
        "cost_reduction_fraction",
        "cost_non_inferiority_margin",
        "cost_metric",
        "hard_guard_repairs_trigger_removal",
        "reconstruction_tolerance",
        "decision_data",
    }
    missing_constants = sorted(required_constants.difference(constants))
    if missing_constants:
        raise ValueError("E17 component-removal declaration is incomplete: " f"{missing_constants}")
    if constants.get("decision_data") != "development_only":
        raise ValueError("E17 component removal must be bound to development_only data")
    cost_metric = str(constants.get("cost_metric"))
    if cost_metric != "wall_seconds":
        raise ValueError("E17 component-removal cost_metric must be wall_seconds")

    numeric_constants: dict[str, float] = {}
    for name in (
        "macro_f1_improvement",
        "local_mrr_improvement",
        "cost_reduction_fraction",
        "cost_non_inferiority_margin",
        "reconstruction_tolerance",
    ):
        try:
            value = float(constants[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"E17 component-removal constant {name!r} is not numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"E17 component-removal constant {name!r} is non-finite")
        numeric_constants[name] = value
    if numeric_constants["macro_f1_improvement"] < 0.0:
        raise ValueError("E17 macro_f1_improvement must be non-negative")
    if numeric_constants["local_mrr_improvement"] < 0.0:
        raise ValueError("E17 local_mrr_improvement must be non-negative")
    if not 0.0 <= numeric_constants["cost_reduction_fraction"] <= 1.0:
        raise ValueError("E17 cost_reduction_fraction must be between zero and one")
    if numeric_constants["reconstruction_tolerance"] < 0.0:
        raise ValueError("E17 reconstruction_tolerance must be non-negative")
    if constants.get("hard_guard_repairs_trigger_removal") is not True:
        raise ValueError("E17 must treat a repaired hard guard as a component-removal trigger")

    e17_records = [
        record for record in records if record.get("experiment_id") == config.experiment_id
    ]
    nondevelopment = sorted(
        {str(record.get("stage")) for record in e17_records if record.get("stage") != "screen"}
    )
    if nondevelopment:
        raise ValueError(
            "E17 component removal may use only screen/development records; observed stages "
            f"{nondevelopment}"
        )
    resolved_arms = _component_arms(
        config,
        promoted_overlays=promoted_component_overlays,
    )
    leaveout_arms = sorted(arm.id for arm in resolved_arms if arm.id.startswith("stack_minus_"))
    required_arms = {"stack_all", *leaveout_arms}
    relevant = [record for record in e17_records if record.get("arm_id") in required_arms]
    observed_arms = {str(record.get("arm_id")) for record in relevant}
    missing_arms = sorted(required_arms.difference(observed_arms))
    if missing_arms:
        raise ValueError(f"E17 component-removal audit is missing arms {missing_arms}")

    f1_scores = _scores_by_arm(
        relevant,
        experiment_id="E17",
        metric="F1",
        arms=required_arms,
    )
    mrr_scores = _scores_by_arm(
        relevant,
        experiment_id="E17",
        metric="local.MRR",
        arms=required_arms,
    )
    costs: dict[str, float] = {}
    reconstruction: dict[str, dict[str, Any]] = {}
    tolerance = numeric_constants["reconstruction_tolerance"]
    for arm in sorted(required_arms):
        arm_records = [record for record in relevant if record.get("arm_id") == arm]
        arm_costs: list[float] = []
        failures: list[dict[str, Any]] = []
        for record in arm_records:
            try:
                cost = float(record.get(cost_metric))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"E17 component-removal arm {arm!r} has no numeric {cost_metric}"
                ) from exc
            if not math.isfinite(cost) or cost < 0.0:
                raise ValueError(
                    f"E17 component-removal arm {arm!r} has invalid {cost_metric}: {cost}"
                )
            arm_costs.append(cost)
            audit = record.get("explanation_reconstruction")
            passed = False
            if isinstance(audit, Mapping):
                try:
                    result_rows = int(audit.get("result_rows"))
                    reconstructed_rows = int(audit.get("reconstructed_rows"))
                    max_abs_error = float(audit.get("max_abs_error"))
                except (TypeError, ValueError):
                    pass
                else:
                    passed = (
                        audit.get("status") == "complete"
                        and result_rows > 0
                        and reconstructed_rows == result_rows
                        and math.isfinite(max_abs_error)
                        and max_abs_error <= tolerance
                    )
            if not passed:
                failures.append(
                    {
                        "task_id": record.get("task_id"),
                        "seed": record.get("seed"),
                        "audit": audit,
                    }
                )
        costs[arm] = sum(arm_costs) / len(arm_costs)
        reconstruction[arm] = {
            "passed": not failures,
            "failed_cells": failures,
        }

    stack_cost = costs["stack_all"]
    if stack_cost <= 0.0:
        raise ValueError("E17 stack_all wall_seconds must be positive for the cost-removal guard")
    stack_f1 = f1_scores["stack_all"]
    stack_mrr = mrr_scores["stack_all"]
    triggers: dict[str, list[dict[str, Any]]] = {}
    for arm in leaveout_arms:
        reasons: list[dict[str, Any]] = []
        f1_delta = f1_scores[arm] - stack_f1
        mrr_delta = mrr_scores[arm] - stack_mrr
        cost_reduction = 1.0 - (costs[arm] / stack_cost)
        if reconstruction[arm]["passed"]:
            if f1_delta > numeric_constants["macro_f1_improvement"]:
                reasons.append({"reason": "macro_f1_improvement", "delta": f1_delta})
            if mrr_delta > numeric_constants["local_mrr_improvement"]:
                reasons.append({"reason": "local_mrr_improvement", "delta": mrr_delta})
            if (
                costs[arm] <= stack_cost * (1.0 - numeric_constants["cost_reduction_fraction"])
                and f1_delta >= numeric_constants["cost_non_inferiority_margin"]
            ):
                reasons.append(
                    {
                        "reason": "cost_reduction_with_quality_non_inferiority",
                        "cost_reduction_fraction": cost_reduction,
                        "macro_f1_delta": f1_delta,
                    }
                )
        if not reconstruction["stack_all"]["passed"] and reconstruction[arm]["passed"]:
            reasons.append({"reason": "repairs_explanation_reconstruction_hard_guard"})
        if reasons:
            triggers[arm] = reasons

    failed_reconstruction = sorted(
        arm for arm, audit in reconstruction.items() if audit["passed"] is not True
    )
    audit_record = {
        "decision_data": "development_only",
        "cost_metric": cost_metric,
        "thresholds": numeric_constants,
        "macro_f1": f1_scores,
        "local_mrr": mrr_scores,
        "cost": costs,
        "explanation_reconstruction": reconstruction,
        "removal_triggers": triggers,
    }
    if triggers:
        raise ValueError(
            "E17 component removal is required before freeze; record the reason, version the "
            "composition, and rerun screen: " + canonical_json(audit_record)
        )
    if failed_reconstruction:
        raise ValueError(
            "E17 explanation reconstruction hard guard failed for arms "
            f"{failed_reconstruction}: {canonical_json(audit_record)}"
        )
    return audit_record


def select_experiment(
    source: ExperimentSource,
    records: Sequence[Mapping[str, Any]],
    *,
    promoted_component_overlays: Optional[Mapping[str, Mapping[str, Any]]] = None,
    baseline_manifest_hash: Optional[str] = None,
) -> dict[str, Any]:
    config = source.config
    component_removal_audit = _e17_component_removal_guard(
        source,
        records,
        promoted_component_overlays=promoted_component_overlays,
    )
    resolved_arms = _component_arms(
        config,
        promoted_overlays=promoted_component_overlays,
    )
    arms = {arm.id: arm for arm in resolved_arms}
    base = ConfigModel.from_mapping(_base_mapping(source), warn_v1=False).model_dump(
        mode="json", by_alias=True
    )
    ci_index = _selection_ci_evidence(records, experiment_id=config.experiment_id)
    expected_ci: set[tuple[str, str, str, str, str]] = set()
    for rule in config.selection.decisions:
        if rule.evidence == "paired_ci_lower":
            expected_ci.update(
                (rule.id, rule.baseline, candidate, rule.metric, rule.direction)
                for candidate in rule.candidates
            )
        for guard in rule.guards:
            if guard.evidence != "paired_ci_lower":
                continue
            expected_ci.update(
                (
                    rule.id,
                    guard.baseline or rule.baseline,
                    candidate,
                    guard.metric,
                    guard.direction,
                )
                for candidate in rule.candidates
            )
    undeclared_ci = sorted(set(ci_index).difference(expected_ci))
    if undeclared_ci:
        raise ValueError(
            f"{config.experiment_id}: selection evidence contains undeclared comparisons "
            f"{undeclared_ci}"
        )
    decisions: list[dict[str, Any]] = []
    combined_selected_overlay: dict[str, Any] = {}
    any_selected = False
    all_selected = True
    for rule in config.selection.decisions:
        required = {rule.baseline, *rule.candidates}
        scores = _scores_by_arm(
            records,
            experiment_id=config.experiment_id,
            metric=rule.metric,
            arms=required,
        )
        primary_cells = _metric_cells_by_arm(
            records,
            experiment_id=config.experiment_id,
            metric=rule.metric,
            arms=required,
        )
        baseline_score = scores[rule.baseline]
        candidate_evaluations: dict[str, dict[str, Any]] = {}
        eligible: list[str] = []
        for arm_id in rule.candidates:
            primary = _baseline_delta_evaluation(
                experiment_id=config.experiment_id,
                decision_id=rule.id,
                baseline=rule.baseline,
                candidate=arm_id,
                metric=rule.metric,
                direction=rule.direction,
                scope="aggregate",
                evidence=rule.evidence,
                min_delta=rule.min_delta,
                min_relative_delta=rule.min_relative_delta,
                strict=rule.strict,
                cells=primary_cells,
                ci_index=ci_index,
            )
            guards = [
                _guard_evaluation(
                    experiment_id=config.experiment_id,
                    decision=rule,
                    guard=guard,
                    candidate=arm_id,
                    records=records,
                    ci_index=ci_index,
                )
                for guard in rule.guards
            ]
            rejection_reasons: list[str] = []
            if primary["passed"] is not True:
                rejection_reasons.append("primary_threshold_failed")
            rejection_reasons.extend(
                f"guard_failed:{guard['id']}" for guard in guards if guard["passed"] is not True
            )
            candidate_evaluations[arm_id] = {
                "primary": primary,
                "guards": guards,
                "eligible": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
            }
            if not rejection_reasons:
                eligible.append(arm_id)

        tie_scores = [
            _tie_break_scores(
                tie_break,
                experiment_id=config.experiment_id,
                candidates=eligible,
                baseline=rule.baseline,
                records=records,
                arms=arms,
                base=base,
            )
            for tie_break in rule.tie_breaks
        ]
        ranking = _lexicographic_candidate_ranking(
            eligible,
            primary_scores=scores,
            primary_direction=rule.direction,
            primary_tolerance=rule.tie_tolerance,
            tie_breaks=rule.tie_breaks,
            tie_scores=tie_scores,
        )
        selected_arm = ranking[0] if ranking else None
        if selected_arm is None and not rule.allow_screened_out:
            raise ValueError(
                f"{config.experiment_id}/{rule.id}: no candidate passed and screened_out is forbidden"
            )
        any_selected = any_selected or selected_arm is not None
        all_selected = all_selected and selected_arm is not None
        selected_overlay: Optional[dict[str, Any]] = None
        selected_signed_delta: Optional[float] = None
        if selected_arm is not None:
            baseline_mapping = deep_merge(base, arms[rule.baseline].overlay)
            candidate_mapping = deep_merge(base, arms[selected_arm].overlay)
            selected_overlay = _mapping_delta(baseline_mapping, candidate_mapping)
            combined_selected_overlay = _strict_overlay_merge(
                combined_selected_overlay,
                selected_overlay,
            )
            selected_signed_delta = float(
                candidate_evaluations[selected_arm]["primary"]["evaluations"][0]["signed_delta"]
            )
        decisions.append(
            {
                "id": rule.id,
                "status": "selected" if selected_arm is not None else "screened_out",
                "baseline": rule.baseline,
                "baseline_score": baseline_score,
                "metric": rule.metric,
                "direction": rule.direction,
                "min_delta": rule.min_delta,
                "min_relative_delta": rule.min_relative_delta,
                "evidence": rule.evidence,
                "strict": rule.strict,
                "candidate_scores": {arm_id: scores[arm_id] for arm_id in sorted(rule.candidates)},
                "selected_arm": selected_arm,
                "selected_score": scores[selected_arm] if selected_arm else None,
                "signed_delta": selected_signed_delta,
                "selected_overlay": selected_overlay,
                "candidate_evaluations": {
                    arm_id: candidate_evaluations[arm_id]
                    for arm_id in sorted(candidate_evaluations)
                },
                "ranking_evidence": {
                    "primary_tolerance": rule.tie_tolerance,
                    "tie_breaks": [
                        {
                            **tie_break.model_dump(mode="json"),
                            "scores": {
                                arm_id: scores_by_arm[arm_id] for arm_id in sorted(scores_by_arm)
                            },
                        }
                        for tie_break, scores_by_arm in zip(rule.tie_breaks, tie_scores)
                    ],
                    "final_tie_break": "arm_id_ascending",
                    "lexicographic_order": ranking,
                },
                "required_controls": list(rule.required_controls),
            }
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
        "status": "selected" if any_selected else "screened_out",
        "decision_mode": "independent",
        "all_decisions_selected": all_selected,
        "combined_selected_overlay": combined_selected_overlay if any_selected else None,
        "selection_evidence_hash": (
            hash_payload([ci_index[key] for key in sorted(ci_index)]) if ci_index else None
        ),
        "component_removal_audit": component_removal_audit,
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
        combined = record.get("combined_selected_overlay")
        if isinstance(combined, Mapping):
            selected[dependency] = dict(combined)
            continue
        overlay: dict[str, Any] = {}
        for decision in record.get("decisions") or []:
            if isinstance(decision, Mapping) and isinstance(
                decision.get("selected_overlay"), Mapping
            ):
                overlay = deep_merge(overlay, decision["selected_overlay"])
        selected[dependency] = overlay
    return selected


def _overlay_artifact_paths(value: Any, *, record_dir: Path) -> set[Path]:
    paths: set[Path] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if "artifact" in str(key).lower() and isinstance(item, (str, os.PathLike)) and item:
                candidate = Path(str(item)).expanduser()
                paths.add(
                    candidate.resolve()
                    if candidate.is_absolute()
                    else (record_dir / candidate).resolve()
                )
            else:
                paths.update(_overlay_artifact_paths(item, record_dir=record_dir))
    elif isinstance(value, list):
        for item in value:
            paths.update(_overlay_artifact_paths(item, record_dir=record_dir))
    return paths


def _evidence_file(
    raw: Mapping[str, Any],
    field: str,
    *,
    record_path: Path,
    component_id: str,
) -> Path:
    value = str(raw.get(field) or "").strip()
    if not value:
        raise ValueError(f"E17 component {component_id!r} has no {field}")
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (record_path.parent / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"E17 component {component_id!r} {field} does not exist: {resolved}"
        )
    return resolved


def _validate_component_confirmation_evidence(
    *,
    source_id: str,
    source: ExperimentSource,
    raw: Mapping[str, Any],
    overlay: Mapping[str, Any],
    pool_contract: Mapping[str, Any],
    pool_design_hash: str,
    record_path: Path,
    suite: LoadedSuite,
) -> None:
    """Bind a claimed E17 component to real screen and confirm artifacts."""

    selection_path = _evidence_file(
        raw,
        "selection_record",
        record_path=record_path,
        component_id=source_id,
    )
    selection = _load_json(selection_path)
    embedded_selection_hash = selection.pop("selection_hash", None)
    computed_selection_hash = hash_payload(selection)
    selection["selection_hash"] = embedded_selection_hash
    if (
        not embedded_selection_hash
        or embedded_selection_hash != computed_selection_hash
        or raw.get("selection_hash") != embedded_selection_hash
    ):
        raise ValueError(f"E17 component {source_id!r} selection record/hash is not authentic")
    expected_selection_root = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "suite_hash": suite.suite_hash,
        "stage": "screen",
        "baseline_id": suite.baseline_id,
        "baseline_manifest_hash": suite.baseline_manifest_hash,
        "dataset_lock_hash": suite.dataset_lock_hash,
        "model_lock_hash": suite.model_lock_hash,
        "specification": suite.specification,
    }
    root_mismatches = {
        key: (expected, selection.get(key))
        for key, expected in expected_selection_root.items()
        if selection.get(key) != expected
    }
    if root_mismatches:
        raise ValueError(
            f"E17 component {source_id!r} selection provenance mismatch: " f"{root_mismatches}"
        )
    frozen = (selection.get("experiments") or {}).get(source_id)
    if not isinstance(frozen, Mapping) or frozen.get("status") != "selected":
        raise ValueError(f"E17 component {source_id!r} has no selected screen outcome")
    expected_design_hash = experiment_design_hash(
        source,
        baseline_manifest_hash=suite.baseline_manifest_hash,
    )
    if frozen.get("experiment_config_hash") != source.raw_hash():
        raise ValueError(f"E17 component {source_id!r} screen config hash mismatch")
    if frozen.get("design_hash") != expected_design_hash:
        raise ValueError(f"E17 component {source_id!r} screen design hash mismatch")
    frozen_overlay = frozen.get("combined_selected_overlay")
    if not isinstance(frozen_overlay, Mapping) or _jsonable(frozen_overlay) != _jsonable(overlay):
        raise ValueError(f"E17 component {source_id!r} overlay differs from its frozen selection")
    selected_candidates = {
        str(decision.get("selected_arm"))
        for decision in (frozen.get("decisions") or [])
        if isinstance(decision, Mapping) and decision.get("selected_arm")
    }
    selected_arm = str(raw.get("selected_arm") or "")
    if selected_arm not in selected_candidates:
        raise ValueError(f"E17 component {source_id!r} selected_arm is not a frozen winner")

    confirmation_path = _evidence_file(
        raw,
        "confirmation_manifest",
        record_path=record_path,
        component_id=source_id,
    )
    observed_confirmation_hash = sha256_file(confirmation_path)
    if raw.get("confirmation_manifest_hash") != observed_confirmation_hash:
        raise ValueError(f"E17 component {source_id!r} confirmation manifest hash mismatch")
    confirmation = _load_json(confirmation_path)
    rows = confirmation.get("rows")
    if confirmation.get("schema_version") != 1 or not isinstance(rows, list):
        raise ValueError(
            f"E17 component {source_id!r} confirmation manifest must be metrics schema 1"
        )

    resolved_arms = _component_arms(source.config)
    expected_arms = _selected_arm_ids(
        source,
        stage="confirm",
        selection_record=selection,
        arms=resolved_arms,
    )
    ready_tasks = [
        task for task in source.config.confirm.tasks if task.availability.status == "ready"
    ]
    expected_cells = {
        (arm_id, task.id, int(seed))
        for arm_id in expected_arms
        for task in ready_tasks
        for seed in source.config.confirm.seeds
    }
    source_rows = [
        row for row in rows if isinstance(row, Mapping) and row.get("experiment_id") == source_id
    ]
    observed_cells: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in source_rows:
        try:
            cell = (str(row.get("arm_id")), str(row.get("task_id")), int(row.get("seed")))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"E17 component {source_id!r} confirmation has an invalid cell identity"
            ) from exc
        if cell in observed_cells:
            raise ValueError(f"E17 component {source_id!r} confirmation duplicates cell {cell}")
        observed_cells[cell] = row
    if set(observed_cells) != expected_cells:
        raise ValueError(
            f"E17 component {source_id!r} confirmation cells differ from the frozen matrix: "
            f"missing={sorted(expected_cells.difference(observed_cells))}, "
            f"extra={sorted(set(observed_cells).difference(expected_cells))}"
        )

    for cell, row in sorted(observed_cells.items()):
        if row.get("status") != "complete" or row.get("metric_error") not in (None, {}):
            raise ValueError(
                f"E17 component {source_id!r} confirmation cell {cell} is not complete"
            )
        output_dir = Path(str(row.get("output_dir") or "")).expanduser()
        if not output_dir.is_absolute():
            output_dir = (confirmation_path.parent / output_dir).resolve()
        manifest_path = output_dir / MANIFEST_NAME
        manifest = _load_json(manifest_path)
        fingerprint_payload = manifest.get("fingerprint_payload")
        if not isinstance(fingerprint_payload, Mapping):
            raise ValueError(f"E17 component {source_id!r} cell {cell} has no fingerprint payload")
        expected_manifest = {
            "experiment_id": source_id,
            "stage": "confirm",
            "arm_id": cell[0],
            "task_id": cell[1],
            "seed": cell[2],
            "status": "complete",
        }
        manifest_mismatches = {
            key: (expected, manifest.get(key))
            for key, expected in expected_manifest.items()
            if manifest.get(key) != expected
        }
        expected_fingerprint = {
            "experiment_config_hash": source.raw_hash(),
            "design_hash": expected_design_hash,
            "selection_hash": embedded_selection_hash,
            "baseline_manifest_hash": suite.baseline_manifest_hash,
            "dataset_lock_hash": suite.dataset_lock_hash,
            "model_lock_hash": suite.model_lock_hash,
            "specification_sha256": suite.specification.get("sha256"),
        }
        fingerprint_mismatches = {
            key: (expected, fingerprint_payload.get(key))
            for key, expected in expected_fingerprint.items()
            if fingerprint_payload.get(key) != expected
        }
        if manifest.get("fingerprint") != hash_payload(fingerprint_payload):
            fingerprint_mismatches["fingerprint"] = (
                hash_payload(fingerprint_payload),
                manifest.get("fingerprint"),
            )
        if manifest_mismatches or fingerprint_mismatches:
            raise ValueError(
                f"E17 component {source_id!r} confirmation cell {cell} provenance mismatch: "
                f"manifest={manifest_mismatches}, fingerprint={fingerprint_mismatches}"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("candidate_pool_fingerprint") or "")):
            raise ValueError(
                f"E17 component {source_id!r} cell {cell} has no realized candidate pool"
            )
        candidate_manifest = manifest.get("candidate_pool_manifest_provenance")
        if not isinstance(candidate_manifest, Mapping):
            raise ValueError(
                f"E17 component {source_id!r} cell {cell} has no candidate-pool provenance"
            )
        candidate_path = Path(str(candidate_manifest.get("path") or "")).expanduser().resolve()
        if not candidate_path.is_file() or candidate_manifest.get("sha256") != sha256_file(
            candidate_path
        ):
            raise ValueError(
                f"E17 component {source_id!r} cell {cell} candidate-pool artifact changed"
            )
        if (
            pool_contract.get("mode") == "bound"
            and fingerprint_payload.get("candidate_pool_design_hash") != pool_design_hash
        ):
            raise ValueError(
                f"E17 component {source_id!r} confirmation cell {cell} is not bound "
                "to the composition candidate pool"
            )


def load_and_bind_confirmed_components(path: Path, suite: LoadedSuite) -> LoadedSuite:
    """Bind E17 to immutable confirmed components, never screen-stage winners."""

    record_path = Path(path).expanduser().resolve()
    payload = _load_json(record_path)
    embedded_hash = payload.pop("confirmed_components_hash", None)
    computed_hash = hash_payload(payload)
    payload["confirmed_components_hash"] = embedded_hash
    if not embedded_hash or embedded_hash != computed_hash:
        raise ValueError(
            "confirmed-components record content does not match confirmed_components_hash"
        )
    expected_root = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "stage": "confirm",
        "baseline_id": suite.baseline_id,
        "dataset_lock_hash": suite.dataset_lock_hash,
        "model_lock_hash": suite.model_lock_hash,
        "specification": suite.specification,
    }
    mismatches = {
        key: (expected, payload.get(key))
        for key, expected in expected_root.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"confirmed-components root provenance mismatch: {mismatches}")
    pool_design_hash = str(payload.get("composition_candidate_pool_design_hash") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", pool_design_hash):
        raise ValueError(
            "confirmed-components record requires a 64-hex "
            "composition_candidate_pool_design_hash"
        )

    e17_sources = [source for source in suite.sources if source.config.experiment_id == "E17"]
    if len(e17_sources) != 1 or e17_sources[0].config.composition is None:
        raise ValueError("confirmed-components record requires exactly one E17 composition")
    e17 = e17_sources[0]
    required = {
        component.source_experiment: component
        for component in e17.config.composition.components
        if component.claim and component.source_experiment is not None
    }
    entries = payload.get("components")
    if not isinstance(entries, Mapping) or set(entries) != set(required):
        raise ValueError(
            "confirmed-components record must contain exactly the claimed E17 sources: "
            f"expected={sorted(required)}, observed={sorted(entries or {})}"
        )
    sources = suite.by_id
    overlays: dict[str, dict[str, Any]] = {}
    parent_selection_hashes: set[str] = set()
    for source_id, component in sorted(required.items()):
        raw = entries.get(source_id)
        if not isinstance(raw, Mapping) or raw.get("status") != "confirmed":
            raise ValueError(f"E17 component {source_id!r} is not explicitly confirmed")
        source = sources.get(source_id)
        if source is None:
            raise ValueError(f"E17 component source {source_id!r} is absent from the suite")
        expected_design_hash = experiment_design_hash(
            source,
            baseline_manifest_hash=suite.baseline_manifest_hash,
        )
        if raw.get("experiment_config_hash") != source.raw_hash():
            raise ValueError(f"E17 component {source_id!r} config changed after confirmation")
        if raw.get("design_hash") != expected_design_hash:
            raise ValueError(f"E17 component {source_id!r} design changed after confirmation")
        for field in ("selection_hash", "confirmation_manifest_hash"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(raw.get(field) or "")):
                raise ValueError(f"E17 component {source_id!r} has invalid {field}")
        parent_selection_hashes.add(str(raw["selection_hash"]))
        selected_arm = str(raw.get("selected_arm") or "")
        if selected_arm not in {arm.id for arm in source.config.arms}:
            raise ValueError(
                f"E17 component {source_id!r} names unknown selected arm {selected_arm!r}"
            )
        overlay = raw.get("overlay")
        if not isinstance(overlay, Mapping):
            raise ValueError(f"E17 component {source_id!r} has no frozen overlay")
        overlay_copy = dict(_jsonable(overlay))
        if raw.get("overlay_hash") != hash_payload(overlay_copy):
            raise ValueError(f"E17 component {source_id!r} overlay hash mismatch")

        pool_contract = raw.get("pool_contract")
        if not isinstance(pool_contract, Mapping):
            raise ValueError(f"E17 component {source_id!r} has no pool contract")
        mode = pool_contract.get("mode")
        if mode == "bound":
            if pool_contract.get("candidate_pool_design_hash") != pool_design_hash:
                raise ValueError(
                    f"E17 component {source_id!r} is bound to an incompatible candidate pool"
                )
        elif mode == "independent":
            if not str(pool_contract.get("reason") or "").strip():
                raise ValueError(f"E17 component {source_id!r} pool independence requires a reason")
        else:
            raise ValueError(
                f"E17 component {source_id!r} pool contract mode must be bound or independent"
            )

        _validate_component_confirmation_evidence(
            source_id=source_id,
            source=source,
            raw=raw,
            overlay=overlay_copy,
            pool_contract=pool_contract,
            pool_design_hash=pool_design_hash,
            record_path=record_path,
            suite=suite,
        )

        declared_artifacts: dict[Path, Mapping[str, Any]] = {}
        artifacts = raw.get("artifacts") or []
        if not isinstance(artifacts, list):
            raise ValueError(f"E17 component {source_id!r} artifacts must be a list")
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise ValueError(f"E17 component {source_id!r} has an invalid artifact entry")
            artifact_path = Path(str(artifact.get("path") or "")).expanduser()
            artifact_path = (
                artifact_path.resolve()
                if artifact_path.is_absolute()
                else (record_path.parent / artifact_path).resolve()
            )
            if not artifact_path.is_file():
                raise FileNotFoundError(f"confirmed E17 artifact does not exist: {artifact_path}")
            if artifact.get("sha256") != sha256_file(artifact_path):
                raise ValueError(f"confirmed E17 artifact hash mismatch: {artifact_path}")
            if mode == "bound" and artifact.get("candidate_pool_design_hash") != pool_design_hash:
                raise ValueError(
                    f"confirmed E17 artifact is not refit/bound to the composition pool: "
                    f"{artifact_path}"
                )
            declared_artifacts[artifact_path] = artifact
        referenced_artifacts = _overlay_artifact_paths(
            overlay_copy,
            record_dir=record_path.parent,
        )
        missing_artifacts = sorted(
            str(item) for item in referenced_artifacts - declared_artifacts.keys()
        )
        if missing_artifacts:
            raise ValueError(
                f"E17 component {source_id!r} overlay has undeclared fitted artifacts: "
                f"{missing_artifacts}"
            )
        overlays[source_id] = overlay_copy

    if len(parent_selection_hashes) != 1:
        raise ValueError(
            "confirmed E17 components must share one frozen parent screen selection: "
            f"{sorted(parent_selection_hashes)}"
        )

    return replace(
        suite,
        confirmed_components_record=record_path,
        confirmed_components_hash=str(embedded_hash),
        confirmed_parent_selection_hash=next(iter(parent_selection_hashes)),
        confirmed_component_overlays=overlays,
    )


def inherited_selection_overlay(
    selection_record: Mapping[str, Any],
    dependencies: Sequence[str],
) -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    for dependency_overlay in selected_experiment_overlays(selection_record, dependencies).values():
        overlay = deep_merge(overlay, dependency_overlay)
    return overlay


_E17_DOWNSTREAM_SELECTION_PHASE = "e17_post_component_confirmation"


def _validate_selection_experiments(
    suite: LoadedSuite,
    experiments: Mapping[str, Any],
) -> None:
    expected_ids = {source.config.experiment_id for source in suite.sources}
    observed_ids = {str(experiment_id) for experiment_id in experiments}
    if observed_ids != expected_ids:
        raise ValueError(
            "selection record experiment set differs from the frozen suite: "
            f"expected={sorted(expected_ids)}, observed={sorted(observed_ids)}"
        )
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
            promoted = suite.confirmed_component_overlays if experiment_id == "E17" else None
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


def _validate_e17_downstream_transition(
    suite: LoadedSuite,
    parent: Mapping[str, Any],
    experiments: Mapping[str, Any],
) -> None:
    parent_hash = str(parent.get("selection_hash") or "")
    if parent_hash != suite.confirmed_parent_selection_hash:
        raise ValueError(
            "confirmed-components evidence is not descended from the canonical parent "
            "screen selection"
        )
    parent_experiments = parent.get("experiments")
    if not isinstance(parent_experiments, Mapping):
        raise ValueError("parent screen selection has no experiment mapping")
    if set(parent_experiments) != set(experiments):
        raise ValueError("downstream E17 selection changed the frozen experiment set")
    if "E17" not in experiments:
        raise ValueError("downstream component-confirmation selection requires E17")
    for experiment_id, frozen in experiments.items():
        if experiment_id == "E17":
            continue
        if canonical_json(frozen) != canonical_json(parent_experiments.get(experiment_id)):
            raise ValueError(
                "post-confirmation E17 screening may not change a previously frozen "
                f"selection: {experiment_id}"
            )
    parent_e17 = parent_experiments.get("E17")
    current_e17 = experiments.get("E17")
    if (
        not isinstance(parent_e17, Mapping)
        or parent_e17.get("status") != "deferred_runtime"
        or parent_e17.get("reason_code") != "confirmed_components_record_required"
    ):
        raise ValueError("canonical parent selection did not defer E17 for component evidence")
    if not isinstance(current_e17, Mapping) or current_e17.get("status") not in {
        "selected",
        "screened_out",
    }:
        raise ValueError(
            "post-confirmation E17 screening must freeze a selected or screened-out E17 outcome"
        )


def _design_record(
    suite: LoadedSuite,
    frozen_selection: Mapping[str, Any],
    *,
    lifecycle: Optional[Mapping[str, Any]] = None,
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
        "model_lock_hash": suite.model_lock_hash,
        "confirmed_components_hash": suite.confirmed_components_hash,
        "specification": suite.specification,
        "experiments": experiments,
        "frozen_selection": _jsonable(frozen_selection),
    }
    if lifecycle is not None:
        record["lifecycle"] = _jsonable(lifecycle)
    record["design_record_hash"] = hash_payload(record)
    return record


def _write_json_once(path: Path, payload: Mapping[str, Any], *, label: str) -> None:
    if path.exists():
        existing = _load_json(path)
        if canonical_json(existing) != canonical_json(payload):
            raise FileExistsError(f"cannot change frozen {label}: {path}")
        return
    _atomic_json(path, payload)


def write_selection_record(
    suite: LoadedSuite,
    experiments: Mapping[str, Any],
    *,
    output_root: Path,
) -> Path:
    suite_root = Path(output_root).expanduser().resolve() / suite.suite_id
    confirm_root = suite_root / "confirm"
    confirmation_exists = confirm_root.exists() and any(
        path.is_file() for path in confirm_root.rglob("*")
    )
    stage_root = suite_root / "screen"
    _validate_selection_experiments(suite, experiments)

    lifecycle: Optional[dict[str, Any]] = None
    if suite.confirmed_components_hash is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", suite.confirmed_components_hash):
            raise ValueError("bound confirmed-components hash must be 64-hex")
        if not confirmation_exists:
            raise FileNotFoundError(
                "post-confirmation E17 screening requires the component confirmation "
                "artifacts in the same suite output root"
            )
        parent_path = stage_root / "selection.json"
        parent_suite = replace(
            suite,
            confirmed_components_record=None,
            confirmed_components_hash=None,
            confirmed_parent_selection_hash=None,
            confirmed_component_overlays=None,
        )
        parent = load_and_validate_selection(parent_path, parent_suite)
        _validate_e17_downstream_transition(suite, parent, experiments)
        lifecycle = {
            "phase": _E17_DOWNSTREAM_SELECTION_PHASE,
            "parent_selection": parent_path.name,
            "parent_selection_hash": parent["selection_hash"],
            "confirmed_components_hash": suite.confirmed_components_hash,
        }
        suffix = suite.confirmed_components_hash
        design_name = f"design.e17.{suffix}.json"
        selection_name = f"selection.e17.{suffix}.json"
    elif confirmation_exists:
        raise FileExistsError(
            "cannot overwrite frozen selection/design after confirm artifacts exist"
        )
    else:
        design_name = "design.json"
        selection_name = "selection.json"

    design = _design_record(suite, experiments, lifecycle=lifecycle)
    design_path = stage_root / design_name
    if lifecycle is None:
        _atomic_json(design_path, design)
    else:
        _write_json_once(design_path, design, label="downstream E17 design")
    record: dict[str, Any] = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "stage": "screen",
        "suite_hash": suite.suite_hash,
        "baseline_id": suite.baseline_id,
        "baseline_manifest_hash": suite.baseline_manifest_hash,
        "dataset_lock_hash": suite.dataset_lock_hash,
        "model_lock_hash": suite.model_lock_hash,
        "confirmed_components_hash": suite.confirmed_components_hash,
        "specification": suite.specification,
        "design_record": design_name,
        "design_record_hash": design["design_record_hash"],
        "experiments": _jsonable(experiments),
    }
    if lifecycle is not None:
        record["lifecycle"] = lifecycle
    record["selection_hash"] = hash_payload(record)
    path = stage_root / selection_name
    if lifecycle is None:
        _atomic_json(path, record)
    else:
        _write_json_once(path, record, label="downstream E17 selection")
    return path


def load_and_validate_selection(
    path: Path,
    suite: LoadedSuite,
) -> dict[str, Any]:
    record_path = Path(path).expanduser().resolve()
    record = _load_json(record_path)
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
    if record.get("model_lock_hash") != suite.model_lock_hash:
        raise ValueError("model lock changed after screening")
    if record.get("confirmed_components_hash") != suite.confirmed_components_hash:
        raise ValueError("confirmed-components record changed after E17 screening")
    if record.get("specification") != suite.specification:
        raise ValueError("authoritative specification tree changed after screening")
    design_name = str(record.get("design_record", "design.json"))
    if not design_name or Path(design_name).name != design_name:
        raise ValueError("selection design_record must be a file in the selection directory")
    design_path = record_path.parent / design_name
    design = _load_json(design_path)
    design_hash = design.pop("design_record_hash", None)
    computed_design_hash = hash_payload(design)
    design["design_record_hash"] = design_hash
    if design_hash != computed_design_hash or design_hash != record.get("design_record_hash"):
        raise ValueError("frozen design record changed after screening")

    experiments = record.get("experiments") or {}
    if not isinstance(experiments, Mapping):
        raise ValueError("selection experiments must be a mapping")
    if design.get("frozen_selection") != experiments:
        raise ValueError("selection outcomes disagree with the immutable design record")
    _validate_selection_experiments(suite, experiments)

    lifecycle = record.get("lifecycle")
    if lifecycle is None:
        if design.get("lifecycle") is not None:
            raise ValueError("selection and design lifecycle declarations disagree")
        return record
    if not isinstance(lifecycle, Mapping) or set(lifecycle) != {
        "phase",
        "parent_selection",
        "parent_selection_hash",
        "confirmed_components_hash",
    }:
        raise ValueError("invalid downstream E17 selection lifecycle")
    if lifecycle.get("phase") != _E17_DOWNSTREAM_SELECTION_PHASE:
        raise ValueError("unknown selection lifecycle phase")
    if design.get("lifecycle") != lifecycle:
        raise ValueError("selection and design lifecycle declarations disagree")
    if lifecycle.get("confirmed_components_hash") != suite.confirmed_components_hash:
        raise ValueError("downstream selection changed confirmed-components evidence")
    if lifecycle.get("parent_selection_hash") != suite.confirmed_parent_selection_hash:
        raise ValueError("downstream selection changed its component-screen parent")
    parent_name = str(lifecycle.get("parent_selection") or "")
    if parent_name != "selection.json":
        raise ValueError("downstream E17 selection must descend from screen/selection.json")
    parent_path = record_path.parent / parent_name
    if parent_path == record_path:
        raise ValueError("downstream E17 selection may not reference itself as parent")
    parent_suite = replace(
        suite,
        confirmed_components_record=None,
        confirmed_components_hash=None,
        confirmed_parent_selection_hash=None,
        confirmed_component_overlays=None,
    )
    parent = load_and_validate_selection(parent_path, parent_suite)
    _validate_e17_downstream_transition(suite, parent, experiments)
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
        if suite.confirmed_components_hash is not None and config.experiment_id != "E17":
            print(
                f"FROZEN_DEPENDENCY\t{config.experiment_id}\t" "reused_from_component_confirmation"
            )
            continue
        if config.implementation.status != "ready":
            detail = ""
            if config.implementation.status == "deferred_unavailable":
                detail = (
                    f"\tmissing={config.implementation.missing_capability}"
                    f"\texpected={config.implementation.expected_dataset}"
                )
            print(
                f"DEFERRED\t{config.experiment_id}\tdepends={dependencies}\t"
                f"{config.implementation.status}"
                f"[{config.implementation.reason_code}]: {config.implementation.reason}"
                f"{detail}"
            )
            continue
        if config.experiment_id == "E17" and suite.confirmed_component_overlays is None:
            print(
                "DEFERRED_RUNTIME\tE17\tconfirmed_components_record_required: "
                "supply --confirmed-components-record after component confirmations"
            )
            continue
        stage_config = config.screen if stage == "screen" else config.confirm
        for task in stage_config.tasks:
            if task.availability.status == "deferred_unavailable":
                print(
                    f"TASK_DEFERRED\t{config.experiment_id}/{task.id}\t"
                    f"{task.availability.status}[{task.availability.reason_code}]: "
                    f"{task.availability.reason}\t"
                    f"missing={task.availability.missing_capability}\t"
                    f"expected={task.availability.expected_dataset}"
                )
        promoted_components = (
            suite.confirmed_component_overlays if config.experiment_id == "E17" else None
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


def _runtime_deferred_selection(
    source: ExperimentSource,
    suite: LoadedSuite,
    *,
    reason_code: str,
    reason: str,
) -> dict[str, Any]:
    base = ConfigModel.from_mapping(_base_mapping(source), warn_v1=False).model_dump(
        mode="json", by_alias=True
    )
    return {
        "experiment_config_hash": source.raw_hash(),
        "base_config_hash": hash_payload(base),
        "design_hash": experiment_design_hash(
            source,
            baseline_manifest_hash=suite.baseline_manifest_hash,
        ),
        "candidate_pool_design_hash": _candidate_design_hash(base),
        "status": "deferred_runtime",
        "reason_code": reason_code,
        "reason": reason,
        "decisions": [],
    }


def run_stage(
    suite: LoadedSuite,
    *,
    stage: str,
    output_root: Path,
    jobs: int,
    resume: bool,
    workdir: Path,
    selection_record_path: Optional[Path] = None,
    confirmed_components_record_path: Optional[Path] = None,
    dry_run: bool = False,
) -> Optional[Path]:
    if stage not in {"screen", "confirm"}:
        raise ValueError("stage must be screen or confirm")
    if confirmed_components_record_path is not None:
        suite = load_and_bind_confirmed_components(confirmed_components_record_path, suite)
    if stage == "confirm" and (
        not isinstance(suite.model_lock_payload, Mapping)
        or suite.model_lock_payload.get("status") != "complete"
    ):
        raise ValueError(
            "confirm requires a complete immutable model lock; screen and dry-run remain available"
        )
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

    downstream_parent: Optional[dict[str, Any]] = None
    e17_only = False
    if suite.confirmed_components_hash is not None:
        if stage == "screen":
            suite_root = Path(output_root).expanduser().resolve() / suite.suite_id
            confirm_root = suite_root / "confirm"
            if not confirm_root.exists() or not any(
                path.is_file() for path in confirm_root.rglob("*")
            ):
                raise FileNotFoundError(
                    "post-confirmation E17 screening requires the component confirmation "
                    "artifacts in the same suite output root"
                )
            parent_suite = replace(
                suite,
                confirmed_components_record=None,
                confirmed_components_hash=None,
                confirmed_parent_selection_hash=None,
                confirmed_component_overlays=None,
            )
            downstream_parent = load_and_validate_selection(
                suite_root / "screen" / "selection.json",
                parent_suite,
            )
            if downstream_parent.get("selection_hash") != suite.confirmed_parent_selection_hash:
                raise ValueError(
                    "confirmed-components evidence does not descend from this output "
                    "root's canonical screen selection"
                )
            e17_only = True
        else:
            lifecycle = selection.get("lifecycle") if isinstance(selection, Mapping) else None
            e17_only = (
                isinstance(lifecycle, Mapping)
                and lifecycle.get("phase") == _E17_DOWNSTREAM_SELECTION_PHASE
            )

    inventory_suite = suite
    if e17_only:
        e17_sources = tuple(
            source for source in suite.sources if source.config.experiment_id == "E17"
        )
        if len(e17_sources) != 1:
            raise ValueError("downstream component-confirmation phase requires exactly one E17")
        inventory_suite = replace(suite, sources=e17_sources)
    build_dataset_inventory(
        inventory_suite,
        stage=stage,
        output_root=output_root,
        selection_record=selection,
    )

    all_manifests: list[dict[str, Any]] = []
    selections: dict[str, Any] = (
        dict(_jsonable(downstream_parent.get("experiments") or {}))
        if downstream_parent is not None
        else {}
    )
    incremental_selection: dict[str, Any] = {
        "experiments": selections,
        "selection_hash": None,
    }
    for source in suite.sources:
        config = source.config
        if e17_only and config.experiment_id != "E17":
            continue
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
                    "reason_code": config.implementation.reason_code,
                    "missing_capability": config.implementation.missing_capability,
                    "expected_dataset": config.implementation.expected_dataset,
                    "decisions": [],
                }
            continue
        if config.experiment_id == "E17" and suite.confirmed_component_overlays is None:
            if stage == "screen":
                selections[config.experiment_id] = _runtime_deferred_selection(
                    source,
                    suite,
                    reason_code="confirmed_components_record_required",
                    reason=(
                        "E17 waits for immutable component confirmation evidence. The "
                        "component screen/confirm dependency subphase may run without E17; "
                        "rerun screen with --confirmed-components-record once the user has "
                        "reviewed and frozen those confirmations."
                    ),
                )
            continue
        source_selection = selection if stage == "confirm" else incremental_selection
        assert source_selection is not None
        promoted_components = (
            suite.confirmed_component_overlays if config.experiment_id == "E17" else None
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
        if config.experiment_id == "E00":
            from exact.experiments.replay import ReplayValidationError, validate_e00_replay

            replay_path = (
                Path(output_root).expanduser().resolve()
                / suite.suite_id
                / stage
                / "replay"
                / "E00.json"
            )
            try:
                replay = validate_e00_replay(manifests)
            except ReplayValidationError as exc:
                _atomic_json(replay_path, exc.record)
                raise
            _atomic_json(replay_path, replay)
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
    "load_and_bind_confirmed_components",
    "load_suite_or_experiment",
    "print_dry_run",
    "run_stage",
    "select_experiment",
    "specification_tree_identity",
    "write_selection_record",
]
