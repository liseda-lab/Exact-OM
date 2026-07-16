"""Declarative Exact-OM configuration v1 to v2 migration."""

from __future__ import annotations

import difflib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

from exact.core.entities.configs.config import CONFIG_VERSION, default_pipeline_entries


@dataclass(frozen=True)
class Drop:
    """A v1 key intentionally removed from v2."""

    reason: str


@dataclass(frozen=True)
class Transform:
    """A v1 key handled by a named multi-key transformation."""

    name: str


MigrationTarget = Union[str, Drop, Transform]


_REASONER_REMOVED = "the retired Java reasoner control has no v2 runtime effect"
_RUNTIME_DERIVED = "runtime component dependencies are derived from the primary pipeline model"
_PIPELINE_TRANSFORM = Transform("model_stack_to_pipeline")


# One table covers every field accepted by the final v1 schema.  Opaque plugin
# payloads (model params, Bio-ML options, profile provider options, and hierarchy
# family definitions) move wholesale at their owning path.
V1_TO_V2: Dict[str, MigrationTarget] = {
    "logging_level": "run.logging_level",
    "seed": "run.seed",
    "use_file_cache": "run.use_file_cache",
    "data": "data",
    "dataset_track": Transform("dataset_track_alias"),
    "k": "evaluation.k",
    "evaluation.backends": "evaluation.backends",
    "evaluation.k": "evaluation.k",
    "evaluation.bioml": "evaluation.bioml",
    "alignment_params.threshold": "matching.threshold",
    "alignment_params.cardinality": "matching.cardinality",
    "alignment_params.target_cardinality": "matching.target_cardinality",
    "alignment_params.review_low": "matching.review_low",
    "alignment_params.review_high": "matching.review_high",
    "alignment_params.save_json": "output.save.json",
    "alignment_params.save_csv": "output.save.csv",
    "alignment_params.save_stats_csv": "output.save.stats_csv",
    "alignment_params.append_stats_to_summary_csv": "output.save.append_stats_to_summary",
    "dataset_params.reasoner": "dataset.reasoner",
    "dataset_params.num_workers": "dataset.num_workers",
    "dataset_params.filter_exact_matches": "dataset.filter_exact_matches",
    "dataset_params.drop_exact_match_sources": "dataset.drop_exact_match_sources",
    "dataset_params.filter_ignored_alignment_classes": (
        "dataset.filter_ignored_alignment_classes"
    ),
    "dataset_params.projection_include_literals": "dataset.projection_include_literals",
    "dataset_params.hierarchy_max_depth": "dataset.hierarchy_max_depth",
    "dataset_params.max_hierarchy_triples_per_family": (
        "dataset.max_hierarchy_triples_per_family"
    ),
    "dataset_params.max_object_triples": "dataset.max_object_triples",
    "dataset_params.max_diff_triples": "dataset.max_diff_triples",
    "dataset_params.max_attr_items": "dataset.max_attr_items",
    "dataset_params.pair_adaptive_feature_log_every": (
        "dataset.pair_adaptive_feature_log_every"
    ),
    "dataset_params.hierarchical_relation_families": (
        "dataset.hierarchical_relation_families"
    ),
    "dataset_params.n_hops": "dataset.n_hops",
    "dataset_params.max_input_tokens_context": "dataset.max_input_tokens_context",
    "dataset_params.all_labels": "dataset.all_labels",
    "dataset_params.delimiter": "dataset.delimiter",
    "dataset_params.which": "dataset.which",
    "dataset_params.candidate_share_k": "dataset.candidate_share_k",
    "dataset_params.context_method": "dataset.legacy.context_method",
    "dataset_params.best_path_method": "dataset.legacy.best_path_method",
    "dataset_params.context_hop_penalty": "dataset.legacy.context_hop_penalty",
    "dataset_params.context_token_ratio": "dataset.legacy.context_token_ratio",
    "dataset_params.context_safety": "dataset.legacy.context_safety",
    "dataset_params.only_taxonomy": "dataset.legacy.only_taxonomy",
    "dataset_params.add_connectivity_bridges": "dataset.legacy.add_connectivity_bridges",
    "dataset_params.bridge_max_hops": "dataset.legacy.bridge_max_hops",
    "dataset_params.verbaliser_name": "llm.verbaliser.model",
    "dataset_params.gen_max_new_tokens": "llm.verbaliser.max_new_tokens",
    "dataset_params.temperature": "llm.verbaliser.temperature",
    "dataset_params.top_p": "llm.verbaliser.top_p",
    "dataset_params.top_k": "llm.verbaliser.top_k",
    "dataset_params.do_sample": "llm.verbaliser.do_sample",
    "dataset_params.batch_size_verbaliser": "llm.verbaliser.batch_size",
    "dataset_params.max_verb_gen_retries": "llm.verbaliser.max_retries",
    "dataset_params.exclude_missing_dr": "llm.verbaliser.exclude_missing_domain_range",
    "dataset_params.reasoner_timeout_secs": Drop(_REASONER_REMOVED),
    "dataset_params.reasoner_force_hermit": Drop(_REASONER_REMOVED),
    "candidates_params.retrieval_strategy": "candidates.retrieval_strategy",
    "candidates_params.lexical_encoder_name": "candidates.lexical_encoder_name",
    "candidates_params.encode_batch_size": "candidates.encode_batch_size",
    "candidates_params.search_batch_size": "candidates.search_batch_size",
    "candidates_params.top_k": "candidates.top_k",
    "candidates_params.use_amp": "candidates.use_amp",
    "sanity_check_params.sanity_check": "output.sanity_checks.enabled",
    "sanity_check_params.n": "output.sanity_checks.n",
    "sanity_check_params.max_ctx_show": "output.sanity_checks.max_ctx_show",
    "sanity_check_params.max_label_show": "output.sanity_checks.max_label_show",
    "plot_params.bins": "output.plots.bins",
    "plot_params.figsize": "output.plots.figsize",
    "plot_params.dpi": "output.plots.dpi",
    "plot_params.kde": "output.plots.kde",
    "plot_params.alpha": "output.plots.alpha",
    "inference_params.batch_size": "inference.batch_size",
    "inference_params.num_workers": "inference.num_workers",
    "inference_params.log_every": "inference.log_every",
    "inference_params.mixed_precision": "inference.mixed_precision",
    "inference_params.which": "inference.which",
    "inference_params.checkpoint_every": "inference.checkpoint_every",
    "inference_params.resume_from_checkpoint": "inference.resume_from_checkpoint",
    "inference_params.enable_checkpoints": "inference.enable_checkpoints",
    "inference_params.resume_additional_model_checkpoints": (
        "inference.resume_additional_model_checkpoints"
    ),
    "inference_params.allow_rationale_toggle_checkpoint_resume": (
        "inference.allow_rationale_toggle_checkpoint_resume"
    ),
    "inference_params.audit_shards_enabled": "inference.audit_shards_enabled",
    "inference_params.audit_shard_compression": "inference.audit_shard_compression",
    "inference_params.audit_shard_records": "inference.audit_shard_records",
    "inference_params.checkpoint_payload": "inference.checkpoint_payload",
    "inference_params.cache_persist_policy": "inference.cache_persist_policy",
    "llm_profiles": "llm.profiles",
    "llm_routing": "llm.routing",
    "model": _PIPELINE_TRANSFORM,
    "model.name": _PIPELINE_TRANSFORM,
    "model.params": _PIPELINE_TRANSFORM,
    "model.component_type": Drop("pipeline entries infer the model component type"),
    "second_model": _PIPELINE_TRANSFORM,
    "second_model.name": _PIPELINE_TRANSFORM,
    "second_model.params": _PIPELINE_TRANSFORM,
    "second_model.component_type": Drop("pipeline entries infer the model component type"),
    "model_chain": _PIPELINE_TRANSFORM,
    "model_chain.*.name": _PIPELINE_TRANSFORM,
    "model_chain.*.params": _PIPELINE_TRANSFORM,
    "model_chain.*.component_type": Drop("pipeline entries infer the model component type"),
    "second_pass_params": Transform("second_pass_params_to_pipeline"),
    "dataset": Drop(_RUNTIME_DERIVED),
    "trainer": Drop(_RUNTIME_DERIVED),
}


@dataclass(frozen=True)
class MigrationChange:
    source: str
    destination: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class MigrationReport:
    moved: list[MigrationChange] = field(default_factory=list)
    transformed: list[MigrationChange] = field(default_factory=list)
    dropped: list[MigrationChange] = field(default_factory=list)

    def render(self) -> str:
        lines = ["Configuration migration report:"]
        if self.moved:
            lines.append("Moved:")
            lines.extend(
                f"  {item.source} -> {item.destination}" for item in self.moved
            )
        if self.transformed:
            lines.append("Transformed:")
            lines.extend(
                f"  {item.source} -> {item.destination}" for item in self.transformed
            )
        if self.dropped:
            lines.append("Dropped:")
            lines.extend(
                f"  {item.source}: {item.reason}" for item in self.dropped
            )
        if not (self.moved or self.transformed or self.dropped):
            lines.append("  No v1 keys required migration.")
        return "\n".join(lines)


_MISSING = object()


def _get_path(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _set_path(mapping: MutableMapping[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = mapping
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = deepcopy(value)


def _unknown_v1_key(path: str, choices: list[str]) -> ValueError:
    leaf = path.rsplit(".", 1)[-1]
    leaf_choices = sorted({choice.rsplit(".", 1)[-1] for choice in choices})
    suggestion = difflib.get_close_matches(leaf, leaf_choices, n=1, cutoff=0.55)
    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
    return ValueError(f"Unknown v1 configuration key '{path}'.{hint}")


def _validate_registry_entry(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    choices = ["name", "params", "component_type"]
    for key in value:
        if str(key) not in choices:
            raise _unknown_v1_key(f"{path}.{key}", choices)
    if "params" in value and not isinstance(value["params"], Mapping):
        raise ValueError(f"{path}.params must be a mapping")


def _validate_v1_keys(raw: Mapping[str, Any]) -> None:
    known_roots = sorted({path.split(".", 1)[0] for path in V1_TO_V2})
    for root, value in raw.items():
        root_text = str(root)
        if root_text not in known_roots:
            raise _unknown_v1_key(root_text, known_roots)
        if root_text in {"model", "second_model"} and value is not None:
            _validate_registry_entry(value, root_text)
            continue
        if root_text == "model_chain":
            if value is not None and not isinstance(value, list):
                raise ValueError("model_chain must be a list")
            for index, entry in enumerate(value or []):
                _validate_registry_entry(entry, f"model_chain.{index}")
            continue
        if root_text in {
            "data",
            "dataset_track",
            "llm_profiles",
            "llm_routing",
            "second_pass_params",
        }:
            continue
        child_paths = [
            path for path in V1_TO_V2 if path.startswith(f"{root_text}.") and "*" not in path
        ]
        if not child_paths:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"{root_text} must be a mapping")
        allowed = [path.split(".", 1)[1] for path in child_paths]
        for key in value:
            if str(key) not in allowed:
                raise _unknown_v1_key(f"{root_text}.{key}", child_paths)


def _merge_registry_entry(
    raw_entry: Optional[Mapping[str, Any]], default_entry: Mapping[str, Any]
) -> Dict[str, Any]:
    base = deepcopy(dict(default_entry))
    override = dict(raw_entry or {})
    merged = {**base, **{key: deepcopy(value) for key, value in override.items() if key not in {"params", "component_type"}}}
    merged["params"] = {
        **deepcopy(dict(base.get("params") or {})),
        **deepcopy(dict(override.get("params") or {})),
    }
    return merged


def _pipeline_from_v1(raw: Mapping[str, Any]) -> list[Dict[str, Any]]:
    model_chain = raw.get("model_chain")
    if model_chain:
        return [
            {
                "name": str(entry.get("name")),
                "params": deepcopy(dict(entry.get("params") or {})),
            }
            for entry in model_chain
        ]

    defaults = [entry.model_dump(mode="python") for entry in default_pipeline_entries()]
    primary = _merge_registry_entry(raw.get("model"), defaults[0])
    explicit_second = raw.get("second_model")
    legacy_second_pass = raw.get("second_pass_params", _MISSING)
    if explicit_second:
        secondary = _merge_registry_entry(explicit_second, defaults[1])
    elif legacy_second_pass is not _MISSING and legacy_second_pass is not None:
        secondary = _merge_registry_entry(
            {"name": "SecondPassReranker", "params": legacy_second_pass}, defaults[1]
        )
    else:
        secondary = deepcopy(defaults[1])
    pipeline = [primary]
    if secondary.get("name") is not None:
        pipeline.append(secondary)
    return pipeline


def migrate_v1_mapping(raw: Mapping[str, Any]) -> tuple[Dict[str, Any], MigrationReport]:
    """Migrate an unversioned v1 mapping to a v2 mapping and a detailed report."""

    if not isinstance(raw, Mapping):
        raise ValueError("configuration root must be a mapping")
    if "config_version" in raw:
        raise ValueError("migrate_v1_mapping expects an unversioned v1 configuration")
    _validate_v1_keys(raw)

    migrated: Dict[str, Any] = {"config_version": CONFIG_VERSION}
    report = MigrationReport()

    for source, target in V1_TO_V2.items():
        if "*" in source or isinstance(target, Transform):
            continue
        value = _get_path(raw, source)
        if value is _MISSING:
            continue
        if source == "data" and value is None:
            continue
        if isinstance(target, Drop):
            report.dropped.append(MigrationChange(source=source, reason=target.reason))
            continue
        _set_path(migrated, target, value)
        if source != target:
            report.moved.append(MigrationChange(source=source, destination=target))

    data = raw.get("data")
    dataset_track = raw.get("dataset_track")
    if dataset_track is not None:
        if data is None:
            migrated["data"] = deepcopy(dataset_track)
        report.transformed.append(MigrationChange("dataset_track", "data"))

    stack_keys = {"model", "second_model", "model_chain", "second_pass_params"}
    if any(key in raw for key in stack_keys):
        migrated["pipeline"] = _pipeline_from_v1(raw)
        if "model_chain" in raw:
            report.transformed.append(MigrationChange("model_chain", "pipeline"))
        else:
            sources = [key for key in ("model", "second_model") if key in raw]
            if sources:
                report.transformed.append(
                    MigrationChange(" + ".join(sources), "pipeline")
                )
        if "second_pass_params" in raw:
            report.transformed.append(
                MigrationChange("second_pass_params", "pipeline")
            )

    for index, entry in enumerate(raw.get("model_chain") or []):
        if "component_type" in entry:
            target = V1_TO_V2["model_chain.*.component_type"]
            assert isinstance(target, Drop)
            report.dropped.append(
                MigrationChange(f"model_chain.{index}.component_type", reason=target.reason)
            )

    return migrated, report


__all__ = [
    "Drop",
    "MigrationChange",
    "MigrationReport",
    "MigrationTarget",
    "Transform",
    "V1_TO_V2",
    "migrate_v1_mapping",
]
