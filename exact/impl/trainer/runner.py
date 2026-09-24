from __future__ import annotations

import hashlib  # noqa: F401
import inspect  # noqa: F401
import json  # noqa: F401
import math
import os
import re
import signal
import time  # noqa: F401
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path  # noqa: F401
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple  # noqa: F401
from urllib.parse import urlsplit, urlunsplit

import pandas as pd  # noqa: F401
import torch  # noqa: F401
from torch.utils.data import DataLoader, Subset  # noqa: F401
from torch.utils.data._utils.collate import default_collate  # noqa: F401

from exact.core.entities.configs.dataset import DatasetMask  # noqa: F401
from exact.core.entities.mappings import EntityMapping  # noqa: F401
from exact.impl.extraction import extract_global_alignment
from exact.io.relations import predict_relations
from exact.io.writers import write as write_alignment_format
from exact.runs.layout import RunLayout
from exact.utils.formatting import format_duration as _format_duration  # noqa: F401
from exact.utils.run_context import current_run_session
from exact.utils.timing import CacheStatus, StageRecord  # noqa: F401

try:
    import zstandard as zstd  # noqa: F401
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    zstd = None

from exact.core.contracts.alignment_io import bind_alignment_io
from exact.core.contracts.trainer import ITrainer

from .anchors import AnchorPreparationMixin
from .audit_io import AuditIOMixin, _semantic_collate_fn
from .checkpointing import CheckpointingMixin
from .fitting import TrainingPoolMixin, source_batches
from .overlays import OverlaysMixin
from .rationales import RationalesMixin

bind_alignment_io(
    relation_typer=predict_relations,
    writer_dispatch=write_alignment_format,
)


@dataclass
class _FinalizationState:
    kind: DatasetMask
    threshold: Optional[float]
    cardinality: Optional[int]
    target_cardinality: Optional[int]
    local_alignment: bool
    all_mappings: List[Tuple[str, str, float]]
    total_examples: int
    processed_examples: int
    restored_examples: int
    checkpoint_enabled: bool
    checkpoint_path: Optional[Path]
    audit_shard_compression: str
    audit_shard_records: int
    log_every: int
    run_progress: Optional[Any]
    inference_seconds: float
    average_seconds_per_example: float
    inference_work_done: int
    inference_status: CacheStatus
    post_inference_status: CacheStatus
    completed_from_checkpoint: bool = False


class SemanticAlignmentRunner(
    AnchorPreparationMixin,
    TrainingPoolMixin,
    CheckpointingMixin,
    AuditIOMixin,
    RationalesMixin,
    OverlaysMixin,
    ITrainer,
):
    """
    External loop orchestrator for SemanticScorer inference.
    Collects all outputs (scores, weights, explanations) for saving & plotting.
    """

    def __init__(
        self,
        dataset,
        model=None,
        model_params: Optional[Dict[str, Any]] = None,
        models: Optional[List[Tuple[Any, Dict[str, Any]]]] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        output_dir: Optional[Path] = None,
        logger: Optional[Any] = None,
        extraction_config: Optional[Dict[str, Any]] = None,
        training_candidates_file_path: Optional[Path] = None,
        fitting_fusion_config: Optional[Dict[str, Any]] = None,
        fitting_graph_config: Optional[Dict[str, Any]] = None,
        fitting_llm_config: Optional[Dict[str, Any]] = None,
        training_reference_file_path: Optional[Path] = None,
        fitting_gate_config: Optional[Dict[str, Any]] = None,
        source_universe: Optional[List[str]] = None,
        supervision_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.fitting_graph_config = fitting_graph_config
        self.fitting_llm_config = fitting_llm_config
        self.training_reference_file_path = training_reference_file_path
        self.fitting_gate_config = fitting_gate_config
        self.source_universe = source_universe
        self.supervision_config = dict(supervision_config or {})
        self.fitting_fusion_config = fitting_fusion_config
        self.training_candidates_file_path = training_candidates_file_path
        self._extraction_config = {
            "mode": "greedy",
            "assignment_component_cap": 500,
            **dict(extraction_config or {}),
        }
        extraction_mode = str(self._extraction_config["mode"]).strip().lower()
        if extraction_mode not in {
            "greedy",
            "mutual_best",
            "assignment",
            "assignment_accepted_utility",
            "assignment_legacy",
            "stable_marriage",
        }:
            raise ValueError(f"Unknown global extraction mode: {extraction_mode!r}")
        self._extraction_config["mode"] = extraction_mode
        self._extraction_includes_prefilter = False
        self._extraction_diagnostics: Dict[str, Any] = {}

        params = dict(model_params or {})
        cache_dir = params.get("cache_dir")
        if cache_dir is None and output_dir is not None:
            params["cache_dir"] = RunLayout.create(output_dir).cache_dir
        ds_signature = getattr(dataset, "dataset_signature", None)
        if ds_signature:
            params.setdefault("dataset_signature", ds_signature)
            params.setdefault("cache_namespace", ds_signature)

        model_specs = None
        if models is not None:
            model_specs = []
            for idx, (m_cls, m_params) in enumerate(models):
                spec_params = dict(m_params or {})
                if idx == 0:
                    cache_dir = spec_params.get("cache_dir")
                    if cache_dir is None and output_dir is not None:
                        spec_params["cache_dir"] = RunLayout.create(output_dir).cache_dir
                    if ds_signature:
                        spec_params.setdefault("dataset_signature", ds_signature)
                        spec_params.setdefault("cache_namespace", ds_signature)
                model_specs.append((m_cls, spec_params))

        super().__init__(
            dataset=dataset,
            model=model,
            model_params=params if model_specs is None else None,
            models=model_specs,
            device=device,
            output_dir=output_dir,
            logger=logger,
            **kwargs,
        )
        self._last_stage_timings: List[StageRecord] = []
        self._inference_seconds_cumulative: float = 0.0
        self._restored_inference_seconds_cumulative: float = 0.0
        self._inference_session_started_at: Optional[float] = None
        self._examples_per_second_ema: Optional[float] = None
        self._last_effective_threshold: Optional[float] = None
        self._last_effective_threshold_origin: str = "configured"
        self._llm_backend_observations: List[Dict[str, Any]] = []
        self._checkpoint_fingerprint_payload: Dict[str, Any] = (
            self._build_checkpoint_fingerprint_payload()
        )
        self._checkpoint_fingerprint: str = self._hash_checkpoint_fingerprint_payload(
            self._checkpoint_fingerprint_payload
        )
        self._audit_shards_enabled: bool = True
        self._audit_shard_compression: str = "zstd"
        self._audit_shard_records: int = 50000
        self._audit_manifest_path: Optional[Path] = None
        self._audit_shard_dir: Optional[Path] = None
        self._audit_shards: List[Dict[str, Any]] = []
        self._audit_total_records: int = 0
        self._audit_current_writer: Optional[Any] = None
        self._audit_current_shard: Optional[Dict[str, Any]] = None
        self._candidate_records_enabled: bool = True
        self._candidate_manifest_path: Optional[Path] = None
        self._candidate_shard_dir: Optional[Path] = None
        self._candidate_shards: List[Dict[str, Any]] = []
        self._candidate_total_records: int = 0
        self._candidate_current_writer: Optional[Any] = None
        self._candidate_current_shard: Optional[Dict[str, Any]] = None
        self._overlay_manifest_path: Optional[Path] = None
        self._checkpoint_payload_mode: str = "compact"
        self._cache_persist_policy: str = "finalize"
        self._cache_persist_skip_logged: Set[str] = set()
        self._restored_candidate_rows: List[Dict[str, Any]] = []
        run_session = current_run_session()
        self._run_id: str = (
            str(run_session.run_id) if getattr(run_session, "run_id", None) else str(uuid.uuid4())
        )
        self._explanation_store: Optional[Any] = None
        self._explanation_shard_mb: float = 32.0

    @staticmethod
    def _json_safe_value(value: Any) -> Any:
        """Normalize nested diagnostics into deterministic strict-JSON values."""

        if isinstance(value, torch.Tensor):
            return SemanticAlignmentRunner._json_safe_value(value.detach().cpu().tolist())
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, Mapping):
            return {
                str(key): SemanticAlignmentRunner._json_safe_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [SemanticAlignmentRunner._json_safe_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [SemanticAlignmentRunner._json_safe_value(item) for item in value]
            return sorted(
                normalized,
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                return SemanticAlignmentRunner._json_safe_value(item_method())
            except (TypeError, ValueError):
                pass
        return str(value)

    @classmethod
    def _gate_candidate_fields(cls, diagnostic: Any) -> Dict[str, Any]:
        if not isinstance(diagnostic, Mapping):
            return {}
        safe = cls._json_safe_value(diagnostic)
        if not isinstance(safe, dict):
            return {}
        fields: Dict[str, Any] = {
            "llm_gate_diagnostics": json.dumps(
                safe,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        }
        scalar_fields = {
            "mode": "llm_gate_mode",
            "statistic_name": "llm_gate_statistic_name",
            "statistic": "llm_gate_statistic",
            "threshold": "llm_gate_threshold",
            "would_route": "llm_gate_would_route",
            "invoked": "llm_gate_invoked",
            "outcome": "llm_gate_outcome",
        }
        for source, target in scalar_fields.items():
            if source in safe and not isinstance(safe[source], (dict, list)):
                fields[target] = safe[source]
        if "llm_gate_outcome" not in fields and ("would_route" in safe or "invoked" in safe):
            would_route = bool(safe.get("would_route", False))
            invoked = bool(safe.get("invoked", False))
            fields["llm_gate_outcome"] = (
                "invoked" if invoked else "routed_not_invoked" if would_route else "not_routed"
            )
        return fields

    @staticmethod
    def _safe_llm_identity_text(value: Any, *, limit: int = 512) -> Optional[str]:
        if value is None or isinstance(value, (Mapping, list, tuple, set, frozenset)):
            return None
        text = str(value).strip()
        if not text or len(text) > limit or any(ord(char) < 32 for char in text):
            return None
        return text

    @classmethod
    def _safe_llm_endpoint(cls, value: Any) -> Optional[str]:
        text = cls._safe_llm_identity_text(value, limit=2048)
        if text is None:
            return None
        try:
            parsed = urlsplit(text)
        except ValueError:
            return None
        if parsed.scheme or parsed.netloc:
            if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
                return None
            hostname = parsed.hostname
            safe_host = f"[{hostname}]" if ":" in hostname else hostname
            try:
                port = parsed.port
            except ValueError:
                return None
            netloc = f"{safe_host}:{port}" if port is not None else safe_host
            return urlunsplit((parsed.scheme.lower(), netloc, parsed.path, "", ""))
        relative = text.split("?", 1)[0].split("#", 1)[0].strip()
        if not relative or "@" in relative:
            return None
        return relative

    @classmethod
    def _safe_llm_endpoint_identity(cls, value: Any) -> Optional[str]:
        text = cls._safe_llm_identity_text(value, limit=2048)
        if text is None:
            return None
        if "://" in text:
            return cls._safe_llm_endpoint(text)
        identity = text.split("?", 1)[0].split("#", 1)[0].strip()
        if (
            not identity
            or "@" in identity
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", identity)
        ):
            return None
        return identity

    @classmethod
    def _safe_llm_provider(cls, value: Any) -> Optional[str]:
        if isinstance(value, Mapping):
            for key in ("name", "id", "provider"):
                candidate = cls._safe_llm_identity_text(value.get(key))
                if candidate is not None:
                    return candidate
            return None
        return cls._safe_llm_identity_text(value)

    @staticmethod
    def _safe_llm_hash(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        candidate = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:_-]{7,199}", candidate):
            return None
        return candidate

    @staticmethod
    def _safe_llm_usage_counter(value: Any) -> Optional[int]:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.item()
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return int(value) if value >= 0 else None
        if isinstance(value, float) and math.isfinite(value) and value >= 0 and value.is_integer():
            return int(value)
        return None

    @classmethod
    def _llm_usage_counters(cls, metadata: Mapping[str, Any]) -> Dict[str, int]:
        sources: List[Mapping[str, Any]] = []
        for field in ("usage", "token_usage", "response_usage"):
            nested = metadata.get(field)
            if isinstance(nested, Mapping):
                sources.append(nested)
        sources.append(metadata)
        aliases = {
            "calls": ("calls", "call_count", "request_count", "requests", "n_calls"),
            "input_tokens": ("input_tokens", "prompt_tokens"),
            "output_tokens": ("output_tokens", "completion_tokens"),
            "total_tokens": ("total_tokens",),
        }
        counters: Dict[str, int] = {}
        for target, names in aliases.items():
            for source in sources:
                found = False
                for name in names:
                    if name not in source:
                        continue
                    found = True
                    value = cls._safe_llm_usage_counter(source.get(name))
                    if value is not None:
                        counters[target] = value
                    break
                if target in counters or found:
                    break
        return counters

    @classmethod
    def _llm_model_revision(cls, value: Any) -> Optional[str]:
        model = cls._safe_llm_identity_text(value)
        if model is None:
            return None
        _, separator, revision = model.rpartition("@")
        return cls._safe_llm_identity_text(revision) if separator else None

    @staticmethod
    def _llm_profile_matches(profile: Any, backend: Any, model: Any) -> bool:
        if profile is None:
            return False
        if backend is not None and str(getattr(profile, "backend", "")) != str(backend):
            return False
        profile_model = getattr(profile, "model", None)
        return model is None or profile_model is None or str(profile_model) == str(model)

    def _llm_decoding_defaults(self, task: str, backend: Optional[str]) -> Dict[str, Any]:
        model = self.model
        decoding: Dict[str, Any] = {}
        if task == "summary":
            decoding["max_input_tokens"] = getattr(model, "max_total_tokens_llm_summary", None)
            token_field = "max_tokens" if backend == "openrouter" else "max_new_tokens"
            decoding[token_field] = getattr(model, "max_new_tokens_llm", None)
        elif task == "rationale":
            decoding["max_input_tokens"] = getattr(model, "max_total_tokens_llm_rationale", None)
            token_field = "max_tokens" if backend == "openrouter" else "max_new_tokens"
            decoding[token_field] = getattr(model, "max_new_tokens_llm_rationale", None)
        elif task == "decision":
            decoding["max_input_tokens"] = getattr(model, "max_total_tokens_llm_decision", None)
            if backend == "openrouter":
                decoding.update(
                    {
                        "max_tokens": 1,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "logprobs": True,
                        "top_logprobs": 20,
                        "logit_bias_magnitude": getattr(model, "hosted_decision_logit_bias", None),
                        "scoring_mode": "chat_logprobs_binary_head",
                    }
                )
            else:
                decoding["scoring_mode"] = "local_next_token_logits"

        if task in {"summary", "rationale"}:
            decoding["temperature"] = getattr(model, "llm_temperature", None)
            decoding["top_p"] = getattr(model, "llm_top_p", None)
            if backend != "openrouter":
                decoding["do_sample"] = getattr(model, "llm_do_sample", None)
        return {key: value for key, value in decoding.items() if value is not None}

    def _enrich_llm_backend_metadata(
        self, task: str, metadata: Mapping[str, Any]
    ) -> Dict[str, Any]:
        enriched = dict(metadata)
        backend = enriched.get("backend")
        effective_model = enriched.get("effective_model")
        if effective_model is None:
            effective_model = enriched.get("resolved_model")
        if effective_model is None:
            effective_model = enriched.get("model")
        if effective_model is not None:
            enriched["effective_model"] = effective_model

        router = getattr(self.model, "_llm_router", None)
        profiles = getattr(router, "profiles", {}) if router is not None else {}
        routing = getattr(router, "routing", None) if router is not None else None

        requested_profile = enriched.get("requested_profile")
        profile_for_task = getattr(routing, "profile_for_task", None)
        if requested_profile is None and callable(profile_for_task):
            requested_profile = profile_for_task(task)
        if requested_profile is not None:
            enriched["requested_profile"] = requested_profile
        requested_profile_obj = (
            profiles.get(str(requested_profile))
            if isinstance(profiles, Mapping) and requested_profile is not None
            else None
        )
        if enriched.get("requested_model") is None and requested_profile_obj is not None:
            enriched["requested_model"] = getattr(requested_profile_obj, "model", None)
        if enriched.get("requested_model") is None and not bool(
            enriched.get("fallback_triggered", False)
        ):
            enriched["requested_model"] = effective_model

        effective_profile = enriched.get("effective_profile") or enriched.get("profile")
        effective_profile_obj = (
            profiles.get(str(effective_profile))
            if isinstance(profiles, Mapping) and effective_profile is not None
            else None
        )
        if not self._llm_profile_matches(effective_profile_obj, backend, effective_model):
            candidate_names: List[str] = []
            fallback_for_task = getattr(routing, "fallback_for_task", None)
            if callable(fallback_for_task):
                fallback_name = fallback_for_task(task)
                if fallback_name:
                    candidate_names.append(str(fallback_name))
            local_name = getattr(self.model, "_local_llm_profile_name", None)
            if local_name:
                candidate_names.append(str(local_name))
            if isinstance(profiles, Mapping):
                candidate_names.extend(sorted(str(name) for name in profiles))
            for candidate_name in dict.fromkeys(candidate_names):
                candidate_profile = profiles.get(candidate_name)
                if self._llm_profile_matches(candidate_profile, backend, effective_model):
                    effective_profile = candidate_name
                    effective_profile_obj = candidate_profile
                    break
        if effective_profile is not None:
            enriched["profile"] = effective_profile

        aliases = {
            "requested_revision": ("requested_model_revision", "request_revision"),
            "resolved_revision": (
                "effective_revision",
                "resolved_model_revision",
                "model_revision",
                "revision",
            ),
            "endpoint_identity": ("runtime_endpoint_identity", "endpoint_id"),
            "request_time": ("requested_at", "request_timestamp", "timestamp"),
        }
        for target, candidates in aliases.items():
            if enriched.get(target) is not None:
                continue
            for candidate in candidates:
                if enriched.get(candidate) is not None:
                    enriched[target] = enriched[candidate]
                    break

        if enriched.get("requested_revision") is None and requested_profile_obj is not None:
            enriched["requested_revision"] = getattr(
                requested_profile_obj,
                "requested_revision",
                getattr(
                    requested_profile_obj,
                    "model_revision",
                    getattr(requested_profile_obj, "revision", None),
                ),
            )
        if enriched.get("requested_revision") is None:
            enriched["requested_revision"] = self._llm_model_revision(
                enriched.get("requested_model")
            )
        if (
            enriched.get("resolved_revision") is None
            and backend != "openrouter"
            and effective_profile_obj is not None
        ):
            enriched["resolved_revision"] = getattr(
                effective_profile_obj,
                "resolved_revision",
                getattr(
                    effective_profile_obj,
                    "model_revision",
                    getattr(effective_profile_obj, "revision", None),
                ),
            )
        if enriched.get("resolved_revision") is None and backend != "openrouter":
            enriched["resolved_revision"] = self._llm_model_revision(effective_model)
        if enriched.get("endpoint_identity") is None and effective_profile_obj is not None:
            enriched["endpoint_identity"] = getattr(
                effective_profile_obj, "endpoint_identity", None
            )

        if enriched.get("tokenizer") is None and effective_profile_obj is not None:
            enriched["tokenizer"] = getattr(effective_profile_obj, "tokenizer", None)
        if enriched.get("tokenizer") is None and backend == "local_hf":
            enriched["tokenizer"] = effective_model
        if enriched.get("tokenizer_revision") is None and effective_profile_obj is not None:
            enriched["tokenizer_revision"] = getattr(
                effective_profile_obj, "tokenizer_revision", None
            )
        if (
            enriched.get("tokenizer_revision") is None
            and backend == "local_hf"
            and enriched.get("tokenizer") == effective_model
        ):
            enriched["tokenizer_revision"] = enriched.get("resolved_revision")

        endpoint = enriched.get("endpoint")
        if backend == "openrouter" and effective_profile_obj is not None:
            base = self._safe_llm_endpoint(getattr(effective_profile_obj, "api_base", None))
            relative = self._safe_llm_endpoint(endpoint or "chat/completions")
            if relative and base and not urlsplit(relative).scheme:
                endpoint = f"{base.rstrip('/')}/{relative.lstrip('/')}"
            elif relative:
                endpoint = relative
        elif backend != "openrouter":
            endpoint = None
        enriched["endpoint"] = endpoint

        if enriched.get("request_seed") is None:
            enriched["request_seed"] = getattr(self.model, "request_seed", None)
        decoding = self._llm_decoding_defaults(task, str(backend) if backend is not None else None)
        supplied_decoding = enriched.get("decoding") or enriched.get("decoding_settings")
        if isinstance(supplied_decoding, Mapping):
            decoding.update(supplied_decoding)
        scoring_mode = enriched.get("decision_scoring_mode")
        if scoring_mode is not None:
            decoding["scoring_mode"] = scoring_mode
        if (
            task == "decision"
            and backend != "openrouter"
            and bool(enriched.get("fallback_triggered", False))
        ):
            decoding["scoring_mode"] = "local_next_token_logits"
        if decoding:
            enriched["decoding"] = decoding
        return enriched

    @classmethod
    def _sanitize_llm_backend_identity(
        cls, metadata: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        identity: Dict[str, Any] = {}
        for field in (
            "backend",
            "profile",
            "requested_profile",
            "requested_model",
            "requested_revision",
            "effective_model",
            "resolved_revision",
            "tokenizer",
            "tokenizer_revision",
            "request_time",
        ):
            value = cls._safe_llm_identity_text(metadata.get(field))
            if value is not None and "://" in value:
                value = cls._safe_llm_endpoint(value)
            if value is not None:
                identity[field] = value

        provider = cls._safe_llm_provider(metadata.get("provider"))
        if provider is not None and "://" in provider:
            provider = cls._safe_llm_endpoint(provider)
        if provider is not None:
            identity["provider"] = provider
        endpoint = cls._safe_llm_endpoint(metadata.get("endpoint"))
        if endpoint is not None:
            identity["endpoint"] = endpoint
        endpoint_identity = cls._safe_llm_endpoint_identity(metadata.get("endpoint_identity"))
        if endpoint_identity is not None:
            identity["endpoint_identity"] = endpoint_identity

        request_seed = metadata.get("request_seed", metadata.get("seed"))
        if isinstance(request_seed, torch.Tensor) and request_seed.numel() == 1:
            request_seed = request_seed.item()
        if isinstance(request_seed, int) and not isinstance(request_seed, bool):
            identity["request_seed"] = int(request_seed)
        if isinstance(metadata.get("fallback_triggered"), bool):
            identity["fallback_triggered"] = metadata["fallback_triggered"]

        decoding_source = metadata.get("decoding") or metadata.get("decoding_settings")
        decoding: Dict[str, Any] = {}
        if isinstance(decoding_source, Mapping):
            float_fields = {"temperature", "top_p", "logit_bias_magnitude"}
            int_fields = {
                "max_tokens",
                "max_new_tokens",
                "max_input_tokens",
                "top_logprobs",
            }
            bool_fields = {"do_sample", "logprobs"}
            for field in sorted(float_fields | int_fields | bool_fields | {"scoring_mode"}):
                value = decoding_source.get(field)
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    value = value.item()
                if (
                    field in float_fields
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    number = float(value)
                    if math.isfinite(number):
                        decoding[field] = number
                elif field in int_fields and isinstance(value, int) and not isinstance(value, bool):
                    decoding[field] = int(value)
                elif field in bool_fields and isinstance(value, bool):
                    decoding[field] = value
                elif field == "scoring_mode":
                    text = cls._safe_llm_identity_text(value)
                    if text is not None:
                        decoding[field] = text
        if decoding:
            identity["decoding"] = decoding

        prompt_hashes: Set[str] = set()
        cache_hashes: Set[str] = set()

        def _collect_hashes(target: Set[str], value: Any) -> None:
            values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
            for item in values:
                safe_hash = cls._safe_llm_hash(item)
                if safe_hash is not None:
                    target.add(safe_hash)

        for field in ("prompt_hash", "prompt_sha1", "prompt_sha256", "prompt_hashes"):
            _collect_hashes(prompt_hashes, metadata.get(field))
        for field in ("cache_hash", "cache_key_hash", "cache_hashes", "cache_identity"):
            _collect_hashes(cache_hashes, metadata.get(field))
        request_debug = metadata.get("request_debug")
        if isinstance(request_debug, Mapping):
            for field in ("prompt_hash", "prompt_sha1", "prompt_sha256"):
                _collect_hashes(prompt_hashes, request_debug.get(field))
        if prompt_hashes:
            identity["prompt_hashes"] = sorted(prompt_hashes)
        if cache_hashes:
            identity["cache_hashes"] = sorted(cache_hashes)

        backend = identity.get("backend")
        has_runtime_identity = any(
            identity.get(field)
            for field in (
                "profile",
                "requested_model",
                "effective_model",
                "provider",
                "endpoint",
                "endpoint_identity",
                "tokenizer",
            )
        )
        if backend in {None, "none"} or not has_runtime_identity:
            return None
        return identity

    def _record_llm_backend_usage(
        self, backend_usage: Any, *, runtime_observation: bool = True
    ) -> None:
        if not isinstance(backend_usage, Mapping):
            return
        for task in ("summary", "decision", "rationale"):
            metadata = backend_usage.get(task)
            if not isinstance(metadata, Mapping) or not metadata:
                continue
            enriched = self._enrich_llm_backend_metadata(task, metadata)
            identity = self._sanitize_llm_backend_identity(enriched)
            if identity is not None:
                observation: Dict[str, Any] = {"task": task, **identity}
                if runtime_observation:
                    observation["_runtime_observation"] = True
                    observation["_usage_counters"] = self._llm_usage_counters(enriched)
                self._llm_backend_observations.append(observation)

    @staticmethod
    def _llm_counter_summary(observations: List[Mapping[str, int]], field: str) -> Dict[str, Any]:
        values = [int(observation[field]) for observation in observations if field in observation]
        complete = bool(observations) and len(values) == len(observations)
        summary: Dict[str, Any] = {
            "status": "available" if complete else "unavailable",
            "observation_count": len(observations),
            "reported_observations": len(values),
        }
        if complete:
            summary["value"] = sum(values)
        elif values:
            summary["reported_value"] = sum(values)
        return summary

    def _llm_backend_usage_stats(self) -> Optional[Dict[str, Any]]:
        task_buckets: Dict[str, Dict[str, Dict[str, Any]]] = {
            "summary": {},
            "decision": {},
            "rationale": {},
        }
        runtime_counters: Dict[str, List[Mapping[str, int]]] = {
            "summary": [],
            "decision": [],
            "rationale": [],
        }
        for observation in self._llm_backend_observations:
            task = observation.get("task")
            if task not in task_buckets:
                continue
            if observation.get("_runtime_observation") is True:
                counters = observation.get("_usage_counters")
                runtime_counters[task].append(counters if isinstance(counters, Mapping) else {})
            core = {
                key: value
                for key, value in observation.items()
                if key
                not in {
                    "task",
                    "prompt_hashes",
                    "cache_hashes",
                    "_runtime_observation",
                    "_usage_counters",
                }
            }
            canonical = json.dumps(
                core,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            record = task_buckets[task].setdefault(canonical, dict(core))
            for hash_field in ("prompt_hashes", "cache_hashes"):
                hashes = observation.get(hash_field) or []
                if hashes:
                    record[hash_field] = sorted(set(record.get(hash_field, [])) | set(hashes))

        identities = {
            task: [bucket[key] for key in sorted(bucket)]
            for task, bucket in task_buckets.items()
            if bucket
        }
        if not identities:
            return None

        counter_fields = ("calls", "input_tokens", "output_tokens", "total_tokens")
        by_task: Dict[str, Dict[str, Any]] = {}
        for task in identities:
            observations = runtime_counters[task]
            availability = {
                field: self._llm_counter_summary(observations, field) for field in counter_fields
            }
            task_usage: Dict[str, Any] = {
                "observation_count": len(observations),
                "counter_availability": availability,
            }
            for field, report in availability.items():
                if report["status"] == "available":
                    task_usage[field] = report["value"]
            by_task[task] = task_usage

        all_runtime = [
            observation
            for task in ("summary", "decision", "rationale")
            for observation in runtime_counters[task]
        ]
        availability = {
            field: self._llm_counter_summary(all_runtime, field) for field in counter_fields
        }
        usage: Dict[str, Any] = {
            "backend_identities": identities,
            "by_task": by_task,
            "counter_availability": availability,
        }
        for field, report in availability.items():
            if report["status"] == "available":
                usage[field] = report["value"]
        return usage

    def _compute_run_stats(
        self,
        df: pd.DataFrame,
        review_low: Optional[float] = None,
        review_high: Optional[float] = None,
    ) -> Dict[str, Any]:
        for record in self.results_json or []:
            if isinstance(record, Mapping):
                self._record_llm_backend_usage(
                    record.get("backend_usage"), runtime_observation=False
                )
        stats = super()._compute_run_stats(
            df,
            review_low=review_low,
            review_high=review_high,
        )
        llm_usage = self._llm_backend_usage_stats()
        if llm_usage is not None:
            stats["llm_usage"] = llm_usage
        reconstruction_rows = 0
        reconstruction_errors: List[float] = []
        for record in self.results_json or []:
            if not isinstance(record, Mapping):
                continue
            contributions = record.get("contributions")
            confidences = record.get("confidences")
            if not isinstance(contributions, Mapping) or not isinstance(confidences, Mapping):
                continue
            replay = record.get("reconstruction") or {}
            score = replay.get("score", confidences.get("S_final"))
            if score is None:
                continue
            names = ("C_label", "C_strsim", "C_struct", "C_llm", "C_oracle")
            if any(name not in contributions for name in names[:-1]):
                continue
            reconstructed = sum(float(contributions.get(name, 0.0)) for name in names)
            tau = float(replay.get("baseline", getattr(self.model, "tau", 0.5)))
            error = abs(reconstructed - (float(score) - tau))
            if not math.isfinite(error):
                raise ValueError("explanation reconstruction produced a non-finite error")
            reconstruction_rows += 1
            reconstruction_errors.append(error)
        result_count = len(self.results_json or [])
        stats["explanation_reconstruction"] = {
            "status": (
                "complete"
                if result_count > 0 and reconstruction_rows == result_count
                else "unavailable" if reconstruction_rows == 0 else "partial"
            ),
            "result_rows": result_count,
            "reconstructed_rows": reconstruction_rows,
            "score_stage": "pair_pre_selector",
            "tolerance": 1e-6,
            "failed_rows": sum(error > 1e-6 for error in reconstruction_errors),
            "max_abs_error": max(reconstruction_errors) if reconstruction_errors else None,
        }
        source_column = next(
            (column for column in ("src_iri", "Src") if column in df.columns),
            None,
        )
        if source_column is not None and "saved_alignment_member" in df.columns:
            kind_column = next(
                (column for column in ("src_kind", "SrcKind") if column in df.columns),
                None,
            )
            identity_columns = [source_column]
            if kind_column is not None:
                identity_columns.append(kind_column)
            observed_groups = df[identity_columns].dropna(subset=[source_column]).copy()
            observed_groups[source_column] = observed_groups[source_column].astype(str)
            if kind_column is not None:
                observed_groups[kind_column] = observed_groups[kind_column].astype(str)
            observed_count = int(observed_groups.drop_duplicates().shape[0])
            accepted = df["saved_alignment_member"].fillna(False).astype(bool)
            accepted_groups = (
                df.loc[accepted, identity_columns].dropna(subset=[source_column]).copy()
            )
            accepted_groups[source_column] = accepted_groups[source_column].astype(str)
            if kind_column is not None:
                accepted_groups[kind_column] = accepted_groups[kind_column].astype(str)
            accepted_count = int(accepted_groups.drop_duplicates().shape[0])

            pool_manifest = getattr(self.dataset, "candidate_pool_manifest", None)
            pool_summary = (
                pool_manifest.get("gold_free_summary")
                if isinstance(pool_manifest, Mapping)
                else None
            )
            declared_count: Optional[int] = None
            if (
                isinstance(pool_summary, Mapping)
                and pool_summary.get("source_entities") is not None
            ):
                declared_count = int(pool_summary["source_entities"])
                if declared_count < 0:
                    raise ValueError("candidate-pool source_entities cannot be negative")
            source_count = max(observed_count, declared_count or 0)
            if source_count > 0:
                stats["coverage"] = accepted_count / source_count
                stats["abstention_rate"] = 1.0 - stats["coverage"]
                stats["decision_source_counts"] = {
                    "declared": declared_count,
                    "observed": observed_count,
                    "accepted": accepted_count,
                    "unscored": max(0, source_count - observed_count),
                }
                applicability = dict(stats.get("metric_applicability") or {})
                applicability["coverage"] = declared_count is not None
                applicability["abstention_rate"] = declared_count is not None
                stats["metric_applicability"] = applicability
        observed_device = self.device
        observed_execution: Dict[str, Any] = {
            "device_type": observed_device.type,
            "device": str(observed_device),
        }
        if observed_device.type == "cuda":
            try:
                observed_execution["device_name"] = torch.cuda.get_device_name(observed_device)
            except (AssertionError, RuntimeError, ValueError):
                pass
        stats["observed_execution"] = observed_execution
        return stats

    def _finalize_run(self, state: _FinalizationState) -> Tuple[List[EntityMapping], float]:
        """Apply the shared post-inference path for fresh and completed checkpoints."""
        post_inference_start = time.perf_counter()
        candidate_df = self._build_candidate_dataframe()
        if state.completed_from_checkpoint and candidate_df.empty and self.results_json:
            candidate_df = self._build_candidate_dataframe_from_records(self.results_json)

        post_progress_started = False
        if state.run_progress is not None and len(getattr(self, "models", [])) > 1:
            n_sources = (
                int(candidate_df["Src"].nunique())
                if not candidate_df.empty and "Src" in candidate_df.columns
                else 0
            )
            detail = (
                f"rows={len(candidate_df)}, sources={n_sources}"
                if state.completed_from_checkpoint
                else "assembling candidate dataframe"
            )
            state.run_progress.start("PostInference", detail)
            post_progress_started = True

        if state.completed_from_checkpoint:
            if not candidate_df.empty:
                n_sources = (
                    int(candidate_df["Src"].nunique()) if "Src" in candidate_df.columns else 0
                )
                self.log(
                    (
                        "Post-inference processing restored candidate scores: "
                        f"rows={len(candidate_df)}, sources={n_sources}."
                    ),
                    "debug",
                )
        else:
            self.log(
                "Post-inference processing started: assembling candidate dataframe.",
                "info",
            )

        if not candidate_df.empty:
            n_sources = int(candidate_df["Src"].nunique()) if "Src" in candidate_df.columns else 0
            if post_progress_started and not state.completed_from_checkpoint:
                state.run_progress.update(
                    "PostInference",
                    fraction=0.15,
                    detail=f"rows={len(candidate_df)}, sources={n_sources}",
                    force=True,
                )
            if not state.completed_from_checkpoint:
                self.log(
                    (
                        "Candidate dataframe assembled for post-inference processing: "
                        f"rows={len(candidate_df)}, sources={n_sources}."
                    ),
                    "debug",
                )
            checkpointed_df = self._load_additional_models_checkpoint(
                state.kind,
                state.local_alignment,
                state.threshold,
                state.cardinality,
            )
            if checkpointed_df is not None:
                candidate_df = checkpointed_df
            else:
                candidate_df = self._apply_additional_models(
                    candidate_df,
                    kind=state.kind,
                    local_alignment=state.local_alignment,
                    threshold=state.threshold,
                    cardinality=state.cardinality,
                    target_cardinality=state.target_cardinality,
                    results_json=self.results_json,
                    log_every=state.log_every,
                    run_progress=state.run_progress,
                )
                self._write_additional_models_checkpoint(
                    state.kind,
                    candidate_df,
                    state.local_alignment,
                    state.threshold,
                    state.cardinality,
                )
            if "llm_source_choice" in candidate_df:
                for source, group in candidate_df.groupby("Src", sort=False):
                    choices = group["llm_source_choice"].dropna().astype(str).unique()
                    if not len(choices):
                        continue
                    if len(choices) != 1:
                        raise ValueError(f"Conflicting comparative choices for {source}")
                    choice = choices[0]
                    candidate_df.loc[group.index, "S_before_source_choice"] = candidate_df.loc[
                        group.index, "S_final"
                    ]
                    rejected = group.index[group["Tgt"].astype(str) != choice]
                    candidate_df.loc[rejected, "S_final"] = 0.0
                    candidate_df.loc[group.index, "source_choice_reason"] = (
                        "displayed_none" if choice == "__NONE__" else "comparative_choice"
                    )
            state.all_mappings = list(
                zip(candidate_df["Src"], candidate_df["Tgt"], candidate_df["S_final"])
            )
            self._selector_target_conflict_enabled = self._selector_target_conflict_enabled_from_df(
                candidate_df
            )
            self._last_effective_threshold_origin = self._effective_alignment_threshold_origin(
                candidate_df, state.threshold
            )
            state.threshold = self._effective_alignment_threshold(candidate_df, state.threshold)
            self._last_effective_threshold = state.threshold

            if not state.completed_from_checkpoint:
                if self.results_json:
                    self.results_df = self._make_summary_dataframe(self.results_json)
                else:
                    self.results_df = candidate_df
        elif not state.completed_from_checkpoint:
            self.results_df = self._make_summary_dataframe(self.results_json)

        if post_progress_started:
            state.run_progress.finish("PostInference", f"rows={len(candidate_df)}")

        score_frame = pd.DataFrame(state.all_mappings, columns=["Src", "Tgt", "Scores"])
        kind_columns = ["SrcKind", "TgtKind"]
        if not candidate_df.empty and all(
            column in candidate_df.columns for column in ["Src", "Tgt", "S_final", *kind_columns]
        ):
            # Checkpoints intentionally retain their historical three-column mapping payload.
            # The candidate frame is the kind-aware source of truth for fresh/migrated runs,
            # so use it when constructing the in-memory mappings passed to typed writers.
            score_frame = candidate_df[["Src", "Tgt", "S_final", *kind_columns]].rename(
                columns={"S_final": "Scores"}
            )
        self._decision_policy = {
            "threshold": state.threshold,
            "threshold_origin": getattr(self, "_last_effective_threshold_origin", "configured"),
            "source_cardinality": state.cardinality,
            "target_cardinality": (
                None
                if getattr(self, "_selector_target_conflict_enabled", None) is False
                else state.target_cardinality
            ),
            "local_alignment": state.local_alignment,
            "extraction": dict(self._extraction_config),
        }
        extraction_mode = str(self._extraction_config.get("mode", "greedy"))
        if extraction_mode == "greedy":
            # Preserve the shipped threshold-then-source-greedy path byte for byte.
            predictions = EntityMapping.read_table_mappings(
                score_frame,
                threshold=state.threshold,
                cardinality=state.cardinality,
            )
        else:
            if state.local_alignment:
                raise ValueError(
                    "non-greedy global extraction cannot be used for local candidate ranking"
                )
            if state.cardinality not in {None, 1} or state.target_cardinality not in {None, 1}:
                raise ValueError(
                    f"matching.extraction.mode={extraction_mode!r} requires one-to-one "
                    "source and target cardinality"
                )
            scored_mappings = EntityMapping.read_table_mappings(score_frame)
            prefilter_frame = self.dataset.dataframe
            prefiltered_mappings: List[EntityMapping] = []
            if (
                prefilter_frame is not None
                and not prefilter_frame.empty
                and DatasetMask.prefiltered in prefilter_frame.columns
            ):
                exact_frame = prefilter_frame[prefilter_frame[DatasetMask.prefiltered]].copy()
                score_column = next(
                    (name for name in ("Scores", "Score") if name in exact_frame.columns),
                    None,
                )
                if score_column is not None and not exact_frame.empty:
                    columns = ["Src", "Tgt", score_column]
                    for kind_column in ("SrcKind", "TgtKind"):
                        if kind_column in exact_frame.columns:
                            columns.append(kind_column)
                    exact_frame = exact_frame[columns].rename(columns={score_column: "Score"})
                    prefiltered_mappings = EntityMapping.read_table_mappings(exact_frame)
            protected_pairs = {
                (str(mapping.head), str(mapping.tail)) for mapping in prefiltered_mappings
            }
            extraction = extract_global_alignment(
                [*prefiltered_mappings, *scored_mappings],
                mode=extraction_mode,
                threshold=state.threshold,
                protected_pairs=protected_pairs,
                source_cardinality=state.cardinality,
                target_cardinality=state.target_cardinality,
                assignment_component_cap=int(
                    self._extraction_config.get("assignment_component_cap", 500)
                ),
            )
            predictions = extraction.mappings
            self._extraction_diagnostics = dict(extraction.diagnostics)
            self._extraction_includes_prefilter = bool(prefiltered_mappings)
            if not candidate_df.empty:
                selected_pairs = {(str(mapping.head), str(mapping.tail)) for mapping in predictions}
                candidate_df["extraction_selected"] = [
                    (str(source), str(target)) in selected_pairs
                    for source, target in candidate_df[["Src", "Tgt"]].itertuples(
                        index=False, name=None
                    )
                ]
        if not candidate_df.empty:
            from exact.runs.decisions import observe_extraction

            observe_extraction(
                candidate_df,
                predictions,
                threshold=state.threshold,
                config=self._decision_policy,
                implementation="exact.impl.trainer.runner:" + extraction_mode,
            )
        self._final_candidate_frame = candidate_df.copy()
        nil_models = [
            model
            for model in self.models[1:]
            if getattr(model, "nil_config", {}).get("mode", "off") != "off"
        ]
        if nil_models:
            from exact.impl.models.selector.nil_head import (
                source_decision_records,
                validate_nil_application,
            )

            nil_config = nil_models[-1].nil_config
            artifact = (
                json.loads(Path(nil_config["artifact"]).read_text())
                if nil_config.get("mode") == "fitted"
                else None
            )
            if artifact:
                validate_nil_application(self.dataset, nil_config["artifact"], artifact, nil_config)
            universe = getattr(self.dataset, "eligible_source_iris", None) or sorted(
                set(candidate_df.Src.astype(str))
            )
            nil_frame = (
                candidate_df
                if "Src" in candidate_df
                else pd.DataFrame(columns=["Src", "Tgt", "S_base"])
            )
            records = source_decision_records(nil_frame, universe, artifact=artifact)
            emitted = {}
            for mapping in predictions:
                emitted.setdefault(str(mapping.head), []).append(str(mapping.tail))
            for record in records:
                record["emitted_targets"] = sorted(emitted.get(record["Src"], []))
                record["action"] = "emit" if record["emitted_targets"] else "abstain"
            destination = self.output_dir / "source_decisions.json"
            temporary = destination.with_suffix(".json.partial")
            temporary.write_text(
                json.dumps(
                    {"schema_version": 1, "mode": nil_config["mode"], "records": records},
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            temporary.replace(destination)
            self._nil_source_output = {"path": str(destination), "source_count": len(records)}
        compact_rationale_records = False
        if not candidate_df.empty:
            self._annotate_candidate_dataframe(
                candidate_df,
                predictions,
                state.threshold,
                state.local_alignment,
            )
            compact_rationale_records = self._ensure_compact_rationale_records(candidate_df)
        self._annotate_final_prediction_records(
            predictions,
            threshold=state.threshold,
            local_alignment=state.local_alignment,
        )

        rationale_enabled = self._should_generate_final_rationales()
        rationale_start = time.perf_counter()
        self._generate_final_rationales(
            log_every=state.log_every,
            kind=state.kind,
            local_alignment=state.local_alignment,
            threshold=state.threshold,
            cardinality=state.cardinality,
        )
        rationale_meta = getattr(self.model, "_last_rationale_backend_meta", {}) or {}
        if rationale_enabled and isinstance(rationale_meta, Mapping) and rationale_meta:
            self._record_llm_backend_usage({"rationale": rationale_meta})
        rationale_elapsed = time.perf_counter() - rationale_start
        post_inference_elapsed = max(
            0.0,
            time.perf_counter() - post_inference_start - rationale_elapsed,
        )

        if self.results_json and not compact_rationale_records:
            self.results_df = self._make_summary_dataframe(self.results_json)
            if not state.completed_from_checkpoint:
                for record in self.results_json:
                    self._sync_record_model_usage(record)
        elif not candidate_df.empty:
            self.results_df = candidate_df

        if (
            not candidate_df.empty
            and state.checkpoint_enabled
            and state.checkpoint_path is not None
        ):
            self._write_final_overlay(
                state.checkpoint_path,
                candidate_df,
                predictions,
                state.threshold,
                state.local_alignment,
                compression=state.audit_shard_compression,
                records_per_shard=state.audit_shard_records,
            )
        if state.checkpoint_enabled and state.checkpoint_path is not None:
            self._write_checkpoint_state(
                state.checkpoint_path,
                state.kind,
                total_examples=state.total_examples,
                processed_examples=state.processed_examples,
                mappings=state.all_mappings,
                results_json=self.results_json,
            )

        explanation_store = getattr(self, "_explanation_store", None)
        if explanation_store is not None and explanation_store.overlay_count:
            compaction = explanation_store.compact()
            self._overlay_manifest_path = None
            self.log(
                (
                    "Compacted transient explanation overlays: "
                    f"records={compaction['records']}, "
                    f"before={compaction['before_bytes']} bytes, "
                    f"after={compaction['after_bytes']} bytes."
                ),
                "info",
            )

        self._last_stage_timings = [
            StageRecord(
                stage="Alignment.Fitting",
                seconds=getattr(self, "_fitting_seconds", 0.0),
                cache_status=CacheStatus.FRESH,
            ),
            StageRecord(
                stage="Alignment.Inference",
                seconds=state.inference_seconds,
                cache_status=state.inference_status,
                work_done=state.inference_work_done,
                work_total=state.total_examples,
                unit="examples",
            ),
            StageRecord(
                stage="Alignment.PostInference",
                seconds=post_inference_elapsed,
                cache_status=state.post_inference_status,
            ),
            StageRecord(
                stage="Postprocess.Rationales",
                seconds=rationale_elapsed,
                cache_status=CacheStatus.FRESH,
            ),
        ]
        return predictions, state.average_seconds_per_example

    @torch.no_grad()
    def predict(
        self,
        kind: DatasetMask = DatasetMask.inference,
        threshold: Optional[float] = 0.7,
        cardinality: Optional[int] = None,
        target_cardinality: Optional[int] = None,
        local_alignment: bool = False,
        batch_size: int = 8,
        num_workers: int = 0,
        log_every: int = 1,
        mixed_precision: bool = False,
        checkpoint_file: Optional[str] = None,
        checkpoint_every: int = 10,
        resume_from_checkpoint: bool = True,
        enable_checkpoints: bool = True,
        resume_additional_model_checkpoints: bool = True,
        allow_rationale_toggle_checkpoint_resume: bool = False,
        audit_shards_enabled: bool = True,
        audit_shard_compression: str = "zstd",
        audit_shard_records: int = 50000,
        explanation_shard_mb: float = 32.0,
        checkpoint_payload: str = "compact",
        cache_persist_policy: str = "finalize",
        save_json: bool = False,
        run_progress: Optional[Any] = None,
        **kwargs,
    ) -> Tuple[List[EntityMapping], float]:
        if os.getenv("EXACT_EXPERIMENT_RUNTIME") and not enable_checkpoints:
            raise ValueError("Experiment runtime requires checkpoints for bounded stop/recovery")
        stop_path = (
            Path(os.environ["EXACT_EXPERIMENT_STOP_FILE"])
            if os.getenv("EXACT_EXPERIMENT_STOP_FILE")
            else None
        )
        if stop_path is not None:
            signal.signal(signal.SIGTERM, lambda *_: stop_path.touch())
        self.dataset.default_kind = kind
        fitting_started = time.perf_counter()
        self.fit_training_pool(batch_size=batch_size)
        self._fitting_seconds = time.perf_counter() - fitting_started
        grouped_decisions = bool(getattr(self.model, "llm_experiment_enabled", False)) and (
            getattr(self.model, "llm_experiment_config", {})
            .get("decision", {})
            .get("mode", "binary")
            != "binary"
            or bool(self.fitting_gate_config)
            or getattr(self.model, "llm_experiment_config", {}).get("gate", {}).get("mode")
            == "learned"
        )
        grouped_decisions = grouped_decisions or (
            getattr(self.model, "lex_enabled", False)
            and getattr(self.model, "lex_config", {}).get("quality")
            in {"candidate_margin", "encoder_agreement"}
        )
        if grouped_decisions:
            columns = [
                column
                for column in ("Src", "SrcKind", "Tgt", "TgtKind")
                if column in self.dataset.dataframe
            ]
            self.dataset._df = self.dataset.dataframe.sort_values(
                columns, kind="stable"
            ).reset_index(drop=True)
            self.dataset._invalidate_active_dataframe_cache()
        self.prepare_anchors(batch_size=batch_size)
        self.prepare_population_gate(batch_size=batch_size)
        self.model.eval()
        for extra_model in getattr(self, "models", [])[1:]:
            try:
                extra_model.eval()
            except Exception:
                pass
        self.results_json.clear()
        self.results_df = None
        self._llm_summary_stats: Optional[Dict[str, Any]] = None
        self._llm_backend_observations = []
        self._llm_calibration_messages_logged: Set[str] = set()
        self._llm_calibration_report: Optional[Dict[str, Any]] = None
        self._candidate_rows: List[Dict[str, Any]] = []
        self._last_stage_timings = []
        self._inference_seconds_cumulative = 0.0
        self._restored_inference_seconds_cumulative = 0.0
        self._inference_session_started_at = None
        self._examples_per_second_ema = None
        self._last_effective_threshold = threshold
        self._last_effective_threshold_origin = "configured"
        self._restored_candidate_rows = []
        self._overlay_manifest_path = None
        if hasattr(self.model, "reset_llm_calibration_tracking"):
            self.model.reset_llm_calibration_tracking()
        if hasattr(self.model, "reset_summary_stats"):
            self.model.reset_summary_stats()
        if hasattr(self.model, "use_llm_calibration"):
            self._llm_calibration_report = {
                "configured": {
                    "enabled": bool(getattr(self.model, "use_llm_calibration", False)),
                    "a": getattr(self.model, "llm_calibration_a", None),
                    "b": getattr(self.model, "llm_calibration_b", None),
                    "info": getattr(self.model, "llm_calibration_info", None),
                },
                "learned": None,
                "messages": [],
                "pending_total_samples": 0,
            }

        checkpoint_enabled = enable_checkpoints
        self._checkpoint_payload_mode = str(checkpoint_payload or "compact").lower()
        if self._checkpoint_payload_mode not in {"compact", "full"}:
            self._checkpoint_payload_mode = "compact"
        self._cache_persist_policy = str(cache_persist_policy or "finalize").lower()
        if self._cache_persist_policy not in {"checkpoint", "finalize", "never"}:
            self._cache_persist_policy = "finalize"
        self._cache_persist_skip_logged = set()
        self._explanation_shard_mb = max(0.001, float(explanation_shard_mb))
        self._prepare_audit_shards(
            None,
            enabled=True,
            compression=audit_shard_compression,
            records_per_shard=audit_shard_records,
            append_existing=True,
        )
        self._prepare_candidate_shards(
            None,
            enabled=False,
            compression=audit_shard_compression,
            records_per_shard=audit_shard_records,
        )
        self._postprocess_checkpoints_enabled = bool(checkpoint_enabled)
        self._additional_model_checkpoint_resume_enabled = bool(
            checkpoint_enabled and resume_additional_model_checkpoints
        )
        self._additional_model_checkpoint_skip_logged = False
        checkpoint_every = max(1, int(checkpoint_every))

        cp_path: Optional[Path] = None
        restored_examples = 0
        all_mappings: List[Tuple[str, str, float]] = []
        restored_json: List[Dict[str, Any]] = []

        if checkpoint_enabled and resume_from_checkpoint:
            cp_path, restored_mappings, restored_json, restored_examples = (
                self._restore_from_available_checkpoints(
                    kind,
                    checkpoint_file,
                    allow_rationale_toggle_checkpoint_resume=allow_rationale_toggle_checkpoint_resume,
                )
            )
            if restored_examples and cp_path:
                self.log(
                    (
                        f"Resuming from checkpoint {cp_path} with "
                        f"{restored_examples} / {len(self.dataset)} examples already processed."
                    ),
                    level="info",
                )
            all_mappings.extend(restored_mappings)
            if restored_json:
                self.results_json.extend(restored_json)
            restored_candidate_rows = getattr(self, "_restored_candidate_rows", []) or []
            if restored_candidate_rows:
                self._candidate_rows.extend(restored_candidate_rows)

        total_examples = len(self.dataset)
        remaining_examples = max(0, total_examples - restored_examples)

        if restored_examples == 0:
            self._prepare_audit_shards(
                cp_path,
                enabled=True,
                compression=audit_shard_compression,
                records_per_shard=audit_shard_records,
                append_existing=False,
            )

        if remaining_examples == 0:
            if checkpoint_enabled and cp_path is not None:
                self._prepare_audit_shards(
                    cp_path,
                    enabled=bool(audit_shards_enabled),
                    compression=audit_shard_compression,
                    records_per_shard=audit_shard_records,
                    append_existing=True,
                )
                self._prepare_candidate_shards(
                    cp_path,
                    enabled=True,
                    compression=audit_shard_compression,
                    records_per_shard=audit_shard_records,
                    append_existing=True,
                )
            self.log(
                ("Checkpoint already contains predictions for all samples. " "Skipping inference."),
                level="info",
            )
            if run_progress is not None:
                run_progress.finish("Inference", "checkpoint already complete")
            return self._finalize_run(
                _FinalizationState(
                    kind=kind,
                    threshold=threshold,
                    cardinality=cardinality,
                    target_cardinality=target_cardinality,
                    local_alignment=local_alignment,
                    all_mappings=all_mappings,
                    total_examples=total_examples,
                    processed_examples=restored_examples,
                    restored_examples=restored_examples,
                    checkpoint_enabled=checkpoint_enabled,
                    checkpoint_path=cp_path,
                    audit_shard_compression=audit_shard_compression,
                    audit_shard_records=audit_shard_records,
                    log_every=log_every,
                    run_progress=run_progress,
                    inference_seconds=0.0,
                    average_seconds_per_example=0.0,
                    inference_work_done=0,
                    inference_status=CacheStatus.SKIPPED,
                    post_inference_status=CacheStatus.RESUMED,
                    completed_from_checkpoint=True,
                )
            )

        if checkpoint_enabled:
            cp_path = self._ensure_checkpoint_path(kind, checkpoint_file, cp_path)
            if cp_path is None:
                checkpoint_enabled = False
            elif not resume_from_checkpoint and checkpoint_file:
                self.log(
                    f"Checkpoint file {cp_path} will be overwritten for this run.",
                    level="info",
                )
        if checkpoint_enabled and cp_path is not None:
            self._prepare_audit_shards(
                cp_path,
                enabled=bool(audit_shards_enabled),
                compression=audit_shard_compression,
                records_per_shard=audit_shard_records,
                append_existing=restored_examples > 0,
            )
            self._prepare_candidate_shards(
                cp_path,
                enabled=True,
                compression=audit_shard_compression,
                records_per_shard=audit_shard_records,
                append_existing=restored_examples > 0,
            )

        dataset_for_dl = self.dataset
        if restored_examples > 0:
            dataset_for_dl = Subset(self.dataset, range(restored_examples, total_examples))

        batching = {"batch_size": batch_size, "shuffle": False}
        if grouped_decisions:
            remaining_frame = (
                self.dataset._active_dataframe().iloc[restored_examples:].reset_index(drop=True)
            )
            batching = {"batch_sampler": list(source_batches(remaining_frame, int(batch_size)))}
        dl = DataLoader(
            dataset_for_dl,
            **batching,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else None,
            collate_fn=_semantic_collate_fn,
        )

        total_batches = len(dl)
        start_time = time.perf_counter()
        self._inference_session_started_at = start_time
        processed_examples = restored_examples
        batches_run = 0

        self.log(
            (
                f"Running Semantic Alignment on {remaining_examples} remaining pairs "
                f"({total_examples} total)"
            ),
            "info",
        )

        last_checkpoint_at = time.monotonic()
        for step, batch in enumerate(dl, start=1):
            src_iri = batch["src_iri"]
            tgt_iri = batch["tgt_iri"]
            src_kinds = list(batch.get("src_kind") or ["class"] * len(src_iri))
            tgt_kinds = list(batch.get("tgt_kind") or src_kinds)
            src_labels = batch["src_labels"]
            tgt_labels = batch["tgt_labels"]
            src_ctxs = batch.get("src_contexts", None)
            tgt_ctxs = batch.get("tgt_contexts", None)
            src_ctx_raw = batch.get("src_ctx_raw_triples")
            tgt_ctx_raw = batch.get("tgt_ctx_raw_triples")
            src_ctx_bridges = batch.get("src_ctx_bridge_triples")
            tgt_ctx_bridges = batch.get("tgt_ctx_bridge_triples")
            labels = batch.get("label")
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()

            with torch.amp.autocast("cuda", enabled=mixed_precision):
                out = self.model.forward(
                    src_iris=src_iri,
                    tgt_iris=tgt_iri,
                    src_label_lists=src_labels,
                    tgt_label_lists=tgt_labels,
                    src_contexts=src_ctxs,
                    tgt_contexts=tgt_ctxs,
                    src_ctx_raw=src_ctx_raw,
                    tgt_ctx_raw=tgt_ctx_raw,
                    src_ctx_bridges=src_ctx_bridges,
                    tgt_ctx_bridges=tgt_ctx_bridges,
                    label=(
                        labels
                        if getattr(self.model, "llm_experiment_config", {})
                        .get("gate", {})
                        .get("mode")
                        == "oracle"
                        else None
                    ),
                )
            self._record_llm_backend_usage(out.get("backend_usage"))
            self._record_llm_calibration(out.get("llm_calibration"))
            pair_batch_stats = out.get("batch_pair_adaptive_stats") or {}
            if step == 1 and pair_batch_stats:
                src_pool = pair_batch_stats.get("src_pool") or {}
                tgt_pool = pair_batch_stats.get("tgt_pool") or {}
                evidence = pair_batch_stats.get("pair_evidence") or {}
                self.log(
                    (
                        "Pair-adaptive pair context is assembled during inference batches. "
                        f"First batch: pairs={pair_batch_stats.get('pairs', len(src_iri))}, "
                        f"unique src/tgt={pair_batch_stats.get('unique_src', 0)}/{pair_batch_stats.get('unique_tgt', 0)}, "
                        f"entity-cache hits src={pair_batch_stats.get('src_cache_hits', 0)}/{pair_batch_stats.get('unique_src', 0)}, "
                        f"tgt={pair_batch_stats.get('tgt_cache_hits', 0)}/{pair_batch_stats.get('unique_tgt', 0)}; "
                        f"src pools h/o/a={src_pool.get('hier_nonempty', 0)}/{src_pool.get('entities', 0)},"
                        f"{src_pool.get('obj_nonempty', 0)}/{src_pool.get('entities', 0)},"
                        f"{src_pool.get('attr_nonempty', 0)}/{src_pool.get('entities', 0)}; "
                        f"tgt pools h/o/a={tgt_pool.get('hier_nonempty', 0)}/{tgt_pool.get('entities', 0)},"
                        f"{tgt_pool.get('obj_nonempty', 0)}/{tgt_pool.get('entities', 0)},"
                        f"{tgt_pool.get('attr_nonempty', 0)}/{tgt_pool.get('entities', 0)}; "
                        f"selected pair evidence hier/sim/diff/attr={evidence.get('hier_selected', 0)}/{pair_batch_stats.get('pairs', len(src_iri))},"
                        f"{evidence.get('sim_selected', 0)}/{pair_batch_stats.get('pairs', len(src_iri))},"
                        f"{evidence.get('diff_selected', 0)}/{pair_batch_stats.get('pairs', len(src_iri))},"
                        f"{evidence.get('attr_selected', 0)}/{pair_batch_stats.get('pairs', len(src_iri))}."
                    ),
                    "debug",
                )

            # Accumulate mappings
            def _tensor_list(name: str, default: float = 0.0):
                value = out.get(name)
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().tolist()
                return [default] * len(src_iri)

            s_label_vals = _tensor_list("s_label")
            s_label_star_vals = _tensor_list("s_label_star")
            s_ctx_vals = _tensor_list("s_ctx")
            s_lctx_vals = _tensor_list("S_lctx")
            s_base_vals = _tensor_list("S_base")
            s_struct_vals = _tensor_list("S_struct")
            s_hier_vals = _tensor_list("s_hier")
            s_sim_vals = _tensor_list("s_sim")
            s_diff_vals = _tensor_list("s_diff")
            s_attr_vals = _tensor_list("s_attr")
            q_label_vals = _tensor_list("q_label")
            q_struct_vals = _tensor_list("Q_struct")
            q_hier_vals = _tensor_list("q_hier")
            q_sim_vals = _tensor_list("q_sim")
            q_diff_vals = _tensor_list("q_diff")
            q_attr_vals = _tensor_list("q_attr")
            s_final = _tensor_list("S_final")
            w_c_vals = _tensor_list("w_c")
            w_struct_vals = _tensor_list("w_struct")
            w_i_vals = _tensor_list("w_i")
            u_vals = _tensor_list("U")
            u_ind_vals = _tensor_list("U_ind")
            u_dis_vals = _tensor_list("U_dis")
            p_llm_vals = _tensor_list("p_llm")
            i_label_vals = _tensor_list("I_label")
            i_struct_vals = _tensor_list("I_struct")
            i_ctx_vals = _tensor_list("I_ctx")
            i_hier_vals = _tensor_list("I_hier")
            i_sim_vals = _tensor_list("I_sim")
            i_diff_vals = _tensor_list("I_diff")
            i_attr_vals = _tensor_list("I_attr")
            i_llm_vals = _tensor_list("I_llm")
            optional_experiment_values: Dict[str, List[Any]] = {}
            for output_name in ("s_strsim", "q_lex", "q_strsim", "I_lex", "I_strsim"):
                if output_name not in out:
                    continue
                output_value = out[output_name]
                if isinstance(output_value, torch.Tensor):
                    values = output_value.detach().cpu().tolist()
                elif isinstance(output_value, (list, tuple)):
                    values = list(output_value)
                else:
                    raise TypeError(f"Scorer output {output_name!r} must be a tensor or sequence")
                if len(values) != len(src_iri):
                    raise ValueError(
                        f"Scorer output {output_name!r} has {len(values)} values for "
                        f"{len(src_iri)} pairs"
                    )
                optional_experiment_values[output_name] = values

            raw_gate_diagnostics = out.get("llm_gate_diagnostics")
            gate_diagnostics: Optional[List[Any]] = None
            if raw_gate_diagnostics is not None:
                if not isinstance(raw_gate_diagnostics, (list, tuple)):
                    raise TypeError("llm_gate_diagnostics must be a per-pair sequence")
                gate_diagnostics = list(raw_gate_diagnostics)
                if len(gate_diagnostics) != len(src_iri):
                    raise ValueError(
                        "llm_gate_diagnostics count does not match the scored pair count"
                    )
            llm_pair_briefs = list(out.get("llm_pair_briefs") or [""] * len(src_iri))
            ground_truth = labels or [None] * len(src_iri)
            candidate_start_idx = len(self._candidate_rows)

            for idx, (s, t, score) in enumerate(zip(src_iri, tgt_iri, s_final)):
                all_mappings.append((s, t, float(score)))
                candidate_row = {
                    "Src": s,
                    "Tgt": t,
                    "SrcKind": str(src_kinds[idx]),
                    "TgtKind": str(tgt_kinds[idx]),
                    "ground_truth": ground_truth[idx],
                    "s_label": float(s_label_vals[idx]),
                    "s_label_star": float(s_label_star_vals[idx]),
                    "s_ctx": float(s_ctx_vals[idx]),
                    "S_lctx": float(s_lctx_vals[idx]),
                    "S_base": float(s_base_vals[idx]),
                    "S_struct": float(s_struct_vals[idx]),
                    "s_hier": float(s_hier_vals[idx]),
                    "s_sim": float(s_sim_vals[idx]),
                    "s_diff": float(s_diff_vals[idx]),
                    "s_attr": float(s_attr_vals[idx]),
                    "q_label": float(q_label_vals[idx]),
                    "Q_struct": float(q_struct_vals[idx]),
                    "q_hier": float(q_hier_vals[idx]),
                    "q_sim": float(q_sim_vals[idx]),
                    "q_diff": float(q_diff_vals[idx]),
                    "q_attr": float(q_attr_vals[idx]),
                    "S_final": float(score),
                    "w_c": float(w_c_vals[idx]),
                    "w_struct": float(w_struct_vals[idx]),
                    "w_i": float(w_i_vals[idx]),
                    "U": float(u_vals[idx]),
                    "U_ind": float(u_ind_vals[idx]),
                    "U_dis": float(u_dis_vals[idx]),
                    "p_llm": float(p_llm_vals[idx]),
                    "I_label": float(i_label_vals[idx]),
                    "I_struct": float(i_struct_vals[idx]),
                    "I_ctx": float(i_ctx_vals[idx]),
                    "I_hier": float(i_hier_vals[idx]),
                    "I_sim": float(i_sim_vals[idx]),
                    "I_diff": float(i_diff_vals[idx]),
                    "I_attr": float(i_attr_vals[idx]),
                    "I_llm": float(i_llm_vals[idx]),
                    "llm_pair_brief": llm_pair_briefs[idx],
                    "src_label_text": self._summarize_label(src_labels[idx]),
                    "tgt_label_text": self._summarize_label(tgt_labels[idx]),
                    "src_context_text": self._summarize_context(src_ctxs[idx] if src_ctxs else []),
                    "tgt_context_text": self._summarize_context(tgt_ctxs[idx] if tgt_ctxs else []),
                }
                for output_name, values in optional_experiment_values.items():
                    candidate_row[output_name] = float(values[idx])
                source_decision = next(
                    (
                        record
                        for record in out.get("llm_grouped_decisions", [])
                        if record["source"] == str(s)
                    ),
                    None,
                )
                if source_decision is not None:
                    candidate_row["llm_grouped_decision"] = json.dumps(
                        source_decision, sort_keys=True
                    )
                    if (
                        source_decision.get("valid")
                        and source_decision.get("integration") == "source_first"
                    ):
                        candidate_row["llm_source_choice"] = source_decision["choice"]
                if gate_diagnostics is not None:
                    candidate_row.update(self._gate_candidate_fields(gate_diagnostics[idx]))
                from exact.runs.decisions import append_event, candidate_values, event, pair_key

                initial = getattr(self.dataset, "_candidate_stage_observations", {}).get(
                    pair_key(candidate_row), {}
                )
                if initial:
                    candidate_row["candidate_decision"] = initial["candidate_decision"]
                candidate_row["candidate_decision"] = append_event(
                    candidate_row,
                    event(
                        "pair_scoring",
                        reason="scorer_returned",
                        implementation=f"{type(self.model).__module__}.{type(self.model).__name__}",
                        values=candidate_values(candidate_row),
                    ),
                )
                if getattr(self.model, "use_llm", None) is False:
                    candidate_row["candidate_decision"] = append_event(
                        candidate_row,
                        event(
                            "llm_signal",
                            status="not_run",
                            outcome="not_applicable",
                            reason="llm_disabled",
                            implementation="exact.pipeline",
                        ),
                    )
                elif gate_diagnostics is not None:
                    invoked = bool(candidate_row.get("llm_gate_invoked"))
                    candidate_row["candidate_decision"] = append_event(
                        candidate_row,
                        event(
                            "llm_signal",
                            status="completed" if invoked else "not_run",
                            outcome=str(candidate_row.get("llm_gate_outcome") or "unknown"),
                            reason="llm_invoked" if invoked else "llm_gate_skipped",
                            implementation=f"{type(self.model).__module__}.{type(self.model).__name__}",
                            values={
                                key: value
                                for key, value in candidate_values(candidate_row).items()
                                if key == "p_llm" or key.startswith("llm_gate_")
                            },
                        ),
                    )
                self._candidate_rows.append(candidate_row)

            processed_examples += len(src_iri)
            batches_run += 1
            elapsed_for_rate = max(1.0e-9, time.perf_counter() - start_time)
            current_rate = (processed_examples - restored_examples) / elapsed_for_rate
            if current_rate > 0.0:
                if self._examples_per_second_ema is None:
                    self._examples_per_second_ema = current_rate
                else:
                    self._examples_per_second_ema = (
                        0.2 * current_rate + 0.8 * self._examples_per_second_ema
                    )

            # Accumulate full JSONs
            explanations = list(out.get("explanations") or [])
            union_records: List[Dict[str, Any]] = []
            batch_candidate_rows = self._candidate_rows[candidate_start_idx:]
            for local_idx, candidate_row in enumerate(batch_candidate_rows):
                explanation = explanations[local_idx] if local_idx < len(explanations) else None
                record = self._union_explanation_record(candidate_row, explanation)
                if explanation is not None:
                    candidate_idx = candidate_start_idx + local_idx
                    if candidate_idx >= len(self._candidate_rows):
                        continue
                    evidence_items = self._selector_evidence_items_for_record(record)
                    if evidence_items:
                        self._candidate_rows[candidate_idx][
                            "selector_evidence_items"
                        ] = evidence_items
                        record["selector_evidence_items"] = evidence_items
                union_records.append(record)
            self.results_json.extend(union_records)
            self._append_audit_records(union_records)

            if (
                checkpoint_enabled
                and cp_path
                and (
                    step % checkpoint_every == 0
                    or step == total_batches
                    or time.monotonic() - last_checkpoint_at >= 300
                    or (stop_path is not None and stop_path.exists())
                )
            ):
                self._write_checkpoint_state(
                    cp_path,
                    kind,
                    total_examples=total_examples,
                    processed_examples=processed_examples,
                    mappings=all_mappings,
                    results_json=self.results_json,
                )
                last_checkpoint_at = time.monotonic()
                self._maybe_persist_model_cache(reason="checkpoint")

            if step % log_every == 0 or step == total_batches:
                elapsed = time.perf_counter() - start_time
                avg_batch_time = elapsed / max(1, step)
                remaining_batches = max(0, total_batches - step)
                remaining_time = remaining_batches * avg_batch_time
                eta_str = _format_duration(remaining_time)
                pair_diag = ""
                if pair_batch_stats:
                    evidence = pair_batch_stats.get("pair_evidence") or {}
                    pair_total = max(1, int(pair_batch_stats.get("pairs", len(src_iri))))
                    pair_diag = (
                        " | pair-adaptive ctx: "
                        f"cache src={pair_batch_stats.get('src_cache_hits', 0)}/{pair_batch_stats.get('unique_src', 0)}, "
                        f"tgt={pair_batch_stats.get('tgt_cache_hits', 0)}/{pair_batch_stats.get('unique_tgt', 0)}; "
                        f"selected hier/sim/diff/attr={evidence.get('hier_selected', 0)}/{pair_total},"
                        f"{evidence.get('sim_selected', 0)}/{pair_total},"
                        f"{evidence.get('diff_selected', 0)}/{pair_total},"
                        f"{evidence.get('attr_selected', 0)}/{pair_total}; "
                        f"struct-active={pair_batch_stats.get('struct_active_pairs', 0)}/{pair_total}; "
                        f"llm-gated={pair_batch_stats.get('llm_gated_pairs', 0)}/{pair_total}"
                    )
                self.log(
                    (
                        f"Batch {step}/{total_batches} done "
                        f"(avg {avg_batch_time:.2f}s/batch, ETA {eta_str})"
                        f"{pair_diag}"
                    ),
                    "debug",
                )
                if run_progress is not None:
                    run_progress.update(
                        "Inference",
                        completed=processed_examples,
                        total=total_examples,
                        detail=f"pairs={processed_examples}/{total_examples}, batches={step}/{total_batches}",
                    )

        self._finalize_llm_calibration()

        inference_elapsed_seconds = time.perf_counter() - start_time
        self._inference_seconds_cumulative = (
            self._restored_inference_seconds_cumulative + inference_elapsed_seconds
        )
        self._inference_session_started_at = None
        total_time = inference_elapsed_seconds
        duration_str = _format_duration(total_time)
        new_examples = max(0, processed_examples - restored_examples)
        avg_t = total_time / max(1, new_examples)
        avg_bt = total_time / max(1, batches_run)
        self.log(
            (
                f"Completed {kind.name} in {duration_str} "
                f"(processed {new_examples} new pairs, ~{avg_t:.3f}s/example, ~{avg_bt:.2f}s/batch)"
            ),
            "info",
        )
        if run_progress is not None:
            run_progress.finish("Inference", f"processed={processed_examples}/{total_examples}")
        if hasattr(self.model, "llm_summary_stats"):
            self._llm_summary_stats = self.model.llm_summary_stats()
            stats = self._llm_summary_stats or {}
            requested = stats.get("requested", 0)
            usable = stats.get("usable", 0)
            empty = stats.get("empty", 0)
            if requested:
                empty_pct = (empty / requested) * 100.0
                self.log(
                    (
                        "LLM summary/brief quality: "
                        f"{requested} requested, {usable} usable, {empty} empty (~{empty_pct:.2f}% empty)"
                    ),
                    "info",
                )
            else:
                self.log("LLM summaries/briefs were not requested for this run.", "debug")

        # Normalize LLM importance: if the LLM did not run (p_llm == 0 or no decision),
        # clamp I_llm to 0 so downstream summaries/plots are consistent with behavior.
        for rec in self.results_json or []:
            imps = rec.get("importances") or {}
            if "I_llm" not in imps:
                continue
            conf = rec.get("confidences") or {}
            pred = rec.get("prediction") or {}
            p_llm = conf.get("p_llm")
            llm_decision = pred.get("llm_decision")
            if p_llm is None or float(p_llm) == 0.0 or llm_decision in ("", None):
                imps["I_llm"] = 0.0

        self._maybe_persist_model_cache(reason="finalize", force=True)
        inference_status = CacheStatus.RESUMED if restored_examples > 0 else CacheStatus.FRESH
        return self._finalize_run(
            _FinalizationState(
                kind=kind,
                threshold=threshold,
                cardinality=cardinality,
                target_cardinality=target_cardinality,
                local_alignment=local_alignment,
                all_mappings=all_mappings,
                total_examples=total_examples,
                processed_examples=processed_examples,
                restored_examples=restored_examples,
                checkpoint_enabled=checkpoint_enabled,
                checkpoint_path=cp_path,
                audit_shard_compression=audit_shard_compression,
                audit_shard_records=audit_shard_records,
                log_every=log_every,
                run_progress=run_progress,
                inference_seconds=inference_elapsed_seconds,
                average_seconds_per_example=avg_t,
                inference_work_done=new_examples,
                inference_status=inference_status,
                post_inference_status=inference_status,
            )
        )

    @staticmethod
    def _effective_alignment_threshold(
        candidate_df: pd.DataFrame,
        threshold: Optional[float],
    ) -> Optional[float]:
        if threshold is None or candidate_df.empty:
            return threshold
        if "selection_accept_threshold" not in candidate_df.columns:
            return threshold
        selected = pd.to_numeric(candidate_df["selection_accept_threshold"], errors="coerce")
        selected = selected[selected.notna() & (selected > 0.0)]
        if selected.empty:
            return threshold
        return float(selected.median())

    @staticmethod
    def _effective_alignment_threshold_origin(
        candidate_df: pd.DataFrame,
        threshold: Optional[float],
    ) -> str:
        if threshold is None or candidate_df.empty:
            return "configured"
        if "selection_accept_threshold" not in candidate_df.columns:
            return "configured"
        selected = pd.to_numeric(candidate_df["selection_accept_threshold"], errors="coerce")
        selected = selected[selected.notna() & (selected > 0.0)]
        return "selector-median override" if not selected.empty else "configured"

    @staticmethod
    def _selector_target_conflict_enabled_from_df(
        candidate_df: pd.DataFrame,
    ) -> Optional[bool]:
        if candidate_df.empty or "selection_target_conflict_enabled" not in candidate_df.columns:
            return None
        if "selection_accept_threshold" in candidate_df.columns:
            thresholds = pd.to_numeric(candidate_df["selection_accept_threshold"], errors="coerce")
            if thresholds[thresholds.notna() & (thresholds > 0.0)].empty:
                return None
        values = candidate_df["selection_target_conflict_enabled"]
        if values.empty:
            return None
        normalized = values.astype(str).str.lower().isin({"true", "1", "yes", "y"})
        return bool(normalized.any())

    @property
    def last_stage_timings(self) -> List[StageRecord]:
        return list(self._last_stage_timings)

    @property
    def inference_seconds_cumulative(self) -> float:
        return self._inference_seconds_cumulative

    @property
    def examples_per_second_ema(self) -> Optional[float]:
        return self._examples_per_second_ema

    @property
    def last_effective_threshold(self) -> Optional[float]:
        return self._last_effective_threshold

    @property
    def last_effective_threshold_origin(self) -> str:
        return self._last_effective_threshold_origin

    def _build_candidate_dataframe(self) -> pd.DataFrame:
        if not self._candidate_rows:
            return pd.DataFrame()
        df = pd.DataFrame(self._candidate_rows)
        return self._merge_dataset_candidate_columns(df)

    def _build_candidate_dataframe_from_records(
        self, records: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for record in records or []:
            src = record.get("src_iri")
            tgt = record.get("tgt_iri")
            if src is None or tgt is None:
                continue
            pred = record.get("prediction") or {}
            row: Dict[str, Any] = {
                "Src": src,
                "Tgt": tgt,
                "ground_truth": pred.get("ground_truth"),
                "src_label_text": (record.get("selected_labels") or {}).get("source", ""),
                "tgt_label_text": (record.get("selected_labels") or {}).get("target", ""),
                "llm_pair_brief": record.get("llm_pair_brief", ""),
            }
            src_kind = record.get("src_kind", record.get("kind"))
            tgt_kind = record.get("tgt_kind", src_kind)
            if src_kind is not None:
                row["SrcKind"] = src_kind
            if tgt_kind is not None:
                row["TgtKind"] = tgt_kind
            for payload_name in ["confidences", "qualities", "weights", "importances"]:
                payload = record.get(payload_name) or {}
                for key, value in payload.items():
                    if isinstance(value, (dict, list, tuple)):
                        continue
                    row[key] = value
            if "S_final" in row:
                rows.append(row)
        if not rows:
            return pd.DataFrame()
        return self._merge_dataset_candidate_columns(pd.DataFrame(rows))

    def _merge_dataset_candidate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        dataset_df = getattr(self.dataset, "dataframe", None)
        if dataset_df is None or dataset_df.empty:
            return df
        prefixes = (
            "cand_",
            "src_hier",
            "tgt_hier",
            "src_obj",
            "tgt_obj",
            "src_attr",
            "tgt_attr",
            "src_lab",
            "tgt_lab",
        )
        identity_columns = {"Src", "Tgt", "SrcKind", "TgtKind"}
        extra_cols = [
            col
            for col in dataset_df.columns
            if col in identity_columns or any(str(col).startswith(prefix) for prefix in prefixes)
        ]
        if len(extra_cols) <= 2:
            return df
        join_columns = ["Src", "Tgt"]
        join_columns.extend(
            column
            for column in ("SrcKind", "TgtKind")
            if column in df.columns and column in dataset_df.columns
        )
        extra_df = dataset_df[extra_cols].drop_duplicates(subset=join_columns)
        merge_cols = [
            col for col in extra_df.columns if col not in df.columns or col in join_columns
        ]
        if not any(column not in join_columns for column in merge_cols):
            return df
        return df.merge(extra_df[merge_cols], on=join_columns, how="left")

    def _apply_additional_models(
        self,
        df: pd.DataFrame,
        kind: DatasetMask = DatasetMask.inference,
        local_alignment: bool = False,
        threshold: Optional[float] = None,
        cardinality: Optional[int] = None,
        target_cardinality: Optional[int] = None,
        results_json: Optional[List[Dict[str, Any]]] = None,
        log_every: int = 10,
        run_progress: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Run any additional models (beyond the first) sequentially.
        Each model is expected to accept the candidate dataframe and return
        either an updated dataframe or a dict containing 'candidate_df'.
        """
        current = df
        if len(getattr(self, "models", [])) <= 1:
            if not current.empty:
                from exact.runs.decisions import append_event, event

                current["candidate_decision"] = [
                    append_event(
                        row,
                        event(
                            "selection",
                            status="not_run",
                            outcome="not_applicable",
                            reason="no_additional_selector",
                            implementation="exact.pipeline",
                        ),
                    )
                    for row in current.to_dict("records")
                ]
            return current
        checkpoint_every_groups = max(50000, max(1, int(log_every)) * 5000)

        def _checkpoint_current(candidate_df: pd.DataFrame) -> None:
            self._write_additional_models_checkpoint(
                kind,
                candidate_df,
                local_alignment,
                threshold,
                cardinality,
                log_level="debug",
                complete=False,
            )

        checkpoint_callback = (
            _checkpoint_current
            if bool(getattr(self, "_postprocess_checkpoints_enabled", False))
            else None
        )
        for idx, extra_model in enumerate(self.models[1:], start=2):
            extra_model.eval()
            model_name = extra_model.__class__.__name__
            model_start = time.perf_counter()
            self.log(
                (
                    f"Running additional model #{idx} ({model_name}) on "
                    f"{len(current)} candidate rows."
                ),
                "info",
            )
            if run_progress is not None:
                run_progress.update(
                    "PostInference",
                    fraction=0.25,
                    detail=f"model={model_name}, rows={len(current)}",
                    force=True,
                )
            sig = inspect.signature(extra_model.forward)
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            call_kwargs = {"candidate_df": current}
            if accepts_var_kw or "dataset" in sig.parameters:
                call_kwargs["dataset"] = self.dataset
            if accepts_var_kw or "logger" in sig.parameters:
                call_kwargs["logger"] = getattr(self, "logger", None)
            if accepts_var_kw or "primary_model" in sig.parameters:
                call_kwargs["primary_model"] = self.model
            if accepts_var_kw or "results_json" in sig.parameters:
                call_kwargs["results_json"] = results_json
            if accepts_var_kw or "local_alignment" in sig.parameters:
                call_kwargs["local_alignment"] = local_alignment
            if accepts_var_kw or "threshold" in sig.parameters:
                call_kwargs["threshold"] = threshold
            if accepts_var_kw or "cardinality" in sig.parameters:
                call_kwargs["cardinality"] = cardinality
            if accepts_var_kw or "target_cardinality" in sig.parameters:
                call_kwargs["target_cardinality"] = target_cardinality
            if accepts_var_kw or "log_every" in sig.parameters:
                call_kwargs["log_every"] = log_every
            if accepts_var_kw or "run_progress" in sig.parameters:
                call_kwargs["run_progress"] = run_progress
            if checkpoint_callback is not None and (
                accepts_var_kw or "checkpoint_callback" in sig.parameters
            ):
                call_kwargs["checkpoint_callback"] = checkpoint_callback
            if checkpoint_callback is not None and (
                accepts_var_kw or "checkpoint_every_groups" in sig.parameters
            ):
                call_kwargs["checkpoint_every_groups"] = checkpoint_every_groups
            out = extra_model.forward(**call_kwargs)
            if isinstance(out, pd.DataFrame):
                current = out
            elif isinstance(out, dict) and "candidate_df" in out:
                current = out["candidate_df"]
                extra_json = out.get("results_json")
                if extra_json and extra_json is not self.results_json:
                    self.results_json.extend(extra_json)
                    results_json = self.results_json
            else:
                self.log(
                    f"Additional model #{idx} returned unsupported type {type(out)}; ignoring its output.",
                    level="warning",
                )
            elapsed = max(0.0, time.perf_counter() - model_start)
            self.log(
                (
                    f"Additional model #{idx} ({model_name}) finished in "
                    f"{_format_duration(elapsed)} with {len(current)} candidate rows."
                ),
                "info",
            )
            if run_progress is not None:
                run_progress.update(
                    "PostInference",
                    fraction=0.85,
                    detail=f"model={model_name} finished, rows={len(current)}",
                    force=True,
                )
        return current
