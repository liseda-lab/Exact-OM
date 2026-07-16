"""Versioned, strictly validated runtime configuration.

The pydantic models in this module are the single source for configuration
defaults and field descriptions.  ``exact/default_config.yaml`` is generated
from these models; importing :mod:`exact` never reads that file.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from exact.core.contracts.dataset import IDataset
from exact.core.contracts.trainer import ITrainer
from exact.core.entities.configs.dataset import BestPathMethod, ContextMethod
from exact.core.entities.registry import ComponentRegistry, ComponentType

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "default_config.yaml"
CONFIG_VERSION = 2


class StrictConfigModel(BaseModel):
    """Base class that rejects misspelled configuration keys with a hint."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_default=True,
        use_enum_values=False,
    )

    @model_validator(mode="before")
    @classmethod
    def reject_unknown_keys(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        accepted: set[str] = set(cls.model_fields)
        aliases: dict[str, str] = {}
        for name, info in cls.model_fields.items():
            alias = info.alias
            if isinstance(alias, str):
                accepted.add(alias)
                aliases[alias] = name
        unknown = [str(key) for key in value if str(key) not in accepted]
        if not unknown:
            return value
        messages: list[str] = []
        for key in sorted(unknown):
            suggestion = difflib.get_close_matches(key, sorted(accepted), n=1, cutoff=0.55)
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            messages.append(f"Unknown configuration key '{key}'.{hint}")
        raise ValueError(" ".join(messages))


class CompatibilityParams(BaseModel):
    """Flat, read-only compatibility view used by pre-v2 runtime consumers."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class DatasetParams(CompatibilityParams):
    """Compatibility view of ``dataset`` plus ``llm.verbaliser``."""


class CandidatesParams(CompatibilityParams):
    """Compatibility view of the v2 ``candidates`` section."""


class SanityCheckParams(CompatibilityParams):
    """Compatibility view of ``output.sanity_checks``."""


class PlotParams(CompatibilityParams):
    """Compatibility view of ``output.plots``."""


class InferenceParams(CompatibilityParams):
    """Compatibility view of the v2 ``inference`` section."""


class AlignmentParams(CompatibilityParams):
    """Compatibility view of ``matching`` plus ``output.save``."""


class RegistryParams(BaseModel):
    """Resolved runtime component specification."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    component_type: ComponentType = ComponentType.MODEL
    name: Any
    params: Dict[str, Any] = Field(default_factory=dict)


class ModelParams(RegistryParams):
    """Compatibility name for a primary runtime model specification."""


class SecondModelParams(RegistryParams):
    """Compatibility name for an optional post-inference model."""

    name: Any = None


class ModelChainEntry(RegistryParams):
    """Compatibility name for one resolved model-chain entry."""


class RunConfig(StrictConfigModel):
    seed: int = Field(42, description="Global random seed used by deterministic components.")
    logging_level: str = Field(
        "INFO", description="Logging verbosity: DEBUG, INFO, WARNING, ERROR, or CRITICAL."
    )
    use_file_cache: bool = Field(
        True, description="Reuse compatible dataset and model caches when available."
    )

    @field_validator("logging_level", mode="before")
    @classmethod
    def normalize_logging_level(cls, value: Any) -> str:
        if isinstance(value, int):
            value = logging.getLevelName(value)
        level = str(value).strip().upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("logging_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        return level


class DataConfig(StrictConfigModel):
    """Dataset-track selection and optional explicit input-path overrides."""

    track: Optional[str] = Field(None, description="Registered dataset-track provider name.")
    task: Optional[str] = Field(None, description="Task identifier within the selected track.")
    root: Path = Field(Path("data"), description="Local track materialization root.")
    revision: Optional[str] = Field(None, description="Optional immutable track revision.")
    descriptor: Optional[Path] = Field(
        None, description="Optional local track descriptor overriding registry discovery."
    )
    source: Optional[Path] = Field(None, description="Explicit source knowledge-source path.")
    target: Optional[Path] = Field(None, description="Explicit target knowledge-source path.")
    refs: Dict[str, Path] = Field(
        default_factory=dict,
        description="Named reference alignments, such as train, test, and full.",
    )
    candidates: Optional[Path] = Field(None, description="Optional candidate alignment path.")

    @field_validator("track", "task")
    @classmethod
    def validate_identifier(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized or normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
            raise ValueError("track and task must be non-empty path-safe identifiers")
        return normalized


class IOConfig(StrictConfigModel):
    input_format: str = Field(
        "auto", description="Input adapter name, or auto to infer it from each input path."
    )
    source_options: Dict[str, Any] = Field(
        default_factory=dict, description="Source-adapter-specific options."
    )
    target_options: Dict[str, Any] = Field(
        default_factory=dict, description="Target-adapter-specific options."
    )
    output_formats: List[str] = Field(
        default_factory=lambda: ["tsv-global", "tsv-local"],
        description="Ordered alignment writer formats produced for each run.",
    )

    @field_validator("input_format", mode="before")
    @classmethod
    def validate_input_format(cls, value: Any) -> str:
        input_format = str(value or "auto").strip().lower()
        if input_format not in {"auto", "owl", "rdf", "csv-kg"}:
            raise ValueError("input_format must be auto, owl, rdf, or csv-kg")
        return input_format

    @field_validator("output_formats", mode="before")
    @classmethod
    def validate_output_formats(cls, value: Any) -> List[str]:
        formats = [str(item).strip().lower() for item in (value or [])]
        formats = list(dict.fromkeys(item for item in formats if item))
        if not formats:
            raise ValueError("output_formats must contain at least one non-empty writer name")
        return formats


class MatchingChannelsConfig(StrictConfigModel):
    hierarchy_embedding_weight: float = Field(
        0.5, description="Embedding contribution to each hierarchy-channel score."
    )
    hierarchy_support_weight: float = Field(
        0.5, description="Cross-side support contribution to each hierarchy-channel score."
    )
    similarity_embedding_weight: float = Field(
        0.5, description="Embedding contribution to relational-similarity scoring."
    )
    similarity_support_weight: float = Field(
        0.5, description="Cross-side support contribution to relational-similarity scoring."
    )
    similarity_per_relation_cap: int = Field(
        2, description="Maximum selected similarity triples per relation and side."
    )
    difference_per_relation_cap: int = Field(
        3, description="Maximum selected difference triples per relation and side."
    )
    stability_factor: float = Field(
        2.0, description="Multiplier applied to support standard deviation in channel quality."
    )
    attribute_property_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "definition": 1.0,
            "identifier": 0.8,
            "comment": 0.6,
            "other": 0.5,
        },
        description="Fallback weights for definition, identifier, comment, and other attributes.",
    )
    attribute_information_word_cap: int = Field(
        20, description="Word-count ceiling used to normalize attribute informativeness."
    )
    attribute_score_floor: float = Field(
        0.5, description="Minimum support-only attribute-channel score."
    )
    uncertainty_indecision_scale: float = Field(
        2.0, description="Scale of distance-to-neutral indecision uncertainty."
    )
    uncertainty_disagreement_quality_power: float = Field(
        0.5, description="Power applied to lexical/structural quality in disagreement uncertainty."
    )


class MatchingConfig(StrictConfigModel):
    threshold: Optional[float] = Field(
        0.7, description="Minimum final score retained in the global alignment."
    )
    cardinality: Optional[int] = Field(
        1, description="Maximum targets retained per source; null disables the limit."
    )
    target_cardinality: Optional[int] = Field(
        1, description="Maximum sources retained per target; null disables the limit."
    )
    review_low: Optional[float] = Field(0.5, description="Lower review-band score bound.")
    review_high: Optional[float] = Field(0.8, description="Upper review-band score bound.")
    entity_kinds: List[str] = Field(
        default_factory=lambda: ["class"],
        description="Knowledge-graph entity kinds eligible for alignment.",
    )
    relation_prediction: str = Field(
        "none", description="Relation typing mode: none or hierarchy_heuristic."
    )
    channels: MatchingChannelsConfig = Field(
        default=MatchingChannelsConfig.model_validate({}),
        description="Pair-adaptive channel constants exposed for reproducible sweeps.",
    )

    @field_validator("entity_kinds", mode="before")
    @classmethod
    def validate_entity_kinds(cls, value: Any) -> List[str]:
        from exact.core.entities.kinds import normalize_entity_kinds

        return [kind.value for kind in normalize_entity_kinds(value)]

    @field_validator("relation_prediction", mode="before")
    @classmethod
    def validate_relation_prediction(cls, value: Any) -> str:
        mode = str(value or "none").strip().lower()
        if mode not in {"none", "hierarchy_heuristic"}:
            raise ValueError("relation_prediction must be none or hierarchy_heuristic")
        return mode


def _default_hierarchy_families() -> Dict[str, Dict[str, Any]]:
    return {
        "is_a": {
            "iri_aliases": ["http://www.w3.org/2000/01/rdf-schema#subClassOf"],
            "label_seeds": ["subclass of", "is a", "type of", "kind of"],
        },
        "part_of": {
            "iri_aliases": ["http://purl.obolibrary.org/obo/BFO_0000050"],
            "label_seeds": ["part of", "component of"],
        },
        "has_part": {
            "iri_aliases": ["http://purl.obolibrary.org/obo/BFO_0000051"],
            "label_seeds": ["has part"],
        },
    }


def _default_dataset_plot_fields() -> List[str]:
    return [
        "src_hier_total_count",
        "tgt_hier_total_count",
        "src_hier_is_a_count",
        "tgt_hier_is_a_count",
        "src_hier_part_of_count",
        "tgt_hier_part_of_count",
        "src_hier_has_part_count",
        "tgt_hier_has_part_count",
        "src_obj_count",
        "tgt_obj_count",
        "src_obj_relation_count",
        "tgt_obj_relation_count",
        "src_obj_ic_mean",
        "tgt_obj_ic_mean",
        "src_attr_count",
        "tgt_attr_count",
        "src_attr_weight_mean",
        "tgt_attr_weight_mean",
        "src_lab_n_labels",
        "tgt_lab_n_labels",
        "src_lab_word_len",
        "tgt_lab_word_len",
        "src_lab_avg_label_words",
        "tgt_lab_avg_label_words",
        "cand_sim",
        "cand_sim_src_mean",
    ]


class DatasetLegacyConfig(StrictConfigModel):
    context_method: ContextMethod = Field(
        ContextMethod.greedy, description="Legacy single-context extraction strategy."
    )
    best_path_method: BestPathMethod = Field(
        BestPathMethod.dp, description="Legacy context path-search strategy."
    )
    context_hop_penalty: float = Field(0.1, description="Legacy context-path hop penalty.")
    context_token_ratio: float = Field(
        1.3, description="Legacy approximate tokens-per-word budgeting ratio."
    )
    context_safety: float = Field(
        0.8, description="Legacy safety factor for context token budgets."
    )
    only_taxonomy: bool = Field(
        False, description="Restrict legacy context extraction to taxonomy edges."
    )
    add_connectivity_bridges: bool = Field(
        True, description="Add explanation-only bridges to legacy extracted contexts."
    )
    bridge_max_hops: Optional[int] = Field(
        None, description="Optional hard cap on legacy bridge path length."
    )


class DatasetConfig(StrictConfigModel):
    _runtime_component: Optional[Type[IDataset]] = PrivateAttr(default=None)

    reasoner: str = Field("asserted", description="Registered hierarchy reasoner name.")
    num_workers: Optional[int] = Field(None, description="Dataset loading worker count.")
    filter_exact_matches: bool = Field(
        True, description="Remove exact lexical matches before semantic scoring."
    )
    drop_exact_match_sources: bool = Field(
        False, description="Remove every candidate for sources with an exact match."
    )
    filter_ignored_alignment_classes: bool = Field(
        True, description="Exclude knowledge-source entities marked use_in_alignment=false."
    )
    projection_include_literals: bool = Field(
        True, description="Include projected literal edges as attribute/support evidence."
    )
    hierarchy_max_depth: int = Field(
        2, description="Maximum hierarchy depth collected for each configured family."
    )
    max_hierarchy_triples_per_family: int = Field(
        6, description="Maximum hierarchy triples per family and entity."
    )
    max_object_triples: int = Field(
        48, description="Maximum non-hierarchical object triples per entity."
    )
    max_diff_triples: int = Field(
        24, description="Maximum distinctive triples retained before pair reranking."
    )
    max_attr_items: int = Field(
        12, description="Maximum literal or annotation snippets per entity."
    )
    pair_adaptive_feature_log_every: int = Field(
        1000, description="Feature-build debug progress interval in unique entities."
    )
    hierarchical_relation_families: Dict[str, Dict[str, Any]] = Field(
        default_factory=_default_hierarchy_families,
        description="Ontology-native hierarchical relation families and aliases.",
    )
    n_hops: int = Field(1, description="Projected neighborhood radius for structural evidence.")
    max_input_tokens_context: int = Field(
        256, description="Shared upper bound for joined context text."
    )
    all_labels: bool = Field(
        True, description="Score all label variants instead of only a primary label."
    )
    delimiter: str = Field("\n", description="Delimiter used to join verbalized triples.")
    which: Optional[List[str]] = Field(
        default_factory=_default_dataset_plot_fields,
        description="Dataset feature names included in preprocessing plots.",
    )
    candidate_share_k: int = Field(
        2, description="Top lexical candidates shared by candidate utilities."
    )
    legacy: DatasetLegacyConfig = Field(
        default=DatasetLegacyConfig.model_validate({}),
        description="Controls retained only for the legacy single-context dataset.",
    )

    def bind_runtime_component(self, component: Type[IDataset]) -> None:
        """Attach the registry class used by transitional runtime callers."""

        self._runtime_component = component

    def __call__(self, *args: Any, **kwargs: Any) -> IDataset:
        """Instantiate the resolved dataset class for pre-v2 consumers."""

        if self._runtime_component is None:
            raise RuntimeError("resolve_dependencies() must run before constructing the dataset")
        return self._runtime_component(*args, **kwargs)


class CandidateFusionConfig(StrictConfigModel):
    token_weight: float = Field(1.0, description="Weight of lexical token cosine in max fusion.")
    gram_weight: float = Field(0.85, description="Weight of character-gram cosine in max fusion.")
    blend_token_weight: float = Field(
        0.65, description="Token contribution to the blended lexical retrieval score."
    )
    blend_gram_weight: float = Field(
        0.35, description="Character-gram contribution to the blended retrieval score."
    )
    semantic_channel_weight: float = Field(
        1.0, description="Weight applied to semantic retrieval before channel max fusion."
    )
    lexical_channel_weight: float = Field(
        1.0, description="Weight applied to lexical retrieval before channel max fusion."
    )
    df_ceiling_floor: int = Field(
        10, description="Minimum document-frequency ceiling for lexical features."
    )
    df_ceiling_ratio: float = Field(
        0.2, description="Target-corpus ratio used for the feature DF ceiling."
    )


def _candidate_rejected_terms() -> List[str]:
    return [
        "alternativeid",
        "comment",
        "created",
        "creator",
        "curator",
        "date",
        "definition",
        "deprecated",
        "description",
        "editor",
        "format",
        "id",
        "identifier",
        "namespace",
        "note",
        "provenance",
        "review",
        "reviewer",
        "semantic",
        "slim",
        "source",
        "status",
        "subset",
        "type",
        "version",
        "xref",
    ]


class CandidateAliasConfig(StrictConfigModel):
    overall_cap: int = Field(12, description="Maximum annotation aliases retained per entity.")
    max_tokens: int = Field(12, description="Maximum normalized tokens in an alias literal.")
    exact_property_cap: int = Field(8, description="Per-property cap for exact synonyms.")
    preferred_property_cap: int = Field(2, description="Per-property cap for preferred terms.")
    related_property_cap: int = Field(4, description="Per-property cap for related synonyms.")
    default_property_cap: int = Field(4, description="Fallback per-property alias cap.")
    accepted_property_terms: List[str] = Field(
        default_factory=lambda: ["alt", "label", "name", "pref", "synonym", "term", "title"],
        description="Property-name tokens accepted as alias signals.",
    )
    rejected_property_terms: List[str] = Field(
        default_factory=_candidate_rejected_terms,
        description="Property-name tokens excluded from candidate aliases.",
    )
    rejected_properties: List[str] = Field(
        default_factory=lambda: [
            "nhc0",
            "p97",
            "p106",
            "p207",
            "p322",
            "p325",
            "p363",
            "iao0000115",
            "iao0000231",
        ],
        description="Compact property identifiers excluded from candidate aliases.",
    )
    exact_properties: List[str] = Field(
        default_factory=lambda: ["p90", "has exact synonym"],
        description="Compact identifiers or names treated as exact-synonym properties.",
    )
    preferred_properties: List[str] = Field(
        default_factory=lambda: ["p107", "p108"],
        description="Compact identifiers treated as preferred-term properties.",
    )
    related_properties: List[str] = Field(
        default_factory=lambda: [
            "has related synonym",
            "has narrow synonym",
            "has broad synonym",
        ],
        description="Property names treated as related-synonym sources.",
    )
    exact_priority: float = Field(0.0, description="Sort priority for exact aliases.")
    preferred_priority: float = Field(0.05, description="Sort priority for preferred aliases.")
    related_priority: float = Field(0.15, description="Sort priority for related aliases.")
    default_priority: float = Field(0.3, description="Sort priority for other accepted aliases.")
    priority_length_target: int = Field(
        3, description="Preferred alias token count for the priority length penalty."
    )
    priority_length_cap: int = Field(
        6, description="Token-count cap used by the alias priority length penalty."
    )
    priority_length_divisor: float = Field(
        30.0, description="Divisor used by the alias priority length penalty."
    )


class CandidatesConfig(StrictConfigModel):
    retrieval_strategy: str = Field(
        "hybrid", description="Candidate retrieval strategy: primary_label or hybrid."
    )
    lexical_encoder_name: Optional[str] = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence encoder used for semantic candidate retrieval.",
    )
    encode_batch_size: int = Field(512, description="Candidate-encoder batch size.")
    search_batch_size: int = Field(4096, description="Cosine-search batch size.")
    top_k: int = Field(20, description="Candidates retained per source entity.")
    use_amp: bool = Field(
        True, description="Use automatic mixed precision for candidate retrieval when supported."
    )
    fusion: CandidateFusionConfig = Field(
        default=CandidateFusionConfig.model_validate({}),
        description="Lexical and retrieval-channel fusion constants.",
    )
    aliases: CandidateAliasConfig = Field(
        default=CandidateAliasConfig.model_validate({}),
        description="Annotation-alias selection caps, priorities, and blocklists.",
    )

    @field_validator("retrieval_strategy", mode="before")
    @classmethod
    def validate_retrieval_strategy(cls, value: Any) -> str:
        strategy = str(value or "hybrid").lower()
        if strategy not in {"primary_label", "hybrid"}:
            raise ValueError("retrieval_strategy must be 'primary_label' or 'hybrid'")
        return strategy


def _default_inference_fields() -> List[str]:
    return [
        "S_final",
        "S_base",
        "S_struct",
        "s_label",
        "s_hier",
        "s_sim",
        "s_diff",
        "s_attr",
        "I_label",
        "I_struct",
        "I_hier",
        "I_sim",
        "I_diff",
        "I_attr",
        "I_llm",
        "w_c",
        "w_i",
        "U",
        "U_ind",
        "U_dis",
        "p_llm",
    ]


class InferenceConfig(StrictConfigModel):
    batch_size: int = Field(512, description="Candidate-pair inference batch size.")
    num_workers: int = Field(0, description="Inference DataLoader worker count.")
    log_every: int = Field(10, description="Inference progress interval in batches.")
    mixed_precision: bool = Field(
        True, description="Use torch autocast during supported GPU inference."
    )
    which: Optional[List[str]] = Field(
        default_factory=_default_inference_fields,
        description="Model output fields included in inference plots.",
    )
    checkpoint_every: int = Field(10, description="Checkpoint interval in inference batches.")
    resume_from_checkpoint: bool = Field(
        True, description="Resume from the latest compatible inference checkpoint."
    )
    enable_checkpoints: bool = Field(True, description="Enable inference checkpoint creation.")
    resume_additional_model_checkpoints: bool = Field(
        True, description="Reuse compatible selector or reranker checkpoints."
    )
    allow_rationale_toggle_checkpoint_resume: bool = Field(
        True, description="Allow resume when only rationale generation changed."
    )
    audit_shards_enabled: bool = Field(
        True, description="Write incremental candidate-level audit shards."
    )
    audit_shard_compression: str = Field(
        "zstd", description="Audit shard compression: zstd or none."
    )
    audit_shard_records: int = Field(
        50000, description="Maximum records per audit shard before rotation."
    )
    checkpoint_payload: str = Field(
        "compact", description="Checkpoint payload mode: compact or full."
    )
    cache_persist_policy: str = Field(
        "finalize", description="Cache persistence policy: checkpoint, finalize, or never."
    )

    @field_validator("audit_shard_compression", mode="before")
    @classmethod
    def validate_audit_shard_compression(cls, value: Any) -> str:
        compression = str(value or "zstd").lower()
        if compression not in {"zstd", "none"}:
            raise ValueError("audit_shard_compression must be 'zstd' or 'none'")
        return compression

    @field_validator("checkpoint_payload", mode="before")
    @classmethod
    def validate_checkpoint_payload(cls, value: Any) -> str:
        payload = str(value or "compact").lower()
        if payload not in {"compact", "full"}:
            raise ValueError("checkpoint_payload must be 'compact' or 'full'")
        return payload

    @field_validator("cache_persist_policy", mode="before")
    @classmethod
    def validate_cache_persist_policy(cls, value: Any) -> str:
        policy = str(value or "finalize").lower()
        if policy not in {"checkpoint", "finalize", "never"}:
            raise ValueError("cache_persist_policy must be 'checkpoint', 'finalize', or 'never'")
        return policy


class LLMProfileConfig(StrictConfigModel):
    backend: str = Field("local_hf", description="Backend implementation name.")
    model: Optional[str] = Field(None, description="Backend model identifier.")
    tokenizer: Optional[str] = Field(None, description="Optional tokenizer identifier override.")
    api_base: str = Field(
        "https://openrouter.ai/api/v1", description="Hosted backend API base URL."
    )
    api_key_env: str = Field(
        "OPENROUTER_API_KEY", description="Environment variable containing the hosted API key."
    )
    api_key_path: Optional[str] = Field(
        None, description="Optional file containing the hosted API key."
    )
    timeout_secs: float = Field(60.0, description="Per-request hosted timeout in seconds.")
    extra_headers: Dict[str, str] = Field(
        default_factory=dict, description="Additional hosted request headers."
    )
    provider: Dict[str, Any] = Field(
        default_factory=dict, description="Provider-routing options forwarded unchanged."
    )


def _default_llm_profiles() -> Dict[str, LLMProfileConfig]:
    hosted = {
        "backend": "openrouter",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_key_path": "~/.config/openrouter/api_key",
        "timeout_secs": 60.0,
    }
    raw: Dict[str, Dict[str, Any]] = {
        "openrouter_gpt4o_mini": {
            **hosted,
            "model": "openai/gpt-4o-mini",
            "tokenizer": "Xenova/gpt-4o",
        },
        "openrouter_claude_sonnet_46": {
            **hosted,
            "model": "anthropic/claude-sonnet-4.6",
        },
        "openrouter_gpt5": {**hosted, "model": "openai/gpt-5"},
        "openrouter_gpt_oss_120b": {**hosted, "model": "openai/gpt-oss-120b"},
        "openrouter_qwen35_122b": {
            **hosted,
            "model": "qwen/qwen3.5-122b-a10b",
            "tokenizer": "Qwen/Qwen3.5-122B-A10B",
        },
        "openrouter_qwen3_235b": {
            **hosted,
            "model": "qwen/qwen3-235b-a22b-2507",
            "tokenizer": "Qwen/Qwen3-235B-A22B-2507",
        },
        "openrouter_llama33_70b": {
            **hosted,
            "model": "meta-llama/llama-3.3-70b-instruct",
            "tokenizer": "meta-llama/Llama-3.3-70B-Instruct",
        },
        "openrouter_deepseek_chat_v31": {
            **hosted,
            "model": "deepseek/deepseek-chat-v3.1",
            "tokenizer": "deepseek-ai/DeepSeek-V3.1",
        },
        "local_verbaliser_default": {
            "backend": "local_hf",
            "model": "Qwen/Qwen2.5-3B-Instruct",
        },
        "local_llm_default": {
            "backend": "local_hf",
            "model": "Qwen/Qwen2.5-7B-Instruct",
        },
    }
    return {name: LLMProfileConfig(**payload) for name, payload in raw.items()}


class LLMRoutingConfig(StrictConfigModel):
    default_profile: Optional[str] = Field(
        "openrouter_gpt4o_mini", description="Default profile for tasks without an override."
    )
    verbaliser_profile: Optional[str] = Field(
        "openrouter_gpt4o_mini", description="Primary verbaliser profile."
    )
    summary_profile: Optional[str] = Field(
        "openrouter_gpt4o_mini", description="Primary summary or pair-brief profile."
    )
    decision_profile: Optional[str] = Field(
        "openrouter_gpt4o_mini", description="Primary binary-decision profile."
    )
    rationale_profile: Optional[str] = Field(
        "openrouter_gpt4o_mini", description="Primary rationale profile."
    )
    verbaliser_fallback_profile: Optional[str] = Field(
        "local_verbaliser_default", description="Fallback verbaliser profile."
    )
    summary_fallback_profile: Optional[str] = Field(
        None, description="Optional task-specific summary fallback profile."
    )
    rationale_fallback_profile: Optional[str] = Field(
        None, description="Optional task-specific rationale fallback profile."
    )
    fallback_profile: Optional[str] = Field(
        "local_llm_default", description="Shared local fallback profile."
    )
    decision_fallback_profile: Optional[str] = Field(
        "local_llm_default", description="Fallback binary-decision profile."
    )


class LLMVerbaliserConfig(StrictConfigModel):
    model: Optional[str] = Field(
        "Qwen/Qwen2.5-3B-Instruct", description="Local fallback verbaliser model."
    )
    max_new_tokens: int = Field(64, description="Maximum tokens generated per relation template.")
    temperature: float = Field(0.1, description="Verbaliser generation temperature.")
    top_p: float = Field(0.9, description="Verbaliser nucleus-sampling cutoff.")
    top_k: Optional[int] = Field(None, description="Optional local top-k sampling cutoff.")
    do_sample: bool = Field(False, description="Enable local stochastic sampling.")
    batch_size: int = Field(4, description="Local batch size or hosted concurrency cap.")
    max_retries: int = Field(1, description="Retries after invalid generated templates.")
    exclude_missing_domain_range: bool = Field(
        False, description="Skip relations without domain/range examples."
    )


class LLMConfig(StrictConfigModel):
    profiles: Dict[str, LLMProfileConfig] = Field(
        default_factory=_default_llm_profiles,
        description="Named local and hosted backend profiles.",
    )
    routing: LLMRoutingConfig = Field(
        default=LLMRoutingConfig.model_validate({}),
        description="Task-to-profile routing and fallbacks.",
    )
    verbaliser: LLMVerbaliserConfig = Field(
        default=LLMVerbaliserConfig.model_validate({}),
        description="Relation-template generation controls.",
    )

    @model_validator(mode="before")
    @classmethod
    def merge_default_profiles(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        merged = dict(value)
        defaults = _default_llm_profiles()
        provided = value.get("profiles")
        if provided is not None:
            profiles: Dict[str, Any] = {
                name: profile.model_dump(mode="python") for name, profile in defaults.items()
            }
            for name, payload in dict(provided or {}).items():
                existing = profiles.get(str(name), {})
                profiles[str(name)] = {**existing, **dict(payload or {})}
            merged["profiles"] = profiles
        return merged


class EvaluationConfig(StrictConfigModel):
    """Ordered evaluator selection and backend-specific options."""

    backends: List[str] = Field(
        default_factory=lambda: ["builtin"], description="Evaluator backends run in order."
    )
    k: List[int] = Field(
        default_factory=lambda: [1, 5, 10], description="Ranking cutoffs reported by evaluation."
    )
    bioml: Dict[str, Any] = Field(
        default_factory=dict, description="Options forwarded to the optional Bio-ML evaluator."
    )

    @field_validator("backends", mode="before")
    @classmethod
    def validate_backends(cls, value: Any) -> List[str]:
        names = [str(name).strip().lower() for name in (value or ["builtin"])]
        names = list(dict.fromkeys(name for name in names if name))
        if not names:
            raise ValueError("evaluation.backends must contain at least one backend")
        return names

    @field_validator("k", mode="before")
    @classmethod
    def ensure_unique_k(cls, value: Any) -> List[int]:
        return list(dict.fromkeys(int(item) for item in (value or [])))


class SaveConfig(StrictConfigModel):
    full_explanations_json: bool = Field(
        False,
        description="Assemble the optional full candidate-explanations JSON export.",
    )
    csv: bool = Field(True, description="Save the flattened alignment summary table.")
    stats_csv: bool = Field(True, description="Save one-row run statistics as CSV.")
    append_stats_to_summary: bool = Field(
        True, description="Append human-readable statistics to the summary table."
    )


class PlotConfig(StrictConfigModel):
    bins: int = Field(30, description="Histogram bin count.")
    figsize: Tuple[int, int] = Field((7, 5), description="Plot figure size in inches.")
    dpi: int = Field(300, description="Plot export resolution.")
    kde: bool = Field(False, description="Overlay kernel-density estimates on histograms.")
    alpha: float = Field(0.6, description="Default plot transparency.")


class SanityChecksConfig(StrictConfigModel):
    enabled: bool = Field(True, description="Emit sample rows after dataset construction.")
    n: int = Field(3, description="Number of sanity examples emitted.")
    max_ctx_show: int = Field(3, description="Maximum context sentences shown per side.")
    max_label_show: int = Field(3, description="Maximum label variants shown per side.")


class ExplanationOutputConfig(StrictConfigModel):
    shard_mb: int = Field(32, description="Target compressed explanation-shard size in MiB.")


class RetentionConfig(StrictConfigModel):
    checkpoints: str = Field(
        "latest", description="Checkpoint retention policy: all, latest, or none."
    )


class OutputConfig(StrictConfigModel):
    save: SaveConfig = Field(
        default=SaveConfig.model_validate({}), description="Alignment save switches."
    )
    plots: PlotConfig = Field(
        default=PlotConfig.model_validate({}), description="Shared plot styling."
    )
    sanity_checks: SanityChecksConfig = Field(
        default=SanityChecksConfig.model_validate({}),
        description="Human-readable dataset diagnostics.",
    )
    explanations: ExplanationOutputConfig = Field(
        default=ExplanationOutputConfig.model_validate({}),
        description="Explanation store layout controls.",
    )
    retention: RetentionConfig = Field(
        default=RetentionConfig.model_validate({}), description="Run-artifact retention policies."
    )


def _default_primary_params() -> Dict[str, Any]:
    return {
        "lexical_model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "context_model_name": "BAAI/bge-large-en-v1.5",
        "llm_model_name": "Qwen/Qwen2.5-7B-Instruct",
        "max_input_tokens_hier": 128,
        "max_input_tokens_sim": 256,
        "max_input_tokens_diff": 256,
        "max_input_tokens_attr_item": 96,
        "fp16_inference": True,
        "pooling_method": "mean",
        "label_pair_pooling": "max",
        "max_input_tokens_lexical": 32,
        "max_input_tokens_context": 256,
        "max_total_tokens_llm_summary": 768,
        "max_total_tokens_llm_decision": 640,
        "max_total_tokens_llm_rationale": 896,
        "max_new_tokens_llm": 64,
        "max_new_tokens_llm_rationale": 256,
        "use_lexical": True,
        "use_context": True,
        "use_llm": True,
        "tau": 0.5,
        "gamma": 2.0,
        "beta": 0.8,
        "tau_LLM": 0.5,
        "force_llm_summaries": False,
        "llm_temperature": 0.1,
        "llm_top_p": 0.9,
        "llm_do_sample": False,
        "llm_summary_batch_size": 8,
        "llm_decision_batch_size": 8,
        "llm_rationale_batch_size": 8,
        "hosted_decision_labels": ["A", "B"],
        "hosted_decision_logit_bias": 20.0,
        "profile_llm": False,
        "return_explanations": True,
        "generate_llm_rationales": True,
        "use_llm_calibration": False,
        "llm_calibration_a": None,
        "llm_calibration_b": None,
        "llm_calibration_info": None,
        "cache_dir": None,
        "cache_namespace": None,
        "dataset_signature": None,
        "persist_cache_to_disk": True,
        "max_cached_labels": None,
        "max_cached_contexts": None,
        "max_cached_summaries": None,
        "max_cached_rationales": None,
    }


def _default_selector_params() -> Dict[str, Any]:
    return {
        "enabled": True,
        "strategy": "calibrated_rank_accept",
        "global_only": True,
        "replace_final_score": True,
        "score_mode": "p_match",
        "use_no_match": True,
        "temperature": 0.75,
        "eps": 1.0e-6,
        "support_weight": 0.60,
        "no_match_threshold": 0.55,
        "calibration": {
            "enabled": "auto",
            "min_positive_sources": 50,
            "background_negative_weight": 0.02,
            "background_negative_weight_grid": [0.02, 0.05, 0.10, 0.20, 0.40],
            "validation_fraction": 0.2,
            "validation_folds": 5,
            "l2": 0.001,
            "learning_rate": 0.05,
            "max_epochs": 200,
            "threshold_grid_step": 0.005,
            "accept_objective": "f1",
            "f_beta": 1.5,
            "min_precision": None,
            "min_recall": None,
            "exact_prefiltered_source_policy": "hard_negative",
            "exact_prefiltered_negative_weight": 1.0,
        },
        "llm": {
            "enabled": False,
            "mode": False,
            "ambiguity_margin": 0.08,
            "max_candidates": 5,
            "trigger_acceptance_margin": 0.025,
            "trigger_rank_margin": 0.03,
            "min_confidence": 0.75,
        },
    }


class PipelineEntry(StrictConfigModel):
    name: str = Field(..., description="Registered model component name.")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Component-specific constructor parameters."
    )

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: Any) -> str:
        name = str(value or "").strip()
        if not name:
            raise ValueError("pipeline entry name must be non-empty")
        return name


def default_pipeline_entries() -> List[PipelineEntry]:
    """Return fresh default pipeline entries for models and migration."""

    return [
        PipelineEntry(name="PairAdaptiveSemanticScorer", params=_default_primary_params()),
        PipelineEntry(name="CandidateSetSelector", params=_default_selector_params()),
    ]


class ConfigModel(StrictConfigModel):
    """Fully resolved version-2 configuration model."""

    config_version: int = Field(
        CONFIG_VERSION, description="Configuration schema version; currently exactly 2."
    )
    run: RunConfig = Field(
        default=RunConfig.model_validate({}), description="Run-wide process settings."
    )
    data: DataConfig = Field(
        default=DataConfig.model_validate({}), description="Input and dataset-track selection."
    )
    io: IOConfig = Field(
        default=IOConfig.model_validate({}), description="Input adapters and output formats."
    )
    matching: MatchingConfig = Field(
        default=MatchingConfig.model_validate({}),
        description="Alignment extraction and scoring controls.",
    )
    dataset: DatasetConfig = Field(
        default=DatasetConfig.model_validate({}),
        description="Dataset construction and evidence extraction.",
    )
    candidates: CandidatesConfig = Field(
        default=CandidatesConfig.model_validate({}), description="Candidate retrieval controls."
    )
    pipeline: List[PipelineEntry] = Field(
        default_factory=default_pipeline_entries,
        description="Ordered primary and post-inference model components.",
        min_length=1,
    )
    inference: InferenceConfig = Field(
        default=InferenceConfig.model_validate({}),
        description="Inference, audit, and checkpoint controls.",
    )
    llm: LLMConfig = Field(
        default=LLMConfig.model_validate({}), description="LLM profiles and generation routing."
    )
    evaluation: EvaluationConfig = Field(
        default=EvaluationConfig.model_validate({}),
        description="Evaluation backends and ranking cutoffs.",
    )
    output: OutputConfig = Field(
        default=OutputConfig.model_validate({}),
        description="Saved artifacts, plots, and diagnostics.",
    )

    _resolved_sequence: Optional[List[RegistryParams]] = PrivateAttr(default=None)
    _dataset_component: Optional[Type[IDataset]] = PrivateAttr(default=None)
    _trainer_component: Optional[Type[ITrainer]] = PrivateAttr(default=None)
    _legacy_dataset_track: Optional[DataConfig] = PrivateAttr(default=None)

    @field_validator("config_version")
    @classmethod
    def validate_config_version(cls, value: int) -> int:
        if value > CONFIG_VERSION:
            raise ValueError(
                f"config_version {value} is too new; this Exact-OM build supports version {CONFIG_VERSION}"
            )
        if value != CONFIG_VERSION:
            raise ValueError(f"config_version must be {CONFIG_VERSION}")
        return value

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, warn_v1: bool = True) -> "ConfigModel":
        """Validate a v2 mapping, automatically migrating an unversioned v1 mapping."""

        payload: Mapping[str, Any] = raw
        version = raw.get("config_version")
        if version is None:
            from exact.core.entities.configs.migration import migrate_v1_mapping

            payload, _ = migrate_v1_mapping(raw)
            if warn_v1:
                logging.getLogger("exact.config").warning(
                    "Loaded deprecated v1 configuration; run 'exact config migrate' to write v2."
                )
        elif not isinstance(version, int):
            raise ValueError("config_version must be an integer")
        elif version > CONFIG_VERSION:
            raise ValueError(
                f"config_version {version} is too new; this Exact-OM build supports version {CONFIG_VERSION}"
            )
        model = cls.model_validate(dict(payload))
        legacy_track = raw.get("dataset_track") if version is None else None
        if legacy_track is not None:
            model._legacy_dataset_track = DataConfig.model_validate(legacy_track)
        return model

    @classmethod
    def load_config(cls, file_path: Path) -> "ConfigModel":
        """Load YAML from ``file_path`` and validate it as v1 or v2."""

        from exact.core.entities.configs.yaml_io import load_yaml_mapping

        return cls.from_mapping(load_yaml_mapping(Path(file_path)))

    def resolve_dependencies(self) -> None:
        """Resolve registered pipeline, dataset, and trainer classes without mutating the schema."""

        importlib.import_module("exact.impl").bootstrap_components()
        resolved: List[RegistryParams] = []
        for entry in self.pipeline:
            component = ComponentRegistry.get(ComponentType.MODEL, entry.name)
            resolved.append(
                RegistryParams(
                    component_type=ComponentType.MODEL,
                    name=component,
                    params=dict(entry.params),
                )
            )
        primary = resolved[0]
        dependencies = ComponentRegistry.get_dependency(primary.name.__name__)
        self._dataset_component = ComponentRegistry.get(
            ComponentType.DATASET, dependencies[ComponentType.DATASET]
        )
        self.dataset.bind_runtime_component(self._dataset_component)
        self._trainer_component = ComponentRegistry.get(
            ComponentType.TRAINER, dependencies[ComponentType.TRAINER]
        )
        self._resolved_sequence = resolved

    def get_model_sequence(self) -> List[RegistryParams]:
        """Return ordered runtime specs, resolved to classes after ``resolve_dependencies``."""

        if self._resolved_sequence is not None:
            return list(self._resolved_sequence)
        return [
            RegistryParams(
                component_type=ComponentType.MODEL,
                name=entry.name,
                params=dict(entry.params),
            )
            for entry in self.pipeline
        ]

    def effective_data_config(self) -> DataConfig:
        """Return the canonical v2 data section."""

        return self.data

    @property
    def dataset_track(self) -> Optional[DataConfig]:
        """One-release compatibility view of an explicitly supplied v1 alias."""

        return self._legacy_dataset_track

    def fingerprint(self) -> str:
        """Return a stable fingerprint of the canonical resolved v2 schema dump."""

        payload = self.model_dump(mode="json", by_alias=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def logging_level(self) -> int:
        return int(getattr(logging, self.run.logging_level))

    @property
    def seed(self) -> int:
        return self.run.seed

    @property
    def use_file_cache(self) -> bool:
        return self.run.use_file_cache

    @property
    def k(self) -> List[int]:
        return list(self.evaluation.k)

    @property
    def llm_profiles(self) -> Dict[str, LLMProfileConfig]:
        return self.llm.profiles

    @property
    def llm_routing(self) -> LLMRoutingConfig:
        return self.llm.routing

    @property
    def model(self) -> RegistryParams:
        return self.get_model_sequence()[0]

    @property
    def second_model(self) -> Optional[RegistryParams]:
        sequence = self.get_model_sequence()
        return sequence[1] if len(sequence) > 1 else None

    @property
    def model_chain(self) -> List[RegistryParams]:
        return self.get_model_sequence()

    @property
    def dataset_component(self) -> Optional[Type[IDataset]]:
        return self._dataset_component

    @property
    def trainer_component(self) -> Optional[Type[ITrainer]]:
        return self._trainer_component

    @property
    def trainer(self) -> Optional[Type[ITrainer]]:
        """Resolved trainer class retained for the current alignment runtime."""

        return self._trainer_component

    # Historical runtime code calls ``configs.dataset(...)`` and ``configs.trainer(...)``.
    # The names intentionally remain properties while the v2 schema stores its dataset
    # settings in the pydantic field of the same name.  Pydantic field access wins, so
    # callers should use the compatibility helpers below during the transition.

    @property
    def dataset_params(self) -> DatasetParams:
        values = self.dataset.model_dump(mode="python", exclude={"legacy"})
        values.update(self.dataset.legacy.model_dump(mode="python"))
        values.update(
            {
                "verbaliser_name": self.llm.verbaliser.model,
                "gen_max_new_tokens": self.llm.verbaliser.max_new_tokens,
                "do_sample": self.llm.verbaliser.do_sample,
                "temperature": self.llm.verbaliser.temperature,
                "top_k": self.llm.verbaliser.top_k,
                "top_p": self.llm.verbaliser.top_p,
                "max_verb_gen_retries": self.llm.verbaliser.max_retries,
                "exclude_missing_dr": self.llm.verbaliser.exclude_missing_domain_range,
                "batch_size_verbaliser": self.llm.verbaliser.batch_size,
            }
        )
        return DatasetParams.model_validate(values)

    @property
    def candidates_params(self) -> CandidatesParams:
        return CandidatesParams.model_validate(self.candidates.model_dump(mode="python"))

    @property
    def inference_params(self) -> InferenceParams:
        return InferenceParams.model_validate(self.inference.model_dump(mode="python"))

    @property
    def plot_params(self) -> PlotParams:
        return PlotParams.model_validate(self.output.plots.model_dump(mode="python"))

    @property
    def sanity_check_params(self) -> SanityCheckParams:
        return SanityCheckParams.model_validate(
            {
                "sanity_check": self.output.sanity_checks.enabled,
                "n": self.output.sanity_checks.n,
                "max_ctx_show": self.output.sanity_checks.max_ctx_show,
                "max_label_show": self.output.sanity_checks.max_label_show,
            }
        )

    @property
    def alignment_params(self) -> AlignmentParams:
        return AlignmentParams.model_validate(
            {
                "threshold": self.matching.threshold,
                "cardinality": self.matching.cardinality,
                "target_cardinality": self.matching.target_cardinality,
                "review_low": self.matching.review_low,
                "review_high": self.matching.review_high,
                "save_json": self.output.save.full_explanations_json,
                "save_csv": self.output.save.csv,
                "save_stats_csv": self.output.save.stats_csv,
                "append_stats_to_summary_csv": self.output.save.append_stats_to_summary,
            }
        )

    @property
    def dataset_runtime(self) -> Optional[Type[IDataset]]:
        """Resolved dataset class for new consumers."""

        return self._dataset_component

    @property
    def trainer_runtime(self) -> Optional[Type[ITrainer]]:
        """Resolved trainer class for new consumers."""

        return self._trainer_component


__all__ = [
    "AlignmentParams",
    "CandidatesConfig",
    "CandidatesParams",
    "CONFIG_VERSION",
    "ConfigModel",
    "DEFAULT_CONFIG_PATH",
    "DataConfig",
    "DatasetConfig",
    "DatasetParams",
    "EvaluationConfig",
    "InferenceConfig",
    "InferenceParams",
    "IOConfig",
    "LLMProfileConfig",
    "LLMRoutingConfig",
    "MatchingConfig",
    "ModelChainEntry",
    "ModelParams",
    "OutputConfig",
    "PipelineEntry",
    "PlotParams",
    "RegistryParams",
    "RunConfig",
    "SanityCheckParams",
    "SecondModelParams",
    "default_pipeline_entries",
]
