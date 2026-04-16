import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Type, Union, List, Dict
import itertools
import random

from pydantic import BaseModel, Field, field_validator, model_validator

from exact import config, read_yaml
from exact.core.entities.registry import ComponentType, ComponentRegistry
from exact.core.entities.configs.dataset import ContextMethod, BestPathMethod

from exact.core.contracts.model import IModel
from exact.core.contracts.trainer import ITrainer
from exact.core.contracts.dataset import IDataset


class RegistryParams(BaseModel):
    """Base model for components managed via the ComponentRegistry."""
    component_type: ComponentType
    name: str 
    params: dict

    @field_validator("name", mode="before")
    def validate_and_load(cls, name: str, values) -> Type[Any]:
        """Validate and load the component from the registry."""
        if name is None:
            return None
        
        component_type = values.data.get("component_type")
        if not component_type:
            raise ValueError("Component type is required for registry-based parameters.")
        return ComponentRegistry.get(component_type, name)

class ModelParams(RegistryParams):
    component_type: ComponentType = ComponentType.MODEL
    name: Type[IModel] = Field(config["model"]["name"], validate_default=True)
    params: dict = Field(config["model"]["params"])


class SecondModelParams(RegistryParams):
    component_type: ComponentType = ComponentType.MODEL
    name: Optional[Type[IModel]] = Field(config.get("second_model", {}).get("name", None), validate_default=True)
    params: dict = Field(config.get("second_model", {}).get("params", {}))


class ModelChainEntry(RegistryParams):
    component_type: ComponentType = ComponentType.MODEL
    name: Type[IModel]
    params: dict = Field(default_factory=dict)

class DatasetParams(BaseModel):
    # General Params
    num_workers: Optional[int] = Field(config["dataset_params"]["num_workers"])
    filter_exact_matches: bool = Field(config["dataset_params"]["filter_exact_matches"])
    drop_exact_match_sources: bool = Field(config["dataset_params"]["drop_exact_match_sources"])
    candidate_share_k: int = Field(config["dataset_params"].get("candidate_share_k", 1))

    # Context Main Params
    n_hops: int = Field(config["dataset_params"]["n_hops"])
    context_method: ContextMethod = Field(config["dataset_params"]["context_method"], validate_default=True)
    best_path_method: BestPathMethod = Field(config["dataset_params"]["best_path_method"], validate_default=True)
    context_hop_penalty: float = Field(config["dataset_params"]["context_hop_penalty"])
    context_token_ratio: float = Field(config["dataset_params"]["context_token_ratio"])
    context_safety: float = Field(config["dataset_params"]["context_safety"])
    max_input_tokens_context: int = Field(config["dataset_params"]["max_input_tokens_context"])
    only_taxonomy: bool = Field(config["dataset_params"]["only_taxonomy"])
    all_labels: bool = Field(config["dataset_params"]["all_labels"])
    add_connectivity_bridges: bool = Field(config["dataset_params"].get("add_connectivity_bridges", True))
    bridge_max_hops: Optional[int] = Field(config["dataset_params"].get("bridge_max_hops", None))
    reasoner_timeout_secs: float = Field(config["dataset_params"].get("reasoner_timeout_secs", 120))
    reasoner_force_hermit: bool = Field(config["dataset_params"].get("reasoner_force_hermit", False))
    projection_include_literals: bool = Field(config["dataset_params"].get("projection_include_literals", False))
    hierarchical_relation_families: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: dict(config["dataset_params"].get("hierarchical_relation_families", {}))
    )
    hierarchy_max_depth: int = Field(config["dataset_params"].get("hierarchy_max_depth", 2))
    max_hierarchy_triples_per_family: int = Field(config["dataset_params"].get("max_hierarchy_triples_per_family", 6))
    max_object_triples: int = Field(config["dataset_params"].get("max_object_triples", 48))
    max_diff_triples: int = Field(config["dataset_params"].get("max_diff_triples", 24))
    max_attr_items: int = Field(config["dataset_params"].get("max_attr_items", 12))
    pair_adaptive_feature_log_every: int = Field(config["dataset_params"].get("pair_adaptive_feature_log_every", 1000))
    # Verbaliser Params
    verbaliser_name: Optional[str] = Field(config["dataset_params"]["verbaliser_name"])
    gen_max_new_tokens: int = Field(config["dataset_params"]["gen_max_new_tokens"])
    do_sample: bool = Field(config["dataset_params"]["do_sample"])
    temperature: float = Field(config["dataset_params"]["temperature"])
    top_k: Optional[int] = Field(config["dataset_params"]["top_k"])
    top_p: float = Field(config["dataset_params"]["top_p"])
    max_verb_gen_retries: int = Field(config["dataset_params"]["max_verb_gen_retries"])
    # Efficiency Params
    batch_size_verbaliser: int = Field(config["dataset_params"]["batch_size_verbaliser"])
    exclude_missing_dr: bool = Field(config["dataset_params"]["exclude_missing_dr"])
    # Context Agregation Params
    delimiter: str = Field(config["dataset_params"]["delimiter"])
    # Plotting Params
    which: Optional[List[str]] = Field(config["dataset_params"]["which"])

class CandidatesParams(BaseModel):
    lexical_encoder_name: Optional[str] = Field(config["candidates_params"]["lexical_encoder_name"])
    encode_batch_size: int = Field(config["candidates_params"]["encode_batch_size"])
    search_batch_size: int = Field(config["candidates_params"]["search_batch_size"])
    top_k: int = Field(config["candidates_params"]["top_k"])
    use_amp: bool = Field(config["candidates_params"]["use_amp"])

class SanityCheckParams(BaseModel):
    sanity_check: bool = Field(config["sanity_check_params"]["sanity_check"])
    n: int = Field(config["sanity_check_params"]["n"])
    max_ctx_show: int = Field(config["sanity_check_params"]["max_ctx_show"])
    max_label_show: int = Field(config["sanity_check_params"]["max_label_show"])

class PlotParams(BaseModel):
    bins: int = Field(config["plot_params"]["bins"])
    figsize: Tuple[int, int] = Field(tuple(config["plot_params"]["figsize"]))
    dpi: int = Field(config["plot_params"]["dpi"])
    kde: bool = Field(config["plot_params"]["kde"])
    alpha: float = Field(config["plot_params"]["alpha"])


class InferenceParams(BaseModel):
    batch_size: int = Field(config["inference_params"]["batch_size"])
    num_workers: int = Field(config["inference_params"]["num_workers"])
    log_every: int = Field(config["inference_params"]["log_every"])
    mixed_precision: bool = Field(config["inference_params"]["mixed_precision"])
    which: Optional[List[str]] = Field(config["inference_params"]["which"])
    checkpoint_every: int = Field(config["inference_params"]["checkpoint_every"])
    resume_from_checkpoint: bool = Field(config["inference_params"]["resume_from_checkpoint"])
    enable_checkpoints: bool = Field(config["inference_params"]["enable_checkpoints"])
    allow_rationale_toggle_checkpoint_resume: bool = Field(
        config["inference_params"].get("allow_rationale_toggle_checkpoint_resume", False)
    )


class AlignmentParams(BaseModel):
    threshold: Optional[float] = Field(config["alignment_params"]["threshold"])
    cardinality: Optional[int] = Field(config["alignment_params"]["cardinality"])
    save_json: bool = Field(config["alignment_params"]["save_json"])
    save_csv: bool = Field(config["alignment_params"]["save_csv"])
    save_stats_csv: bool = Field(config["alignment_params"]["save_stats_csv"])
    append_stats_to_summary_csv: bool = Field(config["alignment_params"]["append_stats_to_summary_csv"])
    review_low: Optional[float] = Field(config["alignment_params"]["review_low"])
    review_high: Optional[float] = Field(config["alignment_params"]["review_high"])


class LLMProfileConfig(BaseModel):
    backend: str = "local_hf"
    model: Optional[str] = None
    tokenizer: Optional[str] = None
    api_base: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    api_key_path: Optional[str] = None
    timeout_secs: float = 60.0
    extra_headers: Dict[str, str] = Field(default_factory=dict)
    provider: Dict[str, Any] = Field(default_factory=dict)


class LLMRoutingConfig(BaseModel):
    default_profile: Optional[str] = None
    verbaliser_profile: Optional[str] = None
    summary_profile: Optional[str] = None
    decision_profile: Optional[str] = None
    rationale_profile: Optional[str] = None
    verbaliser_fallback_profile: Optional[str] = None
    summary_fallback_profile: Optional[str] = None
    rationale_fallback_profile: Optional[str] = None
    fallback_profile: Optional[str] = None
    decision_fallback_profile: Optional[str] = None

class ConfigModel(BaseModel):

    seed: int = Field(config["seed"])
    logging_level: int = Field(config["logging_level"], validate_default=True)
    use_file_cache: bool = Field(config["use_file_cache"])
    dataset_params: DatasetParams = DatasetParams()
    candidates_params: CandidatesParams = CandidatesParams()
    plot_params: PlotParams = PlotParams()
    sanity_check_params: SanityCheckParams = SanityCheckParams()
    alignment_params: AlignmentParams = AlignmentParams()
    inference_params: InferenceParams = InferenceParams()
    llm_profiles: Dict[str, LLMProfileConfig] = Field(
        default_factory=lambda: {
            name: LLMProfileConfig(**payload)
            for name, payload in (config.get("llm_profiles", {}) or {}).items()
        }
    )
    llm_routing: LLMRoutingConfig = Field(
        default_factory=lambda: LLMRoutingConfig(**(config.get("llm_routing", {}) or {}))
    )
    model: ModelParams = ModelParams()
    second_model: Optional[SecondModelParams] = SecondModelParams()
    model_chain: Optional[List[ModelChainEntry]] = None
    k: Optional[List[int]] = Field(config["k"])
    dataset: Optional[Type[IDataset]] = None
    trainer: Optional[Type[ITrainer]] = None

    @field_validator("logging_level", mode="before")
    def parse_logging_level(logging_level: str) -> int:
        return getattr(logging, logging_level.upper())
    
    @field_validator('k', mode="before")
    def ensure_unique_k(cls, k):
        return list(set(k))
    
    def resolve_dependencies(self) -> None:
        primary_model = None
        if self.model_chain and len(self.model_chain) > 0:
            primary_model = self.model_chain[0]
        else:
            primary_model = self.model
        dependencies = ComponentRegistry.get_dependency(primary_model.name.__name__)
        self.dataset = ComponentRegistry.get(ComponentType.DATASET, dependencies[ComponentType.DATASET])
        self.trainer = ComponentRegistry.get(ComponentType.TRAINER, dependencies[ComponentType.TRAINER])

    @staticmethod
    def _merge_registry_entry(raw_entry: Optional[Dict[str, Any]], default_entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge registry-backed config entries while preserving default params.

        YAML overrides for `model.params` commonly specify only a handful of
        keys. Without this merge, those partial overrides replace the full
        default params dict and silently disable unrelated defaults.
        """
        base = dict(default_entry or {})
        override = dict(raw_entry or {})
        merged = {**base, **{k: v for k, v in override.items() if k != "params"}}
        merged["params"] = {
            **dict(base.get("params") or {}),
            **dict(override.get("params") or {}),
        }
        return merged

    @classmethod
    def load_config(cls, file_path: Path) -> "ConfigModel":
        yaml_config = read_yaml(file_path)

        dataset_params = DatasetParams(**yaml_config.get("dataset_params", {}))
        candidates_params = CandidatesParams(**yaml_config.get("candidates_params", {}))
        plot_params = PlotParams(**yaml_config.get("plot_params", {}))
        sanity_check_params = SanityCheckParams(**yaml_config.get("sanity_check_params", {}))
        alignment_params = AlignmentParams(**yaml_config.get("alignment_params", {}))
        inference_params = InferenceParams(**yaml_config.get("inference_params", {}))
        default_llm_profiles = config.get("llm_profiles", {}) or {}
        merged_llm_profiles = {
            **default_llm_profiles,
            **(yaml_config.get("llm_profiles", {}) or {}),
        }
        llm_profiles = {
            name: LLMProfileConfig(**payload)
            for name, payload in merged_llm_profiles.items()
        }
        llm_routing = LLMRoutingConfig(
            **{
                **(config.get("llm_routing", {}) or {}),
                **(yaml_config.get("llm_routing", {}) or {}),
            }
        )
        model_params = ModelParams(
            **cls._merge_registry_entry(yaml_config.get("model", {}) or {}, config.get("model", {}) or {})
        )
        second_model_raw = yaml_config.get("second_model", {}) or {}
        legacy_second_pass = yaml_config.get("second_pass_params")
        if not second_model_raw and legacy_second_pass is not None:
            second_model_raw = {"name": "SecondPassReranker", "params": legacy_second_pass}
        second_model_params = SecondModelParams(
            **cls._merge_registry_entry(second_model_raw, config.get("second_model", {}) or {})
        )
        model_chain = None
        if "model_chain" in yaml_config:
            entries = yaml_config.get("model_chain") or []
            model_chain = [ModelChainEntry(**cls._merge_registry_entry(entry, None)) for entry in entries]

        # filter config for set keys
        filtered_config = {
            k: v
            for k, v in yaml_config.items()
            if v is not None
            and k in cls.model_fields
            and k not in [
                "dataset_params",
                "candidates_params",
                "alignment_params",
                "inference_params",
                "llm_profiles",
                "llm_routing",
                "model",
                "plot_params",
                "sanity_check_params",
                "second_model",
                "model_chain",
            ]
        }

        return cls(
            dataset_params=dataset_params,
            candidates_params=candidates_params,
            plot_params=plot_params,
            sanity_check_params=sanity_check_params,
            alignment_params=alignment_params,
            inference_params=inference_params,
            llm_profiles=llm_profiles,
            llm_routing=llm_routing,
            model=model_params,
            second_model=second_model_params,
            model_chain=model_chain,
            **filtered_config,
        )

    def get_model_sequence(self) -> List[RegistryParams]:
        """
        Returns the ordered list of model specs to be instantiated.
        If model_chain is provided in the YAML it takes precedence,
        otherwise falls back to the primary model plus optional second_model.
        """
        if self.model_chain:
            return list(self.model_chain)
        chain: List[RegistryParams] = [self.model]
        if self.second_model and self.second_model.name is not None:
            chain.append(self.second_model)
        return chain
