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

class DatasetParams(BaseModel):
    # General Params
    num_workers: Optional[int] = Field(config["dataset_params"]["num_workers"])
    filter_exact_matches: bool = Field(config["dataset_params"]["filter_exact_matches"])

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


class AlignmentParams(BaseModel):
    threshold: Optional[float] = Field(config["alignment_params"]["threshold"])
    cardinality: Optional[int] = Field(config["alignment_params"]["cardinality"])
    save_json: bool = Field(config["alignment_params"]["save_json"])
    save_csv: bool = Field(config["alignment_params"]["save_csv"])
    save_stats_csv: bool = Field(config["alignment_params"]["save_stats_csv"])
    append_stats_to_summary_csv: bool = Field(config["alignment_params"]["append_stats_to_summary_csv"])
    review_low: Optional[float] = Field(config["alignment_params"]["review_low"])
    review_high: Optional[float] = Field(config["alignment_params"]["review_high"])

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
    model: ModelParams = ModelParams()
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
        dependencies = ComponentRegistry.get_dependency(self.model.name.__name__)
        self.dataset = ComponentRegistry.get(ComponentType.DATASET, dependencies[ComponentType.DATASET])
        self.trainer = ComponentRegistry.get(ComponentType.TRAINER, dependencies[ComponentType.TRAINER])

    @classmethod
    def load_config(cls, file_path: Path) -> "ConfigModel":
        yaml_config = read_yaml(file_path)

        dataset_params = DatasetParams(**yaml_config.get("dataset_params", {}))
        candidates_params = CandidatesParams(**yaml_config.get("candidates_params", {}))
        plot_params = PlotParams(**yaml_config.get("plot_params", {}))
        sanity_check_params = SanityCheckParams(**yaml_config.get("sanity_check_params", {}))
        alignment_params = AlignmentParams(**yaml_config.get("alignment_params", {}))
        inference_params = InferenceParams(**yaml_config.get("inference_params", {}))
        model_params = ModelParams(**yaml_config.get("model", {}))

        # filter config for set keys
        filtered_config = {
            k: v
            for k, v in yaml_config.items()
            if v is not None
            and k in cls.model_fields
            and k not in ["dataset_params", "candidates_params", "alignment_params", "inference_params", "model", "plot_params", "sanity_check_params"]
        }

        return cls(
            dataset_params=dataset_params,
            candidates_params=candidates_params,
            plot_params=plot_params,
            sanity_check_params=sanity_check_params,
            alignment_params=alignment_params,
            inference_params=inference_params,
            model=model_params,
            **filtered_config,
        )


