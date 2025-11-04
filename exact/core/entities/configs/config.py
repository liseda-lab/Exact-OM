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
    pre_filtering: Optional[bool] = Field(config["dataset_params"]["pre_filtering"])
    sanity_check: Optional[bool] = Field(config["dataset_params"]["sanity_check"])

    # Context Tabular Params
    n_hops: Optional[int] = Field(config["dataset_params"]["n_hops"])
    context_method: Optional[ContextMethod] = Field(config["dataset_params"]["context_method"], validate_default=True)
    context_hop_penalty: Optional[float] = Field(config["dataset_params"]["context_hop_penalty"])
    context_token_ratio: Optional[float] = Field(config["dataset_params"]["context_token_ratio"])
    context_safety: Optional[float] = Field(config["dataset_params"]["context_safety"])
    best_path_src_method: Optional[BestPathMethod] = Field(config["dataset_params"]["best_path_src_method"], validate_default=True)
    verbaliser_name: Optional[str] = Field(config["dataset_params"]["verbaliser_name"])
    gen_max_new_tokens: Optional[int] = Field(config["dataset_params"]["gen_max_new_tokens"])
    do_sample: Optional[bool] = Field(config["dataset_params"]["do_sample"])
    temperature: Optional[float] = Field(config["dataset_params"]["temperature"])
    top_k: Optional[int] = Field(config["dataset_params"]["top_k"])
    top_p: Optional[float] = Field(config["dataset_params"]["top_p"])
    num_beams: Optional[int] = Field(config["dataset_params"]["num_beams"])
    batch_size: Optional[int] = Field(config["dataset_params"]["batch_size"])
    cache_chunk_size: Optional[int] = Field(config["dataset_params"]["cache_chunk_size"])
    delimiter: Optional[str] = Field(config["dataset_params"]["delimiter"])
    exclude_missing_dr: Optional[bool] = Field(config["dataset_params"]["exclude_missing_dr"])
    encoding_max_length: Optional[int] = Field(config["dataset_params"]["encoding_max_length"])
    gen_max_length: Optional[int] = Field(config["dataset_params"]["gen_max_length"])
    summariser_name: Optional[str] = Field(config["dataset_params"]["summariser_name"])
    max_verb_gen_retries: Optional[int] = Field(config["dataset_params"]["max_verb_gen_retries"])
    smallest_batch_first: Optional[bool] = Field(config["dataset_params"]["smallest_batch_first"])
    only_taxonomy: Optional[bool] = Field(config["dataset_params"]["only_taxonomy"])
    sanity_check_n_samples: Optional[int] = Field(config["dataset_params"]["sanity_check_n_samples"])
    all_labels: Optional[bool] = Field(config["dataset_params"]["all_labels"])
        

class InferenceParams(BaseModel):
    epochs: int = Field(config["inference_params"]["epochs"])
    batch_size: int = Field(config["inference_params"]["batch_size"])
    num_workers: int = Field(config["inference_params"]["num_workers"])
    log_every: int = Field(config["inference_params"]["log_every"])
    mixed_precision: bool = Field(config["inference_params"]["mixed_precision"])


class AlignmentParams(BaseModel):
    threshold: Optional[float] = Field(config["alignment_params"]["threshold"])
    cardinality: Optional[int] = Field(config["alignment_params"]["cardinality"])

class ConfigModel(BaseModel):

    seed: int = Field(config["seed"])
    logging_level: int = Field(config["logging_level"], validate_default=True)
    use_file_cache: bool = Field(config["use_file_cache"])
    dataset_params: DatasetParams = DatasetParams()
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
        alignment_params = AlignmentParams(**yaml_config.get("alignment_params", {}))
        inference_params = InferenceParams(**yaml_config.get("inference_params", {}))
        model_params = ModelParams(**yaml_config.get("model", {}))

        # filter config for set keys
        filtered_config = {
            k: v
            for k, v in yaml_config.items()
            if v is not None
            and k in cls.model_fields
            and k not in ["dataset_params", "alignment_params", "inference_params", "model"]
        }

        return cls(
            dataset_params=dataset_params,
            alignment_params=alignment_params,
            inference_params=inference_params,
            model=model_params,
            **filtered_config,
        )


