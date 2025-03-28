import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Type, Union, List, Dict
import itertools
import random

from pydantic import BaseModel, Field, field_validator, model_validator

from matcha_dl import config, read_yaml
from matcha_dl.core.entities.registry import ComponentType, ComponentRegistry
from matcha_dl.core.entities.configs.dataset import Separator, ComparisonType, ContextType, ContextSemantics, Likelihood
from matcha_dl.core.entities.configs.matcha import Sampler, Matchers

from matcha_dl.core.contracts.stopper import IStopper
from matcha_dl.core.contracts.model import IModel
from matcha_dl.core.contracts.loss import ILoss
from matcha_dl.core.contracts.optimizer import IOptimizer
from matcha_dl.core.contracts.trainer import ITrainer
from matcha_dl.core.contracts.dataset import IDataset


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
    
class StoppingParams(RegistryParams):
    component_type: ComponentType = ComponentType.STOPPER
    name: Optional[Type[IStopper]] = Field(config["stopper"]["name"], validate_default=True)
    params: dict = Field(config["stopper"]["params"])

class ModelParams(RegistryParams):
    component_type: ComponentType = ComponentType.MODEL
    name: Type[IModel] = Field(config["model"]["name"], validate_default=True)
    params: dict = Field(config["model"]["params"])

class LossParams(RegistryParams):
    component_type: ComponentType = ComponentType.LOSS
    name: Type[ILoss] = Field(config["loss"]["name"], validate_default=True)
    params: dict = Field(config["loss"]["params"])

class OptimizerParams(RegistryParams):
    component_type: ComponentType = ComponentType.OPTIMIZER
    name: Type[IOptimizer] = Field(config["optimizer"]["name"], validate_default=True)
    params: dict = Field(config["optimizer"]["params"])

class MatchaParams(BaseModel):
    max_heap: str = Field(config["matcha_params"]["max_heap"])
    threshold: float = Field(config["matcha_params"]["threshold"])
    matchers: List[Matchers] = Field(config["matcha_params"]["matchers"])
    negcardinality: int = Field(config["matcha_params"]["negcardinality"])
    negthreshold: float = Field(config["matcha_params"]["negthreshold"])
    samplers: Sampler = Field(config["matcha_params"]["samplers"])
    calculate_scores: bool = Field(config["matcha_params"]["calculate_scores"])

class PlotNegativesParams(BaseModel):
    enabled: bool = Field(config["plot_negatives"]["enabled"])
    figsize: Tuple[int, int] = Field(config["plot_negatives"]["figsize"])
    kde: bool = Field(config["plot_negatives"]["kde"])
    bins: int = Field(config["plot_negatives"]["bins"])
    color: str = Field(config["plot_negatives"]["color"])
    alpha: float = Field(config["plot_negatives"]["alpha"])
    dpi: int = Field(config["plot_negatives"]["dpi"])
    grid: bool = Field(config["plot_negatives"]["grid"])

class DatasetParams(BaseModel):
    validation_set: Optional[float] = Field(config["dataset_params"]["validation_set"])
    example: Optional[List[bool]] = Field(config["dataset_params"]["example"])
    positive_examples: Optional[List[int]] = Field(config["dataset_params"]["positive_examples"])
    negative_examples: Optional[List[int]] = Field(config["dataset_params"]["negative_examples"])
    task_context: Optional[List[bool]] = Field(config["dataset_params"]["task_context"])
    separator: Optional[List[Separator]] = Field(config["dataset_params"]["separator"])
    comparison_type: Optional[List[ComparisonType]] = Field(config["dataset_params"]["comparison_type"])
    label_cardinality: Optional[List[int]] = Field(config["dataset_params"]["label_cardinality"])
    context_type: Optional[List[Optional[ContextType]]] = Field(config["dataset_params"]["context_type"])
    context_cardinality: Optional[List[int]] = Field(config["dataset_params"]["context_cardinality"])
    context_semantics: Optional[List[Optional[ContextSemantics]]] = Field(config["dataset_params"]["context_semantics"])
    likelihood: Optional[List[Likelihood]] = Field(config["dataset_params"]["likelihood"])

    # TODO Add more checks to dataset params
    @field_validator('validation_set', mode="after")
    def validate_validation_set(cls, v: Optional[float]) -> Optional[float]:
        "Ensure validation_set is between 0 and 1."
        if v is not None and (v < 0 or v > 1):
            raise ValueError("validation_set must be between 0 and 1.")
        return v
    
    @field_validator('label_cardinality', 'context_cardinality', 'positive_examples', 'negative_examples',  mode="after")
    def validate_ints(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        """Ensure cardinality lists are either None or contain non-negative integers."""
        if v is not None and any(x < 0 for x in v):
            raise ValueError("Integer params must contain only non-negative integers.")
        return v

    @model_validator(mode="after")
    def validate_prompt_params(self):
        
        ##Ensure all list fields have the same length.
        fields_to_check = [
            'example', 'positive_examples', 'negative_examples', 
            'task_context', 'separator', 'comparison_type', 
            'label_cardinality', 'context_type', 
            'context_cardinality', 'context_semantics', 'likelihood'
        ]

        lengths = [len(getattr(self, field)) for field in fields_to_check if getattr(self, field) is not None]
        if len(set(lengths)) > 1:
            raise ValueError("All lists in DatasetParams must have the same length.")
        
        # Ensure that context_type and context_semantics are not both provided if one is provided
        if self.context_type is not None and self.context_semantics is not None:
            for ct, cs in zip(self.context_type, self.context_semantics):
                if ct is None and cs is not None:
                    raise ValueError("If context_semantics is provided, context_type must also be provided.")
                if ct is not None and cs is None:
                    raise ValueError("If context_type is provided, context_semantics must also be provided.")
        if self.context_type is None and self.context_semantics is not None:
            raise ValueError("If context_semantics is provided, context_type must also be provided.")
        if self.context_type is not None and self.context_semantics is None:
            raise ValueError("If context_type is provided, context_semantics must also be provided.")
            
        
        return self
        

class TrainingParams(BaseModel):
    epochs: int = Field(config["training_params"]["epochs"])
    batch_size: int = Field(config["training_params"]["batch_size"])
    num_workers: int = Field(config["training_params"]["num_workers"])
    shuffle: bool = Field(config["training_params"]["shuffle"])
    val_every: Optional[int] = Field(config["training_params"]["val_every"])
    save_interval: Optional[int] = Field(config["training_params"]["save_interval"])
    gradient_accumulation_steps: int = Field(config["training_params"]["gradient_accumulation_steps"])
    mixed_precision: bool = Field(config["training_params"]["mixed_precision"])

class AlignmentParams(BaseModel):
    threshold: Optional[float] = Field(config["alignment_params"]["threshold"])
    cardinality: Optional[int] = Field(config["alignment_params"]["cardinality"])

class ConfigModel(BaseModel):

    seed: int = Field(config["seed"])
    logging_level: int = Field(config["logging_level"], validate_default=True)
    use_file_cache: bool = Field(config["use_file_cache"])
    use_last_checkpoint: bool = Field(config["use_last_checkpoint"])
    skip_training_if_checkpoint: bool = Field(config["skip_training_if_checkpoint"])
    matcha_params: MatchaParams = MatchaParams()
    plot_negatives_params: PlotNegativesParams = PlotNegativesParams()
    dataset_params: DatasetParams = DatasetParams()
    alignment_params: AlignmentParams = AlignmentParams()
    training_params: TrainingParams = TrainingParams()
    stopper: StoppingParams = StoppingParams()
    model: ModelParams = ModelParams()
    loss: LossParams = LossParams()
    optimizer: OptimizerParams = OptimizerParams()
    k: Optional[List[int]] = Field(config["k"])
    dataset: Optional[Type[IDataset]] = None
    trainer: Optional[Type[ITrainer]] = None

    @field_validator("logging_level", mode="before")
    def parse_logging_level(logging_level: str) -> int:
        return getattr(logging, logging_level.upper())
    
    @field_validator('k', mode="before")
    def ensure_unique_k(cls, k):
        return list(set(k))
    
    # TODO on register have the model register if it requires scores or not, this way this check can be more generic
    # @model_validator(mode="after")
    # def validate_calculate_scores(cls) -> "ConfigModel":
    #     """Ensure calculate_scores is set to True if model is MlpClassifier."""
    #     if cls.model.name.__name__ == "MlpClassifier" and not cls.matcha_params.calculate_scores:
    #         raise ValueError("calculate_scores must be True for MlpClassifier.")
    #     return cls
    
    def resolve_dependencies(self) -> None:
        dependencies = ComponentRegistry.get_dependency(self.model.name.__name__)
        self.dataset = ComponentRegistry.get(ComponentType.DATASET, dependencies[ComponentType.DATASET])
        self.trainer = ComponentRegistry.get(ComponentType.TRAINER, dependencies[ComponentType.TRAINER])

    @classmethod
    def load_config(cls, file_path: Path) -> "ConfigModel":
        yaml_config = read_yaml(file_path)

        matcha_params = MatchaParams(**yaml_config.get("matcha_params", {}))
        plot_negatives_params = PlotNegativesParams(**yaml_config.get("plot_negatives", {}))
        dataset_params = DatasetParams(**yaml_config.get("dataset_params", {}))
        alignment_params = AlignmentParams(**yaml_config.get("alignment_params", {}))
        training_params = TrainingParams(**yaml_config.get("training_params", {}))
        stopping_params = StoppingParams(**yaml_config.get("stopper", {}))
        model_params = ModelParams(**yaml_config.get("model", {}))
        loss_params = LossParams(**yaml_config.get("loss", {}))
        optimizer_params = OptimizerParams(**yaml_config.get("optimizer", {}))

        # filter config for set keys
        filtered_config = {
            k: v
            for k, v in yaml_config.items()
            if v is not None
            and k in cls.model_fields
            and k not in ["matcha_params", "plot_negatives", "dataset_params", "alignment_params", "training_params", "dataset_params", "alignment_params", "stopper", "model", "loss", "optimizer"]
        }

        return cls(
            matcha_params=matcha_params,
            plot_negatives_params=plot_negatives_params,
            dataset_params=dataset_params,
            alignment_params=alignment_params,
            training_params=training_params,
            stopper=stopping_params,
            model=model_params,
            loss=loss_params,
            optimize=optimizer_params,
            **filtered_config,
        )
    

class ConfigTuner:
    def __init__(self, config_file: Path, ignore_params: Optional[List[str]] = None):
        self.config_file = config_file
        self.ignore_params = ignore_params if ignore_params else []
        self.yaml_config = read_yaml(config_file)

    def _is_tunable_param(self, key: str, value: Any) -> bool:
        return isinstance(value, list) and key not in self.ignore_params

    def _extract_tunable_params(self, config: Dict[str, Any], parent_key: str = '') -> Dict[str, List[Any]]:
        tunable_params = {}
        for k, v in config.items():
            full_key = f"{parent_key}.{k}" if parent_key else k
            if self._is_tunable_param(full_key, v):
                tunable_params[full_key] = v
            elif isinstance(v, dict):
                tunable_params.update(self._extract_tunable_params(v, full_key))
        return tunable_params

    def _set_nested_value(self, config: Dict[str, Any], keys: List[str], value: Any) -> None:
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value

    def _generate_tuning_combinations(self, tunable_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        keys, values = zip(*tunable_params.items())
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        return combinations

    def _generate_random_combinations(self, tunable_params: Dict[str, List[Any]], n: int) -> List[Dict[str, Any]]:
        keys = list(tunable_params.keys())
        combinations = []
        for _ in range(n):
            combination = {key: random.choice(tunable_params[key]) for key in keys}
            combinations.append(combination)
        return combinations

    def _apply_combinations(self, base_config: Dict[str, Any], combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        config_models = []
        for combination in combinations:
            config_copy = base_config.copy()
            for key, value in combination.items():
                keys = key.split('.')
                self._set_nested_value(config_copy, keys, value)
            config_models.append(config_copy)
        return config_models

    def load_tuned_all_configs(self) -> List[ConfigModel]:
        tunable_params = self._extract_tunable_params(self.yaml_config)
        static_params = {k: v for k, v in self.yaml_config.items() if k not in tunable_params}

        tuning_combinations = self._generate_tuning_combinations(tunable_params)
        combined_configs = self._apply_combinations(static_params, tuning_combinations)

        config_models = [ConfigModel(**config) for config in combined_configs]
        return config_models

    def load_random_tuned_configs(self, n: int) -> List[ConfigModel]:
        tunable_params = self._extract_tunable_params(self.yaml_config)
        static_params = {k: v for k, v in self.yaml_config.items() if k not in tunable_params}

        random_combinations = self._generate_random_combinations(tunable_params, n)
        combined_configs = self._apply_combinations(static_params, random_combinations)

        config_models = [ConfigModel(**config) for config in combined_configs]
        return config_models
    
    def load_tuned_config(self, max_combinations: Optional[int] = None) -> List[ConfigModel]:
        if max_combinations:
            return self.load_random_tuned_configs(max_combinations)
        return self.load_tuned_all_configs()


