import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, field_validator

from matcha_dl import config, read_yaml
from matcha_dl.core.contracts.dataset import IDataset
from matcha_dl.core.contracts.loss import ILoss
from matcha_dl.core.contracts.model import IModel
from matcha_dl.core.contracts.optimizer import IOptimizer
from matcha_dl.core.contracts.stopper import IStopper
from matcha_dl.core.contracts.trainer import ITrainer
from matcha_dl.core.entities.configs import ComponentRegistry, ComponentType

# TODO replace by new loading method through registry


class RegistryParams(BaseModel):
    """Base model for components managed via the ComponentRegistry."""
    component_type: ComponentType
    name: str 
    params: dict

    @field_validator("name", mode="before")
    def validate_and_load(cls, name: str, values) -> Type[Any]:
        """Validate and load the component from the registry."""
        component_type = values.get("component_type")
        if not component_type:
            raise ValueError("Component type is required for registry-based parameters.")
        return ComponentRegistry.get(component_type, name)
    
class StoppingParams(BaseModel):
    component_type: ComponentType = ComponentType.STOPPER
    name: Type[IStopper] = Field(config["stopper"]["stopper"], validate_default=True)
    params: dict = Field(config["stopper"]["params"])

class ModelParams(BaseModel):
    component_type: ComponentType = ComponentType.MODEL
    name: Type[IModel] = Field(config["model"]["model"], validate_default=True)
    params: dict = Field(config["model"]["params"])

class LossParams(BaseModel):
    component_type: ComponentType = ComponentType.LOSS
    name: Type[ILoss] = Field(config["loss"]["loss"], validate_default=True)
    params: dict = Field(config["loss"]["params"])

class OptimizerParams(BaseModel):
    component_type: ComponentType = ComponentType.OPTIMIZER
    name: Type[IOptimizer] = Field(config["optimizer"]["optimizer"], validate_default=True)
    params: dict = Field(config["optimizer"]["params"])

class MatchaParams(BaseModel):
    max_heap: str = Field(config["matcha_params"]["max_heap"])
    threshold: float = Field(config["matcha_params"]["threshold"])
    matchers: list = Field(config["matcha_params"]["matchers"])
    negcardinality: int = Field(config["matcha_params"]["negcardinality"])
    negthreshold: float = Field(config["matcha_params"]["negthreshold"])
    samplers: str = Field(config["matcha_params"]["samplers"])


class TrainingParams(BaseModel):
    epochs: int = Field(config["training_params"]["epochs"])
    batch_size: Optional[int] = Field(config["training_params"]["batch_size"])
    val_every: Optional[int] = Field(config["training_params"]["val_every"])
    save_interval: int = Field(config["training_params"]["save_interval"])

class ConfigModel(BaseModel):
    seed: int = Field(config["seed"])
    device: Union[int, str] = Field(config["device"], validate_default=True)
    logging_level: int = Field(config["logging_level"], validate_default=True)
    use_file_cache: bool = Field(config["use_file_cache"])
    use_last_checkpoint: bool = Field(config["use_last_checkpoint"])
    threshold: float = Field(config["threshold"])
    matcha_params: MatchaParams = MatchaParams()
    training_params: TrainingParams = TrainingParams()
    stopper: StoppingParams = StoppingParams()
    model: ModelParams = ModelParams()
    loss: LossParams = LossParams()
    optimizer: OptimizerParams = OptimizerParams()
    k: Optional[list[int]] = Field(config["k"])
    dataset: Optional[Type[IDataset]] = None
    trainer: Optional[Type[ITrainer]] = None

    @field_validator("logging_level", mode="before")
    def parse_logging_level(logging_level: str) -> int:
        return getattr(logging, logging_level.upper())

    @field_validator("device", mode="before")
    def parse_device(cls, device: Optional[int]) -> Union[int, str]:
        if device is not None:
            return device
        else:
            return "cpu"
        
    def resolve_dependencies(self) -> None:
        dependencies = ComponentRegistry.get_dependency(self.model.name)
        self.dataset = ComponentRegistry.get(ComponentType.DATASET, dependencies[ComponentType.DATASET])
        self.trainer = ComponentRegistry.get(ComponentType.TRAINER, dependencies[ComponentType.TRAINER])

    @classmethod
    def load_config(cls, file_path: Path) -> "ConfigModel":
        yaml_config = read_yaml(file_path)

        matcha_params = MatchaParams(**yaml_config.get("matcha_params", {}))
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
            and k not in ["matcha_params", "training_params", "stopper", "model", "loss", "optimizer"]
        }

        return cls(
            matcha_params=matcha_params,
            training_params=training_params,
            stopper=stopping_params,
            model=model_params,
            loss=loss_params,
            optimize=optimizer_params,
            **filtered_config,
        )
