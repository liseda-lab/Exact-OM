from enum import Enum
from typing import Any, Dict, Type

from matcha_dl.core.contracts import SelfRegisteringComponent


class ComponentType(Enum):
    """Enum for valid component types."""
    MODEL = "model"
    DATASET = "dataset"
    TRAINER = "trainer"
    LOSS = "loss"
    OPTIMIZER = "optimizer"
    STOPPER = "stopper"
    METRIC = "metric"


class ComponentRegistry:
    """A registry for managing components and dependencies with lazy loading and type validation."""
    _registries: Dict[ComponentType, Dict[str, str]] = {}  # Store components by type
    _type_validators: Dict[ComponentType, Type[SelfRegisteringComponent]] = {}  # Validators
    _dependencies: Dict[str, Dict[ComponentType, str]] = {}  # Dependency mappings

    @classmethod
    def register_validator(cls, component_type: ComponentType, base_class: Type[SelfRegisteringComponent]):
        """Register a base class validator for a specific component type."""
        if not isinstance(component_type, ComponentType):
            raise ValueError(f"Invalid component type '{component_type}'. Must be a ComponentType Enum value.")

        # Check if a validator is already registered
        if component_type in cls._type_validators:
            if cls._type_validators[component_type] == base_class:
                return  # Ignore duplicate validator registration
            else:
                raise ValueError(
                    f"Validator for component type '{component_type.value}' is already registered with a different class: "
                    f"{cls._type_validators[component_type].__name__}."
                )

        cls._type_validators[component_type] = base_class

    @classmethod
    def register(cls, component_type: ComponentType, name: str, component_path: str):
        """Register a component under a specific type with validation."""
        if not isinstance(component_type, ComponentType):
            raise ValueError(f"Invalid component type '{component_type}'. Must be a ComponentType Enum value.")

        # Check if the component is already registered
        if component_type in cls._registries and name in cls._registries[component_type]:
            if cls._registries[component_type][name] == component_path:
                return  # Ignore duplicate registration
            else:
                raise ValueError(
                    f"Component '{name}' is already registered under type '{component_type.value}' "
                    f"with a different path: {cls._registries[component_type][name]}."
                )

        # Validate component class if a validator exists
        if component_type in cls._type_validators:
            base_class = cls._type_validators[component_type]
            module_name, class_name = component_path.rsplit(".", 1)
            component_class = cls._load_class(module_name, class_name)

            if not issubclass(component_class, base_class):
                raise TypeError(
                    f"Cannot register component '{name}' under type '{component_type.value}': "
                    f"it must inherit from '{base_class.__name__}'."
                )

        # Register the component
        if component_type not in cls._registries:
            cls._registries[component_type] = {}
        cls._registries[component_type][name] = component_path

    @classmethod
    def register_dependency(cls, model_name: str, dependencies: Dict[ComponentType, str]):
        """Register dependencies, ensuring they match registered components."""
        if model_name in cls._dependencies:
            if cls._dependencies[model_name] == dependencies:
                return  # Ignore duplicate dependency registration
            else:
                raise ValueError(
                    f"Dependencies for model '{model_name}' are already registered with different values: "
                    f"{cls._dependencies[model_name]}."
                )

        if model_name not in cls._registries.get(ComponentType.MODEL, {}):
            raise ValueError(
                f"Model '{model_name}' is not registered.\n"
                f"Available models: {', '.join(cls.list(ComponentType.MODEL)) or 'None'}."
            )

        for component_type, component_name in dependencies.items():
            if component_type not in cls._registries or component_name not in cls._registries[component_type]:
                raise ValueError(
                    f"Dependency '{component_name}' is not registered under type '{component_type.value}'.\n"
                    f"Available {component_type.value}s: {', '.join(cls.list(component_type)) or 'None'}."
                )

        cls._dependencies[model_name] = dependencies

    @classmethod
    def get_dependency(cls, model_name: str) -> Dict[ComponentType, str]:
        """Retrieve the dependency mapping for a model."""
        if model_name not in cls._dependencies:
            raise ValueError(
                f"Dependencies for model '{model_name}' are not registered.\n"
                f"Available models: {', '.join(cls._dependencies.keys()) or 'None'}."
            )
        return cls._dependencies[model_name]

    @classmethod
    def _load_class(cls, module_name: str, class_name: str) -> Type[Any]:
        """Dynamically load a class from its module and name."""
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @classmethod
    def get(cls, component_type: ComponentType, name: str) -> Any:
        """Retrieve a registered component, loading it lazily if necessary."""
        if component_type not in cls._registries:
            raise ValueError(f"Component type '{component_type.value}' is not recognized.")

        if name not in cls._registries[component_type]:
            raise ValueError(f"Component '{name}' is not registered under type '{component_type.value}'.")

        # Retrieve the component path
        component_path = cls._registries[component_type][name]

        # Lazily load the class
        module_name, class_name = component_path.rsplit(".", 1)
        return cls._load_class(module_name, class_name)

    @classmethod
    def list(cls, component_type: ComponentType) -> list:
        """List all registered components of a specific type."""
        if component_type not in cls._registries:
            return []
        return list(cls._registries[component_type].keys())
