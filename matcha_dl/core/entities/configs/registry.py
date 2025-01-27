from typing import Type, Dict, List, Any
from matcha_dl.core.contracts import SelfRegisteringComponent


class ComponentRegistry:
    """A registry for managing components by type, with lazy loading and type validation."""
    _registries: Dict[str, Dict[str, str]] = {}  # Store import paths for lazy loading
    _type_validators: Dict[str, Type[SelfRegisteringComponent]] = {}  # Validators
    _dependencies: Dict[str, Dict[str, str]] = {}  # Dependency mappings

    @classmethod
    def register_validator(cls, component_type: str, base_class: Type[SelfRegisteringComponent]):
        """Register a base class validator for a specific component type."""
        cls._type_validators[component_type] = base_class

    @classmethod
    def register(cls, component_type: str, name: str, component_path: str):
        """
        Register a component under a specific type with validation.
        
        Parameters:
            component_type: The type of component (e.g., 'model', 'loss').
            name: The name of the component.
            component_path: The Python import path of the component (e.g., 'my_package.models.ResNet').
        """
        # Validate the component type
        if component_type in cls._type_validators:
            base_class = cls._type_validators[component_type]
            module_name, class_name = component_path.rsplit(".", 1)
            component_class = cls._load_class(module_name, class_name)

            # Ensure the component class is a subclass of the validator
            if not issubclass(component_class, base_class):
                raise TypeError(
                    f"Cannot register component '{name}' under type '{component_type}': "
                    f"it must inherit from '{base_class.__name__}'."
                )

        # Register the component
        if component_type not in cls._registries:
            cls._registries[component_type] = {}
        cls._registries[component_type][name] = component_path

    @classmethod
    def _load_class(cls, module_name: str, class_name: str) -> Type[Any]:
        """Dynamically load a class from its module and name."""
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @classmethod
    def get(cls, component_type: str, name: str) -> Type[Any]:
        """
        Retrieve a registered component, loading it lazily if necessary.
        
        Parameters:
            component_type: The type of component to retrieve (e.g., 'model').
            name: The name of the component to retrieve.
        
        Returns:
            The class of the requested component.
        """
        if component_type not in cls._registries:
            available_types = ", ".join(cls._registries.keys())
            raise ValueError(
                f"Component type '{component_type}' is not recognized.\n"
                f"Available types: {available_types or 'None'}."
            )

        if name not in cls._registries[component_type]:
            available_components = ", ".join(cls._registries[component_type].keys())
            raise ValueError(
                f"Component '{name}' is not registered under type '{component_type}'.\n"
                f"Available components: {available_components or 'None'}."
            )

        # Retrieve the component path
        component_path = cls._registries[component_type][name]

        # Lazily load the class
        module_name, class_name = component_path.rsplit(".", 1)
        return cls._load_class(module_name, class_name)
    
    @classmethod
    def register_dependency(cls, model_name: str, dependencies: Dict[str, str]):
        """
        Register a dependency mapping for a model.
        
        Parameters:
            model_name: The name of the model.
            dependencies: A dictionary containing dependency mappings, e.g., {"dataset": "DatasetName", "trainer": "TrainerName"}.
        """
        cls._dependencies[model_name] = dependencies

    @classmethod
    def get_dependency(cls, model_name: str) -> Dict[str, str]:
        """
        Retrieve the dependency mapping for a model.
        
        Parameters:
            model_name: The name of the model.
        
        Returns:
            A dictionary containing the dependency mapping for the model.
        """
        if model_name not in cls._dependencies:
            raise ValueError(
                f"Dependencies for model '{model_name}' are not registered.\n"
                f"Available models: {', '.join(cls._dependencies.keys()) or 'None'}."
            )
        return cls._dependencies[model_name]

    @classmethod
    def list(cls, component_type: str) -> List[str]:
        """List all registered components of a specific type."""
        if component_type not in cls._registries:
            return []
        return list(cls._registries[component_type].keys())

    @classmethod
    def describe(cls) -> str:
        """Provide a summary of all registered components."""
        description = []
        for component_type, components in cls._registries.items():
            component_list = ", ".join(components.keys())
            description.append(f"{component_type.capitalize()}: {component_list or 'None'}")
        return "\n".join(description)

