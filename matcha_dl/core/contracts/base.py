
from abc import ABC
from matcha_dl.core.entities.configs import ComponentRegistry

#TODO on debug check if all modules are being curretly registered, non imported implementations might not be registered.


class SelfRegisteringComponent(ABC):
    """Base class for self-registering components."""
    component_type: str  # To be defined in subclasses

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Ensure the subclass defines the component_type
        if not hasattr(cls, "component_type") or not cls.component_type:
            raise TypeError(
                f"Class '{cls.__name__}' must define a 'component_type' attribute "
                f"to specify its registry type."
            )

        # Determine the fully qualified import path of the subclass
        module_name = cls.__module__
        class_name = cls.__name__
        import_path = f"{module_name}.{class_name}"

        # Register the class using the ComponentRegistry
        ComponentRegistry.register(cls.component_type, class_name, import_path)
