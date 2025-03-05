
from abc import ABC
from typing import Optional, TYPE_CHECKING

import logging

if TYPE_CHECKING:
    from matcha_dl.core.entities.registry import ComponentType

#TODO on debug check if all modules are being curretly registered, non imported implementations might not be registered.


class SelfRegisteringComponent(ABC):
    
    """Base class for self-registering components."""
    component_type: 'ComponentType'  # To be defined in subclasses

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
        
        from matcha_dl.core.entities.registry import ComponentRegistry
        ComponentRegistry.register(cls.component_type, class_name, import_path)


class LoggingClass:
    """Base class for logging classes."""
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger

    def log(self, msg: str, level: str = "info", traceback: bool = False):
        if self.logger is not None:
            log_method = getattr(self.logger, level, None)
            if callable(log_method):
                if traceback:
                    log_method(msg, exc_info=True)
                else:
                    log_method(msg)
            else:
                self.logger.error(f"Invalid log level: {level}")
        else:
            print(msg)
