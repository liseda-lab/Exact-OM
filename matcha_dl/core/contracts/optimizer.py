
from abc import ABC
from matcha_dl.core.contracts.base import SelfRegisteringComponent


class IOptimizer(SelfRegisteringComponent, ABC):
    """
    Base class for all optimizers in the system.
    Automatically registers subclasses with the ComponentRegistry.
    """
    component_type = "optimizer"

    def __init__(self, params, **kwargs):
        """
        Initializes the optimizer with parameters and additional arguments.
        Args:
            params: Parameters to optimize.
            kwargs: Additional optimizer-specific arguments.
        """
        super().__init__()
