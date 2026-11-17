"""Contract for reproducible random-seed initialization."""

from abc import ABC, abstractmethod

from exact.core.contracts.base import SelfRegisteringComponent
from exact.core.entities.registry import ComponentType


class ISeedSetter(SelfRegisteringComponent, ABC):
    """Interface implemented by seed initializers."""

    component_type = ComponentType.SEED_SETTER

    @abstractmethod
    def set_seed(self) -> None:
        """Apply the configured seed to every supported random-number generator."""

        raise NotImplementedError
