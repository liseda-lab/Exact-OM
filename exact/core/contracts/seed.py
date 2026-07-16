"""Contract for reproducible random-seed initialization."""

from abc import ABC, abstractmethod


class ISeedSetter(ABC):
    """Interface implemented by seed initializers."""

    @abstractmethod
    def set_seed(self, seed: int):
        """Apply ``seed`` to every supported random-number generator."""

        pass
