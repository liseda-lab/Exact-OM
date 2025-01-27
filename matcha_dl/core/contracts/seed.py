
from abc import ABC, abstractmethod

class ISeedSetter(ABC):
    @abstractmethod
    def set_seed(self, seed: int):
        pass