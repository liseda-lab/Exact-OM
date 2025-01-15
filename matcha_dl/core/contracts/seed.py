
from abc import ABC, abstractmethod

class SeedSetterContract(ABC):
    @abstractmethod
    def set_seed(self, seed: int):
        pass