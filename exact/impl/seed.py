import random

import numpy as np
import torch

from exact.core.contracts.seed import ISeedSetter


class SeedSetter(ISeedSetter):
    """Apply one seed consistently across Python, NumPy, and Torch."""

    def __init__(self, seed: int):
        self.seed = seed
        self.set_seed()

    def set_seed(self) -> None:
        """Seed every random-number generator used by the core pipeline."""

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
