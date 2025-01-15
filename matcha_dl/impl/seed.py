
import random
import numpy as np
import torch
from matcha_dl.core.contracts.seed import SeedSetterContract

class SeedSetter(SeedSetterContract):
    def __init__(self, seed: int):
        self.seed = seed
        self.set_seed()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False