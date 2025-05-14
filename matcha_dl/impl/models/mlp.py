import itertools
from typing import List, Optional

from matcha_dl.core.contracts.model import IModel, nn
from matcha_dl.core.values import N_CLASSES


class MlpClassifier(IModel):
    def __init__(self, layers: List[int], n_classes: Optional[int] = None):
        super().__init__()

        if n_classes is None:
            n_classes = N_CLASSES

        seq = []
        for h in layers:
            # LazyLinear will infer in_features on first forward()
            seq.append(nn.LazyLinear(h))
            seq.append(nn.ReLU())
        self._hidden = nn.Sequential(*seq)

        # also lazy
        self.classify = nn.LazyLinear(n_classes)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        # first time you call this, all LazyLinears learn their in_features
        return self.sigmoid(self.classify(self._hidden(x)))
