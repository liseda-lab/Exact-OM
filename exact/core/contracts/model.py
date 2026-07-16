"""Torch model contract for registered scoring components."""

import torch

from exact.core.contracts import SelfRegisteringComponent
from exact.core.entities.registry import ComponentType

nn = torch.nn
Tensor = torch.Tensor


class IModel(nn.Module, SelfRegisteringComponent):
    """Base class for trainable and inference-only Exact models."""

    component_type = ComponentType.MODEL

    def __init__(self, **kwargs):
        """Initialize the underlying Torch module."""

        super(IModel, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Compute model output for ``x`` in concrete subclasses."""

        pass
