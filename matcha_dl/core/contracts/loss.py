from abc import abstractmethod

import torch
import torch.nn as nn

from matcha_dl.core.contracts import SelfRegisteringComponent
from matcha_dl.core.entities.registry import ComponentType

Tensor = torch.Tensor


class ILoss(nn.Module, SelfRegisteringComponent):

    component_type = ComponentType.LOSS
    """
    Abstract base class for torch loss functions.
    """

    def __init__(self, **kwargs):
        super(ILoss, self).__init__()

    @abstractmethod
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of the loss function.

        Args:
            input (Tensor): The input tensor.
            target (Tensor): The target tensor.

        Returns:
            Tensor: The loss value as a tensor.
        """
        pass
