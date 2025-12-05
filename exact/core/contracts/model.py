import torch

from exact.core.contracts import SelfRegisteringComponent
from exact.core.entities.registry import ComponentType

nn = torch.nn
Tensor = torch.Tensor


class IModel(nn.Module, SelfRegisteringComponent):

    component_type = ComponentType.MODEL
    
    def __init__(self, **kwargs):
        super(IModel, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass
