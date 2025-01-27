import torch

from matcha_dl.core.contracts import SelfRegisteringComponent

nn = torch.nn
Tensor = torch.Tensor

MODEL = "model"


class IModel(nn.Module, SelfRegisteringComponent):
    def __init__(self, **kwargs):
        super(IModel, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass
