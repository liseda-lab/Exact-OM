
from typing import Union
from matcha_dl.core.contracts.loss import ILoss, Tensor, nn, torch


class ContrastiveLoss(ILoss):
    def __init__(self, margin: float = 1.0, reduction: str = 'mean', device: Union[str, int] = None, **kwargs):
        """
        Contrastive loss for cosine similarity input (converted internally to distance).

        Args:
            margin (float): Margin for dissimilar pairs (range 0–2 for cosine distance).
            reduction (str): Specifies reduction: 'none' | 'mean' | 'sum'.
            device (str or int, optional): Unused but included for interface consistency.
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.device = device

    def forward(self, cosine_similarity: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            cosine_similarity (Tensor): Cosine similarity between embeddings. Range: [-1, 1].
            labels (Tensor): Binary labels: 0 = similar, 1 = dissimilar.

        Returns:
            Tensor: Contrastive loss value.
        """
        # Convert cosine similarity to cosine distance
        distances = 1 - cosine_similarity  # Range [0, 2]

        # Apply contrastive loss formula
        loss = (1 - labels) * 0.5 * distances.pow(2) + \
               labels * 0.5 * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss