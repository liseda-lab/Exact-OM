from typing import List, Union

from matcha_dl.core.contracts.loss import ILoss, Tensor, torch, nn

F = nn.functional


class BCEWeightedLoss(nn.BCELoss, ILoss):
    def __init__(self, pos_weight=None, reduction='mean', device: Union[str, int]=None, **kwargs):
        """
        Initialize the custom loss function.

        Args:
            pos_weight (float or Tensor, optional): Weight to apply to the positive samples. 
                                                    If scalar, same weight applied across all samples.
                                                    If tensor, different weights for different classes.
            reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                       'none': No reduction will be applied.
                                       'mean': The output will be averaged.
                                       'sum': The output will be summed.
                                       Default: 'mean'
        """
        super(BCEWeightedLoss, self).__init__()
        
        if pos_weight is not None:
            self.pos_weight = torch.tensor(pos_weight).to(device)
        
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for the custom loss function.

        Args:
            inputs (Tensor): Model output (logits or probabilities). 
                             The user should ensure this matches the format needed.
                             If logits, ensure you apply sigmoid before calling this function.
            targets (Tensor): Ground truth binary labels (0 or 1). 
                              Shape: same as inputs.
                              
        Returns:
            Tensor: The computed loss.
        """

        print("Inputs: ", inputs)
        print("Inputs Shape: ", inputs.shape)

        print("Targets: ", targets)
        print("Targets Shape: ", targets.shape)
        
        # Compute unweighted binary cross-entropy loss for each sample
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        print("Unreduced BCE Loss: ", bce_loss)
        print("Unreduced BCE Loss Shape: ", bce_loss.shape)
        bce_loss_unreduced = bce_loss

        # Apply pos_weight to positive samples if specified
        if self.pos_weight is not None:
            weights = targets * self.pos_weight + (1 - targets)
            bce_loss = bce_loss * weights

        print("Weighted BCE Loss: ", bce_loss)
        print("Weighted BCE Loss Shape: ", bce_loss.shape)

        print("Loss Comparison:", (bce_loss == bce_loss_unreduced).all())

        # Apply reduction
        if self.reduction == 'mean':
            print("Mean BCE Loss: ", bce_loss.mean())
            print("Mean BCE Loss Shape: ", bce_loss.mean().shape)
            raise Exception("Forced Stop after first loss calculation")
            # return bce_loss.mean()
        elif self.reduction == 'sum':
            return bce_loss.sum()
        else:
            return bce_loss  # No reduction, return loss per sample

class BCELoss(nn.BCELoss, ILoss):
    
    def __init__(self, reduction='mean', device: Union[str, int]=None, **kwargs):
        """
        Initialize the custom loss function.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                       'none': No reduction will be applied.
                                       'mean': The output will be averaged.
                                       'sum': The output will be summed.
                                       Default: 'mean'
        """
        super(BCELoss, self).__init__(reduction=reduction, **kwargs)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass for the custom loss function.

        Args:
            inputs (Tensor): Model output (logits or probabilities). 
                             The user should ensure this matches the format needed.
                             If logits, ensure you apply sigmoid before calling this function.
            targets (Tensor): Ground truth binary labels (0 or 1). 
                              Shape: same as inputs.
                              
        Returns:
            Tensor: The computed loss.
        """
        print("Inputs: ", inputs)
        print("Inputs Shape: ", inputs.shape)

        print("Targets: ", targets)
        print("Targets Shape: ", targets.shape)

        loss_unreduced = F.binary_cross_entropy(inputs, targets, reduction='none')
        print("Unreduced BCE Loss: ", loss_unreduced)
        print("Unreduced BCE Loss Shape: ", loss_unreduced.shape)

        loss = super().forward(inputs, targets)
        print("Reduced BCE Loss: ", loss)
        print("Reduced BCE Loss Shape: ", loss.shape)

        raise Exception("Forced Stop after first loss calculation")

        # return super().forward(inputs, targets)

class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, ILoss):

    def __init__(self, pos_weight, reduction='mean', device: Union[str, int]=None, **kwargs):
        
        """
        Initialize the custom loss function.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                                       'none': No reduction will be applied.
                                       'mean': The output will be averaged.
                                       'sum': The output will be summed.
                                       Default: 'mean'
        """

        if pos_weight is not None:
            self.pos_weight = torch.tensor(pos_weight).to(device)

        super(BCEWithLogitsLoss, self).__init__(pos_weight=pos_weight, reduction=reduction, **kwargs)


    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Forward pass for the custom loss function.

        Args:
            inputs (Tensor): Model output (logits or probabilities). 
                             The user should ensure this matches the format needed.
                             If logits, ensure you apply sigmoid before calling this function.
            targets (Tensor): Ground truth binary labels (0 or 1). 
                              Shape: same as inputs.
                              
        Returns:
            Tensor: The computed loss.
        """
        return super().forward(inputs, targets)