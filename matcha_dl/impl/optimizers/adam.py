
import torch.optim as optim

from matcha_dl.core.contracts.optimizer import IOptimizer

class Adam(IOptimizer, optim.Adam):
    """
    Implementation of the Adam optimizer that integrates with IOptimizer.
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, **kwargs):
        """
        Initializes the Adam optimizer with default parameters.
        Args:
            params: Parameters to optimize.
            lr: Learning rate (default: 0.001).
            betas: Coefficients used for computing running averages of gradient and its square.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay (L2 penalty).
            amsgrad: Whether to use the AMSGrad variant of this algorithm from the paper 
                     "On the Convergence of Adam and Beyond".
        """
        IOptimizer.__init__(self, params, **kwargs)
        optim.Adam.__init__(self, params, lr, betas, eps, weight_decay, amsgrad, **kwargs)
