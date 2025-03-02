from matcha_dl.core.contracts.stopper import IStopper


class StableLossStopping(IStopper):
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        """
        Initialize the EarlyStopper.

        Args:
            patience (int): Number of epochs to wait after the last improvement before stopping.
            min_delta (float): Minimum change in validation loss to be considered as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss: float) -> bool:
        """
        Check if training should be stopped based on validation loss.

        Args:
            val_loss (float): Current validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
    
    
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
