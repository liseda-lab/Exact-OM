"""Shared control signal for imported training and module-mode phase runners."""


class PhaseBoundary(Exception):
    def __init__(self, next_phase, completed_epoch, optimizer_updates):
        self.receipt = dict(
            status="phase_complete",
            next_phase=next_phase,
            completed_epoch=completed_epoch,
            optimizer_updates=optimizer_updates,
        )
        super().__init__(next_phase)
