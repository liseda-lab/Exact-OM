"""A lossless storage candidate must retain all state and the prior durable file."""
import random
import os

import pytest
import torch

from tools.repair.checkpoint_diagnostic import identical
from tools.repair.checkpoint_storage import save_compressed


def test_compressed_checkpoint_retains_optimizer_rng_and_tensor_views(tmp_path):
    parameter = torch.nn.Parameter(torch.arange(24, dtype=torch.float32).reshape(4, 6))
    optimizer = torch.optim.AdamW([parameter])
    parameter.sum().backward()
    optimizer.step()
    shared = torch.arange(30)
    state = dict(model={"weight": parameter.detach()}, optimizer=optimizer.state_dict(),
                 cpu_rng=torch.get_rng_state(), python_rng=random.getstate(),
                 next_offset=7, optimizer_updates=7, phase="optimization",
                 views=[shared[::2], shared[1:]], empty=torch.empty(0))
    path = tmp_path / "state.pt"
    save_compressed(path, state)
    loaded = torch.load(path, weights_only=True)
    assert identical(state, loaded)
    assert loaded["views"][0].untyped_storage().data_ptr() == loaded["views"][1].untyped_storage().data_ptr()


def test_failed_compressed_fsync_keeps_prior_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "state.pt"
    save_compressed(path, {"prior": torch.tensor([9])})
    previous = path.read_bytes()
    def failed(_):
        raise OSError("injected fsync failure")
    monkeypatch.setattr(os, "fsync", failed)
    with pytest.raises(OSError, match="fsync"):
        save_compressed(path, {"next": torch.tensor([10])})
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob(".training-*"))
