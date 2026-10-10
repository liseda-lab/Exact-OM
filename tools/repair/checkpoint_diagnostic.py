"""TRAIN engineering state I/O only: no model construction or optimizer execution."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from tools.repair.batch import read, sha
from tools.repair.historical_regression import binding
from tools.repair.phase_timing import checkpoint_identity, load_training_state, phase, recording
from tools.repair.shared_release import authenticate, immutable
from tools.repair.train import save_training_state


def identical(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and torch.equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(identical(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(identical(a, b) for a, b in zip(left, right))
    return left == right


def run(reference_path, output):
    reference, output = read(reference_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "started.json").exists():
        raise ValueError("Checkpoint diagnostic cannot reset or replay")
    if reference["scope"] != "terminal_train_engineering_state_io_only":
        raise ValueError("Checkpoint diagnostic scope changed")
    deadline = float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"])
    if deadline - time.time() < 10:
        raise TimeoutError("Insufficient checkpoint diagnostic cleanup reserve")
    immutable(
        output / "started.json", dict(epoch=time.time(), reference=binding(Path(reference_path)))
    )
    source = authenticate(reference["checkpoint"])
    trace = output / "phase-timings.jsonl"
    target = output / "roundtrip-state.pt"
    with recording(trace), phase("checkpoint_diagnostic"):
        state = load_training_state(source, map_location="cpu")
        if state["optimizer_updates"] != reference["expected_optimizer_updates"]:
            raise ValueError("Historical optimizer endpoint changed")
        if deadline - time.time() < 5:
            raise TimeoutError("Insufficient serialization cleanup reserve")
        save_training_state(target, state)
        restored = load_training_state(target, map_location="cpu")
        with phase("checkpoint_compare_all_state"):
            exact = identical(state, restored)
    if sha(source) != reference["checkpoint"]["sha256"]:
        raise ValueError("Historical state changed during diagnostic")
    if not exact:
        raise ValueError("Checkpoint roundtrip changed state")
    result = dict(
        schema="exact-repair/checkpoint-io-diagnostic/v1",
        status="complete",
        source=reference["checkpoint"],
        roundtrip=binding(target),
        trace=binding(trace),
        exact_all_state=True,
        checkpoint_identity=checkpoint_identity(state),
        bytes=target.stat().st_size,
        optimizer_updates_executed=0,
        previous_costs_preserved=True,
        selected_model=False,
        test_payloads_opened=False,
        hosted_calls=0,
        limitation="CPU I/O diagnostic; no attribution of historical GPU checkpoint gap or fit endpoint admission",
    )
    immutable(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.reference, args.output)


if __name__ == "__main__":
    main()
