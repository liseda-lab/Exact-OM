"""Lossless, torch-load-compatible checkpoint storage qualification.

The candidate preserves tensor bytes, all optimizer/RNG state, atomic replacement,
and both file/directory fsync. It is not enabled in scientific training by this module.
"""
from __future__ import annotations

import argparse
import copy
import io
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

from tools.repair.batch import read
from tools.repair.checkpoint_diagnostic import identical
from tools.repair.historical_regression import binding
from tools.repair.shared_release import authenticate, immutable


def save_compressed(path, state):
    """Keep the standard torch ZIP format, losslessly compressing its records."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = io.BytesIO()
    torch.save(state, serialized)
    serialized.seek(0)
    descriptor, temporary = tempfile.mkstemp(prefix=".training-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            with zipfile.ZipFile(serialized) as source, zipfile.ZipFile(
                stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                allowZip64=True,
            ) as target:
                for entry in source.infolist():
                    with source.open(entry) as incoming, target.open(entry.filename, "w", force_zip64=True) as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(manifest_path, output):
    import torch
    from tools.repair.train import save_training_state

    torch.set_num_threads(2)
    manifest, output = read(manifest_path), Path(output)
    if manifest["scope"] != "terminal_train_engineering_storage_only":
        raise ValueError("Storage qualification scope changed")
    deadline = float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"])
    immutable(output / "started.json", dict(epoch=time.time(), manifest=binding(Path(manifest_path))))
    if (output / "report.json").exists():
        raise ValueError("Do not replay terminal storage qualification")
    rows = []
    for reference in manifest["checkpoints"]:
        source = authenticate(reference)
        state = torch.load(source, weights_only=True, map_location="cpu")
        if state["optimizer_updates"] != 126 or state["phase"] != "development_loss":
            raise ValueError("Expected complete terminal one-epoch engineering state")
        for best in (False, True):
            value = dict(state)
            if best:
                value["best_state"] = copy.deepcopy(state["model"])
            for repeat in range(2):
                for codec, writer in (("stored", save_training_state), ("deflate1", save_compressed)):
                    if time.time() + 15 >= deadline:
                        raise TimeoutError("Storage qualification cleanup reserve reached")
                    key = f"{source.parent.name}-{best}-{repeat}-{codec}"
                    path = output / (key + ".pt")
                    started = time.monotonic()
                    writer(path, value)
                    write_seconds = time.monotonic() - started
                    started = time.monotonic()
                    restored = torch.load(path, weights_only=True, map_location="cpu")
                    read_seconds = time.monotonic() - started
                    if not identical(value, restored):
                        raise ValueError("Storage roundtrip changed checkpoint state")
                    row = dict(source=reference, best_state_present=best, repeat=repeat, codec=codec,
                               write_seconds=write_seconds, read_seconds=read_seconds,
                               bytes=path.stat().st_size, exact_all_state=True)
                    immutable(output / "rows" / (key + ".json"), row)
                    rows.append(row)
                    path.unlink()  # Disposable measured copies; original remains immutable.
            del value
        authenticate(reference)
    result = dict(schema="exact-repair/checkpoint-storage-qualification/v1", status="complete",
                  rows=rows, optimizer_updates_executed=0, selected_model=False,
                  test_outcomes_opened=False, hosted_calls=0, source_states_unchanged=True,
                  training_writer_changed=False, measured_cpu_only=True,
                  remaining_gate="Compare total write/read overhead; qualify any selected writer under actual GPU load before endpoint admission")
    immutable(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output)


if __name__ == "__main__":
    main()
