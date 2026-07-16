"""Deterministic candidate-pool sampling utilities."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Iterable


def create_validation_samples(
    sample_ratio: float = 0.1,
    random_seed: int = 42,
    dataset_dirs: Iterable[Path] | None = None,
    *,
    data_root: Path = Path("data"),
) -> list[Path]:
    """Create deterministic ``test.cands.val.tsv`` samples.

    Args:
        sample_ratio: Fraction of rows to sample, strictly between zero and one.
        random_seed: Seed for deterministic sampling across datasets.
        dataset_dirs: Explicit dataset directories. When omitted, immediate children
            of ``data_root`` containing ``test.cands.tsv`` are used.
        data_root: Root searched when ``dataset_dirs`` is omitted.

    Returns:
        Paths of validation files written, in deterministic dataset order.
    """

    if not 0 < sample_ratio < 1:
        raise ValueError("sample_ratio must be between 0 and 1 (exclusive)")
    directories = (
        list(dataset_dirs)
        if dataset_dirs is not None
        else [
            path
            for path in Path(data_root).iterdir()
            if path.is_dir() and (path / "test.cands.tsv").is_file()
        ]
    )
    rng = random.Random(random_seed)
    outputs: list[Path] = []
    for directory in sorted(Path(path) for path in directories):
        source_file = directory / "test.cands.tsv"
        target_file = directory / "test.cands.val.tsv"
        with source_file.open("r", encoding="utf-8", newline="") as source:
            reader = csv.reader(source, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration:
                continue
            rows = list(reader)
        if not rows:
            continue
        sample_size = min(len(rows), max(1, math.floor(len(rows) * sample_ratio)))
        sampled = rng.sample(rows, sample_size)
        with target_file.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(sampled)
        outputs.append(target_file)
    return outputs
