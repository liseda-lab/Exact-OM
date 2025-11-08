
import csv
import math
import random
from pathlib import Path
from typing import Iterable, Optional

DATA_DIR = Path(__file__).parent


def create_validation_samples(
    sample_ratio: float = 0.1,
    random_seed: int = 42,
    dataset_dirs: Optional[Iterable[Path]] = None,
) -> None:
    """
    Create a per-dataset stratified sample from every available test.cands.tsv file.

    A new file named test.cands.val.tsv is written inside each dataset directory. The
    sample size is computed as `max(1, floor(num_rows * sample_ratio))` so that even
    small datasets contribute at least one example. Sampling is deterministic thanks
    to the configurable random seed.
    """

    if not 0 < sample_ratio < 1:
        raise ValueError("sample_ratio must be between 0 and 1 (exclusive).")

    dirs = (
        dataset_dirs
        if dataset_dirs is not None
        else (
            d
            for d in DATA_DIR.iterdir()
            if d.is_dir() and (d / "test.cands.tsv").is_file()
        )
    )

    dirs = sorted(dirs)
    if not dirs:
        print(f"No dataset folders with test.cands.tsv found under {DATA_DIR}.")
        return

    rng = random.Random(random_seed)

    for dataset_dir in dirs:
        source_file = dataset_dir / "test.cands.tsv"
        target_file = dataset_dir / "test.cands.val.tsv"

        with source_file.open("r", encoding="utf-8", newline="") as src:
            reader = csv.reader(src, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration:
                print(f"{source_file} is empty; skipping.")
                continue
            rows = [row for row in reader]

        if not rows:
            print(f"{source_file} has a header but no data rows; skipping.")
            continue

        sample_size = max(1, math.floor(len(rows) * sample_ratio))
        sample_size = min(sample_size, len(rows))
        sampled_rows = rng.sample(rows, sample_size)

        with target_file.open("w", encoding="utf-8", newline="") as tgt:
            writer = csv.writer(tgt, delimiter="\t")
            writer.writerow(header)
            writer.writerows(sampled_rows)

        print(
            f"Created {target_file.name} with {sample_size} / {len(rows)} rows "
            f"for dataset {dataset_dir.name}."
        )


if __name__ == "__main__":

    from exact.core.values import DATASET_URL
    from exact.utils.data import DataDownloader  # Imported lazily to avoid heavy deps when unused.

    # Check if the data directory has any dirs inside

    if DataDownloader.has_subdirs(DATA_DIR):
        print("Data directory already exists and contains subdirectories.")
        print("Please remove the data directory or its contents to download the dataset again.")
        exit(1)
    else:
        print("Data directory is empty. Proceeding with download.")

        downloader = DataDownloader(dest_folder=str(DATA_DIR))
        downloader.download_dataset(DATASET_URL)

        print("Dataset downloaded and uncompressed successfully.")
