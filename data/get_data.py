
import csv
import math
import random
import shutil
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

DATA_DIR = Path(__file__).parent


def _ensure_archive_extracted(url: str, extract_dir: Path, downloader) -> None:
    if extract_dir.exists():
        print(f"{extract_dir.name} already present; skipping download and extraction.")
        return

    archive_path = downloader.download_file(url)
    print(f"Extracting {archive_path.name} into {extract_dir}...")
    downloader.extract_archive(archive_path, dest_folder=extract_dir, delete_archive=True)


def _read_alignment(align_path: Path) -> Tuple[str, str, Sequence[Tuple[str, str, str]]]:
    ns = {
        "al": "http://knowledgeweb.semanticweb.org/heterogeneity/alignment",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    }
    root = ET.parse(align_path).getroot()
    locs = [loc.text.strip() for loc in root.findall(".//al:location", ns)]
    if len(locs) < 2:
        raise ValueError(f"Could not find two ontology locations in {align_path}")
    cells = []
    for cell in root.findall(".//al:Cell", ns):
        ent1_el = cell.find("al:entity1", ns)
        ent2_el = cell.find("al:entity2", ns)
        measure_el = cell.find("al:measure", ns)
        if ent1_el is None or ent2_el is None or measure_el is None:
            continue
        ent1 = ent1_el.attrib.get(f"{{{ns['rdf']}}}resource")
        ent2 = ent2_el.attrib.get(f"{{{ns['rdf']}}}resource")
        cells.append((ent1, ent2, measure_el.text.strip()))
    return locs[0], locs[1], cells


def _load_classes(path: Path) -> Sequence[str]:
    try:
        from rdflib import Graph, RDF, RDFS, OWL
        from rdflib.term import URIRef
    except ImportError as exc:
        raise RuntimeError(
            "rdflib is required to parse ontologies. Install with `pip install rdflib`."
        ) from exc

    g = Graph()
    g.parse(path)
    classes = set()
    for typ in (OWL.Class, RDFS.Class, OWL.DeprecatedClass):
        for s in g.subjects(RDF.type, typ):
            if isinstance(s, URIRef):
                classes.add(str(s))
    return tuple(sorted(classes))


def _build_conference_datasets() -> None:
    from exact.core.values import CONFERENCE_URL, REFERENCE_ALIGNMENT_URL
    from exact.utils.data import DataDownloader

    raw_dir = DATA_DIR / "conference_raw"
    ont_dir = raw_dir / "conference"
    align_dir = raw_dir / "reference-alignment"
    downloader = DataDownloader(dest_folder=raw_dir)

    _ensure_archive_extracted(CONFERENCE_URL, ont_dir, downloader)
    _ensure_archive_extracted(REFERENCE_ALIGNMENT_URL, align_dir, downloader)

    files_by_name = {p.name.lower(): p for p in ont_dir.glob("*.owl")}

    created = []
    for align_path in sorted(align_dir.glob("*.rdf")):
        src_loc, tgt_loc, cells = _read_alignment(align_path)
        src_name = Path(urlparse(src_loc).path).name.lower()
        tgt_name = Path(urlparse(tgt_loc).path).name.lower()
        src_path = files_by_name.get(src_name)
        tgt_path = files_by_name.get(tgt_name)
        if src_path is None or tgt_path is None:
            raise FileNotFoundError(
                f"Missing ontology for {align_path.name}: {src_name} or {tgt_name}"
            )

        dataset_dir = DATA_DIR / align_path.stem
        dataset_dir.mkdir(exist_ok=True)
        shutil.copy2(src_path, dataset_dir / src_path.name)
        shutil.copy2(tgt_path, dataset_dir / tgt_path.name)

        tgt_classes = list(_load_classes(tgt_path))
        ent2_values = {ent2 for _, ent2, _ in cells}
        missing = sorted(ent2_values - set(tgt_classes))
        if missing:
            tgt_classes.extend(missing)
        candidates_str = str(tuple(tgt_classes))

        with (dataset_dir / "test.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["SrcEntity", "TgtEntity", "Score"])
            writer.writerows(cells)

        with (dataset_dir / "test.cands.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["SrcEntity", "TgtEntity", "TgtCandidates"])
            for ent1, ent2, _ in cells:
                writer.writerow([ent1, ent2, candidates_str])

        created.append(dataset_dir.name)

    print(f"Created {len(created)} conference datasets: {', '.join(sorted(created))}")


def _download_bioml_if_missing() -> None:
    """
    Preserve the original Bio-ML download behavior: if the data directory is empty,
    fetch and unpack the zenodo archive via DataDownloader. If any subdirectories
    already exist, we assume the user has the data and skip the download.
    """

    from exact.core.values import DATASET_URL
    from exact.utils.data import DataDownloader  # Lazy import to avoid heavy deps when unused.

    if DataDownloader.has_subdirs(DATA_DIR):
        print("Data directory already contains subdirectories; skipping Bio-ML download.")
        return

    print("Data directory is empty. Proceeding with Bio-ML download.")
    downloader = DataDownloader(dest_folder=str(DATA_DIR))
    downloader.download_dataset(DATASET_URL)
    print("Bio-ML dataset downloaded and uncompressed successfully.")


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

    _download_bioml_if_missing()
    _build_conference_datasets()
