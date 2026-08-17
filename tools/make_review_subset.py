#!/usr/bin/env python3
"""Build and validate deterministic review-response subsets."""

from __future__ import annotations

import argparse
import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    data_dir: str
    unique_sources: int
    candidate_rows: int
    reference_rows: int
    candidate_pairs: int
    first_source: str
    last_source: str
    targets_in_candidates: int


DATASETS = {
    "omim-ordo": DatasetSpec(
        data_dir="omim-ordo",
        unique_sources=300,
        candidate_rows=302,
        reference_rows=302,
        candidate_pairs=30_502,
        first_source="http://omim.org/entry/100800",
        last_source="http://omim.org/entry/158320",
        targets_in_candidates=302,
    ),
    "snomed-fma.body": DatasetSpec(
        data_dir="snomed-fma.body",
        unique_sources=300,
        candidate_rows=324,
        reference_rows=324,
        candidate_pairs=32_724,
        first_source="http://snomed.info/id/10013000",
        last_source="http://snomed.info/id/1425000",
        targets_in_candidates=324,
    ),
}


def _split_tsv(path: Path, expected_header: tuple[str, ...]) -> tuple[bytes, list[bytes]]:
    """Return the original header/row bytes after validating the TSV shape."""
    lines = path.read_bytes().splitlines(keepends=True)
    if not lines:
        raise AssertionError(f"{path} is empty")

    header = lines[0]
    columns = tuple(header.rstrip(b"\r\n").decode("utf-8").split("\t"))
    if columns != expected_header:
        raise AssertionError(f"{path} has header {columns!r}; expected {expected_header!r}")

    expected_fields = len(expected_header)
    for line_number, row in enumerate(lines[1:], start=2):
        if len(row.rstrip(b"\r\n").split(b"\t")) != expected_fields:
            raise AssertionError(f"{path}:{line_number} does not have {expected_fields} fields")
    return header, lines[1:]


def _fields(row: bytes) -> tuple[str, ...]:
    return tuple(field.decode("utf-8") for field in row.rstrip(b"\r\n").split(b"\t"))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assert_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: got {actual!r}, expected {expected!r}")


def _join_rows(header: bytes, rows: Iterable[bytes]) -> bytes:
    return header + b"".join(rows)


def build_subset(dataset: str = "omim-ordo") -> dict[str, object]:
    """Build both subset files in memory, validate them, then write them."""
    spec = DATASETS[dataset]
    data_dir = REPO_ROOT / "data" / spec.data_dir
    candidates_input = data_dir / "test.cands.tsv"
    reference_input = data_dir / "test.tsv"
    candidates_output = data_dir / "review300.test.cands.tsv"
    reference_output = data_dir / "review300.test.tsv"
    candidate_header, candidate_rows = _split_tsv(
        candidates_input,
        ("SrcEntity", "TgtEntity", "TgtCandidates"),
    )
    reference_header, reference_rows = _split_tsv(
        reference_input,
        ("SrcEntity", "TgtEntity", "Score"),
    )

    selected_sources = sorted({_fields(row)[0] for row in candidate_rows})[: spec.unique_sources]
    selected_source_set = set(selected_sources)

    selected_candidate_rows = [
        row for row in candidate_rows if _fields(row)[0] in selected_source_set
    ]
    selected_reference_rows = [
        row for row in reference_rows if _fields(row)[0] in selected_source_set
    ]

    candidate_lists: dict[tuple[str, str], tuple[str, ...]] = {}
    candidate_pair_count = 0
    for row in selected_candidate_rows:
        src, tgt, serialized_candidates = _fields(row)
        parsed_candidates = ast.literal_eval(serialized_candidates)
        if not isinstance(parsed_candidates, (tuple, list)):
            raise AssertionError(f"candidate list for {(src, tgt)!r} is not a tuple or list")
        candidates = tuple(str(candidate) for candidate in parsed_candidates)
        candidate_lists[(src, tgt)] = candidates
        candidate_pair_count += len(candidates)

    targets_in_candidates = 0
    for row in selected_reference_rows:
        src, tgt, _score = _fields(row)
        candidates = candidate_lists.get((src, tgt), ())
        targets_in_candidates += int(tgt in candidates)

    candidate_sources = {_fields(row)[0] for row in selected_candidate_rows}
    reference_sources = {_fields(row)[0] for row in selected_reference_rows}

    _assert_equal("unique selected sources", len(selected_sources), spec.unique_sources)
    _assert_equal("unique candidate sources", len(candidate_sources), spec.unique_sources)
    _assert_equal("unique reference sources", len(reference_sources), spec.unique_sources)
    _assert_equal("candidate rows", len(selected_candidate_rows), spec.candidate_rows)
    _assert_equal("reference rows", len(selected_reference_rows), spec.reference_rows)
    _assert_equal("candidate pairs", candidate_pair_count, spec.candidate_pairs)
    _assert_equal("first source IRI", selected_sources[0], spec.first_source)
    _assert_equal("last source IRI", selected_sources[-1], spec.last_source)
    _assert_equal(
        "reference targets found in their candidate lists",
        targets_in_candidates,
        spec.targets_in_candidates,
    )

    candidate_payload = _join_rows(candidate_header, selected_candidate_rows)
    reference_payload = _join_rows(reference_header, selected_reference_rows)
    candidates_output.write_bytes(candidate_payload)
    reference_output.write_bytes(reference_payload)

    return {
        "unique_sources": len(selected_sources),
        "candidate_rows": len(selected_candidate_rows),
        "reference_rows": len(selected_reference_rows),
        "candidate_pairs": candidate_pair_count,
        "first_source": selected_sources[0],
        "last_source": selected_sources[-1],
        "targets_in_candidates": targets_in_candidates,
        "candidates_sha256": _sha256(candidate_payload),
        "reference_sha256": _sha256(reference_payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="omim-ordo")
    args = parser.parse_args()
    summary = build_subset(args.dataset)
    data_dir = REPO_ROOT / "data" / DATASETS[args.dataset].data_dir
    candidates_output = data_dir / "review300.test.cands.tsv"
    reference_output = data_dir / "review300.test.tsv"
    print(f"Wrote {candidates_output.relative_to(REPO_ROOT)}")
    print(f"Wrote {reference_output.relative_to(REPO_ROOT)}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
