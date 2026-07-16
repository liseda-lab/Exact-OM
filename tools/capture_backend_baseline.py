#!/usr/bin/env python3
"""Capture deterministic ontology-backend snapshots.

Run this script with the backend revision being evaluated, then retain its zstd
JSON output as the parity oracle for later backend changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import zstandard

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exact.core.entities.kinds import EntityKind  # noqa: E402
from exact.ontology import load_ontology  # noqa: E402

PART_OF = "http://purl.obolibrary.org/obo/BFO_0000050"
HAS_PART = "http://purl.obolibrary.org/obo/BFO_0000051"


def capture(path: Path) -> dict:
    source = load_ontology(path)
    classes = list(source.entities(EntityKind.CLASS))
    properties = sorted(
        set(source.entities(EntityKind.OBJECT_PROPERTY))
        | set(source.entities(EntityKind.DATA_PROPERTY))
    )
    families = {"is_a": (), "part_of": (PART_OF,), "has_part": (HAS_PART,)}
    return {
        "schema_version": 1,
        "origin_name": path.name,
        "classes": classes,
        "labels": {iri: source.labels(iri) or [source.short_form(iri)] for iri in classes},
        "annotations": {
            iri: [asdict(value) for value in source.annotations(iri)] for iri in classes
        },
        "attributes": {iri: [asdict(value) for value in source.attributes(iri)] for iri in classes},
        "hierarchy": {iri: source.hierarchy_bundle(iri, families) for iri in classes},
        "projection": {
            "taxonomy": [edge.astuple() for edge in source.projection_edges(method="taxonomy")],
            "owl2vecstar": [
                edge.astuple() for edge in source.projection_edges(method="owl2vecstar")
            ],
            "owl2vecstar_literals": [
                edge.astuple()
                for edge in source.projection_edges(method="owl2vecstar", include_literals=True)
            ],
        },
        "excluded_from_alignment": sorted(source.excluded_from_alignment()),
        "property_domains": {prop: source.property_domains(prop) for prop in properties},
        "property_ranges": {prop: source.property_ranges(prop) for prop in properties},
    }


def write_snapshot(path: Path, output_dir: Path) -> Path:
    payload = json.dumps(
        capture(path), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}.backend.json.zst"
    output_path.write_bytes(zstandard.ZstdCompressor(level=10).compress(payload))
    return output_path


def expand_inputs(paths: list[Path]) -> list[Path]:
    ontologies: list[Path] = []
    for path in paths:
        if path.is_dir():
            discovered = sorted(path.glob("*.owl"))
            if not discovered:
                raise FileNotFoundError(f"No .owl ontologies found in dataset directory {path}")
            ontologies.extend(discovered)
        else:
            ontologies.append(path)
    return ontologies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ontologies", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("exp/baselines"))
    args = parser.parse_args()
    for ontology in expand_inputs(args.ontologies):
        print(write_snapshot(ontology, args.output_dir))


if __name__ == "__main__":
    main()
