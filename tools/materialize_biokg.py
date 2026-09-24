#!/usr/bin/env python3
"""Prepare or execute native E13 enrichment from an anchor-free public BioKG release."""
import argparse
from pathlib import Path

from exact.experiments.evidence_inventory import import_biokg_public_graph
from exact.experiments.materialization import (
    materialize_legacy,
    prepare_legacy_materialization,
    verify_legacy_materialization,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--souffle", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    raw, native = args.output / "raw", args.output / "materialized"
    if (native / "materialization.json").exists():
        verify_legacy_materialization(
            native, raw=raw, package=args.package, ontology=args.ontology, executable=args.souffle
        )
    else:
        if not (raw / "conversion.json").exists():
            import_biokg_public_graph(args.package, raw, ontology=args.ontology)
        prepare_legacy_materialization(
            args.package, raw, native, ontology=args.ontology, executable=args.souffle
        )
    if not args.prepare_only:
        result = materialize_legacy(raw, native)
        print(f"{result['added_edges']} consequences; {native / 'enriched'}")


if __name__ == "__main__":
    main()
