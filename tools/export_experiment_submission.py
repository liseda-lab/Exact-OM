#!/usr/bin/env python3
"""Export saved experiment predictions using public pools, without reference labels."""

import argparse
from pathlib import Path

from exact.experiments.submission import TRACKS, export_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track", choices=TRACKS, required=True)
    parser.add_argument("--public-candidates", type=Path)
    parser.add_argument(
        "--source-universe",
        type=Path,
        help="Full eligible ontology signature, one IRI per line; global export only",
    )
    parser.add_argument("--population-manifest", type=Path)
    parser.add_argument("--target-population-manifest", type=Path)
    parser.add_argument(
        "--query-runs", type=Path, help="Prepared inference.json for original-query shards"
    )
    parser.add_argument(
        "--score-field",
        default="S_final",
        help="Saved candidate score; Q_match permits a saved joint DISO NIL probability",
    )
    parser.add_argument("--source-uri")
    parser.add_argument("--target-uri")
    args = parser.parse_args()
    print(
        export_submission(
            args.run,
            args.output,
            args.track,
            public_candidates=args.public_candidates,
            source_universe=args.source_universe,
            population_manifest=args.population_manifest,
            target_population_manifest=args.target_population_manifest,
            query_runs=args.query_runs,
            score_field=args.score_field,
            source_uri=args.source_uri,
            target_uri=args.target_uri,
        )
    )


if __name__ == "__main__":
    main()
