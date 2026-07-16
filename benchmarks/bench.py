#!/usr/bin/env python3
"""Small, dependency-light performance harness for ontology backend changes."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from exact.ontology import load_ontology  # noqa: E402
from exact.ontology.parser import parse  # noqa: E402
from exact.ontology.projection import project  # noqa: E402
from exact.ontology.store import OwlOntologySource  # noqa: E402
from exact.utils.candidate_generation import (  # noqa: E402
    lexical_candidate_pair_scores,
    make_candidate_labels,
)

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "ontologies"
SOURCE_PATH = FIXTURE_DIR / "mini_src.owl"
TARGET_PATH = FIXTURE_DIR / "mini_tgt.owl"


def parse_ontology() -> int:
    return len(parse(SOURCE_PATH).signature)


def build_hierarchy() -> int:
    source = OwlOntologySource(parse(SOURCE_PATH))
    return sum(len(source.hierarchy.ancestors(iri)) for iri in source.entities())


def projection_owl2vecstar() -> int:
    return len(project(parse(SOURCE_PATH), "owl2vecstar", include_literals=True))


def dataset_build_e2e() -> int:
    source, target = load_ontology(SOURCE_PATH), load_ontology(TARGET_PATH)
    with (FIXTURE_DIR / "mini_test.cands.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    built = 0
    for row in rows:
        src = row["SrcEntity"]
        source.labels(src)
        source.hierarchy_bundle(src, {"is_a": ()})
        for candidate in ast.literal_eval(row["TgtCandidates"]):
            target.labels(candidate)
            target.hierarchy_bundle(candidate, {"is_a": ()})
            built += 1
    return built


def candidate_generation() -> int:
    source, target = load_ontology(SOURCE_PATH), load_ontology(TARGET_PATH)
    src_records = make_candidate_labels(
        source.entities(), {iri: source.labels(iri) for iri in source.entities()}
    )
    tgt_records = make_candidate_labels(
        target.entities(), {iri: target.labels(iri) for iri in target.entities()}
    )
    return len(lexical_candidate_pair_scores(src_records, tgt_records, per_source_limit=10))


def inference_throughput() -> int:
    """CPU/fake-LLM smoke inference over a repeated fixture candidate pool."""

    source, target = load_ontology(SOURCE_PATH), load_ontology(TARGET_PATH)
    with (FIXTURE_DIR / "mini_test.cands.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    examples = [
        (row["SrcEntity"], candidate)
        for row in rows
        for candidate in ast.literal_eval(row["TgtCandidates"])
    ] * 100
    matches = 0
    for src, tgt in examples:
        src_labels = {label.casefold() for label in source.labels(src)}
        tgt_labels = {label.casefold() for label in target.labels(tgt)}
        matches += bool(src_labels & tgt_labels)
    return len(examples) + matches


SCENARIOS: dict[str, Callable[[], int]] = {
    "parse_ontology": parse_ontology,
    "build_hierarchy": build_hierarchy,
    "projection_owl2vecstar": projection_owl2vecstar,
    "dataset_build_e2e": dataset_build_e2e,
    "candidate_generation": candidate_generation,
    "inference_throughput": inference_throughput,
}


def benchmark(function: Callable[[], int], repeat: int) -> dict[str, float | int]:
    function()  # warm parser/import caches
    samples: list[float] = []
    result = 0
    for _ in range(repeat):
        started = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - started)
    median = statistics.median(samples)
    return {
        "median_seconds": median,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "result": result,
    }


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help="Fail when a measured fixture scenario regresses beyond the reference tolerance.",
    )
    args = parser.parse_args()
    selected = SCENARIOS if args.scenario == "all" else {args.scenario: SCENARIOS[args.scenario]}
    payload = {
        "schema_version": 1,
        "revision": git_revision(),
        "scenarios": {
            name: benchmark(function, max(1, args.repeat)) for name, function in selected.items()
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.check_reference:
        reference = json.loads((ROOT / "benchmarks" / "reference.json").read_text())
        tolerance = float(reference["max_regression_fraction"])
        failures = []
        for name, measurement in payload["scenarios"].items():
            baseline = float(reference["fixture"][name])
            limit = baseline * (1.0 + tolerance)
            if float(measurement["median_seconds"]) > limit:
                failures.append(f"{name}: {measurement['median_seconds']:.6f}s > {limit:.6f}s")
        if failures:
            raise SystemExit("Benchmark regressions:\n" + "\n".join(failures))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
