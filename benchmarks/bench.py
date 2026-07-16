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
import tempfile
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from exact.io.sources.csv_kg import CsvKgSource  # noqa: E402
from exact.ontology import load_ontology  # noqa: E402
from exact.ontology.parser import parse  # noqa: E402
from exact.ontology.projection import project  # noqa: E402
from exact.ontology.store import OwlOntologySource  # noqa: E402
from exact.runs import ExplanationStore, RunLayout  # noqa: E402
from exact.tracks.descriptor import TrackDescriptor  # noqa: E402
from exact.tracks.hf import HfProvider  # noqa: E402
from exact.utils.candidate_generation import (  # noqa: E402
    lexical_candidate_pair_scores,
    make_candidate_labels,
)

FIXTURE_DIR = ROOT / "tests" / "fixtures" / "ontologies"
SOURCE_PATH = FIXTURE_DIR / "mini_src.owl"
TARGET_PATH = FIXTURE_DIR / "mini_tgt.owl"
_EXPLANATION_RECORDS: list[dict] | None = None
_READ_FIXTURE: tuple[tempfile.TemporaryDirectory[str], Path] | None = None


def _explanation_records() -> list[dict]:
    global _EXPLANATION_RECORDS
    if _EXPLANATION_RECORDS is None:
        _EXPLANATION_RECORDS = [
            {
                "src_iri": f"https://example.org/source/{index // 100}",
                "tgt_iri": f"https://example.org/target/{index}",
                "confidences": {"S_final": (index % 100) / 100.0},
                "prediction": {"selected": index % 7 == 0},
                "description": "repeated benchmark explanation payload " * 4,
                "run_id": "benchmark",
                "explanation_schema_version": 1,
            }
            for index in range(100_000)
        ]
    return _EXPLANATION_RECORDS


def parse_ontology() -> int:
    return len(parse(SOURCE_PATH).signature)


def build_hierarchy() -> int:
    source = OwlOntologySource(parse(SOURCE_PATH))
    return sum(len(source.hierarchy.ancestors(iri)) for iri in source.entities())


def transitive_closure() -> int:
    """Materialize all fixture ancestor and descendant closures."""

    source = OwlOntologySource(parse(SOURCE_PATH))
    return sum(
        len(source.hierarchy.ancestors(iri)) + len(source.hierarchy.descendants(iri))
        for iri in source.entities()
    )


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


def csv_kg_load() -> int:
    """Load and index the committed mini BioKG-shaped fixture."""

    source = CsvKgSource.from_path(ROOT / "tests" / "fixtures" / "kg_csv")
    return len(source.projection_edges(include_literals=True))


class _FixtureHfClient:
    """Local snapshot client used by the hermetic materialization benchmark."""

    snapshot = ROOT / "tests" / "fixtures" / "tracks" / "hf_snapshot"

    def resolve_revision(self, repo_id: str, revision: str) -> str:
        return "fixture-commit"

    def snapshot_download(
        self,
        repo_id: str,
        revision: str,
        destination: Path,
        **kwargs: object,
    ) -> Path:
        return self.snapshot


def _fixture_track_descriptor() -> TrackDescriptor:
    return TrackDescriptor.from_mapping(
        {
            "descriptor_version": 1,
            "name": "benchmark-track",
            "provider": "hf",
            "provider_version": "1",
            "upstream": {
                "repo_id": "fixture/benchmark",
                "revision": "fixture",
                "checksum_manifest": "SHA256SUMS",
            },
            "tasks": {
                "demo": {
                    "source": "source.owl",
                    "target": "target.owl",
                    "refs": {
                        "test": {
                            "path": "references/test.rdf",
                            "transform": "alignment_rdf_to_tsv",
                        }
                    },
                    "candidates": {
                        "path": "pools/pools.jsonl",
                        "transform": "pools_jsonl_to_cands_tsv",
                    },
                }
            },
        }
    )


def track_materialize() -> int:
    """Materialize and lock a local HF-shaped track fixture."""

    with tempfile.TemporaryDirectory(prefix="exact-benchmark-track-") as directory:
        provider = HfProvider(_fixture_track_descriptor(), client=_FixtureHfClient())
        layout = provider.materialize("demo", Path(directory))
        artifacts = (layout.source, layout.target, *layout.refs.values())
        return sum(path.is_file() for path in artifacts)


def explanation_store_write() -> int:
    """Write 100k explanation records to bounded compressed shards."""

    with tempfile.TemporaryDirectory(prefix="exact-benchmark-write-") as directory:
        store = ExplanationStore.create(Path(directory), run_id="benchmark")
        store.append(_explanation_records())
        return store.record_count


def _read_fixture() -> Path:
    global _READ_FIXTURE
    if _READ_FIXTURE is None:
        temporary = tempfile.TemporaryDirectory(prefix="exact-benchmark-read-")
        run_dir = Path(temporary.name)
        store = ExplanationStore.create(run_dir, run_id="benchmark")
        store.append(_explanation_records())
        _READ_FIXTURE = temporary, run_dir
    return _READ_FIXTURE[1]


def explanation_store_read() -> int:
    """Cold-open the index and retrieve one source from a 100k-record store."""

    store = ExplanationStore(RunLayout.open(_read_fixture()).explanations_dir)
    return len(store.get("https://example.org/source/500"))


SCENARIOS: dict[str, Callable[[], int]] = {
    "parse_ontology": parse_ontology,
    "build_hierarchy": build_hierarchy,
    "transitive_closure": transitive_closure,
    "projection_owl2vecstar": projection_owl2vecstar,
    "dataset_build_e2e": dataset_build_e2e,
    "candidate_generation": candidate_generation,
    "inference_throughput": inference_throughput,
    "csv_kg_load": csv_kg_load,
    "track_materialize": track_materialize,
    "explanation_store_write": explanation_store_write,
    "explanation_store_read": explanation_store_read,
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
