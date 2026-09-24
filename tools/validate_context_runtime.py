"""Prepare one pinned ontology context and retain operational evidence (no matching).

Run with the installed environment: python -m tools.validate_context_runtime NCIT.
Full preparation is explicit; all outputs/checkpoints live under data, not /tmp.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import platform
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path

from exact_inspect.context import OntologyContext, prepare_context
from exact_inspect.contracts import file_hash

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "NCIT": (
        "data/experiments-v2/bioml-primary/ontologies/NCIT-bd7a9a8a75e5-Thesaurus.owl",
        "1a7182a7327ebc4181f7d6b0f7e81ed04dd258f1a86bd8f560e4a0d61439d58a",
    ),
    "DOID": (
        "data/experiments-v2/ontology-normalization/doid-611355c44553/doid.annotation-declarations.owl",
        "f552e866a6233add0c02e5fed1ac1b0236c36c7f92247ed8c3a2a522ecc88fbb",
    ),
}


def peak_rss_bytes() -> int:
    """Use the new process image's high-water mark, not a fork parent's inherited peak."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def measure(context: OntologyContext, name: str) -> dict:
    examples = json.loads(
        (
            ROOT / "specs/explanation-framework/evidence/real-entity-context-examples.json"
        ).read_text()
    )["examples"]
    checks = []
    first_query_seconds = None
    for example in examples:
        expected = example["entities"].get(name)
        if expected is None:
            continue
        ref = {
            "ontology_version_id": context.ontology_version_id,
            "iri": expected["identity"]["iri"],
            "kind": "class",
        }
        query_started = time.perf_counter()
        actual = context.entity_context(ref)
        if first_query_seconds is None:
            first_query_seconds = time.perf_counter() - query_started
        definitions = {
            (fact["predicate_iri"], fact["value"].get("lexical_form"))
            for category in ("definitions", "alternate_definitions")
            for fact in actual["categories"][category]["items"]
        }
        parents = {edge["parent"]["iri"] for edge in context.hierarchy(ref)["items"]}
        if expected["definition_status"] == "no_definition_assertion_in_root_document":
            assert not definitions
        assert all(
            (item["predicate_iri"], item["text"]) in definitions for item in expected["definitions"]
        )
        assert {item["iri"] for item in expected["literal_asserted_named_parents"]} == parents
        checks.append(
            {
                "iri": ref["iri"],
                "definition_count": sum(
                    actual["categories"][category]["total_count"]
                    for category in ("definitions", "alternate_definitions")
                ),
                "definition_category_counts": {
                    category: actual["categories"][category]["total_count"]
                    for category in ("definitions", "alternate_definitions")
                },
                "parent_count": len(parents),
                "restriction_count": actual["categories"]["restrictions"]["total_count"],
                "status": "pass",
            }
        )
    refs = [item["entity"] for item in context.search(kind="class", limit=40)["items"]]

    def query(number: int) -> tuple[float, int]:
        ref = refs[number % len(refs)]
        start = time.perf_counter()
        result = (
            context.entity_context(ref)
            if number % 3 == 0
            else (
                context.search(term=ref["iri"], limit=20)
                if number % 3 == 1
                else context.hierarchy(ref)
            )
        )
        return time.perf_counter() - start, len(json.dumps(result).encode())

    sequential = [query(number) for number in range(100)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as readers:
        concurrent_results = list(readers.map(query, range(100)))
    cold_start = time.perf_counter()
    reopened = OntologyContext(context.path)
    reopened.entity_context(refs[0])
    return {
        "parity": checks,
        "cold_first_entity_after_verified_open_seconds": first_query_seconds,
        "queries": 100,
        "four_reader_queries": 100,
        "warm_p95_seconds": sorted(seconds for seconds, _ in sequential)[94],
        "four_reader_p95_seconds": sorted(seconds for seconds, _ in concurrent_results)[94],
        "max_response_bytes": max(size for _, size in sequential + concurrent_results),
        "reopen_first_query_seconds": time.perf_counter() - cold_start,
        "query_process_peak_rss_bytes": peak_rss_bytes(),
    }


def verify_original_doid(output: Path) -> None:
    """Retain the strict native-runtime failure of the unmodified pinned DOID root."""
    import pyowl_core as core

    source = ROOT / "data/experiments-v2/bioml-primary/ontologies/DOID-611355c44553-doid.owl"
    expected = "sha256:611355c445537fcf4bae2c519f1b3598af5a8fea793274316e35525b7d05e945"
    if file_hash(source) != expected:
        raise ValueError("Original DOID bytes differ from their specification pin")
    started = time.perf_counter()
    report = {
        "schema": "exact-context-native-negative/1",
        "source_sha256": expected,
        "pyowl_core_version": importlib.metadata.version("pyowl-core"),
        "backend": "native",
        "imports": "ignore",
        "preserve_source_map": True,
        "partial_rdf_mapping": False,
        "normalization": "none",
    }
    try:
        core.load_snapshot(
            source,
            options=core.LoadOptions(
                backend=core.BackendPreference.NATIVE,
                imports=core.ImportPolicy.IGNORE,
                preserve_source_map=True,
                offline=True,
            ),
        )
        report["status"] = "accepted"
    except Exception as error:
        report.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
    report.update(
        {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes(),
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "doid-original-native.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ontology", choices=PINS)
    parser.add_argument("--output", type=Path, default=ROOT / "data/explanation-framework/context")
    parser.add_argument("--measure-only", action="store_true")
    parser.add_argument("--sqlite-cache-mib", type=int, default=2048)
    parser.add_argument("--work-directory", type=Path)
    parser.add_argument("--verify-original-doid", action="store_true")
    args = parser.parse_args()
    if args.verify_original_doid:
        verify_original_doid(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    source, sha = PINS[args.ontology]
    destination = args.output / args.ontology.lower()
    started = time.time()
    dist = importlib.metadata.distribution("pyowl-core")
    record = next(
        dist.locate_file(item)
        for item in dist.files or []
        if str(item).endswith(".dist-info/RECORD")
    )
    report = {
        "schema": "exact-context-operational/1",
        "ontology": args.ontology,
        "command": shlex.join(
            [".venv/bin/python", "-m", "tools.validate_context_runtime", *sys.argv[1:]]
        ),
        "source_sha256": sha,
        "source_path": source,
        "parser_version": dist.version,
        "installed_distribution_record_sha256": file_hash(Path(str(record))),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started_at": started,
        "status": "running",
        "resource_profile": "64GB target measured on 128GB node",
        "matching_campaign": False,
    }
    report_path = args.output / (
        args.ontology.lower() + (".queries.json" if args.measure_only else ".operational.json")
    )
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        previous_path = args.output / (
            args.ontology.lower() + ".attempt-" + str(int(previous["started_at"])) + ".json"
        )
        previous_path.write_text(json.dumps(previous, indent=2) + "\n")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        derivation = None
        if args.ontology == "DOID":
            derivation = json.loads(
                (
                    ROOT
                    / "data/experiments-v2/ontology-normalization/doid-611355c44553/normalization.json"
                ).read_text()
            )
        open_started = time.perf_counter()
        context = (
            OntologyContext(destination)
            if args.measure_only
            else prepare_context(
                ROOT / source,
                destination,
                expected_hash=sha,
                matcher_scope="root",
                ontology_name=args.ontology,
                source_derivation=derivation,
                cache_mib=args.sqlite_cache_mib,
                work_directory=args.work_directory,
            )
        )
        report["initial_open_or_preparation_seconds"] = time.perf_counter() - open_started
        if args.measure_only:
            report.update(measure(context, args.ontology))
            report["read_targets"] = {
                "warm_p95_at_most_500ms": report["warm_p95_seconds"] <= 0.5,
                "four_reader_p95_at_most_500ms": report["four_reader_p95_seconds"] <= 0.5,
                "first_entity_after_verified_open_at_most_2s": report[
                    "cold_first_entity_after_verified_open_seconds"
                ]
                <= 2,
                "response_at_most_2MiB": report["max_response_bytes"] <= 2 * 1024 * 1024,
                "serving_process_rss_at_most_4GiB": report["query_process_peak_rss_bytes"]
                <= 4 * 1024**3,
            }
            if not all(report["read_targets"].values()):
                raise RuntimeError(
                    "Context read resource target failed; inspect recorded measurements"
                )
        else:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.validate_context_runtime",
                    args.ontology,
                    "--output",
                    str(args.output),
                    "--measure-only",
                ],
                cwd=ROOT,
                check=True,
            )
            report["query_measurement"] = json.loads(
                (args.output / (args.ontology.lower() + ".queries.json")).read_text()
            )
        report.update(
            {
                "status": "pass",
                "ontology_version_id": context.ontology_version_id,
                "context_sha256": context.manifest["artifacts"]["context.sqlite"],
                "artifact_bytes": context.database.stat().st_size,
                "completeness": context.manifest["completeness"],
                "capabilities": context.manifest["capabilities"],
                "diagnostics": context.manifest["diagnostics"],
            }
        )
    except BaseException as error:
        report.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        report.update(
            {
                "wall_seconds": time.time() - started,
                "peak_rss_bytes": peak_rss_bytes(),
            }
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in ("ontology", "status", "wall_seconds", "peak_rss_bytes")
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
