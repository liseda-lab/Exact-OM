"""Prepare native public populations and original local queries before the G4 freeze."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tools.experiment_resources import memory_limit_bytes, process_tree_rss_bytes
from tools.prepared_batch import binding, read, verified, write
from tools.storage_guard import stop_owned_worker


def _check_resources(pause_paths, limit):
    if any(Path(path).exists() for path in pause_paths):
        raise RuntimeError("Final population preparation has an intentional pause/STOP")
    if process_tree_rss_bytes() > limit:
        raise RuntimeError("Final population preparation exceeds allocation memory allowance")


def _run(command, log_path, pause_paths):
    """Bound the native child's memory and honor STOP without touching other workers."""
    limit = memory_limit_bytes()
    _check_resources(pause_paths, limit)
    with Path(log_path).open("a") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while True:
                try:
                    result = process.wait(timeout=2)
                    if result:
                        raise subprocess.CalledProcessError(result, command)
                    _check_resources(pause_paths, limit)
                    return
                except subprocess.TimeoutExpired:
                    _check_resources(pause_paths, limit)
        except BaseException:
            stop_owned_worker(process)
            raise


def validate_protocol(path):
    """Validate public input hashes only, without loading ontologies or labels."""
    protocol = read(path)
    fixed = {
        "schema_version": 1,
        "kind": "final_native_populations",
        "entity_kinds": ["class"],
        "native_threads": 2,
        "selection_eligible": False,
        "hosted_calls": False,
        "rationales": False,
    }
    extra = {"ontologies", "cases", "source", "output", "pause_paths", "filter_ignored"}
    if set(protocol) != set(fixed) | extra or any(
        type(protocol.get(k)) is not type(v) or protocol[k] != v for k, v in fixed.items()
    ):
        raise ValueError("Final population protocol differs from its public-only contract")
    if not isinstance(protocol["filter_ignored"], bool):
        raise ValueError("Alignment eligibility policy must be explicit")
    if set(protocol["ontologies"]) != {"NCIT", "DOID", "SNOMED", "FMA"}:
        raise ValueError("Prepare exactly the four declared final ontologies")
    output = Path(protocol["output"])
    if not output.is_absolute():
        raise ValueError("Population output requires an absolute path")
    for row in protocol["ontologies"].values():
        if set(row) != {"input", "source_options"}:
            raise ValueError("Only public ontology and native source options are permitted")
        ontology = verified(row["input"])
        if not ontology.is_absolute() or ontology.is_relative_to(output):
            raise ValueError("Native input must be outside the fresh output root")
        options = row["source_options"]
        if not isinstance(options, dict) or set(options) - {
            "include_abox",
            "label_properties",
            "imports",
            "import_policy",
        }:
            raise ValueError("Unexpected native source options")
        if options.get("import_policy", "strict") != "strict":
            raise ValueError("Final population requires strict complete native imports")
        for item in options.get("imports", {}).values():
            verified(item)
    expected_cases = {"H0": ("NCIT", "DOID"), "H1": ("SNOMED", "FMA"), "H2": ("SNOMED", "NCIT")}
    if set(protocol["cases"]) != set(expected_cases):
        raise ValueError("All three predeclared final cases are required")
    for name, pair in expected_cases.items():
        row = protocol["cases"][name]
        if set(row) != {"source", "target", "public_queries", "query_count", "source_count"}:
            raise ValueError("Final case must contain public inputs only")
        if (row["source"], row["target"]) != pair:
            raise ValueError("Final ontology pair changed")
        verified(row["public_queries"])
        if any(type(row[k]) is not int or row[k] <= 0 for k in ("query_count", "source_count")):
            raise ValueError("Bound original public query counts are required")
    expected = {
        "tools/prepare_final_populations.py",
        "exact/experiments/public_inference.py",
        "exact/experiments/inputs.py",
        "exact/io/sources/owl.py",
        "exact/ontology/__init__.py",
        "exact/ontology/view_contract.py",
    }
    if set(protocol["source"]) != expected:
        raise ValueError("Native population implementation must be fully bound")
    code = Path(__file__).resolve().parents[1]
    for name, item in protocol["source"].items():
        if verified(item) != code / name:
            raise ValueError("Native preparation source differs from its frozen checkout")
    if not protocol["pause_paths"] or any(
        not Path(p).is_absolute() for p in protocol["pause_paths"]
    ):
        raise ValueError("Absolute supervisor and runtime STOP paths are required")
    return protocol


def prepare_one(recipe_path, name):
    """Run only native loading and eligibility enumeration, with no fitting or models."""
    from exact.experiments.public_inference import prepare_population

    protocol = validate_protocol(recipe_path)
    row = protocol["ontologies"][name]
    _check_resources(protocol["pause_paths"], memory_limit_bytes())
    prepare_population(
        verified(row["input"]),
        Path(protocol["output"]) / name / "population.txt",
        entity_kinds=protocol["entity_kinds"],
        filter_ignored=protocol["filter_ignored"],
        source_options=row["source_options"],
    )


def run_diagnostic(recipe_path):
    """Checkpoint each unique native population and original-query inventory once."""
    from exact.experiments.inputs import prepare_pool
    from exact.experiments.public_inference import validate_population

    recipe_path = Path(recipe_path)
    protocol = validate_protocol(recipe_path)
    output = Path(protocol["output"])
    output.mkdir(parents=True, exist_ok=True)
    populations, cases = {}, {}
    for name, row in protocol["ontologies"].items():
        manifest = output / name / "population.txt.manifest.json"
        _run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--protocol",
                str(recipe_path),
                "--one",
                name,
            ],
            output / (name + ".log"),
            protocol["pause_paths"],
        )
        record, _ = validate_population(manifest)
        if (
            record["ontology"] != row["input"]
            or record.get("source_options", {}) != row["source_options"]
        ):
            raise ValueError("Native population identity differs from its declaration")
        populations[name] = binding(manifest)
    for case_id, row in protocol["cases"].items():
        _check_resources(protocol["pause_paths"], memory_limit_bytes())
        destination = output / case_id / "local"
        receipt = destination / "test.inputs.json"
        if receipt.exists():
            record = read(receipt)
            for item in record["outputs"].values():
                verified(item)
        else:
            record = prepare_pool(
                verified(row["public_queries"]), destination, role="test", expose_labels=False
            )
        if (
            record.get("input_sha256") != row["public_queries"]["sha256"]
            or record.get("labels_exposed") is not False
            or record.get("query_grouping") != "original_rows_preserved_with_positional_qid"
            or record.get("original_query_rows") != row["query_count"]
            or record.get("sources") != row["source_count"]
            or set(record["outputs"])
            != {"candidates", "source_universe", "queries", "public_queries"}
        ):
            raise ValueError("Original public query inventory changed")
        cases[case_id] = {
            "source_population": populations[row["source"]],
            "target_population": populations[row["target"]],
            "local_queries": binding(receipt),
            "query_count": row["query_count"],
        }
    validate_protocol(recipe_path)
    report = output / "preparation-receipt.json"
    write(
        report,
        {
            "status": "complete",
            "selection_eligible": False,
            "recipe": binding(recipe_path),
            "populations": populations,
            "cases": cases,
            "reference_labels_used": False,
            "hosted_requests": 0,
            "hosted_tokens": 0,
        },
        immutable=True,
    )
    write(
        output / "completion.json",
        {
            "status": "complete",
            "selection_eligible": False,
            "diagnostic": binding(report),
        },
        immutable=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--one", choices=["NCIT", "DOID", "SNOMED", "FMA"], required=True)
    args = parser.parse_args()
    prepare_one(args.protocol, args.one)
