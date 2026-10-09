"""Audit a frozen declaration against observable train/development inputs only."""

from __future__ import annotations

import argparse
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.graph import FEATURE_SCHEMA_V3, build_observable_graph
from exact.repair.graph_schema import declared_metadata, training_metadata
from exact.repair.records import canonical_hash, read_record
from tools.repair.batch import read, sha
from tools.repair.historical_regression import binding, verify_binding


def run(plan_path, output):
    plan = read(plan_path)
    if (
        plan["schema"] != "exact-repair/graph-schema-validation-plan/v1"
        or set(plan["releases"]) != {"train", "development"}
        or plan["expected_cases"] != {"train": 128, "development": 32}
        or plan["heldout_use"] is not False
        or plan["model_fitting"] is not False
    ):
        raise ValueError("Schema validation requires the complete train/development boundary")
    declaration = verify_binding(plan["declaration"])
    metadata = declared_metadata(declaration)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, parents, ids = [], {}, set()
    for split, reference in plan["releases"].items():
        release = verify_binding(reference)
        if len(release["rows"]) != plan["expected_cases"][split]:
            raise ValueError("Schema validation release denominator changed")
        for item in release["rows"]:
            if item["split"] != split or item["status"] != "materialized" or item["case_id"] in ids:
                raise ValueError("Schema validation split, status or case identity changed")
            ids.add(item["case_id"])
            parent = item["structural_parent"]
            if parents.setdefault(parent, split) != split:
                raise ValueError("A schema-validation parent crosses train/development")
            # The evaluator record and all held-out records are deliberately unopened.
            problem = read_record(verify_binding(item["observable"]))
            if problem.content_hash != item["input_hash"]:
                raise ValueError("Schema-validation observable input changed")
            graph = build_observable_graph(
                problem.objects,
                fixed_axioms=problem.fixed_axioms,
                source_axioms=problem.source_axioms,
                target_axioms=problem.target_axioms,
                evidence=dict(problem.evidence),
                feature_schema=FEATURE_SCHEMA_V3,
            )
            missing_nodes = sorted(set(graph.metadata[0]) - set(metadata[0]))
            missing_edges = sorted(set(graph.metadata[1]) - set(metadata[1]))
            row = dict(
                case_id=item["case_id"],
                split=split,
                structural_parent=parent,
                observable=item["observable"],
                input_hash=problem.content_hash,
                graph_metadata=graph.metadata,
                nodes=len(graph.nodes),
                edges=len(graph.edges),
                missing_nodes=missing_nodes,
                missing_edges=missing_edges,
                status="unsupported" if missing_nodes or missing_edges else "admitted",
            )
            if row["status"] == "admitted":
                training_metadata(graph.metadata, declaration)
            path = output / "rows" / (canonical_hash(item["case_id"]) + ".json")
            write_artifact(path, row)
            rows.append(binding(path))
            write_artifact(
                output / "progress.json", dict(recorded_rows=len(rows), case_id=item["case_id"])
            )
    failures = [r for r in rows if read(r["path"])["status"] != "admitted"]
    report = dict(
        schema="exact-repair/graph-schema-validation/v1",
        status="failed" if failures else "complete",
        plan=binding(plan_path),
        declaration=plan["declaration"],
        graph_schema_hash=canonical_hash(declaration),
        declared_node_types=len(metadata[0]),
        declared_edge_types=len(metadata[1]),
        scheduled_cases=plan["expected_cases"],
        recorded_rows=len(rows),
        rows=rows,
        unsupported_rows=failures,
        heldout_cases_opened=False,
        evaluator_records_opened=False,
        model_fitting=False,
        source=sha(__file__),
        limitations=[
            "Input-language admission only; not native reasoner coverage or G0-G2 passage.",
            "Unit tests cover encoder backward and checkpoint identity, not proposal-circuit backward.",
            "Label eligibility, full executable protocols, GPU capacity and decoded development selection remain prerequisites.",
        ],
    )
    write_artifact(output / "report.json", report)
    if failures:
        raise ValueError(
            "Observable graphs exceed the frozen declared schema; see all scheduled rows"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)


if __name__ == "__main__":
    main()
