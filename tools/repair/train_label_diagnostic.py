"""Charged fixed-TRAIN qualification of the post-verification label correction."""

import argparse
from pathlib import Path
import time

from exact.repair.workers import bounded_call
from tools.repair.batch import read
from tools.repair.generated_acquisition import _assignment_payload
from tools.repair.historical_regression import binding
from tools.repair.prepare import case_from_dict, read_label_cache
from tools.repair.shared_release import bound, immutable
from exact.repair.learning import SemanticTargetSpec
from exact.repair.records import canonical_hash


def run(plan_path, output):
    plan, output = read(plan_path), Path(output)
    if (
        plan.get("purpose") != "post_verification_label_qualification"
        or plan.get("test_opened") is not False
    ):
        raise ValueError("Expected fixed TRAIN diagnostic")
    rows = []
    for index, declaration in enumerate(plan["rows"]):
        prior = bound(declaration["prior_row"])
        generated = bound(prior["generated"])
        case = case_from_dict(generated["case"])
        assignment = tuple(declaration["assignment"])
        if case.split != "train" or case.case_id != prior["case_id"]:
            raise ValueError("Diagnostic changed TRAIN identity")
        cache = read_label_cache(prior["cache"], Path(prior["cache"]["path"]).parent, case)
        historical = next(label for label in cache.labels if label.assignment == assignment)
        if historical.feasible is not None:
            raise ValueError("Diagnostic is restricted to the declared unfinished labels")
        target = output / f"case-{index:02}"
        receipt = target / "receipt.json"
        if receipt.exists():
            value = read(receipt)
            if value["declaration"] != declaration:
                raise ValueError("Diagnostic resume dependencies changed")
            rows.append(binding(receipt))
            continue
        if (target / "started.json").exists():
            raise ValueError("Interrupted diagnostic needs explicit recovery; no renewed clock")
        immutable(target / "started.json", dict(epoch=time.time(), declaration=declaration))
        protocol = bound(bound(prior["plan"])["protocol"])
        profile = tuple(sorted(protocol["objective"]["edit_weights"].items()))
        semantic_target = SemanticTargetSpec(
            canonical_hash(case.probes),
            protocol["teacher"]["family_weights"]["desired"],
            protocol["teacher"]["family_weights"]["unwanted"],
        )
        result = bounded_call(
            _assignment_payload,
            case,
            assignment,
            profile,
            semantic_target=semantic_target,
            support_enabled=protocol["model"].get("support_enabled", False),
            output_directory=str(target / "cache"),
            evidence_directory=target / "native",
            timeout=plan["call_seconds"],
            memory_mb=8192,
        )
        value = dict(
            declaration=declaration,
            status=result.status,
            detail=result.detail,
            cleanup_complete=result.cleanup_complete,
            resources=dict(result.resource_usage),
            artifact=result.value if result.status == "complete" else None,
            prior_outcome_preserved=True,
            primary_training=False,
        )
        if value["artifact"]:
            new = read_label_cache(value["artifact"], target / "cache", case)
            value.update(feasible=new.labels[0].feasible, usable=new.labels[0].usable)
        immutable(receipt, value)
        rows.append(binding(receipt))
        if not result.cleanup_complete or result.status == "error":
            raise RuntimeError("Native diagnostic implementation/cleanup failure")
    immutable(
        output / "report.json",
        dict(
            status="complete",
            plan=binding(plan_path),
            rows=rows,
            primary_training=False,
            old_labels_overwritten=False,
            test_opened=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)


if __name__ == "__main__":
    main()
