"""Dependency-checked rollback for the diagnosed coherent endpoint defect."""

from __future__ import annotations

import copy

from exact.repair.candidates import _original_mapping_relation
from exact.repair.records import canonical_hash

# Only the bounded-acquisition repair and its archived pre-optimization state
# are eligible. A later implementation must establish a new compatibility proof.
PREDECESSOR = "9c92a5723ed31ba6ce7d65b0b62e2d09a328ccf07041013637fd64aac431e5bc"
INITIAL_IMPLEMENTATION = "e71392f8cbc31a0932cde613f8357e20b0f6cd5efe824a737d8561ce9544da24"
UNCHANGED_DEPENDENCIES = "7928e52d873dbd474276351dd06aa754694ea1627147f1535070f812d1fd6481"


def recover_endpoint_state(failed, baseline, options, warm_start_hash, dependencies):
    """Restore a proven earlier boundary; retain the later cost and attempt history.

    All later optimization/selection depended on the defective clean-case
    generation. Earlier epoch weights were not retained, so evaluating only the
    last weights would change the frozen selection schedule. Replay from the
    saved pre-optimization boundary, reusing only unaffected acquired labels.
    """
    initial_identity = canonical_hash((options, warm_start_hash, INITIAL_IMPLEMENTATION))
    failed_identity = canonical_hash((options, warm_start_hash, PREDECESSOR))
    pending = baseline.get("pending_acquisition", {})
    collection = pending.get("collection_state", {})
    if not (
        dependencies == UNCHANGED_DEPENDENCIES
        and failed.get("schema") == baseline.get("schema") == "exact-repair/training-state/v3"
        and failed.get("recovery_revision")
        == baseline.get("recovery_revision")
        == "exact-phase-resume/v3.1"
        and failed.get("identity") == failed_identity
        and baseline.get("identity") == initial_identity
        and failed.get("phase") == baseline.get("phase") == "acquisition"
        and failed.get("next_epoch") == options["epochs"]
        and failed.get("next_offset") == 0
        and failed.get("best_state") is None
        and failed.get("best_epoch") is None
        and len(failed.get("history", [])) == options["epochs"]
        and baseline.get("next_epoch")
        == baseline.get("next_offset")
        == baseline.get("optimized")
        == 0
        and not baseline.get("history")
        and not baseline.get("optimizer", {}).get("state")
        and baseline.get("best_state") is None
        and pending.get("epoch") == 0
        and "proposed" in pending
        and collection.get("schema") == "exact-repair/collection-state/v3.2"
        and collection.get("collection_identity")
        == canonical_hash(collection.get("collection_dependencies"))
        and failed.get("elapsed_seconds", -1) >= baseline.get("elapsed_seconds", 0)
        and failed.get("execution_count", 0) > baseline.get("execution_count", 0)
    ):
        raise ValueError("Endpoint recovery dependencies or checkpoint boundary are incompatible")
    scheduled = [row for row in failed["history"] if "generated" in row]
    if not scheduled or any(
        row.get("selection_criterion") is not None
        or not any(
            report.get("status") == "generation_error"
            and report.get("detail")
            == "ValueError: cannot infer complete original relation for endpoint retrieval"
            for report in row["generated"].values()
        )
        for row in scheduled
    ):
        raise ValueError("Endpoint recovery requires the diagnosed unavailable selection history")
    completed = baseline.get("acquisition_completed", [])
    retained = {case_id for epoch, case_id in completed if epoch == 0}
    if len(retained) != len(completed):
        raise ValueError("Endpoint recovery requires only epoch-zero acquisitions")
    retained.add(pending.get("case_id"))
    cases = {case.case_id: case for case, _ in options["training"]}
    if not retained <= cases.keys():
        raise ValueError("Retained acquisitions are not in the frozen training split")
    for case_id in retained:
        # The new fallback cannot affect any reused generation. It is never
        # consulted for these original anchored elementary class mappings.
        for obj in cases[case_id].problem.objects:
            if (
                obj.kind == "mapping"
                and _original_mapping_relation(obj, obj.source_entity, obj.target_entity, "class")
                is None
            ):
                raise ValueError("Retained acquisition depends on changed endpoint generation")
    restored = copy.deepcopy(baseline)
    restored["pending_acquisition"]["case_deadline_exhausted"] = True
    restored["elapsed_seconds"] = failed["elapsed_seconds"]
    restored["execution_count"] = failed["execution_count"]
    lineage = dict(
        migration="coherent-endpoint-preoptimization-rollback/v1",
        source_identity=failed_identity,
        restored_identity=initial_identity,
        reused="pre-optimization model/optimizer/RNG, unaffected acquisitions, initial labels",
        invalidated="later optimization and all generated development selection",
        retained_acquisition_cases=sorted(retained),
        discarded_next_epoch=failed["next_epoch"],
        elapsed_seconds=failed["elapsed_seconds"],
        budgets_reset=False,
    )
    restored["recovery_lineage"] = [*failed.get("recovery_lineage", []), lineage]
    return restored
