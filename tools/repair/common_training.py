"""Offline, authenticated adapter from a closed shared release to paired training.

The manifest contains references, not a second copy of symbolic labels. Loading
never acquires missing labels, opens TEST inputs, or resolves a teacher gate.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from exact.repair.records import canonical_hash
from tools.repair.prepare import case_from_dict, read_label_cache
from tools.repair.shared_release import authenticate, bound, validate_completion

SCHEMA = "exact-repair/common-training-preparation/v1"
CONDITIONS = ("symbolic", "symbolic_plus_llm")


def _case_identity(case, declaration, split):
    if case.split != split or any(
        getattr(case, key) != declaration[key]
        for key in ("case_id", "family", "control", "structural_parent")
    ):
        raise ValueError("Shared training case identity/split changed")


def load_release(reference, audit_run):
    """Return all declared TRAIN/DEV cases and only their committed union caches."""
    completion, outputs, _, _ = validate_completion(audit_run)
    release_path = authenticate(reference)
    relative = str(release_path.relative_to(Path(completion["work"])))
    if outputs.get(relative) != reference["sha256"]:
        raise ValueError("Shared release is not a nonce-bound terminal output")
    release = bound(reference)
    if (
        release.get("schema") != "exact-repair/common-generated-shared-release/v1"
        or release.get("status") != "complete"
        or release.get("acquisition_closed") is not True
        or release.get("maximum_rounds") != 2
        or release.get("heldout_outcomes_opened") is not False
    ):
        raise ValueError("Expected a closed two-round shared release with TEST unopened")
    # Authenticate both acquisition lineages; they remain evidence and costs,
    # including unavailable/interrupted rows, rather than new work to replay.
    prior = bound(release["prior_release"])
    for run in [*prior["runs"], *release["refinement_runs"]]:
        validate_completion(run)
    if release.get("operation") == "authorized_unknown_label_correction":
        bound(release["correction_amendment"])
        bound(release["correction_predecessor"])
        for run in [release["correction_qualification"], *release["correction_histories"],
                    *release["correction_runs"]]:
            validate_completion(run)
    for item in release["common"].values():
        if isinstance(item, dict) and "path" in item and "sha256" in item:
            authenticate(item)
    coverage = bound(release["coverage"])
    cases, caches, label_rows, train_rows = [], {}, [], []
    for shard_ref in release["shards"]:
        shard = bound(shard_ref)
        if shard["shared_conditions"] != list(CONDITIONS):
            raise ValueError("Both conditions must share the same shards")
        for row_ref in shard["rows"]:
            row = bound(row_ref)
            declaration = row["input"]
            original = case_from_dict(bound(declaration["evaluator"]))
            _case_identity(original, declaration, "train")
            if original.problem.content_hash != declaration["input_hash"]:
                raise ValueError("Original shared TRAIN input changed")
            # Failed generation still contributes the original case and a missing
            # cache; it must not gain a newly generated inventory at fitting time.
            case = (
                case_from_dict(bound(row["generated"])["case"])
                if row.get("generated")
                else original
            )
            _case_identity(case, row, "train")
            if row.get("cache"):
                cache = read_label_cache(row["cache"], Path(row["cache"]["path"]).parent, case)
                deps = row["collection_dependencies"]
                if (
                    dict(cache.hashes) != deps["hashes"]
                    or list(cache.candidate_counts) != deps["candidate_counts"]
                ):
                    raise ValueError("Shared cache collection dependencies changed")
                caches[case.case_id] = cache
            if len(row["attempts"]) != row["scheduled_attempts"]:
                raise ValueError("Shared assignment denominator changed")
            cases.append(case)
            train_rows.append(row)
            label_rows.append(
                dict(
                    case_id=case.case_id,
                    status=row["scientific_status"],
                    process_status=row["process_status"],
                    release_row=row_ref,
                )
            )
    dev = bound(release["development"])
    for row in dev["rows"]:
        case = case_from_dict(bound(row["evaluator"]))
        _case_identity(case, row, "development")
        if case.problem.content_hash != row["input_hash"] or row.get("cache") is not None:
            raise ValueError("Unexpected DEV input/cache in the unopened shared release")
        cases.append(case)
        label_rows.append(dict(case_id=case.case_id, status="missing_development_cache"))
    if (
        len(train_rows) != release["expected_train_cases"]
        or len(dev["rows"]) != release["expected_development_cases"]
        or sum(row["scheduled_attempts"] for row in train_rows)
        != release["expected_assignment_slots"]
        or coverage["overall"]["cases"] != len(train_rows)
    ):
        raise ValueError("Shared case/assignment denominator changed")
    ids, parents = set(), {}
    for case in cases:
        if (
            case.case_id in ids
            or parents.setdefault(case.structural_parent, case.split) != case.split
        ):
            raise ValueError("Duplicate case or parent crosses TRAIN/DEV")
        ids.add(case.case_id)
    return (
        tuple(cases),
        caches,
        dict(
            common_acquisition_closed=True,
            common_release=reference,
            common_release_audit=audit_run,
            common_data_identity=canonical_hash(
                (reference, release["shards"], release["development"])
            ),
            requested=len(cases),
            produced=len(cases),
            origin="generated",
            cohort=release["cohort"],
            missing_real_coverage=release["missing_real_coverage"],
            label_rows=label_rows,
            selected_counts=dict(Counter(c.split for c in cases)),
            labelled_cases=len(caches),
            expected_assignment_slots=release["expected_assignment_slots"],
            historical_label_costs="linked once through common_release; no new acquisition charge",
            label_seconds=0.0,
            label_cpu_seconds=0.0,
            coverage=release["coverage"],
            fit_execution_authorized=False,
        ),
    )


def load_common_preparation(value):
    """Called by the normal --prepared entry point after its content-hash check."""
    from exact.repair.protocol import load_protocol_v3, training_projection_v3

    if value.get("schema") != SCHEMA:
        raise ValueError("Unsupported common training preparation")
    protocol = training_projection_v3(load_protocol_v3(authenticate(value["protocol"])))
    cases, caches, report = load_release(value["release"], value["audit_run"])
    validate_closed_preparation(cases, report, protocol, case_limit=None, retry_labels=False)
    report.update(protocol_hash=canonical_hash(protocol), case_limit=None)
    return cases, caches, report


def validate_closed_preparation(cases, preparation, protocol, *, case_limit, retry_labels):
    """A closed common release cannot silently turn into condition-specific data."""
    if not preparation.get("common_acquisition_closed"):
        return
    if case_limit is not None or retry_labels or protocol["training"]["sampled_assignments"] != 0:
        raise ValueError("Closed common acquisition forbids case filtering or reacquisition")
    expected = [c.case_id for c in cases]
    rows = [r["case_id"] for r in preparation.get("label_rows", [])]
    if len(rows) != len(set(rows)) or set(rows) != set(expected):
        raise ValueError("Closed preparation must retain every TRAIN/DEV label row")
    dev = [c.case_id for c in cases if c.split == "development"]
    if protocol["training"].get("development_case_ids") != dev:
        raise ValueError("Closed preparation DEV order/denominator changed")


def development_schedule(declarations, *, final_epoch=50, repair_seconds=120.0):
    """Freeze balanced paired identities before seeing any decoded DEV output.

    Each of the 32 cases gets exactly one semantic slot per condition. Seven
    paired slots receive independent order swaps (14 calls >= ceil(.2 * 64)).
    The 120-second repair allowance is a planning candidate pending admission.
    """
    import math

    if len(declarations) != 32 or len({r["case_id"] for r in declarations}) != 32:
        raise ValueError("Expected all 32 distinct declared DEV cases")
    if (
        final_epoch <= 5
        or final_epoch > 50
        or not math.isfinite(repair_seconds)
        or repair_seconds <= 0
    ):
        raise ValueError("Invalid common endpoint or DEV repair limit")
    groups = [(seed, epoch) for seed in (13, 37, 73) for epoch in (5, final_epoch)]
    quotas = (6, 5, 5, 6, 5, 5)
    # Interleave family/control strata; this order uses input identity only.
    buckets = {}
    for row in declarations:
        if row["split"] != "development":
            raise ValueError("Semantic selection can only use declared DEV identities")
        buckets.setdefault((row["family"], row["control"]), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: r["case_id"])
    ordered = [
        bucket[i]
        for i in range(max(map(len, buckets.values())))
        for _, bucket in sorted(buckets.items())
        if i < len(bucket)
    ]
    assigned = {group: [] for group in groups}
    cursor = 0
    for row in ordered:
        while len(assigned[groups[cursor % 6]]) >= quotas[cursor % 6]:
            cursor += 1
        assigned[groups[cursor % 6]].append(row["case_id"])
        cursor += 1
    rows, slots = [], {}
    for group_index, (seed, epoch) in enumerate(groups):
        subset = assigned[(seed, epoch)]
        audited = set(subset[: 2 if group_index == 0 else 1])
        for condition in CONDITIONS:
            for case in declarations:
                key = dict(
                    seed=seed, supervision_condition=condition, epoch=epoch, case_id=case["case_id"]
                )
                semantic = case["case_id"] in subset
                swapped = semantic and case["case_id"] in audited
                reserve = (182.0 + (90.0 if swapped else 0.0)) if semantic else 0.0
                row = dict(
                    selection_slot=key,
                    slot_id=canonical_hash(key),
                    family=case["family"],
                    control=case["control"],
                    structural_parent=case["structural_parent"],
                    semantic_scheduled=semantic,
                    swapped=swapped,
                    repair_seconds=repair_seconds,
                    annotation_reserve_seconds=reserve,
                    case_seconds=repair_seconds + reserve,
                    status="scheduled_not_attempted",
                )
                rows.append(row)
                if semantic:
                    slots[row["slot_id"]] = dict(selection_slot=key, swapped=swapped)
    return dict(
        schema="exact-repair/common-development-schedule/v1",
        execution_authorized=False,
        schedule_status="candidate_pending_teacher_service_and_runtime_admission",
        rows=rows,
        expected_rows=len(rows),
        unique_semantic_slots=len(slots),
        independent_swapped_calls=sum(r["swapped"] for r in rows),
        frozen_annotation_slots=slots,
        post_decode_reserve_seconds=182.0,
        service_request_seconds=90,
        repair_seconds=repair_seconds,
        missing_slots="retain unavailable; never replace or enlarge denominator",
        rule="input-only round robin across family/control strata; matched conditions",
        case_worker_seconds=sum(r["case_seconds"] for r in rows),
        final_epoch=final_epoch,
        device_role="owned RTX2080Ti DEV; oversized routes to owned5090 boundary",
    )
