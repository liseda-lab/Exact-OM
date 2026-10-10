"""Freeze closed TRAIN observations once, including required missing swap masks.

This is offline preparation: no client, reasoner, acquisition or TEST reader.
The fixed response denominator and the original comparison denominator differ.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from exact.repair.records import canonical_hash, read_record
from exact.repair.semantic_fidelity import (
    AGGREGATION_REVISION,
    FIDELITY_TRAINING_SCHEMA,
    AnnotationScheduleV3,
    ValidatedFidelityAggregateV3,
    aggregate_comparisons,
    read_fidelity_training_artifact,
)
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.shared_release import authenticate, bound, immutable, validate_completion


def combine(schedules, observations):
    """The frozen original/swap calls form one quorum, never separate terms."""
    if not schedules or len(schedules) > 2:
        raise ValueError("Expected one original and at most one required audit")
    first = schedules[0]
    if any(
        s.packet != first.packet or s.parser_versions != first.parser_versions
        or len(s.slots) != 1 or s.quorum != 1 or s.max_disagreement != 0
        for s in schedules
    ):
        raise ValueError("Comparison packet/parser or single-call schedule changed")
    orders = [json.loads(s.slots[0]["parameters"]["messages"][1]["content"])["swapped"] for s in schedules]
    if orders not in ([False], [False, True]):
        raise ValueError("Required original/swapped presentation changed")
    schedule = AnnotationScheduleV3(
        first.packet,
        tuple(slot for s in schedules for slot in s.slots),
        first.parser_versions,
        quorum=len(schedules),
        max_disagreement=0.0,
    )
    return ValidatedFidelityAggregateV3(
        schedule, tuple(observations), aggregate_comparisons(observations, schedule=schedule)
    )


def group_rows(rows):
    groups = defaultdict(list)
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate scheduled row")
        ids.add(row["id"])
        groups[row["comparison_id"]].append(row)
    for group in groups.values():
        group.sort(key=lambda r: r["swapped"])
        if [r["swapped"] for r in group] not in ([False], [False, True]):
            raise ValueError("Missing original or repeated presentation")
        if any(
            r[k] != group[0][k]
            for r in group for k in ("case_id", "parent", "family", "control")
        ):
            raise ValueError("Original/swap case dependency changed")
    return groups


def prepare(campaign, review_ref, output):
    from tools.repair.corrective_semantics import validate_manifest
    from tools.repair.train_packet_admission import validate_rows

    campaign, output = Path(campaign), Path(output)
    review = bound(review_ref)
    predecessor = bound(review["preparation"])
    registry = read(campaign / "supervisor/registry.json")
    manifests, rows, authenticated_outputs, contexts = [], [], {}, {}
    for ref, receipt in zip(predecessor["manifests"], review["receipts"], strict=True):
        manifest = validate_manifest(bound(ref))
        validate_rows(manifest)
        terminal, outputs, _, job = validate_completion(receipt)
        registered = next(r for r in registry["runs"] if r["id"] == job["id"])
        if any(registered[k] != receipt[k] for k in ("dispatch_nonce", "step_id")):
            raise ValueError("Live registered annotation owner changed")
        work = Path(terminal["work"])
        for relative, digest in outputs.items():
            path = work / relative
            authenticate(dict(path=str(path), sha256=digest))
            authenticated_outputs[str(path)] = digest
        report = read(work / "report.json")
        if (report["status"] != "complete" or report["scheduled"] != len(manifest["slots"])
                or report["recorded"] != len(manifest["slots"])):
            raise ValueError("Annotation shard denominator or terminal status changed")
        results = {r["id"]: r for r in report["rows"]}
        if len(results) != len(manifest["slots"]) or set(results) != {r["id"] for r in manifest["slots"]}:
            raise ValueError("Annotation row identities changed")
        for row in manifest["slots"]:
            if row["status"] == "eligible":
                packet = read_record(bound(row["packet"]))
                contexts[packet.content_hash] = dict(
                    original_packet=bound(row["original_packet"]), proof=bound(row["transport_proof"])
                )
            rows.append(dict(
                id=row["id"], comparison_id=row["comparison_id"], case_id=row["case_id"],
                parent=row["structural_parent"], family=row["family"], control=row["control"],
                swapped=row["swapped"], packet=row["packet"], packet_status=row["status"],
                result=results[row["id"]], directory=str(work / row["id"]),
                profile=manifest["profiles"][manifest["profile"]],
            ))
        manifests.append(ref)
    groups = group_rows(rows)
    if len(rows) != 308 or len(groups) != 256 or sum(r["swapped"] for r in rows) != 52:
        raise ValueError("Frozen TRAIN denominator changed")
    if {r["id"] for r in review["rows"]} != {r["id"] for r in rows}:
        raise ValueError("Completion review refers to another denominator")

    def terminal_record(path):
        if str(path) not in authenticated_outputs:
            raise ValueError("Annotation artifact is not a terminal output")
        return read_record(read(path))

    aggregates, admitted, masks = [], [], []
    for comparison_id, group in groups.items():
        schedules, observations = [], []
        for row in group:
            directory = Path(row["directory"])
            if row["packet_status"] != "eligible":
                if row["result"].get("artifact") is not None:
                    raise ValueError("Unavailable packet acquired an annotation")
                continue
            schedule = terminal_record(directory / "schedule.json")
            packet = read_record(bound(row["packet"]))
            if schedule.packet != packet or packet.split != "train":
                raise ValueError("Annotation changed the frozen TRAIN packet")
            # The schedule and observations revalidate the exact wire model/provider.
            for slot in schedule.slots:
                if (slot["parameters"]["model"] != row["profile"]["model"]
                        or canonical_hash(slot["parameters"]["provider"]) != canonical_hash(row["profile"]["provider"])):
                    raise ValueError("Selected teacher model/provider changed")
            schedules.append(schedule)
            artifact = row["result"].get("artifact")
            if artifact:
                if artifact != str(directory / "labels.json"):
                    raise ValueError("Annotation artifact belongs to another row")
                authenticate(dict(path=artifact, sha256=authenticated_outputs[artifact]))
                records = read_fidelity_training_artifact(Path(artifact), "train")
                values = [a for pairs in records.values() for _, a in pairs]
                if len(values) != 1 or values[0].schedule != schedule:
                    raise ValueError("Per-call aggregate schedule mismatch")
                observations.extend(values[0].observations)
        aggregate = None
        if len(schedules) == len(group):
            aggregate = combine(schedules, observations)
            aggregates.append(dict(packet=aggregate.schedule.packet.to_dict(), comparison=aggregate.to_dict()))
        eligible = bool(aggregate and aggregate.global_target_eligible)
        if eligible:
            admitted.append(aggregates[-1])
        masks.append(dict(
            comparison_id=comparison_id, case_id=group[0]["case_id"], parent=group[0]["parent"],
            family=group[0]["family"], control=group[0]["control"],
            scheduled_rows=[r["id"] for r in group], required_calls=len(group),
            valid_responses=len(observations), weak_comparison_mask=eligible,
            symbolic_mask_unchanged=True,
            status=("eligible" if eligible else
                    aggregate.result["abstention_reason"] if aggregate else "unavailable_packet"),
            aggregate_hash=aggregate.content_hash if aggregate else None,
        ))
    immutable(output / "all-aggregates.json", dict(comparisons=aggregates))
    immutable(output / "rating-contexts.json", contexts)
    immutable(output / "labels.json", dict(
        schema=FIDELITY_TRAINING_SCHEMA, aggregation_revision=AGGREGATION_REVISION,
        comparisons=admitted,
    ))
    labels = read_fidelity_training_artifact(output / "labels.json", "train")
    immutable(output / "masks.json", dict(rows=masks, scheduled_responses=308, original_comparisons=256))
    eligible = [m for m in masks if m["weak_comparison_mask"]]
    result = dict(
        schema="exact-repair/shared-weak-label-release/v1", status="frozen",
        predecessor=review_ref, manifests=manifests,
        labels=binding(output / "labels.json"), masks=binding(output / "masks.json"),
        all_aggregates=binding(output / "all-aggregates.json"),
        rating_contexts=binding(output / "rating-contexts.json"),
        scheduled_responses=308, original_comparisons=256, scheduled_swaps=52,
        valid_responses=sum(m["valid_responses"] for m in masks),
        eligible_comparisons=len(eligible), unavailable_comparisons=256-len(eligible),
        covered_cases=len({m["case_id"] for m in eligible}),
        covered_parents=len({m["parent"] for m in eligible}),
        family_coverage=dict(Counter(m["family"] for m in eligible)),
        control_coverage=dict(Counter(m["control"] for m in eligible)),
        aggregate_terms=sum(len(v) for v in labels.values()),
        no_individual_and_joint_double_count=True, missing_weak_masks_symbolic=False,
        cohort="generated_only", independent_semantic_evidence=False, missing_real_coverage=True,
        hosted_calls=0, native_calls=0, test_outcomes_opened=False,
        costs_reset=False, phase_reservations=review["phase_reservations"],
        prior_reserved_usd=review["phase_reserved_usd"],
    )
    immutable(output / "prepared.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.campaign, binding(args.review), args.output)


if __name__ == "__main__":
    main()
