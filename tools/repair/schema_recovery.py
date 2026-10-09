"""Frozen-model schema admission and linked, budget-debited evaluation recovery."""

from __future__ import annotations

import argparse
import fcntl
import math
import time
from collections import Counter
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, read_record
from tools.repair.batch import read, sha
from tools.repair.expanded_corpus import binding, bound
from tools.repair.expanded_profile import checked_checkpoint, checkpoint
from tools.repair import fresh_evaluation as fresh

ERROR = "ValueError: Unregistered edge type: ('evidence', 'supports', 'statement')"
CAUSE = "frozen_model_graph_schema_admission"


def remaining_budget(previous, payload, row, resources):
    if (
        previous["status"] != "complete"
        or not previous["cleanup_complete"]
        or payload["status"] != "generation_error"
        or payload["detail"] != ERROR
        or payload["row_id"] != row["id"]
        or previous["row"] != row
    ):
        raise ValueError("Only the confirmed original schema error admits replay")
    usage = dict(previous["resources"])
    used_wall = previous["elapsed_seconds"]
    used_generation = payload["generation_resources"]["wall_seconds"]
    used_cpu = usage["cpu_seconds"]
    if any(not math.isfinite(v) or v < 0 for v in (used_wall, used_generation, used_cpu)):
        raise ValueError("Original schema error has invalid resource accounting")
    return dict(
        wall_seconds=max(0, row["seconds"] - used_wall),
        generation_seconds=max(0, resources["generation_seconds"] - used_generation),
        cpu_seconds=max(0, row["cpu_seconds"] - used_cpu),
        prior_wall_seconds=used_wall,
        prior_generation_seconds=used_generation,
        prior_cpu_seconds=used_cpu,
    )


def validate_remaining_budget(budget, row, resources):
    for key, original, prior in (
        ("wall_seconds", row["seconds"], "prior_wall_seconds"),
        ("generation_seconds", resources["generation_seconds"], "prior_generation_seconds"),
        ("cpu_seconds", row["cpu_seconds"], "prior_cpu_seconds"),
    ):
        if (
            not math.isfinite(budget[key])
            or not math.isfinite(budget[prior])
            or not 0 < budget[key] <= original
            or budget[prior] < 0
            or budget[key] + budget[prior] > original + 1e-8
        ):
            raise ValueError("Schema recovery exceeded an original scientific budget")


def preflight(schedule):
    """Use observable inputs and existing checkpoint metadata; never evaluator labels."""
    import torch
    from exact.repair.graph import EffectivePreparation
    from exact.repair.model import graph_schema_compatibility
    from exact.repair.retrieval import retrieve_vocabulary

    states, protocols, graphs, rows = {}, {}, {}, []
    for arm in schedule["arms"]:
        protocols[arm["id"]] = bound(arm["protocol"])
        if arm["kind"] != "learned":
            continue
        fresh.pilot.check_binding(arm["model"])
        saved = torch.load(arm["model"]["path"], map_location="cpu", weights_only=True)
        states[arm["id"]] = (saved["metadata"], saved["config"]["encoder"])
        del saved
    for row in schedule["rows"]:
        arm = next(a for a in schedule["arms"] if a["id"] == row["arm_id"])
        item = schedule["cases"][row["case_index"]]
        if arm["kind"] != "learned" or item["status"] != "materialized":
            rows.append(dict(row_id=row["id"], status="compatible_or_control", compatible=True))
            continue
        protocol = protocols[arm["id"]]
        options = fresh.pilot.generation_options(protocol, Path("/unused-schema-preflight-cache"))
        preparation = EffectivePreparation(
            **{
                k: options[k]
                for k in (
                    "max_graph_nodes",
                    "max_graph_edges",
                    "max_explanations",
                    "max_text_tokens",
                    "pair_factor_limit_per_object",
                    "pair_max_pairs",
                    "pair_max_factors",
                    "retrieval_config",
                )
            },
            revision="v3",
        )
        key = canonical_hash((item["observable"], preparation.content_hash))
        if key not in graphs:
            problem = read_record(bound(item["observable"]))
            try:
                vocabulary = retrieve_vocabulary(problem, config=preparation.retrieval_config)
                graphs[key] = preparation.graph(problem, vocabulary)
            except ValueError as error:
                if str(error) != "cannot infer complete original relation for endpoint retrieval":
                    raise
                graphs[key] = None
        graph = graphs[key]
        if graph is None:
            rows.append(
                dict(
                    row_id=row["id"],
                    status="unsupported_original_mapping_bundle",
                    compatible=False,
                    observable=item["observable"],
                )
            )
            continue
        metadata, encoder = states[arm["id"]]
        compatibility = graph_schema_compatibility(metadata, encoder, graph)
        rows.append(
            dict(
                row_id=row["id"],
                status="compatible" if compatibility["compatible"] else "unavailable_model_schema",
                **compatibility,
                observable=item["observable"],
                model=arm["model"],
                graph_hash=canonical_hash(graph),
                metadata_hash=canonical_hash(metadata),
            )
        )
    return dict(
        schema="exact-repair/model-schema-preflight/v1",
        rows=rows,
        counts=dict(Counter(row["status"] for row in rows)),
        evaluator_records_opened=False,
        checkpoint_parameters_changed=False,
        model_source_sha256=sha(Path(__file__).resolve().parents[2] / "exact/repair/model.py"),
        graph_source_sha256=sha(Path(__file__).resolve().parents[2] / "exact/repair/graph.py"),
    )


def one(schedule, row, entry, compatibility, output, identity):
    from exact.repair.workers import bounded_call

    receipt = output / "rows" / (row["id"] + ".json")
    expected = canonical_hash((identity, row, entry, compatibility))
    saved = checked_checkpoint(receipt, expected)
    if saved:
        fresh.validate_payloads(saved)
        if not saved["cleanup_complete"] or (output / "inflight" / (row["id"] + ".json")).exists():
            raise RuntimeError("Schema recovery requires cleanup/ownership reconciliation")
        return saved
    previous = bound(entry["previous"]) if entry.get("previous") else None
    if previous:
        fresh.validate_payloads(previous)
        if previous["row"] != row or not previous["cleanup_complete"]:
            raise ValueError("Original row identity or cleanup mismatch")
        payload = bound(previous["result"]) if previous.get("result") else None
        unsupported = (
            payload
            and payload.get("status") == "generation_error"
            and payload.get("detail")
            == "ValueError: cannot infer complete original relation for endpoint retrieval"
        )
        if unsupported:
            revised_path = output / "payloads" / row["id"] / "result.json"
            write_artifact(
                revised_path,
                dict(
                    payload,
                    status="unsupported_original_mapping_bundle",
                    original_result=previous["result"],
                    classification_revision="explicit-input-support/v1",
                ),
            )
            revised_ref = binding(revised_path)
            return checkpoint(
                receipt,
                expected,
                **{
                    k: v
                    for k, v in previous.items()
                    if k not in ("identity", "content_hash", "result", "payloads")
                },
                result=revised_ref,
                payloads=[revised_ref],
                reuse_of=entry["previous"],
                reuse_source=entry.get("reuse_source"),
                additional_elapsed_seconds=0,
                recovery_action="reclassified_original_unsupported_without_rerun",
            )
        if not payload or payload.get("detail") != ERROR:
            return checkpoint(
                receipt,
                expected,
                **{k: v for k, v in previous.items() if k not in ("identity", "content_hash")},
                reuse_of=entry["previous"],
                reuse_source=entry.get("reuse_source"),
                additional_elapsed_seconds=0,
                recovery_action="reused_original_unchanged",
            )
    else:
        payload = None
    common = dict(
        row=row,
        cleanup_complete=True,
        elapsed_seconds=0,
        resources={},
        result=None,
        payloads=[],
        recovery_of=entry.get("previous"),
        repair_cause=CAUSE,
        same_cause_repair_attempt=1,
        compatibility=compatibility,
    )
    item = schedule["cases"][row["case_index"]]
    arm = next(a for a in schedule["arms"] if a["id"] == row["arm_id"])
    if item["status"] != "materialized" or arm["status"] != "available":
        return checkpoint(
            receipt,
            expected,
            **common,
            status="unavailable",
            reason=item["status"] if item["status"] != "materialized" else arm["status"],
        )
    if not compatibility["compatible"]:
        return checkpoint(
            receipt,
            expected,
            **common,
            status=compatibility["status"],
            recovery_action="explicit_unavailability_preserving_original_denominator",
            reason="No untrained relation parameters or changed frozen weights are introduced",
        )
    arm = next(a for a in schedule["arms"] if a["id"] == row["arm_id"])
    resources = bound(arm["protocol"])["resources"]
    budget = (
        remaining_budget(previous, payload, row, resources)
        if previous
        else dict(
            wall_seconds=row["seconds"],
            generation_seconds=resources["generation_seconds"],
            cpu_seconds=row["cpu_seconds"],
            prior_wall_seconds=0,
            prior_generation_seconds=0,
            prior_cpu_seconds=0,
        )
    )
    if min(budget[k] for k in ("wall_seconds", "generation_seconds", "cpu_seconds")) <= 0:
        return checkpoint(
            receipt,
            expected,
            **common,
            status="unavailable_remaining_budget",
            remaining_budget=budget,
            recovery_action="budget_exhausted_no_replay",
        )
    validate_remaining_budget(budget, row, resources)
    guard = output / "inflight" / (row["id"] + ".json")
    if guard.exists():
        raise RuntimeError("Schema recovery requires owner and spent-budget reconciliation")
    checkpoint(guard, expected, row=row, remaining_budget=budget, started_epoch=time.time())
    directory = output / "payloads" / row["id"]
    began = time.monotonic()
    outcome = bounded_call(
        fresh.evaluate_row,
        schedule,
        row,
        str(directory),
        remaining_budget=budget,
        timeout=budget["wall_seconds"],
        cpu_seconds=budget["cpu_seconds"],
        memory_mb=row["memory_mb"],
    )
    common.update(
        elapsed_seconds=time.monotonic() - began,
        cleanup_complete=outcome.cleanup_complete,
        resources=dict(outcome.resource_usage),
        result=outcome.value if outcome.status == "complete" else None,
        payloads=[
            binding(p) for p in sorted(directory.rglob("*")) if p.is_file() and p.suffix != ".lock"
        ],
    )
    saved = checkpoint(
        receipt,
        expected,
        **common,
        status=outcome.status,
        detail=outcome.detail,
        remaining_budget=budget,
        recovery_action="compatible_remaining_budget_execution",
    )
    fresh.validate_payloads(saved)
    if not outcome.cleanup_complete or "cleanup incomplete" in outcome.detail:
        raise RuntimeError("Schema recovery cleanup incomplete; retain in-flight guard")
    guard.unlink()
    return saved


def run(plan_path, output, start, stop):
    from exact.repair.study import runtime_manifest

    plan = read(plan_path)
    if any(entry.get("blocked_by_prior") for entry in plan["rows"][start:stop]):
        raise ValueError(
            "This slice still has an original owner needing completion/ledger reconciliation"
        )
    schedule = bound(plan["schedule"])
    admission = bound(plan["preflight"])
    if len(plan["rows"]) != len(schedule["rows"]) or not 0 <= start < stop <= len(plan["rows"]):
        raise ValueError("Schema recovery schedule changed")
    if admission["model_source_sha256"] != sha(
        Path(__file__).resolve().parents[2] / "exact/repair/model.py"
    ) or admission["graph_source_sha256"] != sha(
        Path(__file__).resolve().parents[2] / "exact/repair/graph.py"
    ):
        raise ValueError("Schema admission source changed")
    for arm in schedule["arms"]:
        for key in ("model", "protocol", "completion", "training_report"):
            if key in arm:
                fresh.pilot.check_binding(arm[key])
    by_id = {r["row_id"]: r for r in admission["rows"]}
    identity = canonical_hash(
        (sha(plan_path), sha(__file__), sha(fresh.__file__), runtime_manifest())
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refs = []
        for index in range(start, stop):
            row, entry = schedule["rows"][index], plan["rows"][index]
            if entry["row_id"] != row["id"]:
                raise ValueError("Schema recovery row order changed")
            item = schedule["cases"][row["case_index"]]
            if item.get("observable"):
                bound(item["observable"])
            write_artifact(output / "progress.json", dict(recorded=len(refs), row=row["id"]))
            saved = one(schedule, row, entry, by_id[row["id"]], output, identity)
            fresh.raise_on_software_failure(saved)
            refs.append(
                dict(
                    **binding(output / "rows" / (row["id"] + ".json")),
                    row_id=row["id"],
                    status=saved["status"],
                )
            )
        return checkpoint(
            output / "report.json",
            identity,
            schema="exact-repair/schema-recovery/v1",
            status="complete",
            plan=binding(plan_path),
            schedule=plan["schedule"],
            rows=refs,
            scheduled=stop - start,
            recorded=len(refs),
            outcomes=dict(Counter(r["status"] for r in refs)),
            gates_passed=False,
            study_complete=False,
            prior_costs_reset=False,
            checkpoint_parameters_changed=False,
            followup=schedule.get("followup", "xr21-expanded-evaluation-001"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    args = parser.parse_args()
    run(args.plan, args.output, args.start, args.stop)


if __name__ == "__main__":
    main()
