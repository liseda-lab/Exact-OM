"""Resolve teacher identity, then audit shared TRAIN packet prerequisites offline.

No outcomes from DEV/TEST, native rechecks, hosted calls or primary optimization.
The audit retains all comparison intentions, including unavailable evidence.
"""

import argparse
import copy
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
import time

from exact.repair.protocol import RepairProtocolV3
from exact.repair.records import canonical_hash
from tools.repair.batch import _stage_remaining, freeze, prepare_dispatch, read
from tools.repair.common_training import load_release
from tools.repair.corrective_campaign import source_identity
from tools.repair.historical_regression import binding
from tools.repair.shared_release import bound, immutable, check_time
from tools.repair.teacher_selection import DATA_PERMISSIONS, qualified_contract


def annotation_template(selected_ref, source, phase, deadline):
    selected = bound(selected_ref)
    profile = selected["selected_profile"]
    keys = (
        "schema",
        "cost_ceiling_usd",
        "request_limits",
        "request_budget_amendment",
        "lineage_id",
        "ledger_directory",
        "profiles",
        "prices_per_million",
        "prompt_version",
        "prompt_hash",
        "rubric_version",
        "criterion_weights",
        "test_profile",
        "selection_model_ids",
        "data_permissions",
    )
    value = {key: copy.deepcopy(source[key]) for key in keys}
    value.update(
        phase=phase,
        data_permissions=DATA_PERMISSIONS[phase],
        authorized=False,
        profile=profile,
        teacher_profile=profile,
        qualified_teacher=selected_ref,
        calibration_gate=selected["gate"],
        request_profiles={profile: source["request_profiles"][profile]},
        deadline_epoch=deadline,
        parent_splits={},
        slots=[],
        seconds=28800 if phase == "train" else 14400,
    )
    qualified_contract(value)
    return value


def comparison_pairs(case, cache):
    """Two fixed distinct complete plan pairs; never choose by benefit magnitude.

    Feasibility is only an annotation prerequisite. Failure after this freeze
    cannot replace a pair with a different, more convenient candidate.
    """
    if cache is None:
        return []
    assignments = {
        tuple(label.assignment)
        for label in cache.labels
        if label.feasible is True and label.benefit is not None
    }

    def identity(assignment):
        return tuple(
            (obj.object_id, obj.candidates[i].candidate_id)
            for obj, i in zip(case.problem.objects, assignment)
        )

    ordered = sorted(assignments, key=lambda a: canonical_hash(identity(a)))
    return [
        dict(assignments=[list(a), list(b)], candidate_ids=[identity(a), identity(b)])
        for a, b in list(combinations(ordered, 2))[:2]
    ]


def evidence_status(case):
    """Inventory what exists, without treating noisy matcher text as definitions."""
    evidence = dict(case.problem.evidence)
    definitions = []
    for key, value in evidence.items():
        if (
            isinstance(value, Mapping)
            and value.get("kind")
            in {"source_definition", "scope_note", "authored_semantic_definition"}
            and all(value.get(k) for k in ("source_id", "release", "text"))
        ):
            definitions.append(key)
    return dict(
        observed_records=len(evidence),
        source_definition_ids=definitions,
        grounded_definition_available=bool(definitions),
        noisy_matcher_records=sum(
            isinstance(v, Mapping) and v.get("matcher") == "label-jaccard-plus-noise"
            for v in evidence.values()
        ),
        declared_consequence_queries=len(case.probes),
        consequence_intent_is_not_domain_definition=True,
    )


def audit(prepared_path, output):
    from collections import Counter

    prepared = read(prepared_path)
    base = bound(prepared["predecessor"])
    selected = bound(prepared["selected_teacher"])
    bound(selected["review"])
    cases, caches, _ = load_release(base["common_release"], base["audit_run"])
    by_id = {c.case_id: c for c in cases if c.split == "train"}
    rows, shards = [], []
    output = Path(output)
    for shard_ref in base["train_annotation_intentions"]:
        intentions = bound(shard_ref)
        shard_rows = []
        for intention in intentions["slots"]:
            check_time()
            case = by_id[intention["case_id"]]
            cache = caches.get(case.case_id)
            pairs = comparison_pairs(case, cache)
            evidence = evidence_status(case)
            index = intention["comparison_index"]
            pair = pairs[index] if index < len(pairs) else None
            status = (
                "unavailable_missing_symbolic_cache"
                if cache is None
                else (
                    "unavailable_distinct_verified_plans"
                    if pair is None
                    else (
                        "unavailable_grounded_definitions"
                        if not evidence["grounded_definition_available"]
                        else "pending_complete_native_packet_verification"
                    )
                )
            )
            row = dict(
                intention,
                status=status,
                pair=pair,
                evidence=evidence,
                case_input_hash=case.problem.content_hash,
                cache_dependencies=dict(cache.hashes) if cache else None,
                primary_weak_label_eligible=False,
                replacement_allowed=False,
            )
            shard_rows.append(row)
            rows.append(row)
        path = output / (Path(shard_ref["path"]).stem + ".json")
        immutable(
            path,
            dict(
                schema="exact-repair/train-packet-prerequisites/v1",
                selected_teacher=prepared["selected_teacher"],
                comparison_intentions=shard_ref,
                rows=shard_rows,
                scheduled=len(shard_rows),
                authorized=False,
                hosted_calls=0,
                new_assignment_attempts=0,
            ),
        )
        shards.append(binding(path))
    originals = [r for r in rows if not r["swapped"]]
    swaps = [r for r in rows if r["swapped"]]
    if len(originals) != 256 or len(swaps) != 52 or len(by_id) != 128:
        raise ValueError("Shared annotation denominator changed")
    result = dict(
        schema="exact-repair/teacher-successor-audit/v1",
        status="complete",
        prepared=binding(Path(prepared_path)),
        selected_teacher=prepared["selected_teacher"],
        shared_release=base["common_release"],
        shards=shards,
        expected_train_cases=128,
        unique_comparisons=256,
        swapped_repeats=52,
        statuses=dict(Counter(r["status"] for r in originals)),
        family_control_statuses=dict(
            Counter("/".join((r["family"], r["control"], r["status"])) for r in originals)
        ),
        expected_development_cases=32,
        expected_dev_semantic_comparisons=64,
        grounded_definition_cases=sum(
            evidence_status(c)["grounded_definition_available"] for c in by_id.values()
        ),
        primary_fitting_admitted=False,
        paid_calls=0,
        optimizer_updates=0,
        test_outcomes_opened=False,
        cohort="generated_only",
        missing_real_coverage=True,
        remaining_action="Prepare an explicit source-bound controlled semantic evidence revision, "
        "retaining the existing parents, inputs, targets, candidates and all unavailable slots. "
        "Do not call noisy matcher vocabulary domain definitions or reuse calibration labels "
        "as corpus coverage. Then bind exact native-verified pairs and packet/token admission "
        "before shared weak-label acquisition; no request is authorized by this audit.",
    )
    immutable(output / "report.json", result)
    return result


def prepare(campaign, selected_ref, output, repository):
    campaign, output, repository = map(lambda p: Path(p).resolve(), (campaign, output, repository))
    registry = read(campaign / "supervisor/registry.json")
    selected = bound(selected_ref)
    source = bound(selected["manifest"])
    base_ref = registry["common_training_preparation"]["preparation"]
    base = bound(base_ref)
    runtime = source_identity()
    if runtime["dirty_hash"] != __import__("hashlib").sha256(b"").hexdigest():
        raise ValueError("Commit tested successor code before freezing")
    immutable(output / "source.json", runtime)
    deadline = read(campaign / "campaign.json")["primary_freeze_epoch"]
    ledger = read(campaign / "ledger.json")
    manifests = {}
    for phase in ("train", "development"):
        path = output / (phase + "-annotation-template.json")
        stage = "acquisition" if phase == "train" else "development"
        phase_deadline = deadline
        if stage in ledger["stages"]:
            phase_deadline = min(
                deadline,
                ledger["stages"][stage]["started_epoch"]
                + ledger["stage_limits"][stage]["elapsed_seconds"],
            )
        immutable(path, annotation_template(selected_ref, source, phase, phase_deadline))
        manifests[phase] = binding(path)
    dev = bound(base["development_schedule"])
    dev.update(
        request_limits=source["request_limits"],
        teacher_profile=selected["selected_profile"],
        qualified_teacher=selected_ref,
        authorized=False,
        teacher_service_observation=selected["review"],
        remaining_gate="Matched DEV repair/service projection and common endpoint remain unfrozen",
    )
    immutable(output / "development-schedule.json", dev)
    protocols = []
    for ref in base["protocols"]:
        value = bound(ref)
        value["identity"].update(
            execution_authorized=False,
            code_hash=runtime["code_hash"],
            dirty_hash=runtime["dirty_hash"],
            run_id=output.name + "-" + Path(ref["path"]).stem,
        )
        value["llm_labels"].update(
            teacher_profile=selected["selected_profile"],
            prompt_version=source["prompt_version"],
            rubric_version=source["rubric_version"],
            model_manifest=selected_ref["path"],
            post_decode_annotation_manifest=manifests["development"]["path"],
            teacher_budget=dict(calls=416, tokens=4160000, cost_usd=3.1, wall_seconds=37440),
            evaluator_budget=dict(calls=288, tokens=2880000, cost_usd=31.9, wall_seconds=25920),
            aggregate_budget=dict(calls=704, tokens=7040000, cost_usd=35, wall_seconds=63360),
        )
        path = output / "protocols" / Path(ref["path"]).name
        immutable(path, RepairProtocolV3.model_validate(value).model_dump(by_alias=True))
        protocols.append(binding(path))
    prepared = dict(
        schema="exact-repair/teacher-successor-preparation/v1",
        status="prepared_not_queued",
        selected_teacher=selected_ref,
        predecessor=base_ref,
        annotation_templates=manifests,
        protocols=protocols,
        development_schedule=binding(output / "development-schedule.json"),
        common_endpoint=base["endpoint"],
        final_endpoint_frozen=False,
        fit_execution_authorized=False,
        hosted_calls=0,
        request_limits=source["request_limits"],
        unique_train_comparisons=256,
        train_swaps=52,
        development_comparisons=64,
        source_commit=runtime["revision"],
    )
    immutable(output / "prepared.json", prepared)
    job_id = "audit-qualified-teacher-shared-packets-001"
    job = dict(
        id=job_id,
        seconds=600,
        slice_seconds=600,
        cleanup_seconds=10,
        budget_stages=["acquisition", "learning"],
        stage="acquisition",
        priority=220,
        gpu_devices=[],
        resources=dict(cpus=2, gpus=0, memory_mb=8192, gres="none"),
        deadline_epoch=deadline,
        deadline_policy="defer",
        commands=[
            [
                "{python}",
                "-m",
                "pytest",
                "-q",
                "tests/repair_teacher_selection_test.py",
                "--basetemp={work}/tests",
                "--junitxml={work}/tests.xml",
                "--tb=short",
            ],
            [
                "{python}",
                "-m",
                "tools.repair.prepare_teacher_successor",
                "audit",
                str(output / "prepared.json"),
                "{work}",
            ],
        ],
    )
    if any(s not in ledger["stages"] for s in job["budget_stages"]):
        raise ValueError("Packet audit cannot start fresh stage clocks")
    balance = _stage_remaining(copy.deepcopy(ledger), job, time.time())
    if job["seconds"] > 0.7 * balance:
        raise ValueError("Packet audit does not fit remaining admission capacity")
    immutable(
        output / "admission.json",
        dict(
            remaining_seconds=balance,
            reserved_seconds=600,
            scheduling_fraction=0.7,
            observed_epoch=time.time(),
            costs_reset=False,
            scientific_annotation_admitted=False,
        ),
    )
    spec = dict(
        repository=str(repository),
        campaign=str(campaign),
        allocation="14451",
        python="/home/pgcotovio/Exact-OM/.venv/bin/python",
        ledger=str(campaign / "ledger.json"),
        capacity=registry["capacity"],
        source_store=str(campaign / "sources"),
        protocol_source=protocols[0]["path"],
        jobs=[job],
        input_files=[
            str(output / "prepared.json"),
            str(output / "admission.json"),
            selected_ref["path"],
            base_ref["path"],
            *[r["path"] for r in manifests.values()],
            *[r["path"] for r in protocols],
        ],
        purpose="Selected teacher quota continuation and complete shared TRAIN packet prerequisites; no paid calls or fits",
    )
    immutable(output / "batch-spec.json", spec)
    batch = freeze(output / "batch-spec.json", campaign / "batches" / output.name)
    descriptor = prepare_dispatch(
        batch,
        job_id,
        campaign / "attempts" / job_id / "001",
        tmux_socket=campaign / "supervisor/tmux.sock",
        depends_on=[
            "lower-cost-teacher-panel-approved-001-" + p
            for p in ("glm", "minimax", "mimo", "gpt4o_mini")
        ],
    )
    result = dict(
        preparation=binding(output / "prepared.json"),
        batch=binding(batch),
        descriptor=descriptor,
        status="prepared_not_queued",
    )
    immutable(output / "dispatch-prepared.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "audit"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--repository", type=Path)
    args = parser.parse_args()
    if args.mode == "audit":
        audit(args.input, args.output)
    else:
        prepare(args.input, binding(args.selected), args.output, args.repository)


if __name__ == "__main__":
    main()
