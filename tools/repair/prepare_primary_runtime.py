"""Input-only primary phase and E1–E6 schedule preparation; no TEST deserialization."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from exact.repair.records import canonical_hash
from tools.repair.batch import read
from tools.repair.corrective_campaign import source_identity
from tools.repair.historical_regression import binding
from tools.repair.primary_runtime import phase_job
from tools.repair.shared_release import bound, immutable, validate_completion

SEEDS = (13, 37, 73)
CONDITIONS = ("symbolic", "symbolic_plus_llm")


def stratified(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("family", "conference"), row.get("control", "unopened")].append(row)
    groups = {k: sorted(v, key=lambda r: canonical_hash(r["case_id"])) for k, v in groups.items()}
    return [
        rows[i]
        for i in range(max(map(len, groups.values())))
        for _, rows in sorted(groups.items())
        if i < len(rows)
    ]


def evaluation_schedule(generated, conference, protocol_binding, amendment):
    if generated["split"] != "test" or generated["outcomes_opened"] or len(generated["rows"]) != 64:
        raise ValueError("Expected closed 64-case TEST declaration")
    if len(conference["rows"]) != 21 or len({r["id"] for r in conference["rows"]}) != 21:
        raise ValueError("Conference pair denominator changed")
    cases = [
        dict(
            case_id=r["case_id"],
            structural_parent=r["structural_parent"],
            family=r["family"],
            control=r["control"],
            cohort="generated",
            split="test",
            input_hash=r["input_hash"],
            observable=r["observable"],
            target_revision=generated["target_revision"],
            exposure="inherited parents; historical fresh/robust/projection not reclassified",
        )
        for r in generated["rows"]
    ]
    # Preserve all 21 declarations but do not inspect referenced ontology, matcher
    # or evaluator files. Only inherited TEST pairs enter confirmatory rows.
    pairs = [
        dict(
            case_id=r["id"],
            structural_parent=r["group_id"],
            cohort="conference",
            split=r["split"],
            ontology_names=r["ontology_names"],
            ontology_bindings=r["ontology_bindings"],
            matcher_archive=r["matcher_archive"],
            matcher_member=r["matcher_member"],
            theory_scope=r["theory_scope"],
            status=(
                "scheduled_unopened" if r["split"] == "test" else "excluded_inherited_" + r["split"]
            ),
            provenance_status=r["license_status"],
            redistribution_license="not_established",
        )
        for r in conference["rows"]
    ]
    heldout = [r for r in pairs if r["split"] == "test"]
    cases += heldout
    if any("ekaw" in r["ontology_names"] and r["split"] != "test" for r in pairs):
        raise ValueError("Whole-ontology ekaw holdout changed")
    subset = {r["case_id"] for r in stratified(cases[:64])[:16]} | {r["case_id"] for r in heldout}
    runs, comparisons = [], []
    index = {}

    def add(case, experiment, arm, seed=None, *, pool=None, **settings):
        row = dict(
            case_id=case["case_id"],
            structural_parent=case["structural_parent"],
            cohort=case["cohort"],
            family=case.get("family", "conference"),
            control=case.get("control", "unopened"),
            experiment=experiment,
            arm=arm,
            seed=seed,
            pool=pool,
            settings=settings,
            case_seconds=45 if case["cohort"] == "generated" else 300,
            gpu_role="owned_2080_or_prequalified_owned_5090_boundary",
            input_identity=canonical_hash(case),
            protocol=protocol_binding,
        )
        row["id"] = canonical_hash(row)
        row["status"] = "scheduled_not_attempted"
        runs.append(row)
        index[case["case_id"], experiment, arm, seed] = row["id"]
        return row["id"]

    def comparison(experiment, case, left, right, seed=None, **fixed):
        row = dict(
            experiment=experiment,
            case_id=case["case_id"],
            cohort=case["cohort"],
            seed=seed,
            left=left,
            right=right,
            fixed=fixed,
        )
        row["id"] = canonical_hash(row)
        comparisons.append(row)

    for case in cases:
        for arm in (
            "native_deletion",
            "observable_score_greedy",
            "observable_local_support",
            "budgeted_complete_plan_query",
            "deletion_after_rich",
        ):
            add(
                case,
                "E1",
                arm,
                generation=(
                    "none" if arm in {"native_deletion", "observable_score_greedy"} else "rich"
                ),
                query_utility=(
                    "complete_plan_nonadditive" if arm == "budgeted_complete_plan_query" else None
                ),
                unavailable_score_rule="retain_row",
                ablation=arm == "deletion_after_rich",
            )
        for seed in SEEDS:
            for condition in CONDITIONS:
                add(
                    case,
                    "E1",
                    condition,
                    seed,
                    generation="rich",
                    execution="staged",
                    risk="on",
                    checkpoint=f"hgt-pair-{condition}-s{seed}",
                    hosted_calls=0,
                )
        pool = add(
            case,
            "E2_pool",
            "uniform_independent",
            generation="rich",
            sampler_seed=20261009,
            model_dependent=False,
            costs="record_once_even_when_shared",
        )
        for arm in ("observable_local_support", "budgeted_complete_plan_query"):
            add(
                case,
                "E2",
                arm,
                pool=pool,
                generation="reuse_frozen_pool",
                query_utility=(
                    "complete_plan_nonadditive" if arm == "budgeted_complete_plan_query" else None
                ),
            )
        for seed in SEEDS:
            for condition in CONDITIONS:
                add(
                    case,
                    "E2",
                    condition,
                    seed,
                    pool=pool,
                    risk="on",
                    generation="reuse_frozen_pool",
                    checkpoint=f"hgt-pair-{condition}-s{seed}",
                    objective_and_cuts="frozen_common",
                )
            for experiment in ("E1", "E2"):
                comparison(
                    "E6_" + experiment,
                    case,
                    index[case["case_id"], experiment, CONDITIONS[0], seed],
                    index[case["case_id"], experiment, CONDITIONS[1], seed],
                    seed,
                    reuse_only=True,
                    no_duplicate_repair_run=True,
                )
        if case["case_id"] not in subset:
            continue
        uniform = add(
            case,
            "E3",
            "uniform_circuit",
            13,
            distribution="canonical-family-mass/v3",
            weights="fixed_uniform",
            decoder="circuit",
            selector="observable_local_support",
        )
        learned = add(
            case,
            "E3",
            "learned_circuit",
            13,
            distribution="canonical-family-mass/v3",
            weights="hgt-pair-symbolic_plus_llm-s13",
            decoder="circuit",
            selector="observable_local_support",
        )
        grammar = add(
            case,
            "E3",
            "uniform_grammar",
            13,
            distribution="canonical-family-mass/v3",
            weights="fixed_uniform",
            decoder="grammar",
            selector="observable_local_support",
        )
        comparison("E3a", case, learned, uniform, 13, language_draw_unique_time_caps="identical")
        comparison(
            "E3b", case, uniform, grammar, 13, proposal_weights_alias_and_constraints="identical"
        )
        off = add(
            case,
            "E4",
            "risk_off",
            13,
            pool=pool,
            risk="off",
            generation="reuse_frozen_pool",
            checkpoint="hgt-pair-symbolic_plus_llm-s13",
            objective_and_cuts="frozen_common",
        )
        comparison(
            "E4",
            case,
            index[case["case_id"], "E2", CONDITIONS[1], 13],
            off,
            13,
            objective_shortlist_window_solver_initial_proofs_cut_rules="identical",
        )
        one = add(
            case,
            "E5",
            "one_stage",
            13,
            generation="rich",
            execution="one_stage",
            risk="on",
            checkpoint="hgt-pair-symbolic_plus_llm-s13",
            hosted_calls=0,
        )
        comparison(
            "E5",
            case,
            index[case["case_id"], "E1", CONDITIONS[1], 13],
            one,
            13,
            final_language_and_total_budget="identical",
        )
    semantic = []
    # 43 Conference and 85 generated slots, each seed's total differs by <=1.
    # Conference has only 42 E1/E2 paired outputs (7 pairs x 3 seeds x 2).
    # The extra fixed control contrast is explicit, not a fake independent repeat.
    for cohort, quotas in (("conference", (15, 14, 14)), ("generated", (28, 29, 28))):
        cohort_cases = stratified([r for r in cases if r["cohort"] == cohort])
        for seed, count in zip(SEEDS, quotas):
            candidates = []
            for pass_index in range(2):
                for case_index, case in enumerate(cohort_cases):
                    experiment = ("E1", "E2")[(case_index + pass_index) % 2]
                    candidates.append(
                        dict(
                            case_id=case["case_id"],
                            cohort=cohort,
                            seed=seed,
                            experiment=experiment,
                            contrast="paired_supervision",
                            left=index[case["case_id"], experiment, CONDITIONS[0], seed],
                            right=index[case["case_id"], experiment, CONDITIONS[1], seed],
                        )
                    )
            for case in cohort_cases:
                candidates.append(
                    dict(
                        case_id=case["case_id"],
                        cohort=cohort,
                        seed=seed,
                        experiment="E1",
                        contrast="combined_vs_observable_control",
                        left=index[case["case_id"], "E1", "observable_local_support", None],
                        right=index[case["case_id"], "E1", CONDITIONS[1], seed],
                    )
                )
            semantic.extend(
                dict(r, id=canonical_hash(r), status="pending_exact_verified_output")
                for r in candidates[:count]
            )
    swap_ids = {
        r["id"]
        for r in sorted(semantic, key=lambda r: canonical_hash(("order-swap", r["id"])))[:26]
    }
    semantic = [dict(r, swapped_audit=r["id"] in swap_ids) for r in semantic]
    seconds = Counter()
    for r in runs:
        seconds[r["cohort"]] += r["case_seconds"]
    return dict(
        schema="exact-repair/primary-evaluation-schedule/v1",
        execution_authorized=False,
        status="prepared_input_only_pending_model_and_runtime_admission",
        cases=cases,
        conference_all_pairs=pairs,
        expected_runs=runs,
        comparisons=comparisons,
        expected_run_count=len(runs),
        generated_case_count=64,
        conference_test_pairs=len(heldout),
        bounded_subset=sorted(subset),
        semantic_slots=semantic,
        unique_semantic_comparisons=128,
        independent_swapped_calls=26,
        scheduled_annotation_attempts=154,
        annotation_request_cap=192,
        annotation_service_seconds=154 * 90,
        annotation_report_row_cap=256,
        annotation_authorized=False,
        independent_judge="google/gemini-2.5-pro",
        input_tokens=8000,
        output_tokens=2000,
        request_seconds=90,
        max_in_flight=4,
        **amendment,
        worker_reservations_seconds=dict(seconds),
        conference_worker_fraction=seconds["conference"] / sum(seconds.values()),
        evaluation_elapsed_seconds=48 * 3600,
        final_annotation_aggregation_reserve_seconds=8 * 3600,
        repair_launch_window_seconds=40 * 3600,
        maximum_repair_workers=2,
        repair_worker_capacity_at_70_percent_seconds=0.7 * 40 * 3600 * 2,
        secondary_gpu_capacity_at_70_percent_seconds=0.7 * 48 * 3600,
        no_hosted_calls_during_repair=True,
        test_outcomes_opened=False,
        missing_rows="retain every expected run and slot; unavailable never replaced",
        quality_claim="exact regret/optimality only on completely checked finite support",
        nonadditive_query_utility_as_unary_weights=False,
        costs="shared pool and identical output costs once by identity; include setup/failed/query/hosted costs",
    )


def validate_schedule(schedule):
    runs = schedule["expected_runs"]
    ids = {r["id"] for r in runs}
    if len(ids) != len(runs) or len(runs) != schedule["expected_run_count"]:
        raise ValueError("Evaluation run denominator or identities changed")
    for row in runs:
        if row["id"] != canonical_hash({k: v for k, v in row.items() if k not in {"id", "status"}}):
            raise ValueError("Evaluation row identity changed")
    slots = schedule["semantic_slots"]
    if len(slots) != 128 or len({r["id"] for r in slots}) != 128:
        raise ValueError("Semantic denominator changed")
    if sum(r["swapped_audit"] for r in slots) != 26:
        raise ValueError("At least20percent independent swaps required")
    for rows in (slots, schedule["comparisons"]):
        if any(r["left"] not in ids or r["right"] not in ids for r in rows):
            raise ValueError("Comparison references missing output")
    counts = Counter(r["seed"] for r in slots if r["cohort"] == "conference")
    if (
        sum(counts.values()) < 43
        or set(counts) != set(SEEDS)
        or max(counts.values()) - min(counts.values()) > 1
    ):
        raise ValueError("Conference semantic reserve or seed balance changed")
    workers = Counter()
    for row in runs:
        workers[row["cohort"]] += row["case_seconds"]
    if (
        workers != schedule["worker_reservations_seconds"]
        or workers["conference"] * 3 < sum(workers.values())
        or sum(workers.values()) > schedule["repair_worker_capacity_at_70_percent_seconds"]
        or sum(workers.values()) > schedule["secondary_gpu_capacity_at_70_percent_seconds"]
    ):
        raise ValueError("Evaluation worker/GPU capacity or Conference reserve exceeded")
    return True


def prepare(campaign, output):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    registry = read(campaign / "supervisor/registry.json")
    common = bound(registry["common_training_preparation"]["preparation"])
    run = next(r for r in registry["runs"] if r["id"] == "audit-common-training-contract-002")
    attempt = Path(run["completion_path"]).parent
    completion = read(attempt / "completion.json")
    receipt = {k: binding(attempt / (k + ".json")) for k in ("completion", "outputs", "step")}
    receipt.update(
        batch=binding(Path(completion["batch"])),
        dispatch_nonce=run["dispatch_nonce"],
        step_id=run["step_id"],
        expected_status="complete",
    )
    _, outputs, _, _ = validate_completion(receipt)
    audit = registry["common_training_preparation"]["report"]
    if outputs["report.json"] != audit["sha256"]:
        raise ValueError("Predecessor report is not nonce-bound")
    bound(audit)
    source = source_identity()
    if source["dirty_hash"] != hashlib.sha256(b"").hexdigest():
        raise ValueError("Commit tested source before preparation")
    immutable(output / "source.json", source)
    dev = bound(common["development_schedule"])
    amendment = {k: dev[k] for k in ("request_limits", "request_budget_amendment")}
    test = binding(campaign / "inputs/test.json")
    conference = binding(campaign / "inputs/conference-publisher-recovery-001/manifest.json")
    schedule = evaluation_schedule(
        bound(test), bound(conference), common["protocols"][0], amendment
    )
    schedule.update(
        generated_declaration=test,
        conference_declaration=conference,
        measurements_deadline_epoch=1792191600.0,
        models_freeze_epoch=1791997200.0,
    )
    validate_schedule(schedule)
    immutable(output / "evaluation-schedule.json", schedule)
    phases = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            rows = [
                r
                for r in dev["rows"]
                if r["selection_slot"]["seed"] == seed
                and r["selection_slot"]["supervision_condition"] == condition
            ]
            passes = {
                epoch: sum(r["case_seconds"] for r in rows if r["selection_slot"]["epoch"] == epoch)
                for epoch in (5, 50)
            }
            phases.append(
                dict(
                    model_id=f"model-{condition}-{seed}",
                    seed=seed,
                    condition=condition,
                    candidate_endpoint=6300,
                    candidate_epochs=50,
                    endpoint_admitted=False,
                    dev_pass_seconds=passes,
                    all_remaining_dev_reserve_seconds=sum(passes.values()),
                    final_dev_reserve_seconds=passes[50],
                    model_gpu_capacity_at_70_percent_seconds=0.7 * 10800,
                    fit_capacity_after_dev_at_70_percent_seconds=max(
                        0, 0.7 * 10800 - sum(passes.values())
                    ),
                    candidate_dev_gpu_projection_passes=sum(passes.values()) <= 0.7 * 10800,
                    phase_sequence=[
                        "fit_to_epoch5",
                        "development_epoch5",
                        "fit_to_common_endpoint",
                        "development_final",
                    ],
                    fit_job=phase_job(f"model-{condition}-{seed}", "fit", 600),
                    development_job=phase_job(
                        f"model-{condition}-{seed}", "development", max(passes.values()) + 5
                    ),
                    oversized_development_job=phase_job(
                        f"model-{condition}-{seed}",
                        "development",
                        max(passes.values()) + 5,
                        oversized=True,
                    ),
                )
            )
    contract = dict(
        schema="exact-repair/primary-runtime-preparation/v1",
        execution_authorized=False,
        phase_implementation="tools.repair.primary_runtime",
        models=phases,
        remaining_reserve_rule="sum all remaining frozen DEV cases before fitting; subtract terminal/begun cases without renewal",
        boundary_rule="terminal nonce-bound step receipt before next owner; shared checkpoint lock; no live GPU transfer",
        oversized_route="resource-limited result; separately owned5090 successor, same case deadline/model/graph and ledger; no automatic replay",
        current_candidate_gpu_admitted=False,
        common_endpoint_rule="measured slower paired condition at70percent remaining model/stage/device capacity; shared complete epoch>=5; no per-arm rescue",
        pending_gates=[
            "teacher incident4d926e051616ed6ca634e9d1",
            "grounded shared weak labels and TRAIN-only scale",
            "current-inventory symbolic and combined loss throughput with concurrent load",
            "revised matched DEV repair allowance from measured service within existing caps",
        ],
        model_elapsed_seconds=21600,
        model_gpu_seconds=10800,
        primary_gpu_seconds=64800,
        development_elapsed_seconds=43200,
        development_worker_seconds=129600,
        secondary_learning_gpu_seconds=129600,
        learning_elapsed_seconds=237600,
        first_admission_and_all_charges_preserved=True,
    )
    immutable(output / "phase-contract.json", contract)
    immutable(output / "ledger-observation.json", read(campaign / "ledger.json"))
    result = dict(
        schema="exact-repair/primary-runtime-evaluation-preparation/v1",
        status="prepared_not_queued",
        source_commit=source["revision"],
        source=binding(output / "source.json"),
        predecessor=registry["common_training_preparation"]["preparation"],
        predecessor_audit=receipt,
        phase_contract=binding(output / "phase-contract.json"),
        evaluation_schedule=binding(output / "evaluation-schedule.json"),
        test_outcomes_opened=False,
        paid_calls=0,
        primary_updates=0,
        execution_authorized=False,
        teacher_incident_id="4d926e051616ed6ca634e9d1",
        steps=[],
    )
    immutable(output / "prepared.json", result)
    return result


def audit(prepared, output):
    prepared = Path(prepared)
    value = read(prepared)
    validate_completion(value["predecessor_audit"])
    schedule = bound(value["evaluation_schedule"])
    validate_schedule(schedule)
    # Recompute only from the two input declarations, never held-out evaluator data.
    common = bound(value["predecessor"])
    dev = bound(common["development_schedule"])
    recomputed = evaluation_schedule(
        bound(schedule["generated_declaration"]),
        bound(schedule["conference_declaration"]),
        common["protocols"][0],
        {k: dev[k] for k in ("request_limits", "request_budget_amendment")},
    )
    if any(schedule[k] != v for k, v in recomputed.items()):
        raise ValueError("Input-only schedule recomputation differs")
    contract = bound(value["phase_contract"])
    if contract["execution_authorized"] or value["execution_authorized"]:
        raise ValueError("Unresolved runtime accidentally admitted")
    result = dict(
        schema="exact-repair/primary-runtime-evaluation-audit/v1",
        status="complete",
        preparation=binding(prepared),
        expected_runs=len(schedule["expected_runs"]),
        expected_comparisons=len(schedule["comparisons"]),
        semantic_slots=128,
        swaps=26,
        conference_pairs=21,
        conference_test_pairs=schedule["conference_test_pairs"],
        conference_worker_fraction=schedule["conference_worker_fraction"],
        primary_execution_admitted=False,
        hosted_calls=0,
        test_outcomes_opened=False,
    )
    immutable(Path(output) / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "audit"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else audit)(args.input, args.output)


if __name__ == "__main__":
    main()
