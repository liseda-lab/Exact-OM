"""Prepare the October corrective campaign without opening held-out outcomes.

This is a small preparation adapter around existing protocols, batches and the
supervisor. Runtime admission and accounting remain in their shared owners.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import hashlib
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pyowl_core as owl

from exact.repair.api import write_artifact
from exact.repair.protocol import RepairProtocolV3
from exact.repair.graph_schema import generic_graph_schema
from exact.repair.records import canonical_hash
from tools.repair.batch import read, sha
from tools.repair.expanded_corpus import binding, bound, immutable
from tools.repair.expanded_profile import MECHANISMS_SELECTED, parent_case, parent_fingerprints
from tools.repair.corpus import coherent_control
from tools.repair.prepare import case_from_dict, case_to_dict, save_preparation

FREEZE = datetime.fromisoformat("2026-10-14T18:00:00+01:00").timestamp()
MEASUREMENTS = datetime.fromisoformat("2026-10-17T00:00:00+01:00").timestamp()
GPU_DEVICES = {
    "GPU-19b25b77-7f5b-79a9-0c39-843ea57bd484": {"gres": "gpu:rtx5090:1"},
    "GPU-7b3f1042-3c17-9c54-5653-415de63df835": {"gres": "gpu:rtx2080ti:1"},
}


def source_identity():
    """Bind actual implementation contents as well as the containing Git commit."""
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    paths = sorted(
        [
            *root.glob("exact/repair/*.py"),
            *root.glob("tools/repair/*.py"),
            root / "specs/exact-repair/15-preliminary-corrections.md",
            root / "specs/exact-repair/16-liseda05-experiment-plan.md",
            root / "specs/exact-repair/protocol/liseda05-20261009-plan.json",
        ]
    )
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    diff = subprocess.check_output(
        [
            "git",
            "diff",
            "--binary",
            "HEAD",
            "--",
            "exact/repair",
            "tools/repair",
            "specs/exact-repair",
        ],
        cwd=root,
    )
    return dict(
        revision=revision,
        code_hash=canonical_hash(hashes),
        dirty_hash=hashlib.sha256(diff).hexdigest(),
        source_files=hashes,
        graph_schema_hash=canonical_hash(generic_graph_schema()),
    )


def observable_queries(problem):
    """Freeze deployable asserted commitments without looking at evaluator targets."""
    axioms = {a for obj in problem.objects for a in obj.original_axioms}
    return [
        dict(
            probe_id="observed-" + canonical_hash(axiom),
            axiom_hex=owl.canonical_bytes(axiom).hex(),
            family="observed_commitment",
            desired=True,
            nonvacuity=None,
        )
        for axiom in sorted(axioms, key=owl.canonical_bytes)
    ]


def semantic_queries(case):
    """Add a declared unwanted consequence by generator meaning, never by outcomes."""
    from exact.repair.learning import TeacherProbe

    names = {}
    for axiom in (
        *case.problem.fixed_axioms,
        *(a for o in case.problem.objects for a in o.original_axioms),
    ):
        for entity in owl.signature(axiom):
            if isinstance(entity, owl.Class):
                names[entity.iri.value.rsplit(":", 1)[-1]] = entity
    pair = {
        "papers": ("Paper_s", "Accepted_t"),
        "overlap": ("Author_s", "Reviewer_s"),
        "disjointness": ("Author_s", "Reviewer_s"),
        "conjunct": ("Overlap_s", "Reviewer_s"),
        "participant": ("Participant_s", "Speaker_s"),
        "domain": ("Software_s", "Person_s"),
        "range": ("Software_s", "Person_s"),
        "filler": ("Software_s", "Person_s"),
    }[case.family]
    if case.mirrored:
        pair = tuple(
            n[:-2] + "_t" if n.endswith("_s") else n[:-2] + "_s" if n.endswith("_t") else n
            for n in pair
        )
    if not all(n in names for n in pair):
        raise ValueError(
            "Declared unwanted query endpoints absent: " + case.case_id + " " + repr(pair)
        )
    axiom = owl.SubClassOf(*(names[n] for n in pair))
    query = TeacherProbe("corrective-unwanted/v1", axiom, "unwanted", False, (names[pair[0]],))
    existing = tuple(p for p in case.probes if p.probe_id == query.probe_id)
    if existing:
        if existing != (query,):
            raise ValueError("Existing corrective query has an incompatible target")
        return case
    return dataclasses.replace(
        case,
        probes=(*case.probes, query),
        variation=(
            *case.variation,
            ("semantic_target_revision", "desired-unwanted-nonvacuity/20261009"),
        ),
    )


def _publish_draft(path, value):
    write_artifact(path, value)
    return binding(path)


def inputs(previous, output, *, replace_draft=False):
    previous, output = Path(previous), Path(output)
    publish = _publish_draft if replace_draft else immutable
    release = previous / "work/xr21-expanded-corpus/corpus/releases"
    exposures, all_cases, result = {}, [], {}
    for split in ("train", "development", "test"):
        old = read(release / (split + ".json"))
        expected = {"train": 128, "development": 32, "test": 64}[split]
        if old["scheduled"] != expected or len(old["rows"]) != expected:
            raise ValueError("Inherited frozen split denominator differs")
        rows, cases = [], []
        for row in old["rows"]:
            if row["status"] != "materialized":
                rows.append({**row, "inherited_unavailability": True})
                continue
            # Opening a predeclared input/target is not executing an outcome;
            # held-out reasoning and model selection are forbidden in preparation.
            inherited = bound(row["evaluator"])
            case = case_from_dict(inherited)
            if (
                row["case_hash"] != inherited["hash"]
                or row["case_id"] != case.case_id
                or row["input_hash"] != case.problem.content_hash
            ):
                raise ValueError("Inherited release/case identity mismatch")
            if case.split != split:
                raise ValueError("Existing release split mismatch")
            case = semantic_queries(case)
            destination = output / split / (canonical_hash(case.case_id) + ".json")
            payload = case_to_dict(case)
            item = publish(destination, payload)
            rows.append(
                {
                    **row,
                    "evaluator": item,
                    "case_hash": payload["hash"],
                    "inherited_case_hash": inherited["hash"],
                    "query_revision": "desired-unwanted-nonvacuity/20261009",
                }
            )
            cases.append(case)
            for fingerprint in parent_fingerprints(case):
                previous_split = exposures.setdefault(fingerprint, split)
                if previous_split != split:
                    raise ValueError("Structural ancestry crosses split boundaries")
        result[split] = dict(
            schema="exact-repair/corrective-corpus-release/v1",
            split=split,
            source=binding(release / (split + ".json")),
            scheduled=len(old["rows"]),
            rows=rows,
            outcomes_opened=False,
            inherited_case_ids=True,
            target_revision="desired-unwanted-nonvacuity/20261009",
        )
        publish(output / (split + ".json"), result[split])
        save_preparation(output / (split + "-preparation.json"), cases, result[split])
        all_cases.extend(cases)
    # Retain exposed profile parents for engineering/calibration only.
    inv = read(previous / "work/xr21-expanded-profile/profile/inventory.json")
    calibration = []
    for parent in inv["selected"]:
        if parent["split"] != "profile":
            continue
        case = semantic_queries(parent_case(parent["family"], parent["depth"], 20261009))
        calibration.extend((case, coherent_control(case)))
    save_preparation(
        output / "calibration-preparation.json",
        calibration,
        dict(scope="exposed development calibration", test_outcomes_opened=False),
    )
    return all_cases, calibration


def protocol(campaign, cases, *, condition="symbolic", seed=13):
    campaign = Path(campaign)
    base = read(
        Path(__file__).resolve().parents[2]
        / "specs/exact-repair/protocol/xr21-review2-conformance.json"
    )

    def resolve(value):
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return (
            "not_applicable/corrective-calibration"
            if isinstance(value, str) and value.startswith("UNFROZEN")
            else value
        )

    value = resolve(base)
    source = source_identity()
    value["identity"].update(
        code_hash=source["code_hash"],
        dirty_hash=source["dirty_hash"],
        run_id=f"corrective-hgt-pair-{condition}-s{seed}",
        implementation_revision="exact-repair/preliminary-corrections/v1",
        execution_authorized=True,
    )
    value["model"].update(backbone="hgt", pair_benefit=True, graph_schema=generic_graph_schema())
    value["generation"].update(execution_schedule="staged_verified_repair", elementary_seconds=30.0)
    value["generation"]["stages"] = [value["generation"]["stages"][0]]
    value["generation"]["stages"][0].update(
        candidate_cap=24,
        draws_per_object=16,
        classes_per_side=4,
        endpoints_per_side=4,
        properties_per_side=2,
        max_depth=1,
        max_constructors=1,
    )
    value["circuit"].update(
        cache_directory=str(campaign / "cache/compiler"),
        call_seconds=20,
        aggregate_seconds=60,
        rss_mb=8192,
    )
    value["corpus"].update(
        families=list(MECHANISMS_SELECTED), groups_per_family=dict(train=8, development=2, test=4)
    )
    value["collection"].update(
        cases_per_round=128,
        plan_attempts_per_case=16,
        rounds=2,
        retries=0,
        plan_quotas=dict(utility=4, proposal=4, diversity=4, quartet=4, uniform=0),
        utility_attempts=4,
        proposal_attempts=4,
        diversity_attempts=4,
        quartet_attempts=4,
        generator_fraction=0.25,
        exploration_fraction=0.0,
    )
    value["resources"].update(
        case_wall_seconds=300,
        case_cpu_seconds=600,
        case_rss_mb=20480,
        diagnosis_seconds=20,
        generation_seconds=60,
        verification_seconds=30,
        startup_seconds=10,
        cleanup_grace_seconds=2,
        concurrency=3,
        campaign_wall_seconds=112 * 3600,
        campaign_cpu_seconds=112 * 3600 * 14,
        campaign_gpu_hours=102,
        allocated_gpus=2,
        campaign_cost_usd=35,
        stage_wall_seconds=dict(corpus=4 * 3600, label=30 * 3600, train=66 * 3600),
        stage_cpu_seconds=dict(corpus=4 * 3600 * 14, label=30 * 3600 * 14, train=66 * 3600 * 14),
        stage_rss_mb=dict(corpus=20480, label=20480, train=24576),
    )
    value["training"].update(
        max_epochs=50,
        patience=50,
        patience_enabled=False,
        seeds=[seed],
        sampled_assignments=0,
        development_epochs=[5, 50],
        max_full_development_evaluations=2,
        development_case_ids=[c.case_id for c in cases if c.split == "development"],
        threads=4,
        batch_size=1,
        device="cuda",
        generation_stage=0,
    )
    value["evaluation"].update(
        train_manifest=str(campaign / "inputs/train.json"),
        development_manifest=str(campaign / "inputs/development.json"),
        test_manifest=str(campaign / "inputs/test.json"),
        independent_judge="google/gemini-2.5-pro; independent, withheld from teacher calibration",
    )
    value["llm_labels"].update(
        ledger_directory=str(campaign / "annotations/ledger"), post_decode_annotation_manifest=None
    )
    if condition == "symbolic_plus_llm":
        value["identity"]["label_schema"] = "exact-repair/symbolic-plus-fidelity/v1"
        value["losses"].update(
            target_basis=condition,
            loss_contract="provenance_additive/v1",
            weak_anchor_weight=0.2,
            weak_comparison_weight=0.2,
        )
        value["llm_labels"].update(
            enabled=True,
            execution_authorized=True,
            teacher_profile="UNFROZEN/calibration_selected_teacher",
            evaluator_profile="repair_independent_test",
            annotation_manifest=str(campaign / "annotations/train/labels.json"),
            development_use_policy="development_selection/v1",
            plan_rating_aggregation=dict(
                revision="per_criterion_median/v1", minimum_ratings=1, max_criterion_range=0.25
            ),
        )
        # Prepared template cannot execute before the teacher/gate is resolved.
        value["identity"]["execution_authorized"] = False
    return RepairProtocolV3.model_validate(value).model_dump(by_alias=True)


def prepare(previous, campaign):
    campaign = Path(campaign).resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    if (campaign / "batches").exists():
        raise ValueError(
            "Already deployed: preserve frozen campaign and prepare an explicit successor"
        )
    ledger_path = campaign / "ledger.json"
    if ledger_path.exists() and read(ledger_path).get("attempts"):
        raise ValueError("Recorded attempts prohibit replacing draft campaign artifacts")
    cases, calibration = inputs(previous, campaign / "inputs", replace_draft=True)
    for condition in ("symbolic", "symbolic_plus_llm"):
        for seed in (13, 37, 73):
            write_artifact(
                campaign / "protocols" / f"hgt-pair-{condition}-s{seed}.json",
                protocol(campaign, cases, condition=condition, seed=seed),
            )
    plan = read(
        Path(__file__).resolve().parents[2]
        / "specs/exact-repair/protocol/liseda05-20261009-plan.json"
    )
    immutable(campaign / "planning-contract.json", plan)
    immutable(
        campaign / "authorization.json",
        dict(
            schema="exact-repair/corrective-authorization/v1",
            date="2026-10-09",
            launch_authorized=True,
            user_instruction="Integrate f29f753, implement corrections, prepare and run batches, then supervisor handoff",
            monetary_ceiling_usd=35,
            calibration_ceiling_usd=2,
            calibration_max_requests=32,
            llm_plan_approved=True,
            teacher_shortlist=[
                "deepseek/deepseek-v4.1-flash",
                "qwen/qwen3.8-flash",
                "z-ai/glm-5.3-flash",
            ],
            reference_teacher="anthropic/claude-sonnet-4.6",
            independent_test_judge="google/gemini-2.5-pro",
            calibration_not_equivalence_proof=True,
            preserve_historical_stops=True,
            allocation="14451",
        ),
    )
    limits = dict(
        calibration=dict(elapsed_seconds=4 * 3600, deadline_epoch=FREEZE),
        acquisition=dict(elapsed_seconds=30 * 3600, deadline_epoch=FREEZE),
        learning=dict(elapsed_seconds=66 * 3600, deadline_epoch=FREEZE),
        primary_training=dict(gpu_seconds=18 * 3600, deadline_epoch=FREEZE),
        development=dict(
            elapsed_seconds=12 * 3600, worker_seconds=36 * 3600, deadline_epoch=FREEZE
        ),
        secondary_learning=dict(gpu_seconds=36 * 3600, deadline_epoch=FREEZE),
        unary_ablation=dict(gpu_seconds=3 * 3600, deadline_epoch=FREEZE),
        evaluation=dict(elapsed_seconds=48 * 3600, deadline_epoch=MEASUREMENTS),
        secondary_evaluation=dict(gpu_seconds=48 * 3600, deadline_epoch=MEASUREMENTS),
    )
    for condition in ("symbolic", "symbolic_plus_llm"):
        for seed in (13, 37, 73):
            limits[f"model-{condition}-{seed}"] = dict(
                elapsed_seconds=6 * 3600, gpu_seconds=3 * 3600, deadline_epoch=FREEZE
            )
    ledger = campaign / "ledger.json"
    if not ledger.exists():
        write_artifact(
            ledger,
            dict(
                schema="exact-repair/cumulative-budget/v1",
                limit_worker_seconds=None,
                stage_limits=limits,
                attempts={},
                historical_costs=binding(
                    Path(previous) / "artifacts/expanded-preliminary-closure-001/costs.json"
                ),
            ),
        )
    source = source_identity()
    revision = source["revision"]
    _publish_draft(campaign / "implementation-source.json", source)
    p = binding(campaign / "protocols/hgt-pair-symbolic-s13.json")
    rows = []
    for case in calibration:
        for arm in ("native_deletion", "support_retention_surrogate"):
            identifier = canonical_hash((case.case_id, arm))[:24]
            rows.append(
                dict(
                    id=identifier,
                    case=case_to_dict(case),
                    arm=arm,
                    protocol=p,
                    seconds=300,
                    cpu_seconds=600,
                    memory_mb=20480,
                    source_revision=revision,
                    schedule="staged_verified_repair",
                    theory_scope="complete_generated_theory",
                    observable_queries=observable_queries(case.problem),
                )
            )
    for shard in range(3):
        _publish_draft(
            campaign / "calibration" / f"study-{shard}.json",
            dict(
                schema="exact-repair/corrective-study/v1",
                source_revision=revision,
                rows=rows[shard::3],
                cohort="exposed_development",
                heldout_outcomes_opened=False,
            ),
        )
    _publish_draft(
        campaign / "campaign.json",
        dict(
            schema="exact-repair/corrective-campaign/v1",
            implementation_source=binding(campaign / "implementation-source.json"),
            planning_contract=binding(campaign / "planning-contract.json"),
            authorization=binding(campaign / "authorization.json"),
            case_counts=dict(Counter(c.split for c in cases)),
            primary_seeds=[13, 37, 73],
            conditions=["symbolic", "symbolic_plus_llm"],
            source_revision=revision,
            paired_shards=True,
            paired_initialization=True,
            paired_optimizer_updates=True,
            primary_freeze_epoch=FREEZE,
            measurements_deadline_epoch=MEASUREMENTS,
            abstract_date="2026-10-18",
            paper_date="2026-10-25",
            protected_date="2026-10-17",
            gpu_devices=GPU_DEVICES,
            capacity=dict(cpus=14, gpus=2, memory_mb=101297),
            annotation_ceiling_usd=35,
            annotation_request_cap=704,
            stages=[
                "hardware/native qualification",
                "throughput and semantic calibration",
                "shared acquisition",
                "six paired fits and full DEV",
                "E1-E6 and Conference",
                "aggregation and evidence export",
            ],
            model_selection_from_test=False,
            old_campaigns_restarted=False,
        ),
    )
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    print(prepare(args.previous, args.campaign))


if __name__ == "__main__":
    main()
