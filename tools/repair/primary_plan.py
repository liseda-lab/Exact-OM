"""Prepare disabled paired primary plans and audit their complete input contracts."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from exact.repair.protocol import RepairProtocolV3, load_protocol_v3, training_projection_v3
from exact.repair.records import canonical_hash
from tools.repair.batch import read
from tools.repair.common_training import load_release, validate_closed_preparation
from tools.repair.corrective_campaign import source_identity
from tools.repair.historical_regression import binding
from tools.repair.primary_runtime import FIT_GPU, DEV_GPU
from tools.repair.shared_release import authenticate, bound, immutable
from tools.repair.training_readiness import require_ready


def phase_plan(seed, condition):
    if seed not in (13, 37, 73) or condition not in ("symbolic", "symbolic_plus_llm"):
        raise ValueError("Undeclared paired model")
    stem = f"primary-{condition}-s{seed}"
    phases = [f"{stem}-{phase}-001" for phase in ("fit5", "dev5", "fit6", "dev6")]
    previous = []
    if condition == "symbolic_plus_llm":
        previous = [f"primary-symbolic-s{seed}-fit6-001"]
    elif seed == 73:
        previous = [f"primary-{c}-s37-dev6-001" for c in ("symbolic", "symbolic_plus_llm")]
    return [
        dict(
            id=phase,
            phase="fit" if index % 2 == 0 else "development",
            depends_on=previous if index == 0 else [phases[index - 1]],
            execution_authorized=False,
            descriptor_prepared=False,
            release_gate="measured_admission" if seed == 13 else "first_pair_technical_review",
        )
        for index, phase in enumerate(phases)
    ]


def prepare(campaign, development, output):
    campaign, output = Path(campaign), Path(output)
    registry = read(campaign / "supervisor/registry.json")
    capacity = bound(registry["shared_capacity_preparation"]["preparation"])
    weak = bound(capacity["shared_release"])
    dev = bound(binding(Path(development)))
    schedule = bound(dev["schedule"])
    source = source_identity()
    if source["dirty_hash"] != __import__("hashlib").sha256(b"").hexdigest():
        raise ValueError("Commit primary plan implementation before preparation")
    if schedule["final_epoch"] != 6 or schedule["repair_seconds"] != 30:
        raise ValueError("Resource proposal is bound to six epochs and30second DEV repair")
    models = []
    for ref in capacity["protocols"]:
        value = bound(ref)
        seed = value["training"]["seeds"][0]
        condition = value["losses"]["target_basis"]
        stem = f"hgt-pair-{condition}-s{seed}"
        own = [
            r
            for r in schedule["rows"]
            if r["selection_slot"]["seed"] == seed
            and r["selection_slot"]["supervision_condition"] == condition
        ]
        value["identity"].update(
            execution_authorized=False,
            code_hash=source["code_hash"],
            dirty_hash=source["dirty_hash"],
            run_id=output.name + "-" + stem,
        )
        value["training"].update(
            max_epochs=6,
            development_epochs=[5, 6],
            sampled_assignments=0,
            patience_enabled=False,
            patience=6,
            max_full_development_evaluations=2,
            development_case_seconds={r["slot_id"]: r["case_seconds"] for r in own},
            final_development_reserve_seconds=sum(
                r["case_seconds"] for r in own if r["selection_slot"]["epoch"] == 6
            ),
        )
        value["llm_labels"]["post_decode_annotation_manifest"] = dev["annotation_manifest"]["path"]
        pp = output / "protocols" / (stem + ".json")
        immutable(pp, RepairProtocolV3.model_validate(value).model_dump(by_alias=True))
        prepared = dict(
            schema="exact-repair/common-training-preparation/v1",
            release=capacity["common_release"],
            audit_run=capacity["common_release_audit"],
            protocol=binding(pp),
        )
        path = output / "preparations" / (stem + ".json")
        immutable(path, dict(prepared, hash=canonical_hash(prepared)))
        manifest = dict(
            schema="exact-repair/primary-phase-runtime/v1",
            execution_authorized=False,
            seed=seed,
            condition=condition,
            protocol=binding(pp),
            preparation=binding(path),
            shared_weak_labels=weak["labels"],
            rating_contexts=weak["rating_contexts"],
            training_readiness=capacity["training_readiness"],
            phase_gpu_uuids=dict(fit=FIT_GPU, development=DEV_GPU),
            state_directory=str(output / "states" / stem),
            admission_status="blocked_existing_resource_caps",
            proposed_common_optimizer_updates=756,
        )
        mp = output / "manifests" / (stem + ".json")
        immutable(mp, manifest)
        models.append(
            dict(
                seed=seed,
                condition=condition,
                manifest=binding(mp),
                phases=phase_plan(seed, condition),
            )
        )
    from tools.repair.prepare_primary_runtime import evaluation_schedule, validate_schedule

    prior = bound(registry["primary_runtime_preparation"]["preparation"])
    prior_test = bound(prior["evaluation_schedule"])
    test = evaluation_schedule(
        bound(prior_test["generated_declaration"]),
        bound(prior_test["conference_declaration"]),
        bound(models[0]["manifest"])["protocol"],
        {k: schedule[k] for k in ("request_limits", "request_budget_amendment")},
    )
    test.update(
        generated_declaration=prior_test["generated_declaration"],
        conference_declaration=prior_test["conference_declaration"],
        measurements_deadline_epoch=1792191600.0,
        models_freeze_epoch=1791997200.0,
    )
    validate_schedule(test)
    immutable(output / "evaluation-schedule.json", test)
    immutable(
        output / "prepared.json",
        dict(
            schema="exact-repair/disabled-primary-plan/v1",
            status="complete_preparation_not_admitted",
            source_commit=source["revision"],
            models=models,
            development=binding(Path(development)),
            evaluation_schedule=binding(output / "evaluation-schedule.json"),
            common_release=capacity["common_release"],
            common_release_audit=capacity["common_release_audit"],
            shared_release=capacity["shared_release"],
            training_readiness=capacity["training_readiness"],
            endpoint_candidate=dict(epochs=6, optimizer_updates=756, development_epochs=[5, 6]),
            first_pair_review_depends_on=[
                f"primary-{c}-s13-dev6-001" for c in ("symbolic", "symbolic_plus_llm")
            ],
            first_pair_review_kind="technical_validity_only_no_performance_threshold",
            first_pair_review_registered=False,
            review_registration_rule=(
                "Register explicit review on actual terminal run IDs "
                "when seed13 descriptors are admitted"
            ),
            later_seed_launch_descriptors_prepared=False,
            primary_fit_admitted=False,
            test_outcomes_opened=False,
            hosted_calls=0,
            remaining_action=(
                "Resolve measured resource admission, then bind an immutable admission receipt "
                "and enable only seed13; retain seed37/73 plans without launch descriptors"
            ),
        ),
    )
    return output / "prepared.json"


def audit(prepared_path, output):
    p = bound(binding(Path(prepared_path)))
    cases, caches, provenance = load_release(p["common_release"], p["common_release_audit"])
    bound(p["shared_release"])
    dev = bound(p["development"])
    annotation = bound(dev["annotation_manifest"])
    from tools.repair.corrective_semantics import validate_manifest

    candidate = copy.deepcopy(annotation)
    candidate["authorized"] = True  # Validation only; never persisted or transmitted.
    validate_manifest(candidate)
    if annotation["authorized"] is not False or len(annotation["cases"]) != 32:
        raise ValueError("DEV contract must retain all32 cases and remain disabled")
    identities = []
    for model in p["models"]:
        m = bound(model["manifest"])
        if m["execution_authorized"] is not False:
            raise ValueError("Proposed primary manifest unexpectedly enabled")
        protocol = training_projection_v3(load_protocol_v3(authenticate(m["protocol"])))
        validate_closed_preparation(
            cases, provenance, protocol, case_limit=None, retry_labels=False
        )
        readiness = require_ready(m["training_readiness"], m["shared_weak_labels"], cases)
        if (
            protocol["training"]["max_epochs"] != 6
            or protocol["training"]["sampled_assignments"] != 0
        ):
            raise ValueError("Paired completed-update endpoint changed")
        identities.append(
            dict(
                seed=model["seed"],
                condition=model["condition"],
                training_ready=readiness["ready"],
                train_cases=sum(c.split == "train" for c in cases),
                train_caches=sum(c.split == "train" and c.case_id in caches for c in cases),
            )
        )
    from tools.repair.prepare_primary_runtime import validate_schedule

    validate_schedule(bound(p["evaluation_schedule"]))
    result = dict(
        schema="exact-repair/primary-plan-audit/v1",
        status="complete",
        models=identities,
        prepared=binding(Path(prepared_path)),
        primary_fit_admitted=False,
        common_data_identical=True,
        all_launch_descriptors_disabled=True,
        hosted_calls=0,
        native_calls=0,
        test_outcomes_opened=False,
        remaining_action=(
            "User resource decision or additional measured equivalent optimization; "
            "no automatic cap amendment"
        ),
    )
    immutable(Path(output) / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "audit"))
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print((prepare if args.mode == "prepare" else audit)(*args.paths))


if __name__ == "__main__":
    main()
