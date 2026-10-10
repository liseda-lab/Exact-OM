"""Complete input-only DEV successor; execution remains behind measured admission."""

from __future__ import annotations

import argparse
from pathlib import Path

from exact.repair.records import canonical_hash
from exact.repair.semantic_fidelity import consequence_basis_from_probes
from tools.repair.annotation_profile import request_profile, input_limits
from tools.repair.batch import read, source_snapshot
from tools.repair.common_training import development_schedule
from tools.repair.corrective_campaign import source_identity
from tools.repair.grounded_supervision import controlled_evidence, render
from tools.repair.historical_regression import binding
from tools.repair.prepare import case_from_dict
from tools.repair.shared_release import bound, immutable


def fixed_counterpart(case):
    """Native deletion is declared from inputs, never selected on DEV outcomes."""
    identities = []
    for obj in case.problem.objects:
        deletions = [candidate for candidate in obj.candidates if not candidate.axioms]
        if len(deletions) != 1:
            raise ValueError("DEV counterpart requires one input-declared deletion per object")
        identities.append(deletions[0].candidate_id)
    return identities


def case_entry(case, source, byte_cap):
    evidence = controlled_evidence(case, source, allowed_split="development")
    return dict(
        parent=case.structural_parent,
        query_basis_hash=canonical_hash(consequence_basis_from_probes(case.probes)),
        counterpart_candidate_ids=fixed_counterpart(case),
        counterpart_rule=(
            "input-declared native deletion; identical/missing/unverified stays unavailable"
        ),
        task="Assess retained meaning only under this explicitly authored fictional setting. "
        "Do not infer external domain facts or reward a named repair action. Use abstain "
        "when the declared setting cannot distinguish the complete plans; "
        "logical validity is separate.",
        original_observation=[
            render(a for obj in case.problem.objects for a in obj.original_axioms)
        ],
        evidence=evidence["evidence"],
        local_context=[],
        coverage=dict(
            complete=True,
            omissions=[],
            stop_reason="complete_controlled_setting",
            byte_budget=byte_cap,
        ),
        evidence_binding_identity=canonical_hash(evidence),
        independent_semantic_evidence=False,
        real_data_coverage=False,
    )


def prepare(campaign, output, repository, final_epoch=6, repair_seconds=30):
    campaign, output, repository = map(Path, (campaign, output, repository))
    registry = read(campaign / "supervisor/registry.json")
    capacity = bound(registry["shared_capacity_preparation"]["preparation"])
    release = bound(capacity["common_release"])
    dev = bound(release["development"])
    source = source_identity()
    if source["dirty_hash"] != __import__("hashlib").sha256(b"").hexdigest():
        raise ValueError("Commit source before preparing DEV")
    code, _ = source_snapshot(repository, source["revision"], campaign / "sources")
    evidence_source = binding(code / "tools/repair/grounded_supervision.py")
    teacher = bound(registry["qualified_teacher_preparation"]["preparation"])
    template = bound(teacher["annotation_templates"]["development"])
    controls = template["request_profiles"][template["profile"]]
    amendment_ref = registry["campaign_context"]["training_data_correction"]
    amendment = bound(amendment_ref)
    controls.update(
        max_input_tokens=amendment["max_input_tokens"],
        max_input_bytes=amendment["max_input_bytes"],
        input_amendment=amendment_ref,
    )
    request_profile(template, template["profile"])
    template["rate_limit_policy"] = dict(
        authorization=registry["campaign_context"]["transient_provider_policy"],
        base_seconds=30,
        max_seconds=300,
    )
    from tools.repair.annotation_backoff import validate_policy

    validate_policy(template)
    cases = [case_from_dict(bound(row["evaluator"])) for row in dev["rows"]]
    if len(cases) != 32 or any(c.split != "development" for c in cases):
        raise ValueError("Expected all32 DEV inputs and no TEST")
    schedule = development_schedule(
        dev["rows"], final_epoch=final_epoch, repair_seconds=repair_seconds
    )
    template.update(
        cases={c.case_id: case_entry(c, evidence_source, input_limits(controls)[1]) for c in cases},
        parent_splits={c.structural_parent: "development" for c in cases},
        frozen_annotation_slots=schedule["frozen_annotation_slots"],
        post_decode_reserve_seconds=schedule["post_decode_reserve_seconds"],
        controlled_evidence_source=evidence_source,
        packet_encoding="semantic-packet-lossless-text-dictionary/v1",
        execution_gate="disabled_until_measured_common_endpoint_and_all_stage_caps_pass",
        authorized=False,
    )
    immutable(output / "development-annotation.json", template)
    schedule.update(
        request_limits=template["request_limits"],
        request_budget_amendment=template["request_budget_amendment"],
        qualified_teacher=template["qualified_teacher"],
        teacher_profile=template["teacher_profile"],
        annotation_manifest=binding(output / "development-annotation.json"),
        status="saved_complete_input_contract_not_admitted",
        development_use_policy="development_selection/v1",
    )
    immutable(output / "development-schedule.json", schedule)
    immutable(
        output / "prepared.json",
        dict(
            schema="exact-repair/development-contract-preparation/v1",
            status="saved_preparation_not_queued",
            source_commit=source["revision"],
            common_release=capacity["common_release"],
            development_inputs=release["development"],
            schedule=binding(output / "development-schedule.json"),
            annotation_manifest=binding(output / "development-annotation.json"),
            training_readiness=capacity["training_readiness"],
            expected_rows=schedule["expected_rows"],
            unique_semantic_slots=64,
            independent_swapped_calls=14,
            final_epoch_candidate=final_epoch,
            counterfactual_endpoint_frozen=False,
            primary_fit_admitted=False,
            hosted_calls=0,
            native_calls=0,
            test_outcomes_opened=False,
            remaining_action=(
                "Resource admission remains unresolved; validate exact post-decode packets "
                "at runtime, retain every unavailable slot; never transmit this disabled template"
            ),
        ),
    )
    return output / "prepared.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("repository", type=Path)
    args = parser.parse_args()
    print(prepare(args.campaign, args.output, args.repository))


if __name__ == "__main__":
    main()
