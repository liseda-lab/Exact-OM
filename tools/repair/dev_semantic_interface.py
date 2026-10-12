"""Explicit, cohort-wide DEV interface correction over saved verified repairs.

This is a new observation revision, not retry-until-success. Historical judgments,
raw receipts, weights, case selection, counterpart plans and shared charges survive.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from exact.repair.records import canonical_hash, canonical_json
from exact.repair.semantic_fidelity import (
    NATIVE_EVIDENCE_PROMPT_VERSION,
    CITATION_PROMPT_VERSION,
    CONTRAST_PROMPT_VERSION,
    COMPLETE_PROMPT_VERSION,
    REFERENCE_PROMPT_VERSION,
    consequence_basis_from_probes,
)
from tools.repair.batch import read, sha
from tools.repair.historical_regression import binding
from tools.repair.shared_release import authenticate, bound, immutable

SCHEMA = "exact-repair/dev-semantic-interface-amendment/v1"
EVIDENCE_REVISION = "native-plan-query-evidence/v1"


def native_evidence(probes, plans, evidence, *, compare_text=False):
    """Expose already-qualified outcomes, without calculating or inventing labels.

    Callers authenticate the saved native-plan context or build it with the full
    verifier. Evidence IDs bind the plan, so swapping A/B never reverses a claim.
    """
    import pyowl_core as owl
    from exact.repair.owl import snapshot_from_axioms

    basis = consequence_basis_from_probes(probes)
    expected = set(basis["queries"])
    expected.update(p.probe_id + "#nonvacuity" for p in probes if p.conditions())
    result = dict(evidence)
    for plan in plans:
        if not plan.eligible or plan.query_basis_hash != canonical_hash(basis):
            raise ValueError("Native evidence requires the exact qualified query basis")
        if set(plan.query_outcomes) != expected:
            raise ValueError("Native evidence query coverage differs from the frozen basis")
        proof = json.loads(plan.content[-1])["consequence_report_id"]
        for probe in probes:
            query = owl.render_document(
                snapshot_from_axioms((probe.axiom,)).materialize().root, format="functional"
            ).decode("utf-8")
            for kind, suffix in (("entailment", ""), ("nonvacuity", "#nonvacuity")):
                if kind == "nonvacuity" and not probe.conditions():
                    continue
                value = plan.query_outcomes[probe.probe_id + suffix]
                identity = (plan.plan_id, plan.theory_hash, proof, probe.probe_id, kind)
                key = "Q" + canonical_hash(identity)[:16]
                meaning = (
                    "true means entailed; false means not entailed (not entailment of negation)."
                    if kind == "entailment"
                    else "true means all listed expressions are satisfiable; false means at least "
                    "one is unsatisfiable. This does not assert entailment of the query."
                )
                item = dict(
                    source_id=proof,
                    release=EVIDENCE_REVISION,
                    kind="native_" + kind,
                    symbolic_value=value,
                    text=canonical_json(
                        dict(
                            plan_id=plan.plan_id,
                            theory_hash=plan.theory_hash,
                            policy_hash=plan.policy_hash,
                            query_basis_hash=plan.query_basis_hash,
                            verification_report_id=plan.report_id,
                            consequence_report_id=proof,
                            query_id=probe.probe_id,
                            query=query,
                            outcome=value,
                            nonvacuity_expressions=probe.conditions() if suffix else (),
                            interpretation=meaning,
                        )
                    ),
                )
                if key in result and result[key] != item:
                    raise ValueError("Native evidence ID collision")
                result[key] = item
    if compare_text:
        a, b = plans
        key = "THEORY_COMPARISON"
        if key in result:
            raise ValueError("Theory comparison evidence ID collision")
        left, right = a.content[0], b.content[0]
        same = left == right
        result[key] = dict(
            source_id=canonical_hash((a.content_hash, b.content_hash)),
            release="verified-complete-theory-text-comparison/v1",
            kind="native_asserted_text_identity",
            symbolic_value="true" if same else "false",
            text=canonical_json(
                dict(
                    first_plan_id=a.plan_id,
                    second_plan_id=b.plan_id,
                    first_plan_theory_hash=a.theory_hash,
                    second_plan_theory_hash=b.theory_hash,
                    complete_theory_texts_identical=same,
                    lines_only_in_first_plan=sorted(
                        set(left.splitlines()) - set(right.splitlines())
                    ),
                    lines_only_in_second_plan=sorted(
                        set(right.splitlines()) - set(left.splitlines())
                    ),
                    interpretation="Exact asserted text comparison, not semantic or logical equivalence. Both complete theories remain present.",
                    omitted_difference_lines=0,
                )
            ),
        )
    return result


def validate_amendment(manifest):
    amendment = bound(manifest["dev_interface_amendment"])
    if (
        amendment.get("schema") != SCHEMA
        or amendment.get("authorized") is not True
        or not amendment.get("user_instruction")
        or amendment.get("costs_reset") is not False
        or amendment.get("selection_rule_unchanged") is not True
        or amendment.get("test_opened") is not False
        or manifest["phase"] != "development"
        or manifest.get("packet_encoding") is not None
        or manifest.get("prompt_version")
        not in {
            NATIVE_EVIDENCE_PROMPT_VERSION,
            CITATION_PROMPT_VERSION,
            CONTRAST_PROMPT_VERSION,
            COMPLETE_PROMPT_VERSION,
            REFERENCE_PROMPT_VERSION,
        }
        or manifest.get("no_verbatim_quotes", False)
        != (
            manifest.get("prompt_version")
            in {
                CITATION_PROMPT_VERSION,
                CONTRAST_PROMPT_VERSION,
                COMPLETE_PROMPT_VERSION,
                REFERENCE_PROMPT_VERSION,
            }
        )
        or manifest.get("theory_text_comparison", False)
        != (
            manifest.get("prompt_version")
            in {CONTRAST_PROMPT_VERSION, COMPLETE_PROMPT_VERSION, REFERENCE_PROMPT_VERSION}
        )
        or manifest.get("complete_criteria", False)
        != (manifest.get("prompt_version") in {COMPLETE_PROMPT_VERSION, REFERENCE_PROMPT_VERSION})
        or manifest.get("intended_commitment_evidence", False)
        != (manifest.get("prompt_version") == REFERENCE_PROMPT_VERSION)
        or manifest.get("fact_references", False)
        != (manifest.get("prompt_version") == REFERENCE_PROMPT_VERSION)
    ):
        raise ValueError("DEV interface correction requires explicit bound authorization")
    previous = bound(amendment["previous_manifest"])
    for key in (
        "profiles",
        "profile",
        "parent_splits",
        "frozen_annotation_slots",
        "rubric_version",
        "criterion_weights",
        "cost_ceiling_usd",
        "ledger_directory",
        "lineage_id",
        "teacher_profile",
        "test_profile",
        "selection_model_ids",
    ):
        if manifest[key] != previous[key]:
            raise ValueError("DEV interface amendment changed frozen design: " + key)
    if manifest.get("evaluator_request_amendment"):
        from tools.repair.evaluator_evidence import request_reallocation

        quota, _, _ = request_reallocation(manifest)
        if quota["old_limits"] != previous["request_limits"] or quota[
            "previous_manifest"
        ] != amendment.get("previous_annotation_manifest"):
            raise ValueError("Evaluator quota amendment is outside the bound interface lineage")
    elif manifest["request_limits"] != previous["request_limits"]:
        raise ValueError("DEV interface amendment changed frozen design: request_limits")
    if set(manifest["cases"]) != set(previous["cases"]):
        raise ValueError("DEV interface amendment changed its case denominator")
    byte_cap = manifest["request_profiles"][manifest["profile"]]["max_input_bytes"]
    for case_id, old in previous["cases"].items():
        expected = {**old, "coverage": {**old["coverage"], "byte_budget": byte_cap}}
        if manifest["cases"][case_id] != expected:
            raise ValueError("DEV interface amendment changed meaning or counterpart evidence")
    slots = amendment["requests"]
    scheduled = {
        key
        for key, row in previous["frozen_annotation_slots"].items()
        if row["selection_slot"]["epoch"] == 6
    }
    if set(slots) != scheduled or not set(amendment["qualification_slots"]) <= scheduled:
        raise ValueError("DEV correction must cover the complete frozen final cohort")
    if manifest.get("comparison_basis") is not None:
        parent = bound(manifest["comparison_basis"])
        if (
            manifest["comparison_basis"] != amendment.get("previous_interface_amendment")
            or parent["previous_manifest"] != amendment["previous_manifest"]
            or set(parent["requests"]) != set(slots)
            or any(
                parent["requests"][key]["source_request"] != slots[key]["source_request"]
                for key in slots
            )
        ):
            raise ValueError(
                "Shared comparison identity requires the exact same frozen repair pair"
            )
    if manifest["rate_limit_policy"]["resume_deadline_epoch"] != amendment["deadline_epoch"]:
        raise ValueError("DEV correction transport window changed")
    return amendment


def rebind_request(source_path, manifest_path, state_directory):
    """Change only the authorized interface binding; authenticate every reused plan."""
    manifest = read(manifest_path)
    amendment = validate_amendment(manifest)
    source = binding(source_path)
    original = read(source_path)
    slot = canonical_hash(original["selection_slot"])
    member = amendment["requests"].get(slot)
    if (
        member is None
        or member["source_request"] != source
        or original["manifest_hash"] != amendment["previous_manifest"]["sha256"]
    ):
        raise ValueError("Decoded request is outside the authorized fixed cohort")
    revised = dict(
        original,
        manifest_hash=sha(manifest_path),
        request_identity=canonical_hash((original["request_identity"], sha(manifest_path))),
        interface_origin=source,
        interface_amendment=manifest["dev_interface_amendment"],
    )
    destination = Path(state_directory) / revised["request_identity"] / "request.json"
    immutable(destination, revised)
    immutable(
        destination.with_name("origin.json"),
        dict(
            source=source,
            amendment=manifest["dev_interface_amendment"],
            historical_annotation_request=member["historical_annotation_request"],
        ),
    )
    for reference in member["verified_plans"]:
        value = bound(reference)
        immutable(destination.parent / "verified-plans" / Path(reference["path"]).name, value)
    return destination


def qualify(manifest_path, state_directory, output):
    """Run a prespecified interface panel; its exact observations serve the cohort."""
    from exact.repair.api import write_artifact
    from exact.repair.records import read_record
    from tools.repair.corrective_semantics import annotate_decoded, prepare_decoded_packet
    from tools.repair.grounded_supervision import packet_admission

    manifest = read(manifest_path)
    amendment = validate_amendment(manifest)
    output = Path(output)
    immutable(output / "manifest-binding.json", binding(manifest_path))
    # Admit the entire fixed cohort before any new paid observation. Identical
    # pairs may need their first native proof; all other compatible proofs reuse.
    admission = []
    for slot, member in sorted(amendment["requests"].items()):
        path = rebind_request(member["source_request"]["path"], manifest_path, state_directory)
        left = (
            min(amendment["deadline_epoch"], float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"]))
            - time.time()
            - 10
        )
        if left <= 92:
            write_artifact(output / "progress.json", dict(status="preflight_deferred", slot=slot))
            return
        packet = prepare_decoded_packet(path, Path(manifest_path), seconds=min(90, left))
        if packet is None:
            raise ValueError("Fixed DEV cohort has unavailable native evidence: " + slot)
        immutable(path.parent / "interface-packet.json", packet.to_dict())
        presentations = (
            [False, True] if manifest["frozen_annotation_slots"][slot].get("swapped") else [False]
        )
        for swapped in presentations:
            check = packet_admission(packet, manifest, swapped)
            admission.append(dict(slot=slot, swapped=swapped, **check))
            if check["status"] != "eligible":
                raise ValueError("Fixed DEV packet exceeds its declared full input profile")
    immutable(
        output / "admission.json",
        dict(
            rows=admission,
            hosted_calls=0,
            complete_plans=True,
            truncated=False,
            manifest=binding(manifest_path),
        ),
    )
    rows = []
    for slot in amendment["qualification_slots"]:
        path = rebind_request(
            amendment["requests"][slot]["source_request"]["path"], manifest_path, state_directory
        )
        left = (
            min(amendment["deadline_epoch"], float(os.environ["EXACT_REPAIR_DEADLINE_EPOCH"]))
            - time.time()
            - 10
        )
        artifact = annotate_decoded(path, Path(manifest_path), seconds=max(0, left))
        receipts = [read(p) for p in (path.parent / "annotation").glob("*/receipt.json")]
        if left <= 92 or any(r["status"] == "rate_limit_deferred" for r in receipts):
            write_artifact(output / "progress.json", dict(status="transport_deferred", slot=slot))
            return
        aggregate = (
            read_record(read(artifact)["comparisons"][0]["comparison"]) if artifact else None
        )
        rows.append(
            dict(
                slot=slot,
                request=binding(path),
                artifact=binding(artifact) if artifact else None,
                eligible=bool(aggregate and aggregate.global_target_eligible),
            )
        )
        write_artifact(output / "progress.json", dict(status="running", rows=rows))
    immutable(
        output / "report.json",
        dict(
            status="qualified" if all(r["eligible"] for r in rows) else "review_required",
            manifest=binding(manifest_path),
            rows=rows,
            test_opened=False,
            selection_rule_unchanged=True,
            costs_reset=False,
            observations_reused_in_full_cohort=True,
        ),
    )


def review(plan_path, report_paths, output):
    """Require the whole six-model, 192-case denominator; never substitute weights."""
    plan = read(plan_path)
    declaration = bound(plan["annotation_manifest"])
    amendment = validate_amendment(declaration)
    historical = bound(amendment["historical_review"])
    if (
        not historical["training_integrity_passed"]
        or not historical["full_dev_lifecycle_qualified"]
    ):
        raise ValueError("Recovered readiness lacks original training/lifecycle qualification")
    expected = {canonical_hash(m["source_report"]): m for m in plan["models"]}
    if len(expected) != 6 or len(report_paths) != 6:
        raise ValueError("The DEV review requires all six frozen models")
    models, seen = [], set()
    for path in report_paths:
        value = read(path)
        key = canonical_hash(value["source"])
        if key not in expected or key in seen:
            raise ValueError("Duplicated or foreign model selection report")
        seen.add(key)
        model = expected[key]
        weights = bound(model["weights"])
        authenticate(weights["artifact"])
        if (
            value["status"] != "complete"
            or value["rows"] != 32
            or value.get("annotation_manifest") != plan["annotation_manifest"]
            or value.get("interface_amendment") != declaration["dev_interface_amendment"]
            or value["selection_rule_unchanged"] is not True
            or value["test_opened"] is not False
            or value["costs_reset"] is not False
            or value["epoch"] != weights["epoch"]
            or ((value["selection_criterion"] is None) != (value["selected_checkpoint"] is None))
            or (
                value["selected_checkpoint"] is not None
                and value["selected_checkpoint"] != weights["artifact"]
            )
        ):
            raise ValueError("DEV selection report changed protocol or saved weights")
        source = bound(model["source_report"])
        refs = {canonical_hash(r): r for r in source["rows"]}
        rows = [read(p) for p in sorted(Path(path).parent.glob("rows/*.json"))]
        if len(rows) != 32 or {canonical_hash(r["source"]) for r in rows} != set(refs):
            raise ValueError("Recovered selection lost a decoded DEV row")
        semantic_ids = {
            slot["selection_slot"]["case_id"]
            for slot in declaration["frozen_annotation_slots"].values()
            if slot["selection_slot"]["epoch"] == weights["epoch"]
            and slot["selection_slot"]["seed"] == model["seed"]
            and slot["selection_slot"]["supervision_condition"] == model["condition"]
        }
        selected = [r["result"]["decoded"] for r in rows if r["result"]["case_id"] in semantic_ids]
        verified = sum(
            r["result"]["decoded"]["logical_status"] == "VERIFIED_FEASIBLE" for r in rows
        )
        available = sum(r.get("weak_benefit") is not None for r in selected)
        if value["selected_checkpoint"] is not None and available != len(semantic_ids):
            raise ValueError("Incomplete semantics cannot select a checkpoint")
        models.append(
            dict(
                seed=model["seed"],
                condition=model["condition"],
                report=binding(path),
                weights=model["weights"],
                semantic_slots=len(semantic_ids),
                semantic_available=available,
                verified_dev_rows=verified,
                selected_checkpoint=value["selected_checkpoint"],
                selection_criterion=value["selection_criterion"],
            )
        )
    count = sum(m["selected_checkpoint"] is not None for m in models)
    immutable(
        Path(output) / "report.json",
        dict(
            schema="exact-repair/dev-interface-readiness/v1",
            status="qualified" if count == 6 else "review_required",
            plan=binding(plan_path),
            historical_review=amendment["historical_review"],
            models=models,
            selected_models_available=count,
            semantic_slots=sum(m["semantic_slots"] for m in models),
            semantic_available=sum(m["semantic_available"] for m in models),
            verified_dev_rows=sum(m["verified_dev_rows"] for m in models),
            full_dev_rows=192,
            training_integrity_passed=True,
            full_dev_lifecycle_qualified=all(m["verified_dev_rows"] == 32 for m in models),
            selection_rule_unchanged=True,
            costs_reset=False,
            test_opened=False,
            historical_observations_preserved=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("state_directory")
    parser.add_argument("output")
    parser.add_argument("--selection-reports", nargs="+")
    args = parser.parse_args()
    if args.selection_reports:
        review(args.manifest, args.selection_reports, args.output)
    else:
        qualify(args.manifest, args.state_directory, args.output)


if __name__ == "__main__":
    main()
