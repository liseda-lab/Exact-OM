#!/usr/bin/env python3
"""Check design artifacts only; this does not establish runtime readiness.

Requires jsonschema (validated with 4.26.0). From a repository checkout:
    python specs/explanation-framework/protocol/validate_specs.py
For a staged pack, use --repo-root to resolve links into the existing repository.
"""

import argparse
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

from jsonschema import Draft7Validator
from collections import Counter



def validate_study_documents(documents, require, errors, counts):
    """Validate design examples, not a running study service or real results."""
    try:
        schema = documents["protocol/study.schema.json"]
        validator = Draft7Validator(schema)
        study = documents["protocol/study-design.json"]
        require(study["primary_metric"] == "answer_present_MRR", "MRR must exclude negative cases")
        require(study["unsubmitted_is_missing"], "Unsubmitted study answers must remain missing")
        require(not study["invitation_consumed_on_first_use"], "Invitation must remain resumable")
        require(not study["tab_hidden_automatically_pauses"], "Hidden tab may be external-tool work")
        require(study.get("protege_preparation_required") is False,
                "Corrective protocol must not require Protege installation")
        external = study.get("external_inspection_policy", {})
        require(external.get("protege_recommended_only") is True
                and external.get("external_use_required") is False
                and external.get("multiple_methods_allowed") is True
                and external.get("method_changes_between_cases_allowed") is True
                and external.get("method_commitment_required") is False,
                "External inspection must remain optional, combinable and changeable")
        require(study.get("consultation_asked_each_case") is True
                and external.get("reporting_unit") == "source_plus_candidate_set_case"
                and external.get("candidate_level_reports_required") is False
                and external.get("same_frozen_information_scope") is True,
                "Consultation granularity or information scope disagrees with amendment 14")
        tutorial = study.get("tutorial_policy", {})
        require(all(tutorial.get(field) is True for field in (
                    "interactive_shared_workspace_required", "synthetic_disjoint_resources_required",
                    "identical_preassignment_exposure", "eventual_core_pass_required",
                    "feedback_and_unlimited_retries", "durable_progress_required"))
                and tutorial.get("comprehension_items") == 5
                and tutorial.get("scored_time_includes_preparation") is False,
                "Interactive training, assessment or timing policy disagrees with amendment 14")
        require(study["participant_identity_fields"] == [], "Do not request participant identity")
        cases = {v["case_id"]: v for name, v in documents.items()
                 if name.startswith("fixtures/study/") and v.get("artifact_type") == "study_case"}
        forbidden = {"ground_truth", "case_kind", "acceptable_candidate_ids", "gold_target_id",
                     "invitation_token", "session_secret", "email", "participant_name"}

        def keys(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield key
                    yield from keys(item)
            elif isinstance(value, list):
                for item in value:
                    yield from keys(item)

        for name, value in documents.items():
            if not name.startswith("fixtures/study/"):
                continue
            require(not forbidden.intersection(keys(value)), f"{name}: secret/private answer field")
            kind = value["artifact_type"]
            if kind == "study_case":
                candidates = value["candidates"]
                require(len({c["candidate_id"] for c in candidates}) == 5,
                        f"{name}: candidate IDs must be unique")
                require([c["display_position"] for c in candidates] == [1, 2, 3, 4, 5],
                        f"{name}: initial positions must be contiguous and ordered")
            if kind == "ranking_response":
                candidate_ids = {c["candidate_id"] for c in cases[value["case_id"]]["candidates"]}
                require(set(value["ranked_candidate_ids"]) <= candidate_ids,
                        f"{name}: ranking contains a candidate outside its presentation")
                require(value["presentation_id"] == cases[value["case_id"]]["presentation_id"],
                        f"{name}: mismatched presentation")
        for name, profile in documents["protocol/study-schedules.synthetic.json"]["profiles"].items():
            total = int(name)
            inventory = {c["case_id"]: c for c in profile["case_inventory"]}
            exposures = {cid: Counter() for cid in inventory}
            require(len(inventory) == total, f"Schedule {name}: invalid inventory size")
            require(sum(c["kind"] == "answer_absent" for c in inventory.values()) == 4,
                    f"Schedule {name}: expected four negatives")
            orders = Counter()
            for schedule in profile["schedules"]:
                counts["study_schedules"] += 1
                orders[tuple(schedule["block_order"])] += 1
                assigned = []
                for condition, ids in schedule["case_allocation"].items():
                    require(len(ids) == total // 2, f"{name}: unequal condition sizes")
                    require(len(set(ids)) == len(ids), f"{name}: repeated case in condition")
                    require(sum(inventory[c]["kind"] == "answer_absent" for c in ids) == 2,
                            f"{name}: condition must include two negatives")
                    ranks = Counter(inventory[c]["system_accepted_rank"] for c in ids
                                    if inventory[c]["kind"] == "answer_present")
                    require(set(ranks) == {1, 2, 3, 4, 5}, f"{name}: positive rank stratum missing")
                    require(max(ranks.values()) - min(ranks.values()) <= 1,
                            f"{name}: unnecessary positive rank imbalance")
                    if total == 24:
                        require(all(n == 2 for n in ranks.values()), "24-case exact rank balance failed")
                    for cid in ids:
                        exposures[cid][condition] += 1
                    assigned += ids
                require(len(set(assigned)) == total and set(assigned) == set(inventory),
                        f"{name}: repeated/omitted case across conditions")
            require(set(orders.values()) == {2} and len(orders) == 2,
                    f"{name}: block order not counterbalanced")
            require(all(v == Counter(explanation=2, ontology_baseline=2) for v in exposures.values()),
                    f"{name}: cases not balanced across conditions")
        for item in documents["protocol/study-scoring-oracles.synthetic.json"]["cases"]:
            counts["scoring_oracles"] += 1
            response = item["response"]
            validator.validate(response)
            if response["workflow_state"] != "submitted":
                actual = {"missing": True, "rr": None}
            elif item["case_kind"] == "answer_present":
                accepted = set(item["acceptable_candidate_ids"])
                require(bool(accepted), item["name"] + ": positive key has no acceptable candidate")
                rank = next((n for n, cid in enumerate(response["ranked_candidate_ids"], 1)
                             if cid in accepted), None)
                actual = {"rr": 1.0 / rank if rank else 0.0}
            else:
                require(not item["acceptable_candidate_ids"], item["name"] + ": negative key has answer")
                answer = response["response_type"]
                actual = {"rr": None, "correct_none": int(answer == "none_of_these"),
                          "false_endorsement": int(answer == "ranked_candidates"),
                          "uncertain": int(answer == "insufficient_evidence")}
            require(actual == item["expected"], item["name"] + ": scoring example disagrees with protocol")
        # Reject the ambiguous empty/none payloads that caused the original UI concern.
        base = next(v for name, v in documents.items()
                    if name.startswith("fixtures/study/") and v.get("artifact_type") == "ranking_response"
                    and v.get("response_type") == "ranked_candidates")
        bad_none = dict(base, response_type="none_of_these")
        bad_empty = dict(base, ranked_candidate_ids=[])
        bad_duplicate = dict(base, ranked_candidate_ids=["candidate-1", "candidate-1"])
        require(not validator.is_valid(bad_none), "None must reject a simultaneous nonempty ranking")
        require(not validator.is_valid(bad_empty), "Ranked submission cannot be empty")
        require(not validator.is_valid(bad_duplicate), "Ranked submission cannot repeat candidates")
    except Exception as error:
        errors.append(f"Study specification validation: {error}")


def validate_integration_followup(documents, require):
    """Check the agreed follow-up design, not implementation or release readiness."""
    followup = documents.get("protocol/integration-followup.json", {})
    sequence = ["backend_contract_implementation_and_handoff", "frontend_integration",
                "joint_acceptance"]
    require(followup.get("status") == "design_only_not_current_runtime_config"
            and followup.get("runtime_extension") == "study-integration/1",
            "Integration follow-up must identify a design-only versioned extension")
    require(followup.get("sequence") == sequence,
            "Follow-up requires backend handoff before frontend integration and joint acceptance")
    blueprint = documents.get("protocol/development.json", {})
    require(blueprint.get("corrective_iteration", {}).get("sequence_scope")
            == "initial_2026_10_02_correction_only"
            and blueprint.get("integration_followup", {}).get("sequence") == sequence
            and blueprint.get("integration_followup", {}).get("waives_backend_or_launch_gates") is False,
            "Development blueprint must distinguish historical and current correction sequences")
    parallel = followup.get("parallel", {})
    require(all(parallel.get(field) is True for field in (
                "isolated_checkouts_required", "shared_file_owner_required", "merge_backend_first"))
            and parallel.get("waives_handoff_or_joint_acceptance") is False,
            "Parallel work must preserve ownership, handoff and integration gates")
    acceptance = {f"J{i:02}" for i in range(1, 13)}
    issues = followup.get("issues", {})
    require(set(issues) == {"R01", "R02", "R03", "R04"}
            and set(followup.get("joint_acceptance_ids", [])) == acceptance
            and all(v.get("acceptance") and set(v["acceptance"]) <= acceptance
                    for v in issues.values()),
            "All four reopened findings require valid joint acceptance references")
    workspace = followup.get("workspace", {})
    require(workspace.get("v2_explanation_descriptor_required") is True
            and workspace.get("v2_prepared_excerpt_fallback_allowed") is False
            and "baseline_descriptor" in workspace and workspace["baseline_descriptor"] is None
            and workspace.get("scope_is_authorization") is False
            and workspace.get("legacy_v1_adapter_preserved") is True,
            "Workspace discovery cannot use a silent v2 fallback or weaken baseline isolation")
    readiness = followup.get("readiness", {})
    require(readiness.get("candidate_initial_context_count") == 5
            and all(readiness.get(field) is True for field in (
                "source_context_required", "admitted_initial_profiles_and_comparisons_required",
                "usable_render_required"))
            and all(readiness.get(field) is False for field in (
                "lazy_all_ontology_prefetch_required", "request_failure_is_absence",
                "premature_submission_allowed", "hidden_tab_automatically_pauses")),
            "Readiness must cover bounded case content without inventing absence or timing")
    position = followup.get("tutorial_position", {})
    require(position.get("views") == ["lesson", "assessment"]
            and position.get("omitted_patch") == "preserve"
            and position.get("explicit_null_patch") == "reject_422"
            and position.get("position_only_mutation_allowed") is True
            and position.get("save_destination_not_previous_view") is True
            and position.get("frozen_content_rewritten") is False,
            "Tutorial position must distinguish assessment, preserve omissions and save the destination")
    exports = followup.get("exports", {})
    require(exports.get("corrected_analysis_schema") == "exact-study-analysis/3"
            and exports.get("corrected_csv_schema") == "exact-study-csv/3"
            and all(exports.get(field) is True for field in (
                "v2_admin_explicit_selection_required", "saved_export_bytes_preserved",
                "source_protocol_versions_preserved")),
            "Corrected export versions must preserve frozen source and archived evidence")
    require(exports.get("observations_unit") == "page_seconds"
            and exports.get("cross_page_monotonic_offsets_comparable") is False
            and exports.get("coverage_status") == "not_established"
            and all(field in exports and exports[field] is None for field in (
                "unique_elapsed_coverage_seconds", "unobserved_elapsed_seconds"))
            and exports.get("active_duration_known") is False,
            "Raw page observations must not claim unique elapsed coverage or active duration")
    policy = followup.get("preserved_protocol", {})
    require(policy.get("protege_required") is False
            and policy.get("external_use_required") is False
            and policy.get("multiple_methods_allowed") is True
            and policy.get("method_changes_between_cases_allowed") is True
            and policy.get("reporting_unit") == "source_plus_candidate_set_case"
            and policy.get("assessment_items") == 5
            and policy.get("server_owned_completion") is True,
            "Follow-up cannot change external-method freedom, reporting unit or training gates")
    evidence = followup.get("evidence", {})
    require(evidence.get("real_service_browser_required") is True
            and evidence.get("fresh_full_tutorial_journey_required") is True
            and evidence.get("mock_success_is_joint_acceptance") is False
            and evidence.get("waives_backend_or_launch_gates") is False,
            "Follow-up requires real joint evidence without waiving existing launch gates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    staged_repo = root.parent.parent
    errors = []
    counts = {"json_files": 0, "schema_fixtures": 0, "local_links": 0,
              "study_schedules": 0, "scoring_oracles": 0}

    def require(condition, message):
        if not condition:
            errors.append(message)

    documents = {}
    for path in sorted(root.rglob("*.json")):
        try:
            documents[path.relative_to(root).as_posix()] = json.loads(path.read_text())
            counts["json_files"] += 1
        except (ValueError, OSError) as error:
            errors.append(f"{path.name}: {error}")
    schema = documents.get("protocol/contract.schema.json")
    if not schema:
        errors.append("Missing/invalid target schema")
    else:
        try:
            Draft7Validator.check_schema(schema)
            validator = Draft7Validator(schema)
            study_schema = documents["protocol/study.schema.json"]
            Draft7Validator.check_schema(study_schema)
            study_validator = Draft7Validator(study_schema)
            require(study_schema["definitions"]["EntityRef"] == schema["definitions"]["EntityRef"],
                    "Study and explanation entity identities disagree")
            for name, value in documents.items():
                if not name.startswith("fixtures/"):
                    continue
                counts["schema_fixtures"] += 1
                fixture_validator = study_validator if name.startswith("fixtures/study/") else validator
                for error in fixture_validator.iter_errors(value):
                    errors.append(f"{name}: {error.json_path}: {error.message}")
                provenance = value.get("fixture_provenance", "")
                require(bool(provenance), f"{name}: missing fixture provenance")
                if value.get("artifact_type") == "entity_context":
                    scope = value["context_scope"]
                    for category in ("definitions", "synonyms", "parents"):
                        page = value[category]
                        require(page["returned_count"] == len(page["items"]),
                                f"{name}/{category}: returned_count mismatch")
                        total = page["total_count"]
                        require(total is None or total >= page["returned_count"],
                                f"{name}/{category}: invalid total_count")
                        require(page["scope"] == scope,
                                f"{name}/{category}: inconsistent scope/policy")
                        if page["status"] == "absent_in_scope":
                            require(not page["items"] and total == 0 and bool(page["reason"]),
                                    f"{name}/{category}: unjustified absence")
                        ids = [item["fact_id"] for item in page["items"]]
                        require(len(ids) == len(set(ids)), f"{name}/{category}: duplicate IDs")
                        require(all(item["subject"] == value["entity"] for item in page["items"]),
                                f"{name}/{category}: incorrect subject")
                if ".synthetic." in name:
                    require("synthetic" in provenance.lower(), f"{name}: synthetic label missing")
                if value.get("artifact_type") == "generated_explanation":
                    manifest = value["manifest"]
                    if manifest["status"] == "failed":
                        require(not value["claims"] and manifest["response_hash"] is None
                                and value["grounding_status"] != "validated",
                                f"{name}: failure claims successful generated output")
                if value.get("artifact_type") == "pair_decision_trace":
                    for event in value["events"]:
                        if event["status"] == "not_recorded":
                            require(event["outcome"] == "unknown" and not event["scores"],
                                    f"{name}: unrecorded event invents an outcome/score")
        except Exception as error:
            errors.append(f"Schema/fixture validation: {error}")

    validate_study_documents(documents, require, errors, counts)
    validate_integration_followup(documents, require)

    blueprint = documents.get("protocol/development.json", {})
    require(blueprint.get("status") == "design_only_not_current_runtime_config",
            "Blueprint must not claim to be executable runtime configuration")
    require(blueprint.get("sequence") == ["B0", "B1+B2", "B3+B4", "B5", "F1"],
            "Blueprint sequencing disagrees with handoff")
    require(blueprint.get("early_frontend_agent") is False, "Unexpected early frontend dependency")
    require(blueprint.get("sequence_scope") == "original_bootstrap_only"
            and blueprint.get("corrective_iteration", {}).get("sequence") == [
                "frontend_workflow_and_contract_inventory", "joint_contract_freeze",
                "backend_corrections_and_frontend_integration", "joint_acceptance"]
            and blueprint.get("corrective_iteration", {}).get("waives_backend_or_launch_gates") is False,
            "Corrective sequencing must be explicit without waiving acceptance gates")
    require(blueprint.get("study", {}).get("named_external_tool_required") is False
            and blueprint.get("study", {}).get("external_setup_required") == [
                "source_ontology_available", "target_ontology_available"],
            "Development blueprint must not reintroduce mandatory external software")
    generation = blueprint.get("generation", {})
    require(generation.get("provider") == "openrouter" and generation.get("llm_calls_on_read") is False,
            "Generation provider/read-path policy changed")
    require(generation.get("profile_bindings_required_before_dispatch") is True,
            "Generation requires actual model bindings before dispatch")
    require(bool(blueprint.get("unresolved_bindings")), "Design-time missing bindings must stay explicit")
    require(counts["schema_fixtures"] >= 4, "Missing target-contract seed fixtures")

    for path in sorted(root.rglob("*.md")):
        # Ignore fenced examples; this pack uses inline Markdown links without nested URL parentheses.
        body = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
        for target in re.findall(r"\]\(([^)\n]+)\)", body):
            target = target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            counts["local_links"] += 1
            destination = (path.parent / unquote(parsed.path)).resolve()
            exists = destination.exists()
            if not exists and args.repo_root:
                try:
                    relative = destination.relative_to(staged_repo)
                    exists = (args.repo_root / relative).exists()
                except ValueError:
                    pass
            require(exists, f"{path.relative_to(root)}: broken local link {target}")

    result = {"scope": "specification artifacts only; not backend acceptance",
              "status": "failed" if errors else "passed", **counts, "errors": errors}
    print(json.dumps(result, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
