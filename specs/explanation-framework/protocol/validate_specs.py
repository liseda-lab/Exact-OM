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

    blueprint = documents.get("protocol/development.json", {})
    require(blueprint.get("status") == "design_only_not_current_runtime_config",
            "Blueprint must not claim to be executable runtime configuration")
    require(blueprint.get("sequence") == ["B0", "B1+B2", "B3+B4", "B5", "F1"],
            "Blueprint sequencing disagrees with handoff")
    require(blueprint.get("early_frontend_agent") is False, "Unexpected early frontend dependency")
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
