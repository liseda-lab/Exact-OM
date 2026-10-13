"""Authenticate the prospective six-model endpoint rule without inventing DEV scores.

This adapter reads model/protocol metadata and checkpoint bytes only. It never
opens TEST inputs or uses a judgment to select a model.
"""

from tools.repair.shared_release import authenticate, bound

SCHEMA = "exact-repair/fixed-endpoint-test-freeze/v1"
RULE = "fixed_common_training_endpoint"
MEMBERS = {(c, s) for c in ("symbolic", "symbolic_plus_llm") for s in (13, 37, 73)}


def models(reference):
    freeze = bound(reference)
    plan = bound(freeze["plan"])
    authorization = bound(plan["authorization"])
    report = bound(freeze["report"])
    if (
        freeze.get("schema") != "exact-repair/noise-aware-dev/v1"
        or freeze.get("selection_rule") != RULE
        or freeze.get("outcome_selection") is not False
        or freeze.get("test_opened") is not False
        or plan.get("rules") != authorization.get("rules")
        or plan["rules"].get("checkpoint_selection") != RULE
        or plan["rules"].get("epoch") != 6
        or plan["rules"].get("optimizer_updates") != 756
        or authorization.get("training_weights_unchanged") is not True
        or report.get("models") != freeze.get("models")
        or report.get("dev_outcomes_selected_checkpoints") is not False
        or report.get("test_opened") is not False
    ):
        raise ValueError("Fixed endpoint authorization or closed-TEST provenance changed")
    rows = freeze["models"]
    if len(rows) != 6 or {(m["condition"], m["seed"]) for m in rows} != MEMBERS:
        raise ValueError("The six paired endpoints must all remain present")
    result = {}
    for row in rows:
        weight = bound(row["weights"])
        if (
            (weight["condition"], weight["seed"]) != (row["condition"], row["seed"])
            or weight["epoch"] != 6
            or weight["optimizer_updates"] != 756
            or weight.get("weights_unchanged") is not True
            or row.get("selection_rule") != RULE
            or row.get("selection_criterion") is not None
            or row.get("selected_checkpoint") != weight["artifact"]
        ):
            raise ValueError("A fixed training endpoint or its selection rule changed")
        authenticate(weight["artifact"])
        result[f"hgt-pair-{row['condition']}-s{row['seed']}"] = weight["artifact"]
    return result


def validate(freeze):
    """Validate the new schema explicitly; legacy receipts retain their old rules."""
    if freeze.get("schema") != SCHEMA:
        return
    if (
        freeze.get("selection_rule") != RULE
        or freeze.get("outcome_selection") is not False
        or freeze.get("test_outcomes_opened") is not False
        or freeze.get("final_freeze") is not True
        or freeze.get("models") != models(freeze["noise_aware_dev_freeze"])
    ):
        raise ValueError("TEST model bindings differ from the authorized fixed endpoints")


def repair_deadline(manifest, first_admission):
    """An authorized earlier opening preserves the absolute measurement deadline."""
    if "opening_amendment" not in manifest:
        return first_admission + 40 * 3600
    amendment = bound(manifest["opening_amendment"])
    authorization = bound(amendment["authorization"])
    schedule = bound(manifest["schedule"])
    original = bound(amendment["original_schedule"])
    if (
        amendment.get("schema") != "exact-repair/early-test-opening/v1"
        or authorization.get("decision") != "Start TEST as soon as ready"
        or manifest["not_before_epoch"] != amendment["new_opening_epoch"]
        or schedule["models_freeze_epoch"] != amendment["new_opening_epoch"]
        or manifest["deadline_epoch"] != original["measurements_deadline_epoch"]
        or schedule["measurements_deadline_epoch"] != original["measurements_deadline_epoch"]
        or schedule["final_annotation_aggregation_reserve_seconds"]
        != original["final_annotation_aggregation_reserve_seconds"]
    ):
        raise ValueError("Early TEST opening changed its authorization, deadline or final reserve")
    return manifest["deadline_epoch"] - schedule["final_annotation_aggregation_reserve_seconds"]
