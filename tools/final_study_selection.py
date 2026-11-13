"""Validate a private owner-selected pool without changing matcher candidates.

Selection approval is separate from evidence preparation and publication admission.
No answer keys or review decisions belong in participant/provider payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def case_signature(case):
    frozen = {
        key: case[key]
        for key in ("case_id", "query_id", "source", "reference_rank", "reference_target")
    }
    frozen["candidates"] = [
        {key: candidate[key] for key in ("target", "score", "original_rank")}
        for candidate in case["candidates"]
    ]
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_frozen_cases(cases, inventory):
    originals = {c["case_id"]: c for c in inventory["cases"]}
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Repeated review case identity")
    for case in cases:
        original = originals.get(case["case_id"])
        if original is None or case_signature(case) != case_signature(original):
            raise ValueError("A reviewed case differs from the frozen matcher/reference inventory")
        if len(case["candidates"]) != 5:
            raise ValueError("Every case must retain its original five candidates")


def resolve_selection(cases, plan, choices=None, case_reviews=None):
    """Return provisional choices plus all blocking errors; never silently pad a pool."""
    by_id = {c["case_id"]: c for c in cases}
    choices = choices or {}
    case_reviews = case_reviews or {}
    ids = list(plan["selected_case_ids"])
    errors, pending = [], []
    slots = plan["decision_slots"]
    slot_ids = {s["case_id"] for s in slots}
    if len(slot_ids) != len(slots) or not slot_ids.issubset(set(ids)):
        errors.append("Decision slots must be unique members of the proposed pool.")
    if set(choices) - slot_ids:
        errors.append("Unknown decision slot.")
    fixed = set(plan["fixed_case_ids"])
    if fixed & slot_ids:
        errors.append("Previously approved cases cannot be replacement slots.")
    substitutions = {}
    for slot in slots:
        selected = choices.get(slot["case_id"])
        if not selected:
            pending.append(slot["case_id"])
            selected = slot["case_id"]
        allowed = {c["case_id"] for c in slot["choices"]}
        if selected not in allowed:
            errors.append("Choice is not an offered alternative: " + slot["case_id"])
        substitutions[slot["case_id"]] = selected
    ids = [substitutions.get(cid, cid) for cid in ids]
    if len(ids) != 40 or len(set(ids)) != 40:
        errors.append("Choose 40 different cases; an alternative cannot fill two slots.")
    if not fixed.issubset(set(ids)):
        errors.append("The twelve previously approved cases must be retained.")
    if any(cid not in by_id for cid in ids):
        errors.append("Unknown selected case.")
    selected_cases = [by_id[cid] for cid in ids if cid in by_id]
    if len({c["source"] for c in selected_cases}) != len(selected_cases):
        errors.append("The pool cannot repeat a source concept.")
    counts = Counter(c["reference_rank"] for c in selected_cases)
    if any(counts.get(rank, 0) < 2 for rank in range(1, 6)):
        errors.append("Keep at least two different sources at every reference rank.")
    for cid in ids:
        if case_reviews.get(cid) == "exclude":
            errors.append("A selected case is marked Exclude: " + cid)
        if case_reviews.get(cid) == "consider" and cid not in pending:
            pending.append(cid)
    return {
        "case_ids": ids,
        "counts_by_rank": {rank: counts.get(rank, 0) for rank in range(1, 6)},
        "pending_case_ids": pending,
        "errors": errors,
        "ready_for_selection_freeze": not pending and not errors,
        "ready_for_publication": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--review-notes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selected-catalogue",
        type=Path,
        help="New private catalogue for the existing resource-preparation tool",
    )
    args = parser.parse_args()
    raw = args.catalogue.read_bytes()
    catalogue, notes = json.loads(raw), json.loads(args.review_notes.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if notes["catalogue_sha256"] != digest:
        raise ValueError("Review notes belong to another catalogue")
    verify_frozen_cases(catalogue["cases"], json.loads(args.inventory.read_bytes()))
    choices = {
        cid: note["pool_choice"] for cid, note in notes["notes"].items() if note.get("pool_choice")
    }
    result = resolve_selection(
        catalogue["cases"],
        catalogue["selection_plan"],
        choices,
        {cid: note.get("decision") for cid, note in notes["notes"].items()},
    )
    if not result["ready_for_selection_freeze"]:
        raise ValueError(json.dumps(result))
    if args.output.exists() or (args.selected_catalogue and args.selected_catalogue.exists()):
        raise ValueError("Use new output paths, preserving earlier reviews")
    if args.selected_catalogue and args.output.resolve() == args.selected_catalogue.resolve():
        raise ValueError("Receipt and catalogue need different output paths")
    result.update(
        schema="exact-final-study-owner-selection/1", private=True, catalogue_sha256=digest
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    args.output.chmod(0o600)
    if args.selected_catalogue:
        catalogue["options"]["forty_unequal"] = result["case_ids"]
        catalogue["option_quotas"]["forty_unequal"] = result["counts_by_rank"]
        catalogue["owner_selection"] = result
        catalogue["status"] = "owner_selected_resource_preparation_pending"
        with args.selected_catalogue.open("x") as stream:
            json.dump(catalogue, stream, ensure_ascii=False)
        args.selected_catalogue.chmod(0o600)
    print(json.dumps({"selected": len(result["case_ids"]), "counts": result["counts_by_rank"]}))


if __name__ == "__main__":
    main()
