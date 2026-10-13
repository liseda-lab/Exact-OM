"""Resume only provider-rejected single-source questions, retaining all other work.

The first panel and its HTTP errors stay immutable. Replacement identities bind
the original source, gate, exact rejected wire and explicit transport amendment.
No valid judgment is resampled and no semantic threshold or gold changes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair import scoped_judge as s
from tools.repair.batch import read
from tools.repair.shared_release import bound


def replacement_wire(original, source_slot, source_record, gate, request, case):
    gate_slot = next(x for x in original["slots"] if x["id"] == source_slot["gate_slot"])
    gate_wire = bound(gate_slot["packet"])
    if (
        not gate["valid"]
        or gate["identity"] != canonical_hash((original, gate_slot, gate_wire, None))
        or gate["answers"]
        != s.parse(bound(gate["response"]), gate_wire, "gate", original["models"]["decisions"])
    ):
        raise ValueError("Foreign or altered original gate")
    if (
        source_record["valid"]
        or source_record["status"] != "unavailable"
        or source_record["identity"] != request["identity"]
        or source_record["identity"]
        != canonical_hash((original, source_slot, request["payload"], gate))
    ):
        raise ValueError("Replacement would resample valid or incompatible work")
    db = Path(original["ledger_directory"]) / "requests.sqlite3"
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
        errors = connection.execute(
            "SELECT status, raw FROM attempts WHERE request_id=?", (source_record["request_id"],)
        ).fetchall()
        cached_gate = connection.execute(
            "SELECT raw FROM attempts WHERE request_id=? AND status=200", (gate["request_id"],)
        ).fetchall()
    if (
        len(errors) != 1
        or errors[0][0] != 400
        or "needs at least 2 choices" not in json.loads(errors[0][1])["error"]["message"]
    ):
        raise ValueError("No confirmed matching provider validation rejection")
    if len(cached_gate) != 1 or hashlib.sha256(cached_gate[0][0]).hexdigest() != gate["raw_sha256"]:
        raise ValueError("Original gate HTTP provenance changed")
    wire = s.filter_scores(s.packet(case, source_slot, original["models"]), gate)
    expected = json.loads(json.dumps(request["payload"]))
    changed = 0
    for key, question in expected["questions"].items():
        if key.endswith(":source") and len(question["criteria"]) == 1:
            question["criteria"][
                "no_supported_source"
            ] = "No supplied meaning source supports this assessment."
            changed += 1
    if not changed or wire != expected:
        raise ValueError("Recovery changes more than the single-source choice contract")
    return wire


def validate(plan):
    original, report = bound(plan["original_plan"]), bound(plan["original_report"])
    authorization = bound(plan["authorization"])
    if (
        plan["schema"] != "exact-repair/scoped-choice-transport-recovery/v1"
        or report["plan_identity"] != canonical_hash(original)
        or authorization["user_text"] != "Ok please run that test"
        or plan["budget_phase"] != "development"
        or plan["models"] != original["models"]
        or plan["ledger_directory"] != original["ledger_directory"]
    ):
        raise ValueError("Incompatible source or authorization")
    cases = {c["id"]: c for c in bound(original["cases"])}
    costs, seen = {}, set()
    route = next(
        r
        for r in bound(plan["models"]["decisions"]["endpoint"])["data"]["endpoints"]
        if r["tag"] == "openai"
    )
    for slot in plan["slots"]:
        source = next(x for x in original["slots"] if x["id"] == slot["replaces"])
        record, gate, request = (
            bound(slot[k]) for k in ("rejected_receipt", "original_gate", "rejected_request")
        )
        if (
            slot["replaces"] in seen
            or record not in report["records"]
            or slot["stage"] != "repair_score"
            or slot["model"] != "decisions"
        ):
            raise ValueError("Duplicate or unbound rejection")
        seen.add(slot["replaces"])
        wire = replacement_wire(original, source, record, gate, request, cases[source["case"]])
        if bound(slot["packet"]) != wire or any(
            slot[k] != source[k] for k in ("case", "repeat", "swapped", "model")
        ):
            raise ValueError("Replacement identities changed")
        tokens = len(json.dumps(wire, ensure_ascii=True).encode()) + 4096
        if tokens > route["context_length"] or any(
            q["type"] == "choice" and len(q["criteria"]) < 2 for q in wire["questions"].values()
        ):
            raise ValueError("Recovery violates provider input contract")
        costs[slot["id"]] = dict(input_token_upper_bound=tokens, reserved_cost_usd=tokens * 2e-7)
    if (
        len(seen) != plan["max_requests"]
        or sum(c["reserved_cost_usd"] for c in costs.values()) > plan["pilot_reservation_usd"]
    ):
        raise ValueError("Recovery reservation exceeded")
    return original, report, cases, costs


def run(path, output):
    plan = read(path)
    original, original_report, cases, costs = validate(plan)
    s.chat.reserve(plan, costs)

    def one(slot):
        return s.old.one(plan, slot, Path(output) / "presentations", parser=s.parse)

    with ThreadPoolExecutor(max_workers=2) as pool:
        replacements = list(pool.map(one, plan["slots"]))
    mapping = {
        slot["replaces"]: r for slot, r in zip(plan["slots"], replacements) if r.get("valid")
    }
    records = []
    for record in original_report["records"]:
        if record["slot"] in mapping:
            replacement = mapping[record["slot"]]
            records.append(
                {
                    **replacement,
                    "actual_request_slot": replacement["slot"],
                    "slot": record["slot"],
                    "replaces_request_id": record["request_id"],
                    "transport_recovery": True,
                }
            )
        else:
            records.append(record)
    report = s.summarize(original, cases, records)
    report.update(
        recovery_plan_identity=canonical_hash(plan),
        original_report=plan["original_report"],
        replacements=replacements,
        reused_records=len(records) - len(mapping),
        corrected_records=len(mapping),
        completed_epoch=time.time(),
        originally_reported_cost_usd=original_report["reported_cost_usd"],
        additional_reported_cost_usd=sum(
            r.get("usage", {}).get("cost", 0) or 0 for r in replacements
        ),
        historical_rejection_costs_preserved=True,
        semantic_contract_changed=False,
    )
    write_artifact(Path(output) / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("output")
    args = parser.parse_args()
    report = run(args.plan, args.output)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "corrected_records",
                    "reused_records",
                    "methods",
                    "reported_cost_usd",
                )
            }
        )
    )
