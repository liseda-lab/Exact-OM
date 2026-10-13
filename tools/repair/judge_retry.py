"""Prospectively frozen, budgeted failure recovery; never resample a valid opinion."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sqlite3
import time

import jsonschema

from exact.llm.ledger import RequestLedger
from exact.llm.routing import extract_chat_text
from exact.repair.records import canonical_hash
from tools.repair import judge_pilot as pilot
from tools.repair.batch import locked, read
from tools.repair.historical_regression import binding
from tools.repair.shared_release import bound, immutable

REMINDER = (
    "The previous attempt did not produce a complete response in the required JSON format. "
    "Assess the same supplied evidence using the unchanged rubric. Return only a complete "
    "JSON object matching the supplied schema. Unknown and abstain remain valid answers."
)


def policy(max_retries=2):
    if type(max_retries) is not int or max_retries not in (1, 2):
        raise ValueError("Allow one or two retries only")
    return dict(
        schema="exact-repair/judge-failure-retry/v1",
        max_retries=max_retries,
        max_format_retries=1,
        backoff_seconds=[5, 15][:max_retries],
        retry_unknown_delivery=True,
        preserve_unknown_cost=True,
        semantic_resampling=False,
    )


def attempt_evidence(plan, record):
    """Read the durable transport outcome, including non-2xx and unknown delivery."""
    ledger = RequestLedger(Path(plan["ledger_directory"]))
    with sqlite3.connect(f"file:{ledger.path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT state,status,raw,usage,error FROM attempts WHERE request_id=? "
            "ORDER BY number DESC LIMIT 1",
            (record["request_id"],),
        ).fetchone()
    return dict(row) if row else {}


def failure_kind(plan, record, wire, model):
    if record["valid"]:
        return None
    evidence = attempt_evidence(plan, record)
    # A replacement worker may prove a crashed local sender exited. A live or
    # remote unresolved sender must never cause another transmission.
    if evidence.get("state") == "sent":
        try:
            RequestLedger(Path(plan["ledger_directory"])).recover_unknown(record["request_id"])
        except ValueError:
            return None
        evidence = attempt_evidence(plan, record)
    if evidence.get("state") == "unknown":
        return "transport_unknown"
    status = evidence.get("status")
    if status in (408, 429) or isinstance(status, int) and 500 <= status < 600:
        return "transient_provider"
    if not isinstance(status, int) or not 200 <= status < 300:
        return None
    try:
        response = json.loads(evidence["raw"])
    except (ValueError, TypeError):
        return "format"
    if not isinstance(response, dict):
        return "format"
    if "error" in response:
        error = response["error"]
        code = error.get("code") if isinstance(error, dict) else None
        # OpenRouter may return a provider failure inside HTTP 200.
        return (
            "transient_provider"
            if str(code) in ("408", "429") or str(code).isdigit() and 500 <= int(code) < 600
            else None
        )
    if (
        response.get("model") not in model["accepted_model_ids"]
        or response.get("provider") != model["provider_name"]
    ):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return "format"
    choice = choices[0]
    if not isinstance(choice, dict):
        return "format"
    message = choice.get("message")
    if not isinstance(message, dict):
        return "format"
    if message.get("refusal") or choice.get("finish_reason") not in (
        "stop",
        "length",
    ):
        return None
    if choice.get("finish_reason") == "length":
        return "format"
    try:
        answer = json.loads(extract_chat_text(response))
        jsonschema.validate(answer, wire["response_format"]["json_schema"]["schema"])
    except (ValueError, KeyError, TypeError, jsonschema.ValidationError):
        return "format"
    # A structurally valid but conflicting judgment is evidence, not a failed call.
    return None


def request_cost(model, wire, max_bytes):
    cost = pilot.request_cost(model, wire, max_bytes)
    route = next(
        r for r in bound(model["endpoint"])["data"]["endpoints"] if r["tag"] == model["route"]
    )
    prices = [route["pricing"]] + [
        p
        for p in route["pricing"].get("overrides", [])
        if p["min_prompt_tokens"] <= cost["input_token_upper_bound"]
    ]
    cache = max(float(p.get("input_cache_write", 0)) for p in prices)
    cost["reserved_cost_usd"] += cache * cost["input_token_upper_bound"]
    return cost


def cost_summary(records):
    """Physical attempts, not just the successful last response, determine cost."""
    return dict(
        reported_cost_usd=sum(
            r.get("attempt_reported_cost_usd", r.get("usage", {}).get("cost") or 0) for r in records
        ),
        cost_complete=all(
            r.get("attempt_cost_complete", r.get("usage", {}).get("cost") is not None)
            for r in records
        ),
    )


def run(plan, slot, directory, execute):
    frozen = plan["retry_policy"]
    if frozen != policy(frozen.get("max_retries")):
        raise ValueError("Unsupported or changed retry policy")
    if slot not in plan["slots"] or slot["stage"] != "chat":
        raise ValueError("Retry slot is foreign or not Chat")
    root = Path(directory) / slot["id"]
    with locked(root / "retry-chain.lock"):
        immutable(
            root / "retry-policy.json", dict(plan_identity=canonical_hash(plan), policy=frozen)
        )
        state = read(Path(plan["ledger_directory"]) / "phase-reservations.json")
        # All mandatory baseline calls must fit before a retry can consume headroom.
        if any(
            state["reservations"]
            .get(canonical_hash((plan["budget_phase"], s["id"])), {})
            .get("request_basis")
            != canonical_hash(plan)
            for s in plan["slots"]
        ):
            raise ValueError("Reserve the entire primary schedule before bounded retries")
        attempts, receipts, format_retries = [], [], 0
        current_plan, current_slot = plan, slot
        for number in range(frozen["max_retries"] + 1):
            record = execute(current_plan, current_slot, directory)
            evidence = attempt_evidence(current_plan, record)
            usage = record.get("usage", {})
            if not usage and evidence.get("raw"):
                try:
                    raw_response = json.loads(evidence["raw"])
                    usage = raw_response.get("usage", {}) if isinstance(raw_response, dict) else {}
                except ValueError:
                    pass
            attempts.append(dict(record, usage=usage or {}))
            receipts.append(binding(Path(directory) / current_slot["id"] / "receipt.json"))
            wire = bound(current_slot["packet"])
            model = plan["models"][slot["model"]]
            kind = failure_kind(current_plan, record, wire, model)
            stop = "success" if record["valid"] else "not_retryable"
            if kind is None:
                break
            stop = "retry_limit"
            if number == frozen["max_retries"] or kind == "format" and format_retries == 1:
                break
            retry_slot = dict(slot, id=slot["id"] + f"-retry-{number + 1}")
            if retry_slot["id"] in {s["id"] for s in plan["slots"]}:
                raise ValueError("Retry identity collides with a primary presentation")
            retry_root = Path(directory) / retry_slot["id"]
            retry_wire = copy.deepcopy(wire)
            if kind == "format":
                retry_wire["messages"].append(dict(role="user", content=REMINDER))
                format_retries += 1
            immutable(retry_root / "wire.json", retry_wire)
            retry_slot["packet"] = binding(retry_root / "wire.json")
            retry_plan = dict(
                plan,
                slots=[retry_slot],
                retry_lineage=dict(
                    root_plan_identity=canonical_hash(plan),
                    logical_slot=slot["id"],
                    retry_number=number + 1,
                    failure_kind=kind,
                    previous_receipt=receipts[-1],
                ),
            )
            immutable(retry_root / "plan.json", retry_plan)
            # Replay a saved retry even after the deadline; no new HTTP call occurs.
            if not (retry_root / "transmission-claimed.json").exists():
                stop = "deadline"
                delay = frozen["backoff_seconds"][number]
                if time.time() + delay + 190 > plan["deadline_epoch"]:
                    break
                try:
                    pilot.reserve(
                        retry_plan,
                        {
                            retry_slot["id"]: request_cost(
                                model, retry_wire, plan["max_input_bytes"]
                            )
                        },
                    )
                except pilot.BudgetExceeded:
                    stop = "budget"
                    break
                time.sleep(delay)
            current_plan, current_slot = retry_plan, retry_slot
        costs = cost_summary(attempts)
        # Exactly one logical observation. The physical ID and complete lineage
        # remain explicit, so retries never count as independent votes.
        return dict(
            record,
            slot=slot["id"],
            effective_slot=record["slot"],
            attempt_receipts=receipts,
            attempt_count=len(attempts),
            retry_stop=stop,
            technical_failure_kind=kind,
            attempt_reported_cost_usd=costs["reported_cost_usd"],
            attempt_cost_complete=costs["cost_complete"],
        )
