"""Read-only, count-only preparation outcomes; no provider content or credentials."""

import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path


def generation_summary(root, source_results, served_results, mode):
    sources = {r["explanation_id"]: r for r in source_results}
    served = {r["explanation_id"]: r for r in served_results}
    jobs = [
        r for r in sources.values() if r["manifest"]["requested_model"] != "none-typed-renderer"
    ]
    summary = {
        "generation_mode": mode,
        "provider_jobs": len(jobs),
        "provider_jobs_by_task": dict(Counter(r["task"] for r in jobs)),
        "saved_response_jobs": sum(r["manifest"].get("response_hash") is not None for r in jobs),
        "served_artifacts": len(served),
        "independently_reviewed_artifacts": sum(
            r.get("support_status") == "independently_reviewed" for r in served.values()
        ),
        "factual_fallback_artifacts": sum(
            r.get("preparation_kind") == "factual_fallback" for r in served.values()
        ),
        "deterministic_artifacts": sum(
            r["manifest"]["requested_model"] == "none-typed-renderer" for r in served.values()
        ),
        "provider_attempt_scope": "Explanation-framework requests in this preparation root, including retries; artifact counts use distinct explanation identities.",
        "provider_attempts": None,
        "response_successes": None,
        "authentication_failures": None,
        "other_http_failures": None,
        "unknown_attempt_outcomes": None,
        "successful_usage_records": None,
        "account_charge_status": "unverified",
    }
    ledger = Path(root) / "provider-ledger" / "requests.sqlite3"
    if mode == "typed-only":
        # The mode does not read or attribute any unrelated inherited job cache.
        summary.update(ledger_status="not_applicable", provider_batch_status="not_performed")
        for key in (
            "provider_attempts",
            "response_successes",
            "authentication_failures",
            "other_http_failures",
            "unknown_attempt_outcomes",
            "successful_usage_records",
        ):
            summary[key] = 0
        return summary
    if ledger.is_file():
        with closing(sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            identities = {
                row[0]
                for row in db.execute("SELECT request_id, identity FROM requests")
                if json.loads(row[1]).get("role") == "explanation_framework"
            }
            attempts = [
                row
                for row in db.execute("SELECT request_id, state, status, usage FROM attempts")
                if row[0] in identities
            ]
        successes = [
            row
            for row in attempts
            if row[1] == "completed" and row[2] is not None and 200 <= row[2] < 300
        ]
        authentication = sum(row[2] in {401, 403} for row in attempts)
        other = sum(
            row[2] is not None and row[2] >= 400 and row[2] not in {401, 403} for row in attempts
        )
        summary.update(
            ledger_status="available",
            provider_attempts=len(attempts),
            response_successes=len(successes),
            authentication_failures=authentication,
            other_http_failures=other,
            unknown_attempt_outcomes=len(attempts) - len(successes) - authentication - other,
            successful_usage_records=sum(bool(json.loads(row[3])) for row in successes if row[3]),
        )
    else:
        summary["ledger_status"] = "unavailable"
    summary["provider_batch_status"] = (
        "independently_reviewed_prose_available"
        if summary["independently_reviewed_artifacts"]
        else (
            "responses_without_approved_prose"
            if summary["saved_response_jobs"]
            else (
                "authentication_failed"
                if summary["provider_attempts"]
                and summary["authentication_failures"] == summary["provider_attempts"]
                else (
                    "no_successful_response"
                    if summary["provider_attempts"]
                    else "dispatch_unverified" if jobs else "not_performed"
                )
            )
        )
    )
    return summary


def generation_limitations(summary):
    if summary["generation_mode"] == "typed-only":
        return [
            "Typed-only preparation: the real OpenRouter synthesis batch and independent prose review were not performed; descriptions are deterministic renderings or factual fallbacks."
        ]
    if not summary["independently_reviewed_artifacts"]:
        if summary["provider_batch_status"] == "authentication_failed":
            return [
                f"All {summary['provider_attempts']} recorded OpenRouter attempts failed authentication; no successful provider response or independently reviewed prose is available. Served descriptions are factual fallbacks or deterministic renderings. Provider account charges have not been verified."
            ]
        return [
            "No independently reviewed provider prose is available. Served descriptions use admitted factual extracts, deterministic renderings or fallbacks; provider-mode selection alone does not establish a successful synthesis batch."
        ]
    if (
        summary["authentication_failures"]
        or summary["other_http_failures"]
        or summary["unknown_attempt_outcomes"]
    ):
        return [
            "The preparation ledger includes failed or unresolved provider attempts. Only separately approved prose is served; catalogue counts distinguish transport outcomes from reviewed artifacts."
        ]
    return []
