"""Observed transport failures stay distinct from successful reviewed prose."""

import json
import sqlite3

from exact_inspect.contracts import file_hash
from tools.review_generation_summary import generation_limitations, generation_summary


def artifact(key, *, reviewed=False, deterministic=False, response=False):
    return {
        "explanation_id": key,
        "task": "entity_profile",
        "support_status": "independently_reviewed" if reviewed else "fallback",
        "preparation_kind": "grounded_description" if reviewed else "factual_fallback",
        "manifest": {
            "requested_model": "none-typed-renderer" if deterministic else "provider-model",
            "response_hash": "saved-response" if response else None,
        },
    }


def ledger(root, attempts):
    path = root / "provider-ledger/requests.sqlite3"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE requests(request_id TEXT, identity TEXT)")
        db.execute("CREATE TABLE attempts(request_id TEXT, state TEXT, status INTEGER, usage TEXT)")
        for index, (state, status, usage, role) in enumerate(attempts):
            key = str(index)
            db.execute("INSERT INTO requests VALUES (?,?)", (key, json.dumps({"role": role})))
            db.execute("INSERT INTO attempts VALUES (?,?,?,?)", (key, state, status, usage))
    return path


def test_authentication_failure_batch_has_no_success_or_charge_claim(tmp_path):
    path = ledger(tmp_path, [("rejected", 401, None, "explanation_framework")] * 86)
    before = file_hash(path)
    sources = [artifact(str(i)) for i in range(86)]
    summary = generation_summary(tmp_path, sources, sources + sources, "provider")
    assert summary["provider_jobs"] == 86
    assert summary["provider_attempts"] == summary["authentication_failures"] == 86
    assert summary["response_successes"] == summary["saved_response_jobs"] == 0
    assert summary["successful_usage_records"] == 0
    assert summary["account_charge_status"] == "unverified"
    assert summary["provider_batch_status"] == "authentication_failed"
    assert summary["factual_fallback_artifacts"] == 86
    assert summary["independently_reviewed_artifacts"] == 0
    assert (
        "All 86 recorded OpenRouter attempts failed authentication"
        in generation_limitations(summary)[0]
    )
    assert file_hash(path) == before


def test_mixed_transport_and_review_outcomes_count_separately(tmp_path):
    ledger(
        tmp_path,
        [
            ("completed", 200, '{"prompt_tokens":12}', "explanation_framework"),
            ("rejected", 403, None, "explanation_framework"),
            ("rejected", 500, None, "explanation_framework"),
            ("unknown", None, None, "explanation_framework"),
            ("completed", 200, "{}", "unrelated_role"),
        ],
    )
    source, reviewed, fallback = (
        artifact("source", response=True),
        artifact("reviewed", reviewed=True, response=True),
        artifact("fallback"),
    )
    summary = generation_summary(tmp_path, [source, fallback], [reviewed, fallback], "provider")
    assert summary["provider_attempts"] == 4
    assert (
        summary["response_successes"]
        == summary["authentication_failures"]
        == summary["other_http_failures"]
        == summary["unknown_attempt_outcomes"]
        == 1
    )
    assert summary["saved_response_jobs"] == summary["successful_usage_records"] == 1
    assert summary["independently_reviewed_artifacts"] == summary["factual_fallback_artifacts"] == 1
    assert "failed or unresolved" in generation_limitations(summary)[0]


def test_missing_ledger_is_unknown_and_typed_mode_ignores_inherited_cache(tmp_path):
    source = artifact("failed")
    missing = generation_summary(tmp_path, [source], [source], "provider")
    assert missing["provider_attempts"] is None
    assert missing["provider_batch_status"] == "dispatch_unverified"
    ledger(tmp_path, [("rejected", 401, None, "explanation_framework")])
    typed = artifact("typed", deterministic=True)
    result = generation_summary(tmp_path, [typed], [typed], "typed-only")
    assert result["provider_attempts"] == result["provider_jobs"] == 0
    assert result["provider_batch_status"] == "not_performed"
