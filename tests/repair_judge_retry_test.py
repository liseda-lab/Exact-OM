import json
from pathlib import Path
import sqlite3
import time

import httpx
import jsonschema
import pytest

from exact.llm.routing import OpenRouterClient, extract_chat_text
from exact.repair.api import write_artifact
from tools.repair import judge_pilot as pilot, judge_retry as retry, probabilistic_judge as worker
from tools.repair.batch import read
from tools.repair.historical_regression import binding
from tools.repair.shared_release import bound, immutable


SCHEMA = dict(
    type="object",
    additionalProperties=False,
    required=["preference", "evidence"],
    properties=dict(
        preference=dict(enum=["A", "B", "tie", "abstain"]), evidence=dict(type="string")
    ),
)


def parse(response, wire, stage, model):
    if (
        response.get("model") not in model["accepted_model_ids"]
        or response.get("provider") != model["provider_name"]
        or response["choices"][0]["finish_reason"] != "stop"
    ):
        raise ValueError("Identity/completion mismatch")
    answer = json.loads(extract_chat_text(response))
    jsonschema.validate(answer, SCHEMA)
    if answer["evidence"] != "D1":
        raise ValueError("Ungrounded semantic judgment")
    return answer


def ok(preference="A", **extra):
    return dict(
        model="model",
        provider="Provider",
        usage=dict(cost=0.001),
        choices=[
            dict(
                finish_reason="stop",
                message=dict(content=json.dumps(dict(preference=preference, evidence="D1"))),
            )
        ],
        **extra,
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    model = dict(
        route="route",
        provider_name="Provider",
        accepted_model_ids=["model"],
        profile=dict(
            backend="openrouter",
            model="model",
            revision="fixed-v1",
            api_key_env="RETRY_TEST_KEY",
            timeout_secs=180,
            provider=dict(only=["route"], allow_fallbacks=False, require_parameters=True),
        ),
    )
    immutable(
        tmp_path / "endpoint.json",
        dict(
            data=dict(
                endpoints=[
                    dict(
                        tag="route",
                        model_id="model",
                        provider_name="Provider",
                        context_length=1000000,
                        max_completion_tokens=10000,
                        supported_parameters=[
                            "reasoning",
                            "max_tokens",
                            "response_format",
                            "structured_outputs",
                        ],
                        pricing=dict(
                            prompt="0.000001", completion="0.000002", input_cache_write="0.000001"
                        ),
                    )
                ]
            )
        ),
    )
    model["endpoint"] = binding(tmp_path / "endpoint.json")
    wire = dict(
        model="model",
        provider=model["profile"]["provider"],
        max_tokens=1000,
        reasoning=dict(effort="low"),
        messages=[
            dict(role="system", content="Compare using the fixed rubric."),
            dict(role="user", content='{"evidence":{"D1":"fixed"}}'),
        ],
        response_format=dict(
            type="json_schema", json_schema=dict(name="test", strict=True, schema=SCHEMA)
        ),
    )
    immutable(tmp_path / "wire.json", wire)
    slot = dict(
        id="case-0",
        case="case",
        model="judge",
        stage="chat",
        repeat=0,
        swapped=False,
        packet=binding(tmp_path / "wire.json"),
    )
    state = dict(
        cost_ceiling_usd=35,
        total_request_limit=100,
        request_limits=dict(train=10, development=10, test=10),
        reservations={"historical": dict(phase="train", reserved_cost_usd=2, state="unresolved")},
    )
    immutable(tmp_path / "previous.json", state)
    immutable(tmp_path / "ledger/phase-reservations.json", state)
    plan = dict(
        slots=[slot],
        models={"judge": model},
        budget_phase="test",
        semantic_role="test_assessment",
        transport_role="test_assessment",
        ledger_directory=str(tmp_path / "ledger"),
        previous_phase_state=binding(tmp_path / "previous.json"),
        deadline_epoch=time.time() + 3600,
        max_input_bytes=1000000,
        retry_policy=retry.policy(),
    )
    calls, replies = [], []

    def handler(request):
        calls.append(json.loads(request.content))
        response = replies[min(len(calls) - 1, len(replies) - 1)]
        if isinstance(response, Exception):
            raise response
        return response

    original = OpenRouterClient.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._client.close()
        self._client = httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(OpenRouterClient, "__init__", init)
    monkeypatch.setenv("RETRY_TEST_KEY", "test-only")
    monkeypatch.setattr(retry.time, "sleep", lambda seconds: None)
    return plan, slot, calls, replies


def admit(plan):
    pilot.reserve(
        plan,
        {
            s["id"]: retry.request_cost(
                plan["models"][s["model"]], bound(s["packet"]), plan["max_input_bytes"]
            )
            for s in plan["slots"]
        },
    )


def run(plan, slot, tmp_path):
    return worker.one(plan, slot, tmp_path / "out", parser=parse)


@pytest.mark.parametrize(
    "first",
    [
        httpx.Response(
            200, json=dict(error=dict(code=502, message="Provider returned an empty response"))
        ),
        httpx.Response(429, json=dict(error=dict(code=429))),
        httpx.Response(503, json=dict(error=dict(code=503), usage=dict(cost=0.0002))),
        httpx.ReadTimeout("unknown delivery"),
    ],
)
def test_transport_recovery_is_one_vote_with_all_attempts_and_safe_resume(setup, tmp_path, first):
    plan, slot, calls, replies = setup
    replies.extend([first, httpx.Response(200, json=ok())])
    admit(plan)
    report = run(plan, slot, tmp_path)
    assert report["valid"] and report["attempt_count"] == 2 and len(calls) == 2
    assert report["slot"] == slot["id"] and report["effective_slot"] == slot["id"] + "-retry-1"
    assert calls[0] == calls[1] and report["answers"]["preference"] == "A"
    records = [bound(b) for b in report["attempt_receipts"]]
    assert len({r["request_id"] for r in records}) == 2
    assert not records[0]["valid"] and records[1]["valid"]
    state = read(tmp_path / "ledger/phase-reservations.json")
    assert (
        len(state["reservations"]) == 3
        and state["reservations"]["historical"]["reserved_cost_usd"] == 2
    )
    assert run(plan, slot, tmp_path) == report and len(calls) == 2
    assert read(tmp_path / "ledger/phase-reservations.json") == state
    expected = 0.0012 if isinstance(first, httpx.Response) and first.status_code == 503 else 0.001
    assert retry.cost_summary([report])["reported_cost_usd"] == pytest.approx(expected)
    assert report["attempt_cost_complete"] == (expected == 0.0012)


def test_failure_limit_survives_replacement_and_changed_policy_rejected(setup, tmp_path):
    plan, slot, calls, replies = setup
    replies.append(httpx.Response(503, json=dict(error=dict(code=503))))
    admit(plan)
    report = run(plan, slot, tmp_path)
    assert (
        not report["valid"]
        and report["attempt_count"] == 3
        and report["retry_stop"] == "retry_limit"
    )
    assert run(plan, slot, tmp_path) == report and len(calls) == 3
    changed = {**plan, "retry_policy": retry.policy(1)}
    with pytest.raises(ValueError):
        run(changed, slot, tmp_path)
    # Losing the last worker receipt still reuses the durable HTTP response.
    Path(report["attempt_receipts"][-1]["path"]).unlink()
    restored = run(plan, slot, tmp_path)
    assert not restored["valid"] and len(calls) == 3


def test_format_reminder_preserves_evidence_and_only_one_format_retry(setup, tmp_path):
    plan, slot, calls, replies = setup
    bad = ok()
    bad["choices"][0]["message"]["content"] = "{broken"
    replies.extend([httpx.Response(200, json=bad), httpx.Response(200, json=ok("abstain"))])
    admit(plan)
    report = run(plan, slot, tmp_path)
    assert report["valid"] and report["answers"]["preference"] == "abstain"
    assert calls[1] == {
        **calls[0],
        "messages": calls[0]["messages"] + [dict(role="user", content=retry.REMINDER)],
    }
    assert report["attempt_count"] == 2


def test_persistent_bad_format_stops_without_a_third_opinion(setup, tmp_path):
    plan, slot, calls, replies = setup
    bad = ok()
    bad["choices"][0]["message"]["content"] = "{}"
    replies.append(httpx.Response(200, json=bad))
    admit(plan)
    report = run(plan, slot, tmp_path)
    assert not report["valid"] and len(calls) == 2


@pytest.mark.parametrize(
    "kind", ["abstain", "tie", "B", "ungrounded", "identity", "auth", "refusal"]
)
def test_valid_opinions_and_permanent_or_semantic_faults_are_not_retried(setup, tmp_path, kind):
    plan, slot, calls, replies = setup
    response = ok(kind if kind in ("abstain", "tie", "B") else "A")
    if kind == "ungrounded":
        response["choices"][0]["message"]["content"] = json.dumps(
            dict(preference="A", evidence="foreign")
        )
    if kind == "identity":
        response["model"] = "wrong-model"
    if kind == "refusal":
        response["choices"][0]["finish_reason"] = "content_filter"
    replies.append(httpx.Response(401 if kind == "auth" else 200, json=response))
    admit(plan)
    report = run(plan, slot, tmp_path)
    assert len(calls) == 1 and report["attempt_count"] == 1
    assert report["valid"] == (kind in ("abstain", "tie", "B"))


@pytest.mark.parametrize("limit", ["phase", "total", "money", "deadline"])
def test_retry_honors_limits_before_transmission(setup, tmp_path, limit, monkeypatch):
    plan, slot, calls, replies = setup
    state = read(tmp_path / "previous.json")
    if limit == "phase":
        state["request_limits"]["test"] = 1
    if limit == "total":
        state["total_request_limit"] = 2
    if limit == "money":
        state["cost_ceiling_usd"] = 2.01
    write_artifact(tmp_path / "previous.json", state)
    write_artifact(tmp_path / "ledger/phase-reservations.json", state)
    plan["previous_phase_state"] = binding(tmp_path / "previous.json")
    replies.append(httpx.Response(503, json=dict(error=dict(code=503))))
    admit(plan)
    if limit == "deadline":
        monkeypatch.setattr(retry.time, "time", lambda: plan["deadline_epoch"] - 192)
    report = run(plan, slot, tmp_path)
    assert not report["valid"] and len(calls) == 1
    assert report["retry_stop"] == ("deadline" if limit == "deadline" else "budget")
    assert len(read(tmp_path / "ledger/phase-reservations.json")["reservations"]) == 2


def test_entire_baseline_must_be_reserved_and_legacy_plans_stay_one_shot(setup, tmp_path):
    plan, slot, calls, replies = setup
    with pytest.raises(ValueError, match="entire primary schedule"):
        run(plan, slot, tmp_path)
    assert not calls
    plan.pop("retry_policy")
    replies.append(httpx.Response(503, json=dict(error=dict(code=503))))
    admit(plan)
    assert not run(plan, slot, tmp_path)["valid"] and len(calls) == 1


def test_live_unresolved_sender_is_never_duplicated(setup, tmp_path):
    plan, slot, calls, replies = setup
    replies.append(httpx.Response(503, json=dict(error=dict(code=503))))
    admit(plan)
    first = worker._one(plan, slot, tmp_path / "out", parser=parse)
    with sqlite3.connect(tmp_path / "ledger/requests.sqlite3") as db:
        db.execute("UPDATE attempts SET state='sent' WHERE request_id=?", (first["request_id"],))
    report = run(plan, slot, tmp_path)
    assert not report["valid"] and report["attempt_count"] == 1 and len(calls) == 1


def test_cached_retry_recovers_missing_receipt_after_deadline(setup, tmp_path, monkeypatch):
    plan, slot, calls, replies = setup
    replies.extend(
        [httpx.Response(503, json=dict(error=dict(code=503))), httpx.Response(200, json=ok())]
    )
    admit(plan)
    first = run(plan, slot, tmp_path)
    Path(first["attempt_receipts"][-1]["path"]).unlink()
    monkeypatch.setattr(retry.time, "time", lambda: plan["deadline_epoch"] + 1)
    restored = run(plan, slot, tmp_path)
    assert restored["valid"] and len(calls) == 2
