from pathlib import Path

import httpx
import pytest

from tests.repair_october_learning_annotation_runner_test import manifest
from tests.repair_semantic_fidelity_test import adapter, judgment, packet, response
from tools.repair import annotation_backoff as backoff
from tools.repair import corrective_semantics as runner


def settings(tmp_path):
    value = manifest(tmp_path)
    approval = tmp_path / "authorization.json"
    runner.write_artifact(
        approval,
        dict(
            schema="exact-repair/transient-provider-authorization/v1",
            status="approved",
            automatic_429_recovery=True,
            lineage_id=value["lineage_id"],
            cost_ceiling_usd=35,
            user_instruction="Automatically back off on temporary provider limits.",
        ),
    )
    value["rate_limit_policy"] = dict(
        authorization=dict(path=str(approval), sha256=runner.sha(approval)),
        base_seconds=0.001,
        max_seconds=0.002,
    )
    return value


def rejection(status=429, **body):
    return httpx.Response(
        status,
        json=dict(error=dict(code=status), **body),
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )


def install(tmp_path, monkeypatch, responses):
    import exact.llm.routing

    replies = iter(responses)
    client, calls = adapter(tmp_path / "ledger", monkeypatch, handler=lambda **kw: next(replies))
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    return client, calls


def test_multiple_429s_are_transport_waits_preserve_every_attempt_and_resume_free(
    tmp_path, monkeypatch
):
    client, calls = install(
        tmp_path, monkeypatch, [rejection(), rejection(), response(judgment(packet()))]
    )
    config, output = settings(tmp_path), tmp_path / "slot"
    path = backoff.annotate(packet(), config, output, slot_id="original", seconds=300)
    assert path == output / "labels.json" and len(calls) == 3
    checkpoint = runner.read(output / "backoff.json")
    assert checkpoint["complete"] and len(checkpoint["failures"]) == 2
    for failure in checkpoint["failures"]:
        assert runner.sha(failure["receipt"]) == failure["receipt_sha256"]
    state = runner.read(tmp_path / "ledger/phase-reservations.json")
    assert len(state["reservations"]) == 3
    assert {r["comparison_id"] for r in state["reservations"].values()} == {"original"}
    from exact.repair.semantic_fidelity import read_fidelity_training_artifact

    labels = read_fidelity_training_artifact(path, "train")
    aggregates = [a for values in labels.values() for _, a in values]
    assert len(aggregates) == 1 and aggregates[0].global_target_eligible
    assert len(aggregates[0].result["unique_observations"]) == 1
    assert backoff.annotate(packet(), config, output, slot_id="original", seconds=1) == path
    assert len(calls) == 3
    path.write_text("{}")
    with pytest.raises(ValueError, match="terminal output changed"):
        backoff.annotate(packet(), config, output, slot_id="original", seconds=300)


@pytest.mark.parametrize("status", [401, 403, 500, 503])
def test_other_failures_are_not_unboundedly_retried(tmp_path, monkeypatch, status):
    _, calls = install(tmp_path, monkeypatch, [rejection(status)])
    with pytest.raises(runner.ConfirmedAnnotationFailure):
        backoff.annotate(packet(), settings(tmp_path), tmp_path / "out", slot_id="s", seconds=300)
    assert len(calls) == 1


def test_generation_evidence_on_429_does_not_authorize_another_observation(tmp_path, monkeypatch):
    _, calls = install(tmp_path, monkeypatch, [rejection(usage=dict(completion_tokens=3))])
    with pytest.raises(runner.ConfirmedAnnotationFailure):
        backoff.annotate(packet(), settings(tmp_path), tmp_path / "out", slot_id="s", seconds=300)
    assert len(calls) == 1


def test_invalid_success_is_not_retried(tmp_path, monkeypatch):
    _, calls = install(tmp_path, monkeypatch, [response("not-json")])
    assert (
        backoff.annotate(packet(), settings(tmp_path), tmp_path / "out", slot_id="s", seconds=300)
        is None
    )
    assert len(calls) == 1


def test_wait_is_checkpointed_without_renewing_deadline(tmp_path, monkeypatch):
    wait = rejection()
    wait = httpx.Response(
        429,
        json=dict(error=dict(code=429, metadata=dict(retry_after_seconds=600))),
        request=wait.request,
    )
    _, calls = install(tmp_path, monkeypatch, [wait])
    config, output = settings(tmp_path), tmp_path / "out"
    assert backoff.annotate(packet(), config, output, slot_id="s", seconds=300) is None
    first = runner.read(output / "backoff.json")
    assert runner.read(output / "receipt.json")["status"] == "rate_limit_deferred"
    assert backoff.annotate(packet(), config, output, slot_id="s", seconds=3000) is None
    assert runner.read(output / "backoff.json")["deadline_epoch"] == first["deadline_epoch"]
    assert len(calls) == 1


def test_frozen_shared_window_resumes_after_short_worker_slice(tmp_path, monkeypatch):
    import time

    clock = [time.time()]
    wait = httpx.Response(
        429,
        json=dict(error=dict(code=429, metadata=dict(retry_after_seconds=600))),
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )
    _, calls = install(tmp_path, monkeypatch, [wait, response(judgment(packet()))])
    config, output = settings(tmp_path), tmp_path / "out"
    config["deadline_epoch"] = clock[0] + 3600
    config["rate_limit_policy"]["resume_deadline_epoch"] = clock[0] + 1800
    monkeypatch.setattr(backoff.time, "time", lambda: clock[0])
    assert backoff.annotate(packet(), config, output, slot_id="s", seconds=120) is None
    first = runner.read(output / "backoff.json")
    clock[0] += 601
    assert (
        backoff.annotate(packet(), config, output, slot_id="s", seconds=120)
        == output / "labels.json"
    )
    last = runner.read(output / "backoff.json")
    assert last["deadline_epoch"] == first["deadline_epoch"]
    assert len(last["failures"]) == 1 and len(calls) == 2


def test_each_rejected_request_counts_towards_existing_budget(tmp_path, monkeypatch):
    _, calls = install(tmp_path, monkeypatch, [rejection()])
    config = settings(tmp_path)
    config["cost_ceiling_usd"] = 0.015  # one 0.012 reservation, but not two
    auth = Path(config["rate_limit_policy"]["authorization"]["path"])
    value = runner.read(auth)
    value["cost_ceiling_usd"] = config["cost_ceiling_usd"]
    runner.write_artifact(auth, value)
    config["rate_limit_policy"]["authorization"]["sha256"] = runner.sha(auth)
    assert backoff.annotate(packet(), config, tmp_path / "out", slot_id="s", seconds=300) is None
    assert len(calls) == 1
    assert runner.read(tmp_path / "out/receipt.json")["status"] == "budget_unavailable"


def test_explicit_authorization_is_bound(tmp_path):
    config = settings(tmp_path)
    Path(config["rate_limit_policy"]["authorization"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="authorization changed"):
        backoff.validate_policy(config)


def test_runner_uses_authorized_backoff_and_publishes_normal_terminal_artifacts(
    tmp_path, monkeypatch
):
    _, calls = install(tmp_path, monkeypatch, [rejection(), response(judgment(packet()))])
    monkeypatch.setattr(runner, "_authentication_preflight", lambda *a: None)
    config = settings(tmp_path)
    p = tmp_path / "packet.json"
    runner.write_artifact(p, packet().to_dict())
    config.update(seconds=300, slots=[dict(id="s", packet=dict(path=str(p), sha256=runner.sha(p)))])
    source = tmp_path / "manifest.json"
    runner.write_artifact(source, config)
    result = runner.run(source, tmp_path / "output")
    assert result[0]["status"] == "complete"
    assert result[0]["artifact"] == str(tmp_path / "output/s/labels.json")
    assert runner.read(tmp_path / "output/report.json")["status"] == "complete"
    assert len(calls) == 2


def test_fresh_comparison_path_recovers_429_without_resampling_valid_labels(tmp_path, monkeypatch):
    _, calls = install(tmp_path, monkeypatch, [rejection(), response(judgment(packet()))])
    config = settings(tmp_path)
    output = tmp_path / "fresh-decoded-comparison"
    result = runner.annotate_comparison(
        packet(), config, output, comparison_id="dev-slot", order_swap_audit=False, seconds=300
    )
    assert result == output / "original/labels.json"
    assert len(calls) == 2
    state = runner.read(output / "original/backoff.json")
    assert state["complete"] and len(state["failures"]) == 1
    assert (
        runner.annotate_comparison(
            packet(), config, output, comparison_id="dev-slot", order_swap_audit=False, seconds=1
        )
        == result
    )
    assert len(calls) == 2
