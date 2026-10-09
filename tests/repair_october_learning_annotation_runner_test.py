"""Corrective annotation wrapper uses the real ledger and mock HTTP only."""

import json

import pytest

from tests.repair_semantic_fidelity_test import adapter, judgment, packet, response
from tools.repair import corrective_semantics as runner


def manifest(tmp_path):
    return dict(
        schema="exact-repair/corrective-annotation/v1",
        authorized=True,
        phase="calibration",
        cost_ceiling_usd=35,
        request_limits=runner.PHASE_LIMITS,
        lineage_id="annotation-runner-fixture",
        ledger_directory=str(tmp_path / "ledger"),
        profile="teacher",
        teacher_profile="teacher",
        test_profile="evaluator",
        profiles={
            name: dict(backend="openrouter", model="vendor/" + name, provider={"only": ["pinned"]})
            for name in ("teacher", "evaluator")
        },
        prices_per_million={"teacher": dict(input=1, output=2)},
        parent_splits={"parent-train": "train"},
        selection_model_ids=["vendor/teacher"],
        data_permissions="mock-only",
    )


def test_annotation_config_is_stable_across_remaining_time_and_completed_replay(
    tmp_path, monkeypatch
):
    import exact.llm.routing

    client, calls = adapter(tmp_path / "ledger", monkeypatch)
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    settings = manifest(tmp_path)
    destination = tmp_path / "annotation"
    first = runner.annotate_packet(
        packet(), settings, destination, slot_id="frozen-slot", seconds=300
    )
    assert first is not None and len(calls) == 1
    second = runner.annotate_packet(
        packet(), settings, destination, slot_id="frozen-slot", seconds=1
    )
    assert second == first and len(calls) == 1
    state = runner.read(tmp_path / "ledger" / "phase-reservations.json")
    assert len(state["reservations"]) == 1
    assert next(iter(state["reservations"].values()))["state"] == "completed"
    saved_run = runner.read(destination / "run.json")
    assert saved_run["record"]["max_seconds_per_request"] == 90


def test_unstarted_short_time_slice_spends_no_allowance_and_changed_context_rejected(tmp_path):
    settings = manifest(tmp_path)
    assert (
        runner.annotate_packet(packet(), settings, tmp_path / "short", slot_id="s", seconds=30)
        is None
    )
    assert not (tmp_path / "ledger" / "phase-reservations.json").exists()
    assert runner._reserve_phase(settings, "s", packet().content_hash, 0.01) is None
    assert runner._reserve_phase(settings, "s", packet().content_hash, 0.01) is False
    with pytest.raises(ValueError, match="changed"):
        runner._reserve_phase(
            {**settings, "profile": "different"}, "s", packet().content_hash, 0.01
        )


def test_unique_comparison_quota_is_distinct_from_request_and_swap_quota(tmp_path):
    settings = {**manifest(tmp_path), "phase": "development"}
    for index in range(64):
        runner._reserve_phase(
            settings, f"slot:{index}", str(index), 0.001, comparison_id=f"comparison:{index}"
        )
    # One independently charged order-swap request shares its original comparison.
    assert (
        runner._reserve_phase(settings, "slot:0:swap", "0", 0.001, comparison_id="comparison:0")
        is None
    )
    with pytest.raises(ValueError, match="unique comparison"):
        runner._reserve_phase(settings, "slot:64", "64", 0.001, comparison_id="comparison:64")


def test_required_order_swap_is_two_paid_attempts_one_comparison_and_replay(tmp_path, monkeypatch):
    import exact.llm.routing

    p = packet()

    def transport(**kwargs):
        context = json.loads(json.loads(kwargs["content"])["messages"][1]["content"])
        swapped = context["swapped"]
        return response(
            judgment(
                p,
                swapped=swapped,
                decision="B" if swapped else "A",
                a=0.25 if swapped else 0.75,
                b=0.75 if swapped else 0.25,
            )
        )

    client, calls = adapter(tmp_path / "ledger", monkeypatch, handler=transport)
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    settings = manifest(tmp_path)
    output = tmp_path / "audit"
    artifact = runner.annotate_comparison(
        p, settings, output, comparison_id="frozen-pair", order_swap_audit=True, seconds=300
    )
    assert artifact is not None and len(calls) == 2
    aggregate = runner.read_record(runner.read(artifact)["comparisons"][0]["comparison"])
    assert aggregate.global_target_eligible and aggregate.decision == "A"
    assert len(aggregate.result["unique_observations"]) == 2
    state = runner.read(tmp_path / "ledger" / "phase-reservations.json")
    assert len(state["reservations"]) == 2
    assert {row["comparison_id"] for row in state["reservations"].values()} == {"frozen-pair"}
    assert (
        runner.annotate_comparison(
            p, settings, output, comparison_id="frozen-pair", order_swap_audit=True, seconds=1
        )
        == artifact
    )
    assert len(calls) == 2


def test_missing_required_order_swap_does_not_admit_original_alone(tmp_path, monkeypatch):
    import exact.llm.routing

    client, calls = adapter(tmp_path / "ledger", monkeypatch)
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    settings = manifest(tmp_path)
    output = tmp_path / "audit"
    assert (
        runner.annotate_packet(
            packet(),
            settings,
            output / "original",
            slot_id="pair:original",
            comparison_id="pair",
            seconds=300,
        )
        is not None
    )
    assert (
        runner.annotate_comparison(
            packet(), settings, output, comparison_id="pair", order_swap_audit=True, seconds=1
        )
        is None
    )
    assert len(calls) == 1
    assert runner.read(output / "audit.json")["status"] == "required_swap_unavailable"


def test_outer_deadline_blocks_hosted_transmission_without_spending(tmp_path, monkeypatch):
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(runner.time.time() + 80))
    assert (
        runner.annotate_packet(
            packet(), manifest(tmp_path), tmp_path / "late", slot_id="s", seconds=300
        )
        is None
    )
    assert not (tmp_path / "ledger" / "phase-reservations.json").exists()


def test_spend_ceiling_masks_slot_without_transmission_or_retry(tmp_path, monkeypatch):
    import exact.llm.routing

    client, calls = adapter(tmp_path / "ledger", monkeypatch)
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    settings = {**manifest(tmp_path), "cost_ceiling_usd": 0.001}
    output = tmp_path / "budget"
    assert runner.annotate_packet(packet(), settings, output, slot_id="s", seconds=300) is None
    assert calls == []
    assert runner.read(output / "receipt.json")["status"] == "budget_unavailable"
    assert runner.annotate_packet(packet(), settings, output, slot_id="s", seconds=300) is None
    assert calls == []


def test_calibration_consumption_is_frozen_before_any_hosted_attempt(tmp_path, monkeypatch):
    path = tmp_path / "calibration" / "manifest.json"
    declaration = {**manifest(tmp_path), "slots": [], "seconds": 0}
    runner.write_artifact(path, declaration)
    monkeypatch.setattr(
        runner,
        "annotate_packet",
        lambda *a, **k: pytest.fail("Empty calibration must not invoke a provider"),
    )
    assert runner.run(path, tmp_path / "output") == []
    used = runner.read(path.parent / "used.json")
    assert used["manifest_path"] == str(path.resolve())
    assert used["manifest_sha256"] == runner.sha(path)
    assert not (tmp_path / "ledger").exists()
    assert runner.run(path, tmp_path / "output") == []
    runner.write_artifact(path, {**declaration, "seconds": 1})
    with pytest.raises(ValueError, match="Frozen corpus artifact changed"):
        runner.run(path, tmp_path / "output")


@pytest.mark.parametrize("status", [401, 403, 404, 429, 503])
def test_confirmed_http_failure_stops_schedule_and_replay_without_renewing_quotas(
    tmp_path, monkeypatch, status
):
    import httpx
    import exact.llm.routing

    def transport(**kwargs):
        if kwargs["method"] == "GET":
            return httpx.Response(
                200, json={"data": {}}, request=httpx.Request("GET", kwargs["url"])
            )
        return httpx.Response(
            status,
            json={"error": {"code": status, "message": "Rejected"}},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )

    client, calls = adapter(tmp_path / "ledger", monkeypatch, handler=transport)
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    packet_path = tmp_path / "packet.json"
    runner.write_artifact(packet_path, packet().to_dict())
    slots = [
        dict(id=f"slot-{i}", packet=dict(path=str(packet_path), sha256=runner.sha(packet_path)))
        for i in range(3)
    ]
    settings = {**manifest(tmp_path), "slots": slots, "seconds": 300}
    path = tmp_path / "calibration" / "manifest.json"
    runner.write_artifact(path, settings)
    output = tmp_path / "output"
    for _ in range(2):
        with pytest.raises(runner.ConfirmedAnnotationFailure, match=f"HTTP {status}"):
            runner.run(path, output)
        assert sum(call["method"] == "POST" for call in calls) == 1
        report = runner.read(output / "report.json")
        assert report["status"] == ("blocked_external" if status in {401, 403} else "failed")
        assert report["scheduled"] == report["recorded"] == 3
        if status in {401, 403}:
            assert report["blocker"]["kind"] == "provider_authentication"
            assert report["blocker"]["retry_permitted"] is False
        assert [row["status"] for row in report["rows"]] == [
            "provider_error",
            "not_attempted_provider_failure",
            "not_attempted_provider_failure",
        ]
        assert report["confirmed_failure"]["http_status"] == status
        state = runner.read(tmp_path / "ledger" / "phase-reservations.json")
        assert len(state["reservations"]) == 1
        assert sum(row["reserved_cost_usd"] for row in state["reservations"].values()) > 0
        assert not (output / "slot-1").exists()
    # The same exact qualification also protects replay of historical masked HTTP errors.
    receipt_path = output / "slot-0" / "receipt.json"
    receipt = runner.read(receipt_path)
    receipt["status"] = "annotation_unavailable"
    receipt.pop("confirmed_failure")
    runner.write_artifact(receipt_path, receipt)
    with pytest.raises(runner.ConfirmedAnnotationFailure):
        runner.run(path, output)
    assert sum(call["method"] == "POST" for call in calls) == 1
    with pytest.raises(runner.ConfirmedAnnotationFailure):
        runner.run(path, tmp_path / "replacement-output")
    assert sum(call["method"] == "POST" for call in calls) == 1


def test_free_authentication_preflight_preserves_every_scientific_slot(tmp_path, monkeypatch):
    import httpx
    import exact.llm.routing

    def rejected(**kwargs):
        assert kwargs["method"] == "GET"
        return httpx.Response(
            401, json={"error": {"code": 401}}, request=httpx.Request("GET", kwargs["url"])
        )

    client, calls = adapter(tmp_path / "ledger", monkeypatch, handler=rejected)
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    path = tmp_path / "manifest.json"
    runner.write_artifact(
        path, {**manifest(tmp_path), "slots": [dict(id="a"), dict(id="b")], "seconds": 300}
    )
    with pytest.raises(runner.ConfirmedAnnotationFailure):
        runner.run(path, tmp_path / "output")
    assert len(calls) == 1 and calls[0]["method"] == "GET"
    assert not (tmp_path / "ledger/phase-reservations.json").exists()
    report = runner.read(tmp_path / "output/report.json")
    assert report["status"] == "blocked_external"
    assert len(report["rows"]) == 2
    assert all(row["status"] == "not_attempted_provider_failure" for row in report["rows"])
    assert "mock-secret" not in (tmp_path / "output/authentication.json").read_text()
    proof = runner._read_bound(report["confirmed_failure"]["evidence"])
    assert proof["method"] == "GET" and proof["http_status"] == 401
    assert (
        runner.hashlib.sha256(proof["raw_response"].encode()).hexdigest()
        == proof["response_sha256"]
    )


def _approved_replacement_fixture(tmp_path, monkeypatch):
    import httpx
    import exact.llm.routing

    old = manifest(tmp_path)
    p = packet()
    packet_path = tmp_path / "packet.json"
    runner.write_artifact(packet_path, p.to_dict())
    old["slots"] = [
        dict(id=f"original-{i}", packet=dict(path=str(packet_path), sha256=runner.sha(packet_path)))
        for i in range(32)
    ]
    old["seconds"] = 300
    old_manifest_path = tmp_path / "original-manifest.json"
    runner.write_artifact(old_manifest_path, old)

    def transport(**kwargs):
        if kwargs["method"] == "GET":
            return httpx.Response(
                200, json={"data": {}}, request=httpx.Request("GET", kwargs["url"])
            )
        if sum(call["method"] == "POST" for call in calls) == 1:
            return httpx.Response(
                401, json={"error": {"code": 401}}, request=httpx.Request("POST", kwargs["url"])
            )
        return response(judgment(p))

    client, calls = adapter(tmp_path / "ledger", monkeypatch, handler=transport)
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    old_output = tmp_path / "old-output"
    with pytest.raises(runner.ConfirmedAnnotationFailure):
        runner.annotate_packet(
            p, old, old_output, slot_id="original-0", comparison_id="original-0", seconds=300
        )
    for index in range(1, 32):
        runner._reserve_phase(
            old, f"original-{index}", p.content_hash, 0.012, comparison_id=f"original-{index}"
        )
    prior = runner.read(tmp_path / "ledger/phase-reservations.json")
    prior_path = tmp_path / "prior-state.json"
    runner.write_artifact(prior_path, prior)
    proposal = dict(
        original_request_limits=runner.PHASE_LIMITS,
        proposed_request_limits=runner.REALLOCATED_LIMITS,
    )
    proposal_path = tmp_path / "proposal.json"
    runner.write_artifact(proposal_path, proposal)
    approval = dict(
        schema="exact-repair/request-budget-amendment-authorization/v1",
        status="approved",
        approval_text="Approve the reallocation",
        proposal=dict(path=str(proposal_path), sha256=runner.sha(proposal_path)),
        **proposal,
        aggregate_max_requests=704,
        calibration_max_usd=2,
        campaign_max_usd=35,
        replacement_calibration_max_requests=32,
        prior_reserved_exposure_usd=sum(
            row["reserved_cost_usd"] for row in prior["reservations"].values()
        ),
        replacement_calibration_max_reserved_usd=0.50512,
    )
    approval_path = tmp_path / "approval.json"
    runner.write_artifact(approval_path, approval)
    reference = lambda path: dict(path=str(path), sha256=runner.sha(path))
    new = {
        **old,
        "request_limits": runner.REALLOCATED_LIMITS,
        "request_budget_amendment": dict(
            authorization=reference(approval_path),
            previous_phase_state=reference(prior_path),
            previous_manifest=reference(old_manifest_path),
        ),
        "slots": [
            dict(
                id="replacement-0",
                comparison_id="original-0",
                packet=reference(packet_path),
                replaces=dict(slot_id="original-0", receipt=reference(old_output / "receipt.json")),
            )
        ],
    }
    return new, prior, calls, approval_path


def test_approved_reallocation_retains_failed_attempts_and_replacement_wire_identity(
    tmp_path, monkeypatch
):
    settings, prior, calls, _ = _approved_replacement_fixture(tmp_path, monkeypatch)
    path = tmp_path / "replacement/manifest.json"
    runner.write_artifact(path, settings)
    output = tmp_path / "replacement-output"
    for _ in range(2):
        rows = runner.run(path, output)
        assert rows[0]["status"] == "complete"
        assert sum(call["method"] == "POST" for call in calls) == 2
        state = runner.read(tmp_path / "ledger/phase-reservations.json")
        assert state["request_limits"] == runner.REALLOCATED_LIMITS
        assert len(state["reservations"]) == 33
        assert all(
            state["reservations"][key] == value for key, value in prior["reservations"].items()
        )
        successor = state["reservations"][runner.canonical_hash(("calibration", "replacement-0"))]
        assert successor["replacement"]["failed_attempt"]["http_status"] == 401
        assert successor["comparison_id"] == "original-0"
    from exact.llm.ledger import RequestLedger

    with RequestLedger(tmp_path / "ledger")._transaction() as db:
        attempts = db.execute("SELECT request_id,state,status FROM attempts").fetchall()
    assert len(attempts) == len({row["request_id"] for row in attempts}) == 2
    assert {(row["state"], row["status"]) for row in attempts} == {
        ("rejected", 401),
        ("completed", 200),
    }
    for phase in ("train", "development", "test"):
        continued = {**settings, "phase": phase}
        assert (
            runner._reserve_phase(continued, phase + "-slot", packet().content_hash, 0.012) is None
        )
    state = runner.read(tmp_path / "ledger/phase-reservations.json")
    assert len(state["reservations"]) == 36
    assert all(state["reservations"][key] == value for key, value in prior["reservations"].items())


@pytest.mark.parametrize(
    "mutation", ["unapproved", "tampered", "unsupported", "reset", "wrong_comparison"]
)
def test_request_amendment_rejects_unapproved_tampered_reset_or_changed_support(
    tmp_path, monkeypatch, mutation
):
    settings, prior, calls, approval_path = _approved_replacement_fixture(tmp_path, monkeypatch)
    if mutation in {"unapproved", "tampered"}:
        approval = runner.read(approval_path)
        approval["status"] = "proposed_not_authorized"
        runner.write_artifact(approval_path, approval)
        if mutation == "unapproved":
            settings["request_budget_amendment"]["authorization"]["sha256"] = runner.sha(
                approval_path
            )
    elif mutation == "unsupported":
        settings["request_limits"] = {**runner.REALLOCATED_LIMITS, "calibration": 65}
    elif mutation == "reset":
        runner.write_artifact(
            tmp_path / "ledger/phase-reservations.json", {**prior, "reservations": {}}
        )
    else:
        settings["slots"][0]["comparison_id"] = "different"
    with pytest.raises(ValueError):
        runner._reserve_phase(
            settings,
            "replacement-0",
            packet().content_hash,
            0.012,
            comparison_id=settings["slots"][0]["comparison_id"],
        )
    assert sum(call["method"] == "POST" for call in calls) == 1


def _policy_fixture(tmp_path, monkeypatch):
    import httpx
    import exact.llm.routing

    def transport(**kwargs):
        if kwargs["method"] == "GET":
            return httpx.Response(
                200, json={"data": {}}, request=httpx.Request("GET", kwargs["url"])
            )
        model = json.loads(kwargs["content"])["model"]
        if model == "vendor/teacher":
            return httpx.Response(
                404,
                json={
                    "error": {
                        "code": 404,
                        "metadata": {
                            "failed_routing_step": "Filter by Guardrails",
                            "input_endpoint_count": 1,
                            "ineligibility_reasons": [
                                {
                                    "reason": "paid-model-training-violation-by-account",
                                    "endpoint_count": 1,
                                    "configure_url": "https://openrouter.ai/settings/privacy",
                                }
                            ],
                        },
                    }
                },
                request=httpx.Request("POST", kwargs["url"]),
            )
        return response(judgment(packet()), model=model)

    client, calls = adapter(tmp_path / "ledger", monkeypatch, handler=transport)
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    packet_path = tmp_path / "packet.json"
    runner.write_artifact(packet_path, packet().to_dict())
    settings = {
        **manifest(tmp_path),
        "seconds": 300,
        "prices_per_million": {name: dict(input=1, output=2) for name in ("teacher", "evaluator")},
        "slots": [
            dict(
                id=f"slot-{i}",
                profile=profile,
                packet=dict(path=str(packet_path), sha256=runner.sha(packet_path)),
            )
            for i, profile in enumerate(("teacher", "evaluator", "teacher", "evaluator"))
        ],
    }
    path = tmp_path / "initial/manifest.json"
    runner.write_artifact(path, settings)
    return settings, path, calls


def test_provider_policy_rejection_masks_only_its_profile_and_keeps_other_slots(
    tmp_path, monkeypatch
):
    _, path, calls = _policy_fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    for _ in range(2):
        rows = runner.run(path, output)
        assert [row["status"] for row in rows] == [
            "provider_error",
            "complete",
            "not_attempted_provider_policy",
            "complete",
        ]
        assert sum(call["method"] == "POST" for call in calls) == 3
        report = runner.read(output / "report.json")
        assert report["status"] == "complete"
        assert report["recorded"] == report["scheduled"] == 4
        assert set(report["provider_policy_deferrals"]) == {"teacher"}
        assert not (output / "slot-2").exists()
        state = runner.read(tmp_path / "ledger/phase-reservations.json")
        assert len(state["reservations"]) == 3


@pytest.mark.parametrize(
    "mutation", [None, "schedule", "wrong_slot", "tampered_receipt", "unqualified"]
)
def test_source_bound_policy_deferral_preserves_earlier_cost_and_never_repeats_failure(
    tmp_path, monkeypatch, mutation
):
    settings, path, calls = _policy_fixture(tmp_path, monkeypatch)
    old = tmp_path / "old-output"
    with pytest.raises(runner.ConfirmedAnnotationFailure):
        runner.annotate_packet(packet(), settings, old, slot_id="slot-0", seconds=300)
    prior = runner.read(tmp_path / "ledger/phase-reservations.json")
    reference = lambda value: dict(path=str(value), sha256=runner.sha(value))
    settings["provider_deferrals"] = {
        "teacher": dict(
            previous_manifest=reference(path),
            failed_slot="slot-0",
            receipt=reference(old / "receipt.json"),
        )
    }
    if mutation == "schedule":
        settings["slots"] = settings["slots"][:-1]
    elif mutation == "wrong_slot":
        settings["provider_deferrals"]["teacher"]["failed_slot"] = "slot-1"
    elif mutation == "tampered_receipt":
        runner.write_artifact(old / "receipt.json", {})
    elif mutation == "unqualified":
        with runner.sqlite3.connect(tmp_path / "ledger/requests.sqlite3") as db:
            db.execute("UPDATE attempts SET raw=?", (b'{"error":{"code":404}}',))
    next_path = tmp_path / "continued/manifest.json"
    runner.write_artifact(next_path, settings)
    output = tmp_path / "continued-output"
    if mutation:
        with pytest.raises(ValueError):
            runner.run(next_path, output)
        assert sum(call["method"] == "POST" for call in calls) == 1
    else:
        rows = runner.run(next_path, output)
        assert [row["status"] for row in rows] == [
            "provider_error",
            "complete",
            "not_attempted_provider_policy",
            "complete",
        ]
        assert sum(call["method"] == "POST" for call in calls) == 3
        after = runner.read(tmp_path / "ledger/phase-reservations.json")
        assert all(
            after["reservations"][key] == value for key, value in prior["reservations"].items()
        )
        assert len(after["reservations"]) == 3
        assert not (output / "slot-0").exists()
