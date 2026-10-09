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
