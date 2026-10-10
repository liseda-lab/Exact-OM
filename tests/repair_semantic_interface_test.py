"""Versioned request correction; strict validation and historical replay stay intact."""

import dataclasses
import json

import pytest

from exact.repair.records import canonical_hash, canonical_json
from exact.repair.semantic_fidelity import (
    INTERFACE_PROMPT,
    INTERFACE_PROMPT_VERSION,
    PROMPT,
    PROMPT_VERSION,
    TEACHER,
    SemanticAnnotationAdapter,
    _parameter_context,
    validate_comparison,
)
from tests.repair_october_learning_annotation_runner_test import manifest
from tests.repair_semantic_fidelity_test import adapter, judgment, packet, run
from tools.repair import corrective_semantics as runner


def revised(settings):
    return {
        **settings,
        "prompt_version": INTERFACE_PROMPT_VERSION,
        "prompt_hash": canonical_hash(INTERFACE_PROMPT),
    }


def test_both_prompt_versions_replay_and_revision_has_distinct_request(tmp_path, monkeypatch):
    old, calls = adapter(tmp_path, monkeypatch)
    legacy = old.annotate(packet(), role=TEACHER)
    modern = SemanticAnnotationAdapter(
        old.router, dataclasses.replace(run(), prompt_version=INTERFACE_PROMPT_VERSION), tmp_path
    )
    fixed = modern.annotate(packet(), role=TEACHER)
    assert len(calls) == 2
    assert legacy.annotator["prompt_hash"] == canonical_hash(PROMPT)
    assert fixed.annotator["prompt_hash"] == canonical_hash(INTERFACE_PROMPT)
    assert (
        legacy.annotator["wire_receipt"]["request_id"]
        != fixed.annotator["wire_receipt"]["request_id"]
    )
    for client, result, version in [
        (old, legacy, PROMPT_VERSION),
        (modern, fixed, INTERFACE_PROMPT_VERSION),
    ]:
        assert client.annotate(packet(), role=TEACHER) == result
        assert (
            _parameter_context(result.annotator["request_parameters"])["prompt_version"] == version
        )
    assert len(calls) == 2
    forged = json.loads(canonical_json(fixed.annotator["request_parameters"]))
    forged["messages"][0]["content"] = PROMPT
    with pytest.raises(ValueError, match="prompt identity"):
        _parameter_context(forged)


def test_entrypoint_binds_explicit_prompt_and_preserves_wire_limits(tmp_path, monkeypatch):
    import exact.llm.routing

    client, calls = adapter(tmp_path / "ledger", monkeypatch)
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    settings = revised(manifest(tmp_path))
    output = tmp_path / "annotation"
    result = runner.annotate_packet(packet(), settings, output, slot_id="s", seconds=300)
    assert result is not None and len(calls) == 1
    payload = json.loads(calls[0]["content"])
    assert payload["messages"][0]["content"] == INTERFACE_PROMPT
    assert payload["max_tokens"] == 2000 and payload["temperature"] == 0
    assert payload["provider"]["allow_fallbacks"] is False
    assert "reasoning" not in payload and "response_format" not in payload
    assert runner.read(output / "run.json")["record"]["prompt_version"] == INTERFACE_PROMPT_VERSION
    assert runner.annotate_packet(packet(), settings, output, slot_id="s", seconds=1) == result
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["hash", "version", "missing_hash", "missing_version"])
def test_bad_manifest_prompt_rejected_before_reservation(tmp_path, mutation):
    settings = revised(manifest(tmp_path))
    if mutation == "hash":
        settings["prompt_hash"] = canonical_hash(PROMPT)
    elif mutation == "version":
        settings["prompt_version"] = "unregistered"
    else:
        del settings["prompt_hash" if mutation == "missing_hash" else "prompt_version"]
    with pytest.raises(ValueError, match="prompt"):
        runner.annotate_packet(packet(), settings, tmp_path / "out", slot_id="s", seconds=300)
    assert not (tmp_path / "ledger").exists()


@pytest.mark.parametrize(
    "defect", ["fences", "prose", "boolean", "definition", "query_path", "quote"]
)
def test_interface_correction_does_not_weaken_validator(defect):
    p = packet()
    value = judgment(p)
    criterion = value["criteria"][0]
    if defect == "prose":
        criterion["symbolic_claims"][0]["value"] = "the query is entailed"
    elif defect == "boolean":
        criterion["symbolic_claims"][0]["value"] = True
    elif defect == "definition":
        criterion["symbolic_claims"][0]["evidence_id"] = "D1"
    elif defect == "query_path":
        criterion["evidence_ids"].append("plan_a.query_outcomes.q1")
    elif defect == "quote":
        criterion["quotes"][0]["span"] = "invented quotation"
    raw = json.dumps(value)
    if defect == "fences":
        raw = "```json\n" + raw + "\n```"
    with pytest.raises(ValueError):
        validate_comparison(raw, p)


def test_definition_only_semantic_judgment_needs_no_invented_boolean():
    p = packet()
    value = judgment(p)
    for criterion in value["criteria"]:
        criterion["evidence_ids"] = ["D1", "D2"]
        criterion["symbolic_claims"] = []
    assert validate_comparison(json.dumps(value), p).global_target_eligible


def recovery_fixture(tmp_path, monkeypatch, *, valid=False):
    """A definitive schema failure, with its actual mock wire receipt and old budget."""
    from exact.repair.api import write_artifact
    from tools.repair.batch import sha

    client, _ = adapter(tmp_path / "ledger", monkeypatch)
    observation = client.annotate(packet(), role=TEACHER)
    wire = json.loads(canonical_json(observation.annotator["wire_receipt"]))
    response = json.loads(wire["raw_response"])
    if not valid:
        response["choices"][0]["message"]["content"] = (
            "```json\n" + json.dumps(judgment(packet())) + "\n```"
        )
    wire["raw_response"] = json.dumps(response)
    import hashlib

    wire["sha256"] = hashlib.sha256(wire["raw_response"].encode()).hexdigest()
    schedule = client.schedule(packet(), role=TEACHER, quorum=1, repetitions=(0,))
    settings = revised(manifest(tmp_path))
    settings["profiles"]["teacher"]["provider"] = json.loads(
        canonical_json(schedule.slots[0]["parameters"]["provider"])
    )
    settings["request_limits"] = runner.REALLOCATED_LIMITS
    settings["slots"] = [
        dict(id="new", comparison_id="comparison", profile="teacher", swapped=False)
    ]
    old_row = dict(
        phase="calibration",
        slot="old",
        packet_hash=packet().content_hash,
        comparison_id="comparison",
        reserved_cost_usd=0.024,
        state="unavailable",
    )
    state = dict(
        lineage=settings["lineage_id"],
        cost_ceiling_usd=35,
        request_limits=runner.REALLOCATED_LIMITS,
        reservations={canonical_hash(("calibration", "old")): old_row},
    )

    def save(name, value):
        p = tmp_path / name
        write_artifact(p, value)
        return dict(path=str(p), sha256=sha(p))

    prior = save("prior.json", state)
    write_artifact(tmp_path / "ledger/phase-reservations.json", state)
    proposal = dict(
        schema="exact-repair/annotation-interface-recovery-proposal/v1",
        profile="teacher",
        request_contract_hash=canonical_hash(
            {k: v for k, v in settings.items() if k != "authorized"}
        ),
        max_requests=1,
        max_reserved_usd=0.024,
        time={"latest_full_panel_admission_epoch": runner.time.time() + 900},
        previous_phase_state=prior,
        slot_lineage={
            "new": dict(
                slot_id="old",
                receipt=save("receipt.json", {"status": "annotation_unavailable"}),
                schedule=save("schedule.json", schedule.to_dict()),
                wire_receipt=save("wire.json", wire),
            )
        },
    )
    bound = save("proposal.json", proposal)
    approval = save(
        "approval.json",
        dict(
            schema="exact-repair/annotation-interface-recovery-authorization/v1",
            status="approved",
            approval_text="fixture approval; mock HTTP only",
            proposal=bound,
        ),
    )
    settings["interface_recovery"] = dict(proposal=bound, authorization=approval)
    monkeypatch.setattr(runner, "_request_budget", lambda _: (runner.REALLOCATED_LIMITS, None))
    return settings, state


def test_recovery_is_one_explicitly_approved_retry_with_prior_costs(tmp_path, monkeypatch):
    settings, before = recovery_fixture(tmp_path, monkeypatch)
    assert (
        runner._reserve_phase(
            settings, "new", packet().content_hash, 0.024, comparison_id="comparison"
        )
        is None
    )
    after = runner.read(tmp_path / "ledger/phase-reservations.json")
    assert len(after["reservations"]) == 2
    assert all(after["reservations"][key] == value for key, value in before["reservations"].items())
    assert sum(row["reserved_cost_usd"] for row in after["reservations"].values()) == 0.048
    new = after["reservations"][canonical_hash(("calibration", "new"))]
    assert new["interface_recovery"]["technical_attempt"] == 2
    assert (
        runner._reserve_phase(
            settings, "new", packet().content_hash, 0.024, comparison_id="comparison"
        )
        is False
    )


@pytest.mark.parametrize(
    "defect", ["approval", "settings", "cost", "prior", "packet", "comparison"]
)
def test_recovery_rejects_unapproved_or_incompatible_work_before_reservation(
    tmp_path, monkeypatch, defect
):
    settings, before = recovery_fixture(tmp_path, monkeypatch)
    packet_hash, comparison_id, cost = packet().content_hash, "comparison", 0.024
    if defect == "approval":
        settings["interface_recovery"]["authorization"] = None
    elif defect == "settings":
        settings["profiles"]["teacher"]["model"] = "unapproved/model"
    elif defect == "cost":
        cost = 0.025
    elif defect == "prior":
        state = runner.read(tmp_path / "ledger/phase-reservations.json")
        state["reservations"] = {}
        runner.write_artifact(tmp_path / "ledger/phase-reservations.json", state)
    elif defect == "packet":
        packet_hash = "changed"
    else:
        comparison_id = "new-comparison"
    snapshot = (tmp_path / "ledger/phase-reservations.json").read_bytes()
    with pytest.raises((ValueError, PermissionError)):
        runner._reserve_phase(settings, "new", packet_hash, cost, comparison_id=comparison_id)
    assert (tmp_path / "ledger/phase-reservations.json").read_bytes() == snapshot


def test_valid_scientific_result_is_not_a_schema_failure_to_retry(tmp_path, monkeypatch):
    settings, _ = recovery_fixture(tmp_path, monkeypatch, valid=True)
    with pytest.raises(ValueError, match="valid scientific judgment"):
        runner._reserve_phase(
            settings, "new", packet().content_hash, 0.024, comparison_id="comparison"
        )


def test_expired_whole_panel_admission_spends_nothing(tmp_path, monkeypatch):
    settings, _ = recovery_fixture(tmp_path, monkeypatch)
    snapshot = (tmp_path / "ledger/phase-reservations.json").read_bytes()
    monkeypatch.setattr(runner.time, "time", lambda: 1e20)
    with pytest.raises(runner.AnnotationBudgetExhausted, match="no longer fits stage"):
        runner._reserve_phase(
            settings, "new", packet().content_hash, 0.024, comparison_id="comparison"
        )
    assert (tmp_path / "ledger/phase-reservations.json").read_bytes() == snapshot
