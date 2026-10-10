"""Input amendments preserve teacher provenance and reach the actual request path."""

from dataclasses import replace

import pytest

from exact.repair.annotation_controls import input_token_bound
from exact.repair.records import canonical_hash
from tests.repair_annotation_controls_test import profile_fixture, save, wire_client
from tests.repair_semantic_fidelity_test import packet
from tools.repair import corrective_semantics as runner
from tools.repair.annotation_profile import request_profile
from tools.repair.grounded_supervision import packet_admission
from tools.repair.train_input_recovery import reprofile


def amended(tmp_path):
    m = profile_fixture(tmp_path)
    controls = m["request_profiles"]["teacher"]
    data = runner.read(controls["endpoint"]["path"])
    data["data"]["endpoints"][0].update(context_length=32000, max_prompt_tokens=30000)
    controls["endpoint"] = save(tmp_path, "endpoint.json", data)
    amendment = dict(
        schema="exact-repair/annotation-input-amendment/v1",
        authorized=True,
        costs_reset=False,
        truncate_evidence=False,
        model="vendor/teacher",
        provider="pinned",
        max_input_tokens=24576,
        max_input_bytes=65536,
        max_output_tokens=6000,
        phases=[m["phase"]],
        cost_ceiling_usd=m["cost_ceiling_usd"],
        request_limits=m["request_limits"],
        previous_profile_hash=canonical_hash(controls),
    )
    controls.update(
        max_input_tokens=24576,
        max_input_bytes=65536,
        input_amendment=save(tmp_path, "amendment.json", amendment),
    )
    return m


def test_long_complete_packet_is_admitted_only_with_supported_frozen_profile(tmp_path):
    m = amended(tmp_path)
    controls = request_profile(m, "teacher")
    p = packet()
    p = replace(p, original_observation=("full retained axiom context " * 1400,))
    raw = p.to_dict()
    revised = reprofile(p, controls)
    assert {k: v for k, v in revised.to_dict().items() if k != "hash"} != raw
    assert replace(revised, coverage=p.coverage) == p
    result = packet_admission(revised, m, False)
    assert 32768 < result["input_bytes"] < 65536 and result["status"] == "eligible"
    with pytest.raises(ValueError, match="byte cap"):
        input_token_bound([dict(role="user", content="x" * 40000)], {}, controls["tokenizer"])
    assert (
        input_token_bound(
            [dict(role="user", content="x" * 40000)], {}, controls["tokenizer"], 65536
        )
        > 0
    )


@pytest.mark.parametrize(
    "defect", ["context", "prompt", "cap", "price", "tamper", "missing_amendment"]
)
def test_invalid_amendment_rejected_before_any_ledger_or_request(tmp_path, defect):
    m = amended(tmp_path)
    controls = m["request_profiles"]["teacher"]
    if defect in {"context", "prompt"}:
        d = runner.read(controls["endpoint"]["path"])
        d["data"]["endpoints"][0][
            "context_length" if defect == "context" else "max_prompt_tokens"
        ] = 10000
        controls["endpoint"] = save(tmp_path, "endpoint.json", d)
    elif defect == "cap":
        controls["max_input_tokens"] = 30000
    elif defect == "price":
        m["cost_ceiling_usd"] = 36
    elif defect == "tamper":
        controls["input_amendment"]["sha256"] = "0" * 64
    else:
        del controls["input_amendment"]
    with pytest.raises(ValueError):
        request_profile(m, "teacher")
    assert not (tmp_path / "ledger").exists()


def test_actual_entrypoint_reserves_amended_input_cost_and_preserves_exact_resume(
    tmp_path, monkeypatch
):
    import exact.llm.routing

    client, calls = wire_client(tmp_path / "ledger", monkeypatch, {"effort": "low"})
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    m = amended(tmp_path)
    m["teacher_panel"] = {"test_only": True}
    monkeypatch.setattr(runner, "_request_budget", lambda m: (runner.PHASE_LIMITS, None))
    result = runner.annotate_packet(packet(), m, tmp_path / "out", slot_id="s", seconds=300)
    assert result and len(calls) == 1
    state = runner.read(tmp_path / "ledger/phase-reservations.json")
    assert next(iter(state["reservations"].values()))["reserved_cost_usd"] == pytest.approx(
        0.036576
    )
    run = runner.read_record(runner.read(tmp_path / "out/run.json"))
    assert run.max_input_bytes == 65536
    assert run.aggregate_budget.tokens == 704 * 30576
    assert runner.annotate_packet(packet(), m, tmp_path / "out", slot_id="s", seconds=1) == result
    assert len(calls) == 1


def test_actual_entrypoint_sends_full_packet_above_legacy_byte_cap(tmp_path, monkeypatch):
    import exact.llm.routing
    import json
    from tests.repair_semantic_fidelity_test import judgment, response

    client, calls = wire_client(tmp_path / "ledger", monkeypatch, {"effort": "low"})
    m = amended(tmp_path)
    m["teacher_panel"] = {"test_only": True}
    p = reprofile(
        replace(packet(), original_observation=("full retained axiom context " * 1400,)),
        m["request_profiles"]["teacher"],
    )

    def transport(**kwargs):
        calls.append(kwargs)
        return response(judgment(p))

    monkeypatch.setattr(client.router.hosted._client, "request", transport)
    monkeypatch.setattr(exact.llm.routing, "LLMRouter", lambda *a, **k: client.router)
    monkeypatch.setattr(runner, "_request_budget", lambda m: (runner.PHASE_LIMITS, None))
    result = runner.annotate_packet(p, m, tmp_path / "out", slot_id="long", seconds=300)
    assert result and len(calls) == 1
    payload = json.loads(calls[0]["content"])
    assert len(payload["messages"][1]["content"].encode()) > 32768
    context = json.loads(payload["messages"][1]["content"])
    assert context["packet"] == json.loads(json.dumps(p.judge_payload()))


def test_recovery_rejects_changed_plan_split_and_missing_original(tmp_path, monkeypatch):
    from tools.repair import train_input_recovery as recovery
    from tools.repair.semantic_packet_encoding import encode

    m = amended(tmp_path)
    m["phase"] = "train"
    control = m["request_profiles"]["teacher"]
    amendment = runner.read(control["input_amendment"]["path"])
    amendment["phases"] = ["train"]
    control["input_amendment"] = save(tmp_path, "train-amendment.json", amendment)
    native = packet()
    m.update(rubric_version=native.rubric_version, criterion_weights=dict(native.criterion_weights))
    previous = dict(
        id="untransmitted",
        original_packet=save(tmp_path, "native.json", native.to_dict()),
        swapped=False,
        status="unavailable_packet_bytes",
        primary_weak_label_eligible=False,
    )
    revised = reprofile(native, control)
    compact, proof = encode(revised)
    row = dict(
        previous,
        original_packet=save(tmp_path, "reprofiled.json", revised.to_dict()),
        packet=save(tmp_path, "encoded.json", compact.to_dict()),
        transport_proof=save(tmp_path, "proof.json", proof),
        **packet_admission(compact, m, False)
    )
    row["primary_weak_label_eligible"] = row["status"] == "eligible"
    phase = dict(reservations={})
    runner.write_artifact(tmp_path / "ledger/phase-reservations.json", phase)
    m.update(
        parent_splits={native.parent_group_id: "train"},
        slots=[row],
        packet_admission=dict(
            schema=recovery.SCHEMA,
            input_amendment=control["input_amendment"],
            phase_before=save(tmp_path, "phase.json", phase),
        ),
    )
    monkeypatch.setattr(recovery, "predecessor", lambda _: (m, [previous]))
    assert recovery.validate_rows(m) == [row]
    row["swapped"] = True
    with pytest.raises(ValueError, match="case, split"):
        recovery.validate_rows(m)
    row["swapped"] = False
    m["slots"] = []
    with pytest.raises(ValueError, match="denominator"):
        recovery.validate_rows(m)
    m["slots"] = [row]
    phase["reservations"]["old"] = dict(phase="train", slot=row["id"])
    m["packet_admission"]["phase_before"] = save(tmp_path, "paid-phase.json", phase)
    runner.write_artifact(tmp_path / "ledger/phase-reservations.json", phase)
    with pytest.raises(ValueError, match="already had"):
        recovery.validate_rows(m)
