"""Closed rows and exact transport proofs gate the hosted entry point."""

import copy
from pathlib import Path

import pytest

from exact.repair.records import canonical_hash
from tests.repair_october_learning_annotation_runner_test import manifest
from tests.repair_semantic_fidelity_test import packet
from tools.repair import corrective_semantics as runner
from tools.repair import train_packet_admission as admission
from tools.repair.historical_regression import binding
from tools.repair.shared_release import immutable
from tools.repair.semantic_packet_encoding import REVISION, encoding_identity


def setup(tmp_path, monkeypatch, eligible=True):
    def save(name, value):
        path = tmp_path / name
        immutable(path, value)
        return binding(path)

    monkeypatch.setattr(admission, "packet_admission", lambda *a: dict(status="eligible"))
    raw = save("native/packet.json", packet().to_dict())
    source = save("native/evidence.json", dict(authored=True))
    missing = dict(
        id="missing",
        packet=None,
        swapped=False,
        status="unavailable_native_plan",
        primary_weak_label_eligible=False,
        native_receipts=[],
        evidence_contract=source,
    )
    available = dict(missing, id="fixed", packet=raw, status="unavailable_packet_bytes")
    originals = [missing, available] if eligible else [missing]
    original_refs = [save("native/" + r["id"] + ".json", r) for r in originals]
    rows = [
        admission.transform(r, ref, {}, tmp_path / "encoded")
        for r, ref in zip(originals, original_refs)
    ]
    outputs = {Path(ref["path"]).name: ref["sha256"] for ref in [raw, source, *original_refs]}
    batch = save("batch.json", dict(jobs=[dict(id="prepare-grounded-train-packets-001")]))
    step = dict(dispatch_nonce="nonce", step_id="14451.63")
    receipt = dict(
        completion=save(
            "completion.json",
            dict(
                **step,
                status="complete",
                exit_code=0,
                job_id="prepare-grounded-train-packets-001",
                batch=batch["path"],
                work=str(tmp_path / "native"),
            ),
        ),
        step=save("step.json", step),
        outputs=save("outputs.json", outputs),
        batch=batch,
        **step,
        expected_status="complete",
    )
    m = dict(manifest(tmp_path), phase="train", seconds=300, slots=rows)
    m["calibration_gate"] = save("gate.json", dict(status="qualified"))
    m["packet_admission"] = dict(
        schema=admission.SCHEMA,
        encoding=save(
            "encoding.json",
            dict(
                revision=REVISION,
                encoding_identity=encoding_identity(),
                encoder=binding(Path(admission.__file__).with_name("semantic_packet_encoding.py")),
            ),
        ),
        budget=save("budget.json", dict(request_limits=m["request_limits"], costs_reset=False)),
        native_receipt=save("native-receipt.json", receipt),
        rows=save("rows.json", dict(rows=rows)),
    )
    return m


def test_unavailable_only_entry_point_never_opens_network_or_reserves(tmp_path, monkeypatch):
    m = setup(tmp_path, monkeypatch, eligible=False)
    path = tmp_path / "manifest.json"
    runner.write_artifact(path, m)
    monkeypatch.setattr(runner, "_authentication_preflight", lambda *a: pytest.fail("network"))
    monkeypatch.setattr(runner, "annotate_packet", lambda *a, **k: pytest.fail("hosted call"))
    rows = runner.run(path, tmp_path / "result")
    assert rows[0]["status"] == "unavailable_native_plan"
    assert rows[0]["attempted"] is False
    assert runner.read(tmp_path / "result/report.json")["status"] == "complete"
    assert not (tmp_path / "ledger").exists()
    assert runner.run(path, tmp_path / "result") == rows


def test_mixed_entry_point_keeps_denominator_and_sends_only_exact_eligible_packet(
    tmp_path, monkeypatch
):
    m = setup(tmp_path, monkeypatch)
    path = tmp_path / "manifest.json"
    runner.write_artifact(path, m)
    calls = []
    monkeypatch.setattr(runner, "_authentication_preflight", lambda *a: None)

    def annotate(p, settings, output, **kw):
        calls.append((p, kw["slot_id"]))
        assert p == runner.read_record(runner.read(m["slots"][1]["packet"]["path"]))
        return Path(output) / "labels.json"

    monkeypatch.setattr(runner, "annotate_packet", annotate)
    rows = runner.run(path, tmp_path / "result")
    assert len(rows) == 2 and [r["status"] for r in rows] == ["unavailable_native_plan", "complete"]
    assert [c[1] for c in calls] == ["fixed"]


@pytest.mark.parametrize("defect", ["status", "order", "packet", "proof", "nonce"])
def test_tamper_rejected_before_account_check(tmp_path, monkeypatch, defect):
    m = setup(tmp_path, monkeypatch)
    if defect == "status":
        m["slots"][0]["status"] = "eligible"
    elif defect == "order":
        m["slots"][1]["swapped"] = True
    elif defect == "packet":
        Path(m["slots"][1]["packet"]["path"]).write_text("{}")
    elif defect == "proof":
        Path(m["slots"][1]["transport_proof"]["path"]).write_text("{}")
    else:
        Path(m["packet_admission"]["native_receipt"]["path"]).write_text("{}")
    path = tmp_path / "manifest.json"
    runner.write_artifact(path, m)
    monkeypatch.setattr(runner, "_authentication_preflight", lambda *a: pytest.fail("network"))
    with pytest.raises(ValueError):
        runner.run(path, tmp_path / "result")


def test_provider_failure_retains_later_scientifically_unavailable_rows(tmp_path, monkeypatch):
    m = setup(tmp_path, monkeypatch)
    m["slots"].reverse()
    runner.write_artifact(tmp_path / "reverse.json", dict(rows=m["slots"]))
    m["packet_admission"]["rows"] = binding(tmp_path / "reverse.json")
    runner.write_artifact(tmp_path / "manifest.json", m)
    monkeypatch.setattr(runner, "_authentication_preflight", lambda *a: None)

    def fail(*a, **kw):
        raise runner.ConfirmedAnnotationFailure(dict(http_status=503, kind="provider_response"))

    monkeypatch.setattr(runner, "annotate_packet", fail)
    with pytest.raises(runner.ConfirmedAnnotationFailure):
        runner.run(tmp_path / "manifest.json", tmp_path / "result")
    report = runner.read(tmp_path / "result/report.json")
    assert [r["status"] for r in report["rows"]] == ["provider_error", "unavailable_native_plan"]


def test_unknown_delivery_consumes_service_and_resume_never_replenishes(tmp_path):
    m = dict(manifest(tmp_path), phase="train")
    state = dict(
        schema="exact-repair/annotation-phase-budget/v1",
        lineage=m["lineage_id"],
        cost_ceiling_usd=35,
        request_limits=runner.PHASE_LIMITS,
        reservations={},
    )
    for i in range(320):
        state["reservations"][str(i)] = dict(
            phase="train",
            slot=str(i),
            comparison_id=str(i % 256),
            packet_hash="p",
            reserved_cost_usd=0.001,
            state="unresolved",
        )
    runner.write_artifact(tmp_path / "ledger/phase-reservations.json", state)
    with pytest.raises(runner.AnnotationBudgetExhausted, match="service"):
        runner._reserve_phase(m, "next", "p", 0.001, comparison_id="0")
    assert runner.read(tmp_path / "ledger/phase-reservations.json") == state


def test_row_cannot_change_scientific_identity_even_with_fresh_container_hash(
    tmp_path, monkeypatch
):
    m = setup(tmp_path, monkeypatch)
    row = copy.deepcopy(m["slots"][1])
    original = runner.read(row["original_row"]["path"])
    row["swapped"] = True
    with pytest.raises(ValueError, match="native identity"):
        admission.validate_row(row, m, original)
    assert canonical_hash(original) == canonical_hash(runner.read(row["original_row"]["path"]))


def test_run_digest_tokenization_headroom_is_reserved(monkeypatch):
    from tools.repair import grounded_supervision as grounded

    monkeypatch.setattr(grounded, "request_profile", lambda *a: dict(reasoning=None, tokenizer={}))
    monkeypatch.setattr(grounded, "input_token_bound", lambda *a: 7980)
    result = grounded.packet_admission(
        packet(),
        dict(
            profile="teacher", prompt_version="semantic-fidelity-prompt/v3.2", lineage_id="fixture"
        ),
        False,
    )
    assert result["status"] == "unavailable_packet_tokens"
    assert result["input_token_bound"] == 8044
