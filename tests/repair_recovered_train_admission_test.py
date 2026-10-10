from pathlib import Path

import pytest

from tests.repair_october_learning_annotation_runner_test import manifest
from tests.repair_semantic_fidelity_test import packet
from tools.repair import recovered_train_admission as admission
from tools.repair import train_packet_admission as entry
from tools.repair.historical_regression import binding
from tools.repair.recovered_train_packets import TASK, TASK_REVISION, clarify
from tools.repair.semantic_packet_encoding import REVISION, encoding_identity
from tools.repair.shared_release import immutable


def fixture(tmp_path, monkeypatch):
    def save(name, value):
        path = tmp_path / name
        immutable(path, value)
        return binding(path)

    monkeypatch.setattr(admission, "packet_admission", lambda *a: dict(status="eligible"))
    monkeypatch.setattr(admission, "request_profile", lambda *a: {})
    monkeypatch.setattr(admission, "reprofile", lambda raw, controls: raw)
    m = dict(manifest(tmp_path), phase="train", teacher_profile="teacher", qualified_teacher={},
             prompt_version="frozen", prompt_hash="frozen", rubric_version="frozen",
             criterion_weights={"meaning_retention": 1}, request_profiles={})
    amendment = save("amendment.json", dict(task_revision=TASK_REVISION, task=TASK,
                                            retry_prior_responses=False))
    raw = save("native/packet.json", clarify(packet()).to_dict())
    old = save("old.json", dict(pair=None, packet=None))
    original = dict(id="new", pair=dict(assignments=[[0], [1]]), packet=raw, swapped=False,
                    status="complete_native_packet", native_receipts=[],
                    prior_native_row=old, task_amendment=amendment)
    original_ref = save("native/row.json", original)
    row = admission.transform(original, original_ref, m, tmp_path / "encoded")
    plan = save("plan.json", dict(task_amendment=amendment))
    report = save("native/report.json", dict(plan=plan, rows=[original_ref]))
    batch = save("batch.json", dict(
        frozen_files={plan["path"]: plan["sha256"]},
        jobs=[dict(id="producer", commands=[["python", "-m", "tools.repair.recovered_train_packets", plan["path"], "work"]])]))
    step = dict(dispatch_nonce="nonce", step_id="14451.83")
    receipt = dict(
        completion=save("completion.json", dict(**step, status="complete", exit_code=0,
                                                  job_id="producer", batch=batch["path"],
                                                  work=str(tmp_path / "native"))),
        step=save("step.json", step),
        outputs=save("outputs.json", {Path(ref["path"]).name: ref["sha256"]
                                      for ref in [raw, original_ref, report]}),
        batch=batch, **step, expected_status="complete")
    phase = dict(reservations={})
    save("ledger/phase-reservations.json", phase)
    template = save("template.json", m)
    m.update(slots=[row], packet_admission=dict(
        schema=admission.SCHEMA, template=template, task_amendment=amendment,
        encoding=save("encoding.json", dict(revision=REVISION, encoding_identity=encoding_identity(),
            encoder=binding(Path(admission.__file__).with_name("semantic_packet_encoding.py")))),
        phase_before=save("phase.json", phase),
        budget=save("budget.json", dict(request_limits=m["request_limits"], costs_reset=False)),
        native_receipts=[receipt], rows=save("rows.json", dict(rows=[row]))))
    return m, save


def test_new_admission_routes_through_terminal_receipts_and_lossless_round_trip(tmp_path, monkeypatch):
    m, _ = fixture(tmp_path, monkeypatch)
    assert entry.validate_rows(m) == m["slots"]


@pytest.mark.parametrize("defect", ["order", "model", "task", "nonce", "prior_paid", "old_pair"])
def test_changed_or_previously_attempted_rows_cannot_enter_hosted_adapter(tmp_path, monkeypatch, defect):
    m, save = fixture(tmp_path, monkeypatch)
    if defect == "order":
        m["slots"][0]["swapped"] = True
        m["packet_admission"]["rows"] = save("changed-rows.json", dict(rows=m["slots"]))
    elif defect == "model":
        m["profile"] = "another"
    elif defect == "task":
        m["packet_admission"]["task_amendment"] = save("changed-task.json", dict(
            task_revision=TASK_REVISION, task="prefer A", retry_prior_responses=False))
    elif defect == "nonce":
        m["packet_admission"]["native_receipts"][0]["dispatch_nonce"] = "different"
    elif defect == "prior_paid":
        phase = dict(reservations={"old": dict(phase="train", slot="new")})
        from exact.repair.api import write_artifact
        write_artifact(tmp_path / "ledger/phase-reservations.json", phase)
        m["packet_admission"]["phase_before"] = save("paid-before.json", phase)
    else:
        # Even changing a bound historical file cannot quietly upgrade a slot.
        (tmp_path / "old.json").write_text('{"pair": {}, "packet": null}')
    with pytest.raises(ValueError):
        entry.validate_rows(m)
