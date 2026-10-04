"""Recovery registration uses real nonce-bound receipts and preserves its predecessor."""

import hashlib
import json

import pytest

from tools.repair.recovery import link_recovery


@pytest.mark.parametrize("queued", [False, True])
def test_recovery_links_once_and_rejects_live_parent(tmp_path, queued):
    old_receipt = tmp_path / "old-completion.json"
    old_receipt.write_text(json.dumps(dict(status="failed")))
    old = dict(
        id="old",
        logical_id="job",
        step_id="12.1",
        completion_path=str(old_receipt),
        pending_recovery="new",
        enabled=True,
    )
    new = dict(
        id="new",
        logical_id="job",
        step_id="12.2",
        status_path=str(tmp_path / "status.json"),
        recovery_of="old",
        repair_attempt=1,
        max_repairs=2,
        dispatch_nonce="nonce",
        recovery_completion_sha256=hashlib.sha256(old_receipt.read_bytes()).hexdigest(),
    )
    (tmp_path / "step.json").write_text(json.dumps(dict(step_id="12.2", dispatch_nonce="nonce")))
    registry = tmp_path / "registry.json"
    worker = tmp_path / "worker.sh"
    worker.write_text("#!/bin/bash\nexit 0\n")
    launch = dict(
        argv=["/usr/bin/srun", "--jobid=12", "/bin/bash", str(worker)],
        run=new,
        nonce="nonce0000000000",
        step_path=str(tmp_path / "step.json"),
        launcher_log=str(tmp_path / "launcher.log"),
        tmux_socket=str(tmp_path / "tmux.sock"),
        bindings=[dict(path=str(worker), sha256=hashlib.sha256(worker.read_bytes()).hexdigest())],
    )
    if queued:
        new.update(
            exit_path=str(tmp_path / "exit_code"), completion_path=str(tmp_path / "completion.json")
        )
        (tmp_path / "step.json").write_text(
            json.dumps(dict(step_id="12.2", dispatch_nonce=launch["nonce"]))
        )
    registry.write_text(
        json.dumps(
            dict(
                runs=[old] if queued else [old, new],
                pending_batches=[dict(id="new", launch=launch)] if queued else [],
            )
        )
    )
    before = registry.read_bytes()
    with pytest.raises(ValueError, match="Prior worker is live"):
        link_recovery(tmp_path, "new", steps={"12.1": "RUNNING", "12.2": "RUNNING"}, step_id="12.2")
    assert registry.read_bytes() == before
    for _ in range(2):
        result = link_recovery(tmp_path, "new", steps={"12.2": "RUNNING"}, step_id="12.2")
        assert result["status"] == "linked"
    rows = json.loads(registry.read_text())["runs"]
    assert len(rows) == 2 and rows[0]["superseded_by"] == "new" and not rows[0]["enabled"]
    assert rows[1]["repair_attempt"] == 1
    (tmp_path / "step.json").write_text(json.dumps(dict(step_id="12.2", dispatch_nonce="wrong")))
    with pytest.raises(ValueError, match="nonce"):
        link_recovery(tmp_path, "new", steps={"12.2": "RUNNING"}, step_id="12.2")


def scientific_recovery(tmp_path, status="generation_error"):
    from exact.experiments.science_health import inspect_science
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    work = tmp_path / "old-work"
    payload = put(work / "result.json", dict(row_id="row", status=status, detail="ValueError: bad edge"))
    row = put(work / "row.json", dict(row=dict(id="row"), status="complete", result=payload))
    report = put(work / "evaluation/report.json", dict(schema="exact-repair/fresh-evaluation/v1",
                 status="complete", scheduled=1, recorded=1, rows=[dict(row, row_id="row")]))
    complete = dict(status="complete", step_id="12.1", dispatch_nonce="old-nonce", work=str(work))
    completion = put(tmp_path / "old-attempt/completion.json", complete)
    put(tmp_path / "old-attempt/outputs.json", {"evaluation/report.json": report["sha256"]})
    old = dict(id="old", logical_id="job", step_id="12.1", dispatch_nonce="old-nonce",
               completion_path=completion["path"], science_report_relative="evaluation/report.json",
               pending_recovery="new", enabled=True)
    failures = inspect_science(old, complete)["failures"]
    new = dict(id="new", logical_id="job", step_id="12.2", recovery_of="old",
               status_path=str(tmp_path / "new-attempt/status.json"), dispatch_nonce="new-nonce",
               repair_attempt=1, max_repairs=2, recovery_completion_sha256=completion["sha256"])
    put(tmp_path / "new-attempt/step.json", dict(step_id="12.2", dispatch_nonce="new-nonce"))
    registry = dict(runs=[old, new], pending_batches=[])
    put(tmp_path / "registry.json", registry)
    return registry, failures


def test_completed_parent_requires_explicit_qualified_scientific_failure(tmp_path):
    registry, failures = scientific_recovery(tmp_path)
    before = (tmp_path / "registry.json").read_bytes()
    with pytest.raises(ValueError, match="no matching qualified scientific failure"):
        link_recovery(tmp_path, "new", steps={"12.2": "RUNNING"}, step_id="12.2")
    assert (tmp_path / "registry.json").read_bytes() == before
    registry["runs"][1]["recovery_scientific_failure"] = failures[0]
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    for _ in range(2):
        result = link_recovery(tmp_path, "new", steps={"12.2": "RUNNING"}, step_id="12.2")
        assert result["status"] == "linked"
    saved = json.loads((tmp_path / "registry.json").read_text())
    assert saved["runs"][0]["superseded_by"] == "new"
    assert saved["runs"][1]["recovery_scientific_failure"] == failures[0]


@pytest.mark.parametrize("tamper", ["payload", "row_ids", "healthy"])
def test_scientific_recovery_rejects_unqualified_or_changed_evidence(tmp_path, tamper):
    registry, failures = scientific_recovery(tmp_path, "evaluated" if tamper == "healthy" else "generation_error")
    expected = failures[0] if failures else dict(signature=dict(status="generation_error", detail="claimed"), row_ids=["row"], evidence={})
    registry["runs"][1]["recovery_scientific_failure"] = expected
    if tamper == "payload":
        (tmp_path / "old-work/result.json").write_text("{}")
    elif tamper == "row_ids":
        expected["row_ids"] = ["different-row"]
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    before = (tmp_path / "registry.json").read_bytes()
    with pytest.raises(ValueError, match="qualified scientific failure"):
        link_recovery(tmp_path, "new", steps={"12.2": "RUNNING"}, step_id="12.2")
    assert (tmp_path / "registry.json").read_bytes() == before


@pytest.mark.parametrize("tamper", [None, "cause", "budget", "completion", "repair_attempt"])
def test_operational_continuation_does_not_consume_software_repair_attempt(tmp_path, tamper):
    from tools.repair.expanded_corpus import binding
    from tools.repair.scaling_endpoint_continuation import CAUSE, SCHEMA

    registry, _ = scientific_recovery(tmp_path)
    old, new = registry["runs"]
    receipt = tmp_path / "old-attempt/completion.json"
    receipt.write_text(json.dumps(dict(status="failed", error=CAUSE if tamper != "cause" else {"type": "ValueError"})))
    plan = dict(schema=SCHEMA, recovery_kind="operational_continuation", original_step=old["step_id"],
                completion=binding(receipt), prior_costs_reset=False, scientific_budgets_changed=False)
    if tamper == "budget":
        plan["scientific_budgets_changed"] = True
    if tamper == "completion":
        plan["completion"]["sha256"] = "wrong"
    path = tmp_path / "continuation.json"
    path.write_text(json.dumps(plan))
    new.update(recovery_kind="operational_continuation", continuation_attempt=1,
               repair_attempt=0 if tamper != "repair_attempt" else 1,
               recovery_completion_sha256=binding(receipt)["sha256"], continuation_plan=binding(path))
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    before = (tmp_path / "registry.json").read_bytes()
    if tamper:
        with pytest.raises(ValueError, match="Operational continuation"):
            link_recovery(tmp_path, "new", steps={"12.2": "RUNNING"}, step_id="12.2")
        assert (tmp_path / "registry.json").read_bytes() == before
    else:
        link_recovery(tmp_path, "new", steps={"12.2": "RUNNING"}, step_id="12.2")
        saved = json.loads((tmp_path / "registry.json").read_text())
        assert saved["runs"][0]["superseded_by"] == "new"
        assert saved["runs"][1]["repair_attempt"] == 0
