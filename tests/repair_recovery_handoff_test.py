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
