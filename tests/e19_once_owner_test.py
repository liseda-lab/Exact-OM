"""Slurm worker ancestry differs from the verified local srun launcher."""

from types import SimpleNamespace

import pytest

from tools import resume_e19_once as once


@pytest.mark.parametrize(
    "kind,pid,host,step,user,allowed",
    [
        ("srun", 101, "node", "14372.30", "owner", True),
        ("srun", 102, "node", "14372.30", "owner", False),
        ("python", 101, "node", "14372.30", "owner", False),
        ("srun", 101, "remote", "14372.30", "owner", False),
        ("srun", 101, "node", "14372.29", "owner", False),
        ("srun", 101, "node", "14372.30", "other", False),
    ],
)
def test_only_actual_local_current_step_launcher_is_allowed(
    tmp_path, monkeypatch, kind, pid, host, step, user, allowed
):
    import psutil

    monkeypatch.setenv("SLURM_STEP_ID", "30")
    uid = once.os.getuid()
    monkeypatch.setattr(once.socket, "gethostname", lambda: "node")
    account = uid if user == "owner" else uid + 1

    def output(command, **kwargs):
        if command[0] == "squeue":
            return "14372.0\n14372.30\n14372.extern\n"
        assert command == ["scontrol", "show", "step", "14372.30", "-o"]
        return f"StepId={step} UserId={account} State=RUNNING SrunHost:Pid={host}:101"

    monkeypatch.setattr(once.subprocess, "check_output", output)
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda: SimpleNamespace(pid=200, parents=lambda: [SimpleNamespace(pid=201)]),
    )
    item = SimpleNamespace(
        pid=pid,
        info={"name": kind, "cmdline": [kind, str(tmp_path / "worker.sh")]},
        uids=lambda: SimpleNamespace(real=uid),
        parents=lambda: [],
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *_: [item])
    args = SimpleNamespace(root=tmp_path, supervisor_step="14372.28")
    if allowed:
        once.check_owner(args)
    else:
        with pytest.raises(ValueError):
            once.check_owner(args)


@pytest.mark.parametrize(
    "kind,parent_kind,parent_uid,expected",
    [
        ("srun", "srun", 1001, True),
        ("python", "srun", 1001, False),
        ("srun", "python", 1001, False),
        ("srun", "srun", 1002, False),
    ],
)
def test_only_same_user_srun_helper_chain_is_accepted(kind, parent_kind, parent_uid, expected):
    launcher = SimpleNamespace(
        pid=101, name=lambda: parent_kind, uids=lambda: SimpleNamespace(real=parent_uid)
    )
    process = SimpleNamespace(
        pid=102,
        info={"name": kind},
        uids=lambda: SimpleNamespace(real=1001),
        parents=lambda: [launcher],
    )
    assert once.current_srun_process(process, 101, 1001) is expected


def test_orphan_or_unrelated_srun_is_rejected():
    process = SimpleNamespace(
        pid=102, info={"name": "srun"}, uids=lambda: SimpleNamespace(real=1001), parents=lambda: []
    )
    assert once.current_srun_process(process, 101, 1001) is False
