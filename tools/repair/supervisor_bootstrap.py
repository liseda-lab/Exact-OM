"""Prepare and explicitly start the existing supervisor in a retained extern-owned tmux."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from tools.repair.batch import read, sha
from tools.supervise_experiments import config_fingerprint, validate_policy, write


def prepare(
    directory,
    *,
    allocation,
    node,
    code,
    repository,
    python,
    instructions,
    codex,
    codex_config,
    recipient="pgcotovio@gmail.com",
):
    directory, code = Path(directory).resolve(), Path(code).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "policy.json").exists():
        raise ValueError(
            "Supervisor policy already exists; prepare a separately reviewed deployment"
        )
    policy = dict(
        schema_version=1,
        allocation=str(allocation),
        node=node,
        code_root=str(code),
        repository=str(Path(repository).resolve()),
        python=str(Path(python).resolve()),
        codex=str(Path(codex).absolute()),
        codex_config=str(Path(codex_config).resolve()),
        authentication="chatgpt",
        model="gpt-6-astra",
        model_reasoning_effort="xhigh",
        interval_seconds=300,
        max_agent_runs_per_day=None,
        max_attempts_per_incident=2,
        agent_timeout_seconds=2400,
        notify_when_idle=True,
        require_actionable_blocker=True,
        instructions=str(Path(instructions).resolve()),
        instructions_sha256=sha(instructions),
    )
    policy["codex_config_semantic_sha256"] = config_fingerprint(policy)
    policy["notifications"] = dict(
        enabled=True,
        transport="command",
        recipient=recipient,
        sender=recipient,
        timeout_seconds=180,
        command=[
            policy["python"],
            str(code / "tools/send_supervisor_email.py"),
            "--policy",
            str(directory / "policy.json"),
            "--directory",
            str(directory),
            "--recipient",
            recipient,
        ],
    )
    validate_policy(policy)
    write(directory / "policy.json", policy)
    entry = directory / "entry.py"
    entry.write_text(
        "import json, os, time\nfrom pathlib import Path\n"
        + "root=Path("
        + repr(str(directory))
        + ")\n"
        + "assert os.environ['SLURM_JOB_ID'] == "
        + repr(str(allocation))
        + "\n"
        + "assert not (root/'STOP').exists() and not (root/'PAUSE').exists()\n"
        + "(root/'startup.json').write_text(json.dumps(dict(status='started', pid=os.getpid(), "
        "step_id=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'], "
        "started_epoch=time.time(), resources=dict(cpus=1,gpus=0,memory_mb=8192))))\n"
        + "os.execv("
        + repr(policy["python"])
        + ", "
        + repr(
            [
                policy["python"],
                str(code / "tools/supervise_experiments.py"),
                "--directory",
                str(directory),
            ]
        )
        + ")\n"
    )
    argv = [
        "/usr/bin/srun",
        "--jobid=" + str(allocation),
        "--overlap",
        "--exact",
        "-N1",
        "-n1",
        "--cpus-per-task=1",
        "--mem=8192",
        "--gres=none",
        "--cpu-bind=none",
        "--unbuffered",
        "--job-name=repair-supervisor",
        "--chdir=" + str(code),
        policy["python"],
        str(entry),
    ]
    launcher = directory / "launch-supervisor.sh"
    launcher.write_text(
        "#!/bin/bash\nset -u\n"
        + shlex.join(argv)
        + " </dev/null >>"
        + shlex.quote(str(directory / "supervisor.log"))
        + " 2>&1\n"
        + "result=$?\nprintf '%s\\n' \"$result\" >"
        + shlex.quote(str(directory / "launcher-exit"))
        + '\nexit "$result"\n'
    )
    launch = dict(
        argv=argv,
        launcher=str(launcher),
        tmux_socket=str(directory / "tmux.sock"),
        bindings=[
            dict(path=str(path), sha256=sha(path))
            for path in (directory / "policy.json", entry, launcher)
        ],
    )
    write(directory / "launch.json", launch)
    return launch


def start(directory):
    """Explicit operational action; never remove a STOP or borrow another owner."""
    directory = Path(directory).resolve()
    policy, launch = read(directory / "policy.json"), read(directory / "launch.json")
    if any((directory / name).exists() for name in ("STOP", "PAUSE")):
        raise ValueError("Supervisor STOP/PAUSE remains authoritative")
    if os.environ.get("SLURM_JOB_ID") != policy["allocation"]:
        raise ValueError("Start only inside the retained allocation")
    own = Path("/proc/self/cgroup").read_text()
    group = re.search(r"^0::(.+)/step_extern/", own, re.MULTILINE)
    if group is None:
        raise ValueError("Persistent tmux must be created from this allocation's extern cgroup")
    for item in launch["bindings"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Reviewed supervisor launch changed")
    base = ["/usr/bin/tmux", "-N", "-S", launch["tmux_socket"]]
    found = subprocess.run(
        base + ["display-message", "-p", "#{pid}"], capture_output=True, text=True
    )
    if found.returncode:
        subprocess.run(
            [
                "/usr/bin/tmux",
                "-S",
                launch["tmux_socket"],
                "new-session",
                "-d",
                "-s",
                "repair-owner",
                "/bin/sleep",
                "infinity",
            ],
            check=True,
        )
        found = subprocess.run(
            base + ["display-message", "-p", "#{pid}"], capture_output=True, text=True, check=True
        )
    pid = int(found.stdout.strip())
    owner = Path("/proc") / str(pid)
    if (
        owner.stat().st_uid != os.getuid()
        or group.group(1) + "/step_extern/" not in (owner / "cgroup").read_text()
    ):
        raise ValueError("Detached server belongs to another allocation or user")
    write(
        directory / "detached-owner.json",
        dict(pid=pid, cgroup=(owner / "cgroup").read_text(), allocation=policy["allocation"]),
    )
    session = "repair-supervisor"
    exists = subprocess.run(base + ["has-session", "-t", session], capture_output=True)
    if exists.returncode == 0:
        raise ValueError("Supervisor launcher already exists; reconcile its live receipt")
    subprocess.run(
        base + ["new-session", "-d", "-s", session, "/bin/bash", launch["launcher"]], check=True
    )
    return dict(status="launcher_started", allocation=policy["allocation"], tmux_pid=pid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "start"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--configuration", type=Path)
    args = parser.parse_args()
    result = (
        prepare(args.directory, **read(args.configuration))
        if args.action == "prepare"
        else start(args.directory)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
