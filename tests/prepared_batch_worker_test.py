"""Prepared workers retain charges and terminal receipts across retry boundaries."""

from __future__ import annotations

import fcntl
import os
from types import SimpleNamespace

import pytest

from exact.experiments import campaign
from exact.experiments.budget import BudgetLedger
from exact.llm.ledger import RequestLedger
from tests.prepared_batch_history_test import _fixture as history_fixture
from tools import experiment_resources, prepared_batch, resume_e19_once


@pytest.mark.parametrize("failed", [False, True])
def test_saved_treatment_import_is_charged_and_failure_never_runs_cells(
    tmp_path, monkeypatch, failed
):
    from tools import recover_e22_measurements

    worker = _worker(tmp_path, monkeypatch)
    donor = tmp_path / "donor"
    source = donor / "runtime" / worker.lock["campaign_id"]
    prepared_batch.write(source / "budget.json", {"work": {"last": {"end": 4}}})
    prepared_batch.write(donor / "status.json", {"at": 5})
    recipe = prepared_batch.read(worker.path)
    recipe.update(
        parent_run_id="E22-prior",
        e22_completed_treatments={"source_runtime": str(source)},
    )
    prepared_batch.write(worker.path, recipe)
    calls = []

    def migrate(*args):
        calls.append("import")
        if failed:
            raise ValueError("Treatment checksum changed")

    def execute(*args, **kwargs):
        calls.append("execute")
        _outputs(worker)

    monkeypatch.setattr(recover_e22_measurements, "import_completed_treatments", migrate)
    monkeypatch.setattr(experiment_resources, "guarded_execute", execute)
    if failed:
        with pytest.raises(ValueError, match="Treatment checksum changed"):
            prepared_batch.run_recipe(worker.path)
    else:
        prepared_batch.run_recipe(worker.path)
    assert calls == (["import"] if failed else ["import", "execute"])
    account = prepared_batch.read(worker.runtime / "budget.json")
    work = account["work"]["preparation/E22/treatment-import/40"]
    assert work["status"] == ("failed" if failed else "complete")
    assert work["requests"] == work["tokens"] == work["actual_usd"] == 0
    if not failed:
        tail = account["work"]["failed-finalization/E22-prior"]
        assert (tail["start"], tail["end"], tail["status"]) == (4, 5, "failed")


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("retained_charge", [False, True])
def test_label_repair_precedes_cells_and_closes_accounting_on_failure(
    tmp_path, monkeypatch, failed, retained_charge
):
    from tools import finalize_prepared_selection, recover_e22_labels

    worker = _worker(tmp_path, monkeypatch)
    recipe = prepared_batch.read(worker.path)
    recipe.update(
        parent_run_id="E22-budget-recovery",
        e22_label_repair={"snapshots": []},
        failed_finalization_interval={"run_id": "E22-policy-recovery-01", "start": 4, "end": 5},
    )
    if retained_charge:
        finalize_prepared_selection.charge_failed_finalization(
            recipe, BudgetLedger(worker.parent, worker.limits), prepared_batch.read(worker.parent)
        )
    prepared_batch.write(worker.path, recipe)
    calls = []

    def prepare(actual_recipe, lock, runtime, code):
        assert actual_recipe == recipe
        assert lock == worker.root / "campaign.lock.yaml" and runtime == worker.runtime
        assert str(code) == recipe["code_root"]
        tail = prepared_batch.read(runtime / "budget.json")["work"][
            "failed-finalization/E22-policy-recovery-01"
        ]
        assert (tail["start"], tail["end"], tail["status"]) == (4, 5, "failed")
        calls.append("prepare")
        if failed:
            raise ValueError("Unverified label-repair snapshot")

    def execute(*args, **kwargs):
        calls.append("execute")
        _outputs(worker)

    monkeypatch.setattr(recover_e22_labels, "prepare_label_recovery", prepare)
    monkeypatch.setattr(experiment_resources, "guarded_execute", execute)
    if failed:
        with pytest.raises(ValueError, match="Unverified label-repair snapshot"):
            prepared_batch.run_recipe(worker.path)
    else:
        prepared_batch.run_recipe(worker.path)
    assert calls == (["prepare"] if failed else ["prepare", "execute"])
    account = prepared_batch.read(worker.runtime / "budget.json")
    work = account["work"]["preparation/E22/label-repair/40"]
    assert work["status"] == ("failed" if failed else "complete")
    assert work["end"] >= work["start"]
    assert work["requests"] == work["tokens"] == work["actual_usd"] == 0
    assert [key for key in account["work"] if key.startswith("failed-finalization/")] == [
        "failed-finalization/E22-policy-recovery-01"
    ]


def _request(directory, label, tokens):
    ledger = RequestLedger(directory)
    key = ledger.plan({"role": "decision", "payload": {"max_tokens": 8, "text": label}})
    number = ledger.sent(key)
    ledger.received(key, number, b'{"result":"fixture"}', 200)
    ledger.usage(key, number, {"prompt_tokens": tokens - 2, "completion_tokens": 2, "cost": 0.001})
    return key


def _charge(path, limits, key, *, requests=1, tokens=12, status="complete"):
    ledger = BudgetLedger(path, limits)
    ledger.admit(key, group="llm", seconds=1, requests=requests, tokens=tokens)
    ledger.finish(
        key, start=1, end=2, status=status, requests=requests, tokens=tokens, actual_usd=0.001
    )
    return ledger.snapshot()


def _worker(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key in (
        "EXACT_OPENROUTER_REQUEST_CAP",
        "EXACT_OPENROUTER_TOKEN_CAP",
        "EXACT_OPENROUTER_RETRY_UNKNOWN",
        "EXACT_EVIDENCE_PREFETCH",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "14372")
    monkeypatch.setenv("SLURM_STEP_ID", "40")
    recipe, lock, _ = history_fixture(tmp_path)
    lock = campaign.CampaignLock.model_validate(lock).model_dump(mode="json")
    root, supervisor = tmp_path / "worker", tmp_path / "supervisor"
    root.mkdir()
    supervisor.mkdir()
    (supervisor / "supervisor-step-id").write_text("14372.39\n")
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "api_key").write_text("test-placeholder-not-a-secret")
    environment = tmp_path / "environment.json"
    environment.write_text("{}")
    recipe.update(
        root=str(root),
        code_root=str(repository),
        supervisor=str(supervisor),
        repository=str(repository),
        environment=prepared_batch.binding(environment),
        allocation="14372",
        campaign_id=lock["campaign_id"],
        dispatch_nonce="fixture-nonce",
    )
    path = root / "recipe.json"
    prepared_batch.write(path, recipe)
    monkeypatch.setattr(prepared_batch.sys, "argv", ["prepared_batch.py", "--recipe", str(path)])
    limits = {
        "envelopes_hours": {"llm": 1, "reserve": 1},
        "requests_cap": 100,
        "tokens_cap": 10000,
        "final_requests_reserved": 10,
        "final_tokens_reserved": 1000,
    }
    parent = tmp_path / "parent/budget.json"
    parent_state = _charge(parent, limits, "historical/closed")
    request = _request(parent.parent / "openrouter", "historical", 12)
    parent_status = parent.parent / "status.json"
    prepared_batch.write(parent_status, {"cumulative_budget": str(parent)})
    registry = {
        "runs": [
            {
                "id": "parent",
                "status_path": str(parent_status),
                "completion_path": str(parent.parent / "completion.json"),
            },
            {
                "id": "E18",
                "status_path": str(root / "status.json"),
                "completion_path": str(root / "completion.json"),
            },
        ]
    }
    prepared_batch.write(supervisor / "registry.json", registry)
    monkeypatch.setattr(resume_e19_once, "check_owner", lambda args: None)

    def git(command, **kwargs):
        assert command[:1] == ["git"], "No live scheduler or other external process is allowed"
        return recipe["commit"] + "\n" if command[1] == "rev-parse" else ""

    monkeypatch.setattr(prepared_batch.subprocess, "check_output", git)
    preparations = []

    def prepare(*args, **kwargs):
        preparations.append(True)
        return lock

    monkeypatch.setattr(prepared_batch, "prepare_lock", prepare)
    monkeypatch.setattr(
        campaign,
        "campaign_plan",
        lambda *a, **kw: {
            "rows": [{"step": "E18", "issues": []}],
            "budget_errors": [],
        },
    )
    runtime = root / "runtime" / recipe["campaign_id"]
    return SimpleNamespace(
        path=path,
        root=root,
        runtime=runtime,
        parent=parent,
        parent_state=parent_state,
        request=request,
        limits=limits,
        lock=lock,
        preparations=preparations,
    )


def _outputs(worker):
    stage = worker.runtime / "screen"
    prepared_batch.write(stage / "selection.json", {"experiments": {"E18": {"status": "selected"}}})
    step = next(s for s in worker.lock["steps"] if s["id"] == "E18")
    for arm in step["arms"]:
        for mode in step.get("execution_modes", ["global_alignment"]):
            task = step["case"] + "-" + mode
            for seed in step.get("seeds", [17]):
                prepared_batch.write(
                    stage / "runs/E18" / arm["id"] / task / f"seed-{seed}/experiment_manifest.json",
                    {
                        "status": "complete",
                        "return_code": 0,
                        "extraction_complete": True,
                        "generate_rationales": False,
                        "arm_id": arm["id"],
                        "task_id": task,
                        "execution_mode": mode,
                        "seed": seed,
                    },
                )


def test_prepared_worker_executes_once_and_keeps_completed_receipts_on_duplicate(
    tmp_path, monkeypatch
):
    worker = _worker(tmp_path, monkeypatch)
    calls = []

    def execute(*args, **kwargs):
        calls.append(True)
        kwargs["check_pause"]()
        _outputs(worker)

    monkeypatch.setattr(experiment_resources, "guarded_execute", execute)
    parent_before = worker.parent.read_bytes()
    prepared_batch.main()
    assert calls == [True] and worker.preparations == [True]
    assert worker.parent.read_bytes() == parent_before
    assert (
        RequestLedger(worker.runtime / "openrouter").cached(worker.request)
        == b'{"result":"fixture"}'
    )
    current = prepared_batch.read(worker.runtime / "budget.json")
    assert current["work"]["historical/closed"] == worker.parent_state["work"]["historical/closed"]
    assert "prepared-dispatch/E18/40" in current["work"]
    assert os.environ["EXACT_OPENROUTER_REQUEST_CAP"] == "90"
    assert os.environ["EXACT_OPENROUTER_TOKEN_CAP"] == "9000"
    assert os.environ["EXACT_EVIDENCE_PREFETCH"] == "0"
    completion = prepared_batch.read(worker.root / "completion.json")
    assert completion["step_id"] == "14372.40"
    assert completion["dispatch_nonce"] == "fixture-nonce"
    completed_bytes = (worker.root / "completion.json").read_bytes()
    status_bytes = (worker.root / "status.json").read_bytes()
    with pytest.raises(ValueError, match="must never be relaunched"):
        prepared_batch.main()
    assert calls == [True] and worker.preparations == [True]
    assert (worker.root / "completion.json").read_bytes() == completed_bytes
    assert (worker.root / "status.json").read_bytes() == status_bytes


def test_prepared_worker_resumes_own_budget_without_reimport_or_charge_reset(tmp_path, monkeypatch):
    worker = _worker(tmp_path, monkeypatch)
    attempts = []
    checkpoint = worker.runtime / "saved-step.txt"
    resumed_request = []

    def execute(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            _charge(
                worker.runtime / "budget.json",
                worker.limits,
                "failed/E18/attempt-1",
                tokens=7,
                status="failed",
            )
            resumed_request.append(
                _request(worker.runtime / "openrouter", "failed-attempt-result", 7)
            )
            checkpoint.write_text("completed scientific work")
            raise RuntimeError("short numerical failure")
        assert checkpoint.read_text() == "completed scientific work"
        _outputs(worker)

    monkeypatch.setattr(experiment_resources, "guarded_execute", execute)
    with pytest.raises(RuntimeError, match="short numerical failure"):
        prepared_batch.main()
    failed = prepared_batch.read(worker.root / "completion.json")
    assert failed["status"] == "failed" and failed["exit_code"] == 1
    assert failed["step_id"] == "14372.40" and failed["dispatch_nonce"] == "fixture-nonce"
    assert "short numerical failure" in prepared_batch.read(worker.root / "status.json")["error"]
    charged = prepared_batch.read(worker.runtime / "budget.json")
    imported = (worker.runtime / "account-import.json").read_bytes()
    parent_before = worker.parent.read_bytes()

    def no_reimport(*args, **kwargs):
        pytest.fail("Resuming an authoritative worker account must not import an older ledger")

    monkeypatch.setattr(prepared_batch, "copy_account", no_reimport)
    monkeypatch.setenv("SLURM_STEP_ID", "41")
    prepared_batch.main()
    current = prepared_batch.read(worker.runtime / "budget.json")
    assert all(current["work"][key] == value for key, value in charged["work"].items())
    assert "prepared-dispatch/E18/41" in current["work"]
    assert worker.parent.read_bytes() == parent_before
    assert (worker.runtime / "account-import.json").read_bytes() == imported
    assert RequestLedger(worker.runtime / "openrouter").cached(resumed_request[0]) is not None
    assert worker.preparations == [True] and len(attempts) == 2
    assert prepared_batch.read(worker.root / "failed-14372.40.json") == failed
    assert prepared_batch.read(worker.root / "completion.json")["step_id"] == "14372.41"


def test_short_worker_admission_failure_emits_nonce_and_numeric_step_receipt(tmp_path, monkeypatch):
    worker = _worker(tmp_path, monkeypatch)
    monkeypatch.setattr(
        campaign,
        "campaign_plan",
        lambda *a, **kw: {
            "rows": [{"step": "E18", "issues": ["missing required evidence"]}],
            "budget_errors": [],
        },
    )

    def never_execute(*args, **kwargs):
        pytest.fail("A failed admission must not start numerical work")

    monkeypatch.setattr(experiment_resources, "guarded_execute", never_execute)
    with pytest.raises(ValueError, match="failed admission"):
        prepared_batch.main()
    failed = prepared_batch.read(worker.root / "completion.json")
    assert failed["status"] == "failed" and failed["exit_code"] == 1
    assert failed["step_id"] == "14372.40" and failed["dispatch_nonce"] == "fixture-nonce"
    assert prepared_batch.read(worker.root / "status.json")["status"] == "failed"
    assert not (worker.runtime / "screen").exists()


def _terminal_recipe(tmp_path, monkeypatch):
    root = tmp_path / "short-job"
    root.mkdir()
    path = root / "recipe.json"
    prepared_batch.write(
        path,
        {
            "root": str(root),
            "allocation": "14372",
            "campaign_id": "fixture-campaign",
            "dispatch_nonce": "short-job-nonce",
        },
    )
    monkeypatch.setenv("SLURM_STEP_ID", "42")
    return root, path


@pytest.mark.parametrize("exit_code", [1, 130, 143])
def test_shell_exit_helper_records_actual_terminal_exit_and_nonce(tmp_path, monkeypatch, exit_code):
    root, path = _terminal_recipe(tmp_path, monkeypatch)
    monkeypatch.setattr(
        prepared_batch.sys,
        "argv",
        [
            "prepared_batch.py",
            "--recipe",
            str(path),
            "--record-exit",
            str(exit_code),
        ],
    )

    def no_numerical_run(*args, **kwargs):
        pytest.fail("Recording a shell exit must never relaunch scientific work")

    monkeypatch.setattr(prepared_batch, "run_recipe", no_numerical_run)
    prepared_batch.main()
    receipt = prepared_batch.read(root / "completion.json")
    assert receipt == {
        "status": "failed",
        "exit_code": exit_code,
        "step_id": "14372.42",
        "dispatch_nonce": "short-job-nonce",
    }
    assert prepared_batch.read(root / "status.json")["exit_code"] == exit_code
    original = (root / "completion.json").read_bytes()
    prepared_batch.record_exit(path, 1)
    assert (root / "completion.json").read_bytes() == original


def test_shell_exit_helper_preserves_existing_completed_comparison(tmp_path, monkeypatch):
    root, path = _terminal_recipe(tmp_path, monkeypatch)
    prepared_batch.write(root / "completion.json", {"status": "complete", "step_id": "14372.41"})
    prepared_batch.write(root / "status.json", {"status": "complete"})
    before = {name: (root / name).read_bytes() for name in ("completion.json", "status.json")}
    prepared_batch.record_exit(path, 143)
    assert all((root / name).read_bytes() == value for name, value in before.items())


def test_shell_exit_helper_cannot_overwrite_locked_live_worker(tmp_path, monkeypatch):
    root, path = _terminal_recipe(tmp_path, monkeypatch)
    prepared_batch.write(root / "status.json", {"status": "running"})
    before = (root / "status.json").read_bytes()
    with (root / "worker.lock").open("a") as live:
        fcntl.flock(live, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared_batch.record_exit(path, 143)
    assert not (root / "completion.json").exists()
    assert (root / "status.json").read_bytes() == before


def test_shell_exit_helper_archives_prior_failed_step_before_recording_retry(tmp_path, monkeypatch):
    root, path = _terminal_recipe(tmp_path, monkeypatch)
    previous = {
        "status": "failed",
        "exit_code": 1,
        "step_id": "14372.41",
        "dispatch_nonce": "short-job-nonce",
    }
    prepared_batch.write(root / "completion.json", previous)
    prepared_batch.record_exit(path, 143)
    assert prepared_batch.read(root / "failed-14372.41.json") == previous
    latest = prepared_batch.read(root / "completion.json")
    assert latest["status"] == "failed" and latest["exit_code"] == 143
    assert latest["step_id"] == "14372.42" and latest["dispatch_nonce"] == "short-job-nonce"


@pytest.mark.parametrize("failed", [False, True])
def test_diagnostic_worker_preserves_account_and_does_not_run_campaign(
    tmp_path, monkeypatch, failed
):
    from tools import run_directional_diagnostic

    worker = _worker(tmp_path, monkeypatch)
    recipe = prepared_batch.read(worker.path)
    output = tmp_path / "diagnostic"
    diagnostic = tmp_path / "diagnostic-recipe.json"
    prepared_batch.write(diagnostic, {"kind": "e24_directional_known_pairs", "output": str(output)})
    recipe["diagnostic"] = prepared_batch.binding(diagnostic)
    prepared_batch.write(worker.path, recipe)
    # Offline diagnostics must not even require access to the hosted credential.
    (tmp_path / "api_key").unlink(missing_ok=True)
    monkeypatch.setattr(
        experiment_resources,
        "guarded_execute",
        lambda *a, **k: pytest.fail("diagnostic ran a selecting campaign"),
    )

    def run(path):
        assert path == diagnostic
        if failed:
            raise RuntimeError("fixture failed after checkpoint")
        artifact = output / "diagnostic.json"
        prepared_batch.write(artifact, {"result": "fixture"})
        prepared_batch.write(
            output / "completion.json",
            {
                "status": "complete",
                "selection_eligible": False,
                "diagnostic": prepared_batch.binding(artifact),
            },
        )

    monkeypatch.setattr(run_directional_diagnostic, "run_diagnostic", run)
    if failed:
        with pytest.raises(RuntimeError, match="fixture failed"):
            prepared_batch.run_recipe(worker.path)
    else:
        prepared_batch.run_recipe(worker.path)
        completion = prepared_batch.read(worker.root / "completion.json")
        assert completion["selection_eligible"] is False
        assert "selection" not in completion
    account = prepared_batch.read(worker.runtime / "budget.json")
    charge = account["work"]["diagnostic/E18/40"]
    assert charge["status"] == ("failed" if failed else "complete")
    assert charge["requests"] == charge["tokens"] == charge["actual_usd"] == 0
