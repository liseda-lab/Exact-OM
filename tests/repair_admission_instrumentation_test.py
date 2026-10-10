"""Timing preserves durability, exact interrupted resume and ownership gates."""

import json

import pytest
import torch

from tests.repair_training_completion_test import cache_for
from tools.repair import train
from tools.repair.admission_fixture import remaining, validate_owner
from tools.repair.corpus import generate_corpus
from tools.repair.evaluation_ownership import resources
from tools.repair.phase_timing import load_training_state, recording
from tools.repair.primary_runtime import DEV_GPU, PhaseBoundary


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_trace_does_not_change_exact_interrupted_resume(tmp_path, monkeypatch):
    cases = generate_corpus(
        split_counts={"train": 2, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
        revision="v3",
    )
    fitting = [(c, cache_for(c)) for c in cases if c.split == "train"]
    dev = train.scheduled_development(cases, {})
    options = dict(
        encoder="none",
        hidden_dim=8,
        heads=2,
        layers=0,
        dropout=0.2,
        epochs=2,
        development_epochs=[2],
        max_full_development_evaluations=1,
        patience_enabled=False,
        sampled_assignments=0,
        proposal_arm="bounded_enumeration",
        deadline_seconds=60,
        total_training_seconds=180,
        execution_phase="fit",
    )
    reference, interrupted = tmp_path / "reference.pt", tmp_path / "interrupted.pt"
    with pytest.raises(PhaseBoundary):
        train.train_cases(fitting, dev, checkpoint_path=reference, **options)
    original = train.save_training_state

    def stop_after_committed_update(path, state):
        original(path, state)
        if state.get("optimizer_updates") == 1:
            raise InterruptedError("injected committed minibatch interruption")

    trace = tmp_path / "timing.jsonl"
    monkeypatch.setenv("EXACT_REPAIR_PHASE_LOG", str(trace))
    monkeypatch.setattr(train, "save_training_state", stop_after_committed_update)
    with pytest.raises(InterruptedError):
        train.train_cases(fitting, dev, checkpoint_path=interrupted, **options)
    prior_elapsed = torch.load(interrupted, weights_only=True)["elapsed_seconds"]
    monkeypatch.setattr(train, "save_training_state", original)
    with pytest.raises(PhaseBoundary):
        train.train_cases(fitting, dev, checkpoint_path=interrupted, **options)
    a, b = (torch.load(p, weights_only=True) for p in (reference, interrupted))
    for key in (
        "model",
        "optimizer",
        "cpu_rng",
        "fit_rng",
        "epoch_order",
        "next_epoch",
        "next_offset",
        "optimizer_updates",
        "optimizer_case_counts",
        "identity",
    ):
        equal(a[key], b[key])
    assert b["execution_count"] == 2 and b["elapsed_seconds"] >= prior_elapsed
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len({r["invocation"] for r in rows}) == 2
    ends = [r for r in rows if r["event"] == "end"]
    assert {
        "checkpoint_snapshot",
        "checkpoint_write",
        "checkpoint_serialize",
        "checkpoint_flush_fsync",
        "checkpoint_replace_directory_fsync",
        "checkpoint_read_deserialize",
        "checkpoint_restore",
        "setup_graph_grammar_pairs",
        "setup_model",
        "setup_identity_optimizer",
    } <= {r["phase"] for r in ends}
    writes = [r for r in ends if r["phase"] == "checkpoint_write"]
    assert all(
        r["identity"]["bytes"] > 0 and r["identity"]["identity"] == a["identity"] for r in writes
    )
    assert any(r["status"] == "InterruptedError" for r in ends)
    assert all(r["elapsed_seconds"] >= 0 and r["cpu_seconds"] >= 0 for r in ends)


def test_failed_serialization_preserves_prior_checkpoint_and_records_failure(tmp_path, monkeypatch):
    path, trace = tmp_path / "state.pt", tmp_path / "trace.jsonl"
    train.save_training_state(path, {"identity": "prior", "value": torch.tensor([7])})
    original = path.read_bytes()

    def broken(state, stream):
        stream.write(b"partial")
        raise OSError("injected full disk")

    monkeypatch.setattr(torch, "save", broken)
    with recording(trace), pytest.raises(OSError):
        train.save_training_state(path, {"identity": "replacement"})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".training-*"))
    with recording(trace):
        assert load_training_state(path, map_location="cpu")["identity"] == "prior"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(r.get("status") == "OSError" and r["phase"] == "checkpoint_write" for r in rows)
    assert not any(r["phase"] == "checkpoint_replace_directory_fsync" for r in rows)


@pytest.mark.parametrize("lane", ["native", "inference"])
def test_fixture_owner_requires_actual_nonce_stage_and_resources(tmp_path, lane):
    (tmp_path / "step.json").write_text(
        json.dumps(dict(step_id="14451.99", dispatch_nonce="owned"))
    )
    manifest = dict(nonces={lane: "owned"})
    owner = dict(
        status="reserved",
        resources=resources(lane),
        budget_stages=["learning"] + (["secondary_learning"] if lane == "inference" else []),
        gpu_devices=[DEV_GPU] if lane == "inference" else [],
    )
    ledger = dict(attempts={str(tmp_path): owner})
    validate_owner(manifest, lane, ledger, "14451.99")
    owner["budget_stages"] = ["calibration"]
    with pytest.raises(ValueError, match="resource/stage/nonce"):
        validate_owner(manifest, lane, ledger, "14451.99")
    with pytest.raises(ValueError, match="one charged owner"):
        validate_owner(manifest, lane, ledger, "14451.98")


def test_nested_case_deadline_preserves_cleanup(monkeypatch):
    monkeypatch.setattr("tools.repair.admission_fixture.time.time", lambda: 100)
    assert remaining(200, 105) == 3
    assert remaining(101, 200) == 0


@pytest.mark.parametrize("started,expired", [(False, False), (True, True), (True, False)])
def test_preparation_never_starts_or_renews_a_stage(tmp_path, monkeypatch, started, expired):
    from exact.repair.api import write_artifact
    from tools.repair import prepare_admission as module

    now = 1000
    monkeypatch.setattr(module.time, "time", lambda: now)
    campaign = tmp_path / "campaign"
    reference = dict(path=str(tmp_path / "predecessor.json"), sha256="bound-in-fixture")
    write_artifact(
        campaign / "supervisor/registry.json",
        dict(
            primary_evaluation_capacity_preparation=dict(completion=reference),
            primary_runtime_preparation=dict(throughput_review=dict(path="review")),
            capacity=dict(cpus=14, gpus=2, memory_mb=101297),
        ),
    )
    write_artifact(campaign / "campaign.json", dict(primary_freeze_epoch=100000))
    limits = dict(learning=dict(elapsed_seconds=10000), secondary_learning=dict(gpu_seconds=10000))
    stages = (
        {
            name: dict(started_epoch=-20000 if expired else now - 100, limits=value)
            for name, value in limits.items()
        }
        if started
        else {}
    )
    stages["primary_training"] = dict(cumulative=dict(gpu_seconds=650.2801702707075))
    ledger = dict(attempts={}, stage_limits=limits, stages=stages)
    write_artifact(campaign / "ledger.json", ledger)

    def bound(r):
        if r["path"] == "review":
            return dict(
                report_bindings={
                    "qualify-current-inventory-5090-profile-recovery-001": dict(path="report")
                }
            )
        if r["path"] == "report":
            return dict(
                status="resource_limited", optimizer_updates=99, phase_state=dict(path="state")
            )
        return dict(audit_receipt={})

    monkeypatch.setattr(module, "bound", bound)
    monkeypatch.setattr(
        module, "validate_completion", lambda r: (dict(work=str(tmp_path)), {}, {}, {})
    )
    specs = []

    def freeze(spec, destination):
        specs.append(module.read(spec))
        path = destination / "batch.json"
        write_artifact(path, specs[-1])
        return path

    monkeypatch.setattr(module, "freeze", freeze)
    monkeypatch.setattr(
        module,
        "prepare_dispatch",
        lambda batch, job, attempt, **kw: dict(id=job, depends_on=kw["depends_on"]),
    )
    if not started or expired:
        with pytest.raises(ValueError, match="new stage|remaining stage"):
            module.prepare(campaign, tmp_path / "out", tmp_path / "repo")
        assert not specs
    else:
        value = module.prepare(campaign, tmp_path / "out", tmp_path / "repo")
        assert value["status"] == "prepared_not_queued"
        jobs = specs[0]["jobs"]
        assert len(jobs) == 3 and all("calibration" not in j["budget_stages"] for j in jobs)
        assert sum(j["seconds"] for j in jobs) == 900
        assert module.read(campaign / "ledger.json") == ledger
        assert module.read(campaign / "supervisor/registry.json").keys() == {
            "primary_evaluation_capacity_preparation",
            "primary_runtime_preparation",
            "capacity",
        }


def test_checkpoint_diagnostic_preserves_all_state_and_refuses_replay(tmp_path, monkeypatch):
    import time
    from exact.repair.api import write_artifact
    from tools.repair.checkpoint_diagnostic import run
    from tools.repair.historical_regression import binding

    source = tmp_path / "old.pt"
    state = dict(
        identity="frozen",
        optimizer_updates=99,
        optimizer=dict(state={1: dict(step=torch.tensor(1), momentum=torch.tensor([0.5]))}),
        cpu_rng=torch.get_rng_state(),
        next_epoch=0,
        next_offset=99,
        elapsed_seconds=650.2801702707075,
    )
    train.save_training_state(source, state)
    before = binding(source)
    reference = tmp_path / "reference.json"
    write_artifact(
        reference,
        dict(
            scope="terminal_train_engineering_state_io_only",
            checkpoint=before,
            expected_optimizer_updates=99,
        ),
    )
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(time.time() + 60))
    report = run(reference, tmp_path / "io")
    assert report["exact_all_state"] and report["optimizer_updates_executed"] == 0
    equal(torch.load(report["roundtrip"]["path"], weights_only=True), state)
    assert binding(source) == before
    with pytest.raises(ValueError, match="reset or replay"):
        run(reference, tmp_path / "io")
