"""DEV engineering cannot open TEST, alter the fixed sample or reset stage clocks."""

from collections import Counter
import time

import pytest

from exact.repair.api import write_artifact
from tests.repair_primary_evaluation_test import public_fixture, schedule
from tools.repair.evaluation_ownership import PROFILE, lane_for, resources, validate_owner
from tools.repair.historical_regression import binding
from tools.repair.primary_evaluation import SCHEMA, execute_rows
from tools.repair.representative_capacity import (
    MODEL,
    SCOPE,
    convert_engineering_weights,
    fixed_cases,
    representative_rows,
    validate_manifest,
    wait_for_peer,
)


def engineering_manifest(tmp_path, lane="native"):
    template = schedule()
    example, public, protocol = public_fixture(tmp_path)
    cases, inputs = [], {}
    for family in range(8):
        for control in ("coherent", "corrupted"):
            case = dict(
                example,
                case_id=f"dev-{family}-{control}",
                family=f"f{family}",
                control=control,
                observable=public,
            )
            cases.append(case)
            from tools.repair.batch import read

            payload = read(public["path"])
            payload["metadata"] = case
            path = tmp_path / (case["case_id"] + ".json")
            write_artifact(path, payload)
            inputs[case["case_id"]] = binding(path)
    write_artifact(tmp_path / "declaration.json", dict(rows=cases))
    plan = dict(
        cases=cases,
        development_declaration=binding(tmp_path / "declaration.json"),
        protocols=[protocol],
        operations=[
            "complete_plan_observable_query",
            "deletion_after_rich",
            "fixed_learned",
            "generated_learned",
            "native_deletion",
            "observable_score_greedy",
            "pool",
            "proposal_support",
            "support_retention_surrogate",
        ],
    )
    rows = representative_rows(plan, template)
    write_artifact(tmp_path / "plan.json", plan)
    write_artifact(tmp_path / "template.json", template)
    write_artifact(
        tmp_path / "schedule.json",
        dict(
            fixed_plan=binding(tmp_path / "plan.json"),
            template_schedule=binding(tmp_path / "template.json"),
            rows=rows,
            public_inputs=inputs,
        ),
    )
    (tmp_path / "weights.pt").write_bytes(b"mock weights; no inference in admission tests")
    model = dict(**binding(tmp_path / "weights.pt"), model_hash="mock")
    write_artifact(
        tmp_path / "conversion.json",
        dict(role=MODEL, selected=False, optimizer_updates=0, artifact=model),
    )
    manifest = dict(
        schema=SCHEMA,
        scope=SCOPE,
        execution_authorized=True,
        ownership_lane=lane,
        cpus=3,
        memory_mb=20000,
        device="cpu",
        gpu_uuid=None,
        concurrent_load_profile=PROFILE,
        dispatch_nonce="owner",
        deadline_epoch=time.time() + 300,
        ledger_path=str(tmp_path / "ledger.json"),
        source_commit="fixture",
        schedule=binding(tmp_path / "schedule.json"),
        engineering_model_conversion=binding(tmp_path / "conversion.json"),
        cases={c["case_id"]: c for c in cases},
        public_inputs=inputs,
        models={MODEL: model},
        rows=[r for r in rows if lane_for(r) == lane],
        hosted_calls=0,
        optimizer_updates=0,
        test_payloads_allowed=False,
    )
    return manifest, plan, rows


def test_fixed_sample_all_operations_and_matched_knobs(tmp_path):
    manifest, plan, rows = engineering_manifest(tmp_path)
    assert len(fixed_cases(plan)) == 16
    assert Counter(map(lane_for, rows)) == dict(pool=16, native=112, inference=112)
    assert len({r["id"] for r in rows}) == 240
    assert set(r["case_seconds"] for r in rows) == {45}
    pools = {r["id"]: r for r in rows if lane_for(r) == "pool"}
    for row in rows:
        if row["pool"]:
            pool = pools[row["pool"]]
            assert all(pool[k] == row[k] for k in ("case_id", "input_identity", "protocol"))
        if row["adapter"]["operation"] == "native_deletion":
            assert row["pool"] is None
    validate_manifest(manifest, check_visibility=False)


@pytest.mark.parametrize("damage", ["test", "replacement", "budget", "model", "hosted", "input"])
def test_engineering_scope_cannot_bypass_closed_test_or_frozen_rows(tmp_path, damage):
    manifest, plan, rows = engineering_manifest(tmp_path)
    if damage == "test":
        next(iter(manifest["cases"].values()))["split"] = "test"
    elif damage == "replacement":
        manifest["rows"].pop()
    elif damage == "budget":
        manifest["rows"][0]["case_seconds"] *= 2
    elif damage == "model":
        manifest["models"] = {"selected-primary": next(iter(manifest["models"].values()))}
    elif damage == "hosted":
        manifest["hosted_calls"] = 1
    else:
        manifest["public_inputs"] = {}
    with pytest.raises(ValueError):
        validate_manifest(manifest, check_visibility=False)


def test_fixed_sample_rejects_later_favorable_replacement(tmp_path):
    _, plan, _ = engineering_manifest(tmp_path)
    plan["cases"][0]["case_id"] = "later-case"
    with pytest.raises(ValueError, match="identity changed"):
        fixed_cases(plan)


def test_engineering_owner_requires_existing_learning_and_no_evaluation_charge(
    tmp_path, monkeypatch
):
    manifest, _, _ = engineering_manifest(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "4")
    attempt = tmp_path / "attempt"
    write_artifact(attempt / "step.json", dict(step_id="123.4", dispatch_nonce="owner"))
    owner = dict(
        status="running", resources=resources("native"), gpu_devices=[], budget_stages=["learning"]
    )
    stage = dict(attempts={str(attempt): owner})
    validate_owner(manifest, stage)
    owner["budget_stages"].append("evaluation")
    with pytest.raises(ValueError, match="only existing learning"):
        validate_owner(manifest, stage)


def test_expired_learning_clock_preserves_all_unattempted_rows(tmp_path, monkeypatch):
    manifest, _, _ = engineering_manifest(tmp_path, lane="pool")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "4")
    monkeypatch.setattr("torch.cuda.device_count", lambda: 0)
    attempt = tmp_path / "attempt"
    write_artifact(attempt / "step.json", dict(step_id="123.4", dispatch_nonce="owner"))
    ledger = dict(
        stages=dict(learning=dict(started_epoch=time.time() - 67 * 3600)),
        attempts={
            str(attempt): dict(
                status="running",
                resources=resources("pool"),
                gpu_devices=[],
                budget_stages=["learning"],
            )
        },
    )
    write_artifact(tmp_path / "ledger.json", ledger)
    result = execute_rows(manifest, tmp_path / "work")
    assert len(result["rows"]) == 16
    assert all(r["status"] == "not_attempted" for r in result["rows"])
    assert result["charged_seconds"] == 0
    assert "evaluation" not in ledger["stages"]


def test_weight_conversion_is_lossless_and_refuses_architecture_change(tmp_path):
    import torch
    from exact.repair.model import RepairModel

    model = RepairModel(
        (("class",), ()),
        encoder="none",
        hidden_dim=8,
        heads=2,
        layers=0,
        revision="v3",
        pairwise=True,
    )
    source = tmp_path / "original.pt"
    torch.save(dict(metadata=model.metadata, config=model.config, model=model.state_dict()), source)
    protocol = dict(
        model=dict(
            backbone="none",
            hidden_width=8,
            attention_heads=2,
            layers=0,
            graph_schema=None,
            support_enabled=False,
        )
    )
    write_artifact(tmp_path / "protocol.json", protocol)
    original = source.read_bytes()
    receipt = convert_engineering_weights(
        binding(source), binding(tmp_path / "protocol.json"), tmp_path / "converted.pt"
    )
    assert receipt["selected"] is False and receipt["optimizer_updates"] == 0
    assert receipt["tensor_equality"] and source.read_bytes() == original
    protocol["model"]["hidden_width"] = 16
    write_artifact(tmp_path / "changed.json", protocol)
    with pytest.raises(ValueError, match="architecture differs"):
        convert_engineering_weights(
            binding(source), binding(tmp_path / "changed.json"), tmp_path / "bad.pt"
        )


@pytest.mark.parametrize("damage", [None, "nonce", "settled", "missing"])
def test_concurrent_barrier_requires_live_authenticated_peer(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(
        "tools.repair.representative_capacity.validate_manifest", lambda *a, **k: None
    )
    monkeypatch.setattr("tools.repair.evaluation_ownership.validate_owner", lambda *a, **k: None)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "4")
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", str(time.time() + 60))
    identity = dict(schedule=dict(path="fixed", sha256="fixed"), source_commit="source")
    peers, attempts = {}, {}
    for lane, step in (("native", "123.4"), ("inference", "123.5")):
        attempt = tmp_path / lane
        peers[lane] = dict(nonce=lane, attempt=str(attempt))
        write_artifact(attempt / "step.json", dict(step_id=step, dispatch_nonce=lane))
        attempts[str(attempt)] = dict(status="running")
        if lane == "inference" and damage != "missing":
            write_artifact(
                tmp_path / "barrier/inference.json",
                dict(
                    **identity,
                    lane=lane,
                    step_id=step,
                    dispatch_nonce="wrong" if damage == "nonce" else lane,
                ),
            )
            if damage == "settled":
                attempts[str(attempt)]["status"] = "settled"
    write_artifact(tmp_path / "ledger.json", dict(attempts=attempts))
    manifest = dict(
        **identity,
        ownership_lane="native",
        dispatch_nonce="native",
        ledger_path=str(tmp_path / "ledger.json"),
        concurrent_barrier=dict(
            directory=str(tmp_path / "barrier"), maximum_seconds=0.01, peers=peers
        ),
    )
    if damage:
        with pytest.raises(TimeoutError if damage == "missing" else ValueError):
            wait_for_peer(manifest)
    else:
        wait_for_peer(manifest)
