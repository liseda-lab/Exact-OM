"""Closed-cohort entry point and matched schedule checks; no hosted/native calls."""

from dataclasses import replace
from pathlib import Path

import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from exact.repair.protocol import load_protocol_v3, RepairProtocolV3
from tools.repair import common_training as common, train
from tools.repair.historical_regression import binding
from tools.repair.prepare import case_to_dict, load_preparation, publish_label_cache
from tests.repair_training_completion_test import cache_for
from tools.repair.corpus import generate_corpus


def fixture(tmp_path):
    cases = generate_corpus(
        split_counts={"train": 2, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
        revision="v3",
    )
    rows, dev = [], []
    for index, case in enumerate(cases):
        path = tmp_path / f"input-{index}.json"
        write_artifact(path, case_to_dict(case))
        declaration = dict(
            case_id=case.case_id,
            family=case.family,
            control=case.control,
            structural_parent=case.structural_parent,
            split=case.split,
            evaluator=binding(path),
            input_hash=case.problem.content_hash,
        )
        if case.split == "development":
            dev.append(dict(declaration, cache=None))
            continue
        cache = replace(cache_for(case), schema="exact-repair/teacher-cache/v3")
        cached = index == 0
        row = dict(
            declaration,
            input=declaration,
            generated=None,
            cache=publish_label_cache(cache, tmp_path / "caches") if cached else None,
            collection_dependencies=dict(
                hashes=dict(cache.hashes), candidate_counts=list(cache.candidate_counts)
            ),
            attempts=[dict(order=i, status="unavailable") for i in range(16)],
            scheduled_attempts=16,
            scientific_status="collected" if cached else "no_committed_result",
            process_status="complete" if cached else "timeout",
        )
        row_path = tmp_path / f"row-{index}.json"
        write_artifact(row_path, row)
        rows.append(binding(row_path))
    for name, value in dict(
        shard=dict(shared_conditions=list(common.CONDITIONS), rows=rows),
        dev=dict(rows=dev),
        coverage=dict(overall=dict(cases=2)),
        prior=dict(runs=[]),
    ).items():
        write_artifact(tmp_path / f"{name}.json", value)
    release = dict(
        schema="exact-repair/common-generated-shared-release/v1",
        status="complete",
        acquisition_closed=True,
        maximum_rounds=2,
        heldout_outcomes_opened=False,
        prior_release=binding(tmp_path / "prior.json"),
        refinement_runs=[],
        common={},
        coverage=binding(tmp_path / "coverage.json"),
        shards=[binding(tmp_path / "shard.json")],
        development=binding(tmp_path / "dev.json"),
        expected_train_cases=2,
        expected_development_cases=1,
        expected_assignment_slots=32,
        cohort="generated_only",
        missing_real_coverage=True,
    )
    write_artifact(tmp_path / "report.json", release)
    for name, value in dict(
        batch=dict(jobs=[dict(id="audit")]),
        step=dict(step_id="14451.45", dispatch_nonce="nonce"),
        completion=dict(
            status="complete",
            exit_code=0,
            step_id="14451.45",
            dispatch_nonce="nonce",
            job_id="audit",
            work=str(tmp_path),
            batch=str(tmp_path / "batch.json"),
        ),
        outputs={"report.json": binding(tmp_path / "report.json")["sha256"]},
    ).items():
        write_artifact(tmp_path / f"{name}.json", value)
    run = {
        key: binding(tmp_path / f"{key}.json") for key in ("batch", "step", "completion", "outputs")
    }
    run.update(step_id="14451.45", dispatch_nonce="nonce", expected_status="complete")
    protocol = load_protocol_v3(
        Path("specs/exact-repair/protocol/xr21-review2-conformance.json")
    ).model_dump(by_alias=True)
    protocol["training"].update(
        development_case_ids=[c.case_id for c in cases if c.split == "development"],
        sampled_assignments=0,
        device="cpu",
    )
    protocol["resources"]["allocated_gpus"] = 0
    write_artifact(tmp_path / "protocol.json", protocol)
    value = dict(
        schema=common.SCHEMA,
        release=binding(tmp_path / "report.json"),
        audit_run=run,
        protocol=binding(tmp_path / "protocol.json"),
    )
    write_artifact(tmp_path / "prepared.json", dict(value, hash=canonical_hash(value)))
    return cases, value, protocol


def test_normal_loader_preserves_missing_train_and_all_dev(tmp_path):
    cases, value, _ = fixture(tmp_path)
    restored, caches, report = load_preparation(tmp_path / "prepared.json")
    assert {c.case_id: c for c in restored} == {c.case_id: c for c in cases}
    assert len(caches) == 1 and len(report["label_rows"]) == 3
    assert report["common_acquisition_closed"] and report["expected_assignment_slots"] == 32
    assert len(train.scheduled_development(restored, caches)) == 1
    assert report["label_seconds"] == 0 and "linked once" in report["historical_label_costs"]
    assert any(r.get("process_status") == "timeout" for r in report["label_rows"])


def test_nested_tamper_and_unbound_receipt_rejected(tmp_path):
    _, value, _ = fixture(tmp_path)
    run = dict(value["audit_run"], dispatch_nonce="wrong")
    with pytest.raises(ValueError, match="nonce"):
        common.load_release(value["release"], run)
    path = tmp_path / "row-0.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="dependency changed"):
        load_preparation(tmp_path / "prepared.json")


@pytest.mark.parametrize("option", ["subset", "reacquire", "retry", "missing_row", "dev_order"])
def test_closed_release_cannot_change_data_between_conditions(tmp_path, option):
    _, _, protocol = fixture(tmp_path)
    cases, _, report = load_preparation(tmp_path / "prepared.json")
    limit, retry = None, False
    if option == "subset":
        limit = 1
    if option == "retry":
        retry = True
    if option == "reacquire":
        protocol["training"]["sampled_assignments"] = 1
    if option == "missing_row":
        report["label_rows"].pop()
    if option == "dev_order":
        protocol["training"]["development_case_ids"] = []
    with pytest.raises(ValueError):
        common.validate_closed_preparation(
            cases, report, protocol, case_limit=limit, retry_labels=retry
        )


def test_prepare_only_entry_point_never_relabels_closed_missing_rows(tmp_path, monkeypatch):
    _, value, protocol = fixture(tmp_path)
    # Explicitly enable this tiny local fixture, not a campaign protocol.
    protocol["identity"]["execution_authorized"] = True

    def resolved(value):
        if isinstance(value, dict):
            return {k: resolved(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolved(v) for v in value]
        return "fixture" if isinstance(value, str) and value.startswith("UNFROZEN") else value

    protocol = resolved(protocol)
    write_artifact(tmp_path / "protocol.json", protocol)
    value["protocol"] = binding(tmp_path / "protocol.json")
    write_artifact(tmp_path / "prepared.json", dict(value, hash=canonical_hash(value)))
    monkeypatch.setattr(train, "_label_payload", lambda *a, **k: pytest.fail("reacquisition"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--protocol",
            str(tmp_path / "protocol.json"),
            "--prepared",
            str(tmp_path / "prepared.json"),
            "--output",
            str(tmp_path / "output"),
            "--prepare-only",
        ],
    )
    assert train.main() == 0
    _, caches, report = load_preparation(tmp_path / "output/preparation.json")
    assert len(caches) == 1 and len(report["label_rows"]) == 3


def dev_rows():
    return [
        dict(
            case_id=f"case-{i:02d}",
            family=f"family-{i // 4}",
            control="coherent" if i % 2 else "corrupted",
            structural_parent=f"parent-{i // 2}",
            split="development",
        )
        for i in range(32)
    ]


def test_dev_schedule_balanced_slots_independent_swaps_and_reserves():
    schedule = common.development_schedule(dev_rows())
    rows = schedule["rows"]
    assert len(rows) == 384 and schedule["unique_semantic_slots"] == 64
    assert schedule["independent_swapped_calls"] == 14
    assert schedule["case_worker_seconds"] == 58988
    assert 78 * 90 < 4 * 3600
    for seed in (13, 37, 73):
        for epoch in (5, 50):
            a, b = [
                [
                    r
                    for r in rows
                    if r["selection_slot"]["seed"] == seed
                    and r["selection_slot"]["epoch"] == epoch
                    and r["selection_slot"]["supervision_condition"] == c
                ]
                for c in common.CONDITIONS
            ]
            assert len(a) == len(b) == 32
            assert [(r["case_seconds"], r["swapped"], r["semantic_scheduled"]) for r in a] == [
                (r["case_seconds"], r["swapped"], r["semantic_scheduled"]) for r in b
            ]
    assert len({r["selection_slot"]["case_id"] for r in rows if r["semantic_scheduled"]}) == 32
    assert (
        common.development_schedule(list(reversed(dev_rows())))["frozen_annotation_slots"]
        == schedule["frozen_annotation_slots"]
    )


def test_protocol_rejects_missing_dev_limit_and_preserves_old_identity():
    p = load_protocol_v3(Path("specs/exact-repair/protocol/xr21-review2-conformance.json"))
    assert "development_case_seconds" not in p.model_dump(by_alias=True)["training"]
    value = p.model_dump(by_alias=True)
    value["training"].update(
        development_epochs=[1],
        max_full_development_evaluations=1,
        development_case_ids=["dev"],
        development_case_seconds={},
    )
    with pytest.raises(ValueError, match="every scheduled slot"):
        RepairProtocolV3.model_validate(value)


def test_interrupted_dev_case_keeps_its_deadline_on_resume(tmp_path, monkeypatch):
    import torch
    from exact.repair.graph_schema import generic_graph_schema

    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
        revision="v3",
    )
    fitting = [(c, cache_for(c)) for c in cases if c.split == "train"]
    dev = train.scheduled_development(cases, {})
    key = canonical_hash(
        dict(seed=13, supervision_condition="symbolic", epoch=1, case_id=dev[0][0].case_id)
    )
    calls = []
    monkeypatch.setattr(train, "decoded_development", lambda *a, **k: dict(exact_regret=None))

    def generated(*args, seconds, **kwargs):
        calls.append(seconds)
        raise InterruptedError("worker interruption within DEV")

    monkeypatch.setattr(train, "generated_development", generated)
    options = dict(
        encoder="hgt",
        graph_schema=generic_graph_schema(),
        hidden_dim=8,
        heads=2,
        layers=1,
        dropout=0,
        pairwise=True,
        epochs=1,
        development_epochs=[1],
        development_case_ids=[dev[0][0].case_id],
        max_full_development_evaluations=1,
        patience_enabled=False,
        proposal_arm="bounded_enumeration",
        sampled_assignments=0,
        seed=13,
        deadline_seconds=90,
        total_training_seconds=180,
        decode_seconds=10,
        development_case_seconds={key: 10},
        checkpoint_path=tmp_path / "state.pt",
    )
    with pytest.raises(InterruptedError):
        train.train_cases(fitting, dev, **options)
    saved = torch.load(options["checkpoint_path"], weights_only=True)
    deadline = saved["development_progress"]["case_deadlines_epoch"][dev[0][0].case_id]
    assert 0 < calls[0] <= 10
    monkeypatch.setattr(train.time, "time", lambda: deadline + 1)
    with pytest.raises(ValueError, match="No eligible development"):
        train.train_cases(fitting, dev, **options)
    assert len(calls) == 1
    saved = torch.load(options["checkpoint_path"], weights_only=True)
    assert (
        saved["history"][0]["generated"][dev[0][0].case_id]["status"] == "case_budget_unavailable"
    )
