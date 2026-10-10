"""Meaning, split, native-failure and token gates for the controlled successor."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools.repair import grounded_supervision as worker
from tools.repair.corpus import generate_corpus
from tests.repair_semantic_fidelity_test import packet


def case(family="papers", sibling=0):
    return generate_corpus(
        families=(family,),
        seed=13,
        parents_per_family=1,
        siblings_per_parent=2,
        split_counts={"train": 1, "development": 0, "test": 0},
    )[sibling]


@pytest.mark.parametrize("family", tuple(worker.SETTINGS))
def test_controlled_setting_preserves_inputs_and_does_not_turn_queries_into_truth(family):
    item = case(family)
    before = item.problem.content_hash
    source = dict(path="source.json", sha256="a" * 64)
    result = worker.controlled_evidence(item, source)
    altered = worker.controlled_evidence(replace(item, probes=()), source)
    assert result["evidence"] == altered["evidence"]
    assert item.problem.content_hash == before == result["input_hash"]
    assert result["independent_semantic_evidence"] is False
    assert all(r["symbolic_value"] is None for r in result["evidence"].values())
    assert "Intended editable commitments" in result["evidence"]["D2"]["text"]


@pytest.mark.parametrize(
    "change",
    [
        dict(split="test"),
        dict(split="development"),
        dict(origin="conference"),
        dict(family="unknown"),
        dict(intended_theory=()),
    ],
)
def test_refuses_unqualified_sources_and_heldout_semantics(change):
    with pytest.raises(ValueError):
        worker.controlled_evidence(replace(case(), **change), dict(sha256="a" * 64))


def test_native_unavailable_receipt_is_reused_without_favorable_replay(tmp_path, monkeypatch):
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            status="timeout", detail="bounded", cleanup_complete=True, value=None
        )

    monkeypatch.setattr(worker, "bounded_call", unavailable)
    monkeypatch.setattr(worker.time, "time", lambda: 1000)
    monkeypatch.setenv("EXACT_REPAIR_DEADLINE_EPOCH", "1018")
    item = case()
    first = worker.native_plan(item, [0], tmp_path, 1040, 30)
    second = worker.native_plan(item, [0], tmp_path, 1040, 30)
    assert first == second and first[0] is None
    assert len(calls) == 1 and calls[0]["timeout"] == 13


def test_expired_case_never_starts_native_work(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.time, "time", lambda: 1000)
    monkeypatch.setattr(
        worker, "bounded_call", lambda *a, **k: pytest.fail("started expired child")
    )
    assert worker.native_plan(case(), [0], tmp_path, 1006, 30) == (None, None)


def test_oversize_packet_stays_complete_and_unavailable(monkeypatch):
    monkeypatch.setattr(worker, "request_profile", lambda *a: dict(reasoning=None, tokenizer={}))

    def oversized(*args):
        raise ValueError("Annotation input exceeds independent byte cap")

    monkeypatch.setattr(worker, "input_token_bound", oversized)
    item = packet()
    original = item.content_hash
    result = worker.packet_admission(
        item,
        dict(
            profile="teacher", prompt_version="semantic-fidelity-prompt/v3.2", lineage_id="campaign"
        ),
        False,
    )
    assert result["status"] == "unavailable_packet_bytes"
    assert item.content_hash == original


def test_tokenizer_integrity_error_is_not_scientific_unavailability(monkeypatch):
    monkeypatch.setattr(worker, "request_profile", lambda *a: dict(reasoning=None, tokenizer={}))

    def corrupt(*args):
        raise ValueError("Pinned annotation tokenizer changed")

    monkeypatch.setattr(worker, "input_token_bound", corrupt)
    with pytest.raises(ValueError, match="tokenizer changed"):
        worker.packet_admission(
            packet(),
            dict(
                profile="teacher",
                prompt_version="semantic-fidelity-prompt/v3.2",
                lineage_id="campaign",
            ),
            False,
        )


def test_full_audit_keeps_308_rows_and_resumes_without_native_replay(tmp_path, monkeypatch):
    from tools.repair.historical_regression import binding
    from tools.repair.shared_release import immutable
    from exact.repair.records import canonical_hash

    def save(name, value):
        path = tmp_path / name
        immutable(path, value)
        return binding(path)

    original = case()
    cases = [
        replace(original, case_id=f"case-{i}", structural_parent=f"parent-{i}") for i in range(128)
    ]
    rows = []
    for item in cases:
        for index in range(2):
            rows.append(
                dict(
                    id=f"{item.case_id}-{index}",
                    case_id=item.case_id,
                    comparison_index=index,
                    structural_parent=item.structural_parent,
                    family=item.family,
                    control=item.control,
                    case_input_hash=item.problem.content_hash,
                    swapped=False,
                    pair=None,
                    status="unavailable_distinct_verified_plans",
                )
            )
    assignments = [[0], [1]]
    rows[0]["pair"] = dict(
        assignments=assignments,
        candidate_ids=[
            [
                [o.object_id, o.candidates[i].candidate_id]
                for o, i in zip(original.problem.objects, a)
            ]
            for a in assignments
        ],
    )
    for row in rows[:52]:
        rows.append(dict(row, id=row["id"] + "-swap", original_id=row["id"], swapped=True))
    shards = [
        save(f"prior/shard-{i}.json", dict(rows=rows[i * 154 : (i + 1) * 154])) for i in range(2)
    ]
    report = save("prior/report.json", dict(shards=shards))
    completion = save("complete.json", dict(audit_receipt={}, report=report))
    template = save(
        "template.json",
        dict(profile="teacher", slots=[], authorized=False, request_limits={"train": 325}),
    )
    base = save("base.json", dict(common_release={}, audit_run={}))
    predecessor = save(
        "predecessor.json", dict(predecessor=base, annotation_templates={"train": template})
    )
    marker = tmp_path / "code.txt"
    marker.write_text("immutable source")
    source = save(
        "source.json",
        dict(
            settings=worker.SETTINGS,
            glossary=worker.GLOSSARY,
            revision=worker.REVISION,
            scope=worker.SCOPE,
            constructor=binding(marker),
            worker=binding(marker),
        ),
    )
    prepared = save(
        "prepared.json",
        dict(
            predecessor=predecessor,
            predecessor_completion=completion,
            controlled_source=source,
            deadline_epoch=worker.time.time() + 1000,
            pair_seconds=65,
            case_seconds=130,
            plan_seconds=30,
        ),
    )
    monkeypatch.setattr(
        worker,
        "validate_completion",
        lambda _: (dict(work=str(tmp_path / "prior")), {"report.json": report["sha256"]}, {}, {}),
    )
    monkeypatch.setattr(worker, "load_release", lambda *args: (cases, {}, {}))
    monkeypatch.setattr(worker, "native_plan", lambda *args: (None, None))
    result = worker.audit(prepared["path"], tmp_path / "output")
    assert result["scheduled"] == 308 and result["unique_comparisons"] == 256
    assert result["statuses"]["unavailable_native_plan"] == 2
    assert result["hosted_calls"] == 0 and result["primary_fitting_admitted"] is False
    monkeypatch.setattr(worker, "native_plan", lambda *args: pytest.fail("replayed saved row"))
    assert worker.audit(prepared["path"], tmp_path / "output") == result
    saved = worker.read(tmp_path / "output/rows" / (canonical_hash(rows[-1]["id"]) + ".json"))
    assert saved["comparison_id"] == rows[-1]["original_id"]
