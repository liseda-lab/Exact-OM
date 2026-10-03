"""Corpus releases preserve independence, missingness and immutable resume."""

import dataclasses

import pytest

from exact.repair.records import canonical_hash
from exact.repair.workers import CallResult
from tools.repair import expanded_corpus as corpus
from tools.repair import expanded_profile as profile
from tools.repair.prepare import case_from_dict, case_to_dict, load_preparation


def parent(family="papers", depth=2, split="train"):
    case = profile.parent_case(family, depth, 13)
    return dict(
        key=case.structural_parent,
        group_id=case.structural_parent,
        family=family,
        depth=depth,
        split=split,
        fingerprints=profile.parent_fingerprints(case),
    )


@pytest.mark.parametrize("family", profile.MECHANISMS_SELECTED)
def test_new_identity_preserves_parent_and_native_pair(family, tmp_path):
    selected = parent(family, split="fresh_evaluation")
    records = corpus.materialize_pair(selected, "new-study", 73)
    rows = corpus.export_pair(records, selected, tmp_path)
    cases = [case_from_dict(record) for record in records]
    assert all(c.split == "test" and c.schema_revision == "v3" for c in cases)
    assert cases[0].structural_parent == cases[1].structural_parent
    assert cases[0].structural_parent.startswith("new-study:fresh-evaluation:")
    assert set(profile.parent_fingerprints(cases[0])) == set(selected["fingerprints"])
    corpus.audit_release_rows(rows, dict(exposed=[], selected=[selected]))
    for row in rows:
        assert not {"probes", "intended_theory", "intended_assignment", "split"} & set(
            corpus.bound(row["observable"])
        )
    assert corpus.export_pair(records, selected, tmp_path) == rows


def test_variant_cannot_cross_parent_or_split(tmp_path):
    selected = parent()
    records = corpus.materialize_pair(selected, "study", 13)
    records[1] = case_to_dict(dataclasses.replace(case_from_dict(records[1]), split="test"))
    with pytest.raises(ValueError, match="cross"):
        corpus.export_pair(records, selected, tmp_path)


def test_variant_cannot_reintroduce_exposed_core(tmp_path):
    selected = parent()
    rows = corpus.export_pair(corpus.materialize_pair(selected, "study", 13), selected, tmp_path)
    # A fingerprint introduced by a materialized clean variant must also be checked.
    rows[1]["fingerprints"] = ["old-core"]
    with pytest.raises(ValueError, match="overlaps"):
        corpus.audit_release_rows(
            rows, dict(selected=[selected], exposed=[dict(fingerprints=["old-core"])])
        )


def test_releases_and_bound_inputs_are_immutable(tmp_path):
    path = tmp_path / "split.json"
    source = corpus.immutable(path, dict(selected=["a"], missing=["b"]))
    with pytest.raises(ValueError, match="Frozen"):
        corpus.immutable(path, dict(selected=["b"], missing=["a"]))
    path.write_text("{}")
    with pytest.raises(ValueError, match="binding"):
        corpus.bound(source)


@pytest.mark.parametrize("failure", [False, True])
def test_full_denominators_resume_without_requery(tmp_path, monkeypatch, failure):
    selected = parent()
    missing = [
        dict(family="papers", split=split, status="unavailable_structural_parent")
        for split, target in corpus.TARGETS.items()
        for _ in range(target // 2 - (split == "train"))
    ]
    inventory = dict(selected=[selected], missing=missing, exposed=[])
    schedule = dict(selected=[selected], missing=missing, seed=13)
    manifest = dict()
    plan = dict(
        study="test-study",
        per_parent_seconds=30,
        per_parent_memory_mb=4096,
        followup_stages=["evaluation"],
        split_schedule=corpus.immutable(tmp_path / "split.json", schedule),
    )
    path = tmp_path / "plan.json"
    corpus.immutable(path, plan)
    monkeypatch.setattr(corpus, "validate_profile", lambda p: (manifest, inventory, {}))
    monkeypatch.setattr(corpus, "split_schedule", lambda *args: schedule)
    monkeypatch.setattr(corpus, "audit_structures", lambda *args: {})
    monkeypatch.setattr("exact.repair.study.runtime_manifest", lambda: dict(test="fixed"))
    calls = []

    def invoke(fn, *args, **kwargs):
        calls.append(args)
        return (
            CallResult("timeout", detail="recorded")
            if failure
            else CallResult("complete", value=fn(*args))
        )

    monkeypatch.setattr("exact.repair.workers.bounded_call", invoke)
    output = tmp_path / "out"
    report = corpus.run(path, output)
    assert report["scheduled_cases"] == 288
    assert sum(report["status_counts"].values()) == 288
    assert report["status_counts"]["unavailable_structural_parent"] == 286
    assert (
        report["status_counts"]["unavailable_materialization" if failure else "materialized"] == 2
    )
    for split, target in corpus.TARGETS.items():
        release = corpus.bound(report["releases"][split]["manifest"])
        assert len(release["rows"]) == target
        cases, caches, info = load_preparation(report["releases"][split]["preparation"]["path"])
        assert not caches and not info["test_outcomes_opened"]
    assert corpus.run(path, output) == report
    assert len(calls) == 1
    # Saved output corruption must be rejected, not silently regenerated.
    (output / "releases/test.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen"):
        corpus.run(path, output)


def test_changed_split_stops_before_materialization(tmp_path, monkeypatch):
    plan = dict(split_schedule=corpus.immutable(tmp_path / "split.json", dict(selected=["test"])))
    path = tmp_path / "plan.json"
    corpus.immutable(path, plan)
    monkeypatch.setattr(corpus, "validate_profile", lambda p: ({}, {}, {}))
    monkeypatch.setattr(corpus, "split_schedule", lambda *args: dict(selected=["train"]))
    with pytest.raises(ValueError, match="Frozen split"):
        corpus.run(path, tmp_path / "output")


def test_materialization_crosses_bounded_transport():
    from exact.repair.workers import bounded_call

    result = bounded_call(
        corpus.materialize_pair, parent("range"), "transport-study", 13, timeout=30, memory_mb=4096
    )
    assert result.status == "complete", result.detail
    assert result.cleanup_complete
    assert len(result.value) == 2
    assert all(case_from_dict(record).schema_revision == "v3" for record in result.value)
