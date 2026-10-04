"""Controlled overlap design, qualified native supports and strict descendants."""

import pytest

from tools.repair import conflict_overlap as overlap
from tools.repair import fresh_evaluation as fresh
from tools.repair.corpus import generate_corpus
from tools.repair.expanded_corpus import immutable
from tools.repair.expanded_profile import parent_fingerprints


@pytest.fixture
def parent(tmp_path):
    case = generate_corpus(
        parents_per_family=1,
        siblings_per_parent=1,
        families=("overlap",),
        split_counts=dict(train=0, development=0, test=1),
    )[0]
    return dict(
        base_case_id=case.case_id,
        case_id=case.case_id + ":robustness:baseline",
        structural_parent=case.structural_parent,
        group_id=case.structural_parent,
        family="overlap",
        control="corrupted",
        condition="baseline",
        split="fresh_evaluation",
        fingerprints=parent_fingerprints(case),
        status="materialized",
        observable=immutable(tmp_path / "parent.json", case.problem.to_dict()),
    )


def test_matched_motifs_keep_ancestry_and_change_only_one_edge(parent):
    left, left_clean = overlap.construct(parent, "nonshared")
    right, right_clean = overlap.construct(parent, "shared")
    assert left.structural_parent == right.structural_parent == parent["structural_parent"]
    assert left.split == right.split == "test"
    assert left.probes == right.probes
    assert left.intended_theory == right.intended_theory
    assert left.problem.evidence == right.problem.evidence
    assert left.problem.fixed_axioms == right.problem.fixed_axioms
    assert left.problem.objects[:3] == right.problem.objects[:3]
    assert left.problem.objects[3].original_axioms != right.problem.objects[3].original_axioms
    assert [c.cost_features for c in left.problem.objects[3].candidates] == [
        c.cost_features for c in right.problem.objects[3].candidates
    ]
    assert left_clean.control == right_clean.control == "coherent"
    assert len(left_clean.problem.objects) == len(right_clean.problem.objects) == 2
    from exact.repair.retrieval import retrieve_vocabulary

    assert retrieve_vocabulary(left_clean.problem).menus
    assert retrieve_vocabulary(right_clean.problem).menus
    assert all(
        "expected_minimal_supports" not in str(case.problem.to_dict()) for case in (left, right)
    )


def test_exposure_audit_unions_transitive_aliases_and_never_claims_heldout(parent, tmp_path):
    cases = overlap.construct(parent, "shared")
    motif = parent_fingerprints(cases[0])[0]
    inventory = immutable(
        tmp_path / "inventory.json",
        dict(
            exposed=[dict(key="historical", fingerprints=["bridge"], family="alias")],
            selected=[dict(key="train", fingerprints=[motif, "bridge"], split="train")],
        ),
    )
    audit = overlap.exposure_audit(cases, parent, dict(profile_inventory=inventory))
    assert audit["cross_split_or_historical"]
    assert {r["key"] for r in audit["matching_parents"]} == {"train", "historical"}
    assert not audit["heldout_claim"] and audit["independent_parent_count"] == 1


@pytest.fixture
def schedule(parent, tmp_path):
    inventory = immutable(
        tmp_path / "inventory.json",
        dict(
            exposed=[],
            selected=[
                dict(
                    key=parent["group_id"],
                    fingerprints=parent["fingerprints"],
                    split="fresh_evaluation",
                )
            ],
        ),
    )
    split = immutable(tmp_path / "split.json", dict(profile_inventory=inventory))
    base = dict(
        schema=fresh.SCHEMA,
        program=split,
        authorization=split,
        corpus_completion=split,
        split_schedule=split,
        cases=[parent],
        arms=[
            dict(id=str(i), status="available", kind="learned" if i < 6 else "control")
            for i in range(9)
        ],
        control_definition={},
        symbolic_limitation="heuristic",
        model_selection_source=split,
    )
    source = immutable(tmp_path / "source-schedule.json", base)
    output = tmp_path / "prepared"
    result = overlap.prepare(source, output)
    return output / "schedule.json", result


def test_prepared_denominators_and_scientific_caps_are_frozen(schedule):
    _, result = schedule
    assert len(result["rows"]) == len({r["id"] for r in result["rows"]}) == 36
    assert len(result["cases"]) == 4 and result["planned_parent_groups"] == 1
    assert {c["condition"] for c in result["cases"]} == {"shared", "nonshared"}
    assert all(
        (r["seconds"], r["cpu_seconds"], r["memory_mb"]) == (300, 600, 8192) for r in result["rows"]
    )
    assert result["generation_seconds"] == 60 and not result["test_outcomes_opened"]


def test_unknown_witnesses_preserve_all_rows_without_requery(schedule, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import exact.repair.workers as workers

    path, planned = schedule
    calls = []

    def unknown(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            status="timeout",
            detail="deadline",
            cleanup_complete=True,
            resource_usage=(),
            value=None,
        )

    monkeypatch.setattr(workers, "bounded_call", unknown)
    monkeypatch.setattr(overlap, "witness_identity", lambda _: "frozen-witness-identity")
    output = tmp_path / "witness"
    overlap.run_witness(path, output)
    overlap.run_witness(path, output)
    assert len(calls) == 4 and all(c["timeout"] == 300 for c in calls)
    effective = overlap.qualified_schedule(path, output / "report.json")
    assert effective["rows"] == planned["rows"]
    assert all(c["status"] == "unavailable_native_conflict_witness" for c in effective["cases"])
    changed = immutable(tmp_path / "changed-schedule.json", dict(planned, seed=37))
    with pytest.raises(ValueError, match="another schedule"):
        overlap.qualified_schedule(changed["path"], output / "report.json")


def test_all_unknown_witnesses_still_produce_complete_paired_denominator(
    schedule, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import exact.repair.workers as workers

    path, _ = schedule
    monkeypatch.setattr(
        workers,
        "bounded_call",
        lambda *args, **kwargs: SimpleNamespace(
            status="timeout",
            detail="deadline",
            cleanup_complete=True,
            resource_usage=(),
            value=None,
        ),
    )
    monkeypatch.setattr(overlap, "witness_identity", lambda _: "frozen-witness-identity")
    witness = tmp_path / "witness"
    overlap.run_witness(path, witness)
    overlap.evaluate(path, witness / "report.json", tmp_path / "evaluation", 0, 36)
    report = overlap.summarize(
        path,
        witness / "report.json",
        [tmp_path / "evaluation/evaluation/report.json"],
        tmp_path / "summary",
    )
    assert report["scheduled"] == report["recorded"] == 36
    assert report["outcomes"] == {"unavailable": 36}
    assert len(report["pairs"]) == 18 and not any(p["available"] for p in report["pairs"])
    with pytest.raises(ValueError, match="Incomplete overlap evaluation denominator"):
        overlap.summarize(path, witness / "report.json", [], tmp_path / "missing")


@pytest.mark.parametrize("condition", overlap.CONDITIONS)
def test_native_exact_minimal_supports_and_coherent_control(parent, tmp_path, condition):
    for case in overlap.construct(parent, condition):
        item = dict(
            case_id=case.case_id,
            control=case.control,
            observable=immutable(tmp_path / (case.control + ".json"), case.problem.to_dict()),
            expected_minimal_supports=(
                overlap.SUPPORTS[condition] if case.control == "corrupted" else ()
            ),
        )
        result = overlap.native_witness(item, tmp_path / case.control)
        assert result["qualified"], result
        assert result["actual_minimal_supports"] == sorted(item["expected_minimal_supports"])
        assert len(result["checks"]) == (16 if case.control == "corrupted" else 1)
        assert all(c["qualified"] for c in result["checks"])
