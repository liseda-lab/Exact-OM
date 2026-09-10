from __future__ import annotations

import json
from pathlib import Path

import pytest

from exact.core.entities.kinds import EntityKind
from exact.experiments.paper_metrics import (
    SourceConfusion,
    SourceEvaluation,
    e17_interaction_bootstrap,
    extract_evaluation_metrics,
    holm_adjust_p_values,
    paired_global_f1_bootstrap,
    recompute_global_prf,
)
from exact.impl.evaluators.builtin import BuiltinEvaluator
from exact.utils.provenance import file_provenance


def _write_table(path: Path, header: str, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows, ""]), encoding="utf-8")
    return path


def _write_report(
    run: Path,
    *,
    builtin: dict[str, float],
    alignment: Path,
    reference: Path,
    train_reference: Path | None = None,
) -> Path:
    refs = {
        "alignment": file_provenance(alignment),
        "full_reference": file_provenance(reference),
    }
    if train_reference is not None:
        refs["train_reference"] = file_provenance(train_reference)
    report = {
        "builtin": builtin,
        "meta": {"refs": refs, "k": [1, 5, 10], "numeric_trap": 99},
    }
    path = run / "evaluation" / "evaluation_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_extract_evaluation_metrics_reads_only_direct_authoritative_json(
    tmp_path: Path,
) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    (evaluation / "evaluation_results.json").write_text(
        json.dumps(
            {
                "builtin": {
                    "P": 0.4,
                    "R": 0.5,
                    "F1": 0.444,
                    "class.F1": 0.5,
                    "debug": {"number": 123},
                },
                "meta": {
                    "refs": {"full_reference": {"path": "unused"}},
                    "k": [1, 5, 10],
                    "numeric_trap": 7,
                },
            }
        ),
        encoding="utf-8",
    )
    local = evaluation / "local"
    local.mkdir()
    (local / "evaluation_results.json").write_text(
        json.dumps(
            {
                "builtin": {"Hits@1": 0.25, "MRR": 0.5},
                "meta": {
                    "refs": {"reference_candidates": {"path": "unused"}},
                    "k": [1],
                },
            }
        ),
        encoding="utf-8",
    )
    (evaluation / "evaluation_results.csv").write_text("Metric,Value\nF1,999\n", encoding="utf-8")
    (tmp_path / "run_stats.json").write_text('{"F1": 888}', encoding="utf-8")
    (tmp_path / "metrics.json").write_text('{"F1": 777}', encoding="utf-8")

    assert extract_evaluation_metrics(tmp_path) == {
        "P": 0.4,
        "R": 0.5,
        "F1": 0.444,
        "class.F1": 0.5,
        "local.Hits@1": 0.25,
        "local.MRR": 0.5,
    }


def test_extract_evaluation_metrics_preserves_multi_backend_namespace_and_finiteness(
    tmp_path: Path,
) -> None:
    report = tmp_path / "evaluation_results.json"
    payload = {
        "builtin": {"P": 0.8, "R": 0.75, "F1": 0.774},
        "bioml": {"equivalence.f1": 0.7, "details": {"count": 10}},
        "meta": {"refs": {"full_reference": {"path": "unused"}}},
    }
    report.write_text(json.dumps(payload), encoding="utf-8")

    assert extract_evaluation_metrics(report) == {
        "bioml.equivalence.f1": 0.7,
        "builtin.F1": 0.774,
        "builtin.P": 0.8,
        "builtin.R": 0.75,
    }

    payload["builtin"]["F1"] = float("nan")
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite metric builtin.F1"):
        extract_evaluation_metrics(report)


def test_recompute_global_prf_matches_builtin_and_emits_explicit_typed_slices(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    alignment = _write_table(
        run / "alignment" / "paper.maps_global.tsv",
        "SrcEntity\tTgtEntity\tScore\tRelation\tSrcKind\tTgtKind",
        [
            "s1\tt1\t0.9\t<\tclass\tclass",
            "s2\tt2\t0.9\t=\tclass\tclass",
            "s3\tt3\t0.9\t<\tclass\tclass",
            "i1\twrong\t0.9\t=\tindividual\tindividual",
            "x\ty\t0.9\t=\tclass\tclass",
            "s0\tt0\t0.9\t=\tclass\tclass",
        ],
    )
    reference = _write_table(
        tmp_path / "full.tsv",
        "SrcEntity\tTgtEntity\tRelation\tSrcKind\tTgtKind",
        [
            "s1\tt1\t=\tclass\tclass",
            "s2\tt2\t=\tclass\tclass",
            "s3\tt3\t<\tclass\tclass",
            "i1\tj1\t=\tindividual\tindividual",
        ],
    )
    train = _write_table(
        tmp_path / "train.tsv",
        "SrcEntity\tTgtEntity\tRelation\tSrcKind\tTgtKind",
        ["s0\tt0\t=\tclass\tclass"],
    )
    builtin = BuiltinEvaluator.global_eval(
        predictions=alignment,
        test_reference=reference,
        train_reference=train,
        entity_kinds=[EntityKind.CLASS, EntityKind.INDIVIDUAL],
    )
    assert builtin["F1"] == 0.667
    _write_report(
        run,
        builtin={key: builtin[key] for key in ("P", "R", "F1")},
        alignment=alignment,
        reference=reference,
        train_reference=train,
    )
    _write_table(
        run / "alignment" / "maps_global.tsv",
        "SrcEntity\tTgtEntity\tScore",
        ["wrong\tpair\t0.1"],
    )

    recomputed = recompute_global_prf(
        run,
        slices=[
            (EntityKind.CLASS, "="),
            (EntityKind.CLASS, "<"),
            (EntityKind.INDIVIDUAL, "equivalence"),
        ],
    )

    assert recomputed.overall.metrics.as_dict() == pytest.approx(
        {"tp": 3, "fp": 2, "fn": 1, "P": 0.6, "R": 0.75, "F1": 2 / 3}
    )
    assert recomputed.slices["class|equivalence"].metrics.as_dict() == pytest.approx(
        {"tp": 1, "fp": 1, "fn": 1, "P": 0.5, "R": 0.5, "F1": 0.5}
    )
    assert recomputed.slices["class|source_subsumed_by_target"].metrics.as_dict() == pytest.approx(
        {"tp": 1, "fp": 1, "fn": 0, "P": 0.5, "R": 1.0, "F1": 2 / 3}
    )
    assert recomputed.slices["individual|equivalence"].metrics.as_dict() == pytest.approx(
        {"tp": 0, "fp": 1, "fn": 1, "P": 0.0, "R": 0.0, "F1": 0.0}
    )
    assert recomputed.overall.by_source["s1"].tp == 1
    assert recomputed.slices["class|equivalence"].by_source["s1"].fn == 1

    reference.write_text(reference.read_text(encoding="utf-8") + "new\trow\t=\tclass\tclass\n")
    with pytest.raises(ValueError, match="full reference SHA-256 mismatch"):
        recompute_global_prf(run)


def test_typed_slices_fail_closed_when_maps_global_has_no_dimensions(tmp_path: Path) -> None:
    run = tmp_path / "run"
    alignment = _write_table(
        run / "alignment" / "maps_global.tsv",
        "SrcEntity\tTgtEntity\tScore",
        ["s1\tt1\t0.9"],
    )
    reference = _write_table(tmp_path / "full.tsv", "SrcEntity\tTgtEntity", ["s1\tt1"])
    _write_report(
        run,
        builtin={"P": 1.0, "R": 1.0, "F1": 1.0},
        alignment=alignment,
        reference=reference,
    )

    assert recompute_global_prf(run).overall.metrics.f1 == 1.0
    with pytest.raises(ValueError, match="explicit Relation column"):
        recompute_global_prf(run, slices=[("class", "=")])


def _source_evaluation(
    counts: dict[str, SourceConfusion],
    *,
    reference_hash: str = "a" * 64,
    alignment_hash: str = "b" * 64,
) -> SourceEvaluation:
    return SourceEvaluation(
        reference_sha256=reference_hash,
        alignment_sha256=alignment_hash,
        by_source=counts,
    )


def test_paired_bootstrap_recomputes_global_f1_and_is_deterministic() -> None:
    baseline = {
        ("task", 17): _source_evaluation(
            {"A": SourceConfusion(9, 0, 1), "B": SourceConfusion(0, 10, 0)}
        )
    }
    candidate = {
        ("task", 17): _source_evaluation(
            {"A": SourceConfusion(8, 0, 2), "B": SourceConfusion(0, 0, 0)},
            alignment_hash="c" * 64,
        )
    }

    first = paired_global_f1_bootstrap(baseline, candidate, resamples=500, seed=23)
    second = paired_global_f1_bootstrap(baseline, candidate, resamples=500, seed=23)

    expected = 8 / 9 - 18 / 29
    assert first.delta == pytest.approx(expected)
    assert first.arm_scores == pytest.approx({"baseline": 18 / 29, "candidate": 8 / 9})
    assert first.as_dict() == second.as_dict()
    assert 0.0 < first.p_value <= 1.0
    assert first.reference_sha256 == "a" * 64
    assert first.paired_seeds == (17,)
    assert first.n_units == 2
    source_macro_delta = ((0.8 + 0.0) / 2) - ((18 / 19 + 0.0) / 2)
    assert first.delta != pytest.approx(source_macro_delta)


def test_bootstrap_uses_task_macro_and_rejects_unpaired_designs() -> None:
    def cell(tp: int, fp: int, fn: int, reference_hash: str, alignment: str) -> SourceEvaluation:
        return _source_evaluation(
            {"source": SourceConfusion(tp, fp, fn)},
            reference_hash=reference_hash,
            alignment_hash=alignment * 64,
        )

    baseline = {
        ("task-1", 1): cell(5, 5, 5, "1" * 64, "a"),
        ("task-1", 2): cell(7, 3, 3, "1" * 64, "b"),
        ("task-2", 1): cell(9, 1, 1, "2" * 64, "c"),
        ("task-2", 2): cell(9, 1, 1, "2" * 64, "d"),
    }
    candidate = {
        ("task-1", 1): cell(7, 3, 3, "1" * 64, "e"),
        ("task-1", 2): cell(9, 1, 1, "1" * 64, "f"),
        ("task-2", 1): cell(8, 2, 2, "2" * 64, "7"),
        ("task-2", 2): cell(8, 2, 2, "2" * 64, "8"),
    }

    result = paired_global_f1_bootstrap(baseline, candidate, resamples=50, seed=5)

    assert result.arm_scores == pytest.approx({"baseline": 0.75, "candidate": 0.8})
    assert result.delta == pytest.approx(0.05)
    assert result.n_tasks == 2
    assert result.n_cells == 4
    assert result.n_units == 2
    assert result.paired_seeds == (1, 2)
    assert result.reference_sha256 is None
    assert result.as_dict()["p_value_adjustment"] == "none"

    missing_seed = dict(candidate)
    missing_seed.pop(("task-2", 2))
    with pytest.raises(ValueError, match="unequal task/seed cells"):
        paired_global_f1_bootstrap(baseline, missing_seed, resamples=10)

    wrong_reference = dict(candidate)
    wrong_reference[("task-1", 1)] = cell(7, 3, 3, "9" * 64, "e")
    with pytest.raises(ValueError, match="unequal evaluation reference hashes"):
        paired_global_f1_bootstrap(baseline, wrong_reference, resamples=10)


def test_e17_interaction_and_holm_adjustment() -> None:
    def arm(tp: int, token: str) -> dict[tuple[str, int], SourceEvaluation]:
        return {
            ("task", 3): _source_evaluation(
                {"source": SourceConfusion(tp, 10 - tp, 10 - tp)},
                alignment_hash=token * 64,
            )
        }

    result = e17_interaction_bootstrap(
        arm(5, "1"),
        arm(6, "2"),
        arm(7, "3"),
        arm(9, "4"),
        resamples=100,
        seed=11,
    )

    assert result.delta == pytest.approx(0.1)
    assert result.ci_low == pytest.approx(0.1)
    assert result.ci_high == pytest.approx(0.1)
    assert result.p_value == pytest.approx(1 / 101)
    assert holm_adjust_p_values({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx(
        {"a": 0.03, "b": 0.06, "c": 0.06}
    )


def test_frozen_population_keeps_empty_groups_and_rejects_changed_membership(tmp_path):
    import hashlib

    run = tmp_path / "run"
    alignment = _write_table(
        run / "alignment/maps_global.tsv", "SrcEntity\tTgtEntity\tScore", ["s1\tt1\t1"]
    )
    reference = _write_table(tmp_path / "reference.tsv", "SrcEntity\tTgtEntity", ["s1\tt1"])
    _write_report(run, builtin={"P": 1, "R": 1, "F1": 1}, alignment=alignment, reference=reference)
    dataset = run / "dataset"
    dataset.mkdir()
    groups = [["empty", "class"], ["s1", "class"]]
    identity = hashlib.sha256("empty\tclass\ns1\tclass".encode()).hexdigest()
    sample = {
        "eligible_source_iris": ["empty", "s1"],
        "source_kind_groups": groups,
        "sha256": identity,
    }
    path = dataset / "candidate_pool_sample_manifest.json"
    path.write_text(json.dumps({"retrieval_config": {"source_sample": sample}}))
    result = recompute_global_prf(run).overall
    assert result.by_source["empty"] == SourceConfusion()
    assert result.source_universe_sha256 == identity
    assert result.metrics.f1 == 1
    sample["eligible_source_iris"] = ["s1"]
    path.write_text(json.dumps({"retrieval_config": {"source_sample": sample}}))
    with pytest.raises(ValueError, match="integrity"):
        recompute_global_prf(run)
