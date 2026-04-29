from pathlib import Path

import pandas as pd

from exact.analysis.candidate_recall import (
    absent_gold_dataframe,
    analyze_candidate_recall,
    flatten_candidate_recall,
    write_absent_gold_tsv,
)


def test_candidate_recall_excludes_train_and_counts_exact_oracle():
    candidates = pd.DataFrame(
        [
            {"Src": "s1", "Tgt": "t1", "cand_sim": 0.9},
            {"Src": "s1", "Tgt": "t_alt", "cand_sim": 0.8},
            {"Src": "s2", "Tgt": "t_wrong", "cand_sim": 0.7},
            {"Src": "s2", "Tgt": "t2", "cand_sim": 0.6},
            {"Src": "s_train", "Tgt": "t_train", "cand_sim": 1.0},
        ]
    )
    reference_pairs = {
        ("s1", "t1"),
        ("s2", "t2"),
        ("s3", "t3"),
        ("s4", "t4"),
        ("s_train", "t_train"),
    }
    train_pairs = {("s_train", "t_train")}
    exact_pairs = {("s4", "t4"), ("s_train", "t_train")}

    analysis = analyze_candidate_recall(candidates, reference_pairs, train_pairs=train_pairs, exact_pairs=exact_pairs)

    assert analysis["counts"]["reference_pairs"] == 4
    assert analysis["counts"]["candidate_pairs"] == 4
    assert analysis["counts"]["generated_hits"] == 2
    assert analysis["counts"]["oracle_hits"] == 3
    assert analysis["counts"]["absent_gold_pairs"] == 2
    assert analysis["counts"]["absent_gold_pairs_after_exact"] == 1
    assert analysis["metrics"]["generated_candidate_recall"] == 0.5
    assert analysis["metrics"]["exact_prefilter_oracle_recall"] == 0.75
    assert analysis["gold_rank"]["present_pairs"] == 2
    assert analysis["gold_rank"]["rank_median"] == 1.5
    assert analysis["gold_rank"]["rank_p90"] == 1.9
    assert absent_gold_dataframe(analysis).to_dict("records") == [
        {"Src": "s3", "Tgt": "t3"},
        {"Src": "s4", "Tgt": "t4"},
    ]
    assert absent_gold_dataframe(analysis, after_exact=True).to_dict("records") == [
        {"Src": "s3", "Tgt": "t3"},
    ]


def test_candidate_recall_writes_absent_gold_and_flattens(tmp_path: Path):
    analysis = analyze_candidate_recall(
        candidates=[("s1", "t1")],
        reference_pairs=[("s1", "t1"), ("s2", "t2")],
    )
    out = tmp_path / "absent.tsv"
    out_after_exact = tmp_path / "absent_after_exact.tsv"

    write_absent_gold_tsv(out, analysis)
    write_absent_gold_tsv(out_after_exact, analysis, after_exact=True)
    flat = flatten_candidate_recall(40, analysis)

    assert out.read_text(encoding="utf-8").splitlines() == ["Src\tTgt", "s2\tt2"]
    assert out_after_exact.read_text(encoding="utf-8").splitlines() == ["Src\tTgt", "s2\tt2"]
    assert flat["top_k"] == 40
    assert flat["reference_pairs"] == 2
    assert flat["generated_candidate_recall"] == 0.5
