from pathlib import Path

import pandas as pd

from exact.analysis.alignment_diagnostics import analyze_alignment_run


def test_alignment_diagnostics_reports_oracle_and_miss_buckets(tmp_path: Path):
    run_dir = tmp_path / "run"
    summary_dir = run_dir / "model" / "alignment" / "default"
    align_dir = run_dir / "model" / "alignment"
    summary_dir.mkdir(parents=True)
    align_dir.mkdir(parents=True, exist_ok=True)

    reference_path = tmp_path / "ref.tsv"
    reference_path.write_text(
        "\n".join(
            [
                "SrcEntity\tTgtEntity\tScore",
                "s1\tt1\t1.0",
                "s2\tt2\t1.0",
                "s3\tt3\t1.0",
                "s4\tt4\t1.0",
            ]
        ),
        encoding="utf-8",
    )
    alignment_path = align_dir / "src2tgt.maps_global.tsv"
    alignment_path.write_text(
        "\n".join(
            [
                "SrcEntity\tTgtEntity\tScore",
                "s1\tt1\t0.9",
                "s2\tt_wrong\t0.8",
                "s5\tt5\t0.7",
            ]
        ),
        encoding="utf-8",
    )
    summary_path = summary_dir / "summary_metrics.csv"
    pd.DataFrame(
        [
            {
                "src_iri": "s1",
                "tgt_iri": "t1",
                "saved_alignment_member": True,
                "selector_abstained": False,
                "selector_llm_used": False,
                "selector_reason": "calibrated",
                "S_pair_final": 0.9,
                "selection_utility": 0.9,
            },
            {
                "src_iri": "s2",
                "tgt_iri": "t2",
                "saved_alignment_member": False,
                "selector_abstained": False,
                "selector_llm_used": False,
                "selector_reason": "calibrated_no_match",
                "S_pair_final": 0.7,
                "selection_utility": 0.7,
            },
            {
                "src_iri": "s2",
                "tgt_iri": "t_wrong",
                "saved_alignment_member": True,
                "selector_abstained": False,
                "selector_llm_used": True,
                "selector_reason": "llm",
                "S_pair_final": 0.8,
                "selection_utility": 0.8,
            },
            {
                "src_iri": "s3",
                "tgt_iri": "t3",
                "saved_alignment_member": False,
                "selector_abstained": True,
                "selector_llm_used": False,
                "selector_reason": "calibrated_no_match",
                "S_pair_final": 0.6,
                "selection_utility": 0.6,
            },
        ]
    ).to_csv(summary_path, sep="\t", index=False)

    diagnostics = analyze_alignment_run(run_dir=run_dir, reference_path=reference_path)

    assert diagnostics["counts"]["true_positive_pairs"] == 1
    assert diagnostics["counts"]["false_positive_pairs"] == 2
    assert diagnostics["counts"]["false_negative_pairs"] == 3
    assert diagnostics["oracle"]["missed_present_in_candidates"] == 2
    assert diagnostics["oracle"]["missed_absent_from_candidates"] == 1
    assert diagnostics["miss_buckets"]["candidate_absent"] == 1
    assert diagnostics["miss_buckets"]["present_wrong_selected"] == 1
    assert diagnostics["miss_buckets"]["present_abstained"] == 1
    assert diagnostics["llm"]["selected_fp"] == 1
