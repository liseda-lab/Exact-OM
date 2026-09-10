from pathlib import Path

import pandas as pd

from exact.core.actions.alignment import _materialize_evaluation_reference


def test_candidate_reference_materializes_equivalence_and_kinds(tmp_path: Path) -> None:
    parent = tmp_path / "candidates.tsv"
    parent.write_text(
        "SrcEntity\tTgtEntity\tTgtCandidates\n" "s\tt\t['t', 'other']\n",
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        [
            {
                "Src": "s",
                "Tgt": "t",
                "Label": "['t', 'other']",
                "SrcKind": "class",
                "TgtKind": "class",
            }
        ]
    )

    output = _materialize_evaluation_reference(
        frame,
        parent_path=parent,
        output_path=tmp_path / "evaluation_inputs" / "full_reference.tsv",
    )

    assert pd.read_csv(output, sep="\t").to_dict(orient="records") == [
        {
            "SrcEntity": "s",
            "TgtEntity": "t",
            "Relation": "=",
            "SrcKind": "class",
            "TgtKind": "class",
        }
    ]


def test_typed_reference_preserves_explicit_relation(tmp_path: Path) -> None:
    parent = tmp_path / "reference.tsv"
    parent.write_text(
        "SrcEntity\tTgtEntity\tRelation\nsource\ttarget\t<\n",
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        [
            {
                "Src": "source",
                "Tgt": "target",
                "Label": "<",
                "SrcKind": "object_property",
                "TgtKind": "object_property",
            }
        ]
    )

    output = _materialize_evaluation_reference(
        frame,
        parent_path=parent,
        output_path=tmp_path / "evaluation_inputs" / "full_reference.tsv",
    )

    assert pd.read_csv(output, sep="\t")["Relation"].tolist() == ["<"]


def test_official_four_column_reference_survives_dataset_and_action_materialization(tmp_path):
    from exact.core.actions.evaluation import run_evaluation
    from exact.core.entities.kinds import EntityKind
    from exact.impl.datasets.base import BaseAlignmentDataset

    class Dataset(BaseAlignmentDataset):
        def log_sanity_examples(self, *args, **kwargs):
            pass

        def plot_feature_distributions(self, *args, **kwargs):
            pass

        def get_features(self, frame):
            return frame

        def __getitem__(self, index):
            raise IndexError(index)

        def __len__(self):
            return 0

    reference = tmp_path / "official.tsv"
    reference.write_text("SrcEntity\tTgtEntity\tRelation\tScore\ns\tt\t=\t0.75\n")
    dataset = Dataset(output_path=tmp_path / "dataset")
    dataset._source_entity_kind_index = {"s": EntityKind.CLASS}
    dataset._target_entity_kind_index = {"t": EntityKind.CLASS}
    dataset.load_reference(reference)
    assert dataset.reference.iloc[0]["Label"] == "="
    assert dataset.reference.iloc[0]["Score"] == 0.75
    output = _materialize_evaluation_reference(
        dataset.reference, parent_path=reference, output_path=tmp_path / "enriched.tsv"
    )
    mappings = tmp_path / "predictions.tsv"
    mappings.write_text("SrcEntity\tTgtEntity\tScore\ns\tt\t0.9\n")
    run_evaluation(
        mappings, tmp_path / "evaluation", full_reference_file_path=output, error_on_fail=True
    )
    assert output.read_text().splitlines()[1] == "s\tt\t=\tclass\tclass"


def test_stripped_local_pool_gets_reporting_gold_only_after_scoring(tmp_path):
    from exact.core.actions.evaluation import (
        materialize_local_ranking_inputs,
        run_evaluation,
    )

    pool = tmp_path / "unlabelled.tsv"
    original = (
        "SrcEntity\tTgtEntity\tTgtCandidates\ns1\t\t['wrong', 'right']\ns2\t\t['other', 'gold']\n"
    )
    pool.write_text(original)
    scores = tmp_path / "scores.tsv"
    scores.write_text(
        "SrcEntity\tTgtEntity\tTgtCandidates\ns1\t\t[('wrong', 0.1), ('right', 0.9)]\ns2\t\t[('other', 0.2), ('gold', 0.8)]\n"
    )
    reference = tmp_path / "valid.tsv"
    reference.write_text("SrcEntity\tTgtEntity\tRelation\tScore\ns1\tright\t=\t1\ns2\tgold\t=\t1\n")
    annotated, queries = materialize_local_ranking_inputs(
        scores, pool, reference, tmp_path / "evaluation/inputs"
    )
    metrics = run_evaluation(
        annotated, tmp_path / "evaluation", reference_candidates=queries, error_on_fail=True
    )
    assert metrics["MRR"] == 1.0
    assert metrics["Hits@1"] == 1.0
    assert pool.read_text() == original
    assert scores.read_text().splitlines()[1].startswith("s1\t\t")


def test_local_reporting_join_keeps_all_positives_and_empty_candidate_groups(tmp_path):
    import ast

    from exact.core.actions.evaluation import (
        materialize_local_ranking_inputs,
        run_evaluation,
    )

    pool = tmp_path / "pool.tsv"
    pool.write_text("SrcEntity\tTgtEntity\tTgtCandidates\ns\t\t['a', 'b']\nempty\t\t[]\n")
    scores = tmp_path / "scores.tsv"
    scores.write_text("SrcEntity\tTgtEntity\tTgtCandidates\ns\t\t[('a', 0.9), ('b', 0.8)]\n")
    reference = tmp_path / "valid.tsv"
    reference.write_text(
        "SrcEntity\tTgtEntity\tRelation\tScore\ns\ta\t=\t1\ns\tb\t=\t1\nempty\tmissing\t=\t1\noutside\tx\t=\t1\n"
    )
    annotated, queries = materialize_local_ranking_inputs(
        scores, pool, reference, tmp_path / "evaluation/inputs"
    )
    rows = pd.read_csv(queries, sep="\t")
    assert len(rows) == 3
    assert set(rows.loc[rows.SrcEntity == "s", "TgtEntity"]) == {"a", "b"}
    assert ast.literal_eval(rows.loc[rows.SrcEntity == "empty", "TgtCandidates"].iloc[0]) == []
    metrics = run_evaluation(
        annotated, tmp_path / "evaluation", reference_candidates=queries, error_on_fail=True
    )
    assert metrics["MRR"] == 0.5  # (1 + 1/2 + empty-pool zero) / three official queries
