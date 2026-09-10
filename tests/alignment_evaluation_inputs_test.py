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
