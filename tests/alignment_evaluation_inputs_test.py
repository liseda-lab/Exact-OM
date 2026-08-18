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
