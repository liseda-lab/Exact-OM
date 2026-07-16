from ast import literal_eval
from pathlib import Path

import pandas as pd

from exact.io.relations import predict_relations
from exact.io.sources.csv_kg import CsvKgSource
from exact.io.writers import write
from exact.io.writers.typed_tsv import TYPED_RELATIONS

FIXTURE = Path(__file__).parent / "fixtures" / "kg_csv"
BASE = "http://example.org/kg/"


def _anchors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Src": BASE + "atrium", "Tgt": BASE + "atrium", "Score": 0.99},
            {"Src": BASE + "heart", "Tgt": BASE + "heart", "Score": 0.98},
            {"Src": BASE + "organ", "Tgt": BASE + "organ", "Score": 0.97},
        ]
    )


def test_hierarchy_heuristic_types_equivalent_and_directional_relations() -> None:
    source = CsvKgSource.from_path(FIXTURE)
    candidates = pd.DataFrame(
        [
            {"Src": BASE + "atrium", "Tgt": BASE + "atrium", "Score": 0.99},
            {"Src": BASE + "atrium", "Tgt": BASE + "heart", "Score": 0.75},
            {"Src": BASE + "heart", "Tgt": BASE + "atrium", "Score": 0.7},
            {"Src": BASE + "heart", "Tgt": BASE + "heart", "Score": 0.98},
        ]
    )
    typed = predict_relations(
        candidates,
        source,
        source,
        mode="hierarchy_heuristic",
        anchors=_anchors(),
    )
    relations = {
        (row.SrcEntity, row.TgtEntity): row.Relation for row in typed.itertuples(index=False)
    }
    assert relations[(BASE + "atrium", BASE + "atrium")] == "="
    assert relations[(BASE + "atrium", BASE + "heart")] == "<"
    assert relations[(BASE + "heart", BASE + "atrium")] == ">"
    assert relations[(BASE + "heart", BASE + "heart")] == "="
    assert typed["relation_confidence"].between(0.0, 1.0).all()


def test_none_mode_preserves_equivalence_default() -> None:
    source = CsvKgSource.from_path(FIXTURE)
    frame = predict_relations(
        pd.DataFrame([{"Src": "s", "Tgt": "t", "Score": 0.5, "Relation": ">"}]),
        source,
        source,
    )
    assert frame.loc[0, "Relation"] == "="
    assert frame.loc[0, "relation_confidence"] == 1.0


def test_mini_biokg_candidates_produce_ranked_typed_submission(tmp_path: Path) -> None:
    source = CsvKgSource.from_path(FIXTURE)
    candidate_table = pd.read_csv(FIXTURE / "candidates.tsv", sep="\t")
    rows = []
    score = 1.0
    allowed: set[tuple[str, str]] = set()
    for candidate_row in candidate_table.itertuples(index=False):
        for target in literal_eval(candidate_row.TgtCandidates):
            allowed.add((candidate_row.SrcEntity, target))
            rows.append({"Src": candidate_row.SrcEntity, "Tgt": target, "Score": score})
            score -= 0.1
    relations = predict_relations(
        pd.DataFrame(rows),
        source,
        source,
        mode="hierarchy_heuristic",
        anchors=_anchors(),
    )
    path = write("typed-tsv", relations, tmp_path)
    submission = pd.read_csv(path, sep="\t")
    assert set(zip(submission.SrcEntity, submission.TgtEntity)) == allowed
    assert set(submission.Relation) <= TYPED_RELATIONS
    assert submission.Score.tolist() == sorted(submission.Score, reverse=True)
