from ast import literal_eval
from pathlib import Path

import pandas as pd
import pytest

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


def test_semantic_graph_closure_types_paths_and_abstains_when_unresolved() -> None:
    source = CsvKgSource.from_path(FIXTURE)
    candidates = pd.DataFrame(
        [
            {"Src": BASE + "organ", "Tgt": BASE + "organ", "Score": 0.2},
            {"Src": BASE + "heart", "Tgt": BASE + "heart", "Score": 0.3},
            {"Src": BASE + "atrium", "Tgt": BASE + "organ", "Score": 0.7},
            {"Src": BASE + "heart", "Tgt": BASE + "atrium", "Score": 0.7},
            {"Src": BASE + "blood", "Tgt": BASE + "missing", "Score": 0.4},
        ]
    )
    typed = predict_relations(
        candidates,
        source,
        source,
        mode="semantic_entailment",
        semantic_backend="graph_closure",
    )
    relations = {
        (row.SrcEntity, row.TgtEntity): row.Relation for row in typed.itertuples(index=False)
    }
    assert (BASE + "organ", BASE + "organ") not in relations
    assert (BASE + "heart", BASE + "heart") not in relations
    assert relations[(BASE + "atrium", BASE + "organ")] == "<"
    assert relations[(BASE + "heart", BASE + "atrium")] == ">"
    assert (BASE + "blood", BASE + "missing") not in relations
    assert typed.attrs["relation_anchor_count"] == 2
    assert len(typed.attrs["relation_abstentions"]) == 3
    assert typed.attrs["relation_abstentions"][-1:] == [
        {
            "source": BASE + "blood",
            "target": BASE + "missing",
            "reason": "not_entailed",
            "forward_confidence": 0.0,
            "reverse_confidence": 0.0,
        }
    ]
    assert typed["relation_evidence"].str.contains('"backend":"graph_closure"', regex=False).all()
    assert typed["relation_confidence"].min() == pytest.approx(1.0)
    assert {row["origin"] for row in typed.attrs["relation_anchors"]} == {"exact_label"}


def test_semantic_reciprocal_anchor_margin_is_configurable() -> None:
    source = CsvKgSource.from_path(FIXTURE)
    candidates = pd.DataFrame([{"Src": BASE + "atrium", "Tgt": BASE + "heart", "Score": 0.96}])

    accepted = predict_relations(
        candidates,
        source,
        source,
        mode="semantic_entailment",
        equivalence_anchor_threshold=0.95,
        equivalence_anchor_margin=0.10,
    )
    rejected = predict_relations(
        candidates,
        source,
        source,
        mode="semantic_entailment",
        equivalence_anchor_threshold=0.95,
        equivalence_anchor_margin=0.97,
    )

    assert accepted.empty  # Its own high-confidence bridge is never independent evidence.
    assert accepted.attrs["relation_anchor_count"] == 1
    assert rejected.empty
    assert rejected.attrs["relation_anchor_count"] == 0


def test_semantic_timeout_abstains_remaining_rows_without_escaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptySource:
        def labels(self, iri: str) -> list[str]:
            return []

        def entities(self, kind) -> list[str]:
            return []

    ticks = iter([0.0, 61.0])
    monkeypatch.setattr(
        "exact.io.relations.time.monotonic",
        lambda: next(ticks, 61.0),
    )
    typed = predict_relations(
        pd.DataFrame(
            [
                {"Src": "s1", "Tgt": "t1", "Score": 0.8},
                {"Src": "s2", "Tgt": "t2", "Score": 0.7},
            ]
        ),
        EmptySource(),
        EmptySource(),
        mode="semantic_entailment",
        timeout_seconds=60.0,
    )
    assert typed.empty
    assert typed.attrs["relation_abstentions"] == [
        {"source": "s1", "target": "t1", "reason": "reasoning_timeout"},
        {"source": "s2", "target": "t2", "reason": "reasoning_timeout"},
    ]


@pytest.mark.parametrize("mode", ["learned_three_way", "semantic_then_learned"])
def test_learned_relation_modes_fail_closed_without_fitted_artifact(mode: str) -> None:
    source = CsvKgSource.from_path(FIXTURE)
    with pytest.raises(Exception, match="requires a fitted multinomial relation artifact"):
        predict_relations(
            pd.DataFrame([{"Src": BASE + "heart", "Tgt": BASE + "heart", "Score": 0.9}]),
            source,
            source,
            mode=mode,
        )


def test_bridge_reasoner_marks_csv_profile_unsupported() -> None:
    source = CsvKgSource.from_path(FIXTURE)
    result = predict_relations(
        pd.DataFrame([{"Src": BASE + "heart", "Tgt": BASE + "heart", "Score": 0.9}]),
        source,
        source,
        mode="semantic_entailment",
        semantic_backend="bridge_reasoner",
    )
    assert result.empty
    assert result.attrs["relation_abstentions"][0]["reason"] == "unsupported_profile"
