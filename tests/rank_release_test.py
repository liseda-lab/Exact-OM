"""Production local query CLI preserves each query occurrence and native anchors."""

import ast
import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from exact.delivery.cli.rank import main, read_queries


def frame(sources):
    return pd.DataFrame(
        [
            dict(
                Src=s,
                Tgt=t,
                S_final=score,
                S_pair_final=score,
                s_label=score,
                S_struct=0.7,
                s_hier=0.7,
                s_sim=0.6,
                s_diff=0.8,
                s_attr=0.5,
                selector_evidence_items=[],
            )
            for s in sources
            for t, score in [("a", 0.8), ("b", 0.3), ("c", 0.1)]
        ]
    )


def settings(path):
    config = dict(
        config_version=2,
        pipeline=[
            dict(name="PairAdaptiveSemanticScorer", params=dict(use_llm=False)),
            dict(
                name="CandidateSetSelector",
                params=dict(calibration=dict(max_epochs=3, validation_folds=2)),
            ),
        ],
        selector=dict(enabled=True, rerank=dict(mode="pairwise", model="current_linear")),
        matching=dict(calibration=dict(mode="none"), nil=dict(mode="off")),
    )
    path.write_text(json.dumps(config))


def setup_fit(tmp_path):
    config, rows, queries, model = [
        tmp_path / name for name in ["config.json", "train.jsonl", "train.tsv", "head.json"]
    ]
    settings(config)
    frame(["train1", "train2", "train3", "train4"]).to_json(rows, orient="records", lines=True)
    with queries.open("w") as stream:
        w = csv.writer(stream, delimiter="\t")
        w.writerow(["SrcEntity", "TgtEntity", "TgtCandidates"])
        for source in ["train1", "train2", "train3", "train4"]:
            w.writerow([source, "a", repr(["c", "b", "a"])])
    args = [
        "fit",
        "--candidates",
        str(rows),
        "--queries",
        str(queries),
        "--config",
        str(config),
        "--model",
        str(model),
        "--confirmed-negatives",
    ]
    return args, model


def test_fit_apply_reload_and_repeated_source_queries(tmp_path):
    args, model = setup_fit(tmp_path)
    assert main(args) == 0
    before = model.read_bytes()
    assert main(args) == 0
    assert model.read_bytes() == before
    provenance = json.loads(before)["fit_provenance"]
    assert provenance["anchor_policy"] == "native_hard_anchor_bypass_no_feature_fabrication"
    rows, queries, out = tmp_path / "test.jsonl", tmp_path / "test.tsv", tmp_path / "submission.tsv"
    frame(["test1"]).to_json(rows, orient="records", lines=True)
    with queries.open("w") as stream:
        w = csv.writer(stream, delimiter="\t")
        w.writerow(["SrcEntity", "TgtCandidates"])
        w.writerow(["test1", repr(["c", "b", "a"])])
        w.writerow(["test1", repr(["b", "a"])])
    call = [
        "apply",
        "--candidates",
        str(rows),
        "--queries",
        str(queries),
        "--model",
        str(model),
        "--output",
        str(out),
    ]
    assert main(call) == 0
    assert main(call) == 0
    with out.open() as stream:
        saved = list(csv.DictReader(stream, delimiter="\t"))
    assert len(saved) == 2
    assert set(ast.literal_eval(saved[0]["TgtCandidates"])) == {"a", "b", "c"}
    assert set(ast.literal_eval(saved[1]["TgtCandidates"])) == {"a", "b"}
    assert all(isinstance(t, str) for row in saved for t in ast.literal_eval(row["TgtCandidates"]))
    scores = json.loads(out.with_suffix(".scores.json").read_text())
    assert scores["candidate_occurrences"] == 5
    assert scores["original_query_count"] == 2
    assert scores["source_count"] == 1
    assert scores["acceptance"] is False


def test_anchor_bypass_is_explicit_and_train_requires_real_features(tmp_path):
    args, model = setup_fit(tmp_path)
    assert main(args) == 0
    rows, queries, anchors, out = [
        tmp_path / name for name in ["test.jsonl", "test.json", "anchors.json", "out.tsv"]
    ]
    scored = frame(["test1"])
    scored = scored[scored.Tgt != "c"]
    scored.to_json(rows, orient="records", lines=True)
    queries.write_text(json.dumps([dict(qid=7, source="test1", candidates=["a", "b", "c"])]))
    anchors.write_text(json.dumps([dict(Src="test1", Tgt="c", S_final=1.0)]))
    call = [
        "apply",
        "--candidates",
        str(rows),
        "--queries",
        str(queries),
        "--model",
        str(model),
        "--output",
        str(out),
    ]
    with pytest.raises(SystemExit):
        main(call)
    assert main(call + ["--anchors", str(anchors)]) == 0
    predicted = json.loads(out.with_suffix(".scores.json").read_text())["queries"][0]
    assert predicted["candidates"][0]["target"] == "c"
    assert predicted["candidates"][0]["protected_exact"] is True
    assert predicted["fixed_anchor_count"] == 1
    train_path = Path(args[args.index("--candidates") + 1])
    training = pd.read_json(train_path, lines=True)
    training.iloc[1:].to_json(train_path, orient="records", lines=True)
    with pytest.raises(SystemExit):
        main(args)


def test_reporting_source_overlap_and_changed_artifacts_rejected(tmp_path):
    args, model = setup_fit(tmp_path)
    assert main(args) == 0
    query_path = tmp_path / "test.json"
    query_path.write_text(json.dumps([dict(qid=0, source="train1", candidates=["a", "b", "c"])]))
    rows = tmp_path / "test.jsonl"
    frame(["train1"]).to_json(rows, orient="records", lines=True)
    call = [
        "apply",
        "--candidates",
        str(rows),
        "--queries",
        str(query_path),
        "--model",
        str(model),
        "--output",
        str(tmp_path / "out.tsv"),
    ]
    with pytest.raises(SystemExit):
        main(call)
    saved = json.loads(model.read_text())
    saved["rank_model"]["bias"] += 1
    model.write_text(json.dumps(saved))
    with pytest.raises(SystemExit):
        main(call)


def test_query_validation_and_explicit_negative_authorization(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps([dict(qid=1, source="x", candidates=["a", "b"], gold_target="missing")])
    )
    with pytest.raises(ValueError, match="gold target"):
        read_queries(path, labels=True)
    path.write_text(json.dumps([dict(qid=1, source="x", candidates=["a", "a"])]))
    with pytest.raises(ValueError, match="unique target"):
        read_queries(path, labels=False)
    args, _ = setup_fit(tmp_path)
    with pytest.raises(SystemExit):
        main(args[:-1])
