"""Final global acceptance contracts without model downloads or provider calls."""

import itertools
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from exact.acceptance.features import RAW, assignment, group_stats, select
from exact.acceptance.io import partial_metrics, prepare, references, read_frame
from exact.acceptance.model import training_population, training_rows
from exact.delivery.cli.acceptance import main


def candidates(n=12):
    rng = np.random.default_rng(3)
    rows = []
    for source in range(n):
        for target in range(4):
            row = {name: float(rng.uniform(0.1, 0.9)) for name in RAW}
            row.update(
                Src=f"s{source:02}", Tgt=f"t{target}", original_exact_anchor=source == target
            )
            row["S_pair_final"] = 0.9 if source == target else 0.1 + float(rng.uniform(0, 0.1))
            rows.append(row)
    return pd.DataFrame(rows)


def test_margins_ties_and_neutral_singleton():
    values = np.array([0.4, 0.4, 0.2, 0.9, 0.1, 0.8])
    codes = np.array([0, 0, 0, 1, 2, 2])
    gap, rank, count = group_stats(values, codes, 3)
    assert rank.tolist() == [0, 1, 2, 0, 1, 0]
    for i, value in enumerate(values):
        others = values[(codes == codes[i]) & (np.arange(len(values)) != i)]
        assert gap[i] == pytest.approx(value - max(others) if len(others) else 0.0)
    assert count.tolist() == [3, 1, 2]


def test_duplicate_missing_and_empty_populations():
    frame = candidates()
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(pd.concat([frame, frame.iloc[:1]]))
    with pytest.raises(ValueError, match="Missing candidate"):
        prepare(frame.drop(columns=["s_hier"]))
    frame.loc[0, "U"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        prepare(frame)
    assert prepare(candidates().iloc[:0]).x.shape == (0, 33)


def test_one_to_one_assignment_matches_bruteforce():
    rng = np.random.default_rng(5)
    for _ in range(5):
        s = np.repeat(np.arange(3), 3)
        t = np.tile(np.arange(3), 3)
        p = rng.uniform(0.1, 0.99, 9)
        chosen = assignment(p, s, t, 0.4)
        best = 0.0
        for choices in itertools.product(range(-1, 3), repeat=3):
            real = [v for v in choices if v >= 0]
            if len(set(real)) != len(real):
                continue
            idx = [i * 3 + v for i, v in enumerate(choices) if v >= 0]
            if all(p[i] > 0.4 for i in idx):
                best = max(best, sum(logit(p[i]) - logit(0.4) for i in idx))
        assert sum(logit(p[i]) - logit(0.4) for i in chosen) == pytest.approx(best)
        assert np.array_equal(chosen, assignment(p, s, t, 0.4))


def test_winner_loses_next_candidate_and_no_anchor_bypass():
    s, t, p = np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([0.85, 0.8, 0.9])
    assert select(p, s, t, threshold=0.5, mode="winner").tolist() == [2]
    assert set(select(p, s, t, threshold=0.5, mode="greedy")) == {1, 2}
    assert len(select(p, s, t, threshold=0.9, mode="greedy")) == 0


def test_both_heldout_endpoints_and_missing_reference_denominators():
    data = prepare(candidates())
    refs = {(f"s{i:02}", f"t{i}") for i in range(4)} | {("absent", "absent")}
    d = training_population(data, refs, folds=3)
    for fold in range(3):
        idx, y, weights = training_rows(data, d, fold, 0.02)
        assert not (d["sf"][data.src[idx]] == fold).any()
        assert not (d["tf"][data.tgt[idx]] == fold).any()
        assert (weights > 0).all()
    assert len(refs) == 5 and d["label"].sum() == 4


def test_partial_or_rule_and_reference_shape(tmp_path):
    metric = partial_metrics(
        {("a", "x"), ("u", "y"), ("v", "w"), ("c", "z")}, {("a", "x"), ("b", "y")}, {("c", "z")}
    )
    assert (metric["TP"], metric["FP"], metric["FN"], metric["unjudged"]) == (1, 1, 1, 1)
    path = tmp_path / "refs.csv"
    pd.DataFrame([("a", "x"), ("a", "y")], columns=["Src", "Tgt"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one-to-one"):
        references(path)


def test_jsonl_decision_output_compatibility(tmp_path):
    path = tmp_path / "scored.jsonl"
    frame = candidates()
    frame.to_json(path, orient="records", lines=True, double_precision=15)
    assert np.array_equal(prepare(read_frame(path)).x, prepare(frame).x)


def test_retained_head_cli_and_anchor_policy(tmp_path):
    from exact.acceptance.retained import RANK_FEATURES, ACCEPT_FEATURES

    rank = dict(
        mean=[0.0] * 17,
        scale=[1.0] * 17,
        weights=[1.0] + [0.0] * 16,
        bias=0.0,
        model_type="current_linear",
    )
    accept = dict(
        mean=[0.0] * 10, scale=[1.0] * 10, weights=[0.0, 0.0, 10.0] + [0.0] * 7, bias=-5.0
    )
    head = dict(
        rank_features=RANK_FEATURES,
        accept_features=ACCEPT_FEATURES,
        rank_model=rank,
        accept_model=accept,
        pair_calibration=dict(mode="platt", slope=1.0, intercept=0.0),
        accept_threshold=0.5,
        absolute_pair_floor=0.1,
        exact_anchor_bypass=False,
        global_reciprocity=True,
        synthetic_runnerup_negatives=0,
    )
    path, pairs, model = tmp_path / "head.json", tmp_path / "pairs.csv", tmp_path / "model"
    path.write_text(json.dumps(head))
    candidates().to_csv(pairs, index=False)
    assert (
        main(
            ["import-retained", "--head", str(path), "--temperature", ".75", "--output", str(model)]
        )
        == 0
    )
    assert (
        main(
            [
                "apply-retained",
                "--candidates",
                str(pairs),
                "--model",
                str(model),
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    mappings = pd.read_csv(tmp_path / "out/mappings.csv")
    assert not mappings.SrcEntity.duplicated().any()
    assert not mappings.TgtEntity.duplicated().any()
    head["exact_anchor_bypass"] = True
    path.write_text(json.dumps(head))
    with pytest.raises(SystemExit):
        main(
            [
                "import-retained",
                "--head",
                str(path),
                "--temperature",
                ".75",
                "--output",
                str(tmp_path / "invalid"),
            ]
        )


def test_cli_fit_apply_import_and_restart(tmp_path, monkeypatch):
    from exact.acceptance import model

    frame = candidates()
    path, train, output = tmp_path / "pairs.csv", tmp_path / "train.tsv", tmp_path / "model"
    frame.to_csv(path, index=False)
    pd.DataFrame(
        [(f"s{i:02}", f"t{i}") for i in range(4)], columns=["SrcEntity", "TgtEntity"]
    ).to_csv(train, sep="\t", index=False)
    args = [
        "fit",
        "--candidates",
        str(path),
        "--train",
        str(train),
        "--output",
        str(output),
        "--folds",
        "2",
        "--family",
        "logistic",
    ]
    original = model.predict
    calls = []

    def interrupted(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise InterruptedError("after durable first-fold model")
        return original(*a, **kw)

    monkeypatch.setattr(model, "predict", interrupted)
    with pytest.raises(SystemExit):
        main(args)
    first = output / "fold-0/model.joblib"
    timestamp = first.stat().st_mtime_ns
    monkeypatch.setattr(model, "predict", original)
    assert main(args) == 0
    assert first.stat().st_mtime_ns == timestamp
    assert main(args) == 0
    for policy in ["full", "partial"]:
        assert (
            main(
                [
                    "apply",
                    "--candidates",
                    str(path),
                    "--model",
                    str(output),
                    "--policy",
                    policy,
                    "--output",
                    str(tmp_path / policy),
                ]
            )
            == 0
        )
        assert (
            json.loads((tmp_path / policy / "completion.json").read_text())["status"] == "complete"
        )
    with pytest.raises(SystemExit):
        main(args + ["--weak-weight", "0.1"])
    imported = tmp_path / "imported"
    assert (
        main(
            [
                "import-model",
                "--model",
                str(output / "model.joblib"),
                "--output",
                str(imported),
                "--threshold",
                ".5",
                "--mode",
                "assignment",
                "--policy",
                "partial",
                "--trusted-model",
            ]
        )
        == 0
    )
    estimator, manifest = model.load(imported)
    assert manifest["refitted"] is False
    with (imported / "model.joblib").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        model.load(imported)
