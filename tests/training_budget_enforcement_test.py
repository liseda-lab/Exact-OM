"""Exercise label budgets through score extraction, calibration and real OOF fitting."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from exact.impl.models.selector.fitting import fingerprint
from exact.impl.trainer.fitting import _validate_budgeted_selector_fit
from tests.grouped_fitting_test import (
    TinyDataset,
    TinyScorer,
    selector,
    tiny_runner,
    training_rows,
)


def budget_runner(tmp_path, budget, *, calibration="platt", selection="passive"):
    frame = training_rows()
    frame["confirmed_label"] = frame.Tgt.str.endswith("0").astype(int)
    reference = {(str(row.Src), str(row.Tgt)) for row in frame.itertuples() if row.confirmed_label}
    # Keep annotated pool misses in the unbudgeted acceptance population, and
    # outside-pool positives belonging to the selected candidate groups.
    reference |= {(f"s{i}", "outside-pool") for i in range(9)}
    reference.add(("pool-miss", "outside-pool"))
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                {"Src": ["pool-miss"] * 2, "Tgt": ["miss-1", "miss-2"], "confirmed_label": [0, 0]}
            ),
        ],
        ignore_index=True,
    )
    pool, refs = tmp_path / "training.tsv", tmp_path / "reference.tsv"
    frame[["Src", "Tgt", "confirmed_label"]].to_csv(pool, sep="\t", index=False)
    pd.DataFrame(sorted(reference), columns=["Src", "Tgt"]).to_csv(refs, sep="\t", index=False)
    head = selector()
    head.training_reference_file_path = str(refs)
    head.matching_calibration.update(
        mode=calibration, artifact=str(tmp_path / f"cal-{budget}.json")
    )
    scorer = TinyScorer()
    runner = tiny_runner(
        tmp_path,
        TinyDataset(pd.DataFrame({"Src": ["report"], "Tgt": ["report-0"]})),
        scorer,
        head,
    )
    runner.supervision_config = {
        "negative_label_policy": "confirmed_negatives",
        "label_budget": budget,
        "label_selection": selection,
    }
    runner.training_candidates_file_path = pool
    return runner, head, scorer, reference


def test_budget_reaches_real_calibration_and_selector_folds_and_shares_raw_scores(tmp_path):
    identities, selected = [], {}
    score_cache = None
    for budget in (2, 4, None):
        runner, head, scorer, reference = budget_runner(tmp_path, budget)
        runner.fit_training_pool(batch_size=5)
        artifact = json.loads(Path(head.rerank_config["artifact"]).read_text())
        calibration = json.loads(Path(head.matching_calibration["artifact"]).read_text())
        units = runner._training_effective_units
        expected = (
            set(units["selected_sources"])
            if budget
            else {f"s{i}" for i in range(9)} | {"pool-miss"}
        )
        selected[budget] = expected
        identities.append(artifact["fit_identity"])
        assert set(artifact["fit_provenance"]["training_sources"]) == expected
        assert set(calibration["fit_provenance"]["training_sources"]) == expected
        assert {row["source"] for row in calibration["oof_predictions"]} == expected
        assert len(calibration["oof_predictions"]) == len(expected) * 2
        assert sum(row["label"] for row in calibration["oof_predictions"]) == (budget or 9)
        assert artifact["fit_provenance"]["training_reference_sha256"] == fingerprint(
            sorted(pair for pair in reference if pair[0] in expected)
        )
        heldout = []
        for fold in artifact["folds"]:
            train, test = set(fold["train_sources"]), set(fold["heldout_sources"])
            assert train | test == expected
            assert not train & test
            heldout.extend(test)
        assert set(heldout) == expected and len(heldout) == len(expected)
        if budget:
            assert (
                artifact["fit_provenance"]["application"]["training_budget"]["selected_sources"]
                == units["selected_sources"]
            )
        else:
            assert "training_budget" not in artifact["fit_provenance"]["application"]
        assert "training_budget" not in units["binding"]
        # All source scores are cached once, independently of the fitted budget.
        paths = list((tmp_path / "fitting").glob("*/training_scores.json"))
        assert len(paths) == 1
        if score_cache is None:
            score_cache = paths[0].read_bytes()
            assert scorer.calls
        else:
            assert paths[0].read_bytes() == score_cache
            assert not scorer.calls
    assert selected[2] < selected[4] < selected[None]
    assert len(set(identities)) == 3


def test_active_budget_is_enforced_and_unselected_confirmed_labels_cannot_reenter(tmp_path):
    runner, head, _, _ = budget_runner(tmp_path, 2, calibration="none", selection="uncertainty")
    runner.fit_training_pool(batch_size=5)
    payload = json.loads(Path(head.rerank_config["artifact"]).read_text())
    assert payload["fit_provenance"]["training_sources"] == ["s0", "s1"]
    membership = json.loads(
        Path(head.rerank_config["artifact"]).with_name("label_budget.json").read_text()
    )
    for corruption, message in (
        ("source", "training sources"),
        ("fold", "OOF split"),
        ("binding", "not bound"),
    ):
        invalid = copy.deepcopy(payload)
        if corruption == "source":
            invalid["fit_provenance"]["training_sources"].append("unselected")
        elif corruption == "fold":
            invalid["folds"][0]["train_sources"].append("unselected")
        else:
            invalid["fit_provenance"]["application"].pop("training_budget")
        with pytest.raises(ValueError, match=message):
            _validate_budgeted_selector_fit(invalid, membership)


@pytest.mark.parametrize("consumer", ["nil", "relation"])
def test_unsupported_independent_label_population_fails_before_scoring(tmp_path, consumer):
    runner, head, scorer, _ = budget_runner(tmp_path, 2)
    if consumer == "nil":
        head.nil_config.update(mode="fitted", training_source_labels="source-labels.tsv")
        message = "NIL training units"
    else:
        runner.relation_config = {
            "relation_prediction": "learned_three_way",
            "relation_training_file": "typed-labels.tsv",
        }
        message = "typed relation training units"
    with pytest.raises(ValueError, match=message):
        runner.fit_training_pool(batch_size=5)
    assert not scorer.calls
