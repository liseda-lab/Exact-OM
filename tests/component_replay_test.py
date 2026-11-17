"""Frozen request support changes scores through the production scorer without calls."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from exact.runtime.numerical_cache import training_content_contract
from exact.impl.models.component_replay import apply_judgments
from exact.utils.fitted_artifacts import fingerprint
from exact.utils.provenance import sha256_file
from tests.numerical_cache_test import runtime
from tests.pair_adaptive_experiments_test import _scorer, _TinyDataset
from tests.scoring_cache_throughput_test import native_training_model


def setup_bank(tmp_path, monkeypatch):
    dataset = _TinyDataset()
    dataset._df = pd.DataFrame({"Src": ["s"] * 3, "Tgt": ["t", "tt", "ta"]})
    dataset._active_dataframe = lambda: dataset._df
    scorer = _scorer(strsim={"enabled": True}, return_explanations=True)
    scorer.attach_dataset(dataset)
    selected = {"Src": "s", "Tgt": "tt"}
    payload = {
        "schema": "exact-component-inference-judgments-v1",
        "model_id": "fixture-luna",
        "dataset_signature": dataset.dataset_signature,
        "provenance": {"role": "valid"},
        "predictions": [{**selected, "probability": 0.9}],
        "expected_pairs": [selected],
        "policy": "constant_0_5",
        "request_bank_sha256": "frozen-bank",
        "candidate_pool_sha256": fingerprint(sorted([("s", "t"), ("s", "tt"), ("s", "ta")])),
    }

    def save():
        path = tmp_path / "inference.json"
        path.write_text(json.dumps(payload))
        monkeypatch.setenv("EXACT_COMPONENT_INFERENCE_JUDGMENTS", str(path))
        monkeypatch.setenv("EXACT_COMPONENT_INFERENCE_JUDGMENTS_SHA256", sha256_file(path))
        monkeypatch.setenv("EXACT_COMPONENT_JUDGE_MODEL", "fixture-luna")

    save()
    args = dict(
        src_iris=["s"] * 3,
        tgt_iris=["t", "tt", "ta"],
        src_label_lists=[["source"]] * 3,
        tgt_label_lists=[["source"], ["other"], ["source"]],
    )
    return scorer, dataset, payload, save, args


def test_production_scorer_replays_fixed_support_complete_pool_and_provenance(
    tmp_path, monkeypatch
):
    scorer, _, _, _, args = setup_bank(tmp_path, monkeypatch)
    scorer.use_llm = True
    for name in (
        "llm_yesno_probs_batched",
        "generate_pair_briefs_batched",
        "llm_binary_decision_probs",
    ):
        monkeypatch.setattr(scorer, name, lambda *args, **kwargs: pytest.fail("made a paid call"))
    first = scorer(**args)
    assert len(first["S_final"]) == 3
    assert first["need_llm"].tolist() == [False, True, False]
    assert torch.equal(first["S_final"][[0, 2]], first["S_base"][[0, 2]])
    assert float(first["S_final"][1]) == pytest.approx(0.5 * float(first["S_base"][1]) + 0.45)
    assert first["llm_gate_diagnostics"][1]["cached_observed_judgment"] is True
    assert first["llm_gate_diagnostics"][0]["llm_probability"] is None
    assert first["batch_pair_adaptive_stats"]["decision_requested_pairs"] == 0
    assert first["batch_pair_adaptive_stats"]["cached_judgment_pairs"] == 1
    assert (
        first["explanations"][1]["experiment_diagnostics"]["llm_gate"]["cached_model"]
        == "fixture-luna"
    )
    # Change the numerical order and analytic threshold. Frozen membership persists.
    scorer.tau_LLM = 2.0
    second = scorer(**{**args, "tgt_label_lists": [["other"], ["source"], ["other"]]})
    assert not torch.equal(first["S_base"], second["S_base"])
    assert second["need_llm"].tolist() == [False, True, False]
    assert torch.equal(second["S_final"][[0, 2]], second["S_base"][[0, 2]])


@pytest.mark.parametrize("mutation", ["pool", "missing", "duplicate", "model", "nan", "checksum"])
def test_inference_bank_mismatch_fails_closed(tmp_path, monkeypatch, mutation):
    scorer, dataset, payload, save, args = setup_bank(tmp_path, monkeypatch)
    if mutation == "pool":
        dataset._df = dataset._df.iloc[:2].copy()
    elif mutation == "missing":
        payload["predictions"] = []
    elif mutation == "duplicate":
        payload["expected_pairs"].append(payload["expected_pairs"][0])
    elif mutation == "model":
        payload["model_id"] = "wrong-model"
    elif mutation == "nan":
        payload["predictions"][0]["probability"] = float("nan")
    else:
        Path(tmp_path / "inference.json").write_text("changed")
    if mutation != "checksum":
        save()
    with pytest.raises(ValueError):
        scorer(**args)


def test_reusing_scorer_after_pool_replacement_revalidates_membership(tmp_path, monkeypatch):
    scorer, dataset, _, _, args = setup_bank(tmp_path, monkeypatch)
    scorer(**args)
    dataset._df = dataset._df.iloc[:2].copy()
    with pytest.raises(ValueError, match="different candidate pool"):
        scorer(**args)


def test_train_numerical_identity_excludes_inference_bank_and_never_applies_it(
    tmp_path, monkeypatch
):
    runtime(tmp_path, monkeypatch)
    model = native_training_model(tmp_path)
    original = training_content_contract(model)
    monkeypatch.setenv("EXACT_COMPONENT_INFERENCE_JUDGMENTS", str(tmp_path / "unneeded.json"))
    monkeypatch.setenv("EXACT_COMPONENT_INFERENCE_JUDGMENTS_SHA256", "bank-a")
    monkeypatch.setenv("EXACT_COMPONENT_JUDGE_MODEL", "luna")
    assert training_content_contract(model) == original
    identity = copy.deepcopy(model.runtime_fingerprint_payload())
    monkeypatch.setenv("EXACT_COMPONENT_INFERENCE_JUDGMENTS_SHA256", "bank-b")
    assert training_content_contract(model) == original
    assert model.runtime_fingerprint_payload() != identity
    model._numerical_scoring_role = "train"
    assert apply_judgments(model, ["s"], ["t"], torch.tensor([0.5]), torch.tensor([1.0])) is None
    assert "component_judgments" not in model.runtime_fingerprint_payload()
