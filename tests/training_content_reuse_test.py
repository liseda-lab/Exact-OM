"""Freeze actual TRAIN evidence before lookup; refit heads for each application."""

import copy
import json
from pathlib import Path
from types import MethodType

import pandas as pd
import pytest

from exact.runtime.numerical_cache import training_content_contract
from exact.impl.models.selector.feature_contract import SPECIFICITY_FEATURES
from tests.grouped_fitting_test import (
    TinyDataset,
    TinyScorer,
    selector,
    tiny_runner,
    training_rows,
)
from tests.numerical_cache_test import runtime
from tests.scoring_cache_throughput_test import native_training_model


def cached_templates(dataset):
    dataset.verbalization_mode = "llm"
    dataset.only_taxonomy = False
    dataset._verb_temp_path.parent.mkdir(parents=True, exist_ok=True)
    dataset._verb_temp_path.write_text(json.dumps({"templates": {"r": "$SRC relates to $TGT"}}))
    dataset._write_template_cache_metadata()


def test_prepare_templates_binds_cached_contents_before_shallow_training_copy(
    tmp_path, monkeypatch
):
    runtime(tmp_path, monkeypatch)
    model = native_training_model(tmp_path)
    dataset = model._attached_dataset
    cached_templates(dataset)
    monkeypatch.setattr(
        dataset, "_ensure_verbaliser", lambda: pytest.fail("generated cached templates")
    )
    assert training_content_contract(model) is None  # Passive discovery stays passive.
    contract = training_content_contract(model, prepare_templates=True)
    assert contract["dataset"]["template_contents"] == {"r": "$SRC relates to $TGT"}
    assert copy.copy(dataset)._verbalization_templates is dataset._verbalization_templates
    assert training_content_contract(model) == contract
    dataset._verbalization_templates["r"] = "$SRC differs from $TGT"
    assert training_content_contract(model) != contract


def test_template_preparation_failure_is_not_hidden_as_a_cache_miss(tmp_path, monkeypatch):
    runtime(tmp_path, monkeypatch)
    model = native_training_model(tmp_path)
    dataset = model._attached_dataset
    dataset.verbalization_mode = "llm"
    dataset.only_taxonomy = False

    def fail():
        raise OSError("required template preparation failed")

    monkeypatch.setattr(dataset, "_ensure_verbaliser", fail)
    with pytest.raises(OSError, match="required template preparation failed"):
        training_content_contract(model, prepare_templates=True)


@pytest.mark.parametrize("setting", ["EXACT_ENCODER_CUDA_GRAPHS", "EXACT_PAIR_CONTEXT_TEXT_BATCH"])
def test_training_contract_keeps_encoder_execution_and_anchor_dependencies(
    tmp_path, monkeypatch, setting
):
    runtime(tmp_path, monkeypatch)
    model = native_training_model(tmp_path)
    original = training_content_contract(model)
    assert original is not None
    monkeypatch.setenv(setting, "changed")
    assert training_content_contract(model) != original
    monkeypatch.delenv(setting)
    original = training_content_contract(model)
    model._exact_anchor_src_to_tgt = {"s": {"t"}}
    assert training_content_contract(model) != original


def test_global_local_reuse_train_scores_and_refit_changed_selector_with_current_binding(
    tmp_path, monkeypatch
):
    identity = runtime(tmp_path, monkeypatch)
    rows = training_rows()
    pool, reference = tmp_path / "train.tsv", tmp_path / "reference.tsv"
    rows[["Src", "Tgt"]].to_csv(pool, sep="\t", index=False)
    rows[rows.Tgt.str.endswith("0")][["Src", "Tgt"]].to_csv(reference, sep="\t", index=False)

    def runner(name, *, local=False):
        model = native_training_model(tmp_path, local=local)
        dataset = model._attached_dataset
        cached_templates(dataset)
        monkeypatch.setattr(
            dataset, "_ensure_verbaliser", lambda: pytest.fail("generated cached templates")
        )
        monkeypatch.setattr(type(dataset), "__getitem__", TinyDataset.__getitem__)

        def features(self, frame):
            assert self._verbalization_templates is not None
            return TinyDataset.get_features(self, frame).assign(
                **{name: 0.0 for name in SPECIFICITY_FEATURES}
            )

        monkeypatch.setattr(dataset, "get_features", MethodType(features, dataset))
        monkeypatch.setattr(dataset, "_annotate_candidate_similarity_stats", lambda: None)
        dataset._df = TinyDataset.get_features(
            dataset, pd.DataFrame({"Src": [name], "Tgt": ["report-target"]})
        )
        model.calls = []
        monkeypatch.setattr(model, "forward", MethodType(TinyScorer.forward, model))
        head = selector("pairwise" if local else "current_listwise")
        head.training_reference_file_path = str(reference)
        result = tiny_runner(tmp_path / name, dataset, model, head)
        result.supervision_config = {"negative_label_policy": "complete_reference"}
        result.training_candidates_file_path = pool
        return result

    first = runner("global-report")
    first.fit_training_pool(batch_size=5)
    assert first.model.calls
    original = next(first.output_dir.glob("fitting/*/training_scores.json"))
    second = runner("local-report", local=True)
    runtime(
        tmp_path,
        monkeypatch,
        **{
            **identity,
            "role": "local_ranking",
            "inputs": {
                **identity["inputs"],
                "candidates": "different-reporting-pool",
            },
        }
    )
    second.fit_training_pool(batch_size=5)
    assert second.model.calls == []
    assert not list(second.output_dir.glob("fitting/*/training_scores.json"))
    reused = json.loads(
        next(second.output_dir.glob("fitting/*/training_scores.reference.json")).read_text()
    )
    assert Path(reused["aggregate_uri"]) == original.resolve()
    first_head = json.loads(Path(first.models[1].rerank_config["artifact"]).read_text())
    second_head = json.loads(Path(second.models[1].rerank_config["artifact"]).read_text())
    assert first_head["fit_identity"] != second_head["fit_identity"]
    assert first_head["fit_provenance"]["application"]["source_ids"] == ["global-report"]
    assert second_head["fit_provenance"]["application"]["source_ids"] == ["local-report"]
    assert set(second_head["fit_provenance"]["training_sources"]) == set(rows.Src)
