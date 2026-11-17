"""TRAIN reuse preserves native evidence, full source groups and real judge support."""

import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from exact.runtime.numerical_cache import training_content_contract
from exact.impl.trainer.study_training import (
    export_component_training,
    load_train_judgments,
    prepare_native_features,
    replay_train_judgments,
)
from exact.utils.provenance import sha256_file
from tests.fitting_storage_retention_test import runner_at
from tests.numerical_cache_test import runtime
from tests.scoring_cache_throughput_test import native_training_model


def feature_view(tmp_path):
    source = tmp_path / "source.ofn"
    if not source.exists():
        source.write_text(
            "Ontology(Declaration(Class(<urn:s>)) Declaration(Class(<urn:parent>)) "
            "SubClassOf(<urn:s> <urn:parent>) "
            "AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> "
            '<urn:s> "source class"))'
        )
    model = native_training_model(tmp_path)
    view = model._attached_dataset
    view._candidates = pd.DataFrame({"Src": ["urn:s"], "Tgt": ["urn:t"], "cand_sim": [0.78]})
    return model, view


def test_native_feature_reuse_preserves_objects_metadata_and_invalidates_evidence(
    tmp_path, monkeypatch
):
    runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("EXACT_COMPONENT_FEATURE_CACHE", "1")
    model, first = feature_view(tmp_path)
    contract = training_content_contract(model)
    cold = prepare_native_features(first, contract)
    native = copy.deepcopy(first._entity_feature_cache)
    assert native[("src", "class", "urn:s")]["hierarchy"]["is_a"]
    assert cold.cand_sim.tolist() == [0.78]
    assert first._training_feature_cache_report == {"entity_hits": 0, "entity_misses": 2}
    _, second = feature_view(tmp_path)
    monkeypatch.setattr(
        second, "_bundle_for_class", lambda *args: pytest.fail("recomputed native entities")
    )
    changed_head = {**contract, "model": {"other": "fusion/judge"}}
    warm = prepare_native_features(second, changed_head)
    pd.testing.assert_frame_equal(cold, warm)
    assert second._entity_feature_cache == native
    assert second._training_feature_cache_report == {"entity_hits": 2, "entity_misses": 0}
    for key in ("dataset", "inputs", "implementation", "dependencies"):
        _, third = feature_view(tmp_path)
        changed = {**contract, key: {**contract[key], "mutation": key}}
        prepare_native_features(third, changed)
        assert third._training_feature_cache_report["entity_misses"] == 2


def test_native_feature_storage_bound_corruption_and_errors(tmp_path, monkeypatch):
    runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("EXACT_COMPONENT_FEATURE_CACHE", "1")
    model, first = feature_view(tmp_path)
    contract = training_content_contract(model)
    cold = prepare_native_features(first, contract)
    with sqlite3.connect(tmp_path / "numerical-cache/evidence.sqlite") as connection:
        connection.execute("UPDATE entries SET payload=x'00'")
    _, second = feature_view(tmp_path)
    pd.testing.assert_frame_equal(cold, prepare_native_features(second, contract))
    assert second._training_feature_cache_report["entity_misses"] == 2
    monkeypatch.setenv("EXACT_NUMERICAL_CACHE_MAX_BYTES", "0")
    for _ in range(2):
        _, view = feature_view(tmp_path)
        prepare_native_features(view, {**contract, "namespace": "bounded"})
        assert view._training_feature_cache_report["entity_hits"] == 0
    _, view = feature_view(tmp_path)

    def fail(*args):
        raise OSError("native ontology failure")

    monkeypatch.setattr(view, "get_entity_features", fail)
    with pytest.raises(OSError, match="native ontology failure"):
        prepare_native_features(view, {**contract, "namespace": "failed"})


def test_feature_cache_concurrent_publication_keeps_one_valid_payload(tmp_path, monkeypatch):
    runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("EXACT_COMPONENT_FEATURE_CACHE", "1")
    model, _ = feature_view(tmp_path)
    contract = training_content_contract(model)

    class View:
        def __init__(self):
            self._entity_feature_cache = {}
            self._candidates = pd.DataFrame({"Src": ["s"], "Tgt": ["t"]})

        def get_entity_features(self, iri, side, kind):
            value = {"labels": [iri], "fact": (side, kind, iri)}
            self._entity_feature_cache[(side, kind, iri)] = value
            return value

        def get_features(self, frame):
            for side, column in (("src", "Src"), ("tgt", "Tgt")):
                for iri in frame[column]:
                    if (side, "class", iri) not in self._entity_feature_cache:
                        self.get_entity_features(iri, side, "class")
            return self._entity_feature_cache

    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(lambda _: prepare_native_features(View(), contract), range(2)))
    assert outputs[0] == outputs[1] == prepare_native_features(View(), contract)
    with sqlite3.connect(tmp_path / "numerical-cache/evidence.sqlite") as connection:
        assert connection.execute("SELECT count(*) FROM entries").fetchone()[0] == 1


def bank_fixture(tmp_path, monkeypatch, *, runner=None):
    runner = runner or runner_at(tmp_path)
    model = runner.model
    model.llm_model_name = "fixture-luna"
    model.beta = 0.8
    raw = pd.read_csv(runner.training_candidates_file_path, sep="\t")
    sources = list(dict.fromkeys(raw.Src))[:6]
    pairs = [{"Src": source, "Tgt": f"t{source[1:]}-0"} for source in sources]
    payload = {
        "schema": "exact-component-train-judgments-v1",
        "model_id": "fixture-luna",
        "policy": {"fusion_weight": "beta_u", "beta": 0.8},
        "source_ids": sources,
        "expected_pairs": pairs,
        "predictions": [{**pair, "probability": 0.25} for pair in pairs],
        "provenance": {
            "role": "train",
            "labels_used": False,
            "request_support": "frozen_label_blind",
            "fold_safe": True,
            "evidence": "train_independent_ontology",
            "upstream_fit_dependencies": [],
        },
        "binding": {
            "training_pool_sha256": sha256_file(runner.training_candidates_file_path),
            "reference_sha256": sha256_file(Path(runner.models[1].training_reference_file_path)),
            "dataset_signature": runner.dataset.dataset_signature,
        },
    }

    def save(value=payload):
        path = tmp_path / "bank.json"
        path.write_text(json.dumps(value))
        monkeypatch.setenv("EXACT_COMPONENT_TRAIN_JUDGMENTS", str(path))
        monkeypatch.setenv("EXACT_COMPONENT_TRAIN_JUDGMENTS_SHA256", sha256_file(path))

    def load():
        return load_train_judgments(
            model,
            raw,
            {
                "dataset_signature": runner.dataset.dataset_signature,
                "source_ids": ["report"],
            },
            pool_path=runner.training_candidates_file_path,
            reference_path=runner.models[1].training_reference_file_path,
        )

    save()
    return runner, payload, save, load


@pytest.mark.parametrize(
    "mutation",
    [
        "checksum",
        "model",
        "input",
        "folds",
        "missing",
        "extra",
        "duplicate",
        "probability",
        "policy",
        "support",
    ],
)
def test_train_bank_rejects_stale_incomplete_or_leaky_predictions(tmp_path, monkeypatch, mutation):
    _, payload, save, load = bank_fixture(tmp_path, monkeypatch)
    assert load()["source_ids"] == payload["source_ids"]
    if mutation == "checksum":
        monkeypatch.setenv("EXACT_COMPONENT_TRAIN_JUDGMENTS_SHA256", "bad")
    else:
        if mutation == "model":
            payload["model_id"] = "another-model"
        elif mutation == "input":
            payload["binding"]["training_pool_sha256"] = "changed"
        elif mutation == "folds":
            payload["provenance"]["upstream_fit_dependencies"] = ["trained-on-heldout-labels"]
        elif mutation == "missing":
            payload["predictions"].pop()
        elif mutation == "extra":
            payload["predictions"].append({"Src": "s0", "Tgt": "t0-1", "probability": 0.3})
        elif mutation == "duplicate":
            payload["predictions"].append(payload["predictions"][0])
        elif mutation == "probability":
            payload["predictions"][0]["probability"] = float("nan")
        elif mutation == "policy":
            payload["policy"]["beta"] = 0.6
        else:
            payload["source_ids"].append("report")
        save()
    with pytest.raises(ValueError):
        load()


def test_train_replay_refits_heads_on_matched_complete_groups_without_paid_calls(
    tmp_path, monkeypatch
):
    runner, payload, _, load = bank_fixture(tmp_path, monkeypatch)
    head = runner.models[1]
    original_fit = head.fit_training_artifact
    captured = []

    def fit(frame, *args, **kwargs):
        captured.append(frame.copy())
        return original_fit(frame, *args, **kwargs)

    monkeypatch.setattr(head, "fit_training_artifact", fit)
    runner.fit_training_pool(batch_size=5)
    assert set(source for call in runner.model.calls for source in call["src_iris"]) == set(
        payload["source_ids"]
    )
    training = captured[0]
    assert len(training) == 12  # Both candidates for each of the six sources.
    judged = training[training.Tgt.str.endswith("0")]
    unjudged = training[~training.Tgt.str.endswith("0")]
    assert judged.S_final.tolist() == pytest.approx([0.848] * 6)
    assert unjudged.S_final.tolist() == pytest.approx([0.2] * 6)
    assert unjudged.p_llm.isna().all() and not unjudged.llm_used.any()
    assert judged.p_llm.tolist() == [0.25] * 6
    assert training.S_final.equals(training.S_pair_final)
    artifact = json.loads(Path(head.rerank_config["artifact"]).read_text())
    assert set(artifact["fit_provenance"]["training_sources"]) == set(payload["source_ids"])
    assert (
        artifact["fit_provenance"]["application"]["component_train_judgments"]["sha256"]
        == load()["sha256"]
    )
    # Replay uses current fusion output, never a cached final score.
    changed = training.assign(S_base=0.5, U=0.8, S_final=99, S_pair_final=99)
    replayed = replay_train_judgments(changed, load())
    assert replayed.loc[judged.index, "S_final"].tolist() == pytest.approx([0.34] * 6)
    assert replayed.loc[unjudged.index, "S_final"].tolist() == [0.5] * 6
    missing = changed[changed.Tgt != "t0-0"]
    with pytest.raises(ValueError, match="removed required"):
        replay_train_judgments(missing, load())


def test_offline_replay_binds_intended_hosted_model_without_enabling_calls(tmp_path, monkeypatch):
    runner, _, _, load = bank_fixture(tmp_path, monkeypatch)
    runner.model.llm_model_name = "legacy-local-default"
    runner.model.use_llm = False
    monkeypatch.setenv("EXACT_COMPONENT_JUDGE_MODEL", "fixture-luna")
    assert load()["model_id"] == "fixture-luna"
    monkeypatch.setenv("EXACT_COMPONENT_JUDGE_MODEL", "different-hosted-model")
    with pytest.raises(ValueError, match="model mismatch"):
        load()


def test_unrouted_train_sources_remain_in_the_fitting_support(tmp_path, monkeypatch):
    runner, payload, save, load = bank_fixture(tmp_path, monkeypatch)
    payload["expected_pairs"] = []
    payload["predictions"] = []
    save()
    assert load()["source_ids"] == payload["source_ids"]
    runner.fit_training_pool(batch_size=5)
    fitted = json.loads(Path(runner.models[1].rerank_config["artifact"]).read_text())
    assert set(fitted["fit_provenance"]["training_sources"]) == set(payload["source_ids"])


def test_native_train_packets_are_score_blind_and_invalidate_older_aggregates(
    tmp_path, monkeypatch
):
    from tests.grouped_fitting_test import TinyScorer

    runner_at(tmp_path).fit_training_pool(batch_size=5)

    class WithFacts(TinyScorer):
        def forward(self, **kwargs):
            result = super().forward(**kwargs)
            result["explanations"] = [
                {
                    "selected_labels": {"source": source, "target": target},
                    "prediction": {"ground_truth": 1},
                    "triple_attributions": {
                        "hierarchy": {
                            "is_a": {
                                "source": [
                                    {
                                        "triple": (source, "is_a", "parent"),
                                        "item_id": source + ":fact",
                                        "score": 0.98,
                                    }
                                ],
                                "target": [],
                            }
                        },
                        "similarity": {"source": [], "target": []},
                        "difference": {"source": [], "target": []},
                    },
                    "attributes": {"source": [], "target": []},
                }
                for source, target in zip(kwargs["src_iris"], kwargs["tgt_iris"])
            ]
            return result

    monkeypatch.setenv("EXACT_COMPONENT_TRAIN_FACTS", "1")
    scorer = WithFacts()
    runner_at(tmp_path, scorer).fit_training_pool(batch_size=5)
    assert scorer.calls
    aggregates = [
        json.loads(path.read_text()) for path in tmp_path.glob("fitting/*/training_scores.json")
    ]
    assert len(aggregates) == 2 and aggregates[0]["identity"] != aggregates[1]["identity"]
    rows = next(p["rows"] for p in aggregates if "llm_structured_evidence_packet" in p["rows"][0])
    for row in rows:
        packet = json.loads(row["llm_structured_evidence_packet"])
        assert packet["facts"][0]["item_id"] == row["Src"] + ":fact"
        assert "score" not in packet["facts"][0] and "ground_truth" not in packet
        assert row["selected_labels"] == {"source": row["Src"], "target": row["Tgt"]}


def test_export_preserves_current_judgments_and_shares_identical_compact_payloads(
    tmp_path, monkeypatch
):
    import zstandard as zstd

    runner, payload, _, load = bank_fixture(tmp_path, monkeypatch)
    runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("EXACT_COMPONENT_EXPORT_TRAIN", "1")
    runner.fit_training_pool(batch_size=5)
    receipt = json.loads((tmp_path / "component_training.json").read_text())
    assert sha256_file(Path(receipt["aggregate_uri"])) == receipt["sha256"]
    with zstd.open(receipt["aggregate_uri"], "rt") as stream:
        exported = json.load(stream)
    assert exported["identity"] == receipt["identity"]
    assert set(exported["source_ids"]) == set(payload["source_ids"])
    frame = pd.DataFrame(exported["rows"])
    assert frame.loc[frame.Tgt.str.endswith("0"), "S_final"].tolist() == pytest.approx([0.848] * 6)
    assert frame.S_final.equals(frame.S_pair_final)
    assert "llm_evidence_packet" not in frame and "src_label_text" not in frame
    assert exported["application"]["component_train_judgments"]["sha256"] == load()["sha256"]
    second = tmp_path / "second-cell"
    shared = export_component_training(
        frame,
        output_dir=second,
        cache_dir=second / "fitting",
        application=exported["application"],
        pool_path=runner.training_candidates_file_path,
        reference_path=runner.models[1].training_reference_file_path,
    )
    assert shared == receipt
    assert not (second / "fitting/component_training_scores.json.zst").exists()
    assert json.loads((second / "component_training.json").read_text()) == receipt


def test_export_storage_limit_preserves_progress_without_publishing_partial_receipts(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("EXACT_COMPONENT_EXPORT_TRAIN", "1")
    monkeypatch.setenv("EXACT_COMPONENT_TRAIN_MAX_BYTES", "1")
    with pytest.raises(OSError, match="storage allowance"):
        runner_at(tmp_path).fit_training_pool(batch_size=5)
    assert list(tmp_path.glob("fitting/*/training_scores.json"))
    assert not (tmp_path / "component_training.json").exists()
    assert not list(tmp_path.rglob("*.partial"))


def test_cached_constant_policy_is_independent_of_live_routing(tmp_path, monkeypatch):
    runner, payload, save, load = bank_fixture(tmp_path, monkeypatch)
    runner.model.llm_experiment_enabled = False
    runner.model.llm_experiment_config = dict(fusion_weight="constant", constant_weight=0.5)
    payload["policy"] = dict(fusion_weight="constant", constant_weight=0.5)
    save()
    assert load()["policy"] == payload["policy"]


def test_original_query_mode_exports_train_without_fitting_native_union_selector(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("EXACT_COMPONENT_EXPORT_TRAIN", "1")
    monkeypatch.setenv("EXACT_COMPONENT_LOCAL_QUERY_RANK", "1")
    runner = runner_at(tmp_path)
    selector = runner.models[1]
    selector.enabled = False
    monkeypatch.setattr(
        selector, "fit_training_artifact", lambda *a, **kw: pytest.fail("native union head fitted")
    )
    runner.fit_training_pool(batch_size=5)
    assert runner.component_training_export is not None
    assert runner._training_pool_fitted is True
    assert not list(tmp_path.rglob("selector.json"))


def test_export_reports_actual_native_fold_membership(tmp_path, monkeypatch):
    monkeypatch.setenv("EXACT_COMPONENT_EXPORT_TRAIN", "1")
    runner = runner_at(tmp_path)
    runner.fit_training_pool(batch_size=5)
    report = json.loads((tmp_path / "component-fit-folds.json").read_text())
    actual = report["components"]["selector"]
    artifact = json.loads(Path(actual["artifact"]["path"]).read_text())
    assert actual["actual_fold_map"] == {
        source: fold["fold"] for fold in artifact["folds"] for source in fold["heldout_sources"]
    }
    assert set(actual["actual_fold_map"]) == set(
        pd.read_csv(runner.training_candidates_file_path, sep="\t").Src
    )
    assert actual["reference_component_overlap"]["targets_crossing_folds"] == 0
