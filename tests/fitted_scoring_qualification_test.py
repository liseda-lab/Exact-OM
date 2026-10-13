"""Fitted throughput checks use complete frozen pools, real heads and no raw labels."""

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pandas as pd
import pytest
import torch

from exact.core.entities.configs.config import ConfigModel
from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer
from exact.impl.models.selector import CandidateSetSelector
from tools import fitted_scoring_qualification as fitted
from tools.benchmark_scoring_throughput import isolated_cache_root
from tools.compare_scoring_parity import compare_records


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def saved_run(tmp_path):
    run = tmp_path / "scientific"
    config = ConfigModel()
    config.data.execution_mode = "global_alignment"
    config.data.source, config.data.target = tmp_path / "s.owl", tmp_path / "t.owl"
    config.data.refs = {"train": tmp_path / "secret-reference.tsv"}
    config.data.train_candidates = tmp_path / "secret-training.tsv"
    config.matching.fusion.mode = "analytic_fitted"
    config.matching.calibration.mode = "platt"
    config.selector.rerank.mode = "analytic"
    config.selector.runtime_global_only = False
    config.candidates.cross_encoder.mode = "on"
    write(run / "config.yaml", config.model_dump(mode="json", by_alias=True))
    fusion = write(run / "fitting/fit/fusion.json", {"parameters": [1]})
    write(
        run / "fitting/fit/selector.json",
        {
            "kind": "fitted_selector",
            "fit_provenance": {
                "application": {"dataset_signature": "pair"},
                "training_sources": ["train"],
            },
        },
    )
    write(
        run / "fitting/score_calibrator.json",
        {
            "kind": "score_calibrator",
            "fit_provenance": {
                "application": {"dataset_signature": "pair"},
                "training_sources": ["training"],
            },
        },
    )
    write(run / "fitting/cross_encoder/retrieval_artifact.json", {})
    write(
        run / "dataset/candidate_pool_manifest.json",
        {"models": {"cross_encoder": {"sha256": "a" * 64}}},
    )
    write(run / "dataset/verbalization_templates.json", {})
    pd.DataFrame(
        [
            {
                "Src": s,
                "Tgt": t,
                "cand_sim": 0.5,
                "src_obj_ic_mean": 0.7,
                "tgt_hier_total_count": 8,
                "Label": "DO_NOT_MATERIALIZE",
            }
            for s, t in [("s2", "b"), ("s2", "a"), ("s1", "b"), ("s1", "c"), ("s3", "d")]
        ]
    ).to_csv(run / "dataset/dataset.csv", index=False)
    write(
        run / "explanations/index.json",
        {
            "total_records": 4,
            "sources": {"s2": {"records": 2}, "s1": {"records": 2}, "s3": {"records": 0}},
        },
    )
    checkpoint = write(
        run / "checkpoints/inference_1.json",
        {
            "dataset_signature": "pair",
            "processed_examples": 4,
            "explanation_records_count": 4,
            "checkpoint_fingerprint_payload": {
                "models": [
                    {
                        "class": "PairAdaptiveSemanticScorer",
                        "payload": {
                            "pair_adaptive_channels": {
                                "experiments": {
                                    "fusion_artifact": fitted.bind(fusion),
                                    "fusion_effective": {"tau": 0.4, "gamma": 1.2, "beta": 0.8},
                                }
                            }
                        },
                    }
                ]
            },
        },
    )
    return SimpleNamespace(
        fitted_run=run, checkpoint=checkpoint, minimum_pairs=3, output=tmp_path / "panel"
    )


def test_preparation_loads_only_candidate_metadata_and_complete_groups(tmp_path, monkeypatch):
    args = saved_run(tmp_path)
    real_read = pd.read_csv

    def read(*a, **kw):
        assert kw["usecols"]("Label") is False
        assert kw["usecols"]("Features") is False
        assert kw["usecols"]("Score") is False
        return real_read(*a, **kw)

    monkeypatch.setattr(pd, "read_csv", read)
    original = Path.open

    def guarded(path, *a, **kw):
        assert not path.name.startswith("secret-")
        assert path.name != "training_scores.json"
        return original(path, *a, **kw)

    monkeypatch.setattr(Path, "open", guarded)
    manifest = fitted.prepare(args)
    panel = json.loads((args.output / "workload.json").read_text())
    assert [(q["source"], q["candidates"]) for q in panel["queries"]] == [
        ("s2", ["b", "a"]),
        ("s1", ["b", "c"]),
    ]
    config = ConfigModel.load_config(args.output / "config.json")
    assert not config.data.refs and config.data.train_candidates is None
    assert str(config.matching.fusion.artifact) == manifest["heads"]["fusion"]["path"]
    assert str(config.selector.rerank.artifact) == manifest["heads"]["selector"]["path"]
    assert "Label" not in (args.output / "candidates.json").read_text()
    metadata = json.loads((args.output / "candidates.json").read_text())
    assert all(
        row["src_obj_ic_mean"] == 0.7 and row["tgt_hier_total_count"] == 8 for row in metadata
    )
    assert manifest["raw_training_or_reference_inputs_read"] is False
    assert manifest["promotable_scientific_result"] is False


def test_changed_fit_and_missing_complete_population_fail_closed(tmp_path):
    args = saved_run(tmp_path)
    (args.fitted_run / "fitting/fit/fusion.json").write_text("changed")
    with pytest.raises(ValueError, match="fusion head changed"):
        fitted.prepare(args)
    assert not args.output.exists()


def test_insufficient_complete_population_is_not_silently_truncated(tmp_path):
    args = saved_run(tmp_path)
    args.minimum_pairs = 5
    with pytest.raises(ValueError, match="Not enough complete"):
        fitted.prepare(args)
    assert not args.output.exists()


def test_qualification_binding_rejects_later_scientific_mutation(tmp_path):
    args = saved_run(tmp_path)
    fitted.prepare(args)
    parameters = {
        "config": args.output / "config.json",
        "workload": args.output / "workload.json",
        "templates": args.fitted_run / "dataset/verbalization_templates.json",
    }
    fitted.load(args.output / "fitted-manifest.json", **parameters)
    (args.fitted_run / "fitting/fit/selector.json").write_text("changed")
    with pytest.raises(ValueError, match="Changed fitted"):
        fitted.load(args.output / "fitted-manifest.json", **parameters)


def test_forbid_scientific_output_and_cache_reuse(tmp_path):
    args = saved_run(tmp_path)
    args.output = args.fitted_run / "qualification"
    with pytest.raises(ValueError, match="inside the scientific run"):
        fitted.prepare(args)
    root = isolated_cache_root(tmp_path / "output", tmp_path / "nvme")
    (root / "existing.sqlite").write_bytes(b"do not touch")
    with pytest.raises(ValueError, match="must be empty"):
        isolated_cache_root(tmp_path / "another", root)
    assert (root / "existing.sqlite").read_bytes() == b"do not touch"


def test_fixture_matches_real_prompt_interface_and_records_changed_evidence():
    scorer = SimpleNamespace(
        hosted_decision_labels=["A", "B"],
        llm_experiment_enabled=False,
        max_total_tokens_llm_summary=8,
        _normalize_text=lambda x: " ".join(x.split()),
    )
    scorer._brief_prompt = MethodType(PairAdaptiveSemanticScorer._brief_prompt, scorer)
    scorer._hosted_decision_prompt = MethodType(
        PairAdaptiveSemanticScorer._hosted_decision_prompt, scorer
    )
    fixtures = []
    for packet in ("short evidence", "short evidence", "different evidence"):
        fixture = fitted.HostedFixtures("cpu", scorer)
        briefs = fixture.briefs(["source"], ["target"], [packet])
        assert briefs == [packet]  # Match the existing short-packet pass-through.
        result = fixture.judgments(["source"], ["target"], briefs, [""])
        assert torch.isfinite(result).all() and 0 <= result[0] <= 1
        assert fixture.records[-1]["prompt"] == scorer._hosted_decision_prompt(
            "source", "target", packet, ""
        )
        fixtures.append(fixture.records)
    assert compare_records(fixtures[0], fixtures[1])["status"] == "passed"
    assert compare_records(fixtures[0], fixtures[2])["status"] == "failed"
    with pytest.raises(ValueError, match="Incomplete decision"):
        fixture.judgments(["source"], ["target"], [], [""])


def test_selector_rows_preserve_metadata_and_reject_nonfinite():
    metadata = [{"Src": "s", "Tgt": "t", "cand_sim": 0.8}]
    rows = fitted.selector_rows(
        metadata, ["s"], ["t"], {"S_final": torch.tensor([0.9]), "S_base": torch.tensor([0.8])}
    )
    assert rows[0]["cand_sim"] == 0.8 and rows[0]["S_final"] == pytest.approx(0.9)
    assert "S_final" not in metadata[0]
    with pytest.raises(ValueError, match="Nonfinite"):
        fitted.selector_rows(metadata, ["s"], ["t"], {"S_final": torch.tensor([float("nan")])})


def test_real_fitted_selector_and_mutual_best_run_without_fitting(tmp_path, monkeypatch):
    config = ConfigModel()
    config.selector.enabled = config.selector.runtime_enabled = True
    config.selector.runtime_global_only = False
    config.selector.rerank.mode = "analytic"
    config.matching.calibration.mode = "platt"
    config.matching.threshold = 0.7
    config.matching.extraction.mode = "mutual_best"
    config.matching.extraction.anchor_conflict_policy = "compete"
    config.resolve_dependencies()
    n = len(CandidateSetSelector.ACCEPT_FEATURE_NAMES)
    artifact = write(
        tmp_path / "selector.json",
        {
            "schema_version": 1,
            "kind": "fitted_selector",
            "fit_identity": "fixture-fit",
            "explanation_schema": "feature_additive",
            "accept_threshold": 0.5,
            "oof_metrics": {},
            "rank_model": {"model_type": "analytic"},
            "accept_model": {
                "weights": [0.0] * n,
                "mean": [0.0] * n,
                "scale": [1.0] * n,
                "bias": 3.0,
            },
            "fit_provenance": {
                "rank_features": CandidateSetSelector.RANK_FEATURE_NAMES,
                "accept_features": CandidateSetSelector.ACCEPT_FEATURE_NAMES,
                "score_calibration": "platt",
                "recipe": {"rerank": config.selector.rerank.model_dump(exclude={"artifact"})},
                "application": {"dataset_signature": "pair"},
                "training_sources": ["training"],
            },
        },
    )
    config.selector.rerank.artifact = str(artifact)
    config.matching.calibration.artifact = str(
        write(
            tmp_path / "calibration.json",
            {
                "schema_version": 1,
                "kind": "score_calibrator",
                "calibrator": {"mode": "platt", "slope": 1.0, "intercept": 0.0},
            },
        )
    )

    def no_fit(*a, **kw):
        pytest.fail("Qualification tried to fit instead of loading the saved head")

    monkeypatch.setattr(CandidateSetSelector, "fit_training_artifact", no_fit)
    monkeypatch.setattr(CandidateSetSelector, "_load_training_reference_pairs", no_fit)
    rows = [
        {"Src": s, "Tgt": t, "S_final": p, "s_label": p, "cand_sim": p}
        for s, t, p in (("s1", "t1", 0.9), ("s1", "t2", 0.1), ("s2", "t1", 0.7), ("s2", "t3", 0.8))
    ]
    records = [{"src_iri": r["Src"], "tgt_iri": r["Tgt"], "confidences": {}} for r in rows]
    result = fitted.apply_selector(
        config, SimpleNamespace(dataset_signature="pair"), None, rows, records, set()
    )
    assert len(result["rows"]) == 4
    assert result["mappings"] == [["s1", "t1"], ["s2", "t3"]]
    assert all("selector_explanation" in row for row in records)
    assert all("score_calibration" in row for row in records)
    assert result["hosted_quality_established"] is False


def test_uncheckpointed_index_tail_is_not_admitted(tmp_path):
    args = saved_run(tmp_path)
    index_path = args.fitted_run / "explanations/index.json"
    index = json.loads(index_path.read_text())
    index["total_records"] += 1
    write(index_path, index)
    with pytest.raises(ValueError, match="durable inference checkpoint"):
        fitted.prepare(args)


def test_loaded_primary_contract_allows_only_declared_diagnostic_change(tmp_path):
    payload = {
        "model": "frozen",
        "pair_adaptive_channels": {
            "experiments": {"diff": {"controlled_perturbations": True, "dump_components": True}}
        },
    }
    checkpoint = write(
        tmp_path / "checkpoint.json",
        {"checkpoint_fingerprint_payload": {"models": [{"payload": payload}]}},
    )
    manifest = {"dependencies": {"checkpoint": fitted.bind(checkpoint)}}
    scorer = SimpleNamespace(_runtime_fingerprint_payload=lambda: payload)
    assert fitted.validate_primary(scorer, manifest, "recorded") == payload
    payload["pair_adaptive_channels"]["experiments"]["diff"]["controlled_perturbations"] = False
    fitted.validate_primary(scorer, manifest, "natural")
    payload["pair_adaptive_channels"]["experiments"]["diff"]["dump_components"] = False
    with pytest.raises(ValueError, match="recipe differs"):
        fitted.validate_primary(scorer, manifest, "natural")


def test_metadata_round_trip_preserves_close_numeric_values():
    value = 0.7123456789012345
    frame = pd.DataFrame([{"cand_sim": value, "optional": float("nan")}])
    records = json.loads(json.dumps(fitted.records_without_nan(frame), allow_nan=False))
    assert records == [{"cand_sim": value, "optional": None}]
