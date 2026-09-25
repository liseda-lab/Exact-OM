"""Bounded E01 score replay preserves populations, scores and extraction semantics."""

import copy
import json

import pandas as pd
import pytest
import zstandard as zstd

from exact.core.entities.configs.yaml_io import dump_yaml_document
from exact.experiments.extraction_replay import (
    build_packet,
    run_replay,
    validate_packet,
)
from exact.runs.store import ExplanationStore


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def replay(tmp_path):
    source = tmp_path / "source"
    universe = tmp_path / "sources.txt"
    universe.write_text("urn:a\nurn:b\n")
    config = {
        "data": {
            "reference_role": "valid",
            "refs": {"valid": "valid.tsv"},
            "execution_mode": "global_alignment",
            "source_universe": str(universe),
        },
        "run": {"source_cap": 2, "seed": 17},
        "matching": {
            "threshold": 0.7,
            "cardinality": 1,
            "target_cardinality": 1,
            "relation_prediction": "none",
            "extraction": {"mode": "greedy", "assignment_component_cap": 500},
        },
        "pipeline": [{"params": {"generate_llm_rationales": False}}],
        "evaluation": {"backends": ["builtin"]},
    }
    (source / "_inputs").mkdir(parents=True)
    (source / "_inputs/resolved.config.yaml").write_text(dump_yaml_document(config))
    write_json(
        source / "experiment_manifest.json",
        {
            "experiment_id": "E01",
            "split_role": "development",
            "source_cap": 2,
            "runtime_versions": {"pyowl_core": "fixture"},
            "status": "complete",
        },
    )
    write_json(
        source / "checkpoints/inference_additional_models_abc.json",
        {"complete": True, "candidate_records_count": 2},
    )
    rows = [
        {"Src": "urn:a", "Tgt": "urn:z", "SrcKind": "class", "TgtKind": "class", "S_final": 0.8},
        {"Src": "urn:b", "Tgt": "urn:q", "SrcKind": "class", "TgtKind": "class", "S_final": 0.75},
    ]
    with zstd.open(
        source / "checkpoints/inference_additional_models_abc.jsonl.zst", "wt"
    ) as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    exacts = [
        {
            "Src": src,
            "Tgt": tgt,
            "SrcKind": "class",
            "TgtKind": "class",
            "prefiltered": True,
            "Scores": 1.0,
        }
        for src, tgt in [("urn:a", "urn:x"), ("urn:a", "urn:y"), ("urn:b", "urn:z")]
    ]
    (source / "dataset/sampled_inputs").mkdir(parents=True)
    pd.DataFrame([{**row, "prefiltered": False} for row in rows] + exacts).to_csv(
        source / "dataset/dataset.csv", index=False
    )
    write_json(source / "dataset/dataset.meta.json", {"dataset_signature": "fixture"})
    pool = {
        "fingerprint": "pool",
        "gold_free_summary": {"source_entities": 2, "candidate_pairs": 2},
    }
    for name in ("candidate_pool_manifest.json", "candidate_pool_sample_manifest.json"):
        write_json(source / "dataset" / name, pool)
    (source / "dataset/sampled_inputs/full_reference.tsv").write_text(
        "SrcEntity\tTgtEntity\tRelation\nurn:a\turn:x\t=\nurn:b\turn:z\t=\n"
    )
    (source / "dataset/sampled_inputs/training_reference.tsv").write_text(
        "SrcEntity\tTgtEntity\tRelation\n"
    )
    write_json(
        source / "stats/run_stats.json",
        {"source_sampling": {"selected_sources": 2}, "coverage": 0, "timing": {"old": 100}},
    )
    ExplanationStore(source / "explanations").append(
        [
            {
                "src_iri": row["Src"],
                "tgt_iri": row["Tgt"],
                "confidences": {"S_final": row["S_final"]},
                "contributions": {"C_label": row["S_final"] - 0.5},
                "reconstruction": {"score": row["S_final"], "baseline": 0.5},
                "prediction": {"saved_alignment_member": False},
            }
            for row in rows
        ]
    )
    packet_path = tmp_path / "packet.json"
    packet = build_packet(
        source, packet_path, expected_sources=2, expected_scored_pairs=2, expected_protected_pairs=3
    )
    return config, source, packet, packet_path


@pytest.mark.parametrize(
    "mode",
    [
        "threshold",
        "greedy",
        "mutual_best",
        "stable_marriage",
        "assignment_accepted_utility",
        "assignment_legacy",
    ],
)
def test_replay_uses_real_extraction_without_models(replay, tmp_path, monkeypatch, mode):
    from exact.impl.trainer.runner import SemanticAlignmentRunner

    monkeypatch.setattr(
        SemanticAlignmentRunner,
        "__init__",
        lambda *args, **kwargs: pytest.fail("model runner constructed"),
    )
    config, source, packet, packet_path = replay
    config = copy.deepcopy(config)
    config["matching"]["extraction"].update(mode=mode, anchor_conflict_policy="compete")
    if mode == "threshold":
        config["matching"].update(cardinality=None, target_cardinality=None)
    resolved = tmp_path / "config.yaml"
    resolved.write_text(dump_yaml_document(config))
    output = tmp_path / mode
    wrapper = tmp_path / "job.yaml"
    wrapper.write_text(
        dump_yaml_document({"job": {"config_file": str(resolved), "output_dir": str(output)}})
    )
    run_replay(wrapper, packet_path)
    alignment = pd.read_csv(output / "alignment/maps_global.tsv", sep="\t")
    assert len(alignment) == (5 if mode == "threshold" else 2)
    if mode != "threshold":
        assert alignment.SrcEntity.is_unique and alignment.TgtEntity.is_unique
    assert set(alignment.Score) <= {1.0, 0.8, 0.75}
    stats = json.loads((output / "stats/run_stats.json").read_text())
    assert stats["extraction_replay"]["new_scoring_calls"] == 0
    assert stats["observed_execution"]["mode"] == "extraction_only_replay"
    assert "timing" not in stats
    assert stats["explanation_reconstruction"]["failed_rows"] == 0
    decisions = json.loads((output / "candidate_decisions.json").read_text())
    assert len(decisions["records"]) == 5
    for record in decisions["records"]:
        assert len([event for event in record["events"] if event["stage"] == "extraction"]) == 1
    assert (output / "evaluation/evaluation_results.json").is_file()
    assert (output / "run_manifest.json").is_file()


def test_replay_rejects_changed_scoring_and_private_reference(replay):
    config, source, packet, _ = replay
    altered = copy.deepcopy(config)
    altered["matching"]["threshold"] = 0.6
    with pytest.raises(ValueError, match="changed scoring"):
        validate_packet(packet, altered)
    altered = copy.deepcopy(config)
    altered["data"]["refs"]["test"] = "private.tsv"
    with pytest.raises(ValueError, match="private"):
        validate_packet(packet, altered)


def test_replay_rejects_mutated_checkpoint(replay):
    config, source, packet, _ = replay
    (source / "checkpoints/inference_additional_models_abc.jsonl.zst").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        validate_packet(packet, config)
