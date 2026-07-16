import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch

from exact.core.contracts.model import IModel
from exact.core.entities.mappings import EntityMapping


def _load_runner_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "exact"
        / "impl"
        / "trainer"
        / "semantic_runner.py"
    )
    spec = importlib.util.spec_from_file_location("semantic_runner_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyDataset:
    dataset_signature = "ontology-signature"
    cache_fingerprint = "candidate-fingerprint-v2"

    def __len__(self):
        return 0


class _DummyModel(IModel):
    def __init__(self, **kwargs):
        super().__init__()

    def runtime_fingerprint_payload(self, **kwargs):
        return {"model": "dummy"}

    def forward(self, *args, **kwargs):
        return {}


def test_primary_checkpoint_fingerprint_includes_dataset_cache_fingerprint(
    tmp_path: Path,
):
    module = _load_runner_module()
    runner = module.SemanticAlignmentRunner(
        dataset=_DummyDataset(),
        model=_DummyModel,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )

    payload = runner._build_checkpoint_fingerprint_payload()

    assert payload["dataset_signature"] == "ontology-signature"
    assert payload["dataset_fingerprint"] == "candidate-fingerprint-v2"


def test_compact_checkpoint_restores_from_single_explanation_store(tmp_path: Path):
    module = _load_runner_module()
    runner = module.SemanticAlignmentRunner(
        dataset=_DummyDataset(),
        model=_DummyModel,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    checkpoint_path = runner.checkpoint_dir / "inference_test.json"
    records = [
        {"src_iri": "s1", "tgt_iri": "t1", "confidences": {"S_final": 0.8}},
        {"src_iri": "s2", "tgt_iri": "t2", "confidences": {"S_final": 0.7}},
        {"src_iri": "s3", "tgt_iri": "t3", "confidences": {"S_final": 0.6}},
    ]

    runner._checkpoint_payload_mode = "compact"
    runner._prepare_audit_shards(
        checkpoint_path,
        enabled=True,
        compression="none",
        records_per_shard=2,
    )
    runner._prepare_candidate_shards(
        checkpoint_path,
        enabled=True,
        compression="none",
        records_per_shard=2,
    )
    runner._append_audit_records(records)
    candidate_rows = [
        runner._candidate_row_from_explanation_record(record) for record in records
    ]
    runner._append_candidate_records([row for row in candidate_rows if row is not None])
    runner._inference_seconds_cumulative = 812.4
    runner._examples_per_second_ema = 1.48
    runner._write_checkpoint_state(
        checkpoint_path,
        module.DatasetMask.inference,
        total_examples=3,
        processed_examples=3,
        mappings=[("s1", "t1", 0.8), ("s2", "t2", 0.7), ("s3", "t3", 0.6)],
        results_json=records,
    )

    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload["checkpoint_payload"] == "compact"
    assert payload["checkpoint_schema_version"] == 3
    assert "results_json" not in payload
    assert "mappings" not in payload
    assert payload["explanation_index_path"] == "../explanations/index.json"
    assert payload["explanation_records_count"] == 3
    assert "audit_manifest_path" not in payload
    assert "candidate_records_manifest_path" not in payload
    assert payload["timing"] == {
        "inference_seconds_cumulative": 812.4,
        "examples_per_second_ema": 1.48,
    }

    runner._inference_seconds_cumulative = 0.0
    runner._restored_inference_seconds_cumulative = 0.0
    runner._examples_per_second_ema = None

    mappings, restored_records, processed = runner._load_checkpoint_state(
        checkpoint_path,
        module.DatasetMask.inference,
    )

    assert processed == 3
    stored_records = [
        {
            **record,
            "run_id": runner._run_id,
            "explanation_schema_version": 1,
        }
        for record in records
    ]
    assert restored_records == stored_records
    assert runner._restored_candidate_rows == candidate_rows
    assert mappings == [("s1", "t1", 0.8), ("s2", "t2", 0.7), ("s3", "t3", 0.6)]
    assert runner.inference_seconds_cumulative == 812.4
    assert runner.examples_per_second_ema == 1.48

    full_json_path = tmp_path / "full_explanations.json"
    runner.write_full_explanations_json(full_json_path)
    assert json.loads(full_json_path.read_text(encoding="utf-8")) == stored_records


def test_legacy_compact_checkpoint_migrates_audit_into_store(tmp_path: Path):
    module = _load_runner_module()
    runner = module.SemanticAlignmentRunner(
        dataset=_DummyDataset(),
        model=_DummyModel,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    checkpoint_path = runner.checkpoint_dir / "inference_legacy.json"
    records = [
        {"src_iri": "s1", "tgt_iri": "t1", "confidences": {"S_final": 0.8}},
        {"src_iri": "s2", "tgt_iri": "t2", "confidences": {"S_final": 0.7}},
    ]

    audit_dir = checkpoint_path.parent / "inference_legacy_audit"
    audit_dir.mkdir()
    (audit_dir / "shard-000000.jsonl").write_text(
        "".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8"
    )
    (audit_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "format": "jsonl",
                "compression": "none",
                "records_per_shard": 2,
                "total_records": 2,
                "shards": [{"path": "shard-000000.jsonl", "records": 2}],
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path.write_text(
        json.dumps(
            {
                "checkpoint_schema_version": 2,
                "kind": "inference",
                "dataset_signature": runner.dataset.dataset_signature,
                "dataset_fingerprint": runner.dataset.cache_fingerprint,
                "checkpoint_fingerprint": runner._checkpoint_fingerprint,
                "checkpoint_fingerprint_payload": runner._checkpoint_fingerprint_payload,
                "total_examples": 2,
                "processed_examples": 2,
                "checkpoint_payload": "compact",
                "audit_manifest_path": "inference_legacy_audit/manifest.json",
            }
        ),
        encoding="utf-8",
    )
    runner._prepare_audit_shards(
        checkpoint_path,
        enabled=True,
        compression="none",
        records_per_shard=2,
    )

    mappings, restored_records, processed = runner._load_checkpoint_state(
        checkpoint_path,
        module.DatasetMask.inference,
    )

    assert processed == 2
    assert restored_records == records
    assert mappings == [("s1", "t1", 0.8), ("s2", "t2", 0.7)]
    assert list(runner._explanation_store.iter_all()) == [
        {
            **record,
            "run_id": runner._run_id,
            "explanation_schema_version": 1,
        }
        for record in records
    ]
    assert runner._restored_candidate_rows == [
        {
            "Src": "s1",
            "Tgt": "t1",
            "ground_truth": None,
            "src_label_text": "",
            "tgt_label_text": "",
            "llm_pair_brief": "",
            "S_final": 0.8,
        },
        {
            "Src": "s2",
            "Tgt": "t2",
            "ground_truth": None,
            "src_label_text": "",
            "tgt_label_text": "",
            "llm_pair_brief": "",
            "S_final": 0.7,
        },
    ]

    runner._write_checkpoint_state(
        checkpoint_path,
        module.DatasetMask.inference,
        total_examples=2,
        processed_examples=2,
        mappings=mappings,
        results_json=restored_records,
    )
    migrated_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert migrated_payload["checkpoint_schema_version"] == 3
    assert migrated_payload["explanation_index_path"]
    assert "audit_manifest_path" not in migrated_payload


def test_checkpoint_restore_discards_uncheckpointed_store_suffix(tmp_path: Path):
    module = _load_runner_module()
    runner = module.SemanticAlignmentRunner(
        dataset=_DummyDataset(),
        model=_DummyModel,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    checkpoint_path = runner.checkpoint_dir / "inference_test.json"
    committed = [{"src_iri": "s1", "tgt_iri": "t1", "confidences": {"S_final": 0.8}}]
    runner._prepare_audit_shards(
        checkpoint_path,
        enabled=True,
        compression="none",
        records_per_shard=1,
    )
    runner._append_audit_records(committed)
    runner._write_checkpoint_state(
        checkpoint_path,
        module.DatasetMask.inference,
        total_examples=2,
        processed_examples=1,
        mappings=[("s1", "t1", 0.8)],
        results_json=committed,
    )
    runner._append_audit_records(
        [{"src_iri": "s2", "tgt_iri": "t2", "confidences": {"S_final": 0.7}}]
    )

    mappings, records, processed = runner._load_checkpoint_state(
        checkpoint_path, module.DatasetMask.inference
    )

    assert processed == 1
    assert mappings == [("s1", "t1", 0.8)]
    assert records == [
        {
            **committed[0],
            "run_id": runner._run_id,
            "explanation_schema_version": 1,
        }
    ]
    assert runner._explanation_store.record_count == 1


def test_final_overlay_uses_store_and_compacts_without_checkpoint_sidecars(
    tmp_path: Path,
):
    module = _load_runner_module()
    runner = module.SemanticAlignmentRunner(
        dataset=_DummyDataset(),
        model=_DummyModel,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    checkpoint_path = runner.checkpoint_dir / "inference_test.json"
    base_record = {
        "src_iri": "s1",
        "tgt_iri": "t1",
        "confidences": {"S_final": 0.8},
        "prediction": {"saved_alignment_member": False},
    }
    runner.results_json.append(base_record)
    runner._prepare_audit_shards(
        checkpoint_path,
        enabled=True,
        compression="none",
        records_per_shard=1,
    )
    runner._append_audit_records([base_record])

    overlay_path = runner._write_final_overlay(
        checkpoint_path,
        pd.DataFrame([{"Src": "s1", "Tgt": "t1", "S_final": 0.8}]),
        [EntityMapping("s1", "t1", score=0.8)],
        threshold=0.7,
        local_alignment=False,
        compression="none",
        records_per_shard=1,
    )

    assert overlay_path == runner.run_layout.explanation_index_path
    assert runner._explanation_store.overlay_count == 1
    assert (
        runner._explanation_store.get("s1")[0]["prediction"]["saved_alignment_member"]
        is True
    )
    runner._explanation_store.compact()
    assert runner._explanation_store.overlay_count == 0
    assert not runner._explanation_store.overlays_dir.exists()
    assert not (checkpoint_path.parent / "inference_test_overlay").exists()
