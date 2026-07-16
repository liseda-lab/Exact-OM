import importlib.util
import json
from pathlib import Path

import torch

from exact.core.contracts.model import IModel


def _load_runner_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "exact" / "impl" / "trainer" / "semantic_runner.py"
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


def test_primary_checkpoint_fingerprint_includes_dataset_cache_fingerprint(tmp_path: Path):
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


def test_compact_checkpoint_restores_from_slim_candidate_sidecar(tmp_path: Path):
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
    candidate_rows = [runner._candidate_row_from_explanation_record(record) for record in records]
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
    assert "results_json" not in payload
    assert "mappings" not in payload
    assert payload["audit_manifest_path"]
    assert payload["candidate_records_manifest_path"]
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
    assert restored_records == records
    assert runner._restored_candidate_rows == candidate_rows
    assert mappings == [("s1", "t1", 0.8), ("s2", "t2", 0.7), ("s3", "t3", 0.6)]
    assert runner.inference_seconds_cumulative == 812.4
    assert runner.examples_per_second_ema == 1.48

    full_json_path = tmp_path / "full_explanations.json"
    runner.write_full_explanations_json(full_json_path)
    assert json.loads(full_json_path.read_text(encoding="utf-8")) == records


def test_legacy_compact_checkpoint_migrates_audit_to_candidate_sidecar(tmp_path: Path):
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

    runner._checkpoint_payload_mode = "compact"
    runner._prepare_audit_shards(
        checkpoint_path,
        enabled=True,
        compression="none",
        records_per_shard=1,
    )
    runner._append_audit_records(records)
    runner._write_checkpoint_state(
        checkpoint_path,
        module.DatasetMask.inference,
        total_examples=2,
        processed_examples=2,
        mappings=[],
        results_json=records,
    )
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload.pop("candidate_records_manifest_path", None)
    payload.pop("candidate_records_count", None)
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    mappings, restored_records, processed = runner._load_checkpoint_state(
        checkpoint_path,
        module.DatasetMask.inference,
    )

    migrated_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert processed == 2
    assert restored_records == records
    assert mappings == [("s1", "t1", 0.8), ("s2", "t2", 0.7)]
    assert migrated_payload["checkpoint_schema_version"] == 2
    assert migrated_payload["candidate_records_manifest_path"]
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
