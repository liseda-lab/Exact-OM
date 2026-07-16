from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import zstandard as zstd

from exact.core.contracts.model import IModel
from exact.core.entities.configs.dataset import DatasetMask
from exact.impl.trainer import SemanticAlignmentRunner
from exact.runs import ExplanationStore, RunLayout, RunReader, refresh_manifest

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "runs" / "legacy_v1"
INTERRUPTION_WORKER = Path(__file__).parent / "helpers" / "wp_l_interrupted_run.py"
CRASH_EXIT_CODE = 86


class _LegacyGoldenDataset:
    dataset_signature = "wp-l-legacy-golden"
    cache_fingerprint = "wp-l-candidates-v1"

    def __len__(self) -> int:
        return 3


class _LegacyGoldenModel(IModel):
    def runtime_fingerprint_payload(self, **kwargs: Any) -> dict[str, str]:
        return {"model": "wp-l-legacy-golden-v1"}

    def runtime_fingerprint(self) -> str:
        payload = json.dumps(self.runtime_fingerprint_payload(), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}


def _manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        path = manifest_path.parent / shard["path"]
        records.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    return records


def _size_records(count: int = 1000, source_count: int = 50):
    records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for index in range(count):
        source_number = index % source_count
        source = f"https://example.org/source/{source_number:04d}"
        target = f"https://example.org/target/{index:06d}"
        source_label = f"biomedical concept {source_number}"
        score = 0.65 + (index % 30) / 100
        pair_brief = f"Candidate {index} shares normalized label evidence and hierarchy context."
        records.append(
            {
                "explanation_schema_version": 1,
                "src_iri": source,
                "tgt_iri": target,
                "src_kind": "class",
                "tgt_kind": "class",
                "kind": "class",
                "llm_pair_brief": pair_brief,
                "selected_labels": {
                    "source": source_label,
                    "target": source_label,
                },
                "prediction": {
                    "ground_truth": index % 7 == 0,
                    "llm_decision": None,
                },
                "confidences": {
                    "s_label": 0.91,
                    "S_base": 0.82,
                    "S_struct": 0.73,
                    "S_final": score,
                },
                "qualities": {"q_label": 0.88, "Q_struct": 0.77},
                "weights": {
                    "w_c": 0.6,
                    "w_struct": 0.4,
                    "w_i": 0.0,
                    "U": 0.1,
                    "U_ind": 0.05,
                    "U_dis": 0.05,
                },
                "importances": {
                    "I_label": 0.55,
                    "I_struct": 0.45,
                    "I_llm": 0.0,
                },
                "context": {
                    "source": [
                        "shared hierarchy relation",
                        "shared description token " * 8,
                    ],
                    "target": [
                        "shared hierarchy relation",
                        "shared description token " * 8,
                    ],
                },
                "run_id": "size-fixture",
            }
        )
        candidates.append(
            {
                "Src": source,
                "Tgt": target,
                "ground_truth": index % 7 == 0,
                "src_label_text": source_label,
                "tgt_label_text": source_label,
                "llm_pair_brief": pair_brief,
                "s_label": 0.91,
                "S_base": 0.82,
                "S_struct": 0.73,
                "S_final": score,
                "q_label": 0.88,
                "Q_struct": 0.77,
                "w_c": 0.6,
                "w_struct": 0.4,
                "w_i": 0.0,
                "U": 0.1,
                "U_ind": 0.05,
                "U_dis": 0.05,
                "I_label": 0.55,
                "I_struct": 0.45,
                "I_llm": 0.0,
            }
        )
    return records, candidates


def _write_legacy_stream(directory: Path, records: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True)
    raw = b"".join(
        (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )
    shard = directory / "shard-000000.jsonl.zst"
    shard.write_bytes(zstd.ZstdCompressor(level=3).compress(raw))
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "format": "jsonl",
                "compression": "zstd",
                "records_per_shard": 50000,
                "total_records": len(records),
                "shards": [{"path": shard.name, "records": len(records)}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _directory_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _normalized_records(reader: RunReader) -> list[dict[str, Any]]:
    records = []
    for record in reader.iter_explanations():
        normalized = dict(record)
        normalized.pop("run_id", None)
        records.append(normalized)
    return records


def test_committed_pre_wp_l_golden_reads_restores_and_migrates(
    tmp_path: Path,
) -> None:
    legacy_reader = RunReader.open(FIXTURE_ROOT)
    golden = list(legacy_reader.iter_explanations())

    assert legacy_reader.layout.version == 1
    assert legacy_reader.mappings("global")["Score"].tolist() == [0.91, 0.78]
    assert legacy_reader.stats() == {"n_mappings": 2, "fixture": "pre-wp-l"}

    runner = SemanticAlignmentRunner(
        dataset=_LegacyGoldenDataset(),
        model=_LegacyGoldenModel,
        device=torch.device("cpu"),
        output_dir=tmp_path / "restore",
    )
    checkpoint = FIXTURE_ROOT / "model" / "checkpoints" / "inference_legacy.json"
    mappings, audit_records, processed = runner._load_checkpoint_state(
        checkpoint,
        DatasetMask.inference,
    )

    assert processed == 3
    assert mappings == [
        ("https://example.org/source/A", "https://example.org/target/alpha", 0.91),
        ("https://example.org/source/A", "https://example.org/target/beta", 0.31),
        ("https://example.org/source/B", "https://example.org/target/gamma", 0.78),
    ]
    assert runner.inference_seconds_cumulative == 4.25
    assert runner.examples_per_second_ema == 0.71

    legacy_export = tmp_path / "legacy-export.json"
    runner.write_full_explanations_json(legacy_export)
    assert json.loads(legacy_export.read_text(encoding="utf-8")) == golden

    checkpoint_dir = checkpoint.parent
    candidates = _manifest_records(checkpoint_dir / "inference_legacy_candidates" / "manifest.json")
    overlays = _manifest_records(checkpoint_dir / "inference_legacy_overlay" / "manifest.json")
    union_records = [
        runner._union_explanation_record(candidate, record)
        for candidate, record in zip(candidates, audit_records)
    ]
    modern = tmp_path / "migrated"
    store = ExplanationStore.create(modern)
    store.append(union_records)
    store.append_overlay(overlays)
    store.compact()
    refresh_manifest(RunLayout.open(store.directory.parent))

    assert list(RunReader.open(modern).iter_explanations()) == golden


def test_default_store_is_smaller_than_pre_wp_l_compressed_parallel_streams(
    tmp_path: Path,
) -> None:
    records, candidates = _size_records()
    modern = ExplanationStore.create(tmp_path / "modern", run_id="size-fixture")
    modern.append(records)

    legacy = tmp_path / "legacy"
    _write_legacy_stream(legacy / "audit", records)
    _write_legacy_stream(legacy / "candidates", candidates)

    modern_bytes = _directory_bytes(modern.directory)
    legacy_bytes = _directory_bytes(legacy)
    assert modern_bytes <= legacy_bytes, (
        f"layout-v2 store used {modern_bytes} bytes; pre-WP-L compressed audit and "
        f"candidate streams used {legacy_bytes} bytes"
    )


def test_hard_interruption_resume_and_finalization_match_uninterrupted_run(
    tmp_path: Path,
) -> None:
    resumed = tmp_path / "resumed"
    uninterrupted = tmp_path / "uninterrupted"
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}

    crashed = subprocess.run(
        [sys.executable, str(INTERRUPTION_WORKER), "crash", str(resumed)],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert crashed.returncode == CRASH_EXIT_CODE, crashed.stderr
    interrupted_checkpoint = resumed / "checkpoints" / "inference_acceptance.json"
    interrupted_payload = json.loads(interrupted_checkpoint.read_text(encoding="utf-8"))
    assert interrupted_payload["processed_examples"] == 2
    assert interrupted_payload["explanation_records_count"] == 2

    for mode, output in (("resume", resumed), ("uninterrupted", uninterrupted)):
        completed = subprocess.run(
            [sys.executable, str(INTERRUPTION_WORKER), mode, str(output)],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    resumed_reader = RunReader.open(resumed)
    uninterrupted_reader = RunReader.open(uninterrupted)
    assert _normalized_records(resumed_reader) == _normalized_records(uninterrupted_reader)
    assert len(list(resumed_reader.iter_explanations())) == 6

    checkpoint_files = sorted((resumed / "checkpoints").glob("*.json"))
    assert checkpoint_files == [interrupted_checkpoint]
    final_checkpoint = json.loads(interrupted_checkpoint.read_text(encoding="utf-8"))
    assert final_checkpoint["processed_examples"] == 6
    assert final_checkpoint["explanation_records_count"] == 6
    assert not (resumed / "explanations" / "overlays").exists()
    assert not list(resumed.rglob("*_overlay"))

    manifest = resumed_reader.manifest()
    listed_paths = {artifact["path"] for artifact in manifest["artifacts"]}
    actual_paths = {
        path.relative_to(resumed).as_posix()
        for path in resumed.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    }
    assert listed_paths == actual_paths
    for artifact in manifest["artifacts"]:
        path = resumed / artifact["path"]
        assert artifact["bytes"] == path.stat().st_size
        if artifact["kind"] in {"alignment", "evaluation"}:
            assert len(artifact["sha256"]) == 64
