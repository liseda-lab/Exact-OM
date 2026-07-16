from __future__ import annotations

import json
import os
from pathlib import Path

from exact.runs import (
    ExplanationStore,
    RunLayout,
    RunManifest,
    RunReader,
    clean_run,
    finalize_artifacts,
    prune_checkpoints,
    refresh_manifest,
)


def _records() -> list[dict]:
    return [
        {
            "src_iri": "source-b",
            "tgt_iri": "target-1",
            "confidences": {"S_final": 0.4},
            "prediction": {"selected": False},
            "run_id": "run-1",
            "explanation_schema_version": 1,
        },
        {
            "src_iri": "source-a",
            "tgt_iri": "target-2",
            "confidences": {"S_final": 0.8},
            "prediction": {"selected": True},
            "run_id": "run-1",
            "explanation_schema_version": 1,
        },
        {
            "src_iri": "source-b",
            "tgt_iri": "target-3",
            "confidences": {"S_final": 0.6},
            "prediction": {"selected": True},
            "run_id": "run-1",
            "explanation_schema_version": 1,
        },
    ]


def test_layout_resolves_v1_and_v2_artifacts(tmp_path: Path) -> None:
    v1 = tmp_path / "v1"
    legacy_alignment = v1 / "model" / "alignment"
    legacy_alignment.mkdir(parents=True)
    mapping = legacy_alignment / "src2tgt.maps_local.tsv"
    mapping.write_text("SrcEntity\tTgtEntity\nsource\ttarget\n", encoding="utf-8")
    explanations = legacy_alignment / "default" / "full_explanations.json"
    explanations.parent.mkdir()
    explanations.write_text('[{"src_iri":"source","tgt_iri":"target"}]', encoding="utf-8")

    legacy = RunLayout.open(v1)
    assert legacy.version == 1
    assert legacy.mapping_path("local") == mapping
    assert legacy.full_explanations_path == explanations

    modern = RunLayout.create(tmp_path / "v2")
    assert modern.version == 2
    assert modern.mapping_path("local") == modern.alignment_dir / "maps_local.tsv"
    assert modern.explanation_index_path.parent.is_dir()


def test_store_resume_recovers_tail_and_matches_uninterrupted_run(
    tmp_path: Path,
) -> None:
    records = _records()
    uninterrupted = ExplanationStore.create(tmp_path / "whole", run_id="run-1")
    uninterrupted.append(records)

    resumed = ExplanationStore.create(tmp_path / "resumed", run_id="run-1")
    resumed.append(records[:1])
    shard = next(iter(resumed._index["shards"].values()))
    shard_path = resumed.directory / shard["path"]
    committed_size = shard_path.stat().st_size
    with shard_path.open("ab") as stream:
        stream.write(b"uncommitted-tail")
    assert shard_path.stat().st_size > committed_size

    resumed = ExplanationStore(resumed.directory, run_id="run-1")
    assert shard_path.stat().st_size == committed_size
    resumed.append(records[1:])
    assert list(resumed.iter_all()) == list(uninterrupted.iter_all())


def test_store_truncates_uncheckpointed_suffix_before_resume(tmp_path: Path) -> None:
    records = _records()
    store = ExplanationStore.create(tmp_path / "run", run_id="run-1")
    store.append(records[:2])
    checkpoint_count = store.record_count
    store.append(records[2:])

    report = store.truncate(checkpoint_count)

    assert report["records"] == checkpoint_count
    assert list(store.iter_all()) == records[:2]
    store.append(records[2:])
    assert list(store.iter_all()) == records


def test_store_get_reads_one_shard_and_compacts_overlays(tmp_path: Path) -> None:
    store = ExplanationStore.create(tmp_path / "run", run_id="run-1")
    store.append(_records())
    store.append_overlay(
        [
            {
                "Src": "source-b",
                "Tgt": "target-1",
                "confidences": {"S_final": 0.95},
                "prediction": {"selected": True},
            }
        ]
    )

    store.shard_reads = 0
    selected = store.get("source-b")
    assert store.shard_reads == 1
    assert selected[0]["confidences"]["S_final"] == 0.95
    assert selected[0]["prediction"]["selected"] is True

    expected = list(store.iter_all())
    report = store.compact()
    assert report["records"] == len(expected)
    assert not store.overlays_dir.exists()
    assert list(store.iter_all()) == expected

    exported = store.export(tmp_path / "full_explanations.json", format="json")
    assert exported.read_text(encoding="utf-8") == json.dumps(
        expected, ensure_ascii=False, separators=(",", ":")
    )


def test_reader_uses_store_for_v2_and_json_fallback_for_v1(tmp_path: Path) -> None:
    v2 = tmp_path / "v2"
    layout = RunLayout.create(v2)
    layout.mapping_path("local").write_text(
        "SrcEntity\tTgtEntity\nsource-a\ttarget-2\n", encoding="utf-8"
    )
    layout.run_stats_path.write_text('{"n_mappings":1}', encoding="utf-8")
    store = ExplanationStore(layout.explanations_dir, run_id="run-1")
    store.append(_records())
    manifest = refresh_manifest(layout, run_id="run-1")

    reader = RunReader.open(v2)
    assert reader.mappings("local").iloc[0]["SrcEntity"] == "source-a"
    assert len(reader.explanations_for("source-b")) == 2
    assert reader.explanation_shard_reads == 1
    assert reader.stats() == {"n_mappings": 1}
    assert reader.manifest()["sessions"] == ["run-1"]
    mapping_artifact = next(
        item for item in manifest.payload["artifacts"] if item["kind"] == "alignment"
    )
    assert len(mapping_artifact["sha256"]) == 64

    v1 = tmp_path / "v1"
    old_alignment = v1 / "model" / "alignment"
    old_alignment.mkdir(parents=True)
    (old_alignment / "src2tgt.maps_local.tsv").write_text(
        "SrcEntity\tTgtEntity\nsource-b\ttarget-1\n", encoding="utf-8"
    )
    old_payload = old_alignment / "default" / "full_explanations.json"
    old_payload.parent.mkdir()
    old_payload.write_text(json.dumps(_records()), encoding="utf-8")
    legacy_reader = RunReader.open(v1)
    assert len(legacy_reader.explanations_for("source-b")) == 2
    assert legacy_reader.manifest()["synthesized"] is True


def test_clean_is_dry_run_capable_and_preserves_foreign_files(tmp_path: Path) -> None:
    layout = RunLayout.create(tmp_path / "run")
    checkpoint = layout.checkpoints_dir / "inference_old.json"
    checkpoint.write_text("{}", encoding="utf-8")
    sidecar = layout.checkpoints_dir / "inference_old_audit" / "shard-000000.jsonl"
    sidecar.parent.mkdir()
    sidecar.write_text("{}\n", encoding="utf-8")
    foreign = layout.checkpoints_dir / "notes.txt"
    foreign.write_text("keep me", encoding="utf-8")
    cache = layout.root / "cache" / "dataset.bin"
    cache.parent.mkdir()
    cache.write_bytes(b"cache")
    manifest = RunManifest.create(layout)
    manifest.register(checkpoint, kind="checkpoint", checksum=False)
    manifest.register(sidecar, kind="checkpoint", checksum=False)
    manifest.register(cache, kind="dataset_cache", checksum=False)
    manifest.write()

    preview = clean_run(layout.root, dry_run=True)
    assert preview.count == 2
    assert checkpoint.exists() and sidecar.exists()

    result = clean_run(layout.root)
    assert result.count == 2
    assert not checkpoint.exists() and not sidecar.exists()
    assert foreign.read_text(encoding="utf-8") == "keep me"
    assert cache.exists()

    clean_run(layout.root, include_dataset_cache=True)
    assert not cache.exists()
    assert foreign.exists()


def test_checkpoint_retention_keeps_latest_valid_manifest_and_sidecars(
    tmp_path: Path,
) -> None:
    layout = RunLayout.create(tmp_path / "run")
    old = layout.checkpoints_dir / "inference_old.json"
    new = layout.checkpoints_dir / "inference_new.json"
    corrupt = layout.checkpoints_dir / "inference_corrupt.json"
    old_sidecar = layout.checkpoints_dir / "inference_old_audit" / "manifest.json"
    new_sidecar = layout.checkpoints_dir / "inference_new_audit" / "manifest.json"
    old_sidecar.parent.mkdir()
    new_sidecar.parent.mkdir()
    old_sidecar.write_text("{}", encoding="utf-8")
    new_sidecar.write_text("{}", encoding="utf-8")
    checkpoint_common = {"processed_examples": 1, "checkpoint_fingerprint": "fixture"}
    old.write_text(
        json.dumps(
            {
                **checkpoint_common,
                "audit_manifest_path": "inference_old_audit/manifest.json",
            }
        )
    )
    new.write_text(
        json.dumps(
            {
                **checkpoint_common,
                "audit_manifest_path": "inference_new_audit/manifest.json",
            }
        )
    )
    corrupt.write_text("{broken", encoding="utf-8")
    os.utime(old, ns=(1, 1))
    os.utime(new, ns=(2, 2))
    os.utime(corrupt, ns=(3, 3))

    result = prune_checkpoints(layout.root, "latest")
    assert result.count == 3
    assert not old.exists() and not old_sidecar.exists() and not corrupt.exists()
    assert new.exists() and new_sidecar.exists()


def test_finalize_compacts_prunes_and_refreshes_manifest(tmp_path: Path) -> None:
    layout = RunLayout.create(tmp_path / "run")
    store = ExplanationStore(layout.explanations_dir, run_id="session")
    store.append(_records())
    store.append_overlay(
        [{"Src": "source-a", "Tgt": "target-2", "prediction": {"selected": False}}]
    )
    checkpoint = layout.checkpoints_dir / "inference_done.json"
    checkpoint.write_text(
        json.dumps({"processed_examples": 1, "checkpoint_fingerprint": "fixture"}),
        encoding="utf-8",
    )

    report = finalize_artifacts(
        layout.root,
        run_id="session",
        save_full_explanations=True,
        checkpoint_retention="none",
    )

    assert report["checkpoints_removed"] == 1
    assert layout.manifest_path.is_file()
    assert layout.full_explanations_path.is_file()
    assert not checkpoint.exists()
    reader = RunReader.open(layout.root)
    assert reader.explanations_for("source-a")[0]["prediction"]["selected"] is False


def test_finalize_does_not_rewrite_already_compacted_store(tmp_path: Path) -> None:
    layout = RunLayout.create(tmp_path / "run")
    store = ExplanationStore(layout.explanations_dir, run_id="session")
    store.append(_records())
    shards_before = sorted(path.name for path in layout.explanation_shards_dir.iterdir())

    report = finalize_artifacts(
        layout.root,
        run_id="session",
        checkpoint_retention="all",
    )

    assert report["compaction"]["records"] == len(_records())
    assert report["compaction"]["before_bytes"] == report["compaction"]["after_bytes"]
    assert sorted(path.name for path in layout.explanation_shards_dir.iterdir()) == shards_before


def test_manifest_indexes_dynamic_deliverables_and_dataset_cache(
    tmp_path: Path,
) -> None:
    layout = RunLayout.create(tmp_path / "run")
    alignment = layout.alignment_dir / "alignment.rdf"
    alignment.write_text("<rdf:RDF />\n", encoding="utf-8")
    evaluation = layout.evaluation_dir / "backend-report.txt"
    evaluation.write_text("report\n", encoding="utf-8")
    cache = layout.cache_dir / "dataset.bin"
    cache.parent.mkdir()
    cache.write_bytes(b"cache")
    dataset = layout.dataset_dir / "dataset.csv"
    dataset.parent.mkdir()
    dataset.write_text("Src,Tgt\nsource,target\n", encoding="utf-8")
    layout.legacy_times_path.write_text("# derived\nTotal: 1.0 minutes\n", encoding="utf-8")

    manifest = refresh_manifest(layout, run_id="session")
    artifacts = {item["path"]: item for item in manifest.payload["artifacts"]}

    assert len(artifacts["alignment/alignment.rdf"]["sha256"]) == 64
    assert len(artifacts["evaluation/backend-report.txt"]["sha256"]) == 64
    assert artifacts["cache/dataset.bin"]["kind"] == "dataset_cache"
    assert artifacts["dataset/dataset.csv"]["kind"] == "dataset"
    assert artifacts["times.txt"]["kind"] == "timing_render"


def test_compressed_store_is_smaller_than_legacy_monolith(tmp_path: Path) -> None:
    records = [
        {
            "src_iri": f"source-{index // 5}",
            "tgt_iri": f"target-{index}",
            "description": "repetitive biomedical explanation " * 20,
            "run_id": "size-test",
            "explanation_schema_version": 1,
        }
        for index in range(1000)
    ]
    store = ExplanationStore.create(tmp_path / "run", run_id="size-test")
    store.append(records)
    stored_bytes = sum(path.stat().st_size for path in store.directory.rglob("*") if path.is_file())
    legacy_bytes = len(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    assert stored_bytes <= legacy_bytes
