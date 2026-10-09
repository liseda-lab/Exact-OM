import json

import pytest

from exact.runs.store import ExplanationStore


def row(i):
    return {"src_iri": "source", "tgt_iri": str(i), "scores": {"final": i / 10}}


@pytest.mark.parametrize("compression", ["none", "zstd"])
def test_same_size_corruption_rejected_before_any_records(tmp_path, compression):
    store = ExplanationStore(tmp_path, compression=compression)
    store.append([row(1), row(2)])
    shard = next(iter(store._index["shards"].values()))
    path = tmp_path / shard["path"]
    data = bytearray(path.read_bytes())
    data[len(data)//2] ^= 1
    path.write_bytes(data)
    reopened = ExplanationStore(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        next(reopened.iter_all())


def test_committed_prefix_survives_failed_index_write_and_reopen(tmp_path, monkeypatch):
    store = ExplanationStore(tmp_path)
    store.append([row(1)])
    def fail(_):
        raise OSError("NAS write failed")
    monkeypatch.setattr(store, "_write_index", fail)
    with pytest.raises(OSError, match="NAS"):
        store.append([row(2)])
    reopened = ExplanationStore(tmp_path)
    assert [r["tgt_iri"] for r in reopened.iter_all()] == ["1"]
    reopened.append([row(2)])
    assert [r["tgt_iri"] for r in ExplanationStore(tmp_path).iter_all()] == ["1", "2"]


def test_failure_after_index_publication_keeps_referenced_data(tmp_path, monkeypatch):
    store = ExplanationStore(tmp_path)
    store.append([row(1)])
    original = store._write_index
    def fail_after_write(value):
        original(value)
        raise OSError("directory fsync failed")
    monkeypatch.setattr(store, "_write_index", fail_after_write)
    with pytest.raises(OSError, match="fsync"):
        store.append([row(2)])
    assert [r["tgt_iri"] for r in ExplanationStore(tmp_path).iter_all()] == ["1", "2"]


def test_legacy_index_can_be_read_and_append_checks_new_frames(tmp_path):
    store = ExplanationStore(tmp_path, compression="none")
    store.append([row(1)])
    raw = json.loads(store.index_path.read_text())
    for shard in raw["shards"].values():
        shard.pop("frames")
    raw.pop("integrity_schema_version")
    store.index_path.write_text(json.dumps(raw))
    reopened = ExplanationStore(tmp_path)
    reopened.append([row(2)])
    assert len(list(ExplanationStore(tmp_path).iter_all())) == 2


def test_external_output_root_quota_prevents_worker_admission(tmp_path, monkeypatch):
    from tools import storage_guard
    extra = tmp_path / "external"
    extra.mkdir()
    (extra / "database-wal").write_bytes(b"x" * 20000)
    monkeypatch.setattr(storage_guard.subprocess, "Popen", lambda *a, **k: pytest.fail("launch"))
    result = storage_guard.run(
        ["worker"], root=tmp_path, pause_paths=[tmp_path / "STOP"],
        min_free=0, max_used=10**9, growth_reserve=0,
        additional_roots=[{"usage_root": str(extra), "min_free_bytes": 0,
                           "max_used_bytes": 10000, "growth_reserve_bytes": 0}],
    )
    assert result == 75
    assert "usage ceiling" in (tmp_path / "STOP").read_text()
