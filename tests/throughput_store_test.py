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


@pytest.mark.parametrize("operation", ["overlay", "truncate", "compact"])
def test_replacement_failure_after_publication_never_deletes_live_payload(tmp_path, monkeypatch, operation):
    store = ExplanationStore(tmp_path)
    store.append([row(1), row(2)])
    original = store._write_index
    def fail_after_write(value):
        original(value)
        raise OSError("directory fsync failed after publication")
    monkeypatch.setattr(store, "_write_index", fail_after_write)
    with pytest.raises(OSError, match="after publication"):
        if operation == "overlay":
            store.append_overlay([{**row(1), "recovered": True}])
        elif operation == "truncate":
            store.truncate(1)
        else:
            store.compact()
    reopened = list(ExplanationStore(tmp_path).iter_all())
    assert [r["tgt_iri"] for r in reopened] == (["1"] if operation == "truncate" else ["1", "2"])
    if operation == "overlay":
        assert reopened[0]["recovered"] is True


def test_replacement_streams_bounded_chunks(tmp_path, monkeypatch):
    store = ExplanationStore(tmp_path)
    consumed = 0
    written = 0
    original = ExplanationStore.append
    def append(self, records):
        nonlocal written
        assert len(records) <= 256
        result = original(self, records)
        written += len(records)
        return result
    monkeypatch.setattr(ExplanationStore, "append", append)
    def rows():
        nonlocal consumed
        for i in range(777):
            consumed += 1
            assert consumed - written <= 256
            yield row(i)
    assert store._replace_records(rows())["records"] == 777
    assert len(list(ExplanationStore(tmp_path).iter_all())) == 777


def test_interleaved_sources_keep_original_order_after_replacement(tmp_path):
    store = ExplanationStore(tmp_path, hash_buckets=1)
    records = [{**row(i), "src_iri": source} for i, source in enumerate(("a", "b", "a", "c", "b"))]
    store.append(records)
    assert [r["tgt_iri"] for r in store.iter_all()] == [str(i) for i in range(5)]
    store.compact()
    store.truncate(4)
    assert [r["tgt_iri"] for r in store.iter_all()] == [str(i) for i in range(4)]


def test_many_shards_reader_opens_only_one_file_at_a_time(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from pathlib import Path
    store = ExplanationStore(tmp_path, hash_buckets=1, shard_mb=.00001)
    store.append([{**row(i), "src_iri": f"source{i}"} for i in range(130)])
    assert len(store._index["shards"]) == 130
    original = Path.open
    live = peak = 0
    @contextmanager
    def tracked(path, *args, **kwargs):
        nonlocal live, peak
        with original(path, *args, **kwargs) as stream:
            counted = path.parent.name == "shards"
            live += counted
            peak = max(peak, live)
            try:
                yield stream
            finally:
                live -= counted
    monkeypatch.setattr(Path, "open", tracked)
    assert [r["tgt_iri"] for r in store.iter_all()] == [str(i) for i in range(130)]
    assert peak == 1


def test_read_only_inspection_does_not_recover_an_active_append(tmp_path, monkeypatch):
    store = ExplanationStore(tmp_path)
    store.append([row(1)])
    original = store._append_bytes
    def during_append(path, encoded):
        original(path, encoded)
        size = path.stat().st_size
        reader = ExplanationStore(tmp_path, read_only=True)
        assert [r["tgt_iri"] for r in reader.iter_all()] == ["1"]
        assert path.stat().st_size == size
        with pytest.raises(ValueError, match="Read-only"):
            reader.append([row(99)])
    monkeypatch.setattr(store, "_append_bytes", during_append)
    store.append([row(2)])
    assert [r["tgt_iri"] for r in store.iter_all()] == ["1", "2"]


def test_failed_compaction_build_cleans_owned_temporary_payload(tmp_path, monkeypatch):
    store = ExplanationStore(tmp_path)
    store.append([row(1)])
    original = ExplanationStore.append
    def fail(self, records, **kwargs):
        result = original(self, records, **kwargs)
        if self.directory.name.startswith(".compact-"):
            raise OSError("ENOSPC in temporary rebuild")
        return result
    monkeypatch.setattr(ExplanationStore, "append", fail)
    with pytest.raises(OSError, match="ENOSPC"):
        store.compact()
    assert not list(tmp_path.glob(".compact-*"))
    assert [r["tgt_iri"] for r in store.iter_all()] == ["1"]


def test_only_recovery_reclaims_dead_owned_compaction(tmp_path, monkeypatch):
    import json
    import socket
    import exact.runs.store as storage
    store = ExplanationStore(tmp_path)
    store.append([row(1)])
    orphan = tmp_path / ".compact-0123456789ab"
    orphan.mkdir()
    (orphan / "owner.json").write_text(json.dumps({"host": socket.gethostname(), "pid": 999999}))
    (orphan / "payload").write_bytes(b"interrupted temporary shard")
    def dead(*args):
        raise ProcessLookupError()
    monkeypatch.setattr(storage.os, "kill", dead)
    assert list(ExplanationStore(tmp_path, read_only=True).iter_all())
    assert orphan.exists()
    assert list(ExplanationStore(tmp_path).iter_all())
    assert not orphan.exists()


@pytest.mark.parametrize("owner", [None, [], {"host": "local", "pid": None}])
def test_uncertain_compaction_owner_is_retained(tmp_path, monkeypatch, owner):
    import json
    import exact.runs.store as storage
    store = ExplanationStore(tmp_path)
    store.append([row(1)])
    orphan = tmp_path / ".compact-0123456789ab"
    orphan.mkdir()
    (orphan / "owner.json").write_text(json.dumps(owner))
    monkeypatch.setattr(storage.socket, "gethostname", lambda: "local")
    assert list(ExplanationStore(tmp_path).iter_all())
    assert orphan.exists()


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


def test_compact_resident_evidence_preserves_selector_and_full_durable_record(tmp_path, monkeypatch):
    import copy
    from types import SimpleNamespace
    import pandas as pd
    from exact.impl.trainer.audit_io import AuditIOMixin
    from exact.impl.models.selector.selector import CandidateSetSelector

    selector = CandidateSetSelector(enabled=True, use_no_match=False, llm={"enabled": False})
    owner = AuditIOMixin()
    owner.model = SimpleNamespace(generate_llm_rationales=False)
    owner.models = [owner.model, selector]
    owner._explanation_store = ExplanationStore(tmp_path)
    records = [{"src_iri": "s", "tgt_iri": t, "confidences": {"S_final": 0.8},
                "attributes": {"source": [], "target": [{"property": "definition", "value": t,
                                                         "item_id": t, "weight": 1.0}]},
                "cross_side_provenance": {}}
               for t in ("one", "two")]
    owner._explanation_store.append(records)
    monkeypatch.setenv("EXACT_COMPACT_RESIDENT_EVIDENCE", "1")
    compact = [owner._resident_evidence_record(r) for r in records]
    assert all("attributes" not in r for r in compact)
    frame = pd.DataFrame({"Src": ["s", "s"], "Tgt": ["one", "two"], "S_pair_final": [0.8, 0.79]})
    full_scores = selector._distinctive_scores(frame, selector._record_lookup(copy.deepcopy(records)))
    compact_scores = selector._distinctive_scores(frame, selector._record_lookup(compact))
    assert full_scores == compact_scores
    assert all("attributes" in r for r in owner._explanation_store.iter_all())
