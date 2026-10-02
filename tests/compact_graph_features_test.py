import copy
import hashlib
import json
from pathlib import Path

import pytest

from exact.impl.models.graph_head import compact_graph_fingerprints, verify_graph_manifests
from exact.utils.fitted_artifacts import fingerprint, freeze_json
from tools import compact_graph_features as compact


def _payload():
    side = {
        "input_sha256": "in",
        "output_sha256": "out",
        "hierarchy_removal": {
            "removed": [["s", "is_a", "t"]] * 15,
            "fraction": 0.5,
            "seed": 17,
            "input_sha256": "before",
            "output_sha256": "after",
        },
    }
    manifest = {"src": side, "tgt": copy.deepcopy(side)}
    rows = [
        {
            "Src": source,
            "Tgt": "target",
            "unrecognized_future_field": {"x": [4, True]},
            "text": 'escaped newline\n        "graph_fingerprints": { and \\ quote "',
            "score": 0.12345678901234567,
            "confirmed_label": 1,
            "graph_features": {
                "active": True,
                "values": [0.1, 2.0, -0.0],
                "graph_fingerprints": copy.deepcopy(manifest),
            },
        }
        for source in ["s1", "s2"]
    ]
    return {"rows": rows, "source_ids": ["s1", "s2"]}


def _prepare(tmp_path, payload=None):
    payload = payload or _payload()
    path = tmp_path / (fingerprint(payload["source_ids"]) + ".json")
    freeze_json(path, payload)
    options = {
        "manifest_directory": tmp_path / "manifests",
        "receipt_path": tmp_path / "receipts" / path.name,
    }
    return payload, path, options


@pytest.mark.parametrize("apply", [False, True])
def test_streaming_compaction_preserves_all_values_and_audits_original(
    tmp_path, monkeypatch, apply
):
    payload, path, options = _prepare(tmp_path)
    original = path.read_bytes()
    monkeypatch.setattr(compact, "_CHUNK", 29)  # Split keys, UTF-8 escapes and closing markers.
    receipt, manifest = compact.compact_shard(path, apply=apply, **options)
    assert receipt["original_sha256"] == hashlib.sha256(original).hexdigest()
    assert receipt["original_bytes"] == len(original)
    output = path if apply else Path(receipt["candidate"])
    assert receipt["compact_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["status"] == ("compacted" if apply else "prepared")
    expected = copy.deepcopy(payload)
    for row in expected["rows"]:
        row["graph_features"]["graph_fingerprints"] = compact_graph_fingerprints(
            row["graph_features"]["graph_fingerprints"]
        )
    assert json.loads(output.read_text()) == expected
    verify_graph_manifests(manifest.compact, options["manifest_directory"])
    if not apply:
        assert path.read_bytes() == original
    assert receipt["row_count"] == 2
    assert receipt["source_ids"] == ["s1", "s2"]
    assert len(list(options["manifest_directory"].glob("*.json"))) == 1


def test_reuses_decoded_manifest_between_shards(tmp_path, monkeypatch):
    _, first, options = _prepare(tmp_path)
    _, manifest = compact.compact_shard(first, **options)
    payload = _payload()
    payload["rows"][0]["Src"], payload["rows"][1]["Src"] = "s3", "s4"
    payload["source_ids"] = ["s3", "s4"]
    _, second, options = _prepare(tmp_path, payload)
    monkeypatch.setattr(compact, "GraphManifest", lambda *_: pytest.fail("Decoded manifest twice"))
    receipt, returned = compact.compact_shard(second, manifest=manifest, **options)
    assert returned is manifest
    assert receipt["source_ids"] == ["s3", "s4"]


@pytest.mark.parametrize("corruption", ["truncated", "mixed", "membership", "filename", "layout"])
def test_rejects_invalid_input_without_replacing_original(tmp_path, corruption):
    payload = _payload()
    if corruption == "mixed":
        payload["rows"][1]["graph_features"]["graph_fingerprints"]["src"]["output_sha256"] = "bad"
    if corruption == "membership":
        payload["source_ids"] = ["s1", "missing"]
    _, path, options = _prepare(tmp_path, payload)
    if corruption == "truncated":
        path.write_bytes(path.read_bytes()[:-10])
    elif corruption == "filename":
        path = path.rename(tmp_path / ("0" * 64 + ".json"))
    elif corruption == "layout":
        path.write_text(json.dumps(payload))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        compact.compact_shard(path, apply=True, **options)
    assert path.read_bytes() == before
    assert not options["receipt_path"].exists()
    assert not path.with_suffix(".json.compact.partial").exists()


def test_rejects_existing_candidate_and_receipt(tmp_path):
    _, path, options = _prepare(tmp_path)
    candidate = path.with_suffix(".json.compact.partial")
    candidate.write_text("other writer")
    with pytest.raises(FileExistsError):
        compact.compact_shard(path, **options)
    assert candidate.read_text() == "other writer"
    candidate.unlink()
    compact.compact_shard(path, **options)
    with pytest.raises(ValueError, match="receipt already exists"):
        compact.compact_shard(path, **options)


def test_failed_candidate_fsync_never_replaces_input(tmp_path, monkeypatch):
    _, path, options = _prepare(tmp_path)
    before = path.read_bytes()
    # Constructing the shared manifest creates sidecars before simulating quota failure.
    # Obtain the exact canonical block from the original fixture.
    start = before.index(compact._MANIFEST_KEY) + len(compact._MANIFEST_KEY)
    end = before.index(b"\n        }", start) + len(b"\n        }")
    manifest = compact.GraphManifest(before[start:end], options["manifest_directory"])
    monkeypatch.setattr(
        compact.os, "fsync", lambda *_: (_ for _ in ()).throw(OSError(122, "quota"))
    )
    with pytest.raises(OSError):
        compact.compact_shard(path, manifest=manifest, apply=True, **options)
    assert path.read_bytes() == before
    assert not options["receipt_path"].exists()
