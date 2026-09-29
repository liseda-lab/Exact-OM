"""Prepared reuse is independent of scoring and never bypasses integrity gates."""

import json

import pytest

from exact.impl.datasets import prepared_cache
from exact.utils.provenance import sha256_file


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("EXACT_DATASET_CACHE_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "development")
    directory = tmp_path / "original"
    directory.mkdir()
    fingerprint = "a" * 40
    (directory / "dataset.csv").write_text("Src,Tgt,Label\ns,t,=\n")
    (directory / "candidate_recall.tsv").write_text("Src\tTgt\tpool_role\ns\tt\tcandidate\n")
    (directory / "candidate_pool_manifest.json").write_text(json.dumps({"fingerprint": "b" * 64}))
    (directory / "dataset.meta.json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "candidate_pool_fingerprint": "b" * 64,
                "candidate_recall_sha256": sha256_file(directory / "candidate_recall.tsv"),
            }
        )
    )
    for name in ("verbalization_templates.json", "verbalization_templates.meta.json"):
        (directory / name).write_text("{}")
    (directory / "scores.json").write_text("must never be reused as prepared inputs")
    return directory, fingerprint


def test_shared_cache_copies_only_verified_prepared_inputs(prepared, tmp_path):
    directory, fingerprint = prepared
    published = prepared_cache.publish(directory, fingerprint)
    destination = tmp_path / "new-arm"
    assert prepared_cache.restore(destination, fingerprint)
    assert {
        p.name for p in destination.iterdir()
    } == prepared_cache._REQUIRED | prepared_cache._TEMPLATES
    for path in destination.iterdir():
        assert path.read_bytes() == (directory / path.name).read_bytes()
    (destination / "dataset.csv").write_text("local mutation")
    assert (published / "dataset.csv").read_bytes() == (directory / "dataset.csv").read_bytes()


@pytest.mark.parametrize("name", sorted(prepared_cache._REQUIRED | prepared_cache._TEMPLATES))
def test_shared_cache_verifies_every_file_before_installing(prepared, tmp_path, name):
    directory, fingerprint = prepared
    published = prepared_cache.publish(directory, fingerprint)
    (published / name).write_text("changed")
    destination = tmp_path / "new-arm"
    with pytest.raises(ValueError, match="has changed"):
        prepared_cache.restore(destination, fingerprint)
    assert not destination.exists()


def test_shared_cache_separates_fingerprints_and_roles(prepared, tmp_path, monkeypatch):
    directory, fingerprint = prepared
    prepared_cache.publish(directory, fingerprint)
    assert not prepared_cache.restore(tmp_path / "different-model", "c" * 40)
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "reporting")
    assert not prepared_cache.restore(tmp_path / "reporting", fingerprint)
    monkeypatch.delenv("EXACT_EXPERIMENT_ROLE")
    with pytest.raises(ValueError, match="explicit experiment role"):
        prepared_cache.restore(tmp_path / "unspecified", fingerprint)


def test_shared_cache_rejects_misbound_metadata(prepared):
    directory, fingerprint = prepared
    metadata = json.loads((directory / "dataset.meta.json").read_text())
    metadata["candidate_recall_sha256"] = "0" * 64
    (directory / "dataset.meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata does not match"):
        prepared_cache.publish(directory, fingerprint)


def test_legacy_dataset_without_raw_pool_is_not_published(prepared):
    directory, fingerprint = prepared
    (directory / "candidate_recall.tsv").unlink()
    assert prepared_cache.publish(directory, fingerprint) is None


def test_local_staging_reuses_verified_bytes_without_rehashing_nas(prepared, tmp_path, monkeypatch):
    directory, fingerprint = prepared
    published = prepared_cache.publish(directory, fingerprint)
    monkeypatch.setenv("EXACT_DATASET_CACHE_LOCAL_DIR", str(tmp_path / "nvme"))
    assert prepared_cache.restore(tmp_path / "first", fingerprint)
    staged = next(path for path in (tmp_path / "nvme/prepared-datasets").iterdir() if path.is_dir())
    assert (staged / "dataset.csv").read_bytes() == (directory / "dataset.csv").read_bytes()
    prepared_cache._VERIFIED.clear()  # A new worker shares only the staged receipt.
    hashed = []
    original = prepared_cache.sha256_file

    def track(path):
        hashed.append(path)
        return original(path)

    monkeypatch.setattr(prepared_cache, "sha256_file", track)
    assert prepared_cache.restore(tmp_path / "second", fingerprint)
    assert hashed and all(path.parent != published for path in hashed)
    (published / "dataset.csv").write_text("damaged NAS file")
    with pytest.raises(ValueError, match="has changed"):
        prepared_cache.restore(tmp_path / "corrupt", fingerprint)


def test_local_staging_rejects_changed_hot_copy_and_can_bypass_full_scratch(
    prepared, tmp_path, monkeypatch
):
    directory, fingerprint = prepared
    prepared_cache.publish(directory, fingerprint)
    monkeypatch.setenv("EXACT_DATASET_CACHE_LOCAL_DIR", str(tmp_path / "nvme"))
    monkeypatch.setenv("EXACT_DATASET_CACHE_LOCAL_MAX_BYTES", "0")
    assert prepared_cache.restore(tmp_path / "uncached", fingerprint)
    assert not any(path.is_dir() for path in (tmp_path / "nvme/prepared-datasets").iterdir())
    monkeypatch.setenv("EXACT_DATASET_CACHE_LOCAL_MAX_BYTES", str(1024**2))
    assert prepared_cache.restore(tmp_path / "first", fingerprint)
    staged = next(path for path in (tmp_path / "nvme/prepared-datasets").iterdir() if path.is_dir())
    (staged / "dataset.csv").write_text("corrupt hot cache")
    with pytest.raises(ValueError, match="has changed"):
        prepared_cache.restore(tmp_path / "corrupt", fingerprint)


def test_local_staging_does_not_reuse_a_republished_different_manifest(
    prepared, tmp_path, monkeypatch
):
    directory, fingerprint = prepared
    published = prepared_cache.publish(directory, fingerprint)
    monkeypatch.setenv("EXACT_DATASET_CACHE_LOCAL_DIR", str(tmp_path / "nvme"))
    assert prepared_cache.restore(tmp_path / "first", fingerprint)
    (published / "dataset.csv").write_text("Src,Tgt,Label\ns,different,=\n")
    manifest = json.loads((published / "manifest.json").read_text())
    manifest["files"]["dataset.csv"] = sha256_file(published / "dataset.csv")
    (published / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="publication changed"):
        prepared_cache.restore(tmp_path / "second", fingerprint)
