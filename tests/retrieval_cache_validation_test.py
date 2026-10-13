"""Fitted-input integrity stays mandatory while cache validation remains bounded."""

import os
import sqlite3
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from exact.experiments import numerical_cache
from exact.experiments import runtime as encoder_runtime
from exact.experiments.numerical_cache import cached_numerical, peek_channel
from exact.impl.datasets.base import BaseAlignmentDataset
from exact.impl.models.pair_adaptive_channels import PairAdaptiveChannelsMixin
from exact.impl.retrieval import artifacts
from tests.encoder_cache_migration_test import Config, Tokenizer, scorer
from tests.numerical_cache_test import Probe, runtime
from tests.retrieval_experiments_test import _artifact


class BoundDataset:
    validate_retrieval_artifacts = BaseAlignmentDataset.validate_retrieval_artifacts
    _configured_retrieval_artifacts = BaseAlignmentDataset._configured_retrieval_artifacts

    def __init__(self, artifact):
        self._retrieval_artifacts = {"cross_encoder": artifact}

    @property
    def cache_fingerprint(self):
        return self._configured_retrieval_artifacts()["cross_encoder"].sha256


def fitted(tmp_path):
    root = _artifact(tmp_path / "fitted", kind="cross_encoder")
    return artifacts.resolve_local_retrieval_artifact(root, expected_kind="cross_encoder")


def test_real_artifact_is_hashed_once_across_hosted_batches(tmp_path, monkeypatch):
    runtime(tmp_path, monkeypatch)
    calls = []
    original = artifacts._artifact_sha256
    original_open = Path.open
    weight_reads = {"opens": 0, "read_calls": 0, "bytes": 0}

    class Reader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size):
            value = self.stream.read(size)
            weight_reads["read_calls"] += 1
            weight_reads["bytes"] += len(value)
            return value

    def track_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if path == tmp_path / "fitted/model/weights.bin" and mode == "rb":
            weight_reads["opens"] += 1
            return Reader(stream)
        return stream

    def count(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(artifacts, "_artifact_sha256", count)
    monkeypatch.setattr(Path, "open", track_open)
    artifact = fitted(tmp_path)

    class Hosted(Probe):
        use_llm = True
        llm_experiment_config = {"gate": {"mode": "source_top_fraction"}}

        @torch.inference_mode()
        @cached_numerical()
        def forward(self, values):
            self.forward_calls += 1
            for _ in range(512):
                result = self.channel(values)
            return result

    model = Hosted()
    model._attached_dataset = BoundDataset(artifact)
    first = model.forward(torch.tensor([0.2]))
    second = model.forward(torch.tensor([0.2]))
    assert torch.equal(first["raw"], second["raw"])
    assert model.forward_calls == 2 and model.channel_calls == 1
    assert len(calls) == artifact.validation_stats()["full_hashes"] == 1
    assert artifact.validation_stats()["inventory_checks"] <= 10
    assert weight_reads == {"opens": 1, "read_calls": 2, "bytes": len(b"synthetic model bytes")}
    assert not hasattr(model, "_numerical_validation_depth")


@pytest.mark.parametrize("mutation", ["same_size", "replace", "delete", "add", "symlink"])
def test_verified_inventory_rejects_mutations(tmp_path, mutation):
    artifact = fitted(tmp_path)
    weights = artifact.model_path / "weights.bin"
    previous = weights.stat()
    if mutation == "same_size":
        weights.write_bytes(b"x" * previous.st_size)
        os.utime(weights, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    elif mutation == "replace":
        replacement = artifact.model_path / "replacement"
        replacement.write_bytes(b"x" * previous.st_size)
        os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        replacement.replace(weights)
    elif mutation == "delete":
        weights.unlink()
    elif mutation == "add":
        (artifact.model_path / "extra.bin").write_bytes(b"added input")
    else:
        target = tmp_path / "external.bin"
        target.write_bytes(weights.read_bytes())
        weights.unlink()
        weights.symlink_to(target)
    for _ in range(2):
        with pytest.raises(ValueError, match="changed after validation|symlinks"):
            artifact.assert_unchanged()


def test_force_new_process_and_resolution_revalidate_bytes(tmp_path, monkeypatch):
    artifact = fitted(tmp_path)
    artifact.assert_unchanged(force=True)
    assert artifact.validation_stats()["full_hashes"] == 2
    object.__setattr__(artifact, "_verified", (-1, artifact._verified[1]))
    artifact.assert_unchanged()
    assert artifact.validation_stats()["full_hashes"] == 3
    calls = []
    original = artifacts._artifact_sha256
    monkeypatch.setattr(
        artifacts, "_artifact_sha256", lambda *args: (calls.append(True), original(*args))[1]
    )
    artifacts.resolve_local_retrieval_artifact(
        artifact.manifest_path, expected_kind="cross_encoder"
    )
    assert len(calls) == 1


def test_change_during_initial_hash_is_not_admitted(tmp_path, monkeypatch):
    root = _artifact(tmp_path / "fitted", kind="cross_encoder")
    original = artifacts._artifact_sha256

    def mutate(*args):
        result = original(*args)
        (root / "model/weights.bin").write_bytes(b"changed during validation")
        return result

    monkeypatch.setattr(artifacts, "_artifact_sha256", mutate)
    with pytest.raises(ValueError, match="changed during validation"):
        artifacts.resolve_local_retrieval_artifact(root, expected_kind="cross_encoder")


@pytest.mark.parametrize("mode", ["hit", "disabled", "no_runtime", "channel", "probe", "gradient"])
@pytest.mark.parametrize("damage", ["changed", "missing", "unreadable"])
def test_mandatory_artifact_failure_never_becomes_cache_miss(tmp_path, monkeypatch, mode, damage):
    runtime(tmp_path, monkeypatch)
    artifact = fitted(tmp_path)
    model = Probe()
    model._attached_dataset = BoundDataset(artifact)
    values = torch.tensor([0.2])
    model.forward(values)
    before = model.forward_calls, model.channel_calls
    if damage == "changed":
        (artifact.model_path / "weights.bin").write_bytes(b"changed input")
    elif damage == "missing":
        artifact.manifest_path.unlink()
    else:
        monkeypatch.setattr(
            artifacts,
            "_artifact_inventory",
            lambda *_: (_ for _ in ()).throw(PermissionError("unreadable fitted input")),
        )
    if mode == "disabled":
        monkeypatch.setenv("EXACT_NUMERICAL_CACHE", "0")
    elif mode == "no_runtime":
        monkeypatch.delenv("EXACT_EXPERIMENT_RUNTIME")
    with pytest.raises((ValueError, OSError)):
        if mode == "gradient":
            with torch.enable_grad():
                model.channel(values)
        else:
            with torch.inference_mode():
                if mode == "probe":
                    peek_channel(model.channel, values)
                elif mode == "channel":
                    model.channel(values)
                else:
                    model.forward(values)
    assert (model.forward_calls, model.channel_calls) == before


def test_optional_database_failure_recomputes_but_artifact_drift_does_not(tmp_path, monkeypatch):
    runtime(tmp_path, monkeypatch)
    artifact = fitted(tmp_path)
    model = Probe()
    model._attached_dataset = BoundDataset(artifact)

    def unavailable(*args):
        raise sqlite3.OperationalError("optional cache unavailable")

    monkeypatch.setattr(numerical_cache, "NumericalCache", unavailable)
    assert torch.equal(model.forward(torch.tensor([0.2]))["score"], torch.tensor([0.7]) ** 2)
    assert model.forward_calls == model.channel_calls == 1
    (artifact.model_path / "weights.bin").write_bytes(b"changed input")
    with pytest.raises(ValueError, match="changed after validation"):
        model.forward(torch.tensor([0.2]))
    assert model.forward_calls == model.channel_calls == 1


@pytest.mark.parametrize(
    "error", [ValueError("changed required identity"), OSError("missing required identity")]
)
def test_scope_integrity_errors_propagate_before_scoring(tmp_path, monkeypatch, error):
    runtime(tmp_path, monkeypatch)
    model = Probe()
    monkeypatch.setattr(
        numerical_cache, "_scope", lambda *args, **kwargs: (_ for _ in ()).throw(error)
    )
    with pytest.raises(type(error), match="required identity"):
        model.forward(torch.tensor([0.2]))
    assert model.forward_calls == model.channel_calls == 0


@pytest.mark.parametrize("cache", ["contract", "reference", "rows"])
@pytest.mark.parametrize("damage", ["changed", "missing"])
def test_training_cache_discovery_validates_mandatory_inputs(tmp_path, monkeypatch, cache, damage):
    runtime(tmp_path, monkeypatch)
    artifact = fitted(tmp_path)
    model = Probe()
    model._attached_dataset = BoundDataset(artifact)
    if damage == "changed":
        (artifact.model_path / "weights.bin").write_bytes(b"changed input")
    else:
        artifact.manifest_path.unlink()
    with pytest.raises((ValueError, OSError)):
        if cache == "contract":
            numerical_cache.training_content_contract(model)
        elif cache == "reference":
            numerical_cache.training_score_reference(model, "train", {}, 16)
        else:
            numerical_cache.shared_training_scores(model, "train", {}, 16)


def test_artifact_change_during_forward_blocks_final_cache_publication(tmp_path, monkeypatch):
    runtime(tmp_path, monkeypatch)
    artifact = fitted(tmp_path)

    class Mutating(Probe):
        @torch.inference_mode()
        @cached_numerical()
        def forward(self, values):
            result = self.channel(values)
            (artifact.model_path / "weights.bin").write_bytes(b"changed while scoring")
            return result

    model = Mutating()
    model._attached_dataset = BoundDataset(artifact)
    with pytest.raises(ValueError, match="changed after validation"):
        model.forward(torch.tensor([0.2]))
    assert (
        model._numerical_cache.connection.execute(
            "SELECT count(*) FROM entries WHERE operation='forward'"
        ).fetchone()[0]
        == 0
    )


def test_channel_probe_is_read_only_and_respects_scope(tmp_path, monkeypatch):
    runtime(tmp_path, monkeypatch)
    model = Probe()
    values = torch.tensor([0.2])
    with torch.inference_mode():
        assert peek_channel(model.channel, values) is None
        assert not hasattr(model, "_numerical_cache")
        model.forward(values)
        assert torch.equal(peek_channel(model.channel, values)["raw"], torch.tensor([0.7]))
        assert peek_channel(model.channel, values + 1) is None
        assert model.channel_calls == 1
        monkeypatch.setenv("EXACT_NUMERICAL_CACHE", "0")
        assert peek_channel(model.channel, values) is None


@pytest.mark.parametrize("operation", ["channel", "probe"])
@pytest.mark.parametrize(
    "feature", ["EXACT_PAIR_CONTEXT_BATCHING", "EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY"]
)
def test_direct_channel_cache_hit_requires_qualification_admission(
    tmp_path, monkeypatch, operation, feature
):
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_BATCHING", "0")
    monkeypatch.setenv("EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY", "0")
    monkeypatch.setenv(feature, "1")
    monkeypatch.setenv("EXACT_PAIR_CONTEXT_ENCODER", "batch")
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "throughput_fixture")
    monkeypatch.setenv("EXACT_PRIMITIVE_NAMESPACE", "qualification-direct-channel")
    payload, reads = {"score": 0.75}, []
    cache = SimpleNamespace(
        pid=os.getpid(),
        path=tmp_path / "numerical/entries.sqlite",
        get=lambda *_args: (reads.append(True), payload)[1],
        batch=nullcontext,
    )
    model = SimpleNamespace(_numerical_cache=cache, device="cpu")
    method = PairAdaptiveChannelsMixin._score_label_channel.__get__(model)
    monkeypatch.setattr(numerical_cache, "_scope", lambda *_a, **_k: (tmp_path, "fixed"))
    invoke = method if operation == "channel" else lambda *args: peek_channel(method, *args)
    with torch.inference_mode():
        assert invoke(["source"], ["target"]) is payload
        monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "scientific")
        with pytest.raises(RuntimeError, match="only admitted"):
            invoke(["source"], ["target"])
    # Deliberately offer the same cached identity after the namespace becomes
    # invalid: the cached result must not bypass admission or touch storage.
    assert len(reads) == 1


@pytest.mark.parametrize("setting", ["SCHEDULE", "ENCODER", "MATRIX_KERNEL", "ARITHMETIC"])
def test_qualification_mode_change_invalidates_batch_scopes(tmp_path, monkeypatch, setting):
    runtime(tmp_path, monkeypatch)
    model = Probe()
    model._numerical_channel_scopes = {}
    with torch.inference_mode():
        model.channel(torch.tensor([0.2]))
        monkeypatch.setenv("EXACT_PAIR_CONTEXT_" + setting, "changed")
        model.channel(torch.tensor([0.2]))
    assert model.channel_calls == 2


@pytest.mark.parametrize(
    "setting,value",
    [
        ("SCHEDULE", "staged"),
        ("ENCODER", "batch"),
        ("MATRIX_KERNEL", "bmm"),
        ("ARITHMETIC", "cpu"),
    ],
)
def test_qualification_mode_change_invalidates_encoder_rows(monkeypatch, setting, value):
    current, tokenizer = scorer(), Tokenizer()
    current._numerical_channel_scopes = {}
    model = torch.nn.Linear(2, 2, bias=False).eval()
    model.config = Config()
    name = "EXACT_PAIR_CONTEXT_" + setting
    monkeypatch.delenv(name, raising=False)
    before = encoder_runtime.encoder_identity(current, tokenizer, model, 16)
    monkeypatch.setenv(name, value)
    assert encoder_runtime.encoder_identity(current, tokenizer, model, 16) != before
