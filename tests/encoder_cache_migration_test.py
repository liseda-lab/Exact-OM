"""Only bound, numerically identical encoders migrate historical rows."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from exact.experiments import runtime
from exact.impl.models.scorer_common import PoolingMethod, ScorerCommonMixin


class Config:
    _commit_hash = "a" * 40

    def to_dict(self):
        return {"hidden_size": 2, "_commit_hash": self._commit_hash}


class Tokenizer:
    def get_vocab(self):
        return {"a": 0, "b": 1}


def scorer():
    result = ScorerCommonMixin()
    result.pooling_method = PoolingMethod.MEAN
    result.fp16 = False
    result._cache_tensor_dtype = torch.float32
    result.device_type = result.device = "cpu"
    return result


def setup_cache(tmp_path, monkeypatch):
    current = Path(runtime.__file__).parents[1] / "impl/models/scorer_common.py"
    previous = tmp_path / "previous.py"
    previous.write_text(current.read_text() + "\n# Prior nonnumerical source revision.\n")
    old_sha = runtime.sha256_file(previous)
    real_sha = runtime.sha256_file
    model = SimpleNamespace(config=Config(), training=False)
    tokenizer = Tokenizer()
    monkeypatch.setenv("EXACT_EMBEDDING_CACHE_DIR", str(tmp_path / "vectors"))
    monkeypatch.delenv("EXACT_ENCODER_COMPATIBLE_SOURCE", raising=False)
    monkeypatch.delenv("EXACT_ENCODER_COMPATIBLE_SOURCE_SHA256", raising=False)
    monkeypatch.setattr(runtime, "sha256_file", lambda p: old_sha if p == current else real_sha(p))
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    runtime.cached_encoder_rows(scorer(), tokenizer, model, ["a", "b"], 16, lambda _: expected)
    monkeypatch.setattr(runtime, "sha256_file", real_sha)
    monkeypatch.setenv("EXACT_ENCODER_COMPATIBLE_SOURCE", str(previous))
    monkeypatch.setenv("EXACT_ENCODER_COMPATIBLE_SOURCE_SHA256", old_sha)
    return model, tokenizer, expected, previous


def test_bound_unchanged_encoder_imports_exact_rows_without_forward(tmp_path, monkeypatch):
    model, tokenizer, expected, _ = setup_cache(tmp_path, monkeypatch)

    def forbidden(_):
        pytest.fail("Compatible vectors must not be re-encoded")

    actual = runtime.cached_encoder_rows(scorer(), tokenizer, model, ["b", "a", "b"], 16, forbidden)
    assert torch.equal(actual, expected[[1, 0, 1]])


@pytest.mark.parametrize("change", ["revision", "precision", "role", "length"])
def test_other_identity_fields_never_migrate(tmp_path, monkeypatch, change):
    model, tokenizer, _, _ = setup_cache(tmp_path, monkeypatch)
    current = scorer()
    length = 16
    if change == "revision":
        model.config._commit_hash = "b" * 40
    elif change == "precision":
        current._cache_tensor_dtype = torch.float16
    elif change == "role":
        monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "final")
    else:
        length = 32
    calls = []

    def compute(texts):
        calls.append(texts)
        return torch.tensor([[5.0, 6.0]])

    actual = runtime.cached_encoder_rows(current, tokenizer, model, ["a"], length, compute)
    assert calls == [["a"]]
    assert torch.equal(actual.float(), torch.tensor([[5.0, 6.0]]))


@pytest.mark.parametrize("change", ["binding", "pooling", "corruption"])
def test_incompatible_or_corrupted_rows_fail_closed(tmp_path, monkeypatch, change):
    model, tokenizer, _, previous = setup_cache(tmp_path, monkeypatch)
    if change == "binding":
        previous.write_text(previous.read_text() + "# changed\n")
    elif change == "pooling":
        previous.write_text(previous.read_text().replace('MEAN = "mean"', 'MEAN = "other"', 1))
        monkeypatch.setenv("EXACT_ENCODER_COMPATIBLE_SOURCE_SHA256", runtime.sha256_file(previous))
    else:
        import sqlite3

        with sqlite3.connect(tmp_path / "vectors/vectors.sqlite3") as db:
            db.execute("UPDATE vectors SET raw=x'00'")
    with pytest.raises(ValueError, match="binding|pooling|Corrupted"):
        runtime.cached_encoder_rows(
            scorer(), tokenizer, model, ["a"], 16, lambda _: pytest.fail("Unexpected forward")
        )
