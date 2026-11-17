from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from exact.impl.datasets import contextgraph as contextgraph_module
from exact.impl.datasets.contextgraph import ContextDataset


class _TokenizerFactory:
    calls: list[tuple[str, dict[str, Any]]] = []

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> object:
        cls.calls.append((model, dict(kwargs)))
        return object()


class _LoadedModel:
    def __init__(self) -> None:
        self.device: torch.device | None = None

    def to(self, device: torch.device) -> "_LoadedModel":
        self.device = device
        return self


class _ModelFactory:
    calls: list[tuple[str, dict[str, Any]]] = []

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> _LoadedModel:
        cls.calls.append((model, dict(kwargs)))
        return _LoadedModel()


def _patch_factories(monkeypatch) -> None:
    _TokenizerFactory.calls = []
    _ModelFactory.calls = []
    monkeypatch.setattr(contextgraph_module, "AutoTokenizer", _TokenizerFactory)
    monkeypatch.setattr(contextgraph_module, "AutoModelForCausalLM", _ModelFactory)


def test_context_verbaliser_loads_resolved_profile_model_at_revision(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_factories(monkeypatch)
    revision = "a" * 40
    dataset = ContextDataset(
        output_path=tmp_path,
        device=torch.device("cpu"),
        verbaliser_name=None,
        llm_profiles={
            "pinned": {
                "backend": "local_hf",
                "model": "org/resolved-verbaliser",
                "revision": revision,
                "extra_headers": {"Authorization": "Bearer must-not-be-persisted"},
            }
        },
        llm_routing={"verbaliser_profile": "pinned"},
    )

    dataset._ensure_verbaliser()

    expected_call = [("org/resolved-verbaliser", {"revision": revision})]
    assert _TokenizerFactory.calls == expected_call
    assert _ModelFactory.calls == expected_call

    fingerprint = dataset._cache_fingerprint_payload()
    assert fingerprint["verbaliser_requested_model"] == "org/resolved-verbaliser"
    assert fingerprint["verbaliser_requested_revision"] == revision
    assert fingerprint["verbaliser_resolved_model"] == "org/resolved-verbaliser"
    assert fingerprint["verbaliser_resolved_revision"] == revision

    dataset._write_template_cache_metadata()
    metadata = json.loads(dataset._verb_temp_meta_path.read_text(encoding="utf-8"))
    assert metadata["requested_model"] == "org/resolved-verbaliser"
    assert metadata["requested_revision"] == revision
    assert metadata["resolved_model"] == "org/resolved-verbaliser"
    assert metadata["resolved_revision"] == revision
    assert "must-not-be-persisted" not in json.dumps(metadata)


def test_context_verbaliser_omits_revision_keyword_when_profile_has_none(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_factories(monkeypatch)
    dataset = ContextDataset(
        output_path=tmp_path,
        device=torch.device("cpu"),
        verbaliser_name="org/legacy-name-must-not-be-loaded",
        llm_profiles={
            "local": {
                "backend": "local_hf",
                "model": "org/resolved-without-revision",
            }
        },
        llm_routing={"verbaliser_profile": "local"},
    )

    dataset._ensure_verbaliser()

    expected_call = [("org/resolved-without-revision", {})]
    assert _TokenizerFactory.calls == expected_call
    assert _ModelFactory.calls == expected_call
    assert dataset._cache_fingerprint_payload()["verbaliser_resolved_revision"] is None


def test_context_verbaliser_revision_changes_template_fingerprint(tmp_path: Path) -> None:
    common = {
        "device": torch.device("cpu"),
        "verbaliser_name": None,
        "llm_routing": {"verbaliser_profile": "local"},
    }
    first = ContextDataset(
        output_path=tmp_path / "first",
        llm_profiles={
            "local": {
                "backend": "local_hf",
                "model": "org/verbaliser",
                "revision": "b" * 40,
            }
        },
        **common,
    )
    second = ContextDataset(
        output_path=tmp_path / "second",
        llm_profiles={
            "local": {
                "backend": "local_hf",
                "model": "org/verbaliser",
                "revision": "c" * 40,
            }
        },
        **common,
    )

    assert first.verbalizer_template_fingerprint != second.verbalizer_template_fingerprint
