import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import torch

from exact.core.entities.configs.config import CandidatesConfig, LLMProfileConfig
from exact.core.entities.configs.migration import migrate_v1_mapping
from exact.core.entities.kinds import EntityKind
from exact.impl.datasets import base as base_module
from exact.impl.datasets.base import BaseAlignmentDataset
from exact.impl.models import scorer_common as scorer_common_module
from exact.impl.models import semantic_llm as semantic_llm_module
from exact.impl.models import semantic_scorer as semantic_scorer_module
from exact.impl.models.semantic_scorer import SemanticScorer
from exact.impl.trainer.runner import SemanticAlignmentRunner
from exact.llm.routing import LLMRouter


class _Source:
    def __init__(self, iri: str) -> None:
        self.iri = iri

    def entities(self, kind: EntityKind) -> list[str]:
        return [self.iri] if kind == EntityKind.CLASS else []


class _RevisionDataset(BaseAlignmentDataset):
    def __getitem__(self, idx: int):
        raise IndexError(idx)

    def __len__(self) -> int:
        return 0 if self.dataframe is None else len(self.dataframe)

    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def plot_feature_distributions(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_sanity_examples(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _candidate_rows_for_kind(self, **kwargs: Any) -> list[dict[str, object]]:
        return [
            {
                "Src": "source",
                "Tgt": "target",
                "Label": 0,
                "cand_sim": 1.0,
                "cand_sim_semantic": 1.0,
                "cand_sim_lexical": 0.0,
                "cand_channels": "semantic",
                "SrcKind": EntityKind.CLASS.value,
                "TgtKind": EntityKind.CLASS.value,
            }
        ]


def test_candidate_encoder_revision_reaches_loader_and_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class _SentenceTransformer:
        def __init__(self, name: str, **kwargs: Any) -> None:
            calls.append((name, kwargs))

        def _first_module(self) -> SimpleNamespace:
            config = SimpleNamespace(_commit_hash="b" * 40)
            return SimpleNamespace(auto_model=SimpleNamespace(config=config))

    monkeypatch.setattr(base_module, "SentenceTransformer", _SentenceTransformer)
    dataset = _RevisionDataset(output_path=tmp_path)
    dataset._source = _Source("source")
    dataset._target = _Source("target")

    dataset.load_candidates(
        lexical_encoder_name="org/encoder",
        encoder_revision="a" * 40,
        top_k=1,
        device=torch.device("cpu"),
        use_amp=False,
    )

    assert calls == [
        (
            "org/encoder",
            {"device": "cpu", "revision": "a" * 40},
        )
    ]
    retrieval = dataset.candidate_pool_manifest["retrieval_config"]
    encoder = dataset.candidate_pool_manifest["models"]["encoder"]
    assert retrieval["encoder_revision"] == "a" * 40
    assert retrieval["encoder_resolved_revision"] == "b" * 40
    assert encoder["requested_revision"] == "a" * 40
    assert encoder["resolved_revision"] == "b" * 40

    dataset._write_cache_metadata()
    cache = json.loads(dataset._cache_meta_path.read_text(encoding="utf-8"))
    assert cache["candidate_encoder"] == {
        "identifier": "org/encoder",
        "requested_revision": "a" * 40,
        "resolved_revision": "b" * 40,
    }


class _Tokenizer:
    def __call__(self, text: str, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[1])


class _Model:
    def to(self, device: str) -> "_Model":
        return self


class _Factory:
    calls: list[tuple[str, dict[str, Any]]] = []
    product: Any = None

    @classmethod
    def from_pretrained(cls, name: str, **kwargs: Any) -> Any:
        cls.calls.append((name, kwargs))
        return cls.product()


def test_semantic_and_local_llm_revisions_bind_loaders_and_runtime_identity(
    monkeypatch,
) -> None:
    class _TokenizerFactory(_Factory):
        calls = []
        product = _Tokenizer

    class _EncoderFactory(_Factory):
        calls = []
        product = _Model

    class _CausalFactory(_Factory):
        calls = []
        product = _Model

    monkeypatch.setattr(semantic_scorer_module, "AutoTokenizer", _TokenizerFactory)
    monkeypatch.setattr(semantic_scorer_module, "AutoModel", _EncoderFactory)
    monkeypatch.setattr(scorer_common_module, "AutoTokenizer", _TokenizerFactory)
    monkeypatch.setattr(scorer_common_module, "AutoModelForCausalLM", _CausalFactory)

    hosted_tokenizer_revision = "4" * 40
    scorer = SemanticScorer(
        lexical_model_name="org/lexical",
        lexical_model_revision="1" * 40,
        context_model_name="org/context",
        context_model_revision="2" * 40,
        llm_model_name="org/llm",
        llm_model_revision="3" * 40,
        llm_profiles={
            "hosted": {
                "backend": "openrouter",
                "model": "provider/model",
                "tokenizer": "org/tokenizer",
                "tokenizer_revision": hosted_tokenizer_revision,
            }
        },
        persist_cache_to_disk=False,
        fp16_inference=False,
        device="cpu",
    )
    scorer._ensure_local_llm()

    assert _TokenizerFactory.calls == [
        ("org/lexical", {"revision": "1" * 40}),
        ("org/context", {"revision": "2" * 40}),
        ("org/llm", {"revision": "3" * 40}),
    ]
    assert _EncoderFactory.calls == [
        ("org/lexical", {"revision": "1" * 40}),
        ("org/context", {"revision": "2" * 40}),
    ]
    assert _CausalFactory.calls == [
        (
            "org/llm",
            {"torch_dtype": torch.float32, "revision": "3" * 40},
        )
    ]

    identity = scorer.runtime_fingerprint_payload()
    assert identity["lexical_model_revision"] == "1" * 40
    assert identity["context_model_revision"] == "2" * 40
    assert identity["llm_model_revision"] == "3" * 40
    local_profile = identity["llm_router"]["profiles"]["__semantic_local_llm__"]
    assert local_profile["revision"] == "3" * 40
    assert (
        identity["llm_router"]["profiles"]["hosted"]["tokenizer_revision"]
        == hosted_tokenizer_revision
    )

    resolved = scorer._llm_router.resolve_task("summary")
    assert resolved.revision == "3" * 40
    assert scorer._resolved_backend_metadata(resolved)["revision"] == "3" * 40


def test_omitted_revisions_preserve_loader_kwargs(monkeypatch) -> None:
    class _TokenizerFactory(_Factory):
        calls = []
        product = _Tokenizer

    class _EncoderFactory(_Factory):
        calls = []
        product = _Model

    monkeypatch.setattr(semantic_scorer_module, "AutoTokenizer", _TokenizerFactory)
    monkeypatch.setattr(semantic_scorer_module, "AutoModel", _EncoderFactory)

    SemanticScorer(
        lexical_model_name="org/default",
        use_context=False,
        use_llm=False,
        persist_cache_to_disk=False,
        device="cpu",
    )

    assert _TokenizerFactory.calls == [("org/default", {})]
    assert _EncoderFactory.calls == [("org/default", {})]
    assert CandidatesConfig.model_validate({}).encoder_revision is None
    assert LLMProfileConfig.model_validate({}).revision is None
    assert LLMProfileConfig.model_validate({}).tokenizer_revision is None


def test_router_fingerprints_and_resolves_profile_revision() -> None:
    router = LLMRouter(
        llm_profiles={
            "local": {
                "backend": "local_hf",
                "model": "org/local",
                "revision": "c" * 40,
            }
        },
        llm_routing={"default_profile": "local"},
    )

    assert router.fingerprint_payload()["profiles"]["local"]["revision"] == "c" * 40
    assert router.resolve_task("summary").revision == "c" * 40


def test_hosted_tokenizer_revision_binds_loader_cache_and_router_fingerprint(
    monkeypatch,
) -> None:
    class _TokenizerFactory(_Factory):
        calls = []
        product = _Tokenizer

    monkeypatch.setattr(semantic_llm_module, "AutoTokenizer", _TokenizerFactory)
    scorer = object.__new__(SemanticScorer)
    scorer._hosted_decision_tokenizers = {}
    scorer._log_once = lambda *_args, **_kwargs: None

    first_revision = "e" * 40
    second_revision = "f" * 40
    first_profile = SimpleNamespace(
        name="hosted-a",
        tokenizer="org/tokenizer",
        tokenizer_revision=first_revision,
    )
    second_profile = SimpleNamespace(
        name="hosted-b",
        tokenizer="org/tokenizer",
        tokenizer_revision=second_revision,
    )

    first = scorer._get_hosted_decision_tokenizer(first_profile)
    assert scorer._get_hosted_decision_tokenizer(first_profile) is first
    second = scorer._get_hosted_decision_tokenizer(second_profile)

    assert second is not first
    assert _TokenizerFactory.calls == [
        ("org/tokenizer", {"revision": first_revision}),
        ("org/tokenizer", {"revision": second_revision}),
    ]
    assert set(scorer._hosted_decision_tokenizers) == {
        ("org/tokenizer", first_revision),
        ("org/tokenizer", second_revision),
    }

    router = LLMRouter(
        llm_profiles={
            "hosted": {
                "backend": "openrouter",
                "model": "provider/model",
                "tokenizer": "org/tokenizer",
                "tokenizer_revision": first_revision,
            }
        }
    )
    profile = router.profiles["hosted"]
    assert profile.tokenizer_revision == first_revision
    assert (
        router.fingerprint_payload()["profiles"]["hosted"]["tokenizer_revision"] == first_revision
    )
    changed_router = LLMRouter(
        llm_profiles={
            "hosted": {
                "backend": "openrouter",
                "model": "provider/model",
                "tokenizer": "org/tokenizer",
                "tokenizer_revision": second_revision,
            }
        }
    )
    assert changed_router.fingerprint_payload() != router.fingerprint_payload()


def test_omitted_hosted_tokenizer_revision_preserves_loader_kwargs(monkeypatch) -> None:
    class _TokenizerFactory(_Factory):
        calls = []
        product = _Tokenizer

    monkeypatch.setattr(semantic_llm_module, "AutoTokenizer", _TokenizerFactory)
    scorer = object.__new__(SemanticScorer)
    scorer._hosted_decision_tokenizers = {}
    scorer._log_once = lambda *_args, **_kwargs: None
    profile = SimpleNamespace(name="hosted", tokenizer="org/tokenizer")

    scorer._get_hosted_decision_tokenizer(profile)

    assert _TokenizerFactory.calls == [("org/tokenizer", {})]
    assert list(scorer._hosted_decision_tokenizers) == [("org/tokenizer", None)]


def test_runtime_identity_enriches_and_sanitizes_tokenizer_revision() -> None:
    tokenizer_revision = "9" * 40
    hosted_profile = SimpleNamespace(
        backend="openrouter",
        model="provider/model",
        revision=None,
        tokenizer="org/tokenizer",
        tokenizer_revision=tokenizer_revision,
        api_base="https://user:secret@example.test/v1?api_key=secret",
    )
    local_profile = SimpleNamespace(
        backend="local_hf",
        model="provider/local-model",
        revision="7" * 40,
        tokenizer=None,
        tokenizer_revision=None,
        api_base=None,
    )
    runner = object.__new__(SemanticAlignmentRunner)
    runner._model = SimpleNamespace(
        request_seed=17,
        _llm_router=SimpleNamespace(
            profiles={"hosted": hosted_profile, "local": local_profile},
            routing=SimpleNamespace(
                profile_for_task=lambda _task: "hosted",
                fallback_for_task=lambda _task: "local",
            ),
        ),
        _local_llm_profile_name="local",
    )

    enriched = runner._enrich_llm_backend_metadata(
        "decision",
        {
            "backend": "openrouter",
            "profile": "hosted",
            "model": "provider/model",
        },
    )
    identity = runner._sanitize_llm_backend_identity(enriched)

    assert enriched["tokenizer_revision"] == tokenizer_revision
    assert identity is not None
    assert identity["tokenizer"] == "org/tokenizer"
    assert identity["tokenizer_revision"] == tokenizer_revision

    fallback_enriched = runner._enrich_llm_backend_metadata(
        "decision",
        {
            "backend": "local_hf",
            "profile": "hosted",
            "model": "provider/local-model",
            "fallback_triggered": True,
        },
    )
    fallback_identity = runner._sanitize_llm_backend_identity(fallback_enriched)

    assert fallback_enriched["profile"] == "local"
    assert fallback_identity is not None
    assert fallback_identity["tokenizer"] == "provider/local-model"
    assert fallback_identity["tokenizer_revision"] == "7" * 40


def test_v1_candidate_revision_migrates_to_v2() -> None:
    migrated, _ = migrate_v1_mapping({"candidates_params": {"encoder_revision": "d" * 40}})

    assert migrated["candidates"]["encoder_revision"] == "d" * 40
