from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from exact.core.contracts.model import IModel
from exact.impl.trainer import SemanticAlignmentRunner


class _FixtureDataset:
    dataset_signature = "runner-experiment-output-test"
    cache_fingerprint = "runner-experiment-output-candidates-v1"

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "src_iri": f"https://example.org/source/{index}",
            "tgt_iri": f"https://example.org/target/{index}",
            "src_kind": "class",
            "tgt_kind": "class",
            "src_labels": [f"source {index}"],
            "tgt_labels": [f"target {index}"],
            "src_ctx_triples": [],
            "tgt_ctx_triples": [],
            "label": index == 0,
        }


class _FixtureModel(IModel):
    def __init__(self, emit_experiment_outputs: bool, **kwargs: Any) -> None:
        super().__init__()
        self.emit_experiment_outputs = bool(emit_experiment_outputs)

    def runtime_fingerprint_payload(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "model": "runner-experiment-output-fixture-v1",
            "emit_experiment_outputs": self.emit_experiment_outputs,
        }

    def runtime_fingerprint(self) -> str:
        payload = json.dumps(self.runtime_fingerprint_payload(), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def forward(self, *, src_iris: list[str], **kwargs: Any) -> dict[str, Any]:
        count = len(src_iris)
        output: dict[str, Any] = {"S_final": torch.tensor([0.9, 0.4][:count], dtype=torch.float32)}
        if not self.emit_experiment_outputs:
            return output
        output.update(
            {
                "s_strsim": torch.tensor([0.88, 0.33][:count]),
                "q_lex": torch.tensor([0.77, 0.22][:count]),
                "q_strsim": torch.tensor([0.66, 0.11][:count]),
                "I_lex": torch.tensor([0.55, 0.44][:count]),
                "I_strsim": torch.tensor([0.45, 0.34][:count]),
                "llm_gate_diagnostics": [
                    {
                        "would_route": True,
                        "threshold": 0.7,
                        "statistic_name": "U",
                        "statistic": 0.8,
                        "mode": "quantile",
                        "invoked": False,
                        "tags": {"b", "a"},
                        "nested": {"non_finite": float("nan"), "tensor": torch.tensor(2)},
                    },
                    {
                        "mode": "analytic",
                        "statistic_name": "U",
                        "statistic": 0.9,
                        "threshold": 0.5,
                        "would_route": True,
                        "invoked": True,
                    },
                ][:count],
            }
        )
        return output


class _BackendUsageModel(_FixtureModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(emit_experiment_outputs=False, **kwargs)
        self.calls = 0
        self.request_seed = 17
        self.llm_temperature = 0.2
        self.llm_top_p = 0.8
        self.llm_do_sample = False
        self.max_total_tokens_llm_summary = 256
        self.max_total_tokens_llm_decision = 384
        self.max_total_tokens_llm_rationale = 512
        self.max_new_tokens_llm = 32
        self.max_new_tokens_llm_rationale = 48
        self.hosted_decision_logit_bias = 40.0
        self._local_llm_profile_name = "local-profile"
        self._llm_router = SimpleNamespace(
            routing=SimpleNamespace(
                profile_for_task=lambda task: "hosted-profile",
                fallback_for_task=lambda task: "local-profile",
            ),
            profiles={
                "hosted-profile": SimpleNamespace(
                    backend="openrouter",
                    model="vendor/requested-model@v1",
                    tokenizer="vendor/tokenizer@r3",
                    api_base="https://api-user:api-password@api.example.test/v1?secret=yes",
                ),
                "local-profile": SimpleNamespace(
                    backend="local_hf",
                    model="vendor/local-model@r4",
                    tokenizer="vendor/local-tokenizer@r4",
                    api_base=None,
                ),
            },
        )

    def forward(self, *, src_iris: list[str], **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        prompt_sha1 = f"{self.calls:040x}"
        common_hosted = {
            "backend": "openrouter",
            "profile": "hosted-profile",
            "requested_profile": "hosted-profile",
            "requested_model": "vendor/requested-model@v1",
            "effective_model": "vendor/effective-model@r2",
            "tokenizer": "vendor/tokenizer@r3",
            "provider": {"name": "safe-provider", "api_key": "provider-secret"},
            "endpoint": (
                "https://api-user:api-password@api.example.test/v1/chat/completions"
                "?api_key=query-secret#debug"
            ),
            "request_seed": 17,
            "decoding": {
                "temperature": 0.2,
                "top_p": 0.8,
                "max_tokens": 32,
                "messages": ["never persist this prompt"],
            },
            "prompt_hash": "a" * 40,
            "cache_key_hash": "b" * 64,
            "request_debug": {
                "prompt_sha1": prompt_sha1,
                "prompt": "never persist this debug prompt",
                "request_payload": {"Authorization": "Bearer debug-secret"},
            },
            "headers": {"Authorization": "Bearer header-secret"},
            "api_key_env": "SECRET_API_KEY_ENV",
            "api_key_path": "/secret/key/path",
            "fallback_error": "provider payload secret",
        }
        return {
            "S_final": torch.full((len(src_iris),), 0.9, dtype=torch.float32),
            "backend_usage": {
                "summary": common_hosted,
                "decision": {
                    "backend": "local_hf",
                    "profile": "hosted-profile",
                    "model": "vendor/local-model@r4",
                    "request_seed": 17,
                    "fallback_triggered": True,
                    "decision_scoring_mode": "chat_logprobs_binary_head",
                },
                "rationale": {
                    **common_hosted,
                    "decoding": {
                        "temperature": 0.2,
                        "top_p": 0.8,
                        "max_tokens": 48,
                    },
                },
            },
        }


def _run(tmp_path: Path, *, emit: bool) -> SemanticAlignmentRunner:
    runner = SemanticAlignmentRunner(
        dataset=_FixtureDataset(),
        model=_FixtureModel,
        model_params={"emit_experiment_outputs": emit},
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    runner.predict(
        threshold=0.0,
        local_alignment=True,
        batch_size=2,
        num_workers=0,
        mixed_precision=False,
        enable_checkpoints=False,
        audit_shards_enabled=False,
        audit_shard_compression="none",
        cache_persist_policy="never",
        log_every=100,
    )
    return runner


def test_runner_conditionally_persists_experiment_outputs_and_canonical_gate_json(
    tmp_path: Path,
) -> None:
    runner = _run(tmp_path / "experiment", emit=True)
    first, second = runner._candidate_rows

    assert first["s_strsim"] == pytest.approx(0.88)
    assert first["q_lex"] == pytest.approx(0.77)
    assert first["q_strsim"] == pytest.approx(0.66)
    assert first["I_lex"] == pytest.approx(0.55)
    assert first["I_strsim"] == pytest.approx(0.45)
    assert first["llm_gate_mode"] == "quantile"
    assert first["llm_gate_would_route"] is True
    assert first["llm_gate_invoked"] is False
    assert first["llm_gate_outcome"] == "routed_not_invoked"
    assert second["llm_gate_outcome"] == "invoked"

    diagnostic = json.loads(first["llm_gate_diagnostics"])
    assert diagnostic["nested"] == {"non_finite": None, "tensor": 2}
    assert diagnostic["tags"] == ["a", "b"]
    assert first["llm_gate_diagnostics"] == json.dumps(
        diagnostic,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    baseline = _run(tmp_path / "baseline", emit=False)
    for row in baseline._candidate_rows:
        assert "s_strsim" not in row
        assert "q_lex" not in row
        assert "q_strsim" not in row
        assert "I_lex" not in row
        assert "I_strsim" not in row
        assert "llm_gate_diagnostics" not in row
        assert "llm_gate_outcome" not in row


def test_runner_persists_deduplicated_sanitized_llm_backend_provenance(
    tmp_path: Path,
) -> None:
    runner = SemanticAlignmentRunner(
        dataset=_FixtureDataset(),
        model=_BackendUsageModel,
        device=torch.device("cpu"),
        output_dir=tmp_path / "llm-provenance",
    )
    predictions, _ = runner.predict(
        threshold=0.0,
        local_alignment=True,
        batch_size=1,
        num_workers=0,
        mixed_precision=False,
        enable_checkpoints=False,
        audit_shards_enabled=False,
        audit_shard_compression="none",
        cache_persist_policy="never",
        log_every=100,
    )
    paths = runner.save_results(predictions, save_json=False, save_stats_csv=False)
    stats = json.loads(paths["run_stats_json"].read_text(encoding="utf-8"))
    identities = stats["llm_usage"]["backend_identities"]

    assert list(identities) == ["summary", "decision", "rationale"]
    assert all(len(identities[task]) == 1 for task in identities)
    summary = identities["summary"][0]
    assert summary == {
        "backend": "openrouter",
        "profile": "hosted-profile",
        "requested_profile": "hosted-profile",
        "requested_model": "vendor/requested-model@v1",
        "effective_model": "vendor/effective-model@r2",
        "provider": "safe-provider",
        "endpoint": "https://api.example.test/v1/chat/completions",
        "tokenizer": "vendor/tokenizer@r3",
        "request_seed": 17,
        "decoding": {
            "max_input_tokens": 256,
            "max_tokens": 32,
            "temperature": 0.2,
            "top_p": 0.8,
        },
        "prompt_hashes": [f"{1:040x}", f"{2:040x}", "a" * 40],
        "cache_hashes": ["b" * 64],
    }
    decision = identities["decision"][0]
    assert decision["profile"] == "local-profile"
    assert decision["requested_profile"] == "hosted-profile"
    assert decision["requested_model"] == "vendor/requested-model@v1"
    assert decision["effective_model"] == "vendor/local-model@r4"
    assert decision["tokenizer"] == "vendor/local-tokenizer@r4"
    assert decision["decoding"]["scoring_mode"] == "local_next_token_logits"

    serialized = json.dumps(stats["llm_usage"], sort_keys=True)
    for forbidden in (
        "never persist",
        "provider-secret",
        "api-password",
        "query-secret",
        "debug-secret",
        "header-secret",
        "SECRET_API_KEY_ENV",
        "/secret/key/path",
        "provider payload secret",
        "Authorization",
    ):
        assert forbidden not in serialized

    baseline = _run(tmp_path / "baseline-stats", emit=False)
    baseline_stats = baseline._compute_run_stats(baseline.results_df)
    assert "llm_usage" not in baseline_stats
