"""Isolated, durable text-predicate client for the E07 Luna diagnostic.

This does not change the experiment router or selected judge. Paid calls require
the existing prompt policy and shared hard spending authority. The direct API
contract is https://developers.openai.com/api/reference/resources/decisions/methods/create.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from time import monotonic
from typing import Any

import httpx

from exact.llm.ledger import RequestLedger, request_identity
from exact.llm.prompt_budget import (
    DECISIONS_ENDPOINT,
    DECISIONS_MODEL,
    decisions_payload_text,
    load_prompt_budget_policy,
    validate_prompt_budget,
)
from exact.llm.routing import LLMProfile
from exact.llm.spending_admission import HostedSpendingError, selected_policy


def predicate_identity(
    input_text: str, instructions: str, *, tokenizer: str, tokenizer_revision: str
) -> dict[str, Any]:
    """Build the exact request identity without credentials, I/O or admission."""
    payload = {
        "model": DECISIONS_MODEL,
        "input": input_text,
        "questions": [{"type": "predicate", "name": "equivalent", "instructions": instructions}],
    }
    decisions_payload_text(payload)
    return {
        "schema_version": 1,
        "role": "decision",
        "endpoint": DECISIONS_ENDPOINT,
        "payload": payload,
        "revision": "openai_decisions_predicate_v1",
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
    }


def _usage(raw: Any) -> dict[str, Any]:
    """Normalize reported counts, retaining missing/invalid counts as unknown."""
    usage: dict[str, Any] = {"provider_usage": raw}
    if not isinstance(raw, dict):
        return usage
    for source, target in (
        ("input_tokens", "prompt_tokens"),
        ("output_tokens", "completion_tokens"),
    ):
        value = raw.get(source)
        if type(value) is int and 0 <= value <= 2**63 - 1:
            usage[target] = value
    total = raw.get("total_tokens")
    if (
        type(total) is int
        and "prompt_tokens" in usage
        and "completion_tokens" in usage
        and total != usage["prompt_tokens"] + usage["completion_tokens"]
    ):
        # Inconsistent provider counts cannot reduce the conservative reservation.
        usage.pop("prompt_tokens")
        usage.pop("completion_tokens")
    cost = raw.get("cost")
    if (
        isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and math.isfinite(cost)
        and cost >= 0
    ):
        usage["cost"] = cost
    return usage


class DecisionsClient:
    def __init__(
        self,
        ledger_dir: Path,
        *,
        tokenizer: str,
        tokenizer_revision: str,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout_secs: float = 60.0,
        cache_only: bool = False,
    ) -> None:
        if not math.isfinite(timeout_secs) or timeout_secs <= 0:
            raise ValueError("Decisions timeout must be finite and positive")
        self.ledger = RequestLedger(Path(ledger_dir))
        self.profile = LLMProfile(
            name="openai_luna_decisions",
            backend="openai_decisions",
            model=DECISIONS_MODEL,
            revision="openai_decisions_predicate_v1",
            tokenizer=tokenizer,
            tokenizer_revision=tokenizer_revision,
        )
        self._api_key = api_key
        self.cache_only = cache_only
        self._client = httpx.Client(transport=transport, timeout=timeout_secs)

    def close(self) -> None:
        self._client.close()

    def predict(self, input_text: str, instructions: str) -> dict[str, Any]:
        """Return a probability or explicit refusal; never silently retry a send."""
        identity = predicate_identity(
            input_text,
            instructions,
            tokenizer=self.profile.tokenizer or "",
            tokenizer_revision=self.profile.tokenizer_revision or "",
        )
        payload = identity["payload"]
        wire = decisions_payload_text(payload).encode("utf-8")
        cache_env = os.getenv("EXACT_HOSTED_CACHE_ONLY")
        if cache_env not in (None, "0", "1"):
            raise ValueError("EXACT_HOSTED_CACHE_ONLY must be unset, '0', or '1'")
        cache_only = self.cache_only or cache_env == "1"
        key = request_identity(identity)[0] if cache_only else self.ledger.plan(identity)
        with self.ledger.request_lock(key):
            cached = self.ledger.cached_attempt(key)
            if cached is not None:
                return self._result(key, cached[0], cached[1], cached=True)
            if cache_only:
                raise RuntimeError(f"Decisions cached-only request {key} has no saved response")
            api_key = self._api_key if self._api_key is not None else os.getenv("OPENAI_API_KEY")
            if not isinstance(api_key, str) or not api_key.strip():
                raise RuntimeError("Direct Decisions API requires OPENAI_API_KEY")
            api_key = api_key.strip()
            if api_key.startswith("sk-or-"):
                raise RuntimeError("Direct Decisions API cannot use an OpenRouter credential")
            policy = selected_policy()
            if policy is None or policy["policy"]["schema_version"] != 2:
                raise HostedSpendingError("Decisions requires the bound v2 hosted spending policy")
            prompt_policy = load_prompt_budget_policy(required=True)
            validate_prompt_budget(
                self.profile, payload, "decisions", role="decision", binding=prompt_policy
            )
            number = self.ledger.sent(key)
            started = monotonic()
            try:
                response = self._client.post(
                    DECISIONS_ENDPOINT,
                    content=wire,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
            except BaseException as exc:
                self.ledger.unknown(
                    key, number, type(exc).__name__, elapsed_seconds=monotonic() - started
                )
                if isinstance(exc, httpx.TransportError):
                    raise RuntimeError(
                        f"Decisions request {key} has unknown delivery; explicit retry required"
                    ) from exc
                raise
            self.ledger.received(
                key,
                number,
                response.content,
                response.status_code,
                elapsed_seconds=monotonic() - started,
            )
            if not response.is_success:
                raise RuntimeError(f"Decisions HTTP {response.status_code}; request {key} retained")
            return self._result(key, number, response.content, cached=False)

    def _result(self, key: str, number: int, raw: bytes, *, cached: bool) -> dict[str, Any]:
        data = json.loads(raw)
        usage = _usage(data.get("usage") if isinstance(data, dict) else None)
        # Account credible usage even for refused, malformed or wrong-model answers.
        # Replaying also recovers a crash between raw commit and usage settlement.
        self.ledger.usage(key, number, usage)
        if not isinstance(data, dict) or data.get("model") != DECISIONS_MODEL:
            raise ValueError("Decisions returned an unexpected or missing model identity")
        answers = data.get("answers")
        if (
            not isinstance(answers, list)
            or len(answers) != 1
            or not isinstance(answers[0], dict)
            or answers[0].get("name") != "equivalent"
            or answers[0].get("type") not in {"predicate", "refusal"}
        ):
            raise ValueError("Decisions returned an invalid or mismatched predicate answer")
        answer = answers[0]
        refused = answer["type"] == "refusal"
        probability = None if refused else answer.get("probability")
        if refused and "probability" in answer:
            raise ValueError("Decisions refusal cannot include a probability")
        if not refused and (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("Decisions probability must be finite and within [0, 1]")
        if usage.get("completion_tokens", 0) != 0:
            raise ValueError("Decisions unexpectedly reported generated output tokens")
        return {
            "probability": float(probability) if probability is not None else None,
            "refused": refused,
            "refusal": "model_refusal" if refused else None,
            "request_id": key,
            "response_id": data.get("id"),
            "usage": usage,
            "cached": cached,
        }
