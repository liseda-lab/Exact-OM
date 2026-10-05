"""Content-bound admission limits for hosted generation, before paid transmission.

The admission estimate uses a pinned local tokenizer plus a framing reserve.
A separate UTF-8 byte cap bounds request size when that tokenizer differs from
the hosted model. Neither estimate claims the provider's exact billed token
count. No tokenizer downloads are allowed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Protocol, cast

POLICY_PATH_ENV = "EXACT_LLM_PROMPT_POLICY_PATH"
POLICY_SHA256_ENV = "EXACT_LLM_PROMPT_POLICY_SHA256"
MAX_INPUT_TOKENS = 12288
MAX_OUTPUT_TOKENS = 1024
MAX_INPUT_BYTES = 32768
COUNTING_METHOD = "pinned_tokenizer_plus_overhead_and_utf8_cap_v1"


class PromptBudgetError(RuntimeError):
    """A local prompt policy rejection; never a sent or unknown hosted request."""


class TokenizerProfile(Protocol):
    """The selected profile fields needed for offline admission."""

    name: str
    model: str | None
    tokenizer: str | None
    tokenizer_revision: str | None


def load_prompt_budget_policy(
    binding: Mapping[str, Any] | None = None, *, required: bool | None = None
) -> dict[str, Any] | None:
    """Read and verify a policy binding; experiment transmissions require one."""
    required = os.getenv("EXACT_EXPERIMENT_MODE") == "1" if required is None else required
    path = binding.get("path") if binding is not None else os.getenv(POLICY_PATH_ENV)
    expected = binding.get("sha256") if binding is not None else os.getenv(POLICY_SHA256_ENV)
    if not path and not expected and not required:
        return None
    if not path or not expected:
        raise PromptBudgetError("Hosted prompt budget requires a policy path and SHA256 binding")
    try:
        location = Path(path).resolve()
        raw = location.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise PromptBudgetError("Hosted prompt policy bytes differ from their SHA256 binding")
        policy = json.loads(raw)
    except PromptBudgetError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise PromptBudgetError("Cannot read the bound hosted prompt policy") from exc
    if not isinstance(policy, dict) or type(policy.get("schema_version")) is not int:
        raise PromptBudgetError("Invalid hosted prompt policy schema")
    if policy["schema_version"] != 1:
        raise PromptBudgetError("Unsupported hosted prompt policy schema")
    for field, maximum in (
        ("max_input_tokens", MAX_INPUT_TOKENS),
        ("max_output_tokens", MAX_OUTPUT_TOKENS),
        ("max_input_bytes", MAX_INPUT_BYTES),
    ):
        value = policy.get(field)
        if type(value) is not int or not 1 <= value <= maximum:
            raise PromptBudgetError(f"Hosted prompt policy {field} must be in 1..{maximum}")
    reserve = policy.get("chat_overhead_tokens")
    if type(reserve) is not int or reserve < 128:
        raise PromptBudgetError("Hosted prompt policy requires at least 128 framing reserve tokens")
    if policy.get("counting_method", COUNTING_METHOD) != COUNTING_METHOD:
        raise PromptBudgetError("Unsupported hosted prompt counting method")
    return {"path": str(location), "sha256": expected, "policy": policy}


@lru_cache(maxsize=16)
def _load_tokenizer(name: str, revision: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        name, revision=revision, local_files_only=True, trust_remote_code=False
    )


def _count_text(tokenizer: Any, text: str, *, special_tokens: bool) -> tuple[int, int]:
    tokens = tokenizer.encode(text, add_special_tokens=special_tokens)
    if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
        raise PromptBudgetError("Pinned tokenizer returned unsupported token IDs")
    return len(tokens), len(text.encode("utf-8"))


def validate_prompt_budget(
    profile: TokenizerProfile,
    payload: Mapping[str, Any],
    endpoint: str,
    *,
    role: str | None = None,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Admit a final sanitized payload, or raise before any paid attempt exists.

    Completion batches sum all input estimates and all requested output tokens.
    Unsupported fields/content fail closed instead of silently omitting tokens.
    Operational policy metadata intentionally stays out of request/cache identity.
    """
    selected = load_prompt_budget_policy(binding)
    if selected is None:
        return None
    policy = selected["policy"]
    label = f"role={role or profile.name!r} profile={profile.name!r} endpoint={endpoint!r}"
    try:
        if endpoint not in {"chat/completions", "completions"}:
            raise PromptBudgetError("Unsupported generation endpoint")
        common = {
            "model",
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "seed",
            "provider",
            "echo",
        }
        accepted = common | ({"messages"} if endpoint == "chat/completions" else {"prompt"})
        if set(payload) - accepted:
            raise PromptBudgetError("Unsupported generation fields cannot be counted safely")
        if payload.get("model") != profile.model:
            raise PromptBudgetError("Generation model differs from the selected tokenizer profile")
        output = payload.get("max_tokens")
        if type(output) is not int or output < 1:
            raise PromptBudgetError("Hosted prompt budget requires positive integer max_tokens")
        messages = None
        prompts: list[str] = []
        if endpoint == "chat/completions":
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise PromptBudgetError("Hosted chat budget requires nonempty text messages")
            for message in messages:
                if (
                    not isinstance(message, dict)
                    or set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user", "assistant"}
                    or not isinstance(message["content"], str)
                ):
                    raise PromptBudgetError(
                        "Unsupported chat message content cannot be counted safely"
                    )
            batch_size = 1
            utf8_bytes = sum(
                len(message[field].encode("utf-8"))
                for message in messages
                for field in ("role", "content")
            )
        else:
            value = payload.get("prompt")
            prompt_values = [value] if isinstance(value, str) else value
            if (
                not isinstance(prompt_values, list)
                or not prompt_values
                or any(not isinstance(prompt, str) for prompt in prompt_values)
            ):
                raise PromptBudgetError("Hosted completion budget requires text prompts")
            prompts = cast(list[str], prompt_values)
            batch_size = len(prompts)
            utf8_bytes = sum(len(prompt.encode("utf-8")) for prompt in prompts)
        if utf8_bytes > policy["max_input_bytes"]:
            raise PromptBudgetError(
                f"Input text bytes {utf8_bytes} exceed limit {policy['max_input_bytes']}"
            )
        output_total = output * batch_size
        if output_total > policy["max_output_tokens"]:
            raise PromptBudgetError(
                f"Requested output tokens {output_total} exceed limit {policy['max_output_tokens']}"
            )
        name, revision = profile.tokenizer, profile.tokenizer_revision
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(revision, str)
            or not re.fullmatch(r"[0-9a-fA-F]{40}", revision)
        ):
            raise PromptBudgetError(
                "Hosted prompt budget requires an explicitly pinned tokenizer revision"
            )
        tokenizer = _load_tokenizer(name, revision)
        if messages is not None:
            # Some approved hosted tokenizers have no chat template. Counting
            # every role and content retains a declared local estimate; the
            # independent byte cap bounds text even for a mismatched tokenizer.
            if getattr(tokenizer, "chat_template", None):
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if not isinstance(text, str):
                    raise PromptBudgetError("Pinned chat template did not return text")
                raw_tokens, _ = _count_text(tokenizer, text, special_tokens=False)
                method = "chat_template"
            else:
                text = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
                raw_tokens, _ = _count_text(tokenizer, text, special_tokens=True)
                method = "text_roles_json"
            estimate = raw_tokens + policy["chat_overhead_tokens"]
        else:
            counts = [_count_text(tokenizer, prompt, special_tokens=True) for prompt in prompts]
            raw_tokens = sum(count[0] for count in counts)
            estimate = raw_tokens + batch_size * policy["chat_overhead_tokens"]
            method = "completion_text"
        if estimate > policy["max_input_tokens"]:
            raise PromptBudgetError(
                f"Input admission estimate {estimate} exceeds limit {policy['max_input_tokens']} "
                f"(local tokenizer={raw_tokens}, UTF-8 bytes={utf8_bytes}, "
                f"reserve={policy['chat_overhead_tokens']} per prompt)"
            )
        return {
            "policy_sha256": selected["sha256"],
            "counting_method": COUNTING_METHOD,
            "rendering_method": method,
            "hosted_model": profile.model,
            "provider_exact_token_count": False,
            "tokenizer": name,
            "tokenizer_revision": revision,
            "local_tokenizer_tokens": raw_tokens,
            "utf8_bytes": utf8_bytes,
            "input_admission_tokens": estimate,
            "requested_output_tokens": output_total,
            "batch_size": batch_size,
        }
    except PromptBudgetError as exc:
        raise PromptBudgetError(f"{exc}; {label}") from exc
    except Exception as exc:
        raise PromptBudgetError(f"Cannot count the final hosted prompt offline; {label}") from exc
