from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import httpx


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_KEY_PATH = "~/.config/openrouter/api_key"
DEFAULT_OPENROUTER_MAX_RETRIES = 2
DEFAULT_OPENROUTER_MAX_KEEPALIVE_CONNECTIONS = 20
DEFAULT_OPENROUTER_MAX_CONNECTIONS = 100
TRANSIENT_HTTP_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def _noop_logger(*args, **kwargs) -> None:
    return None


def _json_headers(api_key: Optional[str] = None, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update({str(k): str(v) for k, v in extra.items()})
    return headers


def _load_json_from_text(text: str) -> Dict[str, Any]:
    blob = (text or "").strip()
    if not blob:
        raise ValueError("Empty JSON response.")
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", blob, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _strip_code_fences(text: str) -> str:
    blob = (text or "").strip()
    if not blob:
        return ""
    blob = re.sub(r"^\s*```(?:json)?\s*", "", blob, flags=re.IGNORECASE)
    blob = re.sub(r"\s*```\s*$", "", blob)
    return blob.strip()


def _extract_json_string_value(text: str, required_key: str) -> Optional[str]:
    blob = _strip_code_fences(text)
    if not blob:
        return None
    pattern = re.compile(
        rf'"{re.escape(required_key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
        flags=re.DOTALL,
    )
    match = pattern.search(blob)
    if not match:
        return None
    raw_value = match.group(1)
    try:
        return json.loads(f'"{raw_value}"').strip()
    except json.JSONDecodeError:
        return raw_value.strip()


def _read_key_file(path_str: str) -> Optional[str]:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.exists() or not path.is_file():
        return None
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def _truncate_http_body(text: str, limit: int = 1200) -> str:
    body = str(text or "").strip()
    if len(body) <= limit:
        return body
    return body[:limit] + "...<truncated>"


@dataclass
class LLMProfile:
    name: str
    backend: str = "local_hf"
    model: Optional[str] = None
    tokenizer: Optional[str] = None
    api_base: str = DEFAULT_OPENROUTER_BASE_URL
    api_key_env: str = "OPENROUTER_API_KEY"
    api_key_path: Optional[str] = None
    timeout_secs: float = 60.0
    extra_headers: Dict[str, str] = field(default_factory=dict)
    provider: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, name: str, raw: Optional[Dict[str, Any]]) -> "LLMProfile":
        raw = dict(raw or {})
        headers = raw.get("extra_headers") or {}
        return cls(
            name=name,
            backend=str(raw.get("backend", "local_hf")).strip() or "local_hf",
            model=raw.get("model"),
            tokenizer=raw.get("tokenizer"),
            api_base=str(raw.get("api_base", DEFAULT_OPENROUTER_BASE_URL)).rstrip("/"),
            api_key_env=str(raw.get("api_key_env", "OPENROUTER_API_KEY")).strip() or "OPENROUTER_API_KEY",
            api_key_path=raw.get("api_key_path"),
            timeout_secs=float(raw.get("timeout_secs", 60.0)),
            extra_headers={str(k): str(v) for k, v in dict(headers).items()},
            provider=dict(raw.get("provider") or {}),
        )


@dataclass
class LLMRouting:
    default_profile: Optional[str] = None
    verbaliser_profile: Optional[str] = None
    summary_profile: Optional[str] = None
    decision_profile: Optional[str] = None
    rationale_profile: Optional[str] = None
    verbaliser_fallback_profile: Optional[str] = None
    summary_fallback_profile: Optional[str] = None
    rationale_fallback_profile: Optional[str] = None
    fallback_profile: Optional[str] = None
    decision_fallback_profile: Optional[str] = None

    @classmethod
    def from_raw(cls, raw: Optional[Dict[str, Any]]) -> "LLMRouting":
        raw = dict(raw or {})
        return cls(
            default_profile=raw.get("default_profile"),
            verbaliser_profile=raw.get("verbaliser_profile"),
            summary_profile=raw.get("summary_profile"),
            decision_profile=raw.get("decision_profile"),
            rationale_profile=raw.get("rationale_profile"),
            verbaliser_fallback_profile=raw.get("verbaliser_fallback_profile"),
            summary_fallback_profile=raw.get("summary_fallback_profile"),
            rationale_fallback_profile=raw.get("rationale_fallback_profile"),
            fallback_profile=raw.get("fallback_profile"),
            decision_fallback_profile=raw.get("decision_fallback_profile"),
        )

    def profile_for_task(self, task: str) -> Optional[str]:
        per_task = {
            "verbaliser": self.verbaliser_profile,
            "summary": self.summary_profile,
            "decision": self.decision_profile,
            "rationale": self.rationale_profile,
        }
        return per_task.get(task) or self.default_profile

    def fallback_for_task(self, task: str) -> Optional[str]:
        per_task = {
            "verbaliser": self.verbaliser_fallback_profile,
            "summary": self.summary_fallback_profile,
            "rationale": self.rationale_fallback_profile,
        }
        if task == "decision":
            return self.decision_fallback_profile or self.fallback_profile
        return per_task.get(task) or self.fallback_profile


@dataclass
class ResolvedLLMTask:
    task: str
    backend: str
    profile_name: Optional[str]
    model: Optional[str]
    decision_capable: bool
    fallback_triggered: bool
    fallback_reason: Optional[str]
    api_key: Optional[str] = None


class OpenRouterClient:
    _capability_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    _prompted_key_paths: Dict[str, Optional[str]] = {}

    def __init__(self, log=None):
        self.log = log or _noop_logger
        self.max_retries = DEFAULT_OPENROUTER_MAX_RETRIES
        self._client = httpx.Client(
            headers={"Accept": "application/json"},
            limits=httpx.Limits(
                max_keepalive_connections=DEFAULT_OPENROUTER_MAX_KEEPALIVE_CONNECTIONS,
                max_connections=DEFAULT_OPENROUTER_MAX_CONNECTIONS,
            ),
        )

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _should_retry_status(status_code: int) -> bool:
        return int(status_code) in TRANSIENT_HTTP_STATUS_CODES

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = min(2.0, 0.25 * (2 ** attempt)) + random.uniform(0.0, 0.05)
        time.sleep(delay)

    def _http_json(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        payload: Optional[Dict[str, Any]] = None,
        timeout_secs: float = 60.0,
    ) -> Dict[str, Any]:
        model = None
        provider = None
        if isinstance(payload, dict):
            model = payload.get("model")
            provider = payload.get("provider")

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(
                    method=method,
                    url=url,
                    headers=headers or {},
                    json=payload,
                    timeout=timeout_secs,
                )
                response.raise_for_status()
                body = response.text
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return _load_json_from_text(body)
            except httpx.HTTPStatusError as exc:
                body_excerpt = _truncate_http_body(exc.response.text if exc.response is not None else "")
                details = (
                    f"OpenRouter HTTP {exc.response.status_code} {exc.response.reason_phrase} for {method} {url}"
                    + (f" model={model!r}" if model is not None else "")
                    + (f" provider={provider!r}" if provider is not None else "")
                )
                if body_excerpt:
                    details += f" body={body_excerpt}"
                if attempt < self.max_retries and self._should_retry_status(exc.response.status_code):
                    self.log(
                        f"{details}; retrying ({attempt + 1}/{self.max_retries}).",
                        "debug",
                    )
                    self._sleep_before_retry(attempt)
                    continue
                raise RuntimeError(details) from exc
            except (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                httpx.WriteError,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
            ) as exc:
                details = (
                    f"OpenRouter transport error for {method} {url}"
                    + (f" model={model!r}" if model is not None else "")
                    + (f" provider={provider!r}" if provider is not None else "")
                    + f": {exc}"
                )
                if attempt < self.max_retries:
                    self.log(
                        f"{details}; retrying ({attempt + 1}/{self.max_retries}).",
                        "debug",
                    )
                    self._sleep_before_retry(attempt)
                    continue
                raise RuntimeError(details) from exc
        raise RuntimeError(f"OpenRouter request failed for {method} {url} after retries were exhausted.")

    def _prompt_key_path(self, env_name: str) -> Optional[str]:
        if env_name in self._prompted_key_paths:
            return self._prompted_key_paths[env_name]
        if not sys.stdin or not sys.stdin.isatty():
            self._prompted_key_paths[env_name] = None
            return None
        try:
            path_str = input(
                f"{env_name} is not set. Enter a path to a file containing the OpenRouter key (leave blank to skip): "
            ).strip()
        except EOFError:
            path_str = ""
        value = path_str or None
        self._prompted_key_paths[env_name] = value
        return value

    def resolve_api_key(self, profile: LLMProfile) -> Optional[str]:
        env_value = os.getenv(profile.api_key_env, "").strip()
        if env_value:
            return env_value
        file_token = _read_key_file(profile.api_key_path or "")
        if file_token:
            return file_token
        file_token = _read_key_file(DEFAULT_OPENROUTER_KEY_PATH)
        if file_token:
            return file_token
        prompted_path = self._prompt_key_path(profile.api_key_env)
        file_token = _read_key_file(prompted_path or "")
        if file_token:
            return file_token
        return None

    def model_capabilities(self, profile: LLMProfile) -> Dict[str, Any]:
        key = (profile.api_base, profile.model or "")
        cached = self._capability_cache.get(key)
        if cached is not None:
            return cached
        payload = self._http_json(
            url=f"{profile.api_base}/models",
            headers=_json_headers(extra=profile.extra_headers),
            timeout_secs=profile.timeout_secs,
        )
        data = payload.get("data") or []
        lookup = {
            str(item.get("id")): item
            for item in data
            if isinstance(item, dict) and item.get("id") is not None
        }
        result = lookup.get(profile.model or "", {})
        self._capability_cache[key] = result
        return result

    def supports_parameter(self, profile: LLMProfile, parameter: str) -> bool:
        meta = self.model_capabilities(profile)
        params = meta.get("supported_parameters") or []
        return str(parameter) in set(str(p) for p in params)

    @staticmethod
    def _merged_provider(profile: LLMProfile, provider: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        merged: Dict[str, Any] = {}
        if profile.provider:
            merged.update(dict(profile.provider))
        if provider:
            merged.update(dict(provider))
        return merged or None

    def chat_completion(
        self,
        profile: LLMProfile,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Iterable[str]] = None,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        provider: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        api_key = self.resolve_api_key(profile)
        if not api_key:
            raise RuntimeError(f"Missing OpenRouter API key for profile '{profile.name}'.")
        payload: Dict[str, Any] = {
            "model": profile.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stop:
            payload["stop"] = list(stop)
        if logprobs:
            payload["logprobs"] = True
            if top_logprobs is not None:
                payload["top_logprobs"] = int(top_logprobs)
        if logit_bias:
            payload["logit_bias"] = {str(k): float(v) for k, v in dict(logit_bias).items()}
        if seed is not None and self.supports_parameter(profile, "seed"):
            payload["seed"] = int(seed)
        merged_provider = self._merged_provider(profile, provider)
        if merged_provider is not None:
            payload["provider"] = merged_provider
        return self._http_json(
            url=f"{profile.api_base}/chat/completions",
            method="POST",
            headers=_json_headers(api_key=api_key, extra=profile.extra_headers),
            payload=payload,
            timeout_secs=profile.timeout_secs,
        )

    def completion(
        self,
        profile: LLMProfile,
        prompt: Union[str, List[str]],
        max_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Iterable[str]] = None,
        logprobs: Optional[int] = None,
        echo: Optional[bool] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        provider: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        api_key = self.resolve_api_key(profile)
        if not api_key:
            raise RuntimeError(f"Missing OpenRouter API key for profile '{profile.name}'.")
        payload: Dict[str, Any] = {
            "model": profile.model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stop:
            payload["stop"] = list(stop)
        if logprobs is not None:
            payload["logprobs"] = int(logprobs)
        if echo is not None:
            payload["echo"] = bool(echo)
        if logit_bias:
            payload["logit_bias"] = {str(k): float(v) for k, v in dict(logit_bias).items()}
        if seed is not None and self.supports_parameter(profile, "seed"):
            payload["seed"] = int(seed)
        merged_provider = self._merged_provider(profile, provider)
        if merged_provider is not None:
            payload["provider"] = merged_provider
        return self._http_json(
            url=f"{profile.api_base}/completions",
            method="POST",
            headers=_json_headers(api_key=api_key, extra=profile.extra_headers),
            payload=payload,
            timeout_secs=profile.timeout_secs,
        )


class LLMRouter:
    def __init__(
        self,
        llm_profiles: Optional[Dict[str, Any]] = None,
        llm_routing: Optional[Dict[str, Any]] = None,
        log=None,
    ):
        self.log = log or _noop_logger
        raw_profiles = dict(llm_profiles or {})
        self.profiles = {
            str(name): LLMProfile.from_raw(str(name), raw)
            for name, raw in raw_profiles.items()
        }
        self.routing = LLMRouting.from_raw(llm_routing)
        self.hosted = OpenRouterClient(log=self.log)

    def fingerprint_payload(self) -> Dict[str, Any]:
        return {
            "routing": {
                "default_profile": self.routing.default_profile,
                "verbaliser_profile": self.routing.verbaliser_profile,
                "summary_profile": self.routing.summary_profile,
                "decision_profile": self.routing.decision_profile,
                "rationale_profile": self.routing.rationale_profile,
                "verbaliser_fallback_profile": self.routing.verbaliser_fallback_profile,
                "summary_fallback_profile": self.routing.summary_fallback_profile,
                "rationale_fallback_profile": self.routing.rationale_fallback_profile,
                "fallback_profile": self.routing.fallback_profile,
                "decision_fallback_profile": self.routing.decision_fallback_profile,
            },
            "profiles": {
                name: {
                    "backend": profile.backend,
                    "model": profile.model,
                    "tokenizer": profile.tokenizer,
                    "api_base": profile.api_base,
                    "api_key_env": profile.api_key_env,
                    "timeout_secs": profile.timeout_secs,
                    "extra_headers": dict(profile.extra_headers),
                    "provider": dict(profile.provider),
                }
                for name, profile in sorted(self.profiles.items())
            },
        }

    def ensure_profile(self, name: str, raw: Optional[Dict[str, Any]]) -> None:
        if name in self.profiles:
            return
        self.profiles[name] = LLMProfile.from_raw(name, raw)

    def _profile(self, name: Optional[str]) -> Optional[LLMProfile]:
        if not name:
            return None
        return self.profiles.get(name)

    def resolve_task(self, task: str, require_logprobs: bool = False) -> ResolvedLLMTask:
        primary_name = self.routing.profile_for_task(task)
        fallback_name = self.routing.fallback_for_task(task)
        primary = self._profile(primary_name)
        fallback = self._profile(fallback_name)

        if primary is None:
            if fallback is None:
                return ResolvedLLMTask(task, "none", None, None, False, False, "no_profile_configured")
            self.log(
                (
                    f"LLM routing selected profile '{primary_name}' for task '{task}', "
                    f"but no such profile is defined. Falling back to '{fallback.name}'."
                ),
                "warning",
            )
            return ResolvedLLMTask(task, fallback.backend, fallback.name, fallback.model, not require_logprobs, True, "missing_primary_profile")

        if primary.backend != "openrouter":
            return ResolvedLLMTask(task, primary.backend, primary.name, primary.model, not require_logprobs, False, None)

        api_key = self.hosted.resolve_api_key(primary)
        if not api_key:
            if fallback is not None:
                self.log(
                    (
                        f"OpenRouter profile '{primary.name}' selected for task '{task}' but no API key was found. "
                        f"Falling back to '{fallback.name}'."
                    ),
                    "warning",
                )
                return ResolvedLLMTask(task, fallback.backend, fallback.name, fallback.model, False, True, "missing_api_key")
            return ResolvedLLMTask(task, "openrouter", primary.name, primary.model, False, False, "missing_api_key")

        if require_logprobs and not self.hosted.supports_parameter(primary, "logprobs"):
            if fallback is not None:
                self.log(
                    (
                        f"OpenRouter model '{primary.model}' does not advertise logprobs support for task '{task}'. "
                        f"Falling back to '{fallback.name}'."
                    ),
                    "warning",
                )
                return ResolvedLLMTask(task, fallback.backend, fallback.name, fallback.model, False, True, "logprobs_unsupported")
            return ResolvedLLMTask(task, "openrouter", primary.name, primary.model, False, False, "logprobs_unsupported", api_key=api_key)

        return ResolvedLLMTask(task, "openrouter", primary.name, primary.model, True, False, None, api_key=api_key)


def extract_chat_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts).strip()
    return ""


def extract_first_token_top_logprobs(payload: Dict[str, Any]) -> List[Tuple[str, float]]:
    choices = payload.get("choices") or []
    if not choices:
        return []
    logprobs = (choices[0] or {}).get("logprobs") or {}
    content = logprobs.get("content") or []
    if not content:
        return []
    first = content[0] or {}
    top = first.get("top_logprobs") or []
    pairs: List[Tuple[str, float]] = []
    for item in top:
        if not isinstance(item, dict):
            continue
        token = str(item.get("token", ""))
        lp = item.get("logprob")
        if lp is None:
            continue
        try:
            pairs.append((token, float(lp)))
        except (TypeError, ValueError):
            continue
    return pairs


def extract_completion_span_logprobs(
    payload: Dict[str, Any],
    span_start: int,
    span_end: int,
) -> List[Dict[str, Any]]:
    choices = payload.get("choices") or []
    if not choices:
        return []
    logprobs = (choices[0] or {}).get("logprobs") or {}
    tokens = logprobs.get("tokens") or []
    token_logprobs = logprobs.get("token_logprobs") or []
    text_offsets = logprobs.get("text_offset") or []
    overlaps: List[Dict[str, Any]] = []
    for token, logprob, offset in zip(tokens, token_logprobs, text_offsets):
        if logprob is None or offset is None:
            continue
        token_text = str(token)
        start = int(offset)
        end = start + len(token_text)
        if end <= span_start or start >= span_end:
            continue
        overlaps.append(
            {
                "token": token_text,
                "logprob": float(logprob),
                "start": start,
                "end": end,
            }
        )
    return overlaps


def extract_completion_suffix_logprob(
    payload: Dict[str, Any],
    suffix_start: int,
    suffix_end: int,
) -> float:
    overlaps = extract_completion_span_logprobs(payload, suffix_start, suffix_end)
    if not overlaps:
        raise ValueError("OpenRouter completion response lacked echoed token logprobs for the suffix span.")
    return float(sum(float(item["logprob"]) for item in overlaps))


def split_completion_choices(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    choices = payload.get("choices") or []
    provider = payload.get("provider")
    model = payload.get("model")
    system_fingerprint = payload.get("system_fingerprint")
    ordered = sorted(
        [choice for choice in choices if isinstance(choice, dict)],
        key=lambda choice: int(choice.get("index", 0) or 0),
    )
    return [
        {
            "choices": [choice],
            "provider": provider,
            "model": model,
            "system_fingerprint": system_fingerprint,
        }
        for choice in ordered
    ]


def parse_structured_json(text: str, required_key: str) -> str:
    normalized = _strip_code_fences(text)
    try:
        payload = _load_json_from_text(normalized)
        value = payload.get(required_key)
        if value is None:
            raise ValueError(f"Missing key '{required_key}' in model response.")
        return str(value).strip()
    except Exception:
        fallback = _extract_json_string_value(normalized, required_key)
        if fallback is None:
            raise
        return fallback
