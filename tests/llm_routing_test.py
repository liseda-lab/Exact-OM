import json

import httpx

from exact.utils.llm_routing import (
    DEFAULT_OPENROUTER_KEY_PATH,
    LLMProfile,
    LLMRouter,
    OpenRouterClient,
    extract_completion_span_logprobs,
    extract_completion_suffix_logprob,
    extract_first_token_top_logprobs,
    parse_structured_json,
    split_completion_choices,
)
from exact.utils.timing import load_recorded_timings, update_recorded_timings


def test_parse_structured_json_extracts_embedded_object():
    text = 'ignored prefix {"summary": "Entity summary."} ignored suffix'
    assert parse_structured_json(text, "summary") == "Entity summary."


def test_parse_structured_json_extracts_value_from_fenced_partial_json():
    text = '```json\n{"rationale":"First sentence. Second sentence without closing brace"\n'
    assert (
        parse_structured_json(text, "rationale")
        == "First sentence. Second sentence without closing brace"
    )


def test_extract_first_token_top_logprobs_reads_openai_shape():
    payload = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "Yes", "logprob": -0.1},
                                {"token": "No", "logprob": -2.4},
                            ]
                        }
                    ]
                }
            }
        ]
    }
    assert extract_first_token_top_logprobs(payload) == [("Yes", -0.1), ("No", -2.4)]


def test_extract_completion_suffix_logprob_single_token_suffix():
    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["Prompt:", " Yes", "!"],
                    "token_logprobs": [None, -0.25, -1.5],
                    "text_offset": [0, 7, 11],
                }
            }
        ]
    }
    assert extract_completion_suffix_logprob(payload, suffix_start=7, suffix_end=11) == -0.25


def test_extract_completion_suffix_logprob_multi_token_suffix():
    payload = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["Prompt:", " multi", " token", "\n"],
                    "token_logprobs": [None, -0.2, -0.3, -1.0],
                    "text_offset": [0, 7, 13, 19],
                }
            }
        ]
    }
    span = extract_completion_span_logprobs(payload, span_start=7, span_end=19)
    assert span == [
        {"token": " multi", "logprob": -0.2, "start": 7, "end": 13},
        {"token": " token", "logprob": -0.3, "start": 13, "end": 19},
    ]
    assert extract_completion_suffix_logprob(payload, suffix_start=7, suffix_end=19) == -0.5


def test_openrouter_completion_merges_profile_provider_preferences(monkeypatch):
    captured = {}
    client = OpenRouterClient()
    profile = LLMProfile(
        name="hosted",
        backend="openrouter",
        model="openai/gpt-4o-mini",
        provider={"sort": "latency"},
    )
    monkeypatch.setattr(client, "resolve_api_key", lambda profile: "test-key")

    def fake_http_json(url, method="GET", headers=None, payload=None, timeout_secs=60.0):
        captured["url"] = url
        captured["payload"] = payload
        return {"choices": []}

    monkeypatch.setattr(client, "_http_json", fake_http_json)
    client.completion(
        profile=profile,
        prompt="Prompt",
        max_tokens=1,
        logprobs=1,
        echo=True,
        provider={"require_parameters": True},
    )
    assert captured["url"].endswith("/completions")
    assert captured["payload"]["provider"] == {
        "sort": "latency",
        "require_parameters": True,
    }


def test_openrouter_chat_completion_includes_seed_when_supported(monkeypatch):
    captured = {}
    client = OpenRouterClient()
    profile = LLMProfile(name="hosted", backend="openrouter", model="openai/gpt-4o-mini")
    monkeypatch.setattr(client, "resolve_api_key", lambda profile: "test-key")
    monkeypatch.setattr(
        client, "supports_parameter", lambda profile, parameter: parameter == "seed"
    )

    def fake_http_json(url, method="GET", headers=None, payload=None, timeout_secs=60.0):
        captured["payload"] = payload
        return {"choices": []}

    monkeypatch.setattr(client, "_http_json", fake_http_json)
    client.chat_completion(
        profile=profile,
        messages=[{"role": "user", "content": "Prompt"}],
        max_tokens=1,
        seed=123,
    )
    assert captured["payload"]["seed"] == 123


def test_openrouter_chat_completion_omits_seed_when_unsupported(monkeypatch):
    captured = {}
    client = OpenRouterClient()
    profile = LLMProfile(name="hosted", backend="openrouter", model="openai/gpt-4o-mini")
    monkeypatch.setattr(client, "resolve_api_key", lambda profile: "test-key")
    monkeypatch.setattr(client, "supports_parameter", lambda profile, parameter: False)

    def fake_http_json(url, method="GET", headers=None, payload=None, timeout_secs=60.0):
        captured["payload"] = payload
        return {"choices": []}

    monkeypatch.setattr(client, "_http_json", fake_http_json)
    client.chat_completion(
        profile=profile,
        messages=[{"role": "user", "content": "Prompt"}],
        max_tokens=1,
        seed=123,
    )
    assert "seed" not in captured["payload"]


def test_openrouter_http_error_includes_body_and_model(monkeypatch):
    client = OpenRouterClient()

    monkeypatch.setattr(client, "resolve_api_key", lambda profile: "test-key")
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/completions")
    response = httpx.Response(
        status_code=404,
        request=request,
        content=b'{"error":{"message":"No endpoint for this model"}}',
    )

    def fake_request(**kwargs):
        raise httpx.HTTPStatusError("Not Found", request=request, response=response)

    monkeypatch.setattr(client._client, "request", fake_request)
    profile = LLMProfile(name="hosted", backend="openrouter", model="qwen/qwen3.5-122b-a10b")
    try:
        client.completion(profile=profile, prompt="Prompt", max_tokens=1)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError as exc:
        message = str(exc)
        assert "HTTP 404" in message
        assert "/completions" in message
        assert "qwen/qwen3.5-122b-a10b" in message
        assert "No endpoint for this model" in message


def test_openrouter_retries_transient_http_status(monkeypatch):
    client = OpenRouterClient()
    client.max_retries = 1
    monkeypatch.setattr(client, "resolve_api_key", lambda profile: "test-key")
    monkeypatch.setattr(client, "_sleep_before_retry", lambda attempt: None)
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    transient = httpx.Response(status_code=503, request=request, content=b'{"error":"busy"}')
    success = httpx.Response(
        status_code=200, request=request, content=json.dumps({"choices": []}).encode("utf-8")
    )
    calls = {"count": 0}

    def fake_request(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.HTTPStatusError("Service Unavailable", request=request, response=transient)
        return success

    monkeypatch.setattr(client._client, "request", fake_request)
    profile = LLMProfile(name="hosted", backend="openrouter", model="openai/gpt-4o-mini")
    payload = client.chat_completion(
        profile=profile, messages=[{"role": "user", "content": "hi"}], max_tokens=1
    )
    assert payload == {"choices": []}
    assert calls["count"] == 2


def test_split_completion_choices_orders_by_index():
    payload = {
        "provider": "unit-test-provider",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {"index": 1, "text": "second"},
            {"index": 0, "text": "first"},
        ],
    }
    split = split_completion_choices(payload)
    assert [item["choices"][0]["text"] for item in split] == ["first", "second"]
    assert all(item["provider"] == "unit-test-provider" for item in split)


def test_router_falls_back_to_local_when_key_missing(monkeypatch):
    router = LLMRouter(
        llm_profiles={
            "hosted": {"backend": "openrouter", "model": "openai/gpt-4o-mini"},
            "local": {"backend": "local_hf", "model": "Qwen/Qwen2.5-7B-Instruct"},
        },
        llm_routing={
            "decision_profile": "hosted",
            "decision_fallback_profile": "local",
        },
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(router.hosted, "_prompt_key_path", lambda env_name: None)
    monkeypatch.setattr("exact.utils.llm_routing._read_key_file", lambda path: None)
    resolved = router.resolve_task("decision", require_logprobs=True)
    assert resolved.backend == "local_hf"
    assert resolved.fallback_triggered is True
    assert resolved.fallback_reason == "missing_api_key"


def test_router_falls_back_to_local_when_logprobs_missing(monkeypatch):
    router = LLMRouter(
        llm_profiles={
            "hosted": {"backend": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
            "local": {"backend": "local_hf", "model": "Qwen/Qwen2.5-7B-Instruct"},
        },
        llm_routing={
            "decision_profile": "hosted",
            "decision_fallback_profile": "local",
        },
    )
    monkeypatch.setattr(router.hosted, "resolve_api_key", lambda profile: "test-key")
    monkeypatch.setattr(router.hosted, "supports_parameter", lambda profile, parameter: False)
    resolved = router.resolve_task("decision", require_logprobs=True)
    assert resolved.backend == "local_hf"
    assert resolved.fallback_triggered is True
    assert resolved.fallback_reason == "logprobs_unsupported"


def test_router_uses_default_key_path_before_prompt(monkeypatch):
    router = LLMRouter(
        llm_profiles={
            "hosted": {"backend": "openrouter", "model": "openai/gpt-4o-mini"},
            "local": {"backend": "local_hf", "model": "Qwen/Qwen2.5-7B-Instruct"},
        },
        llm_routing={
            "summary_profile": "hosted",
            "summary_fallback_profile": "local",
        },
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(router.hosted, "_prompt_key_path", lambda env_name: None)
    monkeypatch.setattr(
        "exact.utils.llm_routing._read_key_file",
        lambda path: "default-file-key" if path == DEFAULT_OPENROUTER_KEY_PATH else None,
    )
    resolved = router.resolve_task("summary")
    assert resolved.backend == "openrouter"
    assert resolved.fallback_triggered is False


def test_update_recorded_timings_merges_without_dropping_previous_steps(tmp_path):
    times_file = tmp_path / "times.txt"
    update_recorded_timings(times_file, {"Dataset": 12.5, "Dataset.Process": 4.0})
    update_recorded_timings(times_file, {"Alignment": 8.0, "Postprocess.Outputs": 1.0})
    timings = load_recorded_timings(times_file)
    assert timings["Dataset"] == 12.5
    assert timings["Dataset.Process"] == 4.0
    assert timings["Alignment"] == 8.0
    assert timings["Postprocess.Outputs"] == 1.0


def test_router_warns_when_primary_profile_is_missing():
    messages = []

    def capture_log(message, level="info"):
        messages.append((level, message))

    router = LLMRouter(
        llm_profiles={
            "local": {"backend": "local_hf", "model": "Qwen/Qwen2.5-7B-Instruct"},
        },
        llm_routing={
            "summary_profile": "missing_hosted_profile",
            "summary_fallback_profile": "local",
        },
        log=capture_log,
    )

    resolved = router.resolve_task("summary")

    assert resolved.backend == "local_hf"
    assert resolved.fallback_triggered is True
    assert resolved.fallback_reason == "missing_primary_profile"
    assert any(
        level == "warning" and "missing_hosted_profile" in message for level, message in messages
    )
