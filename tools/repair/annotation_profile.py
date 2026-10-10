"""Validate frozen repair request profiles against digest-bound public metadata."""

from exact.repair.annotation_controls import SCHEMA_VERSION
from tools.repair.batch import read, sha


def request_profile(manifest, name):
    controls = manifest.get("request_profiles", {}).get(name)
    if controls is None:
        return None
    required_fields = {
        "schema_version",
        "max_output_tokens",
        "reasoning",
        "endpoint",
        "catalog",
        "tokenizer",
    }
    optional_fields = {"max_input_tokens", "max_input_bytes", "input_amendment"}
    if set(controls) not in (required_fields, required_fields | optional_fields):
        raise ValueError("Unknown request profile fields")
    if controls["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unknown response schema")
    cap = controls["max_output_tokens"]
    if type(cap) is not int or not 1 <= cap <= 6000:
        raise ValueError("Invalid finite profile output cap")
    evidence = {}
    for key in ("endpoint", "catalog"):
        binding = controls[key]
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Request profile capability evidence changed")
        evidence[key] = read(binding["path"])
    profile = manifest["profiles"][name]
    provider = profile["provider"]
    if (
        len(provider.get("only", [])) != 1
        or provider.get("allow_fallbacks") is not False
        or provider.get("require_parameters") is not True
    ):
        raise ValueError("Request profile requires one fail-closed provider")
    endpoints = evidence["endpoint"]["data"]
    matches = [e for e in endpoints["endpoints"] if e["provider_name"] == provider["only"][0]]
    if endpoints["id"] != profile["model"] or len(matches) != 1:
        raise ValueError("Ambiguous or wrong pinned endpoint")
    endpoint = matches[0]
    required = {"response_format", "structured_outputs", "max_tokens", "temperature"}
    if controls["reasoning"] is not None:
        required.add("reasoning")
    if not required <= set(endpoint["supported_parameters"]) or endpoint["status"] != 0:
        raise ValueError("Pinned endpoint lacks required structured-output controls")
    if "input_amendment" in controls:
        from tools.repair.shared_release import bound

        amendment = bound(controls["input_amendment"])
        tokens, byte_cap = input_limits(controls)
        if (
            amendment.get("schema") != "exact-repair/annotation-input-amendment/v1"
            or amendment.get("authorized") is not True
            or amendment.get("costs_reset") is not False
            or amendment.get("truncate_evidence") is not False
            or amendment.get("model") != profile["model"]
            or amendment.get("provider") != provider["only"][0]
            or amendment.get("max_input_tokens") != tokens
            or amendment.get("max_input_bytes") != byte_cap
            or amendment.get("max_output_tokens") != cap
            or manifest["phase"] not in amendment.get("phases", [])
            or amendment.get("cost_ceiling_usd") != manifest["cost_ceiling_usd"]
            or amendment.get("request_limits") != manifest["request_limits"]
            or type(tokens) is not int
            or not 1 <= tokens
            or type(byte_cap) is not int
            or not 1 <= byte_cap
            or tokens + cap > endpoint.get("context_length", 0)
            or tokens > (endpoint.get("max_prompt_tokens") or endpoint["context_length"])
        ):
            raise ValueError("Invalid or unsupported annotation input amendment")
    if cap > endpoint["max_completion_tokens"]:
        raise ValueError("Output cap exceeds endpoint maximum")
    models = [m for m in evidence["catalog"]["data"] if m["id"] == profile["model"]]
    if len(models) != 1:
        raise ValueError("Missing exact model capability")
    reasoning = controls["reasoning"]
    support = models[0].get("reasoning") or {}
    if reasoning is None:
        if (
            "reasoning" in endpoint["supported_parameters"]
            or profile["model"] != "openai/gpt-4o-mini-2024-07-18"
        ):
            raise ValueError("Reasoning omission requires the qualified non-reasoning model")
    elif reasoning == {"enabled": False}:
        if support.get("mandatory") is not False:
            raise ValueError("Model does not explicitly support reasoning disable")
    elif reasoning == {"effort": "low"}:
        if "low" not in support.get("supported_efforts", []):
            raise ValueError("Model does not explicitly support low effort")
    else:
        raise ValueError("Unsupported reasoning setting")
    prices = manifest["prices_per_million"][name]
    for param, price in (("prompt", "input"), ("completion", "output")):
        if not 0 < float(endpoint["pricing"][param]) * 1e6 <= prices[price] + 1e-9:
            raise ValueError("Frozen endpoint exceeds price ceiling")
        if provider.get("max_price", {}).get(param) != prices[price]:
            raise ValueError("Provider max_price and reservation differ")
    tokenizer = controls["tokenizer"]
    if sha(tokenizer["path"]) != tokenizer["sha256"]:
        raise ValueError("Input tokenizer immutable bytes changed")
    if tokenizer.get("kind") == "openai_o200k_base":
        if (
            profile["model"] != "openai/gpt-4o-mini-2024-07-18"
            or tokenizer.get("encoding") != "o200k_base"
            or tokenizer.get("repository") != "openai/tiktoken"
            or sha(tokenizer["source"]["path"]) != tokenizer["source"]["sha256"]
            or tokenizer.get("ordinary_token_ids_equal") is not True
        ):
            raise ValueError("GPT-4o requires bound o200k_base conversion evidence")
    elif (
        tokenizer["repository"] != models[0].get("hugging_face_id")
        or len(tokenizer["revision"]) != 40
    ):
        raise ValueError("Input tokenizer must bind the declared model and immutable bytes")
    return controls


def input_limits(controls):
    """Legacy profiles keep their original bounds; amendments bind both guards."""
    if controls is None:
        return 8000, 8000
    return controls.get("max_input_tokens", 8000), controls.get("max_input_bytes", 32768)
