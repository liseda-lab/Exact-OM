"""Explicit isolated recovery attempts, retaining every previous generation cache."""

import json
import re
from collections import Counter
from pathlib import Path

from exact_inspect.artifacts import atomic_json
from exact_inspect.contracts import canonical_hash
from exact_inspect.generation import ExplanationJobs
from tools.review_generation_summary import generation_summary


def attempt_paths(root, attempt):
    if type(attempt) is not int or attempt < 1:
        raise ValueError("Generation attempt must be a positive integer")
    root = Path(root)
    suffix = "" if attempt == 1 else f"-attempt-{attempt}"
    return {
        "generation": root / ("generation" + suffix),
        "requests": root / ("review-requests" + suffix),
        "reviews": root / ("independent-reviews" + suffix),
        "summary": root / ("generation-summary" + suffix + ".json"),
        "status": root / ("generation-review-status" + suffix + ".json"),
        "scopes": root / ("prepared-scopes" + suffix),
        "assets": root if attempt == 1 else root / ("description-assets" + suffix),
        "dependencies": root / "package" / ("description-dependencies" + suffix),
        "explanations": root / "package" / ("explanations" + suffix),
        "partial_catalogue": root / ("catalogue.partial" + suffix + ".json"),
    }


def attempt_scope_id(case_id, attempt):
    """Keep legacy identities while separating every later immutable scope."""
    attempt_paths(Path("."), attempt)
    return case_id if attempt == 1 else f"attempt-{attempt}-{case_id}"


def bind_attempt(root, attempt, inputs):
    record = {
        "schema": "exact-review-generation-attempt/1",
        "generation_attempt": attempt,
        "preparation_inputs_sha256": canonical_hash(inputs),
        "generation_directory": attempt_paths(root, attempt)["generation"].name,
    }
    path = Path(root) / "generation-attempts" / f"attempt-{attempt}.json"
    if path.exists():
        if json.loads(path.read_bytes()) != record:
            raise ValueError("Generation attempt already belongs to another preparation")
    else:
        atomic_json(path, record)


def verified_saved_result(directory, *, expected_identity=None):
    """Verify a cache entry belongs to this request, not merely to some valid receipt."""
    directory = Path(directory)
    try:
        request = json.loads((directory / "request.json").read_bytes())
        result = json.loads((directory / "explanation.json").read_bytes())
        receipt = json.loads((directory / "receipt.json").read_bytes())
        identity = canonical_hash(request)
        manifest = result["manifest"]
        if (
            identity != "sha256:" + directory.name
            or (expected_identity is not None and identity != canonical_hash(expected_identity))
            or receipt["request_hash"] != identity
            or result["explanation_id"] != identity
            or canonical_hash(result) != receipt["output_hash"]
            or manifest["packet_hash"] != canonical_hash(request["packet"])
            or manifest["prompt_hash"] != canonical_hash(request["prompt"])
            or manifest["requested_model"] != request["profile"]["model"]
            or result["task"] != request["packet"]["task"]
            or result["entities"] != request["packet"]["entities"]
        ):
            raise ValueError("Generation cache request/result lineage differs")
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError("Generation cache request/result lineage failed verification") from error


def previous_attempt_summaries(root, attempt):
    """Only observed earlier attempt counts; no retries or ledger changes."""
    roots = []
    for path in Path(root).glob("generation*"):
        match = re.fullmatch(r"generation(?:-attempt-([1-9][0-9]*))?", path.name)
        if match and path.is_dir():
            number = int(match[1] or 1)
            if number < attempt:
                roots.append((number, path))
    summaries = []
    for number, path in sorted(roots):
        results = []
        for saved in sorted(path.glob("*/explanation.json")):
            result = verified_saved_result(saved.parent)
            results.append(result)
        summary = generation_summary(path, results, [], "provider")
        for key in (
            "served_artifacts",
            "independently_reviewed_artifacts",
            "factual_fallback_artifacts",
            "deterministic_artifacts",
        ):
            summary.pop(key)
        summary.update(
            generation_attempt=number,
            artifact_scope="Earlier immutable generation cache; not the current publication's descriptions.",
            cached_job_statuses=dict(Counter(r["manifest"]["status"] for r in results)),
        )
        summaries.append(summary)
    return summaries


def generation_contract(profile):
    from exact_inspect.descriptions import (
        COMPARISON_PROSE_PROMPT_V2,
        PROFILE_PROSE_PROMPT_V2,
    )
    from exact_inspect.generation import ExplanationOutput

    return {
        "schema": "exact-review-generation-contract/1",
        "profile": profile.model_dump(),
        "prompt_hashes": {
            "entity_profile": canonical_hash(PROFILE_PROSE_PROMPT_V2),
            "pair_comparison": canonical_hash(COMPARISON_PROSE_PROMPT_V2),
        },
        "output_schema_hash": canonical_hash(ExplanationOutput.model_json_schema()),
        "implementation": "grounding-prose-review/1",
        "max_semantic_repairs": 0,
        "max_http_retries": 0,
        "retry_unknown_delivery": False,
    }


def bind_generation_contract(root, attempt, profile):
    """A changed prompt/schema needs an explicit new attempt before any dispatch."""
    contract = generation_contract(profile)
    path = Path(root) / "generation-attempts" / f"attempt-{attempt}-contract.json"
    if path.exists():
        if canonical_hash(json.loads(path.read_bytes())) != canonical_hash(contract):
            raise ValueError("Generation contract changed; select an explicit new attempt")
        return
    # Adopt only compatible already-prepared jobs, including interrupted jobs.
    for request in attempt_paths(root, attempt)["generation"].glob("*/request.json"):
        value = json.loads(request.read_bytes())
        task = value["packet"]["task"]
        if (
            canonical_hash(value["profile"]) != canonical_hash(contract["profile"])
            or canonical_hash(value["prompt"]) != contract["prompt_hashes"].get(task)
            or canonical_hash(value["schema"]) != contract["output_schema_hash"]
            or value["implementation"] != contract["implementation"]
            or value["max_repairs"] != 0
        ):
            raise ValueError(
                "Saved generation uses another contract; select an explicit new attempt"
            )
    atomic_json(path, contract)


class ReviewExplanationJobs(ExplanationJobs):
    """Review recovery never implies another HTTP or unknown-delivery retry."""

    cache_only = False

    def generate(self, packet, profile, **kwargs):
        from exact_inspect.descriptions import (
            COMPARISON_PROSE_PROMPT_V2,
            PROFILE_PROSE_PROMPT_V2,
        )
        from exact_inspect.generation import ExplanationOutput

        if (
            kwargs.get("max_repairs", 0) != 0
            or kwargs.get("prompt") is not None
            or packet.selection.get("description_contract") != "exact-description/2"
            or packet.selection.get("description_prose") is not True
        ):
            raise ValueError(
                "Review generation requires its frozen prose contract with zero repairs"
            )
        kwargs["max_repairs"] = 0
        identity = {
            "packet": packet.model_dump(mode="json"),
            "profile": profile.model_dump(),
            "prompt": (
                PROFILE_PROSE_PROMPT_V2
                if packet.task == "entity_profile"
                else COMPARISON_PROSE_PROMPT_V2
            ),
            "schema": ExplanationOutput.model_json_schema(),
            "implementation": "grounding-prose-review/1",
            "max_repairs": 0,
        }
        key = canonical_hash(identity)
        directory = self.root / key[7:]
        request_path = directory / "request.json"
        if request_path.exists() and canonical_hash(json.loads(request_path.read_bytes())) != key:
            raise ValueError("Generation cache request/result lineage failed verification")
        if (directory / "explanation.json").exists():
            return verified_saved_result(directory, expected_identity=identity)
        if self.cache_only:
            raise ValueError(
                "Required generation result is absent; cache-only assembly cannot dispatch"
            )
        super().generate(packet, profile, **kwargs)
        return verified_saved_result(directory, expected_identity=identity)

    def _provider(self, messages, profile):
        import os

        from exact.llm.routing import LLMProfile, OpenRouterClient

        if os.getenv("EXACT_OPENROUTER_RETRY_UNKNOWN") == "1":
            raise ValueError("Review generation forbids implicit unknown-delivery retry")
        client = OpenRouterClient()
        client.ledger_dir = self.root / "provider-ledger"
        client.max_retries = 0
        client.retry_unknown_requests = False
        binding = LLMProfile(
            name=profile.name,
            backend="openrouter",
            model=profile.model,
            revision=profile.revision,
            provider=profile.routing,
            timeout_secs=profile.timeout_seconds,
        )
        try:
            return client.chat_completion(
                binding,
                messages,
                profile.max_tokens,
                temperature=profile.temperature,
                role="explanation_framework",
            )
        finally:
            client._client.close()
