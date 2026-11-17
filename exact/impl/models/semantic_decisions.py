"""Opt-in binary Decisions routing; the chat judge remains the default."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from exact.llm.decisions import DecisionsClient, EQUIVALENCE_INSTRUCTIONS
from exact.llm.prompt_budget import DECISIONS_ENDPOINT, DECISIONS_MODEL, DECISIONS_PROVIDER
from exact.llm.routing import LLMProfile


def validate_profile(scorer: Any) -> LLMProfile:
    """Reject unsupported configuration before loading encoders or making paid calls."""
    router = scorer._llm_router
    profile = router.profiles.get(router.routing.profile_for_task("decision"))
    if (
        not isinstance(profile, LLMProfile)
        or profile.backend != "openrouter"
        or profile.model != DECISIONS_MODEL
        or profile.revision != "openrouter_decisions_noul_v1"
        or profile.api_base.rstrip("/") != DECISIONS_ENDPOINT.rsplit("/", 1)[0]
        or profile.provider != DECISIONS_PROVIDER
        or not profile.tokenizer
        or re.fullmatch(r"[0-9a-fA-F]{40}", profile.tokenizer_revision or "") is None
    ):
        raise ValueError("OpenRouter Decisions requires its exact pinned decision profile")
    if not callable(getattr(scorer, "_hosted_decisions_evidence", None)):
        raise ValueError("OpenRouter Decisions requires the pair-adaptive evidence renderer")
    if getattr(scorer, "generate_llm_rationales", False):
        raise ValueError("OpenRouter Decisions requires rationale generation to be disabled")
    config = getattr(scorer, "llm_experiment_config", {}).get("decision", {})
    if config.get("mode", "binary") != "binary":
        raise ValueError("OpenRouter Decisions supports binary pair judgments only")
    return profile


def score_pairs(
    scorer: Any,
    src_labels: list[str],
    tgt_labels: list[str],
    briefs: list[str],
    tgt_summaries: list[str],
    *,
    observations: list[dict[str, Any]] | None,
    prompt_suffixes: list[str] | None,
) -> torch.Tensor:
    profile = validate_profile(scorer)
    if any(tgt_summaries):
        raise ValueError("OpenRouter Decisions expects a pair brief, not separate target summaries")
    directory = scorer._llm_router.hosted.ledger_dir or os.getenv("EXACT_OPENROUTER_LEDGER_DIR")
    if not directory:
        raise ValueError("OpenRouter Decisions requires the shared hosted request ledger")
    cache_only = os.getenv("EXACT_HOSTED_CACHE_ONLY") == "1"
    api_key = None if cache_only else scorer._llm_router.hosted.resolve_api_key(profile)
    inputs = [
        scorer._hosted_decisions_evidence(source, target, brief) + suffix
        for source, target, brief, suffix in zip(
            src_labels, tgt_labels, briefs, prompt_suffixes or [""] * len(src_labels)
        )
    ]
    scorer._last_decision_backend_meta = {
        "backend": "openrouter",
        "profile": profile.name,
        "model": profile.model,
        "revision": profile.revision,
        "decision_capable": True,
        "fallback_triggered": False,
        "fallback_reason": None,
        "endpoint": "decisions",
        "provider": "OpenAI",
        "decision_probe_passed": None,
        "decision_probe_error": None,
        "decision_scoring_mode": "openrouter_decisions_noul_v1",
        "seed_supported": False,
    }
    workers = max(1, min(int(scorer.llm_decision_batch_size or len(inputs)), len(inputs)))
    client = DecisionsClient(
        Path(directory),
        tokenizer=profile.tokenizer or "",
        tokenizer_revision=profile.tokenizer_revision or "",
        api_key=api_key,
        api_key_path=Path(profile.api_key_path) if profile.api_key_path else None,
        timeout_secs=profile.timeout_secs,
        cache_only=cache_only,
    )
    scorer._record_decision_stat("hosted_attempted", len(inputs))
    try:
        def predict(evidence: str) -> dict[str, Any]:
            return client.predict(evidence, EQUIVALENCE_INSTRUCTIONS)

        if workers == 1:
            replies = [predict(evidence) for evidence in inputs]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                replies = list(executor.map(predict, inputs))
        if any(reply["refused"] for reply in replies):
            raise RuntimeError("OpenRouter Decisions refused a judgment; no score or fallback applied")
    except Exception:
        scorer._record_decision_stat("scoring_failures", len(inputs))
        raise
    finally:
        client.close()
    probabilities = [reply["probability"] for reply in replies]
    if observations is not None:
        observations.extend(replies)
    scorer._record_decision_stat("hosted_scored", len(replies))
    return torch.tensor(probabilities, dtype=torch.float32, device=scorer.device)
