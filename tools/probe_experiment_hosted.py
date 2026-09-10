#!/usr/bin/env python3
"""Validate the pinned hosted decision profile and relocated cache with two requests."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

from exact.core.entities.configs.yaml_io import load_yaml_mapping
from exact.llm.ledger import RequestLedger
from exact.llm.routing import (
    LLMProfile,
    OpenRouterClient,
    extract_chat_text,
    extract_first_token_top_logprobs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        EXACT_EXPERIMENT_MODE="1",
        EXACT_OPENROUTER_REQUEST_CAP="4",
        EXACT_OPENROUTER_TOKEN_CAP="20000",
        TOKENIZERS_PARALLELISM="false",
    )
    config = load_yaml_mapping(args.base_config)
    name = config["llm"]["routing"]["decision_profile"]
    profile = LLMProfile.from_raw(name, config["llm"]["profiles"][name])
    if args.api_key_file is not None:
        os.environ[profile.api_key_env] = args.api_key_file.read_text().strip()
    client = OpenRouterClient()
    client.max_retries = 0
    client.ledger_dir = args.output / "ledger"
    started = time.monotonic()
    report = {"status": "running", "profile": name, "checks": []}
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            profile.tokenizer, revision=profile.tokenizer_revision, local_files_only=True
        )
        metadata = client.model_capabilities(profile)
        (args.output / "model-capabilities.json").write_text(json.dumps(metadata, indent=2))
        for labels in (("A", "B"), ("A", "B", "Z")):
            bias = {}
            for label in labels:
                for text in (label, " " + label):
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    if len(ids) == 1:
                        bias[str(ids[0])] = 20.0
            options = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Return one token from "
                        + ", ".join(labels)
                        + ". Every listed token is equally acceptable; choose any one.",
                    }
                ],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 20,
                "logit_bias": bias,
                "seed": 17,
                "role": "decision",
            }
            response = client.chat_completion(profile, **options)
            observed = {
                token.strip()
                for token, score in extract_first_token_top_logprobs(response)
                if math.isfinite(score)
            }
            if set(labels) - observed or extract_chat_text(response).strip() not in labels:
                raise ValueError(
                    f"Missing usable categorical decision output/logprobs for {labels}"
                )
            before = RequestLedger(client.ledger_dir).summary()
            replay = client.chat_completion(profile, **options)
            if response != replay or RequestLedger(client.ledger_dir).summary() != before:
                raise ValueError("Completed response replay changed data or incurred a request")
            report["checks"].append(
                {
                    "labels": labels,
                    "output": extract_chat_text(response),
                    "categorical_logprobs": True,
                    "cache_replay": True,
                }
            )
        relocated = args.output / "relocated-ledger"
        shutil.copytree(client.ledger_dir, relocated)
        client.ledger_dir = relocated
        client.resolve_api_key = lambda _: None
        if client.chat_completion(profile, **options) != response:
            raise ValueError("Relocated response differs")
        report.update(status="complete", relocated_cache_without_credentials=True)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, message=str(exc))
    finally:
        report["seconds"] = time.monotonic() - started
        report["ledger"] = RequestLedger(args.output / "ledger").summary()
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        client.close()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
