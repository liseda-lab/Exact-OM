"""Explicit Luna prose revision over a retained, immutable development publication.

Generation uses a bounded durable ledger. Human/independent-agent claim decisions are
required separately; no generator result is published or automatically approved.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from exact.llm.routing import LLMProfile, OpenRouterClient
from exact_inspect.artifacts import atomic_json, validate_bundle
from exact_inspect.context import OntologyContext
from exact_inspect.contracts import canonical_hash, canonical_json, file_hash
from exact_inspect.description_review import build_review_request
from exact_inspect.descriptions import (
    COMPARISON_PROSE_PROMPT_V5,
    PROFILE_PROSE_PROMPT_V5,
)
from exact_inspect.generation import FactPacket, factual_fallback, prose_response_format
from exact_inspect.readable_descriptions import enrich
from tools.review_model_comparison import ADDENDUM, corrected_wire, validate_output

MODEL = "openai/gpt-6-luna"
VERSION = "exact-luna-prose-revision/1"
MAX_BYTES = 128 * 1024
MAX_OUTPUT = 4000
MAX_USD = 5.0
SAFETY = """When a definition is malformed, do not silently repair it or copy the broken
phrase into prose. Use an intact supported part or a readable hierarchy statement; otherwise
return no claims. A synonym alone is not a useful generated definition. Shared comparative
meaning must be supported on each side independently: do not transfer an adjective/modifier
from one anatomical feature to another. Preserve the existing differences in what is causal,
associated, a material basis, usual, possible, or merely recorded terminology."""


def read(path):
    return json.loads(Path(path).read_bytes())


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen revision artifact changed: {path.name}")
        return
    atomic_json(path, value)


def snapshot(path):
    s = Path(path).stat()
    return [
        s.st_dev,
        s.st_ino,
        s.st_nlink,
        s.st_mode,
        getattr(s, "st_flags", 0),
        s.st_size,
        s.st_mtime_ns,
        s.st_ctime_ns,
    ]


def wire(packet):
    prompt = (
        PROFILE_PROSE_PROMPT_V5 if packet.task == "entity_profile" else COMPARISON_PROSE_PROMPT_V5
    )
    return {
        "model": MODEL,
        "max_tokens": MAX_OUTPUT,
        "reasoning": {"effort": "medium", "exclude": True},
        "provider": {"only": ["OpenAI"], "allow_fallbacks": False, "require_parameters": True},
        "response_format": prose_response_format(),
        "messages": [
            {"role": "system", "content": prompt + "\n\n" + ADDENDUM + "\n\n" + SAFETY},
            {"role": "user", "content": canonical_json(corrected_wire(packet)).decode()},
        ],
    }


def contexts_for(base, bundle):
    indexed = {a.path: a.sha256 for a in bundle.artifacts}
    return {
        version: OntologyContext(
            base / "package" / rel, verified_database_hash=indexed[rel + "/context.sqlite"]
        )
        for version, rel in bundle.ontologies.items()
    }


def prepare(base, root):
    base, root = base.resolve(), root.resolve()
    if (root / "plan.json").exists():
        raise ValueError("Use the frozen plan; a changed plan requires a new revision root")
    bundle = validate_bundle(base / "package/package.json")
    contexts = contexts_for(base, bundle)
    originals = {}
    jobs = []
    for old, rel in sorted(bundle.explanations.items()):
        original = read(base / "package" / rel)
        packet = FactPacket.model_validate(original["support_packet"])
        packet = packet.model_copy(
            update={"selection": {**packet.selection, "description_prose": True}}
        )
        packet = enrich(packet, contexts, bundle.policy)
        body = wire(packet)
        if len(canonical_json(body)) > MAX_BYTES:
            raise ValueError("Request exceeds the frozen bound; do not truncate admitted facts")
        identity = {"schema": VERSION, "packet": packet.model_dump(mode="json"), "payload": body}
        job = canonical_hash(identity)
        originals[old] = {
            "path": rel,
            "sha256": file_hash(base / "package" / rel),
            "stat": snapshot(base / "package" / rel),
            "job": job,
            "support_status": original["support_status"],
        }
        jobs.append({"job_id": job, "old_id": old, "identity": identity})
    if len({x["job_id"] for x in jobs}) != len(jobs):
        raise ValueError("Duplicate source jobs require explicit deduplication before dispatch")
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as response:
        model = next(x for x in json.load(response)["data"] if x["id"] == MODEL)
    pricing = model["pricing"]
    input_price = max(
        float(pricing.get(k, 0)) for k in ("prompt", "input_cache_read", "input_cache_write")
    )
    cost = sum(
        (len(canonical_json(x["identity"]["payload"])) + 256) * input_price
        + MAX_OUTPUT * float(pricing["completion"])
        for x in jobs
    )
    if cost > MAX_USD or len(jobs) > 260:
        raise ValueError("Bounded publication scope or provider budget exceeded")
    plan = {
        "schema": VERSION,
        "base": str(base),
        "root": str(root),
        "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner_sha256": file_hash(Path(__file__)),
        "publication_sha256": file_hash(base / "publication.json"),
        "package_sha256": file_hash(base / "package/package.json"),
        "composition_sha256": file_hash(base / "composition.json"),
        "base_file_stats": {
            str(base / "package" / a.path): snapshot(base / "package" / a.path)
            for a in bundle.artifacts
        },
        "source_artifacts": originals,
        "jobs": jobs,
        "model": model,
        "max_calls": len(jobs),
        "max_output": MAX_OUTPUT,
        "max_bytes": MAX_BYTES,
        "budget_usd": MAX_USD,
        "estimated_max_usd": cost,
        "scope": "All 140 entity profiles and 120 presented pair comparisons; predecessor ontology/matcher/ELK data are immutable. Existing frozen profiles remain the score-blind comparison input dependencies.",
        "authorization": "User requested preparing the bundle again using Luna; generation and local reviewed revision only.",
        "review_required": True,
        "retries": 0,
        "repairs": 0,
    }
    freeze(root / "plan.json", plan)
    print(
        json.dumps(
            {
                "jobs": len(jobs),
                "profiles": sum(x["identity"]["packet"]["task"] == "entity_profile" for x in jobs),
                "comparisons": sum(
                    x["identity"]["packet"]["task"] == "pair_comparison" for x in jobs
                ),
                "plan_sha256": canonical_hash(plan),
                "estimated_max_usd": cost,
            }
        ),
        flush=True,
    )


def source_result(job, response):
    packet = FactPacket.model_validate(job["identity"]["packet"])
    body = job["identity"]["payload"]
    output = factual_fallback(packet)
    reason = "Provider draft awaits a separately recorded independent semantic review"
    return {
        "artifact_type": "generated_explanation",
        "contract_version": "exact-explain/1.0",
        "explanation_id": job["job_id"],
        "task": packet.task,
        "entities": [e.model_dump() for e in packet.entities],
        "claims": [
            c.model_copy(
                update={
                    "claim_id": canonical_hash(
                        {"generation": job["job_id"], **c.model_dump(exclude={"claim_id"})}
                    )
                }
            ).model_dump()
            for c in output.claims
        ],
        "grounding_status": "validated",
        "limitations": list(dict.fromkeys(packet.missingness + [reason])),
        "fixture_provenance": "prepared from bound ontology facts; original-excerpt fallback",
        "preparation_kind": "factual_fallback",
        "support_status": "fallback",
        "support_reasons": [reason],
        "support_packet": packet.model_dump(mode="json"),
        "manifest": {
            "description_contract": "exact-description/2",
            "ontology_context_hashes": packet.context_hashes,
            "visibility_policy_hash": packet.policy_hash,
            "packet_hash": canonical_hash(packet),
            "prompt_hash": canonical_hash(body["messages"][0]["content"]),
            "output_schema": "exact-explain-generation/1",
            "requested_model": MODEL,
            "returned_model": response.get("model"),
            "provider": response.get("provider"),
            "parameters_hash": canonical_hash({k: v for k, v in body.items() if k != "messages"}),
            "language": packet.language,
            "response_hash": canonical_hash(response),
            "status": "failed",
            "category_alias_normalization": {},
        },
    }


def verify_base(plan):
    base = Path(plan["base"])
    for path, key in [
        ("publication.json", "publication_sha256"),
        ("package/package.json", "package_sha256"),
        ("composition.json", "composition_sha256"),
    ]:
        if file_hash(base / path) != plan[key]:
            raise ValueError("Frozen predecessor changed")
    for path, before in plan["base_file_stats"].items():
        if snapshot(path) != before:
            raise ValueError("Frozen predecessor inode, protection, size or timestamps changed")


def generate(root, key_path, workers=4):
    plan = read(root / "plan.json")
    if file_hash(Path(__file__)) != plan["runner_sha256"]:
        raise ValueError("Generation runner differs from frozen plan")
    if os.getenv("EXACT_OPENROUTER_RETRY_UNKNOWN") == "1":
        raise ValueError("Unknown-delivery retries are forbidden")
    verify_base(plan)
    os.environ["EXACT_OPENROUTER_REQUEST_CAP"] = str(plan["max_calls"])
    os.environ["EXACT_OPENROUTER_TOKEN_CAP"] = str(
        plan["max_calls"] * (MAX_BYTES + MAX_OUTPUT + 1024)
    )

    def call(job):
        folder = root / "generation" / job["job_id"][7:]
        identity = job["identity"]
        freeze(folder / "request.json", identity)
        response_path = folder / "response.json"
        if response_path.exists():
            saved = read(response_path)
            if canonical_hash(saved["response"]) != saved["sha256"]:
                raise ValueError("Saved provider response changed")
            response = saved["response"]
        else:
            client = OpenRouterClient()
            client.ledger_dir = root / "provider-ledger"
            client.max_retries = 0
            client.retry_unknown_requests = False
            profile = LLMProfile(
                name="luna-review-bundle",
                backend="openrouter",
                model=MODEL,
                api_key_path=str(key_path),
                timeout_secs=90,
            )
            try:
                response = client._generation(
                    profile,
                    identity["payload"],
                    "chat/completions",
                    role="luna_review_bundle",
                    requested_seed=None,
                )
            finally:
                client.close()
            if (
                response.get("model") not in (MODEL, MODEL.removeprefix("openai/"))
                or response.get("provider") != "OpenAI"
            ):
                raise ValueError("Unexpected model/provider; retained ledger requires inspection")
            freeze(response_path, {"response": response, "sha256": canonical_hash(response)})
        value, errors = validate_output(response, FactPacket.model_validate(identity["packet"]))
        if errors:
            freeze(
                folder / "unreviewable.json",
                {"contract_errors": errors, "response_hash": canonical_hash(response)},
            )
            return job["old_id"], "unreviewable"
        result = source_result(job, response)
        freeze(folder / "source.json", result)
        if not value["claims"]:
            freeze(folder / "abstained.json", {"response_hash": canonical_hash(response)})
            return job["old_id"], "abstained"
        request = build_review_request(result, response)
        freeze(root / "review-requests" / (job["job_id"][7:] + ".json"), request)
        return job["old_id"], "awaiting_independent_review"

    # One bounded storage-guarded batch; the existing ledger serializes its writes.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(call, job): job for job in plan["jobs"]}
        try:
            for ordinal, future in enumerate(concurrent.futures.as_completed(pending), 1):
                old, status = future.result()
                print(
                    json.dumps(
                        {
                            "completed": ordinal,
                            "total": len(pending),
                            "old_id": old,
                            "status": status,
                        }
                    ),
                    flush=True,
                )
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    verify_base(plan)
    freeze(
        root / "generation-complete.json",
        {
            "plan_sha256": canonical_hash(plan),
            "jobs": len(plan["jobs"]),
            "independent_review_required": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "generate"])
    parser.add_argument("--base", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key-path", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.base, args.root)
    else:
        generate(args.root, args.key_path)


if __name__ == "__main__":
    main()
