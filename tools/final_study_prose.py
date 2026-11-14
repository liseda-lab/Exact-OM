"""Bounded Luna drafts for final-study facts; independent admission is a separate step."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import urllib.request
from pathlib import Path

from exact.llm.routing import LLMProfile, OpenRouterClient
from exact_inspect.contracts import canonical_hash, canonical_json, file_hash
from exact_inspect.description_review import (
    build_review_request,
    validate_reviewed_artifact,
)
from exact_inspect.generation import FactPacket
from tools.review_luna_bundle import (
    MAX_BYTES,
    MAX_OUTPUT,
    MODEL,
    freeze,
    read,
    source_result,
    wire,
)
from tools.review_model_comparison import validate_output

PROFILE_REVISION_GUIDANCE = """Write a useful, concise description rather than only saying that
X is a type of its parent. Preserve distinguishing information explicitly supported by
the supplied original facts: site, cell type, mechanism, gene, inheritance, onset, or
characteristic manifestations. Use two or three short sentences when needed. Do not
omit these details merely to make one sentence. If the packet only supports a parent
classification, do not invent a distinguishing definition: return no claims. Never
borrow details from a candidate, a reference mapping, or external knowledge. Copy
every cited fact identifier exactly from the input; do not retype or shorten it."""


def prepare(inputs, root, reuse_reviewed=None, regenerate_ids=None):
    values = read(inputs)["results"]
    selected = set(read(regenerate_ids)) if regenerate_ids else None
    if selected is not None and (
        not reuse_reviewed
        or not selected
        or not selected <= values.keys()
        or any(values[key]["task"] != "entity_profile" for key in selected)
    ):
        raise ValueError("Targeted regeneration requires existing inputs and entity profiles")
    reused = {}
    if reuse_reviewed:
        previous = read(reuse_reviewed)["results"]
        for old_id, original in values.items():
            saved = previous.get(old_id)
            if (
                saved
                and (selected is None or old_id not in selected)
                and saved["support_packet"] == original["support_packet"]
            ):
                validate_reviewed_artifact(saved)
                reused[old_id] = saved
        freeze(
            root / "reused-results.json",
            {"source_sha256": file_hash(reuse_reviewed), "results": reused},
        )
    jobs = []
    for old_id, original in sorted(values.items()):
        if old_id in reused:
            continue
        if selected is not None and old_id not in selected:
            continue
        packet = FactPacket.model_validate(original["support_packet"])
        body = wire(packet)
        if selected is not None:
            body["messages"][0]["content"] += "\n\n" + PROFILE_REVISION_GUIDANCE
        if len(canonical_json(body)) > MAX_BYTES:
            raise ValueError("Provider packet exceeds the existing bound")
        identity = {
            "schema": "exact-final-prose/1",
            "packet": packet.model_dump(mode="json"),
            "payload": body,
        }
        jobs.append({"job_id": canonical_hash(identity), "old_id": old_id, "identity": identity})
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as response:
        model = next(m for m in json.load(response)["data"] if m["id"] == MODEL)
    pricing = model["pricing"]
    input_price = max(
        float(pricing.get(k, 0)) for k in ("prompt", "input_cache_read", "input_cache_write")
    )
    cost = sum(
        (len(canonical_json(j["identity"]["payload"])) + 256) * input_price
        + MAX_OUTPUT * float(pricing["completion"])
        for j in jobs
    )
    if len(jobs) > 424 or cost > 5:
        raise ValueError(
            f"Final-study bound exceeded: {len(jobs)} requests, ${cost:.2f} conservative maximum"
        )
    plan = {
        "schema": "exact-final-prose-plan/1",
        "inputs_sha256": file_hash(inputs),
        "runner_sha256": file_hash(__file__),
        "jobs": jobs,
        "max_calls": len(jobs),
        "budget_usd": 5,
        "estimated_max_usd": cost,
        "model": model,
        "retries": 0,
        "review_required": True,
        "regenerate_ids_sha256": file_hash(regenerate_ids) if regenerate_ids else None,
    }
    freeze(root / "plan.json", plan)
    print(json.dumps({"jobs": len(jobs), "reused": len(reused), "maximum_usd": cost}), flush=True)


def generate(root, key_path, workers):
    plan = read(root / "plan.json")
    if plan["runner_sha256"] != file_hash(__file__):
        raise ValueError("Runner changed after planning")
    if os.getenv("EXACT_OPENROUTER_RETRY_UNKNOWN") == "1":
        raise ValueError("Unknown-delivery retries are forbidden")
    os.environ["EXACT_OPENROUTER_REQUEST_CAP"] = str(plan["max_calls"])
    os.environ["EXACT_OPENROUTER_TOKEN_CAP"] = str(
        plan["max_calls"] * (MAX_BYTES + MAX_OUTPUT + 1024)
    )

    def call(job):
        folder = root / "generation" / job["job_id"][7:]
        freeze(folder / "request.json", job["identity"])
        response_path = folder / "response.json"
        if response_path.exists():
            saved = read(response_path)
            if canonical_hash(saved["response"]) != saved["sha256"]:
                raise ValueError("Provider response changed")
            response = saved["response"]
        else:
            client = OpenRouterClient()
            client.ledger_dir = root / "provider-ledger"
            client.max_retries = 0
            client.retry_unknown_requests = False
            profile = LLMProfile(
                name="final-study-luna",
                backend="openrouter",
                model=MODEL,
                api_key_path=str(key_path),
                timeout_secs=120,
            )
            try:
                response = client._generation(
                    profile,
                    job["identity"]["payload"],
                    "chat/completions",
                    role="final_study_prose",
                    requested_seed=None,
                )
            finally:
                client.close()
            freeze(response_path, {"response": response, "sha256": canonical_hash(response)})
        if (
            response.get("model") not in (MODEL, MODEL.removeprefix("openai/"))
            or response.get("provider") != "OpenAI"
        ):
            raise ValueError("Unexpected model/provider; inspect retained response")
        value, errors = validate_output(
            response, FactPacket.model_validate(job["identity"]["packet"])
        )
        if errors:
            freeze(folder / "unreviewable.json", {"errors": errors})
            return "unreviewable"
        result = source_result(job, response)
        freeze(folder / "source.json", result)
        if not value["claims"]:
            return "abstained"
        freeze(
            root / "review-requests" / (job["job_id"][7:] + ".json"),
            build_review_request(result, response),
        )
        return "awaiting_independent_review"

    outcomes = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(call, j): j for j in plan["jobs"]}
        for future in concurrent.futures.as_completed(pending):
            job = pending[future]
            try:
                status = future.result()
            except Exception as exc:
                # Exceptions can include transport detail; do not print credentials,
                # request bodies, or arbitrary provider text to the public log.
                status = "failed:" + type(exc).__name__
            outcomes[job["job_id"]] = status
            print(
                json.dumps({"completed": len(outcomes), "total": len(pending), "status": status}),
                flush=True,
            )
    freeze(root / "outcomes.json", outcomes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "generate", "combine"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--key-path", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--reuse-reviewed", type=Path)
    parser.add_argument(
        "--regenerate-ids",
        type=Path,
        help="Explicit entity-profile IDs for a targeted quality revision",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.inputs, args.root, args.reuse_reviewed, args.regenerate_ids)
    elif args.command == "generate":
        generate(args.root, args.key_path, args.workers)
    else:
        combined = {}
        sources = {}
        for name in ("reused-results.json", "reviewed-results.json"):
            path = args.root / name
            saved = read(path)
            sources[name] = file_hash(path)
            for old_id, value in saved["results"].items():
                if old_id in combined:
                    raise ValueError("New and reused reviewed outputs overlap")
                validate_reviewed_artifact(value)
                combined[old_id] = value
        freeze(
            args.root / "combined-reviewed-results.json", {"sources": sources, "results": combined}
        )
        print(json.dumps({"combined_reviewed_results": len(combined)}), flush=True)


if __name__ == "__main__":
    main()
