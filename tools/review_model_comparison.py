"""Bounded, offline-only paired description diagnostic; never publishes prose.

Uses the production durable provider ledger. Inputs are frozen corrected packets;
provider identity, resource caps and review blinding are additional study guards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import time
import urllib.request
from pathlib import Path

from exact.llm.routing import LLMProfile, OpenRouterClient
from exact_inspect.contracts import canonical_hash
from exact_inspect.descriptions import (
    COMPARISON_PROSE_PROMPT_V5,
    PROFILE_PROSE_PROMPT_V5,
)
from exact_inspect.generation import (
    ExplanationOutput,
    FactPacket,
    prose_provider_packet,
    prose_response_format,
)
from exact_inspect.readable_descriptions import label_facts

MODELS = ("openai/gpt-4o-mini-2024-07-18", "openai/gpt-6-luna")
MAX_CALLS = 24
MAX_BYTES = 65536
MAX_OUTPUT = 4000
MAX_USD = 0.50
RULE = "Among attempt-7 artifacts with zero approved claims, sort by original ID; select the first four profiles without definitions, first four profiles with definitions, and all four comparisons. No selection based on new outputs."
ADDENDUM = """Additional corrected-input contract: referenced_labels supplies admitted original
labels for exact typed endpoints of supplied axioms. Cite both the relation axiom and its
endpoint label when using a readable parent/property name. The direction is in the axiom AST,
not its subject index. When there is no definition, a readable asserted subclass or restriction
may provide a useful necessary-condition description; it is not an equivalent definition.
Do not repair malformed definitions, turn association/material basis into causation, drop
qualifiers, or infer broader scope merely because one annotation lists fewer features.
At most two claims and 160 words total. An empty claims list is allowed; no filler."""


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"Refusing to replace frozen study artifact: {path}")
        return
    with path.open("x") as handle:
        handle.write(encoded)


def corrected_wire(packet):
    value = prose_provider_packet(packet)
    value["referenced_labels"] = [
        {k: f[k] for k in ("fact_id", "subject", "category", "value")} for f in label_facts(packet)
    ]
    return value


def payload(case, model):
    value = {
        "model": model,
        "messages": case["messages"],
        "max_tokens": MAX_OUTPUT,
        "provider": {"only": ["OpenAI"], "allow_fallbacks": False, "require_parameters": True},
        "response_format": prose_response_format(),
    }
    if model == MODELS[0]:
        value["temperature"] = 0
    else:
        value["reasoning"] = {"effort": "medium", "exclude": True}
    if len(json.dumps(value, ensure_ascii=False).encode()) > MAX_BYTES:
        raise ValueError("Wire input exceeds the frozen byte cap")
    return value


def validate_output(raw, packet):
    errors = []
    try:
        if raw["choices"][0]["finish_reason"] != "stop":
            errors.append("non_stop_finish")
        value = json.loads(raw["choices"][0]["message"]["content"])
        if set(value) != {"claims", "limitations", "relation"}:
            errors.append("wrong_top_level_fields")
        out = ExplanationOutput.model_validate(value)
        if len(out.claims) > 2 or out.relation is not None:
            errors.append("claim_or_relation_contract")
        if any(x not in packet.missingness for x in out.limitations):
            errors.append("invented_missingness")
        allowed = {f["fact_id"] for f in [*packet.facts, *label_facts(packet)]}
        for claim in value["claims"]:
            if set(claim) != {"text", "fact_ids", "category"}:
                errors.append("wrong_claim_fields")
            if not claim["fact_ids"] or not set(claim["fact_ids"]) <= allowed:
                errors.append("invalid_citation_membership")
        return value, sorted(set(errors))
    except (ValueError, KeyError, IndexError, TypeError):
        return None, ["invalid_json_or_schema"]


def prepare(root, historical, combined):
    if (root / "plan.json").exists():
        raise ValueError("Plan already frozen; use run for exact durable replay")
    composition = read(combined / "composition.json")
    package_path = combined / "package/package.json"
    package = read(package_path)
    bypath = {x["path"]: x for x in package["artifacts"]}
    groups = {"no_definition": [], "definition": [], "comparison": []}
    for path in sorted((historical / "independent-reviews-attempt-7").glob("*.json")):
        review = read(path)
        if any(d["decision"] == "approved" for d in review["decisions"]):
            continue
        request = read(historical / "review-requests-attempt-7" / path.name)
        packet = request["packet"]
        group = (
            "comparison"
            if packet["task"] == "pair_comparison"
            else (
                "definition"
                if any(f["category"] == "definitions" for f in packet["facts"])
                else "no_definition"
            )
        )
        groups[group].append((path, request))
    assert {k: len(v) for k, v in groups.items()} == {
        "no_definition": 12,
        "definition": 6,
        "comparison": 4,
    }
    cases = []
    for group, entries in groups.items():
        for review_path, request in entries[:4]:
            old = "sha256:" + review_path.stem
            replacement = composition["description_replacements"][old]["replacement"]
            relpath = package["explanations"][replacement]
            path = combined / "package" / relpath
            assert digest(path) == bypath[relpath]["sha256"]
            artifact = read(path)
            packet = FactPacket.model_validate(artifact["support_packet"])
            assert (
                canonical_hash(packet.model_dump(mode="json"))
                == artifact["manifest"]["packet_hash"]
            )
            assert packet.entities == FactPacket.model_validate(request["packet"]).entities
            assert packet.facts == request["packet"]["facts"]
            wire = corrected_wire(packet)
            prompt = (
                (
                    PROFILE_PROSE_PROMPT_V5
                    if packet.task == "entity_profile"
                    else COMPARISON_PROSE_PROMPT_V5
                )
                + "\n\n"
                + ADDENDUM
            )
            case = {
                "case_id": f"case-{len(cases)+1:02d}",
                "stratum": group,
                "original_id": old,
                "original_review_sha256": digest(review_path),
                "corrected_id": replacement,
                "source_path": str(path),
                "source_sha256": digest(path),
                "packet": packet.model_dump(mode="json"),
                "packet_hash": canonical_hash(packet.model_dump(mode="json")),
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(wire, ensure_ascii=False, sort_keys=True),
                    },
                ],
            }
            case["messages_hash"] = canonical_hash(case["messages"])
            cases.append(case)
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as response:
        catalogue = {x["id"]: x for x in json.load(response)["data"] if x["id"] in MODELS}
    assert set(catalogue) == set(MODELS)
    estimate = 0.0
    for case in cases:
        for model in MODELS:
            body = payload(case, model)
            price = catalogue[model]["pricing"]
            estimate += (len(json.dumps(body, ensure_ascii=False).encode()) + 256) * float(
                price["prompt"]
            ) + MAX_OUTPUT * float(price["completion"])
    assert estimate < MAX_USD
    plan = {
        "schema": "exact-description-model-diagnostic/1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "runner_sha256": digest(__file__),
        "selection_rule": RULE,
        "eligible_counts": {k: len(v) for k, v in groups.items()},
        "models": list(MODELS),
        "catalogue": catalogue,
        "cases": cases,
        "combined_package_path": str(package_path),
        "combined_package_sha256": digest(package_path),
        "publication_sha256": composition["publication_sha256"],
        "max_calls": MAX_CALLS,
        "max_output_tokens_per_call": MAX_OUTPUT,
        "max_wire_bytes_per_call": MAX_BYTES,
        "worst_case_catalogue_usd": estimate,
        "budget_usd": MAX_USD,
        "replicates": 1,
        "retries": 0,
        "semantic_repairs": 0,
        "primary_metric": "Whole usable response: all claims grounded and useful; at least one claim; missing documentation is not disease scope.",
        "secondary_metrics": [
            "grounded claims",
            "useful grounded claims",
            "safe abstentions",
            "schema/citation errors",
            "qualifier/scope/causation errors",
            "cost",
            "latency",
        ],
        "review": "Root Codex agent, independent of the two API generators, reviews per-case randomized A/B outputs without reading model assignment until signed review is frozen. Single AI reviewer; no clinical validation.",
        "limitations": [
            "Purposive hard-case sample, not representative",
            "One sample per model/case",
            "Same corrected inputs/prompts; not comparable causally to the old uncorrected batch",
            "Luna medium reasoning and default sampling; mini temperature zero",
            "No publication or automatic approval",
        ],
    }
    save(root / "plan.json", plan)
    mapping = {}
    for case in cases:
        models = list(MODELS)
        secrets.SystemRandom().shuffle(models)
        mapping[case["case_id"]] = dict(zip(("A", "B"), models))
    save(root / "assignment.json", mapping)
    print(
        json.dumps(
            {
                "prepared_cases": len(cases),
                "plan_hash": canonical_hash(plan),
                "maximum_cost_usd": estimate,
            }
        ),
        flush=True,
    )


def run(root, key_path):
    plan = read(root / "plan.json")
    assert digest(__file__) == plan["runner_sha256"], "Frozen runner changed"
    assert digest(plan["combined_package_path"]) == plan["combined_package_sha256"]
    for case in plan["cases"]:
        assert digest(case["source_path"]) == case["source_sha256"]
    os.environ["EXACT_OPENROUTER_REQUEST_CAP"] = str(MAX_CALLS)
    os.environ["EXACT_OPENROUTER_TOKEN_CAP"] = str(MAX_CALLS * (MAX_BYTES + 512 + MAX_OUTPUT))
    if os.getenv("EXACT_OPENROUTER_RETRY_UNKNOWN") == "1":
        raise ValueError("Unknown-delivery retry is prohibited")
    assignment = read(root / "assignment.json")
    client = OpenRouterClient()
    client.ledger_dir = root / "provider-ledger"
    client.max_retries = 0
    client.retry_unknown_requests = False
    try:
        for case in plan["cases"]:
            for slot, model in assignment[case["case_id"]].items():
                destination = root / "responses" / f'{case["case_id"]}-{slot}.json'
                if destination.exists():
                    continue
                profile = LLMProfile(
                    name="description-model-diagnostic",
                    backend="openrouter",
                    model=model,
                    api_key_path=str(key_path),
                    timeout_secs=90,
                )
                raw = client._generation(
                    profile,
                    payload(case, model),
                    "chat/completions",
                    role="description_model_diagnostic",
                    requested_seed=None,
                )
                # The existing ledger saves raw bytes before parsing; an unexpected identity
                # stops the study and remains evidence rather than silently changing models.
                if raw.get("model") != model.removeprefix("openai/") and raw.get("model") != model:
                    raise ValueError("Unexpected returned model identity; inspect retained ledger")
                if raw.get("provider") != "OpenAI":
                    raise ValueError("Unexpected provider identity; inspect retained ledger")
                save(destination, raw)
                value, errors = validate_output(raw, FactPacket.model_validate(case["packet"]))
                save(
                    root / "blind" / f'{case["case_id"]}-{slot}.json',
                    {
                        "case_id": case["case_id"],
                        "slot": slot,
                        "output": value,
                        "contract_errors": errors,
                        "output_hash": canonical_hash(value),
                    },
                )
                print(
                    json.dumps(
                        {
                            "case": case["case_id"],
                            "slot": slot,
                            "saved": True,
                            "contract_errors": errors,
                        }
                    ),
                    flush=True,
                )
    finally:
        client.close()
    assert digest(plan["combined_package_path"]) == plan["combined_package_sha256"]
    for case in plan["cases"]:
        assert digest(case["source_path"]) == case["source_sha256"]
    save(
        root / "generation-complete.json",
        {
            "plan_hash": canonical_hash(plan),
            "responses": len(list((root / "responses").glob("*.json"))),
            "frozen_sources_unchanged": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--combined", type=Path)
    parser.add_argument("--key-path", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.root, args.historical, args.combined)
    else:
        run(args.root, args.key_path)
