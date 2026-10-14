"""Preview an explicit bounded recovery batch; dispatch requires its exact approved plan.

The preview reads frozen earlier requests, never credentials or provider responses.
Execution reuses only the selected attempt's immutable cache and never repairs/retries.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from exact_inspect.artifacts import atomic_json
from exact_inspect.contracts import canonical_hash, file_hash
from exact_inspect.description_review import build_review_request
from exact_inspect.descriptions import PROFILE_PROSE_PROMPT_V5
from exact_inspect.generation import (
    ExplanationOutput,
    FactPacket,
    GenerationProfile,
    comparison_packet,
    prose_response_format,
)
from tools.review_generation_attempt import (
    ReviewExplanationJobs,
    attempt_paths,
    bind_attempt,
    bind_generation_contract,
    generation_contract,
)


def frozen_profile(value):
    # Preserve JSON numeric spelling used in immutable request identities. Pydantic
    # coerces explicit 0/60 into 0.0/60.0 even though constructor defaults are ints.
    GenerationProfile.model_validate(value)
    if set(value) != set(GenerationProfile.model_fields):
        raise ValueError("Frozen profile lacks its complete recorded parameters")
    return GenerationProfile.model_construct(**value)


def _frozen_requests(root, source_attempt):
    jobs = {}
    for path in sorted(attempt_paths(root, source_attempt)["generation"].glob("*/request.json")):
        value = json.loads(path.read_bytes())
        if canonical_hash(value)[7:] != path.parent.name:
            raise ValueError("Earlier frozen request differs from its identity")
        packet = FactPacket.model_validate(value["packet"])
        if packet.selection.get("description_prose") is not True or value["max_repairs"] != 0:
            raise ValueError("Earlier batch is not the admitted prose scope")
        jobs[path.parent.name] = (value, packet)
    return jobs


def batch_plan(root, attempt, source_attempt=1):
    root = Path(root).resolve()
    attempt_paths(root, attempt)
    if attempt <= source_attempt:
        raise ValueError("Recovery requires a later explicit attempt")
    if (root / "publication.json").exists():
        raise ValueError("Completed publications cannot acquire another generation attempt")
    inputs = json.loads((root / "preparation-inputs.json").read_bytes())
    selection = json.loads((root / "selection.json").read_bytes())
    if inputs["description_preparation"] != "independent-prose/1":
        raise ValueError("This root is not bound to provider preparation")
    for relative, expected in inputs["contexts"].items():
        if file_hash(root / "package" / relative) != expected:
            raise ValueError("Prepared ontology manifest changed")
    if selection["run_manifest_sha256"] != inputs["run_manifest"]:
        raise ValueError("Selected cases belong to another preparation input")
    jobs = _frozen_requests(root, source_attempt)
    profiles = {
        canonical_hash(p.entities[0]): (key, value, p)
        for key, (value, p) in jobs.items()
        if p.task == "entity_profile"
    }
    if len(profiles) != sum(p.task == "entity_profile" for _, p in jobs.values()):
        raise ValueError("Earlier scope contains repeated independent profile requests")
    bindings = {canonical_hash(value["profile"]) for value, _ in jobs.values()}
    if len(bindings) != 1:
        raise ValueError("Earlier scope has inconsistent provider bindings")
    profile = frozen_profile(next(iter(jobs.values()))[0]["profile"])
    contract = generation_contract(profile)
    requested_profiles, pairs, seen = [], [], set()
    for case_index, selected in enumerate(selection["cases"]):
        entities = [selected["source"], *selected["targets"]]
        for entity in entities:
            identity = canonical_hash(entity)
            if identity in profiles and identity not in seen:
                seen.add(identity)
                key, value, packet = profiles[identity]
                updated = {
                    **value,
                    "provider_response_format": prose_response_format(),
                    "prompt": PROFILE_PROSE_PROMPT_V5,
                    "schema": ExplanationOutput.model_json_schema(),
                    "implementation": contract["implementation"],
                }
                requested_profiles.append(
                    {
                        "source_request": key,
                        "entity": entity,
                        "packet_sha256": canonical_hash(packet),
                        "job_id": canonical_hash(updated),
                    }
                )
        if case_index < inputs["generate_pairs"]:
            pair = entities[:2]
            if any(canonical_hash(e) not in profiles for e in pair):
                raise ValueError("Pair lacks an independently requested profile")
            matches = [
                key
                for key, (_, p) in jobs.items()
                if p.task == "pair_comparison"
                and [e.model_dump(mode="json") for e in p.entities] == pair
            ]
            if len(matches) != 1:
                raise ValueError("Earlier comparison scope does not match selected top candidate")
            pairs.append(
                {
                    "case_id": selected["case_id"],
                    "entities": pair,
                    "source_request": matches[0],
                    "profile_packet_hashes": [
                        canonical_hash(profiles[canonical_hash(e)][2]) for e in pair
                    ],
                }
            )
    if len(seen) != len(profiles) or len(pairs) != sum(
        p.task == "pair_comparison" for _, p in jobs.values()
    ):
        raise ValueError("Earlier scope contains requests outside the selected cases")
    return {
        "schema": "exact-review-generation-batch/1",
        "root": str(root),
        "generation_attempt": attempt,
        "source_attempt": source_attempt,
        "preparation_inputs_sha256": file_hash(root / "preparation-inputs.json"),
        "selection_sha256": file_hash(root / "selection.json"),
        "contract": contract,
        "profiles": requested_profiles,
        "comparisons": pairs,
        "counts": {"entity_profile": len(requested_profiles), "pair_comparison": len(pairs)},
        "comparison_binding": "Exact comparison job hashes are derived at execution from these two immutable packets and their newly saved independent profile results, using comparison_packet; no historical prose normalization.",
        "maximum_new_http_attempts": len(requested_profiles) + len(pairs),
        "destination": "https://openrouter.ai/api/v1/chat/completions",
        "authorization": "Preview only. A further bounded batch requires explicit approval before execute.",
    }


def execute_plan(plan, approved_sha256, *, preflight=False):
    if canonical_hash(plan) != approved_sha256:
        raise ValueError("Explicit approved plan hash does not match")
    current = batch_plan(plan["root"], plan["generation_attempt"], plan["source_attempt"])
    if current != plan:
        raise ValueError("Approved batch inputs or generation contract changed")
    root, attempt = Path(plan["root"]), plan["generation_attempt"]
    paths = attempt_paths(root, attempt)
    profile = frozen_profile(plan["contract"]["profile"])
    bind_attempt(root, attempt, json.loads((root / "preparation-inputs.json").read_bytes()))
    bind_generation_contract(root, attempt, profile)
    # The generation lock plus ledger block concurrent sends and unknown-delivery retries.
    jobs, originals = ReviewExplanationJobs(paths["generation"]), _frozen_requests(
        root, plan["source_attempt"]
    )
    results, packets, counts, exact_jobs = {}, {}, Counter(), []
    selected_profiles, selected_comparisons = plan["profiles"], plan["comparisons"]
    if preflight:
        # One real comparison, its independent profiles, and one missing-definition
        # profile exercise both request shapes before the rest of this SAME plan.
        selected_comparisons = selected_comparisons[:1]
        identities = {canonical_hash(e) for pair in selected_comparisons for e in pair["entities"]}
        missing = next(
            (
                p
                for p in selected_profiles
                if not any(
                    f.get("category") == "definitions"
                    for f in originals[p["source_request"]][1].facts
                )
            ),
            None,
        )
        if missing:
            identities.add(canonical_hash(missing["entity"]))
        selected_profiles = [
            p for p in selected_profiles if canonical_hash(p["entity"]) in identities
        ]

    def generate(packet, expected=None):
        result = jobs.generate(packet, profile, max_repairs=0)
        if expected is not None and result["explanation_id"] != expected:
            raise ValueError("Generated profile differs from approved identity")
        key = result["explanation_id"][7:]
        counts[packet.task] += 1
        exact_jobs.append(
            {
                "task": packet.task,
                "job_id": result["explanation_id"],
                "packet_sha256": canonical_hash(packet),
            }
        )
        saved = jobs.root / key / "response-0.json"
        if saved.is_file():
            response = json.loads(saved.read_bytes())["response"]
            try:
                request = build_review_request(result, response)
            except ValueError as error:
                atomic_json(
                    paths["requests"] / (key + ".unreviewable.json"),
                    {"explanation_id": result["explanation_id"], "reason": str(error)},
                )
                if preflight:
                    raise ValueError(
                        "Real preflight response is unreviewable; no remaining jobs dispatched"
                    ) from error
            else:
                atomic_json(paths["requests"] / (key + ".json"), request)
        else:
            raise ValueError(
                "Generation has no saved provider response; no remaining jobs dispatched"
            )
        atomic_json(
            root / f"generation-progress-attempt-{attempt}.json",
            {
                "counts": dict(counts),
                "approved_plan_sha256": approved_sha256,
                "exact_jobs": exact_jobs,
            },
        )
        return result

    for item in selected_profiles:
        packet = originals[item["source_request"]][1]
        identity = canonical_hash(item["entity"])
        packets[identity] = packet
        results[identity] = generate(packet, item["job_id"])
    for item in selected_comparisons:
        a, b = [canonical_hash(e) for e in item["entities"]]
        packet = comparison_packet(packets[a], packets[b], [results[a], results[b]])
        generate(packet)
    marker = (
        root
        / f"generation-{'preflight' if preflight else 'dispatch-complete'}-attempt-{attempt}.json"
    )
    atomic_json(
        marker,
        {
            "counts": dict(counts),
            "approved_plan_sha256": approved_sha256,
            "exact_jobs": exact_jobs,
            "preflight": preflight,
            "semantic_review": "Separate independent review required; no publication performed.",
        },
    )
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview")
    preview.add_argument("--root", type=Path, required=True)
    preview.add_argument("--attempt", type=int, required=True)
    preview.add_argument("--source-attempt", type=int, default=1)
    preview.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("execute")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--approved-plan-sha256", required=True)
    execute.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.command == "preview":
        if args.output.exists():
            raise ValueError("Preview output already exists")
        plan = batch_plan(args.root, args.attempt, args.source_attempt)
        atomic_json(args.output, plan)
        print(
            json.dumps(
                {
                    "path": str(args.output),
                    "sha256": canonical_hash(plan),
                    "counts": plan["counts"],
                    "prompt_hashes": plan["contract"]["prompt_hashes"],
                    "dispatched": False,
                }
            )
        )
    else:
        print(
            execute_plan(
                json.loads(args.plan.read_bytes()),
                args.approved_plan_sha256,
                preflight=args.preflight,
            )
        )


if __name__ == "__main__":
    main()
