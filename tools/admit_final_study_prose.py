"""Bind separately authored claim decisions to saved final-study Luna drafts.

This command cannot invent an approval. Every reviewable claim must already have
an explicit independent decision; originals and provider responses remain intact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from exact_inspect.contracts import canonical_hash, file_hash
from exact_inspect.description_review import (
    IndependentDescriptionReview,
    admit_independent_review,
    build_review_request,
)
from tools.final_study_prose import freeze, read


def admit(root, decisions_path, reviewer_id, provenance):
    plan = read(root / "plan.json")
    decisions = read(decisions_path)
    by_id = {row["id"]: row for row in decisions}
    requests = sorted((root / "review-requests").glob("*.json"))
    if len(by_id) != len(decisions) or set(by_id) != {p.stem for p in requests}:
        raise ValueError("Independent decisions must cover exactly all reviewable drafts")
    jobs = {j["job_id"].removeprefix("sha256:"): j for j in plan["jobs"]}
    approved, counts = {}, Counter()
    for path in requests:
        row = by_id[path.stem]
        folder = root / "generation" / path.stem
        source = read(folder / "source.json")
        response = read(folder / "response.json")
        if canonical_hash(response["response"]) != response["sha256"]:
            raise ValueError("Saved provider response changed")
        request = build_review_request(source, response["response"])
        if request != read(path):
            raise ValueError("Review request differs from its saved source")
        if len(row["decisions"]) != len(request["claim_inventory"]):
            raise ValueError("Review does not cover every claim")
        receipt = IndependentDescriptionReview.model_validate(
            {
                **request["bindings"],
                "reviewer_id": reviewer_id,
                "reviewer_kind": "independent_agent",
                "review_provenance": provenance,
                "decisions": [
                    {**decision, **identity, "ordinal": n}
                    for n, (decision, identity) in enumerate(
                        zip(row["decisions"], request["claim_inventory"])
                    )
                ],
            }
        )
        freeze(root / "independent-reviews" / path.name, receipt.model_dump(mode="json"))
        counts.update(d.decision for d in receipt.decisions)
        if any(d.decision == "approved" for d in receipt.decisions):
            result = admit_independent_review(source, response["response"], receipt)
            freeze(root / "admitted" / path.name, result)
            approved[jobs[path.stem]["old_id"]] = result
    result = {
        "schema": "exact-final-prose-admission/1",
        "plan_sha256": file_hash(root / "plan.json"),
        "decisions_sha256": file_hash(decisions_path),
        "reviewed_jobs": len(requests),
        "admitted_jobs": len(approved),
        "claim_counts": dict(counts),
        "results": approved,
    }
    freeze(root / "reviewed-results.json", result)
    print({k: v for k, v in result.items() if k != "results"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--provenance", required=True)
    args = parser.parse_args()
    admit(args.root, args.decisions, args.reviewer_id, args.provenance)


if __name__ == "__main__":
    main()
