"""Check frozen prose-review bindings, main API responses and study-resource parity.

This read-only verifier replays separately recorded review decisions; it does not
make semantic approvals. It also applies the existing independent typed-rendering
audit to fallback claims. No provider credential, generation or database mutation
is involved. Run after cache-only publication assembly, with a new receipt path::

    python -m tools.verify_review_prose /path/to/prepared-input /path/to/receipt.json
"""

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

from fastapi.testclient import TestClient

from exact_inspect.artifacts import atomic_json, validate_bundle
from exact_inspect.context import OntologyContext
from exact_inspect.contracts import canonical_hash, file_hash
from exact_inspect.description_review import (
    IndependentDescriptionReview,
    admit_independent_review,
    build_review_request,
)
from exact_inspect.descriptions import validate_prepared_support
from exact_inspect.models import GeneratedExplanationResponse
from exact_inspect.service import create_prepared_app
from exact_inspect.study.resources import ExplanationResource
from tools.audit_review_publication import assess_claim, contained
from tools.review_generation_attempt import attempt_paths, verified_saved_result


def verify(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("Use a new receipt outside the immutable preparation root")
    started = time.monotonic()
    package = root / "package"
    bundle = validate_bundle(package / "package.json")
    attempt = attempt_paths(root, bundle.provenance["generation_attempt"])
    contexts = {v: OntologyContext(package / p) for v, p in bundle.ontologies.items()}
    results = {
        key: json.loads(contained(package, locator).read_bytes())
        for key, locator in bundle.explanations.items()
    }
    counts, reviewed_ids, reviewed_claims = Counter(), set(), set()
    review_hashes = {}
    for path in sorted(attempt["reviews"].glob("*.json")):
        receipt = IndependentDescriptionReview.model_validate_json(path.read_bytes())
        directory = attempt["generation"] / receipt.source_explanation_id[7:]
        source = verified_saved_result(directory)
        responses = [json.loads(p.read_bytes()) for p in directory.glob("response-*.json")]
        raw = next(r["response"] for r in responses if r["sha256"] == receipt.response_hash)
        assert canonical_hash(raw) == receipt.response_hash
        bindings = build_review_request(source, raw)["bindings"]
        assert all(getattr(receipt, key) == value for key, value in bindings.items())
        approved = [d for d in receipt.decisions if d.decision == "approved"]
        if approved:
            expected = admit_independent_review(source, raw, receipt.model_dump(mode="json"))
            assert receipt.source_explanation_id not in results
            reviewed_ids.add(expected["explanation_id"])
            reviewed_claims.update(d.claim_id for d in approved)
            counts[source["task"] + "_with_approved_claims"] += 1
        else:
            expected = source
            assert expected["preparation_kind"] == "factual_fallback"
        assert results[expected["explanation_id"]] == expected
        counts.update("draft_claims_" + d.decision for d in receipt.decisions)
        counts["review_receipts"] += 1
        review_hashes[path.name] = file_hash(path)
    claims = {}
    for key, result in results.items():
        assert result["explanation_id"] == key
        validate_prepared_support(result, contexts, bundle.policy)
        if result["support_status"] == "independently_reviewed":
            assert key in reviewed_ids
        else:
            for claim in result["claims"]:
                assess_claim(claim, result)
                counts["typed_fallback_claim_occurrences"] += 1
        for claim in result["claims"]:
            expected = {
                **claim,
                "entities": result["entities"],
                "preparation_kind": result["preparation_kind"],
                "support_status": result["support_status"],
                "generation_manifest_sha256": canonical_hash(result["manifest"])[7:],
            }
            assert claims.setdefault(claim["claim_id"], expected) == expected
    publication = json.loads((root / "publication.json").read_bytes())
    definition = publication["definition"]
    assert definition["purpose"] == "development_review"
    assert definition["correctness_metrics"] == "disabled"
    assets = {a["asset_id"]: a for a in definition["assets"]}
    scopes = {s["case_id"]: s for s in definition["workspace_scopes"] if s["kind"] == "case"}
    all_scoped, displayed_reviewed = set(), set()
    for case in definition["cases"]:
        asset = assets[case["explanation_refs"][0]]
        path = contained(root, asset["path"])
        assert file_hash(path)[7:] == asset["sha256"]
        resource = ExplanationResource.model_validate_json(path.read_bytes())
        displayed = [*resource.entity_profiles, *resource.pair_comparison]
        for claim in displayed:
            expected = claims[claim.claim_id]
            for field in (
                "text",
                "fact_ids",
                "category",
                "preparation_kind",
                "support_status",
                "generation_manifest_sha256",
            ):
                assert getattr(claim, field) == expected[field]
            assert [e.model_dump() for e in claim.scoped_entities] == expected["entities"]
            if claim.support_status == "independently_reviewed":
                assert claim.claim_id in reviewed_claims
                assert claim.grounding == "independent_review"
                displayed_reviewed.add(claim.claim_id)
        scope = scopes[case["case_id"]]
        index = contained(root, scope["index_path"])
        assert file_hash(index)[7:] == scope["index_sha256"]
        with sqlite3.connect(index.as_uri() + "?mode=ro", uri=True) as db:
            saved = [json.loads(row[0]) for row in db.execute("SELECT payload FROM explanations")]
        scoped_claims = set()
        for result in saved:
            expected = GeneratedExplanationResponse.model_validate(
                results[result["explanation_id"]]
            )
            assert GeneratedExplanationResponse.model_validate(result) == expected
            all_scoped.add(result["explanation_id"])
            scoped_claims.update(c["claim_id"] for c in result["claims"])
        assert scoped_claims == {c.claim_id for c in displayed}
    assert all_scoped == set(results)
    assert displayed_reviewed == reviewed_claims
    with TestClient(create_prepared_app(package / "package.json")) as client:
        for key, result in results.items():
            response = client.get("/api/v1/explanations/" + key)
            assert response.status_code == 200, (key, response.status_code)
            assert GeneratedExplanationResponse.model_validate(response.json()) == (
                GeneratedExplanationResponse.model_validate(result)
            )
            counts["actual_api_responses_checked"] += 1
    report = {
        "schema": "exact-review-prose-verification/1",
        "status": "passed",
        "boundary": "Actual main ASGI API, admitted support and frozen study-resource/index parity; not a browser or PostgreSQL journey or a new semantic approval",
        "publication_sha256": file_hash(root / "publication.json"),
        "package_sha256": file_hash(package / "package.json"),
        "policy_sha256": bundle.policy.policy_hash,
        "verifier_sha256": file_hash(Path(__file__)),
        "review_receipt_sha256": review_hashes,
        "counts": dict(counts),
        "case_count": len(definition["cases"]),
        "presented_pairs": sum(len(c["candidates"]) for c in definition["cases"]),
        "unique_artifacts": len(results),
        "approved_prose_claims": len(reviewed_claims),
        "seconds": time.monotonic() - started,
    }
    atomic_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.output), sort_keys=True))
