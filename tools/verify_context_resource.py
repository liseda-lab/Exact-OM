"""Verify a policy ontology resource against its pinned source's logical axioms."""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import time
from pathlib import Path

from exact_inspect.context import OntologyContext
from exact_inspect.context_resources import validate_ontology_resource
from exact_inspect.context_semantics import (
    ANNOTATION_REGISTRY,
    _allowed,
    decode_payload,
)
from exact_inspect.contracts import VisibilityPolicy, file_hash
from tools.validate_context_runtime import PINS, ROOT, peak_rss_bytes


def main() -> None:
    """Reparse only explicitly requested preparation outputs; serving never calls this."""
    import pyowl_core as core

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ontology", choices=PINS)
    parser.add_argument("--resource", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context", type=Path, help="Reuse verified original native context")
    args = parser.parse_args()
    source_relative, expected_hash = PINS[args.ontology]
    source = ROOT / source_relative
    if file_hash(source) != "sha256:" + expected_hash:
        raise ValueError("Pinned ontology input changed")
    started = time.perf_counter()
    receipt = validate_ontology_resource(
        args.resource, args.resource.with_suffix(args.resource.suffix + ".receipt.json")
    )
    options = core.LoadOptions(
        backend=core.BackendPreference.NATIVE,
        imports=core.ImportPolicy.IGNORE,
        preserve_source_map=True,
        offline=True,
    )
    exported = core.load_snapshot(args.resource, options=options)

    policy = VisibilityPolicy.model_validate(receipt["policy"])
    inspected = {"axioms": 0, "annotation_assertions": 0, "nested_annotations": 0}

    def check_predicate(predicate):
        category = ANNOTATION_REGISTRY.get(predicate, ("annotations", None))[0]
        if not _allowed(policy, receipt["ontology_version_id"], category):
            raise ValueError(f"Ontology resource retained an excluded annotation: {predicate}")

    def check_annotations(annotations):
        for annotation in annotations:
            inspected["nested_annotations"] += 1
            check_predicate(annotation.property.iri.value)
            check_annotations(annotation.annotations)

    def logical_axioms(snapshot, *, verify_policy=False):
        logical = set()
        for axiom in snapshot.iter_axioms():
            annotation_assertion = type(axiom).__name__ == "AnnotationAssertion"
            if verify_policy:
                inspected["axioms"] += 1
                check_annotations(axiom.annotations)
                if annotation_assertion:
                    inspected["annotation_assertions"] += 1
                    check_predicate(axiom.property.iri.value)
            if not annotation_assertion:
                logical.add(
                    core.structural_digest(
                        dataclasses.replace(axiom, annotations=core.CanonicalSet())
                    )
                )
        return logical

    context = OntologyContext(args.context) if args.context else None
    if context is not None:
        if (
            context.manifest.get("policy_filter")
            or context.ontology_version_id != receipt["ontology_version_id"]
            or context.manifest["artifacts"]["context.sqlite"] != receipt["source_context_sha256"]
            or "sha256:" + expected_hash
            not in {item["source_sha256"] for item in context.manifest["sources"]}
        ):
            raise ValueError("Original context does not match the resource and pinned source")
        expected = set()
        with context._connection(vm_step_budget=50_000_000_000) as connection:
            # AnnotationAssertion is the only stored constructor with a predicate.
            for row in connection.execute(
                "SELECT original_digest,payload FROM axioms WHERE predicate IS NULL ORDER BY rowid"
            ):
                payload = decode_payload(row["payload"])
                if payload["ast"].get("annotations"):
                    axiom = core.decode_canonical(base64.b64decode(payload["original_syntax"]))
                    if not dataclasses.is_dataclass(axiom) or isinstance(axiom, type):
                        raise ValueError("Original canonical axiom is not a structural dataclass")
                    digest = core.structural_digest(
                        dataclasses.replace(axiom, annotations=core.CanonicalSet())
                    )
                else:
                    digest = bytes.fromhex(row["original_digest"].removeprefix("sha256:"))
                expected.add(digest)
    else:
        expected = logical_axioms(core.load_snapshot(source, options=options))
    actual = logical_axioms(exported, verify_policy=True)
    if actual != expected or exported.import_manifest.edges:
        raise ValueError("Ontology resource changed logical axioms or retained external imports")
    report = {
        "schema": "exact-ontology-resource-parity/1",
        "ontology": args.ontology,
        "status": "pass",
        "source_sha256": file_hash(source),
        "source_verification": "verified_context_originals" if context else "native_pinned_source",
        "source_context_sha256": (
            context.manifest["artifacts"]["context.sqlite"] if context else None
        ),
        "resource_sha256": file_hash(args.resource),
        "receipt_hash": receipt["receipt_hash"],
        "policy_hash": receipt["policy_hash"],
        "logical_axiom_count": len(actual),
        "external_import_count": 0,
        "excluded_annotation_count": 0,
        "inspected": inspected,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss_bytes(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
