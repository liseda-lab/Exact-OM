"""Explicit policy-filtered OWL export with verifiable admission receipts.

Preparation uses the installed OWL decoder/renderer; participant publication only
verifies immutable bytes and the receipt. Source paths are not required at serving.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import atomic_json
from .context_semantics import ANNOTATION_REGISTRY, _allowed, decode_payload
from .contracts import VisibilityPolicy, canonical_hash, file_hash

RECEIPT_SCHEMA = "exact-ontology-resource/1"
_LOGICAL_CATEGORIES = {
    "hierarchy",
    "restrictions",
    "types",
    "assertions",
    "domains",
    "ranges",
    "characteristics",
    "axioms",
}


def export_ontology_resource(
    context: Any, destination: str | Path, policy: VisibilityPolicy
) -> Path:
    """Write a portable import-free Functional Syntax ontology and adjacent receipt.

    All logical axioms must be permitted. Only annotations can be removed by this
    resource exporter, preserving the semantics of the declared ontology scope.
    Canonical original axioms remain in the authorized preparation package.
    """
    import pyowl_core as core

    version = context.ontology_version_id
    if not policy.allows_ontology(version) or any(
        not _allowed(policy, version, c) for c in _LOGICAL_CATEGORIES
    ):
        raise ValueError("Ontology resource policy must permit the declared logical ontology scope")
    destination = Path(destination)
    receipt_path = destination.with_suffix(destination.suffix + ".receipt.json")
    if receipt_path.exists():
        validate_ontology_resource(
            destination, receipt_path, policy_hash=policy.policy_hash, ontology_ids=[version]
        )
        return receipt_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending_path = receipt_path.with_suffix(receipt_path.suffix + ".pending")
    pending = {
        "source_context_sha256": context.manifest["artifacts"]["context.sqlite"],
        "ontology_version_id": version,
        "policy_hash": policy.policy_hash,
    }
    if pending_path.exists():
        if json.loads(pending_path.read_text(encoding="utf-8")) != pending:
            raise ValueError("Incomplete ontology export has different inputs")
    elif destination.exists():
        raise ValueError("Existing ontology resource has no matching preparation receipt")
    else:
        atomic_json(pending_path, pending)
    template = core.load_snapshot(
        b"Ontology()", options=core.LoadOptions(imports=core.ImportPolicy.IGNORE)
    ).root
    removed: dict[str, int] = {}
    retained: dict[str, int] = {}
    redactions: dict[str, int] = {}

    def filtered(node: Any) -> Any:
        if isinstance(node, core.CanonicalSet):
            items = []
            for item in node:
                if type(item).__name__ == "Annotation":
                    predicate = item.property.iri.value
                    category = ANNOTATION_REGISTRY.get(predicate, ("annotations", None))[0]
                    if not _allowed(policy, version, category):
                        redactions[predicate] = redactions.get(predicate, 0) + 1
                        continue
                items.append(filtered(item))
            return core.CanonicalSet(items)
        if (
            dataclasses.is_dataclass(node)
            and not isinstance(node, type)
            and hasattr(node, "annotations")
        ):
            return dataclasses.replace(
                node,
                **{
                    field.name: filtered(getattr(node, field.name))
                    for field in dataclasses.fields(node)
                    if field.init
                },
            )
        if isinstance(node, tuple):
            return tuple(filtered(item) for item in node)
        return node

    descriptor, temporary = tempfile.mkstemp(prefix=".ontology-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream, context._connection(
            vm_step_budget=50_000_000_000
        ) as connection:
            stream.write(b"Ontology(\n")
            batch: list[Any] = []

            def flush() -> None:
                if not batch:
                    return
                document = dataclasses.replace(template, axioms=core.CanonicalSet(batch))
                rendered = core.render_document(document, format=core.DocumentFormat.FUNCTIONAL)
                if not rendered.startswith(b"Ontology(\n") or not rendered.endswith(b")\n"):
                    raise ValueError("Unsupported installed Functional Syntax renderer framing")
                stream.write(rendered[len(b"Ontology(\n") : -len(b")\n")])
                batch.clear()

            for row in connection.execute("SELECT category,payload FROM axioms ORDER BY rowid"):
                category = row["category"]
                if not _allowed(policy, version, category):
                    removed[category] = removed.get(category, 0) + 1
                    continue
                payload = decode_payload(row["payload"])
                axiom = core.decode_canonical(base64.b64decode(payload["original_syntax"]))
                batch.append(filtered(axiom))
                retained[category] = retained.get(category, 0) + 1
                if len(batch) >= 1000:
                    flush()
            flush()
            stream.write(b")\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "ontology_version_id": version,
            "source_context_sha256": context.manifest["artifacts"]["context.sqlite"],
            "source_documents": [record["source_sha256"] for record in context.manifest["sources"]],
            "policy": policy.model_dump(mode="json"),
            "policy_hash": policy.policy_hash,
            "resource_sha256": file_hash(destination),
            "resource_size_bytes": destination.stat().st_size,
            "format": "application/owl-functional",
            "scope": context.manifest["scope"],
            "imports": "flattened_declared_scope_no_external_resolution",
            "retained_categories": retained,
            "removed_categories": removed,
            "redacted_annotation_predicates": redactions,
            "logical_axioms_preserved": True,
            "exporter": "exact-policy-ontology/1",
        }
        receipt["receipt_hash"] = canonical_hash(receipt)
        atomic_json(receipt_path, receipt)
        pending_path.unlink(missing_ok=True)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return receipt_path


def validate_ontology_resource(
    path: str | Path,
    receipt_path: str | Path,
    *,
    policy_hash: str | None = None,
    ontology_ids: Any = (),
) -> dict[str, Any]:
    """Verify preparation admission and bytes without invoking an OWL parser.

    Receipts are trusted preparation inputs, bound by the study asset's own hash;
    they are not a signature authority for arbitrary participant uploads.
    """
    path, receipt_path = Path(path), Path(receipt_path)
    receipt: dict[str, Any] = json.loads(receipt_path.read_text(encoding="utf-8"))
    claimed = receipt.get("receipt_hash")
    if (
        canonical_hash({key: value for key, value in receipt.items() if key != "receipt_hash"})
        != claimed
    ):
        raise ValueError("Ontology admission receipt hash mismatch")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("exporter") != "exact-policy-ontology/1"
    ):
        raise ValueError("Unsupported ontology admission receipt")
    policy = VisibilityPolicy.model_validate(receipt["policy"])
    if policy.policy_hash != receipt.get("policy_hash") or (
        policy_hash and policy_hash != policy.policy_hash
    ):
        raise ValueError("Ontology resource policy mismatch")
    version = receipt.get("ontology_version_id")
    if not isinstance(version, str):
        raise ValueError("Ontology resource receipt lacks a version identity")
    if (ontology_ids and version not in ontology_ids) or not policy.allows_ontology(version):
        raise ValueError("Ontology resource version is outside the admitted universe")
    if receipt.get("logical_axioms_preserved") is not True or any(
        category in _LOGICAL_CATEGORIES for category in receipt.get("removed_categories", {})
    ):
        raise ValueError("Ontology export did not preserve logical axioms")
    if any(
        not _allowed(policy, version, category)
        for category in receipt.get("retained_categories", {})
    ):
        raise ValueError("Ontology resource contains a forbidden category")
    if (
        receipt.get("format") != "application/owl-functional"
        or receipt.get("imports") != "flattened_declared_scope_no_external_resolution"
    ):
        raise ValueError("Ontology resource is not a self-contained supported export")
    if path.stat().st_size != receipt.get("resource_size_bytes") or file_hash(path) != receipt.get(
        "resource_sha256"
    ):
        raise ValueError("Ontology resource bytes do not match admission receipt")
    return receipt


def main() -> None:
    """Export an admitted original-ontology resource during explicit preparation."""
    import argparse

    from .context import OntologyContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = VisibilityPolicy.model_validate_json(args.policy.read_text(encoding="utf-8"))
    receipt = export_ontology_resource(OntologyContext(args.context), args.output, policy)
    print(receipt)


if __name__ == "__main__":
    main()
