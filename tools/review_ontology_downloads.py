"""Reuse a verified full export and its original receipt during local preparation."""

import json
import os
import tempfile
from pathlib import Path

from exact_inspect.artifacts import atomic_json
from exact_inspect.context_resources import (
    _LOGICAL_CATEGORIES,
    validate_ontology_resource,
)
from exact_inspect.contracts import file_hash
from exact_inspect.navigation_prepare import copy_immutable


def reuse_ontology_download(context, source, destination, policy, parity_path, diagnostics):
    """Require exact immutable lineage and saved full parity before cloning bytes.

    This reuses an earlier verified preparation; it does not claim a new parser or
    parity run. Its untouched original receipt remains the served admission input.
    """
    source, destination, parity_path = map(Path, (source, destination, parity_path))
    diagnostics = Path(diagnostics)
    source_receipt = source.with_suffix(source.suffix + ".receipt.json")
    for path in (source, source_receipt, parity_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Prepared ontology inputs must be regular local files")
    receipt = validate_ontology_resource(
        source,
        source_receipt,
        policy_hash=policy.policy_hash,
        ontology_ids=[context.ontology_version_id],
        context=context,
    )
    parity = json.loads(parity_path.read_bytes())
    inspected = parity.get("inspected")
    if (
        parity.get("schema") != "exact-ontology-resource-parity/1"
        or parity.get("status") != "pass"
        or parity.get("source_verification") != "verified_context_originals"
        or parity.get("source_context_sha256") != receipt["source_context_sha256"]
        or parity.get("resource_sha256") != receipt["resource_sha256"]
        or parity.get("receipt_hash") != receipt["receipt_hash"]
        or parity.get("policy_hash") != policy.policy_hash
        or receipt["source_documents"] != [parity.get("source_sha256")]
        or any(
            type(parity.get(key)) is not int or parity[key] < 0
            for key in ("external_import_count", "excluded_annotation_count", "logical_axiom_count")
        )
        or not isinstance(inspected, dict)
        or type(inspected.get("axioms")) is not int
        or parity.get("external_import_count") != 0
        or parity.get("excluded_annotation_count") != 0
        or inspected.get("axioms") != sum(receipt["retained_categories"].values())
        or parity.get("logical_axiom_count")
        != sum(
            count
            for category, count in receipt["retained_categories"].items()
            if category in _LOGICAL_CATEGORIES
        )
    ):
        raise ValueError("Saved full ontology parity does not bind this export and context")
    receipt_path = destination.with_suffix(destination.suffix + ".receipt.json")
    pending = receipt_path.with_suffix(receipt_path.suffix + ".pending")
    preserved_pending = None
    prior_pending = diagnostics / (destination.name + ".prior-export-pending.json")
    if pending.exists():
        if json.loads(pending.read_bytes()) != {
            "source_context_sha256": context.manifest["artifacts"]["context.sqlite"],
            "ontology_version_id": context.ontology_version_id,
            "policy_hash": policy.policy_hash,
        }:
            raise ValueError("Unfinished ontology export belongs to other inputs")
        preserved_pending = prior_pending
    elif prior_pending.exists():
        preserved_pending = prior_pending
    destination.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or receipt_path.is_symlink():
        raise ValueError("Installed ontology outputs must not be symbolic links")
    if destination.exists() and file_hash(destination) != receipt["resource_sha256"]:
        raise ValueError("Existing ontology bytes differ; refusing to replace them")
    if receipt_path.exists() and source_receipt.read_bytes() != receipt_path.read_bytes():
        raise ValueError("Existing ontology receipt differs; refusing to replace it")
    if preserved_pending and pending.exists():
        if preserved_pending.exists() and preserved_pending.read_bytes() != pending.read_bytes():
            raise ValueError("Prior ontology failure diagnostic differs")
        if not preserved_pending.exists():
            copy_immutable(pending, preserved_pending)
    for original, target in ((source, destination), (source_receipt, receipt_path)):
        if target.exists():
            continue
        descriptor, staged_name = tempfile.mkstemp(prefix=".verified-ontology-", dir=target.parent)
        os.close(descriptor)
        staged = Path(staged_name)
        try:
            copy_immutable(original, staged)
            if file_hash(staged) != file_hash(original):
                raise ValueError("Cloned ontology input failed byte verification")
            os.replace(staged, target)
        finally:
            staged.unlink(missing_ok=True)
    validate_ontology_resource(
        destination,
        receipt_path,
        policy_hash=policy.policy_hash,
        ontology_ids=[context.ontology_version_id],
        context=context,
    )
    atomic_json(
        diagnostics / (destination.name + ".verified-reuse.json"),
        {
            "schema": "exact-review-ontology-reuse/1",
            "status": "verified_reuse_of_prior_full_export",
            "verification": "Current immutable bytes and lineage checked; saved parity reused, not rerun.",
            "ontology_version_id": context.ontology_version_id,
            "current_context_sha256": context.manifest["artifacts"]["context.sqlite"],
            "source_context_sha256": receipt["source_context_sha256"],
            "policy_hash": policy.policy_hash,
            "source_manifest_sha256": file_hash(context.path / "manifest.json"),
            "resource_sha256": file_hash(destination),
            "receipt_sha256": file_hash(receipt_path),
            "receipt_hash": receipt["receipt_hash"],
            "saved_parity_sha256": file_hash(parity_path),
            "retained_categories": receipt["retained_categories"],
            "preserved_pending_sha256": file_hash(preserved_pending) if preserved_pending else None,
        },
    )
    pending.unlink(missing_ok=True)
    return receipt_path
