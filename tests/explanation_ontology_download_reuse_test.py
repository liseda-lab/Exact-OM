"""Full saved exports can be reused only through an exact immutable context chain."""

import copy
import json

import pyowl_core as core
import pytest

from exact_inspect.artifacts import atomic_json
from exact_inspect.context import build_context_package
from exact_inspect.context_export import export_policy_context
from exact_inspect.context_resources import (
    _LOGICAL_CATEGORIES,
    export_ontology_resource,
    validate_ontology_resource,
)
from exact_inspect.context_semantics import decode_payload, typed_node
from exact_inspect.contracts import (
    VisibilityPolicy,
    canonical_hash,
    canonical_json,
    file_hash,
)
from tools.review_ontology_downloads import reuse_ontology_download


@pytest.fixture
def prepared(tmp_path):
    original = build_context_package(
        core.load_snapshot(
            b"""Ontology(
        Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>))
        SubClassOf(<urn:A> <urn:B>)
        AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "Alpha")
        AnnotationAssertion(<http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P207> <urn:A> "PRIVATE")
        )"""
        ),
        tmp_path / "original",
    )
    policy = VisibilityPolicy()
    filtered = export_policy_context(original, tmp_path / "filtered", policy)
    resource = tmp_path / "downloads" / "original.ofn"
    receipt_path = export_ontology_resource(original, resource, policy)
    receipt = json.loads(receipt_path.read_bytes())
    snapshot = core.load_snapshot(resource)
    with filtered._connection() as db:
        expected = {
            canonical_json(decode_payload(row[0])["ast"])
            for row in db.execute("SELECT payload FROM axioms")
        }
    assert {canonical_json(typed_node(a)) for a in snapshot.iter_axioms()} == expected
    assert not snapshot.import_manifest.edges and "PRIVATE" not in resource.read_text()
    parity_path = resource.with_suffix(".parity.json")
    atomic_json(
        parity_path,
        {
            "schema": "exact-ontology-resource-parity/1",
            "status": "pass",
            "source_verification": "verified_context_originals",
            "source_context_sha256": receipt["source_context_sha256"],
            "resource_sha256": receipt["resource_sha256"],
            "receipt_hash": receipt["receipt_hash"],
            "policy_hash": policy.policy_hash,
            "source_sha256": receipt["source_documents"][0],
            "external_import_count": 0,
            "excluded_annotation_count": 0,
            "inspected": {"axioms": len(expected)},
            "logical_axiom_count": sum(
                count
                for category, count in receipt["retained_categories"].items()
                if category in _LOGICAL_CATEGORIES
            ),
        },
    )
    return original, filtered, resource, receipt_path, parity_path, policy


def test_prior_full_export_reuses_exact_filtered_child_and_preserves_failure(prepared, tmp_path):
    original, context, source, original_receipt, parity, policy = prepared
    target = tmp_path / "review" / "source-review.ofn"
    pending = target.with_suffix(".ofn.receipt.json.pending")
    pending_record = {
        "source_context_sha256": context.manifest["artifacts"]["context.sqlite"],
        "ontology_version_id": context.ontology_version_id,
        "policy_hash": policy.policy_hash,
    }
    atomic_json(pending, pending_record)
    frozen = {
        p: file_hash(p) for p in (source, original_receipt, context.database, original.database)
    }
    diagnostics = tmp_path / "diagnostics"
    receipt = reuse_ontology_download(context, source, target, policy, parity, diagnostics)
    assert target.stat().st_ino != source.stat().st_ino
    assert receipt.read_bytes() == original_receipt.read_bytes()
    assert file_hash(target) == frozen[source]
    assert not pending.exists()
    preserved = diagnostics / "source-review.ofn.prior-export-pending.json"
    assert json.loads(preserved.read_bytes()) == pending_record
    assert {p: file_hash(p) for p in frozen} == frozen
    assert export_ontology_resource(context, target, policy) == receipt
    validate_ontology_resource(target, receipt, policy_hash=policy.policy_hash, context=context)
    reuse = json.loads((diagnostics / "source-review.ofn.verified-reuse.json").read_bytes())
    assert reuse["source_context_sha256"] != reuse["current_context_sha256"]
    assert reuse["preserved_pending_sha256"] == file_hash(preserved)
    before = file_hash(diagnostics / "source-review.ofn.verified-reuse.json")
    assert reuse_ontology_download(context, source, target, policy, parity, diagnostics) == receipt
    assert file_hash(diagnostics / "source-review.ofn.verified-reuse.json") == before


@pytest.mark.parametrize(
    "changed", ["parent", "policy", "scope", "documents", "counts", "preparation_key"]
)
def test_export_cache_rejects_same_version_with_different_context_binding(prepared, changed):
    _, context, resource, receipt, _, policy = prepared
    context.manifest = copy.deepcopy(context.manifest)
    if changed == "parent":
        context.manifest["policy_filter"]["source_context_sha256"] = "sha256:" + "0" * 64
    elif changed == "policy":
        context.manifest["policy_filter"]["policy_hash"] = "sha256:" + "0" * 64
    elif changed == "scope":
        context.manifest["scope"] = "closure"
    elif changed == "documents":
        context.manifest["sources"][0]["source_sha256"] = "sha256:" + "0" * 64
    elif changed == "counts":
        context.manifest["completeness"]["category_counts"]["labels"] += 1
        context.manifest["completeness"]["axiom_count"] += 1
    else:
        context.manifest["preparation_key"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="context|lineage"):
        export_ontology_resource(context, resource, policy)


@pytest.mark.parametrize("changed", ["resource", "receipt", "parity", "pending", "existing"])
def test_reuse_does_not_install_or_replace_unverified_inputs(prepared, tmp_path, changed):
    _, context, source, receipt, parity, policy = prepared
    target = tmp_path / "review" / "target-review.ofn"
    target.parent.mkdir()
    if changed == "resource":
        source.write_bytes(source.read_bytes() + b"\n")
    elif changed == "receipt":
        record = json.loads(receipt.read_bytes())
        record["retained_categories"]["labels"] += 1
        record["receipt_hash"] = canonical_hash(
            {key: value for key, value in record.items() if key != "receipt_hash"}
        )
        atomic_json(receipt, record)
    elif changed == "parity":
        record = json.loads(parity.read_bytes())
        record["source_context_sha256"] = "sha256:" + "0" * 64
        atomic_json(parity, record)
    elif changed == "pending":
        atomic_json(target.with_suffix(".ofn.receipt.json.pending"), {"other": "preparation"})
    else:
        target.write_text("Existing unrelated bytes must survive.")
    with pytest.raises(ValueError):
        reuse_ontology_download(context, source, target, policy, parity, tmp_path / "diagnostics")
    assert not target.with_suffix(".ofn.receipt.json").exists()
    if changed == "existing":
        assert target.read_text() == "Existing unrelated bytes must survive."
    else:
        assert not target.exists()
