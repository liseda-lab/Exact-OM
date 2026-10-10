"""Saved real matcher links retain scope and never become invented ontology edges."""

import copy
import json
from pathlib import Path

import pyowl_core as core
import pytest
from fastapi.testclient import TestClient

from exact.runs import ExplanationStore
from exact_inspect.artifacts import publish_bundle
from exact_inspect.context import build_context_package
from exact_inspect.contracts import DomainError, canonical_hash, file_hash
from exact_inspect.decisions import (
    DecisionStore,
    dependency_inventory_hash,
    import_run,
    selected_evidence,
)
from exact_inspect.evidence_metadata import (
    PairEvidenceMetadata,
    filter_pair_metadata,
    pair_evidence_metadata,
    recorded_bridges,
)
from exact_inspect.models import SelectedEvidence
from exact_inspect.service import create_prepared_app


@pytest.fixture
def actual():
    saved = json.loads(
        (Path(__file__).parent / "fixtures/explanation_cross_side_local300.json").read_bytes()
    )
    record = saved["record"]
    assert canonical_hash(record) == saved["provenance"]["retained_record_hash"]
    source = dict(ontology_version_id="source-context", iri=record["src_iri"], kind="class")
    target = dict(ontology_version_id="target-context", iri=record["tgt_iri"], kind="class")
    return record, source, target


def adapt(record, source, target):
    items = selected_evidence(record, source, target)
    return items, pair_evidence_metadata(record, items, source, target, ["sha256:retained-record"])


def test_real_retained_links_map_exact_features_cross_channels_and_entity_endpoints(actual):
    record, source, target = actual
    before = canonical_hash(record)
    items, metadata = adapt(record, source, target)
    for item in items:
        SelectedEvidence.model_validate(item)
    assert len(items) == 29
    assert metadata["bridge_status"] == "recorded"
    assert metadata["comparison_resolution"] == dict(
        recorded=28, resolved=28, unresolved=0, ambiguous=0, unsupported=0
    )
    assert len(metadata["bridges"]) == 28
    by_id = {item["evidence_id"]: item for item in items}
    hierarchy = [b for b in metadata["bridges"] if b["channel"] == "hierarchy"]
    assert len(hierarchy) == 6
    for bridge, raw in zip(hierarchy, record["cross_side_provenance"]["hierarchy"]["is_a"]):
        assert bridge["level"] == "feature_correspondence"
        assert bridge["source_family"] == bridge["target_family"] == "is_a"
        assert bridge["recorded_score"] == raw["score"]
        for side in ("source", "target"):
            row = by_id[bridge[side + "_evidence_ids"][0]]
            assert row["historical_item_alias"] == "legacy:" + raw[side + "_item_id"]
            assert row["side"] == side and row["channel"] == "hierarchy" and row["family"] == "is_a"
    cross = [
        b
        for b in metadata["bridges"]
        if b["source_channel"] == "attributes" and b["target_channel"] == "hierarchy"
    ]
    assert len(cross) == 1 and cross[0]["level"] == "feature_correspondence"
    assert cross[0]["source_family"] is None and cross[0]["target_family"] == "is_a"
    difference = [b for b in metadata["bridges"] if b["channel"] == "difference"]
    assert len(difference) == 3
    assert all(
        b["level"] == "channel_comparison"
        and not b["source_evidence_ids"]
        and b["target_evidence_ids"]
        for b in difference
    )
    lexical = next(b for b in metadata["bridges"] if b["channel"] == "lexical")
    assert lexical["comparison_basis"] == "lexical_score"
    assert not lexical["source_evidence_ids"] and not lexical["target_evidence_ids"]
    assert all(b["interpretation"] == "matcher_comparison" for b in metadata["bridges"])
    assert (
        recorded_bridges({"evidence_bridges": metadata["bridges"]}, items, source, target, [])
        == metadata["bridges"]
    )
    assert canonical_hash(record) == before


def test_exact_family_scope_avoids_channel_union_and_ambiguous_attribute_anchor(actual):
    record, source, target = actual
    alias = record["cross_side_provenance"]["attributes"]["source"][0]["anchor_ref"]
    matching = next(
        item
        for item in record["triple_attributions"]["hierarchy"]["is_a"]["target"]
        if item["item_id"] == alias
    )
    record["triple_attributions"]["hierarchy"]["part_of"]["target"] = [copy.deepcopy(matching)]
    # A same alias in another family must not alter the six exact is_a pair links.
    _, metadata = adapt(record, source, target)
    assert len([b for b in metadata["bridges"] if b["channel"] == "hierarchy"]) == 6
    assert metadata["comparison_resolution"] == dict(
        recorded=28, resolved=27, unresolved=0, ambiguous=1, unsupported=0
    )
    issue = next(r for r in metadata["comparison_records"] if r["status"] == "ambiguous")
    assert (
        issue["reason_code"] == "saved_item_alias_not_unique_in_scope"
        and issue["bridge_id"] is None
    )
    assert not any(
        b["source_channel"] == "attributes" and b["target_channel"] == "hierarchy"
        for b in metadata["bridges"]
    )
    assert alias not in json.dumps(metadata)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "opposite_side",
        "wrong_kind",
        "wrong_channel",
        "bad_score",
        "wrong_endpoint",
    ],
)
def test_bad_or_ambiguous_saved_links_remain_explicit_and_never_guess(change, actual):
    record, source, target = actual
    link = record["cross_side_provenance"]["hierarchy"]["is_a"][0]
    source_alias = link["source_item_id"]
    if change == "missing":
        link["source_item_id"] = "UNRESOLVED-PRIVATE-ALIAS"
    elif change == "duplicate":
        rows = record["triple_attributions"]["hierarchy"]["is_a"]["source"]
        rows.append(copy.deepcopy(next(x for x in rows if x["item_id"] == source_alias)))
    elif change == "opposite_side":
        link["source_item_id"] = link["target_item_id"]
    elif change == "wrong_channel":
        rows = record["triple_attributions"]["hierarchy"]["is_a"]["source"]
        item = next(x for x in rows if x["item_id"] == source_alias)
        rows.remove(item)
        record["triple_attributions"]["similarity"]["source"].append(item)
    elif change == "bad_score":
        link["score"] = True
    elif change == "wrong_endpoint":
        record["cross_side_provenance"]["lexical"][0]["target_ref"] = "__source__"
    items = selected_evidence(record, source, target)
    if change == "wrong_kind":
        for item in items:
            if item["historical_item_alias"] == "legacy:" + source_alias:
                item["entity"] = {**source, "kind": "individual"}
    metadata = pair_evidence_metadata(record, items, source, target, ["artifact"])
    assert metadata["bridge_status"] == "recorded"
    assert metadata["comparison_resolution"]["recorded"] == 28
    assert metadata["comparison_resolution"]["resolved"] < 28
    assert len(metadata["comparison_records"]) == 28
    assert "UNRESOLVED-PRIVATE-ALIAS" not in json.dumps(metadata)
    for row in metadata["comparison_records"]:
        assert (row["bridge_id"] is not None) == (row["status"] == "resolved")
    PairEvidenceMetadata.model_validate(metadata)


def test_recorded_but_entirely_unresolvable_is_not_reported_as_not_recorded(actual):
    record, source, target = actual
    record["cross_side_provenance"] = {"hierarchy": record["cross_side_provenance"]["hierarchy"]}
    metadata = pair_evidence_metadata(record, [], source, target, ["artifact"])
    assert metadata["bridge_status"] == "recorded" and metadata["bridges"] == []
    assert metadata["comparison_resolution"]["unresolved"] == 6
    assert filter_pair_metadata(metadata, [])["bridge_status"] == "recorded"
    empty = pair_evidence_metadata({}, [], source, target, [])
    assert empty["bridge_status"] == "not_recorded" and empty["comparison_resolution"] is None


def test_actual_http_preserves_real_record_comparisons_and_policy_tombstones(actual, tmp_path):
    record, _, _ = actual
    root = tmp_path / "package"
    contexts = []
    for side in ("source", "target"):
        iri = record["src_iri" if side == "source" else "tgt_iri"]
        # Tiny declared signatures intentionally cannot resolve the real originals.
        contexts.append(
            build_context_package(
                core.load_snapshot(f"Ontology(Declaration(Class(<{iri}>)))".encode()), root / side
            )
        )
    run = tmp_path / "run"
    ExplanationStore.create(run).append([record])
    import_run(
        run,
        root / "decisions",
        source_ontology_version_id=contexts[0].ontology_version_id,
        target_ontology_version_id=contexts[1].ontology_version_id,
        run_id="retained-actual",
    )
    package = publish_bundle(
        root,
        ontologies={c.ontology_version_id: c.path.name for c in contexts},
        runs={"retained-actual": "decisions"},
    )
    frozen = file_hash(root / "decisions" / "decisions.sqlite")
    store = DecisionStore(root / "decisions")
    manifest = store.manifest()
    inventory = dependency_inventory_hash(manifest["dependencies"])
    assert inventory == manifest["dependency_inventory_hash"]
    assert (
        manifest["comparison_artifact_ref_semantics"]["legacy_and_default"]
        == "dependency_inventory_hash"
    )
    assert any("/shards/" in path for path in manifest["dependencies"])
    pair_id = store.candidates(record["src_iri"])["items"][0]["pair_id"]
    with TestClient(create_prepared_app(package)) as client:
        response = client.get("/api/v1/runs/retained-actual/pair", params={"pair_id": pair_id})
        assert response.status_code == 200, response.text
        metadata = response.json()["evidence_metadata"]
        assert all(
            row["artifact_refs"] == [inventory]
            for row in metadata["bridges"] + metadata["comparison_records"]
        )
        assert (
            metadata["comparison_resolution"]["recorded"]
            == metadata["comparison_resolution"]["resolved"]
            == 28
        )
        assert metadata["counts"]["selected"] == 29 and metadata["counts"]["not_exported"] == 29
        assert metadata["bridge_status"] == "withheld"
        cross = next(
            b
            for b in metadata["bridges"]
            if b["source_channel"] == "attributes" and b["target_channel"] == "hierarchy"
        )
        assert (
            cross["status"] == "withheld"
            and cross["source_fact_ids"] == cross["target_fact_ids"] == []
        )
        assert cross["target_family"] == "is_a"
        page = client.get(
            "/api/v1/runs/retained-actual/pair-evidence", params={"pair_id": pair_id, "limit": 100}
        )
        assert page.status_code == 200, page.text
        assert page.json()["total_count"] == 29
        assert all(item["historical_item_alias"] is None for item in page.json()["items"])
        assert "hierarchy-target-" not in response.text and "attribute-source-" not in response.text
    assert file_hash(root / "decisions" / "decisions.sqlite") == frozen
    # Neither the inventory contents nor its declaration can drift behind the
    # same verified SQLite bytes; legacy manifests remain separately supported.
    manifest_path = root / "decisions" / "decisions.manifest.json"
    saved = manifest_path.read_bytes()
    changed = copy.deepcopy(manifest)
    changed["dependencies"][next(iter(changed["dependencies"]))] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(DomainError, match="inventory changed"):
        DecisionStore(root / "decisions")
    changed = copy.deepcopy(manifest)
    del changed["dependency_inventory_hash"]
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(DomainError, match="inventory changed"):
        DecisionStore(root / "decisions")
    manifest_path.write_bytes(saved)
    DecisionStore(root / "decisions")
