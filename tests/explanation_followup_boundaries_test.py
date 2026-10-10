"""S1 follow-up integrity, bounded dispatch and retained-origin regressions."""

import json
from types import SimpleNamespace

import pytest

from exact.runs import ExplanationStore
from exact_inspect import decisions
from exact_inspect.artifacts import publish_bundle
from exact_inspect.contracts import (
    DomainError,
    VisibilityPolicy,
    canonical_hash,
    file_hash,
)
from exact_inspect.service import PreparedService
from tools import review_generation_batch as batch
from tools.refresh_review_evidence import recover_literals
from tools.review_generation_attempt import ReviewExplanationJobs
from tools.review_ontology_admission import pinned_imports


def test_first_decision_read_reuses_readiness_and_detects_replacement(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    ExplanationStore.create(raw).append(
        [
            {
                "src_iri": "urn:S",
                "tgt_iri": "urn:T",
                "src_kind": "class",
                "tgt_kind": "class",
                "confidences": {"S_final": 0.7},
            }
        ]
    )
    root = tmp_path / "package"
    decisions.import_run(
        raw, root / "run", source_ontology_version_id="source", target_ontology_version_id="target"
    )
    package = publish_bundle(root, policy=VisibilityPolicy(), runs={"run": "run"})
    service = PreparedService(package)
    real_hash = decisions.file_hash
    calls = []

    def count(path):
        calls.append(path)
        return real_hash(path)

    monkeypatch.setattr(decisions, "file_hash", count)
    store = service.run("run")
    assert store.sources()["total_count"] == 1
    assert calls == []  # Before the fix the entire index was hashed again here.
    path = root / "run/decisions.sqlite"
    data = path.read_bytes()
    path.unlink()
    path.write_bytes(data + b"tampered")
    with pytest.raises(DomainError, match="changed after verification"):
        store.sources()
    with pytest.raises(DomainError):
        PreparedService(package)


@pytest.mark.parametrize("response", [None, {"choices": [{"message": {"content": "not JSON"}}]}])
def test_failed_real_preflight_stops_before_remaining_requests(tmp_path, monkeypatch, response):
    from tests.explanation_review_generation_batch_test import batch_fixture

    batch_fixture(tmp_path)
    plan = batch.batch_plan(tmp_path, 3)
    calls = []

    def provider(self, *args):
        calls.append(True)
        if response is None:
            raise RuntimeError("Provider rejected request")
        return response

    monkeypatch.setattr(ReviewExplanationJobs, "_provider", provider)
    with pytest.raises(ValueError, match="no remaining jobs dispatched"):
        batch.execute_plan(plan, canonical_hash(plan), preflight=True)
    assert len(calls) == 1
    # An explicit repeat preserves the failed cached attempt; it never retries HTTP.
    with pytest.raises(ValueError, match="no remaining jobs dispatched"):
        batch.execute_plan(plan, canonical_hash(plan), preflight=True)
    assert len(calls) == 1
    assert not (tmp_path / "generation-dispatch-complete-attempt-3.json").exists()


def test_literal_refresh_preserves_recorded_identity_and_projection():
    source = {
        "evidence_id": "saved",
        "feature_id": "feature",
        "entity": {"ontology_version_id": "v", "iri": "urn:S", "kind": "class"},
        "semantic_terms": {"property_iri": "urn:definition", "value": "saved lexical value"},
        "fact_ids": [],
        "graph_projection": "not_recorded",
    }
    seen = []

    def resolve(terms, entity):
        seen.append((terms, entity))
        return {"fact_ids": ["original"]}

    result = recover_literals([source], {"v": SimpleNamespace(resolve_feature=resolve)})[0]
    assert result["fact_ids"] == ["original"] and source["fact_ids"] == []
    for key in ("evidence_id", "feature_id", "entity", "semantic_terms", "graph_projection"):
        assert result[key] == source[key]
    assert seen == [(source["semantic_terms"], source["entity"])]


def test_import_relocation_checks_bytes_and_symlink_escape(tmp_path):
    root = tmp_path / "root"
    (root / "imports").mkdir(parents=True)
    path = root / "imports/one.owl"
    path.write_text("Ontology()")
    manifest = {"urn:import": {"path": "/remote/imports/one.owl", "sha256": file_hash(path)}}
    resolver, receipt = pinned_imports(root, manifest)
    assert resolver == {"urn:import": path}
    assert receipt["urn:import"]["path"] == "imports/one.owl"
    path.write_text("changed")
    with pytest.raises(ValueError, match="Pinned ontology import"):
        pinned_imports(root, manifest)
    outside = tmp_path / "outside.owl"
    outside.write_text("Ontology()")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="Pinned ontology import"):
        pinned_imports(root, manifest)


def test_admission_hash_reuse_is_stat_bound_and_detects_mid_read_change(tmp_path, monkeypatch):
    from tools import review_environment as review

    path = tmp_path / "input"
    path.write_text("first")
    inventory = review.AdmissionHashes()
    first = inventory(path)
    assert inventory(path) == first and inventory.bytes_read == 5
    path.write_text("other")
    assert inventory(path) != first and inventory.bytes_read == 10
    original = review.file_hash

    def changing(value):
        digest = original(value)
        value.write_text("changed during checksum")
        return digest

    monkeypatch.setattr(review, "file_hash", changing)
    with pytest.raises(review.ReviewError, match="changed during admission"):
        review.AdmissionHashes()(path)
