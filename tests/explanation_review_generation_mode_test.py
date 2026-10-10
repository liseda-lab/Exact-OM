"""Local typed preparation cannot silently dispatch or reuse a provider-mode root."""

import json
import shutil
from types import SimpleNamespace

import pytest

from exact_inspect.artifacts import atomic_json, publish_bundle, validate_bundle
from exact_inspect.contracts import VisibilityPolicy, file_hash
from exact_inspect.study.models import Publish
from tools import prepare_review_bundle as preparation
from tools.build_explanation_v2_fixture import publication_v2


def preparation_fixture(tmp_path, monkeypatch):
    frozen = publication_v2(tmp_path, forms_version="exact-study-forms/3").model_dump(mode="json")
    atomic_json(tmp_path / "practice-publication.json", frozen)
    policy = VisibilityPolicy.model_validate(frozen["definition"]["visibility_policy"])
    scope = next(s for s in frozen["definition"]["workspace_scopes"] if s["kind"] == "case")
    source_version = frozen["definition"]["cases"][0]["source"]["ontology_version_id"]
    package = tmp_path / "package"
    ontologies = {}
    for ref in scope["contexts"]:
        label = "source" if ref["ontology_version_id"] == source_version else "target"
        relative = f"ontologies/{label}"
        shutil.copytree(tmp_path / ref["path"], package / relative)
        manifest_path = package / relative / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["name"] = "NCIT synthetic test" if label == "source" else "DOID synthetic test"
        atomic_json(manifest_path, manifest)
        ontologies[ref["ontology_version_id"]] = relative
    publish_bundle(package, policy=policy, ontologies=ontologies)
    run = tmp_path / "run"
    atomic_json(run / "run_manifest.json", {"fixture": "synthetic producer input"})
    atomic_json(
        run / "experiment_manifest.json", {"status": "complete", "split_role": "development"}
    )
    atomic_json(
        run / "source_decisions.json",
        {
            "records": [
                {
                    "Src": case["source"]["iri"],
                    "candidates": [
                        {
                            "target": c["entity"]["iri"],
                            "S_final": c["score"],
                            "threshold_positive": True,
                            "emitted": True,
                        }
                        for c in case["candidates"]
                    ],
                }
                for case in frozen["definition"]["cases"]
            ]
        },
    )
    # This boundary test uses invented native ontology data. Only the matcher run
    # adapter is stubbed; all description/resource/workspace preparation is real.
    reader = SimpleNamespace(iter_explanations=lambda: [], explanations_for=lambda _: [])
    monkeypatch.setattr(preparation.RunReader, "open", lambda _: reader)
    monkeypatch.setattr(
        preparation,
        "prepared_private_run",
        lambda *args, **kwargs: SimpleNamespace(
            manifest=lambda: {"counts": {"selected_evidence": 10, "provenance_unavailable": 9}}
        ),
    )
    from exact_inspect.contracts import canonical_hash
    from exact_inspect.evidence_metadata import pair_evidence_metadata

    pairs = {}
    for case in frozen["definition"]["cases"]:
        for candidate in case["candidates"]:
            source, target = case["source"], candidate["entity"]
            pairs[
                canonical_hash({"run_id": "test-recorded-run", "source": source, "target": target})
            ] = {
                "source": source,
                "target": target,
                "evidence_metadata": {
                    **pair_evidence_metadata({}, [], source, target, []),
                    "scope": "prepared_visibility_policy",
                },
            }
    portable = SimpleNamespace(
        manifest=lambda: {"run_id": "test-recorded-run"},
        pair=lambda identity: pairs[identity],
        evidence=lambda _: [],
    )
    monkeypatch.setattr(preparation, "prepared_policy_run", lambda *args, **kwargs: portable)
    # A valid portable locator still needs to exist and be content-bound.
    atomic_json(package / "runs/review/fixture.json", {"fixture": "stubbed run adapter"})

    return run


def test_typed_only_prepares_real_artifacts_without_provider_jobs(tmp_path, monkeypatch):
    run = preparation_fixture(tmp_path, monkeypatch)
    package = tmp_path / "package"

    def forbidden_jobs(*args, **kwargs):
        pytest.fail("Typed-only preparation must not construct provider jobs")

    monkeypatch.setattr(preparation, "ExplanationJobs", forbidden_jobs)
    publication_path = preparation.prepare(
        tmp_path,
        run,
        count=4,
        generate_pairs=12,
        revision="typed-only-test",
        generation_mode="typed-only",
    )
    publication = Publish.model_validate_json(publication_path.read_bytes())
    assert all(key.original_production_ranks == {} for key in publication.case_keys)
    assert any(
        "not performed" in reason for reason in publication.definition.review_bundle.limitations
    )
    limitations = publication.definition.review_bundle.limitations
    assert any("selected typed review entities" in text for text in limitations)
    assert any("Most run-wide" in text for text in limitations)
    assert any("Rejected candidates and differing final decisions" in text for text in limitations)
    catalogue = json.loads((tmp_path / "catalogue.json").read_bytes())
    assert catalogue["generation_mode"] == "typed-only"
    assert catalogue["provider_batch_status"] == "not_performed"
    assert not (tmp_path / "generation").exists()
    assert not (tmp_path / "independent-reviews").exists()
    bundle = validate_bundle(package / "package.json")
    assert bundle.provenance["generation_mode"] == "typed-only"
    for relative in bundle.explanations.values():
        value = json.loads((package / relative).read_bytes())
        assert value["manifest"]["requested_model"] == "none-typed-renderer"
        assert value["manifest"]["response_hash"] is None
        assert value.get("independent_review") is None
    before = {p: file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="inputs changed"):
        preparation.prepare(
            tmp_path, run, count=4, revision="typed-only-test", generation_mode="provider"
        )
    assert {p: file_hash(p) for p in before} == before


def test_review_decision_limitation_requires_explicit_recorded_flags():
    def limitations(rows, counts=None):
        return preparation.review_scope_limitations(
            {"cases": [{"recorded_candidates": rows}]}, counts or {}
        )

    assert not any("threshold-positive" in text for text in limitations([{"S_final": 1.0}]))
    assert not any(
        "threshold-positive" in text
        for text in limitations(
            [
                {"threshold_positive": True, "emitted": True},
                {"threshold_positive": False, "emitted": False},
            ]
        )
    )
    assert not any(
        "Most run-wide" in text
        for text in limitations([], {"selected_evidence": 10, "provenance_unavailable": 4})
    )
