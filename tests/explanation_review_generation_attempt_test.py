"""Recovery is explicit and isolated from immutable failed attempt caches."""

import json

import pyowl_core as core
import pytest

from exact_inspect.context import build_context_package
from exact_inspect.contracts import EntityRef, VisibilityPolicy, file_hash
from exact_inspect.generation import ExplanationJobs, GenerationProfile, entity_packet
from tools.review_generation_attempt import (
    attempt_paths,
    bind_attempt,
    previous_attempt_summaries,
)


def test_explicit_second_attempt_preserves_failed_first_cache_and_identity(tmp_path):
    context = build_context_package(
        core.load_snapshot(
            b'Ontology(Declaration(Class(<urn:A>)) AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "A label"))'
        ),
        tmp_path / "context",
    )
    entity = EntityRef(ontology_version_id=context.ontology_version_id, iri="urn:A", kind="class")
    policy = VisibilityPolicy()
    facts = context.facts(entity, category="labels", policy=policy)["items"]
    packet = entity_packet(
        entity, facts, context_hash=context.manifest["artifacts"]["context.sqlite"], policy=policy
    )
    profile = GenerationProfile(name="test", model="test")
    inputs = {"description_preparation": "independent-prose/1", "frozen_input": "same"}
    bind_attempt(tmp_path, 1, inputs)
    first = attempt_paths(tmp_path, 1)
    assert first["generation"] == tmp_path / "generation"
    calls = []

    def failure(*args):
        calls.append("failed")
        raise RuntimeError("Synthetic HTTP401 failure")

    def success(*args):
        calls.append("response")
        return {
            "model": "test",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "claims": [
                                    {
                                        "text": "A label",
                                        "fact_ids": [facts[0]["fact_id"]],
                                        "category": "key_fact",
                                    }
                                ],
                                "limitations": [],
                                "relation": None,
                            }
                        )
                    }
                }
            ],
        }

    failed = ExplanationJobs(first["generation"]).generate(
        packet, profile, provider=failure, max_repairs=0
    )
    assert failed["manifest"]["status"] == "failed"
    frozen = {p: file_hash(p) for p in first["generation"].rglob("*") if p.is_file()}
    assert (
        ExplanationJobs(first["generation"]).generate(
            packet, profile, provider=success, max_repairs=0
        )
        == failed
    )
    assert calls == ["failed"]

    bind_attempt(tmp_path, 2, inputs)
    second = attempt_paths(tmp_path, 2)
    assert second["generation"].name == "generation-attempt-2"
    assert second["requests"].name == "review-requests-attempt-2"
    assert second["reviews"].name == "independent-reviews-attempt-2"
    recovered = ExplanationJobs(second["generation"]).generate(
        packet, profile, provider=success, max_repairs=0
    )
    assert recovered["manifest"]["status"] == "validated"
    assert recovered["explanation_id"] == failed["explanation_id"]
    assert calls == ["failed", "response"]
    prior = previous_attempt_summaries(tmp_path, 2)
    assert len(prior) == 1 and prior[0]["generation_attempt"] == 1
    assert prior[0]["cached_job_statuses"] == {"failed": 1}
    assert prior[0]["provider_attempts"] is None  # A test callback is not a wire ledger.
    assert "served_artifacts" not in prior[0]
    assert {p: file_hash(p) for p in frozen} == frozen
    with pytest.raises(ValueError, match="another preparation"):
        bind_attempt(tmp_path, 2, {"frozen_input": "changed"})


@pytest.mark.parametrize("attempt", [0, -1, True, "2"])
def test_generation_attempt_requires_explicit_positive_integer(tmp_path, attempt):
    with pytest.raises(ValueError, match="positive integer"):
        attempt_paths(tmp_path, attempt)


def test_recovery_after_partial_scope_preserves_prior_outputs_and_resume_is_cached(
    tmp_path, monkeypatch
):
    from exact_inspect.study.models import Publish
    from tests.explanation_review_generation_mode_test import preparation_fixture
    from tools import prepare_review_bundle as preparation
    from tools.review_generation_attempt import ReviewExplanationJobs

    run = preparation_fixture(tmp_path, monkeypatch)
    calls, scopes = [], []
    active_attempt = 1
    real_build = preparation.build_workspace_scope

    class OfflineJobs(ReviewExplanationJobs):
        def _provider(self, messages, profile):
            calls.append((active_attempt, messages[0]["content"]))
            if active_attempt == 1:
                raise RuntimeError("Synthetic authentication failure")
            # A new actual saved response changes metadata and source job bytes.
            # Empty prose deliberately requires no independent claim approvals.
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"claims": [], "limitations": [], "relation": None}
                            )
                        }
                    }
                ]
            }

    def interrupted_build(*args, **kwargs):
        if scopes:
            raise RuntimeError("Synthetic interruption after first completed case scope")
        result = real_build(*args, **kwargs)
        scopes.append(result)
        return result

    monkeypatch.setattr(preparation, "ExplanationJobs", OfflineJobs)
    monkeypatch.setattr(preparation, "build_workspace_scope", interrupted_build)
    with pytest.raises(RuntimeError, match="Synthetic interruption"):
        preparation.prepare(tmp_path, run, count=4, revision="recover-test")
    assert len(list((tmp_path / "prepared-scopes").glob("*.input.json"))) == 1
    assert not (tmp_path / "publication.json").exists()
    paths = attempt_paths(tmp_path, 1)
    frozen = {
        p: file_hash(p)
        for folder in [
            paths["generation"],
            paths["scopes"],
            paths["dependencies"],
            paths["explanations"],
        ]
        for p in folder.rglob("*")
        if p.is_file()
    }
    first_scope = scopes[0]
    frozen[tmp_path / first_scope.index_path] = file_hash(tmp_path / first_scope.index_path)
    first_asset = tmp_path / "review-01-explanation.json"
    frozen[first_asset] = file_hash(first_asset)
    active_attempt = 2
    monkeypatch.setattr(preparation, "build_workspace_scope", real_build)
    result = preparation.prepare(
        tmp_path, run, count=4, revision="recover-test", generation_attempt=2
    )
    publication = Publish.model_validate_json(result.read_bytes())
    current_scopes = [s for s in publication.definition.workspace_scopes if s.kind == "case"]
    assert len(current_scopes) == 4
    assert all(s.scope_id.startswith("attempt-2-") for s in current_scopes)
    assert first_scope.index_path not in [s.index_path for s in current_scopes]
    assert all(
        a.path.startswith("description-assets-attempt-2/")
        for a in publication.definition.assets
        if a.kind == "explanation" and not a.asset_id.startswith("practice-")
    )
    assert {p: file_hash(p) for p in frozen} == frozen
    assert any(a == 2 for a, _ in calls)
    number = len(calls)
    assert (
        preparation.prepare(tmp_path, run, count=4, revision="recover-test", generation_attempt=2)
        == result
    )
    assert len(calls) == number
