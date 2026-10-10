"""Bounded preview and generation share identities without mutating earlier attempts."""

import json

import pytest

from exact_inspect.artifacts import atomic_json
from exact_inspect.contracts import EntityRef, canonical_hash, file_hash
from exact_inspect.descriptions import (
    COMPARISON_PROSE_PROMPT_V1,
    PROFILE_PROSE_PROMPT_V1,
)
from exact_inspect.generation import (
    ExplanationJobs,
    FactPacket,
    GenerationProfile,
    comparison_packet,
)
from tools import review_generation_batch as batch
from tools.review_generation_attempt import (
    ReviewExplanationJobs,
    bind_generation_contract,
)


def batch_fixture(root):
    profile = GenerationProfile(name="review-description-v2", model="synthetic")
    packets, profiles, calls = [], [], []

    def provider(*args):
        calls.append(True)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"claims": [], "limitations": [], "relation": None})
                    }
                }
            ]
        }

    jobs = ExplanationJobs(root / "generation")
    for iri in ["urn:A", "urn:B"]:
        packet = FactPacket(
            task="entity_profile",
            entities=[EntityRef(ontology_version_id="urn:context", iri=iri, kind="class")],
            context_hashes=["sha256:" + "a" * 64],
            policy_hash="sha256:" + "b" * 64,
            facts=[],
            missingness=[],
            selection={"description_contract": "exact-description/2", "description_prose": True},
        )
        packets.append(packet)
        profiles.append(
            jobs.generate(
                packet, profile, provider=provider, prompt=PROFILE_PROSE_PROMPT_V1, max_repairs=0
            )
        )
    jobs.generate(
        comparison_packet(*packets, profiles),
        profile,
        provider=provider,
        prompt=COMPARISON_PROSE_PROMPT_V1,
        max_repairs=0,
    )
    assert len(calls) == 3
    context = root / "package/ontologies/one/manifest.json"
    atomic_json(context, {"fixture": "immutable context metadata"})
    atomic_json(
        root / "preparation-inputs.json",
        {
            "description_preparation": "independent-prose/1",
            "contexts": {"ontologies/one/manifest.json": file_hash(context)},
            "run_manifest": "sha256:" + "c" * 64,
            "generate_pairs": 1,
        },
    )
    atomic_json(
        root / "selection.json",
        {
            "run_manifest_sha256": "sha256:" + "c" * 64,
            "cases": [
                {
                    "case_id": "case-one",
                    "source": packets[0].entities[0].model_dump(),
                    "targets": [packets[1].entities[0].model_dump()],
                }
            ],
        },
    )
    return profile


def test_preview_binds_corrected_prompts_and_execute_reuses_only_new_attempt(tmp_path, monkeypatch):
    profile = batch_fixture(tmp_path)
    frozen = {p: file_hash(p) for p in (tmp_path / "generation").rglob("*") if p.is_file()}
    plan = batch.batch_plan(tmp_path, 3)
    assert plan["counts"] == {"entity_profile": 2, "pair_comparison": 1}
    assert plan["maximum_new_http_attempts"] == 3
    assert not (tmp_path / "generation-attempt-3").exists()
    assert plan["contract"]["max_http_retries"] == plan["contract"]["max_semantic_repairs"] == 0
    calls = []

    def provider(self, *args):
        calls.append(True)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"claims": [], "limitations": [], "relation": None})
                    }
                }
            ]
        }

    monkeypatch.setattr(ReviewExplanationJobs, "_provider", provider)
    with pytest.raises(ValueError, match="approved plan hash"):
        batch.execute_plan(plan, "sha256:" + "0" * 64)
    marker = batch.execute_plan(plan, canonical_hash(plan))
    assert len(calls) == 3
    receipt = json.loads(marker.read_bytes())
    assert receipt["counts"] == plan["counts"]
    assert len(receipt["exact_jobs"]) == 3
    assert batch.execute_plan(plan, canonical_hash(plan)) == marker
    assert len(calls) == 3
    packet = batch._frozen_requests(tmp_path, 1)[plan["profiles"][0]["source_request"]][1]
    cached = ReviewExplanationJobs(tmp_path / "generation-attempt-3")
    cached.cache_only = True
    assert (
        cached.generate(packet, profile, max_repairs=0)["explanation_id"]
        == plan["profiles"][0]["job_id"]
    )
    assert cached.generate(packet, profile)["explanation_id"] == plan["profiles"][0]["job_id"]
    with pytest.raises(ValueError, match="cache-only assembly cannot dispatch"):
        cached.generate(packet.model_copy(update={"language": "pt"}), profile, max_repairs=0)
    assert len(calls) == 3
    assert {p: file_hash(p) for p in frozen} == frozen
    with pytest.raises(ValueError, match="another contract"):
        bind_generation_contract(tmp_path, 1, profile)
    assert not (tmp_path / "generation-attempts/attempt-1-contract.json").exists()


def test_review_transport_explicitly_disables_wire_retries(tmp_path, monkeypatch):
    import exact.llm.routing as routing

    seen = []

    class Client:
        def __init__(self):
            self._client = self
            self.max_retries = 99
            self.retry_unknown_requests = True

        def chat_completion(self, *args, **kwargs):
            seen.append((self.max_retries, self.retry_unknown_requests, self.ledger_dir))
            return {"synthetic": True}

        def close(self):
            seen.append("closed")

    monkeypatch.setattr(routing, "OpenRouterClient", Client)
    monkeypatch.delenv("EXACT_OPENROUTER_RETRY_UNKNOWN", raising=False)
    jobs = ReviewExplanationJobs(tmp_path / "generation-attempt-3")
    assert jobs._provider([], GenerationProfile(name="synthetic", model="test")) == {
        "synthetic": True
    }
    assert seen == [(0, False, jobs.root / "provider-ledger"), "closed"]
    monkeypatch.setenv("EXACT_OPENROUTER_RETRY_UNKNOWN", "1")
    with pytest.raises(ValueError, match="unknown-delivery"):
        jobs._provider([], GenerationProfile(name="synthetic", model="test"))
    assert len(seen) == 2


@pytest.mark.parametrize("cache_only", [False, True])
@pytest.mark.parametrize(
    "corruption",
    ["relocated_result", "request_identity", "receipt_request_hash", "result_identity"],
)
def test_generation_cache_checks_request_result_lineage_before_replay(
    tmp_path, monkeypatch, cache_only, corruption
):
    import shutil

    from tools.review_generation_attempt import previous_attempt_summaries

    batch_fixture(tmp_path)
    plan = batch.batch_plan(tmp_path, 3)
    calls = []

    def provider(self, *args):
        calls.append(True)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"claims": [], "limitations": [], "relation": None})
                    }
                }
            ]
        }

    monkeypatch.setattr(ReviewExplanationJobs, "_provider", provider)
    batch.execute_plan(plan, canonical_hash(plan))
    packet = batch._frozen_requests(tmp_path, 1)[plan["profiles"][0]["source_request"]][1]
    profile = batch.frozen_profile(plan["contract"]["profile"])
    jobs = ReviewExplanationJobs(tmp_path / "generation-attempt-3")
    original = jobs.generate(packet, profile)
    other = jobs.generate(packet, profile.model_copy(update={"model": "other-model"}))
    directory = jobs.root / original["explanation_id"][7:]
    if corruption == "relocated_result":
        for name in ["explanation.json", "receipt.json"]:
            shutil.copyfile(jobs.root / other["explanation_id"][7:] / name, directory / name)
    elif corruption == "request_identity":
        value = json.loads((directory / "request.json").read_bytes())
        value["profile"]["model"] = "foreign"
        atomic_json(directory / "request.json", value)
    elif corruption == "receipt_request_hash":
        value = json.loads((directory / "receipt.json").read_bytes())
        value["request_hash"] = other["explanation_id"]
        atomic_json(directory / "receipt.json", value)
    else:
        value = json.loads((directory / "explanation.json").read_bytes())
        value["explanation_id"] = other["explanation_id"]
        atomic_json(directory / "explanation.json", value)
        receipt = json.loads((directory / "receipt.json").read_bytes())
        receipt["output_hash"] = canonical_hash(value)
        atomic_json(directory / "receipt.json", receipt)
    before = {p: file_hash(p) for p in directory.iterdir() if p.is_file()}
    count = len(calls)
    jobs.cache_only = cache_only
    with pytest.raises(ValueError, match="lineage"):
        jobs.generate(packet, profile)
    with pytest.raises(ValueError, match="lineage"):
        previous_attempt_summaries(tmp_path, 4)
    assert len(calls) == count
    assert {p: file_hash(p) for p in before} == before
