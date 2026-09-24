"""Contract, recovery, generation and deployment-boundary regressions for prepared inspection."""

import json
import shutil
import subprocess
import sys
import zipfile

import pytest
from fastapi.testclient import TestClient

from exact_inspect.artifacts import (
    BoundedCache,
    BundleLibrary,
    atomic_json,
    export_archive,
    publish_bundle,
    validate_bundle,
)
from exact_inspect.contracts import (
    DomainError,
    EntityRef,
    Page,
    Scope,
    VisibilityPolicy,
    canonical_hash,
    decode_cursor,
    encode_cursor,
)
from exact_inspect.generation import (
    ExplanationJobs,
    GenerationProfile,
    comparison_packet,
    entity_packet,
)
from exact_inspect.preparation import (
    ExecutionLock,
    InputBinding,
    OntologyBinding,
    Preparation,
    reuse_plan,
    stage_identity,
    stage_readiness,
)
from exact_inspect.service import create_prepared_app


def _entity(iri="urn:test:source", ontology="sha256:" + "a" * 64):
    return EntityRef(ontology_version_id=ontology, iri=iri, kind="class")


def _fact(entity, text="A supplied definition.", category="definitions"):
    value = {
        "subject": entity.model_dump(),
        "predicate_iri": "urn:definition",
        "value": {
            "term_type": "literal",
            "lexical_form": text,
            "language": "en",
            "datatype": "http://www.w3.org/2001/XMLSchema#string",
        },
        "category": category,
        "interpretation": "asserted",
        "origins": [],
    }
    return {"fact_id": canonical_hash(value), **value}


def _packet(entity=None, policy=None):
    entity = entity or _entity()
    return entity_packet(
        entity,
        [_fact(entity)],
        context_hash=entity.ontology_version_id,
        policy=policy or VisibilityPolicy(),
    )


def _provider(packet, calls):
    def call(messages, profile):
        calls.append(messages)
        return {
            "model": profile.model,
            "provider": "fixture",
            "usage": {"cost": 0},
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "claims": [
                                    {
                                        "text": packet.facts[0]["value"]["lexical_form"],
                                        "fact_ids": [packet.facts[0]["fact_id"]],
                                        "category": "meaning",
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

    return call


def test_identity_preserves_punning_literal_language_and_policy():
    entity = _entity()
    assert canonical_hash(entity) != canonical_hash(
        entity.model_copy(update={"kind": "individual"})
    )
    fact = _fact(entity)
    changed = {**fact, "value": {**fact["value"], "language": "pt"}}
    assert canonical_hash(fact) != canonical_hash(changed)
    policy = VisibilityPolicy(categories=("definitions",))
    packet = entity_packet(
        entity,
        [fact, _fact(entity, "answer-bearing", "xrefs")],
        context_hash=entity.ontology_version_id,
        policy=policy,
    )
    assert len(packet.facts) == 1
    assert "answer-bearing" not in packet.model_dump_json()


def test_cursor_rejects_query_policy_and_snapshot_changes():
    scope = {"ontology": "a", "query": "x", "policy": "restricted"}
    cursor = encode_cursor(scope, ["label", "iri"])
    assert decode_cursor(cursor, scope) == ["label", "iri"]
    for key in scope:
        with pytest.raises(DomainError) as exc:
            decode_cursor(cursor, {**scope, key: "changed"})
        assert exc.value.status_code == 409
    with pytest.raises(DomainError):
        decode_cursor("invalid?", scope)


def test_page_distinguishes_unknown_count_and_absence():
    scope = Scope(
        ontology_version_id="a",
        context_revision="1",
        basis="root_document",
        visibility_policy_hash=canonical_hash("policy"),
        filter_id="all",
    )
    page = Page(items=[], returned_count=0, scope=scope, status="unavailable_source")
    assert page.total_count is None
    with pytest.raises(ValueError):
        Page(items=[1], returned_count=0, scope=scope)


def test_generation_replay_prompt_invalidation_and_candidate_independence(tmp_path):
    packet = _packet()
    profile = GenerationProfile(name="baseline", model="fixture/model")
    calls = []
    jobs = ExplanationJobs(tmp_path)
    first = jobs.generate(packet, profile, provider=_provider(packet, calls))
    assert first["grounding_status"] == "validated"
    assert jobs.generate(packet, profile, provider=_provider(packet, calls)) == first
    assert len(calls) == 1
    second = jobs.generate(
        packet, profile, provider=_provider(packet, calls), prompt="Revised prompt"
    )
    assert len(calls) == 2 and second["explanation_id"] != first["explanation_id"]
    assert "counterpart" not in json.dumps(packet.model_dump())
    assert "score" not in json.dumps(packet.model_dump())


def test_unknown_citations_and_unsupported_paraphrase_never_become_approved(tmp_path):
    packet = _packet()
    calls = []

    def invalid(messages, profile):
        calls.append(messages)
        return {
            "model": profile.model,
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "claims": [
                                    {
                                        "text": "The target definitively proves the source is equivalent.",
                                        "fact_ids": [canonical_hash("forbidden")],
                                        "category": "meaning",
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

    result = ExplanationJobs(tmp_path).generate(
        packet, GenerationProfile(name="test", model="fixture/model"), provider=invalid
    )
    assert len(calls) == 3
    assert result["manifest"]["status"] == "failed"
    assert result["claims"][0]["text"] == "A supplied definition."
    assert "definitively" not in json.dumps(result)


def test_response_saved_replays_after_validation_crash_and_copy(tmp_path, monkeypatch):
    import exact_inspect.generation as generation

    packet, calls = _packet(), []
    profile = GenerationProfile(name="test", model="fixture/model")
    jobs = ExplanationJobs(tmp_path / "original")
    original_grounding = generation.grounding

    def interrupted(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(generation, "grounding", interrupted)
    with pytest.raises(KeyboardInterrupt):
        jobs.generate(packet, profile, provider=_provider(packet, calls))
    shutil.copytree(tmp_path / "original", tmp_path / "copied")
    shutil.rmtree(tmp_path / "original")
    monkeypatch.setattr(generation, "grounding", original_grounding)
    resumed = ExplanationJobs(tmp_path / "copied").generate(
        packet, profile, provider=_provider(packet, calls)
    )
    assert len(calls) == 1
    assert resumed["manifest"]["status"] == "validated"


def test_comparison_rejects_unrestricted_profiles_and_is_score_blind(tmp_path):
    source, target = _packet(), _packet(_entity("urn:test:target"))
    profile = GenerationProfile(name="test", model="fixture/model")
    jobs = ExplanationJobs(tmp_path)
    profiles = [jobs.generate(p, profile, provider=_provider(p, [])) for p in (source, target)]
    packet = comparison_packet(source, target, profiles)
    assert (
        "score" not in packet.model_dump_json() and "candidate_rank" not in packet.model_dump_json()
    )
    restricted = _packet(policy=VisibilityPolicy(policy_id="restricted"))
    with pytest.raises(ValueError):
        comparison_packet(restricted, target, profiles)


def _bundle(path):
    path.mkdir()
    atomic_json(path / "data.json", {"inert": True})
    return publish_bundle(
        path, audience="development_demo", capabilities={"context": "not_exported"}
    )


def test_portable_import_reopen_and_failure_preserves_selection(tmp_path):
    package = _bundle(tmp_path / "original")
    archive = tmp_path / "bundle.zip"
    export_archive(package, archive)
    library = BundleLibrary(tmp_path / "library")
    imported = library.import_archive(archive)
    selected = library.select(imported.package_id)
    shutil.rmtree(package.parent)
    assert validate_bundle(selected).package_id == imported.package_id
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as stream:
        stream.writestr("../escape", "bad")
    with pytest.raises(DomainError):
        library.import_archive(tmp_path / "bad.zip")
    assert (
        json.loads((library.root / "selection.json").read_bytes())["package_id"]
        == imported.package_id
    )
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("kind", ["duplicate", "symlink", "bomb", "unmanifested"])
def test_archive_abuse_rejected(tmp_path, kind):
    import stat

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        if kind == "duplicate":
            stream.writestr("a", "a")
            with pytest.warns(UserWarning):
                stream.writestr("a", "b")
        elif kind == "symlink":
            entry = zipfile.ZipInfo("a")
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            stream.writestr(entry, "/tmp")
        elif kind == "bomb":
            stream.writestr("a", "0" * 100000)
        else:
            stream.writestr("not-a-manifest", "a")
    with pytest.raises(DomainError):
        BundleLibrary(tmp_path / "library", max_ratio=50).import_archive(archive)


def test_lru_bounds_bytes_entries_and_mutable_readers():
    cache = BoundedCache(max_entries=2, max_bytes=50)
    cache.put("a", {"value": []})
    cache.get("a")["value"].append("mutated")
    assert cache.get("a") == {"value": []}
    cache.put("b", {})
    cache.put("c", {})
    assert cache.get("a") is None and cache.bytes <= 50
    cache.put("huge", "x" * 100)
    assert cache.get("huge") is None


def test_hosted_profiles_have_no_import_and_local_import_is_inert(tmp_path):
    package = _bundle(tmp_path / "package")
    archive = tmp_path / "package.zip"
    export_archive(package, archive)
    demo = TestClient(create_prepared_app(package, profile="public_demo"))
    assert demo.post("/api/v1/bundles/import", content=b"ignored").status_code == 404
    assert "path" not in demo.get("/api/v1/health").text
    local = TestClient(create_prepared_app(None, library_dir=tmp_path / "library"))
    assert local.get("/api/v1/health").json()["status"] == "not_requested"
    denied = local.post("/api/v1/bundles/import", headers={"Origin": "https://attacker.invalid"})
    assert denied.status_code == 403
    response = local.post(
        "/api/v1/bundles/import",
        content=archive.read_bytes(),
        headers={"Content-Type": "application/zip"},
    )
    assert response.status_code == 200, response.text
    selected = local.post("/api/v1/bundles/" + response.json()["package_id"] + "/select")
    assert selected.status_code == 200
    assert local.get("/api/v1/health").json()["status"] == "available"


def test_serve_imports_no_training_or_ontology_runtime(tmp_path):
    package = _bundle(tmp_path / "package")
    code = "from exact_inspect.service import create_prepared_app; import sys; from pathlib import Path; create_prepared_app(Path(sys.argv[1])); assert not any(x in sys.modules for x in ['torch','transformers','pyowl_core','sentence_transformers'])"
    subprocess.run([sys.executable, "-c", code, str(package)], check=True)


def test_stage_locks_and_selective_repair_do_not_require_future_bindings(tmp_path):
    source = tmp_path / "source.owl"
    source.write_text("bytes")
    from exact_inspect.contracts import file_hash

    ontology = OntologyBinding(
        name="source", root=InputBinding(path=str(source), sha256=file_hash(source), size=5)
    )
    lock = ExecutionLock(
        design_revision="1", ontologies=[ontology], runtime={"pyowl-core": "0.2.1"}
    )
    readiness = stage_readiness(lock)
    assert readiness["context-index"]["status"] == "ready"
    assert readiness["run-import"]["status"] == readiness["profiles"]["status"] == "blocked_input"
    bound = lock.model_copy(
        update={"profile": {"name": "one", "model": "model-a"}, "entities": [{"id": "x"}]}
    )
    changed = bound.model_copy(update={"profile": {"name": "two", "model": "model-b"}})
    plan = reuse_plan(bound, changed, cause="Model binding update")
    actions = {r["stage"]: r["action"] for r in plan["stages"]}
    assert actions["context-index"] == actions["run-import"] == "reuse"
    assert actions["profiles"] == actions["comparisons"] == "rebuild"
    relocated = ontology.model_copy(
        update={"root": ontology.root.model_copy(update={"path": "/new/location.owl"})}
    )
    assert stage_identity(lock, "context-index") == stage_identity(
        lock.model_copy(update={"ontologies": [relocated]}), "context-index"
    )


def test_completed_stage_relocates_and_corruption_fails_closed(tmp_path):
    source = tmp_path / "source.owl"
    source.write_text("fixture")
    from exact_inspect.contracts import file_hash

    lock = ExecutionLock(
        design_revision="synthetic",
        ontologies=[
            OntologyBinding(
                name="fixture",
                root=InputBinding(path=str(source), sha256=file_hash(source), size=7),
            )
        ],
    )
    original = tmp_path / "original"
    Preparation(lock, original, input_root=tmp_path).run(["acquire-verify"])
    copied = tmp_path / "copied"
    Preparation(lock, copied, input_root=tmp_path, resume_from=original).run(["acquire-verify"])
    shutil.rmtree(original)
    report = Preparation(lock, copied, input_root=tmp_path).run(["acquire-verify"])
    stage = copied / report["outputs"]["acquire-verify"]
    (stage / "verified.json").write_text("corrupt")
    with pytest.raises(DomainError) as exc:
        Preparation(lock, copied, input_root=tmp_path).run(["acquire-verify"])
    assert exc.value.envelope.code == "corrupt_stage"
