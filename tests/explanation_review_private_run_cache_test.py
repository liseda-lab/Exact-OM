"""Adapter/resolver changes produce new private indexes without replacing old evidence."""

from types import SimpleNamespace

import pytest

from exact.runs import ExplanationStore
from exact_inspect import decisions
from exact_inspect.contracts import file_hash
from tools.review_private_run_cache import prepared_private_run


def test_private_import_versions_adapter_and_typed_resolver_and_preserves_old_bytes(
    tmp_path, monkeypatch
):
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
    contexts = {}
    for version in ["source", "target"]:
        database = tmp_path / (version + ".sqlite")
        database.write_bytes(("exact synthetic " + version).encode())
        contexts[version] = SimpleNamespace(
            database=database,
            manifest={
                "artifacts": {"context.sqlite": file_hash(database)},
                "identity": {"root_sha256": "sha256:" + "a" * 64},
            },
        )
    entities = [
        {"ontology_version_id": "source", "iri": "urn:S", "kind": "class"},
        {"ontology_version_id": "target", "iri": "urn:T", "kind": "class"},
    ]
    real_import = decisions.import_run
    calls = []

    def importer(*args, **kwargs):
        calls.append(True)
        return real_import(*args, **kwargs)

    monkeypatch.setattr(decisions, "import_run", importer)

    def prepare():
        return prepared_private_run(
            tmp_path,
            raw,
            contexts=contexts,
            source_version="source",
            target_version="target",
            selected_entities=entities,
        )

    first = prepare()
    original = {p: file_hash(p) for p in first.database.parent.iterdir() if p.is_file()}
    assert prepare().database == first.database and len(calls) == 1
    monkeypatch.setattr(decisions, "ADAPTER_REVISION", "decision-adapter/synthetic-next")
    upgraded = prepare()
    assert upgraded.database != first.database and len(calls) == 2
    assert upgraded.manifest()["revision"] != first.manifest()["revision"]
    assert {p: file_hash(p) for p in original} == original
    # Typed identities, rather than just IRI, determine which originals resolve.
    entities[0] = {**entities[0], "kind": "object_property"}
    typed = prepare()
    assert typed.database != upgraded.database and len(calls) == 3
    assert typed.manifest()["context_revision"] != upgraded.manifest()["context_revision"]
    assert {p: file_hash(p) for p in original} == original
    # Consumed source dependencies cannot silently drift behind a valid index.
    relative = next(k for k in typed.manifest()["dependencies"] if "/shards/" in k)
    source = raw / relative
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Consumed matcher artifact changed"):
        prepare()
    assert len(calls) == 3
