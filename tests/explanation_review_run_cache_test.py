"""Portable run reuse verifies real SQLite bytes, source, exporter and context bindings."""

import json
from types import SimpleNamespace

from exact.runs import ExplanationStore
from exact_inspect.artifacts import atomic_json
from exact_inspect.contracts import VisibilityPolicy, file_hash
from exact_inspect.decisions import DecisionStore, import_run
from exact_inspect.decisions_export import export_policy_run
from tools.review_run_cache import prepared_policy_run


def test_policy_export_reuses_only_complete_matching_frozen_output(tmp_path):
    raw_run = tmp_path / "raw"
    ExplanationStore.create(raw_run).append(
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
    import_run(
        raw_run,
        tmp_path / "private",
        source_ontology_version_id="source",
        target_ontology_version_id="target",
    )
    source = DecisionStore(tmp_path / "private")
    policy = VisibilityPolicy()
    # No selected evidence needs context lookup in this native decision fixture;
    # the context bytes still form part of the preparation checkpoint identity.
    context_bytes = tmp_path / "context.sqlite"
    context_bytes.write_bytes(b"first context revision")
    context = SimpleNamespace(
        database=context_bytes,
        manifest={
            "artifacts": {"context.sqlite": file_hash(context_bytes)},
            "policy_filter": {"policy_hash": policy.policy_hash},
        },
    )
    destination, checkpoint = tmp_path / "export", tmp_path / "export-inputs.json"
    calls = []

    def exporter(*args, **kwargs):
        calls.append(True)
        return export_policy_run(*args, **kwargs)

    def prepare(adopt=False):
        return prepared_policy_run(
            source,
            destination,
            contexts={"source": context},
            policy=policy,
            checkpoint=checkpoint,
            adopt_existing=adopt,
            exporter=exporter,
        )

    result = prepare()
    assert len(calls) == 1
    original = file_hash(result.database)
    assert prepare().manifest() == result.manifest()
    assert len(calls) == 1 and file_hash(result.database) == original
    checkpoint.unlink()
    assert prepare(adopt=True).manifest() == result.manifest()
    assert len(calls) == 1 and checkpoint.is_file()

    # A context change is not covered by the generic exporter revision alone.
    context_bytes.write_bytes(b"second context revision")
    context.manifest["artifacts"]["context.sqlite"] = file_hash(context_bytes)
    prepare()
    assert len(calls) == 2
    saved_binding = json.loads(checkpoint.read_bytes())
    assert saved_binding["context_hashes"]["source"] == file_hash(context_bytes)

    for changed in ("source", "policy", "exporter", "database", "partial", "no_adoption"):
        manifest_path = destination / "decisions.manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        if changed == "source":
            manifest["policy_filter"]["source_revision"] = "sha256:" + "0" * 64
            atomic_json(manifest_path, manifest)
        elif changed == "policy":
            manifest["policy_filter"]["policy_hash"] = "sha256:" + "0" * 64
            atomic_json(manifest_path, manifest)
        elif changed == "exporter":
            manifest["revision"] = "sha256:" + "0" * 64
            atomic_json(manifest_path, manifest)
        elif changed == "database":
            database = destination / "decisions.sqlite"
            database.write_bytes(database.read_bytes() + b"corrupt")
        elif changed == "partial":
            manifest_path.unlink()
        else:
            checkpoint.unlink()
        count = len(calls)
        repaired = prepare()
        assert len(calls) == count + 1
        assert (
            repaired.manifest()["policy_filter"]["source_revision"] == source.manifest()["revision"]
        )
        assert file_hash(repaired.database) == repaired.manifest()["database_hash"]
