from __future__ import annotations

import json
from pathlib import Path

import pytest

from exact.ontology import load_ontology
from exact.ontology.provenance import ontology_stack_provenance

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"


def test_ontology_stack_provenance_is_complete_and_path_free() -> None:
    path = FIXTURES / "mini_src.owl"
    source = load_ontology(path)

    provenance = source.ontology_stack_provenance()

    assert provenance["kind"] == "owl"
    core = provenance["core"]
    assert core["shared_snapshot"] is True
    assert set(core["fingerprints"]) == {"structural", "logical", "signature"}
    assert len(core["closure"]["import_manifest_sha256"]) == 64
    assert len(core["closure"]["resolver_configuration_sha256"]) == 64
    assert core["closure"]["source_documents"]
    assert provenance["projector"]["profile"] == "mowl-d993536-v1"
    assert provenance["projector"]["selection"]["effective"] == "python"
    assert provenance["reasoner"]["selection"]["effective"] == "asserted"
    handoff = provenance["consumer_handoff"]
    assert handoff["schema_version"] == 1
    assert handoff["core"] == {
        "encoded_view_schemas": {},
        "owner_kind": "direct",
        "storage_backend": "python",
    }
    assert handoff["projector"]["compiler_cache_schema"]
    assert "ingestion_path" not in handoff["projector"]
    assert handoff["reasoner"]["reasoner"] == "asserted"

    encoded = json.dumps(provenance, sort_keys=True)
    assert str(path.resolve()) not in encoded
    assert "object at 0x" not in encoded
    assert "file://" not in encoded


def test_ontology_stack_provenance_redacts_consumer_diagnostics() -> None:
    source = load_ontology(FIXTURES / "mini_src.owl")

    provenance = ontology_stack_provenance(
        source.owl_snapshot(),
        projector_settings=source.projector_settings,
        projector=source.projector,
        reasoner={
            "failure_reason": (
                "/private/tmp/exact-reasoner/ontology.pyocore "
                "https://user:secret@example.org/private object at 0x1234abcd"
            )
        },
    )

    encoded = json.dumps(provenance, sort_keys=True)
    assert "/private/tmp" not in encoded
    assert "user:secret" not in encoded
    assert "0x1234abcd" not in encoded


def test_consumer_handoff_records_projector_public_ingestion_report() -> None:
    source = load_ontology(FIXTURES / "mini_src.owl")
    source.configure_projector(backend="python")
    assert source.projection_edges()

    handoff = source.ontology_stack_provenance()["consumer_handoff"]

    assert handoff["projector"]["ingestion_path"] == "scalar-python"
    assert handoff["projector"]["selected_backend"] == "python"
    assert "schema_name" not in handoff["projector"]


def test_consumer_handoff_rejects_unbounded_reasoner_diagnostics() -> None:
    source = load_ontology(FIXTURES / "mini_src.owl")

    with pytest.raises(ValueError, match="fields are incompatible"):
        ontology_stack_provenance(
            source.owl_snapshot(),
            projector_settings=source.projector_settings,
            projector=source.projector,
            reasoner={
                "consumer_handoff": {
                    "ingestion_path": "encoded-native",
                    "compiler_digest": "digest",
                    "counters": {},
                    "private_arena_id": "forbidden",
                }
            },
        )
