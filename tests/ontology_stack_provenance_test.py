from __future__ import annotations

import json
from pathlib import Path

import pytest

from exact.ontology import load_ontology
from exact.ontology.provenance import _projector_consumer_handoff, ontology_stack_provenance

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
        "encoded_contract": {
            "schema_name": "pyowl-core/structural-columns",
            "schema_version": 1,
            "descriptor_sha256": (
                "9ad29db6a7e616f65cea2957bc5ba8d1f9b99ef0eb1fe1432c09be25786267b5"
            ),
        },
        "encoded_view_schemas": {},
        "owner_kind": "direct",
        "storage_backend": "python",
    }
    assert handoff["projector"]["compiler_cache_schema"]
    assert "ingestion_path" not in handoff["projector"]
    assert handoff["reasoner"]["reasoner"] == "asserted"
    assert handoff["reasoner"]["worker_wire_verified"] is False
    assert handoff["reasoner"]["worker_mmap_verified"] is False
    assert "worker_owl_parse_count" not in handoff["reasoner"]

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
    assert handoff["projector"]["consumer_compile_seconds"] >= 0.0
    assert handoff["projector"]["counters"] == {
        "encoded_buffer_bytes": 0,
        "encoded_buffer_count": 0,
        "encoded_compiler_gil_released": False,
        "encoded_detached_buffer_count": 0,
        "encoded_indexed_buffer_count": 0,
        "encoded_posting_bytes": 0,
        "encoded_referenced_view_count": 0,
        "encoded_segment_count": 0,
        "encoded_staging_copy_bytes": 0,
        "encoded_zero_copy_buffers": 0,
        "materialized_scalar_rows": 0,
    }
    assert "encoded_view_publication_seconds" not in handoff["projector"]
    assert "schema_name" not in handoff["projector"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("encoded_view_publication_seconds", float("nan"), "diagnostic is invalid"),
        ("consumer_compile_seconds", True, "diagnostic is invalid"),
        ("counters", {"private_arena_id": 1}, "counters are incompatible"),
        ("counters", {"materialized_scalar_rows": False}, "counters are invalid"),
    ],
)
def test_consumer_handoff_rejects_malformed_projector_diagnostics(
    field: str, value: object, message: str
) -> None:
    ingestion: dict[str, object] = {
        "path": "scalar-python",
        "encoded_schema_name": None,
        "encoded_schema_version": None,
        "encoded_descriptor_sha256": None,
        field: value,
    }

    with pytest.raises((TypeError, ValueError), match=message):
        _projector_consumer_handoff(
            {
                "package_version": "0.1.0",
                "compiler_cache_schema": "compiler/1",
                "last_projection": {
                    "provenance": {
                        "selected_backend": "python",
                        "native_implementation_version": None,
                        "ingestion": ingestion,
                    }
                },
            }
        )


def test_consumer_handoff_records_bounded_encoded_projector_diagnostics() -> None:
    result = _projector_consumer_handoff(
        {
            "package_version": "0.1.0",
            "compiler_cache_schema": "compiler/1",
            "last_projection": {
                "provenance": {
                    "selected_backend": "native",
                    "native_implementation_version": "native/1",
                    "ingestion": {
                        "path": "encoded-native",
                        "encoded_schema_name": "pyowl-core/structural-columns",
                        "encoded_schema_version": 1,
                        "encoded_descriptor_sha256": "a" * 64,
                        "encoded_view_publication_seconds": 0.25,
                        "consumer_compile_seconds": 0.5,
                        "counters": {
                            "encoded_buffer_count": 11,
                            "encoded_staging_copy_bytes": 0,
                            "materialized_scalar_rows": 0,
                        },
                    },
                }
            },
        }
    )

    assert result == {
        "package_version": "0.1.0",
        "compiler_cache_schema": "compiler/1",
        "selected_backend": "native",
        "implementation_version": "native/1",
        "ingestion_path": "encoded-native",
        "schema_name": "pyowl-core/structural-columns",
        "schema_version": 1,
        "descriptor_sha256": "a" * 64,
        "encoded_view_publication_seconds": 0.25,
        "consumer_compile_seconds": 0.5,
        "counters": {
            "encoded_buffer_count": 11,
            "encoded_staging_copy_bytes": 0,
            "materialized_scalar_rows": 0,
        },
    }


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("consumer_compile_seconds", True, "consumer_compile_seconds is invalid"),
        (
            "encoded_view_publication_seconds",
            float("nan"),
            "encoded_view_publication_seconds is invalid",
        ),
        (
            "encoded_view_publication_seconds",
            0.25,
            "claimed encoded-view publication",
        ),
        ("counters", {"materialized_scalar_rows": False}, "counters are invalid"),
    ],
)
def test_consumer_handoff_rejects_malformed_reasoner_phase_diagnostics(
    field: str, value: object, message: str
) -> None:
    source = load_ontology(FIXTURES / "mini_src.owl")
    handoff: dict[str, object] = {
        "ingestion_path": "scalar-python",
        "compiler_digest": "0" * 64,
        "counters": {},
        field: value,
    }

    with pytest.raises((TypeError, ValueError), match=message):
        ontology_stack_provenance(
            source.owl_snapshot(),
            projector_settings=source.projector_settings,
            projector=source.projector,
            reasoner={"consumer_handoff": handoff},
        )


def test_consumer_handoff_records_public_reasoner_compiler_contract() -> None:
    source = load_ontology(FIXTURES / "mini_src.owl")

    provenance = ontology_stack_provenance(
        source.owl_snapshot(),
        projector_settings=source.projector_settings,
        projector=source.projector,
        reasoner={
            "selection": {"effective": "hermit", "package_version": "0.1.0"},
            "backend": {"effective": "python", "implementation_version": "0.1.0"},
            "consumer_handoff": {
                "ingestion_path": "scalar-python",
                "compiler_digest": "0" * 64,
                "compiler_cache_schema_version": 1,
                "ir_schema_version": 1,
                "implementation_version": "0.1.0",
                "consumer_compile_seconds": 0.25,
                "counters": {
                    "encoded_buffer_count": 0,
                    "materialized_scalar_rows": 7,
                },
            },
        },
    )

    assert provenance["consumer_handoff"]["reasoner"] == {
        "reasoner": "hermit",
        "package_version": "0.1.0",
        "selected_backend": "python",
        "implementation_version": "0.1.0",
        "ingestion_path": "scalar-python",
        "compiler_digest": "0" * 64,
        "counters": {
            "encoded_buffer_count": 0,
            "materialized_scalar_rows": 7,
        },
        "compiler_cache_schema_version": 1,
        "ir_schema_version": 1,
        "consumer_compile_seconds": 0.25,
    }
