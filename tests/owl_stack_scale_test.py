from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import pyowl_core
from pyowl2vec_star_projector import Edge, canonical_edges_sha256

from benchmarks.owl_stack_scale import (
    _consumer_counter_evidence,
    _edge_record,
    _require_consumer_counter_evidence,
    _require_encoded_path,
    main,
    measure,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"


def _encoded_handoff(**counter_overrides: int | bool) -> dict[str, object]:
    counters: dict[str, int | bool] = {
        "base_flattening_bytes": 0,
        "encoded_buffer_bytes": 128,
        "encoded_buffer_count": 11,
        "encoded_compiler_gil_released": True,
        "encoded_staging_copy_bytes": 0,
        "encoded_zero_copy_buffers": 11,
        "materialized_scalar_rows": 0,
        "parser_calls": 0,
        "per_row_ffi_calls": 0,
        "resolver_calls": 0,
        "scalar_axiom_materializations": 0,
        "scalar_term_materializations": 0,
        "structural_copy_bytes": 0,
        "wire_decoder_calls": 0,
        "wire_encoder_calls": 0,
    }
    counters.update(counter_overrides)
    return {
        "ingestion_path": "encoded-native",
        "schema_name": "pyowl-core/structural-columns",
        "schema_version": 1,
        "descriptor_sha256": ("9ad29db6a7e616f65cea2957bc5ba8d1f9b99ef0eb1fe1432c09be25786267b5"),
        "counters": counters,
    }


def test_streaming_edge_digest_matches_public_projector_artifact_contract() -> None:
    edges = (
        Edge("urn:test:A", "urn:test:r", "urn:test:B"),
        Edge("urn:test:C", "urn:test:r", "urn:test:D"),
    )

    digest = hashlib.sha256()
    for edge in edges:
        digest.update(_edge_record(edge))

    assert digest.hexdigest() == canonical_edges_sha256(edges)


def test_scale_measurement_records_path_free_wpn_handoff_evidence() -> None:
    path = FIXTURES / "mini_src.owl"

    result = measure(
        path,
        buffer_edges=32,
        include_literals=False,
        reasoner_name="asserted",
    )

    assert result["load_calls"] == 1
    assert result["identity"] == {
        "source_snapshot": True,
        "projector_snapshot": True,
        "reasoner_snapshot": True,
    }
    operations = result["core_operations"]
    assert operations["consumer_delta"] == {
        "load_snapshot": 0,
        "encode_snapshot": 0,
        "decode_snapshot": 0,
        "open_snapshot": 0,
    }
    projection = result["projection"]
    assert projection["consumer"]["ingestion_path"] == "scalar-python"
    assert projection["consumer_compile_seconds"] >= 0.0
    assert projection["encoded_view_publication_seconds"] is None
    assert projection["publication_compile_timing_note"] is None
    assert projection["consumer"]["counters"]["materialized_scalar_rows"] == 0
    assert projection["edges"] > 0
    assert len(projection["result_sha256"]) == 64
    assert projection["result_sha256"] == result["projection_cache"]["result_sha256"]
    reasoner = result["reasoner"]
    assert reasoner["measured"] is True
    assert reasoner["consumer_compile_seconds"] is None
    assert reasoner["encoded_view_publication_seconds"] is None
    assert reasoner["results"]["entities"] > 0
    assert len(reasoner["results"]["result_sha256"]) == 64
    materialization = result["materialization_and_copy"]
    assert materialization["public_counters"]["projector"]["materialized_scalar_rows"] == 0
    assert materialization["complete_public_counter_coverage"] is False
    acceptance = materialization["acceptance_evidence"]
    assert acceptance["acceptance_ready"] is False
    assert acceptance["projector"]["selected_ingestion_path"] == "scalar-python"
    assert acceptance["unexpected_core_operation_calls"] == {}
    assert result["second_ontology_representation"] is False

    encoded = json.dumps(result, sort_keys=True)
    assert str(path.resolve()) not in encoded
    assert "object at 0x" not in encoded


def test_required_encoded_mode_rejects_scalar_consumer_selection() -> None:
    original_load = pyowl_core.load_snapshot

    with pytest.raises(RuntimeError, match="projector did not select required encoded-native"):
        measure(
            FIXTURES / "mini_src.owl",
            buffer_edges=32,
            include_literals=False,
            projector_backend="python",
            require_encoded_consumers=True,
        )

    assert pyowl_core.load_snapshot is original_load


def test_direct_encoded_counter_evidence_requires_complete_zero_copy_gil_record() -> None:
    evidence = _consumer_counter_evidence(
        consumer="projector",
        handoff=_encoded_handoff(),
    )

    assert evidence["acceptance_ready"] is True
    assert evidence["complete_public_counter_coverage"] is True
    assert evidence["missing_public_counters"] == []
    assert evidence["invalid_public_counters"] == {}
    assert evidence["nonzero_forbidden_public_counters"] == {}
    assert evidence["encoded_buffer_count"] == 11
    assert evidence["encoded_zero_copy_buffers"] == 11
    assert evidence["all_encoded_buffers_zero_copy"] is True
    assert evidence["encoded_schema"]["compatible"] is True
    _require_consumer_counter_evidence(evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_name", "pyowl-core/other"),
        ("schema_version", True),
        ("descriptor_sha256", "0" * 64),
    ],
)
def test_direct_encoded_counter_evidence_rejects_incompatible_schema(
    field: str,
    value: object,
) -> None:
    handoff = _encoded_handoff()
    handoff[field] = value

    evidence = _consumer_counter_evidence(consumer="projector", handoff=handoff)

    assert evidence["encoded_schema"]["compatible"] is False
    assert evidence["acceptance_ready"] is False
    with pytest.raises(RuntimeError, match="ingestion schema is incompatible"):
        _require_encoded_path(consumer="projector", handoff=handoff)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"parser_calls": 1}, "nonzero_forbidden_public_counters"),
        ({"structural_copy_bytes": 64}, "nonzero_forbidden_public_counters"),
        ({"encoded_staging_copy_bytes": 64}, "direct_staging_copy_bytes"),
        ({"encoded_compiler_gil_released": False}, "encoded_compiler_gil_released"),
    ],
)
def test_direct_encoded_counter_evidence_rejects_ineligible_records(
    overrides: dict[str, int | bool],
    field: str,
) -> None:
    evidence = _consumer_counter_evidence(
        consumer="projector",
        handoff=_encoded_handoff(**overrides),
    )

    assert evidence["acceptance_ready"] is False
    if field == "nonzero_forbidden_public_counters":
        assert evidence[field]
    elif field == "direct_staging_copy_bytes":
        assert evidence[field] == overrides["encoded_staging_copy_bytes"]
    else:
        assert evidence[field] == overrides["encoded_compiler_gil_released"]
    with pytest.raises(RuntimeError, match="acceptance evidence failed"):
        _require_consumer_counter_evidence(evidence)


def test_direct_encoded_counter_evidence_rejects_missing_counter() -> None:
    handoff = _encoded_handoff()
    counters = handoff["counters"]
    assert isinstance(counters, dict)
    counters.pop("wire_encoder_calls")

    evidence = _consumer_counter_evidence(consumer="elk", handoff=handoff)

    assert evidence["acceptance_ready"] is False
    assert evidence["missing_public_counters"] == ["wire_encoder_calls"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"parser_calls": False},
        {"encoded_buffer_bytes": False},
        {"encoded_buffer_bytes": -1},
        {"encoded_staging_copy_bytes": False},
        {"encoded_buffer_count": 0},
        {"encoded_zero_copy_buffers": 10},
        {"encoded_private_ir_bytes": 1},
    ],
)
def test_direct_encoded_counter_evidence_rejects_inexact_zero_copy_records(
    overrides: dict[str, int | bool],
) -> None:
    evidence = _consumer_counter_evidence(
        consumer="hermit",
        handoff=_encoded_handoff(**overrides),
    )

    assert evidence["acceptance_ready"] is False
    with pytest.raises(RuntimeError, match="acceptance evidence failed"):
        _require_consumer_counter_evidence(evidence)


def test_cli_emits_versioned_configuration(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "owl_stack_scale.py",
            str(FIXTURES / "mini_src.owl"),
            "--buffer-edges",
            "32",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 5
    assert payload["configuration"]["cache_state"] == ("cold-load; projection cache fill then hit")
    assert payload["measurements"][0]["load_calls"] == 1
