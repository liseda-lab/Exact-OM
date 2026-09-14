from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyowl_core
import pytest
from pyowl2vec_star_projector import Edge, canonical_edges_sha256

from benchmarks.owl_stack_scale import (
    _consumer_counter_evidence,
    _edge_record,
    _is_ncit_doid_request,
    _require_consumer_counter_evidence,
    _require_encoded_path,
    _validate_ncit_doid_acceptance,
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
    encoded_view = pyowl_core.EncodedStructuralView
    return {
        "ingestion_path": "encoded-native",
        "schema_name": encoded_view.SCHEMA_NAME,
        "schema_version": encoded_view.SCHEMA_VERSION,
        "descriptor_sha256": encoded_view.DESCRIPTOR_SHA256.hex(),
        "counters": counters,
    }


def _ncit_doid_acceptance_fixture() -> tuple[dict[str, object], list[dict[str, object]]]:
    environment: dict[str, object] = {
        "packages": {
            "exact-om": "2.1.0",
            "pyowl-core": "0.2.0",
            "pyowl2vec-star-projector": "0.2.0",
        }
    }
    measurements = []
    for name, size, sha256, edges, digest in (
        (
            "source.owl",
            57_163_710,
            "379a37f47c0c8e7c30397769358cca955140d16b2797a1cc75da4b1fc2b354eb",
            42_103,
            "a" * 64,
        ),
        (
            "target.owl",
            6_687_536,
            "76f41cce3616ad1a9ba6353f469e96bde7addba5d43e541651a3ab703f9ba2bc",
            9_388,
            "b" * 64,
        ),
    ):
        measurements.append(
            {
                "input": {"name": name, "bytes": size, "sha256": sha256},
                "load_calls": 1,
                "load_backend": "native",
                "identity": {
                    "source_snapshot": True,
                    "projector_snapshot": True,
                    "reasoner_snapshot": True,
                },
                "second_ontology_representation": False,
                "projection": {
                    "profile": "mowl-d993536-v1",
                    "requested_backend": "native",
                    "edges": edges,
                    "result_sha256": digest,
                },
                "projection_cache": {"edges": edges, "result_sha256": digest},
                "materialization_and_copy": {"acceptance_evidence": {"acceptance_ready": True}},
            }
        )
    return environment, measurements


def test_ncit_doid_acceptance_validator_accepts_frozen_correctness_record() -> None:
    environment, measurements = _ncit_doid_acceptance_fixture()

    acceptance = _validate_ncit_doid_acceptance(environment, measurements)

    assert acceptance["status"] == "passed"
    assert acceptance["performance_claim"] is False
    assert acceptance["inputs_match_frozen_baseline"] is True
    assert acceptance["expected_projection_edges"] == {
        "source": 42_103,
        "target": 9_388,
    }
    assert acceptance["historical_ncit_delta"]["residual"] == {"edges": 0}


def test_ncit_doid_acceptance_auto_selects_only_the_frozen_pair_path() -> None:
    pair = Path("data/bioml_zenodo/ncit-doid")

    assert _is_ncit_doid_request((pair / "source.owl", pair / "target.owl")) is True
    assert _is_ncit_doid_request((pair / "target.owl", pair / "source.owl")) is False
    assert _is_ncit_doid_request((Path("other/source.owl"), Path("other/target.owl"))) is False


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("version", "pyowl-core version"),
        ("hash", "source input sha256"),
        ("count", "source projected"),
        ("backend", "source projector backend"),
    ],
)
def test_ncit_doid_acceptance_validator_rejects_frozen_contract_mismatch(
    mismatch: str,
    message: str,
) -> None:
    environment, measurements = _ncit_doid_acceptance_fixture()
    packages = environment["packages"]
    assert isinstance(packages, dict)

    if mismatch == "version":
        packages["pyowl-core"] = "0.2.0rc1"
    elif mismatch == "hash":
        input_record = measurements[0]["input"]
        assert isinstance(input_record, dict)
        input_record["sha256"] = "0" * 64
    elif mismatch == "count":
        projection = measurements[0]["projection"]
        assert isinstance(projection, dict)
        projection["edges"] = 42_102
    else:
        projection = measurements[0]["projection"]
        assert isinstance(projection, dict)
        projection["requested_backend"] = "python"

    with pytest.raises(RuntimeError, match=message):
        _validate_ncit_doid_acceptance(environment, measurements)


def test_streaming_edge_digest_matches_public_projector_artifact_contract() -> None:
    edges = (
        Edge("urn:test:A", "urn:test:r", "urn:test:B"),
        Edge("urn:test:C", "urn:test:r", "urn:test:D"),
    )

    digest = hashlib.sha256()
    for edge in edges:
        digest.update(_edge_record(edge))

    assert digest.hexdigest() == canonical_edges_sha256(edges)


def test_scale_measurement_records_path_free_wpn_handoff_evidence(monkeypatch) -> None:
    import pyowl2vec_star_projector.api as projector_api

    def forbidden(*args, **kwargs):
        raise AssertionError("Scale benchmark must not enter scalar projection")

    monkeypatch.setattr(projector_api, "prepare_streaming_compilation", forbidden)
    monkeypatch.setattr(projector_api, "prepare_encoded_subset_compilation", forbidden)
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
    assert result["load_backend"] == projection["requested_backend"] == "native"
    assert projection["consumer"]["ingestion_path"] == "encoded-native"
    assert projection["time_to_first_edge_seconds"] is None
    assert projection["edge_delivery"] == "guarded-materialized-list"
    assert projection["buffer_edges_effective"] is None
    assert projection["consumer_compile_seconds"] >= 0.0
    assert projection["encoded_view_publication_seconds"] >= 0.0
    assert projection["publication_compile_timing_note"] is None
    assert projection["consumer"]["counters"]["materialized_scalar_rows"] == 0
    assert projection["consumer"]["counters"]["native_validation_receipt"] is True
    assert projection["consumer"]["counters"]["native_canonical_sort_calls"] >= 1
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
    assert materialization["complete_public_counter_coverage"] is True
    acceptance = materialization["acceptance_evidence"]
    assert acceptance["acceptance_ready"] is True
    assert acceptance["projector"]["selected_ingestion_path"] == "encoded-native"
    assert acceptance["unexpected_core_operation_calls"] == {}
    assert result["second_ontology_representation"] is False

    encoded = json.dumps(result, sort_keys=True)
    assert str(path.resolve()) not in encoded
    assert "object at 0x" not in encoded


@pytest.mark.parametrize("option", ["load_backend", "projector_backend"])
def test_python_selection_is_rejected_before_loading_or_hashing(monkeypatch, option) -> None:
    original_load = pyowl_core.load_snapshot

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid backend must fail before input work")

    monkeypatch.setattr("benchmarks.owl_stack_scale._sha256_file", forbidden)
    with pytest.raises(
        ValueError, match="require native loading and projection; Python is rejected"
    ):
        measure(
            Path("does-not-exist.owl"),
            buffer_edges=32,
            include_literals=False,
            require_encoded_consumers=True,
            **{option: "python"},
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
    assert payload["configuration"]["cache_state"] == (
        "cold-load; guarded projection fills cache then repeated hits"
    )
    assert payload["configuration"]["load_backend"] == "native"
    assert payload["configuration"]["projector_backend"] == "native"
    assert payload["configuration"]["buffer_edges_effective"] is None
    assert payload["measurements"][0]["load_calls"] == 1
