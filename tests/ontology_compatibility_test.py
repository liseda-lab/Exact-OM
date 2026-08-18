"""Machine-readable final-stack compatibility and packaging contract."""

from __future__ import annotations

import json
from pathlib import Path

import pyowl_core

from exact.ontology import projection, provenance, reasoning

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "release" / "core-compatibility.json"
PYPROJECT_PATH = ROOT / "pyproject.toml"

FORBIDDEN_NATIVE_WORK = {
    "base_flattening_bytes",
    "materialized_scalar_rows",
    "parser_calls",
    "per_row_ffi_calls",
    "resolver_calls",
    "scalar_axiom_materializations",
    "scalar_term_materializations",
    "structural_copy_bytes",
    "wire_decoder_calls",
    "wire_encoder_calls",
}
PROJECTOR_SCALAR_LEDGER = {
    "base_flattening_bytes",
    "encoded_buffer_bytes",
    "encoded_buffer_count",
    "encoded_compiler_gil_released",
    "encoded_detached_buffer_count",
    "encoded_indexed_buffer_count",
    "encoded_posting_bytes",
    "encoded_referenced_view_count",
    "encoded_segment_count",
    "encoded_staging_copy_bytes",
    "encoded_zero_copy_buffers",
    "materialized_scalar_rows",
    "parser_calls",
    "per_row_ffi_calls",
    "resolver_calls",
    "scalar_axiom_materializations",
    "scalar_term_materializations",
    "structural_copy_bytes",
    "wire_decoder_calls",
    "wire_encoder_calls",
}


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_final_ontology_stack_uses_only_published_0_2_releases() -> None:
    contract = _contract()

    assert contract["schema"] == "exact-om.ontology-stack-compatibility/1"
    assert contract["status"] == "verified"
    assert contract["performance_claim"] is False
    assert contract["exact"] == {
        "repository": "https://github.com/liseda-lab/Exact-OM",
        "version": "2.1.0",
    }
    assert "release_evidence" not in contract

    assert contract["tested_stack"] == {
        "pyowl-core": {
            "repository": "https://github.com/OAEI-ML/pyOWLCore",
            "version": "0.2.0",
        },
        "pyowl2vec-star-projector": {
            "repository": "https://github.com/OAEI-ML/pyOwl2Vec-Star-projector",
            "version": "0.2.0",
        },
        "pyelk-reasoner": {
            "repository": "https://github.com/OAEI-ML/pyELK",
            "version": "0.2.0",
        },
        "pyhermit": {
            "repository": "https://github.com/OAEI-ML/pyHermiT",
            "version": "0.2.0",
        },
        "oaei-bioml-eval": {
            "repository": "https://github.com/OAEI-ML/OAEI-Bio-ML-eval",
            "version": "0.2.1",
        },
    }


def test_manifest_matches_runtime_schema_and_complete_counter_vocabularies() -> None:
    contract = _contract()
    encoded = contract["encoded_contract"]
    identity = projection.encoded_contract_identity()

    assert contract["core_contract"] == {
        "distribution": "pyowl-core",
        "version": "0.2.0",
        "api_version": list(pyowl_core.API_VERSION),
        "model_schema": pyowl_core.MODEL_SCHEMA_VERSION,
        "wire_writer": list(pyowl_core.WIRE_FORMAT_VERSION),
        "wire_readable_through": list(pyowl_core.WIRE_FORMAT_VERSION),
        "adapter_protocol": pyowl_core.ADAPTER_PROTOCOL_VERSION,
    }
    assert encoded["schema_name"] == identity.core_schema_name
    assert encoded["schema_version"] == identity.core_schema_version
    assert encoded["model_schema"] == pyowl_core.MODEL_SCHEMA_VERSION == 2
    assert encoded["descriptor_source"] == ("pyowl_core.EncodedStructuralView.DESCRIPTOR_SHA256")
    assert encoded["descriptor_sha256"] == identity.core_descriptor_sha256
    assert encoded["in_process_transport"] == "identity"
    assert encoded["worker_transport"] == "one-core-wire-verified-mmap"
    assert encoded["performance_claim"] is False
    assert contract["performance_claim"] is False

    assert contract["cache_policy"] == {
        "core_model_schema": 2,
        "schema_1_ontology_caches": "reject-before-interpretation-and-rebuild",
        "conversion_supported": False,
        "completed_run_artifacts_readable": True,
        "schema_1_resume_supported": False,
    }

    projector = contract["projector_handoff"]
    reasoner = contract["reasoner_handoff"]
    assert set(projector["required_counters"]) == provenance._PROJECTOR_HANDOFF_COUNTERS
    assert set(projector["scalar_counter_ledger"]) == PROJECTOR_SCALAR_LEDGER
    assert projector["tested_owner_ingestion_paths"] == {
        "direct": "encoded-native",
        "decoded": "encoded-native",
        "mmap": "scalar-native",
        "overlay": "encoded-native",
        "composite": "encoded-native",
    }
    assert set(reasoner["required_counters"]) == reasoning._HANDOFF_COUNTERS
    assert set(projector["required_zero_counters"]) == FORBIDDEN_NATIVE_WORK
    assert set(reasoner["required_zero_counters"]) == FORBIDDEN_NATIVE_WORK
    assert projector["advertised_ingestion_path"] == "encoded-native"
    assert reasoner["advertised_ingestion_path"] == "encoded-native"


def test_dependency_ranges_extras_and_artifact_inclusion_stay_coordinated() -> None:
    contract = _contract()
    metadata = PYPROJECT_PATH.read_text(encoding="utf-8")

    assert contract["dependency_constraints"] == {
        "base": {
            "pyowl-core": ">=0.2,<0.3",
            "pyowl2vec-star-projector": ">=0.2,<0.3",
        },
        "reasoning_extra": {
            "pyelk-reasoner": ">=0.2,<0.3",
            "pyhermit": ">=0.2,<0.3",
        },
        "evaluation_extra": {
            "oaei-bioml-eval": ">=0.2.1,<0.3",
        },
    }
    for line in (
        'pyowl-core = ">=0.2,<0.3"',
        'pyowl2vec-star-projector = ">=0.2,<0.3"',
        'pyelk-reasoner = {version = ">=0.2,<0.3", optional = true}',
        'pyhermit = {version = ">=0.2,<0.3", optional = true}',
        'oaei-bioml-eval = {version = ">=0.2.1,<0.3", extras = ["reasoner"], optional = true}',
        'reasoning = ["pyelk-reasoner", "pyhermit"]',
        'bioml-eval = ["oaei-bioml-eval"]',
        '{ path = "release/core-compatibility.json", format = ["sdist", "wheel"] }',
        'exclude = ["release/evidence/**"]',
    ):
        assert metadata.count(line) == 1

    boundary = contract["public_boundary"]
    assert boundary == {
        "exact_requests_encoded_buffers": False,
        "exact_decodes_encoded_buffers": False,
        "exact_flattens_layered_views": False,
        "consumer_path_retry": False,
        "java_runtime": False,
        "oaei_is_ontology_runtime_dependency": False,
    }
