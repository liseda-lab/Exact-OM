"""Machine-readable final-stack compatibility and packaging contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_published_baseline_record_retains_verified_release_versions() -> None:
    contract = _contract()

    assert contract["schema"] == "exact-om.ontology-stack-compatibility/1"
    assert contract["status"] == "verified"
    assert contract["record_scope"] == "published_baseline"
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


def test_published_schema_and_counter_record_remain_unchanged() -> None:
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
    assert set(projector["required_counters"]) == PROJECTOR_SCALAR_LEDGER | {
        "native_batch_edges",
        "native_boundary_calls",
        "native_compiled_edges",
        "native_edge_batches",
        "native_output_vector_edges",
        "native_peak_buffered_edges",
        "native_retained_inverse_properties",
        "native_retained_subrole_properties",
    }
    assert set(projector["scalar_counter_ledger"]) == PROJECTOR_SCALAR_LEDGER
    assert projector["tested_owner_ingestion_paths"] == {
        "direct": "encoded-native",
        "decoded": "encoded-native",
        "mmap": "scalar-native",
        "overlay": "encoded-native",
        "composite": "encoded-native",
    }
    assert set(reasoner["required_counters"]) == PROJECTOR_SCALAR_LEDGER | {
        "encoded_private_ir_bytes"
    }
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


def test_published_baseline_content_is_not_rewritten_by_candidate_metadata() -> None:
    contract = _contract()
    contract.pop("native_pipeline_candidate")
    contract.pop("record_scope")
    actual = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    # Canonical content of the pre-candidate published record, not a wheel hash.
    assert actual == "84a3f958b73822895f13bc87647c7eb479852a769e9bd3e1c07d79749a618167"


def test_native_candidate_requires_its_own_artifact_and_full_input_evidence() -> None:
    baseline = _contract()
    candidate = baseline["native_pipeline_candidate"]
    assert candidate["status"] in {"pending", "installed_validated", "validated"}
    assert candidate["release_state"] == "unreleased-local-wheels"
    assert candidate["performance_claim"] is False
    assert candidate["version_alone_identifies_artifacts"] is False
    verification = candidate["verification"]
    assert verification["artifact_bindings"] in {"pending", "passed"}
    assert verification["installed_T3"] in {"pending", "passed"}
    assert verification["real_input_T4"] in {"pending", "running", "passed", "failed", "timed_out"}
    if candidate["status"] in {"installed_validated", "validated"}:
        assert verification["installed_T3"] == "passed"
    if candidate["status"] == "validated":
        assert verification["real_input_T4"] == "passed"
    if verification["installed_T3"] == "passed":
        assert verification["artifact_bindings"] == "passed"
        assert verification["evidence"]
    assert set(candidate["packages"]) == {
        "pyowl-core",
        "pyowl2vec-star-projector",
        "pyelk-reasoner",
        "pyhermit",
    }
    for name, record in candidate["packages"].items():
        assert record["repository"] == baseline["tested_stack"][name]["repository"]
        assert record["version"] == baseline["tested_stack"][name]["version"]
        for field, size in (("source_commit", 40), ("wheel_sha256", 64)):
            value = record[field]
            if verification["artifact_bindings"] == "passed":
                assert isinstance(value, str) and len(value) == size
                assert set(value) <= set("0123456789abcdef")
            else:
                assert value is None
    contract = candidate["execution_contract"]
    assert contract == "exact/native-pipeline/v2"
    assert (
        projection.projector_cache_identity(projection.ProjectorSettings())["execution_contract"]
        == contract
    )
    assert reasoning.reasoner_cache_identity("asserted")["execution_contract"] == contract
    assert candidate["scientific_identity"]["timing_counters_are_diagnostics"] is True
    assert (
        candidate["scientific_identity"][
            "published_baseline_verification_does_not_verify_candidate"
        ]
        is True
    )
    handoff = candidate["strict_consumer_handoff"]
    assert handoff["ingestion_path"] == "encoded-native"
    assert handoff["encoded_contract_ref"] == "encoded_contract"
    assert handoff["in_process_owner"] == "direct-retained-native-snapshot"
    assert handoff["encoded_unsupported_owners"] == ["decoded", "mmap", "overlay", "composite"]
    assert handoff["worker_wire_supported"] is False
    assert handoff["consumer_path_retry"] is False


def test_candidate_counter_vocabulary_matches_current_provenance_boundaries() -> None:
    baseline = _contract()
    candidate = baseline["native_pipeline_candidate"]
    projector = candidate["projector"]
    reasoner = candidate["reasoner"]
    assert set(projector["counter_vocabulary"]) == provenance._PROJECTOR_HANDOFF_COUNTERS
    assert set(reasoner["counter_vocabulary"]) == reasoning._HANDOFF_COUNTERS
    assert set(reasoner["counter_vocabulary"]) == provenance._REASONER_HANDOFF_COUNTERS
    assert set(projector["boolean_counters"]) == {
        "encoded_compiler_gil_released",
        "native_validation_receipt",
    }
    assert set(reasoner["boolean_counters"]) == reasoning._HANDOFF_BOOLEAN_COUNTERS
    assert set(projector["required_true_counters"]) == set(projector["boolean_counters"])
    assert set(reasoner["required_true_before_query"]) == set(reasoner["boolean_counters"]) - {
        "native_result_validation"
    }
    assert set(reasoner["required_true_after_query"]) == set(reasoner["boolean_counters"])
    assert set(projector["required_zero_counters"]) == FORBIDDEN_NATIVE_WORK | {
        "encoded_indexed_buffer_count"
    }
    assert set(reasoner["required_zero_counters"]) == FORBIDDEN_NATIVE_WORK | {
        "encoded_indexed_buffer_count",
        "native_metadata_domain_copies",
    }
    for name, report in (("projector", projector), ("reasoner", reasoner)):
        vocabulary = report["counter_vocabulary"]
        assert vocabulary == sorted(set(vocabulary))
        assert set(baseline[f"{name}_handoff"]["required_counters"]) < set(vocabulary)
        assert set(report["required_zero_counters"]) <= set(vocabulary)
        assert set(report["boolean_counters"]).isdisjoint(report["required_zero_counters"])
    core = candidate["required_capabilities"]["pyowl-core"]
    for view in (
        "AxiomTypeIndex",
        "AnnotationAssertionIndex",
        "ClassFeatureView",
        "AssertedPropertyHierarchyView",
        "PropertyDomainRangeView",
    ):
        assert f"{view}.supports_native" in core
