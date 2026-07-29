"""Machine-readable final-stack compatibility and packaging contract."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_final_ontology_stack_revisions_and_trees_are_exact() -> None:
    contract = _contract()

    assert contract["schema"] == "exact-om.ontology-stack-compatibility/1"
    assert contract["exact"] == {
        "repository": "https://github.com/liseda-lab/Exact-OM",
        "version": "2.0.0",
        "compatibility_parent_commit": "abba717bd5b3f186678bd6f3e88bf73066c2ae49",
        "compatibility_parent_tree": "a0a7c9e3ddf3a9ad1ae8da307654eeafdd2d3b59",
        "final_revision_binding": "external-pyowl-core-consumer-compatibility-matrix",
        "runtime_baseline_commit": "ab4b76644f6ed58894d0920e47de713ba1ffb358",
        "runtime_baseline_tree": "dc353f2b4e2eb2f987ae4d7552cdb51bf9480226",
    }
    tested = contract["tested_stack"]
    assert tested == {
        "pyowl-core": {
            "repository": "https://github.com/OAEI-ML/pyOWLCore",
            "commit": "005c3ccad129757b3a9be125dc064b812b607ef5",
            "tree": "d4f3f29f6594b59f3d45a4811c38fb761a7028b9",
            "version": "0.1.0.dev0",
        },
        "pyowl2vec-star-projector": {
            "repository": "https://github.com/OAEI-ML/pyOwl2Vec-Star-projector",
            "commit": "9f19db3de54b7bdffe45498479edadd72af37218",
            "tree": "71280102561265fb015d9bf9d02c57448c0490a4",
            "version": "0.1.0rc1",
        },
        "pyelk-reasoner": {
            "repository": "https://github.com/OAEI-ML/pyELK",
            "commit": "70302fcd6abc27d703eeb8f59027fc1392f4709b",
            "tree": "ad28753b4cb0efed5e4ca442575a63ef4d7bd0c8",
            "version": "0.1.0.dev0",
        },
        "pyHermiT": {
            "repository": "https://github.com/OAEI-ML/pyHermiT",
            "commit": "af8f7fc669b28dfc15728c84c78f9094787d288b",
            "tree": "e7523959d80b116520f9387463d2ae82ba22c5f8",
            "version": "0.1.0.dev0",
        },
    }
    assert contract["companion_evidence"] == {
        "oaei-bioml-eval": {
            "repository": "https://github.com/OAEI-ML/OAEI-Bio-ML-eval",
            "role": "companion-evaluation-evidence",
            "ontology_runtime_dependency": False,
            "final_commit": "94713d5068ce78d90f42e7fb100c7631b6490924",
            "source_commit": "1e8a55f27cc5e6de0dad7f149408a8d85a294be6",
            "tree": "10a9fa27852cc02210a494f7daebfe418be17238",
        }
    }


def test_manifest_matches_runtime_schema_and_complete_counter_vocabularies() -> None:
    contract = _contract()
    encoded = contract["encoded_contract"]
    identity = projection.encoded_contract_identity()

    assert encoded["schema_name"] == identity.core_schema_name
    assert encoded["schema_version"] == identity.core_schema_version
    assert encoded["model_schema"] == 1
    assert encoded["descriptor_sha256"] == identity.core_descriptor_sha256
    assert encoded["owner_matrix"] == [
        "direct",
        "decoded",
        "mmap",
        "overlay",
        "composite",
    ]
    assert encoded["in_process_transport"] == "identity"
    assert encoded["worker_transport"] == "one-core-wire-verified-mmap"
    assert encoded["performance_claim"] is False

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
            "pyowl-core": ">=0.1,<0.2",
            "pyowl2vec-star-projector": ">=0.1,<0.2",
        },
        "reasoning_extra": {
            "pyelk-reasoner": ">=0.1,<0.2",
            "pyhermit": ">=0.1,<0.2",
        },
        "evaluation_extra": {
            "oaei-bioml-eval": ">=0.2,<0.3",
        },
    }
    for line in (
        'pyowl-core = ">=0.1,<0.2"',
        'pyowl2vec-star-projector = ">=0.1,<0.2"',
        'pyelk-reasoner = {version = ">=0.1,<0.2", optional = true}',
        'pyhermit = {version = ">=0.1,<0.2", optional = true}',
        'oaei-bioml-eval = {version = ">=0.2,<0.3", extras = ["reasoner"], optional = true}',
        'reasoning = ["pyelk-reasoner", "pyhermit"]',
        'bioml-eval = ["oaei-bioml-eval"]',
        '{ path = "release/core-compatibility.json", format = ["sdist", "wheel"] }',
    ):
        assert metadata.count(line) == 1

    boundary = contract["public_boundary"]
    assert boundary == {
        "exact_requests_encoded_buffers": False,
        "exact_decodes_encoded_buffers": False,
        "exact_flattens_layered_views": False,
        "oaei_is_ontology_runtime_dependency": False,
    }
