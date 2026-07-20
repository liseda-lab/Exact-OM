from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import pyowl_core
from pyowl2vec_star_projector import Edge, canonical_edges_sha256

from benchmarks.owl_stack_scale import _edge_record, main, measure

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"


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
    assert reasoner["results"]["entities"] > 0
    assert len(reasoner["results"]["result_sha256"]) == 64
    materialization = result["materialization_and_copy"]
    assert materialization["public_counters"]["projector"]["materialized_scalar_rows"] == 0
    assert materialization["complete_public_counter_coverage"] is False
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
    assert payload["schema_version"] == 3
    assert payload["configuration"]["cache_state"] == ("cold-load; projection cache fill then hit")
    assert payload["measurements"][0]["load_calls"] == 1
