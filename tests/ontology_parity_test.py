import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
import zstandard

from tools.capture_backend_baseline import capture

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"
BASELINES = Path(__file__).parent / "baselines"
PROVENANCE = json.loads((BASELINES / "provenance.json").read_text(encoding="utf-8"))
SCALE_EVIDENCE = (
    Path(__file__).parents[1] / "benchmarks" / "evidence" / "wp_m_ncit_doid_baseline.json"
)
SCALE_CANDIDATE = (
    Path(__file__).parents[1] / "benchmarks" / "evidence" / "wp_m_ncit_doid_candidate.json"
)


def _load_snapshot(path: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(zstandard.ZstdDecompressor().decompress(path.read_bytes())),
    )


def _edge_set(payload: dict, name: str) -> set[tuple[str, str, str]]:
    return {tuple(edge) for edge in payload["projection"][name]}


def test_wp_m_scale_baseline_is_reviewable_and_content_addressed():
    evidence = json.loads(SCALE_EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["schema"] == "exact-ontology-scale-baseline/1"
    assert evidence["dataset"]["redistributed"] is False
    for role in ("source", "target"):
        item = evidence["dataset"][role]
        result = evidence["results"][role]
        assert len(item["sha256"]) == 64
        assert item["file_bytes"] > 0
        assert item["axiom_records"] > 0
        assert item["class_and_entity_signature_count"] > 0
        assert result["parse_internal_seconds"] > 0
        assert result["projected_edges"] > 0
        assert result["projection_seconds"] > 0


def test_wp_m_scale_candidate_records_failed_gates_without_changing_inputs():
    baseline = json.loads(SCALE_EVIDENCE.read_text(encoding="utf-8"))
    candidate = json.loads(SCALE_CANDIDATE.read_text(encoding="utf-8"))

    assert candidate["schema"] == "exact-ontology-scale-candidate/1"
    assert candidate["status"] == "blocked"
    assert candidate["inputs_match_frozen_baseline"] is True
    assert candidate["accepted"]["single_load"] is True
    assert candidate["accepted"]["snapshot_identity"] is True
    assert candidate["accepted"]["no_second_ontology_representation"] is True
    assert candidate["accepted"]["performance_gate"] is False
    assert candidate["accepted"]["projection_parity_gate"] is False
    for role in ("source", "target"):
        assert (
            candidate["measurements"][role]["input"]["sha256"]
            == baseline["dataset"][role]["sha256"]
        )


@pytest.mark.parametrize("name", ["mini_src", "mini_tgt"])
def test_fixture_snapshot_parity(name):
    ontology = FIXTURES / f"{name}.owl"
    baseline_path = BASELINES / f"{name}.backend.json.zst"
    metadata = PROVENANCE["fixtures"][name]
    assert hashlib.sha256(ontology.read_bytes()).hexdigest() == metadata["source_sha256"]
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == metadata["snapshot_sha256"]

    expected = _load_snapshot(baseline_path)
    actual = json.loads(json.dumps(capture(ontology), sort_keys=True))
    for field in (
        "schema_version",
        "origin_name",
        "classes",
        "labels",
        "annotations",
        "attributes",
        "excluded_from_alignment",
        "property_domains",
        "property_ranges",
    ):
        assert actual[field] == expected[field], field

    assert _edge_set(actual, "taxonomy") == _edge_set(expected, "taxonomy")
    for projection in ("owl2vecstar", "owl2vecstar_literals"):
        observed = _edge_set(actual, projection)
        baseline = _edge_set(expected, projection)
        assert len(observed & baseline) / len(observed | baseline) >= 0.999

    allowed = metadata["allowed_elk_inferences"]
    for iri, expected_bundle in expected["hierarchy"].items():
        actual_bundle = actual["hierarchy"][iri]
        for family in ("part_of", "has_part"):
            assert actual_bundle[family] == expected_bundle[family]
        missing = sorted(set(expected_bundle["is_a"]) - set(actual_bundle["is_a"]))
        added = sorted(set(actual_bundle["is_a"]) - set(expected_bundle["is_a"]))
        assert added == []
        assert missing == allowed.get(iri, [])
