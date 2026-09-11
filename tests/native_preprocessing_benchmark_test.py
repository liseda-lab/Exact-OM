import json
from pathlib import Path

import pytest
import yaml

from tools.benchmark_native_preprocessing import _digest, _sha256, benchmark

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"


def config_file(tmp_path, **changes):
    config = {
        "data": {
            "source": str((FIXTURES / "mini_src.owl").resolve()),
            "target": str((FIXTURES / "mini_tgt.owl").resolve()),
            "refs": {"test": "/must/not/read/private-reference.tsv"},
            "train_candidates": "/must/not/read/training.tsv",
        },
        "dataset": {
            "reasoner": "asserted",
            "projector": {"backend": "auto", "profile": "mowl-d993536-v1"},
            "projection_include_literals": False,
            "n_hops": 1,
        },
        "matching": {"entity_kinds": ["class"]},
        "io": {},
        "run": {"seed": 17},
    }
    config.update(changes)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_real_native_phases_preserve_semantics_without_models(tmp_path, monkeypatch):
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from exact.llm.routing import OpenRouterClient

    def forbidden(*args, **kwargs):
        raise AssertionError("Benchmark attempted model or hosted work")

    monkeypatch.setattr(SentenceTransformer, "__init__", forbidden)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", forbidden)
    monkeypatch.setattr(OpenRouterClient, "_http_json", forbidden)
    config = config_file(tmp_path)
    first = benchmark(config, tmp_path / "first.json", entity_limit=3)
    import exact.ontology.native_projection as native
    import exact.ontology.projection as projection

    original_prepare = native.prepare_native_encoded_compilation
    original_cache_key = projection.cache_key
    original_report = native.NativeProjector._report
    owns_report = "_report" in vars(native.NativeProjector)
    profile = tmp_path / "profile.jsonl"
    second = benchmark(config, tmp_path / "second.json", entity_limit=3, profile_stages=profile)
    assert native.prepare_native_encoded_compilation is original_prepare
    assert projection.cache_key is original_cache_key
    assert native.NativeProjector._report is original_report
    assert ("_report" in vars(native.NativeProjector)) is owns_report
    events = [json.loads(line) for line in profile.read_text().splitlines()]
    starts = {row["call"]: row for row in events if row["status"] == "running"}
    ends = {row["call"]: row for row in events if row["status"] != "running"}
    assert starts.keys() == ends.keys()
    assert {row["stage"] for row in starts.values()} == {
        "adapter_total",
        "cache_key_and_fingerprints",
        "native_projection_total",
        "native_ingestion_selection",
        "native_prepare_compile",
        "edge_policy_iteration",
        "native_report",
        "adapter_report_validation",
    }
    assert all(row["wall_seconds"] >= 0 and row["cpu_seconds"] >= 0 for row in ends.values())
    assert all(row["status"] == "complete" for row in ends.values())
    assert first["status"] == "complete"
    assert first["load_report"]["backend"] == "native"
    assert first["ontology_stack"]["projector"]["selection"]["effective"] == "native"
    assert first["signature"]["counts"]["class"] == 32
    assert first["exclusions"]["count"] == 2
    assert first["edges"]["count"] == 42
    assert isinstance(first["projection_spill"], dict)
    assert first["entities"]["count"] == first["features"]["count"] == 3
    for field in ("signature", "exclusions", "labels", "edges", "entities", "features"):
        assert first[field]["sha256"] == second[field]["sha256"]
    assert [phase["name"] for phase in first["phases"]] == [
        "load",
        "signature",
        "projection",
        "provenance",
        "edge_digest",
        "exclusions",
        "labels",
        "graph_wrapper",
        "feature_wrapper",
        "entity_features",
        "entity_features_cached",
    ]
    assert all(phase["wall_seconds"] >= 0 for phase in first["phases"])
    assert first["process_peak_rss_bytes"] > 0
    assert json.loads((tmp_path / "first.json").read_text()) == first
    with pytest.raises(FileExistsError):
        benchmark(config, tmp_path / "first.json")


def test_import_hashes_verified_before_load(tmp_path):
    imported = tmp_path / "import.owl"
    imported.write_text("not parsed")
    config = config_file(
        tmp_path,
        io={
            "target_options": {
                "imports": {"urn:import": {"path": str(imported), "sha256": "0" * 64}}
            }
        },
    )
    with pytest.raises(ValueError, match="checksum"):
        benchmark(config, tmp_path / "output.json", side="target")
    assert not (tmp_path / "output.json").exists()


def test_pinned_import_is_used_and_records_complete_closure(tmp_path):
    imported = tmp_path / "import.ofn"
    imported.write_text("Ontology(<urn:import> Declaration(Class(<urn:Imported>)))")
    root = tmp_path / "root.ofn"
    root.write_text(
        "Ontology(<urn:root> Import(<urn:import>) Declaration(Class(<urn:Root>)) SubClassOf(<urn:Root> <urn:Imported>))"
    )
    config = config_file(
        tmp_path,
        data={"source": str(root)},
        io={
            "source_options": {
                "imports": {"urn:import": {"path": imported.name, "sha256": _sha256(imported)}}
            }
        },
    )
    report = benchmark(config, tmp_path / "report.json", entity_limit=1)
    assert report["signature"]["counts"]["class"] == 2
    assert report["ontology_stack"]["core"]["closure"]["complete"]
    assert report["ontology_stack"]["core"]["closure"]["document_count"] == 2
    assert report["input"]["imports"]["urn:import"]["sha256"] == _sha256(imported)


def test_failure_retains_completed_phases_and_explicit_entity_guard(tmp_path):
    selected = tmp_path / "entities.txt"
    selected.write_text("urn:absent\n")
    with pytest.raises(ValueError, match="eligible members"):
        benchmark(config_file(tmp_path), tmp_path / "failed.json", entities_path=selected)
    report = json.loads((tmp_path / "failed.json").read_text())
    assert report["status"] == "failed"
    assert report["phases"][-1]["name"] == "graph_wrapper"
    assert all(phase["status"] == "complete" for phase in report["phases"])
    assert report["failure"]["type"] == "ValueError"
    assert report["ontology_stack"]["projector"]["selection"]["effective"] == "native"


def test_semantic_digest_binds_all_edge_fields():
    assert _digest([("a", "r", "b")]) != _digest([("a", "r", "c")])
    assert _digest([("a", "r", "b")]) != _digest([("a", "s", "b")])


def test_failed_native_load_keeps_a_phase_report(tmp_path):
    invalid = tmp_path / "invalid.owl"
    invalid.write_text("not an ontology")
    config = config_file(tmp_path, data={"source": str(invalid)})
    with pytest.raises(Exception):
        benchmark(config, tmp_path / "failed-load.json")
    report = json.loads((tmp_path / "failed-load.json").read_text())
    assert report["status"] == "failed"
    assert report["phases"][0]["name"] == "load"
    assert report["phases"][0]["status"] == "failed"
    assert report["phases"][0]["wall_seconds"] >= 0


def test_boundary_profile_restores_callables_after_failure(tmp_path, monkeypatch):
    import exact.ontology.native_projection as native
    import exact.ontology.projection as projection
    from tools.benchmark_native_preprocessing import _profile_boundaries

    def fail(*args, **kwargs):
        raise RuntimeError("profile fixture failure")

    monkeypatch.setattr(native, "prepare_native_encoded_compilation", fail)
    original_key = projection.cache_key
    path = tmp_path / "failed-profile.jsonl"
    with pytest.raises(RuntimeError, match="profile fixture failure"):
        with _profile_boundaries(path):
            native.prepare_native_encoded_compilation()
    assert native.prepare_native_encoded_compilation is fail
    assert projection.cache_key is original_key
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["running", "failed"]
    assert rows[-1]["wall_seconds"] >= 0
    with pytest.raises(FileExistsError):
        with _profile_boundaries(path):
            pass
    assert projection.cache_key is original_key
