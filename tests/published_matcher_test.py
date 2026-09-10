from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.experiments.paper_metrics import recompute_global_prf
from exact.experiments.published_matcher import run_cell, validate_binding
from exact.utils.provenance import sha256_path


def _cell(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "parameters.txt").write_text("print_output|false\n")
    (bundle / "matcher.jar").write_bytes(b"pinned test fixture")
    (tmp_path / "source.owl").write_text("source")
    (tmp_path / "target.owl").write_text("target")
    (tmp_path / "sources.txt").write_text("urn:s1\nurn:empty\nurn:s2\n")
    (tmp_path / "valid.tsv").write_text("SrcEntity\tTgtEntity\nurn:s1\turn:t1\nurn:s2\turn:t2\n")
    return SimpleNamespace(
        published_matcher={
            "matcher": "logmap",
            "bundle": {"path": str(bundle), "sha256": sha256_path(bundle)},
            "jar": "matcher.jar",
            "timeout_seconds": 60,
            "java_heap_gb": 2,
            "java_threads": 1,
        },
        source_cap=None,
        seed=17,
        recovery=None,
        config_hash="fixture",
        reference_role="valid",
        output_dir=tmp_path / "output",
        resolved_config={
            "data": {
                "root": str(tmp_path),
                "source": "source.owl",
                "target": "target.owl",
                "execution_mode": "global_alignment",
                "source_universe": "sources.txt",
                "refs": {"valid": "valid.tsv"},
            },
            "matching": {"entity_kinds": ["class"]},
            "evaluation": {"backends": ["builtin"]},
        },
    )


def test_published_output_uses_shared_evaluator_and_preserves_empty_sources(tmp_path, monkeypatch):
    cell = _cell(tmp_path)

    def invoke(command, cwd, output, timeout, stop):
        assert command[-1] == "false" and "MATCHER" in command
        assert not any("valid.tsv" in value for value in command)
        cells = ""
        for source, target, relation in [
            ("urn:s1", "urn:t1", "="),
            ("urn:s2", "urn:wrong", "="),
            ("urn:outside", "urn:t1", "="),
            ("urn:s2", "urn:t2", "&lt;"),
        ]:
            cells += f'<map><Cell><entity1 rdf:resource="{source}"/><entity2 rdf:resource="{target}"/><relation>{relation}</relation><measure>0.9</measure></Cell></map>'
        (cwd / "logmap2_mappings.rdf").write_text(
            f'<rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><Alignment>{cells}</Alignment></rdf:RDF>'
        )
        return 0, 0.01, 1000

    monkeypatch.setattr("exact.experiments.published_matcher._invoke", invoke)
    code, seconds, peak = run_cell(cell)
    result = recompute_global_prf(cell.output_dir).overall
    assert (code, peak) == (0, 1000)
    assert seconds >= 0.01
    assert result.metrics.f1 == 0.5
    assert set(result.by_source) == {"urn:s1", "urn:s2", "urn:empty"}
    assert result.by_source["urn:empty"].as_dict() == {"tp": 0, "fp": 0, "fn": 0}
    assert result.source_universe_sha256
    assert (cell.output_dir / "published/logmap/logmap2_mappings.rdf").exists()


def test_changed_bundle_and_local_mode_fail_before_launch(tmp_path, monkeypatch):
    cell = _cell(tmp_path)
    monkeypatch.setattr(
        "exact.experiments.published_matcher._invoke", lambda *args: pytest.fail("launched")
    )
    cell.resolved_config["data"]["execution_mode"] = "local_ranking"
    with pytest.raises(ValueError, match="global alignment only"):
        run_cell(cell)
    cell.resolved_config["data"]["execution_mode"] = "global_alignment"
    (Path(cell.published_matcher["bundle"]["path"]) / "matcher.jar").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bundle changed"):
        run_cell(cell)


def test_unbounded_or_escaping_matcher_is_rejected(tmp_path):
    cell = _cell(tmp_path)
    with pytest.raises(ValueError, match="timeout_seconds"):
        validate_binding({**cell.published_matcher, "timeout_seconds": 0})
    with pytest.raises(ValueError, match="inside its pinned bundle"):
        validate_binding({**cell.published_matcher, "jar": "../matcher.jar"})
