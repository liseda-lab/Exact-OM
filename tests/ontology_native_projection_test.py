"""Native compiler execution, semantic parity, and fail-closed fallback checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyowl2vec_star_projector as upstream
import pyowl2vec_star_projector.api as upstream_api
import pyowl_core
import pytest

import exact.ontology.native_projection as native
from exact.core.entities.configs.config import ProjectorConfig
from exact.io.sources.csv_kg import CsvKgSource
from exact.ontology import load_ontology
from exact.ontology.projection import ProjectorSettings, SharedProjectionAdapter

FIXTURES = Path(__file__).parent / "fixtures"


def _forbid(*args, **kwargs):
    raise AssertionError("scalar compilation or materialization is forbidden")


@pytest.mark.parametrize("method", ["owl2vecstar", "taxonomy"])
@pytest.mark.parametrize("include_literals", [False, True])
def test_native_projection_matches_reference_without_scalar_work(
    monkeypatch, method, include_literals
):
    source = load_ontology(FIXTURES / "ontologies/mini_src.owl")
    view = source.owl_snapshot()
    oracle = upstream.Projector()
    expected = (
        oracle.project_taxonomy(view, backend="python", duplicates="unique", order="canonical")
        if method == "taxonomy"
        else oracle.project(
            view,
            options=upstream.ProjectionOptions(
                backend="python",
                include_literals=include_literals,
                duplicates="unique",
                order="canonical",
            ),
        )
    )
    for name in (
        "prepare_streaming_compilation",
        "prepare_encoded_subset_compilation",
        "iter_asserted_taxonomy",
    ):
        monkeypatch.setattr(upstream_api, name, _forbid)
    monkeypatch.setattr(type(view), "iter_axioms", _forbid)
    edges = source.projection_edges(method=method, include_literals=include_literals)
    assert [edge.astuple() for edge in edges] == [
        (edge.source, edge.relation, edge.destination) for edge in expected
    ]
    assert source.projector.last_view is view
    report = source.projector.last_report.to_dict()["provenance"]
    assert report["selected_backend"] == "native"
    assert report["ingestion"]["path"] == "encoded-native"
    assert report["ingestion"]["counters"]["scalar_axiom_materializations"] == 0
    # A cached result must not make another compiler call.
    monkeypatch.setattr(native, "prepare_native_encoded_compilation", _forbid)
    assert source.projection_edges(method=method, include_literals=include_literals) == edges


@pytest.mark.parametrize("method", ["owl2vecstar", "taxonomy"])
@pytest.mark.parametrize("failure", ["decline", "unsupported", "unavailable"])
def test_native_failure_never_invokes_scalar_fallback(monkeypatch, method, failure):
    source = load_ontology(FIXTURES / "ontologies/mini_src.owl")
    view = source.owl_snapshot()
    for name in (
        "prepare_streaming_compilation",
        "prepare_encoded_subset_compilation",
        "iter_asserted_taxonomy",
    ):
        monkeypatch.setattr(upstream_api, name, _forbid)
    monkeypatch.setattr(type(view), "iter_axioms", _forbid)

    def fail(*args, **kwargs):
        if failure == "unsupported":
            raise native.NativeEncodedDirectUnsupported("fixture shape")
        if failure == "unavailable":
            raise upstream.NativeBackendUnavailableError("fixture native unavailable")
        return None, "fixture native decline"

    monkeypatch.setattr(native, "prepare_native_encoded_compilation", fail)
    with pytest.raises(upstream.NativeBackendUnavailableError, match="fixture"):
        source.projection_edges(method=method)
    assert source._projection.cache_keys == ()
    assert source.projector.last_report is None


def test_reported_scalar_work_cannot_be_cached(monkeypatch):
    source = load_ontology(FIXTURES / "ontologies/mini_src.owl")
    source.projection_edges()
    payload = source.projector.last_report.to_dict()
    payload["provenance"]["ingestion"]["counters"]["scalar_axiom_materializations"] = 1

    class TamperedProjector(native.NativeProjector):
        def project(self, *args, **kwargs):
            return []

        @property
        def last_report(self):
            return SimpleNamespace(to_dict=lambda: payload)

    fake = TamperedProjector()
    adapter = SharedProjectionAdapter(source.owl_snapshot(), projector=fake)
    with pytest.raises(RuntimeError, match="forbidden scalar work"):
        adapter.edges()
    assert adapter.cache_keys == ()


def test_legacy_auto_is_only_an_alias_for_native_and_python_is_rejected():
    assert ProjectorSettings.from_value({"backend": "auto"}).backend == "native"
    assert ProjectorConfig.model_validate({"backend": "auto"}).backend == "native"
    assert ProjectorConfig().backend == "native"
    with pytest.raises(ValueError):
        ProjectorSettings.from_value({"backend": "python"})
    with pytest.raises(ValueError):
        ProjectorConfig.model_validate({"backend": "python"})


def test_csv_declared_edges_do_not_enter_owl_compilation(monkeypatch):
    monkeypatch.setattr(native.NativeProjector, "project", _forbid)
    source = CsvKgSource.from_path(FIXTURES / "kg_csv")
    edges = source.projection_edges(include_literals=True)
    assert edges
    assert edges == sorted(edges, key=lambda edge: edge.astuple())


def test_imported_closure_is_projected_natively(monkeypatch, tmp_path):
    imported = tmp_path / "imported.ofn"
    imported.write_text(
        "Ontology(<urn:imported> Declaration(Class(<urn:B>)) Declaration(Class(<urn:C>)) SubClassOf(<urn:B> <urn:C>))"
    )
    root = tmp_path / "root.ofn"
    root.write_text(
        f"Ontology(<urn:root> Import(<{imported.as_uri()}>) Declaration(Class(<urn:A>)) SubClassOf(<urn:A> <urn:B>))"
    )
    source = load_ontology(
        root, resolver=pyowl_core.MappingResolver({imported.as_uri(): imported.read_bytes()})
    )
    view = source.owl_snapshot()
    assert view.is_complete
    monkeypatch.setattr(type(view), "iter_axioms", _forbid)
    edges = {edge.astuple() for edge in source.projection_edges(method="taxonomy")}
    assert ("urn:A", "http://subclassof", "urn:B") in edges
    assert ("urn:B", "http://subclassof", "urn:C") in edges
    assert source.projector.last_report.provenance.ingestion.path == "encoded-native"


def test_plain_upstream_projector_injection_is_rejected():
    source = load_ontology(FIXTURES / "ontologies/mini_src.owl")
    with pytest.raises(TypeError, match="injectable scalar fallback"):
        SharedProjectionAdapter(source.owl_snapshot(), projector=upstream.Projector())
