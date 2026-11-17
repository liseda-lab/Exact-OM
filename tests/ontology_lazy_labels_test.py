"""Graph construction must not traverse labels outside a bounded query population."""

from pathlib import Path

from exact.core.entities.ontology import OntologyGraph
from exact.ontology import load_ontology

FIXTURE = Path(__file__).parent / "fixtures/ontologies/mini_src.owl"
SRC = "http://example.org/mini/src#"


def test_native_graph_keeps_labels_lazy_and_memoizes_requested_entities(monkeypatch):
    source = load_ontology(FIXTURE)
    source.configure_projector(backend="native")
    original = source.labels
    queried = []

    def labels(iri):
        queried.append(iri)
        return original(iri)

    monkeypatch.setattr(source, "labels", labels)
    graph = OntologyGraph(source)
    assert graph.edges and queried == []
    assert graph.get_labels(SRC + "Heart") == ["coração", "heart"]
    assert graph.get_labels(SRC + "Heart") == ["coração", "heart"]
    assert queried == [SRC + "Heart"]
    assert graph.get_labels(SRC + "UnlabelledClass") == ["UnlabelledClass"]
    assert graph.get_labels('"plain literal"') == ["plain literal"]
    assert queried == [SRC + "Heart", SRC + "UnlabelledClass"]


def test_explicit_label_warmup_preserves_lazy_results_and_graph_statistics():
    source = load_ontology(FIXTURE)
    source.configure_projector(backend="native")
    graph = OntologyGraph(source)
    edges = [edge.astuple() for edge in graph.edges]
    statistics = dict(graph.edge_ic)
    labels = graph.get_labels(SRC + "Heart")
    graph.precompute_all_labels()
    assert graph.get_labels(SRC + "Heart") == labels
    assert [edge.astuple() for edge in graph.edges] == edges
    assert graph.edge_ic == statistics
    assert {iri for edge in edges for iri in edge}.issubset(graph.label_cache)
