from pathlib import Path

from exact.core.entities.ontology import OntologyGraph
from exact.core.entities.graph import Edge
from exact.ontology import load_ontology


class CountedEdges:
    def __init__(self, rows):
        self.rows, self.scans, self.visits = rows, 0, 0

    def __iter__(self):
        self.scans += 1
        for row in self.rows:
            self.visits += 1
            yield row


def graph_with_edges(rows):
    graph = object.__new__(OntologyGraph)
    graph.edges = CountedEdges(rows)
    graph._example_triples_cache = {}
    graph._relations_cache = {'r', 's'}
    graph.get_relations = lambda human_readable=False: {'r', 's'}
    graph.get_relations_missing_domain = lambda human_readable=False: ['s']
    graph.get_relations_missing_range = lambda human_readable=False: []
    graph.get_labels = lambda iri: ['label:' + iri]
    return graph


def test_examples_scan_once_preserve_order_and_cache_options():
    edges = [Edge('a','r','b'), Edge('c','s','d'), Edge('e','r','f'), Edge('g','s','h')]
    graph = graph_with_edges(edges)
    assert graph.get_example_triples(1, human_readable=False) == {'r':[('a','r','b')], 's':[('c','s','d')]}
    assert graph.edges.scans == 1 and graph.edges.visits == 2
    assert graph.get_example_triples(1)['label:r'] == [('label:a','label:r','label:b')]
    assert graph.edges.scans == 1
    assert graph.get_example_triples(2, human_readable=False)['r'] == [('a','r','b'), ('e','r','f')]
    assert graph.edges.scans == 2
    assert graph.get_example_triples(1, exclude_missing_dr=True, human_readable=False) == {'r':[('a','r','b')]}
    assert graph.edges.scans == 3
    graph.get_labels = lambda iri: ['changed:' + iri]
    assert 'changed:r' in graph.get_example_triples(1)
    assert graph.edges.scans == 3


def test_nonpositive_counts_are_empty_without_scanning():
    graph = graph_with_edges([Edge('a','r','b')])
    assert graph.get_example_triples(0, human_readable=False) == {'r':[], 's':[]}
    assert graph.get_example_triples(-3, human_readable=False) == {'r':[], 's':[]}
    assert graph.edges.scans == 0


def test_native_produced_edge_examples_match_positive_count_legacy():
    source = load_ontology(Path(__file__).parent / 'fixtures/ontologies/mini_src.owl')
    native_edges = source.projection_edges(include_literals=True)
    assert native_edges
    graph = graph_with_edges(native_edges)
    graph.get_relations = lambda human_readable=False: {edge.rel for edge in native_edges}
    relations = graph.get_relations()
    graph._relations_cache = relations
    expected = {rel:[edge.astuple() for edge in native_edges if edge.rel == rel][:3] for rel in relations}
    assert graph.get_example_triples(3, human_readable=False) == expected
    assert graph.edges.scans == 1


def test_cold_relation_inventory_and_examples_share_one_scan():
    graph = graph_with_edges([Edge('a','r','b'), Edge('c','s','d'), Edge('e','r','f')])
    graph._relations_cache = None
    del graph.get_relations
    assert graph.get_example_triples(1, human_readable=False) == {'r':[('a','r','b')], 's':[('c','s','d')]}
    assert graph.edges.scans == 1
    assert graph.get_relations(False) == {'r','s'}
    assert graph.edges.scans == 1
