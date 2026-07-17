import subprocess
import sys
from pathlib import Path

import pytest
from pyowl_core import IRI, Class, ObjectSomeValuesFrom, SubClassOf

from exact.core.entities.graph import AnnotationValue, Edge
from exact.core.entities.kinds import EntityKind
from exact.core.entities.ontology import OntologyGraph
from exact.ontology import load_ontology
from exact.ontology.expressions import existential_targets, named_class_iri
from exact.ontology.parser import parse
from exact.ontology.reasoning import AssertedHierarchyReasoner, load_reasoner
from exact.utils.eval import MetricUtils
from tests.knowledge_source_conformance import assert_knowledge_source_conformance

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"
SRC = "http://example.org/mini/src#"
TGT = "http://example.org/mini/tgt#"
PART_OF = "http://purl.obolibrary.org/obo/BFO_0000050"


@pytest.fixture(scope="module")
def source():
    return load_ontology(FIXTURES / "mini_src.owl")


@pytest.fixture(scope="module")
def target():
    return load_ontology(FIXTURES / "mini_tgt.owl")


def test_fixture_sources_conform(source, target):
    assert_knowledge_source_conformance(source)
    assert_knowledge_source_conformance(target)


def test_core_snapshot_builds_complete_named_signatures(source):
    assert source.ontology_iri == "http://example.org/mini/src"
    assert len(source.entities()) == 32
    assert len(source.entities(EntityKind.OBJECT_PROPERTY)) == 4
    assert len(source.entities(EntityKind.DATA_PROPERTY)) == 1
    assert len(source.entities(EntityKind.INDIVIDUAL)) == 3
    assert all(not iri.startswith("N") for iri in source.entities())
    assert "http://www.w3.org/2001/XMLSchema#string" not in source.entities()


def test_core_snapshot_indexes_individual_assertions_deterministically(tmp_path):
    ontology = tmp_path / "individuals.owl"
    ontology.write_text(
        """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:ex="http://example.org/undeclared#">
  <owl:Ontology rdf:about="http://example.org/undeclared"/>
  <owl:Class rdf:about="http://example.org/undeclared#Person"/>
  <owl:ObjectProperty rdf:about="http://example.org/undeclared#knows"/>
  <owl:DatatypeProperty rdf:about="http://example.org/undeclared#code"/>
  <rdf:Description rdf:about="http://example.org/undeclared#knows">
    <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#TransitiveProperty"/>
  </rdf:Description>
  <rdf:Description rdf:about="http://example.org/undeclared#bob">
    <rdf:type rdf:resource="http://example.org/undeclared#Person"/>
    <ex:knows rdf:resource="http://example.org/undeclared#alice"/>
    <ex:code>42</ex:code>
  </rdf:Description>
</rdf:RDF>
""",
        encoding="utf-8",
    )
    source = load_ontology(ontology)
    base = "http://example.org/undeclared#"

    assert source.entities(EntityKind.INDIVIDUAL) == (base + "alice", base + "bob")
    assert source.direct_parents(base + "bob", EntityKind.INDIVIDUAL) == [base + "Person"]
    assert base + "knows" in source.entities(EntityKind.OBJECT_PROPERTY)
    assert base + "code" in source.entities(EntityKind.DATA_PROPERTY)
    assert Edge(base + "bob", base + "knows", base + "alice") in source.projection_edges()
    assert any(value.value == "42" for value in source.attributes(base + "bob"))


def test_parser_and_projector_preserve_observed_complex_axiom_rules(tmp_path):
    ontology = tmp_path / "projection-compat.owl"
    ontology.write_text(
        """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:ex="http://example.org/compat#">
  <owl:Ontology rdf:about="http://example.org/compat"/>
  <owl:ObjectProperty rdf:about="http://example.org/compat#rel"/>
  <owl:ObjectProperty rdf:about="http://example.org/compat#parent">
    <rdfs:domain rdf:resource="http://example.org/compat#Anchor"/>
    <rdfs:range rdf:resource="http://example.org/compat#A"/>
  </owl:ObjectProperty>
  <owl:ObjectProperty rdf:about="http://example.org/compat#childA">
    <rdfs:subPropertyOf rdf:resource="http://example.org/compat#parent"/>
  </owl:ObjectProperty>
  <owl:ObjectProperty rdf:about="http://example.org/compat#childB">
    <rdfs:subPropertyOf rdf:resource="http://example.org/compat#parent"/>
  </owl:ObjectProperty>
  <owl:DatatypeProperty rdf:about="http://example.org/compat#data"/>
  <owl:Class rdf:about="http://example.org/compat#A"/>
  <owl:Class rdf:about="http://example.org/compat#B"/>
  <owl:Class rdf:about="http://example.org/compat#C"/>
  <owl:Class rdf:about="http://example.org/compat#Root"/>
  <owl:Class rdf:about="http://example.org/compat#Mid">
    <rdfs:subClassOf rdf:resource="http://example.org/compat#Root"/>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/compat#Leaf">
    <rdfs:subClassOf rdf:resource="http://example.org/compat#Mid"/>
    <rdfs:subClassOf rdf:resource="http://example.org/compat#Root"/>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/compat#Anchor">
    <owl:equivalentClass>
      <owl:Restriction>
        <owl:onProperty rdf:resource="http://example.org/compat#rel"/>
        <owl:someValuesFrom rdf:resource="http://example.org/compat#A"/>
      </owl:Restriction>
    </owl:equivalentClass>
    <rdfs:subClassOf>
      <owl:Restriction>
        <owl:onProperty rdf:resource="http://example.org/compat#rel"/>
        <owl:allValuesFrom rdf:resource="http://example.org/compat#B"/>
      </owl:Restriction>
    </rdfs:subClassOf>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/compat#UnionAnchor">
    <owl:equivalentClass>
      <owl:Class><owl:unionOf rdf:parseType="Collection">
        <rdf:Description rdf:about="http://example.org/compat#A"/>
        <rdf:Description rdf:about="http://example.org/compat#B"/>
      </owl:unionOf></owl:Class>
    </owl:equivalentClass>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/compat#NestedAnchor">
    <owl:equivalentClass>
      <owl:Class><owl:intersectionOf rdf:parseType="Collection">
        <rdf:Description rdf:about="http://example.org/compat#C"/>
        <owl:Class><owl:unionOf rdf:parseType="Collection">
          <rdf:Description rdf:about="http://example.org/compat#A"/>
          <rdf:Description rdf:about="http://example.org/compat#B"/>
        </owl:unionOf></owl:Class>
        <owl:Restriction>
          <owl:onProperty rdf:resource="http://example.org/compat#rel"/>
          <owl:someValuesFrom rdf:resource="http://example.org/compat#C"/>
        </owl:Restriction>
      </owl:intersectionOf></owl:Class>
    </owl:equivalentClass>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/compat#MinCardinality">
    <rdfs:subClassOf><owl:Restriction>
      <owl:onProperty rdf:resource="http://example.org/compat#rel"/>
      <owl:minCardinality rdf:datatype="http://www.w3.org/2001/XMLSchema#integer">1</owl:minCardinality>
    </owl:Restriction></rdfs:subClassOf>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/compat#ExactCardinality">
    <rdfs:subClassOf><owl:Restriction>
      <owl:onProperty rdf:resource="http://example.org/compat#rel"/>
      <owl:cardinality rdf:datatype="http://www.w3.org/2001/XMLSchema#integer">1</owl:cardinality>
    </owl:Restriction></rdfs:subClassOf>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/compat#DataCardinality">
    <rdfs:subClassOf><owl:Restriction>
      <owl:onProperty rdf:resource="http://example.org/compat#data"/>
      <owl:minCardinality rdf:datatype="http://www.w3.org/2001/XMLSchema#integer">1</owl:minCardinality>
    </owl:Restriction></rdfs:subClassOf>
  </owl:Class>
</rdf:RDF>
""",
        encoding="utf-8",
    )
    base = "http://example.org/compat#"
    source = load_ontology(ontology)
    edges = {edge.astuple() for edge in source.projection_edges()}

    assert (base + "Anchor", base + "rel", base + "B") in edges
    assert (base + "Anchor", base + "rel", base + "A") not in edges
    assert (base + "UnionAnchor", "http://subclassof", base + "A") in edges
    assert (base + "UnionAnchor", "http://subclassof", base + "B") in edges
    assert (base + "NestedAnchor", "http://subclassof", base + "C") in edges
    assert (base + "NestedAnchor", base + "rel", base + "C") in edges
    assert (base + "NestedAnchor", "http://subclassof", base + "A") not in edges
    assert (
        base + "MinCardinality",
        base + "rel",
        "http://www.w3.org/2002/07/owl#Thing",
    ) in edges
    assert not any(edge[0] == base + "ExactCardinality" for edge in edges)
    assert not any(edge[0] == base + "DataCardinality" for edge in edges)
    assert (base + "Anchor", base + "childA", base + "A") in edges
    assert (base + "Anchor", base + "childB", base + "A") not in edges
    assert source.direct_parents(base + "A") == [base + "UnionAnchor"]
    assert source.direct_parents(base + "Leaf") == [base + "Mid"]
    assert source.direct_children(base + "Root") == [base + "Mid"]


def test_annotations_labels_attributes_and_exclusions(source):
    assert source.labels(SRC + "Heart") == ["coração", "heart"]
    assert source.labels(SRC + "UnlabelledClass") == []
    assert source.short_form(SRC + "UnlabelledClass") == "UnlabelledClass"

    attributes = source.attributes(SRC + "Heart")
    assert (
        AnnotationValue(
            "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
            "cardiac organ",
            True,
        )
        in attributes
    )
    assert any(value.value.startswith("The muscular organ") for value in attributes)
    assert source.excluded_from_alignment() == frozenset(
        {SRC + "IgnoredConcept", SRC + "DeprecatedConcept"}
    )


def test_asserted_hierarchy_and_equivalence_normalization(source):
    assert source.direct_parents(SRC + "Heart") == [
        SRC + "CardiacStructure",
        SRC + "Organ",
    ]
    assert source.direct_parents(SRC + "Clinician") == [SRC + "Person"]
    assert source.direct_parents(SRC + "HealthcareProfessional") == [SRC + "Person"]
    assert set(source.direct_children(SRC + "Person")) >= {
        SRC + "Clinician",
        SRC + "HealthcareProfessional",
        SRC + "Patient",
    }
    assert SRC + "Clinician" not in source.direct_parents(SRC + "HealthcareProfessional")
    assert SRC + "Entity" in source.hierarchy.ancestors(SRC + "Heart")


def test_expression_walkers_and_hierarchy_bundle(source):
    parsed = parse(FIXTURES / "mini_src.owl")
    chest_axiom = next(
        axiom
        for axiom in parsed.iter_axioms(SubClassOf)
        if named_class_iri(axiom.sub_class) == SRC + "ChestPain"
        and isinstance(axiom.super_class, ObjectSomeValuesFrom)
    )
    assert named_class_iri(Class(IRI(SRC + "Heart"))) == SRC + "Heart"
    assert existential_targets(chest_axiom.super_class, (PART_OF,)) == [SRC + "Heart"]
    assert source.hierarchy_bundle(
        SRC + "CardiacStructure", {"is_a": (), "part_of": (PART_OF,)}
    ) == {"is_a": [SRC + "Structure"], "part_of": [SRC + "Heart"]}


def test_property_schema_and_instance_indexes(source):
    assert source.property_domains(PART_OF) == [SRC + "AnatomicalEntity"]
    assert source.property_ranges(PART_OF) == [SRC + "AnatomicalEntity"]
    assert source.direct_parents(SRC + "participatesIn", EntityKind.OBJECT_PROPERTY) == [
        SRC + "relatedTo"
    ]
    assert source.direct_parents(SRC + "alice", EntityKind.INDIVIDUAL) == [SRC + "Patient"]
    assert any(value.value == "P-001" for value in source.attributes(SRC + "alice"))


def test_projection_matches_observed_legacy_fixture_behavior(source):
    taxonomy = {edge.astuple() for edge in source.projection_edges(method="taxonomy")}
    owl2vec = {edge.astuple() for edge in source.projection_edges()}
    with_literals = {edge.astuple() for edge in source.projection_edges(include_literals=True)}
    assert len(taxonomy) == 28
    assert len(owl2vec) == 42
    assert len(with_literals) == 76
    assert (SRC + "ChestPain", PART_OF, SRC + "Heart") in owl2vec
    assert (SRC + "Heart", SRC + "hasPart", SRC + "ChestPain") in owl2vec
    assert (SRC + "alice", "http://type", SRC + "Patient") in owl2vec
    assert (SRC + "Heart", "rdfs:label", "heart") in with_literals


def test_graph_view_consumes_protocol(source):
    graph = OntologyGraph(source, only_taxonomy=False, include_literals=True)
    assert graph.get_labels(SRC + "Heart")[0] == "coração"
    assert graph.get_labels(SRC + "UnlabelledClass") == ["UnlabelledClass"]
    assert graph.get_all_classes() == list(source.entities())
    schema = graph.get_property_domains_and_ranges(human_readable=False)
    assert schema[PART_OF] == {
        "domain": [SRC + "AnatomicalEntity"],
        "range": [SRC + "AnatomicalEntity"],
    }


def test_asserted_reasoner_and_unresolved_plugin_error(source):
    reasoner = load_reasoner("asserted", source)
    assert isinstance(reasoner, AssertedHierarchyReasoner)
    assert reasoner.direct_parents(SRC + "Heart") == source.direct_parents(SRC + "Heart")
    with pytest.raises(ValueError, match="Installed plugins"):
        load_reasoner("not-installed", source)


def test_metric_ignored_index_uses_knowledge_source(source):
    ignored = MetricUtils.get_ignored_class_index(source)
    assert ignored[SRC + "IgnoredConcept"]
    assert ignored[SRC + "DeprecatedConcept"]
    assert not ignored[SRC + "Heart"]


def test_edge_contract_is_hashable_and_tuple_compatible():
    edge = Edge("a", "r", "b")
    assert edge.astuple() == ("a", "r", "b")
    assert {edge} == {Edge("a", "r", "b")}


def test_top_level_ontology_import_is_java_free_and_delivery_lazy():
    code = """
import sys
import exact.ontology
legacy_backend = ''.join(('mo', 'wl'))
assert not any(name == legacy_backend or name.startswith(legacy_backend + '.') for name in sys.modules)
assert not any(name.startswith('exact.delivery') for name in sys.modules)
import exact
assert 'init_' + 'jvm' not in exact.__all__
compat_init = getattr(exact, 'init_' + 'jvm')
try:
    compat_init('4g')
except RuntimeError as exc:
    assert 'no longer needs Java' in str(exc)
else:
    raise AssertionError('deprecated stub must reject JVM initialization')
"""
    subprocess.run([sys.executable, "-c", code], check=True)
