import subprocess
import sys
from pathlib import Path

import pytest

import exact.io.sources as sources_module
from exact.core.entities.graph import AnnotationValue, Edge
from exact.core.entities.kinds import EntityKind
from exact.io.sources import SourceOptionsError, SourceRegistry, infer_format, resolve
from exact.io.sources.csv_kg import CsvKgSource
from exact.io.sources.datalog import parse_fact
from exact.io.sources.rdf import RdfSource
from tests.knowledge_source_conformance import assert_knowledge_source_conformance

FIXTURES = Path(__file__).parent / "fixtures"
KG = FIXTURES / "kg_csv"
RDF = FIXTURES / "rdf" / "mini.ttl"
OWL = FIXTURES / "ontologies" / "mini_src.owl"
KG_BASE = "http://example.org/kg/"
RDF_BASE = "http://example.org/rdf/"


@pytest.fixture(scope="module")
def rdf_source() -> RdfSource:
    source = resolve(RDF)
    assert isinstance(source, RdfSource)
    return source


@pytest.fixture(scope="module")
def csv_source() -> CsvKgSource:
    source = resolve(KG)
    assert isinstance(source, CsvKgSource)
    return source


def test_rdf_and_csv_sources_pass_shared_conformance(rdf_source, csv_source) -> None:
    assert_knowledge_source_conformance(rdf_source)
    assert_knowledge_source_conformance(csv_source)


def test_rdf_source_maps_schema_annotations_and_projection(rdf_source: RdfSource) -> None:
    assert rdf_source.direct_parents(RDF_BASE + "Heart") == [RDF_BASE + "Organ"]
    assert rdf_source.direct_children(RDF_BASE + "Heart") == [RDF_BASE + "Atrium"]
    assert rdf_source.labels(RDF_BASE + "Heart") == ["Heart", "Cardiac organ"]
    assert AnnotationValue(
        RDF_BASE + "description",
        "A muscular pump",
        True,
        lang="en",
    ) in rdf_source.attributes(RDF_BASE + "Heart")
    assert rdf_source.hierarchy_bundle(
        RDF_BASE + "Atrium", {"is_a": (), "part_of": (RDF_BASE + "partOf",)}
    ) == {
        "is_a": [RDF_BASE + "Heart"],
        "part_of": [RDF_BASE + "Heart"],
    }
    assert rdf_source.property_domains(RDF_BASE + "partOf") == [RDF_BASE + "Entity"]
    assert rdf_source.property_ranges(RDF_BASE + "partOf") == [RDF_BASE + "Entity"]
    assert RDF_BASE + "Person" in rdf_source.direct_parents(
        RDF_BASE + "alice", EntityKind.INDIVIDUAL
    )
    assert rdf_source.excluded_from_alignment() == frozenset({RDF_BASE + "Deprecated"})
    assert (
        Edge(RDF_BASE + "Atrium", RDF_BASE + "partOf", RDF_BASE + "Heart")
        in rdf_source.projection_edges()
    )
    assert Edge(
        RDF_BASE + "Heart", RDF_BASE + "description", "A muscular pump"
    ) in rdf_source.projection_edges(include_literals=True)


def test_csv_source_merges_datalog_and_descriptor_semantics(
    csv_source: CsvKgSource,
) -> None:
    assert csv_source.entities() == (
        KG_BASE + "atrium",
        KG_BASE + "heart",
        KG_BASE + "organ",
    )
    assert csv_source.entities(EntityKind.INDIVIDUAL) == (KG_BASE + "blood",)
    assert csv_source.labels(KG_BASE + "heart") == ["Cardiac organ", "Heart"]
    assert csv_source.direct_parents(KG_BASE + "atrium") == [KG_BASE + "heart"]
    assert csv_source.ancestors(KG_BASE + "atrium") == {
        KG_BASE + "heart",
        KG_BASE + "organ",
    }
    assert csv_source.hierarchy_bundle(
        KG_BASE + "atrium", {"is_a": (), "part_of": ("part_of",)}
    ) == {
        "is_a": [KG_BASE + "heart"],
        "part_of": [KG_BASE + "heart"],
    }
    assert AnnotationValue("description", "muscular pump", True) in csv_source.attributes(
        KG_BASE + "heart"
    )
    assert Edge(KG_BASE + "atrium", "part_of", KG_BASE + "heart") in (csv_source.projection_edges())
    assert Edge(KG_BASE + "heart", "description", "muscular pump") not in (
        csv_source.projection_edges()
    )


def test_source_auto_dispatch_and_owl_options() -> None:
    assert infer_format(OWL) == "owl"
    assert infer_format(RDF) == "rdf"
    assert infer_format(KG) == "csv-kg"
    assert infer_format(Path("alignment.rdf")) == "owl"
    schema_only = resolve(OWL, options={"include_abox": False})
    assert schema_only.entities(EntityKind.INDIVIDUAL) == ()
    assert schema_only.entities(EntityKind.CLASS)


def test_rdf_configured_type_selector_limits_matching_signature() -> None:
    selected = resolve(
        RDF,
        format="rdf",
        options={"entity_selector": {"class": ["http://www.w3.org/2002/07/owl#Class"]}},
    )
    assert RDF_BASE + "Heart" in selected.entities(EntityKind.CLASS)
    assert RDF_BASE + "alice" not in selected.entities(EntityKind.CLASS)
    assert selected.entities(EntityKind.INDIVIDUAL) == ()


def test_csv_descriptor_and_datalog_fail_safely() -> None:
    with pytest.raises(SourceOptionsError, match="safe relative path"):
        resolve(
            KG,
            format="csv-kg",
            options={"triples_files": [{"path": "../outside.csv"}]},
        )
    assert parse_fact("related(a, b).") == Edge("a", "related", "b")
    assert parse_fact("part_of(http://example.org/x#A, http://example.org/x#B).") == Edge(
        "http://example.org/x#A", "part_of", "http://example.org/x#B"
    )
    assert parse_fact("  # comment") is None
    with pytest.raises(SourceOptionsError, match="rules are not supported"):
        parse_fact("ancestor(X, Y) :- parent(X, Y).")
    with pytest.raises(SourceOptionsError, match="binary facts"):
        parse_fact("ternary(a, b, c).")


def test_source_entry_point_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = CsvKgSource.from_path(KG)

    class PluginFactory:
        def __call__(self, path: Path, *, options):
            assert path == RDF
            assert options == {"token": "fixture"}
            return expected

    class EntryPoint:
        name = "fixture-source"

        @staticmethod
        def load():
            return PluginFactory()

    class EntryPoints(list):
        def select(self, *, group: str):
            assert group == "exact.sources"
            return self

    monkeypatch.setattr(
        sources_module.metadata,
        "entry_points",
        lambda: EntryPoints([EntryPoint()]),
    )
    registry = SourceRegistry()
    assert "fixture-source" in registry.names()
    assert (
        registry.resolve(RDF, source_format="fixture-source", options={"token": "fixture"})
        is expected
    )


def test_importing_io_does_not_eagerly_import_rdflib() -> None:
    code = "import sys; import exact.io; assert 'rdflib' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_importing_ontology_parser_does_not_eagerly_import_rdflib() -> None:
    code = "import sys; import exact.ontology.parser; assert 'rdflib' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_parsing_owl_never_loads_rdflib() -> None:
    code = (
        "import sys; "
        "from pathlib import Path; "
        "from exact.ontology.parser import parse; "
        f"parse(Path({str(OWL)!r})); "
        "assert 'rdflib' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
