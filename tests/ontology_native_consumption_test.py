"""Native loading and typed annotation consumption preserve the complete OWL owner."""

from io import BytesIO

import pyowl_core as core
import pytest

from exact.core.entities.graph import AnnotationValue
from exact.core.values import ANNOTATION_IRI
from exact.ontology import load_ontology
from exact.ontology.parser import parse
from exact.ontology.store import OWL_DEPRECATED, RDFS_LABEL

ROOT = f"""Ontology(<urn:root> Import(<urn:imported>)
Annotation(<urn:note> "ontology metadata")
Declaration(Class(<urn:C>))
Declaration(NamedIndividual(<urn:C>))
Declaration(DataProperty(<urn:count>))
SubClassOf(<urn:C> <urn:Parent>)
AnnotationAssertion(Annotation(<urn:nested> "nested metadata")
<{RDFS_LABEL}> <urn:C> "Class"@en)
AnnotationAssertion(<{RDFS_LABEL}> <urn:C> "Classe"@fr)
AnnotationAssertion(<urn:note> <urn:C> "description")
AnnotationAssertion(<urn:link> <urn:C> <urn:Elsewhere>)
AnnotationAssertion(<{ANNOTATION_IRI}> <urn:C> " false ")
DataPropertyAssertion(<urn:count> <urn:C> "2"^^<http://www.w3.org/2001/XMLSchema#integer>)
)""".encode()
IMPORTED = f"""Ontology(<urn:imported>
Declaration(Class(<urn:Imported>))
AnnotationAssertion(<{RDFS_LABEL}> <urn:Imported> "Imported")
AnnotationAssertion(<{OWL_DEPRECATED}> <urn:Imported> "true")
)""".encode()


def _snapshot():
    return core.load_snapshot(
        ROOT,
        document_iri="urn:root",
        options=core.LoadOptions(backend=core.BackendPreference.NATIVE),
        resolver=core.MappingResolver({"urn:imported": IMPORTED}),
    )


@pytest.mark.parametrize("loader", [load_ontology, parse])
@pytest.mark.parametrize(
    "options", [None, core.LoadOptions(offline=True, collect_provenance=False)]
)
def test_document_load_requires_native_and_retains_other_options(monkeypatch, loader, options):
    original = core.load_snapshot
    calls = []

    def checked(*args, **kwargs):
        calls.append(kwargs["options"])
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "load_snapshot", checked)
    result = loader(BytesIO(b"Ontology(<urn:tiny> Declaration(Class(<urn:C>)))"), options=options)
    owner = result.owl_snapshot() if hasattr(result, "owl_snapshot") else result
    assert owner.load_options.backend is core.BackendPreference.NATIVE
    assert calls[0].collect_provenance is (options.collect_provenance if options else True)
    assert calls[0].offline is True
    assert calls[0].allow_partial_rdf_mapping is False


@pytest.mark.parametrize("loader", [load_ontology, parse])
def test_document_load_rejects_python_before_parsing(monkeypatch, loader):
    def forbidden(*args, **kwargs):
        raise AssertionError("a rejected parser choice must not parse anything")

    monkeypatch.setattr(core, "load_snapshot", forbidden)
    with pytest.raises(ValueError, match="native pyowl-core parser"):
        loader(ROOT, options=core.LoadOptions(backend=core.BackendPreference.PYTHON))


@pytest.mark.parametrize("overlay", [False, True])
def test_annotations_use_only_native_type_partition_without_origins_or_full_scan(
    monkeypatch, overlay
):
    snapshot = _snapshot()
    added = core.AnnotationAssertion(
        core.AnnotationProperty(core.IRI(RDFS_LABEL)),
        core.IRI("urn:C"),
        core.Literal("Overlay", core.Datatype(core.IRI(core.XSD_STRING_IRI))),
    )
    owner = (
        core.apply_delta(snapshot, core.OntologyDelta(add_axioms=(added,))) if overlay else snapshot
    )
    source = load_ontology(owner)

    def forbidden(*args, **kwargs):
        raise AssertionError("annotation lookup traversed unrelated axioms or an unused index")

    monkeypatch.setattr(type(snapshot), "iter_axioms", forbidden)
    monkeypatch.setattr(core.AnnotationAssertionIndex, "_build", classmethod(forbidden))
    assert source.labels("urn:C") == (
        ["Class", "Classe", "Overlay"] if overlay else ["Class", "Classe"]
    )
    assert source.labels("urn:Imported") == ["Imported"]
    assert source.annotations("urn:C", ["urn:link"]) == [
        AnnotationValue("urn:link", "urn:Elsewhere", False)
    ]
    assert source.annotations("urn:root") == []
    assert source.annotations("urn:C", ["urn:nested"]) == []
    assert source.labels("invalid IRI") == []
    assert AnnotationValue("urn:note", "description", True) in source.attributes("urn:C")
    assert AnnotationValue(
        "urn:count", "2", True, datatype="http://www.w3.org/2001/XMLSchema#integer"
    ) in source.attributes("urn:C")
    assert source.excluded_from_alignment() == frozenset({"urn:C", "urn:Imported"})
    assert source.owl_snapshot() is owner
    assert source._axioms.options.include_origins is False
    assert "_class_view" not in source.__dict__


def test_hierarchy_and_domain_views_do_not_collect_unused_origins():
    source = load_ontology(_snapshot())
    assert source.direct_parents("urn:C") == ["urn:Parent"]
    assert source._class_view.options.include_origins is False
    assert source._property_view.options.include_origins is False
    assert source._domain_range.options.include_origins is False
