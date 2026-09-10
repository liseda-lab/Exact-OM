"""Entity enumeration uses the complete public signature without count indexes."""

import pyowl_core as core
import pytest

from exact.core.entities.kinds import EntityKind
from exact.ontology import load_ontology

ROOT = b"""Ontology(<urn:root> Import(<urn:imported>)
Annotation(<urn:ontologyOnly> "root")
Declaration(Class(<urn:ZZ>))
Declaration(Class(<urn:AAAAAA>))
Declaration(Class(<urn:Removable>))
Declaration(NamedIndividual(<urn:ZZ>))
SubClassOf(<urn:ZZ> <http://www.w3.org/2002/07/owl#Thing>)
SubClassOf(<urn:ZZ> ObjectSomeValuesFrom(<urn:usedProperty> <urn:ReferencedOnly>))
SubObjectPropertyOf(<urn:usedProperty> <urn:parentProperty>)
DataPropertyDomain(<urn:dataOnly> <urn:ZZ>))"""
IMPORTED = b"""Ontology(<urn:imported>
Annotation(<urn:annotationOnly> "import")
Declaration(Class(<urn:Imported>))
AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label>
<urn:Imported> "Imported label"))"""


@pytest.mark.parametrize("backend", [core.BackendPreference.NATIVE, core.BackendPreference.PYTHON])
@pytest.mark.parametrize("overlay", [False, True])
def test_complete_typed_signature_preserves_lexical_order_without_count_view(
    monkeypatch, backend, overlay
):
    try:
        snapshot = core.load_snapshot(
            ROOT,
            document_iri="urn:root",
            options=core.LoadOptions(backend=backend),
            resolver=core.MappingResolver({"urn:imported": IMPORTED}),
        )
    except core.BackendUnavailableError:
        if backend is core.BackendPreference.NATIVE:
            pytest.skip("native pyowl-core backend unavailable")
        raise
    removed = core.Declaration(core.Class(core.IRI("urn:Removable")))
    owner = (
        core.apply_delta(
            snapshot,
            core.OntologyDelta(
                add_axioms=(core.Declaration(core.Class(core.IRI("urn:OverlayAdded"))),),
                remove_axioms=(removed,),
            ),
        )
        if overlay
        else snapshot
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("entity enumeration requested the unused SignatureView count index")

    monkeypatch.setattr(core.SignatureView, "_build", classmethod(forbidden))
    original_signature = type(owner).signature
    signature_calls = []

    def signature(self, *args, **kwargs):
        signature_calls.append(self)
        return original_signature(self, *args, **kwargs)

    monkeypatch.setattr(type(owner), "signature", signature)
    source = load_ontology(owner)
    assert source.owl_snapshot() is owner
    expected_classes = (
        "http://www.w3.org/2002/07/owl#Thing",
        "urn:AAAAAA",
        "urn:Imported",
        "urn:ReferencedOnly",
        "urn:ZZ",
    )
    changed = "urn:OverlayAdded" if overlay else "urn:Removable"
    assert source.entities() == tuple(sorted((*expected_classes, changed)))
    assert source.entities(EntityKind.INDIVIDUAL) == ("urn:ZZ",)
    assert source.entities(EntityKind.OBJECT_PROPERTY) == (
        "urn:parentProperty",
        "urn:usedProperty",
    )
    assert source.entities(EntityKind.DATA_PROPERTY) == ("urn:dataOnly",)
    assert source.entities(EntityKind.ANNOTATION_PROPERTY) == (
        "http://www.w3.org/2000/01/rdf-schema#label",
        "urn:annotationOnly",
        "urn:ontologyOnly",
    )
    assert source.labels("urn:Imported") == ["Imported label"]
    assert source.direct_parents("urn:usedProperty", EntityKind.OBJECT_PROPERTY) == [
        "urn:parentProperty"
    ]
    assert source.entities() == tuple(sorted((*expected_classes, changed)))
    assert signature_calls == [owner]
    assert snapshot.contains(removed)
    assert owner.contains(removed) is not overlay
