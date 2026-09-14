"""The typed hierarchy factory keeps the shared query and scope contract."""

import pyowl_core as core
import pytest

from exact.core.entities.kinds import EntityKind
from exact.ontology import load_ontology
from exact.ontology.store import OwlOntologySource

ROOT = b"""Ontology(<urn:root> Import(<urn:imported>)
SubClassOf(<urn:A> <urn:B>)
SubClassOf(<urn:A> ObjectSomeValuesFrom(<urn:p> <urn:C>))
EquivalentClasses(<urn:B> <urn:Alias>)
EquivalentClasses(<urn:C> ObjectIntersectionOf(<urn:D> <urn:E>))
DisjointUnion(<urn:Union> <urn:D> <urn:E>)
Declaration(NamedIndividual(<urn:A>))
AnnotationAssertion(<urn:note> <urn:A> "not a hierarchy row"))"""
IMPORTED = b"""Ontology(<urn:imported>
SubClassOf(<urn:B> <urn:Imported>)
EquivalentClasses(<urn:Imported> <urn:ImportedAlias>))"""


def _owners():
    root = core.load_snapshot(
        ROOT,
        document_iri="urn:root",
        options=core.LoadOptions(backend=core.BackendPreference.NATIVE),
        resolver=core.MappingResolver({"urn:imported": IMPORTED}),
    )
    overlay = core.apply_delta(
        root,
        core.OntologyDelta(
            remove_axioms=(
                core.SubClassOf(core.Class(core.IRI("urn:A")), core.Class(core.IRI("urn:B"))),
            ),
            add_axioms=(
                core.SubClassOf(core.Class(core.IRI("urn:A")), core.Class(core.IRI("urn:C"))),
            ),
        ),
    )
    return root, overlay, core.compose_views(root, overlay)


@pytest.mark.parametrize(
    "scope", [core.AxiomScope.CLOSURE, core.AxiomScope.ROOT, core.AxiomScope.DOCUMENT]
)
def test_native_feature_scope_keeps_original_owner(monkeypatch, scope):
    root, overlay, composite = _owners()
    options = dict(
        scope=scope, equivalent_operands=True, include_builtins=False, require_native_pipeline=True
    )
    if scope is core.AxiomScope.DOCUMENT:
        options["document_key"] = next(
            record.document_key
            for record in root.import_manifest.documents
            if record.document_key != root.root_document_key
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("native feature query traversed Python ontology roots")

    monkeypatch.setattr(type(root), "iter_axioms", forbidden)
    actual = root.view(core.ClassFeatureView, **options)
    assert actual._ontology is root
    assert root.view(core.ClassFeatureView, **options) is actual
    if scope is core.AxiomScope.CLOSURE:
        expected = ("urn:Alias", "urn:B", "urn:Imported", "urn:ImportedAlias")
    elif scope is core.AxiomScope.ROOT:
        expected = ("urn:Alias", "urn:B")
    else:
        expected = ()
    assert (
        tuple(item.iri.value for item in actual.ancestors(core.Class(core.IRI("urn:A"))))
        == expected
    )
    # Changed class overlays and composite scopes cannot take the native receipt
    # path yet. Rejection happens before scalar graph construction.
    if scope is core.AxiomScope.CLOSURE:
        with pytest.raises(core.BackendProtocolError):
            overlay.view(core.ClassFeatureView, **options)
        with pytest.raises(core.BackendProtocolError):
            composite.view(core.ClassFeatureView, **options)


def test_exact_facade_uses_native_features_and_retains_owner(monkeypatch):
    root, _, _ = _owners()
    source = load_ontology(root)

    def forbidden(*args, **kwargs):
        raise AssertionError("facade class features decoded a complete typed partition")

    monkeypatch.setattr(type(root), "iter_axioms", forbidden)
    monkeypatch.setattr(OwlOntologySource, "_axioms", property(forbidden))
    assert source.direct_parents("urn:A") == ["urn:Alias", "urn:B"]
    assert source.direct_parents("urn:C") == ["urn:D", "urn:E"]
    assert source.hierarchy_bundle("urn:A", {"is_a": [], "related": ["urn:p"]}) == {
        "is_a": ["urn:Alias", "urn:B"],
        "related": ["urn:C"],
    }
    assert source.owl_snapshot() is root
    assert isinstance(source._class_view, core.ClassFeatureView)
    assert source._class_view.options.require_native_pipeline is True
    assert source._class_view.native_report["backend"] == "native"


def test_selected_typed_features_preserve_punning_and_subject_filters(monkeypatch):
    unrelated = "\n".join(
        f"ClassAssertion(<urn:OtherClass> <urn:other{i}>) "
        f'DataPropertyAssertion(<urn:data> <urn:other{i}> "unrelated")'
        for i in range(100)
    )
    source = load_ontology(
        (
            f"""Ontology(
        Declaration(Class(<urn:shared>))
        Declaration(NamedIndividual(<urn:shared>))
        ClassAssertion(ObjectIntersectionOf(<urn:A> ObjectSomeValuesFrom(<urn:p> <urn:B>)) <urn:shared>)
        ClassAssertion(<urn:shared> <urn:member>)
        ClassAssertion(<urn:A> _:anonymous)
        DataPropertyAssertion(<urn:data> <urn:shared> "value")
        SubAnnotationPropertyOf(<urn:note> <urn:annotationParent>)
        SubAnnotationPropertyOf(<urn:annotationChild> <urn:note>)
        {unrelated}
    )"""
        ).encode()
    )
    snapshot = source.owl_snapshot()

    def forbidden(*args, **kwargs):
        raise AssertionError("selected feature query enumerated Python ontology axioms")

    monkeypatch.setattr(type(snapshot), "iter_axioms", forbidden)
    index = source._axioms
    original_iter = type(index).iter
    requests = []

    def selected(self, axiom_type, **kwargs):
        assert kwargs.get("referencing") is not None
        requests.append((axiom_type, kwargs["referencing"]))
        yield from original_iter(self, axiom_type, **kwargs)

    monkeypatch.setattr(type(index), "iter", selected)
    assert source.direct_parents("urn:shared", EntityKind.INDIVIDUAL) == ["urn:A", "urn:B"]
    assert source.direct_children("urn:A", EntityKind.INDIVIDUAL) == ["urn:shared"]
    assert source.direct_children("urn:shared", EntityKind.INDIVIDUAL) == ["urn:member"]
    assert [value.value for value in source.attributes("urn:shared")] == ["value"]
    assert source.direct_parents("urn:note", EntityKind.ANNOTATION_PROPERTY) == [
        "urn:annotationParent"
    ]
    assert source.direct_children("urn:note", EntityKind.ANNOTATION_PROPERTY) == [
        "urn:annotationChild"
    ]
    assert len(requests) == 6
    assert index.options.require_native_pipeline is True


@pytest.mark.parametrize("method", ["direct_parents", "direct_children"])
@pytest.mark.parametrize("declarations_only", [False, True])
def test_ambiguous_object_data_property_hierarchy_fails_explicitly(method, declarations_only):
    declarations = """Declaration(ObjectProperty(<urn:P>))
        Declaration(DataProperty(<urn:P>))"""
    axioms = """EquivalentObjectProperties(<urn:P> <urn:O>)
        EquivalentDataProperties(<urn:P> <urn:D>)
        SubObjectPropertyOf(<urn:O> <urn:OP>)
        SubDataPropertyOf(<urn:D> <urn:DP>)"""
    source = load_ontology(f"Ontology({declarations if declarations_only else axioms})".encode())
    with pytest.raises(core.BackendProtocolError, match="ambiguous object/data IRI punning"):
        getattr(source, method)("urn:P", EntityKind.OBJECT_PROPERTY)
