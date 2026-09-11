"""The typed hierarchy factory keeps the shared query and scope contract."""

import pyowl_core as core
import pytest

from exact.ontology import load_ontology
from exact.ontology.index_views import TypedClassHierarchyView

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


def _queries(view):
    classes = [
        core.Class(core.IRI(f"urn:{name}"))
        for name in ("A", "B", "Alias", "C", "D", "E", "Union", "Imported", "ImportedAlias")
    ]
    return (
        tuple(view.iter_edges()),
        tuple(view.equivalence_sets()),
        tuple(
            (
                tuple(view.asserted_parents(item)),
                tuple(view.asserted_children(item)),
                tuple(view.equivalents(item)),
                view.component(item),
            )
            for item in classes
        ),
        view.ignored_complex_endpoint_count,
    )


@pytest.mark.parametrize("owner_index", [0, 1, 2])
@pytest.mark.parametrize(
    "scope", [core.AxiomScope.CLOSURE, core.AxiomScope.ROOT, core.AxiomScope.DOCUMENT]
)
@pytest.mark.parametrize("equivalence", ["preserve", "bidirectional", "component"])
@pytest.mark.parametrize("origins", [False, True])
def test_shared_hierarchy_semantics_and_scopes_without_generic_axiom_iteration(
    monkeypatch, owner_index, scope, equivalence, origins
):
    root, overlay, composite = _owners()
    owner = (root, overlay, composite)[owner_index]
    options = dict(
        scope=scope,
        include_origins=origins,
        equivalence_handling=equivalence,
        include_disjoint_union=True,
    )
    if scope is core.AxiomScope.DOCUMENT:
        options["document_key"] = next(
            record.document_key
            for record in root.import_manifest.documents
            if record.document_key != root.root_document_key
        )
    if owner_index == 2 and scope is not core.AxiomScope.CLOSURE:
        with pytest.raises(ValueError) as expected:
            owner.view(core.AssertedClassHierarchyView, **options)
        with pytest.raises(type(expected.value)):
            owner.view(TypedClassHierarchyView, **options)
        return
    reference = _queries(owner.view(core.AssertedClassHierarchyView, **options))

    def forbidden(*args, **kwargs):
        raise AssertionError("typed hierarchy construction must not scan generic axiom roots")

    monkeypatch.setattr(type(root), "iter_axioms", forbidden)
    actual = owner.view(TypedClassHierarchyView, **options)
    if origins:
        # Explicit provenance queries still use core's original-owner origin lookup.
        monkeypatch.undo()
    assert _queries(actual) == reference
    assert actual._ontology is owner
    assert owner.view(TypedClassHierarchyView, **options) is actual
    assert actual.report.schema_name == TypedClassHierarchyView.SCHEMA_NAME
    assert core.AssertedClassHierarchyView.SCHEMA_VERSION == 1


def test_exact_facade_uses_typed_hierarchy_and_retains_owner(monkeypatch):
    root, _, _ = _owners()
    source = load_ontology(root)

    def forbidden(*args, **kwargs):
        raise AssertionError("facade class features must not decode the entire closure")

    monkeypatch.setattr(type(root), "iter_axioms", forbidden)
    assert source.direct_parents("urn:A") == ["urn:Alias", "urn:B"]
    assert source.direct_parents("urn:C") == ["urn:D", "urn:E"]
    assert source.owl_snapshot() is root
    assert isinstance(source._class_view, TypedClassHierarchyView)
