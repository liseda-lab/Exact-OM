"""Recover saved literal projections without guessing qualifiers or typed subjects."""

import pyowl_core as core
import pytest

from exact_inspect.context import build_context_package


def test_saved_literal_projection_requires_an_unambiguous_original(tmp_path):
    snapshot = core.load_snapshot(
        b"""Ontology(
      Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>))
      Declaration(ObjectProperty(<urn:A>))
      AnnotationAssertion(<urn:definition> <urn:A> "Unique meaning"@en)
      AnnotationAssertion(<urn:definition> <urn:A> "Same"@en)
      AnnotationAssertion(<urn:definition> <urn:A> "Same"@pt)
      AnnotationAssertion(<urn:definition> <urn:A> "1"^^<http://www.w3.org/2001/XMLSchema#integer>)
      AnnotationAssertion(<urn:definition> <urn:A> "1"^^<http://www.w3.org/2001/XMLSchema#string>)
      AnnotationAssertion(<urn:definition> <urn:B> "Neighbour only")
    )""",
        options=core.LoadOptions(backend=core.BackendPreference.NATIVE),
    )
    context = build_context_package(snapshot, tmp_path / "context")
    entity = {"ontology_version_id": context.ontology_version_id, "iri": "urn:A", "kind": "class"}

    def resolve(value, **qualifiers):
        return context.resolve_feature(
            {"entity_iri": "urn:A", "property_iri": "urn:definition", "value": value, **qualifiers},
            entity,
        )["fact_ids"]

    assert len(resolve("Unique meaning")) == 1
    assert resolve("Unique meaning", language="pt") == []
    assert resolve("Unique meaning", datatype=None) == []
    assert resolve("Same") == []
    assert len(resolve("Same", language="en")) == 1
    assert resolve("1") == []
    assert len(resolve("1", datatype="http://www.w3.org/2001/XMLSchema#integer")) == 1
    assert resolve("Neighbour only") == []
    assert resolve("Unique meaning", entity_iri="urn:B") == []
    with pytest.raises(KeyError, match="Unknown ontology entity"):
        context.resolve_feature(
            {"entity_iri": "urn:A", "property_iri": "urn:definition", "value": "Unique meaning"},
            {**entity, "kind": "individual"},
        )
