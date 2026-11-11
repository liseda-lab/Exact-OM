"""Referenced-label preparation obeys the same serialized bound as admission."""

from exact_inspect.contracts import EntityRef, VisibilityPolicy, canonical_json
from exact_inspect.generation import entity_packet
from exact_inspect.readable_descriptions import enrich, label_facts


def test_label_array_punctuation_counts_toward_byte_budget():
    policy = VisibilityPolicy()
    entity = EntityRef(ontology_version_id="source", iri="urn:source", kind="class")
    refs = ["urn:a", "urn:b"]
    packet = entity_packet(entity, [], context_hash="sha256:" + "0" * 64, policy=policy)
    packet = packet.model_copy(
        update={
            "facts": [
                {
                    "subject": entity.model_dump(),
                    "value": {
                        "ast": [
                            {"type": "Class", "iri": {"type": "IRI", "value": iri}} for iri in refs
                        ]
                    },
                }
            ]
        }
    )

    class Context:
        def _label(self, ref, language, policy):
            return {"status": "available", "fact_id": ref["iri"]}

        def fact(self, fid, ref, *, policy):
            f = {
                "fact_id": fid,
                "subject": ref,
                "category": "labels",
                "value": {"term_type": "literal", "lexical_form": ""},
            }
            f["value"]["lexical_form"] = "x" * (24575 - len(canonical_json(f)))
            assert len(canonical_json(f)) == 24575
            return f

    result = enrich(packet, {"source": Context()}, policy)
    assert len(label_facts(result)) == 1
    assert len(canonical_json(result.selection["referenced_labels"])) <= 48 * 1024


def test_study_reconstructs_label_packets_without_transport_field_inflation(tmp_path):
    import pyowl_core as core

    from exact_inspect.context import build_context_package
    from exact_inspect.readable_descriptions import revise
    from exact_inspect.study.builder import build_explanation_resource
    from tools.study_preparation_support import deterministic_result, packet_for

    iris = [f"urn:parent:{i}" for i in range(35)]
    text = (
        "Ontology(Declaration(Class(<urn:A>)) "
        'AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "A") '
        'AnnotationAssertion(<http://purl.obolibrary.org/obo/IAO_0000115> <urn:A> "A recorded definition.") '
        + " ".join(
            f'Declaration(Class(<{iri}>)) AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <{iri}> "'
            + "label " * 150
            + '")'
            for iri in iris
        )
        + " SubClassOf(<urn:A> ObjectIntersectionOf("
        + " ".join(f"<{iri}>" for iri in iris)
        + ")))"
    )
    context = build_context_package(core.load_snapshot(text.encode()), tmp_path / "context")
    entity = EntityRef(ontology_version_id=context.ontology_version_id, iri="urn:A", kind="class")
    contexts = {context.ontology_version_id: context}
    policy = VisibilityPolicy()
    packet, _ = packet_for(entity, contexts, policy)
    result = revise(deterministic_result(packet), contexts, policy)
    from exact_inspect.generation import FactPacket

    packet = FactPacket.model_validate(result["support_packet"])
    resource = build_explanation_resource(
        contexts, [entity], policy=policy, packets=[packet], explanations=[result]
    )
    assert resource.entity_profiles
    assert (
        len(canonical_json([f.model_dump(mode="json") for f in resource.referenced_labels]))
        > 48 * 1024
    )
