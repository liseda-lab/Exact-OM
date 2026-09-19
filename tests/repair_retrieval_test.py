"""Observed vocabulary retrieval and reproducible graph/menu handoffs."""

import dataclasses
import pickle

import pyowl_core as owl
import pytest

from exact.repair.api import prepare_repair
from exact.repair.graph import GraphExplanation, build_observable_graph, structural_id
from exact.repair.owl import snapshot_from_axioms
from exact.repair.records import canonical_hash
from exact.repair.retrieval import RetrievalConfig, retrieve_vocabulary


def cls(name):
    return owl.Class(owl.IRI(f"urn:retrieve:{name}"))


def prop(name):
    return owl.ObjectProperty(owl.IRI(f"urn:retrieve:{name}"))


def label(entity, text, predicate="label"):
    return owl.AnnotationAssertion(
        owl.AnnotationProperty(owl.IRI(f"http://www.w3.org/2000/01/rdf-schema#{predicate}")),
        entity.iri,
        owl.Literal(text, owl.XSD_STRING),
    )


def problem(*, evidence=None, retrieve=False, config=None):
    source = snapshot_from_axioms(
        (
            owl.SubClassOf(cls("S"), cls("Parent")),
            owl.SubClassOf(cls("S"), owl.ObjectSomeValuesFrom(prop("hasPart"), cls("Part"))),
            label(cls("S"), "accepted manuscript"),
        )
    )
    target = snapshot_from_axioms(
        (
            owl.Declaration(cls("T")),
            owl.Declaration(cls("Other")),
            owl.Declaration(cls("Alternative")),
            owl.Declaration(prop("hasAuthor")),
            label(cls("Alternative"), "accepted manuscript"),
            label(prop("hasAuthor"), "manuscript authors", "comment"),
        )
    )
    return prepare_repair(
        source,
        target,
        [{"Src": "urn:retrieve:S", "Tgt": "urn:retrieve:T", "object_id": "m"}],
        evidence=evidence,
        retrieve=retrieve,
        retrieval_config=config,
    )


def test_standalone_labels_structure_and_typed_side_menus():
    result = retrieve_vocabulary(problem())
    menu = result.for_object("m")
    assert {cls("S"), cls("Parent"), cls("Part")} <= set(menu.source_classes)
    assert {cls("T"), cls("Alternative")} <= set(menu.target_classes)
    assert prop("hasPart") in menu.source_properties
    assert prop("hasAuthor") in menu.target_properties
    assert cls("Alternative") not in menu.source_classes
    assert all(isinstance(entity, owl.Class) for entity in menu.classes)
    assert all(isinstance(entity, owl.ObjectProperty) for entity in menu.properties)
    assert ("target", cls("Alternative")) in menu.endpoint_alternatives


def test_matching_alternatives_use_all_channels_but_keep_side_and_kind():
    evidence = {
        "inference_artifacts": [
            {
                "content": {
                    "alternatives": [
                        {
                            "src_iri": "urn:retrieve:S",
                            "tgt_iri": "urn:retrieve:Other",
                            "score": 0.8,
                        },
                        {"Src": "urn:retrieve:S", "Tgt": "urn:retrieve:Parent", "Score": 1.0},
                        {
                            "Src": "urn:retrieve:S",
                            "Tgt": "urn:retrieve:hasAuthor",
                            "Kind": "object_property",
                        },
                        {"Src": "urn:retrieve:S", "Tgt": "urn:retrieve:Unobserved"},
                    ]
                }
            }
        ],
        "teacher": {"Src": "urn:retrieve:S", "Tgt": "urn:retrieve:Forbidden"},
    }
    result = retrieve_vocabulary(problem(evidence=evidence))
    menu = result.for_object("m")
    assert cls("Other") in menu.target_classes
    assert menu.target_classes.index(cls("Other")) < menu.target_classes.index(cls("Alternative"))
    assert cls("Parent") not in menu.target_classes
    assert cls("Forbidden") not in menu.classes
    assert any(
        row.get("reason") == "outside_typed_side_signature"
        for row in result.provenance["omissions"]
    )


def test_retrieved_descriptions_enter_graph_before_encoding():
    input_problem = problem()
    result = retrieve_vocabulary(input_problem)
    graph = build_observable_graph(
        input_problem.objects,
        evidence=result.graph_evidence(input_problem.evidence),
        retrieved_symbols=result.symbols,
        explanations=result.explanations,
        source_axioms=input_problem.source_axioms,
        target_axioms=input_problem.target_axioms,
    )
    evidence_id = f"evidence:{structural_id(cls('Alternative'))}"
    description = next(node for node in graph.nodes if node.node_id == evidence_id)
    assert any("token:accepted" in name for name, _ in description.features)
    assert (evidence_id, "supports", structural_id(cls("Alternative"))) in graph.edges
    assert not any(node.node_id == "evidence:global:retrieval" for node in graph.nodes)


def test_preparation_retains_controls_and_captures_exact_immutable_menus():
    config = RetrievalConfig(classes_per_side=2, endpoints_per_side=1)
    prepared = problem(retrieve=True, config=config)
    tags = {tag for candidate in prepared.objects[0].candidates for tag in candidate.action_tags}
    assert {"keep", "delete", "retain_subsumption", "replace_endpoint"} <= tags
    result = retrieve_vocabulary(prepared)
    assert result.provenance["config"]["classes_per_side"] == 2
    assert dict(prepared.evidence)["retrieval"] == result.capture()
    assert retrieve_vocabulary(pickle.loads(pickle.dumps(prepared))) == result
    _ = result.menu_index
    assert pickle.loads(pickle.dumps(result)) == result
    with pytest.raises(TypeError):
        result.menu_index["m"] = 7


def test_captured_input_identity_invalidates_on_observed_evidence_change():
    prepared = problem(retrieve=True)
    old = retrieve_vocabulary(prepared)
    evidence = dict(prepared.evidence)
    evidence["alternatives"] = [{"Src": "urn:retrieve:S", "Tgt": "urn:retrieve:Other"}]
    changed = dataclasses.replace(prepared, evidence=tuple(evidence.items()))
    new = retrieve_vocabulary(changed)
    assert old.provenance["input_identity"] != new.provenance["input_identity"]
    assert cls("Other") in new.for_object("m").target_classes


def test_diagnosis_complete_support_and_future_exclusion():
    base = problem()
    support = owl.Declaration(cls("Other"))
    diagnosis = GraphExplanation("d", ("m",), (support,), cls("S"))
    result = retrieve_vocabulary(base, explanations=(diagnosis,))
    assert result.explanations == (diagnosis,)
    assert cls("Other") in result.for_object("m").target_classes
    with pytest.raises(ValueError, match="post-decision"):
        retrieve_vocabulary(
            base, explanations=(dataclasses.replace(diagnosis, available_before_decision=False),)
        )
    evidence = {
        "explanations": [
            {
                "explanation_id": "full",
                "support_object_ids": ["m"],
                "support_axiom_hashes": [canonical_hash(support)],
            },
            {
                "explanation_id": "partial",
                "support_object_ids": ["missing"],
                "support_axiom_hashes": [canonical_hash(support)],
            },
            {
                "explanation_id": "future",
                "support_object_ids": ["m"],
                "available_before_decision": False,
            },
        ]
    }
    result = retrieve_vocabulary(problem(evidence=evidence))
    assert [item.explanation_id for item in result.explanations] == ["full"]
    reasons = {row.get("reason") for row in result.provenance["omissions"]}
    assert "incomplete_support" in reasons
    prepared = problem(evidence=evidence)
    assert "explanations[2]" in dict(prepared.evidence)["evidence_omissions"]
    assert "future" not in repr(result.graph_evidence(prepared.evidence))
    with pytest.raises(ValueError, match="observed asserted"):
        retrieve_vocabulary(
            base,
            explanations=(
                dataclasses.replace(diagnosis, support_axioms=(owl.Declaration(cls("Hidden")),)),
            ),
        )


def test_budgets_are_deterministic_and_omissions_are_explicit():
    base = problem()
    config = RetrievalConfig(
        classes_per_side=1, properties_per_side=0, neighborhood_axioms=1, lexical_postings=1
    )
    result = retrieve_vocabulary(base, config=config)
    shuffled = dataclasses.replace(
        base,
        source_axioms=tuple(reversed(base.source_axioms)),
        target_axioms=tuple(reversed(base.target_axioms)),
    )
    replay = retrieve_vocabulary(shuffled, config=config)
    assert result.menus == replay.menus
    assert result.for_object("m").source_classes == (cls("S"),)
    assert result.for_object("m").target_classes == (cls("T"),)
    assert result.for_object("m").properties == ()
    channels = {row["channel"] for row in result.provenance["omissions"]}
    assert {"neighborhood", "lexical", "source_classes"} <= channels
    for options in (
        {"classes_per_side": 0},
        {"neighborhood_hops": -1},
        {"evidence_nodes": 0},
        {"properties_per_side": True},
    ):
        with pytest.raises(ValueError):
            RetrievalConfig(**options)


def test_locked_and_nonclass_objects_have_no_endpoint_replacements():
    empty = snapshot_from_axioms(())
    prepared = prepare_repair(
        empty,
        empty,
        [
            {
                "Src": "urn:retrieve:S",
                "Tgt": "urn:retrieve:T",
                "object_id": "locked",
                "locked": True,
            },
            {
                "Src": "urn:retrieve:hasPart",
                "Tgt": "urn:retrieve:hasAuthor",
                "Kind": "object_property",
                "object_id": "p",
            },
        ],
    )
    result = retrieve_vocabulary(prepared)
    assert result.for_object("locked").endpoint_alternatives == ()
    assert result.for_object("p").endpoint_alternatives == ()
    assert len(prepared.objects[0].candidates) == 1
    assert {
        tag for candidate in prepared.objects[1].candidates for tag in candidate.action_tags
    } == {"keep", "delete"}


def test_unpartitioned_standalone_records_keep_observed_vocabulary_and_unknown_side():
    base = problem()
    unpartitioned = dataclasses.replace(base, source_axioms=(), target_axioms=())
    result = retrieve_vocabulary(unpartitioned)
    assert cls("Parent") in result.for_object("m").classes
    assert prop("hasPart") in result.for_object("m").properties
    descriptions = dict(result.descriptions)
    assert descriptions[structural_id(cls("Parent"))]["sides"] == ("unknown",)
    assert result.provenance["unpartitioned_symbols"]


def test_evidence_scan_budget_and_cache_restore_avoid_unbounded_retrieval(monkeypatch):
    base = problem(evidence={"many": [{"Src": "urn:retrieve:S", "Tgt": "urn:retrieve:Other"}] * 50})
    result = retrieve_vocabulary(base, config=RetrievalConfig(evidence_nodes=3))
    assert result.provenance["examined_evidence_nodes"] <= 3
    assert any(row["channel"] == "evidence_scan" for row in result.provenance["omissions"])
    prepared = problem(retrieve=True)
    import exact.repair.retrieval as module

    def fail_signature(value):
        raise AssertionError("cached menus should not walk ontology signatures")

    monkeypatch.setattr(module.owl, "signature", fail_signature)
    restored = retrieve_vocabulary(prepared)
    assert restored.capture() == dict(prepared.evidence)["retrieval"]


def test_nested_current_matcher_source_decisions_supply_typed_alternatives():
    evidence = {
        "source_decisions": {
            "records": [
                {
                    "Src": "urn:retrieve:S",
                    "candidates": [
                        {
                            "target": "urn:retrieve:Other",
                            "SrcKind": "class",
                            "TgtKind": "class",
                            "S_final": 0.8,
                        },
                        {
                            "target": "urn:retrieve:hasAuthor",
                            "SrcKind": "class",
                            "TgtKind": "object_property",
                        },
                    ],
                }
            ]
        }
    }
    retrieved = retrieve_vocabulary(problem(evidence=evidence))
    assert cls("Other") in retrieved.for_object("m").target_classes
    assert cls("hasAuthor") not in retrieved.for_object("m").classes
