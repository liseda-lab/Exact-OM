"""Generic input-language admission, including types absent from fitting graphs."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pyowl_core as owl
import pytest
import torch

from exact.repair.graph import (
    FEATURE_SCHEMA_V3,
    GraphExplanation,
    GraphNode,
    build_observable_graph,
)
from exact.repair.graph_schema import declared_metadata, generic_graph_schema, training_metadata
from exact.repair.model import ModelGraphSchemaError, RepairModel
from exact.repair.protocol import RepairProtocolV3, load_protocol_v3, training_projection_v3
from exact.repair.records import canonical_hash
from tests.repair_model_schema_test import graphs


def test_generic_schema_contains_builder_contract_without_case_discovery():
    declaration = generic_graph_schema()
    nodes, edges = declared_metadata(declaration)
    for owner in ("mapping", "statement", "class", "constructor", "axiom", "evidence"):
        assert ("evidence", "supports", owner) in edges
        assert (owner, "reverse_supports", "evidence") in edges
    assert ("explanation", "support", "mapping") in edges
    assert ("explanation", "witness", "constructor") in edges
    assert ("axiom", "sub_class", "class") in edges
    assert ("constructor", "filler", "class") in edges
    assert all((kind, "self", kind) in edges for kind in nodes)
    assert training_metadata(graphs()[0].metadata, declaration) == (nodes, edges)


@pytest.mark.parametrize("encoder", ["hgt", "rgcn", "none"])
def test_declared_models_handle_unobserved_relations_backward_and_portable_checkpoint(
    tmp_path, encoder
):
    torch.set_num_threads(1)
    schema = generic_graph_schema()
    original, fresh = graphs()
    model = RepairModel(
        training_metadata(original.metadata, schema),
        graph_schema=schema,
        encoder=encoder,
        hidden_dim=8,
        heads=2,
        layers=1,
        dropout=0,
        revision="v3",
    )
    memory = model.encode(fresh)
    loss = sum(row.square().sum() for row in memory.rows.values())
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.input_projection["statement"].weight.grad is not None
    state = dict(
        metadata=model.metadata,
        config=model.config,
        weights=model.state_dict(),
        graph_schema_hash=model.graph_schema_hash,
    )
    path = tmp_path / "model.pt"
    torch.save(state, path)
    restored = torch.load(path, weights_only=True)
    loaded = RepairModel(restored["metadata"], **restored["config"])
    loaded.load_state_dict(restored["weights"], strict=True)
    assert loaded.graph_schema_hash == canonical_hash(schema) == restored["graph_schema_hash"]
    assert all(torch.equal(row, loaded.encode(fresh).rows[key]) for key, row in memory.rows.items())
    unsupported = replace(fresh, edges=fresh.edges + (("e", "undeclared", "s"),))
    with pytest.raises(ModelGraphSchemaError):
        loaded.encode(unsupported)
    unsupported = replace(fresh, nodes=fresh.nodes + (GraphNode("x", "unsupported", ()),))
    with pytest.raises(ModelGraphSchemaError):
        loaded.encode(unsupported)


@pytest.mark.parametrize(
    "mutation", ["missing_relation", "extra_relation", "language_hash", "duplicate", "extra_field"]
)
def test_strict_protocol_rejects_altered_schema_and_does_not_extend_warm_starts(mutation):
    schema = generic_graph_schema()
    raw = load_protocol_v3(
        Path("specs/exact-repair/protocol/xr21-review2-conformance.json")
    ).model_dump(by_alias=True)
    raw["model"]["graph_schema"] = schema
    if mutation == "missing_relation":
        schema["edge_types"].pop()
    elif mutation == "extra_relation":
        schema["edge_types"].append(["class", "invented", "class"])
    elif mutation == "language_hash":
        schema["language_hash"] = "0" * 64
    elif mutation == "duplicate":
        schema["node_types"].append(schema["node_types"][0])
    else:
        schema["allow_unknown"] = True
    with pytest.raises(ValueError):
        RepairProtocolV3.model_validate(raw)
    with pytest.raises(ValueError, match="Warm-start"):
        training_metadata(graphs()[0].metadata, generic_graph_schema(), graphs()[0].metadata)


def test_protocol_projection_binds_schema_without_changing_historical_serialization():
    from tools.repair.train import _protocol_arguments

    original = load_protocol_v3(Path("specs/exact-repair/protocol/xr21-review2-conformance.json"))
    raw = original.model_dump(by_alias=True)
    assert "graph_schema" not in raw["model"]
    assert original.resolved_hash == canonical_hash(raw)
    declared = copy.deepcopy(raw)
    declared["model"]["graph_schema"] = generic_graph_schema()
    protocol = RepairProtocolV3.model_validate(declared)
    assert protocol.resolved_hash != original.resolved_hash
    assert (
        _protocol_arguments(training_projection_v3(protocol))["graph_schema"]
        == generic_graph_schema()
    )
    assert RepairProtocolV3.model_validate(raw).model_dump(by_alias=True) == raw


def test_generic_builder_admits_syntax_evidence_supports_and_witnesses():
    from tools.repair.corpus import generate_corpus

    case = generate_corpus(
        split_counts={"train": 1, "development": 0, "test": 0},
        siblings_per_parent=1,
        families=("range",),
        revision="v3",
    )[0]
    obj = case.problem.objects[0]
    concept = owl.Class(owl.IRI("urn:schema:concept"))
    prop = owl.ObjectProperty(owl.IRI("urn:schema:property"))
    witness = owl.ObjectSomeValuesFrom(prop, concept)
    graph = build_observable_graph(
        case.problem.objects,
        fixed_axioms=case.problem.fixed_axioms,
        evidence={obj.object_id: {"score": 0.5}, "global": {"text": "observed"}},
        explanations=(GraphExplanation("schema", (obj.object_id,), witness=witness),),
        retrieved_symbols=(concept, prop),
        feature_schema=FEATURE_SCHEMA_V3,
    )
    assert training_metadata(graph.metadata, generic_graph_schema()) == declared_metadata(
        generic_graph_schema()
    )


def test_training_checkpoint_binds_schema_and_rejects_changed_resume(tmp_path, monkeypatch):
    from tests.repair_training_completion_test import cache_for
    from tools.repair.corpus import generate_corpus
    from tools.repair import train

    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
        revision="v3",
    )
    fitting = [(c, cache_for(c)) for c in cases if c.split == "train"]
    development = [(c, cache_for(c)) for c in cases if c.split == "development"]
    schema = generic_graph_schema()
    path = tmp_path / "state.pt"
    options = dict(
        encoder="none",
        hidden_dim=8,
        heads=2,
        layers=0,
        revision="v3",
        proposal_arm="bounded_enumeration",
        graph_schema=schema,
        checkpoint_path=path,
        deadline_seconds=60,
    )
    save = train.save_training_state

    def stop_at_checkpoint(destination, state):
        save(destination, state)
        raise InterruptedError("schema checkpoint saved before fitting")

    monkeypatch.setattr(train, "save_training_state", stop_at_checkpoint)
    with pytest.raises(InterruptedError):
        train.train_cases(fitting, development, **options)
    saved = torch.load(path, weights_only=True)
    assert saved["graph_schema"] == schema
    assert saved["graph_schema_hash"] == canonical_hash(schema)
    assert saved["metadata"] == declared_metadata(schema)
    # A compatible resume reaches another checkpoint without schema migration.
    with pytest.raises(InterruptedError):
        train.train_cases(fitting, development, **options)
    saved["graph_schema_hash"] = "0" * 64
    save(path, saved)
    with pytest.raises(ValueError, match="Resume checkpoint declared graph schema changed"):
        train.train_cases(fitting, development, **options)


def test_validation_rejects_heldout_release_before_opening_payloads(tmp_path):
    from tools.repair.validate_graph_schema import run

    plan = dict(
        schema="exact-repair/graph-schema-validation-plan/v1",
        releases={"train": {}, "development": {}, "test": {}},
        expected_cases={"train": 128, "development": 32},
        heldout_use=False,
        model_fitting=False,
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="boundary"):
        run(path, tmp_path / "output")
