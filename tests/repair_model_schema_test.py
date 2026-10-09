from dataclasses import replace

import pytest
import torch

from exact.repair.graph import GraphNode, ObservableGraph
from exact.repair.model import ModelGraphSchemaError, RepairModel, graph_schema_compatibility


def graphs():
    nodes = (GraphNode("e", "evidence", ()), GraphNode("s", "statement", ()))
    original = ObservableGraph(
        nodes,
        (("e", "self", "e"), ("s", "self", "s")),
        (("object", "s"),),
        feature_schema="exact-repair/observable-features/v3",
    )
    fresh = replace(
        original, edges=original.edges + (("e", "supports", "s"), ("s", "reverse_supports", "e"))
    )
    return original, fresh


@pytest.mark.parametrize("encoder", ["hgt", "rgcn"])
def test_frozen_graph_encoder_rejects_new_relations_without_inventing_weights(encoder):
    original, fresh = graphs()
    model = RepairModel(
        original.metadata, encoder=encoder, hidden_dim=8, heads=2, layers=1, revision="v3"
    )
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    evidence = graph_schema_compatibility(model.metadata, encoder, fresh)
    assert (
        not evidence["compatible"]
        and ("evidence", "supports", "statement") in evidence["missing_edge_types"]
    )
    with pytest.raises(ModelGraphSchemaError):
        model.encode(fresh)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in weights.items())


def test_no_graph_encodes_new_edges_identically_and_preserves_full_readout_graph():
    original, fresh = graphs()
    model = RepairModel(
        original.metadata, encoder="none", hidden_dim=8, heads=2, layers=0, dropout=0, revision="v3"
    )
    same_weights_full_schema = RepairModel(fresh.metadata, **model.config)
    same_weights_full_schema.load_state_dict(model.state_dict(), strict=True)
    prior = model.encode(original)
    result = model.encode(fresh)
    expected = same_weights_full_schema.encode(fresh)
    assert result.graph == fresh and result.graph.adjacency["e"] == frozenset({"e", "s"})
    assert graph_schema_compatibility(model.metadata, "none", fresh)["compatible"]
    for node in ("e", "s"):
        assert torch.equal(result.rows[node], expected.rows[node])
        assert torch.equal(result.rows[node], prior.rows[node])
    assert model.metadata == original.metadata


def test_no_graph_still_rejects_untrained_node_projection():
    original, _ = graphs()
    fresh = replace(original, nodes=original.nodes + (GraphNode("x", "unseen", ()),))
    model = RepairModel(
        original.metadata, encoder="none", hidden_dim=8, heads=2, layers=0, revision="v3"
    )
    with pytest.raises(ModelGraphSchemaError):
        model.encode(fresh)
