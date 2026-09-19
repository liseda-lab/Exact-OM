"""Native compiler transport preserves the DAG and enforces the parent deadline."""

from time import monotonic

import pytest
import torch

from exact.repair.circuit import ConditionedMixture, ProposalEncoding, compile_encoding
from exact.repair.compilation import (
    CircuitArtifact,
    EvaluationNode,
    _restore_evaluation_nodes,
    compilation_cache_info,
    compile_bounded,
)


def test_immutable_compiler_transport_preserves_mass_gradients_and_sample_order():
    encoding = ProposalEncoding(
        (("first", ("a", "b")), ("second", ("c", "d"))),
        ((True, False, True, False), (True, False, False, True), (False, True, True, False)),
        ("ac", "ad", "bc"),
        "immutable-transport-oracle",
    )
    native = compile_encoding(encoding)
    transported = compile_bounded(encoding, seconds=10)
    before_hits = compilation_cache_info()["hits"]
    assert compile_bounded(encoding, seconds=10) is transported
    assert compilation_cache_info()["hits"] == before_hits + 1
    assert isinstance(transported.manager, CircuitArtifact)
    assert isinstance(transported.root, EvaluationNode)
    assert transported.manager.sdd and transported.manager.vtree
    assert transported.root.global_model_count() == native.root.global_model_count()
    assert transported.root.count() == native.root.count()
    logits = torch.tensor(
        [[0.3, -0.4, 1.1, 0.2], [-0.2, 0.8, 0.5, -0.6]], dtype=torch.float64, requires_grad=True
    )
    components = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    first = ConditionedMixture(native, logits, components)
    second = ConditionedMixture(transported, logits, components)
    assert torch.allclose(first.component_log_normalizers, second.component_log_normalizers)
    assert first.sample(20, seed=71) == second.sample(20, seed=71)
    first_grad = torch.autograd.grad(
        first.log_probability(encoding.assignments[0]), (logits, components), retain_graph=True
    )
    second_grad = torch.autograd.grad(
        second.log_probability(encoding.assignments[0]), (logits, components)
    )
    assert all(torch.allclose(a, b) for a, b in zip(first_grad, second_grad))


def test_transport_reconstruction_checks_deadline_before_native_free_nodes():
    artifact = CircuitArtifact(b"recorded sdd", b"recorded vtree", 1, 0, 1, 2)
    table = ((0, "true", 0, ()),)
    with pytest.raises(TimeoutError, match="reconstruction"):
        _restore_evaluation_nodes(artifact, table, 0, deadline=monotonic() - 1)
    node = _restore_evaluation_nodes(artifact, table, 0, deadline=monotonic() + 1)
    assert node.is_true() and node.global_model_count() == 2
