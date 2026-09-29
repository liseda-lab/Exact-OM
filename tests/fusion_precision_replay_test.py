"""Neutral validation reproduces shipped mixed precision without changing fits."""

from types import SimpleNamespace

import pytest
import torch

from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer
from exact.impl.models.selector.fusion_fitting import (
    fusion_scores,
    validate_neutral_fusion,
)


def evidence(dtype):
    names = ["label", "hier__is_a", "strsim", "lex", "struct"]
    scores = torch.tensor([[0.8095703125, 0.2, 0.5, 0, 0]], dtype=torch.float64)
    quality = torch.tensor([[1, 0.6, 0, 0, 0]], dtype=torch.float64)
    active = quality > 0
    model = SimpleNamespace(
        tau=0.5, gamma=2.0, fusion_enabled=False, _fusion_multiplier=lambda _: 1.0
    )
    authority = PairAdaptiveSemanticScorer._sigma_authority
    lex = scores[:, 0].to(dtype)
    structural = scores[:, 1].float()
    lm = authority(model, lex, quality[:, 0].float(), active[:, 0], channel="lex")
    sm = authority(model, structural, quality[:, 1].float(), active[:, 1], channel="struct")
    weight = sm / (lm + sm)
    expected = ((1 - weight) * lex + weight * structural).double()
    return (scores, quality, active), names, expected


def test_fp16_producer_requires_explicit_precision_binding():
    channels, names, expected = evidence(torch.float16)
    with pytest.raises(ValueError, match="does not replay"):
        validate_neutral_fusion(channels, names, expected)
    validate_neutral_fusion(channels, names, expected, fp16_lexical=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_declared_producer_accepts_both_actual_batch_paths(dtype):
    channels, names, expected = evidence(dtype)
    validate_neutral_fusion(channels, names, expected, fp16_lexical=True)


def test_mixed_legacy_batches_and_corrupted_scores():
    half, names, expected_half = evidence(torch.float16)
    single, _, expected_single = evidence(torch.float32)
    channels = tuple(torch.cat((a, b)) for a, b in zip(half, single))
    expected = torch.cat((expected_half, expected_single))
    validate_neutral_fusion(channels, names, expected, fp16_lexical=True)
    with pytest.raises(ValueError, match="does not replay"):
        validate_neutral_fusion(channels, names, expected + 0.001, fp16_lexical=True)


def test_precision_validation_does_not_change_fitting_scores_or_gradients():
    channels, names, expected = evidence(torch.float16)
    tau = torch.tensor(0.52, dtype=torch.float64, requires_grad=True)
    params = (tau, torch.tensor(1.9), torch.ones(len(names)))
    before = fusion_scores(channels, names, params, mode="analytic_fitted")
    grad_before = torch.autograd.grad(before.sum(), tau)[0]
    validate_neutral_fusion(channels, names, expected, fp16_lexical=True)
    after = fusion_scores(channels, names, params, mode="analytic_fitted")
    grad_after = torch.autograd.grad(after.sum(), tau)[0]
    assert torch.equal(before, after)
    assert torch.equal(grad_before, grad_after)
