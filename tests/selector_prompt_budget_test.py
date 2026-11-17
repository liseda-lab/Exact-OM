"""A rejected paid request must never masquerade as selector abstention."""

from types import SimpleNamespace

import pandas as pd
import pytest

from exact.impl.models.selector.acceptance import AcceptanceMixin
from exact.llm.prompt_budget import PromptBudgetError


@pytest.mark.parametrize("method", ["_llm_arbitrate_group", "_llm_direct_choice_group"])
def test_prompt_rejection_propagates_from_both_arbitration_modes(method):
    error = PromptBudgetError("oversized final prompt")

    def rejected(*args):
        raise error

    selector = SimpleNamespace(
        _llm_prompt=lambda *args: ({"user": "request"}, {}, {}),
        _run_llm_prompt=rejected,
    )
    arguments = dict(
        src="source", group=pd.DataFrame(), primary_model=object(), logger=None, record_lookup={}
    )
    if method == "_llm_arbitrate_group":
        arguments.update(selector_scores=[0.8], no_match_risk=0.2)
    with pytest.raises(PromptBudgetError) as caught:
        getattr(AcceptanceMixin, method)(selector, **arguments)
    assert caught.value is error
