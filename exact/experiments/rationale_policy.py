"""Experiment output policy; production model defaults remain unchanged."""

from __future__ import annotations

from typing import Any, Mapping

from exact.core.entities.configs.config import ConfigModel


def apply_rationale_policy(
    mapping: Mapping[str, Any], *, generate_rationales: bool = False
) -> dict[str, Any]:
    """Resolve defaults and disable prose generation unless explicitly requested.

    An opt-in preserves each component's setting, including disabled LLM controls.
    Call before freezing the resolved configuration and computing its identity.
    """
    resolved = ConfigModel.from_mapping(mapping, warn_v1=False).model_dump(
        mode="json", by_alias=True
    )
    if not generate_rationales:
        for component in resolved["pipeline"]:
            params = component["params"]
            if (
                component["name"] == "PairAdaptiveSemanticScorer"
                or "generate_llm_rationales" in params
            ):
                params["generate_llm_rationales"] = False
    return resolved


def require_rationale_policy(
    mapping: Mapping[str, Any], *, generate_rationales: bool = False
) -> None:
    """Reject stale/prebuilt experimental configs without changing their hashes."""
    resolved = ConfigModel.from_mapping(mapping, warn_v1=False).model_dump(
        mode="json", by_alias=True
    )
    if not generate_rationales and resolved != apply_rationale_policy(resolved):
        raise ValueError(
            "Experimental rationale generation requires explicit generate_rationales: true; "
            "rebuild the experiment plan with rationales disabled."
        )
