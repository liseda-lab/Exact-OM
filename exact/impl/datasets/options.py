"""Small normalization helpers for dataset runtime option mappings."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def mapping_options(value: Any, label: str) -> Dict[str, Any]:
    """Normalize plain mappings and validated config-model values."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        if isinstance(dumped, Mapping):
            return dict(dumped)
    raise TypeError(f"{label} must be a mapping")


def candidate_config(
    params: Dict[str, Any],
    name: str,
    explicit: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve and persist one nested candidate-generation configuration."""

    raw = explicit if explicit is not None else params.get(name)
    normalized = mapping_options(raw, f"candidate {name}")
    if explicit is not None:
        params[name] = normalized
    return normalized


__all__ = ["candidate_config", "mapping_options"]
