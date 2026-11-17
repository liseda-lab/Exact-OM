"""Shared strict Pydantic base for versioned Exact-OM configuration."""

from __future__ import annotations

import difflib
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, model_validator


class StrictConfigModel(BaseModel):
    """Base class that rejects misspelled configuration keys with a hint."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_default=True,
        use_enum_values=False,
    )

    @model_validator(mode="before")
    @classmethod
    def reject_unknown_keys(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        accepted: set[str] = set(cls.model_fields)
        aliases: dict[str, str] = {}
        for name, info in cls.model_fields.items():
            alias = info.alias
            if isinstance(alias, str):
                accepted.add(alias)
                aliases[alias] = name
        unknown = [str(key) for key in value if str(key) not in accepted]
        if not unknown:
            return value
        messages: list[str] = []
        for key in sorted(unknown):
            suggestion = difflib.get_close_matches(key, sorted(accepted), n=1, cutoff=0.55)
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            messages.append(f"Unknown configuration key '{key}'.{hint}")
        raise ValueError(" ".join(messages))


__all__ = ["StrictConfigModel"]
