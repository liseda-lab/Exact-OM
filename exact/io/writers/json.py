"""Programmatic JSON mapping writer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from exact.io.writers._frames import validated_mapping_frame
from exact.io.writers.base import WriterOptionsError


class JsonWriter:
    """Write normalized mapping records as UTF-8 JSON."""

    name = "json"
    default_filename = "alignment.json"

    def write(
        self,
        mappings: Any,
        path: Path,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Path:
        normalized = dict(options or {})
        unknown = sorted(set(normalized) - {"indent"})
        if unknown:
            raise WriterOptionsError(f"Unknown JSON writer option(s): {', '.join(unknown)}")
        indent = normalized.get("indent", 2)
        if indent is not None and (not isinstance(indent, int) or indent < 0):
            raise WriterOptionsError("JSON indent must be a non-negative integer or null")
        frame = validated_mapping_frame(mappings)
        records = [
            {
                "Src": row.SrcEntity,
                "Tgt": row.TgtEntity,
                "Score": float(row.Score),
                "Relation": str(row.Relation),
                "Kind": str(row.Kind),
            }
            for row in frame.itertuples(index=False)
        ]
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(records, stream, ensure_ascii=False, indent=indent)
            stream.write("\n")
        return Path(path)


WRITER = JsonWriter()

__all__ = ["JsonWriter", "WRITER"]
