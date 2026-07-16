"""Runtime settings for the Exact alignment inspection service."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, ClassVar

try:
    from pydantic import field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError as exc:  # pragma: no cover - exercised in an isolated wheel test
    raise RuntimeError(
        "exact-inspect requires the visualization dependencies. Install `exact-om[viz]`."
    ) from exc


class InspectSettings(BaseSettings):
    """Configuration loaded from arguments or ``EXACT_INSPECT_*`` variables."""

    model_config = SettingsConfigDict(
        env_prefix="EXACT_INSPECT_",
        case_sensitive=False,
        extra="ignore",
    )

    run_dir: Path | None = None
    analysis_dir: Path | None = None
    frontend_dir: Path | None = None
    source_ontology_path: Path | None = None
    target_ontology_path: Path | None = None
    enable_ontology_info: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    _legacy_names: ClassVar[dict[str, tuple[str, ...]]] = {
        "run_dir": ("EXACT_STUDY_RUN_DIR",),
        "analysis_dir": ("EXACT_STUDY_ANALYSIS_DIR",),
        "frontend_dir": ("EXACT_STUDY_FRONTEND_DIR",),
        "source_ontology_path": ("EXACT_STUDY_SOURCE_ONTOLOGY_PATH",),
        "target_ontology_path": ("EXACT_STUDY_TARGET_ONTOLOGY_PATH",),
        "enable_ontology_info": ("EXACT_STUDY_ENABLE_ONTOLOGY_INFO",),
        "host": ("EXACT_STUDY_HOST",),
        "port": ("EXACT_STUDY_PORT",),
        "log_level": ("EXACT_STUDY_LOG_LEVEL", "EXACT_STUDY_LOGGING_LEVEL"),
    }

    def __init__(self, **values: Any) -> None:
        legacy_present: list[str] = []
        for field, legacy_names in self._legacy_names.items():
            new_name = f"EXACT_INSPECT_{field.upper()}"
            for legacy_name in legacy_names:
                if legacy_name not in os.environ:
                    continue
                legacy_present.append(legacy_name)
                if field not in values and new_name not in os.environ:
                    values[field] = os.environ[legacy_name]
                break
        if legacy_present:
            warnings.warn(
                "EXACT_STUDY_* settings are deprecated; use EXACT_INSPECT_* instead "
                f"({', '.join(sorted(legacy_present))}).",
                DeprecationWarning,
                stacklevel=2,
            )
        if (
            "port" not in values
            and "EXACT_INSPECT_PORT" not in os.environ
            and "EXACT_STUDY_PORT" not in os.environ
            and "PORT" in os.environ
        ):
            values["port"] = os.environ["PORT"]
        super().__init__(**values)

    @field_validator(
        "run_dir",
        "analysis_dir",
        "frontend_dir",
        "source_ontology_path",
        "target_ontology_path",
        mode="after",
    )
    @classmethod
    def _resolve_path(cls, value: Path | None) -> Path | None:
        return value.expanduser().resolve() if value is not None else None

    @field_validator("log_level", mode="after")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = str(value).upper()
        if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError(f"Unsupported log level: {value}")
        return level


Settings = InspectSettings

__all__ = ["InspectSettings", "Settings"]
