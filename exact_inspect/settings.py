"""Configuration for the optional local Exact Explain application."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class InspectSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXACT_INSPECT_", extra="ignore")

    package: Path | None = None
    frontend_dir: Path | None = None
    library_dir: Path = Field(default_factory=lambda: Path.home() / ".exact-explain" / "library")
    profile: Literal["local_app", "public_demo"] = "local_app"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["critical", "error", "warning", "info", "debug"] = "info"

    @field_validator("package", "frontend_dir", "library_dir", mode="after")
    @classmethod
    def resolve_path(cls, value: Path | None) -> Path | None:
        return value.expanduser().resolve() if value is not None else None


Settings = InspectSettings
