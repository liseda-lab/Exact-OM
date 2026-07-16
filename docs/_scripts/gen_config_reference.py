"""Generate configuration reference pages from the live Pydantic schema."""

from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Union, get_args, get_origin

import mkdocs_gen_files
import yaml
from pydantic import BaseModel
from pydantic.fields import PydanticUndefined

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "exact" / "default_config.yaml"
OUTPUT_DIR = Path("reference/configuration")
MISSING = object()


def _slug(value: str) -> str:
    """Convert a configuration section name into a stable page filename."""
    return value.strip().lower().replace("_", "-")


def _model_type(annotation: Any) -> type[BaseModel] | None:
    """Return the first Pydantic model nested in a type annotation, if any."""
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation
    for argument in get_args(annotation):
        nested = _model_type(argument)
        if nested is not None:
            return nested
    return None


def _type_name(annotation: Any) -> str:
    """Render a compact, deterministic representation of a Python annotation."""
    if annotation is Any:
        return "Any"
    if annotation is None or annotation is type(None):
        return "None"

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, types.UnionType):
        return " | ".join(_type_name(argument) for argument in arguments)
    if origin is not None:
        name = getattr(origin, "__name__", str(origin).replace("typing.", ""))
        if arguments:
            return f"{name}[{', '.join(_type_name(argument) for argument in arguments)}]"
        return name
    return getattr(annotation, "__name__", str(annotation).replace("typing.", ""))


def _yaml_default(defaults: Mapping[str, Any], field_name: str) -> Any:
    """Read a field's default from a YAML mapping without conflating null and missing."""
    return defaults[field_name] if field_name in defaults else MISSING


def _schema_default(field: Any) -> Any:
    """Return a serializable Pydantic default when the YAML has no entry."""
    if field.default is not PydanticUndefined:
        return field.default
    if field.default_factory is not None:
        try:
            return field.default_factory()
        except Exception:
            return MISSING
    return MISSING


def _render_value(value: Any) -> str:
    """Render a default value safely inside a Markdown table cell."""
    if value is MISSING:
        return "_not set_"
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    try:
        rendered = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        rendered = repr(value)
    return f"`{rendered.replace('|', '&#124;')}`"


def _render_description(description: str | None) -> str:
    """Normalize a field description for a Markdown table cell."""
    if not description:
        return "—"
    return " ".join(description.split()).replace("|", "&#124;")


def _page_header(title: str, model: type[BaseModel] | None = None) -> list[str]:
    """Build a page heading and optional model documentation."""
    lines = [f"# {title}", "", "_This page is generated; do not edit it by hand._", ""]
    if model is not None:
        doc = inspect.cleandoc(model.__doc__) if model.__doc__ else None
        if doc:
            lines.extend([doc, ""])
    return lines


def _render_fields(model: type[BaseModel], defaults: Mapping[str, Any]) -> list[str]:
    """Render the fields for one Pydantic model as a Markdown table."""
    lines = [
        "| Field | Type | Default | Description |",
        "| --- | --- | --- | --- |",
    ]
    for name, field in model.model_fields.items():
        default = _yaml_default(defaults, name)
        if default is MISSING:
            default = _schema_default(field)
        field_type = _type_name(field.annotation).replace("|", "&#124;")
        lines.append(
            f"| `{name}` | `{field_type}` | {_render_value(default)} | "
            f"{_render_description(field.description)} |"
        )
    lines.append("")
    return lines


def _write_page(path: Path, lines: list[str], source: Path) -> None:
    """Write a generated page and associate its edit link with its schema source."""
    with mkdocs_gen_files.open(path, "w") as page:
        page.write("\n".join(lines))
    mkdocs_gen_files.set_edit_path(path, source.relative_to(ROOT))


def generate() -> None:
    """Generate an overview and one page for every top-level config section."""
    sys.path.insert(0, str(ROOT))
    from exact.core.entities.configs.config import ConfigModel

    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as stream:
        defaults = yaml.safe_load(stream) or {}
    if not isinstance(defaults, Mapping):
        raise TypeError(f"Expected a mapping in {DEFAULT_CONFIG_PATH}")

    schema_source = Path(inspect.getsourcefile(ConfigModel) or "")
    nav = mkdocs_gen_files.Nav()

    overview = _page_header("Configuration reference")
    overview.extend(
        [
            "The pages in this section are rebuilt from `ConfigModel` and ",
            "`exact/default_config.yaml` during every MkDocs build. Defaults therefore track ",
            "the executable configuration rather than a hand-maintained table.",
            "",
            "## Sections",
            "",
        ]
    )

    root_defaults: dict[str, Any] = {}
    root_fields: list[str] = []
    section_pages: list[tuple[str, str]] = []

    for section_name, field in ConfigModel.model_fields.items():
        section_model = _model_type(field.annotation)
        section_default = _yaml_default(defaults, section_name)
        if section_model is None or not isinstance(section_default, Mapping):
            root_fields.append(section_name)
            if section_default is not MISSING:
                root_defaults[section_name] = section_default
            continue

        filename = f"{_slug(section_name)}.md"
        title = section_name.replace("_", " ").title()
        section_pages.append((title, filename))
        lines = _page_header(title, section_model)
        lines.extend(_render_fields(section_model, section_default))
        _write_page(OUTPUT_DIR / filename, lines, schema_source)

    for title, filename in section_pages:
        overview.append(f"- [{title}]({filename})")
    overview.append("- [Root options](root-options.md)")
    overview.append("")
    _write_page(OUTPUT_DIR / "index.md", overview, schema_source)

    root_lines = _page_header("Root options", ConfigModel)
    root_lines.extend(
        [
            "These fields are scalar values or registry-backed structures rather than nested ",
            "configuration sections.",
            "",
            "| Field | Type | Default | Description |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name in root_fields:
        field = ConfigModel.model_fields[name]
        default = _yaml_default(root_defaults, name)
        if default is MISSING:
            default = _schema_default(field)
        field_type = _type_name(field.annotation).replace("|", "&#124;")
        root_lines.append(
            f"| `{name}` | `{field_type}` | {_render_value(default)} | "
            f"{_render_description(field.description)} |"
        )
    root_lines.append("")
    _write_page(OUTPUT_DIR / "root-options.md", root_lines, schema_source)

    nav["Overview"] = "index.md"
    for title, filename in section_pages:
        nav[title] = filename
    nav["Root options"] = "root-options.md"
    with mkdocs_gen_files.open(OUTPUT_DIR / "SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


generate()
