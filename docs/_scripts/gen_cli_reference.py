"""Generate command-line option tables from Exact's argparse definitions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[2]
CLI_DIR = ROOT / "exact" / "delivery" / "cli"
OUTPUT_DIR = Path("reference/cli")


@dataclass(frozen=True)
class Argument:
    """Documentation extracted from one ``ArgumentParser.add_argument`` call."""

    flags: tuple[str, ...]
    required: str
    value: str
    default: str
    help: str


def _literal(node: ast.AST | None, default: Any = None) -> Any:
    """Return a literal AST value, falling back to source-like text."""
    if node is None:
        return default
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        return ast.unparse(node)


def _keywords(call: ast.Call) -> dict[str, ast.AST]:
    """Index named arguments from a call."""
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}


def _is_call(call: ast.Call, name: str) -> bool:
    """Check the terminal name of a direct or attribute call."""
    function = call.func
    return (isinstance(function, ast.Name) and function.id == name) or (
        isinstance(function, ast.Attribute) and function.attr == name
    )


def _clean(value: Any) -> str:
    """Make a value safe for a Markdown table cell."""
    if value is None or value == "":
        return "—"
    return " ".join(str(value).split()).replace("|", "&#124;")


def _extract_argument(call: ast.Call) -> Argument | None:
    """Extract display data from an ``add_argument`` call."""
    raw_flags = tuple(str(_literal(node)) for node in call.args)
    if not raw_flags:
        return None

    keywords = _keywords(call)
    help_text = _literal(keywords.get("help"))
    if help_text in {"argparse.SUPPRESS", "==SUPPRESS=="}:
        return None
    positional = not any(flag.startswith("-") for flag in raw_flags)
    action = str(_literal(keywords.get("action"), ""))
    required = bool(_literal(keywords.get("required"), positional))

    value_parts: list[str] = []
    value_type = _literal(keywords.get("type"))
    if value_type:
        value_parts.append(str(value_type))
    nargs = _literal(keywords.get("nargs"))
    if nargs:
        value_parts.append(f"nargs={nargs}")
    choices = _literal(keywords.get("choices"))
    if choices:
        value_parts.append(f"choices={choices}")
    if action and action not in {"store", "None"}:
        value_parts.append(action)

    default = _literal(keywords.get("default"))
    if "default" not in keywords:
        if action == "store_true":
            default = False
        elif action == "store_false":
            default = True

    return Argument(
        flags=raw_flags,
        required="yes" if required else "no",
        value=_clean(", ".join(value_parts)),
        default=_clean(default),
        help=_clean(help_text),
    )


def _inspect_parser(path: Path) -> tuple[str, list[Argument]]:
    """Extract a parser description and arguments from a CLI module without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    description = ""
    arguments: list[Argument] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _is_call(node, "ArgumentParser") and not description:
            description = str(_literal(_keywords(node).get("description"), ""))
        if _is_call(node, "add_argument"):
            argument = _extract_argument(node)
            if argument is not None:
                arguments.append(argument)
    return description, arguments


def _title(stem: str) -> str:
    """Convert a CLI module stem into a page title."""
    return stem.replace("_", " ").title()


def _write_command_page(path: Path, source: Path, description: str, args: list[Argument]) -> None:
    """Write the reference page for one command module."""
    title = _title(source.stem)
    lines = [
        f"# {title}",
        "",
        "_This page is generated from the command's `argparse` definitions._",
        "",
    ]
    if description:
        lines.extend([description, ""])
    lines.extend(
        [
            "| Option | Required | Value | Default | Description |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for argument in args:
        flags = ", ".join(f"`{flag}`" for flag in argument.flags)
        lines.append(
            f"| {flags} | {argument.required} | {argument.value} | "
            f"{argument.default} | {argument.help} |"
        )
    lines.append("")

    with mkdocs_gen_files.open(path, "w") as page:
        page.write("\n".join(lines))
    mkdocs_gen_files.set_edit_path(path, source.relative_to(ROOT))


def generate() -> None:
    """Generate the CLI overview and per-module option tables."""
    nav = mkdocs_gen_files.Nav()
    pages: list[tuple[str, str, str]] = []

    for source in sorted(CLI_DIR.glob("*.py")):
        if source.name == "__init__.py":
            continue
        description, arguments = _inspect_parser(source)
        if not arguments:
            continue
        filename = f"{source.stem.replace('_', '-')}.md"
        _write_command_page(OUTPUT_DIR / filename, source, description, arguments)
        pages.append((_title(source.stem), filename, description))

    overview = [
        "# Command-line reference",
        "",
        "_This page is generated; do not edit it by hand._",
        "",
        "Each page below is rebuilt from the corresponding parser in `exact/delivery/cli/`.",
        "",
    ]
    for title, filename, description in pages:
        suffix = f" — {description}" if description else ""
        overview.append(f"- [{title}]({filename}){suffix}")
    overview.append("")

    index_path = OUTPUT_DIR / "index.md"
    with mkdocs_gen_files.open(index_path, "w") as page:
        page.write("\n".join(overview))
    mkdocs_gen_files.set_edit_path(index_path, Path("docs/_scripts/gen_cli_reference.py"))

    nav["Overview"] = "index.md"
    for title, filename, _ in pages:
        nav[title] = filename
    with mkdocs_gen_files.open(OUTPUT_DIR / "SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


generate()
