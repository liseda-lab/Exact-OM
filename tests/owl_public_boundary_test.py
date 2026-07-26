"""Static guardrails for Exact's public shared-OWL dependency boundary."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "exact"
FORBIDDEN_PREFIXES = (
    "pyowl_core._native",
    "pyowl_core.backends",
    "pyowl_core.document.native_storage",
    "pyowl2vec_star_projector.compiler",
    "pyowl2vec_star_projector.native",
)


def _import_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.module:
        return tuple(f"{node.module}.{alias.name}" for alias in node.names)
    return ()


def test_runtime_uses_only_public_shared_owl_modules() -> None:
    violations: list[str] = []
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            for imported in _import_names(node):
                if any(
                    imported == prefix or imported.startswith(f"{prefix}.")
                    for prefix in FORBIDDEN_PREFIXES
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {imported}")
    assert violations == []
