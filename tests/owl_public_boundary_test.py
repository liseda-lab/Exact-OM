"""Static guardrails for Exact's public shared-OWL dependency boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "exact"
FORBIDDEN_PREFIXES = (
    "pyowl_core._native",
    "pyowl_core.backends",
    "pyowl_core.document.native_storage",
    "pyowl2vec_star_projector.compiler",
    "pyowl2vec_star_projector.native",
)
FORBIDDEN_ENCODED_SYMBOLS = frozenset(
    {
        "EncodedStructuralPublicationV1",
        "decode_canonical",
        "decode_varint",
        "get_encoded_structural_view",
    }
)
ONTOLOGY_HANDOFF_FILES = (
    *sorted((SOURCE / "ontology").glob("*.py")),
    SOURCE / "io" / "sources" / "owl.py",
    SOURCE / "core" / "entities" / "ontology.py",
)


def _import_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.module:
        return tuple(f"{node.module}.{alias.name}" for alias in node.names)
    return ()


def _forbidden_imports(path: Path, source: str) -> list[str]:
    violations: list[str] = []
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        for imported in _import_names(node):
            if any(
                imported == prefix or imported.startswith(f"{prefix}.")
                for prefix in FORBIDDEN_PREFIXES
            ):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {imported}")
    return violations


def test_runtime_uses_only_public_owl_modules() -> None:
    violations: list[str] = []
    for path in sorted(SOURCE.rglob("*.py")):
        violations.extend(_forbidden_imports(path, path.read_text(encoding="utf-8")))
    assert violations == []


@pytest.mark.parametrize(
    ("relative_path", "source", "allowed"),
    [
        (
            "ontology/native_projection.py",
            "from pyowl2vec_star_projector.native import prepare_native_encoded_compilation",
            False,
        ),
        (
            "ontology/projection.py",
            "from pyowl2vec_star_projector.native import prepare_native_encoded_compilation",
            False,
        ),
        (
            "ontology/native_projection.py",
            "import pyowl2vec_star_projector.native",
            False,
        ),
        (
            "ontology/native_projection.py",
            "from pyowl2vec_star_projector.native import iter_native_passthrough",
            False,
        ),
        (
            "ontology/native_projection.py",
            "import pyowl2vec_star_projector.compiler",
            False,
        ),
        ("ontology/native_projection.py", "import pyowl_core._native", False),
    ],
)
def test_native_implementation_imports_are_rejected(relative_path, source, allowed):
    violations = _forbidden_imports(SOURCE / relative_path, source)
    assert bool(violations) is not allowed


def test_ontology_handoff_never_decodes_buffers_or_flattens_layered_views() -> None:
    violations: list[str] = []
    for path in ONTOLOGY_HANDOFF_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            symbol: str | None = None
            if isinstance(node, ast.Name):
                symbol = node.id
            elif isinstance(node, ast.Attribute):
                symbol = node.attr
            if symbol in FORBIDDEN_ENCODED_SYMBOLS:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {symbol}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "materialize"
            ):
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: layered view materialization"
                )
    assert violations == []
