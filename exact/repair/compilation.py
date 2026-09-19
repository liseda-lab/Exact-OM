"""Hard native compilation deadlines and immutable, deadline-checked circuit transport."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

from .circuit import CompiledCircuit, ProposalEncoding, compile_encoding
from .workers import bounded_call


@dataclass(frozen=True, slots=True)
class CircuitArtifact:
    """Compiler-owned public serialization retained for independent audit/replay.

    This replaces a native manager only in a transported evaluation circuit. It
    does not compile, simplify or alter the logical DAG produced by PySDD.
    """

    sdd: bytes
    vtree: bytes
    variable_count: int
    node_count: int
    root_model_count: int
    root_global_model_count: int


@dataclass(frozen=True, slots=True, eq=False)
class EvaluationNode:
    """Read-only PySDD node interface needed by weighted counting and sampling."""

    id: int
    kind: str
    literal: int = 0
    branches: tuple[tuple[EvaluationNode, EvaluationNode], ...] = ()
    root_artifact: CircuitArtifact | None = None

    def is_false(self) -> bool:
        return self.kind == "false"

    def is_true(self) -> bool:
        return self.kind == "true"

    def is_literal(self) -> bool:
        return self.kind == "literal"

    def elements(self) -> tuple[tuple[EvaluationNode, EvaluationNode], ...]:
        return self.branches

    def count(self) -> int:
        if self.root_artifact is None:
            raise ValueError("transported circuit counts are recorded only for its root")
        return self.root_artifact.node_count

    def model_count(self) -> int:
        if self.root_artifact is None:
            raise ValueError("transported circuit counts are recorded only for its root")
        return self.root_artifact.root_model_count

    def global_model_count(self) -> int:
        if self.root_artifact is None:
            raise ValueError("transported circuit counts are recorded only for its root")
        return self.root_artifact.root_global_model_count


def _serialize_compile(
    encoding: Any, order: tuple[int, ...] | None, max_nodes: int, seconds: float
) -> tuple[CircuitArtifact, tuple, int, str, str]:
    if isinstance(encoding, ProposalEncoding):
        compiled = compile_encoding(encoding, variable_order=order)
    else:
        from .grammar import compile_grammar

        compiled = compile_grammar(
            encoding, variable_order=order, max_nodes=max_nodes, max_seconds=seconds
        )
    if compiled.node_count > max_nodes:
        raise ValueError("Compiled circuit exceeds the declared node budget")
    # Copy the compiler's public DAG verbatim in child-before-parent order. This
    # is serialization of an existing compiler result, not a local compiler.
    table = []
    visited = set()
    pending = [(compiled.root, False)]
    while pending:
        node, expanded = pending.pop()
        if node.id in visited:
            continue
        row: tuple[int, str, int, tuple[tuple[int, int], ...]]
        if node.is_false():
            row = (node.id, "false", 0, ())
        elif node.is_true():
            row = (node.id, "true", 0, ())
        elif node.is_literal():
            row = (node.id, "literal", node.literal, ())
        elif expanded:
            row = (node.id, "decision", 0, tuple((p.id, s.id) for p, s in node.elements()))
        else:
            pending.append((node, True))
            pending.extend(
                (child, False)
                for pair in node.elements()
                for child in pair
                if child.id not in visited
            )
            continue
        visited.add(node.id)
        table.append(row)
    with tempfile.TemporaryDirectory(prefix="exact-repair-circuit-") as directory:
        sdd, vtree = Path(directory) / "formula.sdd", Path(directory) / "order.vtree"
        compiled.root.save(str(sdd).encode())
        compiled.manager.vtree().save(str(vtree).encode())
        artifact = CircuitArtifact(
            sdd.read_bytes(),
            vtree.read_bytes(),
            encoding.variable_count,
            compiled.node_count,
            compiled.root.model_count(),
            compiled.root.global_model_count(),
        )
    return artifact, tuple(table), compiled.root.id, compiled.cache_key, compiled.compiler_version


def _restore_evaluation_nodes(
    artifact: CircuitArtifact, table: tuple, root_id: int, *, deadline: float
) -> EvaluationNode:
    """Rebuild immutable Python records with no uninterruptible parent native work."""
    nodes: dict[int, EvaluationNode] = {}
    for identifier, kind, literal, references in table:
        if monotonic() >= deadline:
            raise TimeoutError("Circuit transport reconstruction deadline exhausted")
        if identifier in nodes or kind not in {"false", "true", "literal", "decision"}:
            raise ValueError("Invalid serialized compiler DAG")
        if kind == "literal" and not 1 <= abs(literal) <= artifact.variable_count:
            raise ValueError("Compiler DAG contains an invalid literal")
        branches = []
        for left, right in references:
            if monotonic() >= deadline:
                raise TimeoutError("Circuit transport reconstruction deadline exhausted")
            branches.append((nodes[left], nodes[right]))
        nodes[identifier] = EvaluationNode(
            identifier, kind, literal, tuple(branches), artifact if identifier == root_id else None
        )
    if monotonic() >= deadline:
        raise TimeoutError("Circuit transport reconstruction deadline exhausted")
    return nodes[root_id]


@lru_cache(maxsize=32)
def _compile_bounded_cached(
    encoding: Any,
    *,
    seconds: float = 20.0,
    max_nodes: int = 100000,
    variable_order: Sequence[int] | None = None,
) -> CompiledCircuit:
    """Compile in a killable worker; transport its immutable public evaluation DAG.

    Native compilation, model counting and public artifact serialization all run
    under supervision. The parent only reconstructs checked Python records under
    the same deadline. Neural log weights and autograd remain in the caller.
    """
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    started = monotonic()
    outcome = bounded_call(
        _serialize_compile,
        encoding,
        None if variable_order is None else tuple(variable_order),
        max_nodes,
        seconds,
        timeout=seconds,
    )
    if outcome.status == "timeout":
        raise TimeoutError("Circuit compilation deadline exhausted")
    if outcome.status != "complete":
        raise RuntimeError(f"Circuit compilation failed: {outcome.detail}")
    artifact, table, root_id, key, version = outcome.value
    root = _restore_evaluation_nodes(artifact, table, root_id, deadline=started + seconds)
    return CompiledCircuit(
        encoding, artifact, root, key, version, monotonic() - started, artifact.node_count
    )


def compile_bounded(
    encoding: Any,
    *,
    seconds: float = 20.0,
    max_nodes: int = 100000,
    variable_order: Sequence[int] | None = None,
) -> CompiledCircuit:
    """Reuse identical compiled languages; resource limits remain part of cache identity."""
    return _compile_bounded_cached(
        encoding,
        seconds=seconds,
        max_nodes=max_nodes,
        variable_order=None if variable_order is None else tuple(variable_order),
    )


def compilation_cache_info() -> dict[str, int]:
    """Expose cache reuse counters without changing compiled-object identity."""
    info = _compile_bounded_cached.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "entries": info.currsize,
        "capacity": info.maxsize or 0,
    }
