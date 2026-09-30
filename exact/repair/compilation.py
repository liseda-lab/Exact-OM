"""Hard native compilation deadlines and immutable, deadline-checked circuit transport."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

from .circuit import CompiledCircuit, ProposalEncoding, compile_encoding
from .records import canonical_hash
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
    telemetry: tuple = ()


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
    encoding: Any,
    order: tuple[int, ...] | None,
    max_nodes: int,
    seconds: float,
    vtree_type: str = "balanced",
    collect: bool = True,
    max_live_nodes: int | None = None,
    max_reachable_nodes: int | None = None,
    max_elements: int | None = None,
) -> tuple[CircuitArtifact, tuple, int, str, str]:
    if isinstance(encoding, ProposalEncoding):
        compiled = compile_encoding(encoding, variable_order=order)
    else:
        from .grammar import compile_grammar

        compiled = compile_grammar(
            encoding,
            variable_order=order,
            max_nodes=max_nodes,
            max_seconds=seconds,
            vtree_type=vtree_type,
            collect=collect,
            max_live_nodes=max_live_nodes,
            max_reachable_nodes=max_reachable_nodes,
            max_elements=max_elements,
        )
    if compiled.node_count > max_nodes:
        raise ValueError("Compiled circuit exceeds the declared node budget")
    if max_reachable_nodes is not None and compiled.node_count > max_reachable_nodes:
        raise ValueError("Compiled circuit exceeds the reachable node budget")
    if max_elements is not None and compiled.root.size() > max_elements:
        raise ValueError("Compiled circuit exceeds the element budget")
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
            compiled.telemetry,
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


def _compile_persistent_worker(
    encoding: Any,
    *,
    seconds: float = 20.0,
    max_nodes: int = 100000,
    variable_order: Sequence[int] | None = None,
    cache_directory: str | None = None,
    cache_bytes: int = 268435456,
    vtree_type: str = "balanced",
    collect: bool = True,
    max_live_nodes: int | None = None,
    max_reachable_nodes: int | None = None,
    max_elements: int | None = None,
) -> CompiledCircuit:
    """Compile in a killable worker; transport its immutable public evaluation DAG.

    Native compilation, model counting and public artifact serialization all run
    under supervision. The parent only reconstructs checked Python records under
    the same deadline. Neural log weights and autograd remain in the caller.
    """
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    started = monotonic()
    implementation_hash = canonical_hash(
        tuple(
            (name, hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest())
            for name in ("compilation.py", "grammar.py", "circuit.py", "candidates.py")
        )
    )
    identity = canonical_hash(
        (
            "compiler-artifact/v3",
            encoding,
            variable_order,
            version("pysdd"),
            implementation_hash,
            vtree_type,
            collect,
        )
    )
    directory = Path(cache_directory) if cache_directory else None
    lock = None
    payload = None
    cache_hit = False
    cache_rejected = False
    try:
        if directory is not None:
            import fcntl
            import time

            directory.mkdir(parents=True, exist_ok=True)
            lock = (directory / (identity + ".lock")).open("a+b")
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if monotonic() >= started + seconds:
                        raise TimeoutError("Circuit cache claim deadline exhausted")
                    time.sleep(min(0.01, max(0, started + seconds - monotonic())))
            path = directory / (identity + ".json")
            if path.exists():
                try:
                    if path.stat().st_size > cache_bytes:
                        raise ValueError("artifact exceeds cache admission budget")
                    raw = path.read_bytes()
                    envelope = json.loads(raw)
                    value = envelope["payload"]
                    if (
                        envelope["schema"] != "exact-repair/compiler-artifact/v3"
                        or envelope["identity"] != identity
                        or envelope["sha256"] != canonical_hash(value)
                        or value["compiler_version"] != version("pysdd")
                    ):
                        raise ValueError("artifact identity or integrity mismatch")
                    data = dict(value["artifact"])
                    data["sdd"] = base64.b64decode(data["sdd"], validate=True)
                    data["vtree"] = base64.b64decode(data["vtree"], validate=True)
                    artifact = CircuitArtifact(**data)
                    if (
                        artifact.variable_count != encoding.variable_count
                        or artifact.node_count > max_nodes
                    ):
                        raise ValueError("artifact binding or resource admission mismatch")
                    if max_live_nodes is not None and artifact.node_count > max_live_nodes:
                        raise ValueError("artifact live node limit admission mismatch")
                    if (
                        max_reachable_nodes is not None
                        and artifact.node_count > max_reachable_nodes
                    ):
                        raise ValueError("artifact reachable node limit admission mismatch")
                    if (
                        max_elements is not None
                        and sum(len(row[3]) for row in value["table"]) > max_elements
                    ):
                        raise ValueError("artifact element limit admission mismatch")
                    payload = (
                        artifact,
                        value["table"],
                        value["root_id"],
                        value["key"],
                        value["compiler_version"],
                    )
                    cache_hit = True
                except (ValueError, KeyError, TypeError, OSError):
                    cache_rejected = True
                    payload = None
        if payload is None:
            remaining = seconds - (monotonic() - started)
            # This entire function runs in the independently supervised worker:
            # cache I/O, JSON parsing, native compilation and reconstruction all
            # consume the same deadline, with no native operation in the parent.
            payload = _serialize_compile(
                encoding,
                None if variable_order is None else tuple(variable_order),
                max_nodes,
                remaining,
                vtree_type,
                collect,
                max_live_nodes,
                max_reachable_nodes,
                max_elements,
            )
            if directory is not None:
                artifact, table, root_id, key, compiler_version = payload
                data = asdict(artifact)
                data["sdd"] = base64.b64encode(artifact.sdd).decode()
                data["vtree"] = base64.b64encode(artifact.vtree).decode()
                value = dict(
                    artifact=data,
                    table=table,
                    root_id=root_id,
                    key=key,
                    compiler_version=compiler_version,
                    cold_limits={
                        "seconds": seconds,
                        "max_nodes": max_nodes,
                        "max_live_nodes": max_live_nodes,
                        "max_reachable_nodes": max_reachable_nodes,
                        "max_elements": max_elements,
                    },
                )
                raw = json.dumps(
                    dict(
                        schema="exact-repair/compiler-artifact/v3",
                        identity=identity,
                        sha256=canonical_hash(value),
                        payload=value,
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                if len(raw) <= cache_bytes and monotonic() < started + seconds:
                    # Immutable successful artifacts only; atomic replacement also
                    # safely repairs a corrupt prior artifact under the same claim.
                    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
                        temporary = Path(handle.name)
                        handle.write(raw)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, directory / (identity + ".json"))
                    entries = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
                    used = sum(p.stat().st_size for p in entries)
                    for old in entries:
                        if used <= cache_bytes:
                            break
                        if old.name != identity + ".json":
                            try:
                                size = old.stat().st_size
                                old.unlink()
                                used -= size
                            except FileNotFoundError:
                                pass
    finally:
        if lock is not None:
            lock.close()
    artifact, table, root_id, key, version_name = payload
    key = canonical_hash(
        (
            "compiled-artifact/v3",
            identity,
            key,
            hashlib.sha256(artifact.sdd).hexdigest(),
            hashlib.sha256(artifact.vtree).hexdigest(),
        )
    )
    root = _restore_evaluation_nodes(artifact, table, root_id, deadline=started + seconds)
    return CompiledCircuit(
        encoding,
        artifact,
        root,
        key,
        version_name,
        monotonic() - started,
        artifact.node_count,
        tuple(artifact.telemetry)
        + (
            ("persistent_cache_hit", cache_hit),
            ("cache_identity", identity),
            ("cache_rejected", cache_rejected),
            ("compiler_implementation_hash", implementation_hash),
        ),
    )


@lru_cache(maxsize=32)
def _compile_bounded_cached(
    encoding: Any,
    *,
    seconds: float,
    max_nodes: int,
    variable_order: tuple[int, ...] | None,
    cache_directory: str | None,
    cache_bytes: int,
    vtree_type: str,
    collect: bool,
    memory_mb: float | None = None,
    max_live_nodes: int | None = None,
    max_reachable_nodes: int | None = None,
    max_elements: int | None = None,
) -> CompiledCircuit:
    outcome = bounded_call(
        _compile_persistent_worker,
        encoding,
        seconds=seconds,
        max_nodes=max_nodes,
        variable_order=variable_order,
        cache_directory=cache_directory,
        cache_bytes=cache_bytes,
        vtree_type=vtree_type,
        collect=collect,
        max_live_nodes=max_live_nodes,
        max_reachable_nodes=max_reachable_nodes,
        max_elements=max_elements,
        timeout=seconds,
        memory_mb=memory_mb,
    )
    if outcome.status == "timeout":
        raise TimeoutError("Circuit compilation/cache deadline exhausted")
    if outcome.status != "complete":
        raise RuntimeError(f"Circuit compilation failed ({outcome.status}): {outcome.detail}")
    circuit = outcome.value
    if not isinstance(circuit, CompiledCircuit):
        raise TypeError("compilation worker returned an invalid circuit")
    receipt = dict(circuit.telemetry)
    _DISK_STATS["hits" if receipt.get("persistent_cache_hit") else "misses"] += 1
    _DISK_STATS["rejected"] += int(receipt.get("cache_rejected", False))
    return circuit


def compile_bounded(
    encoding: Any,
    *,
    seconds: float = 20.0,
    max_nodes: int = 100000,
    variable_order: Sequence[int] | None = None,
    cache_directory: str | Path | None = None,
    cache_bytes: int = 268435456,
    vtree_type: str = "balanced",
    collect: bool = True,
    memory_mb: float | None = None,
    max_live_nodes: int | None = None,
    max_reachable_nodes: int | None = None,
    max_elements: int | None = None,
) -> CompiledCircuit:
    """Reuse identical compiled languages; resource limits remain part of cache identity."""
    if seconds <= 0 or cache_bytes < 1:
        raise ValueError("compilation and cache budgets must be positive")
    if cache_directory is None:
        cache_directory = os.environ.get(
            "EXACT_REPAIR_CIRCUIT_CACHE",
            str(Path(tempfile.gettempdir()) / f"exact-repair-circuits-v3-{os.getuid()}"),
        )
    return _compile_bounded_cached(
        encoding,
        seconds=seconds,
        max_nodes=max_nodes,
        variable_order=None if variable_order is None else tuple(variable_order),
        cache_directory=str(cache_directory) if cache_directory else None,
        cache_bytes=cache_bytes,
        vtree_type=vtree_type,
        collect=collect,
        memory_mb=memory_mb,
        max_live_nodes=max_live_nodes,
        max_reachable_nodes=max_reachable_nodes,
        max_elements=max_elements,
    )


_DISK_STATS = {"hits": 0, "misses": 0, "rejected": 0}


def compilation_cache_info() -> dict[str, int]:
    """Expose cache reuse counters without changing compiled-object identity."""
    info = _compile_bounded_cached.cache_info()
    return {
        "hits": info.hits + _DISK_STATS["hits"],
        "disk_hits": _DISK_STATS["hits"],
        "disk_rejected": _DISK_STATS["rejected"],
        "misses": info.misses,
        "entries": info.currsize,
        "capacity": info.maxsize or 0,
    }
