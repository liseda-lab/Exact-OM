"""Hard native compilation deadlines and immutable, deadline-checked circuit transport."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from time import monotonic, process_time
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


def _phase_start() -> tuple[float, float]:
    return monotonic(), process_time()


def _phase(name: str, started: tuple[float, float]) -> dict[str, Any]:
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(
        phase=name,
        wall_seconds=monotonic() - started[0],
        cpu_seconds=process_time() - started[1],
        peak_process_rss_bytes=(
            int(peak * 1024)
            if sys.platform.startswith("linux")
            else int(peak) if sys.platform == "darwin" else None
        ),
        rss_scope="worker process lifetime high water; not a phase allocation delta",
    )


def _publish_cache(
    directory: Path, identity: str, raw: bytes, capacity: int, deadline: float
) -> dict[str, Any]:
    """Short maintenance transaction; disappearing immutable entries are harmless.

    Native compilation stays outside the shared maintenance lock. Publication and
    eviction are serialized only to enforce the directory capacity after a commit.
    Failure leaves the already-validated in-memory circuit usable and is explicit.
    """
    import fcntl
    import time

    if len(raw) > capacity:
        return dict(
            status="not_admitted",
            detail="artifact exceeds cache byte capacity",
            bytes=len(raw),
            capacity_bytes=capacity,
        )
    temporary = None
    try:
        with (directory / ".maintenance.lock").open("a+b") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if monotonic() >= deadline:
                        return dict(
                            status="maintenance_timeout",
                            detail="bounded cache maintenance lock",
                            capacity_bytes=capacity,
                        )
                    time.sleep(min(0.005, max(0, deadline - monotonic())))
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=".publish-", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            target = directory / (identity + ".json")
            os.replace(temporary, target)
            temporary = None
            entries: list[tuple[int, str, int, Path]] = []
            for path in directory.glob("*.json"):
                if monotonic() >= deadline:
                    return dict(
                        status="maintenance_timeout",
                        detail="capacity metadata scan incomplete",
                        capacity_bytes=capacity,
                        observed_entries=len(entries),
                    )
                try:
                    metadata = path.stat()
                except FileNotFoundError:
                    continue
                entries.append((metadata.st_mtime_ns, path.name, metadata.st_size, path))
            used = sum(row[2] for row in entries)
            evicted = 0
            for _, _, size, path in sorted(entries):
                if used <= capacity:
                    break
                if monotonic() >= deadline:
                    return dict(
                        status="maintenance_timeout",
                        detail="capacity cleanup incomplete",
                        capacity_bytes=capacity,
                        observed_bytes=used,
                        evicted_entries=evicted,
                    )
                if path != target:
                    try:
                        path.unlink()
                        evicted += 1
                    except FileNotFoundError:
                        pass
                    used -= size
            return dict(
                status="complete",
                capacity_bytes=capacity,
                observed_bytes=used,
                evicted_entries=evicted,
                bytes=len(raw),
            )
    except OSError as error:
        return dict(
            status="maintenance_error",
            detail=f"{type(error).__name__}: {error}",
            capacity_bytes=capacity,
        )
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


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
    phase_started = _phase_start()
    if isinstance(encoding, ProposalEncoding):
        compiled = compile_encoding(
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
    if compiled.manager.count() > max_nodes:
        raise ValueError("Compiled circuit exceeds the declared allocated node budget")
    if max_live_nodes is not None and compiled.manager.live_count() > max_live_nodes:
        raise ValueError("Compiled circuit exceeds the declared live node budget")
    if max_reachable_nodes is not None and compiled.node_count > max_reachable_nodes:
        raise ValueError("Compiled circuit exceeds the reachable node budget")
    if max_elements is not None and compiled.root.size() > max_elements:
        raise ValueError("Compiled circuit exceeds the element budget")
    native_phase = _phase("native_compile", phase_started)
    phase_started = _phase_start()
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
    dag_phase = _phase("dag_serialization", phase_started)
    phase_started = _phase_start()
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
    save_phase = _phase("native_save", phase_started)
    counters = dict(
        manager_allocated_nodes=int(compiled.manager.count()),
        manager_live_nodes=int(compiled.manager.live_count()),
        manager_dead_nodes=int(compiled.manager.dead_count()),
        manager_elements=int(compiled.manager.size()),
        root_reachable_nodes=compiled.node_count,
        root_elements=int(compiled.root.size()),
        serialized_sdd_bytes=len(artifact.sdd),
        serialized_vtree_bytes=len(artifact.vtree),
        minimization_seconds=None,
        peak_manager_allocated_nodes=dict(compiled.telemetry).get("peak_allocated"),
    )
    artifact = replace(
        artifact,
        telemetry=tuple(artifact.telemetry)
        + (
            ("native_phases", (native_phase, dag_phase, save_phase)),
            ("backend_measurements", counters),
        ),
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
    admission_memory_mb: float | None = None,
) -> CompiledCircuit:
    """Supervise compile/cache/save/restore work and preserve the originating cold receipt."""
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("max_nodes must be a positive integer")
    if any(
        limit is not None and (type(limit) is not int or limit < 1)
        for limit in (max_live_nodes, max_reachable_nodes, max_elements)
    ):
        raise ValueError("circuit structural limits must be positive integers")
    if type(collect) is not bool:
        raise ValueError("compiler collect must be Boolean")
    layouts = (
        {"right", "balanced"}
        if isinstance(encoding, ProposalEncoding)
        else {"right", "balanced", "grouped"}
    )
    if vtree_type not in layouts:
        raise ValueError("unsupported vtree_type for this compiler encoding")
    if type(seconds) not in (int, float) or not isfinite(seconds) or seconds <= 0:
        raise ValueError("compiler wall limit must be finite and positive")
    total_started = _phase_start()
    started = total_started[0]
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
    limits = dict(
        seconds=seconds,
        max_nodes=max_nodes,
        max_live_nodes=max_live_nodes,
        max_reachable_nodes=max_reachable_nodes,
        max_elements=max_elements,
        memory_mb=admission_memory_mb,
        cache_bytes=cache_bytes,
    )
    directory = Path(cache_directory) if cache_directory else None
    lock = None
    payload = None
    cold_receipt = None
    cache_hit = cache_rejected = False
    phases = []
    publication = dict(status="disabled")
    try:
        phase_started = _phase_start()
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
                    envelope = json.loads(path.read_bytes())
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
                    if artifact.variable_count != encoding.variable_count:
                        raise ValueError("artifact binding mismatch")
                    measured = dict(artifact.telemetry)["backend_measurements"]
                    limits_and_counts = (
                        ("allocated node", max_nodes, measured["manager_allocated_nodes"]),
                        ("live node", max_live_nodes, measured["manager_live_nodes"]),
                        ("reachable node", max_reachable_nodes, artifact.node_count),
                        ("element", max_elements, measured["root_elements"]),
                    )
                    for name, limit, count in limits_and_counts:
                        if limit is not None and count > limit:
                            from .grammar import CircuitBudgetExceeded

                            raise CircuitBudgetExceeded(f"artifact {name} limit admission mismatch")
                    payload = (
                        artifact,
                        value["table"],
                        value["root_id"],
                        value["key"],
                        value["compiler_version"],
                    )
                    cold_receipt = value.get("cold_receipt")
                    cache_hit = True
                except (ValueError, KeyError, TypeError, OSError):
                    cache_rejected = True
                    payload = None
        phases.append(_phase("cache_lookup_load", phase_started))
        if payload is None:
            payload = _serialize_compile(
                encoding,
                None if variable_order is None else tuple(variable_order),
                max_nodes,
                seconds - (monotonic() - started),
                vtree_type,
                collect,
                max_live_nodes,
                max_reachable_nodes,
                max_elements,
            )
            phases.extend(dict(row) for row in dict(payload[0].telemetry)["native_phases"])
        artifact, table, root_id, key, version_name = payload
        phase_started = _phase_start()
        root = _restore_evaluation_nodes(artifact, table, root_id, deadline=started + seconds)
        restoration = _phase("evaluation_restore", phase_started)
        phases.append(restoration)
        if not cache_hit:
            cold_receipt = dict(
                schema="exact-repair/cold-compiler-receipt/review-1",
                structural_identity=identity,
                limits=limits,
                phases=[dict(row) for row in dict(artifact.telemetry)["native_phases"]]
                + [restoration],
                measurements=dict(artifact.telemetry)["backend_measurements"],
                scope="native compilation, DAG serialization, native save and initial restore; cache publication is per-call",
            )
            if directory is not None:
                phase_started = _phase_start()
                data = asdict(artifact)
                data["sdd"], data["vtree"] = (
                    base64.b64encode(artifact.sdd).decode(),
                    base64.b64encode(artifact.vtree).decode(),
                )
                value = dict(
                    artifact=data,
                    table=table,
                    root_id=root_id,
                    key=key,
                    compiler_version=version_name,
                    cold_limits=limits,
                    cold_receipt=cold_receipt,
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
                phases.append(_phase("cache_serialization", phase_started))
                phase_started = _phase_start()
                publication = _publish_cache(
                    directory, identity, raw, cache_bytes, min(started + seconds, monotonic() + 0.5)
                )
                phases.append(_phase("cache_publication_maintenance", phase_started))
    finally:
        if lock is not None:
            lock.close()
    elapsed = _phase("worker_total", total_started)
    receipt = dict(
        schema="exact-repair/compiler-measurements/review-1",
        structural_identity=identity,
        resource_contract="compiler-resource-contract/review-2",
        cache_mode="warm_load" if cache_hit else "cold_compile",
        cold_receipt=cold_receipt,
        admission_limits=limits,
        phases=phases,
        total=elapsed,
        cache_publication=publication,
        artifact_bytes=len(artifact.sdd) + len(artifact.vtree),
        supervised_call=None,
        memory_supervision=(
            "worker-tree RSS sampling" if admission_memory_mb is not None else "not requested"
        ),
    )
    key = canonical_hash(
        (
            "compiled-artifact/v3",
            identity,
            key,
            hashlib.sha256(artifact.sdd).hexdigest(),
            hashlib.sha256(artifact.vtree).hexdigest(),
        )
    )
    return CompiledCircuit(
        encoding,
        artifact,
        root,
        key,
        version_name,
        elapsed["wall_seconds"],
        artifact.node_count,
        tuple(artifact.telemetry)
        + (
            ("persistent_cache_hit", cache_hit),
            ("cache_identity", identity),
            ("cache_rejected", cache_rejected),
            ("compiler_implementation_hash", implementation_hash),
            ("measurement_receipt", receipt),
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
    supervised_started = monotonic()
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
        admission_memory_mb=memory_mb,
        timeout=seconds,
        memory_mb=memory_mb,
    )
    supervised = dict(
        wall_seconds=monotonic() - supervised_started,
        resources=dict(outcome.resource_usage),
        status=outcome.status,
        scope="input/startup/worker/result transfer and cleanup",
    )
    if outcome.status != "complete":
        error = (
            TimeoutError("Circuit compilation/cache deadline exhausted")
            if outcome.status == "timeout"
            else RuntimeError(f"Circuit compilation failed ({outcome.status}): {outcome.detail}")
        )
        setattr(
            error,
            "measurement_receipt",
            dict(
                schema="exact-repair/compiler-measurements/review-1",
                resource_contract="compiler-resource-contract/review-2",
                structural_identity=None,
                encoding_identity=encoding.content_hash,
                cache_mode="failed_attempt",
                cold_receipt=None,
                phases=None,
                total=None,
                supervised_call=supervised,
                failure=outcome.detail,
                artifact_bytes=None,
            ),
        )
        raise error
    circuit = outcome.value
    if not isinstance(circuit, CompiledCircuit):
        raise TypeError("compilation worker returned an invalid circuit")
    receipt = dict(circuit.telemetry)
    receipt["measurement_receipt"] = {
        **receipt["measurement_receipt"],
        "supervised_call": supervised,
    }
    circuit = replace(circuit, telemetry=tuple(receipt.items()))
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
    started = _phase_start()
    previous_hits = _compile_bounded_cached.cache_info().hits
    compiled = _compile_bounded_cached(
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
    if _compile_bounded_cached.cache_info().hits == previous_hits:
        return compiled
    telemetry = dict(compiled.telemetry)
    previous_receipt = telemetry["measurement_receipt"]
    current = _phase("memory_cache_lookup", started)
    telemetry["measurement_receipt"] = {
        **previous_receipt,
        "cache_mode": "memory_reuse",
        "phases": [current],
        "total": current,
        "supervised_call": None,
        "cache_publication": {"status": "not performed"},
    }
    return replace(
        compiled, compilation_seconds=current["wall_seconds"], telemetry=tuple(telemetry.items())
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
