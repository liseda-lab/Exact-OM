#!/usr/bin/env python3
"""Record shared-OWL-stack scale evidence for legally available ontologies."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal, cast

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

import pyowl_core  # noqa: E402
from pyowl2vec_star_projector import ProjectionOptions  # noqa: E402

from exact.ontology import load_ontology  # noqa: E402


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


LoadBackend = Literal["auto", "python", "native"]
ProjectorBackend = Literal["auto", "python", "native"]
ReasonerName = Literal["asserted", "elk", "hermit"]

_REQUIRED_ENCODED_COUNTERS = (
    "encoded_buffer_bytes",
    "encoded_buffer_count",
    "encoded_compiler_gil_released",
    "encoded_staging_copy_bytes",
    "encoded_zero_copy_buffers",
)
_REQUIRED_ZERO_COUNTERS = (
    "base_flattening_bytes",
    "materialized_scalar_rows",
    "parser_calls",
    "per_row_ffi_calls",
    "resolver_calls",
    "scalar_axiom_materializations",
    "scalar_term_materializations",
    "structural_copy_bytes",
    "wire_decoder_calls",
    "wire_encoder_calls",
)
_OPTIONAL_ZERO_COUNTERS = ("encoded_private_ir_bytes",)


def _fingerprints(snapshot: pyowl_core.OntologyView) -> dict[str, str]:
    return {
        name: f"{value.algorithm}:{value.schema}:{value.hex}"
        for name, value in (
            ("structural", snapshot.structural_fingerprint),
            ("logical", snapshot.logical_fingerprint),
            ("signature", snapshot.signature_fingerprint),
        )
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(distribution: str, fallback: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def _cpu_seconds() -> float:
    """Return process plus completed-child CPU time for worker-aware evidence."""

    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime)


def _json_record(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def _edge_record(edge: object) -> bytes:
    """Encode one edge using the projectors' public artifact record contract."""

    source = getattr(edge, "source", None)
    relation = getattr(edge, "relation", None)
    destination = getattr(edge, "destination", None)
    if source is None:
        source = getattr(edge, "src")
    if relation is None:
        relation = getattr(edge, "rel")
    if destination is None:
        destination = getattr(edge, "dst")
    return (
        json.dumps(
            {
                "source": cast(str, source),
                "relation": cast(str, relation),
                "destination": cast(str, destination),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


@dataclass(slots=True)
class _CoreOperationProbe:
    """Count public core acquisition/wire entry points without decoding a view."""

    load_snapshot: int = 0
    encode_snapshot: int = 0
    decode_snapshot: int = 0
    open_snapshot: int = 0

    def snapshot(self) -> dict[str, int]:
        return asdict(self)


def _counter_delta(after: Mapping[str, int], before: Mapping[str, int]) -> dict[str, int]:
    result = {name: after[name] - before[name] for name in before}
    if any(value < 0 for value in result.values()):  # pragma: no cover - local probe invariant
        raise RuntimeError("core operation counters moved backwards")
    return result


@contextmanager
def _probe_core_operations() -> Iterator[_CoreOperationProbe]:
    """Instrument stable top-level calls used by Exact; always restore the module."""

    probe = _CoreOperationProbe()
    originals: dict[str, Callable[..., object]] = {}
    for name in probe.snapshot():
        original = getattr(pyowl_core, name, None)
        if callable(original):
            originals[name] = original

    def wrapper(name: str, original: Callable[..., object]) -> Callable[..., object]:
        def counted(*args: object, **kwargs: object) -> object:
            setattr(probe, name, getattr(probe, name) + 1)
            return original(*args, **kwargs)

        return counted

    try:
        for name, original in originals.items():
            setattr(pyowl_core, name, wrapper(name, original))
        yield probe
    finally:
        for name, original in originals.items():
            setattr(pyowl_core, name, original)


def _dataclass_dict(value: object) -> dict[str, object] | None:
    if value is None or isinstance(value, type) or not is_dataclass(value):
        return None
    rendered = asdict(cast(Any, value))
    return cast(dict[str, object], rendered)


def _reasoner_handoff(provenance: Mapping[str, object]) -> dict[str, object] | None:
    handoff = provenance.get("consumer_handoff")
    if not isinstance(handoff, Mapping):
        return None
    return {str(name): value for name, value in handoff.items()}


def _public_counters(*records: Mapping[str, object] | None) -> dict[str, int | bool]:
    counters: dict[str, int | bool] = {}
    for record in records:
        if record is None:
            continue
        nested = record.get("counters")
        candidates = nested if isinstance(nested, Mapping) else record
        for raw_name, value in candidates.items():
            name = str(raw_name)
            if isinstance(value, bool):
                counters[name] = value
            elif isinstance(value, int) and value >= 0:
                counters[name] = value
    return dict(sorted(counters.items()))


def _sum_matching_counters(
    counters: Mapping[str, int | bool],
    *,
    fragments: tuple[str, ...],
) -> int | None:
    selected = [
        value
        for name, value in counters.items()
        if not isinstance(value, bool) and any(fragment in name for fragment in fragments)
    ]
    return sum(selected) if selected else None


def _require_encoded_path(
    *,
    consumer: str,
    handoff: Mapping[str, object] | None,
) -> None:
    path = None if handoff is None else handoff.get("ingestion_path")
    if path != "encoded-native":
        raise RuntimeError(
            f"{consumer} did not select required encoded-native ingestion (selected {path!r})"
        )


def _is_zero_counter(value: object) -> bool:
    return type(value) is int and value == 0


def _is_nonnegative_counter(value: object) -> bool:
    return type(value) is int and value >= 0


def _consumer_counter_evidence(
    *,
    consumer: str,
    handoff: Mapping[str, object] | None,
) -> dict[str, object]:
    """Evaluate one direct-view consumer's public WP-N handoff evidence."""

    path = None if handoff is None else handoff.get("ingestion_path")
    raw_counters = None if handoff is None else handoff.get("counters")
    counters = dict(raw_counters) if isinstance(raw_counters, Mapping) else {}
    required = (*_REQUIRED_ENCODED_COUNTERS, *_REQUIRED_ZERO_COUNTERS)
    missing = [name for name in required if name not in counters]
    invalid = {
        name: counters[name]
        for name in (*required, *_OPTIONAL_ZERO_COUNTERS)
        if name in counters
        and name != "encoded_compiler_gil_released"
        and not _is_nonnegative_counter(counters[name])
    }
    forbidden = {
        name: counters[name]
        for name in (*_REQUIRED_ZERO_COUNTERS, *_OPTIONAL_ZERO_COUNTERS)
        if name in counters and not _is_zero_counter(counters[name])
    }
    staging = counters.get("encoded_staging_copy_bytes")
    gil_released = counters.get("encoded_compiler_gil_released")
    buffer_count = counters.get("encoded_buffer_count")
    zero_copy_buffers = counters.get("encoded_zero_copy_buffers")
    all_buffers_zero_copy = (
        type(buffer_count) is int
        and buffer_count > 0
        and type(zero_copy_buffers) is int
        and zero_copy_buffers == buffer_count
    )
    ready = (
        path == "encoded-native"
        and not missing
        and not invalid
        and not forbidden
        and _is_zero_counter(staging)
        and gil_released is True
        and all_buffers_zero_copy
    )
    return {
        "consumer": consumer,
        "selected_ingestion_path": path,
        "complete_public_counter_coverage": path == "encoded-native" and not missing,
        "missing_public_counters": missing,
        "invalid_public_counters": invalid,
        "nonzero_forbidden_public_counters": forbidden,
        "direct_staging_copy_bytes": staging,
        "encoded_compiler_gil_released": gil_released,
        "encoded_buffer_count": buffer_count,
        "encoded_zero_copy_buffers": zero_copy_buffers,
        "all_encoded_buffers_zero_copy": all_buffers_zero_copy,
        "acceptance_ready": ready,
    }


def _require_consumer_counter_evidence(evidence: Mapping[str, object]) -> None:
    if evidence.get("acceptance_ready") is True:
        return
    consumer = evidence.get("consumer")
    raise RuntimeError(
        f"{consumer} encoded handoff acceptance evidence failed: "
        f"missing={evidence.get('missing_public_counters')!r}, "
        f"invalid={evidence.get('invalid_public_counters')!r}, "
        f"nonzero={evidence.get('nonzero_forbidden_public_counters')!r}, "
        f"staging={evidence.get('direct_staging_copy_bytes')!r}, "
        f"gil_released={evidence.get('encoded_compiler_gil_released')!r}, "
        f"all_buffers_zero_copy={evidence.get('all_encoded_buffers_zero_copy')!r}"
    )


def _hierarchy_measurement(source: Any) -> dict[str, object]:
    started = time.perf_counter()
    cpu_started = _cpu_seconds()
    digest = hashlib.sha256(b"exact:direct-parent-results:v1\0")
    first_result_seconds: float | None = None
    entities = 0
    relationships = 0
    for iri in source.entities():
        parents = sorted(source.direct_parents(iri))
        if first_result_seconds is None:
            first_result_seconds = time.perf_counter() - started
        digest.update(_json_record({"entity": iri, "direct_parents": parents}))
        entities += 1
        relationships += len(parents)
    return {
        "entities": entities,
        "relationships": relationships,
        "result_sha256": digest.hexdigest(),
        "time_to_first_result_seconds": first_result_seconds,
        "complete_result_seconds": time.perf_counter() - started,
        "cpu_seconds": _cpu_seconds() - cpu_started,
    }


def measure(
    path: Path,
    *,
    buffer_edges: int,
    include_literals: bool,
    load_backend: LoadBackend = "python",
    projector_backend: ProjectorBackend = "python",
    reasoner_name: ReasonerName | None = None,
    reasoner_backend: str = "auto",
    reasoner_workers: int = 0,
    reasoner_timeout_seconds: float | None = None,
    reasoner_worker_wire: bool = False,
    require_encoded_consumers: bool = False,
) -> dict[str, Any]:
    """Measure one source while asserting the WP-N handoff invariants."""

    print(f"[owl-stack-scale] hashing {path.name}", file=sys.stderr, flush=True)
    input_bytes = path.stat().st_size
    input_sha256 = _sha256_file(path)
    before_rss = _max_rss_bytes()
    whole_started = time.perf_counter()
    whole_cpu_started = _cpu_seconds()

    with _probe_core_operations() as core_operations:
        print(f"[owl-stack-scale] loading {path.name}", file=sys.stderr, flush=True)
        load_started = time.perf_counter()
        load_cpu_started = _cpu_seconds()
        source = load_ontology(
            path,
            options=pyowl_core.LoadOptions(
                backend=pyowl_core.BackendPreference(load_backend),
            ),
        )
        load_seconds = time.perf_counter() - load_started
        load_cpu_seconds = _cpu_seconds() - load_cpu_started
        after_load_operations = core_operations.snapshot()

        snapshot = source.owl_snapshot()
        source.configure_projector(
            backend=projector_backend,
            profile=source.projector_settings.profile,
        )
        initial_fingerprints = _fingerprints(snapshot)
        initial_axioms = snapshot.report.effective_axiom_count

        options = ProjectionOptions(
            profile=source.projector_settings.profile,
            include_literals=include_literals,
            duplicates="unique",
            order="canonical",
            compatibility_state="isolated",
            backend=projector_backend,
        )
        projection_started = time.perf_counter()
        projection_cpu_started = _cpu_seconds()
        projection_operations_before = core_operations.snapshot()
        print(f"[owl-stack-scale] projecting {path.name}", file=sys.stderr, flush=True)
        iterator = source.projector.iter_edges(
            snapshot,
            options=options,
            buffer_edges=buffer_edges,
        )
        edge_digest = hashlib.sha256()
        first_edge_seconds: float | None = None
        edge_count = 0
        for projected_edge in iterator:
            if first_edge_seconds is None:
                first_edge_seconds = time.perf_counter() - projection_started
            edge_digest.update(_edge_record(projected_edge))
            edge_count += 1
        projection_seconds = time.perf_counter() - projection_started
        projection_cpu_seconds = _cpu_seconds() - projection_cpu_started
        projection_operation_delta = _counter_delta(
            core_operations.snapshot(), projection_operations_before
        )
        spill = asdict(source.projector.last_spill_metrics)
        first_projection_report = source.projector.last_report
        if first_projection_report is None:
            raise RuntimeError("projection completed without a public report")
        first_projection_provenance = first_projection_report.provenance.to_dict()
        first_ingestion = first_projection_report.provenance.ingestion
        first_projector_handoff = {
            "ingestion_path": first_ingestion.path,
            "schema_name": first_ingestion.encoded_schema_name,
            "schema_version": first_ingestion.encoded_schema_version,
            "descriptor_sha256": first_ingestion.encoded_descriptor_sha256,
            "compiler_cache_schema": first_projection_report.provenance.compiler_cache_schema,
            "compiler_digest": None,
            "implementation_version": (
                first_projection_report.provenance.native_implementation_version
            ),
            "encoded_view_publication_seconds": (first_ingestion.encoded_view_publication_seconds),
            "consumer_compile_seconds": first_ingestion.consumer_compile_seconds,
            "counters": dict(first_ingestion.counters),
        }
        first_encoded_counters = _dataclass_dict(source.projector.last_encoded_counters)

        cache_operations_before = core_operations.snapshot()
        cache_started = time.perf_counter()
        print(f"[owl-stack-scale] filling cache for {path.name}", file=sys.stderr, flush=True)
        first_cache_edges = source.projection_edges(include_literals=include_literals)
        cache_fill_seconds = time.perf_counter() - cache_started
        first_cache_count = len(first_cache_edges)
        first_cache_digest = hashlib.sha256()
        for cached_edge in first_cache_edges:
            first_cache_digest.update(_edge_record(cached_edge))
        cache_started = time.perf_counter()
        second_cache_edges = source.projection_edges(include_literals=include_literals)
        cache_hit_seconds = time.perf_counter() - cache_started
        second_cache_count = len(second_cache_edges)
        second_cache_digest = hashlib.sha256()
        for hit_edge in second_cache_edges:
            second_cache_digest.update(_edge_record(hit_edge))
        cache_operation_delta = _counter_delta(core_operations.snapshot(), cache_operations_before)

        reasoner_configuration: dict[str, object] | None = None
        reasoner_results: dict[str, object] | None = None
        reasoner_operations_before = core_operations.snapshot()
        if reasoner_name is not None:
            print(
                f"[owl-stack-scale] configuring {reasoner_name} for {path.name}",
                file=sys.stderr,
                flush=True,
            )
            reasoner_started = time.perf_counter()
            reasoner_cpu_started = _cpu_seconds()
            source.configure_reasoner(
                reasoner_name,
                backend=reasoner_backend,
                workers=reasoner_workers,
                timeout_seconds=reasoner_timeout_seconds,
                worker_wire=reasoner_worker_wire,
            )
            reasoner_configuration = {
                "wall_seconds": time.perf_counter() - reasoner_started,
                "cpu_seconds": _cpu_seconds() - reasoner_cpu_started,
            }
            print(
                f"[owl-stack-scale] querying {reasoner_name} for {path.name}",
                file=sys.stderr,
                flush=True,
            )
            reasoner_results = _hierarchy_measurement(source)

        reasoner = source.reasoner
        reasoner_provenance = source.reasoner_provenance
        reasoner_handoff = _reasoner_handoff(reasoner_provenance)
        reasoner_operation_delta = _counter_delta(
            core_operations.snapshot(), reasoner_operations_before
        )
        all_operations = core_operations.snapshot()

        stack_provenance = source.ontology_stack_provenance()
        final_fingerprints = _fingerprints(snapshot)
        final_axioms = snapshot.report.effective_axiom_count
        after_rss = _max_rss_bytes()
        identity = {
            "source_snapshot": source.owl_snapshot() is snapshot,
            "projector_snapshot": source.projector.last_view is snapshot,
            "reasoner_snapshot": getattr(reasoner, "ontology", None) is snapshot,
        }

        close_reasoner = getattr(reasoner, "close", None)
        if callable(close_reasoner):
            close_reasoner()

        if require_encoded_consumers:
            _require_encoded_path(consumer="projector", handoff=first_projector_handoff)
            if reasoner_name in {"elk", "hermit"}:
                _require_encoded_path(consumer=reasoner_name, handoff=reasoner_handoff)

    result_sha256 = edge_digest.hexdigest()
    first_cache_sha256 = first_cache_digest.hexdigest()
    second_cache_sha256 = second_cache_digest.hexdigest()
    if all_operations["load_snapshot"] != 1 or not all(identity.values()):
        raise RuntimeError(
            "shared snapshot invariant failed: "
            f"loads={all_operations['load_snapshot']}, {identity}"
        )
    if initial_fingerprints != final_fingerprints or initial_axioms != final_axioms:
        raise RuntimeError("projection/reasoning mutated the shared snapshot")
    if (
        edge_count != first_cache_count
        or first_cache_count != second_cache_count
        or result_sha256 != first_cache_sha256
        or first_cache_sha256 != second_cache_sha256
    ):
        raise RuntimeError("projection cache changed its result cardinality or digest")

    consumer_operations = _counter_delta(all_operations, after_load_operations)
    projector_public_counters = _public_counters(first_projector_handoff)
    reasoner_public_counters = _public_counters(reasoner_handoff)
    public_consumer_counters = {
        **{f"projector.{name}": value for name, value in projector_public_counters.items()},
        **{f"reasoner.{name}": value for name, value in reasoner_public_counters.items()},
    }
    materialized_scalar_rows = _sum_matching_counters(
        public_consumer_counters,
        fragments=("materialized_scalar_rows", "model_rows_materialized"),
    )
    copied_structural_bytes = _sum_matching_counters(
        public_consumer_counters,
        fragments=("copy_bytes", "copied_bytes"),
    )
    projector_counter_evidence = _consumer_counter_evidence(
        consumer="projector",
        handoff=first_projector_handoff,
    )
    if reasoner_name in {"elk", "hermit"}:
        inferred_reasoner_selected = True
        reasoner_counter_evidence = _consumer_counter_evidence(
            consumer=reasoner_name,
            handoff=reasoner_handoff,
        )
    else:
        inferred_reasoner_selected = False
        reasoner_counter_evidence = None
    all_measured_consumers_encoded = first_projector_handoff["ingestion_path"] == (
        "encoded-native"
    ) and (
        not inferred_reasoner_selected
        or (
            reasoner_handoff is not None
            and reasoner_handoff.get("ingestion_path") == "encoded-native"
        )
    )
    complete_public_counter_coverage = projector_counter_evidence[
        "complete_public_counter_coverage"
    ] is True and (
        reasoner_counter_evidence is None
        or reasoner_counter_evidence["complete_public_counter_coverage"] is True
    )
    all_counter_evidence_ready = projector_counter_evidence["acceptance_ready"] is True and (
        reasoner_counter_evidence is None or reasoner_counter_evidence["acceptance_ready"] is True
    )
    expected_consumer_operations = {name: 0 for name in consumer_operations}
    if reasoner_worker_wire and inferred_reasoner_selected:
        expected_consumer_operations["encode_snapshot"] = 1
    unexpected_core_operations = {
        name: {"expected": expected_consumer_operations[name], "actual": value}
        for name, value in consumer_operations.items()
        if value != expected_consumer_operations[name]
    }
    if require_encoded_consumers:
        _require_consumer_counter_evidence(projector_counter_evidence)
        if reasoner_counter_evidence is not None:
            _require_consumer_counter_evidence(reasoner_counter_evidence)
        if unexpected_core_operations:
            raise RuntimeError(
                "encoded consumer handoff performed unexpected core operations: "
                f"{unexpected_core_operations!r}"
            )
    return {
        "input": {
            "name": path.name,
            "bytes": input_bytes,
            "sha256": input_sha256,
        },
        "load_calls": all_operations["load_snapshot"],
        "load_seconds": load_seconds,
        "load_cpu_seconds": load_cpu_seconds,
        "load_backend": load_backend,
        "load_report": {
            "source_bytes": snapshot.report.total_source_bytes,
            "documents": snapshot.report.document_count,
            "axioms": snapshot.report.effective_axiom_count,
            "acquisition_cache_hits": snapshot.report.acquisition_cache_hits,
            "document_cache_hits": snapshot.report.document_cache_hits,
            "resolution_attempts": snapshot.report.resolution_attempts,
            "timings": dict(snapshot.report.timings),
        },
        "fingerprints": initial_fingerprints,
        "identity": identity,
        "core_operations": {
            "total": all_operations,
            "consumer_delta": consumer_operations,
            "projection_delta": projection_operation_delta,
            "projection_cache_delta": cache_operation_delta,
            "reasoner_delta": reasoner_operation_delta,
        },
        "consumer_handoff": stack_provenance["consumer_handoff"],
        "materialization_and_copy": {
            "public_counters": {
                "projector": projector_public_counters,
                "reasoner": reasoner_public_counters,
            },
            "materialized_scalar_rows": materialized_scalar_rows,
            "copied_structural_bytes": copied_structural_bytes,
            "core_wire_delta": {
                name: consumer_operations[name]
                for name in ("encode_snapshot", "decode_snapshot", "open_snapshot")
            },
            "core_parser_entry_delta": consumer_operations["load_snapshot"],
            "complete_public_counter_coverage": complete_public_counter_coverage,
            "acceptance_evidence": {
                "all_measured_consumers_encoded": all_measured_consumers_encoded,
                "all_consumer_counter_evidence_ready": all_counter_evidence_ready,
                "expected_core_operation_calls": expected_consumer_operations,
                "unexpected_core_operation_calls": unexpected_core_operations,
                "projector": projector_counter_evidence,
                "reasoner": reasoner_counter_evidence,
                "acceptance_ready": (
                    all_measured_consumers_encoded
                    and complete_public_counter_coverage
                    and all_counter_evidence_ready
                    and not unexpected_core_operations
                ),
            },
        },
        "projection": {
            "requested_backend": projector_backend,
            "include_literals": include_literals,
            "edges": edge_count,
            "result_sha256": result_sha256,
            "wall_seconds": projection_seconds,
            "cpu_seconds": projection_cpu_seconds,
            "time_to_first_edge_seconds": first_edge_seconds,
            "encoded_view_publication_seconds": (first_ingestion.encoded_view_publication_seconds),
            "consumer_compile_seconds": first_ingestion.consumer_compile_seconds,
            "publication_compile_timing_note": (
                None
                if first_ingestion.consumer_compile_seconds is not None
                and (
                    first_ingestion.path != "encoded-native"
                    or first_ingestion.encoded_view_publication_seconds is not None
                )
                else "included in time_to_first_edge_seconds because the public consumer "
                "report omitted a phase timing"
            ),
            "edges_per_second": (
                edge_count / projection_seconds if projection_seconds > 0.0 else None
            ),
            "spill": spill,
            "consumer": first_projector_handoff,
            "encoded_counters": first_encoded_counters,
            "provenance": first_projection_provenance,
        },
        "projection_cache": {
            "edges": first_cache_count,
            "result_sha256": first_cache_sha256,
            "fill_seconds": cache_fill_seconds,
            "hit_seconds": cache_hit_seconds,
            "hit_faster_than_fill": cache_hit_seconds <= cache_fill_seconds,
        },
        "reasoner": {
            "measured": reasoner_name is not None,
            "requested": reasoner_name,
            "requested_backend": reasoner_backend if reasoner_name is not None else None,
            "worker_wire": reasoner_worker_wire if reasoner_name is not None else False,
            "configuration": reasoner_configuration,
            "encoded_view_publication_seconds": (
                reasoner_handoff.get("encoded_view_publication_seconds")
                if reasoner_handoff is not None
                else None
            ),
            "consumer_compile_seconds": (
                reasoner_handoff.get("consumer_compile_seconds")
                if reasoner_handoff is not None
                else None
            ),
            "results": reasoner_results,
            "consumer": reasoner_handoff,
            "provenance": reasoner_provenance,
        },
        "rss": {
            "before_bytes": before_rss,
            "peak_bytes": after_rss,
            "incremental_peak_bytes": max(0, after_rss - before_rss),
        },
        "wall_seconds": time.perf_counter() - whole_started,
        "cpu_seconds": _cpu_seconds() - whole_cpu_started,
        "second_ontology_representation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ontology", type=Path, nargs="+")
    parser.add_argument("--buffer-edges", type=int, default=250_000)
    parser.add_argument("--include-literals", action="store_true")
    parser.add_argument(
        "--load-backend",
        choices=("auto", "python", "native"),
        default="python",
    )
    parser.add_argument(
        "--projector-backend",
        choices=("auto", "python", "native"),
        default="python",
    )
    parser.add_argument(
        "--reasoner",
        choices=("asserted", "elk", "hermit"),
        help="measure a complete direct-parent result in addition to projection",
    )
    parser.add_argument("--reasoner-backend", default="auto")
    parser.add_argument("--reasoner-workers", type=int, default=0)
    parser.add_argument("--reasoner-timeout-seconds", type=float)
    parser.add_argument("--reasoner-worker-wire", action="store_true")
    parser.add_argument(
        "--require-encoded-consumers",
        action="store_true",
        help="fail unless every selected native consumer reports encoded-native ingestion",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        "schema_version": 4,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                "exact-om": _package_version("Exact-OM", "source-checkout"),
                "pyowl-core": _package_version("pyowl-core", str(pyowl_core.__version__)),
                "pyowl2vec-star-projector": _package_version(
                    "pyowl2vec-star-projector", "source-checkout"
                ),
                "pyelk-reasoner": _package_version("pyelk-reasoner", "not-installed"),
                "pyhermit": _package_version("pyHermiT", "not-installed"),
            },
        },
        "configuration": {
            "buffer_edges": args.buffer_edges,
            "include_literals": args.include_literals,
            "load_backend": args.load_backend,
            "projector_backend": args.projector_backend,
            "reasoner": args.reasoner,
            "reasoner_backend": args.reasoner_backend if args.reasoner is not None else None,
            "reasoner_workers": args.reasoner_workers if args.reasoner is not None else None,
            "reasoner_timeout_seconds": (
                args.reasoner_timeout_seconds if args.reasoner is not None else None
            ),
            "reasoner_worker_wire": (
                args.reasoner_worker_wire if args.reasoner is not None else False
            ),
            "require_encoded_consumers": args.require_encoded_consumers,
            "cache_state": "cold-load; projection cache fill then hit",
        },
        "measurements": [
            measure(
                path.expanduser().resolve(),
                buffer_edges=args.buffer_edges,
                include_literals=args.include_literals,
                load_backend=cast(LoadBackend, args.load_backend),
                projector_backend=cast(ProjectorBackend, args.projector_backend),
                reasoner_name=cast(ReasonerName | None, args.reasoner),
                reasoner_backend=args.reasoner_backend,
                reasoner_workers=args.reasoner_workers,
                reasoner_timeout_seconds=args.reasoner_timeout_seconds,
                reasoner_worker_wire=args.reasoner_worker_wire,
                require_encoded_consumers=args.require_encoded_consumers,
            )
            for path in args.ontology
        ],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
