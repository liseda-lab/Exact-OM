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
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

import pyowl_core  # noqa: E402
from pyowl2vec_star_projector import ProjectionOptions  # noqa: E402

from exact.ontology import load_ontology  # noqa: E402


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _fingerprints(snapshot: pyowl_core.OntologySnapshot) -> dict[str, str]:
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


def measure(path: Path, *, buffer_edges: int, include_literals: bool) -> dict[str, Any]:
    """Measure one source while asserting single-load and identity invariants."""

    print(f"[owl-stack-scale] hashing {path.name}", file=sys.stderr, flush=True)
    input_bytes = path.stat().st_size
    input_sha256 = _sha256_file(path)
    load_calls = 0
    original_load = pyowl_core.load_snapshot

    def counted_load(*args: object, **kwargs: object) -> pyowl_core.OntologySnapshot:
        nonlocal load_calls
        load_calls += 1
        loader: Any = original_load
        return cast(pyowl_core.OntologySnapshot, loader(*args, **kwargs))

    before_rss = _max_rss_bytes()
    print(f"[owl-stack-scale] loading {path.name}", file=sys.stderr, flush=True)
    started = time.perf_counter()
    pyowl_core.load_snapshot = counted_load  # type: ignore[assignment]
    try:
        source = load_ontology(
            path,
            options=pyowl_core.LoadOptions(backend=pyowl_core.BackendPreference.PYTHON),
        )
    finally:
        pyowl_core.load_snapshot = original_load
    load_seconds = time.perf_counter() - started
    snapshot = source.owl_snapshot()
    source.configure_projector(
        backend="python",
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
        backend="python",
    )
    projection_started = time.perf_counter()
    print(f"[owl-stack-scale] projecting {path.name}", file=sys.stderr, flush=True)
    iterator = source.projector.iter_edges(
        snapshot,
        options=options,
        buffer_edges=buffer_edges,
    )
    try:
        next(iterator)
    except StopIteration:
        first_edge_seconds: float | None = None
        edge_count = 0
    else:
        first_edge_seconds = time.perf_counter() - projection_started
        edge_count = 1 + sum(1 for _ in iterator)
    projection_seconds = time.perf_counter() - projection_started
    spill = asdict(source.projector.last_spill_metrics)

    cache_started = time.perf_counter()
    print(f"[owl-stack-scale] filling cache for {path.name}", file=sys.stderr, flush=True)
    first_cache_count = len(source.projection_edges(include_literals=include_literals))
    cache_fill_seconds = time.perf_counter() - cache_started
    cache_started = time.perf_counter()
    second_cache_count = len(source.projection_edges(include_literals=include_literals))
    cache_hit_seconds = time.perf_counter() - cache_started

    reasoner = source.reasoner
    final_fingerprints = _fingerprints(snapshot)
    final_axioms = snapshot.report.effective_axiom_count
    after_rss = _max_rss_bytes()
    identity = {
        "source_snapshot": source.owl_snapshot() is snapshot,
        "projector_snapshot": source.projector.last_view is snapshot,
        "reasoner_snapshot": getattr(reasoner, "ontology", None) is snapshot,
    }
    if load_calls != 1 or not all(identity.values()):
        raise RuntimeError(f"shared snapshot invariant failed: loads={load_calls}, {identity}")
    if initial_fingerprints != final_fingerprints or initial_axioms != final_axioms:
        raise RuntimeError("projection/reasoning mutated the shared snapshot")
    if first_cache_count != second_cache_count:
        raise RuntimeError("projection cache changed its result cardinality")

    return {
        "input": {
            "name": path.name,
            "bytes": input_bytes,
            "sha256": input_sha256,
        },
        "load_calls": load_calls,
        "load_seconds": load_seconds,
        "load_report": {
            "source_bytes": snapshot.report.total_source_bytes,
            "documents": snapshot.report.document_count,
            "axioms": snapshot.report.effective_axiom_count,
            "acquisition_cache_hits": snapshot.report.acquisition_cache_hits,
            "document_cache_hits": snapshot.report.document_cache_hits,
        },
        "fingerprints": initial_fingerprints,
        "identity": identity,
        "projection": {
            "include_literals": include_literals,
            "edges": edge_count,
            "wall_seconds": projection_seconds,
            "time_to_first_edge_seconds": first_edge_seconds,
            "edges_per_second": (
                edge_count / projection_seconds if projection_seconds > 0.0 else None
            ),
            "spill": spill,
        },
        "projection_cache": {
            "edges": first_cache_count,
            "fill_seconds": cache_fill_seconds,
            "hit_seconds": cache_hit_seconds,
            "hit_faster_than_fill": cache_hit_seconds <= cache_fill_seconds,
        },
        "rss": {
            "before_bytes": before_rss,
            "peak_bytes": after_rss,
            "incremental_peak_bytes": max(0, after_rss - before_rss),
        },
        "second_ontology_representation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ontology", type=Path, nargs="+")
    parser.add_argument("--buffer-edges", type=int, default=250_000)
    parser.add_argument("--include-literals", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        "schema_version": 2,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                "exact-om": _package_version("Exact-OM", "source-checkout"),
                "pyowl-core": _package_version("pyowl-core", str(pyowl_core.__version__)),
                "pyowl2vec-star-projector": _package_version(
                    "pyowl2vec-star-projector", "source-checkout"
                ),
            },
        },
        "measurements": [
            measure(
                path.expanduser().resolve(),
                buffer_edges=args.buffer_edges,
                include_literals=args.include_literals,
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
