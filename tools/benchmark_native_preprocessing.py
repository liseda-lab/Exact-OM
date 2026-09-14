#!/usr/bin/env python3
"""Measure native OWL preprocessing only; never read mappings or instantiate models.

Use a resolved Exact config and run each ontology in a fresh process. Phase times
are incremental in the recorded order; warm feature lookup is not cold throughput.
An external timeout may stop native code; the report retains the last started phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(rows: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _rss() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"process_peak_rss_bytes": int(usage * (1 if sys.platform == "darwin" else 1024))}


def _write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def _profile_boundaries(path: Path | None) -> Iterator[None]:
    """Time diagnostic boundaries without changing arguments, results or iteration."""
    if path is None:
        yield
        return
    import pyowl2vec_star_projector as shared_projector

    import exact.ontology.native_projection as native
    import exact.ontology.projection as projection

    boundaries = (
        (projection.SharedProjectionAdapter, "edges", "adapter_total", False),
        (projection, "cache_key", "cache_key_and_fingerprints", False),
        (native.NativeProjector, "project", "native_projection_total", False),
        (native.NativeProjector, "project_taxonomy", "native_projection_total", False),
        (shared_projector.Projector, "iter_edges", "native_projection_iterator", True),
        (
            shared_projector.Projector,
            "iter_taxonomy_edges",
            "native_projection_iterator",
            True,
        ),
        (projection, "require_native_report", "adapter_report_validation", False),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    originals: list[tuple[Any, str, Any, bool]] = []
    with path.open("x", encoding="utf-8") as stream:
        started = time.perf_counter()
        stack: list[int] = []
        calls = 0

        def emit(row: dict[str, Any]) -> None:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        @contextmanager
        def span(name: str) -> Iterator[None]:
            nonlocal calls
            calls += 1
            call = calls
            wall, cpu = time.perf_counter(), time.process_time()
            row = {
                "stage": name,
                "call": call,
                "parent_call": stack[-1] if stack else None,
                "elapsed_start_seconds": wall - started,
            }
            emit({**row, "status": "running"})
            stack.append(call)
            status = "failed"
            try:
                yield
                status = "complete"
            finally:
                stack.pop()
                emit(
                    {
                        **row,
                        "status": status,
                        "wall_seconds": time.perf_counter() - wall,
                        "cpu_seconds": time.process_time() - cpu,
                    }
                )

        def instrument(original: Callable, name: str, iterable: bool) -> Callable:
            @wraps(original)
            def call(*args, **kwargs):
                with span(name):
                    return original(*args, **kwargs)

            @wraps(original)
            def iterate(*args, **kwargs):
                with span(name):
                    yield from original(*args, **kwargs)

            return iterate if iterable else call

        try:
            for owner, attribute, name, iterable in boundaries:
                original = getattr(owner, attribute)
                originals.append((owner, attribute, original, attribute in vars(owner)))
                setattr(owner, attribute, instrument(original, name, iterable))
            yield
        finally:
            for owner, attribute, original, owned in reversed(originals):
                if owned:
                    setattr(owner, attribute, original)
                else:
                    delattr(owner, attribute)


def benchmark(
    config_path: Path,
    output: Path,
    *,
    side: str = "source",
    entity_limit: int = 64,
    entities_path: Path | None = None,
    profile_stages: Path | None = None,
) -> dict[str, Any]:
    """Record model-free costs and semantic digests using the actual Exact facade."""
    started = time.perf_counter()
    if side not in {"source", "target"} or entity_limit < 1:
        raise ValueError("side must be source/target and entity_limit must be positive")
    feature_rows_path = output.with_suffix(".features.jsonl")
    for evidence_path in (output, feature_rows_path):
        if evidence_path.exists():
            raise FileExistsError(f"Refusing to overwrite benchmark evidence: {evidence_path}")
    if profile_stages is not None and profile_stages.exists():
        raise FileExistsError(f"Refusing to overwrite profiling evidence: {profile_stages}")
    config = yaml.safe_load(config_path.read_text())
    parameters = dict(config["dataset"])
    parameters.update(parameters.pop("legacy", {}) or {})
    if parameters.get("reasoner", "asserted") != "asserted":
        raise ValueError("This preprocessing benchmark requires the asserted reasoner")
    projector = parameters.get("projector", {})
    if projector.get("backend", "auto") not in {"auto", "native"}:
        raise ValueError("This benchmark requires a native projector")
    ontology = Path(config["data"][side]).expanduser().resolve()
    options = config.get("io", {}).get(f"{side}_options", {})
    imports = {}
    for iri, binding in options.get("imports", {}).items():
        imported = Path(binding["path"]).expanduser()
        imported = imported if imported.is_absolute() else ontology.parent / imported
        actual = _sha256(imported)
        if actual != binding["sha256"]:
            raise ValueError(f"Import checksum mismatch: {iri}")
        imports[iri] = {"path": str(imported.resolve()), "sha256": actual}
    output.parent.mkdir(parents=True, exist_ok=True)
    implementation = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "pid": os.getpid(),
        "host": platform.node(),
        "python": platform.python_version(),
        "config_sha256": _sha256(config_path),
        "side": side,
        "input": {"path": str(ontology), "sha256": _sha256(ontology), "imports": imports},
        "implementation": {
            name: _sha256(implementation / name)
            for name in (
                "exact/ontology/store.py",
                "exact/ontology/projection.py",
                "exact/ontology/native_projection.py",
                "exact/ontology/view_contract.py",
                "exact/ontology/parser.py",
                "exact/core/entities/ontology.py",
                "exact/impl/datasets/pair_adaptive_context.py",
                "tools/benchmark_native_preprocessing.py",
            )
        },
        "settings": {
            "dataset": parameters,
            "source_options": options,
            "requested_projector_backend": projector.get("backend", "auto"),
            "required_projector_backend": "native",
            "entity_limit": entity_limit,
            "entity_selection": "explicit_IRIs" if entities_path else "sha256(seed,kind,IRI)",
        },
        "scope": "Single-ontology preprocessing; no reference, candidate, encoder or LLM calls",
        "boundary_profile": {
            "path": str(profile_stages) if profile_stages else None,
            "enabled": profile_stages is not None,
            "scope": "First projection only; nested inclusive spans; boundary logging overhead included",
            "adapter_remainder": "Includes row conversion, cache publication and other uninstrumented adapter work",
        },
        "timing_semantics": "Sequential incremental phases; RSS is cumulative process high-water mark",
        "phases": [],
    }

    def phase(name: str, action: Callable[[], Any]) -> Any:
        row: dict[str, Any] = {
            "name": name,
            "status": "running",
            "started_elapsed_seconds": time.perf_counter() - started,
            **_rss(),
        }
        report["phases"].append(row)
        _write(output, report)
        tick = time.perf_counter()
        try:
            result = action()
        except BaseException:
            row.update(status="failed", wall_seconds=time.perf_counter() - tick, **_rss())
            raise
        row.update(status="complete", wall_seconds=time.perf_counter() - tick, **_rss())
        report["elapsed_seconds"] = time.perf_counter() - started
        _write(output, report)
        print(f"{name}: {row['wall_seconds']:.3f}s", flush=True)
        return result

    try:
        from exact.core.entities.kinds import EntityKind
        from exact.core.entities.ontology import OntologyGraph
        from exact.io.sources import resolve

        source = phase("load", lambda: resolve(ontology, format="owl", options=options))
        snapshot = source.owl_snapshot()
        report["load_report"] = {
            "backend": snapshot.report.backend,
            "complete": snapshot.is_complete,
            "effective_axiom_count": snapshot.report.effective_axiom_count,
            "total_source_bytes": snapshot.report.total_source_bytes,
            "diagnostics": [item.to_dict() for item in snapshot.report.diagnostics],
        }
        if snapshot.report.backend != "native" or not snapshot.is_complete:
            raise ValueError("Native complete ontology loading is required")
        source.configure_projector(
            backend="native", profile=projector.get("profile", "mowl-d993536-v1")
        )
        signatures = phase(
            "signature", lambda: {kind.value: list(source.entities(kind)) for kind in EntityKind}
        )
        report["signature"] = {
            "counts": {kind: len(rows) for kind, rows in signatures.items()},
            "sha256": _digest(sorted(signatures.items())),
        }
        include_literals = bool(parameters.get("projection_include_literals", False))
        with _profile_boundaries(profile_stages):
            edges = phase(
                "projection", lambda: source.projection_edges(include_literals=include_literals)
            )
        spill = getattr(getattr(source, "projector", None), "last_spill_metrics", None)
        report["projection_spill"] = asdict(spill) if spill is not None else None
        report["ontology_stack"] = phase("provenance", source.ontology_stack_provenance)
        if report["ontology_stack"]["projector"]["selection"]["effective"] != "native":
            raise ValueError("Projection did not use the native backend")
        _write(output, report)
        report["edges"] = {
            "count": len(edges),
            "sha256": phase(
                "edge_digest", lambda: _digest(sorted(edge.astuple() for edge in edges))
            ),
        }
        excluded = phase("exclusions", source.excluded_from_alignment)
        report["exclusions"] = {"count": len(excluded), "sha256": _digest(sorted(excluded))}
        kinds = config.get("matching", {}).get("entity_kinds", ["class"])
        population = sorted((kind, iri) for kind in kinds for iri in signatures[kind])
        report["labels"] = {
            "entities": len(population),
            "sha256": phase(
                "labels",
                lambda: _digest((kind, iri, source.labels(iri)) for kind, iri in population),
            ),
        }
        graph = phase(
            "graph_wrapper", lambda: OntologyGraph(source, include_literals=include_literals)
        )
        report["graph"] = {
            "nodes": len(graph.graph),
            "edges": len(graph.edges),
            "cached_labels": len(graph.label_cache),
        }
        eligible = [
            (kind, iri) for kind, iri in population if kind != "class" or iri not in excluded
        ]
        if entities_path is not None:
            selected = [
                line.strip() for line in entities_path.read_text().splitlines() if line.strip()
            ]
            if len(set(selected)) != len(selected) or len(selected) > entity_limit:
                raise ValueError("Explicit entity list contains duplicates or exceeds entity_limit")
            eligible_by_iri = {iri: kind for kind, iri in eligible}
            if not selected or set(selected) - eligible_by_iri.keys():
                raise ValueError("Explicit entities must be eligible members of this ontology")
            entities = [(eligible_by_iri[iri], iri) for iri in selected]
        else:
            seed = config.get("run", {}).get("seed", 17)
            entities = sorted(
                eligible,
                key=lambda item: hashlib.sha256(f"{seed}\0{item[0]}\0{item[1]}".encode()).digest(),
            )[:entity_limit]
        report["entities"] = {
            "count": len(entities),
            "rows": [list(item) for item in entities],
            "sha256": _digest(entities),
        }

        def make_dataset():
            from exact.impl.datasets.pair_adaptive_context import (
                PairAdaptiveContextDataset,
            )

            settings = {
                **parameters,
                "projector": {**projector, "backend": "native"},
                "verbalization_mode": "deterministic",
                "verbaliser_name": None,
                "device": "cpu",
                "num_workers": 0,
            }
            dataset = PairAdaptiveContextDataset(
                output_path=output.parent / (output.stem + ".scratch"), **settings
            )
            dataset._source, dataset._source_graph = source, graph
            return dataset

        dataset = phase("feature_wrapper", make_dataset)
        features = phase(
            "entity_features",
            lambda: [
                (kind, iri, dataset.get_entity_features(iri, "src", kind)) for kind, iri in entities
            ],
        )
        report["features"] = {"count": len(features), "sha256": _digest(features)}
        # Retain the exact hashed rows so a mismatch is diagnosable without another
        # ontology load. List order is semantic here; do not normalize it away.
        with feature_rows_path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in features:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        report["feature_rows"] = {"path": str(feature_rows_path), **report["features"]}
        warm = phase(
            "entity_features_cached",
            lambda: [
                (kind, iri, dataset.get_entity_features(iri, "src", kind)) for kind, iri in entities
            ],
        )
        if _digest(warm) != report["features"]["sha256"]:
            raise ValueError("Cached entity features changed semantics")
        # Observe only indexes already demanded by these features: reporting must
        # never construct an additional ontology-scale index or alter timings.
        report["native_indexes"] = {
            name.removeprefix("_"): dict(view.native_report)
            for name in (
                "_axioms",
                "_class_view",
                "_property_view",
                "_domain_range",
            )
            if (view := vars(source).get(name)) is not None
        }
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter() - started, **_rss())
        _write(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", choices=("source", "target"), default="source")
    parser.add_argument("--entity-limit", type=int, default=64)
    parser.add_argument("--entities", type=Path, help="Optional frozen entity IRIs, one per line")
    parser.add_argument(
        "--profile-stages", type=Path, help="Optional new JSONL file for native boundary timings"
    )
    args = parser.parse_args()
    benchmark(
        args.config,
        args.output,
        side=args.side,
        entity_limit=args.entity_limit,
        entities_path=args.entities,
        profile_stages=args.profile_stages,
    )


if __name__ == "__main__":
    main()
