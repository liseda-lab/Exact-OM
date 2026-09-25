"""Reproducible CPU-serving measurements and honest backend handoff evidence."""

from __future__ import annotations

import importlib.metadata
import platform
import resource
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, validate_bundle
from .contracts import CONTRACT_VERSION


def verify_backend(package: Path, output: Path) -> dict[str, Any]:
    """Verify the prepared package and measure 100 bounded reads without running experiments."""
    from fastapi.testclient import TestClient

    from .service import create_prepared_app

    start = time.perf_counter()
    app = create_prepared_app(package)
    client = TestClient(app)
    health = client.get("/api/v1/health")
    startup = time.perf_counter() - start
    health.raise_for_status()
    # Service admission above hashes every artifact before health is measured.
    # A separate integrity report must not warm those reads beforehand.
    integrity_started = time.perf_counter()
    manifest = validate_bundle(package)
    integrity_seconds = time.perf_counter() - integrity_started
    queries: list[tuple[str, dict[str, Any]]] = [("/api/v1/health", {})]
    first_queries = []
    for ontology_id in manifest.ontologies:
        first = time.perf_counter()
        response = client.get(
            "/api/v1/entities", params={"ontology_version_id": ontology_id, "limit": 5}
        )
        first_queries.append(time.perf_counter() - first)
        response.raise_for_status()
        queries.append(("/api/v1/entities", {"ontology_version_id": ontology_id, "limit": 20}))
        for entity in response.json()["items"]:
            ref = entity.get("entity", entity)
            params = {k: ref[k] for k in ("ontology_version_id", "iri", "kind")}
            queries.extend(
                (route, params)
                for route in ("/api/v1/entity-context", "/api/v1/entity-facts", "/api/v1/hierarchy")
            )
    for run_id in manifest.runs:
        response = client.get(f"/api/v1/runs/{run_id}/sources", params={"limit": 1})
        response.raise_for_status()
        queries.append((f"/api/v1/runs/{run_id}/sources", {"limit": 20}))
        for source in response.json()["items"]:
            ref = source.get("entity", source.get("source", source))
            iri = ref.get("iri", source.get("source_iri"))
            if iri:
                params = {"source": iri, "limit": 20}
                queries.append((f"/api/v1/runs/{run_id}/candidates", params))
                candidates = client.get(f"/api/v1/runs/{run_id}/candidates", params=params)
                candidates.raise_for_status()
                for pair in candidates.json()["items"][:3]:
                    queries.append((f"/api/v1/runs/{run_id}/pair", {"pair_id": pair["pair_id"]}))

    def measure(index: int) -> dict[str, Any]:
        route, params = queries[index % len(queries)]
        start = time.perf_counter()
        response = client.get(route, params=params)
        return {
            "route": route,
            "seconds": time.perf_counter() - start,
            "bytes": len(response.content),
            "status": response.status_code,
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        samples = list(pool.map(measure, range(100)))
    reads = [sample["seconds"] for sample in samples]
    p95 = statistics.quantiles(reads, n=100)[94]
    context_reads = [s["seconds"] for s in samples if "/runs/" not in s["route"]]
    pair_reads = [s["seconds"] for s in samples if "/runs/" in s["route"]]

    def percentile(values: list[float]) -> float | None:
        return sorted(values)[min(len(values) - 1, int(len(values) * 0.95))] if values else None

    context_p95, pair_p95 = percentile(context_reads), percentile(pair_reads)
    rss_bytes = _peak_rss_bytes()
    checks = {
        "package_integrity": "passed",
        "cold_service_health": "passed" if startup <= 10 else "failed",
        "cold_entity_queries": (
            "passed"
            if first_queries and max(first_queries) <= 2
            else "not_run" if not first_queries else "failed"
        ),
        "bounded_reads": (
            "passed"
            if all(s["status"] == 200 and s["bytes"] <= 2 * 1024**2 for s in samples)
            else "failed"
        ),
        "warm_read_p95": (
            "passed"
            if context_p95 is not None
            and context_p95 <= 0.5
            and (pair_p95 is None or pair_p95 <= 1)
            else "failed"
        ),
        "serving_memory": "passed" if rss_bytes <= 4 * 1024**3 else "failed",
    }
    report = {
        "schema_version": "exact-explain-verification/1",
        "package_id": manifest.package_id,
        "checks": checks,
        "measurements": {
            "query_count": len(samples),
            "readers": 4,
            "integrity_verification_seconds": integrity_seconds,
            "cold_health_seconds": startup,
            "context_p95_seconds": context_p95,
            "pair_p95_seconds": pair_p95,
            "cold_entity_seconds": first_queries,
            "warm_p95_seconds": p95,
            "peak_rss_kib": rss_bytes // 1024,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("fastapi", "pydantic", "pyowl-core")
            },
        },
        "samples": samples,
        "scope": "Local in-process TestClient; run in a fresh process for serving RSS. OS page cache is not flushed. Cold health includes package hash verification; the independent integrity recheck follows it. Fixed mixed reads exclude provider latency. Full native-data, fresh matcher, provider and PostgreSQL gates require separate receipts.",
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "verification.json", report)
    atomic_json(output / "openapi.json", app.openapi())
    from .preparation import ExecutionLock
    from .study.resources import ExplanationResource

    atomic_json(output / "execution-lock.schema.json", ExecutionLock.model_json_schema())
    atomic_json(output / "study-resource.schema.json", ExplanationResource.model_json_schema())
    fixture_routes = list(dict.fromkeys(route for route, _ in queries))
    for number, route in enumerate(fixture_routes):
        params = next(params for candidate, params in queries if candidate == route)
        response = client.get(route, params=params)
        atomic_json(
            output / "fixtures" / f"response-{number}.json",
            {
                "request": {"path": route, "query": params},
                "status": response.status_code,
                "response": response.json(),
            },
        )
    atomic_json(
        output / "backend-handoff.json",
        {
            "contract_version": CONTRACT_VERSION,
            "package_id": manifest.package_id,
            "visibility_policy_hash": manifest.policy.policy_hash,
            "frontend_admitted": False,
            "gates": {
                "G0": "not_run",
                "G1": "not_run",
                "G2": "not_run",
                "G3": "not_run",
                "G4": "not_run",
                "G5": "not_run",
            },
            "evidence": ["verification.json", "openapi.json"],
            "commands": {
                "serve": "exact-inspect serve --package PACKAGE/package.json --profile local_app",
                "demo": "exact-inspect serve --package PACKAGE/package.json --profile public_demo",
                "verify": "exact-inspect verify-backend --package PACKAGE/package.json --output VERIFICATION",
                "export": "exact-inspect export --package PACKAGE/package.json --output bundle.zip",
            },
            "capabilities": manifest.capabilities,
            "remaining_gate_evidence": [
                "Full pinned native ontology receipts",
                "Fresh pinned current-schema Exact run",
                "12-pair actual provider output and claim audit",
                "Artifact kill/repair and database restart/restore",
                "Acceptance owner review of all gate receipts",
            ],
        },
    )
    client.close()
    return report


def _peak_rss_bytes() -> int:
    # Linux getrusage can inherit the pre-exec parent high-water mark. VmHWM
    # measures this serving process, including its actual admission work.
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    # ru_maxrss is kilobytes on Linux but bytes on macOS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024
