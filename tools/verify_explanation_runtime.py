"""Measure a fresh Exact CLI server over loopback; no parsing, matching or generation.

This complements verify-backend's contract/fixture export with actual interpreter,
framework, package-admission and HTTP startup costs. The child is always stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

_RESPONSE_LIMIT = 2 * 1024**2
_FORBIDDEN_IMPORTS = (
    "torch",
    "transformers",
    "sentence_transformers",
    "pyowl_core",
    "pyowl2vec_star_projector",
    "exact.llm",
    "openai",
    "litellm",
)


def _get(client, route, params=None):
    started = time.perf_counter()
    body = bytearray()
    status = None
    error = None
    try:
        with client.stream("GET", route, params=params) as response:
            status = response.status_code
            for chunk in response.iter_bytes(chunk_size=64 * 1024):
                body.extend(chunk)
                if len(body) > _RESPONSE_LIMIT:
                    break
    except httpx.HTTPError as exc:
        error = type(exc).__name__
    return {
        "route": route,
        "query": params or {},
        "seconds": time.perf_counter() - started,
        "bytes": len(body),
        "status": status,
        "error": error,
        "within_response_budget": len(body) <= _RESPONSE_LIMIT,
    }, bytes(body)


def _json_get(client, route, params=None):
    sample, body = _get(client, route, params)
    if sample["status"] != 200 or sample["error"] or not sample["within_response_budget"]:
        raise RuntimeError("A discovery request failed or exceeded the response budget")
    return sample, json.loads(body)


def _queries(client, manifest, package_root):
    """Prefer frozen case entities; ontology-first pages are an explicit fallback."""
    refs = {}
    origins = {}
    inspected = 0
    profile_count = 0
    skipped = 0
    run_queries = []

    def include(ref, origin):
        if ref.get("ontology_version_id") not in manifest.get("ontologies", {}):
            return
        key = tuple(ref[name] for name in ("ontology_version_id", "iri", "kind"))
        if key not in refs and len(refs) >= 32:
            return
        refs[key] = {name: ref[name] for name in ("ontology_version_id", "iri", "kind")}
        origins.setdefault(key, set()).add(origin)

    # These small immutable resources are read after measured CLI admission.
    for _, locator in sorted(manifest.get("explanations", {}).items())[:64]:
        path = package_root / locator
        if path.stat().st_size > _RESPONSE_LIMIT:
            skipped += 1
            continue
        resource = json.loads(path.read_bytes())
        inspected += 1
        profile_count += int(resource.get("task") == "entity_profile")
        for ref in resource.get("entities", []):
            include(ref, "prepared_" + str(resource.get("task", "explanation")))
    for run_id in sorted(manifest.get("runs", {})):
        base = "/api/v1/runs/" + quote(run_id, safe="")
        _, sources = _json_get(client, base + "/sources", {"limit": 20})
        run_queries.append((base + "/sources", {"limit": 20}))
        pair_details = 0
        for source in sources["items"]:
            ref = source["entity"]
            include(ref, "saved_run_source")
            params = {"source": ref["iri"], "source_kind": ref["kind"], "limit": 3}
            run_queries.append((base + "/candidates", params))
            _, candidates = _json_get(client, base + "/candidates", params)
            for pair in candidates["items"]:
                include(pair["source"], "saved_candidate_source")
                include(pair["target"], "saved_candidate_target")
                if pair_details < 3:
                    run_queries.append((base + "/pair", {"pair_id": pair["pair_id"]}))
                    run_queries.append(
                        (base + "/pair-evidence", {"pair_id": pair["pair_id"], "limit": 20})
                    )
                    pair_details += 1
    queries = [("/api/v1/health", {})]
    first_entities = []
    for ontology_id in sorted(manifest.get("ontologies", {})):
        representative = next(
            (ref for ref in refs.values() if ref["ontology_version_id"] == ontology_id), None
        )
        if representative is not None:
            sample, context = _json_get(client, "/api/v1/entity-context", representative)
            label = context.get("preferred_label", {}).get("value")
        else:
            sample, page = _json_get(
                client,
                "/api/v1/entities",
                {"ontology_version_id": ontology_id, "kind": "class", "limit": 5},
            )
            for row in page["items"]:
                include(row["entity"], "ontology_first_page_fallback")
            label = next(
                (
                    row.get("preferred_label", {}).get("value")
                    for row in page["items"]
                    if row.get("preferred_label", {}).get("value")
                ),
                None,
            )
        first_entities.append(sample)
        queries.append(("/api/v1/entities", {"ontology_version_id": ontology_id, "limit": 20}))
        if label:
            queries.append(
                (
                    "/api/v1/entities",
                    {"ontology_version_id": ontology_id, "term": label[:32], "limit": 20},
                )
            )
    for ref in refs.values():
        queries.extend(
            (route, ref)
            for route in ("/api/v1/entity-context", "/api/v1/entity-facts", "/api/v1/hierarchy")
        )
    queries.extend(run_queries)
    for explanation_id in sorted(manifest.get("explanations", {}))[:1]:
        queries.append(("/api/v1/explanations/" + quote(explanation_id, safe=""), {}))
    selection = {
        "method": "prepared_explanation_entities_then_saved_run_entities_then_ontology_fallback",
        "entity_limit": 32,
        "explanation_inspection_limit": 64,
        "run_source_limit": 20,
        "candidates_per_source_limit": 3,
        "pair_details_per_run_limit": 3,
        "explanations_inspected": inspected,
        "profiles_inspected": profile_count,
        "oversized_explanations_skipped": skipped,
        "entities": [{"entity": ref, "origins": sorted(origins[key])} for key, ref in refs.items()],
        "sampling": "round_robin; uniform plan-index stride when the plan exceeds 100 reads",
    }
    return queries, first_entities, selection


def _process_memory(pid):
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


def _percentile(values):
    return sorted(values)[min(len(values) - 1, int(len(values) * 0.95))] if values else None


def _hardware():
    result = {
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "cpu": platform.processor(),
    }
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                result["memory_bytes"] = int(line.split()[1]) * 1024
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                result["cpu"] = line.partition(":")[2].strip()
                break
    return result


def verify_runtime(
    package: Path,
    output: Path,
    *,
    profile: str = "local_app",
    hardware_profile: str = "detected",
    storage_profile: str = "unspecified",
    startup_timeout: float = 600,
    request_timeout: float = 30,
) -> dict[str, Any]:
    """Run exactly 100 fixed mixed HTTP reads with four readers in a fresh CLI process."""
    if profile not in {"local_app", "public_demo"}:
        raise ValueError("Only read-only operational profiles are supported")
    if not 0 < startup_timeout <= 1800 or not 0 < request_timeout <= 120:
        raise ValueError("Startup/request timeouts must be positive and bounded")
    package, output = Path(package).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "server.log"
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [
        sys.executable,
        "-X",
        "importtime",
        "-m",
        "exact_inspect.cli",
        "serve",
        "--package",
        str(package),
        "--profile",
        profile,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "WARNING",
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    process = None
    startup = None
    memory = None
    manifest = {}
    manifest_hash = None
    queries = []
    first_entities = []
    selection = {}
    samples = []
    failure = None
    termination = "not_started"
    phase = "startup"
    with log_path.open("wb") as log:
        started = time.perf_counter()
        try:
            process = subprocess.Popen(command, stdout=log, stderr=log, env=environment)
            base_url = f"http://127.0.0.1:{port}"
            with httpx.Client(base_url=base_url, timeout=0.25, trust_env=False) as polling:
                while time.perf_counter() - started < startup_timeout:
                    if process.poll() is not None:
                        raise RuntimeError("CLI server exited before health")
                    sample, body = _get(polling, "/api/v1/health")
                    if sample["status"] == 200 and not sample["error"]:
                        health = json.loads(body)
                        if health.get("status") == "available":
                            startup = time.perf_counter() - started
                            break
                    time.sleep(0.05)
                else:
                    raise TimeoutError("CLI server did not become ready before the timeout")
            # Read metadata only after the measured child has admitted all package bytes.
            data = package.read_bytes()
            manifest_hash = "sha256:" + hashlib.sha256(data).hexdigest()
            manifest = json.loads(data)
            if health["package_id"] != manifest["package_id"]:
                raise RuntimeError("Loopback service returned another package")
            phase = "read_discovery"
            with httpx.Client(
                base_url=base_url,
                timeout=request_timeout,
                trust_env=False,
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
            ) as client:
                queries, first_entities, selection = _queries(client, manifest, package.parent)
                phase = "mixed_reads"

                def measure(index):
                    plan_index = (
                        index % len(queries) if len(queries) <= 100 else index * len(queries) // 100
                    )
                    route, params = queries[plan_index]
                    sample, _ = _get(client, route, params)
                    return {"index": index, "plan_index": plan_index, **sample}

                with ThreadPoolExecutor(max_workers=4) as pool:
                    samples = list(pool.map(measure, range(100)))
        except Exception as exc:
            failure = {"phase": phase, "type": type(exc).__name__, "message": str(exc)}
        finally:
            if process is not None:
                memory = _process_memory(process.pid)
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                        termination = "terminated"
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                        termination = "killed_after_shutdown_timeout"
                else:
                    process.wait()
                    termination = "already_exited"
    imports = sorted(
        {
            line.rpartition("|")[2].strip()
            for line in log_path.read_text(errors="replace").splitlines()
            if line.startswith("import time:") and "imported package" not in line
        }
    )
    forbidden = [
        name
        for name in imports
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _FORBIDDEN_IMPORTS)
    ]
    context_p95 = _percentile([s["seconds"] for s in samples if "/runs/" not in s["route"]])
    pair_p95 = _percentile([s["seconds"] for s in samples if "/runs/" in s["route"]])
    checks = {
        "cold_service_health": "passed" if startup is not None and startup <= 10 else "failed",
        "cold_entity_queries": (
            "passed"
            if first_entities and max(s["seconds"] for s in first_entities) <= 2
            else "failed" if first_entities else "not_run"
        ),
        "bounded_reads": (
            "passed"
            if len(samples) == 100
            and all(
                s["status"] == 200 and not s["error"] and s["within_response_budget"]
                for s in samples
            )
            else "failed"
        ),
        "warm_read_p95": (
            "passed"
            if context_p95 is not None
            and context_p95 <= 0.5
            and (pair_p95 is None or pair_p95 <= 1)
            else "failed"
        ),
        "serving_memory": (
            "passed"
            if memory is not None and memory <= 4 * 1024**3
            else "failed" if memory is not None else "not_run"
        ),
        "no_runtime_imports": "failed" if forbidden else "passed" if imports else "not_run",
    }
    report = {
        "schema_version": "exact-explain-http-runtime/1",
        "started_at": started_at,
        "status": (
            "passed" if not failure and all(v == "passed" for v in checks.values()) else "failed"
        ),
        "package_id": manifest.get("package_id"),
        "package_manifest_sha256": manifest_hash,
        "profile": profile,
        "hardware_profile": hardware_profile,
        "storage_profile": storage_profile,
        "command": command,
        "command_shell": shlex.join(command),
        "checks": checks,
        "measurements": {
            "query_count": len(samples),
            "readers": 4,
            "cold_health_seconds": startup,
            "cold_entity_queries": first_entities,
            "context_p95_seconds": context_p95,
            "pair_p95_seconds": pair_p95,
            "peak_rss_bytes": memory,
            "memory_source": "child /proc/PID/status VmHWM",
        },
        "runtime": {
            "python": platform.python_version(),
            "hardware": _hardware(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("uvicorn", "fastapi", "httpx", "pydantic")
            },
        },
        "import_observation": {
            "method": "CPython -X importtime stderr",
            "module_count": len(imports),
            "forbidden_prefixes": list(_FORBIDDEN_IMPORTS),
            "forbidden_modules": forbidden,
            "provider_calls": "not_directly_instrumented",
            "gpu_allocation": "not_directly_instrumented",
        },
        "query_plan_selection": selection,
        "query_plan": [{"route": route, "query": params} for route, params in queries],
        "samples": samples,
        "failure": failure,
        "child": {
            "pid": process.pid if process else None,
            "returncode": process.returncode if process else None,
            "termination": termination,
            "host": "127.0.0.1",
            "port": port,
        },
        "scope": "Fresh CLI interpreter and real loopback HTTP, including framework imports, full artifact admission, and import-audit overhead. OS page cache is not flushed; startup health polling adds up to one polling interval. Memory belongs to the child, not the preparing/test process. No matching, ontology parsing, model loading or generation is requested. Network provider calls and GPU allocations are not independently instrumented; import observations are reported separately.",
    }
    from exact_inspect.artifacts import atomic_json

    atomic_json(output / "http-runtime.json", report)
    return report


def main() -> None:
    """Write an operational receipt without publishing or mutating the input package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=("local_app", "public_demo"), default="local_app")
    parser.add_argument("--hardware-profile", default="detected")
    parser.add_argument("--storage-profile", default="unspecified")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=30)
    args = parser.parse_args()
    report = verify_runtime(
        args.package,
        args.output,
        profile=args.profile,
        hardware_profile=args.hardware_profile,
        storage_profile=args.storage_profile,
        startup_timeout=args.startup_timeout,
        request_timeout=args.request_timeout,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": report["checks"],
                "receipt": str(args.output / "http-runtime.json"),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
