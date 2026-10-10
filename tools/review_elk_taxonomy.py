"""Bounded offline ELK probe; never publish incomplete taxonomy as inference.

Use a policy-exported OWL resource and its frozen admission receipt. The parent
process enforces a wall-clock budget; parser memory is separately bounded. This
tool never changes a live publication or launches a matcher experiment.
"""

import argparse
import dataclasses
import importlib.metadata
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

from exact_inspect.artifacts import atomic_json
from exact_inspect.contracts import canonical_hash, file_hash


def classify(source, output, *, seconds, memory):
    import pyelk
    import pyowl_core as core

    receipt = json.loads(Path(str(source) + ".receipt.json").read_bytes())
    if receipt["resource_sha256"] != file_hash(source):
        raise ValueError("Policy-exported ontology differs from its frozen receipt")
    if receipt["schema"] != "exact-ontology-resource/1" or not receipt["logical_axioms_preserved"]:
        raise ValueError("A policy-preserving ontology resource is required")
    result = {
        "schema": "exact-review-elk-taxonomy/1",
        "status": "running",
        "input_sha256": receipt["resource_sha256"],
        "receipt_sha256": file_hash(Path(str(source) + ".receipt.json")),
        "scope": receipt["scope"],
        "source_documents": receipt["source_documents"],
        "ontology_version_id": receipt["ontology_version_id"],
        "policy_hash": receipt["policy_hash"],
        "versions": {p: importlib.metadata.version(p) for p in ("pyelk-reasoner", "pyowl-core")},
        "options": {
            "backend": "rust",
            "require_native_pipeline": True,
            "unsupported": "error",
            "wall_seconds": seconds,
            "parser_max_memory_bytes": memory,
        },
        "phases": [],
        "taxonomy": None,
    }
    started = time.monotonic()
    reasoner = None

    def phase(name, operation):
        before = time.monotonic()
        event = {"phase": name, "status": "running"}
        result["phases"].append(event)
        atomic_json(output, result)
        try:
            value = operation()
        except Exception:
            event.update(status="failed", seconds=time.monotonic() - before)
            raise
        event.update(status="complete", seconds=time.monotonic() - before)
        return value

    try:
        snapshot = phase(
            "native_parse",
            lambda: core.load_snapshot(
                source,
                options=core.LoadOptions(
                    backend=core.BackendPreference.NATIVE,
                    imports=core.ImportPolicy.RESOLVE_STRICT,
                    limits=core.ParseLimits(deadline_seconds=seconds, max_memory_bytes=memory),
                ),
            ),
        )
        reasoner = phase(
            "native_admission",
            lambda: pyelk.Reasoner(
                snapshot,
                config=pyelk.ReasonerConfig(
                    backend="rust", require_native_pipeline=True, unsupported="error", workers=2
                ),
            ),
        )
        if not phase("consistency", lambda: reasoner.is_consistent().require_complete()):
            raise ValueError("Admitted ontology is inconsistent")
        taxonomy = phase("classification", lambda: reasoner.classify().require_complete())
        node = lambda value: [member.iri.value for member in value.members]
        result["taxonomy"] = {
            "nodes": [node(n) for n in taxonomy.nodes],
            "direct_edges": [[node(a), node(b)] for a, b in taxonomy.direct_edges],
            "top": node(taxonomy.top),
            "bottom": node(taxonomy.bottom),
        }
        result.update(status="complete", taxonomy_sha256=canonical_hash(result["taxonomy"]))
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, reason=str(error))
        issues = getattr(error, "reasons", None)
        if issues:
            result["unsupported"] = [
                dataclasses.asdict(i) if dataclasses.is_dataclass(i) else str(i) for i in issues
            ]
    finally:
        if reasoner is not None:
            reasoner.close()
    result.update(
        seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    )
    atomic_json(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--memory", type=int, default=2 * 1024**3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600 or args.memory <= 0:
        parser.error("Provide a positive memory budget and a wall budget of at most one hour")
    if args.worker:
        classify(args.source, args.output, seconds=args.seconds, memory=args.memory)
        return
    if args.output.exists():
        parser.error("Preserve the previous receipt and choose a new output")
    try:
        process = subprocess.run(
            [sys.executable, "-m", "tools.review_elk_taxonomy", *sys.argv[1:], "--worker"],
            timeout=args.seconds,
            check=False,
        )
        if process.returncode:
            raise RuntimeError("ELK worker exited before completing its receipt")
    except (subprocess.TimeoutExpired, RuntimeError) as error:
        result = json.loads(args.output.read_bytes()) if args.output.exists() else {}
        result.update(
            status="failed",
            error_type=type(error).__name__,
            taxonomy=None,
            reason="Bounded ELK worker did not complete",
            wall_seconds=args.seconds,
        )
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
