"""Replay strict native admission against a relocated, hash-pinned DOID closure.

This is a diagnostic, not a parser or a normalization path. Original and derived
documents are tested separately; a failed run never produces a context package.
"""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from exact_inspect.artifacts import atomic_json
from exact_inspect.contracts import file_hash


def pinned_imports(directory, manifest):
    """Relocate the retained map without trusting remote absolute path prefixes."""
    directory = Path(directory).resolve()
    records, resolver = {}, {}
    for iri, record in sorted(manifest.items()):
        source = Path(record["path"])
        relative = (
            Path("imports") / source.name if source.parent.name == "imports" else Path(source.name)
        )
        path = (directory / relative).resolve()
        expected = "sha256:" + record["sha256"].removeprefix("sha256:")
        if not path.is_relative_to(directory) or file_hash(path) != expected:
            raise ValueError("Pinned ontology import is missing, changed or outside its root")
        resolver[iri] = path
        records[iri] = {
            "path": relative.as_posix(),
            "sha256": expected,
            "bytes": path.stat().st_size,
        }
    return resolver, records


def probe(directory, *, timeout=90, memory=2 * 1024**3):
    import pyowl_core as core

    directory = Path(directory)
    resolver, records = pinned_imports(
        directory, json.loads((directory / "import-map.json").read_bytes())
    )
    attempts = []
    for name, expected, kind in (
        (
            "original.owl",
            "611355c445537fcf4bae2c519f1b3598af5a8fea793274316e35525b7d05e945",
            "original",
        ),
        (
            "doid.annotation-declarations.owl",
            "f552e866a6233add0c02e5fed1ac1b0236c36c7f92247ed8c3a2a522ecc88fbb",
            "declared_derivative",
        ),
    ):
        path = directory / name
        if file_hash(path) != "sha256:" + expected:
            raise ValueError("DOID root differs from the recorded pinned input")
        for scope in ("root", "closure"):
            start = time.monotonic()
            attempt = {
                "input": name,
                "input_kind": kind,
                "root_sha256": "sha256:" + expected,
                "scope": scope,
            }
            try:
                snapshot = core.load_snapshot(
                    path,
                    options=core.LoadOptions(
                        backend=core.BackendPreference.NATIVE,
                        imports=(
                            core.ImportPolicy.IGNORE
                            if scope == "root"
                            else core.ImportPolicy.RESOLVE_STRICT
                        ),
                        offline=True,
                        limits=core.ParseLimits(deadline_seconds=timeout, max_memory_bytes=memory),
                    ),
                    resolver=core.MappingResolver(resolver),
                )
                attempt.update(
                    status="parsed",
                    imports_complete=snapshot.is_complete,
                    axiom_count=sum(1 for _ in snapshot.iter_axioms()),
                )
            except (core.UnsupportedSyntaxError, core.ParseError) as error:
                attempt.update(
                    status="rejected",
                    error_type=type(error).__name__,
                    code=getattr(error, "code", None),
                    message=str(error),
                )
            attempt["seconds"] = time.monotonic() - start
            attempts.append(attempt)
    return {
        "schema": "exact-review-ontology-admission-probe/1",
        "pyowl_core_version": importlib.metadata.version("pyowl-core"),
        "backend": "native",
        "allow_partial_rdf_mapping": False,
        "limits": {"seconds_per_attempt": timeout, "max_memory_bytes": memory},
        "imports": records,
        "attempts": attempts,
        "context_publication": "none; diagnostic only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("Receipt exists; preserve it and use a new destination")
    atomic_json(args.receipt, probe(args.directory))


if __name__ == "__main__":
    main()
