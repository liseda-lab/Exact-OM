#!/usr/bin/env python3
"""Check wheel/sdist release boundaries and record their immutable hashes."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

REQUIRED = {
    "exact/__init__.py",
    "exact/default_config.yaml",
    "exact/decisions/runtime.json",
    "exact/native/score.py",
    "exact/acceptance/model.py",
    "exact/tracks/builtin/anatomy.yaml",
    "exact/tracks/builtin/biokg.yaml",
    "exact/tracks/builtin/bioml_hf.yaml",
    "exact/tracks/builtin/bioml_zenodo.yaml",
    "exact/tracks/builtin/conference.yaml",
    "exact/tracks/builtin/diso.yaml",
    "exact/tracks/builtin/oaei_kg.yaml",
}
FORBIDDEN = (
    "exact/repair/",
    "exact/experiments/",
    "exact_inspect/",
    "study_visualizer_runtime/",
    "explanations_visualizer/",
    "specs/",
    "data/",
    ".agents/",
    ".codex/",
    "docs/archive/",
    "docs/verification/",
    "release/evidence/",
)


def check(path):
    path = Path(path)
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata_path = next(n for n in names if n.endswith(".dist-info/METADATA"))
            metadata = BytesParser().parsebytes(archive.read(metadata_path))
            entrypoints = configparser.ConfigParser()
            entrypoints.read_string(
                archive.read(metadata_path.replace("METADATA", "entry_points.txt")).decode()
            )
            if set(entrypoints["console_scripts"]) != {"exact", "exact-eval", "bioml-eval"}:
                raise ValueError("Unexpected console entrypoints")
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            names = [
                str(PurePosixPath(m.name).relative_to(PurePosixPath(m.name).parts[0]))
                for m in members
            ]
            member = next(
                m for m in members if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")
            )
            metadata = BytesParser().parsebytes(archive.extractfile(member).read())
    else:
        raise ValueError(f"Unsupported artifact: {path}")
    missing = REQUIRED - set(names)
    forbidden = [
        n
        for n in names
        if n.startswith(FORBIDDEN) or n.endswith((".pyc", "/AGENTS.md")) or "__pycache__/" in n
    ]
    if missing or forbidden:
        raise ValueError(
            f"Invalid distribution: missing={sorted(missing)}, forbidden={forbidden[:20]}"
        )
    if metadata["Name"].lower().replace("_", "-") != "exact-om":
        raise ValueError("Unexpected project name")
    if not any("license" in n.lower() for n in names):
        raise ValueError("Distribution has no license file")
    if set(metadata.get_all("Provides-Extra", [])) & {"repair", "viz", "study"}:
        raise ValueError("Unfinished product extras must not be published")
    return {
        "path": path.name,
        "version": metadata["Version"],
        "files": len(names),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifacts = sorted([*args.dist.glob("*.whl"), *args.dist.glob("*.tar.gz")])
    if len(artifacts) != 2 or len(list(args.dist.glob("*.whl"))) != 1:
        raise ValueError("Expected exactly one wheel and one sdist in distribution directory")
    result = {"status": "passed", "artifacts": [check(p) for p in artifacts]}
    if len({r["version"] for r in result["artifacts"]}) != 1:
        raise ValueError("Wheel and sdist versions differ")
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
