#!/usr/bin/env python3
"""Build the local explorer and copy its static export into the Python package."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-install", action="store_true", help="Reuse an existing npm ci installation"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    frontend = root / "explanations_visualizer"
    npm = shutil.which("npm")
    node = shutil.which("node")
    if not npm or not node:
        parser.error("Node.js 24 and npm are required to build from source")
    version = subprocess.check_output([node, "--version"], text=True).strip()
    if not version.startswith("v24."):
        parser.error(f"Node.js 24 is required to build the frontend (found {version})")
    if not args.skip_install:
        subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=frontend, check=True)
    subprocess.run([npm, "run", "build"], cwd=frontend, check=True)
    export = frontend / "out"
    for route in ("index.html", "browse/index.html", "library/index.html"):
        if not (export / route).is_file():
            raise RuntimeError(f"Missing frontend route: {route}")
    if any((export / page).exists() for page in ("participate", "admin", "legacy")):
        raise RuntimeError("Unexpected product in the local frontend export")
    destination = root / "exact_inspect" / "static"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(export, destination)
    print(json.dumps({"frontend": str(destination)}))


if __name__ == "__main__":
    main()
