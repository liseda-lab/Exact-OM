#!/usr/bin/env python3
"""Exercise an installed Exact distribution without importing the source checkout."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from generate_sbom import spdx_document
from packaging.utils import canonicalize_name
from verify_distribution_artifacts import _published_stack

_ONTOLOGY = b"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Ontology rdf:about="urn:exact:distribution-smoke"/>
  <owl:Class rdf:about="urn:A">
    <rdfs:subClassOf rdf:resource="urn:B"/>
  </owl:Class>
  <owl:Class rdf:about="urn:B"/>
</rdf:RDF>"""

_BASE_ABSENT_MODULES = (
    "deeponto",
    "fastapi",
    "jpype",
    "mowl",
    "oaei_bioml_eval",
    "pydantic_settings",
    "pyelk",
    "pyhermit",
    "pyhornedowl",
    "uvicorn",
)
_BASE_ABSENT_DISTRIBUTIONS = {
    "deeponto",
    "jpype1",
    "mowl-borg",
    "oaei-bioml-eval",
    "py-horned-owl",
    "pyelk-reasoner",
    "pyhermit",
}


def _reject_jvm_launch(event: str, args: tuple[object, ...]) -> None:
    if event != "subprocess.Popen" or not args:
        return
    executable = os.path.basename(os.fsdecode(args[0])).casefold()
    if executable in {"java", "java.exe", "javac", "javac.exe"}:
        raise RuntimeError(f"distribution smoke attempted to launch a JVM command: {executable}")


def _run_help(command: str) -> None:
    subprocess.run(
        [command, "--help"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise SystemExit(f"required distribution is not installed: {name}") from error


def _compatibility_contract() -> dict[str, Any]:
    distribution = importlib.metadata.distribution("exact-om")
    matches = [
        item
        for item in distribution.files or ()
        if item.as_posix() == "release/core-compatibility.json"
    ]
    if len(matches) != 1:
        raise SystemExit(f"installed compatibility manifest is missing or ambiguous: {matches}")
    path = Path(distribution.locate_file(matches[0]))
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_not_editable() -> None:
    distribution = importlib.metadata.distribution("exact-om")
    direct_urls = [item for item in distribution.files or () if item.name == "direct_url.json"]
    for item in direct_urls:
        payload = json.loads(Path(distribution.locate_file(item)).read_text(encoding="utf-8"))
        directory = payload.get("dir_info")
        if isinstance(directory, dict) and directory.get("editable") is True:
            raise SystemExit("distribution smoke is running against an editable Exact install")


def _assert_installed_versions(contract: dict[str, Any], names: tuple[str, ...]) -> None:
    exact = contract.get("exact")
    if not isinstance(exact, dict) or exact.get("version") != _distribution_version("exact-om"):
        raise SystemExit("installed Exact version does not match the compatibility manifest")
    versions = _published_stack(contract)
    for name in names:
        if versions[name] != _distribution_version(name):
            raise SystemExit(f"installed {name} version does not match the compatibility manifest")


def _assert_public_core_contract(contract: dict[str, Any]) -> None:
    import pyowl_core

    core = contract["core_contract"]
    encoded = contract["encoded_contract"]
    if list(pyowl_core.API_VERSION) != core["api_version"]:
        raise SystemExit("installed core API version differs from the compatibility manifest")
    if int(pyowl_core.MODEL_SCHEMA_VERSION) != core["model_schema"]:
        raise SystemExit("installed core model schema differs from the compatibility manifest")
    if list(pyowl_core.WIRE_FORMAT_VERSION) != core["wire_writer"]:
        raise SystemExit("installed core wire writer differs from the compatibility manifest")
    if int(pyowl_core.ADAPTER_PROTOCOL_VERSION) != core["adapter_protocol"]:
        raise SystemExit("installed core adapter protocol differs from the compatibility manifest")
    descriptor = pyowl_core.EncodedStructuralView.DESCRIPTOR_SHA256
    if not isinstance(descriptor, bytes) or len(descriptor) != 32:
        raise SystemExit("core encoded descriptor does not satisfy the public bytes contract")
    if descriptor.hex() != encoded["descriptor_sha256"]:
        raise SystemExit(
            "installed core encoded descriptor differs from the compatibility manifest"
        )


def _projection() -> tuple[object, ...]:
    from exact.ontology import load_ontology
    from exact.ontology.projection import require_native_report

    source = load_ontology(_ONTOLOGY)
    source.configure_projector(backend="native")
    edges = tuple(source.projection_edges())
    expected = ("urn:A", "http://subclassof", "urn:B")
    if tuple(edge.astuple() for edge in edges) != (expected,):
        raise SystemExit("native projection differs from the fixture subclass edge")
    if source.projector.last_view is not source.owl_snapshot():
        raise SystemExit("native projector did not retain the Exact core owner")
    require_native_report(source.projector)
    return edges


def _base_smoke(contract: dict[str, Any], sbom_output: Path | None) -> None:
    _assert_installed_versions(contract, ("pyowl-core", "pyowl2vec-star-projector"))
    _assert_public_core_contract(contract)
    for command in ("exact", "exact-eval", "bioml-eval"):
        _run_help(command)
    for module in _BASE_ABSENT_MODULES:
        if importlib.util.find_spec(module) is not None:
            raise SystemExit(f"optional or forbidden module is installed in base: {module}")
    _projection()

    payload = spdx_document("exact-om")
    names = {canonicalize_name(str(item["name"])) for item in payload["packages"]}
    missing = {"pyowl-core", "pyowl2vec-star-projector"} - names
    forbidden = _BASE_ABSENT_DISTRIBUTIONS & names
    if missing or forbidden:
        raise SystemExit(
            f"invalid base SBOM graph: missing={sorted(missing)}, forbidden={sorted(forbidden)}"
        )
    if sbom_output is not None:
        sbom_output.parent.mkdir(parents=True, exist_ok=True)
        sbom_output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _native_smoke(contract: dict[str, Any]) -> None:
    _assert_installed_versions(contract, ("pyowl-core", "pyowl2vec-star-projector"))
    _assert_public_core_contract(contract)
    _projection()


def _viz_smoke() -> None:
    for module in ("fastapi", "pydantic_settings", "uvicorn"):
        if importlib.util.find_spec(module) is None:
            raise SystemExit(f"viz dependency is missing: {module}")
    for command in ("exact-inspect", "exact-study-viz"):
        _run_help(command)


def _reasoning_smoke(contract: dict[str, Any]) -> None:
    from exact.ontology import load_ontology

    _assert_installed_versions(contract, ("pyelk-reasoner", "pyhermit"))
    for name in ("elk", "hermit"):
        source = load_ontology(_ONTOLOGY)
        snapshot = source.owl_snapshot()
        source.configure_reasoner(name, backend="rust" if name == "elk" else "native")
        if source.reasoner.ontology is not snapshot:
            raise SystemExit(f"{name} reasoner did not retain the Exact core owner")
        if source.reasoner.ancestors("urn:A") != {"urn:B"}:
            raise SystemExit(f"{name} reasoner returned an unexpected fixture hierarchy")


def _bioml_smoke(contract: dict[str, Any]) -> None:
    from exact.impl.evaluators.bioml import _load_bioml_api

    _assert_installed_versions(contract, ("oaei-bioml-eval",))
    api = _load_bioml_api()
    if api.missing:
        raise SystemExit(
            f"published BioML evaluation package is missing capabilities: {sorted(api.missing)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("base", "native", "viz", "reasoning", "bioml-eval"))
    parser.add_argument("--sbom-output", type=Path)
    args = parser.parse_args()

    sys.addaudithook(_reject_jvm_launch)
    os.environ.pop("JAVA_HOME", None)
    _assert_not_editable()
    contract = _compatibility_contract()
    if contract.get("performance_claim") is not False:
        raise SystemExit("installed compatibility manifest enables a performance claim")

    if args.mode == "base":
        _base_smoke(contract, args.sbom_output)
    elif args.mode == "native":
        _native_smoke(contract)
    elif args.mode == "viz":
        _viz_smoke()
    elif args.mode == "reasoning":
        _reasoning_smoke(contract)
    else:
        _bioml_smoke(contract)
    if any(name in sys.modules for name in ("jpype", "mowl", "deeponto")):
        raise SystemExit("a Java-backed ontology runtime was imported during the smoke")


if __name__ == "__main__":
    main()
