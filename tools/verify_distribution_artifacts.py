#!/usr/bin/env python3
"""Verify Exact wheel/sdist metadata and emit release-candidate support files."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

CONTRACT_PATH = "release/core-compatibility.json"
CONTRACT_SCHEMA = "exact-om.ontology-stack-compatibility/1"
HASH_SCHEMA = "exact-om.distribution-hashes/1"

_BASE_REQUIREMENTS = {
    "pyowl-core": ">=0.2,<0.3",
    "pyowl2vec-star-projector": ">=0.2,<0.3",
}
_EXTRA_REQUIREMENTS = {
    "pyelk-reasoner": (">=0.2,<0.3", "reasoning"),
    "pyhermit": (">=0.2,<0.3", "reasoning"),
    "oaei-bioml-eval": (">=0.2.1,<0.3", "bioml-eval"),
    "fastapi": (None, "viz"),
    "pydantic-settings": (None, "viz"),
    "uvicorn": (None, "viz"),
}
_STACK_RANGES = {
    "pyowl-core": SpecifierSet("==0.2.0"),
    "pyowl2vec-star-projector": SpecifierSet(">=0.2,<0.3"),
    "pyelk-reasoner": SpecifierSet(">=0.2,<0.3"),
    "pyhermit": SpecifierSet(">=0.2,<0.3"),
    "oaei-bioml-eval": SpecifierSet(">=0.2.1,<0.3"),
}


def _single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {pattern!r} in {directory}, found {matches}")
    return matches[0]


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wheel_records(wheel: Path) -> tuple[dict[str, Any], bytes]:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if any(name.startswith("release/evidence/") for name in names):
            raise SystemExit("wheel must exclude release/evidence")
        if "exact_inspect/static/index.html" not in names:
            raise SystemExit("wheel does not contain the built Exact Inspect frontend")
        if CONTRACT_PATH not in names:
            raise SystemExit(f"wheel does not contain {CONTRACT_PATH}")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise SystemExit(f"wheel has an unexpected METADATA set: {metadata_names}")
        contract_bytes = archive.read(CONTRACT_PATH)
        metadata_bytes = archive.read(metadata_names[0])
    return json.loads(contract_bytes), metadata_bytes


def _sdist_contract(sdist: Path) -> dict[str, Any]:
    with tarfile.open(sdist, "r:gz") as archive:
        if any("/release/evidence/" in member.name for member in archive.getmembers()):
            raise SystemExit("sdist must exclude release/evidence")
        members = [
            member for member in archive.getmembers() if member.name.endswith(f"/{CONTRACT_PATH}")
        ]
        if len(members) != 1:
            raise SystemExit(f"sdist has an unexpected compatibility manifest set: {members}")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise SystemExit("could not read the compatibility manifest from the sdist")
        return json.load(stream)


def _metadata(metadata_bytes: bytes) -> tuple[str, list[Requirement]]:
    message = BytesParser(policy=policy.default).parsebytes(metadata_bytes)
    version = message.get("Version")
    if not version:
        raise SystemExit("wheel metadata has no Version")
    requirements = [Requirement(value) for value in message.get_all("Requires-Dist", [])]
    return str(version), requirements


def _requirements_by_name(requirements: list[Requirement]) -> dict[str, list[Requirement]]:
    result: dict[str, list[Requirement]] = {}
    for requirement in requirements:
        result.setdefault(canonicalize_name(requirement.name), []).append(requirement)
    return result


def _marker_matches(requirement: Requirement, extra: str) -> bool:
    if requirement.marker is None:
        return True
    environment = {key: str(value) for key, value in default_environment().items()}
    environment["extra"] = extra
    return requirement.marker.evaluate(environment)


def _require_metadata_contract(requirements: list[Requirement]) -> None:
    by_name = _requirements_by_name(requirements)
    for name, specifier in _BASE_REQUIREMENTS.items():
        rows = by_name.get(name, [])
        if len(rows) != 1:
            raise SystemExit(f"expected one base requirement for {name}, found {rows}")
        requirement = rows[0]
        if requirement.specifier != SpecifierSet(specifier):
            raise SystemExit(f"unexpected {name} constraint: {requirement.specifier}")
        if requirement.marker is not None:
            raise SystemExit(f"base requirement {name} is conditional: {requirement.marker}")

    known_extras = {extra for _, extra in _EXTRA_REQUIREMENTS.values()}
    for name, (specifier, extra) in _EXTRA_REQUIREMENTS.items():
        rows = by_name.get(name, [])
        if len(rows) != 1:
            raise SystemExit(f"expected one optional requirement for {name}, found {rows}")
        requirement = rows[0]
        if specifier is not None and requirement.specifier != SpecifierSet(specifier):
            raise SystemExit(f"unexpected {name} constraint: {requirement.specifier}")
        if requirement.marker is None or not _marker_matches(requirement, extra):
            raise SystemExit(f"{name} is not enabled by the {extra!r} extra")
        if _marker_matches(requirement, ""):
            raise SystemExit(f"optional requirement {name} is active in the base install")
        for other in known_extras - {extra}:
            if _marker_matches(requirement, other):
                raise SystemExit(f"{name} is unexpectedly enabled by the {other!r} extra")


def _published_stack(contract: dict[str, Any]) -> dict[str, str]:
    tested = contract.get("tested_stack")
    if not isinstance(tested, dict):
        raise SystemExit("compatibility manifest has no tested_stack object")
    versions: dict[str, str] = {}
    for name, allowed in _STACK_RANGES.items():
        record = tested.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("version"), str):
            raise SystemExit(f"compatibility manifest has no tested version for {name}")
        raw_version = record["version"]
        version = Version(raw_version)
        if version not in allowed:
            raise SystemExit(f"tested {name} version {version} is outside {allowed}")
        if version.is_prerelease or version.is_devrelease or version.local is not None:
            raise SystemExit(f"tested {name} version is not a final published release: {version}")
        versions[name] = raw_version
    return versions


def _require_compatibility_contract(contract: dict[str, Any], exact_version: str) -> dict[str, str]:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise SystemExit(f"unexpected compatibility manifest schema: {contract.get('schema')!r}")
    if contract.get("status") != "verified":
        raise SystemExit("compatibility manifest is not finalized with status=verified")
    if contract.get("performance_claim") is not False:
        raise SystemExit("compatibility manifest must set performance_claim=false")
    exact = contract.get("exact")
    if not isinstance(exact, dict) or exact.get("version") != exact_version:
        raise SystemExit(
            "compatibility manifest Exact version does not match the wheel metadata: "
            f"{None if not isinstance(exact, dict) else exact.get('version')!r} != {exact_version!r}"
        )
    if Version(exact_version) != Version("2.1.0"):
        raise SystemExit(f"release artifacts must be Exact-OM 2.1.0, found {exact_version}")

    core = contract.get("core_contract")
    encoded = contract.get("encoded_contract")
    if not isinstance(core, dict) or core.get("api_version") != [0, 2]:
        raise SystemExit("compatibility manifest must record core API version [0, 2]")
    if core.get("model_schema") != 2:
        raise SystemExit("compatibility manifest must record core model schema 2")
    if not isinstance(encoded, dict) or encoded.get("schema_version") != 2:
        raise SystemExit("compatibility manifest must record encoded structural schema 2")
    if encoded.get("model_schema") != 2:
        raise SystemExit("encoded contract must be bound to core model schema 2")
    if encoded.get("performance_claim") is not False:
        raise SystemExit("encoded contract must set performance_claim=false")
    descriptor = encoded.get("descriptor_sha256")
    if not isinstance(descriptor, str) or len(descriptor) != 64:
        raise SystemExit("encoded descriptor must be a 64-character SHA-256 hex digest")
    try:
        bytes.fromhex(descriptor)
    except ValueError as error:
        raise SystemExit("encoded descriptor is not hexadecimal") from error
    return _published_stack(contract)


def _hash_record(wheel: Path, sdist: Path, exact_version: str) -> dict[str, object]:
    return {
        "schema": HASH_SCHEMA,
        "exact_version": exact_version,
        "artifacts": [
            {
                "filename": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (wheel, sdist)
        ],
    }


def _constraints(versions: dict[str, str]) -> str:
    return "".join(f"{name}=={version}\n" for name, version in sorted(versions.items()))


def _write_or_check(path: Path, expected: bytes, *, write: bool) -> None:
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)
        return
    if not path.is_file():
        raise SystemExit(f"required release support file is missing: {path}")
    if path.read_bytes() != expected:
        raise SystemExit(f"release support file does not match the tested artifacts: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--write-support-files",
        action="store_true",
        help="Write artifact-sha256.json and stack-constraints.txt beside the distributions.",
    )
    args = parser.parse_args()

    wheel = _single(args.dist_dir, "*.whl")
    sdist = _single(args.dist_dir, "*.tar.gz")
    wheel_contract, metadata_bytes = _wheel_records(wheel)
    sdist_contract = _sdist_contract(sdist)
    if wheel_contract != sdist_contract:
        raise SystemExit("wheel and sdist compatibility manifests differ")
    exact_version, requirements = _metadata(metadata_bytes)
    _require_metadata_contract(requirements)
    versions = _require_compatibility_contract(wheel_contract, exact_version)

    hash_payload = _hash_record(wheel, sdist, exact_version)
    _write_or_check(
        args.dist_dir / "artifact-sha256.json",
        _json_bytes(hash_payload),
        write=args.write_support_files,
    )
    _write_or_check(
        args.dist_dir / "stack-constraints.txt",
        _constraints(versions).encode("utf-8"),
        write=args.write_support_files,
    )
    for artifact in hash_payload["artifacts"]:
        assert isinstance(artifact, dict)
        print(f"verified {artifact['filename']}: sha256={artifact['sha256']}")


if __name__ == "__main__":
    main()
