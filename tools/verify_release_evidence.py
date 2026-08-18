#!/usr/bin/env python3
"""Fail closed unless final Exact 2.1 release evidence matches tested artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read JSON evidence {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"evidence root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_pass_rows(name: str, value: object) -> None:
    if not isinstance(value, dict) or not value:
        raise SystemExit(f"{name} must be a non-empty object")
    for row_name, raw_row in value.items():
        if not isinstance(raw_row, dict) or raw_row.get("status") != "passed":
            raise SystemExit(f"{name}.{row_name} has not passed")
        evidence = raw_row.get("evidence")
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(item, str) or not item.strip() for item in evidence)
        ):
            raise SystemExit(f"{name}.{row_name} has no stable evidence identifier")


def _artifact_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or len(rows) != 2:
        raise SystemExit("distribution hash record must contain exactly two artifacts")
    result: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, dict) or not isinstance(raw_row.get("filename"), str):
            raise SystemExit("distribution hash record contains an invalid artifact row")
        filename = raw_row["filename"]
        kind = "wheel" if filename.endswith(".whl") else "sdist"
        if kind in result or (kind == "sdist" and not filename.endswith(".tar.gz")):
            raise SystemExit(
                f"distribution hash record contains an unexpected artifact: {filename}"
            )
        result[kind] = raw_row
    if set(result) != {"wheel", "sdist"}:
        raise SystemExit("distribution hash record must identify one wheel and one sdist")
    return result


def _require_artifact_evidence(
    closure: dict[str, Any], hash_record: dict[str, Any], *, lock: Path, ncit: Path
) -> None:
    recorded = closure.get("artifacts")
    if not isinstance(recorded, dict):
        raise SystemExit("closure certificate has no artifact evidence")
    actual_rows = _artifact_rows(hash_record)
    for kind, actual in actual_rows.items():
        expected = recorded.get(kind)
        if not isinstance(expected, dict):
            raise SystemExit(f"closure certificate has no {kind} record")
        for field in ("filename", "sha256"):
            if expected.get(field) != actual.get(field):
                raise SystemExit(f"closure {kind} {field} does not match the tested artifact")
    if recorded.get("lock_sha256") != _sha256(lock):
        raise SystemExit("closure lock hash does not match poetry.lock")
    if recorded.get("ncit_doid_evidence_sha256") != _sha256(ncit):
        raise SystemExit("closure NCIT–DOID hash does not match the acceptance record")


def _require_ncit_evidence(payload: dict[str, Any], compatibility: dict[str, Any]) -> None:
    acceptance = payload.get("ncit_doid_acceptance")
    if (
        not isinstance(acceptance, dict)
        or acceptance.get("status") != "passed"
        or acceptance.get("performance_claim") is not False
    ):
        raise SystemExit("NCIT–DOID correctness acceptance has not passed")
    configuration = payload.get("configuration")
    if (
        not isinstance(configuration, dict)
        or configuration.get("load_backend") != "native"
        or configuration.get("projector_backend") != "native"
        or configuration.get("require_encoded_consumers") is not True
    ):
        raise SystemExit("NCIT–DOID evidence was not produced by the required native command")
    environment = payload.get("environment")
    packages = environment.get("packages") if isinstance(environment, dict) else None
    tested = compatibility.get("tested_stack")
    exact = compatibility.get("exact")
    if (
        not isinstance(packages, dict)
        or not isinstance(tested, dict)
        or not isinstance(exact, dict)
    ):
        raise SystemExit("NCIT–DOID or compatibility package evidence is incomplete")
    if packages.get("exact-om") != exact.get("version"):
        raise SystemExit("NCIT–DOID Exact version differs from the compatibility manifest")
    for name in ("pyowl-core", "pyowl2vec-star-projector"):
        record = tested.get(name)
        if not isinstance(record, dict) or packages.get(name) != record.get("version"):
            raise SystemExit(f"NCIT–DOID {name} version differs from the compatibility manifest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument(
        "--closure",
        type=Path,
        default=Path("release/evidence/legacy-compliance-closure.json"),
    )
    parser.add_argument(
        "--ncit",
        type=Path,
        default=Path("release/evidence/pyowl-core-0.2-ncit-doid.json"),
    )
    parser.add_argument(
        "--compatibility",
        type=Path,
        default=Path("release/core-compatibility.json"),
    )
    parser.add_argument("--lock", type=Path, default=Path("poetry.lock"))
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    closure = _load(args.closure)
    compatibility = _load(args.compatibility)
    ncit = _load(args.ncit)
    hash_record = _load(args.dist_dir / "artifact-sha256.json")
    if closure.get("schema") != "exact-om.legacy-compliance-closure/1":
        raise SystemExit("unexpected closure certificate schema")
    if closure.get("status") != "passed":
        raise SystemExit("legacy compliance closure is still pending or failed")
    if closure.get("tested_python_versions") != ["3.10", "3.11", "3.12"]:
        raise SystemExit("closure certificate does not cover Python 3.10–3.12")
    if any(
        closure.get(field) is not False
        for field in ("experiments_in_scope", "exact_repair_in_scope", "performance_claim")
    ):
        raise SystemExit("closure certificate has invalid scope or performance-claim flags")
    exact = closure.get("exact")
    compatible_exact = compatibility.get("exact")
    candidate = closure.get("candidate")
    if (
        not isinstance(candidate, dict)
        or candidate.get("implementation_commit") != args.revision
        or isinstance(candidate.get("source_date_epoch"), bool)
        or not isinstance(candidate.get("source_date_epoch"), int)
        or candidate["source_date_epoch"] <= 0
    ):
        raise SystemExit("closure implementation candidate or SOURCE_DATE_EPOCH is invalid")
    if (
        not isinstance(exact, dict)
        or not isinstance(compatible_exact, dict)
        or exact.get("version") != "2.1.0"
        or exact.get("version") != compatible_exact.get("version")
        or exact.get("commit") != args.revision
    ):
        raise SystemExit("closure Exact version/commit does not match the release candidate")

    _require_pass_rows("preserved_feature_matrix", closure.get("preserved_feature_matrix"))
    _require_pass_rows("audit_defects", closure.get("audit_defects"))
    _require_artifact_evidence(closure, hash_record, lock=args.lock, ncit=args.ncit)
    _require_ncit_evidence(ncit, compatibility)
    print(f"verified Exact 2.1 release evidence for {args.revision}")


if __name__ == "__main__":
    main()
