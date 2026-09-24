"""Assemble a content-bound backend handoff without changing acceptance decisions.

This copies public contracts, prepared response fixtures and admitted ontologies, never study publication
keys, provider requests, database dumps or arbitrary execution inputs. Gate evidence
outside this public allowlist is referenced by hash rather than copied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from exact_inspect.artifacts import (
    atomic_json,
    read_metadata,
    relative_path,
    validate_bundle,
)
from exact_inspect.context_resources import validate_ontology_resource
from exact_inspect.contracts import (
    CONTRACT_VERSION,
    VisibilityPolicy,
    canonical_hash,
    file_hash,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = (
    "exact_inspect/client.py",
    "exact_inspect/study/migrations/001_initial.sql",
    "exact_inspect/study/owl_ast_schema.json",
    "specs/explanation-framework/protocol/contract.schema.json",
    "specs/explanation-framework/protocol/study.runtime-openapi.json",
    "specs/explanation-framework/protocol/study-resource.runtime-schema.json",
    "docs/guides/explanation-backend.md",
    "docs/guides/ontology-context-preparation.md",
    "docs/guides/explanation-study-service.md",
    "docs/verification/explanation-study-postgres.json",
    "docs/verification/explanation-matcher-fidelity.json",
    "docs/verification/explanation-restriction-probe.json",
    "deploy/render/exact_study.Dockerfile",
    "deploy/render/exact_study_requirements.txt",
    "deploy/render/study.render.yaml",
    "deploy/render/exact_inspect_prepared.Dockerfile",
    "deploy/render/exact_inspect_prepared_requirements.txt",
)
GATES = tuple(f"G{index}" for index in range(6))
PRIVATE_FIELDS = {
    "answer_key",
    "case_keys",
    "researcher_case_keys",
    "acceptable_candidate_ids",
    "case_kind",
    "api_key",
    "access_token",
    "signing_secret",
    "researcher_token",
    "password",
    "cookie",
    "authorization",
    "database_url",
    "invitation_secret",
    "invitation_url",
    "invite_token",
}


def _public_data(value: Any) -> None:
    """Guard copied data; public schema declarations are handled separately."""
    pending = [value]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if any(key.lower().replace("-", "_") in PRIVATE_FIELDS for key in node):
                raise ValueError("Private data cannot enter a public handoff artifact")
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)


def _binding(path: Path, locator: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Handoff input is missing or linked")
    return {"path": locator, "sha256": file_hash(path), "size_bytes": path.stat().st_size}


def _result(status: str) -> str:
    if status in {"pass", "passed"}:
        return "pass"
    if status in {"fail", "failed", "blocked_input"}:
        return "fail"
    return "not_run"


def publish_handoff(
    package: Path,
    verification_dir: Path,
    status_ledger: Path,
    output_dir: Path,
    *,
    repository_root: Path = ROOT,
    execution_lock: Path | None = None,
    claim_audit: Path | None = None,
    ontology_resources: list[Path] | None = None,
) -> dict[str, Any]:
    """Publish an immutable artifact, preserving failed, blocked and unrun gates."""
    package, verification_dir, output_dir = map(Path, (package, verification_dir, output_dir))
    repository_root = Path(repository_root).resolve()
    manifest = validate_bundle(package)
    verification = read_metadata(verification_dir / "verification.json")
    if verification.get("package_id") != manifest.package_id:
        raise ValueError("Verification belongs to another package")
    required_checks = {
        "package_integrity",
        "cold_service_health",
        "cold_entity_queries",
        "bounded_reads",
        "warm_read_p95",
        "serving_memory",
    }
    if verification.get(
        "schema_version"
    ) != "exact-explain-verification/1" or not required_checks <= set(
        verification.get("checks", {})
    ):
        raise ValueError("Handoff requires the complete backend verification receipt")
    ledger = read_metadata(Path(status_ledger))
    if set(ledger.get("gates", {})) != set(GATES):
        raise ValueError("Handoff requires explicit G0-G5 dispositions")
    gates, evidence = {}, {}
    for gate_id in GATES:
        source_gate = ledger["gates"][gate_id]
        status = source_gate["status"]
        references = source_gate.get("evidence_ids", [])
        if _result(status) == "pass" and not references:
            raise ValueError("A passed gate requires bound evidence")
        for reference in references:
            record = ledger.get("evidence", {}).get(reference)
            if record is None:
                raise ValueError("Gate references unknown evidence")
            locator = record.get("path", record.get("artifact"))
            bound: dict[str, Any] = {
                "status": record["status"],
                "result": _result(record["status"]),
                "available": False,
            }
            if locator:
                path = relative_path(repository_root, locator)
                if path.is_file():
                    bound.update(_binding(path, locator), available=True, copied=False)
                    expected = record.get("sha256")
                    if expected and bound["sha256"] != expected:
                        raise ValueError("Gate evidence checksum changed")
                    if bound["result"] == "pass" and path.suffix == ".json":
                        receipt = read_metadata(path)
                        if receipt.get("status") in {
                            "failed",
                            "fail",
                            "running",
                            "not_run",
                            "blocked_input",
                        }:
                            raise ValueError(
                                "Claimed passed evidence records an unsuccessful result"
                            )
                        tests = receipt.get("test_suite", {})
                        if tests.get("errors", 0) or tests.get("failures", 0):
                            raise ValueError("Claimed passed evidence contains failed tests")
            if _result(status) == "pass" and (not bound["available"] or bound["result"] != "pass"):
                raise ValueError("A passed gate has missing or unsuccessful evidence")
            evidence[reference] = bound
        gates[gate_id] = {
            "status": status,
            "result": _result(status),
            "evidence_ids": references,
            "remaining": source_gate.get("remaining", []),
            "checks": {
                key.removesuffix("_status"): {"status": value, "result": _result(value)}
                for key, value in source_gate.items()
                if key.endswith("_status")
            },
        }
    if gates["G4"]["result"] == "pass" and any(
        _result(value) != "pass" for value in verification.get("checks", {}).values()
    ):
        raise ValueError("Resource verification does not support a passed G4")
    if gates["G5"]["result"] == "pass" and any(
        gates[gate]["result"] != "pass" for gate in GATES[:-1]
    ):
        raise ValueError("G5 acceptance requires every prerequisite gate")
    if output_dir.exists():
        raise FileExistsError("Choose a new handoff output directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=".handoff-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)

        def copy_public(source: Path, locator: str, role: str, *, schema: bool = False) -> None:
            binding = _binding(source, locator)
            if source.suffix == ".json" and not schema:
                _public_data(read_metadata(source))
            target = relative_path(staging, locator)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if file_hash(target) != binding["sha256"]:
                raise ValueError("Handoff input changed while copying")
            inventory.append({**binding, "role": role})

        for locator in PUBLIC_FILES:
            source = relative_path(repository_root, locator)
            schema = locator.endswith(("schema.json", "openapi.json"))
            copy_public(source, "reference/" + locator, "public_reference", schema=schema)
        specs = repository_root / "specs/explanation-framework"
        if not (specs / "08-frontend-implementation.md").is_file():
            raise ValueError("The frontend assignment is required in the handoff")
        for source in sorted(specs.glob("[0-9][0-9]-*.md")):
            copy_public(
                source, "reference/" + str(source.relative_to(repository_root)), "frontend_brief"
            )
        for source in sorted((specs / "fixtures").rglob("*.json")):
            copy_public(
                source, "reference/" + str(source.relative_to(repository_root)), "design_fixture"
            )
        for name in (
            "verification.json",
            "openapi.json",
            "execution-lock.schema.json",
            "study-resource.schema.json",
        ):
            copy_public(
                verification_dir / name,
                "verification/" + name,
                "runtime_verification",
                schema=name != "verification.json",
            )
        responses = sorted((verification_dir / "fixtures").glob("*.json"))
        if not responses:
            raise ValueError("Prepared API response fixtures are required")
        for source in responses:
            copy_public(source, "verification/fixtures/" + source.name, "prepared_response")
        optional = {}
        for name, optional_source in (
            ("execution_lock", execution_lock),
            ("claim_audit", claim_audit),
        ):
            if optional_source is not None:
                value = read_metadata(Path(optional_source))
                _public_data(value)
                if (
                    name == "execution_lock"
                    and VisibilityPolicy.model_validate(value.get("policy", {})).policy_hash
                    != manifest.policy.policy_hash
                ):
                    raise ValueError("Execution lock policy differs from the package")
                if name == "claim_audit":
                    ids = [record.get("explanation_id") for record in value.get("records", [])]
                    if len(ids) != len(set(ids)) or set(ids) != set(manifest.explanations):
                        raise ValueError("Claim audit does not describe the package explanations")
                    if gates["G3"]["result"] == "pass" and not value.get("all_supported"):
                        raise ValueError("Claim audit does not support a passed G3")
                # The full lock may bind private local inputs; only record its hash.
                optional[name] = _binding(
                    Path(optional_source), str(Path(optional_source).resolve())
                )
                if name == "claim_audit":
                    copy_public(Path(optional_source), "verification/claim-audit.json", name)
                    optional[name]["path"] = "verification/claim-audit.json"

        artifact_hashes = {artifact.path: artifact.sha256 for artifact in manifest.artifacts}
        ontologies = {}
        source_context_hashes = {}
        for identity, locator in manifest.ontologies.items():
            inner = read_metadata(relative_path(package.parent, locator) / "manifest.json")
            if inner.get("ontology_version_id") != identity:
                raise ValueError("Ontology identity differs from package binding")
            if inner.get("policy_filter", {}).get("policy_hash") != manifest.policy.policy_hash:
                raise ValueError("Handoff context lacks its physical visibility-policy binding")
            if any(
                artifact_hashes.get(locator + "/" + name) != digest
                for name, digest in inner["artifacts"].items()
            ):
                raise ValueError("Inner ontology artifacts differ from the package inventory")
            ontologies[identity] = {
                "name": inner.get("name"),
                "root_sha256": inner.get("identity", {}).get("root_sha256"),
                "source_derivation": inner.get("source_derivation"),
                "manifest_sha256": file_hash(
                    relative_path(package.parent, locator) / "manifest.json"
                ),
                "artifacts": inner["artifacts"],
                "scope": inner["scope"],
                "completeness": inner["completeness"],
                "capabilities": inner["capabilities"],
            }
            source_context_hashes[identity] = inner["policy_filter"]["source_context_sha256"]
        resources = {}
        for source in ontology_resources or []:
            source = Path(source)
            receipt_path = source.with_suffix(source.suffix + ".receipt.json")
            receipt = validate_ontology_resource(
                source,
                receipt_path,
                policy_hash=manifest.policy.policy_hash,
                ontology_ids=manifest.ontologies,
            )
            identity = receipt["ontology_version_id"]
            if receipt["source_context_sha256"] != source_context_hashes[identity]:
                raise ValueError("Ontology resource source context differs from the package")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", identity) or identity in resources:
                raise ValueError("Ontology resource identity is invalid or duplicated")
            locator = "resources/" + identity.removeprefix("sha256:") + ".ofn"
            receipt_locator = locator + ".receipt.json"
            copy_public(source, locator, "ontology_resource")
            copy_public(receipt_path, receipt_locator, "ontology_resource_receipt")
            # Bind copied bytes too, in case an input changes during publication.
            copied = validate_ontology_resource(
                staging / locator,
                staging / receipt_locator,
                policy_hash=manifest.policy.policy_hash,
                ontology_ids=[identity],
            )
            if copied != receipt:
                raise ValueError("Ontology admission receipt changed while copying")
            resources[identity] = {
                "path": locator,
                "receipt_path": receipt_locator,
                "sha256": receipt["resource_sha256"],
                "receipt_hash": receipt["receipt_hash"],
                "source_context_sha256": receipt["source_context_sha256"],
                "format": receipt["format"],
            }
        runs = {}
        for identity, locator in manifest.runs.items():
            inner = read_metadata(
                relative_path(package.parent, locator) / "decisions.manifest.json"
            )
            if inner.get("run_id") != identity:
                raise ValueError("Run identity differs from package binding")
            if inner.get("policy_filter", {}).get("policy_hash") != manifest.policy.policy_hash:
                raise ValueError("Handoff run lacks its physical visibility-policy binding")
            if artifact_hashes.get(locator + "/" + inner["database"]) != inner["database_hash"]:
                raise ValueError("Inner run database differs from the package inventory")
            runs[identity] = {
                key: inner[key]
                for key in (
                    "revision",
                    "database_hash",
                    "source_ontology_version_id",
                    "target_ontology_version_id",
                )
            }
        relative_package = os.path.relpath(package.resolve(), output_dir.resolve())
        passed = all(gate["result"] == "pass" for gate in gates.values())
        result = {
            "schema_version": "exact-explanation-handoff/1",
            "contract_version": CONTRACT_VERSION,
            "publication_status": "complete",
            "acceptance_status": (
                "passed"
                if passed
                else (
                    "blocked_input"
                    if any(gate["status"] == "blocked_input" for gate in gates.values())
                    else "pending"
                )
            ),
            "frontend_admitted": passed and ledger.get("frontend_admitted") is True,
            "gates": gates,
            "evidence": evidence,
            "status_ledger": _binding(Path(status_ledger), str(Path(status_ledger).resolve())),
            "package": {
                **_binding(package, relative_package),
                "package_id": manifest.package_id,
                "schema_version": manifest.schema_version,
                "audience": manifest.audience,
            },
            "policy": {
                "hash": manifest.policy.policy_hash,
                "value": manifest.policy.model_dump(mode="json"),
            },
            "ontologies": ontologies,
            "ontology_resources": resources,
            "runs": runs,
            "explanation_ids": sorted(manifest.explanations),
            "runtime": verification["runtime"],
            "resource_checks": verification["checks"],
            "resource_measurements": verification["measurements"],
            "capabilities": manifest.capabilities,
            "profiles": {
                "local_app": {"import_enabled": True},
                "public_demo": {
                    "import_enabled": False,
                    "requires_audience": "development_demo",
                    "package_supported": manifest.audience == "development_demo",
                },
                "study": {
                    "isolated_service": True,
                    "postgresql_required": True,
                    "import_enabled": False,
                },
            },
            "commands": {
                "cwd": "handoff directory for package commands; repository root for build, prepare and study fixture commands",
                "serve": [
                    "exact-inspect",
                    "serve",
                    "--package",
                    relative_package,
                    "--profile",
                    "local_app",
                ],
                "demo": [
                    "exact-inspect",
                    "serve",
                    "--package",
                    relative_package,
                    "--profile",
                    "public_demo",
                ],
                "verify": [
                    "exact-inspect",
                    "verify-backend",
                    "--package",
                    relative_package,
                    "--output",
                    "verification-repeat",
                ],
                "build": ["poetry", "install", "-E", "viz", "-E", "study"],
                "study": [
                    "uvicorn",
                    "exact_inspect.study.api:app_from_env",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                    "--no-access-log",
                ],
                "synthetic_study_client": [
                    "python",
                    "-m",
                    "pytest",
                    "tests/explanation_study_test.py::test_complete_synthetic_http_client_flow",
                    "-q",
                ],
                "invitation_issuance": [
                    "python",
                    "-m",
                    "exact_inspect.study",
                    "invitations",
                    "--study",
                    "STUDY_REVISION",
                    "--count",
                    "1",
                    "--output",
                    "PRIVATE_OUTPUT.json",
                ],
                "resume": [
                    "exact-inspect",
                    "prepare",
                    "--lock",
                    "EXECUTION_LOCK.json",
                    "--output-root",
                    "PREPARATION_ROOT",
                    "--resume",
                ],
                "selective_repair": [
                    "exact-inspect",
                    "prepare",
                    "--lock",
                    "REVISED_LOCK.json",
                    "--output-root",
                    "NEW_ROOT",
                    "--resume-from",
                    "PREVIOUS_ROOT",
                ],
            },
            "fixture_runner": {
                **_binding(
                    repository_root / "tests/explanation_study_test.py",
                    "tests/explanation_study_test.py",
                ),
                "copied": False,
                "reason": "Repository test runner owns synthetic private adjudication; only public study fixtures are copied.",
            },
            "inventory": inventory,
            "inputs": optional,
            "limitations": ledger.get("deferred", []),
            "frontend_brief": "reference/specs/explanation-framework/08-frontend-implementation.md",
            "data_policy": "Fixed public reference inventory, prepared API responses and policy-admitted ontology resources only; no provider request ledger, database dump, invitation credential or study publication envelope is copied.",
        }
        _public_data(result)
        result["handoff_id"] = canonical_hash(result)
        atomic_json(staging / "backend-handoff.json", result)
        os.replace(staging, output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--verification-dir", required=True, type=Path)
    parser.add_argument("--status-ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--claim-audit", type=Path)
    parser.add_argument("--ontology-resource", action="append", type=Path, default=[])
    args = parser.parse_args()
    result = publish_handoff(
        args.package,
        args.verification_dir,
        args.status_ledger,
        args.output_dir,
        repository_root=args.repository_root,
        execution_lock=args.execution_lock,
        claim_audit=args.claim_audit,
        ontology_resources=args.ontology_resource,
    )
    print(
        json.dumps(
            {
                "path": str(args.output_dir / "backend-handoff.json"),
                "handoff_id": result["handoff_id"],
                "acceptance_status": result["acceptance_status"],
                "frontend_admitted": result["frontend_admitted"],
            }
        )
    )


if __name__ == "__main__":
    main()
