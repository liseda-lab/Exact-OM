"""Publish only content-bound public handoff material and never infer acceptance."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from exact_inspect.artifacts import atomic_json
from exact_inspect.contracts import DomainError, canonical_hash, file_hash
from exact_inspect.verification import verify_backend
from tests.explanation_preparation_test import prepared  # noqa: F401,F811
from tools.publish_explanation_handoff import GATES, PUBLIC_FILES, ROOT, publish_handoff


@pytest.fixture
def handoff_inputs(prepared, tmp_path):  # noqa: F811
    _, package, _, _ = prepared
    verification = tmp_path / "verification"
    verify_backend(package, verification)
    repository = tmp_path / "repository"
    for locator in (
        *PUBLIC_FILES,
        "tests/explanation_study_test.py",
        "specs/explanation-framework/08-frontend-implementation.md",
    ):
        target = repository / locator
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / locator, target)
    shutil.copytree(
        ROOT / "specs/explanation-framework/fixtures",
        repository / "specs/explanation-framework/fixtures",
    )
    # The evidence is explicitly synthetic; no real operational gate is claimed.
    proof = repository / "proof.json"
    atomic_json(proof, {"status": "passed", "synthetic": True})
    original = repository / "original-source.json"
    atomic_json(
        original, {"status": "failed", "synthetic": True, "error_type": "UnsupportedSyntaxError"}
    )
    ledger = {
        "frontend_admitted": False,
        "gates": {gate: {"status": "not_run", "evidence_ids": []} for gate in GATES},
        "evidence": {
            "fixture": {"status": "passed", "path": "proof.json", "sha256": file_hash(proof)},
            "original_source": {
                "status": "blocked_input",
                "path": "original-source.json",
                "sha256": file_hash(original),
            },
        },
        "deferred": ["Frontend and full experiments are not part of this fixture."],
    }
    ledger["gates"]["G0"] = {"status": "passed", "evidence_ids": ["fixture"]}
    ledger["gates"]["G1"] = {
        "status": "blocked_input",
        "strict_original_doid_status": "blocked_input",
        "evidence_ids": ["original_source"],
    }
    path = repository / "status.json"
    atomic_json(path, ledger)
    return package, verification, path, repository


def test_handoff_binds_public_inventory_without_promoting_original_source_gate(
    handoff_inputs, tmp_path
):
    package, verification, ledger, repository = handoff_inputs
    destination = tmp_path / "handoff"
    result = publish_handoff(package, verification, ledger, destination, repository_root=repository)
    assert result["publication_status"] == "complete"
    assert "http_runtime" not in result
    assert result["acceptance_status"] == "blocked_input" and not result["frontend_admitted"]
    assert result["gates"]["G1"]["status"] == "blocked_input"
    assert result["gates"]["G1"]["checks"]["strict_original_doid"]["result"] == "fail"
    assert result["gates"]["G5"]["result"] == "not_run"
    assert result["package"]["sha256"] == file_hash(package)
    assert result["policy"]["hash"] and result["ontologies"]
    for artifact in result["inventory"]:
        assert file_hash(destination / artifact["path"]) == artifact["sha256"]
    assert not result["fixture_runner"]["copied"]
    assert not (destination / "reference/tests/explanation_study_test.py").exists()
    schema = json.loads(
        (
            destination
            / "reference/specs/explanation-framework/protocol/study.runtime-openapi.json"
        ).read_text()
    )
    # Public definitions may name private request fields without exposing their values.
    assert "case_keys" in schema["components"]["schemas"]["Publish"]["properties"]
    with pytest.raises(FileExistsError):
        publish_handoff(package, verification, ledger, destination, repository_root=repository)


@pytest.mark.parametrize(
    "violation",
    [
        "changed_receipt",
        "missing_receipt",
        "wrong_package",
        "answer_key",
        "secret",
        "traversal",
        "false_pass",
        "wrong_lock",
        "wrong_claim_audit",
        "failed_measurement",
    ],
)
def test_handoff_rejects_unbound_or_private_inputs(handoff_inputs, tmp_path, violation):
    package, verification, ledger_path, repository = handoff_inputs
    ledger = json.loads(ledger_path.read_text())
    options = {}
    if violation == "changed_receipt":
        atomic_json(repository / "proof.json", {"status": "passed", "changed": True})
    elif violation == "missing_receipt":
        (repository / "proof.json").unlink()
    elif violation == "wrong_package":
        report = json.loads((verification / "verification.json").read_text())
        report["package_id"] = "sha256:" + "0" * 64
        atomic_json(verification / "verification.json", report)
    elif violation in {"answer_key", "secret"}:
        fixture = next((verification / "fixtures").glob("*.json"))
        value = json.loads(fixture.read_text())
        value["response"][
            "acceptable_candidate_ids" if violation == "answer_key" else "api_key"
        ] = ["private-value"]
        atomic_json(fixture, value)
    elif violation == "traversal":
        ledger["evidence"]["fixture"]["path"] = "../proof.json"
    elif violation == "wrong_lock":
        lock = tmp_path / "wrong-lock.json"
        atomic_json(lock, {"policy": {"policy_id": "another-policy"}})
        options["execution_lock"] = lock
    elif violation == "wrong_claim_audit":
        audit = tmp_path / "wrong-audit.json"
        atomic_json(audit, {"records": [{"explanation_id": "different-package"}]})
        options["claim_audit"] = audit
    elif violation == "failed_measurement":
        ledger["gates"]["G4"] = {"status": "passed", "evidence_ids": ["fixture"]}
        report = json.loads((verification / "verification.json").read_text())
        report["checks"]["bounded_reads"] = "failed"
        atomic_json(verification / "verification.json", report)
    else:
        ledger["gates"]["G1"]["status"] = "passed"
    atomic_json(ledger_path, ledger)
    destination = tmp_path / "rejected"
    with pytest.raises((ValueError, DomainError)):
        publish_handoff(
            package, verification, ledger_path, destination, repository_root=repository, **options
        )
    assert not destination.exists()


@pytest.fixture
def ontology_resource(prepared, tmp_path):  # noqa: F811
    from exact_inspect.context import OntologyContext
    from exact_inspect.context_resources import export_ontology_resource

    lock, _, output, _ = prepared
    stage = (
        output / json.loads((output / "preparation.json").read_text())["outputs"]["context-index"]
    )
    locator = next(iter(json.loads((stage / "contexts.json").read_text()).values()))
    resource = tmp_path / "ontology.ofn"
    export_ontology_resource(OntologyContext(stage / locator), resource, lock.policy)
    return resource


def test_handoff_publishes_ontology_without_importing_parser_or_models(
    handoff_inputs, ontology_resource, http_receipt, tmp_path
):
    package, verification, ledger, repository = handoff_inputs
    destination = tmp_path / "parser-free-handoff"
    script = """import importlib.abc, runpy, sys
forbidden = ('pyowl_core', 'torch', 'transformers', 'sentence_transformers', 'openai')
class NoPreparationImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            raise AssertionError('Publication imported preparation dependency: ' + fullname)
sys.meta_path.insert(0, NoPreparationImports())
runpy.run_module('tools.publish_explanation_handoff', run_name='__main__')
assert not any(name.split('.')[0] in forbidden for name in sys.modules)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "--package",
            str(package),
            "--verification-dir",
            str(verification),
            "--status-ledger",
            str(ledger),
            "--repository-root",
            str(repository),
            "--output-dir",
            str(destination),
            "--ontology-resource",
            str(ontology_resource),
            "--http-verification",
            str(http_receipt),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads((destination / "backend-handoff.json").read_text())
    assert result["http_runtime"]["source_receipt_sha256"] == file_hash(http_receipt)
    identity, resource = next(iter(result["ontology_resources"].items()))
    assert resource["path"] == "resources/" + identity.removeprefix("sha256:") + ".ofn"
    assert file_hash(destination / resource["path"]) == resource["sha256"]
    assert "forbidden-target" not in (destination / resource["path"]).read_text()
    assert resource["receipt_path"] == resource["path"] + ".receipt.json"
    artifacts = {artifact["path"]: artifact for artifact in result["inventory"]}
    assert artifacts[resource["path"]]["role"] == "ontology_resource"
    assert artifacts[resource["receipt_path"]]["role"] == "ontology_resource_receipt"
    assert "reference/deploy/render/exact_inspect_prepared.Dockerfile" in artifacts
    assert "reference/deploy/render/exact_inspect_prepared_requirements.txt" in artifacts
    assert (destination / result["package"]["path"]).resolve() == package.resolve()
    assert result["acceptance_status"] == "blocked_input" and not result["frontend_admitted"]


@pytest.mark.parametrize("mismatch", ["context", "ontology", "policy", "bytes", "duplicate"])
def test_handoff_rejects_ontology_resource_mismatch(
    handoff_inputs, ontology_resource, tmp_path, mismatch
):
    package, verification, ledger, repository = handoff_inputs
    receipt_path = ontology_resource.with_suffix(".ofn.receipt.json")
    receipt = json.loads(receipt_path.read_text())
    resources = [ontology_resource]
    if mismatch == "context":
        receipt["source_context_sha256"] = "sha256:" + "0" * 64
    elif mismatch == "ontology":
        receipt["ontology_version_id"] = "sha256:" + "0" * 64
    elif mismatch == "policy":
        from exact_inspect.contracts import VisibilityPolicy

        receipt["policy"]["policy_id"] = "another-policy"
        receipt["policy_hash"] = VisibilityPolicy.model_validate(receipt["policy"]).policy_hash
    elif mismatch == "bytes":
        ontology_resource.write_text("Ontology()")
    else:
        resources.append(ontology_resource)
    receipt.pop("receipt_hash")
    receipt["receipt_hash"] = canonical_hash(receipt)
    atomic_json(receipt_path, receipt)
    destination = tmp_path / "rejected-resource"
    with pytest.raises(ValueError):
        publish_handoff(
            package,
            verification,
            ledger,
            destination,
            repository_root=repository,
            ontology_resources=resources,
        )
    assert not destination.exists()


@pytest.fixture
def http_receipt(handoff_inputs, tmp_path):
    package, _, _, _ = handoff_inputs
    receipt = {
        "schema_version": "exact-explain-http-runtime/1",
        "package_id": json.loads(package.read_text())["package_id"],
        "status": "passed",
        "hardware_profile": "alternate-128GiB",
        "storage_profile": "local-ssd",
        "scope": "Fresh process with OS page cache not flushed; synthetic test measurements.",
        "checks": {
            name: "passed"
            for name in (
                "cold_service_health",
                "cold_entity_queries",
                "bounded_reads",
                "warm_read_p95",
                "serving_memory",
                "no_runtime_imports",
            )
        },
        "runtime": {
            "python": "3.12.3",
            "packages": {"fastapi": "0.116.2", "httpx": "0.28.1"},
            "hardware": {"logical_cpus": 12, "memory_bytes": 128 * 1024**3},
            "executable": "/private-runtime/python",
        },
        "measurements": {
            "cold_health_seconds": 9.0,
            "cold_entity_queries": [
                {
                    "seconds": 0.1,
                    "status": 200,
                    "bytes": 1000,
                    "within_response_budget": True,
                    "query": {"iri": "private-query"},
                    "error": "private-error",
                }
            ],
            "context_p95_seconds": 0.2,
            "pair_p95_seconds": 0.3,
            "peak_rss_bytes": 64 * 1024**2,
            "memory_source": "child /proc/PID/status VmHWM",
            "query_count": 100,
            "readers": 4,
            "log_path": "/private-log",
        },
        "command": ["private-command", "--api-key", "private-secret"],
        "command_shell": "private-command-shell",
        "query_plan": [{"query": "private-query-plan"}],
        "child": {"stdout": "private-child-output"},
    }
    path = tmp_path / "private-http-runtime.json"
    atomic_json(path, receipt)
    return path


def test_handoff_binds_separate_public_http_measurements(handoff_inputs, http_receipt, tmp_path):
    package, verification, ledger_path, repository = handoff_inputs
    ledger = json.loads(ledger_path.read_text())
    ledger["gates"]["G4"] = {"status": "passed", "evidence_ids": ["fixture"]}
    atomic_json(ledger_path, ledger)
    destination = tmp_path / "http-handoff"
    result = publish_handoff(
        package,
        verification,
        ledger_path,
        destination,
        repository_root=repository,
        http_verification=http_receipt,
    )
    http = result["http_runtime"]
    assert http["source_receipt_sha256"] == file_hash(http_receipt)
    assert http["hardware_profile"] == "alternate-128GiB"
    assert http["storage_profile"] == "local-ssd"
    assert "OS page cache not flushed" in http["scope"]
    assert http["runtime"]["hardware"]["memory_bytes"] == 128 * 1024**3
    assert http["measurements"]["cold_health_seconds"] == 9.0
    assert http["measurements"]["cold_entity_queries"] == [
        {"seconds": 0.1, "status": 200, "bytes": 1000, "within_response_budget": True}
    ]
    original = json.loads((verification / "verification.json").read_text())
    assert result["resource_measurements"] == original["measurements"]
    assert result["runtime"] == original["runtime"]
    public = destination / http["receipt"]["path"]
    assert file_hash(public) == http["receipt"]["sha256"]
    assert json.loads(public.read_text()) == {
        key: value for key, value in http.items() if key != "receipt"
    }
    assert (
        next(item for item in result["inventory"] if item["path"] == http["receipt"]["path"])[
            "role"
        ]
        == "http_runtime_verification"
    )
    assert "private-" not in public.read_text()
    assert "private-" not in json.dumps(http)


@pytest.mark.parametrize(
    "violation", ["package", "schema", "missing_check", "failed_check", "status"]
)
def test_handoff_rejects_incompatible_http_evidence(
    handoff_inputs, http_receipt, tmp_path, violation
):
    package, verification, ledger_path, repository = handoff_inputs
    ledger = json.loads(ledger_path.read_text())
    ledger["gates"]["G4"] = {"status": "passed", "evidence_ids": ["fixture"]}
    atomic_json(ledger_path, ledger)
    value = json.loads(http_receipt.read_text())
    if violation == "package":
        value["package_id"] = "sha256:" + "0" * 64
    elif violation == "schema":
        value["schema_version"] = "unknown"
    elif violation == "missing_check":
        value["checks"].pop("cold_service_health")
    elif violation == "failed_check":
        value["checks"]["cold_service_health"] = "failed"
    else:
        value["status"] = "failed"
    atomic_json(http_receipt, value)
    destination = tmp_path / "rejected-http"
    with pytest.raises(ValueError, match="HTTP verification"):
        publish_handoff(
            package,
            verification,
            ledger_path,
            destination,
            repository_root=repository,
            http_verification=http_receipt,
        )
    assert not destination.exists()


def test_handoff_preserves_failed_http_receipt_without_promoting_unrun_gate(
    handoff_inputs, http_receipt, tmp_path
):
    package, verification, ledger, repository = handoff_inputs
    value = json.loads(http_receipt.read_text())
    value["status"] = "failed"
    value["checks"]["cold_service_health"] = "failed"
    atomic_json(http_receipt, value)
    result = publish_handoff(
        package,
        verification,
        ledger,
        tmp_path / "failed-http",
        repository_root=repository,
        http_verification=http_receipt,
    )
    assert result["http_runtime"]["status"] == "failed"
    assert result["http_runtime"]["checks"]["cold_service_health"] == "failed"
    assert result["gates"]["G4"]["result"] == "not_run"
