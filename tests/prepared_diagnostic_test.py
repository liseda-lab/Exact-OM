"""Diagnostic prerequisites, native admission and accounting stay bound across retries."""

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.ontology import versions
from tests.prepared_batch_worker_test import _worker
from tools import prepared_batch, run_bridge_diagnostic, validate_native_owl


def _chain(root, report):
    prepared_batch.write(root / "report.json", report)
    prepared_batch.write(
        root / "diagnostic-completion.json",
        {
            "status": "complete",
            "selection_eligible": False,
            "diagnostic": prepared_batch.binding(root / "report.json"),
        },
    )
    prepared_batch.write(
        root / "completion.json",
        {
            "status": "complete",
            "exit_code": 0,
            "diagnostic": prepared_batch.binding(root / "diagnostic-completion.json"),
        },
    )
    return {
        "id": "admission",
        "completion_path": str(root / "completion.json"),
        "status_path": str(root / "status.json"),
    }


@pytest.mark.parametrize("changed", ["diagnostic-completion.json", "report.json"])
def test_dependency_validation_checks_both_nested_diagnostic_bindings(tmp_path, changed):
    run = _chain(tmp_path, {"status": "passed"})
    assert prepared_batch.completed_run(run)["status"] == "complete"
    (tmp_path / changed).write_text('{"status":"changed"}')
    with pytest.raises(ValueError, match="binding changed"):
        prepared_batch.completed_run(run)


@pytest.mark.parametrize(
    "admission_state",
    [
        "failed",
        "different_code",
        "different_input",
        "passed",
        "prepared_passed",
        "different_preparation",
        "missing_preparation",
        "unexpected_preparation",
        "different_reasoning_input",
        "different_reasoning_import",
        "different_resource_options",
    ],
)
def test_e14_requires_compatible_passed_admission_before_account_copy_or_execution(
    tmp_path, monkeypatch, admission_state
):
    worker = _worker(tmp_path, monkeypatch)
    installed = {"pyowl-core": "core-code", "pyhermit": "reasoner-code"}
    monkeypatch.setattr(versions, "ontology_execution_identity", lambda _: installed)
    input_path = tmp_path / "ontology.ofn"
    input_path.write_text("Ontology()")
    owl = {side: prepared_batch.binding(input_path) for side in ("source", "target")}
    report = {
        "status": "failed" if admission_state == "failed" else "passed",
        "installed_code": installed,
        "inputs": owl,
        "imports": {},
    }
    if admission_state == "different_code":
        report["installed_code"] = {**installed, "pyhermit": "old-code"}
    if admission_state == "different_input":
        report["inputs"] = {**owl, "target": {**owl["target"], "sha256": "0" * 64}}
    if admission_state == "different_resource_options":
        report["resource_options"] = {"max_native_symbol_index_bytes": 1024**3}
    preparation = {"path": str(tmp_path / "preparation.json"), "sha256": "a" * 64}
    uses_preparation = admission_state in {
        "prepared_passed",
        "different_preparation",
        "missing_preparation",
        "different_reasoning_input",
        "different_reasoning_import",
    }
    if uses_preparation or admission_state == "unexpected_preparation":
        derived = {side: {**value, "sha256": "b" * 64} for side, value in owl.items()}
        report.update(
            reasoning_preparation=preparation,
            reasoning_inputs=derived,
            reasoning_imports={},
        )
        monkeypatch.setitem(
            sys.modules,
            "tools.prepare_bridge_ontology",
            SimpleNamespace(verify_preparation=lambda *args: (derived, {})),
        )
        if admission_state == "different_preparation":
            report["reasoning_preparation"] = {**preparation, "sha256": "c" * 64}
        elif admission_state == "missing_preparation":
            report.pop("reasoning_preparation")
        elif admission_state == "different_reasoning_input":
            report["reasoning_inputs"] = owl
        elif admission_state == "different_reasoning_import":
            report["reasoning_imports"] = {"urn:unadmitted": owl["target"]}
    upstream = _chain(tmp_path / "admission", report)
    recipe = prepared_batch.read(worker.path)
    registry_path = Path(recipe["supervisor"]) / "registry.json"
    registry = prepared_batch.read(registry_path)
    registry["runs"].append(upstream)
    prepared_batch.write(registry_path, registry)
    protocol = tmp_path / "protocol.json"
    output = tmp_path / "diagnostic-output"
    prepared_batch.write(
        protocol,
        {
            "kind": "e14_native_known_pairs",
            "owl": owl,
            "imports": {},
            "output": str(output),
            **({"reasoning_preparation": preparation} if uses_preparation else {}),
        },
    )
    recipe.update(
        diagnostic=prepared_batch.binding(protocol),
        scientific_step="E14-bridge",
        depends_on=["admission"],
        admission_run="admission",
    )
    prepared_batch.write(worker.path, recipe)
    # This lane must work without any hosted credential, even when its inherited
    # account has historical hosted attempts that still need to be preserved.
    (Path(recipe["repository"]) / "api_key").unlink()
    calls = []

    def execute(path):
        calls.append(path)
        prepared_batch.write(output / "report.json", {"status": "complete"})
        prepared_batch.write(
            output / "completion.json",
            {
                "status": "complete",
                "selection_eligible": False,
                "diagnostic": prepared_batch.binding(output / "report.json"),
            },
        )

    monkeypatch.setattr(run_bridge_diagnostic, "run_diagnostic", execute)
    if admission_state not in {"passed", "prepared_passed"}:
        with pytest.raises(ValueError, match="admission does not match"):
            prepared_batch.run_recipe(worker.path)
        assert calls == [] and not worker.runtime.exists()
        assert prepared_batch.read(worker.parent) == worker.parent_state
        return
    prepared_batch.run_recipe(worker.path)
    assert calls == [protocol]
    account = prepared_batch.read(worker.runtime / "budget.json")
    assert account["work"]["historical/closed"] == worker.parent_state["work"]["historical/closed"]
    work = account["work"]["diagnostic/E14-bridge/40"]
    assert work["status"] == "complete"
    assert work["requests"] == work["tokens"] == work["actual_usd"] == 0
    with sqlite3.connect(worker.runtime / "openrouter/requests.sqlite3") as connection:
        assert connection.execute("SELECT request_id FROM requests").fetchall() == [
            (worker.request,)
        ]
    completion = prepared_batch.read(worker.root / "completion.json")
    assert completion["selection_eligible"] is False and completion["generate_rationales"] is False


@pytest.mark.parametrize("drift", ["installed_code", "input", "compile_setting"])
def test_native_admission_resumes_completed_ontology_and_rejects_identity_drift(
    tmp_path, monkeypatch, drift
):
    identity = {"pyhermit": "native-wheel-v1"}
    monkeypatch.setattr(versions, "ontology_execution_identity", lambda _: dict(identity))
    loads = []
    interrupted = [True]

    def load(path, **kwargs):
        loads.append(Path(path).stem)
        return Path(path).stem

    class Reasoner:
        def __init__(self, view, config):
            assert config.backend == "native" and config.require_native_pipeline
            assert config.timeout is None
            if view == "target" and interrupted[0]:
                raise RuntimeError("fixture compilation interruption")

        def is_consistent(self):
            return True

        def is_defined(self, entity):
            return True

        def entails(self, axiom):
            return True

        def diagnostics(self):
            return {"native": True}

        def dispose(self):
            pass

    monkeypatch.setitem(
        sys.modules, "pyhermit", SimpleNamespace(Reasoner=Reasoner, ReasonerConfig=SimpleNamespace)
    )
    monkeypatch.setitem(
        sys.modules,
        "pyowl_core",
        SimpleNamespace(
            load_snapshot=load,
            LoadOptions=SimpleNamespace,
            BackendPreference=SimpleNamespace(NATIVE="native"),
            MappingResolver=lambda value: value,
            IRI=lambda value: value,
            Class=lambda value: value,
            SubClassOf=lambda a, b: (a, b),
        ),
    )
    inputs = {}
    for side in ("source", "target"):
        path = tmp_path / (side + ".ofn")
        path.write_text("Ontology()")
        inputs[side] = prepared_batch.binding(path)
    recipe = {
        "kind": "e14_native_admission",
        "inputs": inputs,
        "imports": {},
        "output": str(tmp_path / "output"),
        "max_memory_bytes": 64 * 1024**2,
        "query_classes": {"source": "urn:a", "target": "urn:b"},
    }
    path = tmp_path / "recipe.json"
    prepared_batch.write(path, recipe)
    with pytest.raises(RuntimeError, match="fixture compilation interruption"):
        validate_native_owl.run_diagnostic(path)
    assert loads == ["source", "target"]
    assert not (tmp_path / "output/completion.json").exists()
    interrupted[0] = False
    result = validate_native_owl.run_diagnostic(path)
    assert result["status"] == "passed" and result["reused_ontologies"] == ["source"]
    assert loads == ["source", "target", "target"]
    if drift == "installed_code":
        identity["pyhermit"] = "native-wheel-v2"
    elif drift == "input":
        input_path = tmp_path / "source.ofn"
        input_path.write_text("Ontology(Declaration(Class(<urn:a>)))")
        recipe["inputs"]["source"] = prepared_batch.binding(input_path)
    else:
        recipe["max_compile_work"] = 100
    prepared_batch.write(path, recipe)
    with pytest.raises(ValueError, match="admission identity changed"):
        validate_native_owl.run_diagnostic(path)
    assert loads == ["source", "target", "target"]
