import copy
import hashlib
import json

import pytest

from tests.compact_graph_features_test import _prepare
from tools.compact_graph_features import compact_shard
from tools.prepared_batch import binding
from tools.recover_graph_storage import verify_compacted_shards, verify_identity


def _snapshot(tmp_path):
    cell = tmp_path / "cell"
    directory = cell / "fitting" / ("a" * 64)
    directory.mkdir(parents=True)
    _, path, options = _prepare(directory)
    options["manifest_directory"] = cell / "fitting/graph-manifests"
    receipt, _ = compact_shard(path, apply=True, **options)
    snapshot = {
        "schema_version": 1,
        "training_identity": "a" * 64,
        "files": [{"file": binding(path), "receipt": binding(options["receipt_path"])}],
    }
    return cell, path, options["receipt_path"], receipt, snapshot


def test_verified_raw_import_contains_graph_sidecars_and_no_fitted_artifacts(tmp_path):
    cell, path, _, _, snapshot = _snapshot(tmp_path)
    outputs, count = verify_compacted_shards(snapshot, cell)
    assert count == 2
    assert outputs["fitting/" + "a" * 64 + "/" + path.name] == path
    assert len(outputs) == 2  # Shared src/tgt graph manifest has one stored file.
    assert any(name.startswith("fitting/graph-manifests/") for name in outputs)
    assert not any(name.endswith("graph.json") for name in outputs)


@pytest.mark.parametrize("changed", ["row", "receipt", "sidecar", "duplicate", "identity"])
def test_recovery_rejects_unverified_or_changed_storage_inputs(tmp_path, changed):
    cell, path, receipt_path, receipt, snapshot = _snapshot(tmp_path)
    if changed == "row":
        value = json.loads(path.read_text())
        value["rows"][0]["score"] = 123.0
        path.write_text(json.dumps(value))
        receipt["compact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt["compact_bytes"] = path.stat().st_size
        receipt_path.write_text(json.dumps(receipt))
        snapshot["files"] = [{"file": binding(path), "receipt": binding(receipt_path)}]
    elif changed == "receipt":
        receipt["status"] = "prepared"
        receipt_path.write_text(json.dumps(receipt))
        snapshot["files"][0]["receipt"] = binding(receipt_path)
    elif changed == "sidecar":
        next((cell / "fitting/graph-manifests").iterdir()).write_text("{}")
    elif changed == "duplicate":
        snapshot["files"].append(copy.deepcopy(snapshot["files"][0]))
    else:
        snapshot["training_identity"] = "b" * 64
    with pytest.raises(ValueError):
        verify_compacted_shards(snapshot, cell)


def test_identity_migration_changes_only_implementation(tmp_path):
    old = {
        "artifact_id": "old",
        "implementation": {"sha256": "old"},
        "seed": 17,
        "inputs": {"ontology": "same"},
        "config": {"budget": 50},
    }
    new = {**old, "artifact_id": "new", "implementation": {"sha256": "new"}}
    verify_identity(old, new, old["implementation"], stage="extraction")
    for field, bad in [
        ("seed", 18),
        ("inputs", {"ontology": "different"}),
        ("config", {"budget": 100}),
    ]:
        with pytest.raises(ValueError, match="identity differs"):
            verify_identity(old, {**new, field: bad}, old["implementation"], stage="extraction")


def test_reviewed_shared_utility_delta_updates_both_identity_scopes(tmp_path):
    from tools.recover_graph_storage import verify_code

    old, new = tmp_path / "old", tmp_path / "new"
    modules = {
        "exact/impl/models/graph_head.py": "value = 1\n",
        "exact/utils/fitted_artifacts.py": "value = 1\n",
        "exact/impl/trainer/fitting.py": "def fit_training_pool(self):\n    raw = load(path)\n    identity = fingerprint(raw)\n",
        "exact/impl/models/pair_adaptive_scorer.py": "def _runtime_fingerprint_payload(self):\n    return {'unchanged': True}\n",
        "exact/experiments/reporting.py": "value = 1\n",
    }
    for root in (old, new):
        for name, text in modules.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    changed = ["exact/impl/models/graph_head.py", "exact/utils/fitted_artifacts.py"]
    for name in changed:
        (new / name).write_text("value = 2\n")
    repair = {
        "migration": "graph-provenance-storage-v1",
        "scientific_choices_unchanged": True,
        "reporting_labels_exposed": False,
        "changes": {
            name: {"before": binding(old / name)["sha256"], "after": binding(new / name)["sha256"]}
            for name in changed
        },
    }
    predictor, evaluator = verify_code(old, new, repair)
    assert "exact/utils/fitted_artifacts.py" in predictor["files"]
    assert "exact/utils/fitted_artifacts.py" in evaluator["files"]
    original = {
        "artifact_id": "old-eval",
        "implementation": evaluator,
        "parents": ["old-extraction"],
    }
    expected = {
        **original,
        "artifact_id": "new-eval",
        "implementation": {"sha256": "new-evaluator"},
        "parents": ["new-extraction"],
    }
    verify_identity(
        original,
        expected,
        predictor,
        stage="evaluation",
        old_extraction="old-extraction",
        old_evaluation=evaluator,
    )
    (new / "exact/experiments/reporting.py").write_text("changed_metric = True\n")
    with pytest.raises(ValueError, match="outside the reviewed"):
        verify_code(old, new, repair)


@pytest.mark.parametrize("failed", [False, True])
def test_storage_recovery_is_charged_and_failure_cannot_run_science(tmp_path, monkeypatch, failed):
    from tests.prepared_batch_worker_test import _outputs, _worker
    from tools import experiment_resources, prepared_batch, recover_graph_storage

    worker = _worker(tmp_path, monkeypatch)
    recipe = prepared_batch.read(worker.path)
    recipe["parent_run_id"] = "prior"
    recipe["graph_storage_repair"] = {"fixture": True}
    recipe["failed_finalization_interval"] = {"run_id": "prior", "start": 4, "end": 5}
    prepared_batch.write(worker.path, recipe)
    calls = []

    def migrate(*args):
        calls.append("import")
        if failed:
            raise ValueError("Compacted feature checksum changed")

    def execute(*args, **kwargs):
        calls.append("execute")
        _outputs(worker)

    monkeypatch.setattr(recover_graph_storage, "import_saved", migrate)
    monkeypatch.setattr(experiment_resources, "guarded_execute", execute)
    if failed:
        with pytest.raises(ValueError, match="checksum changed"):
            prepared_batch.run_recipe(worker.path)
    else:
        prepared_batch.run_recipe(worker.path)
    assert calls == (["import"] if failed else ["import", "execute"])
    account = prepared_batch.read(worker.runtime / "budget.json")
    work = account["work"]["preparation/" + recipe["scientific_step"] + "/graph-storage/40"]
    assert work["status"] == ("failed" if failed else "complete")
    assert work["requests"] == work["tokens"] == work["actual_usd"] == 0
    assert account["work"]["failed-finalization/prior"]["status"] == "failed"


@pytest.mark.parametrize("copy_failure", [False, True])
def test_deferred_lineage_retains_parent_until_account_and_launch_are_saved(
    tmp_path, monkeypatch, copy_failure
):
    from exact.llm.ledger import RequestLedger
    from tests.prepared_batch_worker_test import _outputs, _worker
    from tools import experiment_resources, finalize_prepared_selection, prepared_batch

    worker = _worker(tmp_path, monkeypatch)
    recipe = prepared_batch.read(worker.path)
    recipe.update(run_id="E18", parent_run_id="parent", deferred_lineage_registration=True)
    prepared_batch.write(worker.path, recipe)
    registry_path = worker.root.parent / "supervisor/registry.json"
    registry = prepared_batch.read(registry_path)
    registry["runs"][1].update(step_id="14372.40", dispatch_nonce=recipe["dispatch_nonce"])
    prepared_batch.write(registry_path, registry)
    original_copy, original_register = (
        prepared_batch.copy_account,
        finalize_prepared_selection.register_lineage,
    )
    order = []

    def copy_account(*args, **kwargs):
        assert prepared_batch.read(registry_path)["runs"][0].get("enabled", True)
        order.append("copy")
        if copy_failure:
            raise OSError(122, "quota")
        return original_copy(*args, **kwargs)

    def register(value):
        assert (worker.root / "launch.json").exists()
        assert (
            prepared_batch.read(worker.runtime / "budget.json")["work"]["historical/closed"]
            == worker.parent_state["work"]["historical/closed"]
        )
        assert (
            RequestLedger(worker.runtime / "openrouter").cached(worker.request)
            == b'{"result":"fixture"}'
        )
        order.append("register")
        return original_register(value)

    def execute(*args, **kwargs):
        assert prepared_batch.read(registry_path)["runs"][0]["enabled"] is False
        order.append("execute")
        _outputs(worker)

    monkeypatch.setattr(prepared_batch, "copy_account", copy_account)
    monkeypatch.setattr(finalize_prepared_selection, "register_lineage", register)
    monkeypatch.setattr(experiment_resources, "guarded_execute", execute)
    if copy_failure:
        with pytest.raises(OSError):
            prepared_batch.run_recipe(worker.path)
        assert order == ["copy"]
        assert prepared_batch.read(registry_path)["runs"][0].get("enabled", True)
    else:
        prepared_batch.run_recipe(worker.path)
        assert order == ["copy", "register", "execute"]


def test_deferred_lineage_resume_keeps_authoritative_account(tmp_path, monkeypatch):
    from tests.prepared_batch_worker_test import _outputs, _worker
    from tools import experiment_resources, prepared_batch

    worker = _worker(tmp_path, monkeypatch)
    recipe = prepared_batch.read(worker.path)
    recipe.update(run_id="E18", parent_run_id="parent", deferred_lineage_registration=True)
    prepared_batch.write(worker.path, recipe)
    registry_path = worker.root.parent / "supervisor/registry.json"
    registry = prepared_batch.read(registry_path)
    registry["runs"][1].update(step_id="14372.40", dispatch_nonce=recipe["dispatch_nonce"])
    prepared_batch.write(registry_path, registry)
    calls = []

    def execute(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("interrupted scientific step")
        _outputs(worker)

    monkeypatch.setattr(experiment_resources, "guarded_execute", execute)
    with pytest.raises(RuntimeError, match="interrupted"):
        prepared_batch.run_recipe(worker.path)
    retained = prepared_batch.read(worker.runtime / "budget.json")
    monkeypatch.setattr(
        prepared_batch, "copy_account", lambda *_: pytest.fail("Reimported stale parent")
    )
    monkeypatch.setenv("SLURM_STEP_ID", "41")
    registry = prepared_batch.read(registry_path)
    registry["runs"][1]["step_id"] = "14372.41"
    prepared_batch.write(registry_path, registry)
    prepared_batch.run_recipe(worker.path)
    current = prepared_batch.read(worker.runtime / "budget.json")
    assert all(current["work"][key] == value for key, value in retained["work"].items())
    assert prepared_batch.read(registry_path)["runs"][0]["superseded_by"] == "E18"
