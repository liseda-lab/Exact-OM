import copy

import pytest

from tests.compact_graph_features_test import _prepare
from tools.compact_graph_features import compact_shard
from tools.prepared_batch import binding, read, write
from tools.recover_training_retention import (
    verify_code,
    verify_raw_shards,
    verify_terminal,
)


def _snapshot(tmp_path):
    cell = tmp_path / "cell"
    directory = cell / "fitting" / ("a" * 64)
    directory.mkdir(parents=True)
    _, path, options = _prepare(directory)
    options["manifest_directory"] = cell / "fitting/graph-manifests"
    compact_shard(path, apply=True, **options)
    payload = read(path)
    snapshot = dict(
        schema_version=1,
        training_identity="a" * 64,
        files=[
            dict(
                file=binding(path), source_ids=payload["source_ids"], row_count=len(payload["rows"])
            )
        ],
        source_count=2,
        row_count=len(payload["rows"]),
        graph_manifests=[binding(p) for p in options["manifest_directory"].glob("*.json")],
    )
    return cell, snapshot


def test_verified_raw_shards_restore_no_finished_fit(tmp_path):
    cell, snapshot = _snapshot(tmp_path)
    outputs, count = verify_raw_shards(snapshot, cell)
    assert count == 2 and len(outputs) == 2
    assert not any(name.endswith(("graph.json", "training_scores.json")) for name in outputs)


@pytest.mark.parametrize(
    "defect", ["checksum", "membership", "duplicate", "sidecar", "identity", "count"]
)
def test_raw_snapshot_fail_closed(tmp_path, defect):
    from pathlib import Path

    cell, snapshot = _snapshot(tmp_path)
    if defect == "checksum":
        Path(snapshot["files"][0]["file"]["path"]).write_text("{}")
    elif defect == "membership":
        snapshot["files"][0]["source_ids"] = ["invented"]
    elif defect == "duplicate":
        snapshot["files"].append(copy.deepcopy(snapshot["files"][0]))
    elif defect == "sidecar":
        snapshot["graph_manifests"] = []
    elif defect == "identity":
        snapshot["training_identity"] = "b" * 64
    else:
        snapshot["row_count"] += 1
    with pytest.raises(ValueError):
        verify_raw_shards(snapshot, cell)


def test_only_storage_change_and_identical_training_identity(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    modules = {
        "exact/impl/trainer/fitting.py": "def fit_training_pool(self):\n    raw = load(path)\n    identity = fingerprint(raw)\n",
        "exact/impl/models/pair_adaptive_scorer.py": "def _runtime_fingerprint_payload(self):\n    return {'unchanged': True}\n",
        "exact/experiments/reporting.py": "value = 1\n",
    }
    for root in (old, new):
        for name, text in modules.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    path = "exact/impl/trainer/fitting.py"
    (new / path).write_text(modules[path] + "    cleanup_shards()\n")
    repair = dict(
        migration="training-storage-retention-v1",
        scientific_choices_unchanged=True,
        reporting_labels_exposed=False,
        changes={
            path: dict(before=binding(old / path)["sha256"], after=binding(new / path)["sha256"])
        },
    )
    verify_code(old, new, repair)
    (new / path).write_text(
        (new / path).read_text().replace("fingerprint(raw)", "fingerprint(changed_rows)")
    )
    repair["changes"][path]["after"] = binding(new / path)["sha256"]
    with pytest.raises(ValueError, match="identity construction"):
        verify_code(old, new, repair)


def test_source_failure_uses_bound_completion_and_wrapper_not_misleading_exit(tmp_path):
    write(
        tmp_path / "completion.json",
        dict(status="failed", exit_code=1, step_id="14372.90", dispatch_nonce="n"),
    )
    write(tmp_path / "step.json", dict(step_id="14372.90", dispatch_nonce="n"))
    (tmp_path / "wrapper.exit").write_text("75\n")
    (tmp_path / "exit-code").write_text("0\n")
    settings = dict(
        source_completion=binding(tmp_path / "completion.json"),
        source_launch=binding(tmp_path / "step.json"),
        source_wrapper_exit=binding(tmp_path / "wrapper.exit"),
    )
    verify_terminal(settings)
    (tmp_path / "wrapper.exit").write_text("0\n")
    settings["source_wrapper_exit"] = binding(tmp_path / "wrapper.exit")
    with pytest.raises(ValueError, match="terminal storage-guard"):
        verify_terminal(settings)


@pytest.mark.parametrize("failed", [False, True])
def test_import_accounted_before_science(tmp_path, monkeypatch, failed):
    from tests.prepared_batch_worker_test import _outputs, _worker
    from tools import experiment_resources, prepared_batch, recover_training_retention

    worker = _worker(tmp_path, monkeypatch)
    recipe = read(worker.path)
    recipe["training_retention_repair"] = {"fixture": True}
    write(worker.path, recipe)
    calls = []

    def migrate(*args):
        calls.append("import")
        if failed:
            raise ValueError("Raw feature checksum changed")

    def execute(*args, **kwargs):
        calls.append("execute")
        _outputs(worker)

    monkeypatch.setattr(recover_training_retention, "import_saved", migrate)
    monkeypatch.setattr(experiment_resources, "guarded_execute", execute)
    if failed:
        with pytest.raises(ValueError, match="checksum changed"):
            prepared_batch.run_recipe(worker.path)
    else:
        prepared_batch.run_recipe(worker.path)
    assert calls == (["import"] if failed else ["import", "execute"])
    item = read(worker.runtime / "budget.json")["work"][
        "preparation/" + recipe["scientific_step"] + "/training-retention/40"
    ]
    assert item["status"] == ("failed" if failed else "complete")
    assert item["tokens"] == item["requests"] == 0
