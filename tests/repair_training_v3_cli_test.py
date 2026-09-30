"""Exercise the v3 CLI resource wiring without launching a scientific run."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from exact.repair.workers import CallResult
from tests.repair_training_completion_test import cache_for
from tools.repair import train
from tools.repair.corpus import generate_corpus
from tools.repair.prepare import load_preparation


def test_v3_cli_accounts_shared_campaign_and_stage_cpu_and_preserves_resume(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    template = (
        Path(__file__).parents[1] / "specs/exact-repair/protocol/xr21-review2-conformance.json"
    )
    protocol = json.loads(template.read_text().replace('"UNFROZEN"', '"conformance-fixture"'))
    protocol["identity"]["execution_authorized"] = True
    protocol["corpus"].update(
        families=["range"],
        generated_objects=[1],
        groups_per_family={"train": 1, "development": 1, "test": 0},
        corruptions_per_group=1,
        clean_controls_per_group=0,
    )
    # An allocated device accrues wall cost even though these mocked workers run
    # no CUDA operations. Only the shared campaign owns this allowance.
    protocol["resources"]["allocated_gpus"] = 1
    protocol["resources"]["campaign_gpu_hours"] = 1.0
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    output = tmp_path / "run"
    args = ["repair-train", "--protocol", str(path), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 0},
        siblings_per_parent=1,
        families=("range",),
    )
    calls = []

    def worker(function, *worker_args, **options):
        calls.append((function.__name__, options))
        if function.__name__ == "generated_from_protocol":
            value, cpu = cases, 0.25
        elif function is train.label_case:
            value = replace(cache_for(worker_args[0]), schema="exact-repair/teacher-cache/v3")
            cpu = 0.5
        elif function is train._train_payload:
            train_rows, dev_rows, model_options = worker_args
            assert len(train_rows) == len(dev_rows) == 1
            assert model_options["revision"] == "v3"
            assert model_options["deadline_seconds"] < options["timeout"]
            descriptor = train._publish_training_checkpoint(
                {
                    "config": {"feature_dim": 8},
                    "state_dict": {},
                    "metadata": (),
                    "model_schema": "exact-repair/model/v3",
                },
                output / "checkpoints",
            )
            value = (descriptor, {"schema": "exact-repair/training/v3", "status": "fixture"})
            cpu = 0.75
        else:
            raise AssertionError(function)
        return CallResult(
            "complete",
            value,
            resource_usage=(
                ("cpu_seconds", cpu),
                ("wall_seconds", 0.01),
                ("peak_sampled_tree_rss_bytes", 1024),
            ),
        )

    monkeypatch.setattr(train, "bounded_call", worker)
    assert train.main() == 0
    stages = {
        name: json.loads((output / file).read_text())
        for name, file in (
            ("corpus", "corpus-budget.json"),
            ("label", "label-budget.json"),
            ("train", "training-budget.json"),
        )
    }
    assert [name for name, _ in calls] == [
        "generated_from_protocol",
        "label_case",
        "label_case",
        "_train_payload",
    ]
    assert [options["cpu_seconds"] for _, options in calls] == [600, 300, 300, 1800]
    assert all(options["memory_mb"] == 8192 for _, options in calls)
    assert {name: ledger["spent_cpu_seconds"] for name, ledger in stages.items()} == {
        "corpus": 0.25,
        "label": 1.0,
        "train": 0.75,
    }
    assert all(
        "active" not in ledger and ledger["allocated_gpus"] == 0 for ledger in stages.values()
    )
    assert all(
        attempt["cpu_accounting"] == "observed"
        for ledger in stages.values()
        for attempt in ledger["attempts"]
    )
    campaign = json.loads((output / "campaign-budget.json").read_text())
    assert "active" not in campaign and campaign["allocated_gpus"] == 1
    assert campaign["spent_gpu_hours"] == pytest.approx(campaign["spent_seconds"] / 3600)
    assert len(campaign["attempts"]) == 1
    restored_cases, caches, preparation = load_preparation(output / "preparation.json")
    assert len(restored_cases) == len(caches) == 2
    assert preparation["label_cpu_seconds"] == 1.0
    assert (output / "model.pt").exists()
    report = json.loads((output / "report.json").read_text())
    assert report["measured_training_resources"]["cpu_seconds"] == 0.75
    assert report["selected_checkpoint_artifact"]["schema"] == "exact-repair/checkpoint-artifact/v3"

    # Replacement preparation reuses compatible manifests and retains all CPU
    # charges. Its small coordinator overhead still accrues campaign GPU wall.
    monkeypatch.setattr(sys, "argv", [*args, "--prepare-only"])
    calls.clear()
    assert train.main() == 0
    assert not calls
    for name, file in (
        ("corpus", "corpus-budget.json"),
        ("label", "label-budget.json"),
        ("train", "training-budget.json"),
    ):
        assert json.loads((output / file).read_text()) == stages[name]
    resumed = json.loads((output / "campaign-budget.json").read_text())
    assert len(resumed["attempts"]) == 2
    assert resumed["spent_seconds"] >= campaign["spent_seconds"]
    assert resumed["spent_gpu_hours"] >= campaign["spent_gpu_hours"]


def test_v3_selected_checkpoint_uses_small_descriptor_and_checks_artifact(tmp_path, monkeypatch):
    """A selected model larger than the worker frame never enters that frame."""
    import pickle
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    # Just larger than the 16 MiB result frame: no actual training is performed.
    weights = torch.zeros(4_200_000)
    model = SimpleNamespace(
        metadata=(),
        config={"revision": "v3", "feature_dim": 8},
        state_dict=lambda: {"large": weights},
    )
    monkeypatch.setattr(
        train, "train_cases", lambda *args, **kwargs: (model, {"status": "fixture"})
    )
    descriptor, report = train._train_payload(
        [], [], {"revision": "v3", "checkpoint_path": tmp_path / "training-state.pt"}
    )
    assert len(pickle.dumps((descriptor, report))) < 1024
    assert descriptor["size_bytes"] > 16 * 1024 * 1024
    restored = train._read_training_checkpoint(descriptor, tmp_path / "checkpoints")
    assert restored["state_dict"]["large"].device.type == "cpu"
    assert torch.equal(restored["state_dict"]["large"], weights)
    path = Path(descriptor["path"])
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    with pytest.raises(ValueError, match="integrity mismatch"):
        train._read_training_checkpoint(descriptor, tmp_path / "checkpoints")
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        train._read_training_checkpoint(descriptor, tmp_path / "checkpoints")
    with pytest.raises(ValueError, match="outside its declared output"):
        train._read_training_checkpoint(descriptor, tmp_path / "other")
