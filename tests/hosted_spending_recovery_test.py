import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.impl.trainer.checkpointing import CheckpointingMixin
from tools.prepared_batch import binding
from tools.recover_hosted_spending import (
    _ALLOWED,
    relocate_inference_checkpoint,
    verify_code,
    verify_identity,
)


def _sources(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    for root in (old, new):
        for name in _ALLOWED | {"exact/impl/scorer.py", "exact/experiments/reporting.py"}:
            if root == old and name == "exact/utils/hosted_spending.py":
                continue
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unchanged = True\n")
    for name in _ALLOWED:
        (new / name).write_text("accounting = True\n")
    repair = dict(
        migration="hosted-spending-checkpoint-v1",
        scientific_choices_unchanged=True,
        reporting_labels_exposed=False,
        changes={
            name: dict(
                before=binding(old / name)["sha256"] if (old / name).exists() else None,
                after=binding(new / name)["sha256"],
            )
            for name in _ALLOWED
        },
    )
    return old, new, repair


def test_only_exact_bound_administrative_changes_can_migrate(tmp_path):
    old, new, repair = _sources(tmp_path)
    assert (
        verify_code(old, new, repair)["files"]["exact/impl/scorer.py"]
        == binding(old / "exact/impl/scorer.py")["sha256"]
    )
    (new / "exact/llm/ledger.py").write_text("unexpected = True\n")
    with pytest.raises(ValueError, match="source hashes"):
        verify_code(old, new, repair)


@pytest.mark.parametrize("name", ["exact/impl/scorer.py", "exact/experiments/reporting.py"])
def test_unrelated_scientific_or_reporting_changes_reject_reuse(tmp_path, name):
    old, new, repair = _sources(tmp_path)
    (new / name).write_text("unexpected = True\n")
    with pytest.raises(ValueError, match="outside"):
        verify_code(old, new, repair)


@pytest.mark.parametrize(
    "field", ["inputs", "parameters", "parents", "dependencies", "seed", "role"]
)
def test_checkpoint_cannot_cross_input_config_seed_or_role(field):
    original = dict(
        artifact_id="old",
        implementation={"files": {"old": "sha"}},
        inputs={"data": "sha"},
        parameters={"cap": 300},
        parents=["input"],
        dependencies={"torch": "version"},
        seed=17,
        role="development",
    )
    expected = copy.deepcopy(original)
    expected.update(artifact_id="new", implementation={"files": {"new": "sha"}})
    verify_identity(original, expected, original["implementation"])
    expected[field] = "changed"
    with pytest.raises(ValueError, match="identity differs"):
        verify_identity(original, expected, original["implementation"])


def _checkpoint(tmp_path):
    source, target = tmp_path / "old-cell", tmp_path / "new-cell"
    routing = source / "routing" / ("a" * 64)
    routing.mkdir(parents=True)
    selection = routing / "selection.json"
    selection.write_text(json.dumps({"mode": "forced_sample", "selected_sources": ["s"]}))
    (routing / "rows.json").write_text(json.dumps({"rows": [{"source_iri": "s", "score": 0.7}]}))
    (routing / "prepass.json").write_text(json.dumps({"selection_artifact": str(selection)}))
    manifest = tmp_path / "routing-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "directory": str(routing),
                "files": {p.name: binding(p) for p in routing.iterdir()},
            }
        )
    )
    model_payload = {
        "pair_adaptive_channels": {
            "experiments": {
                "llm": {
                    "gate": {
                        "mode": "forced_sample",
                        "artifact": str(selection),
                        "forced_sample_size": 1,
                    }
                },
                "gate_artifact": {
                    **binding(selection),
                    "bytes": selection.stat().st_size,
                    "rows": None,
                },
            }
        }
    }
    fingerprint = CheckpointingMixin._hash_checkpoint_fingerprint_payload
    payload = {
        "dataset_signature": "same-dataset",
        "processed_examples": 1,
        "mappings": [{"src": "s", "tgt": "t", "score": 0.7}],
        "timing": {"inference_seconds_cumulative": 19.5},
        "checkpoint_fingerprint_payload": {
            "dataset_signature": "same-dataset",
            "models": [
                {
                    "class": "PairAdaptiveSemanticScorer",
                    "payload": model_payload,
                    "fingerprint": fingerprint(model_payload),
                }
            ],
        },
    }
    payload["checkpoint_fingerprint"] = fingerprint(payload["checkpoint_fingerprint_payload"])
    return payload, source, target, manifest


def _trainer(payload):
    trainer = CheckpointingMixin()
    trainer.dataset = SimpleNamespace(dataset_signature="same-dataset")
    trainer._checkpoint_fingerprint_payload = payload["checkpoint_fingerprint_payload"]
    trainer._checkpoint_fingerprint = payload["checkpoint_fingerprint"]
    trainer.log = lambda *args, **kwargs: None
    return trainer


def test_real_trainer_accepts_relocated_paths_and_retains_predictions(tmp_path):
    payload, source, target, manifest = _checkpoint(tmp_path)
    before = copy.deepcopy(payload)
    migrated, outputs, receipt = relocate_inference_checkpoint(
        payload, source, target, binding(manifest)
    )
    trainer = _trainer(migrated)
    checkpoint = tmp_path / "inference.json"
    checkpoint.write_text(json.dumps(payload))
    assert trainer._load_checkpoint_state(checkpoint, SimpleNamespace(name="TEST"))[2] == 0
    checkpoint.write_text(json.dumps(migrated))
    mappings, _, restored = trainer._load_checkpoint_state(checkpoint, SimpleNamespace(name="TEST"))
    assert restored == 1 and mappings == [("s", "t", 0.7)]
    assert trainer._restored_inference_seconds_cumulative == 19.5
    assert payload == before
    assert migrated["mappings"] == payload["mappings"]
    assert len(outputs) == 2 and not any(name.endswith("prepass.json") for name in outputs)
    gate = migrated["checkpoint_fingerprint_payload"]["models"][0]["payload"][
        "pair_adaptive_channels"
    ]["experiments"]
    assert (
        Path(gate["llm"]["gate"]["artifact"]) == target / "routing" / ("a" * 64) / "selection.json"
    )
    assert gate["gate_artifact"]["sha256"] == receipt["source_gate"]["sha256"]
    # Explanation references stay relative to the restored run, with no rewrite.
    payload["explanation_index_path"] = "../explanations/index.json"
    relocated, _, _ = relocate_inference_checkpoint(payload, source, target, binding(manifest))
    assert relocated["explanation_index_path"] == "../explanations/index.json"
    assert (
        target / "checkpoints" / relocated["explanation_index_path"]
    ).resolve() == target / "explanations/index.json"


def test_real_trainer_rejects_scientific_gate_change_after_relocation(tmp_path):
    payload, source, target, manifest = _checkpoint(tmp_path)
    migrated, _, _ = relocate_inference_checkpoint(payload, source, target, binding(manifest))
    expected = copy.deepcopy(migrated)
    model = expected["checkpoint_fingerprint_payload"]["models"][0]
    model["payload"]["pair_adaptive_channels"]["experiments"]["llm"]["gate"][
        "forced_sample_size"
    ] = 2
    fingerprint = CheckpointingMixin._hash_checkpoint_fingerprint_payload
    model["fingerprint"] = fingerprint(model["payload"])
    expected["checkpoint_fingerprint"] = fingerprint(expected["checkpoint_fingerprint_payload"])
    checkpoint = tmp_path / "inference.json"
    checkpoint.write_text(json.dumps(migrated))
    assert (
        _trainer(expected)._load_checkpoint_state(checkpoint, SimpleNamespace(name="TEST"))[2] == 0
    )


def test_restored_routing_prepass_reuses_rows_and_reconstructs_matching_fingerprint(tmp_path):
    import pandas as pd

    from exact.impl.trainer.fitting import TrainingPoolMixin
    from tests.grouped_fitting_test import TinyDataset, TinyScorer

    class PairAdaptiveSemanticScorer(TinyScorer):
        request_seed = 17

        def runtime_fingerprint_payload(self, **kwargs):
            artifact = getattr(self, "_gate_artifact", None)
            return {
                "pair_adaptive_channels": {
                    "experiments": {
                        "llm": copy.deepcopy(self.llm_experiment_config),
                        "gate_artifact": dict(artifact.provenance) if artifact else None,
                    }
                }
            }

        def runtime_fingerprint(self):
            return CheckpointingMixin._hash_checkpoint_fingerprint_payload(
                self.runtime_fingerprint_payload()
            )

        def _validate_gate_artifact_contract(self, mode):
            assert mode == "forced_sample"

    class Runner(TrainingPoolMixin, CheckpointingMixin):
        def _json_safe_value(self, value):
            return value

        def log(self, *args, **kwargs):
            pass

    def runner(directory):
        value = Runner()
        value.output_dir = directory
        value.dataset = TinyDataset(
            pd.DataFrame({"Src": ["a", "a", "b"], "Tgt": ["x-0", "x-1", "y-1"]})
        )
        value.model = PairAdaptiveSemanticScorer()
        value.fitting_gate_config = {
            "mode": "forced_sample",
            "forced_sample_size": 1,
            "artifact": None,
        }
        value.model.llm_experiment_config = {"gate": copy.deepcopy(value.fitting_gate_config)}
        return value

    old, new = runner(tmp_path / "old"), runner(tmp_path / "new")
    old.prepare_population_gate(batch_size=2)
    assert old.model.calls
    routing = Path(old.model.llm_experiment_config["gate"]["artifact"]).parent
    manifest = tmp_path / "routing.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "directory": str(routing),
                "files": {p.name: binding(p) for p in routing.iterdir()},
            }
        )
    )
    payload = {
        "dataset_signature": old.dataset.dataset_signature,
        "processed_examples": 1,
        "mappings": [{"src": "a", "tgt": "x-0", "score": 0.9}],
        "checkpoint_fingerprint_payload": old._checkpoint_fingerprint_payload,
        "checkpoint_fingerprint": old._checkpoint_fingerprint,
    }
    relocated, outputs, _ = relocate_inference_checkpoint(
        payload, old.output_dir, new.output_dir, binding(manifest)
    )
    for name, source in outputs.items():
        destination = new.output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    new.prepare_population_gate(batch_size=2)
    assert new.model.calls == []
    assert new._checkpoint_fingerprint == relocated["checkpoint_fingerprint"]
    assert (
        binding(Path(new.model.llm_experiment_config["gate"]["artifact"]))["sha256"]
        == binding(routing / "selection.json")["sha256"]
    )
    checkpoint = new.output_dir / "inference.json"
    checkpoint.write_text(json.dumps(relocated))
    assert new._load_checkpoint_state(checkpoint, SimpleNamespace(name="TEST"))[2] == 1


@pytest.mark.parametrize("change", ["selection", "rows", "inventory", "provenance", "fingerprint"])
def test_relocation_rejects_changed_routing_or_inconsistent_saved_fingerprint(tmp_path, change):
    payload, source, target, manifest = _checkpoint(tmp_path)
    bound = binding(manifest)
    routing = source / "routing" / ("a" * 64)
    if change in {"selection", "rows"}:
        (routing / (change + ".json")).write_text("{}")
    elif change == "inventory":
        (routing / "extra.json").write_text("{}")
    elif change == "provenance":
        model = payload["checkpoint_fingerprint_payload"]["models"][0]
        model["payload"]["pair_adaptive_channels"]["experiments"]["gate_artifact"]["sha256"] = (
            "b" * 64
        )
        fingerprint = CheckpointingMixin._hash_checkpoint_fingerprint_payload
        model["fingerprint"] = fingerprint(model["payload"])
        payload["checkpoint_fingerprint"] = fingerprint(payload["checkpoint_fingerprint_payload"])
    else:
        payload["checkpoint_fingerprint"] = "invalid"
    with pytest.raises(ValueError):
        relocate_inference_checkpoint(payload, source, target, bound)
