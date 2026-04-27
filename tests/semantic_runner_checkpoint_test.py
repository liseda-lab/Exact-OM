from pathlib import Path
import importlib.util

import torch

from exact.core.contracts.model import IModel


def _load_runner_module():
    module_path = Path(__file__).resolve().parents[1] / "exact" / "impl" / "trainer" / "semantic_runner.py"
    spec = importlib.util.spec_from_file_location("semantic_runner_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyDataset:
    dataset_signature = "ontology-signature"
    cache_fingerprint = "candidate-fingerprint-v2"

    def __len__(self):
        return 0


class _DummyModel(IModel):
    def __init__(self, **kwargs):
        super().__init__()

    def runtime_fingerprint_payload(self, **kwargs):
        return {"model": "dummy"}

    def forward(self, *args, **kwargs):
        return {}


def test_primary_checkpoint_fingerprint_includes_dataset_cache_fingerprint(tmp_path: Path):
    module = _load_runner_module()
    runner = module.SemanticAlignmentRunner(
        dataset=_DummyDataset(),
        model=_DummyModel,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )

    payload = runner._build_checkpoint_fingerprint_payload()

    assert payload["dataset_signature"] == "ontology-signature"
    assert payload["dataset_fingerprint"] == "candidate-fingerprint-v2"
