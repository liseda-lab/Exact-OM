import hashlib
import json
from pathlib import Path

import torch

from exact.core.contracts.model import IModel
from exact.core.entities.configs.dataset import DatasetMask
from exact.impl.trainer import SemanticAlignmentRunner


class FixtureDataset:
    dataset_signature = "wp-d-fixture"
    cache_fingerprint = "wp-d-candidates-v1"

    def __len__(self) -> int:
        return 2


class FixtureModel(IModel):
    def __init__(self, **kwargs):
        super().__init__()

    def runtime_fingerprint_payload(self, **kwargs):
        return {"model": "fixture-v1"}

    def runtime_fingerprint(self) -> str:
        payload = json.dumps(self.runtime_fingerprint_payload(), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def forward(self, *args, **kwargs):
        return {}


def test_checkpoint_written_before_decomposition_restores(tmp_path: Path) -> None:
    runner = SemanticAlignmentRunner(
        dataset=FixtureDataset(),
        model=FixtureModel,
        device=torch.device("cpu"),
        output_dir=tmp_path,
    )
    checkpoint = Path(__file__).parent / "fixtures" / "checkpoints" / "baseline_v2_full.json"

    mappings, records, processed = runner._load_checkpoint_state(
        checkpoint,
        DatasetMask.inference,
    )

    assert processed == 2
    assert records == []
    assert mappings == [
        ("https://example.org/source/A", "https://example.org/target/A", 0.91),
        ("https://example.org/source/B", "https://example.org/target/B", 0.73),
    ]
    assert runner.inference_seconds_cumulative == 12.5
    assert runner.examples_per_second_ema == 0.16
