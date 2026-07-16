from pathlib import Path

import pandas as pd
import torch

from exact.core.contracts.model import IModel
from exact.core.entities.mappings import EntityMapping
from exact.impl.trainer import SemanticAlignmentRunner


class _Dataset:
    dataset_signature = "layout-fixture"

    def __len__(self) -> int:
        return 0


class _Model(IModel):
    def forward(self, *args, **kwargs):
        return {}


def test_trainer_writes_only_canonical_layout_v2_paths(tmp_path: Path) -> None:
    runner = SemanticAlignmentRunner(
        dataset=_Dataset(),
        model=_Model,
        device=torch.device("cpu"),
        output_dir=tmp_path / "run",
    )
    runner.results_df = pd.DataFrame(
        [{"src_iri": "source", "tgt_iri": "target", "S_final": 0.9}]
    )

    paths = runner.save_results(
        [EntityMapping("source", "target", score=0.9)],
        sub_dir="legacy-task-name",
        save_json=False,
    )

    assert paths["alignment_tsv"] == runner.run_layout.mapping_path("global")
    assert paths["summary_csv"] == runner.run_layout.summary_metrics_path
    assert paths["run_stats_json"] == runner.run_layout.run_stats_path
    assert not (runner.output_dir / "model").exists()
