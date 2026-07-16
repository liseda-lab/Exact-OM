from pathlib import Path

import pandas as pd
import torch

from exact.core.contracts.model import IModel
from exact.core.entities.mappings import EntityMapping
from exact.impl.trainer import SemanticAlignmentRunner
from exact.io.sources.csv_kg import CsvKgSource
from exact.runs import ExplanationStore

KG_FIXTURE = Path(__file__).parent / "fixtures" / "kg_csv"
KG_BASE = "http://example.org/kg/"


class _Dataset:
    dataset_signature = "layout-fixture"
    source: CsvKgSource
    target: CsvKgSource

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
    legacy_explanation = {
        "src_iri": "source",
        "tgt_iri": "target",
        "confidences": {"S_final": 0.9},
        "prediction": {},
    }
    runner.results_json.append(legacy_explanation)
    runner.results_df = pd.DataFrame([{"src_iri": "source", "tgt_iri": "target", "S_final": 0.9}])

    paths = runner.save_results(
        [EntityMapping("source", "target", score=0.9)],
        sub_dir="legacy-task-name",
        save_json=False,
    )

    assert paths["alignment_tsv"] == runner.run_layout.mapping_path("global")
    assert paths["summary_csv"] == runner.run_layout.summary_metrics_path
    assert paths["run_stats_json"] == runner.run_layout.run_stats_path
    assert not (runner.output_dir / "model").exists()
    assert legacy_explanation == {
        "src_iri": "source",
        "tgt_iri": "target",
        "confidences": {"S_final": 0.9},
        "prediction": {},
    }


def test_trainer_dispatches_typed_formats_and_persists_relation_metadata(
    tmp_path: Path,
) -> None:
    dataset = _Dataset()
    dataset.source = CsvKgSource.from_path(KG_FIXTURE)
    dataset.target = dataset.source
    runner = SemanticAlignmentRunner(
        dataset=dataset,
        model=_Model,
        device=torch.device("cpu"),
        output_dir=tmp_path / "typed-run",
    )
    pairs = [
        EntityMapping(KG_BASE + "atrium", KG_BASE + "atrium", score=0.99),
        EntityMapping(KG_BASE + "atrium", KG_BASE + "heart", score=0.75),
    ]
    records = [
        {
            "src_iri": mapping.head,
            "tgt_iri": mapping.tail,
            "confidences": {"S_final": mapping.score},
            "prediction": {},
        }
        for mapping in pairs
    ]
    runner.results_json.extend(records)
    runner.results_df = runner._make_summary_dataframe(records)
    store = ExplanationStore(runner.run_layout.explanations_dir, compression="none")
    store.append(records)
    runner._explanation_store = store

    paths = runner.save_results(
        pairs,
        output_formats=["typed-tsv", "json"],
        relation_prediction="hierarchy_heuristic",
        save_json=False,
    )

    assert paths["alignment_tsv"] == paths["alignment_typed_tsv"]
    typed = pd.read_csv(paths["alignment_typed_tsv"], sep="\t")
    assert typed["Relation"].tolist() == [
        "equivalent",
        "source_subsumed_by_target",
    ]
    assert paths["alignment_json"].is_file()
    assert not runner.run_layout.mapping_path("global").exists()
    explanation = {item["tgt_iri"]: item for item in store.get(KG_BASE + "atrium")}[
        KG_BASE + "heart"
    ]
    assert explanation["relation"] == "<"
    assert explanation["prediction"]["relation"] == "<"
    assert 0.0 <= explanation["relation_confidence"] <= 1.0
