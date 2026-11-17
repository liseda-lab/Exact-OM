from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from exact.core.entities.kinds import EntityKind
from exact.impl.datasets.pair_adaptive_context import PairAdaptiveContextDataset
from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer
from exact.impl.trainer import SemanticAlignmentRunner

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"
SOURCE_PATH = FIXTURES / "mini_src.owl"
TARGET_PATH = FIXTURES / "mini_tgt.owl"
SRC = "http://example.org/mini/src#"
TGT = "http://example.org/mini/tgt#"


def test_mixed_kind_runner_persists_and_exports_typed_explanations(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "mixed-kind-run"
    kinds = [
        EntityKind.CLASS,
        EntityKind.OBJECT_PROPERTY,
        EntityKind.DATA_PROPERTY,
        EntityKind.INDIVIDUAL,
    ]
    dataset = PairAdaptiveContextDataset(
        output_path=run_dir,
        entity_kinds=kinds,
        only_taxonomy=True,
        verbaliser_name=None,
        projection_include_literals=True,
        cache_ok=False,
    )
    dataset.load_ontologies(SOURCE_PATH, TARGET_PATH)
    dataset._candidates = pd.DataFrame(
        [
            (SRC + "Heart", TGT + "CardiacOrgan", EntityKind.CLASS.value),
            (
                SRC + "participatesIn",
                TGT + "enrolledIn",
                EntityKind.OBJECT_PROPERTY.value,
            ),
            (SRC + "hasCode", TGT + "identifier", EntityKind.DATA_PROPERTY.value),
            (SRC + "alice", TGT + "bob", EntityKind.INDIVIDUAL.value),
        ],
        columns=["Src", "Tgt", "SrcKind"],
    )
    dataset._candidates["TgtKind"] = dataset._candidates["SrcKind"]
    dataset._candidates["Label"] = 0
    dataset._candidates["cand_sim"] = 1.0
    dataset.process()

    runner = SemanticAlignmentRunner(
        dataset=dataset,
        model=PairAdaptiveSemanticScorer,
        model_params={
            "use_lexical": False,
            "use_context": False,
            "use_llm": False,
            "llm_model_name": None,
            "return_explanations": True,
            "persist_cache_to_disk": False,
        },
        device=torch.device("cpu"),
        output_dir=run_dir,
    )
    predictions, _ = runner.predict(
        threshold=0.0,
        cardinality=None,
        target_cardinality=None,
        batch_size=4,
        num_workers=0,
        enable_checkpoints=False,
        audit_shard_compression="none",
        mixed_precision=False,
        log_every=100,
    )
    paths = runner.save_results(
        predictions,
        output_formats=["json"],
        save_json=True,
        save_csv=False,
        save_stats_csv=False,
    )

    expected_kinds = {kind.value for kind in kinds}
    assert {mapping.kind.value for mapping in predictions} == expected_kinds
    alignment_rows = json.loads(paths["alignment_json"].read_text(encoding="utf-8"))
    assert {row["Kind"] for row in alignment_rows} == expected_kinds

    stored = list(runner._explanation_store.iter_all())
    assert len(stored) == 4
    assert {record["kind"] for record in stored} == expected_kinds
    assert all(record["kind"] == record["src_kind"] == record["tgt_kind"] for record in stored)

    exported = json.loads(runner.run_layout.full_explanations_path.read_text(encoding="utf-8"))
    assert {record["kind"] for record in exported} == expected_kinds
