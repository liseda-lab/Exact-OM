import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from exact.core.actions.alignment import run_alignment
from exact.core.entities.configs.config import ConfigModel
from exact.impl.trainer import SemanticAlignmentRunner
from exact.runs import RunLayout


@pytest.mark.parametrize("mode", ["global_alignment", "local_ranking"])
def test_public_action_uses_declared_mode_with_same_frozen_pool(tmp_path, monkeypatch, mode):
    fixtures = Path(__file__).parent / "fixtures" / "ontologies"
    source, target = "http://example.org/mini/src#", "http://example.org/mini/tgt#"
    pool = tmp_path / "pool.tsv"
    pd.DataFrame(
        [
            (source + "Heart", target + "CardiacOrgan"),
            (source + "Heart", target + "BodyOrgan"),
            (source + "AnatomicalEntity", target + "CardiacOrgan"),
        ],
        columns=["SrcEntity", "TgtEntity"],
    ).to_csv(pool, sep="\t", index=False)
    calls = []
    original = SemanticAlignmentRunner.predict

    def predict(self, **kwargs):
        calls.append({**kwargs, "effective_extraction": dict(self._extraction_config)})
        return original(self, **kwargs)

    monkeypatch.setattr(SemanticAlignmentRunner, "predict", predict)
    config = ConfigModel.from_mapping(
        {
            "config_version": 2,
            "run": {"use_file_cache": False, "seed": 17},
            "data": {"execution_mode": mode, "candidate_provenance": "frozen_generated"},
            "dataset": {
                "filter_exact_matches": False,
                "which": [],
                "projector": {"backend": "native"},
            },
            "matching": {
                "threshold": 0.1,
                "extraction": {"mode": "mutual_best", "anchor_conflict_policy": "compete"},
            },
            "llm": {"verbaliser": {"model": None}},
            "pipeline": [
                {
                    "name": "PairAdaptiveSemanticScorer",
                    "params": {
                        "use_lexical": False,
                        "use_context": False,
                        "use_llm": False,
                        "llm_model_name": None,
                        "persist_cache_to_disk": False,
                    },
                }
            ],
            "inference": {"batch_size": 4, "mixed_precision": False, "which": []},
            "output": {"sanity_checks": {"enabled": False}},
        }
    )
    run_dir = tmp_path / mode
    run_alignment(
        source_file_path=fixtures / "mini_src.owl",
        target_file_path=fixtures / "mini_tgt.owl",
        output_dir_path=run_dir,
        candidates_file_path=pool,
        configs_file_path=config,
    )
    assert calls[0]["local_alignment"] == (mode == "local_ranking")
    assert calls[0]["effective_extraction"] == config.matching.extraction.for_execution_mode(
        mode
    ).model_dump(mode="python")
    assert config.matching.extraction.mode == "mutual_best"
    assert config.matching.extraction.anchor_conflict_policy == "compete"
    stats = json.loads(RunLayout.open(run_dir).run_stats_path.read_text())
    assert stats["execution"]["mode"] == mode
    assert stats["execution"]["candidate_provenance"] == "frozen_generated"
    assert stats["execution"]["global_extraction"] == (mode == "global_alignment")
    assert stats["execution"]["extraction_scope"] == {
        "declared_global": config.matching.extraction.model_dump(mode="python"),
        "effective": calls[0]["effective_extraction"],
        "global_policy_applied": mode == "global_alignment",
    }
    if mode == "local_ranking":
        assert "cardinality" not in calls[0] and "target_cardinality" not in calls[0]
        local = pd.read_csv(run_dir / "alignment/maps_local.tsv", sep="\t")
        reached = {
            (row.SrcEntity, target)
            for row in local.itertuples()
            for target, _score in ast.literal_eval(row.TgtCandidates)
        }
        assert reached == set(pd.read_csv(pool, sep="\t").itertuples(index=False, name=None))
