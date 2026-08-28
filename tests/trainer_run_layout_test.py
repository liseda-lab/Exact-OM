import json
from ast import literal_eval
from pathlib import Path

import pandas as pd
import pytest
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
    store = ExplanationStore(runner.run_layout.explanations_dir, compression="none")
    store.append([legacy_explanation])
    runner._explanation_store = store
    stored_before_save = store.get("source")

    paths = runner.save_results(
        [EntityMapping("source", "target", score=0.9)],
        sub_dir="legacy-task-name",
        save_json=False,
    )

    assert paths["alignment_tsv"] == runner.run_layout.mapping_path("global")
    assert "alignment_global_audit" not in paths
    assert not (runner.alignment_dir / "paper.maps_global.tsv").exists()
    assert paths["summary_csv"] == runner.run_layout.summary_metrics_path
    assert paths["run_stats_json"] == runner.run_layout.run_stats_path
    assert not (runner.output_dir / "model").exists()
    assert legacy_explanation == {
        "src_iri": "source",
        "tgt_iri": "target",
        "confidences": {"S_final": 0.9},
        "prediction": {},
    }
    assert store.get("source") == stored_before_save


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
        paper_audit=True,
    )

    assert paths["alignment_tsv"] == paths["alignment_typed_tsv"]
    typed = pd.read_csv(paths["alignment_typed_tsv"], sep="\t")
    assert typed["Relation"].tolist() == [
        "equivalent",
        "source_subsumed_by_target",
    ]
    assert paths["alignment_json"].is_file()
    assert not runner.run_layout.mapping_path("global").exists()
    assert paths["alignment_global_audit"].is_file()
    explanation = {item["tgt_iri"]: item for item in store.get(KG_BASE + "atrium")}[
        KG_BASE + "heart"
    ]
    assert explanation["relation"] == "<"
    assert explanation["prediction"]["relation"] == "<"
    assert 0.0 <= explanation["relation_confidence"] <= 1.0


def test_semantic_abstention_rejects_mapping_across_all_output_surfaces(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = _Dataset()
    dataset.source = CsvKgSource.from_path(KG_FIXTURE)
    dataset.target = dataset.source
    dataset.candidate_pool_manifest = {"gold_free_summary": {"source_entities": 3}}
    runner = SemanticAlignmentRunner(
        dataset=dataset,
        model=_Model,
        device=torch.device("cpu"),
        output_dir=tmp_path / "semantic-abstention-run",
    )
    accepted_key = (KG_BASE + "organ", KG_BASE + "organ")
    abstained_key = (KG_BASE + "blood", KG_BASE + "missing")
    pairs = [
        EntityMapping(*accepted_key, score=0.99),
        EntityMapping(*abstained_key, score=0.88),
    ]
    records = [
        {
            "src_iri": mapping.head,
            "tgt_iri": mapping.tail,
            "confidences": {"S_final": mapping.score},
            "prediction": {
                "saved_alignment_member": True,
                "rationale_positive": True,
                "rationale_decision_label": "Match",
            },
        }
        for mapping in pairs
    ]
    runner.results_json.extend(records)
    runner.results_df = runner._make_summary_dataframe(records)
    store = ExplanationStore(runner.run_layout.explanations_dir, compression="none")
    store.append(records)
    runner._explanation_store = store

    candidate_path = tmp_path / "local-candidates.tsv"
    pd.DataFrame(
        [
            {
                "SrcEntity": source,
                "TgtEntity": target,
                "TgtCandidates": repr([target]),
            }
            for source, target in (accepted_key, abstained_key)
        ]
    ).to_csv(candidate_path, sep="\t", index=False)
    accepted_evidence = '{"backend":"graph_closure","path":["src","tgt"]}'
    abstention_evidence = '{"forward":[],"reverse":[]}'

    def fake_relation_typer(candidates, source, target, **kwargs):
        typed = candidates.iloc[[0]].copy()
        typed["Relation"] = "<"
        typed["relation_confidence"] = 0.84
        typed["relation_semantic_backend"] = "graph_closure"
        typed["relation_evidence"] = accepted_evidence
        typed.attrs["relation_abstentions"] = [
            {
                "source": abstained_key[0],
                "target": abstained_key[1],
                "reason": "not_entailed",
                "forward_confidence": 0.2,
                "reverse_confidence": 0.1,
                "relation_evidence": abstention_evidence,
            }
        ]
        return typed

    monkeypatch.setattr(
        "exact.core.contracts.trainer.type_alignment_relations",
        fake_relation_typer,
    )
    paths = runner.save_results(
        pairs,
        candidates_one2many_path=candidate_path,
        output_formats=["tsv-local", "typed-tsv"],
        relation_prediction="semantic_entailment",
        save_json=False,
        save_stats_csv=False,
    )

    assert [(mapping.head, mapping.tail) for mapping in pairs] == [accepted_key]
    typed = pd.read_csv(paths["alignment_typed_tsv"], sep="\t")
    assert list(zip(typed["SrcEntity"], typed["TgtEntity"])) == [accepted_key]
    local = pd.read_csv(paths["alignment_tsv_local"], sep="\t")
    local_scores = {
        row.SrcEntity: literal_eval(row.TgtCandidates)[0][1]
        for row in local.itertuples(index=False)
    }
    assert local_scores[accepted_key[0]] == 0.99
    assert local_scores[abstained_key[0]] == 0.0

    by_key = {(record["src_iri"], record["tgt_iri"]): record for record in runner.results_json}
    accepted = by_key[accepted_key]
    assert accepted["relation"] == "<"
    assert accepted["relation_evidence"] == accepted_evidence
    assert accepted["relation_semantic_backend"] == "graph_closure"
    assert accepted["prediction"]["saved_alignment_member"] is True
    abstained = by_key[abstained_key]
    assert abstained["relation"] is None
    assert abstained["prediction"]["saved_alignment_member"] is False
    assert abstained["prediction"]["rationale_positive"] is False
    assert abstained["relation_abstention_reason"] == "not_entailed"
    assert abstained["relation_evidence"] == abstention_evidence
    assert abstained["relation_abstention"] == {
        "forward_confidence": 0.2,
        "reason": "not_entailed",
        "relation_evidence": abstention_evidence,
        "reverse_confidence": 0.1,
    }

    summary = runner.results_df.set_index(["src_iri", "tgt_iri"])
    assert bool(summary.loc[accepted_key, "saved_alignment_member"]) is True
    assert summary.loc[accepted_key, "relation_semantic_backend"] == "graph_closure"
    assert summary.loc[accepted_key, "relation_evidence"] == accepted_evidence
    assert bool(summary.loc[abstained_key, "saved_alignment_member"]) is False
    assert bool(summary.loc[abstained_key, "relation_abstained"]) is True
    stored = {(record["src_iri"], record["tgt_iri"]): record for record in store.iter_all()}
    assert stored[accepted_key]["relation_evidence"] == accepted_evidence
    assert stored[accepted_key]["relation_semantic_backend"] == "graph_closure"
    assert stored[abstained_key]["prediction"]["saved_alignment_member"] is False
    assert stored[abstained_key]["relation_abstention_reason"] == "not_entailed"
    assert stored[abstained_key]["relation_evidence"] == abstention_evidence

    stats = json.loads(paths["run_stats_json"].read_text(encoding="utf-8"))
    assert stats["coverage"] == pytest.approx(1 / 3)
    assert stats["abstention_rate"] == pytest.approx(2 / 3)
    assert stats["decision_source_counts"] == {
        "declared": 3,
        "observed": 2,
        "accepted": 1,
        "unscored": 1,
    }
    assert stats["metric_applicability"]["coverage"] is True


def test_trainer_forwards_semantic_relation_configuration_and_anchors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = _Dataset()
    dataset.source = CsvKgSource.from_path(KG_FIXTURE)
    dataset.target = dataset.source
    runner = SemanticAlignmentRunner(
        dataset=dataset,
        model=_Model,
        device=torch.device("cpu"),
        output_dir=tmp_path / "semantic-run",
    )
    anchors = pd.DataFrame([{"Src": KG_BASE + "heart", "Tgt": KG_BASE + "heart", "Score": 0.99}])
    captured = {}

    def fake_relation_typer(candidates, source, target, **kwargs):
        captured.update(
            {
                "source": source,
                "target": target,
                **kwargs,
            }
        )
        typed = candidates.copy()
        typed["Relation"] = "="
        typed["relation_confidence"] = 0.73
        return typed

    monkeypatch.setattr(
        "exact.core.contracts.trainer.type_alignment_relations",
        fake_relation_typer,
    )
    runner.save_results(
        [EntityMapping(KG_BASE + "heart", KG_BASE + "heart", score=0.99)],
        output_formats=["typed-tsv"],
        relation_prediction="semantic_entailment",
        relation_anchors=anchors,
        relation_semantic_backend="bridge_reasoner",
        relation_equivalence_anchor_threshold=0.91,
        relation_equivalence_anchor_margin=0.17,
        relation_confidence_threshold=0.61,
        relation_reasoning_timeout_seconds=47.5,
        save_json=False,
        save_csv=False,
        save_stats_csv=False,
    )

    assert captured.pop("anchors") is anchors
    assert captured == {
        "source": dataset.source,
        "target": dataset.target,
        "mode": "semantic_entailment",
        "semantic_backend": "bridge_reasoner",
        "equivalence_anchor_threshold": 0.91,
        "equivalence_anchor_margin": 0.17,
        "relation_confidence_threshold": 0.61,
        "timeout_seconds": 47.5,
    }
