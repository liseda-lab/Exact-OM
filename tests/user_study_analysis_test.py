import json
from pathlib import Path

import pandas as pd

from exact.analysis import user_study as mod


def _record(
    src: str,
    tgt: str,
    src_label: str,
    tgt_label: str,
    score: float,
    ground_truth: bool = False,
    llm_rationale: str = "",
    include_similarity: bool = False,
) -> dict:
    similarity_items = (
        [{
            "item_id": f"{src}->{tgt}:sim:source",
            "triple": [src_label, "label", "shared term"],
            "subject_iri": src,
            "object_iri": f"{src}#shared",
            "support": 0.9,
            "importance": 0.2,
        }]
        if include_similarity
        else []
    )
    similarity_target_items = (
        [{
            "item_id": f"{src}->{tgt}:sim:target",
            "triple": [tgt_label, "label", "shared term"],
            "subject_iri": tgt,
            "object_iri": f"{tgt}#shared",
            "support": 0.9,
            "importance": 0.2,
        }]
        if include_similarity
        else []
    )
    return {
        "explanation_schema_version": 3,
        "src_iri": src,
        "tgt_iri": tgt,
        "selected_labels": {"source": src_label, "target": tgt_label},
        "confidences": {
            "S_final": score,
            "s_label": max(0.0, score - 0.05),
            "S_struct": min(1.0, score - 0.02),
            "p_llm": 0.0,
        },
        "importances": {
            "I_label": 0.6,
            "I_struct": 0.4,
            "I_llm": 0.0,
            "I_hier": 0.15,
            "I_sim": 0.05 if include_similarity else 0.0,
            "I_diff": 0.12,
            "I_attr": 0.08,
        },
        "weights": {"U": 0.35, "U_dis": 0.08},
        "prediction": {
            "ground_truth": 1 if ground_truth else 0,
            "threshold_positive": score >= 0.8,
            "saved_alignment_member": ground_truth,
            "rationale_positive": ground_truth,
            "rationale_decision_label": "Match" if ground_truth else "No match",
            "llm_decision": "",
            "llm_rationale": llm_rationale,
        },
        "llm_pair_brief": f"Brief for {src_label} vs {tgt_label}",
        "backend_usage": {"summary": {"model": "summary-model"}, "decision": {"model": "decision-model"}},
        "models": {},
        "attributes": {
            "source": [
                {
                    "item_id": f"{src}->{tgt}:attr:source",
                    "property": "label",
                    "value": src_label,
                    "text": f"label: {src_label}",
                    "entity_iri": src,
                    "support": 0.8,
                    "importance": 0.12,
                }
            ],
            "target": [
                {
                    "item_id": f"{src}->{tgt}:attr:target",
                    "property": "definition",
                    "value": f"Definition for {tgt_label}",
                    "text": f"definition: Definition for {tgt_label}",
                    "entity_iri": tgt,
                    "support": 0.82,
                    "importance": 0.15,
                }
            ],
        },
        "context_triples": {
            "hierarchy_source": {"is_a": [[src_label, "is_a", f"{src_label} parent"]], "part_of": [], "has_part": []},
            "hierarchy_target": {"is_a": [[tgt_label, "is_a", f"{tgt_label} parent"]], "part_of": [], "has_part": []},
            "similarity_source": [[src_label, "label", "shared term"]] if include_similarity else [],
            "similarity_target": [[tgt_label, "label", "shared term"]] if include_similarity else [],
            "difference_source": [[src_label, "has_trait", f"{src_label} trait"]],
            "difference_target": [[tgt_label, "has_trait", f"{tgt_label} trait"]],
        },
        "triple_attributions": {
            "hierarchy": {
                "is_a": {
                    "source": [
                        {
                            "item_id": f"{src}->{tgt}:hier:source",
                            "triple": [src_label, "is_a", f"{src_label} parent"],
                            "subject_iri": src,
                            "object_iri": f"{src}#parent",
                            "support": 0.7,
                            "importance": 0.2,
                        }
                    ],
                    "target": [
                        {
                            "item_id": f"{src}->{tgt}:hier:target",
                            "triple": [tgt_label, "is_a", f"{tgt_label} parent"],
                            "subject_iri": tgt,
                            "object_iri": f"{tgt}#parent",
                            "support": 0.7,
                            "importance": 0.2,
                        }
                    ],
                },
                "part_of": {"source": [], "target": []},
                "has_part": {"source": [], "target": []},
            },
            "similarity": {
                "source": similarity_items,
                "target": similarity_target_items,
            },
            "difference": {
                "source": [
                    {
                        "item_id": f"{src}->{tgt}:diff:source",
                        "triple": [src_label, "has_trait", f"{src_label} trait"],
                        "subject_iri": src,
                        "object_iri": f"{src}#trait",
                        "support": 0.6,
                        "importance": 0.25,
                    }
                ],
                "target": [
                    {
                        "item_id": f"{src}->{tgt}:diff:target",
                        "triple": [tgt_label, "has_trait", f"{tgt_label} trait"],
                        "subject_iri": tgt,
                        "object_iri": f"{tgt}#trait",
                        "support": 0.6,
                        "importance": 0.25,
                    }
                ],
            },
        },
        "cross_side_provenance": {
            "lexical": [{"source_ref": "__source__", "target_ref": "__target__", "score": max(0.0, score - 0.05)}],
            "hierarchy": {
                "is_a": [
                    {
                        "source_item_id": f"{src}->{tgt}:hier:source",
                        "target_item_id": f"{src}->{tgt}:hier:target",
                        "score": 0.7,
                    }
                ]
            },
            "similarity": (
                [
                    {
                        "source_item_id": f"{src}->{tgt}:sim:source",
                        "target_item_id": f"{src}->{tgt}:sim:target",
                        "score": 0.9,
                    }
                ]
                if include_similarity
                else []
            ),
            "attributes": {
                "source": [
                    {
                        "item_id": f"{src}->{tgt}:attr:source",
                        "anchor_kind": "label",
                        "anchor_ref": "__target__",
                        "score": 0.8,
                    }
                ],
                "target": [
                    {
                        "item_id": f"{src}->{tgt}:attr:target",
                        "anchor_kind": "label",
                        "anchor_ref": "__source__",
                        "score": 0.82,
                    }
                ],
            },
            "difference": {
                "source": [
                    {
                        "item_id": f"{src}->{tgt}:diff:source",
                        "anchor_kind": "endpoint",
                        "anchor_ref": "__target__",
                        "score": 0.6,
                    }
                ],
                "target": [
                    {
                        "item_id": f"{src}->{tgt}:diff:target",
                        "anchor_kind": "endpoint",
                        "anchor_ref": "__source__",
                        "score": 0.6,
                    }
                ],
            },
        },
    }


def _build_run_dir(
    tmp_path: Path,
    missing_record_source: str | None = None,
    empty_rationale_source: str | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "model" / "alignment" / "default").mkdir(parents=True)
    (run_dir / "config.yaml").write_text("{}\n", encoding="utf-8")

    records = []
    ranking_rows = []
    for rank in range(1, 6):
        for source_idx in range(5):
            src = f"src-r{rank}-{source_idx}"
            src_label = f"Source rank {rank} example {source_idx}"
            candidates = []
            for cand_rank in range(1, 6):
                tgt = f"tgt-r{rank}-{source_idx}-{cand_rank}"
                score = 1.0 - (cand_rank * 0.01)
                candidates.append((tgt, score))
            raw_order = [candidates[rank - 1]] + [cand for idx, cand in enumerate(candidates) if idx != rank - 1]
            gold_tgt = candidates[rank - 1][0]
            ranking_rows.append(
                {
                    "SrcEntity": src,
                    "TgtEntity": gold_tgt,
                    "TgtCandidates": repr(raw_order),
                }
            )
            for tgt, score in candidates:
                if missing_record_source == src and tgt == candidates[0][0]:
                    continue
                llm_rationale = ""
                if empty_rationale_source != src:
                    llm_rationale = f"Rationale for {src} -> {tgt}"
                records.append(
                    _record(
                        src=src,
                        tgt=tgt,
                        src_label=src_label,
                        tgt_label=f"Target {tgt}",
                        score=score,
                        ground_truth=(tgt == gold_tgt),
                        llm_rationale=llm_rationale,
                        include_similarity=(rank == 1 and source_idx == 0 and tgt == candidates[0][0]),
                    )
                )

    pd.DataFrame(ranking_rows).to_csv(
        run_dir / "model" / "alignment" / "src2tgt.maps_local.tsv",
        sep="\t",
        index=False,
    )
    (run_dir / "model" / "alignment" / "default" / "full_explanations.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def test_loader_sorts_candidates_by_score_and_computes_rank(tmp_path):
    run_dir = _build_run_dir(tmp_path)
    artifacts = mod.load_run_analysis(run_dir=run_dir, top_k=5)
    row = artifacts.source_df.loc[artifacts.source_df["src_iri"] == "src-r2-0"].iloc[0]
    assert int(row["gold_rank"]) == 2
    assert row["top1_tgt_iri"] == "tgt-r2-0-1"


def test_incomplete_panel_is_excluded_but_missing_rationale_is_allowed(tmp_path):
    run_dir = _build_run_dir(
        tmp_path,
        missing_record_source="src-r3-0",
        empty_rationale_source="src-r2-0",
    )
    artifacts = mod.load_run_analysis(run_dir=run_dir, top_k=5)
    eligible = mod._eligible_panels(artifacts.source_df, top_k=5)
    assert "src-r3-0" not in set(eligible["src_iri"])
    assert "src-r2-0" in set(eligible["src_iri"])


def test_failure_taxonomy_assignment_order():
    source_df = pd.DataFrame(
        [
            {
                "src_iri": "s-missing",
                "source_label": "s",
                "gold_tgt_iri": "g",
                "gold_tgt_label": "g",
                "top1_tgt_iri": "t",
                "top1_tgt_label": "t",
                "gold_rank": 2,
                "missing_panel_record_count": 1,
                "panel_complete": False,
                "score_gap": 0.1,
                "top1_score": 0.9,
                "gold_score": 0.8,
                "delta_s_label": -0.2,
                "delta_S_struct": 0.1,
                "delta_I_label": -0.1,
                "delta_I_struct": 0.1,
                "delta_U": 0.0,
                "delta_U_dis": 0.0,
                "delta_hierarchy_count": 0,
                "delta_difference_count": 0,
                "delta_similarity_count": 0,
                "delta_attribute_count": 0,
                "top1_I_struct": 0.2,
                "gold_I_struct": 0.2,
                "top1_hierarchy_count": 0,
                "top1_similarity_count": 0,
                "top1_difference_count": 0,
                "top1_attribute_count": 0,
                "gold_hierarchy_count": 0,
                "gold_similarity_count": 0,
                "gold_difference_count": 0,
                "gold_attribute_count": 0,
                "gold_U_dis": 0.0,
                "top1_U_dis": 0.0,
                "panel_has_similarity": False,
                "top1_record_present": False,
                "gold_record_present": True,
                "top1_llm_active": False,
                "gold_llm_active": False,
            },
            {
                "src_iri": "s-below",
                "source_label": "s",
                "gold_tgt_iri": "g",
                "gold_tgt_label": "g",
                "top1_tgt_iri": "t",
                "top1_tgt_label": "t",
                "gold_rank": 6,
                "missing_panel_record_count": 0,
                "panel_complete": True,
                "score_gap": 0.1,
                "top1_score": 0.9,
                "gold_score": 0.7,
                "delta_s_label": -0.2,
                "delta_S_struct": 0.1,
                "delta_I_label": -0.1,
                "delta_I_struct": 0.1,
                "delta_U": 0.0,
                "delta_U_dis": 0.0,
                "delta_hierarchy_count": 0,
                "delta_difference_count": 0,
                "delta_similarity_count": 0,
                "delta_attribute_count": 0,
                "top1_I_struct": 0.2,
                "gold_I_struct": 0.2,
                "top1_hierarchy_count": 0,
                "top1_similarity_count": 0,
                "top1_difference_count": 0,
                "top1_attribute_count": 0,
                "gold_hierarchy_count": 0,
                "gold_similarity_count": 0,
                "gold_difference_count": 0,
                "gold_attribute_count": 0,
                "gold_U_dis": 0.0,
                "top1_U_dis": 0.0,
                "panel_has_similarity": False,
                "top1_record_present": True,
                "gold_record_present": True,
                "top1_llm_active": False,
                "gold_llm_active": False,
            },
            {
                "src_iri": "s-near",
                "source_label": "s",
                "gold_tgt_iri": "g",
                "gold_tgt_label": "g",
                "top1_tgt_iri": "t",
                "top1_tgt_label": "t",
                "gold_rank": 2,
                "missing_panel_record_count": 0,
                "panel_complete": True,
                "score_gap": 0.015,
                "top1_score": 0.905,
                "gold_score": 0.89,
                "delta_s_label": -0.01,
                "delta_S_struct": 0.03,
                "delta_I_label": -0.1,
                "delta_I_struct": 0.1,
                "delta_U": 0.0,
                "delta_U_dis": 0.0,
                "delta_hierarchy_count": 0,
                "delta_difference_count": 0,
                "delta_similarity_count": 0,
                "delta_attribute_count": 0,
                "top1_I_struct": 0.4,
                "gold_I_struct": 0.5,
                "top1_hierarchy_count": 2,
                "top1_similarity_count": 0,
                "top1_difference_count": 1,
                "top1_attribute_count": 1,
                "gold_hierarchy_count": 2,
                "gold_similarity_count": 0,
                "gold_difference_count": 1,
                "gold_attribute_count": 1,
                "gold_U_dis": 0.0,
                "top1_U_dis": 0.0,
                "panel_has_similarity": False,
                "top1_record_present": True,
                "gold_record_present": True,
                "top1_llm_active": False,
                "gold_llm_active": False,
            },
            {
                "src_iri": "s-gold-struct",
                "source_label": "s",
                "gold_tgt_iri": "g",
                "gold_tgt_label": "g",
                "top1_tgt_iri": "t",
                "top1_tgt_label": "t",
                "gold_rank": 2,
                "missing_panel_record_count": 0,
                "panel_complete": True,
                "score_gap": 0.05,
                "top1_score": 0.9,
                "gold_score": 0.85,
                "delta_s_label": -0.01,
                "delta_S_struct": 0.03,
                "delta_I_label": -0.1,
                "delta_I_struct": 0.1,
                "delta_U": 0.0,
                "delta_U_dis": 0.0,
                "delta_hierarchy_count": 1,
                "delta_difference_count": 0,
                "delta_similarity_count": 0,
                "delta_attribute_count": 0,
                "top1_I_struct": 0.4,
                "gold_I_struct": 0.5,
                "top1_hierarchy_count": 2,
                "top1_similarity_count": 0,
                "top1_difference_count": 1,
                "top1_attribute_count": 1,
                "gold_hierarchy_count": 3,
                "gold_similarity_count": 0,
                "gold_difference_count": 1,
                "gold_attribute_count": 1,
                "gold_U_dis": 0.0,
                "top1_U_dis": 0.0,
                "panel_has_similarity": False,
                "top1_record_present": True,
                "gold_record_present": True,
                "top1_llm_active": False,
                "gold_llm_active": False,
            },
            {
                "src_iri": "s-top1-struct",
                "source_label": "s",
                "gold_tgt_iri": "g",
                "gold_tgt_label": "g",
                "top1_tgt_iri": "t",
                "top1_tgt_label": "t",
                "gold_rank": 2,
                "missing_panel_record_count": 0,
                "panel_complete": True,
                "score_gap": 0.07,
                "top1_score": 0.9,
                "gold_score": 0.83,
                "delta_s_label": -0.01,
                "delta_S_struct": -0.04,
                "delta_I_label": -0.1,
                "delta_I_struct": -0.1,
                "delta_U": 0.0,
                "delta_U_dis": 0.0,
                "delta_hierarchy_count": -1,
                "delta_difference_count": 0,
                "delta_similarity_count": 0,
                "delta_attribute_count": 0,
                "top1_I_struct": 0.5,
                "gold_I_struct": 0.3,
                "top1_hierarchy_count": 3,
                "top1_similarity_count": 0,
                "top1_difference_count": 1,
                "top1_attribute_count": 1,
                "gold_hierarchy_count": 2,
                "gold_similarity_count": 0,
                "gold_difference_count": 1,
                "gold_attribute_count": 1,
                "gold_U_dis": 0.0,
                "top1_U_dis": 0.0,
                "panel_has_similarity": False,
                "top1_record_present": True,
                "gold_record_present": True,
                "top1_llm_active": False,
                "gold_llm_active": False,
            },
            {
                "src_iri": "s-lexical",
                "source_label": "s",
                "gold_tgt_iri": "g",
                "gold_tgt_label": "g",
                "top1_tgt_iri": "t",
                "top1_tgt_label": "t",
                "gold_rank": 2,
                "missing_panel_record_count": 0,
                "panel_complete": True,
                "score_gap": 0.08,
                "top1_score": 0.92,
                "gold_score": 0.84,
                "delta_s_label": -0.05,
                "delta_S_struct": 0.01,
                "delta_I_label": -0.1,
                "delta_I_struct": 0.1,
                "delta_U": 0.0,
                "delta_U_dis": 0.0,
                "delta_hierarchy_count": 0,
                "delta_difference_count": 0,
                "delta_similarity_count": 0,
                "delta_attribute_count": 0,
                "top1_I_struct": 0.4,
                "gold_I_struct": 0.4,
                "top1_hierarchy_count": 1,
                "top1_similarity_count": 0,
                "top1_difference_count": 1,
                "top1_attribute_count": 1,
                "gold_hierarchy_count": 1,
                "gold_similarity_count": 0,
                "gold_difference_count": 1,
                "gold_attribute_count": 1,
                "gold_U_dis": 0.0,
                "top1_U_dis": 0.0,
                "panel_has_similarity": False,
                "top1_record_present": True,
                "gold_record_present": True,
                "top1_llm_active": False,
                "gold_llm_active": False,
            },
            {
                "src_iri": "s-fallback",
                "source_label": "s",
                "gold_tgt_iri": "g",
                "gold_tgt_label": "g",
                "top1_tgt_iri": "t",
                "top1_tgt_label": "t",
                "gold_rank": 2,
                "missing_panel_record_count": 0,
                "panel_complete": True,
                "score_gap": 0.08,
                "top1_score": 0.92,
                "gold_score": 0.84,
                "delta_s_label": -0.01,
                "delta_S_struct": 0.0,
                "delta_I_label": -0.1,
                "delta_I_struct": 0.0,
                "delta_U": 0.0,
                "delta_U_dis": 0.0,
                "delta_hierarchy_count": 0,
                "delta_difference_count": 0,
                "delta_similarity_count": 0,
                "delta_attribute_count": 0,
                "top1_I_struct": 0.4,
                "gold_I_struct": 0.4,
                "top1_hierarchy_count": 1,
                "top1_similarity_count": 0,
                "top1_difference_count": 1,
                "top1_attribute_count": 1,
                "gold_hierarchy_count": 1,
                "gold_similarity_count": 0,
                "gold_difference_count": 1,
                "gold_attribute_count": 1,
                "gold_U_dis": 0.0,
                "top1_U_dis": 0.0,
                "panel_has_similarity": False,
                "top1_record_present": True,
                "gold_record_present": True,
                "top1_llm_active": False,
                "gold_llm_active": False,
            },
        ]
    )
    pair_df = pd.DataFrame([{"nonlex_total": 2}, {"nonlex_total": 6}, {"nonlex_total": 8}])
    failure_df = mod._failure_taxonomy(source_df, pair_df, top_k=5)
    categories = dict(zip(failure_df["src_iri"], failure_df["primary_failure_category"]))
    assert categories["s-missing"] == "missing_panel_record"
    assert categories["s-below"] == "gold_below_top5"
    assert categories["s-near"] == "near_tie"
    assert categories["s-gold-struct"] == "gold_structurally_better_but_loses"
    assert categories["s-top1-struct"] == "top1_structurally_better"
    assert categories["s-lexical"] == "lexical_overweight"
    assert categories["s-fallback"] == "evidence_sparse_other"


def test_backfill_rationales_updates_only_missing_records(tmp_path, monkeypatch):
    records = [
        _record("s1", "t1", "Source 1", "Target 1", 0.9, ground_truth=True, llm_rationale="existing rationale"),
        _record("s1", "t2", "Source 1", "Target 2", 0.8, ground_truth=False, llm_rationale=""),
    ]

    class DummyModel:
        def __init__(self):
            self._last_rationale_backend_meta = {"backend": "openrouter", "model": "gpt-test"}

        def generate_final_rationales_for_records(self, rows, progress_callback=None):
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "rationale",
                        "event": "start",
                        "total_records": len(rows),
                        "cached_records": 0,
                        "uncached_records": len(rows),
                        "uncached_unique_prompts": len(rows),
                        "backend": "openrouter",
                        "model": "gpt-test",
                        "concurrency": 1,
                    }
                )
                progress_callback(
                    {
                        "stage": "rationale",
                        "event": "progress",
                        "total_records": len(rows),
                        "cached_records": 0,
                        "total_uncached_records": len(rows),
                        "completed_uncached_records": len(rows),
                        "total_unique_prompts": len(rows),
                        "completed_unique_prompts": len(rows),
                    }
                )
            return [f"generated:{row['tgt_iri']}" for row in rows]

    monkeypatch.setattr(mod, "_build_rationale_model", lambda configs, cache_dir, device, logger: DummyModel())
    updated = mod._backfill_rationales(
        records,
        configs=object(),
        output_dir=tmp_path,
        logger=mod._setup_logger(),
        generate_rationales=True,
    )
    assert updated[0]["prediction"]["llm_rationale"] == "existing rationale"
    assert updated[1]["prediction"]["llm_rationale"] == "generated:t2"
    assert updated[1]["backend_usage"]["rationale"]["model"] == "gpt-test"
    assert updated[1]["models"]["llm_rationale_model"] == "gpt-test"


def test_backfill_explanation_fields_repairs_from_saved_record_only(tmp_path, monkeypatch):
    original = _record("s1", "t1", "Source 1", "Target 1", 0.9, ground_truth=True)
    damaged = json.loads(json.dumps(original))
    damaged.pop("explanation_schema_version", None)
    damaged.pop("cross_side_provenance", None)
    damaged["triple_attributions"]["hierarchy"]["is_a"]["source"][0].pop("item_id", None)
    damaged["attributes"]["source"][0].pop("item_id", None)

    class DummyModel:
        def __init__(self, repaired):
            self.repaired = repaired
            self.targeted_calls = 0

        def reconstruct_explanation_fields_from_record(self, record):
            return json.loads(json.dumps(self.repaired))

        def reconstruct_explanation_fields_for_pair(self, *args, **kwargs):
            self.targeted_calls += 1
            raise AssertionError("targeted pair rehydrate should not be used")

    class DummyConfigs:
        class _ModelSpec:
            class name:
                __name__ = "PairAdaptiveSemanticScorer"

        def get_model_sequence(self):
            return [self._ModelSpec()]

    monkeypatch.setattr(mod, "_build_explanation_backfill_model", lambda *args, **kwargs: DummyModel(original))
    updated = mod._backfill_explanation_fields(
        [damaged],
        run_dir=tmp_path,
        output_dir=tmp_path,
        logger=mod._setup_logger(),
        configs=DummyConfigs(),
        backfill_explanations=True,
    )
    assert "explanation_schema_version" not in damaged
    assert updated[0]["explanation_schema_version"] == 3
    assert updated[0]["cross_side_provenance"]["lexical"]
    assert updated[0]["triple_attributions"]["hierarchy"]["is_a"]["source"][0]["item_id"]
    assert updated[0]["attributes"]["source"][0]["item_id"]
    assert updated[0]["prediction"] == damaged["prediction"]


def test_backfill_explanation_fields_can_fall_back_to_targeted_pair_rehydrate(tmp_path, monkeypatch):
    original = _record("s1", "t1", "Source 1", "Target 1", 0.9, ground_truth=True)
    damaged = json.loads(json.dumps(original))
    damaged.pop("explanation_schema_version", None)
    damaged.pop("cross_side_provenance", None)
    damaged["triple_attributions"] = {}
    damaged["attributes"] = {}

    state = {"targeted_calls": 0}

    class DummyModel:
        def reconstruct_explanation_fields_from_record(self, record):
            raise RuntimeError("saved record is insufficient")

        def reconstruct_explanation_fields_for_pair(self, *args, **kwargs):
            state["targeted_calls"] += 1
            return json.loads(json.dumps(original))

    class DummyConfigs:
        class _ModelSpec:
            class name:
                __name__ = "PairAdaptiveSemanticScorer"

        def get_model_sequence(self):
            return [self._ModelSpec()]

    monkeypatch.setattr(mod, "_build_explanation_backfill_model", lambda *args, **kwargs: DummyModel())
    updated = mod._backfill_explanation_fields(
        [damaged],
        run_dir=tmp_path,
        output_dir=tmp_path,
        logger=mod._setup_logger(),
        configs=DummyConfigs(),
        backfill_explanations=True,
    )
    assert state["targeted_calls"] == 1
    assert updated[0]["explanation_schema_version"] == 3
    assert updated[0]["cross_side_provenance"]["hierarchy"]["is_a"]
    assert updated[0]["attributes"]["target"][0]["item_id"]


def test_full_pipeline_writes_balanced_mapping_and_notebook(tmp_path):
    run_dir = _build_run_dir(tmp_path)
    output_dir = tmp_path / "analysis"
    outputs = mod.run_user_study_analysis(
        run_dir=run_dir,
        output_dir=output_dir,
        top_k=5,
        per_rank=4,
        shortlist_per_rank=4,
        generate_rationales=False,
    )
    mapping = json.loads((output_dir / "study_mapping.json").read_text(encoding="utf-8"))
    assert "pairs" in mapping
    assert len(mapping["pairs"]) == 20
    saw_context_edge = False
    saw_core_bridge = False
    saw_supporting_bridge = False
    saw_bridge_edge = False
    for pair in mapping["pairs"]:
        assert len(pair["paths"]) == 5
        for path in pair["paths"]:
            assert set(["id", "rank", "ground_truth", "score", "metrics", "llm", "nodes", "edges"]).issubset(path.keys())
            assert set(
                [
                    "decision_basis",
                    "evidence_strength",
                    "evidence_agreement",
                    "explanation_coverage",
                    "lead_over_next_candidate",
                ]
            ).issubset(path["metrics"].keys())
            assert path["nodes"][0]["type"] == "Source"
            assert path["nodes"][-1]["type"] == "Target"
            assert all(
                set(["source", "target", "label", "score", "type", "bridge", "level", "level_label"]).issubset(edge.keys())
                for edge in path["edges"]
            )
            assert all(isinstance(edge["bridge"], bool) for edge in path["edges"])
            assert all(isinstance(edge["level"], int) for edge in path["edges"])
            assert all(isinstance(edge["level_label"], str) and edge["level_label"] for edge in path["edges"])
            assert {edge["type"] for edge in path["edges"]}.issubset(
                {
                    "hierarchy",
                    "similarity",
                    "difference",
                    "attribute",
                    "bridge-support",
                    "bridge-contrast",
                }
            )
            for edge in path["edges"]:
                if str(edge["type"]).startswith("bridge-"):
                    assert edge["score"] in {"weak", "moderate", "strong"}
                    assert edge["bridge"] is True
                    assert edge["level"] in {2, 3, 4}
                    assert edge["level_label"] in {"Core bridge", "Supporting bridge", "Optional bridge"}
                else:
                    assert isinstance(edge["score"], (int, float))
                    assert edge["bridge"] is False
                    assert edge["level"] == 1
                    assert edge["level_label"] == "Context edge"
            bridge_keys = [
                (edge["source"], edge["target"], edge["type"])
                for edge in path["edges"]
                if str(edge["type"]).startswith("bridge-")
            ]
            assert len(bridge_keys) == len(set(bridge_keys))
            saw_context_edge = saw_context_edge or any(not edge["bridge"] for edge in path["edges"])
            saw_core_bridge = saw_core_bridge or any(edge["level"] == 2 for edge in path["edges"] if edge["bridge"])
            saw_supporting_bridge = saw_supporting_bridge or any(edge["level"] in {3, 4} for edge in path["edges"] if edge["bridge"])
            saw_bridge_edge = saw_bridge_edge or any(str(edge["type"]).startswith("bridge-") for edge in path["edges"])
            node_types = {node["type"] for node in path["nodes"]}
            assert node_types.issubset(
                {
                    "Source",
                    "Target",
                    "source-context",
                    "target-context",
                }
            )
            context_node_ids = {
                node["id"]
                for node in path["nodes"]
                if node["type"] in {"source-context", "target-context"}
            }
            assert all(not node_id.startswith("label: ") for node_id in context_node_ids)
            assert all(not node_id.startswith("definition: ") for node_id in context_node_ids)
    assert saw_context_edge
    assert saw_core_bridge
    assert saw_supporting_bridge
    assert saw_bridge_edge
    selected_records = json.loads((output_dir / "study_selected_records.json").read_text(encoding="utf-8"))
    assert len(selected_records) == 100
    assert Path(outputs["notebook"]).exists()
    notebook = json.loads(Path(outputs["notebook"]).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
