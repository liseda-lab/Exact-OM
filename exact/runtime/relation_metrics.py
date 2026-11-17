"""E14 typed endpoints, oracle-pair diagnostics and lossless typed submission scores."""

from __future__ import annotations

import json

import pandas as pd

from exact.io.relation_head import RELATIONS
from exact.io.relations import predict_relations
from exact.utils.provenance import sha256_file


def _frame(frame):
    aliases = {
        "Src": "SrcEntity",
        "Tgt": "TgtEntity",
        "src_iri": "SrcEntity",
        "tgt_iri": "TgtEntity",
    }
    result = frame.rename(
        columns={key: value for key, value in aliases.items() if value not in frame}
    ).copy()
    if "S_final" in result:
        result["Score"] = result.S_final
    return result


def relation_metrics(reference, predictions):
    """Typed macro F1 over the three declared relations; abstentions remain false negatives."""

    def triples(frame):
        return set(
            zip(
                frame.SrcEntity.astype(str), frame.TgtEntity.astype(str), frame.Relation.astype(str)
            )
        )

    gold, predicted = triples(reference), triples(predictions)
    per_relation = {}
    for relation in RELATIONS:
        expected = {row for row in gold if row[2] == relation}
        actual = {row for row in predicted if row[2] == relation}
        tp, fp, fn = len(expected & actual), len(actual - expected), len(expected - actual)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_relation[relation] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "P": precision,
            "R": recall,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    return {
        "macro_F1": sum(row["F1"] for row in per_relation.values()) / 3,
        "by_relation": per_relation,
        "reference_pairs": len({row[:2] for row in gold}),
        "predicted_pairs": len({row[:2] for row in predicted}),
        "unlisted_prediction_pairs": len(
            {row[:2] for row in predicted} - {row[:2] for row in gold}
        ),
        "precision_scope": "relative_to_known_reference",
        "unlisted_predictions_are_verified_negatives": False,
    }


def write_relation_diagnostics(trainer, accepted, *, options, save_scores=False):
    """Use existing native graphs; do not score models or inspect final references.

    Oracle inputs contain pairs/kinds only. Anchor discovery uses the same scored
    candidate population as normal inference, never the reporting gold pairs.
    """
    dataset = trainer.dataset
    candidates = getattr(trainer, "_final_candidate_frame", None)
    if candidates is None:
        candidates = trainer.results_df
    candidate_frame = _frame(candidates) if candidates is not None else accepted.copy()
    required = {"SrcEntity", "TgtEntity", "Score"}
    if not required <= set(candidate_frame):
        candidate_frame = accepted.copy()
    else:
        candidate_frame = candidate_frame[
            [
                name
                for name in ("SrcEntity", "TgtEntity", "Score", "SrcKind", "TgtKind")
                if name in candidate_frame
            ]
        ].copy()
    from exact.core.entities.configs.dataset import DatasetMask

    dataset_frame = getattr(dataset, "dataframe", None)
    if dataset_frame is not None and DatasetMask.prefiltered in dataset_frame:
        exact = _frame(dataset_frame[dataset_frame[DatasetMask.prefiltered].fillna(False)])
        if "Scores" in exact:
            exact["Score"] = exact.Scores
        if not exact.empty and required <= set(exact):
            candidate_frame = pd.concat(
                [
                    candidate_frame,
                    exact[
                        list(required) + [name for name in ("SrcKind", "TgtKind") if name in exact]
                    ],
                ],
                ignore_index=True,
            )
    candidate_frame = candidate_frame.drop_duplicates(["SrcEntity", "TgtEntity"])
    directory = trainer.alignment_dir
    mode = options["mode"]
    # Full distributions are model outputs, independent of threshold/cardinality.
    # No probabilities are invented for semantic abstentions or heuristic evidence.
    if save_scores and mode in {"none", "learned_three_way"} and not candidate_frame.empty:
        if mode == "none":
            probabilities = [[1.0, 0.0, 0.0] for _ in range(len(candidate_frame))]
        else:
            from exact.io.relation_head import predict_relation_head

            probabilities, _, _, _ = predict_relation_head(
                candidate_frame, dataset.source, dataset.target, options.get("artifact")
            )
        rows = [
            (row.SrcEntity, row.TgtEntity, relation, float(row.Score) * float(probability))
            for row, probabilities_row in zip(candidate_frame.itertuples(), probabilities)
            for relation, probability in zip(RELATIONS, probabilities_row)
        ]
        pd.DataFrame(rows, columns=["SrcEntity", "TgtEntity", "Relation", "Score"]).to_csv(
            directory / "relation_scores.tsv", sep="\t", index=False
        )
    role = getattr(trainer, "relation_evaluation_role", None)
    reference = getattr(dataset, "reference", None)
    if role not in {"train", "valid", "internal_check"} or reference is None or reference.empty:
        return
    gold = _frame(reference).rename(columns={"Label": "Relation"})
    labels = {
        "equivalent": "=",
        "source_subsumed_by_target": "<",
        "source_subsumes_target": ">",
        "subsumed_by": "<",
        "subsumes": ">",
    }
    gold["Relation"] = gold.Relation.map(lambda value: labels.get(str(value), str(value)))
    if not gold.Relation.isin(RELATIONS).all() or not gold.Relation.isin(["<", ">"]).any():
        return
    eligible = getattr(dataset, "eligible_source_iris", None)
    if eligible is not None:
        gold = gold[gold.SrcEntity.isin(eligible)]
    # The selected candidate score never enters oracle type prediction.
    oracle = gold[
        [name for name in ("SrcEntity", "TgtEntity", "SrcKind", "TgtKind") if name in gold]
    ].drop_duplicates()
    oracle["Score"] = 0.0
    typed = predict_relations(
        oracle, dataset.source, dataset.target, **{**options, "anchor_candidates": candidate_frame}
    )
    oracle_path = directory / "relations.oracle.tsv"
    typed.to_csv(oracle_path, sep="\t", index=False)
    gold_path = directory / "relations.reference.tsv"
    gold.to_csv(gold_path, sep="\t", index=False)
    payload = {
        "schema_version": 1,
        "evaluation_only": True,
        "reference_role": role,
        "oracle_anchor_source": "scored_candidate_population",
        "reference_sha256": sha256_file(gold_path),
        "oracle_sha256": sha256_file(oracle_path),
        "alignment_sha256": sha256_file(directory / "paper.maps_global.tsv"),
        "oracle_pairs": relation_metrics(gold, typed),
        "full_pipeline": relation_metrics(gold, accepted),
    }
    (directory / "relation_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
