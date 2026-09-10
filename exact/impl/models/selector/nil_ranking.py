"""Common probability scale for real candidates and an explicit NIL item."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Sequence

import pandas as pd

from .grouping import iter_source_groups


@dataclass(frozen=True)
class JointNilDistribution:
    """Pre-normalization probabilities and their joint categorical scale."""

    real_probabilities: tuple[float, ...]
    nil_probability: float
    real_q: tuple[float, ...]
    nil_q: float


def joint_nil_distribution(
    real_probabilities: Sequence[float],
    nil_probability: float,
    *,
    eps: float = 1.0e-6,
) -> JointNilDistribution:
    """Apply softmax to candidate/NIL logits as required by E04.

    Inputs must already be probabilities from the same acceptance semantics;
    accepting raw matcher scores here would make the NIL comparison invalid.
    """

    real = tuple(float(value) for value in real_probabilities)
    p_nil = float(nil_probability)
    if not real:
        raise ValueError("joint NIL ranking requires at least one real candidate")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in (*real, p_nil)):
        raise ValueError("real and NIL inputs must be finite probabilities in [0, 1]")
    epsilon = float(eps)
    if not math.isfinite(epsilon) or epsilon <= 0.0 or epsilon >= 0.5:
        raise ValueError("eps must be finite and between zero and 0.5")

    def _logit(probability: float) -> float:
        clipped = min(1.0 - epsilon, max(epsilon, probability))
        return math.log(clipped / (1.0 - clipped))

    logits = [_logit(value) for value in (*real, p_nil)]
    maximum = max(logits)
    masses = [math.exp(value - maximum) for value in logits]
    denominator = sum(masses)
    q = [value / denominator for value in masses]
    return JointNilDistribution(
        real_probabilities=real,
        nil_probability=p_nil,
        real_q=tuple(float(value) for value in q[:-1]),
        nil_q=float(q[-1]),
    )


class NilRankingMixin:
    """Attach a joint real/NIL distribution without changing global acceptance."""

    def _apply_joint_nil_ranking(self, df: pd.DataFrame, dataset=None) -> pd.DataFrame:
        mode = str(self.nil_config.get("mode", "off")).strip().lower()
        if mode == "off":
            self._nil_meta = {"mode": "off", "applied": False}
            return df
        if mode == "fitted":
            import json
            from pathlib import Path

            from .nil_head import source_decision_records

            path = self.nil_config.get("artifact")
            if not path or not Path(path).is_file():
                raise ValueError("Fitted natural NIL requires a training-source artifact")
            artifact = json.loads(Path(path).read_text())
            if artifact["application"].get("dataset_signature") != getattr(
                dataset, "dataset_signature", None
            ):
                raise ValueError("NIL artifact application dataset mismatch")
            records = source_decision_records(df, df.Src.astype(str).unique(), artifact=artifact)
            for record in records:
                indices = df.index[df.Src.astype(str) == record["Src"]]
                probabilities = df.loc[indices, "P_rank"].astype(float).clip(lower=0)
                conditional = (
                    probabilities / float(probabilities.sum())
                    if float(probabilities.sum()) > self.eps
                    else pd.Series(1.0 / len(indices), index=indices)
                )
                for index, value in conditional.items():
                    df.at[index, "Q_match"] = record["in_pool_probability"] * value
                    df.at[index, "Q_nil"] = record["ontology_nil_probability"]
                    df.at[index, "Q_pool_miss"] = record["pool_miss_probability"]
                    df.at[index, "P_rank"] = record["in_pool_probability"] * value
                    df.at[index, "nil_absence_semantics"] = record["absence_semantics"]
                    df.at[index, "nil_ranking_scale"] = artifact["probability_scale"]
                    df.at[index, "selection_nil_winner"] = (
                        record["absence_semantics"] == "ontology_nil"
                    )
                    if record["action"] == "abstain":
                        df.at[index, "S_select"] = 0.0
                        df.at[index, "selection_abstained"] = True
            self._nil_meta = {
                "mode": mode,
                "applied": True,
                "artifact": str(path),
                "source_groups": len(records),
                "probability_scale": artifact["probability_scale"],
            }
            return df
        ranking_scale = (
            str(self.nil_config.get("ranking_scale", "joint_accept_probability")).strip().lower()
        )
        if ranking_scale != "joint_accept_probability":
            raise ValueError("NIL ranking scale must be 'joint_accept_probability'")

        nil_winner_groups = 0
        group_count = 0
        for _, _, _, group in iter_source_groups(df):
            group_count += 1
            indices = list(group.index)
            real_probabilities = [float(group.at[index, "P_match"]) for index in indices]
            nil_probabilities = [
                float(group.at[index, "selection_no_match_prob"]) for index in indices
            ]
            if max(nil_probabilities) - min(nil_probabilities) > self.eps:
                raise ValueError("NIL probability must be source-level and identical on its rows")
            distribution = joint_nil_distribution(
                real_probabilities,
                nil_probabilities[0],
                eps=self.eps,
            )
            ranked: List[tuple[float, str, Any]] = [
                (-distribution.real_q[position], str(group.at[index, "Tgt"]), index)
                for position, index in enumerate(indices)
            ]
            # Candidate IRI wins an exact probability tie with NIL. This is a
            # deterministic conservative abstention tie rule and is persisted.
            ranked.append((-distribution.nil_q, "\uffffNIL", None))
            rank_by_index = {
                index: rank for rank, (_, _, index) in enumerate(sorted(ranked), start=1)
            }
            nil_winner = rank_by_index[None] == 1
            nil_winner_groups += int(nil_winner)
            for position, index in enumerate(indices):
                df.at[index, "P_rank_without_nil"] = float(group.at[index, "P_rank"])
                df.at[index, "P_match_pre_nil"] = distribution.real_probabilities[position]
                df.at[index, "P_nil"] = distribution.nil_probability
                df.at[index, "Q_match"] = distribution.real_q[position]
                df.at[index, "Q_nil"] = distribution.nil_q
                df.at[index, "P_rank"] = distribution.real_q[position]
                df.at[index, "nil_rank"] = int(rank_by_index[None])
                df.at[index, "candidate_joint_rank"] = int(rank_by_index[index])
                df.at[index, "selection_nil_winner"] = bool(nil_winner)
                df.at[index, "nil_mode"] = mode
                df.at[index, "nil_ranking_scale"] = ranking_scale

        self._nil_meta = {
            "mode": mode,
            "ranking_scale": ranking_scale,
            "applied": True,
            "source_groups": group_count,
            "nil_winner_groups": nil_winner_groups,
            "tie_rule": "candidate_iri_before_nil",
        }
        return df


__all__ = ["JointNilDistribution", "NilRankingMixin", "joint_nil_distribution"]
