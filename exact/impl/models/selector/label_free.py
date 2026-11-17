"""Deterministic target-label-free candidate-set selection primitives.

The methods here consume only frozen candidate scores and evidence columns.  In
particular, they never read a reference alignment or fit on target-pair labels.
"""

from __future__ import annotations

import statistics
from typing import Any, List, Optional, Sequence

import pandas as pd

from .calibration_helpers import select_distribution_threshold
from .grouping import groupby_target, iter_source_groups


class LabelFreeSelectorMixin:
    """Inference-only implementations of the E15 rule-based selector arms."""

    @staticmethod
    def _label_free_order(group: pd.DataFrame) -> List[Any]:
        return sorted(
            list(group.index),
            key=lambda idx: (
                -float(group.at[idx, "S_pair_final"]),
                str(group.at[idx, "Tgt"]),
                str(group.at[idx, "TgtKind"]) if "TgtKind" in group.columns else "class",
                str(idx),
            ),
        )

    @staticmethod
    def _median_absolute_deviation(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        centre = float(statistics.median(values))
        return float(statistics.median(abs(float(value) - centre) for value in values))

    def _write_label_free_group(
        self,
        *,
        df: pd.DataFrame,
        group: pd.DataFrame,
        winner_idx: Any,
        accepted: bool,
        source_probability: float,
        margin: float,
        threshold: float,
        threshold_mode: str,
        reason: str,
        reciprocal: bool = False,
        channel_agreement: int = 0,
        margin_cutoff: float = 0.0,
    ) -> None:
        idxs = list(group.index)
        scores = {
            idx: self._clip01(self._safe_float(group.at[idx, "S_pair_final"], 0.0)) for idx in idxs
        }
        rank_probabilities = self._softmax(
            [scores[idx] for idx in idxs], temperature=self.temperature
        )
        probability_by_idx = {
            idx: float(probability) for idx, probability in zip(idxs, rank_probabilities)
        }
        entropy = self._normalized_entropy(rank_probabilities)
        for idx in idxs:
            is_winner = idx == winner_idx
            if not accepted:
                selected_score = 0.0
            elif self.emit_candidate_scores:
                selected_score = scores[idx]
            else:
                selected_score = scores[idx] if is_winner else 0.0
            df.at[idx, "S_select"] = float(selected_score)
            df.at[idx, "P_select"] = float(selected_score)
            df.at[idx, "P_rank"] = probability_by_idx[idx]
            df.at[idx, "P_match"] = scores[idx]
            df.at[idx, "selection_source_p_match"] = float(source_probability)
            df.at[idx, "selection_winner"] = bool(accepted and is_winner)
            df.at[idx, "selection_accept_threshold"] = float(threshold)
            df.at[idx, "selection_margin"] = float(margin)
            df.at[idx, "selection_entropy"] = float(entropy)
            df.at[idx, "selection_no_match_prob"] = float(1.0 - source_probability)
            df.at[idx, "selection_evidence_support"] = scores[idx]
            df.at[idx, "selection_abstained"] = bool(not accepted)
            df.at[idx, "selection_llm_used"] = False
            df.at[idx, "selection_reason"] = reason if accepted else f"{reason}_no_match"
            df.at[idx, "selection_utility"] = scores[idx]
            df.at[idx, "selection_reciprocal"] = bool(reciprocal and is_winner)
            df.at[idx, "selection_channel_agreement"] = int(channel_agreement if is_winner else 0)
            df.at[idx, "selection_margin_cutoff"] = float(margin_cutoff)
            df.at[idx, "selection_threshold_mode"] = str(threshold_mode)

    def _run_score_partition_selector(
        self,
        *,
        df: pd.DataFrame,
        threshold: Optional[float],
    ) -> pd.DataFrame:
        """Apply the registered E03 fixed/Otsu/knee distribution rule."""

        scores = [
            self._clip01(self._safe_float(value, 0.0)) for value in df["S_pair_final"].tolist()
        ]
        threshold_mode = str(self.matching_calibration.get("threshold_mode", "fixed"))
        selected_threshold = select_distribution_threshold(
            scores,
            threshold_mode,
            fixed_threshold=float(threshold) if threshold is not None else 0.7,
        )
        for _, _, _, group in iter_source_groups(df):
            order = self._label_free_order(group)
            winner_idx = order[0]
            top_score = self._clip01(self._safe_float(group.at[winner_idx, "S_pair_final"], 0.0))
            second_score = (
                self._clip01(self._safe_float(group.at[order[1], "S_pair_final"], 0.0))
                if len(order) > 1
                else 0.0
            )
            self._write_label_free_group(
                df=df,
                group=group,
                winner_idx=winner_idx,
                accepted=bool(top_score >= selected_threshold),
                source_probability=top_score,
                margin=top_score - second_score,
                threshold=selected_threshold,
                threshold_mode=threshold_mode,
                reason="score_partition",
            )
        self._calibration_meta = {
            "strategy": "label_free",
            "label_free_mode": "score_partition",
            "threshold_mode": threshold_mode,
            "selected_threshold": float(selected_threshold),
            "target_labels_used": False,
        }
        return df

    def _independent_channel_agreement(
        self,
        group: pd.DataFrame,
        winner_idx: Any,
    ) -> tuple[int, int]:
        agreeing = 0
        usable = 0
        for column in ["s_label", "S_struct", "cand_sim"]:
            if column not in group.columns:
                continue
            values = [self._safe_float(group.at[idx, column], 0.0) for idx in group.index]
            if len(values) > 1 and max(values) - min(values) <= self.eps:
                continue
            usable += 1
            channel_winner = min(
                list(group.index),
                key=lambda idx: (
                    -self._safe_float(group.at[idx, column], 0.0),
                    str(group.at[idx, "Tgt"]),
                    str(idx),
                ),
            )
            if channel_winner == winner_idx:
                agreeing += 1
        return agreeing, usable

    def _run_reciprocal_consensus_selector(
        self,
        *,
        df: pd.DataFrame,
        threshold: Optional[float],
    ) -> pd.DataFrame:
        """Select reciprocal, robust-margin, multi-channel-consensus winners."""

        source_groups: List[tuple[pd.DataFrame, List[Any], float]] = []
        margins: List[float] = []
        for _, _, _, group in iter_source_groups(df):
            order = self._label_free_order(group)
            top = self._clip01(self._safe_float(group.at[order[0], "S_pair_final"], 0.0))
            second = (
                self._clip01(self._safe_float(group.at[order[1], "S_pair_final"], 0.0))
                if len(order) > 1
                else 0.0
            )
            margin = float(top - second)
            margins.append(margin)
            source_groups.append((group, order, margin))

        margin_median = float(statistics.median(margins)) if margins else 0.0
        margin_mad = self._median_absolute_deviation(margins)
        # Frozen conservative analytical constant; unlike a learned cutoff it
        # remains target-label-free and has deterministic MAD=0 behaviour.
        margin_cutoff = float(margin_median + 0.5 * margin_mad)

        target_winners: set[Any] = set()
        for _, group in groupby_target(df):
            target_winners.add(
                min(
                    list(group.index),
                    key=lambda idx: (
                        -self._safe_float(group.at[idx, "S_pair_final"], 0.0),
                        str(group.at[idx, "Src"]),
                        str(idx),
                    ),
                )
            )

        fixed_floor = float(threshold) if threshold is not None else 0.0
        for group, order, margin in source_groups:
            winner_idx = order[0]
            row = group.loc[winner_idx]
            top_score = self._clip01(self._safe_float(row.get("S_pair_final"), 0.0))
            agreeing, usable = self._independent_channel_agreement(group, winner_idx)
            required_agreement = min(2, usable)
            channels_agree = usable > 0 and agreeing >= required_agreement
            reciprocal = winner_idx in target_winners
            src_kind = str(row.get("SrcKind", "class"))
            tgt_kind = str(row.get("TgtKind", src_kind))
            kind_conflict = src_kind != tgt_kind
            semantic_conflict = self._safe_float(row.get("s_diff"), 0.5) < 0.5
            accepted = bool(
                reciprocal
                and margin >= margin_cutoff
                and channels_agree
                and not kind_conflict
                and not semantic_conflict
                and top_score >= fixed_floor
            )
            self._write_label_free_group(
                df=df,
                group=group,
                winner_idx=winner_idx,
                accepted=accepted,
                source_probability=top_score,
                margin=margin,
                threshold=fixed_floor,
                threshold_mode="median_mad_consensus",
                reason="reciprocal_consensus",
                reciprocal=reciprocal,
                channel_agreement=agreeing,
                margin_cutoff=margin_cutoff,
            )
        self._calibration_meta = {
            "strategy": "label_free",
            "label_free_mode": "reciprocal_consensus",
            "margin_median": margin_median,
            "margin_mad": margin_mad,
            "margin_cutoff": margin_cutoff,
            "target_labels_used": False,
        }
        return df


__all__ = ["LabelFreeSelectorMixin"]
