from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch

from exact.core.contracts.model import IModel
from exact.utils.data import read_table


class CandidateSetSelector(IModel):
    """
    Unsupervised listwise second-stage selector for global alignment.

    The primary scorer still estimates pairwise compatibility. This model
    compares all candidates for a source jointly, adds an explicit NO_MATCH
    abstention risk, and replaces S_final with the source-local selection
    score when enabled.
    """

    DEFAULT_SUPPORT_WEIGHT = 0.60
    DEFAULT_NO_MATCH_THRESHOLD = 0.55
    FIXED_LLM_BOOST = 0.15
    DEFAULT_LLM = {
        "enabled": True,
        "ambiguity_margin": 0.08,
        "max_candidates": 5,
        "max_new_tokens": 256,
        "trigger_acceptance_margin": 0.025,
        "trigger_rank_margin": 0.03,
        "min_confidence": 0.75,
        "trigger_rejected_high_support": True,
        "rejected_high_support_pair_min": 0.91,
        "rejected_high_support_acceptance_gap": 0.25,
        "rejected_high_support_rank_margin_max": 0.12,
    }
    DEFAULT_CALIBRATION = {
        "enabled": "auto",
        "min_positive_sources": 50,
        "background_negative_weight": 0.02,
        "l2": 1.0e-3,
        "learning_rate": 0.05,
        "max_epochs": 200,
        "threshold_grid_step": 0.005,
        "accept_objective": "recall_at_precision",
        "f_beta": 1.5,
        "min_precision": 0.82,
        "min_recall": None,
    }
    RANK_FEATURE_NAMES = [
        "logit_pair",
        "s_label",
        "S_struct",
        "s_hier",
        "s_sim",
        "s_diff",
        "s_attr",
        "cand_sim",
        "cand_sim_prob",
        "cand_share_log_ratio",
        "z_pair",
        "pair_gap",
        "channel_gap",
        "pair_rank_score",
        "reciprocity",
        "distinctive",
        "safety",
    ]
    ACCEPT_FEATURE_NAMES = [
        "top_utility",
        "top_rank_prob",
        "top_pair_score",
        "utility_margin",
        "rank_prob_margin",
        "rank_entropy",
        "top_distinctive",
        "top_label",
        "top_struct",
        "top_diff",
    ]

    def __init__(
        self,
        enabled: bool = False,
        global_only: bool = True,
        replace_final_score: bool = True,
        use_no_match: bool = True,
        temperature: float = 0.75,
        eps: float = 1.0e-6,
        support_weight: Optional[float] = None,
        no_match_threshold: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
        no_match: Optional[Dict[str, float]] = None,
        llm: Optional[Dict[str, Any]] = None,
        strategy: str = "heuristic",
        calibration: Optional[Dict[str, Any]] = None,
        training_reference_file_path: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.global_only = bool(global_only)
        self.replace_final_score = bool(replace_final_score)
        self.use_no_match = bool(use_no_match)
        self.temperature = max(float(temperature), float(eps))
        self.eps = max(float(eps), 1.0e-12)
        self.support_weight = self._resolve_support_weight(support_weight, weights or {})
        self.no_match_threshold = self._resolve_no_match_threshold(no_match_threshold, no_match or {})
        # Compatibility attributes for external callers that inspect model params.
        self.weights = {"support_weight": self.support_weight, "auxiliary_evidence": 1.0 - self.support_weight}
        self.no_match = {"threshold": self.no_match_threshold}
        self.llm = self._normalize_llm(llm or {})
        self.strategy = str(strategy or "heuristic")
        self.calibration = self._normalize_calibration(calibration or {})
        self.training_reference_file_path = (
            str(training_reference_file_path) if training_reference_file_path else None
        )
        self._llm_prompts_used = 0
        self._llm_warning_logged = False
        self._calibration_meta: Dict[str, Any] = {}
        self._ignored_legacy_kwargs = dict(kwargs or {})

    def runtime_fingerprint_payload(self, **_: Any) -> Dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "enabled": self.enabled,
            "global_only": self.global_only,
            "replace_final_score": self.replace_final_score,
            "use_no_match": self.use_no_match,
            "temperature": self.temperature,
            "support_weight": self.support_weight,
            "no_match_threshold": self.no_match_threshold,
            "llm": dict(self.llm),
            "strategy": self.strategy,
            "calibration": dict(self.calibration),
            "training_reference_file_path": self.training_reference_file_path,
        }

    def runtime_fingerprint(self) -> str:
        blob = json.dumps(self.runtime_fingerprint_payload(), sort_keys=True, default=str)
        return self._sha1(blob)

    def _resolve_support_weight(
        self,
        support_weight: Optional[float],
        weights: Dict[str, float],
    ) -> float:
        if support_weight is not None:
            return self._clip01(float(support_weight))
        if "support_weight" in weights:
            return self._clip01(float(weights["support_weight"]))

        competition_keys = ["zscore", "gap", "channel_gap", "candidate_generation", "reciprocity"]
        competition_fallback = max([float(weights[key]) for key in competition_keys if key in weights] or [0.0])
        penalty_fallback = max([float(weights[key]) for key in ["granularity", "contradiction"] if key in weights] or [0.0])
        legacy = {
            "absolute_support": float(
                weights.get("absolute_support", weights.get("pairwise", weights.get("absolute", 0.0)))
            ),
            "candidate_competition": float(
                weights.get("candidate_competition", weights.get("competition", competition_fallback))
            ),
            "distinctive_evidence": float(
                weights.get("distinctive_evidence", weights.get("evidence", weights.get("distinctive", 0.0)))
            ),
            "equivalence_safety": float(
                weights.get("equivalence_safety", weights.get("penalty", penalty_fallback))
            ),
        }
        if any(value > 0.0 for value in legacy.values()):
            total = sum(max(0.0, value) for value in legacy.values())
            return self._clip01(max(0.0, legacy["absolute_support"]) / max(total, self.eps))
        return self.DEFAULT_SUPPORT_WEIGHT

    def _resolve_no_match_threshold(
        self,
        no_match_threshold: Optional[float],
        no_match: Dict[str, float],
    ) -> float:
        if no_match_threshold is not None:
            return self._clip01(float(no_match_threshold))
        if "threshold" in no_match:
            return self._clip01(float(no_match["threshold"]))
        if "prior" in no_match:
            return self._clip01(1.0 - float(no_match["prior"]))
        return self.DEFAULT_NO_MATCH_THRESHOLD

    def _normalize_llm(self, llm: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(self.DEFAULT_LLM)
        normalized.update({key: value for key, value in llm.items() if key in normalized})
        if "ambiguity_margin" in llm:
            normalized["ambiguity_margin"] = float(llm["ambiguity_margin"])
        else:
            margins = [
                float(llm[key])
                for key in ["trigger_top2_margin", "trigger_no_match_margin"]
                if key in llm
            ]
            if margins:
                normalized["ambiguity_margin"] = max(margins)
        normalized["ambiguity_margin"] = self._clip01(float(normalized["ambiguity_margin"]))
        normalized["max_candidates"] = max(1, int(normalized["max_candidates"]))
        normalized["max_new_tokens"] = int(normalized["max_new_tokens"])
        normalized["trigger_acceptance_margin"] = self._clip01(
            float(normalized["trigger_acceptance_margin"])
        )
        normalized["trigger_rank_margin"] = self._clip01(float(normalized["trigger_rank_margin"]))
        normalized["min_confidence"] = self._clip01(float(normalized["min_confidence"]))
        normalized["trigger_rejected_high_support"] = self._safe_bool(
            normalized["trigger_rejected_high_support"]
        )
        normalized["rejected_high_support_pair_min"] = self._clip01(
            float(normalized["rejected_high_support_pair_min"])
        )
        normalized["rejected_high_support_acceptance_gap"] = self._clip01(
            float(normalized["rejected_high_support_acceptance_gap"])
        )
        normalized["rejected_high_support_rank_margin_max"] = self._clip01(
            float(normalized["rejected_high_support_rank_margin_max"])
        )
        return normalized

    def _normalize_calibration(self, calibration: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(self.DEFAULT_CALIBRATION)
        normalized.update({key: value for key, value in calibration.items() if key in normalized})
        normalized["enabled"] = str(normalized["enabled"]).lower()
        normalized["min_positive_sources"] = max(1, int(normalized["min_positive_sources"]))
        normalized["background_negative_weight"] = max(
            0.0, float(normalized["background_negative_weight"])
        )
        normalized["l2"] = max(0.0, float(normalized["l2"]))
        normalized["learning_rate"] = max(1.0e-6, float(normalized["learning_rate"]))
        normalized["max_epochs"] = max(1, int(normalized["max_epochs"]))
        normalized["threshold_grid_step"] = self._clip(
            float(normalized["threshold_grid_step"]), 1.0e-4, 0.5
        )
        objective = str(normalized["accept_objective"] or "recall_at_precision").lower()
        if objective not in {"f1", "f_beta", "recall_at_precision"}:
            objective = "recall_at_precision"
        normalized["accept_objective"] = objective
        normalized["f_beta"] = max(1.0e-6, float(normalized["f_beta"]))
        normalized["min_precision"] = self._optional_probability(normalized["min_precision"])
        normalized["min_recall"] = self._optional_probability(normalized["min_recall"])
        return normalized

    def _log(self, logger: Optional[Any], msg: str, level: str = "info") -> None:
        if logger is None:
            print(msg)
            return
        log_method = getattr(logger, level, None) or logger.info
        log_method(msg)

    def forward(
        self,
        candidate_df: pd.DataFrame,
        primary_model: Optional[IModel] = None,
        dataset: Any = None,
        logger: Optional[Any] = None,
        results_json: Optional[List[Dict[str, Any]]] = None,
        local_alignment: bool = False,
        threshold: Optional[float] = None,
        cardinality: Optional[int] = None,
        log_every: int = 10,
        checkpoint_callback: Optional[Callable[[pd.DataFrame], None]] = None,
        checkpoint_every_groups: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not self.enabled or candidate_df.empty:
            return {"candidate_df": candidate_df, "results_json": results_json}
        if self.global_only and local_alignment:
            self._log(logger, "Candidate-set selector skipped for local alignment.", "info")
            return {"candidate_df": candidate_df, "results_json": results_json}
        if "S_final" not in candidate_df.columns:
            self._log(
                logger,
                "Candidate-set selector skipped because candidate dataframe lacks S_final.",
                "warning",
            )
            return {"candidate_df": candidate_df, "results_json": results_json}

        start = time.perf_counter()
        self._llm_prompts_used = 0
        df = candidate_df.copy()
        if "S_pair_final" not in df.columns:
            df["S_pair_final"] = df["S_final"]

        record_lookup = self._record_lookup(results_json)
        distinctive = self._distinctive_scores(df, record_lookup)
        reciprocity = self._reciprocity_scores(df)

        defaults = {
            "S_select": 0.0,
            "P_select": 0.0,
            "selection_margin": 0.0,
            "selection_entropy": 0.0,
            "selection_no_match_prob": 0.0,
            "selection_abstained": False,
            "selection_llm_used": False,
            "selection_reason": "not_selected",
            "selection_distinctive": 0.0,
            "selection_utility": 0.0,
            "P_rank": 0.0,
            "P_match": 0.0,
            "selection_winner": False,
            "selection_accept_threshold": 0.0,
        }
        for col, value in defaults.items():
            if col not in df.columns:
                df[col] = value

        if self.strategy == "calibrated_rank_accept":
            calibrated_df = self._run_calibrated_selector(
                df=df,
                distinctive=distinctive,
                reciprocity=reciprocity,
                primary_model=primary_model,
                logger=logger,
                record_lookup=record_lookup,
                threshold=threshold,
                log_every=log_every,
                checkpoint_callback=checkpoint_callback,
                checkpoint_every_groups=checkpoint_every_groups,
            )
            if calibrated_df is not None:
                df = calibrated_df
            else:
                df = self._run_heuristic_selector(
                    df=df,
                    distinctive=distinctive,
                    reciprocity=reciprocity,
                    primary_model=primary_model,
                    logger=logger,
                    record_lookup=record_lookup,
                    log_every=log_every,
                    checkpoint_callback=checkpoint_callback,
                    checkpoint_every_groups=checkpoint_every_groups,
                )
        else:
            df = self._run_heuristic_selector(
                df=df,
                distinctive=distinctive,
                reciprocity=reciprocity,
                primary_model=primary_model,
                logger=logger,
                record_lookup=record_lookup,
                log_every=log_every,
                checkpoint_callback=checkpoint_callback,
                checkpoint_every_groups=checkpoint_every_groups,
            )

        if self.replace_final_score:
            df["S_final"] = df["S_select"]

        self._sync_results_json(df, results_json)
        elapsed = time.perf_counter() - start
        n_sources = int(df["Src"].nunique()) if "Src" in df.columns else 0
        n_groups = n_sources
        n_abstained = self._count_source_groups(df, "selection_abstained")
        n_llm = self._count_source_groups(df, "selection_llm_used")
        self._log(
            logger,
            (
                "Candidate-set selector: "
                f"sources={n_groups}/{n_sources}, "
                f"abstained_sources={n_abstained}, llm_sources={n_llm}, elapsed={elapsed:.2f}s."
            ),
            "info",
        )
        return {"candidate_df": df, "results_json": results_json}

    def _run_heuristic_selector(
        self,
        df: pd.DataFrame,
        distinctive: Dict[int, float],
        reciprocity: Dict[int, float],
        primary_model: Optional[IModel],
        logger: Optional[Any],
        record_lookup: Dict[Tuple[str, str], Dict[str, Any]],
        log_every: int,
        checkpoint_callback: Optional[Callable[[pd.DataFrame], None]],
        checkpoint_every_groups: Optional[int],
    ) -> pd.DataFrame:
        n_sources = int(df["Src"].nunique())
        n_groups = 0
        n_abstained = 0
        n_llm = 0
        progress_interval = self._progress_interval(log_every)
        checkpoint_interval = self._checkpoint_interval(checkpoint_every_groups)
        progress_start = time.perf_counter()

        for src, group in df.groupby("Src", sort=False):
            n_groups += 1
            idxs = list(group.index)
            group_result = self._score_group(
                src=src,
                group=df.loc[idxs],
                distinctive={idx: distinctive.get(idx, 0.0) for idx in idxs},
                reciprocity={idx: reciprocity.get(idx, 0.0) for idx in idxs},
                primary_model=primary_model,
                logger=logger,
                record_lookup=record_lookup,
            )
            for row_idx, payload in group_result["rows"].items():
                for key, value in payload.items():
                    df.at[row_idx, key] = value
            if bool(group_result.get("abstained")):
                n_abstained += 1
            if bool(group_result.get("llm_used")):
                n_llm += 1
            if n_groups == n_sources or n_groups % progress_interval == 0:
                elapsed_progress = max(1.0e-8, time.perf_counter() - progress_start)
                rate = n_groups / elapsed_progress
                remaining = max(0, n_sources - n_groups)
                eta = self._format_duration(remaining / rate) if rate > 0 else self._format_duration(0.0)
                self._log(
                    logger,
                    (
                        "Candidate-set selector progress: "
                        f"sources={n_groups}/{n_sources}, "
                        f"abstained_sources={n_abstained}, llm_sources={n_llm}, "
                        f"avg={rate:.2f} sources/s, ETA {eta}"
                    ),
                    "debug",
                )
            self._maybe_checkpoint_progress(
                df=df,
                n_groups=n_groups,
                n_sources=n_sources,
                checkpoint_interval=checkpoint_interval,
                checkpoint_callback=checkpoint_callback,
                logger=logger,
            )
        return df

    def _run_calibrated_selector(
        self,
        df: pd.DataFrame,
        distinctive: Dict[int, float],
        reciprocity: Dict[int, float],
        primary_model: Optional[IModel],
        logger: Optional[Any],
        record_lookup: Dict[Tuple[str, str], Dict[str, Any]],
        threshold: Optional[float],
        log_every: int,
        checkpoint_callback: Optional[Callable[[pd.DataFrame], None]],
        checkpoint_every_groups: Optional[int],
    ) -> Optional[pd.DataFrame]:
        if self.calibration.get("enabled") in {"false", "0", "off", "disabled"}:
            self._log(logger, "Calibrated selector disabled; using heuristic selector.", "info")
            return None

        ref_pairs = self._load_training_reference_pairs(logger)
        if not ref_pairs:
            self._log(
                logger,
                "Calibrated selector skipped because no training reference was provided.",
                "warning",
            )
            return None

        rank_features = self._rank_feature_rows(df, distinctive, reciprocity)
        rank_groups, n_positive_sources = self._rank_training_groups(df, ref_pairs)
        min_positive = int(self.calibration["min_positive_sources"])
        if n_positive_sources < min_positive:
            self._log(
                logger,
                (
                    "Calibrated selector skipped because usable positive training sources "
                    f"({n_positive_sources}) are below min_positive_sources={min_positive}."
                ),
                "warning",
            )
            return None

        rank_model = self._fit_rank_model(rank_features, rank_groups)
        if rank_model is None:
            self._log(logger, "Calibrated selector failed to fit rank model; using heuristic selector.", "warning")
            return None
        utilities = {
            idx: self._linear_score(rank_features[idx], rank_model)
            for idx in rank_features
        }

        source_decisions = self._source_decisions(df, utilities, rank_features, distinctive, ref_pairs)
        accept_model = self._fit_accept_model(source_decisions)
        if accept_model is None:
            self._log(
                logger,
                "Calibrated selector failed to fit acceptance model; using heuristic selector.",
                "warning",
            )
            return None
        for decision in source_decisions.values():
            decision["p_match"] = self._sigmoid(
                self._linear_score(decision["accept_features"], accept_model)
            )

        accept_threshold, threshold_metrics = self._tune_accept_threshold(source_decisions)
        if bool(threshold_metrics.get("fallback_to_f1", False)):
            self._log(
                logger,
                (
                    "Calibrated selector accept objective fell back to F1 because no "
                    f"threshold satisfied min_precision={threshold_metrics.get('min_precision')} "
                    f"and min_recall={threshold_metrics.get('min_recall')}."
                ),
                "warning",
            )
        n_sources = int(df["Src"].nunique())
        n_llm = 0
        n_abstained = 0
        progress_interval = self._progress_interval(log_every)
        checkpoint_interval = self._checkpoint_interval(checkpoint_every_groups)
        progress_start = time.perf_counter()
        score_threshold = float(threshold) if threshold is not None else accept_threshold

        for n_groups, (src, group) in enumerate(df.groupby("Src", sort=False), start=1):
            decision = source_decisions.get(str(src))
            if decision is None:
                continue
            p_match = float(decision["p_match"])
            accepted = p_match >= accept_threshold
            winner_idx = int(decision["winner_idx"])
            reason = "calibrated" if accepted else "calibrated_no_match"
            llm_used = False

            if self._should_use_llm_calibrated(
                acceptance_margin=abs(p_match - accept_threshold),
                rank_margin=float(decision["rank_prob_margin"]),
                primary_model=primary_model,
                accepted=accepted,
                top_pair_score=float(decision.get("top_pair_score", 0.0)),
            ):
                llm_choice = self._llm_direct_choice_group(
                    src=str(src),
                    group=group,
                    primary_model=primary_model,
                    logger=logger,
                    record_lookup=record_lookup,
                )
                if llm_choice and llm_choice.get("applied"):
                    llm_used = True
                    n_llm += 1
                    if bool(llm_choice.get("no_match")):
                        accepted = False
                        reason = "llm_no_match"
                    else:
                        chosen_idx = llm_choice.get("winner_idx")
                        if chosen_idx is not None:
                            winner_idx = int(chosen_idx)
                            accepted = True
                            p_match = max(p_match, accept_threshold)
                            reason = "llm"

            final_score = self._threshold_compatible_score(
                p_match=p_match,
                accept_threshold=accept_threshold,
                score_threshold=score_threshold,
            )
            if not accepted:
                final_score = 0.0
                n_abstained += 1

            for row_idx in list(group.index):
                row_rank_prob = float(decision["rank_probs"].get(row_idx, 0.0))
                is_winner = row_idx == winner_idx
                row_score = final_score if is_winner and accepted else 0.0
                df.at[row_idx, "S_select"] = row_score
                df.at[row_idx, "P_select"] = row_score
                df.at[row_idx, "P_rank"] = row_rank_prob
                df.at[row_idx, "P_match"] = float(p_match)
                df.at[row_idx, "selection_winner"] = bool(is_winner and accepted)
                df.at[row_idx, "selection_accept_threshold"] = float(accept_threshold)
                df.at[row_idx, "selection_margin"] = float(decision["utility_margin"])
                df.at[row_idx, "selection_entropy"] = float(decision["rank_entropy"])
                df.at[row_idx, "selection_no_match_prob"] = float(1.0 - p_match)
                df.at[row_idx, "selection_abstained"] = bool(not accepted)
                df.at[row_idx, "selection_llm_used"] = bool(llm_used)
                df.at[row_idx, "selection_reason"] = reason
                df.at[row_idx, "selection_distinctive"] = float(distinctive.get(row_idx, 0.0))
                df.at[row_idx, "selection_utility"] = float(utilities.get(row_idx, 0.0))

            if n_groups == n_sources or n_groups % progress_interval == 0:
                elapsed_progress = max(1.0e-8, time.perf_counter() - progress_start)
                rate = n_groups / elapsed_progress
                remaining = max(0, n_sources - n_groups)
                eta = self._format_duration(remaining / rate) if rate > 0 else self._format_duration(0.0)
                self._log(
                    logger,
                    (
                        "Calibrated selector progress: "
                        f"sources={n_groups}/{n_sources}, abstained_sources={n_abstained}, "
                        f"llm_sources={n_llm}, avg={rate:.2f} sources/s, ETA {eta}"
                    ),
                    "debug",
                )
            self._maybe_checkpoint_progress(
                df=df,
                n_groups=n_groups,
                n_sources=n_sources,
                checkpoint_interval=checkpoint_interval,
                checkpoint_callback=checkpoint_callback,
                logger=logger,
            )

        self._calibration_meta = {
            "strategy": self.strategy,
            "training_reference_file_path": self.training_reference_file_path,
            "n_reference_pairs": len(ref_pairs),
            "n_positive_sources": n_positive_sources,
            "rank_feature_names": list(self.RANK_FEATURE_NAMES),
            "accept_feature_names": list(self.ACCEPT_FEATURE_NAMES),
            "accept_threshold": accept_threshold,
            "accept_objective": threshold_metrics.get("accept_objective"),
            "accept_selected_metrics": threshold_metrics.get("selected_metrics", {}),
            "accept_best_f1_metrics": threshold_metrics.get("best_f1", {}),
            "accept_best_f_beta_metrics": threshold_metrics.get("best_f_beta", {}),
            "accept_recall_at_precision_metrics": threshold_metrics.get("recall_at_precision", {}),
            "accept_fallback_to_f1": bool(threshold_metrics.get("fallback_to_f1", False)),
            "threshold_metrics": threshold_metrics,
            "llm_groups": n_llm,
            "abstained_groups": n_abstained,
            "calibration": dict(self.calibration),
        }
        self._log(
            logger,
            (
                "Calibrated selector: "
                f"usable_positive_sources={n_positive_sources}, "
                f"accept_objective={threshold_metrics.get('accept_objective')}, "
                f"accept_threshold={accept_threshold:.3f}, "
                f"calibration_P={threshold_metrics.get('P', 0.0):.3f}, "
                f"calibration_R={threshold_metrics.get('R', 0.0):.3f}, "
                f"calibration_F1={threshold_metrics.get('F1', 0.0):.3f}, "
                f"calibration_F_beta={threshold_metrics.get('F_beta', 0.0):.3f}, "
                f"abstained_sources={n_abstained}, llm_sources={n_llm}."
            ),
            "info",
        )
        return df

    @staticmethod
    def _progress_interval(log_every: int) -> int:
        return max(1000, max(1, int(log_every)) * 100)

    @staticmethod
    def _checkpoint_interval(checkpoint_every_groups: Optional[int]) -> int:
        if checkpoint_every_groups is None:
            return 0
        return max(1, int(checkpoint_every_groups))

    def _maybe_checkpoint_progress(
        self,
        df: pd.DataFrame,
        n_groups: int,
        n_sources: int,
        checkpoint_interval: int,
        checkpoint_callback: Optional[Callable[[pd.DataFrame], None]],
        logger: Optional[Any],
    ) -> None:
        if checkpoint_interval <= 0 or checkpoint_callback is None:
            return
        if n_groups >= n_sources or n_groups % checkpoint_interval != 0:
            return
        try:
            checkpoint_callback(df)
        except Exception as exc:  # noqa: BLE001
            self._log(logger, f"Candidate-set selector checkpoint failed: {exc}", "warning")

    def _load_training_reference_pairs(self, logger: Optional[Any]) -> set[Tuple[str, str]]:
        if not self.training_reference_file_path:
            return set()
        path = Path(self.training_reference_file_path)
        if not path.exists():
            self._log(logger, f"Training reference file not found: {path}", "warning")
            return set()
        try:
            ref_df = read_table(path)
        except Exception as exc:  # noqa: BLE001
            self._log(logger, f"Failed to read training reference file {path}: {exc}", "warning")
            return set()
        if ref_df.shape[1] < 2:
            self._log(logger, f"Training reference file {path} has fewer than two columns.", "warning")
            return set()
        src_col, tgt_col = ref_df.columns[:2]
        pairs = {
            (str(row[src_col]), str(row[tgt_col]))
            for _, row in ref_df.iterrows()
            if str(row[src_col]) and str(row[tgt_col])
        }
        return pairs

    def _rank_training_groups(
        self,
        df: pd.DataFrame,
        ref_pairs: set[Tuple[str, str]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        groups: List[Dict[str, Any]] = []
        n_positive_sources = 0
        for src, group in df.groupby("Src", sort=False):
            idxs = list(group.index)
            pos = [idx for idx in idxs if (str(src), str(group.at[idx, "Tgt"])) in ref_pairs]
            if not pos:
                continue
            n_positive_sources += 1
            groups.append({"indices": idxs, "positive_indices": pos})
        return groups, n_positive_sources

    def _rank_feature_rows(
        self,
        df: pd.DataFrame,
        distinctive: Dict[int, float],
        reciprocity: Dict[int, float],
    ) -> Dict[int, List[float]]:
        rows: Dict[int, List[float]] = {}
        for _, group in df.groupby("Src", sort=False):
            idxs = list(group.index)
            pair_scores = [
                self._clip01(self._safe_float(group.at[idx, "S_pair_final"], 0.0))
                for idx in idxs
            ]
            mean_score = sum(pair_scores) / max(1, len(pair_scores))
            std_score = self._std(pair_scores)
            order_by_pair = {
                idx: rank
                for rank, idx in enumerate(
                    sorted(idxs, key=lambda row_idx: self._safe_float(group.at[row_idx, "S_pair_final"], 0.0), reverse=True),
                    start=1,
                )
            }
            channel_gaps = self._channel_gaps(group, idxs)
            for pos, idx in enumerate(idxs):
                row = group.loc[idx]
                pair = pair_scores[pos]
                if std_score < self.eps:
                    z_pair = 0.0
                else:
                    z_pair = (pair - mean_score) / std_score
                if len(pair_scores) <= 1:
                    pair_gap = 0.0
                else:
                    other = max(pair_scores[:pos] + pair_scores[pos + 1 :])
                    pair_gap = pair - other
                contradiction = 1.0 - self._clip01(self._safe_float(row.get("s_diff"), 1.0))
                granularity = self._granularity_penalty(row)
                safety = self._clip01(1.0 - ((contradiction + granularity) / 2.0))
                rank = order_by_pair.get(idx, len(idxs))
                rank_score = 1.0 - ((rank - 1) / max(1, len(idxs) - 1))
                rows[idx] = [
                    self._logit(pair),
                    self._safe_float(row.get("s_label"), 0.0),
                    self._safe_float(row.get("S_struct"), 0.0),
                    self._safe_float(row.get("s_hier"), 0.0),
                    self._safe_float(row.get("s_sim"), 0.0),
                    self._safe_float(row.get("s_diff"), 0.0),
                    self._safe_float(row.get("s_attr"), 0.0),
                    self._safe_float(row.get("cand_sim"), pair),
                    self._safe_float(row.get("cand_sim_prob"), 0.0),
                    self._safe_float(row.get("cand_share_log_ratio"), 0.0),
                    z_pair,
                    pair_gap,
                    channel_gaps.get(idx, 0.0),
                    rank_score,
                    self._reciprocity_score01(reciprocity.get(idx, 0.0)),
                    distinctive.get(idx, 0.0),
                    safety,
                ]
        return rows

    def _fit_rank_model(
        self,
        features: Dict[int, List[float]],
        groups: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        train_indices = sorted({idx for group in groups for idx in group["indices"]})
        if not train_indices:
            return None
        mean, scale = self._feature_mean_scale([features[idx] for idx in train_indices])
        local_pos = {idx: pos for pos, idx in enumerate(train_indices)}
        # predict() runs under torch.no_grad(); calibration still needs autograd.
        with torch.enable_grad():
            x = torch.tensor(
                [self._standardize(features[idx], mean, scale) for idx in train_indices],
                dtype=torch.float32,
            )
            dim = int(x.shape[1])
            weights = torch.zeros(dim, dtype=torch.float32, requires_grad=True)
            bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
            opt = torch.optim.Adam([weights, bias], lr=float(self.calibration["learning_rate"]))
            l2 = float(self.calibration["l2"])
            n_groups = max(1, len(groups))
            for _ in range(int(self.calibration["max_epochs"])):
                opt.zero_grad()
                loss = torch.zeros((), dtype=torch.float32)
                for group in groups:
                    positions = [local_pos[idx] for idx in group["indices"] if idx in local_pos]
                    positive_positions = [
                        local_pos[idx] for idx in group["positive_indices"] if idx in local_pos
                    ]
                    if not positions or not positive_positions:
                        continue
                    pos_tensor = torch.tensor(positions, dtype=torch.long)
                    logits = x[pos_tensor].matmul(weights) + bias
                    log_probs = torch.log_softmax(logits, dim=0)
                    positive_local = torch.tensor(
                        [positions.index(pos_idx) for pos_idx in positive_positions],
                        dtype=torch.long,
                    )
                    loss = loss - torch.logsumexp(log_probs[positive_local], dim=0)
                loss = loss / n_groups + l2 * torch.sum(weights * weights)
                loss.backward()
                opt.step()
        return {
            "weights": weights.detach().cpu().tolist(),
            "bias": float(bias.detach().cpu().item()),
            "mean": mean,
            "scale": scale,
        }

    def _source_decisions(
        self,
        df: pd.DataFrame,
        utilities: Dict[int, float],
        rank_features: Dict[int, List[float]],
        distinctive: Dict[int, float],
        ref_pairs: set[Tuple[str, str]],
    ) -> Dict[str, Dict[str, Any]]:
        ref_sources = {src for src, _ in ref_pairs}
        decisions: Dict[str, Dict[str, Any]] = {}
        for src, group in df.groupby("Src", sort=False):
            idxs = list(group.index)
            util_values = [utilities[idx] for idx in idxs]
            probs = self._softmax(util_values, temperature=self.temperature)
            prob_by_idx = {idx: float(prob) for idx, prob in zip(idxs, probs)}
            order = sorted(idxs, key=lambda idx: utilities[idx], reverse=True)
            winner_idx = order[0]
            second_idx = order[1] if len(order) > 1 else None
            utility_margin = utilities[winner_idx] - (utilities[second_idx] if second_idx is not None else 0.0)
            rank_prob_margin = prob_by_idx[winner_idx] - (prob_by_idx[second_idx] if second_idx is not None else 0.0)
            entropy = self._normalized_entropy(probs)
            top_row = group.loc[winner_idx]
            top_pair_score = self._clip01(self._safe_float(top_row.get("S_pair_final"), 0.0))
            accept_features = [
                utilities[winner_idx],
                prob_by_idx[winner_idx],
                top_pair_score,
                utility_margin,
                rank_prob_margin,
                entropy,
                distinctive.get(winner_idx, 0.0),
                self._safe_float(top_row.get("s_label"), 0.0),
                self._safe_float(top_row.get("S_struct"), 0.0),
                self._safe_float(top_row.get("s_diff"), 0.0),
            ]
            src_text = str(src)
            winner_pair = (src_text, str(top_row.get("Tgt")))
            if src_text in ref_sources:
                label = 1.0 if winner_pair in ref_pairs else 0.0
                sample_weight = 1.0
            else:
                label = 0.0
                sample_weight = float(self.calibration["background_negative_weight"])
            decisions[src_text] = {
                "indices": idxs,
                "winner_idx": winner_idx,
                "rank_probs": prob_by_idx,
                "utility_margin": float(utility_margin),
                "rank_prob_margin": float(rank_prob_margin),
                "rank_entropy": float(entropy),
                "top_pair_score": float(top_pair_score),
                "accept_features": accept_features,
                "label": label,
                "sample_weight": sample_weight,
                "rank_feature": rank_features[winner_idx],
            }
        return decisions

    def _fit_accept_model(self, decisions: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        samples = [
            decision
            for decision in decisions.values()
            if float(decision.get("sample_weight", 0.0)) > 0.0
        ]
        if not samples:
            return None
        positives = sum(1 for sample in samples if float(sample.get("label", 0.0)) > 0.5)
        if positives <= 0:
            return None
        mean, scale = self._feature_mean_scale([sample["accept_features"] for sample in samples])
        x = torch.tensor(
            [self._standardize(sample["accept_features"], mean, scale) for sample in samples],
            dtype=torch.float32,
        )
        y = torch.tensor([float(sample["label"]) for sample in samples], dtype=torch.float32)
        sample_weight = torch.tensor(
            [float(sample["sample_weight"]) for sample in samples],
            dtype=torch.float32,
        )
        dim = int(x.shape[1])
        weights = torch.zeros(dim, dtype=torch.float32, requires_grad=True)
        positive_weight = max(1.0e-6, float(torch.sum(sample_weight * y).item()))
        negative_weight = max(1.0e-6, float(torch.sum(sample_weight * (1.0 - y)).item()))
        prior = self._clip(positive_weight / (positive_weight + negative_weight), self.eps, 1.0 - self.eps)
        # predict() runs under torch.no_grad(); calibration still needs autograd.
        with torch.enable_grad():
            bias = torch.tensor(self._logit(prior), dtype=torch.float32, requires_grad=True)
            opt = torch.optim.Adam([weights, bias], lr=float(self.calibration["learning_rate"]))
            l2 = float(self.calibration["l2"])
            denom = torch.clamp(sample_weight.sum(), min=1.0e-6)
            for _ in range(int(self.calibration["max_epochs"])):
                opt.zero_grad()
                logits = x.matmul(weights) + bias
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    y,
                    reduction="none",
                )
                loss = torch.sum(losses * sample_weight) / denom + l2 * torch.sum(weights * weights)
                loss.backward()
                opt.step()
        return {
            "weights": weights.detach().cpu().tolist(),
            "bias": float(bias.detach().cpu().item()),
            "mean": mean,
            "scale": scale,
        }

    def _tune_accept_threshold(self, decisions: Dict[str, Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        samples = [
            decision
            for decision in decisions.values()
            if float(decision.get("sample_weight", 0.0)) > 0.0
        ]
        if not samples:
            return 0.5, {
                "threshold": 0.5,
                "selected_threshold": 0.5,
                "P": 0.0,
                "R": 0.0,
                "F1": 0.0,
                "F_beta": 0.0,
                "TP": 0.0,
                "FP": 0.0,
                "FN": 0.0,
                "accept_objective": str(self.calibration["accept_objective"]),
                "fallback_to_f1": False,
                "selected_metrics": {},
                "best_f1": {},
                "best_f_beta": {},
                "recall_at_precision": {},
            }
        step = float(self.calibration["threshold_grid_step"])
        thresholds = [round(i * step, 10) for i in range(0, int(1.0 / step) + 1)]
        if thresholds[-1] < 1.0:
            thresholds.append(1.0)
        beta = max(1.0e-6, float(self.calibration["f_beta"]))
        beta2 = beta * beta
        threshold_metrics: List[Dict[str, float]] = []
        for threshold in thresholds:
            tp = fp = fn = 0.0
            for sample in samples:
                weight = float(sample.get("sample_weight", 0.0))
                label = float(sample.get("label", 0.0)) > 0.5
                pred = float(sample.get("p_match", 0.0)) >= threshold
                if pred and label:
                    tp += weight
                elif pred and not label:
                    fp += weight
                elif (not pred) and label:
                    fn += weight
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f_beta = (
                (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall)
                if ((beta2 * precision) + recall) > 0
                else 0.0
            )
            threshold_metrics.append(
                {
                    "threshold": float(threshold),
                    "P": precision,
                    "R": recall,
                    "F1": f1,
                    "F_beta": f_beta,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                }
            )

        best_f1 = max(threshold_metrics, key=lambda item: (item["F1"], item["threshold"]))
        best_f_beta = max(
            threshold_metrics,
            key=lambda item: (item["F_beta"], item["threshold"]),
        )

        min_precision = self.calibration.get("min_precision")
        min_recall = self.calibration.get("min_recall")
        precision_floor = float(min_precision) if min_precision is not None else 0.0
        recall_floor = float(min_recall) if min_recall is not None else 0.0
        recall_candidates = [
            item
            for item in threshold_metrics
            if item["P"] >= precision_floor and item["R"] >= recall_floor
        ]
        best_recall_at_precision = (
            max(
                recall_candidates,
                key=lambda item: (item["R"], item["F1"], item["P"], item["threshold"]),
            )
            if recall_candidates
            else None
        )

        objective = str(self.calibration["accept_objective"])
        fallback_to_f1 = False
        if objective == "f1":
            selected = best_f1
        elif objective == "f_beta":
            selected = best_f_beta
        else:
            if best_recall_at_precision is None:
                selected = best_f1
                fallback_to_f1 = True
            else:
                selected = best_recall_at_precision

        diagnostics: Dict[str, Any] = dict(selected)
        diagnostics.update(
            {
                "selected_threshold": float(selected["threshold"]),
                "accept_objective": objective,
                "f_beta": beta,
                "min_precision": min_precision,
                "min_recall": min_recall,
                "fallback_to_f1": fallback_to_f1,
                "selected_metrics": dict(selected),
                "best_f1": dict(best_f1),
                "best_f_beta": dict(best_f_beta),
                "recall_at_precision": dict(best_recall_at_precision or {}),
            }
        )
        return float(selected["threshold"]), diagnostics

    def _threshold_compatible_score(
        self,
        p_match: float,
        accept_threshold: float,
        score_threshold: float,
    ) -> float:
        if score_threshold is None:
            return self._clip01(p_match)
        return self._clip01(float(score_threshold) + (float(p_match) - float(accept_threshold)))

    def _feature_mean_scale(self, rows: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
        if not rows:
            return [], []
        n_cols = len(rows[0])
        means: List[float] = []
        scales: List[float] = []
        for col in range(n_cols):
            values = [float(row[col]) for row in rows]
            mean = sum(values) / len(values)
            var = sum((value - mean) ** 2 for value in values) / max(1, len(values))
            scale = math.sqrt(var)
            means.append(mean)
            scales.append(scale if scale > self.eps else 1.0)
        return means, scales

    def _standardize(
        self,
        row: Sequence[float],
        mean: Sequence[float],
        scale: Sequence[float],
    ) -> List[float]:
        return [
            (float(value) - float(mean[idx])) / max(float(scale[idx]), self.eps)
            for idx, value in enumerate(row)
        ]

    def _linear_score(self, row: Sequence[float], model: Mapping[str, Any]) -> float:
        standardized = self._standardize(row, model["mean"], model["scale"])
        weights = model["weights"]
        score = float(model.get("bias", 0.0))
        for value, weight in zip(standardized, weights):
            score += float(value) * float(weight)
        return float(score)

    def _score_group(
        self,
        src: str,
        group: pd.DataFrame,
        distinctive: Dict[int, float],
        reciprocity: Dict[int, float],
        primary_model: Optional[IModel],
        logger: Optional[Any],
        record_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        idxs = list(group.index)
        n = len(idxs)
        pair_scores = [
            self._clip01(self._safe_float(group.at[idx, "S_pair_final"], 0.0))
            for idx in idxs
        ]
        mean_score = sum(pair_scores) / max(1, n)
        std_score = self._std(pair_scores)
        if std_score < self.eps:
            zscore_terms = [0.5 for _ in pair_scores]
        else:
            zscore_terms = [
                self._sigmoid((value - mean_score) / std_score)
                for value in pair_scores
            ]

        gaps = []
        for pos, value in enumerate(pair_scores):
            if n <= 1:
                gaps.append(0.0)
                continue
            other = max(pair_scores[:pos] + pair_scores[pos + 1 :])
            gaps.append(value - other)

        channel_gaps = self._channel_gaps(group, idxs)
        selector_scores: List[float] = []
        no_match_terms: List[Dict[str, float]] = []
        for pos, idx in enumerate(idxs):
            row = group.loc[idx]
            contradiction = 1.0 - self._clip01(self._safe_float(row.get("s_diff"), 1.0))
            granularity = self._granularity_penalty(row)
            competition_terms = (
                zscore_terms[pos],
                self._gap_score(gaps[pos]),
                channel_gaps.get(idx, 0.0),
                self._candidate_generation_score(row),
                self._reciprocity_score01(reciprocity.get(idx, 0.0)),
            )
            competition_score = self._clip01(sum(competition_terms) / len(competition_terms))
            penalty_signal = (granularity + contradiction) / 2.0
            safety_score = self._clip01(1.0 - penalty_signal)
            auxiliary_score = self._clip01(
                (competition_score + distinctive.get(idx, 0.0) + safety_score) / 3.0
            )
            score = (
                self.support_weight * pair_scores[pos]
                + (1.0 - self.support_weight) * auxiliary_score
            )
            selector_scores.append(float(self._clip01(score)))
            no_match_terms.append(
                {
                    "absolute_support": pair_scores[pos],
                    "candidate_competition": competition_score,
                    "distinctive_evidence": distinctive.get(idx, 0.0),
                    "equivalence_safety": safety_score,
                    "auxiliary_evidence": auxiliary_score,
                }
            )

        selector_probs = self._softmax(selector_scores, temperature=self.temperature)
        no_match_risk = self._no_match_risk(
            selector_scores=selector_scores,
            selector_probs=selector_probs,
            no_match_terms=no_match_terms,
        )

        llm_payload = None
        if self._should_use_llm(selector_scores, no_match_risk, primary_model):
            llm_payload = self._llm_arbitrate_group(
                src=src,
                group=group,
                selector_scores=selector_scores,
                no_match_risk=no_match_risk,
                primary_model=primary_model,
                logger=logger,
                record_lookup=record_lookup,
            )
            if llm_payload and llm_payload.get("applied"):
                selector_scores = list(llm_payload.get("selector_scores", selector_scores))
                no_match_risk = float(llm_payload.get("no_match_risk", no_match_risk))
                selector_probs = self._softmax(selector_scores, temperature=self.temperature)

        no_match_prob = float(no_match_risk) if self.use_no_match else 0.0
        no_match_wins = bool(self.use_no_match and no_match_risk >= self.no_match_threshold)
        final_candidate_scores = [0.0 for _ in selector_scores] if no_match_wins else list(selector_scores)

        best_real = max(final_candidate_scores or [0.0])
        winner_pos = None
        if final_candidate_scores and not no_match_wins:
            winner_pos = max(range(len(final_candidate_scores)), key=lambda pos: final_candidate_scores[pos])
        sorted_final = sorted(selector_scores + ([no_match_prob] if self.use_no_match else []), reverse=True)
        margin = sorted_final[0] - (sorted_final[1] if len(sorted_final) > 1 else 0.0)
        final_entropy = self._normalized_entropy(selector_probs)
        llm_used = bool(llm_payload and llm_payload.get("applied"))
        reason = "llm" if llm_used else "numeric"
        if no_match_wins:
            reason = "llm_no_match" if llm_used else "no_match"

        rows: Dict[int, Dict[str, Any]] = {}
        for pos, idx in enumerate(idxs):
            prob = float(final_candidate_scores[pos]) if pos < len(final_candidate_scores) else 0.0
            rows[idx] = {
                "S_select": prob,
                "P_select": prob,
                "selection_margin": float(margin),
                "selection_entropy": float(final_entropy),
                "selection_no_match_prob": float(no_match_prob),
                "selection_abstained": bool(no_match_wins),
                "selection_llm_used": bool(llm_used),
                "selection_reason": reason,
                "selection_distinctive": float(distinctive.get(idx, 0.0)),
                "selection_utility": float(selector_scores[pos]),
                "P_rank": float(selector_probs[pos]) if pos < len(selector_probs) else 0.0,
                "P_match": float(1.0 - no_match_prob),
                "selection_winner": bool(winner_pos is not None and pos == winner_pos),
            }
        return {
            "rows": rows,
            "abstained": bool(no_match_wins),
            "llm_used": bool(llm_used),
            "best_real": float(best_real),
        }

    def _no_match_risk(
        self,
        selector_scores: Sequence[float],
        selector_probs: Sequence[float],
        no_match_terms: Sequence[Mapping[str, float]],
    ) -> float:
        if not self.use_no_match:
            return 0.0
        entropy = self._normalized_entropy(selector_probs)
        max_support = max(selector_scores or [0.0])
        max_evidence = max(
            [float(term.get("distinctive_evidence", 0.0)) for term in no_match_terms] or [0.0]
        )
        max_competition = max(
            [float(term.get("candidate_competition", 0.0)) for term in no_match_terms] or [0.0]
        )
        risk_terms = (
            entropy,
            1.0 - max_support,
            1.0 - max_evidence,
            1.0 - max_competition,
        )
        risk = sum(risk_terms) / len(risk_terms)
        return float(self._clip01(risk))

    def _should_use_llm(
        self,
        selector_scores: Sequence[float],
        no_match_risk: float,
        primary_model: Optional[IModel],
    ) -> bool:
        if not bool(self.llm.get("enabled", True)):
            return False
        if primary_model is None:
            return False
        if not selector_scores:
            return False
        sorted_scores = sorted(selector_scores, reverse=True)
        top2_margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
        no_match_margin = 1.0
        if self.use_no_match:
            no_match_margin = abs(float(no_match_risk) - self.no_match_threshold)
        ambiguity_margin = float(self.llm.get("ambiguity_margin", 0.08))
        return (
            top2_margin < ambiguity_margin
            or no_match_margin < ambiguity_margin
        )

    def _should_use_llm_calibrated(
        self,
        acceptance_margin: float,
        rank_margin: float,
        primary_model: Optional[IModel],
        accepted: bool = True,
        top_pair_score: float = 0.0,
    ) -> bool:
        if not bool(self.llm.get("enabled", True)):
            return False
        if primary_model is None:
            return False
        near_boundary_or_tie = (
            float(acceptance_margin) <= float(self.llm.get("trigger_acceptance_margin", 0.025))
            or float(rank_margin) <= float(self.llm.get("trigger_rank_margin", 0.03))
        )
        if near_boundary_or_tie:
            return True
        if not bool(self.llm.get("trigger_rejected_high_support", True)) or bool(accepted):
            return False
        return (
            float(top_pair_score) >= float(self.llm.get("rejected_high_support_pair_min", 0.91))
            and float(acceptance_margin) <= float(self.llm.get("rejected_high_support_acceptance_gap", 0.25))
            and float(rank_margin) <= float(self.llm.get("rejected_high_support_rank_margin_max", 0.12))
        )

    def _llm_arbitrate_group(
        self,
        src: str,
        group: pd.DataFrame,
        selector_scores: Sequence[float],
        no_match_risk: float,
        primary_model: IModel,
        logger: Optional[Any],
        record_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        prompt, id_to_index, id_to_tgt = self._llm_prompt(src, group, record_lookup)
        if prompt is None:
            return None
        try:
            text = self._run_llm_prompt(primary_model, prompt)
            payload = self._parse_llm_payload(text)
        except Exception as exc:  # noqa: BLE001
            if not self._llm_warning_logged:
                self._log(logger, f"Candidate-set selector LLM arbitration failed: {exc}", "warning")
                self._llm_warning_logged = True
            return None

        if not payload:
            return None
        winner = str(payload.get("winner", "")).strip()
        relation = str(payload.get("relation", "")).strip().lower()
        confidence = self._clip01(self._safe_float(payload.get("confidence"), 0.0))
        if confidence <= 0.0:
            return None

        updated = list(selector_scores)
        updated_no_match = float(no_match_risk)
        boost = self.FIXED_LLM_BOOST * confidence
        applied = False
        if winner.upper() == "NO_MATCH" or relation in {"broader", "narrower", "sibling", "related", "none"}:
            if not self.use_no_match:
                return None
            updated_no_match = self._clip01(updated_no_match + boost)
            applied = True
        elif relation == "equivalent":
            winner_key = winner
            if winner_key not in id_to_index:
                for cand_id, tgt in id_to_tgt.items():
                    if winner == tgt:
                        winner_key = cand_id
                        break
            winner_idx = id_to_index.get(winner_key)
            if winner_idx is not None and 0 <= winner_idx < len(updated):
                updated[winner_idx] = self._clip01(updated[winner_idx] + boost)
                penalty = boost / max(1, len(updated) - 1)
                for idx in range(len(updated)):
                    if idx != winner_idx:
                        updated[idx] = self._clip01(updated[idx] - penalty)
                applied = True

        if applied:
            self._llm_prompts_used += 1
            return {
                "applied": True,
                "selector_scores": updated,
                "no_match_risk": updated_no_match,
                "raw": payload,
            }
        return None

    def _llm_direct_choice_group(
        self,
        src: str,
        group: pd.DataFrame,
        primary_model: Optional[IModel],
        logger: Optional[Any],
        record_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if primary_model is None:
            return None
        prompt, id_to_index, id_to_tgt = self._llm_prompt(src, group, record_lookup)
        if prompt is None:
            return None
        try:
            text = self._run_llm_prompt(primary_model, prompt)
            payload = self._parse_llm_payload(text)
        except Exception as exc:  # noqa: BLE001
            if not self._llm_warning_logged:
                self._log(logger, f"Candidate-set selector LLM arbitration failed: {exc}", "warning")
                self._llm_warning_logged = True
            return None
        if not payload:
            return None
        confidence = self._clip01(self._safe_float(payload.get("confidence"), 0.0))
        if confidence < float(self.llm.get("min_confidence", 0.75)):
            return None
        winner = str(payload.get("winner", "")).strip()
        relation = str(payload.get("relation", "")).strip().lower()
        self._llm_prompts_used += 1
        if winner.upper() == "NO_MATCH" or relation in {"broader", "narrower", "sibling", "related", "none"}:
            return {"applied": True, "no_match": True, "raw": payload}
        if relation != "equivalent":
            return None
        winner_key = winner
        if winner_key not in id_to_index:
            for cand_id, tgt in id_to_tgt.items():
                if winner == tgt:
                    winner_key = cand_id
                    break
        winner_pos = id_to_index.get(winner_key)
        if winner_pos is None:
            return None
        idxs = list(group.index)
        if not (0 <= int(winner_pos) < len(idxs)):
            return None
        return {
            "applied": True,
            "no_match": False,
            "winner_idx": idxs[int(winner_pos)],
            "raw": payload,
        }

    def _llm_prompt(
        self,
        src: str,
        group: pd.DataFrame,
        record_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, str]], Dict[str, int], Dict[str, str]]:
        max_candidates = max(1, int(self.llm.get("max_candidates", 10)))
        ordered = group.sort_values("S_pair_final", ascending=False).head(max_candidates)
        if ordered.empty:
            return None, {}, {}
        source_label = str(ordered.iloc[0].get("src_label_text") or src)
        lines = [
            "Choose the target candidate that is equivalent to the source ontology entity.",
            "Reject broader, narrower, sibling, and merely related concepts.",
            "Return exactly one JSON object with keys: winner, relation, confidence, decisive_evidence, rejected.",
            "Use winner as one of the candidate ids or NO_MATCH.",
            "",
            f"Source id: {src}",
            f"Source label: {source_label}",
            "",
            "Candidates:",
        ]
        id_to_index: Dict[str, int] = {}
        id_to_tgt: Dict[str, str] = {}
        for prompt_pos, (row_idx, row) in enumerate(ordered.iterrows(), start=1):
            cand_id = f"C{prompt_pos}"
            tgt = str(row.get("Tgt"))
            id_to_index[cand_id] = list(group.index).index(row_idx)
            id_to_tgt[cand_id] = tgt
            label = str(row.get("tgt_label_text") or tgt)
            score = self._safe_float(row.get("S_pair_final"), 0.0)
            record = record_lookup.get((src, tgt)) or {}
            brief = str(record.get("llm_pair_brief") or row.get("llm_pair_brief") or "")
            brief = brief[:1200]
            lines.extend(
                [
                    f"{cand_id}. id: {tgt}",
                    f"label: {label}",
                    f"pairwise_score: {score:.4f}",
                    f"evidence: {brief}" if brief else "evidence: unavailable",
                    "",
                ]
            )
        lines.append(
            'JSON schema: {"winner":"C1|C2|NO_MATCH","relation":"equivalent|broader|narrower|sibling|related|none","confidence":0.0,"decisive_evidence":"...","rejected":{"C2":"..."}}'
        )
        return {
            "system": "You are an ontology alignment expert.",
            "user": "\n".join(lines),
        }, id_to_index, id_to_tgt

    def _run_llm_prompt(self, primary_model: IModel, prompt: Dict[str, str]) -> str:
        custom = getattr(primary_model, "candidate_set_select", None)
        if callable(custom):
            return str(custom(prompt))

        router = getattr(primary_model, "_llm_router", None)
        if router is not None and hasattr(primary_model, "_run_hosted_chat_prompts"):
            resolved = router.resolve_task("decision", require_logprobs=False)
            if resolved.backend == "openrouter" and resolved.decision_capable:
                profile = router.profiles.get(resolved.profile_name or "")
                if profile is not None:
                    outputs = primary_model._run_hosted_chat_prompts(
                        prompts=[prompt],
                        profile=profile,
                        max_tokens=int(self.llm.get("max_new_tokens", 256)),
                        temperature=0.0,
                        top_p=1.0,
                        concurrency=1,
                    )
                    return outputs[0] if outputs else ""

        if not all(hasattr(primary_model, name) for name in ("_ensure_local_llm", "llm_tok", "llm")):
            raise RuntimeError("Primary model does not expose a usable LLM arbitration backend.")
        primary_model._ensure_local_llm()
        rendered = primary_model._render_llm_prompt(prompt)
        tok = primary_model.llm_tok
        llm = primary_model.llm
        device = next(llm.parameters()).device
        enc = tok(
            [rendered],
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=int(getattr(primary_model, "max_total_tokens_llm_decision", 1024)),
        ).to(device)
        with torch.no_grad():
            out = llm.generate(
                **enc,
                max_new_tokens=int(self.llm.get("max_new_tokens", 256)),
                temperature=0.0,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        if hasattr(primary_model, "_strip_llm_prompt_tokens"):
            decoded_tokens = primary_model._strip_llm_prompt_tokens(enc, out)
            return tok.batch_decode(decoded_tokens, skip_special_tokens=True)[0]
        input_len = int(enc["attention_mask"].sum(dim=1)[0].item())
        return tok.decode(out[0, input_len:], skip_special_tokens=True)

    def _parse_llm_payload(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        cleaned = self._strip_code_fences(text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    @staticmethod
    def _format_duration(total_seconds: float) -> str:
        seconds = max(0, int(round(total_seconds)))
        days, remainder = divmod(seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{days}d:{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _count_source_groups(df: pd.DataFrame, column: str) -> int:
        if df.empty or column not in df.columns:
            return 0
        if "Src" not in df.columns:
            return int(df[column].astype(bool).sum())
        return int(df.groupby("Src", sort=False)[column].any().sum())

    @staticmethod
    def _sha1(value: str) -> str:
        import hashlib

        return hashlib.sha1(value.encode("utf-8")).hexdigest()

    def _channel_gaps(self, group: pd.DataFrame, idxs: Sequence[int]) -> Dict[int, float]:
        channels = ["s_label", "S_struct", "s_hier", "s_sim", "s_attr", "cand_sim"]
        available = [col for col in channels if col in group.columns]
        if not available:
            return {idx: 0.0 for idx in idxs}
        result: Dict[int, float] = {}
        for idx in idxs:
            gaps: List[float] = []
            for col in available:
                vals = [self._safe_float(group.at[row_idx, col], 0.0) for row_idx in idxs]
                pos = list(idxs).index(idx)
                if len(vals) <= 1:
                    gaps.append(0.0)
                    continue
                other = max(vals[:pos] + vals[pos + 1 :])
                gaps.append(max(0.0, vals[pos] - other))
            result[idx] = sum(gaps) / max(1, len(gaps))
        return result

    def _granularity_penalty(self, row: pd.Series) -> float:
        src_spec = self._specificity(row, "src")
        tgt_spec = self._specificity(row, "tgt")
        return abs(src_spec - tgt_spec)

    def _candidate_generation_score(self, row: pd.Series) -> float:
        if "cand_sim_prob" not in row or row.get("cand_sim_prob") is None:
            return 0.0
        prob = self._safe_float(row.get("cand_sim_prob"), 0.0)
        return self._clip01(prob)

    def _reciprocity_score01(self, value: float) -> float:
        return self._clip01(0.5 * (float(value) + 1.0))

    def _gap_score(self, gap: float) -> float:
        return self._sigmoid(float(gap) / max(self.temperature, self.eps))

    def _specificity(self, row: pd.Series, prefix: str) -> float:
        obj_ic = self._clip01(self._safe_float(row.get(f"{prefix}_obj_ic_mean"), 0.0))
        hier = self._scaled_count(row.get(f"{prefix}_hier_total_count"), 20.0)
        obj = self._scaled_count(row.get(f"{prefix}_obj_count"), 50.0)
        attr = self._scaled_count(row.get(f"{prefix}_attr_count"), 20.0)
        return float(0.6 * obj_ic + 0.2 * hier + 0.1 * obj + 0.1 * attr)

    @staticmethod
    def _scaled_count(value: Any, cap: float) -> float:
        try:
            count = max(0.0, float(value))
        except (TypeError, ValueError):
            count = 0.0
        return min(1.0, math.log1p(count) / math.log1p(cap))

    def _reciprocity_scores(self, df: pd.DataFrame) -> Dict[int, float]:
        logits = {
            idx: self._logit(self._clip01(self._safe_float(row.get("S_pair_final"), 0.0)))
            for idx, row in df.iterrows()
        }
        result: Dict[int, float] = {}
        for _, group in df.groupby("Tgt", sort=False):
            idxs = list(group.index)
            probs = self._softmax([logits[idx] for idx in idxs], temperature=self.temperature)
            for idx, prob in zip(idxs, probs):
                result[idx] = 2.0 * float(prob) - 1.0
        return result

    def _distinctive_scores(
        self,
        df: pd.DataFrame,
        record_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[int, float]:
        result: Dict[int, float] = {}
        for _, group in df.groupby("Src", sort=False):
            idxs = list(group.index)
            item_map: Dict[int, List[Tuple[str, float]]] = {}
            document_frequency: Dict[str, int] = {}
            for idx, row in group.iterrows():
                record = record_lookup.get((str(row["Src"]), str(row["Tgt"]))) or {}
                items = self._record_evidence_items(record)
                item_map[idx] = items
                for key in {item_key for item_key, _ in items}:
                    document_frequency[key] = document_frequency.get(key, 0) + 1
            raw_scores: Dict[int, float] = {}
            n = max(1, len(idxs))
            for idx in idxs:
                total = 0.0
                for key, strength in item_map.get(idx, []):
                    idf = math.log((n + 1.0) / (document_frequency.get(key, 0) + 1.0))
                    total += max(0.0, strength) * max(0.0, idf)
                raw_scores[idx] = total
            max_score = max(raw_scores.values(), default=0.0)
            for idx, value in raw_scores.items():
                result[idx] = float(value / max_score) if max_score > self.eps else 0.0
        return result

    def _record_evidence_items(self, record: Mapping[str, Any]) -> List[Tuple[str, float]]:
        if not record:
            return []
        linked_ids = self._linked_item_ids(record.get("cross_side_provenance") or {})
        items: List[Tuple[str, float]] = []

        def _add_item(channel: str, side: str, item: Any, default_strength: float = 1.0) -> None:
            if item is None:
                return
            item_id = ""
            if isinstance(item, Mapping):
                item_id = str(item.get("item_id") or "")
            if side == "source" and item_id and item_id not in linked_ids:
                return
            if side == "source" and not item_id:
                return
            key = self._evidence_key(channel, item)
            if not key:
                return
            strength = self._evidence_strength(item, default_strength)
            side_weight = 1.0 if side == "target" else 0.75
            items.append((key, strength * side_weight))

        attributions = record.get("triple_attributions") or {}
        hierarchy = attributions.get("hierarchy") or {}
        if isinstance(hierarchy, Mapping):
            for family, payload in hierarchy.items():
                payload = payload or {}
                for item in payload.get("target") or []:
                    _add_item(f"hierarchy:{family}", "target", item)
                for item in payload.get("source") or []:
                    _add_item(f"hierarchy:{family}", "source", item)
        for channel in ["similarity", "difference"]:
            payload = attributions.get(channel) or {}
            for item in payload.get("target") or []:
                _add_item(channel, "target", item)
            for item in payload.get("source") or []:
                _add_item(channel, "source", item)

        attrs = record.get("attributes") or {}
        for item in attrs.get("target") or []:
            _add_item("attribute", "target", item)
        for item in attrs.get("source") or []:
            _add_item("attribute", "source", item)

        context_triples = record.get("context_triples") or {}
        for key, value in context_triples.items():
            if key.endswith("_target"):
                if isinstance(value, Mapping):
                    for family, triples in value.items():
                        for triple in triples or []:
                            _add_item(f"{key}:{family}", "target", triple, default_strength=0.5)
                else:
                    for triple in value or []:
                        _add_item(key, "target", triple, default_strength=0.5)
        return items

    def _linked_item_ids(self, payload: Any) -> set[str]:
        linked: set[str] = set()

        def _visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for key in ["item_id", "source_item_id", "target_item_id"]:
                    item_id = value.get(key)
                    if item_id:
                        linked.add(str(item_id))
                for child in value.values():
                    _visit(child)
            elif isinstance(value, list):
                for child in value:
                    _visit(child)

        _visit(payload)
        return linked

    def _evidence_key(self, channel: str, item: Any) -> str:
        if isinstance(item, Mapping):
            if item.get("triple"):
                triple = item.get("triple") or []
                text = "|".join(self._normalize_text(part) for part in list(triple)[:3])
                return f"{channel}:triple:{text}"
            prop = self._normalize_text(item.get("property", item.get("prop", "")))
            value = self._normalize_text(item.get("value", item.get("text", "")))
            if prop or value:
                return f"{channel}:attr:{prop}:{value}"
            text = self._normalize_text(item)
            return f"{channel}:item:{text}" if text else ""
        if isinstance(item, (list, tuple)):
            text = "|".join(self._normalize_text(part) for part in list(item)[:3])
            return f"{channel}:triple:{text}" if text else ""
        text = self._normalize_text(item)
        return f"{channel}:text:{text}" if text else ""

    def _evidence_strength(self, item: Any, default: float) -> float:
        if not isinstance(item, Mapping):
            return float(default)
        for key in ["importance", "score", "edge_ic", "weight", "unsupported_mass"]:
            if key in item:
                return self._clip01(self._safe_float(item.get(key), default))
        return float(default)

    def _sync_results_json(
        self,
        df: pd.DataFrame,
        results_json: Optional[List[Dict[str, Any]]],
    ) -> None:
        if not results_json:
            return
        row_lookup = {
            (str(row["Src"]), str(row["Tgt"])): row
            for _, row in df.iterrows()
        }
        for record in results_json:
            key = (str(record.get("src_iri")), str(record.get("tgt_iri")))
            row = row_lookup.get(key)
            if row is None:
                continue
            conf = record.get("confidences") or {}
            if "S_pair_final" not in conf:
                conf["S_pair_final"] = float(row.get("S_pair_final", conf.get("S_final", 0.0)))
            conf["S_select"] = float(row.get("S_select", 0.0))
            conf["P_select"] = float(row.get("P_select", 0.0))
            conf["selection_margin"] = float(row.get("selection_margin", 0.0))
            conf["selection_entropy"] = float(row.get("selection_entropy", 0.0))
            conf["selection_no_match_prob"] = float(row.get("selection_no_match_prob", 0.0))
            conf["selection_distinctive"] = float(row.get("selection_distinctive", 0.0))
            conf["selection_utility"] = float(row.get("selection_utility", 0.0))
            conf["P_rank"] = float(row.get("P_rank", 0.0))
            conf["P_match"] = float(row.get("P_match", 0.0))
            conf["selection_accept_threshold"] = float(row.get("selection_accept_threshold", 0.0))
            if self.replace_final_score:
                conf["S_final"] = float(row.get("S_select", 0.0))
            record["confidences"] = conf

            pred = record.get("prediction") or {}
            pred["selector_abstained"] = bool(row.get("selection_abstained", False))
            pred["selector_llm_used"] = bool(row.get("selection_llm_used", False))
            pred["selector_reason"] = str(row.get("selection_reason", ""))
            pred["selector_winner"] = bool(row.get("selection_winner", False))
            pred["selector_strategy"] = self.strategy
            record["prediction"] = pred

    def _record_lookup(
        self,
        results_json: Optional[List[Dict[str, Any]]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record in results_json or []:
            src = record.get("src_iri")
            tgt = record.get("tgt_iri")
            if src is None or tgt is None:
                continue
            lookup[(str(src), str(tgt))] = record
        return lookup

    def _softmax(self, values: Sequence[float], temperature: float) -> List[float]:
        if not values:
            return []
        temp = max(float(temperature), self.eps)
        scaled = [float(value) / temp for value in values]
        max_value = max(scaled)
        exp_values = [math.exp(value - max_value) for value in scaled]
        denom = sum(exp_values)
        if denom <= self.eps:
            return [1.0 / len(values) for _ in values]
        return [value / denom for value in exp_values]

    def _normalized_entropy(self, probs: Sequence[float]) -> float:
        if len(probs) <= 1:
            return 0.0
        entropy = -sum(float(p) * math.log(max(float(p), self.eps)) for p in probs)
        return float(entropy / math.log(len(probs)))

    def _logit(self, value: float) -> float:
        clipped = self._clip(float(value), self.eps, 1.0 - self.eps)
        return math.log(clipped / (1.0 - clipped))

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def _clip01(self, value: float) -> float:
        return self._clip(value, 0.0, 1.0)

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _std(values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    def _optional_probability(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return self._clip01(self._safe_float(value, 0.0))

    @staticmethod
    def _safe_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "off", "no", "none", "null"}
        return bool(value)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or pd.isna(value):
                return float(default)
        except (TypeError, ValueError):
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _normalize_text(value: Any) -> str:
        text = str(value or "").strip().lower()
        return re.sub(r"\s+", " ", text)
