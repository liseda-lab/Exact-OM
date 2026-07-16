from __future__ import annotations

import inspect  # noqa: F401
import json  # noqa: F401
import math  # noqa: F401
import random  # noqa: F401
import re  # noqa: F401
import time  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import (  # noqa: F401
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import pandas as pd  # noqa: F401
import torch  # noqa: F401

from exact.core.contracts.model import IModel
from exact.utils.data import read_table  # noqa: F401
from exact.utils.formatting import (  # noqa: F401
    clip01,
    format_duration,
    strip_code_fences,
)
from exact.utils.provenance import file_provenance  # noqa: F401

from .acceptance import AcceptanceMixin
from .calibration import CalibrationMixin
from .features import FeatureEngineeringMixin


class CandidateSetSelector(CalibrationMixin, AcceptanceMixin, FeatureEngineeringMixin, IModel):
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
        "enabled": False,
        "mode": "off",
        "ambiguity_margin": 0.08,
        "max_candidates": 5,
        "max_new_tokens": 256,
        "trigger_acceptance_margin": 0.025,
        "trigger_rank_margin": 0.03,
        "min_confidence": 0.75,
    }
    DEFAULT_CALIBRATION = {
        "enabled": "auto",
        "min_positive_sources": 50,
        "background_negative_weight": 0.02,
        "background_negative_weight_grid": [0.02, 0.05, 0.10, 0.20, 0.40],
        "validation_fraction": 0.2,
        "validation_folds": 5,
        "l2": 1.0e-3,
        "learning_rate": 0.05,
        "max_epochs": 200,
        "threshold_grid_step": 0.005,
        "accept_objective": "f1",
        "f_beta": 1.5,
        "min_precision": None,
        "min_recall": None,
        "exact_prefiltered_source_policy": "hard_negative",
        "exact_prefiltered_negative_weight": 1.0,
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
    _strip_code_fences = staticmethod(strip_code_fences)
    _format_duration = staticmethod(format_duration)
    _clip01 = staticmethod(clip01)

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
        score_mode: str = "p_match",
        calibration: Optional[Dict[str, Any]] = None,
        training_reference_file_path: Optional[Any] = None,
        request_seed: Optional[int] = None,
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
        self.no_match_threshold = self._resolve_no_match_threshold(
            no_match_threshold, no_match or {}
        )
        # Compatibility attributes for external callers that inspect model params.
        self.weights = {
            "support_weight": self.support_weight,
            "auxiliary_evidence": 1.0 - self.support_weight,
        }
        self.no_match = {"threshold": self.no_match_threshold}
        self.llm = self._normalize_llm(llm or {})
        self.strategy = str(strategy or "heuristic")
        self.score_mode = self._normalize_score_mode(score_mode)
        self.calibration = self._normalize_calibration(calibration or {})
        self.training_reference_file_path = (
            str(training_reference_file_path) if training_reference_file_path else None
        )
        self.training_reference_provenance: Optional[Dict[str, Any]] = None
        if self.training_reference_file_path:
            reference_path = Path(self.training_reference_file_path).expanduser()
            if reference_path.is_file():
                self.training_reference_provenance = file_provenance(reference_path)
        self.request_seed = int(request_seed) if request_seed is not None else 0
        self._llm_prompts_used = 0
        self._llm_warning_logged = False
        self._calibration_meta: Dict[str, Any] = {}
        self._ignored_legacy_kwargs = dict(kwargs or {})

    def runtime_fingerprint_payload(self, **_: Any) -> Dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "implementation_fingerprint": self._implementation_fingerprint(),
            "enabled": self.enabled,
            "global_only": self.global_only,
            "replace_final_score": self.replace_final_score,
            "use_no_match": self.use_no_match,
            "temperature": self.temperature,
            "support_weight": self.support_weight,
            "no_match_threshold": self.no_match_threshold,
            "llm": dict(self.llm),
            "strategy": self.strategy,
            "score_mode": self.score_mode,
            "calibration": dict(self.calibration),
            "training_reference_file_path": self.training_reference_file_path,
            "training_reference_sha256": (
                self.training_reference_provenance.get("sha256")
                if self.training_reference_provenance
                else None
            ),
            "request_seed": self.request_seed,
        }

    def runtime_fingerprint(self) -> str:
        blob = json.dumps(self.runtime_fingerprint_payload(), sort_keys=True, default=str)
        return self._sha1(blob)

    @classmethod
    def _implementation_fingerprint(cls) -> str:
        try:
            source = inspect.getsource(cls)
        except (OSError, TypeError):
            source = f"{cls.__module__}.{cls.__qualname__}"
        return cls._sha1(source)

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
        competition_fallback = max(
            [float(weights[key]) for key in competition_keys if key in weights] or [0.0]
        )
        penalty_fallback = max(
            [float(weights[key]) for key in ["granularity", "contradiction"] if key in weights]
            or [0.0]
        )
        legacy = {
            "absolute_support": float(
                weights.get(
                    "absolute_support", weights.get("pairwise", weights.get("absolute", 0.0))
                )
            ),
            "candidate_competition": float(
                weights.get(
                    "candidate_competition", weights.get("competition", competition_fallback)
                )
            ),
            "distinctive_evidence": float(
                weights.get(
                    "distinctive_evidence", weights.get("evidence", weights.get("distinctive", 0.0))
                )
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
        explicit_enabled = "enabled" in llm
        explicit_mode = "mode" in llm
        mode = str(normalized.get("mode") or "off").strip().lower()
        if explicit_enabled and not bool(normalized.get("enabled")):
            mode = "off"
        elif (
            explicit_enabled
            and bool(normalized.get("enabled"))
            and not explicit_mode
            and mode == "off"
        ):
            mode = "veto"
        if mode in {"false", "0", "disabled", "none"}:
            mode = "off"
        if mode not in {"off", "veto", "choose"}:
            mode = "veto"
        normalized["mode"] = mode
        if mode == "off":
            normalized["enabled"] = False
        else:
            normalized["enabled"] = bool(normalized.get("enabled", True))
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
        return normalized

    @staticmethod
    def _normalize_score_mode(score_mode: str) -> str:
        mode = str(score_mode or "p_match").strip().lower()
        if mode in {"p", "probability", "prob", "match", "raw"}:
            return "p_match"
        if mode in {"threshold", "threshold_compatible", "legacy"}:
            return "threshold_compatible"
        return "p_match"

    def _normalize_calibration(self, calibration: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(self.DEFAULT_CALIBRATION)
        normalized.update({key: value for key, value in calibration.items() if key in normalized})
        normalized["enabled"] = str(normalized["enabled"]).lower()
        normalized["min_positive_sources"] = max(1, int(normalized["min_positive_sources"]))
        normalized["background_negative_weight"] = max(
            0.0, float(normalized["background_negative_weight"])
        )
        grid_values = normalized.get("background_negative_weight_grid")
        if grid_values is None:
            grid_values = [normalized["background_negative_weight"]]
        if isinstance(grid_values, str):
            grid_values = [value.strip() for value in grid_values.split(",") if value.strip()]
        try:
            grid = [max(0.0, float(value)) for value in list(grid_values)]
        except (TypeError, ValueError):
            grid = [float(normalized["background_negative_weight"])]
        if not grid:
            grid = [float(normalized["background_negative_weight"])]
        normalized["background_negative_weight_grid"] = sorted(set(grid))
        normalized["validation_fraction"] = self._clip(
            float(normalized["validation_fraction"]),
            0.0,
            0.5,
        )
        normalized["validation_folds"] = max(1, int(normalized["validation_folds"]))
        normalized["l2"] = max(0.0, float(normalized["l2"]))
        normalized["learning_rate"] = max(1.0e-6, float(normalized["learning_rate"]))
        normalized["max_epochs"] = max(1, int(normalized["max_epochs"]))
        normalized["threshold_grid_step"] = self._clip(
            float(normalized["threshold_grid_step"]), 1.0e-4, 0.5
        )
        objective = str(normalized["accept_objective"] or "f1").lower()
        if objective not in {"f1", "f_beta", "recall_at_precision"}:
            objective = "f1"
        normalized["accept_objective"] = objective
        normalized["f_beta"] = max(1.0e-6, float(normalized["f_beta"]))
        normalized["min_precision"] = self._optional_probability(normalized["min_precision"])
        normalized["min_recall"] = self._optional_probability(normalized["min_recall"])
        exact_policy = str(normalized["exact_prefiltered_source_policy"] or "hard_negative").lower()
        if exact_policy not in {"hard_negative", "exclude", "legacy"}:
            exact_policy = "hard_negative"
        normalized["exact_prefiltered_source_policy"] = exact_policy
        normalized["exact_prefiltered_negative_weight"] = max(
            0.0,
            float(normalized["exact_prefiltered_negative_weight"]),
        )
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
        target_cardinality: Optional[int] = None,
        log_every: int = 10,
        checkpoint_callback: Optional[Callable[[pd.DataFrame], None]] = None,
        checkpoint_every_groups: Optional[int] = None,
        run_progress: Optional[Any] = None,
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

        n_rows = int(len(df))
        n_sources = int(df["Src"].nunique()) if "Src" in df.columns else 0
        self._log(
            logger,
            (
                "Candidate-set selector started: "
                f"strategy={self.strategy}, rows={n_rows}, sources={n_sources}, "
                f"use_no_match={self.use_no_match}, llm_enabled={bool(self.llm.get('enabled', False))}."
            ),
            "info",
        )
        if run_progress is not None:
            run_progress.update(
                "PostInference",
                fraction=0.30,
                detail=f"selector={self.strategy}, rows={n_rows}, sources={n_sources}",
                force=True,
            )
        record_lookup = self._record_lookup(results_json)
        distinctive = self._distinctive_scores(
            df, record_lookup, logger=logger, log_every=log_every
        )
        if run_progress is not None:
            run_progress.update(
                "PostInference",
                fraction=0.40,
                detail="selector evidence scan completed",
                force=True,
            )
        reciprocity = self._reciprocity_scores(df)

        defaults = {
            "S_select": 0.0,
            "P_select": 0.0,
            "selection_margin": 0.0,
            "selection_entropy": 0.0,
            "selection_no_match_prob": 0.0,
            "selection_evidence_support": 0.0,
            "selection_abstained": False,
            "selection_llm_used": False,
            "selection_reason": "not_selected",
            "selection_distinctive": 0.0,
            "selection_utility": 0.0,
            "P_rank": 0.0,
            "P_match": 0.0,
            "selection_winner": False,
            "selection_accept_threshold": 0.0,
            "selection_target_conflict_enabled": False,
            "selection_target_cardinality": 0,
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
                dataset=dataset,
                logger=logger,
                record_lookup=record_lookup,
                threshold=threshold,
                target_cardinality=target_cardinality,
                log_every=log_every,
                checkpoint_callback=checkpoint_callback,
                checkpoint_every_groups=checkpoint_every_groups,
                run_progress=run_progress,
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
                    run_progress=run_progress,
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
                run_progress=run_progress,
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
        run_progress: Optional[Any],
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
                eta = (
                    self._format_duration(remaining / rate)
                    if rate > 0
                    else self._format_duration(0.0)
                )
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
                if run_progress is not None:
                    run_progress.update(
                        "PostInference",
                        fraction=0.40 + 0.45 * (n_groups / max(1, n_sources)),
                        detail=f"selector sources={n_groups}/{n_sources}",
                        force=n_groups == n_sources,
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
        dataset: Any,
        logger: Optional[Any],
        record_lookup: Dict[Tuple[str, str], Dict[str, Any]],
        threshold: Optional[float],
        target_cardinality: Optional[int],
        log_every: int,
        checkpoint_callback: Optional[Callable[[pd.DataFrame], None]],
        checkpoint_every_groups: Optional[int],
        run_progress: Optional[Any],
    ) -> Optional[pd.DataFrame]:
        if self.calibration.get("enabled") in {"false", "0", "off", "disabled"}:
            self._log(logger, "Calibrated selector disabled; using heuristic selector.", "info")
            return None

        ref_pairs = self._load_training_reference_pairs(logger)
        ref_pairs = self._filter_training_reference_pairs_for_dataset(ref_pairs, dataset, logger)
        if not ref_pairs:
            self._log(
                logger,
                "Calibrated selector skipped because no training reference was provided.",
                "warning",
            )
            return None

        exact_pairs_for_reporting = self._exact_prefiltered_pairs(dataset)
        if self.calibration["exact_prefiltered_source_policy"] == "legacy":
            calibration_ref_pairs = set(ref_pairs)
            exact_prefiltered_ref_pairs: set[Tuple[str, str]] = set()
            exact_prefiltered_ref_sources: set[str] = set()
        else:
            (
                calibration_ref_pairs,
                exact_prefiltered_ref_pairs,
                exact_prefiltered_ref_sources,
            ) = self._exclude_exact_prefiltered_reference_sources(ref_pairs, dataset, logger)

        rank_features = self._rank_feature_rows(df, distinctive, reciprocity)
        _, n_positive_sources = self._rank_training_groups(df, calibration_ref_pairs)
        min_positive = int(self.calibration["min_positive_sources"])
        utilities: Dict[int, float] = {}
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

        self._log(
            logger,
            (
                "Calibrated selector fitting started: "
                f"candidate_rows={len(df)}, candidate_sources={int(df['Src'].nunique())}, "
                f"reference_pairs={len(ref_pairs)}, usable_positive_sources={n_positive_sources}."
            ),
            "info",
        )
        calibration_choice = self._select_accept_model_by_oof_validation(
            df=df,
            rank_features=rank_features,
            distinctive=distinctive,
            calibration_ref_pairs=calibration_ref_pairs,
            exact_prefiltered_sources=exact_prefiltered_ref_sources,
            primary_model=primary_model,
            logger=logger,
            record_lookup=record_lookup,
            target_cardinality=target_cardinality,
            protected_exact_pairs=exact_pairs_for_reporting,
        )
        if calibration_choice is None:
            train_ref_pairs, validation_ref_pairs, split_meta = (
                self._split_reference_pairs_by_source(calibration_ref_pairs)
            )
            rank_groups, split_positive_sources = self._rank_training_groups(df, train_ref_pairs)
            if split_positive_sources < min_positive:
                self._log(
                    logger,
                    (
                        "Calibrated selector skipped because usable positive training sources "
                        f"({split_positive_sources}) are below min_positive_sources={min_positive} "
                        "after the validation split."
                    ),
                    "warning",
                )
                return None

            rank_model = self._fit_rank_model(
                rank_features,
                rank_groups,
                logger=logger,
                label="held-out rank",
            )
            if rank_model is None:
                self._log(
                    logger,
                    "Calibrated selector failed to fit rank model; using heuristic selector.",
                    "warning",
                )
                return None
            utilities = {
                idx: self._linear_score(rank_features[idx], rank_model) for idx in rank_features
            }

            calibration_choice = self._select_accept_model_by_validation(
                df=df,
                utilities=utilities,
                rank_features=rank_features,
                distinctive=distinctive,
                train_ref_pairs=train_ref_pairs,
                validation_ref_pairs=validation_ref_pairs,
                exact_prefiltered_sources=exact_prefiltered_ref_sources,
                primary_model=primary_model,
                logger=logger,
                record_lookup=record_lookup,
                target_cardinality=target_cardinality,
                protected_exact_pairs=exact_pairs_for_reporting,
            )
        else:
            train_ref_pairs = set(calibration_ref_pairs)
            validation_ref_pairs = set(calibration_ref_pairs)
            split_meta = dict(calibration_choice.get("validation_split", {}))
        if calibration_choice is None:
            self._log(
                logger,
                "Calibrated selector failed to fit acceptance model; using heuristic selector.",
                "warning",
            )
            return None
        utilities = calibration_choice.get("utilities", utilities)
        source_decisions = calibration_choice["source_decisions"]
        accept_threshold = float(calibration_choice["accept_threshold"])
        threshold_metrics = dict(calibration_choice["threshold_metrics"])
        oof_accept_threshold = calibration_choice.get("oof_accept_threshold")
        oof_threshold_metrics = dict(calibration_choice.get("oof_threshold_metrics", {}))
        final_refit_threshold_metrics = dict(
            calibration_choice.get("final_refit_threshold_metrics", {})
        )
        target_conflict_enabled = bool(calibration_choice.get("target_conflict_enabled", False))
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
        evidence_support_floor = self._evidence_support_floor(
            score_threshold=score_threshold,
            accept_threshold=accept_threshold,
        )

        for n_groups, (src, group) in enumerate(df.groupby("Src", sort=False), start=1):
            decision = source_decisions.get(str(src))
            if decision is None:
                continue
            p_match = float(decision["p_match"])
            accepted = p_match >= accept_threshold
            winner_idx = int(decision["winner_idx"])
            evidence_support = float(decision.get("evidence_support", 0.0))
            reason = "calibrated" if accepted else "calibrated_no_match"
            llm_used = False

            if self._should_use_llm_calibrated(
                acceptance_margin=abs(p_match - accept_threshold),
                rank_margin=float(decision["rank_prob_margin"]),
                primary_model=primary_model,
                accepted=accepted,
                evidence_support=evidence_support,
                evidence_support_floor=evidence_support_floor,
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
                    elif accepted and str(self.llm.get("mode", "veto")).lower() != "veto":
                        chosen_idx = llm_choice.get("winner_idx")
                        if chosen_idx is not None:
                            winner_idx = int(chosen_idx)
                            reason = "llm"

            final_score = self._final_selector_score(
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
                df.at[row_idx, "selection_target_conflict_enabled"] = bool(target_conflict_enabled)
                df.at[row_idx, "selection_target_cardinality"] = int(target_cardinality or 0)
                df.at[row_idx, "selection_margin"] = float(decision["utility_margin"])
                df.at[row_idx, "selection_entropy"] = float(decision["rank_entropy"])
                df.at[row_idx, "selection_no_match_prob"] = float(1.0 - p_match)
                df.at[row_idx, "selection_evidence_support"] = evidence_support
                df.at[row_idx, "selection_abstained"] = bool(not accepted)
                df.at[row_idx, "selection_llm_used"] = bool(llm_used)
                df.at[row_idx, "selection_reason"] = reason
                df.at[row_idx, "selection_distinctive"] = float(distinctive.get(row_idx, 0.0))
                df.at[row_idx, "selection_utility"] = float(utilities.get(row_idx, 0.0))

            if n_groups == n_sources or n_groups % progress_interval == 0:
                elapsed_progress = max(1.0e-8, time.perf_counter() - progress_start)
                rate = n_groups / elapsed_progress
                remaining = max(0, n_sources - n_groups)
                eta = (
                    self._format_duration(remaining / rate)
                    if rate > 0
                    else self._format_duration(0.0)
                )
                self._log(
                    logger,
                    (
                        "Calibrated selector progress: "
                        f"sources={n_groups}/{n_sources}, abstained_sources={n_abstained}, "
                        f"llm_sources={n_llm}, avg={rate:.2f} sources/s, ETA {eta}"
                    ),
                    "debug",
                )
                if run_progress is not None:
                    run_progress.update(
                        "PostInference",
                        fraction=0.40 + 0.45 * (n_groups / max(1, n_sources)),
                        detail=(
                            f"calibrated selector sources={n_groups}/{n_sources}, "
                            f"abstained={n_abstained}, llm={n_llm}"
                        ),
                        force=n_groups == n_sources,
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
            "training_reference": self.training_reference_provenance,
            "training_reference_sha256": (
                self.training_reference_provenance.get("sha256")
                if self.training_reference_provenance
                else None
            ),
            "n_reference_pairs": len(ref_pairs),
            "n_calibration_reference_pairs": len(calibration_ref_pairs),
            "n_calibration_train_reference_pairs": len(train_ref_pairs),
            "n_validation_reference_pairs": len(validation_ref_pairs),
            "validation_split": dict(split_meta),
            "n_exact_prefiltered_reference_pairs": len(exact_prefiltered_ref_pairs),
            "n_exact_prefiltered_reference_sources": len(exact_prefiltered_ref_sources),
            "exact_training_reference_tp": len(exact_pairs_for_reporting.intersection(ref_pairs)),
            "exact_training_reference_fp": len(exact_pairs_for_reporting.difference(ref_pairs)),
            "exact_prefiltered_source_policy": self.calibration["exact_prefiltered_source_policy"],
            "exact_prefiltered_negative_weight": self.calibration[
                "exact_prefiltered_negative_weight"
            ],
            "selected_background_negative_weight": self.calibration["background_negative_weight"],
            "selected_llm_mode": self.llm.get("mode"),
            "selected_target_conflict_enabled": target_conflict_enabled,
            "selected_target_cardinality": target_cardinality,
            "score_mode": self.score_mode,
            "n_positive_sources": n_positive_sources,
            "rank_feature_names": list(self.RANK_FEATURE_NAMES),
            "accept_feature_names": list(self.ACCEPT_FEATURE_NAMES),
            "accept_threshold": accept_threshold,
            "oof_accept_threshold": oof_accept_threshold,
            "accept_objective": threshold_metrics.get("accept_objective"),
            "accept_selected_metrics": threshold_metrics.get("selected_metrics", {}),
            "accept_best_f1_metrics": threshold_metrics.get("best_f1", {}),
            "accept_best_f_beta_metrics": threshold_metrics.get("best_f_beta", {}),
            "accept_recall_at_precision_metrics": threshold_metrics.get("recall_at_precision", {}),
            "accept_fallback_to_f1": bool(threshold_metrics.get("fallback_to_f1", False)),
            "threshold_metrics": threshold_metrics,
            "oof_threshold_metrics": oof_threshold_metrics,
            "final_refit_threshold_metrics": final_refit_threshold_metrics,
            "llm_groups": n_llm,
            "abstained_groups": n_abstained,
            "calibration": dict(self.calibration),
        }
        refit_diagnostic = ""
        if oof_accept_threshold is not None:
            refit_diagnostic = (
                f"oof_accept_threshold={float(oof_accept_threshold):.3f}, "
                f"oof_validation_P={oof_threshold_metrics.get('P', 0.0):.3f}, "
                f"oof_validation_R={oof_threshold_metrics.get('R', 0.0):.3f}, "
                f"oof_validation_F1={oof_threshold_metrics.get('F1', 0.0):.3f}, "
                f"final_refit_selected_sources={threshold_metrics.get('final_refit_selected_sources', 0)}, "
            )
        self._log(
            logger,
            (
                "Calibrated selector: "
                f"usable_positive_sources={n_positive_sources}, "
                f"exact_prefiltered_reference_sources={len(exact_prefiltered_ref_sources)}, "
                f"exact_train_TP={len(exact_pairs_for_reporting.intersection(ref_pairs))}, "
                f"exact_train_FP={len(exact_pairs_for_reporting.difference(ref_pairs))}, "
                f"exact_prefiltered_policy={self.calibration['exact_prefiltered_source_policy']}, "
                f"validation_sources={split_meta.get('n_validation_sources', 0)}, "
                f"background_negative_weight={self.calibration['background_negative_weight']:.3f}, "
                f"llm_mode={self.llm.get('mode')}, "
                f"target_conflict_enabled={target_conflict_enabled}, "
                f"accept_objective={threshold_metrics.get('accept_objective')}, "
                f"metrics_scope={threshold_metrics.get('validation_scope', 'validation')}, "
                f"{refit_diagnostic}"
                f"accept_threshold={accept_threshold:.3f}, "
                f"selected_P={threshold_metrics.get('P', 0.0):.3f}, "
                f"selected_R={threshold_metrics.get('R', 0.0):.3f}, "
                f"selected_F1={threshold_metrics.get('F1', 0.0):.3f}, "
                f"selected_TP={threshold_metrics.get('TP', 0.0):.1f}, "
                f"selected_FP={threshold_metrics.get('FP', 0.0):.1f}, "
                f"selected_FN={threshold_metrics.get('FN', 0.0):.1f}, "
                f"selected_F_beta={threshold_metrics.get('F_beta', 0.0):.3f}, "
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
