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


class CalibrationMixin:
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
            self._log(
                logger, f"Training reference file {path} has fewer than two columns.", "warning"
            )
            return set()
        src_col, tgt_col = ref_df.columns[:2]
        pairs = {
            (str(row[src_col]), str(row[tgt_col]))
            for _, row in ref_df.iterrows()
            if str(row[src_col]) and str(row[tgt_col])
        }
        return pairs

    def _filter_training_reference_pairs_for_dataset(
        self,
        ref_pairs: set[Tuple[str, str]],
        dataset: Any,
        logger: Optional[Any],
    ) -> set[Tuple[str, str]]:
        if dataset is None or not ref_pairs:
            return ref_pairs
        try:
            src_ignored = set(getattr(dataset, "source_ignored_alignment_classes", set()) or set())
            tgt_ignored = set(getattr(dataset, "target_ignored_alignment_classes", set()) or set())
        except Exception:
            return ref_pairs
        if not src_ignored and not tgt_ignored:
            return ref_pairs
        filtered = {
            (src, tgt)
            for src, tgt in ref_pairs
            if str(src) not in src_ignored and str(tgt) not in tgt_ignored
        }
        removed = len(ref_pairs) - len(filtered)
        if removed:
            self._log(
                logger,
                f"Calibrated selector filtered {removed} training references with ignored alignment classes.",
                "debug",
            )
        return filtered

    def _exclude_exact_prefiltered_reference_sources(
        self,
        ref_pairs: set[Tuple[str, str]],
        dataset: Any,
        logger: Optional[Any],
    ) -> Tuple[set[Tuple[str, str]], set[Tuple[str, str]], set[str]]:
        exact_pairs = self._exact_prefiltered_pairs(dataset)
        if not exact_pairs:
            return set(ref_pairs), set(), set()

        exact_ref_pairs = set(ref_pairs).intersection(exact_pairs)
        exact_ref_sources = {src for src, _ in exact_ref_pairs}
        if not exact_ref_sources:
            return set(ref_pairs), set(), set()

        filtered_ref_pairs = {pair for pair in ref_pairs if pair[0] not in exact_ref_sources}
        self._log(
            logger,
            (
                "Calibrated selector excluded exact-prefiltered training references: "
                f"pairs={len(exact_ref_pairs)}, sources={len(exact_ref_sources)}."
            ),
            "debug",
        )
        return filtered_ref_pairs, exact_ref_pairs, exact_ref_sources

    def _exact_prefiltered_pairs(self, dataset: Any) -> set[Tuple[str, str]]:
        if dataset is None:
            return set()

        pairs: set[Tuple[str, str]] = set()
        try:
            dataset_df = getattr(dataset, "dataframe", None)
        except Exception:
            dataset_df = None

        if isinstance(dataset_df, pd.DataFrame) and not dataset_df.empty:
            pairs.update(self._exact_prefiltered_pairs_from_dataframe(dataset_df))
            if pairs:
                return pairs

        try:
            exact_matches = getattr(dataset, "exact_matches", None)
        except Exception:
            exact_matches = None
        if isinstance(exact_matches, pd.DataFrame) and not exact_matches.empty:
            if {"Src", "Tgt"}.issubset(exact_matches.columns):
                for src, tgt in exact_matches[["Src", "Tgt"]].dropna().itertuples(index=False):
                    pairs.add((str(src), str(tgt)))
        return pairs

    def _exact_prefiltered_pairs_from_dataframe(
        self, dataset_df: pd.DataFrame
    ) -> set[Tuple[str, str]]:
        if not {"Src", "Tgt"}.issubset(dataset_df.columns):
            return set()
        if "prefiltered" not in dataset_df.columns:
            return set()

        prefiltered = self._bool_series(dataset_df["prefiltered"])
        score_column = None
        for candidate in ("Scores", "Score"):
            if candidate in dataset_df.columns:
                score_column = candidate
                break
        if score_column is not None:
            scores = pd.to_numeric(dataset_df[score_column], errors="coerce").fillna(0.0)
            prefiltered = prefiltered & (scores >= 1.0 - self.eps)

        rows = dataset_df.loc[prefiltered, ["Src", "Tgt"]].dropna()
        return {(str(src), str(tgt)) for src, tgt in rows.itertuples(index=False)}

    @staticmethod
    def _bool_series(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float) != 0.0
        return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})

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

    def _split_reference_pairs_by_source(
        self,
        ref_pairs: set[Tuple[str, str]],
    ) -> Tuple[set[Tuple[str, str]], set[Tuple[str, str]], Dict[str, Any]]:
        clean_pairs = {(str(src), str(tgt)) for src, tgt in ref_pairs if str(src) and str(tgt)}
        sources = sorted({src for src, _ in clean_pairs})
        validation_fraction = float(self.calibration.get("validation_fraction", 0.0))
        min_positive = int(self.calibration.get("min_positive_sources", 1))
        meta: Dict[str, Any] = {
            "seed": int(self.request_seed),
            "validation_fraction": validation_fraction,
            "source_disjoint": True,
            "held_out": False,
            "n_sources": len(sources),
            "n_train_sources": len(sources),
            "n_validation_sources": len(sources),
        }
        if len(sources) <= 1 or validation_fraction <= 0.0:
            meta["source_disjoint"] = False
            return set(clean_pairs), set(clean_pairs), meta

        max_validation_sources = len(sources) - min(min_positive, max(1, len(sources) - 1))
        if max_validation_sources <= 0:
            meta["source_disjoint"] = False
            return set(clean_pairs), set(clean_pairs), meta

        requested_validation_sources = max(1, int(round(len(sources) * validation_fraction)))
        n_validation_sources = min(requested_validation_sources, max_validation_sources)
        shuffled_sources = list(sources)
        random.Random(int(self.request_seed)).shuffle(shuffled_sources)
        validation_sources = set(shuffled_sources[:n_validation_sources])
        train_sources = set(sources).difference(validation_sources)

        train_ref_pairs = {pair for pair in clean_pairs if pair[0] in train_sources}
        validation_ref_pairs = {pair for pair in clean_pairs if pair[0] in validation_sources}
        if not train_ref_pairs or not validation_ref_pairs:
            meta["source_disjoint"] = False
            return set(clean_pairs), set(clean_pairs), meta

        meta.update(
            {
                "held_out": True,
                "n_train_sources": len(train_sources),
                "n_validation_sources": len(validation_sources),
                "n_train_pairs": len(train_ref_pairs),
                "n_validation_pairs": len(validation_ref_pairs),
            }
        )
        return train_ref_pairs, validation_ref_pairs, meta

    def _reference_source_folds(
        self,
        ref_pairs: set[Tuple[str, str]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        clean_pairs = {(str(src), str(tgt)) for src, tgt in ref_pairs if str(src) and str(tgt)}
        sources = sorted({src for src, _ in clean_pairs})
        requested_folds = max(1, int(self.calibration.get("validation_folds", 1)))
        validation_fraction = float(self.calibration.get("validation_fraction", 0.0))
        meta: Dict[str, Any] = {
            "seed": int(self.request_seed),
            "validation_fraction": validation_fraction,
            "validation_folds": requested_folds,
            "source_disjoint": True,
            "held_out": False,
            "strategy": "kfold",
            "n_sources": len(sources),
            "n_folds": 0,
            "fold_source_counts": [],
        }
        if requested_folds <= 1 or validation_fraction <= 0.0 or len(sources) <= 1:
            meta["source_disjoint"] = False
            return [], meta

        n_folds = min(requested_folds, len(sources))
        if n_folds <= 1:
            meta["source_disjoint"] = False
            return [], meta

        shuffled_sources = list(sources)
        random.Random(int(self.request_seed)).shuffle(shuffled_sources)
        source_folds = [set(shuffled_sources[fold_idx::n_folds]) for fold_idx in range(n_folds)]
        folds: List[Dict[str, Any]] = []
        for fold_idx, validation_sources in enumerate(source_folds):
            if not validation_sources:
                continue
            train_sources = set(sources).difference(validation_sources)
            train_ref_pairs = {pair for pair in clean_pairs if pair[0] in train_sources}
            validation_ref_pairs = {pair for pair in clean_pairs if pair[0] in validation_sources}
            if not train_ref_pairs or not validation_ref_pairs:
                continue
            folds.append(
                {
                    "fold": fold_idx,
                    "train_ref_pairs": train_ref_pairs,
                    "validation_ref_pairs": validation_ref_pairs,
                    "validation_sources": set(validation_sources),
                }
            )

        if len(folds) <= 1:
            meta["source_disjoint"] = False
            return [], meta

        meta.update(
            {
                "held_out": True,
                "n_folds": len(folds),
                "fold_source_counts": [len(fold["validation_sources"]) for fold in folds],
                "n_train_sources": len(sources),
                "n_validation_sources": len(sources),
                "n_train_pairs": len(clean_pairs),
                "n_validation_pairs": len(clean_pairs),
            }
        )
        return folds, meta

    def _select_accept_model_by_oof_validation(
        self,
        df: pd.DataFrame,
        rank_features: Dict[int, List[float]],
        distinctive: Dict[int, float],
        calibration_ref_pairs: set[Tuple[str, str]],
        exact_prefiltered_sources: set[str],
        primary_model: Optional[IModel],
        logger: Optional[Any],
        record_lookup: Dict[Tuple[str, str], Dict[str, Any]],
        target_cardinality: Optional[int],
        protected_exact_pairs: set[Tuple[str, str]],
    ) -> Optional[Dict[str, Any]]:
        folds, fold_meta = self._reference_source_folds(calibration_ref_pairs)
        if not folds:
            return None

        original_background_weight = float(self.calibration["background_negative_weight"])
        original_llm_mode = str(self.llm.get("mode", "veto"))
        min_positive = int(self.calibration["min_positive_sources"])
        best_choice: Optional[Dict[str, Any]] = None
        weights_grid = list(self.calibration.get("background_negative_weight_grid", []))
        self._log(
            logger,
            (
                "Calibrated selector OOF validation started: "
                f"background_weights={weights_grid}, folds={len(folds)}, "
                f"rank_features={len(rank_features)}."
            ),
            "debug",
        )

        for weight_idx, background_weight in enumerate(weights_grid, start=1):
            weight_start = time.perf_counter()
            self.calibration["background_negative_weight"] = float(background_weight)
            eval_decisions: Dict[str, Dict[str, Any]] = {}
            fold_failed = False
            fold_positive_sources: List[int] = []
            self._log(
                logger,
                (
                    "Calibrated selector OOF background weight started: "
                    f"{weight_idx}/{len(weights_grid)} weight={float(background_weight):.3f}."
                ),
                "debug",
            )

            for fold_pos, fold in enumerate(folds, start=1):
                fold_start = time.perf_counter()
                train_ref_pairs = set(fold["train_ref_pairs"])
                validation_ref_pairs = set(fold["validation_ref_pairs"])
                validation_sources = set(fold["validation_sources"])
                rank_groups, n_positive_sources = self._rank_training_groups(df, train_ref_pairs)
                fold_positive_sources.append(n_positive_sources)
                self._log(
                    logger,
                    (
                        "Calibrated selector OOF fold started: "
                        f"weight={float(background_weight):.3f}, fold={fold_pos}/{len(folds)}, "
                        f"rank_groups={len(rank_groups)}, positive_sources={n_positive_sources}, "
                        f"validation_sources={len(validation_sources)}."
                    ),
                    "debug",
                )
                if n_positive_sources < min_positive:
                    fold_failed = True
                    self._log(
                        logger,
                        (
                            "Calibrated selector OOF fold failed: "
                            f"positive_sources={n_positive_sources} below min_positive_sources={min_positive}."
                        ),
                        "warning",
                    )
                    break
                rank_model = self._fit_rank_model(
                    rank_features,
                    rank_groups,
                    logger=logger,
                    label=f"OOF weight={float(background_weight):.3f} fold={fold_pos}/{len(folds)} rank",
                )
                if rank_model is None:
                    fold_failed = True
                    self._log(
                        logger, "Calibrated selector OOF fold failed to fit rank model.", "warning"
                    )
                    break
                utilities = {
                    idx: self._linear_score(rank_features[idx], rank_model) for idx in rank_features
                }

                decision_start = time.perf_counter()
                train_decisions = self._source_decisions(
                    df,
                    utilities,
                    rank_features,
                    distinctive,
                    train_ref_pairs,
                    exact_prefiltered_sources=exact_prefiltered_sources,
                )
                self._log(
                    logger,
                    (
                        "Calibrated selector OOF train decisions built: "
                        f"sources={len(train_decisions)}, duration={self._format_duration(time.perf_counter() - decision_start)}."
                    ),
                    "debug",
                )
                for src in validation_sources:
                    if src in train_decisions:
                        train_decisions[src]["sample_weight"] = 0.0
                accept_model = self._fit_accept_model(
                    train_decisions,
                    logger=logger,
                    label=f"OOF weight={float(background_weight):.3f} fold={fold_pos}/{len(folds)} accept",
                )
                if accept_model is None:
                    fold_failed = True
                    self._log(
                        logger,
                        "Calibrated selector OOF fold failed to fit accept model.",
                        "warning",
                    )
                    break

                decision_start = time.perf_counter()
                validation_decisions_all = self._source_decisions(
                    df,
                    utilities,
                    rank_features,
                    distinctive,
                    validation_ref_pairs,
                    exact_prefiltered_sources=exact_prefiltered_sources,
                )
                self._log(
                    logger,
                    (
                        "Calibrated selector OOF validation decisions built: "
                        f"sources={len(validation_decisions_all)}, "
                        f"duration={self._format_duration(time.perf_counter() - decision_start)}."
                    ),
                    "debug",
                )
                validation_decisions = {
                    src: decision
                    for src, decision in validation_decisions_all.items()
                    if src in validation_sources and float(decision.get("sample_weight", 0.0)) > 0.0
                }
                if not validation_decisions:
                    continue
                self._assign_p_match(validation_decisions, accept_model)
                for src, decision in validation_decisions.items():
                    decision["validation_fold"] = int(fold["fold"])
                    eval_decisions[src] = decision
                self._log(
                    logger,
                    (
                        "Calibrated selector OOF fold completed: "
                        f"weight={float(background_weight):.3f}, fold={fold_pos}/{len(folds)}, "
                        f"eval_sources={len(eval_decisions)}, "
                        f"duration={self._format_duration(time.perf_counter() - fold_start)}."
                    ),
                    "debug",
                )

            if fold_failed or not eval_decisions:
                continue

            threshold_start = time.perf_counter()
            accept_threshold, threshold_metrics = self._tune_accept_threshold(eval_decisions)
            self._log(
                logger,
                (
                    "Calibrated selector OOF threshold tuned: "
                    f"weight={float(background_weight):.3f}, threshold={accept_threshold:.3f}, "
                    f"F1={threshold_metrics.get('F1', 0.0):.3f}, "
                    f"duration={self._format_duration(time.perf_counter() - threshold_start)}."
                ),
                "debug",
            )
            for llm_mode in self._validation_llm_mode_candidates(primary_model):
                mode_metrics = dict(threshold_metrics)
                if llm_mode == "veto":
                    mode_metrics = self._metrics_with_llm_veto(
                        df=df,
                        decisions=eval_decisions,
                        threshold=accept_threshold,
                        primary_model=primary_model,
                        logger=logger,
                        record_lookup=record_lookup,
                    )
                for target_conflict_enabled in self._target_conflict_mode_candidates(
                    target_cardinality
                ):
                    target_metrics = dict(mode_metrics)
                    if target_conflict_enabled:
                        target_metrics = self._metrics_with_target_conflict(
                            decisions=eval_decisions,
                            threshold=accept_threshold,
                            target_cardinality=int(target_cardinality or 1),
                            protected_exact_pairs=protected_exact_pairs,
                            vetoed_sources=set(mode_metrics.get("llm_vetoed_source_ids", [])),
                        )
                    target_metrics["background_negative_weight"] = float(background_weight)
                    target_metrics["llm_mode"] = llm_mode
                    target_metrics["target_conflict_enabled"] = bool(target_conflict_enabled)
                    target_metrics["validation_scope"] = "oof"
                    target_metrics["validation_folds"] = int(fold_meta.get("n_folds", 0))
                    target_metrics["validation_sources"] = len(eval_decisions)
                    target_metrics["fold_positive_sources_min"] = min(fold_positive_sources or [0])
                    choice_key = (
                        float(target_metrics.get("F1", 0.0)),
                        float(target_metrics.get("P", 0.0)),
                        float(accept_threshold),
                        -float(background_weight),
                        1.0 if llm_mode == "veto" else 0.0,
                        1.0 if target_conflict_enabled else 0.0,
                    )
                    if best_choice is None or choice_key > best_choice["choice_key"]:
                        best_choice = {
                            "choice_key": choice_key,
                            "accept_threshold": float(accept_threshold),
                            "threshold_metrics": target_metrics,
                            "background_negative_weight": float(background_weight),
                            "llm_mode": llm_mode,
                            "target_conflict_enabled": bool(target_conflict_enabled),
                            "validation_split": dict(fold_meta),
                        }
            self._log(
                logger,
                (
                    "Calibrated selector OOF background weight completed: "
                    f"{weight_idx}/{len(weights_grid)} weight={float(background_weight):.3f}, "
                    f"eval_sources={len(eval_decisions)}, "
                    f"duration={self._format_duration(time.perf_counter() - weight_start)}."
                ),
                "debug",
            )

        if best_choice is None:
            self.calibration["background_negative_weight"] = original_background_weight
            self.llm["mode"] = original_llm_mode
            return None

        self.calibration["background_negative_weight"] = float(
            best_choice["background_negative_weight"]
        )
        self.llm["mode"] = str(best_choice["llm_mode"])
        rank_groups, _ = self._rank_training_groups(df, calibration_ref_pairs)
        rank_model = self._fit_rank_model(
            rank_features,
            rank_groups,
            logger=logger,
            label="final OOF refit rank",
        )
        if rank_model is None:
            self.calibration["background_negative_weight"] = original_background_weight
            self.llm["mode"] = original_llm_mode
            return None
        utilities = {
            idx: self._linear_score(rank_features[idx], rank_model) for idx in rank_features
        }
        source_decisions = self._source_decisions(
            df,
            utilities,
            rank_features,
            distinctive,
            calibration_ref_pairs,
            exact_prefiltered_sources=exact_prefiltered_sources,
        )
        accept_model = self._fit_accept_model(
            source_decisions,
            logger=logger,
            label="final OOF refit accept",
        )
        if accept_model is None:
            self.calibration["background_negative_weight"] = original_background_weight
            self.llm["mode"] = original_llm_mode
            return None
        self._assign_p_match(source_decisions, accept_model)

        oof_accept_threshold = float(best_choice["accept_threshold"])
        oof_threshold_metrics = dict(best_choice["threshold_metrics"])
        final_accept_threshold, final_threshold_metrics = self._tune_accept_threshold(
            source_decisions
        )
        final_mode_metrics = dict(final_threshold_metrics)
        if str(best_choice["llm_mode"]) == "veto":
            final_mode_metrics = self._metrics_with_llm_veto(
                df=df,
                decisions=source_decisions,
                threshold=final_accept_threshold,
                primary_model=primary_model,
                logger=logger,
                record_lookup=record_lookup,
            )
        if bool(best_choice["target_conflict_enabled"]):
            final_mode_metrics = self._metrics_with_target_conflict(
                decisions=source_decisions,
                threshold=final_accept_threshold,
                target_cardinality=int(target_cardinality or 1),
                protected_exact_pairs=protected_exact_pairs,
                vetoed_sources=set(final_mode_metrics.get("llm_vetoed_source_ids", [])),
            )
        final_mode_metrics["background_negative_weight"] = float(
            best_choice["background_negative_weight"]
        )
        final_mode_metrics["llm_mode"] = str(best_choice["llm_mode"])
        final_mode_metrics["target_conflict_enabled"] = bool(best_choice["target_conflict_enabled"])
        final_mode_metrics["validation_scope"] = "final_refit"
        final_mode_metrics["oof_selected_threshold"] = oof_accept_threshold
        final_mode_metrics["oof_validation_F1"] = float(oof_threshold_metrics.get("F1", 0.0))
        final_mode_metrics["oof_validation_P"] = float(oof_threshold_metrics.get("P", 0.0))
        final_mode_metrics["oof_validation_R"] = float(oof_threshold_metrics.get("R", 0.0))
        final_mode_metrics["oof_validation_sources"] = int(
            oof_threshold_metrics.get("validation_sources", 0)
        )
        final_mode_metrics["final_refit_selected_sources"] = int(
            sum(
                1
                for decision in source_decisions.values()
                if float(decision.get("p_match", 0.0)) >= float(final_accept_threshold)
            )
        )

        best_choice["rank_model"] = rank_model
        best_choice["accept_model"] = accept_model
        best_choice["oof_accept_threshold"] = oof_accept_threshold
        best_choice["oof_threshold_metrics"] = oof_threshold_metrics
        best_choice["accept_threshold"] = float(final_accept_threshold)
        best_choice["threshold_metrics"] = final_mode_metrics
        best_choice["final_refit_threshold_metrics"] = dict(final_mode_metrics)
        best_choice["utilities"] = utilities
        best_choice["source_decisions"] = source_decisions
        return best_choice

    def _select_accept_model_by_validation(
        self,
        df: pd.DataFrame,
        utilities: Dict[int, float],
        rank_features: Dict[int, List[float]],
        distinctive: Dict[int, float],
        train_ref_pairs: set[Tuple[str, str]],
        validation_ref_pairs: set[Tuple[str, str]],
        exact_prefiltered_sources: set[str],
        primary_model: Optional[IModel],
        logger: Optional[Any],
        record_lookup: Dict[Tuple[str, str], Dict[str, Any]],
        target_cardinality: Optional[int],
        protected_exact_pairs: set[Tuple[str, str]],
    ) -> Optional[Dict[str, Any]]:
        original_background_weight = float(self.calibration["background_negative_weight"])
        original_llm_mode = str(self.llm.get("mode", "veto"))
        train_sources = {src for src, _ in train_ref_pairs}
        validation_sources = {src for src, _ in validation_ref_pairs}
        heldout_validation_sources = validation_sources.difference(train_sources)
        best_choice: Optional[Dict[str, Any]] = None
        weights_grid = list(self.calibration.get("background_negative_weight_grid", []))
        self._log(
            logger,
            (
                "Calibrated selector held-out validation started: "
                f"background_weights={weights_grid}, train_pairs={len(train_ref_pairs)}, "
                f"validation_pairs={len(validation_ref_pairs)}."
            ),
            "debug",
        )

        for weight_idx, background_weight in enumerate(weights_grid, start=1):
            weight_start = time.perf_counter()
            self.calibration["background_negative_weight"] = float(background_weight)
            self._log(
                logger,
                (
                    "Calibrated selector held-out background weight started: "
                    f"{weight_idx}/{len(weights_grid)} weight={float(background_weight):.3f}."
                ),
                "debug",
            )
            decision_start = time.perf_counter()
            train_decisions = self._source_decisions(
                df,
                utilities,
                rank_features,
                distinctive,
                train_ref_pairs,
                exact_prefiltered_sources=exact_prefiltered_sources,
            )
            self._log(
                logger,
                (
                    "Calibrated selector held-out train decisions built: "
                    f"sources={len(train_decisions)}, duration={self._format_duration(time.perf_counter() - decision_start)}."
                ),
                "debug",
            )
            for src in heldout_validation_sources:
                if src in train_decisions:
                    train_decisions[src]["sample_weight"] = 0.0
            accept_model = self._fit_accept_model(
                train_decisions,
                logger=logger,
                label=f"held-out weight={float(background_weight):.3f} accept",
            )
            if accept_model is None:
                continue

            decision_start = time.perf_counter()
            validation_decisions_all = self._source_decisions(
                df,
                utilities,
                rank_features,
                distinctive,
                validation_ref_pairs,
                exact_prefiltered_sources=exact_prefiltered_sources,
            )
            self._log(
                logger,
                (
                    "Calibrated selector held-out validation decisions built: "
                    f"sources={len(validation_decisions_all)}, "
                    f"duration={self._format_duration(time.perf_counter() - decision_start)}."
                ),
                "debug",
            )
            validation_decisions = {
                src: decision
                for src, decision in validation_decisions_all.items()
                if src in validation_sources and float(decision.get("sample_weight", 0.0)) > 0.0
            }
            eval_decisions = validation_decisions or {
                src: decision
                for src, decision in train_decisions.items()
                if float(decision.get("sample_weight", 0.0)) > 0.0
            }
            if not eval_decisions:
                continue

            self._assign_p_match(eval_decisions, accept_model)
            threshold_start = time.perf_counter()
            accept_threshold, threshold_metrics = self._tune_accept_threshold(eval_decisions)
            self._log(
                logger,
                (
                    "Calibrated selector held-out threshold tuned: "
                    f"weight={float(background_weight):.3f}, threshold={accept_threshold:.3f}, "
                    f"F1={threshold_metrics.get('F1', 0.0):.3f}, "
                    f"duration={self._format_duration(time.perf_counter() - threshold_start)}."
                ),
                "debug",
            )

            for llm_mode in self._validation_llm_mode_candidates(primary_model):
                mode_metrics = dict(threshold_metrics)
                if llm_mode == "veto":
                    mode_metrics = self._metrics_with_llm_veto(
                        df=df,
                        decisions=eval_decisions,
                        threshold=accept_threshold,
                        primary_model=primary_model,
                        logger=logger,
                        record_lookup=record_lookup,
                    )
                for target_conflict_enabled in self._target_conflict_mode_candidates(
                    target_cardinality
                ):
                    target_metrics = dict(mode_metrics)
                    if target_conflict_enabled:
                        target_metrics = self._metrics_with_target_conflict(
                            decisions=eval_decisions,
                            threshold=accept_threshold,
                            target_cardinality=int(target_cardinality or 1),
                            protected_exact_pairs=protected_exact_pairs,
                            vetoed_sources=set(mode_metrics.get("llm_vetoed_source_ids", [])),
                        )
                    target_metrics["background_negative_weight"] = float(background_weight)
                    target_metrics["llm_mode"] = llm_mode
                    target_metrics["target_conflict_enabled"] = bool(target_conflict_enabled)
                    target_metrics["validation_scope"] = (
                        "validation" if validation_decisions else "train"
                    )
                    choice_key = (
                        float(target_metrics.get("F1", 0.0)),
                        float(target_metrics.get("P", 0.0)),
                        float(accept_threshold),
                        -float(background_weight),
                        1.0 if llm_mode == "veto" else 0.0,
                        1.0 if target_conflict_enabled else 0.0,
                    )
                    if best_choice is None or choice_key > best_choice["choice_key"]:
                        best_choice = {
                            "choice_key": choice_key,
                            "accept_model": accept_model,
                            "accept_threshold": float(accept_threshold),
                            "threshold_metrics": target_metrics,
                            "background_negative_weight": float(background_weight),
                            "llm_mode": llm_mode,
                            "target_conflict_enabled": bool(target_conflict_enabled),
                        }
            self._log(
                logger,
                (
                    "Calibrated selector held-out background weight completed: "
                    f"{weight_idx}/{len(weights_grid)} weight={float(background_weight):.3f}, "
                    f"eval_sources={len(eval_decisions)}, "
                    f"duration={self._format_duration(time.perf_counter() - weight_start)}."
                ),
                "debug",
            )

        if best_choice is None:
            self.calibration["background_negative_weight"] = original_background_weight
            self.llm["mode"] = original_llm_mode
            return None

        self.calibration["background_negative_weight"] = float(
            best_choice["background_negative_weight"]
        )
        self.llm["mode"] = str(best_choice["llm_mode"])
        source_decisions = self._source_decisions(
            df,
            utilities,
            rank_features,
            distinctive,
            train_ref_pairs,
            exact_prefiltered_sources=exact_prefiltered_sources,
        )
        self._assign_p_match(source_decisions, best_choice["accept_model"])
        best_choice["source_decisions"] = source_decisions
        return best_choice

    def _fit_rank_model(
        self,
        features: Dict[int, List[float]],
        groups: List[Dict[str, Any]],
        logger: Optional[Any] = None,
        label: str = "rank",
    ) -> Optional[Dict[str, Any]]:
        train_indices = sorted({idx for group in groups for idx in group["indices"]})
        if not train_indices:
            return None
        mean, scale = self._feature_mean_scale([features[idx] for idx in train_indices])
        local_pos = {idx: pos for pos, idx in enumerate(train_indices)}
        max_epochs = int(self.calibration["max_epochs"])
        log_interval = max(25, max(1, max_epochs // 4))
        start = time.perf_counter()
        log_fit_detail = max_epochs >= 500 or len(groups) >= 500 or len(train_indices) >= 1000
        if log_fit_detail:
            self._log(
                logger,
                (
                    f"Calibrated selector {label} model fit started: "
                    f"groups={len(groups)}, rows={len(train_indices)}, epochs={max_epochs}."
                ),
                "debug",
            )
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
            for epoch in range(1, max_epochs + 1):
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
                if epoch == max_epochs or epoch % log_interval == 0:
                    elapsed = max(1.0e-8, time.perf_counter() - start)
                    if elapsed >= 2.0 or (log_fit_detail and epoch == max_epochs):
                        self._log(
                            logger,
                            (
                                f"Calibrated selector {label} model fit progress: "
                                f"epoch={epoch}/{max_epochs}, loss={float(loss.detach().cpu().item()):.6f}, "
                                f"duration={self._format_duration(elapsed)}."
                            ),
                            "debug",
                        )
        elapsed_total = time.perf_counter() - start
        if log_fit_detail or elapsed_total >= 1.0:
            self._log(
                logger,
                (
                    f"Calibrated selector {label} model fit completed: "
                    f"duration={self._format_duration(elapsed_total)}."
                ),
                "debug",
            )
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
        exact_prefiltered_sources: Optional[set[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        ref_sources = {src for src, _ in ref_pairs}
        exact_sources = set(exact_prefiltered_sources or set())
        exact_policy = str(self.calibration.get("exact_prefiltered_source_policy", "hard_negative"))
        exact_negative_weight = float(
            self.calibration.get("exact_prefiltered_negative_weight", 1.0)
        )
        decisions: Dict[str, Dict[str, Any]] = {}
        for src, group in df.groupby("Src", sort=False):
            idxs = list(group.index)
            util_values = [utilities[idx] for idx in idxs]
            probs = self._softmax(util_values, temperature=self.temperature)
            prob_by_idx = {idx: float(prob) for idx, prob in zip(idxs, probs)}
            order = sorted(idxs, key=lambda idx: utilities[idx], reverse=True)
            winner_idx = order[0]
            second_idx = order[1] if len(order) > 1 else None
            utility_margin = utilities[winner_idx] - (
                utilities[second_idx] if second_idx is not None else 0.0
            )
            rank_prob_margin = prob_by_idx[winner_idx] - (
                prob_by_idx[second_idx] if second_idx is not None else 0.0
            )
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
                self._safe_float(top_row.get("s_diff"), 0.5),
            ]
            evidence_support = self._evidence_support_from_features(accept_features)
            src_text = str(src)
            winner_pair = (src_text, str(top_row.get("Tgt")))
            if src_text in exact_sources:
                label = 0.0
                if exact_policy == "exclude":
                    sample_weight = 0.0
                else:
                    sample_weight = exact_negative_weight
            elif src_text in ref_sources:
                label = 1.0 if winner_pair in ref_pairs else 0.0
                sample_weight = 1.0
            else:
                label = 0.0
                sample_weight = float(self.calibration["background_negative_weight"])
            decisions[src_text] = {
                "indices": idxs,
                "winner_idx": winner_idx,
                "winner_pair": winner_pair,
                "has_reference": bool(src_text in ref_sources),
                "rank_probs": prob_by_idx,
                "utility_margin": float(utility_margin),
                "rank_prob_margin": float(rank_prob_margin),
                "rank_entropy": float(entropy),
                "top_pair_score": float(top_pair_score),
                "accept_features": accept_features,
                "evidence_support": float(evidence_support),
                "label": label,
                "sample_weight": sample_weight,
                "rank_feature": rank_features[winner_idx],
            }
        return decisions

    def _fit_accept_model(
        self,
        decisions: Dict[str, Dict[str, Any]],
        logger: Optional[Any] = None,
        label: str = "accept",
    ) -> Optional[Dict[str, Any]]:
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
        max_epochs = int(self.calibration["max_epochs"])
        log_interval = max(25, max(1, max_epochs // 4))
        start = time.perf_counter()
        log_fit_detail = max_epochs >= 500 or len(samples) >= 500
        if log_fit_detail:
            self._log(
                logger,
                (
                    f"Calibrated selector {label} model fit started: "
                    f"samples={len(samples)}, positives={positives}, epochs={max_epochs}."
                ),
                "debug",
            )
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
        prior = self._clip(
            positive_weight / (positive_weight + negative_weight), self.eps, 1.0 - self.eps
        )
        # predict() runs under torch.no_grad(); calibration still needs autograd.
        with torch.enable_grad():
            bias = torch.tensor(self._logit(prior), dtype=torch.float32, requires_grad=True)
            opt = torch.optim.Adam([weights, bias], lr=float(self.calibration["learning_rate"]))
            l2 = float(self.calibration["l2"])
            denom = torch.clamp(sample_weight.sum(), min=1.0e-6)
            for epoch in range(1, max_epochs + 1):
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
                if epoch == max_epochs or epoch % log_interval == 0:
                    elapsed = max(1.0e-8, time.perf_counter() - start)
                    if elapsed >= 2.0 or (log_fit_detail and epoch == max_epochs):
                        self._log(
                            logger,
                            (
                                f"Calibrated selector {label} model fit progress: "
                                f"epoch={epoch}/{max_epochs}, loss={float(loss.detach().cpu().item()):.6f}, "
                                f"duration={self._format_duration(elapsed)}."
                            ),
                            "debug",
                        )
        elapsed_total = time.perf_counter() - start
        if log_fit_detail or elapsed_total >= 1.0:
            self._log(
                logger,
                (
                    f"Calibrated selector {label} model fit completed: "
                    f"duration={self._format_duration(elapsed_total)}."
                ),
                "debug",
            )
        return {
            "weights": weights.detach().cpu().tolist(),
            "bias": float(bias.detach().cpu().item()),
            "mean": mean,
            "scale": scale,
        }

    def _tune_accept_threshold(
        self, decisions: Dict[str, Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
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
                has_reference = bool(sample.get("has_reference", label))
                pred = float(sample.get("p_match", 0.0)) >= threshold
                if pred and label:
                    tp += weight
                elif pred and not label:
                    fp += weight
                    if has_reference:
                        fn += weight
                elif (not pred) and label:
                    fn += weight
                elif (not pred) and has_reference:
                    fn += weight
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            )
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

        best_f1 = max(
            threshold_metrics,
            key=lambda item: (item["F1"], item["P"], item["threshold"]),
        )
        best_f_beta = max(
            threshold_metrics,
            key=lambda item: (item["F_beta"], item["P"], item["threshold"]),
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
