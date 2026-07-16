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

from .grouping import count_source_groups, groupby_target, iter_source_groups


class FeatureEngineeringMixin:
    def _evidence_support_from_features(self, accept_features: Sequence[float]) -> float:
        features = {
            name: self._clip01(self._safe_float(value, 0.0))
            for name, value in zip(self.ACCEPT_FEATURE_NAMES, accept_features)
        }
        rank_confidence = max(
            features.get("top_rank_prob", 0.0),
            1.0 - features.get("rank_entropy", 1.0),
        )
        terms = [
            features.get("top_pair_score", 0.0),
            features.get("top_label", 0.0),
            features.get("top_struct", 0.0),
            rank_confidence,
        ]
        support = sum(terms) / max(1, len(terms))

        # The difference channel is neutral at 0.5. Values below neutral are
        # treated as explicit conflict and dampen otherwise strong support.
        diff = features.get("top_diff", 0.5)
        if diff < 0.5:
            support *= diff / 0.5
        return float(self._clip01(support))

    def _feature_mean_scale(
        self, rows: Sequence[Sequence[float]]
    ) -> Tuple[List[float], List[float]]:
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
            self._clip01(self._safe_float(group.at[idx, "S_pair_final"], 0.0)) for idx in idxs
        ]
        mean_score = sum(pair_scores) / max(1, n)
        std_score = self._std(pair_scores)
        if std_score < self.eps:
            zscore_terms = [0.5 for _ in pair_scores]
        else:
            zscore_terms = [
                self._sigmoid((value - mean_score) / std_score) for value in pair_scores
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
        final_candidate_scores = (
            [0.0 for _ in selector_scores] if no_match_wins else list(selector_scores)
        )

        best_real = max(final_candidate_scores or [0.0])
        winner_pos = None
        if final_candidate_scores and not no_match_wins:
            winner_pos = max(
                range(len(final_candidate_scores)), key=lambda pos: final_candidate_scores[pos]
            )
        sorted_final = sorted(
            selector_scores + ([no_match_prob] if self.use_no_match else []), reverse=True
        )
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
        for _, group in groupby_target(df):
            idxs = list(group.index)
            probs = self._softmax([logits[idx] for idx in idxs], temperature=self.temperature)
            for idx, prob in zip(idxs, probs):
                result[idx] = 2.0 * float(prob) - 1.0
        return result

    def _distinctive_scores(
        self,
        df: pd.DataFrame,
        record_lookup: Dict[Tuple[str, str], Dict[str, Any]],
        logger: Optional[Any] = None,
        log_every: int = 10,
    ) -> Dict[int, float]:
        result: Dict[int, float] = {}
        n_sources = count_source_groups(df)
        progress_interval = self._progress_interval(log_every)
        start = time.perf_counter()
        if n_sources:
            self._log(
                logger,
                f"Candidate-set selector distinctive-evidence scan started: sources={n_sources}.",
                "debug",
            )
        for group_idx, (_, _, _, group) in enumerate(iter_source_groups(df), start=1):
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
            if n_sources and (group_idx == n_sources or group_idx % progress_interval == 0):
                elapsed = max(1.0e-8, time.perf_counter() - start)
                rate = group_idx / elapsed
                remaining = max(0, n_sources - group_idx)
                eta = (
                    self._format_duration(remaining / rate)
                    if rate > 0
                    else self._format_duration(0.0)
                )
                self._log(
                    logger,
                    (
                        "Candidate-set selector distinctive-evidence progress: "
                        f"sources={group_idx}/{n_sources}, avg={rate:.2f} sources/s, ETA {eta}"
                    ),
                    "debug",
                )
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
        row_lookup = {(str(row["Src"]), str(row["Tgt"])): row for _, row in df.iterrows()}
        for record in results_json:
            key = (str(record.get("src_iri")), str(record.get("tgt_iri")))
            row = row_lookup.get(key)
            if row is None:
                continue
            conf = record.get("confidences") or {}
            for key in [
                "cand_sim",
                "cand_sim_semantic",
                "cand_sim_lexical",
            ]:
                if key in row:
                    conf[key] = float(row.get(key, 0.0))
            if "S_pair_final" not in conf:
                conf["S_pair_final"] = float(row.get("S_pair_final", conf.get("S_final", 0.0)))
            conf["S_select"] = float(row.get("S_select", 0.0))
            conf["P_select"] = float(row.get("P_select", 0.0))
            conf["selection_margin"] = float(row.get("selection_margin", 0.0))
            conf["selection_entropy"] = float(row.get("selection_entropy", 0.0))
            conf["selection_no_match_prob"] = float(row.get("selection_no_match_prob", 0.0))
            conf["selection_evidence_support"] = float(row.get("selection_evidence_support", 0.0))
            conf["selection_distinctive"] = float(row.get("selection_distinctive", 0.0))
            conf["selection_utility"] = float(row.get("selection_utility", 0.0))
            conf["P_rank"] = float(row.get("P_rank", 0.0))
            conf["P_match"] = float(row.get("P_match", 0.0))
            conf["selection_accept_threshold"] = float(row.get("selection_accept_threshold", 0.0))
            conf["selection_target_conflict_enabled"] = bool(
                row.get("selection_target_conflict_enabled", False)
            )
            conf["selection_target_cardinality"] = int(
                row.get("selection_target_cardinality", 0) or 0
            )
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
