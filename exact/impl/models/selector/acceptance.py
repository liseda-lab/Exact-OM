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

from .grouping import groupby_source, iter_source_groups, source_group_id


class AcceptanceMixin:
    def _validation_llm_mode_candidates(self, primary_model: Optional[IModel]) -> List[str]:
        mode = str(self.llm.get("mode", "off")).lower()
        if not bool(self.llm.get("enabled", True)) or mode == "off" or primary_model is None:
            return ["off"]
        return ["off", "veto"]

    @staticmethod
    def _target_conflict_mode_candidates(target_cardinality: Optional[int]) -> List[bool]:
        if target_cardinality is None or int(target_cardinality) <= 0:
            return [False]
        return [False, True]

    def _assign_p_match(
        self,
        decisions: Dict[str, Dict[str, Any]],
        accept_model: Mapping[str, Any],
    ) -> None:
        for decision in decisions.values():
            decision["p_match"] = self._sigmoid(
                self._linear_score(decision["accept_features"], accept_model)
            )

    def _metrics_with_llm_veto(
        self,
        df: pd.DataFrame,
        decisions: Dict[str, Dict[str, Any]],
        threshold: float,
        primary_model: Optional[IModel],
        logger: Optional[Any],
        record_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        if primary_model is None:
            return self._decision_metrics_at_threshold(decisions, threshold)

        original_mode = str(self.llm.get("mode", "veto"))
        self.llm["mode"] = "veto"
        vetoed_sources: set[str] = set()
        grouped = {source_group_id(key): group for key, group in groupby_source(df)}
        try:
            for group_id, decision in decisions.items():
                p_match = float(decision.get("p_match", 0.0))
                if p_match < float(threshold):
                    continue
                if not self._should_use_llm_calibrated(
                    acceptance_margin=abs(p_match - float(threshold)),
                    rank_margin=float(decision.get("rank_prob_margin", 0.0)),
                    primary_model=primary_model,
                    accepted=True,
                    evidence_support=float(decision.get("evidence_support", 0.0)),
                    evidence_support_floor=0.0,
                ):
                    continue
                group = grouped.get(group_id)
                if group is None:
                    continue
                src = str(decision.get("source", group.iloc[0]["Src"]))
                llm_choice = self._llm_direct_choice_group(
                    src=str(src),
                    group=group,
                    primary_model=primary_model,
                    logger=logger,
                    record_lookup=record_lookup,
                )
                if llm_choice and llm_choice.get("applied") and bool(llm_choice.get("no_match")):
                    vetoed_sources.add(group_id)
        finally:
            self.llm["mode"] = original_mode

        metrics = self._decision_metrics_at_threshold(
            decisions, threshold, vetoed_sources=vetoed_sources
        )
        metrics["llm_vetoed_sources"] = len(vetoed_sources)
        metrics["llm_vetoed_source_ids"] = sorted(vetoed_sources)
        return metrics

    def _metrics_with_target_conflict(
        self,
        decisions: Dict[str, Dict[str, Any]],
        threshold: float,
        target_cardinality: int,
        protected_exact_pairs: set[Tuple[str, str]],
        vetoed_sources: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        vetoed = set(vetoed_sources or set())
        target_groups: Dict[str, List[Tuple[bool, float, str, str]]] = {}
        for src, decision in decisions.items():
            if str(src) in vetoed:
                continue
            if float(decision.get("sample_weight", 0.0)) <= 0.0:
                continue
            if float(decision.get("p_match", 0.0)) < float(threshold):
                continue
            winner_pair = decision.get("winner_pair")
            if not winner_pair:
                continue
            pred_src, pred_tgt = str(winner_pair[0]), str(winner_pair[1])
            target_groups.setdefault(pred_tgt, []).append(
                (False, float(decision.get("p_match", 0.0)), pred_src, pred_tgt)
            )
        for src, tgt in protected_exact_pairs:
            target_groups.setdefault(str(tgt), []).append((True, 1.0, str(src), str(tgt)))

        kept_pairs: set[Tuple[str, str]] = set()
        for rows in target_groups.values():
            rows = sorted(rows, key=lambda row: (row[0], row[1]), reverse=True)
            for _, _, src, tgt in rows[: max(1, int(target_cardinality))]:
                kept_pairs.add((src, tgt))

        tp = fp = fn = 0.0
        for src, sample in decisions.items():
            weight = float(sample.get("sample_weight", 0.0))
            if weight <= 0.0:
                continue
            winner_pair = sample.get("winner_pair")
            pred = bool(winner_pair and (str(winner_pair[0]), str(winner_pair[1])) in kept_pairs)
            label = float(sample.get("label", 0.0)) > 0.5
            has_reference = bool(sample.get("has_reference", label))
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
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        beta = max(1.0e-6, float(self.calibration["f_beta"]))
        beta2 = beta * beta
        f_beta = (
            (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall)
            if ((beta2 * precision) + recall) > 0
            else 0.0
        )
        selected = {
            "threshold": float(threshold),
            "selected_threshold": float(threshold),
            "P": precision,
            "R": recall,
            "F1": f1,
            "F_beta": f_beta,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "target_conflict_removed_predictions": int(
                sum(
                    1
                    for decision in decisions.values()
                    if decision.get("winner_pair")
                    and float(decision.get("p_match", 0.0)) >= float(threshold)
                    and (str(decision["winner_pair"][0]), str(decision["winner_pair"][1]))
                    not in kept_pairs
                )
            ),
        }
        selected.update(
            {
                "accept_objective": str(self.calibration["accept_objective"]),
                "f_beta": beta,
                "min_precision": self.calibration.get("min_precision"),
                "min_recall": self.calibration.get("min_recall"),
                "fallback_to_f1": False,
                "selected_metrics": dict(selected),
                "best_f1": dict(selected),
                "best_f_beta": dict(selected),
                "recall_at_precision": dict(selected),
            }
        )
        return selected

    def _decision_metrics_at_threshold(
        self,
        decisions: Dict[str, Dict[str, Any]],
        threshold: float,
        vetoed_sources: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        vetoed = set(vetoed_sources or set())
        tp = fp = fn = 0.0
        for src, sample in decisions.items():
            weight = float(sample.get("sample_weight", 0.0))
            if weight <= 0.0:
                continue
            label = float(sample.get("label", 0.0)) > 0.5
            has_reference = bool(sample.get("has_reference", label))
            pred = float(sample.get("p_match", 0.0)) >= float(threshold) and str(src) not in vetoed
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
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        beta = max(1.0e-6, float(self.calibration["f_beta"]))
        beta2 = beta * beta
        f_beta = (
            (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall)
            if ((beta2 * precision) + recall) > 0
            else 0.0
        )
        selected = {
            "threshold": float(threshold),
            "selected_threshold": float(threshold),
            "P": precision,
            "R": recall,
            "F1": f1,
            "F_beta": f_beta,
            "TP": tp,
            "FP": fp,
            "FN": fn,
        }
        selected.update(
            {
                "accept_objective": str(self.calibration["accept_objective"]),
                "f_beta": beta,
                "min_precision": self.calibration.get("min_precision"),
                "min_recall": self.calibration.get("min_recall"),
                "fallback_to_f1": False,
                "selected_metrics": dict(selected),
                "best_f1": dict(selected),
                "best_f_beta": dict(selected),
                "recall_at_precision": dict(selected),
            }
        )
        return selected

    def _rank_feature_rows(
        self,
        df: pd.DataFrame,
        distinctive: Dict[int, float],
        reciprocity: Dict[int, float],
    ) -> Dict[int, List[float]]:
        rows: Dict[int, List[float]] = {}
        for _, _, _, group in iter_source_groups(df):
            idxs = list(group.index)
            pair_scores = [
                self._clip01(self._safe_float(group.at[idx, "S_pair_final"], 0.0)) for idx in idxs
            ]
            mean_score = sum(pair_scores) / max(1, len(pair_scores))
            std_score = self._std(pair_scores)
            order_by_pair = {
                idx: rank
                for rank, idx in enumerate(
                    sorted(
                        idxs,
                        key=lambda row_idx: self._safe_float(
                            group.at[row_idx, "S_pair_final"], 0.0
                        ),
                        reverse=True,
                    ),
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

    def _threshold_compatible_score(
        self,
        p_match: float,
        accept_threshold: float,
        score_threshold: float,
    ) -> float:
        if score_threshold is None:
            return self._clip01(p_match)
        return self._clip01(float(score_threshold) + (float(p_match) - float(accept_threshold)))

    def _final_selector_score(
        self,
        p_match: float,
        accept_threshold: float,
        score_threshold: float,
    ) -> float:
        if self.score_mode == "p_match":
            return self._clip01(float(p_match))
        return self._threshold_compatible_score(
            p_match=p_match,
            accept_threshold=accept_threshold,
            score_threshold=score_threshold,
        )

    def _evidence_support_floor(
        self,
        score_threshold: Optional[float],
        accept_threshold: float,
    ) -> float:
        floors = [0.5, self._clip01(float(accept_threshold))]
        if score_threshold is not None:
            floors.append(self._clip01(float(score_threshold)))
        min_precision = self.calibration.get("min_precision")
        if min_precision is not None:
            floors.append(self._clip01(float(min_precision)))
        return float(max(floors))

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
        if str(self.llm.get("mode", "veto")).lower() == "off":
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
        return top2_margin < ambiguity_margin or no_match_margin < ambiguity_margin

    def _should_use_llm_calibrated(
        self,
        acceptance_margin: float,
        rank_margin: float,
        primary_model: Optional[IModel],
        accepted: bool = True,
        evidence_support: float = 0.0,
        evidence_support_floor: float = 1.0,
    ) -> bool:
        if not bool(self.llm.get("enabled", True)):
            return False
        if str(self.llm.get("mode", "veto")).lower() == "off":
            return False
        if primary_model is None:
            return False
        if not bool(accepted):
            return False
        near_boundary_or_tie = float(acceptance_margin) <= float(
            self.llm.get("trigger_acceptance_margin", 0.025)
        ) or float(rank_margin) <= float(self.llm.get("trigger_rank_margin", 0.03))
        return near_boundary_or_tie

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
                self._log(
                    logger, f"Candidate-set selector LLM arbitration failed: {exc}", "warning"
                )
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
        if winner.upper() == "NO_MATCH" or relation in {
            "broader",
            "narrower",
            "sibling",
            "related",
            "none",
        }:
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
                self._log(
                    logger, f"Candidate-set selector LLM arbitration failed: {exc}", "warning"
                )
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
        if winner.upper() == "NO_MATCH" or relation in {
            "broader",
            "narrower",
            "sibling",
            "related",
            "none",
        }:
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
        return (
            {
                "system": "You are an ontology alignment expert.",
                "user": "\n".join(lines),
            },
            id_to_index,
            id_to_tgt,
        )

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

        if not all(
            hasattr(primary_model, name) for name in ("_ensure_local_llm", "llm_tok", "llm")
        ):
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
    def _count_source_groups(df: pd.DataFrame, column: str) -> int:
        if df.empty or column not in df.columns:
            return 0
        if "Src" not in df.columns:
            return int(df[column].astype(bool).sum())
        return int(groupby_source(df)[column].any().sum())

    @staticmethod
    def _sha1(value: str) -> str:
        import hashlib

        return hashlib.sha1(value.encode("utf-8")).hexdigest()
