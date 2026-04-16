from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .semantic_scorer import SemanticScorer


class PairAdaptiveSemanticScorer(SemanticScorer):
    """
    Default pair-adaptive scorer.

    Hierarchy is ontology-native and family-aware, non-hierarchical structure is
    projection-backed, attributes are auxiliary/support-only, and the LLM
    consumes one pair brief instead of separate source/target summaries.
    """

    def __init__(
        self,
        max_input_tokens_hier: int = 128,
        max_input_tokens_sim: int = 256,
        max_input_tokens_diff: int = 256,
        max_input_tokens_attr_item: int = 96,
        max_hierarchy_triples_per_family: int = 6,
        max_object_triples: int = 48,
        max_diff_triples: int = 24,
        max_attr_items: int = 12,
        hierarchical_relation_families: Optional[Dict[str, Dict[str, Any]]] = None,
        attribute_property_weights: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        requested_context_cap = int(kwargs.get("max_input_tokens_context", 256))
        kwargs["max_input_tokens_context"] = max(
            requested_context_cap,
            int(max_input_tokens_hier),
            int(max_input_tokens_sim),
            int(max_input_tokens_diff),
            int(max_input_tokens_attr_item),
        )
        super().__init__(**kwargs)
        self.max_input_tokens_hier = int(max_input_tokens_hier)
        self.max_input_tokens_sim = int(max_input_tokens_sim)
        self.max_input_tokens_diff = int(max_input_tokens_diff)
        self.max_input_tokens_attr_item = int(max_input_tokens_attr_item)
        self.max_hierarchy_triples_per_family = int(max_hierarchy_triples_per_family)
        self.max_object_triples = int(max_object_triples)
        self.max_diff_triples = int(max_diff_triples)
        self.max_attr_items = int(max_attr_items)
        self.hierarchical_relation_families = dict(hierarchical_relation_families or {})
        self.attribute_property_weights = dict(attribute_property_weights or {})
        self._attached_dataset = None

    def attach_dataset(self, dataset: Any) -> None:
        self._attached_dataset = dataset
        if not self.hierarchical_relation_families and hasattr(dataset, "hierarchical_relation_families"):
            self.hierarchical_relation_families = dict(dataset.hierarchical_relation_families or {})
        if hasattr(dataset, "max_hierarchy_triples_per_family"):
            self.max_hierarchy_triples_per_family = int(dataset.max_hierarchy_triples_per_family)
        if hasattr(dataset, "max_object_triples"):
            self.max_object_triples = int(dataset.max_object_triples)
        if hasattr(dataset, "max_diff_triples"):
            self.max_diff_triples = int(dataset.max_diff_triples)
        if hasattr(dataset, "max_attr_items"):
            self.max_attr_items = int(dataset.max_attr_items)

    @staticmethod
    def _sim01(value: torch.Tensor) -> torch.Tensor:
        return ((value + 1.0) / 2.0).clamp(0.0, 1.0)

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _safe_mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return float(sum(float(v) for v in values) / max(1, len(values)))

    @staticmethod
    def _safe_std(values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.0
        tensor = torch.tensor(list(values), dtype=torch.float32)
        return float(torch.std(tensor, unbiased=False).item())

    def _batch_pool_stats(
        self,
        feature_map: Dict[str, Dict[str, Any]],
        family_names: Sequence[str],
    ) -> Dict[str, int]:
        hier_nonempty = 0
        obj_nonempty = 0
        attr_nonempty = 0
        for feats in feature_map.values():
            hierarchy = feats.get("hierarchy", {}) or {}
            if any(hierarchy.get(family, []) for family in family_names):
                hier_nonempty += 1
            if feats.get("object_triples"):
                obj_nonempty += 1
            if feats.get("attributes"):
                attr_nonempty += 1
        return {
            "entities": int(len(feature_map)),
            "hier_nonempty": int(hier_nonempty),
            "obj_nonempty": int(obj_nonempty),
            "attr_nonempty": int(attr_nonempty),
        }

    def _batch_selected_evidence_stats(
        self,
        pair_payloads: Sequence[Dict[str, Any]],
        family_names: Sequence[str],
    ) -> Dict[str, int]:
        hier_selected = 0
        sim_selected = 0
        diff_selected = 0
        attr_selected = 0
        for payload in pair_payloads:
            hierarchy = payload.get("hierarchy", {}) or {}
            if any(
                hierarchy.get(family, {}).get("src_selected") or hierarchy.get(family, {}).get("tgt_selected")
                for family in family_names
            ):
                hier_selected += 1
            sim_payload = payload.get("sim", {}) or {}
            if sim_payload.get("src_selected") or sim_payload.get("tgt_selected"):
                sim_selected += 1
            diff_payload = payload.get("diff", {}) or {}
            if diff_payload.get("src_selected") or diff_payload.get("tgt_selected"):
                diff_selected += 1
            attr_payload = payload.get("attr", {}) or {}
            if attr_payload.get("src_selected") or attr_payload.get("tgt_selected"):
                attr_selected += 1
        return {
            "pairs": int(len(pair_payloads)),
            "hier_selected": int(hier_selected),
            "sim_selected": int(sim_selected),
            "diff_selected": int(diff_selected),
            "attr_selected": int(attr_selected),
        }

    @staticmethod
    def _top_two_scores(mat: torch.Tensor) -> Tuple[float, float]:
        if mat.numel() == 0:
            return 0.0, 0.0
        flat = mat.flatten()
        if flat.numel() == 1:
            val = float(flat[0].item())
            return val, val
        vals, _ = torch.topk(flat, k=2)
        return float(vals[0].item()), float(vals[1].item())

    @staticmethod
    def _normalize_text(text: Any) -> str:
        return str(text or "").strip()

    def _brief_key(self, src_label: str, tgt_label: str, packet: str) -> str:
        payload = "\u241F".join([src_label or "", tgt_label or "", packet or ""])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _family_display_name(self, family: str) -> str:
        return family.replace("_", " ").strip()

    def _hierarchy_template(self, family: str) -> str:
        templates = {
            "is_a": "$SRC is a kind of $TGT.",
            "part_of": "$SRC is part of $TGT.",
            "has_part": "$SRC has part $TGT.",
        }
        return templates.get(family, f"$SRC {self._family_display_name(family)} $TGT.")

    def _trim_items_to_budget(self, items: Sequence[str], budget_tokens: int) -> List[str]:
        if budget_tokens <= 0:
            return [self._normalize_text(item) for item in items if self._normalize_text(item)]
        kept: List[str] = []
        used = 0
        for item in items:
            text = self._normalize_text(item)
            if not text:
                continue
            approx = max(1, len(text.split()))
            if kept and used + approx > budget_tokens:
                break
            kept.append(text)
            used += approx
        return kept

    def _join_channel_sentences(self, sentences: Sequence[str], budget_tokens: int) -> str:
        trimmed = self._trim_items_to_budget(sentences, budget_tokens)
        return self._join_context(list(trimmed))

    def _encode_label_matrix(self, left: Sequence[str], right: Sequence[str]) -> torch.Tensor:
        if not left or not right:
            return torch.zeros((len(left), len(right)), device=self.device)
        e_left = torch.nn.functional.normalize(self.encode_labels_batch(list(left)), dim=-1)
        e_right = torch.nn.functional.normalize(self.encode_labels_batch(list(right)), dim=-1)
        return self._sim01(e_left @ e_right.T)

    def _encode_context_matrix(self, left: Sequence[str], right: Sequence[str]) -> torch.Tensor:
        if not left or not right or not self.use_context:
            return torch.zeros((len(left), len(right)), device=self.device)
        e_left = torch.nn.functional.normalize(self.encode_contexts_batch(list(left)), dim=-1)
        e_right = torch.nn.functional.normalize(self.encode_contexts_batch(list(right)), dim=-1)
        return self._sim01(e_left @ e_right.T)

    def _context_similarity_from_sentences(
        self,
        src_sentences: Sequence[str],
        tgt_sentences: Sequence[str],
        budget_tokens: int,
    ) -> float:
        if not self.use_context or not src_sentences or not tgt_sentences:
            return self.tau
        src_text = self._join_channel_sentences(src_sentences, budget_tokens)
        tgt_text = self._join_channel_sentences(tgt_sentences, budget_tokens)
        if not src_text or not tgt_text:
            return self.tau
        enc = torch.nn.functional.normalize(self.encode_contexts_batch([src_text, tgt_text]), dim=-1)
        return float(self._sim01(torch.sum(enc[0] * enc[1], dim=-1)).item())

    def _select_diverse_indices(
        self,
        items: Sequence[Any],
        scores: Sequence[float],
        limit: int,
        per_relation_cap: Optional[int],
        relation_getter,
        tie_breaker=None,
    ) -> List[int]:
        if not items or limit <= 0:
            return []
        order = list(range(len(items)))
        order.sort(
            key=lambda idx: (
                float(scores[idx]),
                float(tie_breaker(items[idx]) if tie_breaker is not None else 0.0),
            ),
            reverse=True,
        )
        selected: List[int] = []
        rel_counts: Dict[str, int] = {}
        for idx in order:
            rel_key = str(relation_getter(items[idx]) or "")
            if per_relation_cap is not None and rel_counts.get(rel_key, 0) >= per_relation_cap:
                continue
            selected.append(idx)
            rel_counts[rel_key] = rel_counts.get(rel_key, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    def _family_relation_aliases(self, family: str) -> List[str]:
        cfg = self.hierarchical_relation_families.get(family, {}) or {}
        aliases = list(cfg.get("label_seeds") or [])
        if not aliases:
            aliases = [self._family_display_name(family)]
        return aliases

    def _verbalize_hierarchy_items(
        self,
        family: str,
        items: Sequence[Tuple[str, str, str, float]],
    ) -> List[str]:
        template = self._hierarchy_template(family)
        out: List[str] = []
        for head, _, tail, _ in items:
            out.append(template.replace("$SRC", str(head)).replace("$TGT", str(tail)))
        return out

    def _verbalize_object_items(self, items: Sequence[Dict[str, Any]]) -> List[str]:
        triples = [tuple(item.get("triple", ("", "", ""))) for item in items]
        dataset = self._attached_dataset
        if dataset is not None and hasattr(dataset, "_verbalize_triples"):
            try:
                return list(dataset._verbalize_triples(triples))
            except Exception:
                pass
        out: List[str] = []
        for head, rel, tail in triples:
            out.append(f"{head} {rel} {tail}.")
        return out

    def _score_label_channel(
        self,
        src_label_lists: List[List[str]],
        tgt_label_lists: List[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, str]]]:
        n_pairs = len(src_label_lists)
        if not self.use_lexical:
            best_pairs = [
                (
                    (src_labels[0] if src_labels else ""),
                    (tgt_labels[0] if tgt_labels else ""),
                )
                for src_labels, tgt_labels in zip(src_label_lists, tgt_label_lists)
            ]
            neutral = torch.full((n_pairs,), self.tau, device=self.device)
            q_label = torch.zeros(n_pairs, device=self.device)
            return neutral, q_label, best_pairs

        flat_src = [label for labels in src_label_lists for label in labels]
        flat_tgt = [label for labels in tgt_label_lists for label in labels]
        if not flat_src or not flat_tgt:
            neutral = torch.full((n_pairs,), self.tau, device=self.device)
            q_label = torch.zeros(n_pairs, device=self.device)
            best_pairs = [
                (
                    (src_labels[0] if src_labels else ""),
                    (tgt_labels[0] if tgt_labels else ""),
                )
                for src_labels, tgt_labels in zip(src_label_lists, tgt_label_lists)
            ]
            return neutral, q_label, best_pairs

        e_src = torch.nn.functional.normalize(self.encode_labels_batch(flat_src), dim=-1)
        e_tgt = torch.nn.functional.normalize(self.encode_labels_batch(flat_tgt), dim=-1)

        src_slices = []
        start = 0
        for labels in src_label_lists:
            end = start + len(labels)
            src_slices.append(e_src[start:end] if labels else e_src[0:0])
            start = end
        tgt_slices = []
        start = 0
        for labels in tgt_label_lists:
            end = start + len(labels)
            tgt_slices.append(e_tgt[start:end] if labels else e_tgt[0:0])
            start = end

        s_vals: List[torch.Tensor] = []
        q_vals: List[float] = []
        best_pairs: List[Tuple[str, str]] = []
        for src_labels, tgt_labels, src_embs, tgt_embs in zip(
            src_label_lists, tgt_label_lists, src_slices, tgt_slices
        ):
            if src_embs.shape[0] == 0 or tgt_embs.shape[0] == 0:
                s_vals.append(torch.tensor(float(self.tau), device=self.device))
                q_vals.append(0.0)
                best_pairs.append(
                    (
                        (src_labels[0] if src_labels else ""),
                        (tgt_labels[0] if tgt_labels else ""),
                    )
                )
                continue
            mat = self._sim01(src_embs @ tgt_embs.T)
            score, pair = self._select_label_pair(mat, src_labels, tgt_labels)
            z1, z2 = self._top_two_scores(mat)
            q_label = 1.0 if mat.numel() <= 1 else self._clip01((z1 - z2) / max(1e-8, (1.0 - z2)))
            s_vals.append(score)
            q_vals.append(q_label)
            best_pairs.append(pair)
        return torch.stack(s_vals), torch.tensor(q_vals, dtype=torch.float32, device=self.device), best_pairs

    def _score_hierarchy_family(
        self,
        family: str,
        src_items: Sequence[Tuple[str, str, str, float]],
        tgt_items: Sequence[Tuple[str, str, str, float]],
    ) -> Dict[str, Any]:
        payload = {
            "score": self.tau,
            "quality": 0.0,
            "strength": 0.0,
            "coverage": 0.0,
            "specificity": 0.0,
            "embedding": self.tau,
            "src_selected": [],
            "tgt_selected": [],
            "src_sentences": [],
            "tgt_sentences": [],
        }
        if not self.use_context or not src_items or not tgt_items:
            return payload

        src_tails = [item[2] for item in src_items]
        tgt_tails = [item[2] for item in tgt_items]
        support_mat = self._encode_label_matrix(src_tails, tgt_tails)
        row_best = support_mat.max(dim=1).values if support_mat.numel() else torch.zeros(len(src_items), device=self.device)
        col_best = support_mat.max(dim=0).values if support_mat.numel() else torch.zeros(len(tgt_items), device=self.device)

        src_idx = self._select_diverse_indices(
            src_items,
            row_best.detach().cpu().tolist(),
            self.max_hierarchy_triples_per_family,
            per_relation_cap=None,
            relation_getter=lambda _: family,
            tie_breaker=lambda item: float(item[3]),
        )
        tgt_idx = self._select_diverse_indices(
            tgt_items,
            col_best.detach().cpu().tolist(),
            self.max_hierarchy_triples_per_family,
            per_relation_cap=None,
            relation_getter=lambda _: family,
            tie_breaker=lambda item: float(item[3]),
        )

        if not src_idx or not tgt_idx:
            return payload

        src_selected = [src_items[i] for i in src_idx]
        tgt_selected = [tgt_items[i] for i in tgt_idx]
        src_sentences = self._verbalize_hierarchy_items(family, src_selected)
        tgt_sentences = self._verbalize_hierarchy_items(family, tgt_selected)
        reduced = support_mat[src_idx][:, tgt_idx]
        src_support = reduced.max(dim=1).values.detach().cpu().tolist()
        tgt_support = reduced.max(dim=0).values.detach().cpu().tolist()
        str_f = 0.5 * (self._safe_mean(src_support) + self._safe_mean(tgt_support))
        emb_f = self._context_similarity_from_sentences(src_sentences, tgt_sentences, self.max_input_tokens_hier)
        spec_vals = [float(item[3]) for item in src_selected] + [float(item[3]) for item in tgt_selected]
        inf_f = self._safe_mean(spec_vals)
        cov_f = self._clip01((len(src_selected) + len(tgt_selected)) / max(1.0, 2.0 * self.max_hierarchy_triples_per_family))
        q_f = self._clip01((cov_f + str_f + inf_f) / 3.0)
        s_f = self._clip01(0.5 * emb_f + 0.5 * str_f)

        src_imp = [float(value) for value in src_support]
        tgt_imp = [float(value) for value in tgt_support]
        total_imp = sum(src_imp) + sum(tgt_imp) or 1.0
        payload.update(
            {
                "score": s_f,
                "quality": q_f,
                "strength": str_f,
                "coverage": cov_f,
                "specificity": inf_f,
                "embedding": emb_f,
                "src_selected": [
                    {
                        "triple": list(item[:3]),
                        "specificity": float(item[3]),
                        "support": float(src_support[pos]),
                        "importance": float(src_imp[pos] / total_imp),
                    }
                    for pos, item in enumerate(src_selected)
                ],
                "tgt_selected": [
                    {
                        "triple": list(item[:3]),
                        "specificity": float(item[3]),
                        "support": float(tgt_support[pos]),
                        "importance": float(tgt_imp[pos] / total_imp),
                    }
                    for pos, item in enumerate(tgt_selected)
                ],
                "src_sentences": src_sentences,
                "tgt_sentences": tgt_sentences,
            }
        )
        return payload

    def _object_support_matrix(
        self,
        src_items: Sequence[Dict[str, Any]],
        tgt_items: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        if not src_items or not tgt_items:
            return torch.zeros((len(src_items), len(tgt_items)), device=self.device)
        src_rels = [str(item["triple"][1]) for item in src_items]
        tgt_rels = [str(item["triple"][1]) for item in tgt_items]
        src_neighbors = [str(item["triple"][2]) for item in src_items]
        tgt_neighbors = [str(item["triple"][2]) for item in tgt_items]
        rel_mat = self._encode_label_matrix(src_rels, tgt_rels)
        nbr_mat = self._encode_label_matrix(src_neighbors, tgt_neighbors)
        return (0.5 * rel_mat + 0.5 * nbr_mat).clamp(0.0, 1.0)

    def _score_similarity_channel(
        self,
        src_items: Sequence[Dict[str, Any]],
        tgt_items: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = {
            "score": self.tau,
            "quality": 0.0,
            "strength": 0.0,
            "coverage": 0.0,
            "stability": 0.0,
            "embedding": self.tau,
            "src_selected": [],
            "tgt_selected": [],
            "src_sentences": [],
            "tgt_sentences": [],
            "support_matrix": None,
        }
        if not self.use_context or not src_items or not tgt_items:
            return payload

        support_mat = self._object_support_matrix(src_items, tgt_items)
        row_best = support_mat.max(dim=1).values if support_mat.numel() else torch.zeros(len(src_items), device=self.device)
        col_best = support_mat.max(dim=0).values if support_mat.numel() else torch.zeros(len(tgt_items), device=self.device)
        src_idx = self._select_diverse_indices(
            src_items,
            row_best.detach().cpu().tolist(),
            min(self.max_object_triples, len(src_items)),
            per_relation_cap=2,
            relation_getter=lambda item: item["triple"][1],
            tie_breaker=lambda item: item.get("score", 0.0),
        )
        tgt_idx = self._select_diverse_indices(
            tgt_items,
            col_best.detach().cpu().tolist(),
            min(self.max_object_triples, len(tgt_items)),
            per_relation_cap=2,
            relation_getter=lambda item: item["triple"][1],
            tie_breaker=lambda item: item.get("score", 0.0),
        )
        if not src_idx or not tgt_idx:
            return payload

        src_selected = [src_items[i] for i in src_idx]
        tgt_selected = [tgt_items[i] for i in tgt_idx]
        reduced = support_mat[src_idx][:, tgt_idx]
        src_support = reduced.max(dim=1).values.detach().cpu().tolist()
        tgt_support = reduced.max(dim=0).values.detach().cpu().tolist()
        str_sim = 0.5 * (self._safe_mean(src_support) + self._safe_mean(tgt_support))
        src_sentences = self._verbalize_object_items(src_selected)
        tgt_sentences = self._verbalize_object_items(tgt_selected)
        emb_sim = self._context_similarity_from_sentences(src_sentences, tgt_sentences, self.max_input_tokens_sim)
        cov_sim = self._clip01((len(src_selected) + len(tgt_selected)) / max(1.0, 2.0 * self.max_object_triples))
        stability_vals = list(src_support) + list(tgt_support)
        stab_sim = self._clip01(1.0 - min(1.0, 2.0 * self._safe_std(stability_vals)))
        q_sim = self._clip01((cov_sim + str_sim + stab_sim) / 3.0)
        s_sim = self._clip01(0.5 * emb_sim + 0.5 * str_sim)
        src_imp = [float(value) for value in src_support]
        tgt_imp = [float(value) for value in tgt_support]
        total_imp = sum(src_imp) + sum(tgt_imp) or 1.0
        payload.update(
            {
                "score": s_sim,
                "quality": q_sim,
                "strength": str_sim,
                "coverage": cov_sim,
                "stability": stab_sim,
                "embedding": emb_sim,
                "src_selected": [
                    {
                        "triple": list(item["triple"]),
                        "support": float(src_support[pos]),
                        "edge_ic": float(item.get("score", 0.0)),
                        "importance": float(src_imp[pos] / total_imp),
                    }
                    for pos, item in enumerate(src_selected)
                ],
                "tgt_selected": [
                    {
                        "triple": list(item["triple"]),
                        "support": float(tgt_support[pos]),
                        "edge_ic": float(item.get("score", 0.0)),
                        "importance": float(tgt_imp[pos] / total_imp),
                    }
                    for pos, item in enumerate(tgt_selected)
                ],
                "src_sentences": src_sentences,
                "tgt_sentences": tgt_sentences,
                "support_matrix": support_mat,
                "row_best": row_best.detach().cpu().tolist(),
                "col_best": col_best.detach().cpu().tolist(),
            }
        )
        return payload

    def _score_difference_channel(
        self,
        src_items: Sequence[Dict[str, Any]],
        tgt_items: Sequence[Dict[str, Any]],
        support_mat: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        payload = {
            "score": self.tau,
            "quality": 0.0,
            "conflict": 0.0,
            "coverage": 0.0,
            "strength": 0.0,
            "stability": 0.0,
            "src_selected": [],
            "tgt_selected": [],
            "src_sentences": [],
            "tgt_sentences": [],
        }
        if not self.use_context:
            return payload

        if support_mat is None:
            support_mat = self._object_support_matrix(src_items, tgt_items)
        row_best = support_mat.max(dim=1).values if src_items and tgt_items else torch.zeros(len(src_items), device=self.device)
        col_best = support_mat.max(dim=0).values if src_items and tgt_items else torch.zeros(len(tgt_items), device=self.device)

        src_unsupported = [
            float(item.get("score", 0.0)) * (1.0 - float(row_best[idx].item() if idx < row_best.numel() else 0.0))
            for idx, item in enumerate(src_items)
        ]
        tgt_unsupported = [
            float(item.get("score", 0.0)) * (1.0 - float(col_best[idx].item() if idx < col_best.numel() else 0.0))
            for idx, item in enumerate(tgt_items)
        ]

        src_idx = self._select_diverse_indices(
            src_items,
            src_unsupported,
            min(self.max_diff_triples, len(src_items)),
            per_relation_cap=3,
            relation_getter=lambda item: item["triple"][1],
            tie_breaker=lambda item: item.get("score", 0.0),
        )
        tgt_idx = self._select_diverse_indices(
            tgt_items,
            tgt_unsupported,
            min(self.max_diff_triples, len(tgt_items)),
            per_relation_cap=3,
            relation_getter=lambda item: item["triple"][1],
            tie_breaker=lambda item: item.get("score", 0.0),
        )

        src_selected = [src_items[i] for i in src_idx]
        tgt_selected = [tgt_items[i] for i in tgt_idx]
        if not src_selected and not tgt_selected:
            return payload

        def _conflict(selected_items: Sequence[Dict[str, Any]], values: Sequence[float]) -> float:
            if not selected_items:
                return 0.0
            weights = [float(item.get("score", 0.0)) for item in selected_items]
            weighted = [float(values[idx]) for idx in range(len(selected_items))]
            return sum(weighted) / max(1e-8, sum(weights))

        src_vals = [src_unsupported[i] for i in src_idx]
        tgt_vals = [tgt_unsupported[i] for i in tgt_idx]
        c_x = _conflict(src_selected, src_vals)
        c_y = _conflict(tgt_selected, tgt_vals)
        c_diff = self._clip01(0.5 * (c_x + c_y))
        s_diff = self._clip01(1.0 - c_diff)
        cov_diff = self._clip01((len(src_selected) + len(tgt_selected)) / max(1.0, 2.0 * self.max_diff_triples))
        str_diff = 0.5 * (
            self._safe_mean([float(item.get("score", 0.0)) for item in src_selected]) +
            self._safe_mean([float(item.get("score", 0.0)) for item in tgt_selected])
        )
        stab_vals = list(src_vals) + list(tgt_vals)
        stab_diff = self._clip01(1.0 - min(1.0, 2.0 * self._safe_std(stab_vals)))
        q_diff = self._clip01((cov_diff + str_diff + stab_diff) / 3.0)
        src_sentences = self._verbalize_object_items(src_selected)
        tgt_sentences = self._verbalize_object_items(tgt_selected)
        total_imp = sum(src_vals) + sum(tgt_vals) or 1.0
        payload.update(
            {
                "score": s_diff,
                "quality": q_diff,
                "conflict": c_diff,
                "coverage": cov_diff,
                "strength": str_diff,
                "stability": stab_diff,
                "src_selected": [
                    {
                        "triple": list(item["triple"]),
                        "edge_ic": float(item.get("score", 0.0)),
                        "unsupported_mass": float(src_vals[pos]),
                        "importance": float(src_vals[pos] / total_imp),
                    }
                    for pos, item in enumerate(src_selected)
                ],
                "tgt_selected": [
                    {
                        "triple": list(item["triple"]),
                        "edge_ic": float(item.get("score", 0.0)),
                        "unsupported_mass": float(tgt_vals[pos]),
                        "importance": float(tgt_vals[pos] / total_imp),
                    }
                    for pos, item in enumerate(tgt_selected)
                ],
                "src_sentences": src_sentences,
                "tgt_sentences": tgt_sentences,
            }
        )
        return payload

    def _attribute_property_weight(self, prop_name: str) -> float:
        normalized = prop_name.lower()
        for key, weight in self.attribute_property_weights.items():
            if key.lower() in normalized:
                return float(weight)
        if any(token in normalized for token in ["definition", "def", "synopsis", "xref"]):
            return 1.0
        if any(token in normalized for token in ["identifier", "id", "code", "dbxref"]):
            return 0.8
        if any(token in normalized for token in ["comment", "note", "remark"]):
            return 0.6
        return 0.5

    def _attribute_weight(self, item: Dict[str, Any]) -> float:
        prop = self._normalize_text(item.get("prop"))
        value = self._normalize_text(item.get("value"))
        words = max(1, len(value.split()))
        info = min(1.0, float(torch.log1p(torch.tensor(float(words))).item() / torch.log1p(torch.tensor(20.0)).item()))
        return self._clip01(self._attribute_property_weight(prop) * info)

    def _score_attribute_channel(
        self,
        src_attrs: Sequence[Dict[str, Any]],
        tgt_attrs: Sequence[Dict[str, Any]],
        src_labels: Sequence[str],
        tgt_labels: Sequence[str],
        hierarchy_payloads: Dict[str, Dict[str, Any]],
        sim_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "score": self.tau,
            "quality": 0.0,
            "coverage": 0.0,
            "informativeness": 0.0,
            "stability": 0.0,
            "src_selected": [],
            "tgt_selected": [],
        }
        if not self.use_context:
            return payload

        src_items = list(src_attrs[: self.max_attr_items])
        tgt_items = list(tgt_attrs[: self.max_attr_items])
        if not src_items and not tgt_items:
            return payload

        tgt_bank = list(tgt_labels)
        src_bank = list(src_labels)
        for family_payload in hierarchy_payloads.values():
            tgt_bank.extend(family_payload.get("tgt_sentences", []))
            src_bank.extend(family_payload.get("src_sentences", []))
        tgt_bank.extend(sim_payload.get("tgt_sentences", []))
        src_bank.extend(sim_payload.get("src_sentences", []))
        tgt_bank.extend([self._normalize_text(item.get("text")) for item in tgt_items])
        src_bank.extend([self._normalize_text(item.get("text")) for item in src_items])
        tgt_bank = [text for text in tgt_bank if text]
        src_bank = [text for text in src_bank if text]

        def _side_support(
            side_items: Sequence[Dict[str, Any]],
            bank: Sequence[str],
        ) -> Tuple[float, List[Dict[str, Any]], List[float], List[float]]:
            if not side_items or not bank:
                return 0.0, [], [], []
            texts = [self._normalize_text(item.get("text")) for item in side_items]
            weights = [self._attribute_weight(item) for item in side_items]
            mat = self._encode_context_matrix(texts, list(bank))
            best = mat.max(dim=1).values.detach().cpu().tolist() if mat.numel() else [0.0 for _ in texts]
            denom = sum(weights) or 1.0
            score = sum(w * s for w, s in zip(weights, best)) / denom
            weighted_support = [w * s for w, s in zip(weights, best)]
            selected = []
            for item, weight, support, weighted in zip(side_items, weights, best, weighted_support):
                selected.append(
                    {
                        "property": self._normalize_text(item.get("prop")),
                        "value": self._normalize_text(item.get("value")),
                        "text": self._normalize_text(item.get("text")),
                        "support": float(support),
                        "weight": float(weight),
                        "importance": float(weighted),
                    }
                )
            return float(score), selected, weights, best

        src_score, src_selected, src_weights, src_supports = _side_support(src_items, tgt_bank)
        tgt_score, tgt_selected, tgt_weights, tgt_supports = _side_support(tgt_items, src_bank)
        side_scores = [score for score, items in [(src_score, src_selected), (tgt_score, tgt_selected)] if items]
        if not side_scores:
            return payload

        r_attr = self._safe_mean(side_scores)
        s_attr = max(self.tau, r_attr)
        cov_attr = self._clip01((len(src_items) + len(tgt_items)) / max(1.0, 2.0 * self.max_attr_items))
        inf_attr = 0.5 * (self._safe_mean(src_weights) + self._safe_mean(tgt_weights))
        stab_attr = self._clip01(1.0 - min(1.0, 2.0 * self._safe_std(list(src_supports) + list(tgt_supports))))
        q_attr = self._clip01((cov_attr + inf_attr + stab_attr) / 3.0)
        total_imp = sum(item["importance"] for item in src_selected + tgt_selected) or 1.0
        for item in src_selected:
            item["importance"] = float(item["importance"] / total_imp)
        for item in tgt_selected:
            item["importance"] = float(item["importance"] / total_imp)
        payload.update(
            {
                "score": float(s_attr),
                "quality": q_attr,
                "coverage": cov_attr,
                "informativeness": inf_attr,
                "stability": stab_attr,
                "src_selected": src_selected,
                "tgt_selected": tgt_selected,
            }
        )
        return payload

    def _brief_prompt(self, src_label: str, tgt_label: str, packet: str) -> Dict[str, str]:
        return {
            "system": "You are an ontology alignment analyst that returns strict JSON.",
            "user": (
                "Condense the pair evidence below into one compact ontology-alignment brief. "
                "Return exactly one JSON object with one key: \"summary\".\n"
                "The brief must keep these section titles in plain text:\n"
                "Label evidence\nHierarchy evidence\nRelational similarity evidence\n"
                "Distinctive conflicting evidence\nAuxiliary attribute evidence\n\n"
                f"Source entity: {src_label}\n"
                f"Target entity: {tgt_label}\n\n"
                f"Evidence packet:\n{packet}\n\n"
                "Return only JSON."
            ),
        }

    def _generate_briefs_uncached(
        self,
        prompts: List[Dict[str, str]],
        resolved_backend,
    ) -> List[str]:
        if not prompts:
            return []
        if resolved_backend.backend == "openrouter":
            profile = self._llm_router.profiles.get(resolved_backend.profile_name or "")
            if profile is None:
                raise RuntimeError("OpenRouter summary profile was resolved but not found.")
            return self._run_hosted_chat_prompts(
                prompts=prompts,
                profile=profile,
                max_tokens=self.max_new_tokens_llm,
                temperature=self.llm_temperature,
                top_p=self.llm_top_p,
                concurrency=self.llm_summary_batch_size,
            )
        self._ensure_local_llm()
        rendered = [self._render_llm_prompt(prompt) for prompt in prompts]
        outputs = [""] * len(rendered)
        chunk = self.llm_summary_batch_size or len(rendered)
        chunk = chunk if chunk > 0 else len(rendered)
        for start in range(0, len(rendered), chunk):
            end = min(start + chunk, len(rendered))
            enc = self.llm_tok(
                rendered[start:end],
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_total_tokens_llm_summary,
            ).to(self.device)
            with torch.amp.autocast(device_type=self.device_type, enabled=self.fp16):
                out = self.llm.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens_llm,
                    temperature=self.llm_temperature,
                    top_p=self.llm_top_p,
                    do_sample=self.llm_do_sample,
                    pad_token_id=self.llm_tok.eos_token_id,
                )
            new_tokens = self._strip_llm_prompt_tokens(enc, out)
            decoded = self.llm_tok.batch_decode(new_tokens, skip_special_tokens=True)
            for offset, text in enumerate(decoded):
                outputs[start + offset] = text
        return outputs

    def _decision_prompt(self, src_label: str, tgt_label: str, src_summary: str, tgt_summary: str) -> Dict[str, str]:
        pair_brief = src_summary
        return {
            "system": "You are an ontology alignment expert.",
            "user": (
                "Determine whether the following two ontology entities refer to the same concept.\n"
                "Answer with a single token: Yes or No.\n\n"
                f"Source entity: {src_label}\n"
                f"Target entity: {tgt_label}\n\n"
                f"Pair brief:\n{pair_brief}\n\n"
                "Answer:"
            ),
        }

    def _hosted_decision_prompt(
        self,
        src_label: str,
        tgt_label: str,
        src_summary: str,
        tgt_summary: str,
    ) -> Dict[str, str]:
        positive_label, negative_label = self.hosted_decision_labels
        pair_brief = src_summary
        return {
            "system": "You are a binary classifier for ontology alignment.",
            "user": (
                f"Return exactly one token: {positive_label} or {negative_label}.\n"
                f"{positive_label} = the source and target entities are equivalent.\n"
                f"{negative_label} = the source and target entities are not equivalent.\n\n"
                f"Source entity: {src_label}\n"
                f"Target entity: {tgt_label}\n\n"
                f"Pair brief:\n{pair_brief}"
            ),
        }

    def _rationale_prompt(
        self,
        src_label: str,
        tgt_label: str,
        src_summary: str,
        tgt_summary: str,
        decision: str,
    ) -> Dict[str, str]:
        pair_brief = src_summary
        return {
            "system": "You are an ontology alignment expert.",
            "user": (
                "Write a concise but specific rationale explaining the final alignment decision. "
                "Use only the pair brief below. Mention supporting and conflicting evidence when available. "
                "Return exactly one JSON object with one key: \"rationale\".\n\n"
                f"Source entity: {src_label}\n"
                f"Target entity: {tgt_label}\n"
                f"Final decision: {decision}\n\n"
                f"Pair brief:\n{pair_brief}\n\n"
                "Return only JSON."
            ),
        }

    @torch.inference_mode()
    def generate_pair_briefs_batched(
        self,
        src_labels: List[str],
        tgt_labels: List[str],
        evidence_packets: List[str],
    ) -> List[str]:
        if not self.use_llm:
            return list(evidence_packets)
        if not src_labels:
            return []
        summary_backend = self._llm_router.resolve_task("summary")
        self._last_summary_backend_meta = self._resolved_backend_metadata(summary_backend)
        outputs = [""] * len(src_labels)
        pending: Dict[str, Dict[str, Any]] = {}

        short_threshold = int(self.max_total_tokens_llm_summary * 0.75)
        for idx, (src_label, tgt_label, packet) in enumerate(zip(src_labels, tgt_labels, evidence_packets)):
            packet = self._normalize_text(packet)
            if not packet:
                outputs[idx] = ""
                continue
            if len(packet.split()) <= short_threshold:
                outputs[idx] = packet
                continue
            key = self._brief_key(src_label, tgt_label, packet)
            cached = self._summary_cache.get(key)
            if cached is not None:
                outputs[idx] = cached
                continue
            entry = pending.setdefault(
                key,
                {"src_label": src_label, "tgt_label": tgt_label, "packet": packet, "indices": []},
            )
            entry["indices"].append(idx)

        if pending:
            pending_keys = list(pending.keys())
            prompts = [
                self._brief_prompt(
                    pending[key]["src_label"],
                    pending[key]["tgt_label"],
                    pending[key]["packet"],
                )
                for key in pending_keys
            ]
            generated = self._generate_briefs_uncached(prompts, summary_backend)
            for key, brief in zip(pending_keys, generated):
                clean = self._parse_structured_text(brief, "summary", self._clean_summary_text)
                self._cache_store(self._summary_cache, key, clean, self.max_cached_summaries)
                for idx in pending[key]["indices"]:
                    outputs[idx] = clean

        self._record_summary_stats(outputs)
        return outputs

    def generate_final_rationales_for_records(
        self,
        records: List[Dict[str, Any]],
        progress_callback=None,
    ) -> List[str]:
        if not (self.use_llm and self.generate_llm_rationales):
            return ["" for _ in records]
        src_labels: List[str] = []
        tgt_labels: List[str] = []
        pair_briefs: List[str] = []
        decisions: List[str] = []
        for record in records:
            labels = record.get("selected_labels") or {}
            prediction = record.get("prediction") or {}
            src_labels.append(str(labels.get("source", "")))
            tgt_labels.append(str(labels.get("target", "")))
            pair_briefs.append(str(record.get("llm_pair_brief", "")))
            decisions.append(str(prediction.get("rationale_decision_label", "")))
        return self.generate_rationales_batched(
            src_labels=src_labels,
            tgt_labels=tgt_labels,
            src_summaries=pair_briefs,
            tgt_summaries=["" for _ in pair_briefs],
            decisions=decisions,
            progress_callback=progress_callback,
        )

    def _build_evidence_packet(
        self,
        src_label: str,
        tgt_label: str,
        hierarchy_payloads: Dict[str, Dict[str, Any]],
        sim_payload: Dict[str, Any],
        diff_payload: Dict[str, Any],
        attr_payload: Dict[str, Any],
    ) -> str:
        lines = [
            "Label evidence",
            f"Source label: {src_label}",
            f"Target label: {tgt_label}",
            "",
            "Hierarchy evidence",
        ]
        active_hier = False
        for family, payload in hierarchy_payloads.items():
            src_sentences = payload.get("src_sentences", [])
            tgt_sentences = payload.get("tgt_sentences", [])
            if not src_sentences and not tgt_sentences:
                continue
            active_hier = True
            lines.append(f"{self._family_display_name(family)} support (score={payload.get('score', self.tau):.3f})")
            for sentence in src_sentences[:3]:
                lines.append(f"Source: {sentence}")
            for sentence in tgt_sentences[:3]:
                lines.append(f"Target: {sentence}")
        if not active_hier:
            lines.append("No hierarchy evidence selected.")

        lines.extend(["", "Relational similarity evidence"])
        if sim_payload.get("src_sentences") or sim_payload.get("tgt_sentences"):
            lines.append(f"Similarity score={sim_payload.get('score', self.tau):.3f}")
            for sentence in list(sim_payload.get("src_sentences", []))[:4]:
                lines.append(f"Source: {sentence}")
            for sentence in list(sim_payload.get("tgt_sentences", []))[:4]:
                lines.append(f"Target: {sentence}")
        else:
            lines.append("No non-hierarchical relational similarity evidence selected.")

        lines.extend(["", "Distinctive conflicting evidence"])
        if diff_payload.get("src_sentences") or diff_payload.get("tgt_sentences"):
            lines.append(f"Difference compatibility score={diff_payload.get('score', self.tau):.3f}")
            for sentence in list(diff_payload.get("src_sentences", []))[:4]:
                lines.append(f"Source-only signal: {sentence}")
            for sentence in list(diff_payload.get("tgt_sentences", []))[:4]:
                lines.append(f"Target-only signal: {sentence}")
        else:
            lines.append("No distinctive conflicting evidence selected.")

        lines.extend(["", "Auxiliary attribute evidence"])
        if attr_payload.get("src_selected") or attr_payload.get("tgt_selected"):
            lines.append(f"Attribute support score={attr_payload.get('score', self.tau):.3f}")
            for item in list(attr_payload.get("src_selected", []))[:4]:
                lines.append(f"Source attribute: {item.get('text', '')}")
            for item in list(attr_payload.get("tgt_selected", []))[:4]:
                lines.append(f"Target attribute: {item.get('text', '')}")
        else:
            lines.append("No attribute evidence selected.")
        return "\n".join(line for line in lines if line is not None).strip()

    @torch.inference_mode()
    def forward(
        self,
        src_iris: List[str],
        tgt_iris: List[str],
        src_label_lists: List[List[str]],
        tgt_label_lists: List[List[str]],
        src_contexts: Optional[List[List[str]]] = None,
        tgt_contexts: Optional[List[List[str]]] = None,
        src_ctx_raw: Optional[List[List[str]]] = None,
        tgt_ctx_raw: Optional[List[List[str]]] = None,
        src_ctx_bridges: Optional[List[List[str]]] = None,
        tgt_ctx_bridges: Optional[List[List[str]]] = None,
        label: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        n_pairs = len(src_label_lists)
        assert len(tgt_label_lists) == n_pairs
        assert len(src_iris) == n_pairs and len(tgt_iris) == n_pairs
        dataset = self._attached_dataset
        if dataset is None:
            raise RuntimeError("PairAdaptiveSemanticScorer requires an attached dataset.")

        self._log_once(
            "pair_adaptive_inference_mode",
            (
                "Pair-adaptive scoring assembles pair-specific evidence inside inference batches "
                "from per-entity ontology pools (hierarchy, non-hierarchical triples, attributes)."
            ),
            "info",
        )

        src_unique_iris = list(dict.fromkeys(src_iris))
        tgt_unique_iris = list(dict.fromkeys(tgt_iris))
        cache_probe = getattr(dataset, "has_entity_features_cached", None)
        src_cache_hits = sum(
            1 for iri in src_unique_iris
            if callable(cache_probe) and bool(cache_probe(iri, "src"))
        )
        tgt_cache_hits = sum(
            1 for iri in tgt_unique_iris
            if callable(cache_probe) and bool(cache_probe(iri, "tgt"))
        )
        src_feature_map = {
            iri: dataset.get_entity_features(iri, "src")
            for iri in src_unique_iris
        }
        tgt_feature_map = {
            iri: dataset.get_entity_features(iri, "tgt")
            for iri in tgt_unique_iris
        }
        self._computed_llm_calibration = None
        self._calibration_messages = []
        self._last_summary_backend_meta = {}
        self._last_decision_backend_meta = {}
        self._last_rationale_backend_meta = {}

        s_label, q_label, best_pairs = self._score_label_channel(src_label_lists, tgt_label_lists)
        s_label_star = s_label

        family_names = list(self.hierarchical_relation_families.keys() or [])
        if "is_a" not in family_names:
            family_names = ["is_a"] + family_names

        pair_payloads: List[Dict[str, Any]] = []
        struct_channel_scores: Dict[str, List[float]] = {}
        struct_channel_qualities: Dict[str, List[float]] = {}
        for family in family_names:
            struct_channel_scores[f"hier__{family}"] = []
            struct_channel_qualities[f"hier__{family}"] = []
        for key in ["sim_obj", "diff", "attr_aux"]:
            struct_channel_scores[key] = []
            struct_channel_qualities[key] = []

        for idx, (src_iri, tgt_iri) in enumerate(zip(src_iris, tgt_iris)):
            src_feats = src_feature_map[src_iri]
            tgt_feats = tgt_feature_map[tgt_iri]
            src_best_label, tgt_best_label = best_pairs[idx]
            if not src_best_label:
                src_best_label = src_feats["labels"][0] if src_feats.get("labels") else ""
            if not tgt_best_label:
                tgt_best_label = tgt_feats["labels"][0] if tgt_feats.get("labels") else ""

            hierarchy_payloads: Dict[str, Dict[str, Any]] = {}
            for family in family_names:
                family_payload = self._score_hierarchy_family(
                    family,
                    src_feats.get("hierarchy", {}).get(family, []),
                    tgt_feats.get("hierarchy", {}).get(family, []),
                )
                hierarchy_payloads[family] = family_payload
                struct_channel_scores[f"hier__{family}"].append(float(family_payload["score"]))
                struct_channel_qualities[f"hier__{family}"].append(float(family_payload["quality"]))

            sim_payload = self._score_similarity_channel(
                src_feats.get("object_triples", []),
                tgt_feats.get("object_triples", []),
            )
            diff_payload = self._score_difference_channel(
                src_feats.get("object_triples", []),
                tgt_feats.get("object_triples", []),
                support_mat=sim_payload.get("support_matrix"),
            )
            attr_payload = self._score_attribute_channel(
                src_feats.get("attributes", []),
                tgt_feats.get("attributes", []),
                src_feats.get("labels", []),
                tgt_feats.get("labels", []),
                hierarchy_payloads,
                sim_payload,
            )
            struct_channel_scores["sim_obj"].append(float(sim_payload["score"]))
            struct_channel_qualities["sim_obj"].append(float(sim_payload["quality"]))
            struct_channel_scores["diff"].append(float(diff_payload["score"]))
            struct_channel_qualities["diff"].append(float(diff_payload["quality"]))
            struct_channel_scores["attr_aux"].append(float(attr_payload["score"]))
            struct_channel_qualities["attr_aux"].append(float(attr_payload["quality"]))

            packet = self._build_evidence_packet(
                src_best_label,
                tgt_best_label,
                hierarchy_payloads,
                sim_payload,
                diff_payload,
                attr_payload,
            )
            pair_payloads.append(
                {
                    "src_label": src_best_label,
                    "tgt_label": tgt_best_label,
                    "hierarchy": hierarchy_payloads,
                    "sim": sim_payload,
                    "diff": diff_payload,
                    "attr": attr_payload,
                    "packet": packet,
                }
            )

        src_pool_stats = self._batch_pool_stats(src_feature_map, family_names)
        tgt_pool_stats = self._batch_pool_stats(tgt_feature_map, family_names)
        pair_evidence_stats = self._batch_selected_evidence_stats(pair_payloads, family_names)

        channel_score_tensors = {
            key: torch.tensor(values, dtype=torch.float32, device=self.device)
            for key, values in struct_channel_scores.items()
        }
        channel_quality_tensors = {
            key: torch.tensor(values, dtype=torch.float32, device=self.device)
            for key, values in struct_channel_qualities.items()
        }

        sigma_tensors: Dict[str, torch.Tensor] = {}
        for key, score_tensor in channel_score_tensors.items():
            quality_tensor = channel_quality_tensors[key]
            sigma_tensors[key] = quality_tensor * (score_tensor - self.tau).abs().pow(self.gamma)

        sigma_sum = torch.zeros(n_pairs, device=self.device)
        for tensor in sigma_tensors.values():
            sigma_sum = sigma_sum + tensor

        struct_weights: Dict[str, torch.Tensor] = {}
        for key, tensor in sigma_tensors.items():
            struct_weights[key] = torch.where(
                sigma_sum > 1e-8,
                tensor / sigma_sum.clamp_min(1e-8),
                torch.zeros_like(tensor),
            )

        S_struct = torch.full((n_pairs,), float(self.tau), device=self.device)
        if struct_weights:
            structural_terms = [struct_weights[key] * channel_score_tensors[key] for key in struct_weights]
            structural_sum = torch.stack(structural_terms, dim=0).sum(dim=0)
            S_struct = torch.where(sigma_sum > 1e-8, structural_sum, S_struct)

        Q_struct = torch.zeros(n_pairs, device=self.device)
        if struct_weights:
            quality_terms = [struct_weights[key] * channel_quality_tensors[key] for key in struct_weights]
            Q_struct = torch.where(
                sigma_sum > 1e-8,
                torch.stack(quality_terms, dim=0).sum(dim=0),
                torch.zeros_like(sigma_sum),
            )

        hier_keys = [f"hier__{family}" for family in family_names]
        hier_sigma = sum((sigma_tensors[key] for key in hier_keys), torch.zeros(n_pairs, device=self.device))
        s_hier = torch.full((n_pairs,), float(self.tau), device=self.device)
        if hier_keys:
            hier_weighted = sum(
                (sigma_tensors[key] * channel_score_tensors[key] for key in hier_keys),
                torch.zeros(n_pairs, device=self.device),
            )
            active_hier = hier_sigma > 1e-8
            if torch.any(active_hier):
                s_hier = s_hier.clone()
                s_hier[active_hier] = hier_weighted[active_hier] / hier_sigma[active_hier]

        s_sim = channel_score_tensors["sim_obj"]
        s_diff = channel_score_tensors["diff"]
        s_attr = channel_score_tensors["attr_aux"]
        q_hier = torch.zeros(n_pairs, device=self.device)
        if hier_keys:
            active_family_count = sum(
                (channel_quality_tensors[key] > 0).to(torch.float32) for key in hier_keys
            )
            family_quality_sum = sum(
                (channel_quality_tensors[key] for key in hier_keys),
                torch.zeros(n_pairs, device=self.device),
            )
            q_hier = torch.where(
                active_family_count > 0,
                family_quality_sum / active_family_count.clamp_min(1.0),
                torch.zeros_like(family_quality_sum),
            )

        sig_lex = q_label * (s_label - self.tau).abs().pow(self.gamma)
        sig_struct = Q_struct * (S_struct - self.tau).abs().pow(self.gamma)
        active_lex = sig_lex > 1e-8
        active_struct = sig_struct > 1e-8
        both_active = active_lex & active_struct
        only_lex = active_lex & ~active_struct
        only_struct = active_struct & ~active_lex

        w_struct = torch.zeros(n_pairs, device=self.device)
        if torch.any(both_active):
            w_struct = w_struct.clone()
            w_struct[both_active] = sig_struct[both_active] / (
                sig_lex[both_active] + sig_struct[both_active]
            ).clamp_min(1e-8)
        if torch.any(only_struct):
            w_struct = w_struct.clone()
            w_struct[only_struct] = 1.0

        S_base = torch.full((n_pairs,), float(self.tau), device=self.device)
        if torch.any(only_lex):
            S_base = S_base.clone()
            S_base[only_lex] = s_label[only_lex]
        if torch.any(only_struct):
            S_base = S_base.clone()
            S_base[only_struct] = S_struct[only_struct]
        if torch.any(both_active):
            S_base = S_base.clone()
            S_base[both_active] = (1.0 - w_struct[both_active]) * s_label[both_active] + w_struct[both_active] * S_struct[both_active]

        U_ind = (2.0 * (self.tau - (S_base - self.tau).abs())).clamp(0.0, 1.0)
        U_dis = ((q_label * Q_struct).clamp_min(0.0).sqrt() * (s_label - S_struct).abs()).clamp(0.0, 1.0)
        U = torch.maximum(U_ind, U_dis)
        if self.use_llm:
            w_i = (self.beta * U).clamp(0.0, 1.0)
            need_llm = U >= self.tau_LLM
        else:
            w_i = torch.zeros_like(U)
            need_llm = torch.zeros_like(U, dtype=torch.bool)

        pair_packets = [payload["packet"] for payload in pair_payloads]
        pair_briefs = list(pair_packets)
        llm_decisions = [""] * n_pairs
        llm_rationales = [""] * n_pairs
        batch_calibration_samples = 0

        decision_idxs: List[int] = []
        brief_idxs: List[int] = []
        if self.use_llm:
            decision_idxs = torch.nonzero(need_llm).flatten().tolist()
            brief_idxs = list(range(n_pairs)) if self.force_llm_summaries else list(decision_idxs)
        if self.use_llm and brief_idxs:
            brief_src = [pair_payloads[i]["src_label"] for i in brief_idxs]
            brief_tgt = [pair_payloads[i]["tgt_label"] for i in brief_idxs]
            brief_packets = [pair_packets[i] for i in brief_idxs]
            generated_briefs = self.generate_pair_briefs_batched(brief_src, brief_tgt, brief_packets)
            for offset, idx in enumerate(brief_idxs):
                pair_briefs[idx] = generated_briefs[offset]

        p_llm = torch.zeros(n_pairs, device=self.device)
        S_final = S_base.clone()
        if self.use_llm and decision_idxs:
            src_best = [pair_payloads[i]["src_label"] for i in decision_idxs]
            tgt_best = [pair_payloads[i]["tgt_label"] for i in decision_idxs]
            decision_briefs = [pair_briefs[i] for i in decision_idxs]
            p_yes_needed = self.llm_yesno_probs_batched(
                src_best,
                tgt_best,
                decision_briefs,
                ["" for _ in decision_briefs],
            )
            if self.use_llm_calibration:
                if self._llm_calibration_can_apply:
                    p_yes_needed = self._apply_llm_calibration(p_yes_needed)
                    self._calibration_messages.append("Applied configured LLM calibration coefficients.")
                else:
                    samples = self._collect_calibration_samples(decision_idxs, p_yes_needed, label)
                    if samples is not None:
                        probs_fit, labels_fit = samples
                        count = int(probs_fit.shape[0])
                        self._calibration_pending_probs.extend(probs_fit.detach().cpu().tolist())
                        self._calibration_pending_labels.extend(labels_fit.detach().cpu().tolist())
                        batch_calibration_samples += count
                        self._calibration_messages.append(
                            f"Collected {count} calibration samples this batch (total={len(self._calibration_pending_probs)})."
                        )
            p_llm[decision_idxs] = p_yes_needed
            decisions_needed = ["Yes" if float(prob) >= 0.5 else "No" for prob in p_yes_needed]
            for offset, idx in enumerate(decision_idxs):
                llm_decisions[idx] = decisions_needed[offset]
            S_final[need_llm] = (1.0 - w_i[need_llm]) * S_base[need_llm] + w_i[need_llm] * p_llm[need_llm]

        llm_used_mask = torch.zeros(n_pairs, dtype=torch.bool, device=self.device)
        if decision_idxs:
            llm_used_mask[decision_idxs] = True
        w_i_effective = w_i * llm_used_mask.to(w_i.dtype)

        I_label = (1.0 - w_i_effective) * (1.0 - w_struct)
        I_struct = (1.0 - w_i_effective) * w_struct
        I_hier = I_struct * sum((struct_weights[key] for key in hier_keys), torch.zeros(n_pairs, device=self.device))
        I_sim = I_struct * struct_weights["sim_obj"]
        I_diff = I_struct * struct_weights["diff"]
        I_attr = I_struct * struct_weights["attr_aux"]
        I_ctx = I_struct
        I_llm = w_i_effective
        w_c = w_struct
        struct_active_pairs = int((sigma_sum > 1e-8).sum().item())
        llm_gated_pairs = int(need_llm.to(torch.int32).sum().item())
        brief_requested_pairs = int(len(brief_idxs))
        decision_requested_pairs = int(len(decision_idxs))

        result = {
            "s_label": s_label,
            "s_label_star": s_label_star,
            "s_ctx": S_struct,
            "s_hier": s_hier,
            "s_sim": s_sim,
            "s_diff": s_diff,
            "s_attr": s_attr,
            "q_label": q_label,
            "q_hier": q_hier,
            "q_sim": channel_quality_tensors["sim_obj"],
            "q_diff": channel_quality_tensors["diff"],
            "q_attr": channel_quality_tensors["attr_aux"],
            "Q_struct": Q_struct,
            "S_base": S_base,
            "S_lctx": S_base,
            "S_struct": S_struct,
            "p_llm": p_llm,
            "S_final": S_final,
            "w_c": w_c,
            "w_struct": w_struct,
            "U": U,
            "U_ind": U_ind,
            "U_dis": U_dis,
            "w_i": w_i,
            "need_llm": need_llm,
            "I_label": I_label,
            "I_struct": I_struct,
            "I_hier": I_hier,
            "I_sim": I_sim,
            "I_diff": I_diff,
            "I_attr": I_attr,
            "I_ctx": I_ctx,
            "I_llm": I_llm,
            "llm_decisions": llm_decisions,
            "llm_rationales": llm_rationales,
            "llm_pair_briefs": pair_briefs,
            "llm_evidence_packets": pair_packets,
            "llm_calibration": self._llm_calibration_payload(batch_samples=batch_calibration_samples),
            "llm_summary_stats": self.llm_summary_stats(),
            "llm_decision_stats": self.llm_decision_stats(),
            "llm_summaries": {
                "source": pair_briefs,
                "target": ["" for _ in pair_briefs],
            },
            "backend_usage": {
                "summary": dict(self._last_summary_backend_meta),
                "decision": dict(self._last_decision_backend_meta),
                "rationale": dict(self._last_rationale_backend_meta),
            },
            "batch_pair_adaptive_stats": {
                "pairs": int(n_pairs),
                "unique_src": int(len(src_unique_iris)),
                "unique_tgt": int(len(tgt_unique_iris)),
                "src_cache_hits": int(src_cache_hits),
                "src_cache_misses": int(len(src_unique_iris) - src_cache_hits),
                "tgt_cache_hits": int(tgt_cache_hits),
                "tgt_cache_misses": int(len(tgt_unique_iris) - tgt_cache_hits),
                "src_pool": src_pool_stats,
                "tgt_pool": tgt_pool_stats,
                "pair_evidence": pair_evidence_stats,
                "struct_active_pairs": int(struct_active_pairs),
                "llm_gated_pairs": int(llm_gated_pairs),
                "brief_requested_pairs": int(brief_requested_pairs),
                "decision_requested_pairs": int(decision_requested_pairs),
            },
        }

        if self.return_explanations:
            explanations = []
            for idx, payload in enumerate(pair_payloads):
                family_scores = {
                    family: float(payload["hierarchy"][family]["score"])
                    for family in family_names
                }
                family_qualities = {
                    family: float(payload["hierarchy"][family]["quality"])
                    for family in family_names
                }
                family_weights = {
                    family: float(struct_weights[f"hier__{family}"][idx])
                    for family in family_names
                }
                hier_internal_weight = sum((struct_weights[key][idx] for key in hier_keys), torch.tensor(0.0, device=self.device))
                family_importances = {
                    family: float(I_struct[idx] * struct_weights[f"hier__{family}"][idx])
                    for family in family_names
                }
                family_contribs = {
                    family: float(I_struct[idx] * struct_weights[f"hier__{family}"][idx] * (channel_score_tensors[f"hier__{family}"][idx] - self.tau))
                    for family in family_names
                }
                explanations.append(
                    {
                        "src_iri": src_iris[idx],
                        "tgt_iri": tgt_iris[idx],
                        "models": {
                            "lexical_model": self.lexical_model_name if self.use_lexical else None,
                            "context_model": self.context_model_name if self.use_context else None,
                            "llm_model": None,
                            "llm_summary_model": self._last_summary_backend_meta.get("model") if self.use_llm else None,
                            "llm_decision_model": self._last_decision_backend_meta.get("model") if self.use_llm else None,
                            "llm_rationale_model": self._last_rationale_backend_meta.get("model") if self.use_llm else None,
                            "llm_local_fallback_model": self.llm_model_name if self.use_llm else None,
                        },
                        "llm_calibration": self._llm_calibration_payload(batch_samples=0),
                        "confidences": {
                            "s_label": float(s_label[idx]),
                            "s_label_star": float(s_label_star[idx]),
                            "s_hier": float(s_hier[idx]),
                            "s_sim": float(s_sim[idx]),
                            "s_diff": float(s_diff[idx]),
                            "s_attr": float(s_attr[idx]),
                            "Q_struct": float(Q_struct[idx]),
                            "S_base": float(S_base[idx]),
                            "S_struct": float(S_struct[idx]),
                            "p_llm": float(p_llm[idx]),
                            "S_lctx": float(S_base[idx]),
                            "S_final": float(S_final[idx]),
                            "family_scores": family_scores,
                        },
                        "qualities": {
                            "q_label": float(q_label[idx]),
                            "q_hier": float(q_hier[idx]),
                            "q_sim": float(channel_quality_tensors["sim_obj"][idx]),
                            "q_diff": float(channel_quality_tensors["diff"][idx]),
                            "q_attr": float(channel_quality_tensors["attr_aux"][idx]),
                            "family_qualities": family_qualities,
                        },
                        "weights": {
                            "w_label": float((1.0 - w_struct[idx]).item()),
                            "w_struct": float(w_struct[idx]),
                            "w_hier": float(hier_internal_weight.item()),
                            "w_sim": float(struct_weights["sim_obj"][idx]),
                            "w_diff": float(struct_weights["diff"][idx]),
                            "w_attr": float(struct_weights["attr_aux"][idx]),
                            "w_c": float(w_c[idx]),
                            "w_i": float(w_i[idx]),
                            "U": float(U[idx]),
                            "U_ind": float(U_ind[idx]),
                            "U_dis": float(U_dis[idx]),
                            "family_weights": family_weights,
                        },
                        "importances": {
                            "I_label": float(I_label[idx]),
                            "I_struct": float(I_struct[idx]),
                            "I_hier": float(I_hier[idx]),
                            "I_sim": float(I_sim[idx]),
                            "I_diff": float(I_diff[idx]),
                            "I_attr": float(I_attr[idx]),
                            "I_ctx": float(I_ctx[idx]),
                            "I_llm": float(I_llm[idx]),
                            "family_importances": family_importances,
                        },
                        "contributions": {
                            "C_label": float(I_label[idx] * (s_label[idx] - self.tau)),
                            "C_struct": float(I_struct[idx] * (S_struct[idx] - self.tau)),
                            "C_hier": float(sum((family_contribs[family] for family in family_names), 0.0)),
                            "C_sim": float(I_struct[idx] * struct_weights["sim_obj"][idx] * (s_sim[idx] - self.tau)),
                            "C_diff": float(I_struct[idx] * struct_weights["diff"][idx] * (s_diff[idx] - self.tau)),
                            "C_attr": float(I_struct[idx] * struct_weights["attr_aux"][idx] * (s_attr[idx] - self.tau)),
                            "C_llm": float(I_llm[idx] * (p_llm[idx] - S_base[idx])),
                            "family_contributions": family_contribs,
                        },
                        "prediction": {
                            "global_match": bool(S_final[idx] >= self.threshold),
                            "ground_truth": label[idx] if label is not None and idx < len(label) else None,
                            "llm_decision": llm_decisions[idx],
                            "llm_rationale": llm_rationales[idx],
                            "threshold_positive": bool(S_final[idx] >= self.threshold),
                            "saved_alignment_member": False,
                            "rationale_decision_label": "",
                        },
                        "selected_labels": {
                            "source": payload["src_label"],
                            "target": payload["tgt_label"],
                        },
                        "backend_usage": {
                            "summary": dict(self._last_summary_backend_meta),
                            "decision": dict(self._last_decision_backend_meta),
                            "rationale": dict(self._last_rationale_backend_meta),
                        },
                        "context_sentences": {
                            "hierarchy_source": {
                                family: list(payload["hierarchy"][family].get("src_sentences", []))
                                for family in family_names
                            },
                            "hierarchy_target": {
                                family: list(payload["hierarchy"][family].get("tgt_sentences", []))
                                for family in family_names
                            },
                            "similarity_source": list(payload["sim"].get("src_sentences", [])),
                            "similarity_target": list(payload["sim"].get("tgt_sentences", [])),
                            "difference_source": list(payload["diff"].get("src_sentences", [])),
                            "difference_target": list(payload["diff"].get("tgt_sentences", [])),
                        },
                        "context_triples": {
                            "hierarchy_source": {
                                family: [item["triple"] for item in payload["hierarchy"][family].get("src_selected", [])]
                                for family in family_names
                            },
                            "hierarchy_target": {
                                family: [item["triple"] for item in payload["hierarchy"][family].get("tgt_selected", [])]
                                for family in family_names
                            },
                            "similarity_source": [item["triple"] for item in payload["sim"].get("src_selected", [])],
                            "similarity_target": [item["triple"] for item in payload["sim"].get("tgt_selected", [])],
                            "difference_source": [item["triple"] for item in payload["diff"].get("src_selected", [])],
                            "difference_target": [item["triple"] for item in payload["diff"].get("tgt_selected", [])],
                        },
                        "attributes": {
                            "source": list(payload["attr"].get("src_selected", [])),
                            "target": list(payload["attr"].get("tgt_selected", [])),
                        },
                        "llm_pair_evidence_packet": pair_packets[idx],
                        "llm_pair_brief": pair_briefs[idx],
                        "llm_summaries": {
                            "source": pair_briefs[idx],
                            "target": "",
                        },
                        "triple_attributions": {
                            "hierarchy": {
                                family: {
                                    "source": list(payload["hierarchy"][family].get("src_selected", [])),
                                    "target": list(payload["hierarchy"][family].get("tgt_selected", [])),
                                }
                                for family in family_names
                            },
                            "similarity": {
                                "source": list(payload["sim"].get("src_selected", [])),
                                "target": list(payload["sim"].get("tgt_selected", [])),
                            },
                            "difference": {
                                "source": list(payload["diff"].get("src_selected", [])),
                                "target": list(payload["diff"].get("tgt_selected", [])),
                            },
                        },
                    }
                )
            result["explanations"] = explanations

        return result
