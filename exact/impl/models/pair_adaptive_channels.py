from __future__ import annotations

import hashlib  # noqa: F401
import json  # noqa: F401
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: F401

import torch  # noqa: F401

from exact.utils.formatting import clip01, safe_mean  # noqa: F401


class PairAdaptiveChannelsMixin:
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
        return (
            torch.stack(s_vals),
            torch.tensor(q_vals, dtype=torch.float32, device=self.device),
            best_pairs,
        )

    def _score_hierarchy_family(
        self,
        family: str,
        src_items: Sequence[Any],
        tgt_items: Sequence[Any],
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
            "links": [],
        }
        if not self.use_context or not src_items or not tgt_items:
            return payload

        src_tails = [self._hier_item_triple(item)[2] for item in src_items]
        tgt_tails = [self._hier_item_triple(item)[2] for item in tgt_items]
        support_mat = self._encode_label_matrix(src_tails, tgt_tails)
        row_best = (
            support_mat.max(dim=1).values
            if support_mat.numel()
            else torch.zeros(len(src_items), device=self.device)
        )
        col_best = (
            support_mat.max(dim=0).values
            if support_mat.numel()
            else torch.zeros(len(tgt_items), device=self.device)
        )

        src_idx = self._select_diverse_indices(
            src_items,
            row_best.detach().cpu().tolist(),
            self.max_hierarchy_triples_per_family,
            per_relation_cap=None,
            relation_getter=lambda _: family,
            tie_breaker=lambda item: self._hier_item_specificity(item),
        )
        tgt_idx = self._select_diverse_indices(
            tgt_items,
            col_best.detach().cpu().tolist(),
            self.max_hierarchy_triples_per_family,
            per_relation_cap=None,
            relation_getter=lambda _: family,
            tie_breaker=lambda item: self._hier_item_specificity(item),
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
        emb_f = self._context_similarity_from_sentences(
            src_sentences, tgt_sentences, self.max_input_tokens_hier
        )
        spec_vals = [self._hier_item_specificity(item) for item in src_selected] + [
            self._hier_item_specificity(item) for item in tgt_selected
        ]
        inf_f = self._safe_mean(spec_vals)
        cov_f = self._clip01(
            (len(src_selected) + len(tgt_selected))
            / max(1.0, 2.0 * self.max_hierarchy_triples_per_family)
        )
        q_f = self._clip01((cov_f + str_f + inf_f) / 3.0)
        s_f = self._clip01(0.5 * emb_f + 0.5 * str_f)

        src_imp = [float(value) for value in src_support]
        tgt_imp = [float(value) for value in tgt_support]
        total_imp = sum(src_imp) + sum(tgt_imp) or 1.0
        src_selected_rows = [
            self._with_item_id(
                "hierarchy",
                "source",
                {
                    "triple": list(self._hier_item_triple(item)),
                    "specificity": self._hier_item_specificity(item),
                    "subject_iri": self._hier_item_subject_iri(item),
                    "object_iri": self._hier_item_object_iri(item),
                    "support": float(src_support[pos]),
                    "importance": float(src_imp[pos] / total_imp),
                },
                family=family,
            )
            for pos, item in enumerate(src_selected)
        ]
        tgt_selected_rows = [
            self._with_item_id(
                "hierarchy",
                "target",
                {
                    "triple": list(self._hier_item_triple(item)),
                    "specificity": self._hier_item_specificity(item),
                    "subject_iri": self._hier_item_subject_iri(item),
                    "object_iri": self._hier_item_object_iri(item),
                    "support": float(tgt_support[pos]),
                    "importance": float(tgt_imp[pos] / total_imp),
                },
                family=family,
            )
            for pos, item in enumerate(tgt_selected)
        ]
        payload.update(
            {
                "score": s_f,
                "quality": q_f,
                "strength": str_f,
                "coverage": cov_f,
                "specificity": inf_f,
                "embedding": emb_f,
                "src_selected": src_selected_rows,
                "tgt_selected": tgt_selected_rows,
                "src_sentences": src_sentences,
                "tgt_sentences": tgt_sentences,
                "links": self._matrix_provenance_links(
                    src_selected_rows, tgt_selected_rows, reduced
                ),
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
            "links": [],
        }
        if not self.use_context or not src_items or not tgt_items:
            return payload

        support_mat = self._object_support_matrix(src_items, tgt_items)
        row_best = (
            support_mat.max(dim=1).values
            if support_mat.numel()
            else torch.zeros(len(src_items), device=self.device)
        )
        col_best = (
            support_mat.max(dim=0).values
            if support_mat.numel()
            else torch.zeros(len(tgt_items), device=self.device)
        )
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
        emb_sim = self._context_similarity_from_sentences(
            src_sentences, tgt_sentences, self.max_input_tokens_sim
        )
        cov_sim = self._clip01(
            (len(src_selected) + len(tgt_selected)) / max(1.0, 2.0 * self.max_object_triples)
        )
        stability_vals = list(src_support) + list(tgt_support)
        stab_sim = self._clip01(1.0 - min(1.0, 2.0 * self._safe_std(stability_vals)))
        q_sim = self._clip01((cov_sim + str_sim + stab_sim) / 3.0)
        s_sim = self._clip01(0.5 * emb_sim + 0.5 * str_sim)
        src_imp = [float(value) for value in src_support]
        tgt_imp = [float(value) for value in tgt_support]
        total_imp = sum(src_imp) + sum(tgt_imp) or 1.0
        src_selected_rows = [
            self._with_item_id(
                "similarity",
                "source",
                {
                    "triple": list(item["triple"]),
                    "support": float(src_support[pos]),
                    "edge_ic": float(item.get("score", 0.0)),
                    "subject_iri": self._normalize_text(item.get("subject_iri")),
                    "object_iri": self._normalize_text(item.get("object_iri")),
                    "rel_iri": self._normalize_text(item.get("rel_iri")),
                    "importance": float(src_imp[pos] / total_imp),
                },
            )
            for pos, item in enumerate(src_selected)
        ]
        tgt_selected_rows = [
            self._with_item_id(
                "similarity",
                "target",
                {
                    "triple": list(item["triple"]),
                    "support": float(tgt_support[pos]),
                    "edge_ic": float(item.get("score", 0.0)),
                    "subject_iri": self._normalize_text(item.get("subject_iri")),
                    "object_iri": self._normalize_text(item.get("object_iri")),
                    "rel_iri": self._normalize_text(item.get("rel_iri")),
                    "importance": float(tgt_imp[pos] / total_imp),
                },
            )
            for pos, item in enumerate(tgt_selected)
        ]
        payload.update(
            {
                "score": s_sim,
                "quality": q_sim,
                "strength": str_sim,
                "coverage": cov_sim,
                "stability": stab_sim,
                "embedding": emb_sim,
                "src_selected": src_selected_rows,
                "tgt_selected": tgt_selected_rows,
                "src_sentences": src_sentences,
                "tgt_sentences": tgt_sentences,
                "support_matrix": support_mat,
                "row_best": row_best.detach().cpu().tolist(),
                "col_best": col_best.detach().cpu().tolist(),
                "links": self._matrix_provenance_links(
                    src_selected_rows, tgt_selected_rows, reduced
                ),
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
            "source_links": [],
            "target_links": [],
        }
        if not self.use_context:
            return payload

        if support_mat is None:
            support_mat = self._object_support_matrix(src_items, tgt_items)
        row_best = (
            support_mat.max(dim=1).values
            if src_items and tgt_items
            else torch.zeros(len(src_items), device=self.device)
        )
        col_best = (
            support_mat.max(dim=0).values
            if src_items and tgt_items
            else torch.zeros(len(tgt_items), device=self.device)
        )

        src_unsupported = [
            float(item.get("score", 0.0))
            * (1.0 - float(row_best[idx].item() if idx < row_best.numel() else 0.0))
            for idx, item in enumerate(src_items)
        ]
        tgt_unsupported = [
            float(item.get("score", 0.0))
            * (1.0 - float(col_best[idx].item() if idx < col_best.numel() else 0.0))
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
        cov_diff = self._clip01(
            (len(src_selected) + len(tgt_selected)) / max(1.0, 2.0 * self.max_diff_triples)
        )
        str_diff = 0.5 * (
            self._safe_mean([float(item.get("score", 0.0)) for item in src_selected])
            + self._safe_mean([float(item.get("score", 0.0)) for item in tgt_selected])
        )
        stab_vals = list(src_vals) + list(tgt_vals)
        stab_diff = self._clip01(1.0 - min(1.0, 2.0 * self._safe_std(stab_vals)))
        q_diff = self._clip01((cov_diff + str_diff + stab_diff) / 3.0)
        src_sentences = self._verbalize_object_items(src_selected)
        tgt_sentences = self._verbalize_object_items(tgt_selected)
        total_imp = sum(src_vals) + sum(tgt_vals) or 1.0
        src_selected_rows = [
            self._with_item_id(
                "difference",
                "source",
                {
                    "triple": list(item["triple"]),
                    "edge_ic": float(item.get("score", 0.0)),
                    "subject_iri": self._normalize_text(item.get("subject_iri")),
                    "object_iri": self._normalize_text(item.get("object_iri")),
                    "rel_iri": self._normalize_text(item.get("rel_iri")),
                    "unsupported_mass": float(src_vals[pos]),
                    "importance": float(src_vals[pos] / total_imp),
                },
            )
            for pos, item in enumerate(src_selected)
        ]
        tgt_selected_rows = [
            self._with_item_id(
                "difference",
                "target",
                {
                    "triple": list(item["triple"]),
                    "edge_ic": float(item.get("score", 0.0)),
                    "subject_iri": self._normalize_text(item.get("subject_iri")),
                    "object_iri": self._normalize_text(item.get("object_iri")),
                    "rel_iri": self._normalize_text(item.get("rel_iri")),
                    "unsupported_mass": float(tgt_vals[pos]),
                    "importance": float(tgt_vals[pos] / total_imp),
                },
            )
            for pos, item in enumerate(tgt_selected)
        ]
        payload.update(
            {
                "score": s_diff,
                "quality": q_diff,
                "conflict": c_diff,
                "coverage": cov_diff,
                "strength": str_diff,
                "stability": stab_diff,
                "src_selected": src_selected_rows,
                "tgt_selected": tgt_selected_rows,
                "src_sentences": src_sentences,
                "tgt_sentences": tgt_sentences,
                "source_links": [
                    {
                        "item_id": self._normalize_text(item.get("item_id")),
                        "anchor_kind": "endpoint",
                        "anchor_ref": "__target__",
                        "score": float(item.get("unsupported_mass", 0.0)),
                    }
                    for item in src_selected_rows
                    if self._normalize_text(item.get("item_id"))
                ],
                "target_links": [
                    {
                        "item_id": self._normalize_text(item.get("item_id")),
                        "anchor_kind": "endpoint",
                        "anchor_ref": "__source__",
                        "score": float(item.get("unsupported_mass", 0.0)),
                    }
                    for item in tgt_selected_rows
                    if self._normalize_text(item.get("item_id"))
                ],
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
        info = min(
            1.0,
            float(
                torch.log1p(torch.tensor(float(words))).item()
                / torch.log1p(torch.tensor(20.0)).item()
            ),
        )
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
            "source_links": [],
            "target_links": [],
        }
        if not self.use_context:
            return payload

        src_items = [
            self._with_item_id("attribute", "source", item)
            for item in list(src_attrs[: self.max_attr_items])
        ]
        tgt_items = [
            self._with_item_id("attribute", "target", item)
            for item in list(tgt_attrs[: self.max_attr_items])
        ]
        if not src_items and not tgt_items:
            return payload

        tgt_bank = [
            {"kind": "label", "anchor_ref": "__target__", "text": self._normalize_text(label)}
            for label in tgt_labels
            if self._normalize_text(label)
        ]
        src_bank = [
            {"kind": "label", "anchor_ref": "__source__", "text": self._normalize_text(label)}
            for label in src_labels
            if self._normalize_text(label)
        ]
        for family_payload in hierarchy_payloads.values():
            tgt_bank.extend(
                {
                    "kind": "hierarchy",
                    "anchor_ref": self._normalize_text(item.get("item_id")),
                    "text": self._normalize_text(sentence),
                }
                for item, sentence in zip(
                    list(family_payload.get("tgt_selected", [])),
                    list(family_payload.get("tgt_sentences", [])),
                )
                if self._normalize_text(item.get("item_id")) and self._normalize_text(sentence)
            )
            src_bank.extend(
                {
                    "kind": "hierarchy",
                    "anchor_ref": self._normalize_text(item.get("item_id")),
                    "text": self._normalize_text(sentence),
                }
                for item, sentence in zip(
                    list(family_payload.get("src_selected", [])),
                    list(family_payload.get("src_sentences", [])),
                )
                if self._normalize_text(item.get("item_id")) and self._normalize_text(sentence)
            )
        tgt_bank.extend(
            {
                "kind": "similarity",
                "anchor_ref": self._normalize_text(item.get("item_id")),
                "text": self._normalize_text(sentence),
            }
            for item, sentence in zip(
                list(sim_payload.get("tgt_selected", [])),
                list(sim_payload.get("tgt_sentences", [])),
            )
            if self._normalize_text(item.get("item_id")) and self._normalize_text(sentence)
        )
        src_bank.extend(
            {
                "kind": "similarity",
                "anchor_ref": self._normalize_text(item.get("item_id")),
                "text": self._normalize_text(sentence),
            }
            for item, sentence in zip(
                list(sim_payload.get("src_selected", [])),
                list(sim_payload.get("src_sentences", [])),
            )
            if self._normalize_text(item.get("item_id")) and self._normalize_text(sentence)
        )
        tgt_bank.extend(
            {
                "kind": "attribute",
                "anchor_ref": self._normalize_text(item.get("item_id")),
                "text": self._normalize_text(item.get("text")),
            }
            for item in tgt_items
            if self._normalize_text(item.get("item_id")) and self._normalize_text(item.get("text"))
        )
        src_bank.extend(
            {
                "kind": "attribute",
                "anchor_ref": self._normalize_text(item.get("item_id")),
                "text": self._normalize_text(item.get("text")),
            }
            for item in src_items
            if self._normalize_text(item.get("item_id")) and self._normalize_text(item.get("text"))
        )

        def _side_support(
            side_items: Sequence[Dict[str, Any]],
            bank: Sequence[Dict[str, Any]],
        ) -> Tuple[float, List[Dict[str, Any]], List[float], List[float], List[Dict[str, Any]]]:
            if not side_items or not bank:
                return 0.0, [], [], [], []
            texts = [self._normalize_text(item.get("text")) for item in side_items]
            weights = [self._attribute_weight(item) for item in side_items]
            bank_texts = [self._normalize_text(item.get("text")) for item in bank]
            mat = self._encode_context_matrix(texts, bank_texts)
            best = (
                mat.max(dim=1).values.detach().cpu().tolist()
                if mat.numel()
                else [0.0 for _ in texts]
            )
            best_idx = (
                mat.argmax(dim=1).detach().cpu().tolist() if mat.numel() else [0 for _ in texts]
            )
            denom = sum(weights) or 1.0
            score = sum(w * s for w, s in zip(weights, best)) / denom
            weighted_support = [w * s for w, s in zip(weights, best)]
            selected = []
            links = []
            for item, weight, support, weighted, anchor_idx in zip(
                side_items, weights, best, weighted_support, best_idx
            ):
                anchor = dict(bank[anchor_idx]) if bank and 0 <= int(anchor_idx) < len(bank) else {}
                selected.append(
                    {
                        "item_id": self._normalize_text(item.get("item_id")),
                        "property": self._normalize_text(item.get("prop")),
                        "value": self._normalize_text(item.get("value")),
                        "text": self._normalize_text(item.get("text")),
                        "entity_iri": self._normalize_text(item.get("entity_iri")),
                        "support": float(support),
                        "weight": float(weight),
                        "importance": float(weighted),
                    }
                )
                item_id = self._normalize_text(item.get("item_id"))
                anchor_kind = self._normalize_text(anchor.get("kind"))
                anchor_ref = self._normalize_text(anchor.get("anchor_ref"))
                if item_id and anchor_kind and anchor_ref:
                    links.append(
                        {
                            "item_id": item_id,
                            "anchor_kind": anchor_kind,
                            "anchor_ref": anchor_ref,
                            "score": float(support),
                        }
                    )
            return float(score), selected, weights, best, links

        src_score, src_selected, src_weights, src_supports, src_links = _side_support(
            src_items, tgt_bank
        )
        tgt_score, tgt_selected, tgt_weights, tgt_supports, tgt_links = _side_support(
            tgt_items, src_bank
        )
        side_scores = [
            score
            for score, items in [(src_score, src_selected), (tgt_score, tgt_selected)]
            if items
        ]
        if not side_scores:
            return payload

        r_attr = self._safe_mean(side_scores)
        s_attr = max(self.tau, r_attr)
        cov_attr = self._clip01(
            (len(src_items) + len(tgt_items)) / max(1.0, 2.0 * self.max_attr_items)
        )
        inf_attr = 0.5 * (self._safe_mean(src_weights) + self._safe_mean(tgt_weights))
        stab_attr = self._clip01(
            1.0 - min(1.0, 2.0 * self._safe_std(list(src_supports) + list(tgt_supports)))
        )
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
                "source_links": src_links,
                "target_links": tgt_links,
            }
        )
        return payload
