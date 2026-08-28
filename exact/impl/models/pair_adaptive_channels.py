from __future__ import annotations

import hashlib  # noqa: F401
import json  # noqa: F401
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: F401

import torch  # noqa: F401

from exact.impl.models.pair_adaptive_experiments import (
    abbreviation_similarity,
    isub_similarity,
    jaro_winkler_similarity,
    token_set_similarity,
)
from exact.utils.candidate_generation import normalize_candidate_text
from exact.utils.formatting import clip01, safe_mean  # noqa: F401


class PairAdaptiveChannelsMixin:
    def _score_label_channel(
        self,
        src_label_lists: List[List[str]],
        tgt_label_lists: List[List[str]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[Tuple[str, str]],
        List[Dict[str, Any]],
    ]:
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
            return (
                neutral,
                q_label,
                best_pairs,
                [self._empty_label_quality() for _ in range(n_pairs)],
            )

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
            return (
                neutral,
                q_label,
                best_pairs,
                [self._empty_label_quality() for _ in range(n_pairs)],
            )

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
        quality_payloads: List[Dict[str, Any]] = []
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
                quality_payloads.append(self._empty_label_quality())
                continue
            mat = self._sim01(src_embs @ tgt_embs.T)
            score, pair = self._select_label_pair(mat, src_labels, tgt_labels)
            z1, z2 = self._top_two_scores(mat)
            margin = 1.0 if mat.numel() <= 1 else self._clip01((z1 - z2) / max(1e-8, (1.0 - z2)))
            entropy_raw, entropy, entropy_quality_defined = (
                self._label_entropy_components(mat) if self.lex_enabled else (0.0, 0.0, False)
            )
            agreement = self._label_encoder_agreement(src_labels, tgt_labels, mat)
            quality_mode = (
                self.lex_config.get("quality", "margin") if self.lex_enabled else "margin"
            )
            if quality_mode == "margin":
                q_label = margin
            elif quality_mode == "entropy":
                q_label = entropy
            elif quality_mode == "encoder_agreement":
                q_label = agreement
            elif quality_mode == "constant":
                q_label = 1.0
            else:
                raise ValueError(f"unsupported lexical quality mode: {quality_mode!r}")
            s_vals.append(score)
            q_vals.append(q_label)
            best_pairs.append(pair)
            quality_payloads.append(
                {
                    "active": True,
                    "mode": quality_mode,
                    "top1": z1,
                    "top2": z2,
                    "margin": margin,
                    "entropy_raw": entropy_raw,
                    "entropy": entropy,
                    "entropy_quality_defined": entropy_quality_defined,
                    "encoder_agreement": agreement,
                    "label_pairs": int(mat.numel()),
                    "selected": q_label,
                }
            )
        return (
            torch.stack(s_vals),
            torch.tensor(q_vals, dtype=torch.float32, device=self.device),
            best_pairs,
            quality_payloads,
        )

    @staticmethod
    def _empty_label_quality() -> Dict[str, Any]:
        return {
            "active": False,
            "mode": "margin",
            "top1": 0.0,
            "top2": 0.0,
            "margin": 0.0,
            "entropy_raw": 0.0,
            "entropy": 0.0,
            "entropy_quality_defined": False,
            "encoder_agreement": 0.0,
            "label_pairs": 0,
            "selected": 0.0,
        }

    def _label_entropy_quality(self, matrix: torch.Tensor) -> float:
        return self._label_entropy_components(matrix)[1]

    def _label_entropy_components(self, matrix: torch.Tensor) -> Tuple[float, float, bool]:
        """Return raw entropy, normalized quality, and whether entropy is defined.

        E26 treats a single observed label-pair score as missing distributional
        evidence, not as perfect certainty. Keeping the boolean explicit also
        prevents downstream calibration code from silently binning that case as
        maximally reliable.
        """

        if matrix.numel() <= 1:
            return 0.0, 0.0, False
        top_m = max(2, int(self.lex_config.get("entropy_top_m", 5)))
        values = torch.topk(matrix.flatten(), k=min(top_m, matrix.numel())).values.float()
        probabilities = torch.softmax(values, dim=0)
        entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
        normalizer = torch.log(torch.tensor(float(values.numel()), device=values.device))
        quality = self._clip01(1.0 - float((entropy / normalizer.clamp_min(1.0e-12)).item()))
        return float(entropy.item()), quality, True

    def _label_encoder_agreement(
        self,
        src_labels: Sequence[str],
        tgt_labels: Sequence[str],
        lexical_matrix: torch.Tensor,
    ) -> float:
        if not (self.lex_enabled and self.lex_config.get("quality") == "encoder_agreement"):
            return 0.0
        if not self.use_context:
            raise ValueError("lex.quality=encoder_agreement requires the context encoder")
        src_context = torch.nn.functional.normalize(
            self.encode_contexts_batch(list(src_labels)), dim=-1
        )
        tgt_context = torch.nn.functional.normalize(
            self.encode_contexts_batch(list(tgt_labels)), dim=-1
        )
        context_matrix = self._sim01(src_context @ tgt_context.T)
        lexical_flat = lexical_matrix.flatten()
        context_flat = context_matrix.flatten()
        lexical_best = float(torch.max(lexical_flat).item())
        context_best = float(torch.max(context_flat).item())
        return self._clip01(1.0 - abs(lexical_best - context_best))

    def _score_string_channel(
        self,
        src_label_lists: List[List[str]],
        tgt_label_lists: List[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """Score E06's independent, dependency-free lexical string signal."""

        scores: List[float] = []
        qualities: List[float] = []
        payloads: List[Dict[str, Any]] = []
        weights = {
            "isub": float(self.strsim_config.get("isub_weight", 1.0)),
            "jaro_winkler": float(self.strsim_config.get("jaro_winkler_weight", 1.0)),
            "token_set": float(self.strsim_config.get("token_set_weight", 1.0)),
        }
        abbreviation_enabled = self.strsim_config.get("abbreviation", "off") == "initialism"
        for src_labels, tgt_labels in zip(src_label_lists, tgt_label_lists):
            pair_rows: List[Dict[str, Any]] = []
            for src_label in src_labels:
                for tgt_label in tgt_labels:
                    component_scores = {
                        "isub": weights["isub"] * isub_similarity(src_label, tgt_label),
                        "jaro_winkler": weights["jaro_winkler"]
                        * jaro_winkler_similarity(src_label, tgt_label),
                        "token_set": weights["token_set"]
                        * token_set_similarity(src_label, tgt_label),
                    }
                    base_name, base_score = max(
                        component_scores.items(), key=lambda item: (item[1], item[0])
                    )
                    abbreviation_score = (
                        abbreviation_similarity(src_label, tgt_label)
                        if abbreviation_enabled
                        else 0.0
                    )
                    effective = max(base_score, abbreviation_score)
                    winner = "abbreviation" if abbreviation_score > base_score else base_name
                    pair_rows.append(
                        {
                            "source": src_label,
                            "target": tgt_label,
                            "score": self._clip01(effective),
                            "winner": winner,
                            "components": {
                                **{
                                    key: self._clip01(value)
                                    for key, value in component_scores.items()
                                },
                                "abbreviation": self._clip01(abbreviation_score),
                            },
                        }
                    )
            if not pair_rows:
                scores.append(float(self.tau))
                qualities.append(0.0)
                payloads.append({"active": False, "winner": None, "pairs": []})
                continue
            ordered = sorted(pair_rows, key=lambda row: float(row["score"]), reverse=True)
            z1 = float(ordered[0]["score"])
            z2 = float(ordered[1]["score"]) if len(ordered) > 1 else z1
            quality = 1.0 if len(ordered) <= 1 else self._clip01((z1 - z2) / max(1.0e-8, 1.0 - z2))
            scores.append(z1)
            qualities.append(quality)
            payloads.append(
                {
                    "active": True,
                    "score": z1,
                    "quality": quality,
                    "top1": z1,
                    "top2": z2,
                    "winner": ordered[0]["winner"],
                    "selected_labels": {
                        "source": ordered[0]["source"],
                        "target": ordered[0]["target"],
                    },
                    "components": ordered[0]["components"],
                }
            )
        return (
            torch.tensor(scores, dtype=torch.float32, device=self.device),
            torch.tensor(qualities, dtype=torch.float32, device=self.device),
            payloads,
        )

    def _score_hierarchy_family(
        self,
        family: str,
        src_items: Sequence[Any],
        tgt_items: Sequence[Any],
        src_iri: Optional[str] = None,
        tgt_iri: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "score": self.tau,
            "quality": 0.0,
            "strength": 0.0,
            "coverage": 0.0,
            "specificity": 0.0,
            "embedding": self.tau,
            "label_score": self.tau,
            "ancestor_overlap": 0.0,
            "anchor_coverage": 0.0,
            "sibling_conflict": 0.0,
            "sibling_coverage": 0.0,
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
        base_q_f = self._clip01((cov_f + str_f + inf_f) / 3.0)
        base_s_f = self._clip01(
            self.hierarchy_embedding_weight * emb_f + self.hierarchy_support_weight * str_f
        )
        s_f = base_s_f
        q_f = base_q_f
        overlap = 0.0
        anchor_coverage = 0.0
        sibling_conflict = 0.0
        sibling_coverage = 0.0
        if (
            self.hier_enabled
            and self.hier_config.get("mode") == "labels_overlap"
            and family == "is_a"
            and src_iri
            and tgt_iri
        ):
            overlap, anchor_coverage, sibling_conflict, sibling_coverage = (
                self._hierarchy_anchor_terms(src_iri, tgt_iri)
            )
            overlap_weight = float(self.hier_config.get("overlap_weight", 0.5))
            effective_weight = self._clip01(overlap_weight * anchor_coverage)
            s_f = (1.0 - effective_weight) * base_s_f + effective_weight * overlap
            if bool(self.hier_config.get("siblings", False)) and sibling_coverage > 0.0:
                s_f -= overlap_weight * sibling_coverage * sibling_conflict
            s_f = self._clip01(s_f)
            q_f = self._clip01((cov_f + str_f + inf_f + anchor_coverage) / 4.0)

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
                "label_score": base_s_f,
                "ancestor_overlap": overlap,
                "anchor_coverage": anchor_coverage,
                "sibling_conflict": sibling_conflict,
                "sibling_coverage": sibling_coverage,
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

    def _hierarchy_nodes(
        self,
        iri: str,
        side: str,
        *,
        upward: bool,
        depth: Optional[int],
    ) -> Dict[str, int]:
        dataset = self._attached_dataset
        source = dataset.source if side == "src" else dataset.target
        kind = dataset.entity_kind_for(iri, side, warn_unknown=False)
        method = source.direct_parents if upward else source.direct_children
        distances: Dict[str, int] = {}
        frontier = deque([(str(iri), 0)])
        while frontier:
            node, node_depth = frontier.popleft()
            if depth is not None and node_depth >= depth:
                continue
            for neighbor in method(node, kind):
                neighbor = str(neighbor)
                next_depth = node_depth + 1
                previous = distances.get(neighbor)
                if previous is not None and previous <= next_depth:
                    continue
                distances[neighbor] = next_depth
                frontier.append((neighbor, next_depth))
        return distances

    def _hierarchy_ic(self, iri: str, side: str) -> float:
        cache = getattr(self, "_hierarchy_ic_cache", None)
        if cache is None:
            cache = {}
            self._hierarchy_ic_cache = cache
        key = (side, str(iri))
        if key in cache:
            return float(cache[key])
        dataset = self._attached_dataset
        source = dataset.source if side == "src" else dataset.target
        kind = dataset.entity_kind_for(iri, side, warn_unknown=False)
        total = max(1, len(source.entities(kind)))
        descendants = self._hierarchy_nodes(
            iri,
            side,
            upward=False,
            depth=None,
        )
        if total <= 1:
            value = 1.0
        else:
            probability = min(1.0, (len(descendants) + 1.0) / (total + 1.0))
            value = (
                -torch.log(torch.tensor(probability)).item()
                / torch.log(torch.tensor(float(total + 1))).item()
            )
        cache[key] = self._clip01(value)
        return float(cache[key])

    def _hierarchy_anchor_terms(
        self, src_iri: str, tgt_iri: str
    ) -> Tuple[float, float, float, float]:
        depth = self.hier_config.get("depth", 2)
        depth = None if depth is None else int(depth)
        src_ancestors = self._hierarchy_nodes(src_iri, "src", upward=True, depth=depth)
        tgt_ancestors = self._hierarchy_nodes(tgt_iri, "tgt", upward=True, depth=depth)
        src_to_tgt = getattr(self, "_exact_anchor_src_to_tgt", {})
        tgt_to_src = getattr(self, "_exact_anchor_tgt_to_src", {})

        mapped_src: Dict[str, float] = {}
        for src_ancestor in src_ancestors:
            for mapped_target in src_to_tgt.get(src_ancestor, ()):
                weight = 0.5 * (
                    self._hierarchy_ic(src_ancestor, "src")
                    + self._hierarchy_ic(mapped_target, "tgt")
                )
                mapped_src[mapped_target] = max(mapped_src.get(mapped_target, 0.0), weight)
        target_weights: Dict[str, float] = {}
        for target in tgt_ancestors:
            for mapped_source in tgt_to_src.get(target, ()):
                weight = 0.5 * (
                    self._hierarchy_ic(mapped_source, "src") + self._hierarchy_ic(target, "tgt")
                )
                target_weights[target] = max(target_weights.get(target, 0.0), weight)
        anchored_targets = set(target_weights)
        union = set(mapped_src) | anchored_targets
        intersection = set(mapped_src) & anchored_targets
        denominator = sum(
            max(mapped_src.get(target, 0.0), target_weights.get(target, 0.0)) for target in union
        )
        numerator = sum(min(mapped_src[target], target_weights[target]) for target in intersection)
        overlap = numerator / denominator if denominator > 0.0 else 0.0
        src_anchored_count = sum(ancestor in src_to_tgt for ancestor in src_ancestors)
        tgt_anchored_count = sum(ancestor in tgt_to_src for ancestor in tgt_ancestors)
        coverage_terms = []
        if src_ancestors:
            coverage_terms.append(src_anchored_count / len(src_ancestors))
        if tgt_ancestors:
            coverage_terms.append(tgt_anchored_count / len(tgt_ancestors))
        coverage = self._safe_mean(coverage_terms)

        sibling_conflict = 0.0
        sibling_coverage = 0.0
        if bool(self.hier_config.get("siblings", False)):
            src_parents = self._hierarchy_nodes(src_iri, "src", upward=True, depth=1)
            tgt_parents = self._hierarchy_nodes(tgt_iri, "tgt", upward=True, depth=1)
            src_siblings = set()
            tgt_siblings = set()
            for parent in src_parents:
                src_siblings.update(self._hierarchy_nodes(parent, "src", upward=False, depth=1))
            for parent in tgt_parents:
                tgt_siblings.update(self._hierarchy_nodes(parent, "tgt", upward=False, depth=1))
            src_siblings.discard(src_iri)
            tgt_siblings.discard(tgt_iri)
            mapped_siblings = [
                mapped for sibling in src_siblings for mapped in src_to_tgt.get(sibling, ())
            ]
            if src_siblings:
                sibling_coverage = len(
                    {sibling for sibling in src_siblings if sibling in src_to_tgt}
                ) / len(src_siblings)
            if mapped_siblings:
                sibling_conflict = sum(
                    mapped not in tgt_siblings for mapped in mapped_siblings
                ) / len(mapped_siblings)
        return (
            self._clip01(overlap),
            self._clip01(coverage),
            self._clip01(sibling_conflict),
            self._clip01(sibling_coverage),
        )

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
            per_relation_cap=self.similarity_per_relation_cap,
            relation_getter=lambda item: item["triple"][1],
            tie_breaker=lambda item: item.get("score", 0.0),
        )
        tgt_idx = self._select_diverse_indices(
            tgt_items,
            col_best.detach().cpu().tolist(),
            min(self.max_object_triples, len(tgt_items)),
            per_relation_cap=self.similarity_per_relation_cap,
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
        stab_sim = self._clip01(
            1.0 - min(1.0, self.stability_factor * self._safe_std(stability_vals))
        )
        q_sim = self._clip01((cov_sim + str_sim + stab_sim) / 3.0)
        s_sim = self._clip01(
            self.similarity_embedding_weight * emb_sim + self.similarity_support_weight * str_sim
        )
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
            "formulation": "normalised",
            "n_triples_src": len(src_items),
            "n_triples_tgt": len(tgt_items),
            "unsupported_mass_src": 0.0,
            "unsupported_mass_tgt": 0.0,
            "c_x": 0.0,
            "unsupported_mean_src": 0.0,
            "unsupported_mean_tgt": 0.0,
            "diff_absolute": 0.0,
            "c_y": 0.0,
            "diff_pivot_reason": "explicit_neutral_fallback",
        }
        if not self.use_context:
            return payload

        formulation = (
            self.diff_config.get("formulation", "normalised") if self.diff_enabled else "normalised"
        )
        payload["formulation"] = formulation
        if formulation == "off":
            payload["diff_pivot_reason"] = "channel_off"
            return payload
        if self.diff_enabled and not src_items and not tgt_items:
            payload["diff_pivot_reason"] = "empty_both"
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
            per_relation_cap=self.difference_per_relation_cap,
            relation_getter=lambda item: item["triple"][1],
            tie_breaker=lambda item: item.get("score", 0.0),
        )
        tgt_idx = self._select_diverse_indices(
            tgt_items,
            tgt_unsupported,
            min(self.max_diff_triples, len(tgt_items)),
            per_relation_cap=self.difference_per_relation_cap,
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
        unsupported_mass_src = sum(src_unsupported)
        unsupported_mass_tgt = sum(tgt_unsupported)
        unsupported_mean_src = self._safe_mean(src_unsupported)
        unsupported_mean_tgt = self._safe_mean(tgt_unsupported)
        diff_absolute = self._clip01(0.5 * (unsupported_mean_src + unsupported_mean_tgt))
        if formulation == "absolute":
            c_diff = diff_absolute
        elif formulation == "asymmetric":
            c_diff = self._clip01(c_y)
        else:
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
        stab_diff = self._clip01(1.0 - min(1.0, self.stability_factor * self._safe_std(stab_vals)))
        q_diff = self._clip01((cov_diff + str_diff + stab_diff) / 3.0)
        if not src_items:
            pivot_reason = "empty_source"
        elif not tgt_items:
            pivot_reason = "empty_target"
        elif abs(s_diff - self.tau) < 1.0e-6:
            pivot_reason = "genuinely_neutral_mass"
            if formulation == "normalised" and abs(c_x + c_y - 1.0) < 1.0e-6:
                pivot_reason = (
                    "balanced_unsupported_mass"
                    if abs(c_x - c_y) < 1.0e-6
                    else "normalisation_balance"
                )
        else:
            pivot_reason = "non_pivot"
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
                "formulation": formulation,
                "n_triples_src": len(src_items),
                "n_triples_tgt": len(tgt_items),
                "unsupported_mass_src": unsupported_mass_src,
                "unsupported_mass_tgt": unsupported_mass_tgt,
                "unsupported_mean_src": unsupported_mean_src,
                "unsupported_mean_tgt": unsupported_mean_tgt,
                "diff_absolute": diff_absolute,
                "c_x": c_x,
                "c_y": c_y,
                "diff_pivot_reason": pivot_reason,
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
        category_names = {"definition", "identifier", "comment", "other"}
        for key, weight in self.attribute_property_weights.items():
            if key.lower() in category_names:
                continue
            if key.lower() in normalized:
                return float(weight)
        if any(token in normalized for token in ["definition", "def", "synopsis", "xref"]):
            return float(self.attribute_property_weights.get("definition", 1.0))
        if any(token in normalized for token in ["identifier", "id", "code", "dbxref"]):
            return float(self.attribute_property_weights.get("identifier", 0.8))
        if any(token in normalized for token in ["comment", "note", "remark"]):
            return float(self.attribute_property_weights.get("comment", 0.6))
        return float(self.attribute_property_weights.get("other", 0.5))

    def _attribute_weight(self, item: Dict[str, Any]) -> float:
        prop = self._normalize_text(item.get("prop"))
        value = self._normalize_text(item.get("value"))
        words = max(1, len(value.split()))
        info = min(
            1.0,
            float(
                torch.log1p(torch.tensor(float(words))).item()
                / torch.log1p(
                    torch.tensor(float(max(1, self.attribute_information_word_cap)))
                ).item()
            ),
        )
        return self._clip01(self._attribute_property_weight(prop) * info)

    def _signed_identifier_group(self, item: Dict[str, Any]) -> Optional[str]:
        prop_iri = self._normalize_text(item.get("prop_iri"))
        allowlist = {
            self._normalize_text(value)
            for value in self.attr_config.get("signed_property_allowlist", [])
            if self._normalize_text(value)
        }
        if not prop_iri or prop_iri not in allowlist:
            return None
        namespace = self._normalize_text(item.get("identifier_namespace"))
        if not namespace:
            raise ValueError(
                f"signed identifier property {prop_iri!r} lacks descriptor namespace semantics"
            )
        return namespace

    def _signed_identifier_disagreement(
        self,
        src_items: Sequence[Dict[str, Any]],
        tgt_items: Sequence[Dict[str, Any]],
    ) -> Tuple[float, int, List[str]]:
        src_groups: Dict[str, set[str]] = {}
        tgt_groups: Dict[str, set[str]] = {}
        for item, groups in ((item, src_groups) for item in src_items):
            group = self._signed_identifier_group(item)
            value = self._normalize_text(item.get("identifier_normalized"))
            if group and not value:
                raise ValueError("signed identifier evidence lacks descriptor-normalized value")
            if group and value:
                groups.setdefault(group, set()).add(value)
        for item, groups in ((item, tgt_groups) for item in tgt_items):
            group = self._signed_identifier_group(item)
            value = self._normalize_text(item.get("identifier_normalized"))
            if group and not value:
                raise ValueError("signed identifier evidence lacks descriptor-normalized value")
            if group and value:
                groups.setdefault(group, set()).add(value)
        comparable = sorted(set(src_groups) & set(tgt_groups))
        if not comparable:
            return 0.0, 0, []
        disagreements = [
            0.0 if src_groups[group] & tgt_groups[group] else 1.0 for group in comparable
        ]
        return self._safe_mean(disagreements), len(comparable), comparable

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
            "bank": "full",
            "polarity": "support_only",
            "identifier_disagreement": 0.0,
            "identifier_comparable_groups": 0,
            "identifier_groups": [],
            "identifier_support": 0.0,
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

        bank_mode = self.attr_config.get("bank", "full") if self.attr_enabled else "full"
        polarity = (
            self.attr_config.get("polarity", "support_only")
            if self.attr_enabled
            else "support_only"
        )
        tgt_bank = []
        src_bank = []
        if bank_mode in {"full", "attrs_labels"}:
            tgt_bank.extend(
                {
                    "kind": "label",
                    "anchor_ref": "__target__",
                    "text": self._normalize_text(label),
                }
                for label in tgt_labels
                if self._normalize_text(label)
            )
            src_bank.extend(
                {
                    "kind": "label",
                    "anchor_ref": "__source__",
                    "text": self._normalize_text(label),
                }
                for label in src_labels
                if self._normalize_text(label)
            )
        if bank_mode == "full":
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
                        "property_iri": self._normalize_text(item.get("prop_iri")),
                        "identifier_namespace": self._normalize_text(
                            item.get("identifier_namespace")
                        ),
                        "identifier_normalized": self._normalize_text(
                            item.get("identifier_normalized")
                        ),
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
        s_attr = self._clip01(max(self.attribute_score_floor, r_attr))
        identifier_disagreement = 0.0
        identifier_comparable_groups = 0
        identifier_groups: List[str] = []
        identifier_support = 0.0
        if polarity == "signed":
            (
                identifier_disagreement,
                identifier_comparable_groups,
                identifier_groups,
            ) = self._signed_identifier_disagreement(src_items, tgt_items)
            signed_supports = [
                float(selected.get("support", 0.0))
                for item, selected in [
                    *zip(src_items, src_selected),
                    *zip(tgt_items, tgt_selected),
                ]
                if self._signed_identifier_group(item) is not None
            ]
            identifier_support = self._safe_mean(signed_supports)
            if identifier_comparable_groups and identifier_support >= 0.8:
                s_attr = self._clip01(s_attr - self.tau * identifier_disagreement)
        cov_attr = self._clip01(
            (len(src_items) + len(tgt_items)) / max(1.0, 2.0 * self.max_attr_items)
        )
        inf_attr = 0.5 * (self._safe_mean(src_weights) + self._safe_mean(tgt_weights))
        stab_attr = self._clip01(
            1.0
            - min(
                1.0,
                self.stability_factor * self._safe_std(list(src_supports) + list(tgt_supports)),
            )
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
                "bank": bank_mode,
                "polarity": polarity,
                "identifier_disagreement": identifier_disagreement,
                "identifier_comparable_groups": identifier_comparable_groups,
                "identifier_groups": identifier_groups,
                "identifier_support": identifier_support,
            }
        )
        return payload
