from __future__ import annotations

import hashlib  # noqa: F401
import json  # noqa: F401
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: F401

import torch  # noqa: F401

from exact.utils.formatting import clip01, safe_mean  # noqa: F401


class PairAdaptiveEvidenceMixin:
    def attach_dataset(self, dataset: Any) -> None:
        self._attached_dataset = dataset
        if not self.hierarchical_relation_families and hasattr(
            dataset, "hierarchical_relation_families"
        ):
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
                hierarchy.get(family, {}).get("src_selected")
                or hierarchy.get(family, {}).get("tgt_selected")
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
        payload = "\u241f".join([src_label or "", tgt_label or "", packet or ""])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _stable_item_id(
        self,
        channel: str,
        side: str,
        item: Dict[str, Any],
        family: Optional[str] = None,
    ) -> str:
        triple = [self._normalize_text(value) for value in list(item.get("triple") or [])[:3]]
        payload = {
            "channel": self._normalize_text(channel),
            "side": self._normalize_text(side),
            "family": self._normalize_text(family),
            "triple": triple,
            "property": self._normalize_text(item.get("property", item.get("prop"))),
            "value": self._normalize_text(item.get("value")),
            "text": self._normalize_text(item.get("text")),
        }
        digest = hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return f"{payload['channel']}-{payload['side']}-{digest}"

    def _with_item_id(
        self,
        channel: str,
        side: str,
        item: Dict[str, Any],
        family: Optional[str] = None,
    ) -> Dict[str, Any]:
        row = dict(item or {})
        row["item_id"] = self._stable_item_id(channel, side, row, family=family)
        if family and "family" not in row:
            row["family"] = family
        return row

    def _hier_item_triple(self, item: Any) -> Tuple[str, str, str]:
        if isinstance(item, dict):
            triple = list(item.get("triple") or ["", "", ""])
            triple = (triple + ["", "", ""])[:3]
            return (
                self._normalize_text(triple[0]),
                self._normalize_text(triple[1]),
                self._normalize_text(triple[2]),
            )
        triple = list(item[:3]) if item is not None else ["", "", ""]
        triple = (triple + ["", "", ""])[:3]
        return (
            self._normalize_text(triple[0]),
            self._normalize_text(triple[1]),
            self._normalize_text(triple[2]),
        )

    def _hier_item_specificity(self, item: Any) -> float:
        if isinstance(item, dict):
            return float(item.get("specificity", 0.0))
        try:
            return float(item[3])
        except Exception:
            return 0.0

    def _hier_item_subject_iri(self, item: Any) -> str:
        if isinstance(item, dict):
            return self._normalize_text(item.get("subject_iri"))
        try:
            return self._normalize_text(item[4])
        except Exception:
            return ""

    def _hier_item_object_iri(self, item: Any) -> str:
        if isinstance(item, dict):
            return self._normalize_text(item.get("object_iri"))
        try:
            return self._normalize_text(item[5])
        except Exception:
            return ""

    def _matrix_provenance_links(
        self,
        src_items: Sequence[Dict[str, Any]],
        tgt_items: Sequence[Dict[str, Any]],
        support_matrix: Optional[torch.Tensor],
    ) -> List[Dict[str, Any]]:
        if not src_items or not tgt_items or support_matrix is None:
            return []
        if support_matrix.ndim != 2:
            return []
        if support_matrix.shape[0] != len(src_items) or support_matrix.shape[1] != len(tgt_items):
            return []

        links: Dict[Tuple[str, str], Dict[str, Any]] = {}

        def _upsert(src_idx: int, tgt_idx: int) -> None:
            score = float(support_matrix[src_idx, tgt_idx].item())
            if score <= 0.0:
                return
            source_item_id = self._normalize_text(src_items[src_idx].get("item_id"))
            target_item_id = self._normalize_text(tgt_items[tgt_idx].get("item_id"))
            if not source_item_id or not target_item_id:
                return
            key = (source_item_id, target_item_id)
            payload = {
                "source_item_id": source_item_id,
                "target_item_id": target_item_id,
                "score": score,
            }
            prev = links.get(key)
            if prev is None or score > float(prev.get("score", 0.0)):
                links[key] = payload

        for src_idx in range(support_matrix.shape[0]):
            tgt_idx = int(torch.argmax(support_matrix[src_idx]).item())
            _upsert(src_idx, tgt_idx)
        for tgt_idx in range(support_matrix.shape[1]):
            src_idx = int(torch.argmax(support_matrix[:, tgt_idx]).item())
            _upsert(src_idx, tgt_idx)

        return sorted(
            links.values(),
            key=lambda row: (
                -float(row.get("score", 0.0)),
                self._normalize_text(row.get("source_item_id")),
                self._normalize_text(row.get("target_item_id")),
            ),
        )

    def _build_cross_side_provenance(
        self,
        s_label_value: float,
        hierarchy_payloads: Dict[str, Dict[str, Any]],
        sim_payload: Dict[str, Any],
        diff_payload: Dict[str, Any],
        attr_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "lexical": [
                {
                    "source_ref": "__source__",
                    "target_ref": "__target__",
                    "score": float(s_label_value),
                }
            ],
            "hierarchy": {
                family: list(payload.get("links", []))
                for family, payload in hierarchy_payloads.items()
                if payload.get("links")
            },
            "similarity": list(sim_payload.get("links", [])),
            "attributes": {
                "source": list(attr_payload.get("source_links", [])),
                "target": list(attr_payload.get("target_links", [])),
            },
            "difference": {
                "source": list(diff_payload.get("source_links", [])),
                "target": list(diff_payload.get("target_links", [])),
            },
        }

    def _hydrate_record_item_ids(self, record: Dict[str, Any]) -> Dict[str, Any]:
        hydrated = dict(record or {})
        triple_attributions = dict(hydrated.get("triple_attributions") or {})
        hierarchy = dict(triple_attributions.get("hierarchy") or {})
        for family, family_payload in hierarchy.items():
            family_payload = dict(family_payload or {})
            family_payload["source"] = [
                self._with_item_id("hierarchy", "source", item, family=family)
                for item in list(family_payload.get("source") or [])
            ]
            family_payload["target"] = [
                self._with_item_id("hierarchy", "target", item, family=family)
                for item in list(family_payload.get("target") or [])
            ]
            hierarchy[family] = family_payload
        triple_attributions["hierarchy"] = hierarchy
        for channel in ["similarity", "difference"]:
            payload = dict(triple_attributions.get(channel) or {})
            payload["source"] = [
                self._with_item_id(channel, "source", item)
                for item in list(payload.get("source") or [])
            ]
            payload["target"] = [
                self._with_item_id(channel, "target", item)
                for item in list(payload.get("target") or [])
            ]
            triple_attributions[channel] = payload
        hydrated["triple_attributions"] = triple_attributions
        attributes = dict(hydrated.get("attributes") or {})
        attributes["source"] = [
            self._with_item_id("attribute", "source", item)
            for item in list(attributes.get("source") or [])
        ]
        attributes["target"] = [
            self._with_item_id("attribute", "target", item)
            for item in list(attributes.get("target") or [])
        ]
        hydrated["attributes"] = attributes
        hydrated["explanation_schema_version"] = max(
            3,
            int(hydrated.get("explanation_schema_version", 0) or 0),
        )
        return hydrated

    def _explanation_family_names_from_record(self, record: Dict[str, Any]) -> List[str]:
        family_names = list(self.hierarchical_relation_families.keys() or [])
        if "is_a" not in family_names:
            family_names = ["is_a"] + family_names
        hierarchy_payload = (record.get("triple_attributions") or {}).get("hierarchy") or {}
        for family in hierarchy_payload.keys():
            if family not in family_names:
                family_names.append(family)
        return family_names

    def _saved_record_hierarchy_items(
        self,
        record: Dict[str, Any],
        family: str,
        side: str,
    ) -> List[Dict[str, Any]]:
        hierarchy = (record.get("triple_attributions") or {}).get("hierarchy") or {}
        payload = dict(hierarchy.get(family) or {})
        out: List[Dict[str, Any]] = []
        for item in list(payload.get(side) or []):
            triple = list(item.get("triple") or [])
            if len(triple) < 3:
                continue
            out.append(
                {
                    "triple": (
                        self._normalize_text(triple[0]),
                        self._normalize_text(triple[1]),
                        self._normalize_text(triple[2]),
                    ),
                    "specificity": float(item.get("specificity", 0.0)),
                    "subject_iri": self._normalize_text(item.get("subject_iri")),
                    "object_iri": self._normalize_text(item.get("object_iri")),
                }
            )
        return out

    def _saved_record_object_items(
        self,
        record: Dict[str, Any],
        channel: str,
        side: str,
    ) -> List[Dict[str, Any]]:
        payload = (record.get("triple_attributions") or {}).get(channel) or {}
        out: List[Dict[str, Any]] = []
        for item in list(payload.get(side) or []):
            triple = list(item.get("triple") or [])
            if len(triple) < 3:
                continue
            out.append(
                {
                    "triple": (
                        self._normalize_text(triple[0]),
                        self._normalize_text(triple[1]),
                        self._normalize_text(triple[2]),
                    ),
                    "score": float(item.get("edge_ic", item.get("unsupported_mass", 0.0))),
                    "subject_iri": self._normalize_text(item.get("subject_iri")),
                    "object_iri": self._normalize_text(item.get("object_iri")),
                    "rel_iri": self._normalize_text(item.get("rel_iri")),
                }
            )
        return out

    def _saved_record_attribute_items(
        self,
        record: Dict[str, Any],
        side: str,
    ) -> List[Dict[str, Any]]:
        attributes = (record.get("attributes") or {}).get(side) or []
        out: List[Dict[str, Any]] = []
        for item in list(attributes):
            out.append(
                {
                    "prop": self._normalize_text(item.get("property", item.get("prop"))),
                    "value": self._normalize_text(item.get("value")),
                    "text": self._normalize_text(item.get("text")),
                    "weight": float(item.get("weight", 0.0)),
                    "entity_iri": self._normalize_text(item.get("entity_iri")),
                }
            )
        return out

    def reconstruct_explanation_fields_from_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        hydrated = self._hydrate_record_item_ids(record)
        labels = dict(hydrated.get("selected_labels") or {})
        src_label = self._normalize_text(labels.get("source", hydrated.get("src_iri")))
        tgt_label = self._normalize_text(labels.get("target", hydrated.get("tgt_iri")))
        family_names = self._explanation_family_names_from_record(hydrated)
        hierarchy_payloads: Dict[str, Dict[str, Any]] = {}
        for family in family_names:
            hierarchy_payloads[family] = self._score_hierarchy_family(
                family,
                self._saved_record_hierarchy_items(hydrated, family, "source"),
                self._saved_record_hierarchy_items(hydrated, family, "target"),
            )
        sim_payload = self._score_similarity_channel(
            self._saved_record_object_items(hydrated, "similarity", "source"),
            self._saved_record_object_items(hydrated, "similarity", "target"),
        )
        diff_payload = self._score_difference_channel(
            self._saved_record_object_items(hydrated, "difference", "source"),
            self._saved_record_object_items(hydrated, "difference", "target"),
            support_mat=sim_payload.get("support_matrix"),
        )
        attr_payload = self._score_attribute_channel(
            self._saved_record_attribute_items(hydrated, "source"),
            self._saved_record_attribute_items(hydrated, "target"),
            [src_label] if src_label else [],
            [tgt_label] if tgt_label else [],
            hierarchy_payloads,
            sim_payload,
        )
        s_label_value = float((hydrated.get("confidences") or {}).get("s_label", self.tau))
        hydrated["cross_side_provenance"] = self._build_cross_side_provenance(
            s_label_value,
            hierarchy_payloads,
            sim_payload,
            diff_payload,
            attr_payload,
        )
        return hydrated

    def reconstruct_explanation_fields_for_pair(
        self,
        src_iri: str,
        tgt_iri: str,
        src_labels: Optional[List[str]] = None,
        tgt_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        dataset = self._attached_dataset
        if dataset is None:
            raise RuntimeError("PairAdaptiveSemanticScorer requires an attached dataset.")
        src_feats = dataset.get_entity_features(src_iri, "src")
        tgt_feats = dataset.get_entity_features(tgt_iri, "tgt")
        src_label_list = list(src_labels or src_feats.get("labels") or [src_iri])
        tgt_label_list = list(tgt_labels or tgt_feats.get("labels") or [tgt_iri])
        out = self.forward(
            src_iris=[src_iri],
            tgt_iris=[tgt_iri],
            src_label_lists=[src_label_list],
            tgt_label_lists=[tgt_label_list],
            label=None,
        )
        explanations = list(out.get("explanations") or [])
        if not explanations:
            raise RuntimeError(f"Failed to rehydrate explanation for pair ({src_iri}, {tgt_iri}).")
        return dict(explanations[0])

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
        enc = torch.nn.functional.normalize(
            self.encode_contexts_batch([src_text, tgt_text]), dim=-1
        )
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
        items: Sequence[Any],
    ) -> List[str]:
        template = self._hierarchy_template(family)
        out: List[str] = []
        for item in items:
            head, _, tail = self._hier_item_triple(item)
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
