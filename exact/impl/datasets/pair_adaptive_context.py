from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import time

from org.semanticweb.owlapi.model import IRI
from org.semanticweb.owlapi.search import EntitySearcher

from exact.impl.datasets.contextgraph import ContextDataset
from exact.core.entities.ontology import OntologyGraph


class PairAdaptiveContextDataset(ContextDataset):
    """
    Pair-adaptive dataset that keeps per-entity caches but does not materialize
    pair-specific contexts into the dataframe. The scorer pulls cached ontology
    features by entity IRI and computes the pair-specific channels on the fly.
    """

    def __init__(
        self,
        projection_include_literals: bool = False,
        hierarchical_relation_families: Optional[Dict[str, Dict[str, Any]]] = None,
        hierarchy_max_depth: int = 2,
        max_hierarchy_triples_per_family: int = 6,
        max_object_triples: int = 48,
        max_diff_triples: int = 24,
        max_attr_items: int = 12,
        pair_adaptive_feature_log_every: int = 1000,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.projection_include_literals = bool(projection_include_literals)
        self.hierarchical_relation_families = hierarchical_relation_families or {}
        self.hierarchy_max_depth = int(hierarchy_max_depth)
        self.max_hierarchy_triples_per_family = int(max_hierarchy_triples_per_family)
        self.max_object_triples = int(max_object_triples)
        self.max_diff_triples = int(max_diff_triples)
        self.max_attr_items = int(max_attr_items)
        self.pair_adaptive_feature_log_every = max(1, int(pair_adaptive_feature_log_every))
        self._entity_feature_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._direct_superclass_cache: Dict[Tuple[str, str], List[str]] = {}
        self._hierarchy_axiom_targets_cache: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        self._relation_family_cache: Dict[str, Optional[str]] = {}
        self._normalized_relation_families: Dict[str, Dict[str, set[str]]] = {}
        for family, cfg in self.hierarchical_relation_families.items():
            self._normalized_relation_families[family] = {
                "aliases": {
                    OntologyGraph.normalize_label(x)
                    for x in (cfg.get("iri_aliases") or [])
                    if OntologyGraph.normalize_label(x)
                },
                "seeds": {
                    OntologyGraph.normalize_label(x)
                    for x in (cfg.get("label_seeds") or [])
                    if OntologyGraph.normalize_label(x)
                },
            }

    def emit_feature_metrics_on_build(self) -> bool:
        return True

    def _cache_fingerprint_payload(self) -> Dict[str, Any]:
        payload = super()._cache_fingerprint_payload()
        payload.update(
            {
                "projection_include_literals": self.projection_include_literals,
                "hierarchical_relation_families": self.hierarchical_relation_families,
                "hierarchy_max_depth": self.hierarchy_max_depth,
                "max_hierarchy_triples_per_family": self.max_hierarchy_triples_per_family,
                "max_object_triples": self.max_object_triples,
                "max_diff_triples": self.max_diff_triples,
                "max_attr_items": self.max_attr_items,
            }
        )
        return payload

    @property
    def source_graph(self) -> OntologyGraph:
        if self._source_graph is None:
            self.log("Loading source graph for pair-adaptive dataset…", level="info")
            self._source_graph = OntologyGraph(
                self.source.ontology,
                self.source_reasoner,
                only_taxonomy=False,
                include_literals=self.projection_include_literals,
            )
        return self._source_graph

    @property
    def target_graph(self) -> OntologyGraph:
        if self._target_graph is None:
            self.log("Loading target graph for pair-adaptive dataset…", level="info")
            self._target_graph = OntologyGraph(
                self.target.ontology,
                self.target_reasoner,
                only_taxonomy=False,
                include_literals=self.projection_include_literals,
            )
        return self._target_graph

    def _normalize_relation_key(self, text: str) -> str:
        return OntologyGraph.normalize_label(text or "")

    def _relation_family(self, graph: OntologyGraph, rel_iri: str) -> Optional[str]:
        cached = self._relation_family_cache.get(rel_iri)
        if rel_iri in self._relation_family_cache:
            return cached
        rel_key = self._normalize_relation_key(rel_iri)
        rel_label = self._normalize_relation_key(graph.get_labels(rel_iri)[0] if rel_iri else "")
        for family, cfg in self._normalized_relation_families.items():
            if rel_key in cfg["aliases"] or rel_label in cfg["seeds"]:
                self._relation_family_cache[rel_iri] = family
                return family
        self._relation_family_cache[rel_iri] = None
        return None

    def _direct_superclass_iris(self, iri: str, graph: OntologyGraph, side: str) -> List[str]:
        cache_key = (side, iri)
        cached = self._direct_superclass_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        reasoner = self.source_reasoner if side == "src" else self.target_reasoner
        ontology = self.source.ontology if side == "src" else self.target.ontology
        factory = ontology.getOWLOntologyManager().getOWLDataFactory()
        owl_class = factory.getOWLClass(IRI.create(iri))
        out: List[str] = []
        try:
            superset = reasoner.getSuperClasses(owl_class, True).getFlattened()
            for sup in superset:
                if not sup.isOWLThing():
                    out.append(str(sup.getIRI().toString()))
        except Exception:
            pass
        self._direct_superclass_cache[cache_key] = list(out)
        return out

    def _ontology_for_side(self, side: str):
        return self.source.ontology if side == "src" else self.target.ontology

    @staticmethod
    def _iter_java_items(value: Any) -> Iterable[Any]:
        if value is None:
            return []
        try:
            iterator = value.iterator()
            out: List[Any] = []
            while iterator.hasNext():
                out.append(iterator.next())
            return out
        except Exception:
            try:
                return list(value)
            except Exception:
                return []

    @staticmethod
    def _named_class_iri(expr: Any) -> Optional[str]:
        if expr is None:
            return None
        try:
            if expr.isOWLClass():
                return str(expr.asOWLClass().getIRI().toString())
        except Exception:
            pass
        try:
            owl_class = expr.asOWLClass()
            return str(owl_class.getIRI().toString())
        except Exception:
            return None

    @staticmethod
    def _property_iri(prop_expr: Any) -> Optional[str]:
        if prop_expr is None:
            return None
        try:
            named = prop_expr.getNamedProperty()
            return str(named.getIRI().toString())
        except Exception:
            pass
        try:
            if not prop_expr.isAnonymous():
                return str(prop_expr.asOWLObjectProperty().getIRI().toString())
        except Exception:
            pass
        try:
            return str(prop_expr.toStringID())
        except Exception:
            return None

    def _iter_hierarchy_targets(
        self,
        expr: Any,
        graph: OntologyGraph,
    ) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        named_iri = self._named_class_iri(expr)
        if named_iri:
            out.append(("is_a", named_iri))
            return out

        expr_type = ""
        try:
            expr_type = str(expr.getClassExpressionType().toString())
        except Exception:
            expr_type = ""

        if "Intersection" in expr_type or hasattr(expr, "getOperandsAsList") or hasattr(expr, "getOperands"):
            operands = []
            try:
                operands = self._iter_java_items(expr.getOperandsAsList())
            except Exception:
                operands = self._iter_java_items(getattr(expr, "getOperands", lambda: [])())
            for operand in operands:
                out.extend(self._iter_hierarchy_targets(operand, graph))

        if hasattr(expr, "getProperty") and hasattr(expr, "getFiller"):
            prop_iri = self._property_iri(expr.getProperty())
            filler_iri = self._named_class_iri(expr.getFiller())
            family = self._relation_family(graph, prop_iri or "")
            if family and filler_iri:
                out.append((family, filler_iri))
            else:
                out.extend(self._iter_hierarchy_targets(expr.getFiller(), graph))

        return out

    def _hierarchy_axiom_targets(
        self,
        iri: str,
        graph: OntologyGraph,
        side: str,
    ) -> List[Tuple[str, str]]:
        cache_key = (side, iri)
        cached = self._hierarchy_axiom_targets_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        ontology = self._ontology_for_side(side)
        factory = ontology.getOWLOntologyManager().getOWLDataFactory()
        owl_class = factory.getOWLClass(IRI.create(iri))
        out: List[Tuple[str, str]] = []

        try:
            for axiom in self._iter_java_items(ontology.getSubClassAxiomsForSubClass(owl_class)):
                try:
                    out.extend(self._iter_hierarchy_targets(axiom.getSuperClass(), graph))
                except Exception:
                    continue
        except Exception:
            pass

        try:
            for axiom in self._iter_java_items(ontology.getEquivalentClassesAxioms(owl_class)):
                expressions = []
                try:
                    expressions = self._iter_java_items(axiom.getClassExpressionsAsList())
                except Exception:
                    expressions = self._iter_java_items(axiom.getClassExpressions())
                for expr in expressions:
                    named = self._named_class_iri(expr)
                    if named and named == iri:
                        continue
                    out.extend(self._iter_hierarchy_targets(expr, graph))
        except Exception:
            pass

        deduped: List[Tuple[str, str]] = []
        seen = set()
        for family, target_iri in out:
            key = (family, target_iri)
            if key in seen or not target_iri:
                continue
            seen.add(key)
            deduped.append(key)
        self._hierarchy_axiom_targets_cache[cache_key] = list(deduped)
        return deduped

    def _hierarchy_bundle(self, iri: str, graph: OntologyGraph, side: str) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        frontier = [(iri, 0)]
        seen = {iri}
        while frontier:
            node, depth = frontier.pop(0)
            if depth >= self.hierarchy_max_depth:
                continue
            for sup in self._direct_superclass_iris(node, graph, side):
                if sup in seen:
                    continue
                seen.add(sup)
                frontier.append((sup, depth + 1))
                spec = 1.0 / float(depth + 1)
                out["is_a"].append(
                    {
                        "triple": (graph.get_labels(node)[0], "is_a", graph.get_labels(sup)[0]),
                        "specificity": spec,
                        "subject_iri": node,
                        "object_iri": sup,
                    }
                )

        family_frontier = [(iri, 0)]
        family_seen = {(family, iri) for family in self.hierarchical_relation_families if family != "is_a"}
        while family_frontier:
            node, depth = family_frontier.pop(0)
            if depth >= self.hierarchy_max_depth:
                continue
            for family, target_iri in self._hierarchy_axiom_targets(node, graph, side):
                if family == "is_a":
                    continue
                key = (family, target_iri)
                if key in family_seen:
                    continue
                family_seen.add(key)
                family_frontier.append((target_iri, depth + 1))
                spec = 1.0 / float(depth + 1)
                out[family].append(
                    {
                        "triple": (graph.get_labels(node)[0], family, graph.get_labels(target_iri)[0]),
                        "specificity": spec,
                        "subject_iri": node,
                        "object_iri": target_iri,
                    }
                )

        for family, triples in list(out.items()):
            triples.sort(key=lambda item: float(item.get("specificity", 0.0)), reverse=True)
            out[family] = triples[: self.max_hierarchy_triples_per_family]
        return dict(out)

    def _hierarchy_family_names(self) -> List[str]:
        family_names = list(self.hierarchical_relation_families.keys() or [])
        if "is_a" not in family_names:
            family_names = ["is_a"] + family_names
        return family_names

    def _object_bundle(self, iri: str, graph: OntologyGraph) -> List[Dict[str, Any]]:
        triples = []
        raw = graph.get_raw_neighborhood(iri, self.n_hops)
        for src, rel, dst in raw:
            if self._relation_family(graph, rel) is not None:
                continue
            head = graph.get_labels(src)[0]
            rel_label = graph.get_labels(rel)[0]
            tail = graph.get_labels(dst)[0]
            if graph._looks_like_literal_or_blank(dst):
                continue
            triples.append(
                {
                    "triple": (head, rel_label, tail),
                    "rel_iri": rel,
                    "score": graph.edge_ic_norm((src, rel, dst)),
                    "subject_iri": src,
                    "object_iri": dst,
                }
            )
        triples.sort(key=lambda item: item["score"], reverse=True)
        return triples[: self.max_object_triples]

    def _annotation_bundle(self, iri: str, graph: OntologyGraph, side: str) -> List[Dict[str, Any]]:
        ontology = self.source.ontology if side == "src" else self.target.ontology
        factory = ontology.getOWLOntologyManager().getOWLDataFactory()
        owl_class = factory.getOWLClass(IRI.create(iri))
        label_prop = str(factory.getRDFSLabel().getIRI().toString())
        items: List[Dict[str, Any]] = []
        seen = set()
        try:
            annotations = EntitySearcher.getAnnotations(owl_class, ontology)
            iterator = annotations.iterator()
            while iterator.hasNext():
                ann = iterator.next()
                prop_iri = str(ann.getProperty().getIRI().toString())
                if prop_iri == label_prop:
                    continue
                value = ann.getValue()
                if not value.isLiteral():
                    continue
                literal = str(value.asLiteral().get().getLiteral()).strip()
                if not literal:
                    continue
                prop_label = graph.get_labels(prop_iri)[0]
                dedupe_key = (prop_label, literal)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                items.append(
                    {
                        "prop": prop_label,
                        "value": literal,
                        "text": f"{prop_label}: {literal}",
                        "weight": min(1.0, max(0.1, len(literal.split()) / 12.0)),
                        "entity_iri": iri,
                    }
                )
        except Exception:
            pass

        if self.projection_include_literals:
            for src, rel, dst in graph.get_raw_neighborhood(iri, self.n_hops):
                if self._relation_family(graph, rel) is not None:
                    continue
                literal_node = None
                if src == iri and graph._looks_like_literal_or_blank(dst):
                    literal_node = dst
                elif dst == iri and graph._looks_like_literal_or_blank(src):
                    literal_node = src
                if literal_node is None:
                    continue
                literal = graph.get_labels(literal_node)[0].strip()
                if not literal:
                    continue
                prop_label = graph.get_labels(rel)[0]
                dedupe_key = (prop_label, literal)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                items.append(
                    {
                        "prop": prop_label,
                        "value": literal,
                        "text": f"{prop_label}: {literal}",
                        "weight": min(1.0, max(0.1, len(literal.split()) / 12.0)),
                        "entity_iri": iri,
                    }
                )
        items.sort(key=lambda item: item["weight"], reverse=True)
        return items[: self.max_attr_items]

    def get_entity_features(self, iri: str, side: str) -> Dict[str, Any]:
        key = (side, iri)
        cached = self._entity_feature_cache.get(key)
        if cached is not None:
            return cached
        graph = self.source_graph if side == "src" else self.target_graph
        feats = {
            "labels": graph.get_labels(iri)[:] if self.all_labels else [graph.get_labels(iri)[0]],
            "hierarchy": self._hierarchy_bundle(iri, graph, side),
            "object_triples": self._object_bundle(iri, graph),
            "attributes": self._annotation_bundle(iri, graph, side),
        }
        self._entity_feature_cache[key] = feats
        return feats

    def has_entity_features_cached(self, iri: str, side: str) -> bool:
        return (side, iri) in self._entity_feature_cache

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / max(1, len(values)))

    def _entity_pool_metrics(self, feats: Dict[str, Any]) -> Dict[str, float]:
        hierarchy = feats.get("hierarchy", {}) or {}
        object_items = list(feats.get("object_triples", []) or [])
        attributes = list(feats.get("attributes", []) or [])

        family_counts = {
            family: len(hierarchy.get(family, []) or [])
            for family in self._hierarchy_family_names()
        }
        hierarchy_total = sum(family_counts.values())

        object_scores = [float(item.get("score", 0.0)) for item in object_items]
        object_relations = {
            str((item.get("triple") or ["", "", ""])[1] or "").strip()
            for item in object_items
            if item.get("triple")
        }
        attr_weights = [float(item.get("weight", 0.0)) for item in attributes]

        metrics: Dict[str, float] = {
            "hier_total_count": float(hierarchy_total),
            "hier_is_empty": float(hierarchy_total == 0),
            "obj_count": float(len(object_items)),
            "obj_relation_count": float(len(object_relations)),
            "obj_ic_mean": self._safe_mean(object_scores),
            "obj_ic_max": max(object_scores, default=0.0),
            "obj_is_empty": float(len(object_items) == 0),
            "attr_count": float(len(attributes)),
            "attr_weight_mean": self._safe_mean(attr_weights),
            "attr_is_empty": float(len(attributes) == 0),
        }
        for family, count in family_counts.items():
            metrics[f"hier_{family}_count"] = float(count)
        return metrics

    def _pair_adaptive_metric_columns(self) -> List[str]:
        cols = [
            "src_hier_total_count", "tgt_hier_total_count",
            "src_hier_is_empty", "tgt_hier_is_empty",
            "src_obj_count", "tgt_obj_count",
            "src_obj_relation_count", "tgt_obj_relation_count",
            "src_obj_ic_mean", "tgt_obj_ic_mean",
            "src_obj_ic_max", "tgt_obj_ic_max",
            "src_obj_is_empty", "tgt_obj_is_empty",
            "src_attr_count", "tgt_attr_count",
            "src_attr_weight_mean", "tgt_attr_weight_mean",
            "src_attr_is_empty", "tgt_attr_is_empty",
        ]
        for family in self._hierarchy_family_names():
            cols.extend([
                f"src_hier_{family}_count",
                f"tgt_hier_{family}_count",
            ])
        return cols

    def get_features(self, df) -> "DataFrame":
        self.log("Generating labels for pair-adaptive dataset…", level="info")
        src_iris: List[str] = df["Src"].tolist()
        tgt_iris: List[str] = df["Tgt"].tolist()
        usrc = list(dict.fromkeys(src_iris))
        utgt = list(dict.fromkeys(tgt_iris))

        build_started = time.time()

        def _build_feature_map(iris: List[str], side: str) -> Dict[str, Dict[str, Any]]:
            total = len(iris)
            progress_every = max(1, int(self.pair_adaptive_feature_log_every))
            label = "source" if side == "src" else "target"
            feat_map: Dict[str, Dict[str, Any]] = {}
            side_started = time.time()
            self.log(
                f"Pair-adaptive feature build starting for {label}: {total} unique entities",
                level="info",
            )
            for idx, iri in enumerate(iris, start=1):
                feat_map[iri] = self.get_entity_features(iri, side)
                if idx == total or idx % progress_every == 0:
                    elapsed = time.time() - side_started
                    rate = idx / elapsed if elapsed > 1e-8 else 0.0
                    remaining = max(0, total - idx)
                    eta = (remaining / rate) if rate > 1e-8 else 0.0
                    self.log(
                        f"Pair-adaptive feature build {label}: {idx}/{total} entities "
                        f"({(100.0 * idx / max(1, total)):.1f}%) in {elapsed:.1f}s, ETA {eta:.1f}s",
                        level="debug",
                    )
            return feat_map

        src_feat_map = _build_feature_map(usrc, "src")
        tgt_feat_map = _build_feature_map(utgt, "tgt")
        src_lab_map = {iri: src_feat_map[iri]["labels"] for iri in usrc}
        tgt_lab_map = {iri: tgt_feat_map[iri]["labels"] for iri in utgt}
        src_pool_metrics_map = {
            iri: self._entity_pool_metrics(src_feat_map[iri])
            for iri in usrc
        }
        tgt_pool_metrics_map = {
            iri: self._entity_pool_metrics(tgt_feat_map[iri])
            for iri in utgt
        }

        df = df.copy()
        df["SrcLabels"] = [src_lab_map[iri] for iri in src_iris]
        df["TgtLabels"] = [tgt_lab_map[iri] for iri in tgt_iris]
        df["SrcCtx"] = [[] for _ in src_iris]
        df["TgtCtx"] = [[] for _ in tgt_iris]
        df["SrcCtxRaw"] = [[] for _ in src_iris]
        df["TgtCtxRaw"] = [[] for _ in tgt_iris]
        df["SrcCtxBridge"] = [[] for _ in src_iris]
        df["TgtCtxBridge"] = [[] for _ in tgt_iris]
        df["Features"] = [
            [src_labels, [], tgt_labels, []]
            for src_labels, tgt_labels in zip(df["SrcLabels"], df["TgtLabels"])
        ]

        src_label_metrics_map = {
            iri: self._compute_metrics_for_label_list(src_lab_map[iri])
            for iri in usrc
        }
        tgt_label_metrics_map = {
            iri: self._compute_metrics_for_label_list(tgt_lab_map[iri])
            for iri in utgt
        }
        for key in ["n_labels", "char_len", "word_len", "max_label_words", "avg_label_words", "is_empty"]:
            df[f"src_lab_{key}"] = [src_label_metrics_map[iri][key] for iri in src_iris]
            df[f"tgt_lab_{key}"] = [tgt_label_metrics_map[iri][key] for iri in tgt_iris]
        for key in ["n_triples", "char_len", "word_len", "tok_len"]:
            df[f"src_ctx_{key}"] = 0.0
            df[f"tgt_ctx_{key}"] = 0.0
        df["src_ctx_is_empty"] = 1.0
        df["tgt_ctx_is_empty"] = 1.0
        for key in sorted(src_pool_metrics_map[usrc[0]].keys()) if usrc else []:
            df[f"src_{key}"] = [src_pool_metrics_map[iri][key] for iri in src_iris]
        for key in sorted(tgt_pool_metrics_map[utgt[0]].keys()) if utgt else []:
            df[f"tgt_{key}"] = [tgt_pool_metrics_map[iri][key] for iri in tgt_iris]
        self.log(
            f"Pair-adaptive feature generation finished in {time.time() - build_started:.2f}s",
            level="info",
        )
        return df

    def save_feature_metrics(self, df: Optional["DataFrame"] = None, filename: str = "feature_metrics.csv"):
        if df is None:
            df = self.dataframe
        cols = [
            "Src", "Tgt",
            "cand_sim", "cand_sim_src_mean", "cand_sim_prob",
            "cand_share_top", "cand_share_rest", "cand_share_log_ratio",
            "src_lab_n_labels", "src_lab_char_len", "src_lab_word_len", "src_lab_max_label_words", "src_lab_avg_label_words", "src_lab_is_empty",
            "tgt_lab_n_labels", "tgt_lab_char_len", "tgt_lab_word_len", "tgt_lab_max_label_words", "tgt_lab_avg_label_words", "tgt_lab_is_empty",
        ]
        cols.extend(self._pair_adaptive_metric_columns())
        cols = [col for col in cols if col in df.columns]
        out_path = (self.output_path / filename).resolve()
        df[cols].to_csv(out_path, index=False)
        self.log(f"Saved pair-adaptive feature metrics CSV to {out_path}", level="info")
        return out_path

    def supported_plot_metrics(self) -> List[str]:
        metrics = [
            "src_lab_n_labels", "tgt_lab_n_labels",
            "src_lab_word_len", "tgt_lab_word_len",
            "src_lab_char_len", "tgt_lab_char_len",
            "src_lab_max_label_words", "tgt_lab_max_label_words",
            "src_lab_avg_label_words", "tgt_lab_avg_label_words",
            "cand_sim", "cand_sim_src_mean", "cand_sim_prob",
            "cand_share_top", "cand_share_rest", "cand_share_log_ratio",
        ]
        metrics.extend(self._pair_adaptive_metric_columns())
        return metrics

    def default_plot_metrics(self) -> List[str]:
        metrics = [
            "src_lab_n_labels", "tgt_lab_n_labels",
            "src_lab_word_len", "tgt_lab_word_len",
            "src_hier_total_count", "tgt_hier_total_count",
            "src_obj_count", "tgt_obj_count",
            "src_obj_relation_count", "tgt_obj_relation_count",
            "src_obj_ic_mean", "tgt_obj_ic_mean",
            "src_attr_count", "tgt_attr_count",
            "src_attr_weight_mean", "tgt_attr_weight_mean",
            "cand_sim", "cand_sim_src_mean",
        ]
        for family in self._hierarchy_family_names():
            metrics.extend([
                f"src_hier_{family}_count",
                f"tgt_hier_{family}_count",
            ])
        return metrics

    def plot_feature_distributions(
        self,
        which: Optional[List[str]] = None,
        bins: int = 30,
        kde: bool = False,
        dpi: int = 300,
        alpha: float = 0.6,
        **kwargs,
    ) -> None:
        df = self.dataframe
        selected = self._resolve_plot_metrics(which, df)
        if not selected:
            self.log("No pair-adaptive dataset plot metrics available; skipping feature plots.", level="warning")
            return
        plot_dir = (self.plot_dir / "features").resolve()
        plot_dir.mkdir(parents=True, exist_ok=True)

        for col in selected:
            series = df[col].dropna()
            if series.empty:
                self.log(f"Skipping empty pair-adaptive metric '{col}'.", level="info")
                continue
            if int(series.nunique(dropna=True)) <= 1:
                self.log(f"Skipping non-informative pair-adaptive plot '{col}' (constant values).", level="info")
                continue
            plt.figure(figsize=(7, 5))
            sns.histplot(series, bins=bins, kde=kde, stat="probability", alpha=alpha)
            plt.title(col.replace("_", " ").title())
            plt.xlabel(col)
            plt.ylabel("Probability")
            out = plot_dir / f"{col}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=dpi)
            plt.close()
            self.log(f"Saved plot: {out}", level="debug")

    def log_sanity_examples(self, n: int = 6, max_ctx_show: int = 3, max_label_show: int = 5, **kwargs) -> None:
        df = self.dataframe
        if df is None or df.empty:
            self.log("Dataset empty; no pair-adaptive sanity examples.", level="warning")
            return

        problematic = df[
            (df.get("src_hier_is_empty", 0) == 1)
            | (df.get("tgt_hier_is_empty", 0) == 1)
            | (df.get("src_obj_is_empty", 0) == 1)
            | (df.get("tgt_obj_is_empty", 0) == 1)
            | (df.get("src_attr_is_empty", 0) == 1)
            | (df.get("tgt_attr_is_empty", 0) == 1)
            | (df.get("src_lab_is_empty", 0) == 1)
            | (df.get("tgt_lab_is_empty", 0) == 1)
        ]
        if len(problematic) < n:
            rest = df.drop(problematic.index)
            if not rest.empty:
                extra = rest.sample(min(n - len(problematic), len(rest)), random_state=self.request_seed)
                problematic = pd.concat([problematic, extra])
        show = problematic.head(n)
        family_names = self._hierarchy_family_names()

        def _show_lines(tag: str, lines: List[str], limit: int) -> None:
            if not lines:
                self.log(f"  {tag}: <EMPTY>", level="warning")
                return
            for idx, line in enumerate(lines[:limit]):
                self.log(f"  {tag}[{idx}]: {line!r}", level="debug")
            if len(lines) > limit:
                self.log(f"  {tag}: … (+{len(lines) - limit} more)", level="debug")

        self.log(f"### Pair-adaptive sanity examples (n={len(show)})", level="info")
        for row_idx, row in show.iterrows():
            src_iri = row["Src"]
            tgt_iri = row["Tgt"]
            src_feats = self.get_entity_features(src_iri, "src")
            tgt_feats = self.get_entity_features(tgt_iri, "tgt")
            self.log(f"- Pair {row_idx}: Src={src_iri} | Tgt={tgt_iri}", level="info")

            _show_lines("SRC labels", list(src_feats.get("labels", [])), max_label_show)
            _show_lines("TGT labels", list(tgt_feats.get("labels", [])), max_label_show)

            src_family_counts = {
                family: int(row.get(f"src_hier_{family}_count", 0))
                for family in family_names
            }
            tgt_family_counts = {
                family: int(row.get(f"tgt_hier_{family}_count", 0))
                for family in family_names
            }
            self.log(f"  SRC hierarchy counts: {src_family_counts}", level="info")
            self.log(f"  TGT hierarchy counts: {tgt_family_counts}", level="info")

            for family in family_names:
                src_hier = list(src_feats.get("hierarchy", {}).get(family, []) or [])
                tgt_hier = list(tgt_feats.get("hierarchy", {}).get(family, []) or [])
                if src_hier:
                    _show_lines(
                        f"SRC hierarchy[{family}]",
                        [
                            f"{item['triple'][0]} --{item['triple'][1]}--> {item['triple'][2]} "
                            f"(spec={float(item.get('specificity', 0.0)):.2f})"
                            for item in src_hier
                        ],
                        max_ctx_show,
                    )
                if tgt_hier:
                    _show_lines(
                        f"TGT hierarchy[{family}]",
                        [
                            f"{item['triple'][0]} --{item['triple'][1]}--> {item['triple'][2]} "
                            f"(spec={float(item.get('specificity', 0.0)):.2f})"
                            for item in tgt_hier
                        ],
                        max_ctx_show,
                    )

            _show_lines(
                "SRC object triples",
                [
                    f"{item['triple'][0]} --{item['triple'][1]}--> {item['triple'][2]} (ic={float(item.get('score', 0.0)):.3f})"
                    for item in list(src_feats.get("object_triples", []) or [])
                ],
                max_ctx_show,
            )
            _show_lines(
                "TGT object triples",
                [
                    f"{item['triple'][0]} --{item['triple'][1]}--> {item['triple'][2]} (ic={float(item.get('score', 0.0)):.3f})"
                    for item in list(tgt_feats.get("object_triples", []) or [])
                ],
                max_ctx_show,
            )
            _show_lines(
                "SRC attributes",
                [str(item.get("text", "")) for item in list(src_feats.get("attributes", []) or [])],
                max_ctx_show,
            )
            _show_lines(
                "TGT attributes",
                [str(item.get("text", "")) for item in list(tgt_feats.get("attributes", []) or [])],
                max_ctx_show,
            )

            metrics = {
                "cand_sim": row.get("cand_sim"),
                "src_lab_n_labels": row.get("src_lab_n_labels"),
                "tgt_lab_n_labels": row.get("tgt_lab_n_labels"),
                "src_hier_total_count": row.get("src_hier_total_count"),
                "tgt_hier_total_count": row.get("tgt_hier_total_count"),
                "src_obj_count": row.get("src_obj_count"),
                "tgt_obj_count": row.get("tgt_obj_count"),
                "src_attr_count": row.get("src_attr_count"),
                "tgt_attr_count": row.get("tgt_attr_count"),
            }
            self.log(f"  Metrics: {metrics}", level="info")
