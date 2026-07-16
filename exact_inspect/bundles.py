from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

from exact.ontology import load_ontology
from exact.runs import RunReader

from .helpers import (
    attribute_items,
    attribute_node_display,
    categorize_bridge_relevance,
    channel_items,
    edge_display_level,
    hierarchy_items,
    mean_nonempty,
    ordered_path_nodes,
    safe_float,
    safe_text,
)

DEFINITION_ATTRIBUTE_PROPERTIES = {
    "definition",
    "description",
    "comment",
    "editor note",
    "definition citation",
}


def _first_value(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return value
    return None


def _record_pair(record: Mapping[str, Any]) -> Tuple[str, str]:
    return (
        safe_text(_first_value(record, ("src_iri", "Src", "SrcEntity", "source_iri", "source"))),
        safe_text(_first_value(record, ("tgt_iri", "Tgt", "TgtEntity", "target_iri", "target"))),
    )


def _record_score(record: Mapping[str, Any]) -> float:
    confidences = dict(record.get("confidences") or {})
    return safe_float(
        (
            _first_value(
                record,
                ("S_final", "S_select", "Score", "score"),
            )
            if any(record.get(key) is not None for key in ("S_final", "S_select", "Score", "score"))
            else _first_value(confidences, ("S_final", "S_select", "S_base", "score"))
        ),
        0.0,
    )


def _normalise_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(record)
    src, tgt = _record_pair(payload)
    payload["src_iri"] = src
    payload["tgt_iri"] = tgt
    labels = dict(payload.get("selected_labels") or {})
    labels.setdefault("source", safe_text(_first_value(payload, ("src_label", "SrcLabel"))) or src)
    labels.setdefault("target", safe_text(_first_value(payload, ("tgt_label", "TgtLabel"))) or tgt)
    payload["selected_labels"] = labels
    payload.setdefault("confidences", {"S_final": _record_score(payload)})
    payload.setdefault("prediction", {})
    return payload


def _open_mode_payload(reader: RunReader) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build the historical UI payload exclusively through :class:`RunReader`."""

    mapping_rows = reader.mappings("local").to_dict(orient="records")
    mapping_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for raw in mapping_rows:
        src = safe_text(_first_value(raw, ("SrcEntity", "Src", "src_iri", "source")))
        tgt = safe_text(_first_value(raw, ("TgtEntity", "Tgt", "tgt_iri", "target")))
        if src and tgt:
            mapping_by_pair[(src, tgt)] = dict(raw)

    records: List[Dict[str, Any]] = []
    indexed_pairs: set[Tuple[str, str]] = set()
    for raw in reader.iter_explanations():
        record = _normalise_record(raw)
        pair = _record_pair(record)
        if not all(pair):
            continue
        indexed_pairs.add(pair)
        prediction = dict(record.get("prediction") or {})
        prediction.setdefault("global_match", pair in mapping_by_pair)
        record["prediction"] = prediction
        records.append(record)

    # A run may contain a mapping without an explanation (for example an older
    # export interrupted before explanation generation). Keep it inspectable.
    for pair, row in mapping_by_pair.items():
        if pair in indexed_pairs:
            continue
        score = safe_float(_first_value(row, ("Score", "S_final", "S_select", "score")), 0.0)
        records.append(
            {
                "src_iri": pair[0],
                "tgt_iri": pair[1],
                "selected_labels": {"source": pair[0], "target": pair[1]},
                "confidences": {"S_final": score},
                "prediction": {"global_match": True, "threshold_positive": True},
            }
        )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["src_iri"], []).append(record)
    pairs: List[Dict[str, Any]] = []
    for src, source_records in grouped.items():
        ranked = sorted(source_records, key=lambda item: (-_record_score(item), item["tgt_iri"]))
        paths: List[Dict[str, Any]] = []
        for rank, record in enumerate(ranked, start=1):
            pair = (src, record["tgt_iri"])
            prediction = dict(record.get("prediction") or {})
            ground_truth = prediction.get("ground_truth")
            paths.append(
                {
                    "id": pair[1],
                    "rank": rank,
                    "ground_truth": ground_truth,
                    "score": _record_score(record),
                    "metrics": {},
                    "selected": pair in mapping_by_pair,
                }
            )
        pairs.append({"id": src, "paths": paths})
    return {"pairs": pairs}, records


def _hash_id(prefix: str, raw: str) -> str:
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _edge_score_rank(value: Any) -> int:
    text = safe_text(value).lower()
    if text == "strong":
        return 3
    if text == "moderate":
        return 2
    if text == "weak":
        return 1
    return 0


def _append_unique_edge(
    edges: List[Dict[str, Any]],
    edge_index: Dict[Tuple[str, str, str, str], int],
    source: str,
    target: str,
    label: str,
    edge_type: str,
    score: Any,
    origin: str = "explanation",
) -> None:
    key = (source, target, label, edge_type)
    payload = {
        "id": _hash_id("edge", "|".join(key)),
        "source": source,
        "target": target,
        "label": label,
        "type": edge_type,
        "score": score,
        "origin": origin,
    }
    idx = edge_index.get(key)
    if idx is None:
        edge_index[key] = len(edges)
        edges.append(payload)
        return
    if _edge_score_rank(score) > _edge_score_rank(edges[idx].get("score")):
        edges[idx] = payload


def _bridge_strength(channel_importance: float, local_mass: float) -> str:
    relevance = safe_float(channel_importance, 0.0) * (0.5 + 0.5 * safe_float(local_mass, 0.0))
    return categorize_bridge_relevance(relevance)


def _normalized_attribute_property(item: Mapping[str, Any]) -> str:
    return safe_text(item.get("property")).lower().replace("_", " ").replace("-", " ")


def _is_definition_like_attribute(item: Mapping[str, Any]) -> bool:
    prop = _normalized_attribute_property(item)
    if prop in DEFINITION_ATTRIBUTE_PROPERTIES:
        return True
    text = safe_text(item.get("text")).lower()
    return text.startswith("definition:") or text.startswith("description:")


class PrecomputedOntologyLookup:
    def __init__(
        self,
        run_dir: Path,
        analysis_dir: Path,
        logger: logging.Logger,
        enabled: bool = True,
    ) -> None:
        self.run_dir = run_dir
        self.analysis_dir = analysis_dir
        self.logger = logger
        self.enabled = bool(enabled)
        self.bundle_manifest_path = self.run_dir / "study_bundle.json"
        self.bundle_manifest = self._load_bundle_manifest()
        self.cache_path = self._resolve_cache_path()
        self.cache = self._load_cache()

    def _load_bundle_manifest(self) -> Dict[str, Any]:
        if not self.bundle_manifest_path.exists():
            return {}
        try:
            return dict(json.loads(self.bundle_manifest_path.read_text(encoding="utf-8")) or {})
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Failed to parse study bundle manifest %s: %s", self.bundle_manifest_path, exc
            )
            return {}

    def _resolve_cache_path(self) -> Path:
        manifest_cache = safe_text(self.bundle_manifest.get("ontology_cache_path"))
        if manifest_cache:
            candidate = Path(manifest_cache)
            if not candidate.is_absolute():
                candidate = (self.run_dir / candidate).resolve()
            return candidate
        return (self.analysis_dir / "ontology_cache.json").resolve()

    def _load_cache(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        if not self.enabled:
            return {"source": {}, "target": {}}
        if not self.cache_path.exists():
            self.logger.warning(
                "Ontology info enabled, but no precomputed ontology cache was found at %s; ontology info will be empty.",
                self.cache_path,
            )
            return {"source": {}, "target": {}}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to parse ontology cache %s: %s", self.cache_path, exc)
            return {"source": {}, "target": {}}
        return {
            "source": dict((payload.get("source") or {})),
            "target": dict((payload.get("target") or {})),
        }

    def entry_count(self) -> int:
        return len(self.cache.get("source", {})) + len(self.cache.get("target", {}))

    def _entry(self, entity_iri: str, side: str) -> Dict[str, Any]:
        return dict((self.cache.get(side) or {}).get(entity_iri) or {})

    def annotation_payload(self, entity_iri: str, side: str) -> Dict[str, Any]:
        entry = self._entry(entity_iri, side)
        return {
            "labels": list(entry.get("labels") or []),
            "definitions": list(entry.get("definitions") or []),
            "synonyms": list(entry.get("synonyms") or []),
            "annotations": list(entry.get("annotations") or []),
        }

    def direct_parents(self, entity_iri: str, side: str, limit: int = 3) -> List[Dict[str, Any]]:
        return [
            dict(row) for row in list(self._entry(entity_iri, side).get("parents") or [])[:limit]
        ]

    def direct_children(self, entity_iri: str, side: str, limit: int = 3) -> List[Dict[str, Any]]:
        return [
            dict(row) for row in list(self._entry(entity_iri, side).get("children") or [])[:limit]
        ]

    def direct_neighbors(self, entity_iri: str, side: str, limit: int = 4) -> List[Dict[str, Any]]:
        return [
            dict(row) for row in list(self._entry(entity_iri, side).get("neighbors") or [])[:limit]
        ]


def _resolve_ontology_paths(
    run_dir: Path,
    source_path: Optional[Path],
    target_path: Optional[Path],
    logger: logging.Logger,
) -> Tuple[Optional[Path], Optional[Path]]:
    if source_path is not None or target_path is not None:
        return (
            source_path.expanduser().resolve() if source_path else None,
            target_path.expanduser().resolve() if target_path else None,
        )
    for spec_path in sorted(run_dir.glob("*.yaml")) + sorted(run_dir.glob("*.yml")):
        try:
            payload = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.debug("Skipping unreadable ontology spec %s: %s", spec_path, exc)
            continue
        dataset = dict(payload.get("dataset") or {})
        source_name = _first_value(dataset, ("source", "source_ontology", "source_ontology_file"))
        target_name = _first_value(dataset, ("target", "target_ontology", "target_ontology_file"))
        if not source_name or not target_name:
            continue
        root = Path(str(dataset.get("data_dir") or spec_path.parent))
        if not root.is_absolute():
            root = (spec_path.parent / root).resolve()
        resolved_source = Path(str(source_name))
        resolved_target = Path(str(target_name))
        if not resolved_source.is_absolute():
            resolved_source = (root / resolved_source).resolve()
        if not resolved_target.is_absolute():
            resolved_target = (root / resolved_target).resolve()
        if resolved_source.is_file() and resolved_target.is_file():
            return resolved_source, resolved_target
    return None, None


class OntologyLookup:
    """Prefer a bundle cache and lazily fall back to the Java-free ontology API."""

    def __init__(
        self,
        *,
        run_dir: Path,
        analysis_dir: Path,
        logger: logging.Logger,
        enabled: bool,
        source_path: Optional[Path] = None,
        target_path: Optional[Path] = None,
    ) -> None:
        self.cached = PrecomputedOntologyLookup(
            run_dir=run_dir,
            analysis_dir=analysis_dir,
            logger=logger,
            enabled=enabled,
        )
        self.run_dir = run_dir
        self.logger = logger
        self.enabled = enabled
        self.source_path, self.target_path = _resolve_ontology_paths(
            run_dir, source_path, target_path, logger
        )
        self._sources: Dict[str, Any] = {}
        self._edges: Dict[str, List[Any]] = {}

    @property
    def cache_path(self) -> Path:
        return self.cached.cache_path

    def entry_count(self) -> int:
        return self.cached.entry_count()

    def _native(self, side: str) -> Any:
        if not self.enabled:
            return None
        if side in self._sources:
            return self._sources[side]
        path = self.source_path if side == "source" else self.target_path
        if path is None:
            self._sources[side] = None
            return None
        try:
            source = load_ontology(path)
        except Exception as exc:  # noqa: BLE001 - optional enrichment must not break serving
            self.logger.warning("Could not load %s ontology %s: %s", side, path, exc)
            source = None
        self._sources[side] = source
        return source

    @staticmethod
    def _label(source: Any, iri: str) -> str:
        labels = source.labels(iri)
        return safe_text(labels[0] if labels else source.short_form(iri)) or iri

    def annotation_payload(self, entity_iri: str, side: str) -> Dict[str, Any]:
        cached = self.cached.annotation_payload(entity_iri, side)
        if any(cached.values()):
            return cached
        source = self._native(side)
        if source is None:
            return cached
        values = source.annotations(entity_iri)
        annotations = [
            {"property": value.property_iri, "value": str(value.value)}
            for value in values
            if value.is_literal
        ]
        definitions = [
            row["value"]
            for row in annotations
            if any(token in row["property"].lower() for token in ("definition", "comment"))
        ]
        synonyms = [
            row["value"]
            for row in annotations
            if any(token in row["property"].lower() for token in ("synonym", "altlabel"))
        ]
        return {
            "labels": source.labels(entity_iri),
            "definitions": definitions[:5],
            "synonyms": synonyms[:10],
            "annotations": annotations[:12],
        }

    def direct_parents(self, entity_iri: str, side: str, limit: int = 3) -> List[Dict[str, Any]]:
        cached = self.cached.direct_parents(entity_iri, side, limit)
        if cached:
            return cached
        source = self._native(side)
        if source is None:
            return []
        return [
            {"entity_iri": iri, "label": self._label(source, iri), "relation": "is_a"}
            for iri in source.direct_parents(entity_iri)[:limit]
        ]

    def direct_children(self, entity_iri: str, side: str, limit: int = 3) -> List[Dict[str, Any]]:
        cached = self.cached.direct_children(entity_iri, side, limit)
        if cached:
            return cached
        source = self._native(side)
        if source is None:
            return []
        return [
            {"entity_iri": iri, "label": self._label(source, iri), "relation": "is_a"}
            for iri in source.direct_children(entity_iri)[:limit]
        ]

    def direct_neighbors(self, entity_iri: str, side: str, limit: int = 4) -> List[Dict[str, Any]]:
        cached = self.cached.direct_neighbors(entity_iri, side, limit)
        if cached:
            return cached
        source = self._native(side)
        if source is None:
            return []
        if side not in self._edges:
            self._edges[side] = source.projection_edges(method="owl2vecstar")
        rows: List[Dict[str, Any]] = []
        for edge in self._edges[side]:
            if edge.src == entity_iri:
                other, direction = edge.dst, "out"
            elif edge.dst == entity_iri:
                other, direction = edge.src, "in"
            else:
                continue
            rows.append(
                {
                    "entity_iri": other,
                    "label": self._label(source, other),
                    "relation": self._label(source, edge.rel),
                    "relation_iri": edge.rel,
                    "direction": direction,
                    "score": 0.0,
                }
            )
            if len(rows) >= limit:
                break
        return rows


class StudyVisualizerService:
    def __init__(
        self,
        run_dir: Path,
        analysis_dir: Optional[Path] = None,
        enable_ontology_info: bool = True,
        source_ontology_path: Optional[Path] = None,
        target_ontology_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.analysis_dir = (analysis_dir or (self.run_dir / "analysis" / "user_study")).resolve()
        self.logger = logger or logging.getLogger("exact_inspect")
        self.enable_ontology_info = bool(enable_ontology_info)
        self.output_dir = self.analysis_dir
        self.mapping_path = self.analysis_dir / "study_mapping.json"
        self.mode = "bundle" if self.mapping_path.is_file() else "open"
        self.bundle_manifest_path = self.run_dir / "study_bundle.json"
        self.bundle_manifest = self._load_bundle_manifest()
        self.reader: Optional[RunReader] = None
        self.selected_records_path: Optional[Path] = None
        if self.mode == "bundle":
            self.selected_records_path = self._resolve_selected_records_path()
            self.study_mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))
            self.selected_records = json.loads(
                self.selected_records_path.read_text(encoding="utf-8")
            )
        else:
            self.reader = RunReader.open(self.run_dir)
            self.study_mapping, self.selected_records = _open_mode_payload(self.reader)
        self.selected_record_index: Dict[Tuple[str, str], Dict[str, Any]] = {
            _record_pair(record): _normalise_record(record) for record in self.selected_records
        }
        self.config_path = self._resolve_config_path()
        self.ontology = OntologyLookup(
            run_dir=self.run_dir,
            analysis_dir=self.analysis_dir,
            logger=self.logger,
            enabled=self.enable_ontology_info,
            source_path=source_ontology_path,
            target_path=target_ontology_path,
        )
        self._source_cache: Dict[str, Dict[str, Any]] = {}
        self._path_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._pair_mapping_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._selected_sources: List[str] = []
        self._source_label_lookup: Dict[str, str] = {}
        self._build_mapping_indexes()

    def _resolve_selected_records_path(self) -> Path:
        preferred = self.analysis_dir / "study_selected_records_with_rationales.json"
        fallback = self.analysis_dir / "study_selected_records.json"
        if preferred.exists():
            return preferred
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            f"Selected study records not found in {self.analysis_dir}; expected {preferred.name} or {fallback.name}"
        )

    def _load_bundle_manifest(self) -> Dict[str, Any]:
        if not self.bundle_manifest_path.exists():
            return {}
        try:
            payload = json.loads(self.bundle_manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Failed to parse study bundle manifest %s: %s", self.bundle_manifest_path, exc
            )
            return {}
        return dict(payload or {})

    def _resolve_config_path(self) -> Optional[Path]:
        manifest_config = safe_text(self.bundle_manifest.get("config_path"))
        if manifest_config:
            candidate = Path(manifest_config)
            if not candidate.is_absolute():
                candidate = (self.run_dir / candidate).resolve()
            if candidate.exists():
                return candidate
        for candidate in [self.run_dir / "config.yaml", self.run_dir / "config.yml"]:
            if candidate.exists():
                return candidate.resolve()
        return None

    def health_payload(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "mode": self.mode,
            "layout_version": self.reader.layout.version if self.reader else None,
            "run_dir": str(self.run_dir),
            "analysis_dir": str(self.analysis_dir),
            "config_path": str(self.config_path) if self.config_path else None,
            "selected_source_count": len(self._selected_sources),
            "selected_path_count": len(self._pair_mapping_lookup),
            "ontology_info_enabled": bool(self.enable_ontology_info),
            "bundle_manifest_present": bool(self.bundle_manifest),
            "ontology_cache_path": str(self.ontology.cache_path),
            "ontology_cache_entries": self.ontology.entry_count(),
        }

    def _build_mapping_indexes(self) -> None:
        for pair in list(self.study_mapping.get("pairs") or []):
            src = safe_text(pair.get("id"))
            if not src:
                continue
            self._selected_sources.append(src)
            source_label = self._source_label_lookup.get(src, "")
            for path in list(pair.get("paths") or []):
                tgt = safe_text(path.get("id"))
                if tgt:
                    self._pair_mapping_lookup[(src, tgt)] = dict(path)
                    if not source_label:
                        record = self.selected_record_index.get((src, tgt)) or {}
                        source_label = safe_text(
                            ((record.get("selected_labels") or {})).get("source")
                        )
            self._source_label_lookup[src] = source_label or src
        self._selected_sources = list(dict.fromkeys(self._selected_sources))
        self.logger.info(
            "Study visualizer indexed %d selected sources and %d selected source-target paths",
            len(self._selected_sources),
            len(self._pair_mapping_lookup),
        )

    def available_sources(self) -> List[str]:
        return list(self._selected_sources)

    def source_options(self) -> List[Dict[str, str]]:
        return [
            {
                "source_id": source_id,
                "source_label": self._source_label_lookup.get(source_id) or source_id,
            }
            for source_id in self._selected_sources
        ]

    @staticmethod
    def _endpoint_node_id(side: str, entity_iri: str) -> str:
        return f"{side}-endpoint::{entity_iri}"

    def _entity_node_id(self, side: str, entity_iri: str) -> str:
        return f"{side}-entity::{entity_iri}"

    def _literal_node_id(self, side: str, item_id: str, text: str) -> str:
        token = item_id or _hash_id(f"{side}-literal", text)
        return f"{side}-literal::{token}"

    def _additional_node_id(self, side: str, entity_iri: str) -> str:
        return f"{side}-ontology-extra::{entity_iri}"

    def _add_node(
        self,
        nodes: List[Dict[str, Any]],
        node_lookup: Dict[str, Dict[str, Any]],
        node_id: str,
        label: str,
        node_type: str,
        origin: str,
        expandable: bool,
        entity_iri: Optional[str],
        ontology_side: Optional[str],
        node_kind: str,
    ) -> str:
        existing = node_lookup.get(node_id)
        if existing is not None:
            if label and not existing.get("label"):
                existing["label"] = label
            if entity_iri and not existing.get("entity_iri"):
                existing["entity_iri"] = entity_iri
                existing["expandable"] = bool(
                    entity_iri and existing.get("origin") == "explanation"
                )
            return node_id
        payload = {
            "id": node_id,
            "label": label,
            "type": node_type,
            "origin": origin,
            "expandable": bool(expandable),
            "entity_iri": entity_iri,
            "ontology_side": ontology_side,
            "node_kind": node_kind,
            "details": [],
        }
        nodes.append(payload)
        node_lookup[node_id] = payload
        return node_id

    def _add_detail(
        self, node_lookup: Dict[str, Dict[str, Any]], node_id: str, detail: Mapping[str, Any]
    ) -> None:
        node = node_lookup.get(node_id)
        if node is None:
            return
        details = list(node.get("details") or [])
        details.append(dict(detail))
        node["details"] = details[:20]

    def _endpoint_node(
        self,
        record: Mapping[str, Any],
        side: str,
        nodes: List[Dict[str, Any]],
        node_lookup: Dict[str, Dict[str, Any]],
    ) -> str:
        entity_iri = safe_text(record.get("src_iri" if side == "source" else "tgt_iri"))
        label = safe_text((record.get("selected_labels") or {}).get(side)) or entity_iri
        node_id = self._endpoint_node_id(side, entity_iri)
        return self._add_node(
            nodes=nodes,
            node_lookup=node_lookup,
            node_id=node_id,
            label=label,
            node_type="Source" if side == "source" else "Target",
            origin="explanation",
            expandable=bool(entity_iri),
            entity_iri=entity_iri or None,
            ontology_side=side,
            node_kind="endpoint",
        )

    def _entity_context_node(
        self,
        label: str,
        side: str,
        entity_iri: Optional[str],
        nodes: List[Dict[str, Any]],
        node_lookup: Dict[str, Dict[str, Any]],
    ) -> str:
        if entity_iri:
            node_id = self._entity_node_id(side, entity_iri)
        else:
            node_id = self._literal_node_id(side, "", label)
        return self._add_node(
            nodes=nodes,
            node_lookup=node_lookup,
            node_id=node_id,
            label=label,
            node_type="source-context" if side == "source" else "target-context",
            origin="explanation",
            expandable=bool(entity_iri),
            entity_iri=entity_iri or None,
            ontology_side=side if entity_iri else None,
            node_kind="entity-context" if entity_iri else "literal-context",
        )

    def _attribute_context_node(
        self,
        item: Mapping[str, Any],
        side: str,
        nodes: List[Dict[str, Any]],
        node_lookup: Dict[str, Dict[str, Any]],
    ) -> str:
        label = attribute_node_display(item)
        item_id = safe_text(item.get("item_id"))
        node_id = self._literal_node_id(side, item_id, label)
        return self._add_node(
            nodes=nodes,
            node_lookup=node_lookup,
            node_id=node_id,
            label=label,
            node_type="source-context" if side == "source" else "target-context",
            origin="explanation",
            expandable=False,
            entity_iri=None,
            ontology_side=None,
            node_kind="literal-context",
        )

    def _triple_node(
        self,
        label: str,
        side: str,
        entity_iri: Optional[str],
        endpoint_id: str,
        endpoint_label: str,
        endpoint_iri: str,
        nodes: List[Dict[str, Any]],
        node_lookup: Dict[str, Dict[str, Any]],
    ) -> str:
        if label in {endpoint_label, endpoint_iri}:
            return endpoint_id
        return self._entity_context_node(label, side, entity_iri, nodes, node_lookup)

    def _graph_from_record(
        self,
        record: Mapping[str, Any],
        mapping_path: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        labels = dict(record.get("selected_labels") or {})
        importances = dict(record.get("importances") or {})
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        node_lookup: Dict[str, Dict[str, Any]] = {}
        edge_index: Dict[Tuple[str, str, str, str], int] = {}
        item_node_lookup: Dict[str, str] = {}
        item_importance_lookup: Dict[str, float] = {}
        entity_node_lookup: Dict[Tuple[str, str], str] = {}

        src_endpoint = self._endpoint_node(record, "source", nodes, node_lookup)
        tgt_endpoint = self._endpoint_node(record, "target", nodes, node_lookup)
        src_label = safe_text(labels.get("source")) or safe_text(record.get("src_iri"))
        tgt_label = safe_text(labels.get("target")) or safe_text(record.get("tgt_iri"))
        src_iri = safe_text(record.get("src_iri"))
        tgt_iri = safe_text(record.get("tgt_iri"))
        entity_node_lookup[("source", src_iri)] = src_endpoint
        entity_node_lookup[("target", tgt_iri)] = tgt_endpoint

        def _remember_entity(side: str, entity_iri: str, node_id: str) -> None:
            if entity_iri:
                entity_node_lookup[(side, entity_iri)] = node_id

        def _triple_item_anchor(subject_id: str, object_id: str, endpoint_id: str) -> str:
            if subject_id == endpoint_id and object_id != endpoint_id:
                return object_id
            if object_id == endpoint_id and subject_id != endpoint_id:
                return subject_id
            return object_id

        def _add_context_edge(side: str, channel: str, item: Mapping[str, Any]) -> None:
            triple = list(item.get("triple") or [])
            if len(triple) < 3:
                return
            relation = safe_text(triple[1]) or channel
            endpoint_id = src_endpoint if side == "source" else tgt_endpoint
            endpoint_label = src_label if side == "source" else tgt_label
            endpoint_iri = src_iri if side == "source" else tgt_iri
            subject_id = self._triple_node(
                label=safe_text(triple[0]),
                side=side,
                entity_iri=safe_text(item.get("subject_iri")) or None,
                endpoint_id=endpoint_id,
                endpoint_label=endpoint_label,
                endpoint_iri=endpoint_iri,
                nodes=nodes,
                node_lookup=node_lookup,
            )
            object_id = self._triple_node(
                label=safe_text(triple[2]),
                side=side,
                entity_iri=safe_text(item.get("object_iri")) or None,
                endpoint_id=endpoint_id,
                endpoint_label=endpoint_label,
                endpoint_iri=endpoint_iri,
                nodes=nodes,
                node_lookup=node_lookup,
            )
            _remember_entity(side, safe_text(item.get("subject_iri")), subject_id)
            _remember_entity(side, safe_text(item.get("object_iri")), object_id)
            score = safe_float(
                item.get("importance", item.get("support", item.get("unsupported_mass", 1.0))), 1.0
            )
            _append_unique_edge(edges, edge_index, subject_id, object_id, relation, channel, score)
            verbalized = f"{safe_text(triple[0])} --{relation}--> {safe_text(triple[2])}"
            self._add_detail(
                node_lookup,
                subject_id,
                {"channel": channel, "role": "subject", "triple": verbalized, "score": score},
            )
            self._add_detail(
                node_lookup,
                object_id,
                {"channel": channel, "role": "object", "triple": verbalized, "score": score},
            )
            item_id = safe_text(item.get("item_id"))
            if item_id:
                item_node_lookup[item_id] = _triple_item_anchor(subject_id, object_id, endpoint_id)
                item_importance_lookup[item_id] = safe_float(item.get("importance"), 0.0)

        for item in hierarchy_items(record, "source"):
            _add_context_edge("source", "hierarchy", item)
        for item in hierarchy_items(record, "target"):
            _add_context_edge("target", "hierarchy", item)
        for item in channel_items(record, "similarity", "source"):
            _add_context_edge("source", "similarity", item)
        for item in channel_items(record, "similarity", "target"):
            _add_context_edge("target", "similarity", item)
        for item in channel_items(record, "difference", "source"):
            _add_context_edge("source", "difference", item)
        for item in channel_items(record, "difference", "target"):
            _add_context_edge("target", "difference", item)

        for side in ["source", "target"]:
            endpoint_id = src_endpoint if side == "source" else tgt_endpoint
            for item in attribute_items(record, side):
                if _is_definition_like_attribute(item):
                    continue
                node_id = self._attribute_context_node(item, side, nodes, node_lookup)
                relation = safe_text(item.get("property")) or "attribute"
                score = safe_float(item.get("importance", item.get("support", 1.0)), 1.0)
                _append_unique_edge(
                    edges, edge_index, endpoint_id, node_id, relation, "attribute", score
                )
                self._add_detail(
                    node_lookup,
                    node_id,
                    {
                        "channel": "attribute",
                        "property": relation,
                        "text": safe_text(item.get("text")),
                        "score": score,
                    },
                )
                item_id = safe_text(item.get("item_id"))
                if item_id:
                    item_node_lookup[item_id] = node_id
                    item_importance_lookup[item_id] = safe_float(item.get("importance"), 0.0)

        i_label = safe_float(importances.get("I_label"), 0.0)
        i_hier = safe_float(importances.get("I_hier"), 0.0)
        i_sim = safe_float(importances.get("I_sim"), 0.0)
        i_diff = safe_float(importances.get("I_diff"), 0.0)
        i_attr = safe_float(importances.get("I_attr"), 0.0)
        family_importances = dict(importances.get("family_importances") or {})
        provenance = dict(record.get("cross_side_provenance") or {})

        for _link in list(provenance.get("lexical") or []):
            bridge_score = _bridge_strength(i_label, 1.0)
            _append_unique_edge(
                edges,
                edge_index,
                src_endpoint,
                tgt_endpoint,
                "label match",
                "bridge-support",
                bridge_score,
            )

        for family, links in dict(provenance.get("hierarchy") or {}).items():
            family_importance = safe_float(family_importances.get(family), i_hier)
            for link in list(links or []):
                src_node = item_node_lookup.get(safe_text(link.get("source_item_id")))
                tgt_node = item_node_lookup.get(safe_text(link.get("target_item_id")))
                if not src_node or not tgt_node:
                    continue
                local_mass = mean_nonempty(
                    [
                        safe_float(
                            item_importance_lookup.get(safe_text(link.get("source_item_id"))), 0.0
                        ),
                        safe_float(
                            item_importance_lookup.get(safe_text(link.get("target_item_id"))), 0.0
                        ),
                    ],
                    default=0.0,
                )
                bridge_score = _bridge_strength(family_importance, local_mass)
                _append_unique_edge(
                    edges,
                    edge_index,
                    src_node,
                    tgt_node,
                    "shared hierarchy",
                    "bridge-support",
                    bridge_score,
                )

        for link in list(provenance.get("similarity") or []):
            src_node = item_node_lookup.get(safe_text(link.get("source_item_id")))
            tgt_node = item_node_lookup.get(safe_text(link.get("target_item_id")))
            if not src_node or not tgt_node:
                continue
            local_mass = mean_nonempty(
                [
                    safe_float(
                        item_importance_lookup.get(safe_text(link.get("source_item_id"))), 0.0
                    ),
                    safe_float(
                        item_importance_lookup.get(safe_text(link.get("target_item_id"))), 0.0
                    ),
                ],
                default=0.0,
            )
            bridge_score = _bridge_strength(i_sim, local_mass)
            _append_unique_edge(
                edges,
                edge_index,
                src_node,
                tgt_node,
                "similar evidence",
                "bridge-support",
                bridge_score,
            )

        for side in ["source", "target"]:
            opposite = tgt_endpoint if side == "source" else src_endpoint
            for link in list((dict(provenance.get("attributes") or {})).get(side) or []):
                item_node = item_node_lookup.get(safe_text(link.get("item_id")))
                if not item_node:
                    continue
                anchor_ref = safe_text(link.get("anchor_ref"))
                anchor_node = item_node_lookup.get(anchor_ref) or opposite
                local_mass = mean_nonempty(
                    [
                        safe_float(item_importance_lookup.get(safe_text(link.get("item_id"))), 0.0),
                        safe_float(item_importance_lookup.get(anchor_ref), 0.0),
                    ],
                    default=safe_float(
                        item_importance_lookup.get(safe_text(link.get("item_id"))), 0.0
                    ),
                )
                bridge_score = _bridge_strength(i_attr, local_mass)
                _append_unique_edge(
                    edges,
                    edge_index,
                    item_node,
                    anchor_node,
                    "attribute evidence",
                    "bridge-support",
                    bridge_score,
                )

        for link in list((dict(provenance.get("difference") or {})).get("source") or []):
            item_node = item_node_lookup.get(safe_text(link.get("item_id")))
            if item_node:
                bridge_score = _bridge_strength(
                    i_diff,
                    safe_float(item_importance_lookup.get(safe_text(link.get("item_id"))), 0.0),
                )
                _append_unique_edge(
                    edges,
                    edge_index,
                    item_node,
                    tgt_endpoint,
                    "distinctive evidence",
                    "bridge-contrast",
                    bridge_score,
                )
        for link in list((dict(provenance.get("difference") or {})).get("target") or []):
            item_node = item_node_lookup.get(safe_text(link.get("item_id")))
            if item_node:
                bridge_score = _bridge_strength(
                    i_diff,
                    safe_float(item_importance_lookup.get(safe_text(link.get("item_id"))), 0.0),
                )
                _append_unique_edge(
                    edges,
                    edge_index,
                    item_node,
                    src_endpoint,
                    "distinctive evidence",
                    "bridge-contrast",
                    bridge_score,
                )

        for edge in edges:
            is_bridge, level, level_label = edge_display_level(edge)
            edge["bridge"] = is_bridge
            edge["level"] = level
            edge["level_label"] = level_label

        ordered_nodes = ordered_path_nodes(nodes)
        pair_key = (safe_text(record.get("src_iri")), safe_text(record.get("tgt_iri")))
        return {
            "nodes": ordered_nodes,
            "edges": edges,
            "node_lookup": node_lookup,
            "entity_node_lookup": entity_node_lookup,
            "mapping_metrics": dict(mapping_path.get("metrics") or {}) if mapping_path else {},
            "pair_key": pair_key,
        }

    def _build_target_bundle(
        self,
        src_iri: str,
        tgt_iri: str,
        mapping_path: Mapping[str, Any],
    ) -> Dict[str, Any]:
        record = self.selected_record_index.get((src_iri, tgt_iri))
        if record is None:
            raise KeyError(f"Selected study record missing for ({src_iri}, {tgt_iri})")
        graph = self._graph_from_record(record, mapping_path)
        pair_key = (src_iri, tgt_iri)
        payload = {
            "target_id": tgt_iri,
            "target_label": safe_text((record.get("selected_labels") or {}).get("target"))
            or tgt_iri,
            "rank": int(mapping_path.get("rank", 0) or 0),
            "ground_truth": int(mapping_path.get("ground_truth", 0) or 0),
            "score": safe_float(
                mapping_path.get("score", (record.get("confidences") or {}).get("S_final", 0.0)),
                0.0,
            ),
            "metrics": dict(mapping_path.get("metrics") or {}),
            "llm": {
                "pair_brief": safe_text(record.get("llm_pair_brief")),
                "rationale": safe_text((record.get("prediction") or {}).get("llm_rationale")),
                "decision": safe_text((record.get("prediction") or {}).get("llm_decision")),
                "p_llm": safe_float((record.get("confidences") or {}).get("p_llm"), 0.0),
            },
            "graph": {
                "nodes": graph["nodes"],
                "edges": graph["edges"],
            },
            "prediction": {
                "threshold_positive": bool(
                    (record.get("prediction") or {}).get("threshold_positive")
                ),
                "global_match": bool((record.get("prediction") or {}).get("global_match")),
            },
            "record": record,
        }
        self._path_cache[pair_key] = {
            "target_bundle": payload,
            "node_lookup": graph["node_lookup"],
            "entity_node_lookup": graph["entity_node_lookup"],
        }
        return payload

    def get_source_bundle(self, source_iri: str) -> Dict[str, Any]:
        cached = self._source_cache.get(source_iri)
        if cached is not None:
            return cached
        pair = next(
            (
                dict(pair)
                for pair in list(self.study_mapping.get("pairs") or [])
                if safe_text(pair.get("id")) == source_iri
            ),
            None,
        )
        if pair is None:
            raise KeyError(source_iri)
        paths = sorted(list(pair.get("paths") or []), key=lambda row: int(row.get("rank", 0) or 0))
        targets = [
            self._build_target_bundle(source_iri, safe_text(path.get("id")), path) for path in paths
        ]
        source_label = ""
        if targets:
            source_label = safe_text(
                ((targets[0].get("record") or {}).get("selected_labels") or {}).get("source")
            )
        bundle = {
            "source_id": source_iri,
            "source_label": source_label or source_iri,
            "targets": [
                {
                    "target_id": target["target_id"],
                    "target_label": target["target_label"],
                    "rank": target["rank"],
                    "ground_truth": target["ground_truth"],
                    "score": target["score"],
                    "metrics": target["metrics"],
                    "llm": target["llm"],
                    "graph": target["graph"],
                }
                for target in targets
            ],
            "default_target_rank": 1,
        }
        self._source_cache[source_iri] = bundle
        return bundle

    def _resolve_target_bundle(self, source_iri: str, target_iri: str) -> Dict[str, Any]:
        self.get_source_bundle(source_iri)
        cached = self._path_cache.get((source_iri, target_iri))
        if cached is None:
            raise KeyError((source_iri, target_iri))
        return cached

    def get_node_info(self, source_iri: str, target_iri: str, node_id: str) -> Dict[str, Any]:
        path_cache = self._resolve_target_bundle(source_iri, target_iri)
        target_bundle = dict(path_cache["target_bundle"])
        node_lookup = dict(path_cache["node_lookup"])
        node = node_lookup.get(node_id)
        if node is None:
            raise KeyError(node_id)
        record = dict(target_bundle.get("record") or {})
        side = safe_text(node.get("ontology_side"))
        entity_iri = safe_text(node.get("entity_iri"))
        ontology_payload: Dict[str, Any] = {}
        if entity_iri and side and self.enable_ontology_info:
            ontology_payload = self.ontology.annotation_payload(entity_iri, side)
        explanation_payload = {
            "details": list(node.get("details") or []),
        }
        if node.get("node_kind") == "endpoint":
            explanation_payload["attributes"] = list(
                (record.get("attributes") or {}).get(side) or []
            )
        return {
            "node": {key: value for key, value in node.items() if key != "details"},
            "explanation": explanation_payload,
            "ontology": ontology_payload,
            "expandable": bool(node.get("expandable")),
        }

    def expand_node(self, source_iri: str, target_iri: str, node_id: str) -> Dict[str, Any]:
        if not self.enable_ontology_info:
            return {"clicked_node_id": node_id, "expandable": False, "nodes": [], "edges": []}
        path_cache = self._resolve_target_bundle(source_iri, target_iri)
        node_lookup = dict(path_cache["node_lookup"])
        entity_node_lookup = dict(path_cache["entity_node_lookup"])
        node = node_lookup.get(node_id)
        if node is None:
            raise KeyError(node_id)
        if not node.get("expandable"):
            return {"clicked_node_id": node_id, "expandable": False, "nodes": [], "edges": []}
        side = safe_text(node.get("ontology_side"))
        entity_iri = safe_text(node.get("entity_iri"))
        if not side or not entity_iri:
            return {"clicked_node_id": node_id, "expandable": False, "nodes": [], "edges": []}

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        patch_node_lookup: Dict[str, Dict[str, Any]] = {}
        edge_index: Dict[Tuple[str, str, str, str], int] = {}
        same_side_endpoint_type = "Source" if side == "source" else "Target"
        existing_side_labels = {
            " ".join(safe_text(existing.get("label")).lower().split())
            for existing in node_lookup.values()
            if safe_text(existing.get("ontology_side")) == side
            or safe_text(existing.get("type")) == same_side_endpoint_type
        }

        def _new_extra_node(entity_iri_value: str, label: str) -> Optional[str]:
            if not entity_iri_value or entity_iri_value == entity_iri:
                return None
            existing = entity_node_lookup.get((side, entity_iri_value))
            if existing:
                return None
            label_key = " ".join(safe_text(label).lower().split())
            if label_key and label_key in existing_side_labels:
                return None
            node_key = self._additional_node_id(side, entity_iri_value)
            return self._add_node(
                nodes=nodes,
                node_lookup=patch_node_lookup,
                node_id=node_key,
                label=label,
                node_type="ontology-extra",
                origin="ontology-extra",
                expandable=False,
                entity_iri=entity_iri_value,
                ontology_side=side,
                node_kind="entity-context",
            )

        for parent in self.ontology.direct_parents(entity_iri, side, limit=3):
            parent_id = _new_extra_node(parent["entity_iri"], parent["label"])
            if not parent_id:
                continue
            _append_unique_edge(
                edges,
                edge_index,
                node_id,
                parent_id,
                parent.get("relation", "is_a"),
                "ontology-extra",
                None,
                origin="ontology-extra",
            )

        for child in self.ontology.direct_children(entity_iri, side, limit=3):
            child_id = _new_extra_node(child["entity_iri"], child["label"])
            if not child_id:
                continue
            _append_unique_edge(
                edges,
                edge_index,
                child_id,
                node_id,
                child.get("relation", "is_a"),
                "ontology-extra",
                None,
                origin="ontology-extra",
            )

        for nbr in self.ontology.direct_neighbors(entity_iri, side, limit=4):
            nbr_id = _new_extra_node(nbr["entity_iri"], nbr["label"])
            if not nbr_id:
                continue
            if safe_text(nbr.get("direction")) == "out":
                source_id, target_id = node_id, nbr_id
            else:
                source_id, target_id = nbr_id, node_id
            _append_unique_edge(
                edges,
                edge_index,
                source_id,
                target_id,
                safe_text(nbr.get("relation")) or "related to",
                "ontology-extra",
                safe_float(nbr.get("score"), 0.0),
                origin="ontology-extra",
            )

        return {
            "clicked_node_id": node_id,
            "expandable": True,
            "nodes": nodes,
            "edges": edges,
        }


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _ontology_entry(source: Any, iri: str) -> Dict[str, Any]:
    annotations = [
        {"property": value.property_iri, "value": str(value.value)}
        for value in source.annotations(iri)
        if value.is_literal
    ]
    return {
        "labels": source.labels(iri),
        "definitions": [
            row["value"]
            for row in annotations
            if any(token in row["property"].lower() for token in ("definition", "comment"))
        ][:5],
        "synonyms": [
            row["value"]
            for row in annotations
            if any(token in row["property"].lower() for token in ("synonym", "altlabel"))
        ][:10],
        "annotations": annotations[:12],
        "parents": [
            {
                "entity_iri": parent,
                "label": (source.labels(parent) or [source.short_form(parent)])[0],
                "relation": "is_a",
            }
            for parent in source.direct_parents(iri)[:3]
        ],
        "children": [
            {
                "entity_iri": child,
                "label": (source.labels(child) or [source.short_form(child)])[0],
                "relation": "is_a",
            }
            for child in source.direct_children(iri)[:3]
        ],
        "neighbors": [],
    }


def _entity_iris(records: Iterable[Mapping[str, Any]], side: str) -> List[str]:
    found: set[str] = set()
    endpoint_key = "src_iri" if side == "source" else "tgt_iri"
    for record in records:
        endpoint = safe_text(record.get(endpoint_key))
        if endpoint:
            found.add(endpoint)
        for item in hierarchy_items(record, side):
            found.update(
                iri
                for iri in (safe_text(item.get("subject_iri")), safe_text(item.get("object_iri")))
                if iri
            )
        for channel in ("similarity", "difference"):
            for item in channel_items(record, channel, side):
                found.update(
                    iri
                    for iri in (
                        safe_text(item.get("subject_iri")),
                        safe_text(item.get("object_iri")),
                    )
                    if iri
                )
    return sorted(found)


def export_bundle(
    run_dir: Path,
    bundle_dir: Path,
    *,
    analysis_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    bundle_name: Optional[str] = None,
    source_ontology_path: Optional[Path] = None,
    target_ontology_path: Optional[Path] = None,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Export a portable inspection bundle from a historical or v2 run."""

    logger = logger or logging.getLogger("exact_inspect.bundle")
    run_dir = Path(run_dir).expanduser().resolve()
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if bundle_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Bundle directory already exists: {bundle_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(bundle_dir)
    bundle_analysis_dir = bundle_dir / "analysis" / "user_study"
    bundle_analysis_dir.mkdir(parents=True)

    analysis_dir = (
        Path(analysis_dir).expanduser().resolve()
        if analysis_dir
        else run_dir / "analysis" / "user_study"
    )
    mapping_path = analysis_dir / "study_mapping.json"
    selected_path = next(
        (
            candidate
            for candidate in (
                analysis_dir / "study_selected_records_with_rationales.json",
                analysis_dir / "study_selected_records.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    if mapping_path.is_file() and selected_path is not None:
        study_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        records = [
            _normalise_record(record)
            for record in json.loads(selected_path.read_text(encoding="utf-8"))
        ]
        selected_name = selected_path.name
    else:
        reader = RunReader.open(run_dir)
        study_mapping, records = _open_mode_payload(reader)
        selected_name = "study_selected_records.json"

    (bundle_analysis_dir / "study_mapping.json").write_text(
        json.dumps(study_mapping, indent=2), encoding="utf-8"
    )
    (bundle_analysis_dir / selected_name).write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )

    resolved_config = Path(config_path).expanduser().resolve() if config_path else None
    if resolved_config is None:
        resolved_config = next(
            (
                candidate
                for candidate in (run_dir / "config.yaml", run_dir / "config.yml")
                if candidate.is_file()
            ),
            None,
        )
    if resolved_config is not None:
        _copy_file(resolved_config, bundle_dir / "config.yaml")

    source_ontology_path, target_ontology_path = _resolve_ontology_paths(
        run_dir,
        source_ontology_path,
        target_ontology_path,
        logger,
    )
    ontology_cache: Dict[str, Dict[str, Dict[str, Any]]] = {"source": {}, "target": {}}
    for side, path in (("source", source_ontology_path), ("target", target_ontology_path)):
        if path is None:
            continue
        source = load_ontology(path)
        for iri in _entity_iris(records, side):
            ontology_cache[side][iri] = _ontology_entry(source, iri)
    (bundle_analysis_dir / "ontology_cache.json").write_text(
        json.dumps(ontology_cache, indent=2), encoding="utf-8"
    )

    manifest = {
        "bundle_name": bundle_name or run_dir.name,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "source_analysis_dir": str(analysis_dir),
        "config_path": "config.yaml" if resolved_config else None,
        "analysis_dir": "analysis/user_study",
        "study_mapping_path": "analysis/user_study/study_mapping.json",
        "selected_records_path": f"analysis/user_study/{selected_name}",
        "ontology_cache_path": "analysis/user_study/ontology_cache.json",
    }
    (bundle_dir / "study_bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Inspection bundle written to %s", bundle_dir)
    return bundle_dir


InspectionService = StudyVisualizerService


class StudyOntologyLookup(OntologyLookup):
    """Constructor-compatible alias for the pre-2.0 bundle helper."""

    def __init__(
        self,
        run_dir: Path,
        config_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
        enabled: bool = True,
    ) -> None:
        del config_path
        super().__init__(
            run_dir=Path(run_dir).resolve(),
            analysis_dir=Path(run_dir).resolve() / "analysis" / "user_study",
            logger=logger or logging.getLogger("exact_inspect"),
            enabled=enabled,
        )


__all__ = [
    "InspectionService",
    "OntologyLookup",
    "PrecomputedOntologyLookup",
    "StudyOntologyLookup",
    "StudyVisualizerService",
    "export_bundle",
]
