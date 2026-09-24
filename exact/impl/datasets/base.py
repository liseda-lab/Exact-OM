# mypy: ignore-errors

import hashlib
import json
import math
import os
import warnings
from abc import abstractmethod
from ast import literal_eval
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # Compatibility with minimal/runtime-specific installations.
    CrossEncoder = None  # type: ignore[misc, assignment]

from exact.core.contracts.dataset import IDataset
from exact.core.contracts.knowledge import KnowledgeSource
from exact.core.entities.configs.dataset import DatasetMask
from exact.core.entities.kinds import (
    MATCHABLE_ENTITY_KINDS,
    EntityKind,
    build_entity_kind_index,
    infer_entity_kind,
    normalize_entity_kinds,
)
from exact.core.entities.ontology import OntologyGraph
from exact.impl.datasets import prepared_cache
from exact.impl.datasets.options import candidate_config, mapping_options
from exact.impl.retrieval import (
    LocalRetrievalArtifact,
    resolve_local_retrieval_artifact,
)
from exact.io.sources import resolve as resolve_source
from exact.ontology.projection import ProjectorSettings, projector_cache_identity
from exact.ontology.reasoning import (
    reasoner_cache_identity,
    require_native_reasoner_support,
)
from exact.runs.layout import RunLayout
from exact.utils.candidate_generation import (
    adaptive_candidate_count,
    candidate_annotation_priority,
    candidate_token_key,
    lexical_candidate_pair_scores,
    make_candidate_labels,
    normalize_candidate_text,
    rank_channel_scores,
    select_candidate_annotation_literals,
)
from exact.utils.data import read_table
from exact.utils.provenance import dataset_signature_for_paths, file_provenance

DataFrame = pd.DataFrame
_DATASET_CACHE_SCHEMA_VERSION = 4
_ONTOLOGY_BACKEND_VERSION = 5


def _sentence_transformer_resolved_revision(
    model: Any, requested_revision: Optional[str]
) -> Optional[str]:
    """Return the concrete HF revision exposed by a loaded sentence encoder."""

    candidates: List[Any] = [model]
    first_module = getattr(model, "_first_module", None)
    if callable(first_module):
        try:
            candidates.append(first_module())
        except (AttributeError, KeyError, TypeError):
            pass
    index = 0
    seen: set[int] = set()
    while index < len(candidates):
        component = candidates[index]
        index += 1
        if component is None or id(component) in seen:
            continue
        seen.add(id(component))
        auto_model = getattr(component, "auto_model", None)
        if auto_model is not None:
            candidates.append(auto_model)
        config = getattr(component, "config", None)
        if config is not None:
            candidates.append(config)

    for component in candidates:
        if component is None:
            continue
        for field in ("resolved_revision", "_commit_hash", "revision"):
            value = getattr(component, field, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return requested_revision


class BaseAlignmentDataset(IDataset):
    def prepare_retrieval_training(self, configs, **kwargs) -> None:
        from exact.impl.retrieval.training import prepare_retrieval_training

        prepare_retrieval_training(self, configs, **kwargs)

    def prepare_pool_miss_diagnostic(self, reference_path, **kwargs) -> None:
        from exact.impl.models.selector.nil_head import prepare_pool_miss_diagnostic

        prepare_pool_miss_diagnostic(self, reference_path, **kwargs)

    def __init__(
        self,
        output_path: Path,
        filter_exact_matches: bool = False,
        cardinality: int = 1,
        num_workers: Optional[int] = None,
        drop_exact_match_sources: bool = True,
        filter_ignored_alignment_classes: bool = False,
        **kwargs,
    ) -> None:
        self._run_layout = RunLayout.create(Path(output_path))
        self._output_path: Path = self._run_layout.dataset_dir
        self._output_path.mkdir(parents=True, exist_ok=True)
        self.plot_dir: Path = self.output_path / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self._df: DataFrame = None
        self._df_save_path: Path = self.output_path / "dataset.csv"
        self._cache_meta_path: Path = self.output_path / "dataset.meta.json"
        self._active_df_cache_key: Optional[Tuple[int, DatasetMask]] = None
        self._active_df_cache: Optional[DataFrame] = None

        self._num_workers: int = num_workers

        self._default_kind: DatasetMask = DatasetMask.inference

        self._source: KnowledgeSource | None = None
        self._target: KnowledgeSource | None = None
        self._source_graph: OntologyGraph = None
        self._target_graph: OntologyGraph = None
        self._candidates: DataFrame = None
        self._reference: DataFrame = None
        self._exact_matches: DataFrame = None

        self._filter_exact_matches: bool = filter_exact_matches
        self._drop_exact_match_sources: bool = drop_exact_match_sources
        self._filter_ignored_alignment_classes: bool = bool(filter_ignored_alignment_classes)
        self._source_ignored_alignment_classes: Optional[set[str]] = None
        self._target_ignored_alignment_classes: Optional[set[str]] = None

        self._cardinality: int = cardinality

        self._cache_ok = kwargs.get("cache_ok", True)
        self._candidate_share_k: int = int(kwargs.get("candidate_share_k", 1))
        self._candidate_generation_params: Dict[str, Any] = dict(
            kwargs.get("candidate_generation_params") or {}
        )
        self._input_format: str = str(kwargs.get("input_format", "auto") or "auto")
        self._source_options: Dict[str, Any] = mapping_options(
            kwargs.get("source_options"), "source_options"
        )
        self._target_options: Dict[str, Any] = mapping_options(
            kwargs.get("target_options"), "target_options"
        )

        self._candidates_generated = False
        self._source_path: Optional[Path] = None
        self._target_path: Optional[Path] = None
        self._dataset_signature: Optional[str] = None
        self._cache_warning_emitted: bool = False
        self._cache_state: str = "cold"
        self._only_taxonomy_hint: bool = bool(kwargs.get("only_taxonomy", False))
        self._reasoner_name: str = str(kwargs.get("reasoner", "asserted"))
        self._projector_settings = ProjectorSettings.from_value(kwargs.get("projector"))
        raw_entity_kinds = kwargs.get("entity_kinds")
        if raw_entity_kinds is None:
            matching = kwargs.get("matching")
            if isinstance(matching, dict):
                raw_entity_kinds = matching.get("entity_kinds")
            elif matching is not None:
                raw_entity_kinds = getattr(matching, "entity_kinds", None)
        self._entity_kinds = normalize_entity_kinds(raw_entity_kinds)
        self._source_entity_kind_index: Dict[str, EntityKind] = {}
        self._target_entity_kind_index: Dict[str, EntityKind] = {}
        self._unknown_kind_warnings: set[Tuple[str, str]] = set()
        self._candidate_pool_sizes: Dict[str, Dict[str, int]] = {}
        self._candidate_pool_manifest: Dict[str, Any] = {}
        self._candidate_pool_manifest_path = self.output_path / "candidate_pool_manifest.json"
        self._active_candidate_config: Dict[str, Any] = {}
        self._retrieval_artifacts: Dict[str, LocalRetrievalArtifact] = {}
        self._candidate_data_lock_binding = self._candidate_provenance_binding(
            kwargs.get("candidate_data_lock"),
            label="candidate_data_lock",
            require_sha256=True,
        )
        self._candidate_spec_lock_binding = self._candidate_provenance_binding(
            kwargs.get("candidate_spec_lock"),
            label="candidate_spec_lock",
            require_sha256=True,
        )
        self._candidate_model_lock_binding = self._candidate_provenance_binding(
            kwargs.get("candidate_model_lock"),
            label="candidate_model_lock",
            require_sha256=False,
        )
        self._source_restriction_active = False

        super().__init__(logger=kwargs.get("logger"))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:
        return self.dataframe.__str__()

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    def cardinality_1_to_many(self) -> bool:
        return True if self._cardinality > 1 else False

    @property
    def dataframe(self) -> DataFrame:
        return self._df

    @property
    def default_kind(self) -> DatasetMask:
        return self._default_kind

    @default_kind.setter
    def default_kind(self, kind: DatasetMask) -> None:
        self._default_kind = kind
        self._invalidate_active_dataframe_cache()

    def _invalidate_active_dataframe_cache(self) -> None:
        self._active_df_cache_key = None
        self._active_df_cache = None

    def _active_dataframe(self) -> DataFrame:
        if self._df is None:
            raise RuntimeError("Dataset not processed. Call process() first.")
        key = (id(self._df), self.default_kind)
        if self._active_df_cache_key == key and self._active_df_cache is not None:
            return self._active_df_cache
        active = self._df[self._df[self.default_kind]].reset_index(drop=True)
        self._active_df_cache_key = key
        self._active_df_cache = active
        return active

    @property
    def filter_exact_matches(self) -> bool:
        return self._filter_exact_matches

    @property
    def drop_exact_match_sources(self) -> bool:
        return self._drop_exact_match_sources

    @property
    def filter_ignored_alignment_classes(self) -> bool:
        return self._filter_ignored_alignment_classes

    @property
    def source_ignored_alignment_classes(self) -> set[str]:
        return self._ignored_alignment_class_iris("src")

    @property
    def target_ignored_alignment_classes(self) -> set[str]:
        return self._ignored_alignment_class_iris("tgt")

    @property
    def source(self) -> KnowledgeSource:
        return self._source

    @property
    def target(self) -> KnowledgeSource:
        return self._target

    @property
    def exact_matches(self) -> DataFrame:
        return self._exact_matches

    @property
    def candidates(self) -> DataFrame:
        return self._candidates

    @property
    def candidate_recall_cache_path(self) -> Optional[Path]:
        expected = self._load_cache_metadata().get("candidate_recall_sha256")
        if expected is None:
            return None  # Legacy caches did not preserve the raw retrieval pool.
        path = self.output_path / "candidate_recall.tsv"
        if not path.is_file() or file_provenance(path)["sha256"] != expected:
            raise ValueError("Cached candidate-recall pool is missing or has changed")
        return path

    def candidate_recall_frames(self) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
        """Read reporting-only raw pools without changing cached inference inputs."""
        if self._candidates is not None:
            return self._candidates, self._exact_matches
        path = self.candidate_recall_cache_path
        if path is None:
            return None, None
        frame = pd.read_csv(path, sep="\t", float_precision="round_trip")
        groups = self._active_candidate_config.get("source_sample", {}).get("source_kind_groups")
        if groups is not None:
            selected = {tuple(group) for group in groups}
            frame = frame.loc[
                [
                    (str(src), str(kind)) in selected
                    for src, kind in frame[["Src", "SrcKind"]].itertuples(index=False, name=None)
                ]
            ]
        return tuple(
            frame.loc[frame["pool_role"] == role].drop(columns="pool_role").reset_index(drop=True)
            for role in ("candidate", "exact")
        )

    @property
    def candidates_generated(self) -> bool:
        return self._candidates_generated

    @property
    def entity_kinds(self) -> Tuple[EntityKind, ...]:
        """Configured matching kinds, defaulting to historical class-only mode."""

        return self._entity_kinds

    @property
    def primary_entity_kind(self) -> EntityKind:
        """Fallback kind for unknown IRIs in legacy candidate/reference files."""

        return self.entity_kinds[0]

    @property
    def candidate_pool_sizes(self) -> Dict[str, Dict[str, int]]:
        """Return defensive per-kind retrieval pool statistics."""

        return {kind: dict(values) for kind, values in self._candidate_pool_sizes.items()}

    @property
    def candidate_pool_manifest(self) -> Dict[str, Any]:
        """Return the gold-free candidate-pool manifest, loading it after cache hits."""

        if not self._candidate_pool_manifest and self._candidate_pool_manifest_path.is_file():
            try:
                value = json.loads(self._candidate_pool_manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = {}
            if isinstance(value, dict):
                self._candidate_pool_manifest = value
        return deepcopy(self._candidate_pool_manifest)

    @property
    def candidate_pool_fingerprint(self) -> Optional[str]:
        """Return the actual ranked-pool fingerprint used by downstream artifacts."""

        value = self.candidate_pool_manifest.get("fingerprint")
        return str(value) if isinstance(value, str) and value else None

    def bind_candidate_provenance(
        self,
        *,
        data_lock: Any = None,
        spec_lock: Any = None,
        model_lock: Any = None,
    ) -> None:
        """Bind immutable data/model identities before candidates are loaded."""

        if self._candidates_generated or self._candidates is not None:
            raise RuntimeError("candidate provenance must be bound before candidates are loaded")
        if data_lock is not None:
            self._candidate_data_lock_binding = self._candidate_provenance_binding(
                data_lock,
                label="candidate_data_lock",
                require_sha256=True,
            )
        if spec_lock is not None:
            self._candidate_spec_lock_binding = self._candidate_provenance_binding(
                spec_lock,
                label="candidate_spec_lock",
                require_sha256=True,
            )
        if model_lock is not None:
            self._candidate_model_lock_binding = self._candidate_provenance_binding(
                model_lock,
                label="candidate_model_lock",
                require_sha256=False,
            )

    def _persisted_candidate_pool_fingerprint(self) -> Optional[str]:
        if not self._candidate_pool_manifest_path.is_file():
            return None
        try:
            value = json.loads(self._candidate_pool_manifest_path.read_text(encoding="utf-8")).get(
                "fingerprint"
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            return None
        return str(value) if isinstance(value, str) and value else None

    @property
    def reference(self) -> DataFrame:
        return self._reference

    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def num_workers(self) -> Optional[int]:
        return self._num_workers

    @property
    def source_graph(self) -> OntologyGraph:
        if self.source is None:
            self.log("Source ontology not loaded.", level="error")
            raise ValueError("Source ontology not loaded.")
        elif self._source_graph is None:
            self._source_graph = OntologyGraph(self.source)
        return self._source_graph

    @property
    def target_graph(self) -> OntologyGraph:
        if self.target is None:
            self.log("Target ontology not loaded.", level="error")
            raise ValueError("Target ontology not loaded.")
        elif self._target_graph is None:
            self._target_graph = OntologyGraph(self.target)
        return self._target_graph

    def entity_kind_for(
        self,
        iri: str,
        side: str,
        *,
        warn_unknown: bool = True,
    ) -> EntityKind:
        """Resolve an entity kind from a loaded source or target signature."""

        side_key = "src" if side in {"src", "source"} else "tgt"
        source = self.source if side_key == "src" else self.target
        if source is None:
            return self.primary_entity_kind
        index = (
            self._source_entity_kind_index if side_key == "src" else self._target_entity_kind_index
        )
        warning_key = (side_key, str(iri))

        def _warn(message: str) -> None:
            if warning_key in self._unknown_kind_warnings:
                return
            self._unknown_kind_warnings.add(warning_key)
            warnings.warn(message, UserWarning, stacklevel=3)
            self.log(message, level="warning")

        return infer_entity_kind(
            source,
            str(iri),
            primary=self.primary_entity_kind,
            index=index,
            warn=warn_unknown and warning_key not in self._unknown_kind_warnings,
            warning_callback=_warn,
        )

    @staticmethod
    def _kind_value(value: Any) -> str:
        return EntityKind(value).value

    def _ensure_mapping_kinds(
        self,
        df: Optional[DataFrame],
        *,
        label: str,
        filter_selected: bool = True,
        filter_cross_kind: bool = True,
    ) -> Optional[DataFrame]:
        """Add/infer kind columns and reject invalid mapping rows."""

        if df is None or not {"Src", "Tgt"}.issubset(df.columns):
            return df
        normalized = df.copy()
        if "SrcKind" not in normalized.columns:
            normalized["SrcKind"] = [
                self.entity_kind_for(str(iri), "src").value for iri in normalized["Src"]
            ]
        else:
            normalized["SrcKind"] = normalized["SrcKind"].map(self._kind_value)
        if "TgtKind" not in normalized.columns:
            normalized["TgtKind"] = [
                self.entity_kind_for(str(iri), "tgt").value for iri in normalized["Tgt"]
            ]
        else:
            normalized["TgtKind"] = normalized["TgtKind"].map(self._kind_value)

        keep = pd.Series(True, index=normalized.index)
        if filter_selected:
            allowed = {kind.value for kind in self.entity_kinds}
            keep &= normalized["SrcKind"].isin(allowed) & normalized["TgtKind"].isin(allowed)
        if filter_cross_kind:
            keep &= normalized["SrcKind"] == normalized["TgtKind"]
        removed = int((~keep).sum())
        if removed:
            self.log(
                f"#Filtered {removed} {label} rows outside configured within-kind pools.",
                level="warning",
            )
        return normalized.loc[keep].reset_index(drop=True)

    @staticmethod
    def _mapping_key_columns(*frames: Optional[DataFrame]) -> List[str]:
        columns = ["Src", "Tgt"]
        if all(
            frame is not None and {"SrcKind", "TgtKind"}.issubset(frame.columns) for frame in frames
        ):
            columns.extend(["SrcKind", "TgtKind"])
        return columns

    def _ignored_alignment_class_iris(self, side: str) -> set[str]:
        if not self.filter_ignored_alignment_classes:
            return set()

        side_key = "src" if side in {"src", "source"} else "tgt"
        cached = (
            self._source_ignored_alignment_classes
            if side_key == "src"
            else self._target_ignored_alignment_classes
        )
        if cached is not None:
            return cached

        source = self.source if side_key == "src" else self.target
        if source is None:
            return set()
        ignored = set(source.excluded_from_alignment())

        if side_key == "src":
            self._source_ignored_alignment_classes = ignored
        else:
            self._target_ignored_alignment_classes = ignored
        if ignored:
            self.log(
                f"#Ignored alignment classes ({side_key}): {len(ignored)}",
                level="info",
            )
        return ignored

    def _filter_ignored_iris(self, iris: Sequence[Any], side: str) -> List[Any]:
        if not self.filter_ignored_alignment_classes:
            return list(iris)
        ignored = self._ignored_alignment_class_iris(side)
        if not ignored:
            return list(iris)
        return [iri for iri in iris if str(iri) not in ignored]

    def _filter_candidates_ignored_classes(self, df: Optional[DataFrame]) -> Optional[DataFrame]:
        if df is None or df.empty or not self.filter_ignored_alignment_classes:
            return df
        if not {"Src", "Tgt"}.issubset(df.columns):
            return df
        src_ignored = self.source_ignored_alignment_classes
        tgt_ignored = self.target_ignored_alignment_classes
        if not src_ignored and not tgt_ignored:
            return df
        keep = ~(df["Src"].astype(str).isin(src_ignored) | df["Tgt"].astype(str).isin(tgt_ignored))
        removed = int((~keep).sum())
        if removed:
            self.log(
                f"#Filtered {removed} candidate rows with ignored alignment classes.",
                level="info",
            )
        return df.loc[keep].reset_index(drop=True)

    def _filter_mappings_ignored_classes(
        self, df: Optional[DataFrame], label: str
    ) -> Optional[DataFrame]:
        if df is None or df.empty or not self.filter_ignored_alignment_classes:
            return df
        if not {"Src", "Tgt"}.issubset(df.columns):
            return df
        src_ignored = self.source_ignored_alignment_classes
        tgt_ignored = self.target_ignored_alignment_classes
        if not src_ignored and not tgt_ignored:
            return df
        keep = ~(df["Src"].astype(str).isin(src_ignored) | df["Tgt"].astype(str).isin(tgt_ignored))
        removed = int((~keep).sum())
        if removed:
            self.log(
                f"#Filtered {removed} {label} rows with ignored alignment classes.",
                level="info",
            )
        return df.loc[keep].reset_index(drop=True)

    def get_exact_matches(self) -> None:
        """
        Extract exact equivalence mappings by normalizing labels on both ontologies
        and intersecting the normalized label sets. Saves a DataFrame with columns:
        ['Src','Tgt','Score'] where Score=1.0 for exact matches.
        """
        if self._source is None or self._target is None:
            self.log("Ontologies must be loaded before exact matching.", level="error")
            raise ValueError("Ontologies must be loaded first.")

        self.log(
            "#Building ontology graphs (or reusing cached) to get labels...",
            level="debug",
        )

        # Build normalized label→IRIs indexes. The compact key preserves the
        # historical exact-match behavior; the token key adds conservative
        # near-exact rescue for punctuation and word-order variants.
        self.log("#Indexing normalized labels for exact matching...", level="debug")

        def index_norm(map_, key_fn: Callable[[str], Any]) -> Dict[Any, List[str]]:
            idx: Dict[Any, List[str]] = {}
            for iri, labels in map_.items():
                for lab in labels:
                    nl = key_fn(lab)
                    if not nl:
                        continue
                    idx.setdefault(nl, []).append(iri)
            return idx

        def token_exact_key(label: str) -> Tuple[str, ...]:
            key = candidate_token_key(label)
            return key if len(key) >= 2 else tuple()

        def pairs_from_indexes(
            src_idx: Dict[Any, List[str]], tgt_idx: Dict[Any, List[str]]
        ) -> set[Tuple[str, str]]:
            pairs: set[Tuple[str, str]] = set()
            shared = set(src_idx.keys()).intersection(tgt_idx.keys())
            for key in shared:
                for s in src_idx[key]:
                    for t in tgt_idx[key]:
                        pairs.add((s, t))
            return pairs

        exact_rows: List[Dict[str, Any]] = []
        compact_count = 0
        near_exact_count = 0
        for kind in self.entity_kinds:
            if self.candidates is not None:
                candidates = self._ensure_mapping_kinds(
                    self.candidates,
                    label="candidate",
                )
                src_iris = set(
                    candidates.loc[candidates["SrcKind"] == kind.value, "Src"].astype(str)
                )
                tgt_iris = set(
                    candidates.loc[candidates["TgtKind"] == kind.value, "Tgt"].astype(str)
                )
            else:
                src_iris = set(self.source.entities(kind))
                tgt_iris = set(self.target.entities(kind))

            if hasattr(self, "eligible_source_iris"):
                src_iris = src_iris.intersection(self.eligible_source_iris)
            src_map = {iri: self.source_graph.get_labels(iri) for iri in src_iris}
            tgt_map = {iri: self.target_graph.get_labels(iri) for iri in tgt_iris}
            if self.filter_ignored_alignment_classes:
                src_ignored = self.source_ignored_alignment_classes
                tgt_ignored = self.target_ignored_alignment_classes
                src_map = {
                    iri: labels for iri, labels in src_map.items() if str(iri) not in src_ignored
                }
                tgt_map = {
                    iri: labels for iri, labels in tgt_map.items() if str(iri) not in tgt_ignored
                }

            compact_pairs = pairs_from_indexes(
                index_norm(src_map, OntologyGraph.normalize_label),
                index_norm(tgt_map, OntologyGraph.normalize_label),
            )
            token_pairs = pairs_from_indexes(
                index_norm(src_map, token_exact_key),
                index_norm(tgt_map, token_exact_key),
            )
            compact_count += len(compact_pairs)
            near_exact_count += len(token_pairs.difference(compact_pairs))
            exact_rows.extend(
                {
                    "Src": source_iri,
                    "Tgt": target_iri,
                    "Score": 1.0,
                    "SrcKind": kind.value,
                    "TgtKind": kind.value,
                }
                for source_iri, target_iri in sorted(compact_pairs.union(token_pairs))
            )

        self._exact_matches = pd.DataFrame(
            exact_rows,
            columns=["Src", "Tgt", "Score", "SrcKind", "TgtKind"],
        )
        self._exact_matches = self._filter_mappings_ignored_classes(
            self._exact_matches, label="exact"
        )
        self.log(
            "#Exact matches found: "
            f"{len(self._exact_matches)} "
            f"(compact={compact_count}, token_near_exact={near_exact_count})",
            level="info",
        )

    def load_ontologies(self, source_path: Path, target_path: Path) -> None:

        require_native_reasoner_support(self._reasoner_name)
        self._source_path = Path(source_path).resolve()
        self._target_path = Path(target_path).resolve()
        self._dataset_signature = None
        self._source_graph = None
        self._target_graph = None
        self._source_ignored_alignment_classes = None
        self._target_ignored_alignment_classes = None
        self._source_entity_kind_index = {}
        self._target_entity_kind_index = {}
        self._unknown_kind_warnings.clear()

        self.log(
            f"Loading ontologies: src={self._source_path} tgt={self._target_path}",
            level="info",
        )

        self._source = resolve_source(
            self._source_path,
            format=self._input_format,
            options=self._source_options,
        )
        self._configure_projector(self._source)
        self._configure_reasoner(self._source)
        self._source_entity_kind_index = build_entity_kind_index(self._source)

        self.log("#Loaded Source...", level="debug")

        self._target = resolve_source(
            self._target_path,
            format=self._input_format,
            options=self._target_options,
        )
        self._configure_projector(self._target)
        self._configure_reasoner(self._target)
        self._target_entity_kind_index = build_entity_kind_index(self._target)

        self.log("#Loaded Target...", level="debug")

    def ontology_stack_provenance(self) -> Dict[str, Any]:
        """Return source/target provenance without retaining another OWL graph."""

        def describe(source: KnowledgeSource | None) -> Dict[str, Any]:
            provider = getattr(source, "ontology_stack_provenance", None)
            if callable(provider):
                value = provider()
                if not isinstance(value, Mapping):
                    raise TypeError("ontology stack provenance must be a mapping")
                return dict(value)
            return {
                "schema_version": 1,
                "kind": "generic",
                "shared_snapshot": False,
            }

        return {
            "schema_version": 1,
            "cache": {
                "schema_version": _DATASET_CACHE_SCHEMA_VERSION,
                "ontology_backend_version": _ONTOLOGY_BACKEND_VERSION,
                "state": self._cache_state,
            },
            "source": describe(self._source),
            "target": describe(self._target),
        }

    def _configure_projector(self, source: KnowledgeSource) -> None:
        configure = getattr(source, "configure_projector", None)
        if callable(configure):
            configure(
                backend=self._projector_settings.backend,
                profile=self._projector_settings.profile,
            )

    def _configure_reasoner(self, source: KnowledgeSource) -> None:
        configure = getattr(source, "configure_reasoner", None)
        if callable(configure):
            configure(self._reasoner_name)

    @property
    def dataset_signature(self) -> Optional[str]:
        if self._dataset_signature is not None:
            return self._dataset_signature
        if self._source_path is None or self._target_path is None:
            return None
        self._dataset_signature = dataset_signature_for_paths(self._source_path, self._target_path)
        return self._dataset_signature

    def _cache_fingerprint_payload(self) -> Dict[str, Any]:
        payload = {
            "component": self.__class__.__name__,
            "dataset_signature": self.dataset_signature,
            "filter_exact_matches": self.filter_exact_matches,
            "drop_exact_match_sources": self.drop_exact_match_sources,
            "filter_ignored_alignment_classes": self.filter_ignored_alignment_classes,
            "cardinality": self._cardinality,
            "candidate_share_k": self._candidate_share_k,
            "candidate_generation_version": 6,
            "exact_prefilter_materialization_version": 2,
            "ignored_alignment_filter_version": 1,
            "ontology_backend_version": _ONTOLOGY_BACKEND_VERSION,
            "projector": projector_cache_identity(self._projector_settings),
            "input_format": self._input_format,
            "source_options": self._source_options,
            "target_options": self._target_options,
            "entity_kinds": [kind.value for kind in self.entity_kinds],
            "entity_kind_schema_version": 1,
            "candidate_generation_params": self._candidate_generation_params,
            "retrieval_artifacts": {
                name: {
                    "kind": artifact.kind,
                    "sha256": artifact.sha256,
                }
                for name, artifact in self._configured_retrieval_artifacts().items()
            },
            "only_taxonomy_hint": self._only_taxonomy_hint,
            "reasoner": self._reasoner_name,
            "reasoner_identity": reasoner_cache_identity(self._reasoner_name),
        }
        if self._candidate_data_lock_binding is not None:
            payload["candidate_data_lock"] = self._candidate_data_lock_binding
        if self._candidate_spec_lock_binding is not None:
            payload["candidate_spec_lock"] = self._candidate_spec_lock_binding
        if self._candidate_model_lock_binding is not None:
            payload["candidate_model_lock"] = self._candidate_model_lock_binding
        return payload

    def _configured_retrieval_artifacts(self) -> Dict[str, LocalRetrievalArtifact]:
        """Resolve enabled fitted models locally and bind cache identity to their bytes."""

        if self._retrieval_artifacts:
            for artifact in self._retrieval_artifacts.values():
                artifact.assert_unchanged()
            return dict(self._retrieval_artifacts)

        finetune = mapping_options(
            self._candidate_generation_params.get("encoder_finetune"),
            "candidate encoder_finetune",
        )
        if str(finetune.get("mode", "off")).lower() == "contrastive":
            self._retrieval_artifacts["encoder"] = resolve_local_retrieval_artifact(
                finetune.get("artifact"),
                expected_kind="contrastive_encoder",
                negative_policy=str(finetune.get("negative_policy", "complete_reference")),
            )

        cross_encoder = mapping_options(
            self._candidate_generation_params.get("cross_encoder"),
            "candidate cross_encoder",
        )
        if str(cross_encoder.get("mode", "off")).lower() == "on":
            self._retrieval_artifacts["cross_encoder"] = resolve_local_retrieval_artifact(
                cross_encoder.get("artifact"),
                expected_kind="cross_encoder",
            )
        for artifact in self._retrieval_artifacts.values():
            artifact.assert_unchanged()
        return dict(self._retrieval_artifacts)

    @property
    def cache_fingerprint(self) -> Optional[str]:
        payload = self._cache_fingerprint_payload()
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _load_cache_metadata(self) -> Dict[str, Any]:
        if not self._cache_meta_path.exists():
            return {}
        try:
            value = json.loads(self._cache_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_cache_metadata(self) -> None:
        candidate_recall_sha256 = None
        if self._candidates is not None:
            frames = []
            for role, frame in (("candidate", self._candidates), ("exact", self._exact_matches)):
                if frame is not None:
                    columns = [
                        name
                        for name in frame.columns
                        if name
                        in {
                            "Src",
                            "Tgt",
                            "SrcKind",
                            "TgtKind",
                            "cand_sim",
                            "Score",
                            "Scores",
                            "similarity",
                        }
                    ]
                    frames.append(frame[columns].assign(pool_role=role))
            path = self.output_path / "candidate_recall.tsv"
            temporary = path.with_suffix(".tsv.tmp")
            pd.concat(frames, ignore_index=True).to_csv(temporary, sep="\t", index=False)
            os.replace(temporary, path)
            candidate_recall_sha256 = file_provenance(path)["sha256"]
        payload = {
            "candidate_recall_sha256": candidate_recall_sha256,
            "cache_schema_version": _DATASET_CACHE_SCHEMA_VERSION,
            "ontology_backend_version": _ONTOLOGY_BACKEND_VERSION,
            "fingerprint": self.cache_fingerprint,
            "candidate_pool_fingerprint": self.candidate_pool_fingerprint,
            "dataset_signature": self.dataset_signature,
            "component": self.__class__.__name__,
            "candidate_encoder": {
                "identifier": self._active_candidate_config.get("lexical_encoder_name"),
                "requested_revision": self._active_candidate_config.get("encoder_revision"),
                "resolved_revision": self._active_candidate_config.get("encoder_resolved_revision"),
            },
        }
        self._cache_meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._cache_state = "cold"

    def load_candidates(
        self,
        file_path: Optional[Path] = None,
        top_k: Optional[int] = 100,
        lexical_encoder_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        encoder_revision: Optional[str] = None,
        encode_batch_size: Optional[int] = 512,
        search_batch_size: Optional[int] = 4096,
        use_amp: Optional[bool] = True,
        retrieval_strategy: str = "hybrid",
        fusion: Optional[Mapping[str, Any]] = None,
        aliases: Optional[Mapping[str, Any]] = None,
        adaptive_k: Optional[Mapping[str, Any]] = None,
        encoder_finetune: Optional[Mapping[str, Any]] = None,
        cross_encoder: Optional[Mapping[str, Any]] = None,
        multi_view: Optional[Mapping[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:

        if file_path is not None:
            fusion_config = candidate_config(self._candidate_generation_params, "fusion", fusion)
            adaptive_config = candidate_config(
                self._candidate_generation_params, "adaptive_k", adaptive_k
            )
            finetune_config = candidate_config(
                self._candidate_generation_params, "encoder_finetune", encoder_finetune
            )
            cross_encoder_config = candidate_config(
                self._candidate_generation_params, "cross_encoder", cross_encoder
            )
            multi_view_config = candidate_config(
                self._candidate_generation_params, "multi_view", multi_view
            )
            active_transform = (
                str(fusion_config.get("mode", "max")).lower() != "max"
                or bool(adaptive_config.get("enabled", False))
                or str(finetune_config.get("mode", "off")).lower() != "off"
                or str(multi_view_config.get("mode", "labels")).lower() != "labels"
            )
            if active_transform:
                raise ValueError(
                    "Retrieval experiment controls require generated candidates; "
                    "they cannot be applied to an opaque provided pool"
                )
            if not file_path.exists():
                self.log(f"Candidates file not found at {file_path}", level="error")
                raise FileNotFoundError(f"Candidates file not found at {file_path}")

            from exact.utils.mappings import candidate_table_views

            pairs, _ = candidate_table_views(read_table(str(file_path)))
            if hasattr(self, "eligible_source_iris"):
                pairs = pairs.loc[pairs["Src"].astype(str).isin(self.eligible_source_iris)]
            self._candidates = self._ensure_mapping_kinds(pairs, label="candidate")
            self._candidates = self._filter_candidates_ignored_classes(self._candidates)
            if str(cross_encoder_config.get("mode", "off")).lower() == "on":
                if CrossEncoder is None:
                    raise ImportError(
                        "Cross-encoder reranking requires sentence_transformers.CrossEncoder"
                    )
                counts = self._candidates.groupby("Src", sort=False).size()
                if not counts.empty and counts.max() > int(cross_encoder_config.get("top_k", 20)):
                    raise ValueError("Frozen cross-encoder pool exceeds its declared top_k bound")
                before = set(self._candidates[["Src", "Tgt"]].itertuples(index=False, name=None))
                if "cand_sim" not in self._candidates:
                    self._candidates["cand_sim"] = 0.0
                self._candidate_generation_params["cross_encoder"] = cross_encoder_config
                artifact = self._configured_retrieval_artifacts()["cross_encoder"]
                model = CrossEncoder(str(artifact.model_path), device=str(device or "cpu"))
                rows = self._rerank_with_cross_encoder(
                    self._candidates.to_dict("records"),
                    sources=list(self._candidates.Src.astype(str).unique()),
                    model=model,
                    top_k=int(counts.max()) if not counts.empty else 1,
                    adaptive_config={"enabled": False},
                    encode_batch_size=int(encode_batch_size or 32),
                )
                self._candidates = pd.DataFrame(
                    rows,
                    columns=list(self._candidates.columns)
                    + [
                        name
                        for name in (
                            "cand_sim_retrieval",
                            "cand_sim_cross_encoder",
                            "cand_channels",
                        )
                        if name not in self._candidates
                    ],
                )
                if (
                    set(self._candidates[["Src", "Tgt"]].itertuples(index=False, name=None))
                    != before
                ):
                    raise ValueError("Cross-encoder changed the frozen candidate pair universe")
            self._annotate_candidate_similarity_stats()
            self._active_candidate_config = {
                "origin": "provided",
                "path": str(file_path.resolve()),
                "fusion": fusion_config,
                "adaptive_k": adaptive_config,
                "encoder_finetune": finetune_config,
                "cross_encoder": cross_encoder_config,
                "multi_view": multi_view_config,
            }
            self._refresh_candidate_pool_manifest(
                origin="provided",
                candidate_file=file_path,
            )
            self.log("#Loaded Candidates...", level="info")

            if self.filter_exact_matches:
                self.log("Get Exact Matches...", level="info")
                self.get_exact_matches()

        else:
            self.log("#No Candidates file provided, generation candidates...", level="info")

            if self.filter_exact_matches:
                self.log("Get Exact Matches...", level="info")
                self.get_exact_matches()

            self.generate_candidates(
                top_k=top_k,
                lexical_encoder_name=lexical_encoder_name,
                encoder_revision=encoder_revision,
                encode_batch_size=encode_batch_size,
                search_batch_size=search_batch_size,
                use_amp=use_amp,
                retrieval_strategy=retrieval_strategy,
                fusion=fusion,
                aliases=aliases,
                adaptive_k=adaptive_k,
                encoder_finetune=encoder_finetune,
                cross_encoder=cross_encoder,
                multi_view=multi_view,
                device=device,
            )

    def generate_candidates(
        self,
        top_k: int = 100,
        lexical_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder_revision: Optional[str] = None,
        encode_batch_size: int = 512,
        search_batch_size: int = 4096,
        use_amp: bool = True,
        retrieval_strategy: str = "hybrid",
        fusion: Optional[Mapping[str, Any]] = None,
        aliases: Optional[Mapping[str, Any]] = None,
        adaptive_k: Optional[Mapping[str, Any]] = None,
        encoder_finetune: Optional[Mapping[str, Any]] = None,
        cross_encoder: Optional[Mapping[str, Any]] = None,
        multi_view: Optional[Mapping[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Build one-to-many candidate table using label embedding and lexical retrieval.
        """
        if self._source is None or self._target is None:
            self.log("Ontologies must be loaded before candidate generation.", level="error")
            raise ValueError("Ontologies not loaded.")

        strategy = str(retrieval_strategy or "hybrid").lower()
        if strategy not in {"primary_label", "hybrid"}:
            raise ValueError(
                f"Unsupported candidate retrieval_strategy={retrieval_strategy!r}; "
                "expected 'primary_label' or 'hybrid'."
            )

        fusion_config = candidate_config(self._candidate_generation_params, "fusion", fusion)
        alias_config = candidate_config(self._candidate_generation_params, "aliases", aliases)
        adaptive_config = candidate_config(
            self._candidate_generation_params, "adaptive_k", adaptive_k
        )
        finetune_config = candidate_config(
            self._candidate_generation_params, "encoder_finetune", encoder_finetune
        )
        cross_encoder_config = candidate_config(
            self._candidate_generation_params, "cross_encoder", cross_encoder
        )
        multi_view_config = candidate_config(
            self._candidate_generation_params, "multi_view", multi_view
        )
        self._candidate_generation_params.update(
            {
                "retrieval_strategy": strategy,
                "lexical_encoder_name": lexical_encoder_name,
                "encoder_revision": encoder_revision,
                "encode_batch_size": int(encode_batch_size),
                "search_batch_size": int(search_batch_size),
                "top_k": int(top_k),
                "use_amp": bool(use_amp),
            }
        )
        self._active_candidate_config = {
            key: self._candidate_generation_params.get(key)
            for key in (
                "retrieval_strategy",
                "lexical_encoder_name",
                "encoder_revision",
                "encode_batch_size",
                "search_batch_size",
                "top_k",
                "use_amp",
                "fusion",
                "aliases",
                "adaptive_k",
                "encoder_finetune",
                "cross_encoder",
                "multi_view",
            )
        }
        self._retrieval_artifacts = {}
        artifacts = self._configured_retrieval_artifacts()

        adaptive_enabled = bool(adaptive_config.get("enabled", False))
        final_pool_cap = (
            int(adaptive_config.get("k_max", top_k)) if adaptive_enabled else int(top_k)
        )
        final_pool_min = (
            int(adaptive_config.get("k_min", top_k)) if adaptive_enabled else int(top_k)
        )
        if adaptive_enabled and (final_pool_min < 1 or final_pool_cap < final_pool_min):
            raise ValueError("adaptive_k requires 1 <= k_min <= k_max")

        finetune_mode = str(finetune_config.get("mode", "off")).lower()
        if finetune_mode not in {"off", "contrastive"}:
            raise ValueError(f"Unsupported encoder_finetune mode: {finetune_mode!r}")
        encoder_name = lexical_encoder_name
        if finetune_mode == "contrastive":
            encoder_name = str(artifacts["encoder"].model_path)
        if encoder_name is None or not str(encoder_name).strip():
            raise ValueError("Candidate retrieval requires a configured encoder")

        cross_encoder_mode = str(cross_encoder_config.get("mode", "off")).lower()
        if cross_encoder_mode not in {"off", "on"}:
            raise ValueError(f"Unsupported candidate cross_encoder mode: {cross_encoder_mode!r}")
        cross_encoder_top_k = int(cross_encoder_config.get("top_k", top_k))
        if cross_encoder_mode == "on" and cross_encoder_top_k < final_pool_cap:
            raise ValueError(
                "cross_encoder.top_k must be at least the fixed/adaptive candidate-pool cap"
            )

        multi_view_mode = str(multi_view_config.get("mode", "labels")).lower()
        if multi_view_mode not in {"labels", "labels_types", "labels_relations"}:
            raise ValueError(f"Unsupported candidate multi_view mode: {multi_view_mode!r}")

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(
            f"#Candidate generation strategy={strategy} via torch top-k cosine…",
            level="info",
        )
        self.log(f"  Encoder: {encoder_name}", level="debug")
        self.log(f"  Device:  {dev}", level="debug")

        pools: List[Tuple[EntityKind, List[str], List[str]]] = []
        self._candidate_pool_sizes = {}
        for kind in self.entity_kinds:
            src_iris = list(self.source.entities(kind))
            if hasattr(self, "eligible_source_iris"):
                src_iris = [iri for iri in src_iris if str(iri) in self.eligible_source_iris]
            tgt_iris = list(self.target.entities(kind))
            if self.filter_ignored_alignment_classes:
                n_src_before = len(src_iris)
                n_tgt_before = len(tgt_iris)
                src_iris = self._filter_ignored_iris(src_iris, "src")
                tgt_iris = self._filter_ignored_iris(tgt_iris, "tgt")
                self.log(
                    (
                        f"## Alignment-use {kind.value} filter: "
                        f"source={len(src_iris)}/{n_src_before}, "
                        f"target={len(tgt_iris)}/{n_tgt_before}"
                    ),
                    level="debug",
                )
            self.log(
                f"## Source {kind.value}: {len(src_iris)} | "
                f"Target {kind.value}: {len(tgt_iris)}",
                level="debug",
            )

            if (
                not self.cardinality_1_to_many
                and self.drop_exact_match_sources
                and self._exact_matches is not None
            ):
                exact_rows = self._exact_matches
                if "SrcKind" in exact_rows.columns:
                    exact_rows = exact_rows[exact_rows["SrcKind"] == kind.value]
                exact_sources = set(exact_rows["Src"].dropna().astype(str))
                n_src_before = len(src_iris)
                src_iris = [iri for iri in src_iris if str(iri) not in exact_sources]
                n_removed = n_src_before - len(src_iris)
                self.log(
                    (
                        f"## Candidate-generation {kind.value} sources: "
                        f"{len(src_iris)}/{n_src_before} (skipped_exact={n_removed})"
                    ),
                    level="debug",
                )
            self._candidate_pool_sizes[kind.value] = {
                "source_entities": len(src_iris),
                "target_entities": len(tgt_iris),
                "source_labels": 0,
                "target_labels": 0,
            }
            pools.append((kind, src_iris, tgt_iris))

        if not any(src_iris and tgt_iris for _, src_iris, tgt_iris in pools):
            self._candidates = pd.DataFrame(
                columns=[
                    "Src",
                    "Tgt",
                    "SrcKind",
                    "TgtKind",
                    "Label",
                    "cand_sim",
                    "cand_sim_semantic",
                    "cand_sim_lexical",
                    "cand_channels",
                ]
            )
            self._candidates_generated = True
            self._refresh_candidate_pool_manifest(origin="generated")
            return

        encoder_revision_kwargs = (
            {"revision": encoder_revision} if encoder_revision is not None else {}
        )
        st = SentenceTransformer(str(encoder_name), device=str(dev), **encoder_revision_kwargs)
        resolved_encoder_revision = _sentence_transformer_resolved_revision(st, encoder_revision)
        self._active_candidate_config["encoder_resolved_revision"] = resolved_encoder_revision
        cross_encoder_model = None
        if cross_encoder_mode == "on":
            if CrossEncoder is None:
                raise ImportError(
                    "cross-encoder reranking requires sentence_transformers.CrossEncoder"
                )
            cross_encoder_model = CrossEncoder(
                str(artifacts["cross_encoder"].model_path),
                device=str(dev),
            )
        all_rows: List[Dict[str, object]] = []
        for kind, src_iris, tgt_iris in pools:
            if not src_iris or not tgt_iris:
                continue
            all_rows.extend(
                self._candidate_rows_for_kind(
                    kind=kind,
                    src_iris=src_iris,
                    tgt_iris=tgt_iris,
                    strategy=strategy,
                    encoder=st,
                    top_k=int(top_k),
                    encode_batch_size=int(encode_batch_size),
                    search_batch_size=int(search_batch_size),
                    use_amp=bool(use_amp),
                    device=dev,
                    fusion_config=fusion_config,
                    alias_config=alias_config,
                    adaptive_config=adaptive_config,
                    cross_encoder_config=cross_encoder_config,
                    cross_encoder_model=cross_encoder_model,
                    multi_view_mode=multi_view_mode,
                )
            )

        self.log("  Assembling candidate DataFrame…", level="debug")
        candidate_columns = [
            "Src",
            "Tgt",
            "Label",
            "cand_sim",
            "cand_sim_semantic",
            "cand_sim_lexical",
            "cand_channels",
        ]
        if cross_encoder_mode == "on":
            candidate_columns.extend(
                [
                    "cand_sim_retrieval",
                    "cand_sim_cross_encoder",
                ]
            )
        candidate_columns.extend(["SrcKind", "TgtKind"])
        cand_df = pd.DataFrame(
            all_rows,
            columns=candidate_columns,
        )

        self._candidates = self._filter_candidates_ignored_classes(cand_df)
        self._annotate_candidate_similarity_stats()
        self._candidates_generated = True
        self._refresh_candidate_pool_manifest(origin="generated")
        self.log(
            f"#Candidate generation complete: {len(self._candidates)} rows "
            f"(pool cap {final_pool_cap} per source and kind).",
            level="debug",
        )

    def _candidate_rows_for_kind(
        self,
        *,
        kind: EntityKind,
        src_iris: List[str],
        tgt_iris: List[str],
        strategy: str,
        encoder: SentenceTransformer,
        top_k: int,
        encode_batch_size: int,
        search_batch_size: int,
        use_amp: bool,
        device: torch.device,
        fusion_config: Mapping[str, Any],
        alias_config: Mapping[str, Any],
        adaptive_config: Mapping[str, Any],
        cross_encoder_config: Mapping[str, Any],
        cross_encoder_model: Any,
        multi_view_mode: str,
    ) -> List[Dict[str, object]]:
        """Build one isolated semantic/lexical retrieval index for a kind."""

        adaptive_enabled = bool(adaptive_config.get("enabled", False))
        final_pool_cap = (
            int(adaptive_config.get("k_max", top_k)) if adaptive_enabled else int(top_k)
        )
        cross_encoder_enabled = str(cross_encoder_config.get("mode", "off")).lower() == "on"
        initial_pool_k = (
            int(cross_encoder_config.get("top_k", final_pool_cap))
            if cross_encoder_enabled
            else final_pool_cap
        )
        label_scope = "primary labels" if strategy == "primary_label" else "all labels"
        self.log(f"  Extracting {label_scope} for {kind.value} entities…", level="debug")
        if strategy == "primary_label":
            src_labels_by_iri = {
                iri: [self.source_graph.get_primary_label(iri)] for iri in src_iris
            }
            tgt_labels_by_iri = {
                iri: [self.target_graph.get_primary_label(iri)] for iri in tgt_iris
            }
            src_lexical_texts_by_iri = src_labels_by_iri
            tgt_lexical_texts_by_iri = tgt_labels_by_iri
            channel_k = initial_pool_k
        else:
            src_labels_by_iri = {iri: self.source_graph.get_labels(iri) for iri in src_iris}
            tgt_labels_by_iri = {iri: self.target_graph.get_labels(iri) for iri in tgt_iris}
            src_lexical_texts_by_iri = {
                iri: self._candidate_texts_for_iri(self.source_graph, iri, "src", alias_config)
                for iri in src_iris
            }
            tgt_lexical_texts_by_iri = {
                iri: self._candidate_texts_for_iri(self.target_graph, iri, "tgt", alias_config)
                for iri in tgt_iris
            }
            channel_k = max(initial_pool_k * 3, 30)

        if kind == EntityKind.INDIVIDUAL and multi_view_mode != "labels":
            src_views = self._candidate_multiview_texts_by_iri(
                src_iris,
                side="src",
                mode=multi_view_mode,
            )
            tgt_views = self._candidate_multiview_texts_by_iri(
                tgt_iris,
                side="tgt",
                mode=multi_view_mode,
            )
            src_lexical_texts_by_iri = {
                iri: list(src_lexical_texts_by_iri.get(iri, ())) + src_views.get(iri, [])
                for iri in src_iris
            }
            tgt_lexical_texts_by_iri = {
                iri: list(tgt_lexical_texts_by_iri.get(iri, ())) + tgt_views.get(iri, [])
                for iri in tgt_iris
            }

        src_records = make_candidate_labels(src_iris, src_labels_by_iri, kind=kind)
        tgt_records = make_candidate_labels(tgt_iris, tgt_labels_by_iri, kind=kind)
        src_lexical_records = make_candidate_labels(src_iris, src_lexical_texts_by_iri, kind=kind)
        tgt_lexical_records = make_candidate_labels(tgt_iris, tgt_lexical_texts_by_iri, kind=kind)
        self._candidate_pool_sizes[kind.value].update(
            {
                "source_labels": len(src_lexical_records),
                "target_labels": len(tgt_lexical_records),
            }
        )

        self.log(
            f"  Candidate labels: source={len(src_records)} target={len(tgt_records)}; "
            f"lexical texts: source={len(src_lexical_records)} target={len(tgt_lexical_records)}; "
            f"channel_k={channel_k}",
            level="debug",
        )
        semantic_scores = self._semantic_label_pair_scores(
            src_records=src_records,
            tgt_records=tgt_records,
            encoder=encoder,
            top_k=channel_k,
            encode_batch_size=encode_batch_size,
            search_batch_size=search_batch_size,
            use_amp=use_amp,
            device=device,
        )
        lexical_scores = {}
        if strategy == "hybrid":
            self.log("  Running lexical token/char candidate retrieval…", level="debug")
            lexical_scores = lexical_candidate_pair_scores(
                src_records=src_lexical_records,
                tgt_records=tgt_lexical_records,
                per_source_limit=channel_k,
                fusion_config=fusion_config,
            )

        rows = rank_channel_scores(
            sources=[str(iri) for iri in src_iris],
            semantic_scores=semantic_scores,
            lexical_scores=lexical_scores,
            top_k=initial_pool_k if cross_encoder_enabled else int(top_k),
            fusion_config=fusion_config,
            adaptive_config=None if cross_encoder_enabled else adaptive_config,
        )
        if cross_encoder_enabled:
            rows = self._rerank_with_cross_encoder(
                rows,
                sources=[str(iri) for iri in src_iris],
                model=cross_encoder_model,
                top_k=int(top_k),
                adaptive_config=adaptive_config,
                encode_batch_size=encode_batch_size,
            )
        for row in rows:
            row["SrcKind"] = kind.value
            row["TgtKind"] = kind.value
        self._candidate_pool_sizes[kind.value].update(
            {
                "candidate_rows": len(rows),
                "covered_sources": len({str(row["Src"]) for row in rows}),
            }
        )
        return rows

    def _candidate_multiview_texts_by_iri(
        self,
        iris: Sequence[str],
        *,
        side: str,
        mode: str,
    ) -> Dict[str, List[str]]:
        """Build reference-free retrieval views for individual entities only."""

        source = self.source if side == "src" else self.target
        if mode == "labels":
            return {str(iri): [] for iri in iris}

        if mode == "labels_types":
            result: Dict[str, List[str]] = {}
            for iri in iris:
                type_iris: set[str] = set()
                for type_iri in source.direct_parents(str(iri), EntityKind.INDIVIDUAL):
                    type_iris.add(str(type_iri))
                    type_iris.update(
                        str(parent)
                        for parent in source.direct_parents(str(type_iri), EntityKind.CLASS)
                    )
                result[str(iri)] = [
                    f"type {label}"
                    for type_iri in sorted(type_iris)
                    for label in source.labels(type_iri)
                    if str(label).strip()
                ]
            return result

        if mode != "labels_relations":
            raise ValueError(f"Unsupported candidate multi_view mode: {mode!r}")

        requested = {str(iri) for iri in iris}
        kind_index = (
            self._source_entity_kind_index if side == "src" else self._target_entity_kind_index
        )
        texts: Dict[str, set[str]] = {iri: set() for iri in requested}
        for edge in source.projection_edges(method="owl2vecstar", include_literals=False):
            src = str(edge.src)
            dst = str(edge.dst)
            if (
                kind_index.get(src) != EntityKind.INDIVIDUAL
                or kind_index.get(dst) != EntityKind.INDIVIDUAL
            ):
                continue
            relation_labels = source.labels(str(edge.rel))
            if src in requested:
                for relation_label in relation_labels:
                    for neighbor_label in source.labels(dst):
                        texts[src].add(f"out {relation_label} {neighbor_label}")
            if dst in requested:
                for relation_label in relation_labels:
                    for neighbor_label in source.labels(src):
                        texts[dst].add(f"in {relation_label} {neighbor_label}")

        cap = max(1, int(getattr(self, "max_object_triples", 48)))
        return {
            iri: sorted(values, key=lambda value: (normalize_candidate_text(value), value))[:cap]
            for iri, values in texts.items()
        }

    def _rerank_with_cross_encoder(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        sources: Sequence[str],
        model: Any,
        top_k: int,
        adaptive_config: Mapping[str, Any],
        encode_batch_size: int,
    ) -> List[Dict[str, object]]:
        """Rerank a bounded fused pool with a validated local cross-encoder."""

        if not rows:
            return []
        if model is None:
            raise RuntimeError("cross_encoder mode is on but no local model was loaded")

        pairs = [
            (
                self.source_graph.get_primary_label(str(row["Src"])),
                self.target_graph.get_primary_label(str(row["Tgt"])),
            )
            for row in rows
        ]
        predictions = np.asarray(
            model.predict(
                pairs,
                batch_size=int(encode_batch_size),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        )
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions[:, 0]
        if predictions.ndim != 1 or predictions.shape[0] != len(rows):
            raise ValueError(
                "Cross-encoder artifact must emit exactly one scalar score per candidate pair"
            )
        if not np.isfinite(predictions).all():
            raise ValueError("Cross-encoder artifact emitted a non-finite candidate score")

        by_source: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for raw_row, prediction in zip(rows, predictions):
            row = dict(raw_row)
            row["cand_sim_retrieval"] = float(row["cand_sim"])
            row["cand_sim_cross_encoder"] = float(prediction)
            row["cand_sim"] = float(prediction)
            channels = str(row.get("cand_channels", "") or "")
            row["cand_channels"] = f"{channels}|cross_encoder" if channels else "cross_encoder"
            by_source[str(row["Src"])].append(row)

        reranked: List[Dict[str, object]] = []
        for src in sources:
            candidates = by_source.get(str(src), [])
            candidates.sort(
                key=lambda row: (
                    -float(row["cand_sim_cross_encoder"]),
                    -float(row["cand_sim_retrieval"]),
                    str(row["Tgt"]),
                )
            )
            limit = adaptive_candidate_count(
                [float(row["cand_sim_cross_encoder"]) for row in candidates],
                top_k=top_k,
                adaptive_config=adaptive_config,
            )
            reranked.extend(candidates[:limit])
        return reranked

    def _refresh_candidate_pool_manifest(
        self,
        *,
        origin: str,
        candidate_file: Optional[Path] = None,
        frame: Optional[DataFrame] = None,
        persist: bool = True,
    ) -> None:
        """Build and atomically persist a gold-free ranked-pool manifest."""

        candidate_frame = self._candidates if frame is None else frame
        if candidate_frame is None:
            return

        frame = candidate_frame
        kind_names = list(dict.fromkeys(kind.value for kind in self.entity_kinds))
        if "SrcKind" in frame.columns:
            for value in frame["SrcKind"].dropna().astype(str):
                if value not in kind_names:
                    kind_names.append(value)

        per_kind: Dict[str, Dict[str, Any]] = {}
        all_counts: List[int] = []
        for kind_name in kind_names:
            if "SrcKind" in frame.columns:
                kind_frame = frame[frame["SrcKind"].astype(str) == kind_name]
            else:
                kind_frame = frame
            pool_meta = self._candidate_pool_sizes.get(kind_name, {})
            try:
                kind = EntityKind(kind_name)
            except ValueError:
                kind = None
            expected_sources = int(pool_meta.get("source_entities", 0))
            if expected_sources <= 0 and kind is not None and self.source is not None:
                expected_sources = len(self.source.entities(kind))
            kind_payload, counts = self._candidate_kind_manifest(
                kind_frame,
                expected_sources=expected_sources,
            )
            per_kind[kind_name] = kind_payload
            all_counts.extend(counts)

        artifacts = self._configured_retrieval_artifacts()
        encoder_identifier = self._active_candidate_config.get("lexical_encoder_name")
        encoder_record: Dict[str, Any] = {
            "identifier": str(encoder_identifier) if encoder_identifier is not None else None,
            "requested_revision": self._active_candidate_config.get("encoder_revision"),
            "resolved_revision": self._active_candidate_config.get("encoder_resolved_revision"),
            "identifier_sha256": (
                hashlib.sha256(str(encoder_identifier).encode("utf-8")).hexdigest()
                if encoder_identifier is not None
                else None
            ),
        }
        if "encoder" in artifacts:
            encoder_record["artifact"] = artifacts["encoder"].provenance()
        if self._candidate_model_lock_binding is not None:
            encoder_record["model_lock"] = deepcopy(self._candidate_model_lock_binding)

        inputs: Dict[str, Any] = {
            "source": self._candidate_input_provenance(self._source_path),
            "target": self._candidate_input_provenance(self._target_path),
            "data_lock": deepcopy(self._candidate_data_lock_binding),
        }
        if self._candidate_spec_lock_binding is not None:
            inputs["spec_lock"] = deepcopy(self._candidate_spec_lock_binding)
        if candidate_file is not None:
            inputs["candidate_file"] = self._candidate_input_provenance(candidate_file)

        ontology_inventory = {
            "per_kind": {
                kind.value: {
                    "source_entities": len(self.source.entities(kind)),
                    "target_entities": len(self.target.entities(kind)),
                }
                for kind in MATCHABLE_ENTITY_KINDS
            }
        }

        summary = self._candidate_count_summary(all_counts)
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "origin": str(origin),
            "retrieval_config": self._json_safe(self._active_candidate_config),
            "models": {
                "encoder": encoder_record,
                "cross_encoder": (
                    artifacts["cross_encoder"].provenance()
                    if "cross_encoder" in artifacts
                    else None
                ),
            },
            "inputs": inputs,
            "ontology_inventory": ontology_inventory,
            "per_kind": per_kind,
            "gold_free_summary": {
                **summary,
                "candidate_pairs": int(len(frame)),
                "covered_sources": (
                    int(frame["Src"].astype(str).nunique()) if "Src" in frame.columns else 0
                ),
            },
        }
        fingerprint_config = deepcopy(self._active_candidate_config)
        fingerprint_config.pop("path", None)
        for config_name, artifact_name in (
            ("encoder_finetune", "encoder"),
            ("cross_encoder", "cross_encoder"),
        ):
            config_value = fingerprint_config.get(config_name)
            if isinstance(config_value, Mapping) and artifact_name in artifacts:
                fingerprint_config[config_name] = {
                    **dict(config_value),
                    "artifact": artifacts[artifact_name].sha256,
                }
        fingerprint_basis = {
            "schema_version": payload["schema_version"],
            "origin": payload["origin"],
            "retrieval_config": self._json_safe(fingerprint_config),
            "model_hashes": {name: artifact.sha256 for name, artifact in artifacts.items()},
            "input_hashes": {
                name: value.get("sha256")
                for name, value in inputs.items()
                if isinstance(value, Mapping)
            },
            "per_kind": per_kind,
        }
        if self._candidate_model_lock_binding is not None:
            fingerprint_basis["model_lock"] = self._candidate_model_lock_binding
        canonical = json.dumps(
            self._json_safe(fingerprint_basis),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        payload["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self._candidate_pool_manifest = self._json_safe(payload)
        if not persist:
            return
        temporary = self._candidate_pool_manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._candidate_pool_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self._candidate_pool_manifest_path)

    def _candidate_kind_manifest(
        self,
        frame: DataFrame,
        *,
        expected_sources: int,
    ) -> Tuple[Dict[str, Any], List[int]]:
        source_groups = (
            {
                str(src): source_rows
                for src, source_rows in frame.groupby(
                    frame["Src"].astype(str),
                    sort=False,
                )
            }
            if "Src" in frame.columns and not frame.empty
            else {}
        )
        grouped_counts = {src: len(source_rows) for src, source_rows in source_groups.items()}
        counts = [int(value) for value in grouped_counts.values()]
        counts.extend([0] * max(0, int(expected_sources) - len(counts)))

        digest = hashlib.sha256()
        score_columns = [
            column
            for column in (
                "cand_sim",
                "cand_sim_semantic",
                "cand_sim_lexical",
                "cand_sim_retrieval",
                "cand_sim_cross_encoder",
                "cand_channels",
            )
            if column in frame.columns
        ]
        if source_groups:
            for src in sorted(source_groups):
                source_rows = source_groups[src]
                for rank, (_, row) in enumerate(source_rows.iterrows(), start=1):
                    record: Dict[str, Any] = {
                        "src": src,
                        "tgt": str(row["Tgt"]),
                        "rank": rank,
                    }
                    for column in score_columns:
                        value = row[column]
                        if pd.isna(value):
                            record[column] = None
                        elif isinstance(value, (float, np.floating)):
                            record[column] = float(value).hex()
                        else:
                            record[column] = str(value)
                    digest.update(
                        json.dumps(
                            record,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=True,
                        ).encode("utf-8")
                    )
                    digest.update(b"\n")

        return (
            {
                **self._candidate_count_summary(counts),
                "candidate_pairs": int(len(frame)),
                "covered_sources": len(grouped_counts),
                "pool_sha256": digest.hexdigest(),
            },
            counts,
        )

    @staticmethod
    def _candidate_count_summary(counts: Sequence[int]) -> Dict[str, Any]:
        if not counts:
            return {
                "source_entities": 0,
                "mean_pool_size": 0.0,
                "pool_size_q50": 0.0,
                "pool_size_q90": 0.0,
                "pool_size_q95": 0.0,
                "pool_size_max": 0,
            }
        values = np.asarray(counts, dtype=np.float64)
        return {
            "source_entities": int(len(counts)),
            "mean_pool_size": float(values.mean()),
            "pool_size_q50": float(np.quantile(values, 0.50)),
            "pool_size_q90": float(np.quantile(values, 0.90)),
            "pool_size_q95": float(np.quantile(values, 0.95)),
            "pool_size_max": int(values.max()),
        }

    @staticmethod
    def _candidate_input_provenance(path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if path is None:
            return None
        resolved = Path(path).expanduser().resolve()
        if resolved.is_file():
            return file_provenance(resolved)
        return {
            "path": str(resolved),
            "sha256": None,
            "bytes": None,
            "rows": None,
        }

    @classmethod
    def _candidate_provenance_binding(
        cls,
        value: Any,
        *,
        label: str,
        require_sha256: bool,
    ) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if isinstance(value, (str, Path)):
            text = str(value).strip()
            if not text:
                return None
            if len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text):
                return {"sha256": text.lower()}
            path = Path(text).expanduser()
            if path.is_file():
                return file_provenance(path.resolve())
            raise ValueError(
                f"{label} must be an existing lock file, SHA-256 digest, or immutable mapping"
            )
        if not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a path, SHA-256 digest, or mapping")

        binding = cls._json_safe(dict(value))
        sha = str(binding.get("sha256", binding.get("fingerprint", "")) or "").lower()
        valid_sha = len(sha) == 64 and all(char in "0123456789abcdef" for char in sha)
        if require_sha256:
            if not valid_sha:
                raise ValueError(f"{label} must declare an immutable SHA-256 digest")
            return binding

        revision = str(binding.get("revision", "") or "").lower()
        identifier = str(binding.get("identifier", binding.get("model_id", "")) or "").strip()
        immutable_revision = len(revision) in {40, 64} and all(
            char in "0123456789abcdef" for char in revision
        )
        if not valid_sha and not (identifier and immutable_revision):
            raise ValueError(
                f"{label} must declare sha256 or identifier/model_id plus an immutable commit"
            )
        return binding

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _candidate_texts_for_iri(
        self,
        graph: OntologyGraph,
        iri: str,
        side: str,
        alias_config: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        labels = list(graph.get_labels(iri) or [])
        texts: List[str] = []
        seen = set()

        def _add(text: Any) -> None:
            normalized = OntologyGraph.normalize_label(str(text or ""))
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            texts.append(str(text).strip())

        for label in labels:
            _add(label)

        source = self.source if side == "src" else self.target
        annotation_values = [
            (value.property_iri, value.value.strip()) for value in source.attributes(str(iri))
        ]

        for literal in select_candidate_annotation_literals(
            annotation_values,
            seen_normalized=set(seen),
            overall_cap=int((alias_config or {}).get("overall_cap", 12)),
            alias_config=alias_config,
        ):
            _add(literal)
        return texts

    @staticmethod
    def _candidate_annotation_priority(literal: str, prop_iri: str) -> Optional[float]:
        return candidate_annotation_priority(literal, prop_iri)

    def _semantic_label_pair_scores(
        self,
        src_records: Sequence[Any],
        tgt_records: Sequence[Any],
        encoder: SentenceTransformer,
        top_k: int,
        encode_batch_size: int,
        search_batch_size: int,
        use_amp: bool,
        device: torch.device,
    ) -> Dict[Tuple[str, str], float]:
        if not src_records or not tgt_records or top_k <= 0:
            return {}

        self.log("  Encoding labels…", level="debug")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        src_texts = [record.text for record in src_records]
        tgt_texts = [record.text for record in tgt_records]
        src_emb = encoder.encode(
            src_texts,
            batch_size=encode_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        tgt_emb = encoder.encode(
            tgt_texts,
            batch_size=encode_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        self.log("  Running torch Top-K cosine search…", level="debug")
        idxs, sims = self.topk_cosine_search_torch(
            E_src=src_emb.astype(np.float32, copy=False),
            E_tgt=tgt_emb.astype(np.float32, copy=False),
            top_k=top_k,
            batch_size_src=search_batch_size,
            device=device,
            amp=use_amp,
        )

        scores: Dict[Tuple[str, str], float] = {}
        for i, src_record in enumerate(src_records):
            for tgt_idx, score in zip(idxs[i], sims[i]):
                tgt_record = tgt_records[int(tgt_idx)]
                key = (str(src_record.iri), str(tgt_record.iri))
                current = scores.get(key, -1.0)
                if float(score) > current:
                    scores[key] = float(score)
        return scores

    @torch.inference_mode()
    def topk_cosine_search_torch(
        self,
        E_src: np.ndarray,  # [Ns, d], float32
        E_tgt: np.ndarray,  # [Nt, d], float32
        top_k: int = 10,
        batch_size_src: int = 2048,  # tune per VRAM; 2–8K typical for d=384
        device: Optional[torch.device] = None,
        amp: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (indices, scores) where:
        indices: [Ns, top_k] target indices
        scores:  [Ns, top_k] cosine similarities
        """
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_autocast = amp and dev.type == "cuda"
        autocast_ctx = torch.amp.autocast if use_autocast else nullcontext
        autocast_kwargs = {"device_type": dev.type} if use_autocast else {}

        # Move target once, normalize
        T = torch.from_numpy(E_tgt).to(dev, non_blocking=True)
        T = torch.nn.functional.normalize(T, dim=-1)

        Ns, d = E_src.shape
        if E_src.shape[0] == 0 or E_tgt.shape[0] == 0 or top_k <= 0:
            return (
                np.empty((E_src.shape[0], 0), dtype=np.int64),
                np.empty((E_src.shape[0], 0), dtype=np.float32),
            )

        K = min(int(top_k), int(E_tgt.shape[0]))
        all_idx = np.empty((Ns, K), dtype=np.int64)
        all_scr = np.empty((Ns, K), dtype=np.float32)

        n_batches = math.ceil(Ns / batch_size_src)
        for b in range(n_batches):
            s0 = b * batch_size_src
            s1 = min((b + 1) * batch_size_src, Ns)
            S = torch.from_numpy(E_src[s0:s1]).to(dev, non_blocking=True)
            S = torch.nn.functional.normalize(S, dim=-1)

            # cosine = inner product of normalized vectors
            # [Bs, d] @ [d, Nt] -> [Bs, Nt]
            with autocast_ctx(**autocast_kwargs):
                sims = S @ T.T

            vals, idx = torch.topk(sims, k=K, dim=-1, largest=True, sorted=True)
            all_idx[s0:s1] = idx.cpu().numpy()
            all_scr[s0:s1] = vals.to(dtype=torch.float32).cpu().numpy()

        return all_idx, all_scr

    def _annotate_candidate_similarity_stats(self) -> None:
        if self._candidates is None:
            return
        if "cand_sim" not in self._candidates.columns:
            return
        group_columns: str | List[str] = "Src"
        if (
            "SrcKind" in self._candidates.columns
            and self._candidates["SrcKind"].nunique(dropna=False) > 1
        ):
            group_columns = ["Src", "SrcKind"]
        groups = self._candidates.groupby(group_columns, sort=False)
        self._candidates["cand_sim_src_mean"] = groups["cand_sim"].transform("mean")

        src_sum = groups["cand_sim"].transform("sum")
        eps = np.finfo(np.float32).eps
        src_sum = src_sum.replace(0.0, eps)
        self._candidates["cand_sim_prob"] = self._candidates["cand_sim"] / src_sum

        k_share = max(1, int(getattr(self, "_candidate_share_k", 1)))
        ranks = groups.cumcount()
        top_mask = ranks < k_share
        top_contribution = self._candidates["cand_sim_prob"].where(top_mask, 0.0)
        if isinstance(group_columns, list):
            share_top = top_contribution.groupby(
                [self._candidates[column] for column in group_columns], sort=False
            ).transform("sum")
        else:
            share_top = top_contribution.groupby(
                self._candidates[group_columns], sort=False
            ).transform("sum")
        share_rest = (1.0 - share_top).clip(lower=0.0)
        self._candidates["cand_share_top"] = share_top
        self._candidates["cand_share_rest"] = share_rest
        self._candidates["cand_share_log_ratio"] = np.log((share_top + eps) / (share_rest + eps))

    def load_reference(self, file_path: Path) -> None:

        self._reference = read_table(file_path)
        if len(self._reference.columns) < 2:
            raise ValueError("reference tables need source and target columns")
        source, target = self._reference.columns[:2]
        self._reference = self._reference.rename(columns={source: "Src", target: "Tgt"})
        if "Relation" in self._reference:
            # Relations are labels; an optional confidence Score remains separate.
            self._reference = self._reference.rename(columns={"Relation": "Label"})
        elif "Label" not in self._reference:
            extra = [
                name
                for name in self._reference.columns
                if name not in {"Src", "Tgt", "Score", "SrcKind", "TgtKind"}
            ]
            if len(extra) == 1:
                self._reference = self._reference.rename(columns={extra[0]: "Label"})
            elif not extra:
                self._reference["Label"] = "="
            else:
                raise ValueError(f"ambiguous reference label columns: {extra}")
        self._reference = self._ensure_mapping_kinds(
            self._reference,
            label="reference",
        )
        self._reference = self._filter_mappings_ignored_classes(self._reference, label="reference")
        self.log("#Loaded Reference...", level="debug")

    def save(self) -> Path:

        if self.dataframe is None:
            self.log("Dataset is empty.", level="error")
            raise ValueError("Dataset is empty.")
        if self._source_restriction_active:
            if self._df_save_path.exists():
                self.log(
                    "#Source-restricted in-memory view; preserving full dataset cache.",
                    level="debug",
                )
                return self._df_save_path
            raise RuntimeError(
                "A source-restricted view cannot be persisted as the full dataset cache"
            )

        if self.has_cache():
            self.log("#Dataset already saved; skipping...", level="debug")
            return self._df_save_path

        self.dataframe.to_csv(str(self._df_save_path), index=False)
        self._write_cache_metadata()
        prepared_cache.publish(self.output_path, self.cache_fingerprint)

        self.log(f"#Dataset saved to {self._df_save_path}", level="debug")

        return self._df_save_path

    def load(self):
        self._df = pd.read_csv(self._df_save_path, converters={"Features": literal_eval})
        self._df = self._ensure_mapping_kinds(
            self._df,
            label="cached dataset",
            filter_selected=False,
        )

        self.log("#Loaded cached dataset...", level="info")
        if "cand_sim" in self._df.columns:
            self._candidates_generated = True

    def has_cache(self) -> bool:
        shared = False
        if self._cache_ok and not self._df_save_path.exists():
            shared = prepared_cache.restore(self.output_path, self.cache_fingerprint)
        if self._cache_ok and self._df_save_path.exists():
            meta = self._load_cache_metadata()
            cache_schema = meta.get("cache_schema_version")
            ontology_backend = meta.get("ontology_backend_version")
            versions_current = (
                type(cache_schema) is int
                and cache_schema == _DATASET_CACHE_SCHEMA_VERSION
                and type(ontology_backend) is int
                and ontology_backend == _ONTOLOGY_BACKEND_VERSION
            )
            pool_fingerprint = self._persisted_candidate_pool_fingerprint()
            pool_manifest_current = (
                isinstance(pool_fingerprint, str)
                and len(pool_fingerprint) == 64
                and meta.get("candidate_pool_fingerprint") == pool_fingerprint
            )
            if (
                versions_current
                and meta.get("fingerprint") == self.cache_fingerprint
                and pool_manifest_current
            ):
                self._cache_state = "hit"
                return True
            self._cache_state = "invalidated"
            if shared:
                raise ValueError("Shared dataset cache failed native/schema compatibility checks")
            if not self._cache_warning_emitted:
                reason = "missing metadata" if not meta else "fingerprint mismatch"
                if versions_current and not pool_manifest_current:
                    reason = "candidate-pool manifest missing or incompatible"
                if not versions_current:
                    reason = (
                        "ontology-derived cache schema is incompatible with pyowl-core model "
                        "schema 2; rebuild required from the original source bytes. Exact "
                        "will never convert or reinterpret schema-1 structural identities "
                        "or consumer-local dense IDs"
                    )
                self.log(
                    f"Existing dataset cache at {self._df_save_path} is invalid for the current configuration ({reason}); rebuilding.",
                    level="warning",
                )
                self._cache_warning_emitted = True
            return False
        return False

    def process(self) -> "IDataset":

        self._candidates = self._ensure_mapping_kinds(
            self._candidates,
            label="candidate",
        )
        self._candidates = self._filter_candidates_ignored_classes(self._candidates)
        if self._reference is not None:
            self._reference = self._ensure_mapping_kinds(
                self._reference,
                label="reference",
            )
            self._reference = self._filter_mappings_ignored_classes(
                self._reference, label="reference"
            )
        if self._exact_matches is not None:
            self._exact_matches = self._ensure_mapping_kinds(
                self._exact_matches,
                label="exact",
            )
            self._exact_matches = self._filter_mappings_ignored_classes(
                self._exact_matches, label="exact"
            )
        self._annotate_candidate_similarity_stats()

        if self.has_cache():
            self.load()
            from exact.runs.decisions import observe_dataset

            observe_dataset(self, restored=True)
            return self

        # Inference set

        self.log("Creating Inference set", level="debug")

        if self.candidates is None:
            self.log("Candidates not loaded.", level="error")
            raise ValueError("Candidates not loaded.")

        inference_set = self.candidates

        if self.reference is not None:
            # Update Labels based on full_reference
            self.log("#Updating Labels based on Reference...", level="debug")
            mapping_keys = self._mapping_key_columns(inference_set, self.reference)
            inference_set = inference_set.merge(
                self.reference,
                on=mapping_keys,
                how="left",
                suffixes=("", "_y"),
            )
            inference_set["Label"] = inference_set["Label_y"].combine_first(inference_set["Label"])
            inference_set.drop(columns=["Label_y"], inplace=True)

            exact_prefilter_pairs = self._exact_mapping_pairs()
            # Warn how many unique source-target pairs full reference has that are not
            # recoverable either by scored candidates or by exact prefiltering.
            unique_pairs_inference = set(
                zip(inference_set["Src"].astype(str), inference_set["Tgt"].astype(str))
            )
            unique_pairs_full_ref = set(
                zip(self.reference["Src"].astype(str), self.reference["Tgt"].astype(str))
            )
            raw_missing_pairs = unique_pairs_full_ref - unique_pairs_inference
            missing_pairs = raw_missing_pairs.difference(exact_prefilter_pairs)

            if missing_pairs:
                missing_percentage = len(missing_pairs) / len(unique_pairs_full_ref) * 100
                self.log(
                    f"#Warning: {len(missing_pairs)} reference pairs are not covered by candidates or exact prefiltering ({missing_percentage:.2f}%)",
                    level="warning",
                )

                # Warn how many unique source entities full reference has that are not in the candidates with percentage
                unique_src_inference = set(inference_set["Src"].astype(str).unique())
                unique_src_full_ref = set(self.reference["Src"].astype(str).unique())
                missing_src = {src for src, _ in missing_pairs if src not in unique_src_inference}

                if missing_src:
                    missing_percentage = len(missing_src) / len(unique_src_full_ref) * 100
                    self.log(
                        f"#Warning: {len(missing_src)} reference sources are not covered by candidates or exact prefiltering ({missing_percentage:.2f}%)",
                        level="warning",
                    )

                # Warn how many unique target entities full reference has that are not in the candidates with percentage
                unique_tgt_inference = set(inference_set["Tgt"].astype(str).unique())
                unique_tgt_full_ref = set(self.reference["Tgt"].astype(str).unique())
                missing_tgt = {tgt for _, tgt in missing_pairs if tgt not in unique_tgt_inference}

                if missing_tgt:
                    missing_percentage = len(missing_tgt) / len(unique_tgt_full_ref) * 100
                    self.log(
                        f"#Warning: {len(missing_tgt)} reference targets are not covered by candidates or exact prefiltering ({missing_percentage:.2f}%)",
                        level="warning",
                    )

        pre_filtered_set = pd.DataFrame()
        pre_filtered_mappings = self.exact_matches

        if pre_filtered_mappings is not None and not pre_filtered_mappings.empty:
            exact_key_columns = self._mapping_key_columns(pre_filtered_mappings)
            pre_filtered_mappings = pre_filtered_mappings.drop_duplicates(subset=exact_key_columns)
            exact_prefilter_rows = self._exact_prefilter_rows(pre_filtered_mappings)

            pair_filtering = self.cardinality_1_to_many or not self.drop_exact_match_sources

            if pair_filtering:
                # Remove only the exact (Src, Tgt) matches from the inference pool.
                exact_pairs_idx = pd.MultiIndex.from_frame(pre_filtered_mappings[exact_key_columns])
                inference_pairs_idx = pd.MultiIndex.from_frame(inference_set[exact_key_columns])
                match_mask = pd.Series(
                    inference_pairs_idx.isin(exact_pairs_idx), index=inference_set.index
                )
                n_removed = int(match_mask.sum())

                if n_removed:
                    inference_set = inference_set[~match_mask]
                pre_filtered_set = exact_prefilter_rows
                self.log(
                    (
                        "#Exact prefilter: "
                        f"mappings={len(pre_filtered_set)}, removed_candidate_rows={n_removed}"
                    ),
                    level="debug",
                )

            else:
                # Ranking task or cardinality=1-to-1: drop every candidate for sources with an exact match.
                if "SrcKind" in exact_key_columns:
                    exact_sources = set(
                        pre_filtered_mappings[["Src", "SrcKind"]]
                        .dropna()
                        .itertuples(index=False, name=None)
                    )
                    src_mask = pd.Series(
                        [
                            (src, kind) in exact_sources
                            for src, kind in inference_set[["Src", "SrcKind"]].itertuples(
                                index=False, name=None
                            )
                        ],
                        index=inference_set.index,
                    )
                else:
                    exact_sources = set(pre_filtered_mappings["Src"].dropna().astype(str))
                    src_mask = inference_set["Src"].astype(str).isin(exact_sources)
                n_removed = int(src_mask.sum())
                n_sources_removed = (
                    int(inference_set.loc[src_mask, "Src"].nunique()) if n_removed else 0
                )
                if n_removed:
                    inference_set = inference_set[~src_mask]
                pre_filtered_set = exact_prefilter_rows
                removal_note = (
                    "source_candidates_already_skipped"
                    if n_removed == 0
                    else f"removed_candidate_rows={n_removed}, removed_sources={n_sources_removed}"
                )
                self.log(
                    ("#Exact prefilter: " f"mappings={len(pre_filtered_set)}, {removal_note}"),
                    level="debug",
                )

        if not pre_filtered_set.empty:
            pre_filtered_set[DatasetMask.inference] = False
            pre_filtered_set[DatasetMask.prefiltered] = True

        inference_set = self.get_features(inference_set)
        inference_set[DatasetMask.inference] = True
        inference_set[DatasetMask.prefiltered] = False

        self.log(f"#Inference Set: {len(inference_set)} samples", level="debug")

        if pre_filtered_set.empty:
            self._df = inference_set.reset_index(drop=True)
        else:
            self._df = pd.concat([inference_set, pre_filtered_set], ignore_index=True, sort=False)

        self.log("#Processing Done", level="debug")

        from exact.runs.decisions import observe_dataset

        observe_dataset(self)
        return self

    def freeze_source_universe(self, iris: Sequence[str], *, cap: Optional[int], seed: int) -> None:
        """Freeze eligible groups before retrieval, retaining sources with empty pools."""
        canonical = sorted(set(str(iri).strip() for iri in iris if str(iri).strip()))
        if not canonical:
            raise ValueError("source universe must contain at least one entity")
        groups = set()
        for iri in canonical:
            kind = self._source_entity_kind_index.get(iri)
            if kind is None or kind not in self.entity_kinds:
                raise ValueError(f"source universe entity has no eligible ontology kind: {iri}")
            if kind == EntityKind.CLASS and iri in self.source_ignored_alignment_classes:
                raise ValueError(f"source universe includes an excluded alignment entity: {iri}")
            groups.add((iri, kind.value))
        ranked = sorted(
            groups,
            key=lambda item: (
                hashlib.sha256(f"{int(seed)}\x1f{item[0]}\x1f{item[1]}".encode()).digest(),
                item,
            ),
        )
        if cap is not None:
            if cap < 1:
                raise ValueError("source cap must be positive")
            ranked = ranked[:cap]
        self._eligible_source_groups = set(ranked)
        self.eligible_source_groups = tuple(sorted(ranked))
        self.eligible_source_iris = tuple(sorted(iri for iri, _ in ranked))
        identity = hashlib.sha256(
            "\n".join(f"{iri}\t{kind}" for iri, kind in sorted(ranked)).encode()
        ).hexdigest()
        self._candidate_generation_params = {
            **self._candidate_generation_params,
            "source_universe_sha256": identity,
            "source_cap": cap,
            "source_seed": seed,
        }

    def restrict_sources(self, cap: int, seed: int) -> set[str]:
        """Restrict processed in-memory frames to deterministic source-kind groups.

        Selection is gold-free: only source groups present in the processed
        dataframe (or candidate pool) enter the hash ranking. The reusable
        full-dataset cache and its persisted pool manifest are never rewritten.
        """

        if int(cap) < 1:
            raise ValueError("source cap must be at least one")
        universe = self._df if self._df is not None else self._candidates
        if universe is None:
            raise RuntimeError("Dataset not processed. Call process() first.")
        if "Src" not in universe.columns:
            raise ValueError("Dataset source capping requires a Src column")

        def source_groups(frame: DataFrame) -> set[Tuple[str, str]]:
            if frame.empty:
                return set()
            if "SrcKind" in frame.columns:
                return {
                    (str(src), str(kind))
                    for src, kind in frame[["Src", "SrcKind"]]
                    .dropna(subset=["Src"])
                    .itertuples(index=False, name=None)
                }
            return {
                (str(src), self.primary_entity_kind.value)
                for src in frame["Src"].dropna().astype(str)
            }

        groups = getattr(self, "_eligible_source_groups", None)
        if groups is None:
            groups = source_groups(universe)
        ranked_groups = sorted(
            groups,
            key=lambda item: (
                hashlib.sha256(f"{int(seed)}\x1f{item[0]}\x1f{item[1]}".encode("utf-8")).digest(),
                item[0],
                item[1],
            ),
        )
        selected_groups = set(ranked_groups[: min(int(cap), len(ranked_groups))])
        selected_sources = {src for src, _ in selected_groups}
        self.eligible_source_iris = tuple(sorted(selected_sources))

        def restrict(frame: Optional[DataFrame]) -> Optional[DataFrame]:
            if frame is None or "Src" not in frame.columns:
                return frame
            if "SrcKind" in frame.columns:
                mask = pd.Series(
                    [
                        (str(src), str(kind)) in selected_groups
                        for src, kind in frame[["Src", "SrcKind"]].itertuples(
                            index=False,
                            name=None,
                        )
                    ],
                    index=frame.index,
                )
            else:
                mask = frame["Src"].astype(str).isin(selected_sources)
            return frame.loc[mask].reset_index(drop=True).copy()

        self._df = restrict(self._df)
        self._candidates = restrict(self._candidates)
        self._reference = restrict(self._reference)
        self._exact_matches = restrict(self._exact_matches)
        self._source_restriction_active = True
        self._invalidate_active_dataframe_cache()

        selected_by_kind: Dict[str, int] = defaultdict(int)
        for _, kind in selected_groups:
            selected_by_kind[kind] += 1
        for kind, count in selected_by_kind.items():
            self._candidate_pool_sizes.setdefault(kind, {})["source_entities"] = int(count)
        for kind, values in self._candidate_pool_sizes.items():
            values["source_entities"] = int(selected_by_kind.get(kind, 0))
        if self._candidates is not None:
            for kind, values in self._candidate_pool_sizes.items():
                kind_frame = (
                    self._candidates[self._candidates["SrcKind"].astype(str) == kind]
                    if "SrcKind" in self._candidates.columns
                    else self._candidates
                )
                values["candidate_rows"] = int(len(kind_frame))
                values["covered_sources"] = int(kind_frame["Src"].astype(str).nunique())

        sample_payload = "\n".join(f"{src}\t{kind}" for src, kind in sorted(selected_groups))
        self._active_candidate_config = {
            **self._active_candidate_config,
            "source_sample": {
                "cap": int(cap),
                "seed": int(seed),
                "selected_groups": len(selected_groups),
                "eligible_source_iris": sorted(selected_sources),
                "source_kind_groups": sorted(selected_groups),
                "sha256": hashlib.sha256(sample_payload.encode("utf-8")).hexdigest(),
            },
        }
        pool_frame = self._candidates
        if pool_frame is None and self._df is not None and "cand_sim" in self._df.columns:
            pool_frame = self._df[self._df["cand_sim"].notna()].reset_index(drop=True)
        if pool_frame is not None:
            self._refresh_candidate_pool_manifest(
                origin="sampled",
                frame=pool_frame,
                persist=False,
            )
        return selected_sources

    def _exact_mapping_pairs(self) -> set[Tuple[str, str]]:
        if not self.filter_exact_matches:
            return set()
        exact_matches = self.exact_matches
        if (
            exact_matches is None
            or exact_matches.empty
            or not {"Src", "Tgt"}.issubset(exact_matches.columns)
        ):
            return set()
        return {
            (str(src), str(tgt))
            for src, tgt in exact_matches[["Src", "Tgt"]].dropna().itertuples(index=False)
        }

    def _exact_prefilter_rows(self, exact_mappings: DataFrame) -> DataFrame:
        columns = self._mapping_key_columns(exact_mappings)
        rows = exact_mappings[columns].dropna().copy()
        if rows.empty:
            return rows

        score_column = None
        for candidate in ("Scores", "Score"):
            if candidate in exact_mappings.columns:
                score_column = candidate
                break
        if score_column is None:
            scores = pd.Series(1.0, index=rows.index)
        else:
            scores = pd.to_numeric(
                exact_mappings.loc[rows.index, score_column], errors="coerce"
            ).fillna(1.0)

        rows["Scores"] = scores.astype(float)
        return rows

    @abstractmethod
    def get_features(self, df: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def plot_feature_distributions(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def log_sanity_examples(self, *args, **kwargs) -> None:
        pass
