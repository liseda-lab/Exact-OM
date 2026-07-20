# mypy: ignore-errors

import hashlib
import json
import math
import os
import warnings
from abc import abstractmethod
from ast import literal_eval
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor

from exact.core.contracts.dataset import IDataset
from exact.core.contracts.knowledge import KnowledgeSource
from exact.core.entities.configs.dataset import DatasetMask
from exact.core.entities.kinds import (
    EntityKind,
    build_entity_kind_index,
    infer_entity_kind,
    normalize_entity_kinds,
)
from exact.core.entities.ontology import OntologyGraph
from exact.impl.datasets.options import candidate_config, mapping_options
from exact.io.sources import resolve as resolve_source
from exact.ontology.projection import (
    ProjectorSettings,
    projector_cache_identity,
)
from exact.ontology.reasoning import reasoner_cache_identity
from exact.runs.layout import RunLayout
from exact.utils.candidate_generation import (
    candidate_annotation_priority,
    candidate_token_key,
    lexical_candidate_pair_scores,
    make_candidate_labels,
    rank_channel_scores,
    select_candidate_annotation_literals,
)
from exact.utils.data import read_table

DataFrame = pd.DataFrame


class BaseAlignmentDataset(IDataset):
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

    def _path_fingerprint(self, path: Path) -> str:
        try:
            stat = path.stat()
            return f"{path.as_posix()}::{int(stat.st_mtime)}::{stat.st_size}"
        except OSError:
            return path.as_posix()

    @property
    def dataset_signature(self) -> Optional[str]:
        if self._dataset_signature is not None:
            return self._dataset_signature
        if self._source_path is None or self._target_path is None:
            return None
        blob = f"{self._path_fingerprint(self._source_path)}||{self._path_fingerprint(self._target_path)}"
        self._dataset_signature = hashlib.sha1(blob.encode("utf-8")).hexdigest()
        return self._dataset_signature

    def _cache_fingerprint_payload(self) -> Dict[str, Any]:
        return {
            "component": self.__class__.__name__,
            "dataset_signature": self.dataset_signature,
            "filter_exact_matches": self.filter_exact_matches,
            "drop_exact_match_sources": self.drop_exact_match_sources,
            "filter_ignored_alignment_classes": self.filter_ignored_alignment_classes,
            "cardinality": self._cardinality,
            "candidate_share_k": self._candidate_share_k,
            "candidate_generation_version": 5,
            "exact_prefilter_materialization_version": 2,
            "ignored_alignment_filter_version": 1,
            "ontology_backend_version": 4,
            "projector": projector_cache_identity(self._projector_settings),
            "input_format": self._input_format,
            "source_options": self._source_options,
            "target_options": self._target_options,
            "entity_kinds": [kind.value for kind in self.entity_kinds],
            "entity_kind_schema_version": 1,
            "candidate_generation_params": self._candidate_generation_params,
            "only_taxonomy_hint": self._only_taxonomy_hint,
            "reasoner": self._reasoner_name,
            "reasoner_identity": reasoner_cache_identity(self._reasoner_name),
        }

    @property
    def cache_fingerprint(self) -> Optional[str]:
        payload = self._cache_fingerprint_payload()
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _load_cache_metadata(self) -> Dict[str, Any]:
        if not self._cache_meta_path.exists():
            return {}
        try:
            return json.loads(self._cache_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_cache_metadata(self) -> None:
        payload = {
            "cache_schema_version": 3,
            "ontology_backend_version": 4,
            "fingerprint": self.cache_fingerprint,
            "dataset_signature": self.dataset_signature,
            "component": self.__class__.__name__,
        }
        self._cache_meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_candidates(
        self,
        file_path: Optional[Path] = None,
        top_k: Optional[int] = 100,
        lexical_encoder_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        encode_batch_size: Optional[int] = 512,
        search_batch_size: Optional[int] = 4096,
        use_amp: Optional[bool] = True,
        retrieval_strategy: str = "hybrid",
        fusion: Optional[Mapping[str, Any]] = None,
        aliases: Optional[Mapping[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:

        if file_path is not None:
            if not file_path.exists():
                self.log(f"Candidates file not found at {file_path}", level="error")
                raise FileNotFoundError(f"Candidates file not found at {file_path}")

            def get_cands(df: pd.DataFrame) -> pd.DataFrame:

                return pd.DataFrame(
                    [
                        [source, cand, 0]
                        for source, _, target_cands in df.values
                        for cand in literal_eval(target_cands)
                    ],
                    columns=["Src", "Tgt", "Label"],
                )

            # Load One2Many candidates file
            candidates = read_table(str(file_path))
            candidates.columns = ["Src", "Tgt", "Candidates"]

            self.log("#Loaded Candidates Path...", level="debug")

            # Get One2One candidates df
            self._candidates = self._ensure_mapping_kinds(
                get_cands(candidates),
                label="candidate",
            )
            self._candidates = self._filter_candidates_ignored_classes(self._candidates)
            self._annotate_candidate_similarity_stats()
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
                encode_batch_size=encode_batch_size,
                search_batch_size=search_batch_size,
                use_amp=use_amp,
                retrieval_strategy=retrieval_strategy,
                fusion=fusion,
                aliases=aliases,
                device=device,
            )

    def generate_candidates(
        self,
        top_k: int = 100,
        lexical_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encode_batch_size: int = 512,
        search_batch_size: int = 4096,
        use_amp: bool = True,
        retrieval_strategy: str = "hybrid",
        fusion: Optional[Mapping[str, Any]] = None,
        aliases: Optional[Mapping[str, Any]] = None,
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

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(
            f"#Candidate generation strategy={strategy} via torch top-k cosine…",
            level="info",
        )
        self.log(f"  Encoder: {lexical_encoder_name}", level="debug")
        self.log(f"  Device:  {dev}", level="debug")

        pools: List[Tuple[EntityKind, List[str], List[str]]] = []
        self._candidate_pool_sizes = {}
        for kind in self.entity_kinds:
            src_iris = list(self.source.entities(kind))
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
            return

        st = SentenceTransformer(lexical_encoder_name, device=str(dev))
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
                )
            )

        self.log("  Assembling candidate DataFrame…", level="debug")
        cand_df = pd.DataFrame(
            all_rows,
            columns=[
                "Src",
                "Tgt",
                "Label",
                "cand_sim",
                "cand_sim_semantic",
                "cand_sim_lexical",
                "cand_channels",
                "SrcKind",
                "TgtKind",
            ],
        )

        self._candidates = self._filter_candidates_ignored_classes(cand_df)
        self._annotate_candidate_similarity_stats()
        self._candidates_generated = True
        self.log(
            f"#Candidate generation complete: {len(self._candidates)} rows "
            f"(Top-{top_k} per source and kind).",
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
    ) -> List[Dict[str, object]]:
        """Build one isolated semantic/lexical retrieval index for a kind."""

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
            channel_k = int(top_k)
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
            channel_k = max(int(top_k) * 3, 30)

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
            top_k=int(top_k),
            fusion_config=fusion_config,
        )
        for row in rows:
            row["SrcKind"] = kind.value
            row["TgtKind"] = kind.value
        return rows

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
        self._reference.columns = ["Src", "Tgt", "Label"]
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

        if self.has_cache():
            self.log("#Dataset already saved; skipping...", level="debug")
            return self._df_save_path

        self.dataframe.to_csv(str(self._df_save_path), index=False)
        self._write_cache_metadata()

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
        if self._cache_ok and self._df_save_path.exists():
            meta = self._load_cache_metadata()
            if meta.get("fingerprint") == self.cache_fingerprint:
                return True
            if not self._cache_warning_emitted:
                try:
                    legacy = (
                        not meta
                        or int(meta.get("cache_schema_version", 0)) < 3
                        or int(meta.get("ontology_backend_version", 0)) < 4
                    )
                except (TypeError, ValueError):
                    legacy = True
                reason = "missing metadata" if not meta else "fingerprint mismatch"
                if legacy:
                    reason = (
                        "pre-WP-N ontology/compiler metadata is incompatible with the "
                        "encoded schema cache contract; Exact will rebuild from source "
                        "bytes and never reinterpret consumer-local encoded IDs"
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

        return self

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
