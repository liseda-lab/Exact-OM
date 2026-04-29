from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any, Sequence
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os

import math
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from torch import Tensor
from ast import literal_eval
import torch
from sentence_transformers import SentenceTransformer

from exact.core.contracts.base import SelfRegisteringComponent, LoggingClass
from exact.core.entities.registry import ComponentType
from exact.core.entities.configs.dataset import DatasetMask
from exact.core.values import ANNOTATION_IRI
from exact.core.entities.ontology import OntologyGraph
from exact.utils.candidate_generation import (
    candidate_annotation_priority,
    candidate_token_key,
    lexical_candidate_pair_scores,
    make_candidate_labels,
    rank_channel_scores,
    select_candidate_annotation_literals,
)
from exact.utils.data import read_table

from mowl.datasets import PathDataset as OWLDataset
from mowl.owlapi import OWLOntology
from org.semanticweb.HermiT import Reasoner
from org.semanticweb.elk.owlapi import ElkReasonerFactory
from org.semanticweb.owlapi.model import IRI
from org.semanticweb.owlapi.reasoner.structural import StructuralReasonerFactory
from org.semanticweb.owlapi.reasoner import InferenceType
from org.semanticweb.owlapi.search import EntitySearcher


# from jpype import java


DataFrame = pd.DataFrame


class IDataset(SelfRegisteringComponent, LoggingClass, Dataset):

    component_type = ComponentType.DATASET

    def __init__(self, 
                 output_path: Path,
                 filter_exact_matches: bool = False,
                 cardinality: int = 1,
                 num_workers: Optional[int] = None,
                 drop_exact_match_sources: bool = True,
                 filter_ignored_alignment_classes: bool = False,
                 **kwargs
        ) -> None:


        self._output_path: Path = output_path / "dataset"
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

        self._source: OWLDataset = None
        self._target: OWLDataset = None
        self._source_reasoner: Reasoner = None
        self._target_reasoner: Reasoner = None
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

        self._candidates_generated = False
        self._source_path: Optional[Path] = None
        self._target_path: Optional[Path] = None
        self._dataset_signature: Optional[str] = None
        self._cache_warning_emitted: bool = False

        # Hint for skipping heavy reasoning when only taxonomy is requested.
        self._only_taxonomy_hint: bool = bool(kwargs.get("only_taxonomy", False))
        self._reasoner_timeout_secs: float = float(kwargs.get("reasoner_timeout_secs", 120.0))
        self._reasoner_force_hermit: bool = bool(kwargs.get("reasoner_force_hermit", False))
        self._source_reasoner_attempted: bool = False
        self._target_reasoner_attempted: bool = False

        LoggingClass.__init__(self, logger=kwargs.get("logger"))

    # ------------------------------------------------------------------
    # Reasoner helpers (lazy, timeout, ELK-first when only_taxonomy=True).
    # ------------------------------------------------------------------
    def _get_reasoner(self, label: str, ontology: OWLOntology) -> Optional[Reasoner]:
        cache = self._source_reasoner if label == "source" else self._target_reasoner
        attempted = self._source_reasoner_attempted if label == "source" else self._target_reasoner_attempted

        if cache is not None or attempted:
            return cache

        # Mark attempted to avoid repeated long builds
        if label == "source":
            self._source_reasoner_attempted = True
        else:
            self._target_reasoner_attempted = True

        def _build():
            return self._make_reasoner(ontology, label=label)

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_build)
                cache = fut.result(timeout=self._reasoner_timeout_secs)
        except Exception as exc:
            self.log(
                f"Reasoner creation for {label} failed or timed out after {self._reasoner_timeout_secs:.0f}s: {exc}",
                level="warning",
            )
            cache = None

        if label == "source":
            self._source_reasoner = cache
        else:
            self._target_reasoner = cache

        return cache

    def _make_reasoner(self, ontology: OWLOntology, label: str) -> Reasoner:
        only_tax = self._only_taxonomy_hint
        start = time.time()
        self.log(
            f"Creating reasoner for {label} (only_taxonomy={only_tax}, force_hermit={self._reasoner_force_hermit})…",
            level="info",
        )
        # Order: lightweight first; HermiT only when explicitly forced.
        attempts = ["elk", "structural"]
        if self._reasoner_force_hermit:
            attempts.append("hermit")

        last_exc: Optional[Exception] = None
        for kind in attempts:
            try:
                if kind == "elk":
                    elk = ElkReasonerFactory().createReasoner(ontology)
                    elk.precomputeInferences(InferenceType.CLASS_HIERARCHY)
                    self.log(
                        f"ELK reasoner for {label} ready in {time.time() - start:.2f}s",
                        level="info",
                    )
                    return elk
                if kind == "structural":
                    struct = StructuralReasonerFactory().createReasoner(ontology)
                    struct.precomputeInferences(InferenceType.CLASS_HIERARCHY)
                    self.log(
                        f"Structural reasoner for {label} ready in {time.time() - start:.2f}s",
                        level="info",
                    )
                    return struct
                # hermit
                hermit = Reasoner.ReasonerFactory().createReasoner(ontology)
                if only_tax:
                    hermit.precomputeInferences(InferenceType.CLASS_HIERARCHY)
                else:
                    hermit.precomputeInferences(
                        InferenceType.CLASS_HIERARCHY, InferenceType.OBJECT_PROPERTY_HIERARCHY
                    )
                self.log(
                    f"HermiT reasoner for {label} ready in {time.time() - start:.2f}s",
                    level="info",
                )
                return hermit
            except Exception as exc:
                last_exc = exc
                self.log(f"{kind.capitalize()} reasoner failed for {label}: {exc}", level="warning")
                continue

        self.log(
            f"All reasoner attempts failed for {label}; returning None. Last error: {last_exc}",
            level="warning",
        )
        return None

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
    def source(self) -> OWLDataset:
        return self._source
    
    @property
    def target(self) -> OWLDataset:
        return self._target
    
    @property
    def source_reasoner(self) -> Reasoner:
        if self.source is None:
            self.log("Source ontology not loaded.", level="error")
            raise ValueError("Source ontology not loaded.")
        return self._get_reasoner(label="source", ontology=self.source.ontology)

    @property
    def target_reasoner(self) -> Reasoner:
        if self.target is None:
            self.log("Target ontology not loaded.", level="error")
            raise ValueError("Target ontology not loaded.")
        return self._get_reasoner(label="target", ontology=self.target.ontology)
    
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
            self._source_graph = OntologyGraph(self.source.ontology, self.source_reasoner)
        return self._source_graph
    
    @property
    def target_graph(self) -> OntologyGraph:
        if self.target is None:
            self.log("Target ontology not loaded.", level="error")
            raise ValueError("Target ontology not loaded.")
        elif self._target_graph is None:
            self._target_graph = OntologyGraph(self.target.ontology, self.target_reasoner)
        return self._target_graph

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

        dataset = self.source if side_key == "src" else self.target
        if dataset is None:
            return set()
        ontology = dataset.ontology
        ignored: set[str] = set()
        try:
            for owl_class in ontology.getClassesInSignature():
                class_iri = str(owl_class.getIRI().toString())
                axioms = ontology.getAnnotationAssertionAxioms(owl_class.getIRI())
                iterator = axioms.iterator()
                while iterator.hasNext():
                    axiom = iterator.next()
                    try:
                        prop_iri = str(axiom.getProperty().getIRI().toString())
                    except Exception:
                        continue
                    if prop_iri != ANNOTATION_IRI:
                        continue
                    value = axiom.getValue()
                    literal = None
                    if hasattr(value, "isLiteral") and value.isLiteral():
                        literal = str(value.asLiteral().get().getLiteral())
                    elif hasattr(value, "getLiteral"):
                        literal = str(value.getLiteral())
                    if literal is not None and literal.strip().lower() == "false":
                        ignored.add(class_iri)
        except Exception as exc:
            self.log(
                f"Failed to read use_in_alignment annotations for {side_key}: {exc}",
                level="warning",
            )
            ignored = set()

        if side_key == "src":
            self._source_ignored_alignment_classes = ignored
        else:
            self._target_ignored_alignment_classes = ignored
        if ignored:
            self.log(
                f"#Ignored alignment classes ({side_key}): {len(ignored)} use_in_alignment=false",
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
        keep = ~(
            df["Src"].astype(str).isin(src_ignored)
            | df["Tgt"].astype(str).isin(tgt_ignored)
        )
        removed = int((~keep).sum())
        if removed:
            self.log(f"#Filtered {removed} candidate rows with ignored alignment classes.", level="info")
        return df.loc[keep].reset_index(drop=True)

    def _filter_mappings_ignored_classes(self, df: Optional[DataFrame], label: str) -> Optional[DataFrame]:
        if df is None or df.empty or not self.filter_ignored_alignment_classes:
            return df
        if not {"Src", "Tgt"}.issubset(df.columns):
            return df
        src_ignored = self.source_ignored_alignment_classes
        tgt_ignored = self.target_ignored_alignment_classes
        if not src_ignored and not tgt_ignored:
            return df
        keep = ~(
            df["Src"].astype(str).isin(src_ignored)
            | df["Tgt"].astype(str).isin(tgt_ignored)
        )
        removed = int((~keep).sum())
        if removed:
            self.log(f"#Filtered {removed} {label} rows with ignored alignment classes.", level="info")
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

        self.log("#Building ontology graphs (or reusing cached) to get labels...", level="debug")

        if self.candidates is not None:
            # Check matches only for candidates source and target IRIs
            src_iris = set(self.candidates["Src"].unique())
            tgt_iris = set(self.candidates["Tgt"].unique())

            src_map = {iri: self.source_graph.get_labels(iri)
                        for iri in src_iris}
            tgt_map = {iri: self.target_graph.get_labels(iri)
                        for iri in tgt_iris}

        else:
            src_map = self.source_graph.get_labels_map()
            tgt_map = self.target_graph.get_labels_map()

        if self.filter_ignored_alignment_classes:
            src_ignored = self.source_ignored_alignment_classes
            tgt_ignored = self.target_ignored_alignment_classes
            if src_ignored:
                src_map = {iri: labels for iri, labels in src_map.items() if str(iri) not in src_ignored}
            if tgt_ignored:
                tgt_map = {iri: labels for iri, labels in tgt_map.items() if str(iri) not in tgt_ignored}

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

        def pairs_from_indexes(src_idx: Dict[Any, List[str]], tgt_idx: Dict[Any, List[str]]) -> set[Tuple[str, str]]:
            pairs: set[Tuple[str, str]] = set()
            shared = set(src_idx.keys()).intersection(tgt_idx.keys())
            for key in shared:
                for s in src_idx[key]:
                    for t in tgt_idx[key]:
                        pairs.add((s, t))
            return pairs

        compact_pairs = pairs_from_indexes(
            index_norm(src_map, OntologyGraph.normalize_label),
            index_norm(tgt_map, OntologyGraph.normalize_label),
        )
        token_pairs = pairs_from_indexes(
            index_norm(src_map, token_exact_key),
            index_norm(tgt_map, token_exact_key),
        )
        near_exact_pairs = token_pairs.difference(compact_pairs)
        exact_rows = [[s, t, 1.0] for s, t in sorted(compact_pairs.union(token_pairs))]

        self._exact_matches = pd.DataFrame(exact_rows, columns=["Src", "Tgt", "Score"])
        self._exact_matches = self._filter_mappings_ignored_classes(self._exact_matches, label="exact")
        self.log(
            "#Exact matches found: "
            f"{len(self._exact_matches)} "
            f"(compact={len(compact_pairs)}, token_near_exact={len(near_exact_pairs)})",
            level="info",
        )

    def load_ontologies(self, source_path: Path, target_path: Path) -> None:
        
        self._source_path = Path(source_path).resolve()
        self._target_path = Path(target_path).resolve()
        self._dataset_signature = None

        self.log(
            f"Loading ontologies: src={self._source_path} tgt={self._target_path}",
            level="info",
        )

        self._source = OWLDataset(str(self._source_path))
        
        self.log("#Loaded Source...", level="debug")
        
        self._target = OWLDataset(str(self._target_path))
        
        self.log("#Loaded Target...", level="debug")

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
            "candidate_generation_params": self._candidate_generation_params,
            "only_taxonomy_hint": self._only_taxonomy_hint,
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
            "fingerprint": self.cache_fingerprint,
            "dataset_signature": self.dataset_signature,
            "component": self.__class__.__name__,
        }
        self._cache_meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_candidates(self, 
                        file_path: Optional[Path] = None,
                        top_k: Optional[int] = 100,
                        lexical_encoder_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
                        encode_batch_size: Optional[int] = 512,
                        search_batch_size: Optional[int] = 4096,
                        use_amp: Optional[bool] = True,
                        retrieval_strategy: str = "hybrid",
                        device: Optional[torch.device] = None,
                        ) -> None:

        if file_path is not None:

            if not file_path.exists():
                self.log(f"Candidates file not found at {file_path}", level="error")
                raise FileNotFoundError(f"Candidates file not found at {file_path}")

            def get_cands(df: pd.DataFrame) -> pd.DataFrame:

                return pd.DataFrame([
                        [source, cand, 0]
                        for source, _, target_cands in df.values
                        for cand in literal_eval(target_cands)
                    ], columns=["Src", "Tgt", "Label"])

            # Load One2Many candidates file
            candidates = read_table(str(file_path))
            candidates.columns = ["Src", "Tgt", "Candidates"]

            self.log("#Loaded Candidates Path...", level="debug")

            # Get One2One candidates df
            self._candidates = self._filter_candidates_ignored_classes(get_cands(candidates))
            self._annotate_candidate_similarity_stats()
            self.log("#Loaded Candidates...", level="info")

            if self.filter_exact_matches:
                self.log(f"Get Exact Matches...", level="info")
                self.get_exact_matches()

        else:
            self.log("#No Candidates file provided, generation candidates...", level="info")

            if self.filter_exact_matches:
                self.log(f"Get Exact Matches...", level="info")
                self.get_exact_matches()

            self.generate_candidates(
                top_k=top_k,
                lexical_encoder_name=lexical_encoder_name,
                encode_batch_size=encode_batch_size,
                search_batch_size=search_batch_size,
                use_amp=use_amp,
                retrieval_strategy=retrieval_strategy,
                device=device
            )


    def generate_candidates(
        self,
        top_k: int = 100,
        lexical_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encode_batch_size: int = 512,
        search_batch_size: int = 4096,
        use_amp: bool = True,
        retrieval_strategy: str = "hybrid",
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

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"#Candidate generation strategy={strategy} via torch top-k cosine…", level="info")
        self.log(f"  Encoder: {lexical_encoder_name}", level="debug")
        self.log(f"  Device:  {dev}", level="debug")

        src_iris = self.source_graph.get_all_classes()
        tgt_iris = self.target_graph.get_all_classes()
        if self.filter_ignored_alignment_classes:
            n_src_before = len(src_iris)
            n_tgt_before = len(tgt_iris)
            src_iris = self._filter_ignored_iris(src_iris, "src")
            tgt_iris = self._filter_ignored_iris(tgt_iris, "tgt")
            self.log(
                (
                    "## Alignment-use class filter: "
                    f"source={len(src_iris)}/{n_src_before}, target={len(tgt_iris)}/{n_tgt_before}"
                ),
                level="debug",
            )
        self.log(f"## Source classes: {len(src_iris)} | Target classes: {len(tgt_iris)}", level="debug")

        if (not self.cardinality_1_to_many 
                and self.drop_exact_match_sources 
                and self._exact_matches is not None):
            exact_sources = set(self._exact_matches["Src"].dropna().astype(str))
            n_src_before = len(src_iris)
            self.log("#Skipping exact-match sources during candidate generation...", level="debug")
            src_iris = [iri for iri in src_iris if str(iri) not in exact_sources]
            n_removed = n_src_before - len(src_iris)
            self.log(
                (
                    "## Candidate-generation sources: "
                    f"{len(src_iris)}/{n_src_before} (skipped_exact={n_removed})"
                ),
                level="debug",
            )
        
        if not src_iris or not tgt_iris:
            self._candidates = pd.DataFrame(
                columns=[
                    "Src",
                    "Tgt",
                    "Label",
                    "cand_sim",
                    "cand_sim_semantic",
                    "cand_sim_lexical",
                    "cand_channels",
                ]
            )
            self._candidates_generated = True
            return

        label_scope = "primary labels" if strategy == "primary_label" else "all labels"
        self.log(f"  Extracting {label_scope} for all classes…", level="debug")
        if strategy == "primary_label":
            src_labels_by_iri = {iri: [self.source_graph.get_primary_label(iri)] for iri in src_iris}
            tgt_labels_by_iri = {iri: [self.target_graph.get_primary_label(iri)] for iri in tgt_iris}
            src_lexical_texts_by_iri = src_labels_by_iri
            tgt_lexical_texts_by_iri = tgt_labels_by_iri
            channel_k = int(top_k)
        else:
            src_labels_by_iri = {iri: self.source_graph.get_labels(iri) for iri in src_iris}
            tgt_labels_by_iri = {iri: self.target_graph.get_labels(iri) for iri in tgt_iris}
            src_lexical_texts_by_iri = {
                iri: self._candidate_texts_for_iri(self.source_graph, iri, "src")
                for iri in src_iris
            }
            tgt_lexical_texts_by_iri = {
                iri: self._candidate_texts_for_iri(self.target_graph, iri, "tgt")
                for iri in tgt_iris
            }
            channel_k = max(int(top_k) * 3, 30)

        src_records = make_candidate_labels(src_iris, src_labels_by_iri)
        tgt_records = make_candidate_labels(tgt_iris, tgt_labels_by_iri)
        src_lexical_records = make_candidate_labels(src_iris, src_lexical_texts_by_iri)
        tgt_lexical_records = make_candidate_labels(tgt_iris, tgt_lexical_texts_by_iri)

        self.log(
            f"  Candidate labels: source={len(src_records)} target={len(tgt_records)}; "
            f"lexical texts: source={len(src_lexical_records)} target={len(tgt_lexical_records)}; "
            f"channel_k={channel_k}",
            level="debug",
        )
        st = SentenceTransformer(lexical_encoder_name, device=str(dev))
        semantic_scores = self._semantic_label_pair_scores(
            src_records=src_records,
            tgt_records=tgt_records,
            encoder=st,
            top_k=channel_k,
            encode_batch_size=encode_batch_size,
            search_batch_size=search_batch_size,
            use_amp=use_amp,
            device=dev,
        )
        lexical_scores = {}
        if strategy == "hybrid":
            self.log("  Running lexical token/char candidate retrieval…", level="debug")
            lexical_scores = lexical_candidate_pair_scores(
                src_records=src_lexical_records,
                tgt_records=tgt_lexical_records,
                per_source_limit=channel_k,
            )

        self.log("  Assembling candidate DataFrame…", level="debug")
        rows = rank_channel_scores(
            sources=[str(iri) for iri in src_iris],
            semantic_scores=semantic_scores,
            lexical_scores=lexical_scores,
            top_k=int(top_k),
        )
        cand_df = pd.DataFrame(
            rows,
            columns=[
                "Src",
                "Tgt",
                "Label",
                "cand_sim",
                "cand_sim_semantic",
                "cand_sim_lexical",
                "cand_channels",
            ],
        )

        self._candidates = self._filter_candidates_ignored_classes(cand_df)
        self._annotate_candidate_similarity_stats()
        self._candidates_generated = True
        self.log(f"#Candidate generation complete: {len(self._candidates)} rows (Top-{top_k} per source).", level="debug")

    def _candidate_texts_for_iri(self, graph: OntologyGraph, iri: str, side: str) -> List[str]:
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

        ontology = self.source.ontology if side == "src" else self.target.ontology
        factory = ontology.getOWLOntologyManager().getOWLDataFactory()
        owl_class = factory.getOWLClass(IRI.create(str(iri)))
        label_prop = str(factory.getRDFSLabel().getIRI().toString())
        annotation_values: List[Tuple[str, str]] = []
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
                annotation_values.append((prop_iri, literal))
        except Exception:
            return texts

        for literal in select_candidate_annotation_literals(
            annotation_values,
            seen_normalized=set(seen),
            overall_cap=12,
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
        E_src: np.ndarray,           # [Ns, d], float32
        E_tgt: np.ndarray,           # [Nt, d], float32
        top_k: int = 10,
        batch_size_src: int = 2048,  # tune per VRAM; 2–8K typical for d=384
        device: Optional[torch.device] = None,
        amp: bool = True
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
        groups = self._candidates.groupby("Src", sort=False)
        self._candidates["cand_sim_src_mean"] = groups["cand_sim"].transform("mean")

        src_sum = groups["cand_sim"].transform("sum")
        eps = np.finfo(np.float32).eps
        src_sum = src_sum.replace(0.0, eps)
        self._candidates["cand_sim_prob"] = self._candidates["cand_sim"] / src_sum

        k_share = max(1, int(getattr(self, "_candidate_share_k", 1)))
        ranks = groups.cumcount()
        top_mask = ranks < k_share
        share_top_series = (
            self._candidates.loc[top_mask]
            .groupby("Src", sort=False)["cand_sim_prob"]
            .sum()
        )
        share_top = self._candidates["Src"].map(share_top_series).fillna(0.0)
        share_rest = (1.0 - share_top).clip(lower=0.0)
        self._candidates["cand_share_top"] = share_top
        self._candidates["cand_share_rest"] = share_rest
        self._candidates["cand_share_log_ratio"] = np.log((share_top + eps) / (share_rest + eps))

    def load_reference(self, file_path: Path) -> None:

        self._reference = read_table(file_path)
        self._reference.columns = ["Src", "Tgt", "Label"]
        self._reference = self._filter_mappings_ignored_classes(self._reference, label="reference")
        self.log("#Loaded Reference...", level="debug")

    def save(self) -> Path:

        if self.dataframe is None:
            self.log("Dataset is empty.", level="error")
            raise ValueError("Dataset is empty.")
        
        if self.has_cache():
            self.log("#Dataset already saved skyping...", level="debug")
            return self._df_save_path

        self.dataframe.to_csv(str(self._df_save_path), index=False)
        self._write_cache_metadata()

        self.log(f"#Dataset saved to {self._df_save_path}", level="debug")

        return self._df_save_path
    
    def load(self):
        self._df = pd.read_csv(self._df_save_path, converters={"Features": literal_eval})

        self.log("#Loaded Cached Dataset...", level="info")
        if "cand_sim" in self._df.columns:
            self._candidates_generated = True

    def has_cache(self) -> bool:
        if self._cache_ok and self._df_save_path.exists():
            meta = self._load_cache_metadata()
            if meta.get("fingerprint") == self.cache_fingerprint:
                return True
            if not self._cache_warning_emitted:
                reason = "missing metadata" if not meta else "fingerprint mismatch"
                self.log(
                    f"Existing dataset cache at {self._df_save_path} is invalid for the current configuration ({reason}); rebuilding.",
                    level="warning",
                )
                self._cache_warning_emitted = True
            return False
        return False

    def process(self) -> "IDataset":

        self._candidates = self._filter_candidates_ignored_classes(self._candidates)
        if self._reference is not None:
            self._reference = self._filter_mappings_ignored_classes(self._reference, label="reference")
        if self._exact_matches is not None:
            self._exact_matches = self._filter_mappings_ignored_classes(self._exact_matches, label="exact")
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
            inference_set = inference_set.merge(self.reference, on=["Src", "Tgt"], how="left", suffixes=("", "_y"))
            inference_set["Label"] = inference_set["Label_y"].combine_first(inference_set["Label"])
            inference_set.drop(columns=["Label_y"], inplace=True)

            exact_prefilter_pairs = self._exact_mapping_pairs()
            # Warn how many unique source-target pairs full reference has that are not
            # recoverable either by scored candidates or by exact prefiltering.
            unique_pairs_inference = set(zip(inference_set["Src"].astype(str), inference_set["Tgt"].astype(str)))
            unique_pairs_full_ref = set(zip(self.reference["Src"].astype(str), self.reference["Tgt"].astype(str)))
            raw_missing_pairs = unique_pairs_full_ref - unique_pairs_inference
            missing_pairs = raw_missing_pairs.difference(exact_prefilter_pairs)

            if missing_pairs:
                missing_percentage = len(missing_pairs) / len(unique_pairs_full_ref) * 100
                self.log(f"#Warning: {len(missing_pairs)} reference pairs are not covered by candidates or exact prefiltering ({missing_percentage:.2f}%)", level="warning")

                # Warn how many unique source entities full reference has that are not in the candidates with percentage
                unique_src_inference = set(inference_set["Src"].astype(str).unique())
                unique_src_full_ref = set(self.reference["Src"].astype(str).unique())
                missing_src = {src for src, _ in missing_pairs if src not in unique_src_inference}

                if missing_src:
                    missing_percentage = len(missing_src) / len(unique_src_full_ref) * 100
                    self.log(f"#Warning: {len(missing_src)} reference sources are not covered by candidates or exact prefiltering ({missing_percentage:.2f}%)", level="warning")

                # Warn how many unique target entities full reference has that are not in the candidates with percentage
                unique_tgt_inference = set(inference_set["Tgt"].astype(str).unique())
                unique_tgt_full_ref = set(self.reference["Tgt"].astype(str).unique())
                missing_tgt = {tgt for _, tgt in missing_pairs if tgt not in unique_tgt_inference}

                if missing_tgt:
                    missing_percentage = len(missing_tgt) / len(unique_tgt_full_ref) * 100
                    self.log(f"#Warning: {len(missing_tgt)} reference targets are not covered by candidates or exact prefiltering ({missing_percentage:.2f}%)", level="warning")

        pre_filtered_set = pd.DataFrame()
        pre_filtered_mappings = self.exact_matches

        if pre_filtered_mappings is not None and not pre_filtered_mappings.empty:
            pre_filtered_mappings = pre_filtered_mappings.drop_duplicates(subset=["Src", "Tgt"])
            exact_prefilter_rows = self._exact_prefilter_rows(pre_filtered_mappings)

            pair_filtering = self.cardinality_1_to_many or not self.drop_exact_match_sources

            if pair_filtering:
                # Remove only the exact (Src, Tgt) matches from the inference pool.
                exact_pairs_idx = pd.MultiIndex.from_frame(pre_filtered_mappings[["Src", "Tgt"]])
                inference_pairs_idx = pd.MultiIndex.from_frame(inference_set[["Src", "Tgt"]])
                match_mask = pd.Series(inference_pairs_idx.isin(exact_pairs_idx), index=inference_set.index)
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
                exact_sources = set(pre_filtered_mappings["Src"].dropna().astype(str))
                src_mask = inference_set["Src"].astype(str).isin(exact_sources)
                n_removed = int(src_mask.sum())
                n_sources_removed = int(inference_set.loc[src_mask, "Src"].nunique()) if n_removed else 0
                if n_removed:
                    inference_set = inference_set[~src_mask]
                pre_filtered_set = exact_prefilter_rows
                removal_note = (
                    "source_candidates_already_skipped"
                    if n_removed == 0
                    else f"removed_candidate_rows={n_removed}, removed_sources={n_sources_removed}"
                )
                self.log(
                    (
                        "#Exact prefilter: "
                        f"mappings={len(pre_filtered_set)}, {removal_note}"
                    ),
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
        if exact_matches is None or exact_matches.empty or not {"Src", "Tgt"}.issubset(exact_matches.columns):
            return set()
        return {
            (str(src), str(tgt))
            for src, tgt in exact_matches[["Src", "Tgt"]].dropna().itertuples(index=False)
        }

    def _exact_prefilter_rows(self, exact_mappings: DataFrame) -> DataFrame:
        rows = exact_mappings[["Src", "Tgt"]].dropna().copy()
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
            scores = pd.to_numeric(exact_mappings.loc[rows.index, score_column], errors="coerce").fillna(1.0)

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
