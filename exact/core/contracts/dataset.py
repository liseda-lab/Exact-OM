from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from contextlib import nullcontext

import math
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
from exact.core.entities.ontology import OntologyGraph
from exact.utils.data import read_table

from mowl.datasets import PathDataset as OWLDataset
from mowl.owlapi import OWLOntology
from org.semanticweb.HermiT import Reasoner
from org.semanticweb.owlapi.reasoner import InferenceType


# from jpype import java


DataFrame = pd.DataFrame


class IDataset(SelfRegisteringComponent, LoggingClass, Dataset):

    component_type = ComponentType.DATASET

    def __init__(self, 
                 output_path: Path,
                 filter_exact_matches: bool = False,
                 cardinality: int = 1,
                 num_workers: Optional[int] = None,
                 **kwargs
        ) -> None:


        self._output_path: Path = output_path / "dataset"
        self._output_path.mkdir(parents=True, exist_ok=True)
        self.plot_dir: Path = self.output_path / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self._df: DataFrame = None
        self._df_save_path: Path = self.output_path / "dataset.csv"

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

        self._cardinality: int = cardinality

        self._cache_ok = kwargs.get("cache_ok", True)

        self._candidates_generated = False

        LoggingClass.__init__(self, logger=kwargs.get("logger"))

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

    @property
    def filter_exact_matches(self) -> bool:
        return self._filter_exact_matches

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
        if self._source_reasoner is None:
            self._source_reasoner = Reasoner.ReasonerFactory().createReasoner(self.source.ontology)
            self._source_reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY, InferenceType.OBJECT_PROPERTY_HIERARCHY)
        return self._source_reasoner
    
    @property
    def target_reasoner(self) -> Reasoner:
        if self.target is None:
            self.log("Target ontology not loaded.", level="error")
            raise ValueError("Target ontology not loaded.")
        if self._target_reasoner is None:
            self._target_reasoner = Reasoner.ReasonerFactory().createReasoner(self.target.ontology)
            self._target_reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY, InferenceType.OBJECT_PROPERTY_HIERARCHY)
        return self._target_reasoner
    
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

        # Build normalized label→IRIs indexes
        self.log("#Indexing normalized labels for exact matching...", level="debug")
        def index_norm(map_) -> Dict[str, List[str]]:
            idx = {}
            for iri, labels in map_.items():
                for lab in labels:
                    nl = OntologyGraph.normalize_label(lab)
                    if not nl:
                        continue
                    idx.setdefault(nl, []).append(iri)
            return idx

        src_idx = index_norm(src_map)
        tgt_idx = index_norm(tgt_map)

        # Intersect by normalized label keys
        shared = set(src_idx.keys()).intersection(tgt_idx.keys())
        exact_rows = []
        for key in shared:
            for s in src_idx[key]:
                for t in tgt_idx[key]:
                    exact_rows.append([s, t, 1.0])

        self._exact_matches = pd.DataFrame(exact_rows, columns=["Src", "Tgt", "Score"])
        self.log(f"#Exact matches found: {len(self._exact_matches)}", level="info")

    def load_ontologies(self, source_path: Path, target_path: Path) -> None:
        
        self._source = OWLDataset(str(source_path))
        
        self.log("#Loaded Source...", level="debug")
        
        self._target = OWLDataset(str(target_path))
        
        self.log("#Loaded Target...", level="debug")

    def load_candidates(self, 
                        file_path: Optional[Path] = None,
                        top_k: Optional[int] = 100,
                        lexical_encoder_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
                        encode_batch_size: Optional[int] = 512,
                        search_batch_size: Optional[int] = 4096,
                        use_amp: Optional[bool] = True,
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
            self._candidates = get_cands(candidates)
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
                device=device
            )


    def generate_candidates(
        self,
        top_k: int = 100,
        lexical_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encode_batch_size: int = 512,
        search_batch_size: int = 4096,
        use_amp: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Build one-to-many candidate table using GPU torch top-k cosine over MiniLM embeddings.
        Stores into self._candidates with columns ["Src", "Tgt", "Score"].
        """
        if self._source is None or self._target is None:
            self.log("Ontologies must be loaded before candidate generation.", level="error")
            raise ValueError("Ontologies not loaded.")

        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log("#Candidate generation via torch top-k cosine…", level="info")
        self.log(f"  Encoder: {lexical_encoder_name}", level="debug")
        self.log(f"  Device:  {dev}", level="debug")

        src_iris = self.source_graph.get_all_classes()
        tgt_iris = self.target_graph.get_all_classes()
        self.log(f"## Source classes: {len(src_iris)} | Target classes: {len(tgt_iris)}", level="debug")

        if not self.cardinality_1_to_many and self._exact_matches is not None:
            self.log("#Filtering Exact Matches from Candidate Generation...", level="debug")
            src_iris = [iri for iri in src_iris if iri not in set(self._exact_matches["Src"])]
            self.log(f"## Filtered Source classes for Candidate Generation: {len(src_iris)}", level="debug")
        
        # 1) Get primary labels
        self.log("  Extracting primary labels for all classes…", level="debug")

        src_texts = [self.source_graph.get_primary_label(iri) for iri in src_iris]
        tgt_texts = [self.target_graph.get_primary_label(iri) for iri in tgt_iris]

        # 2) Encode with MiniLM (fast)
        self.log("  Encoding labels…", level="debug")
        st = SentenceTransformer(lexical_encoder_name, device=str(dev))
        # Already does batching internally; normalize embeddings for cosine
        src_emb = st.encode(src_texts, batch_size=encode_batch_size, convert_to_numpy=True, normalize_embeddings=True)
        tgt_emb = st.encode(tgt_texts, batch_size=encode_batch_size, convert_to_numpy=True, normalize_embeddings=True)

        # 3) Search Top-K with batched torch matmul
        self.log("  Running torch Top-K cosine search…", level="debug")
        idxs, sims = self.topk_cosine_search_torch(
            E_src=src_emb.astype(np.float32, copy=False),
            E_tgt=tgt_emb.astype(np.float32, copy=False),
            top_k=top_k,
            batch_size_src=search_batch_size,
            device=dev,
            amp=use_amp
        )

        # 4) Build candidate rows
        self.log("  Assembling candidate DataFrame…", level="debug")
        rows = []
        for i, s_iri in enumerate(src_iris):
            tgt_idx_row = idxs[i]
            sim_row = sims[i]
            for j, score in zip(tgt_idx_row, sim_row):
                rows.append([s_iri, tgt_iris[int(j)], 0, float(score)])

        cand_df = pd.DataFrame(rows, columns=["Src", "Tgt", "Label", "cand_sim"])

        self._candidates = cand_df
        self._candidates_generated = True
        self.log(f"#Candidate generation complete: {len(self._candidates)} rows (Top-{top_k} per source).", level="debug")

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
        K = top_k
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


    def load_reference(self, file_path: Path) -> None:

        self._reference = read_table(file_path)
        self._reference.columns = ["Src", "Tgt", "Label"]
        self.log("#Loaded Reference...", level="debug")

    def save(self) -> Path:

        if self.dataframe is None:
            self.log("Dataset is empty.", level="error")
            raise ValueError("Dataset is empty.")
        
        if self.has_cache():
            self.log("#Dataset already saved skyping...", level="debug")
            return self._df_save_path

        self.dataframe.to_csv(str(self._df_save_path), index=False)

        self.log(f"#Dataset saved to {self._df_save_path}", level="debug")

        return self._df_save_path
    
    def load(self):
        self._df = pd.read_csv(self._df_save_path, converters={"Features": literal_eval})

        self.log("#Loaded Cached Dataset...", level="info")

    def has_cache(self) -> bool:
        if self._cache_ok:
            return self._df_save_path.exists()
        return False

    def process(self) -> "IDataset":

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

            # Warn how many unique source-target pairs full reference has that are not in the candidates with percentage
            unique_pairs_inference = set(zip(inference_set["Src"], inference_set["Tgt"]))
            unique_pairs_full_ref = set(zip(self.reference["Src"], self.reference["Tgt"]))
            missing_pairs = unique_pairs_full_ref - unique_pairs_inference  

            if missing_pairs:
                missing_percentage = len(missing_pairs) / len(unique_pairs_full_ref) * 100
                self.log(f"#Warning: {len(missing_pairs)} unique source-target pairs from full reference are not in the candidates ({missing_percentage:.2f}%)", level="warning")

                # Warn how many unique source entities full reference has that are not in the candidates with percentage
                unique_src_inference = set(inference_set["Src"].unique())
                unique_src_full_ref = set(self.reference["Src"].unique())
                missing_src = unique_src_full_ref - unique_src_inference

                if missing_src:
                    missing_percentage = len(missing_src) / len(unique_src_full_ref) * 100
                    self.log(f"#Warning: {len(missing_src)} unique source entities from full reference are not in the candidates ({missing_percentage:.2f}%)", level="warning")

                # Warn how many unique target entities full reference has that are not in the candidates with percentage
                unique_tgt_inference = set(inference_set["Tgt"].unique())
                unique_tgt_full_ref = set(self.reference["Tgt"].unique())
                missing_tgt = unique_tgt_full_ref - unique_tgt_inference

                if missing_tgt:
                    missing_percentage = len(missing_tgt) / len(unique_tgt_full_ref) * 100
                    self.log(f"#Warning: {len(missing_tgt)} unique target entities from full reference are not in the candidates ({missing_percentage:.2f}%)", level="warning")

        pre_filtered_set = pd.DataFrame(columns=inference_set.columns)
        pre_filtered_mappings = self.exact_matches

        if pre_filtered_mappings is not None and not pre_filtered_mappings.empty:
            pre_filtered_mappings = pre_filtered_mappings.drop_duplicates(subset=["Src", "Tgt"])

            if self.candidates_generated and self.cardinality_1_to_many:
                # Global task: remove only the exact (Src, Tgt) matches from the inference pool.
                self.log("#Filtering Exact (Src, Tgt) matches from Inference Set...", level="debug")
                exact_pairs_idx = pd.MultiIndex.from_frame(pre_filtered_mappings[["Src", "Tgt"]])
                inference_pairs_idx = pd.MultiIndex.from_frame(inference_set[["Src", "Tgt"]])
                match_mask = pd.Series(inference_pairs_idx.isin(exact_pairs_idx), index=inference_set.index)

                if match_mask.any():
                    pre_filtered_set = inference_set[match_mask].copy()
                    pre_filtered_set["Score"] = 1.0
                    inference_set = inference_set[~match_mask]

            else:
                # Ranking task or cardinality=1-to-1: drop every candidate for sources with an exact match.
                self.log("#Filtering Exact-match sources from Inference Set...", level="debug")
                exact_src_map = (
                    pre_filtered_mappings.drop_duplicates(subset=["Src"])
                    .set_index("Src")["Tgt"]
                )

                src_mask = inference_set["Src"].isin(exact_src_map.index)

                if src_mask.any():
                    pre_filtered_set = inference_set[src_mask].copy()
                    expected_tgt = pre_filtered_set["Src"].map(exact_src_map)
                    pre_filtered_set["Score"] = (pre_filtered_set["Tgt"] == expected_tgt).astype(float)
                    inference_set = inference_set[~src_mask]

        pre_filtered_set[DatasetMask.inference] = False
        pre_filtered_set[DatasetMask.prefiltered] = True

        self.log(f"#Filtered {len(pre_filtered_set)} mappings from Inference Set", level="debug")

        inference_set = self.get_features(inference_set)
        inference_set[DatasetMask.inference] = True
        inference_set[DatasetMask.prefiltered] = False

        self.log(f"#Inference Set: {len(inference_set)} samples", level="debug")

        self._df = pd.concat([inference_set, pre_filtered_set], ignore_index=True)

        self.log("#Processing Done", level="debug")

        return self

    @abstractmethod
    def get_features(self, df: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def plot_feature_distributions(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def log_sanity_examples(self, *args, **kwargs) -> None:
        pass
