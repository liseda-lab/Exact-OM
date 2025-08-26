import torch
from itertools import chain
from torch import Tensor
from torch import device as tdevice
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum
import random
import json

import pandas as pd
DataFrame = pd.DataFrame
Series = pd.Series
import numpy as np

from pathlib import Path

from typing import List, Tuple, Optional, Dict, Any

from concurrent.futures import ProcessPoolExecutor

from matcha_dl.impl.datasets.tabular import TabularDataset
from matcha_dl.core.entities.dataset import StaticPrompts
from matcha_dl.core.entities.ontology import OntologyGraph
from matcha_dl.core.entities.ontology import Entity
from matcha_dl.utils.models import extract_answer
from matcha_dl.utils.logs import capture_stdout
from matcha_dl.utils.paths import extract_entity_context
from matcha_dl.core.entities.configs.dataset import AggregationStrategy, BatchLengthSortMode, ContextMethod, BestPathMethod
from matcha_dl.core.entities.dataset import DatasetMask

# TODO Harmonize TabularDataset

class ContextTabularDataset(TabularDataset):
    def __init__(self,
                 n_hops: int,
                 verbaliser_name: Optional[str],
                 context_method: ContextMethod = ContextMethod.bfs,
                 best_path_src_method: Optional[BestPathMethod] = None,
                 context_hop_penalty: Optional[float] = 0.1,
                 context_token_ratio: Optional[float] = 1.3,
                 context_safety: Optional[float] = 0.8,
                 gen_max_new_tokens: int = 5000,
                 batch_size: int = 32,
                 do_sample: bool = False,
                 num_beams: int = 1,
                 early_stopping: bool = False,
                 max_verb_gen_retries: int = 0,
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.JOIN, 
                 delimiter: str = "\n",
                 exclude_missing_dr: bool = False,
                 device: tdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 cache_chunk_size: int = 10000,
                 encoding_max_length: Optional[int] = 256,
                 smallest_batch_first: bool = False,
                 batch_length_sort_mode: BatchLengthSortMode = BatchLengthSortMode.max,
                 only_taxonomy: bool = False,
                 gen_max_length: Optional[int] = None,
                 summariser_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 sanity_check_n_samples: int = 5,
                 max_log_lenght: int = 100,
                 all_labels: bool = False,
                 all_context_labels: bool = False,
                **kwargs):

        """
    
        Initializes a ContextGraphDataset that processes entity relationships through a graph.

        Parameters:
            n_hops: Number of hops to traverse in the entity graph.
            verbaliser_name: The model name for triple-level verbalisation.
            gen_max_new_tokens: Maximum number of new tokens to generate.
            aggregation_strategy: Strategy to combine triples:
            - JOIN: Concatenates triples with the delimiter
            - SUMMARISE: Uses a language model to summarize the graph
            delimiter: The string delimiter used when joining triples.
            exclude_missing_dr: Whether to exclude missing data relations.
            device: PyTorch device to use for model inference.
            max_length: Maximum sequence length for tokenization.
            summariser_name: Model name for summarisation (required if using SUMMARISE strategy).
            **kwargs: Additional arguments passed to TabularDataset.

        """

        super().__init__(**kwargs)
        self.n_hops: int = n_hops
        self.encoding_max_length: int = encoding_max_length
        self.gen_max_length: int = gen_max_length
        self.gen_max_new_tokens: int = gen_max_new_tokens
        self.smallest_batch_first: bool = smallest_batch_first
        self.batch_length_sort_mode: BatchLengthSortMode = batch_length_sort_mode
        self.batch_size = batch_size
        self.do_sample: bool = do_sample
        self.num_beams: int = num_beams
        self.early_stopping: bool = early_stopping
        self.max_verb_gen_retries: int = max_verb_gen_retries
        self.aggregation_strategy: AggregationStrategy = aggregation_strategy
        self.delimiter: str = delimiter + " "
        self.exclude_missing_dr: bool = exclude_missing_dr
        self.device: tdevice = device
        self.cache_chunk_size: int = cache_chunk_size
        self.temperature: Optional[float] = temperature
        self.top_p: Optional[float] = top_p
        self.top_k: Optional[int] = top_k
        self.only_taxonomy: bool = only_taxonomy
        self.sanity_check_n_samples: int = sanity_check_n_samples
        self.max_log_length: int = max_log_lenght
        self.all_labels: bool = all_labels
        self.all_context_labels: bool = all_context_labels

        # Greedy search parameters
        self.context_method: ContextMethod = context_method
        self.best_path_src_method: Optional[BestPathMethod] = best_path_src_method
        self.context_hop_penalty: Optional[float] = context_hop_penalty
        self.context_token_ratio: Optional[float] = context_token_ratio
        self.context_safety: Optional[float] = context_safety
        self.context_budget: int = int(
            self.encoding_max_length * self.context_safety
        ) if self.encoding_max_length is not None and self.context_safety is not None else None

        if verbaliser_name is not None:

            self.verbaliser_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(verbaliser_name)
            self.verbaliser_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(verbaliser_name)
            self.verbaliser_model.to(self.device)
        
        else:
            self.verbaliser_tokenizer: Optional[AutoTokenizer] = None
            self.verbaliser_model: Optional[AutoModelForCausalLM] = None

        if self.aggregation_strategy is AggregationStrategy.SUMMARISE:
            if summariser_name is None:
                erro_msg = "summariser_name must be provided for summarisation."
                self.log(erro_msg, level="error")
                raise ValueError(erro_msg)
            if summariser_name == verbaliser_name:
                self.summariser_model = self.verbaliser_model
                self.summariser_tokenizer = self.verbaliser_tokenizer
            else:
                self.summariser_tokenizer = AutoTokenizer.from_pretrained(summariser_name)
                self.summariser_model = AutoModelForCausalLM.from_pretrained(summariser_name)
                self.summariser_model.to(self.device)
        else:
            if delimiter is None:
                erro_msg = "Delimiter param must be provided for join strategy."
                self.log(erro_msg, level="error")
                raise ValueError(erro_msg)
            
            self.summariser_tokenizer: Optional[AutoTokenizer] = None
            self.summariser_model: Optional[AutoModelForCausalLM] = None

        self._source_graph: Optional[OntologyGraph] = None
        self._target_graph: Optional[OntologyGraph] = None
        self._tokenizer: Optional[AutoTokenizer] = None

        self._verbalization_templates: Optional[Dict[str, str]] = None

        self._join_cache: Dict[Tuple[Tuple[str, str, str], ...], str] = {}
        self._summary_cache: Dict[str, str] = {}

        # Internal storage
        self._memmap_fields: Dict[Any, List[np.memmap]] = {}
        self._off_fields:    Dict[Any, List[np.ndarray]] = {}
        self._label_cache: Dict[Any, Tensor] = {}

        self._mem_map_cache_dir = self.output_path / "memmap_cache"
        self._mem_map_cache_dir.mkdir(parents=True, exist_ok=True)

        self._meta_mem_map_path = self._mem_map_cache_dir / "meta.json"
        
        self._verb_temp_path = self.output_path / "verbalization_templates.json"

        self.K: int = 0  # Maximum length of any label or context

        # build approximate cost function
        def _triple_token_cost(triple):
            text = self._verbalize_triples([triple])
            return int(len(text.split()) * self.context_token_ratio)
        
        self.context_cost_fn = _triple_token_cost

    @property
    def source_graph(self) -> OntologyGraph:
        if self._source_graph is None:
            self.log("Loading source graph from ontology..", level="debug")
            with capture_stdout(self.log, level="debug"):
                self._source_graph = OntologyGraph(self.source.ontology, self.source_reasoner, self.only_taxonomy)
            self.log("Source graph loaded.", level="debug")
            self.log(f"Source graph has {len(self._source_graph)} triples.", level="debug")
        return self._source_graph
    
    @property
    def target_graph(self) -> OntologyGraph:
        if self._target_graph is None:
            self.log("Loading target graph from ontology..", level="debug")
            with capture_stdout(self.log, level="debug"):
                self._target_graph = OntologyGraph(self.target.ontology, self.target_reasoner, self.only_taxonomy)
            self.log("Target graph loaded.", level="debug")
            self.log(f"Target graph has {len(self._target_graph)} triples.", level="debug")
        return self._target_graph
    
    @property
    def verbalization_templates(self) -> Dict[str, str]:

        if self._verbalization_templates is None:

            if self.only_taxonomy is True:
                self.log("### Using taxonomy only verbalisation templates", level="debug")
                self._verbalization_templates = {
                    "subclassof": "$SRC is a subclass of $TGT",
                    "subclass_of": "$SRC is a subclass of $TGT"
                }
                return self._verbalization_templates

            if hasattr(self, "_verb_temp_path") and self._verb_temp_path.exists():
                self.log("### Loading verbalisation templates from file...", level="debug")
                with open(self._verb_temp_path, "r") as f:
                    self._verbalization_templates = json.load(f)
                self.log("### Verbalisation templates loaded.", level="debug")
                return self._verbalization_templates
            
            if self.verbaliser_model is None or self.verbaliser_tokenizer is None:
                self.log(f"#### No verbaliser model/tokenizer set, using stub for all triples", level="warning")
                templates = {}
                for rel in self.source_graph.get_relations().union(self.target_graph.get_relations()):
                    templates[rel] = "$SRC " + rel.replace("_", " ").lower() + " $TGT"

            else:


                self.log("### Generating verbalisation templates from ontology...", level="debug")
                examples = self.source_graph.get_example_triples(1, exclude_missing_dr=self.exclude_missing_dr)
                examples.update(self.target_graph.get_example_triples(1, exclude_missing_dr=self.exclude_missing_dr))
                self.log(f"#### Found {len(examples)} unique relations", level="debug")

                keys = list(examples.keys())
                # initial batch of prompts
                base_prompts = [
                    StaticPrompts.get_verbalization(head, rel.replace("_", " ").lower(), tail)
                    for key in keys
                    for (head, rel, tail) in [examples[key][0]]
                ]

                self.log("#### Batch generating initial verbalisations", level="debug")
                sentences = self._batch_generate(
                    base_prompts,
                    self.verbaliser_model,
                    self.verbaliser_tokenizer,
                    name="Verbaliser"
                )

                templates: Dict[str, str] = {}
                to_retry: List[str] = []

                # first pass: lower+replace and validate
                for key, sent in zip(keys, sentences):
                    head, _, tail = examples[key][0]
                    tmpl = sent.lower().replace(head.lower(), "$SRC").replace(tail.lower(), "$TGT")
                    if "$SRC" in tmpl and "$TGT" in tmpl:
                        templates[key] = tmpl
                    else:
                        to_retry.append((key, sent))

                # recursive retries: use the previous (faulty) sentence + corrective note
                retry_round = 0
                if self.max_verb_gen_retries > 0:
                    while to_retry and retry_round < self.max_verb_gen_retries:
                        retry_round += 1
                        self.log(f"#### Retry #{retry_round} for {len(to_retry)} templates", level="warning")

                        retry_prompts = []
                        retry_keys = []
                        for key, prev_tmpl in to_retry:
                            head, _, tail = examples[key][0]

                            retry_prompts.append(
                                StaticPrompts.get_corrective_verbalization(prev_tmpl, head, tail)
                            )
                            retry_keys.append(key)

                        self.log("#### Batch generating retry verbalisations", level="debug")
                        new_sentences = self._batch_generate(
                            retry_prompts,
                            self.verbaliser_model,
                            self.verbaliser_tokenizer,
                            name="Verbaliser-Retry"
                        )

                        new_to_retry = []
                        for key, sent in zip(retry_keys, new_sentences):
                            head, _, tail = examples[key][0]
                            tmpl = sent.lower().replace(head.lower(), "$SRC").replace(tail.lower(), "$TGT")
                            if "$SRC" in tmpl and "$TGT" in tmpl:
                                templates[key] = tmpl
                            else:
                                new_to_retry.append((key, sent))
                                self.log(f"Still missing placeholders for '{key}'", level="warning")

                        to_retry = new_to_retry

                # final fallback for any that still failed
                for key, _ in to_retry:
                    self.log(f"Falling back stub for '{key}'", level="warning")
                    templates[key] = "$SRC " + key.replace("_", " ").lower() + " $TGT"

                # save out
                if hasattr(self, "_verb_temp_path"):
                    self.log("#### Saving verbalisation templates to file...", level="debug")
                    with open(self._verb_temp_path, "w") as f:
                        json.dump(templates, f, indent=2)
                    self.log("#### Templates saved.", level="debug")

            self._verbalization_templates = templates

        return self._verbalization_templates

    
    @property
    def tokenizer(self) -> Optional[AutoTokenizer]:
        return self._tokenizer
    
    @tokenizer.setter
    def tokenizer(self, tokenizer: AutoTokenizer) -> None:

        self.log("###Setting tokenizer for ContextTabularDataset...", level="debug")
        self._tokenizer = tokenizer
        tokenizer_id = getattr(self.tokenizer, 'name_or_path', self.tokenizer.__class__.__name__)

        self.log("###Pre-tokenizing and caching features and labels...", level="debug")

        meta = {}
        if self._meta_mem_map_path.exists():
            self.log("####Meta memmap file exists, loading meta info", level="debug")

            try:
                meta = json.load(open(self._meta_mem_map_path, "r"))
            except Exception:
                self.log("####Failed to load meta info, will rebuild cache", level="debug")
                meta = {}

        rebuild = (
            meta.get('tokenizer_id') != tokenizer_id or
            meta.get('cache_chunk_size') != self.cache_chunk_size
        )

        for kind in DatasetMask:

            if kind is DatasetMask.prefiltered:
                self.log(f"####Skipping {kind} candidates", level="debug")
                continue
            
            dfk = self.dataframe[self.dataframe[kind]]

            # Check if DataFrame is empty
            if dfk.empty:
                self.log(f"####Skipping empty DataFrame for kind '{kind}'", level="debug")
                continue
            
            # Loading Labels
            self.log(f"####Loading labels for kind '{kind}'", level="debug")
            self._label_cache[kind] = torch.tensor(dfk['Label'].values, dtype=torch.float32)

            # Build or load four‐field memmaps
            paths = self._get_paths(kind)
            if not rebuild and all((paths[k].exists() for k in paths)):
                # load four offsets & four memmaps
                offs = [ np.load(paths[f"off_{i}"], mmap_mode='r') for i in range(4) ]
                flats = [
                    np.memmap(paths[f"flat_{i}"], mode='r', dtype=np.int32, shape=(int(offs[i][-1]),))
                    for i in range(4)
                ]
                self._off_fields[kind]    = offs
                self._memmap_fields[kind] = flats
            else:
                self._build_cache_for_kind(kind, paths)

                self.log("###Saving meta information for token cache...", level="debug")
                json.dump(
                {'tokenizer_id': tokenizer_id, 'cache_chunk_size': self.cache_chunk_size},
                open(self._meta_mem_map_path, 'w')
            )
        
        self.log("###Pre-tokenization and caching complete.", level="debug")

    def __len__(self) -> int:
        return len(self._label_cache[self.default_kind])


    def __getitem__(self, idx: int) -> Any:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}.")

        kind = self.default_kind
        offs = self._off_fields[kind]
        mems = self._memmap_fields[kind]

        # We have N*K sequences flattened per field,
        # so sample idx starts at idx*self.K and spans K sequences.
        base = idx * self.K
        seqs = []
        for f in range(4):
            field_seqs = []
            for k in range(self.K):
                flat_idx = base + k
                start = offs[f][flat_idx]
                end   = offs[f][flat_idx + 1]
                field_seqs.append( mems[f][start:end].tolist() )
            seqs.append(field_seqs)

        return {'input_ids': seqs}, self._label_cache[kind][idx]
    
    def get_features(self, dataset: DataFrame) -> DataFrame:
        return self.generate_definitions(dataset)

    def generate_definitions(self, dataset: DataFrame) -> DataFrame:
        # Log start
        self.log("##Generating Entity definitions...", level="debug")
        self.log("###Loading source and target entities uniquely...", level="debug")

        # Extract IRIs and cache Entity instances / build entity maps (unique)
        src_iris = dataset["Src"].tolist()
        tgt_iris = dataset["Tgt"].tolist()

        unique_src_iris = list(dict.fromkeys(src_iris))
        unique_tgt_iris = list(dict.fromkeys(tgt_iris))

        src_map = {
            iri: Entity(
                class_iri=iri,
                ontology=self.source.ontology,
                reasoner=self.source_reasoner,
                ontology_graph=self.source_graph
            )
            for iri in unique_src_iris
        }
        tgt_map = {
            iri: Entity(
                class_iri=iri,
                ontology=self.target.ontology,
                reasoner=self.target_reasoner,
                ontology_graph=self.target_graph
            )
            for iri in unique_tgt_iris
        }

        unique_src_entities = list(src_map.values())
        unique_tgt_entities = list(tgt_map.values())

        # Generate context subgraphs
        self.log("###Generating context subgraphs for each unique entity...", level="debug")

        if self.context_method is ContextMethod.greedy:
            self.log("####Using 'greedy' context extraction method.", level="debug")
            if self.best_path_src_method is None:
                self.log("####No best path source method set, using default 'dp'.", level="debug")
                self.best_path_src_method = BestPathMethod.dp
            else:
                self.log(f"####Using best path source method: '{self.best_path_src_method}'.", level="debug")

                # Set cost function for greedy extraction
                self.log("####Setting cost function for context extraction. Computing cost for every triple..", level="debug")
                self.source_graph.cost_fn = self.context_cost_fn
                self.target_graph.cost_fn = self.context_cost_fn

        else:
            self.log("####Using 'BFS' context extraction method.", level="debug")

        src_subs_unique = [
            ent.get_context_subgraph(
                self.n_hops,
                human_readable=True,
                method=self.context_method,
                best_path_method=self.best_path_src_method,
                budget=self.context_budget,
                hop_penalty=self.context_hop_penalty,
                all_labels=self.all_context_labels,
            )
            for ent in unique_src_entities
        ]
        tgt_subs_unique = [
            ent.get_context_subgraph(
                self.n_hops,
                human_readable=True,
                method=self.context_method,
                best_path_method=self.best_path_src_method,
                budget=self.context_budget,
                hop_penalty=self.context_hop_penalty,
                all_labels=self.all_context_labels,
            )
            for ent in unique_tgt_entities
        ]

        # Verify that the subgraphs are not empty
        empty_src_iris = [iri for iri, sub in zip(unique_src_iris, src_subs_unique) if not sub]
        if empty_src_iris:
            self.log(f"####Empty source subgraphs for {len(empty_src_iris)} entities ({len(empty_src_iris)/len(src_subs_unique):.0%})", level="warning")
            if len(empty_src_iris) > self.sanity_check_n_samples:
                empty_src_iris = empty_src_iris[:self.sanity_check_n_samples] + ["..."]
            self.log(f"####Empty source subgraphs for IRIs: {empty_src_iris}", level="debug")

        empty_tgt_iris = [iri for iri, sub in zip(unique_tgt_iris, tgt_subs_unique) if not sub]
        if empty_tgt_iris:
            self.log(f"####Empty target subgraphs for {len(empty_tgt_iris)} entities ({len(empty_tgt_iris)/len(tgt_subs_unique):.0%})", level="warning")
            if len(empty_tgt_iris) > self.sanity_check_n_samples:
                empty_tgt_iris = empty_tgt_iris[:self.sanity_check_n_samples] + ["..."]
            self.log(f"####Empty target subgraphs for IRIs: {empty_tgt_iris}", level="debug")

        self.log(f"###Generated context subgraphs for {len(src_subs_unique)} unique source and {len(tgt_subs_unique)} target entities.", level="debug")

        # Aggregate via join or summarisation

        combined_subs  = src_subs_unique + tgt_subs_unique
        combined_entities = unique_src_entities + unique_tgt_entities

        if self.aggregation_strategy is AggregationStrategy.JOIN:
            self.log("### Verbalising & joining unique subgraphs…", level="debug")
            all_feats = self._verbalise_subgraphs(combined_subs)
        elif self.aggregation_strategy is AggregationStrategy.SUMMARISE:
            self.log("### Verbalising & summarising unique subgraphs…", level="debug")
            all_feats = self._summarise_subgraphs(combined_subs, combined_entities)
        else:
            all_feats = self._separate_subgraphs(combined_subs, combined_entities)

        # Split back into source and target
        src_feats_unique = all_feats[:len(src_subs_unique)]
        tgt_feats_unique = all_feats[len(src_subs_unique):]

        # Verify that the verbalisation exists for all unique entities
        src_feats_map = dict(zip(unique_src_iris, src_feats_unique))
        missing = set(unique_src_iris) - set(src_feats_map)
        if missing:
            self.log(f"####Missing source verbalisations for {len(missing)} entities", level="warning")
            if len(missing) > self.sanity_check_n_samples:
                missing = list(missing)[:self.sanity_check_n_samples] + ["..."]
            self.log(f"####Missing source verbalisations for IRIs: {missing}", level="debug")
            for iri in missing:
                src_feats_map[iri] = ""

        tgt_feats_map = dict(zip(unique_tgt_iris, tgt_feats_unique))
        missing = set(unique_tgt_iris) - set(tgt_feats_map)
        if missing:
            self.log(f"####Missing target verbalisations for {len(missing)} entities", level="warning")
            if len(missing) > self.sanity_check_n_samples:
                missing = list(missing)[:self.sanity_check_n_samples] + ["..."]
            self.log(f"####Missing target verbalisations for IRIs: {missing}", level="debug")
            for iri in missing:
                tgt_feats_map[iri] = ""

        # map back to full-length lists aligned with DataFrame rows
        # build direct IRI→feature maps for source and target
        src_feats_map = dict(zip(unique_src_iris, src_feats_unique))
        tgt_feats_map = dict(zip(unique_tgt_iris, tgt_feats_unique))

        # then just lookup each row’s iris—this can’t go out of bounds
        src_feats = [src_feats_map[iri] for iri in src_iris]
        tgt_feats = [tgt_feats_map[iri] for iri in tgt_iris]

        # Attach four‐tuple features: [src_label, src_ctx, tgt_label, tgt_ctx]
        if self.all_labels:
            src_labels = [src_map[iri].labels[:] for iri in src_iris]
            tgt_labels = [tgt_map[iri].labels[:] for iri in tgt_iris]
        else:
            src_labels = [[src_map[iri].labels[0]] for iri in src_iris]
            tgt_labels = [[tgt_map[iri].labels[0]] for iri in tgt_iris]

        # compute K = maximum length of any label-list or ctx-list
        K_labels = max(len(x) for x in src_labels + tgt_labels)
        K_ctx = max(len(x) for x in src_feats + tgt_feats)
        self.K = max(K_labels, K_ctx)

        # pad every list to length K with "" so tokeniser will turn it into padding
        def pad_to_K(lst):
            return lst + [""]*(self.K - len(lst))
        
        src_labels = [pad_to_K(x) for x in src_labels]
        tgt_labels = [pad_to_K(x) for x in tgt_labels]
        src_feats = [pad_to_K(x) for x in src_feats]
        tgt_feats = [pad_to_K(x) for x in tgt_feats]

        dataset["Features"] = [
             [s_lbl, s_ctx, t_lbl, t_ctx]
             for s_lbl, s_ctx, t_lbl, t_ctx
             in zip(src_labels, src_feats, tgt_labels, tgt_feats)
        ]

        self.log("###Features verbalised and attached to DataFrame.", level="debug")

        # ─── SANITY CHECKS ───
        if self.sanity_check:
            k = min(self.sanity_check_n_samples, len(src_subs_unique))
            self.log("### Sanity check (SOURCE): raw vs verbalised", level="debug")

            # sample k unique triples
            src_samples = random.sample(
                list(zip(unique_src_entities, src_subs_unique, src_feats_unique)), k
            )
            for ent, raw_triples, verb_list in src_samples:
                self.log(f"[SRC] {ent.class_iri} ({ent.labels[0]})", level="debug")
                # raw_triples is still a list of triples
                truncated_raw = raw_triples
                if len(raw_triples) > self.max_log_length:
                    truncated_raw = raw_triples[: self.max_log_length] + [("...")]
                self.log(f"  raw triples: {truncated_raw!r}", level="debug")

                # verb_list is now a list of up to K strings
                for i, ctx in enumerate(verb_list):
                    txt = ctx
                    if len(txt) > self.max_log_length:
                        txt = txt[: self.max_log_length] + "..."
                    self.log(f"  verbalisation[{i}]: {txt!r}", level="debug")

            k = min(5, len(tgt_subs_unique))
            self.log("### Sanity check (TARGET): raw vs verbalised", level="debug")
            tgt_samples = random.sample(
                list(zip(unique_tgt_entities, tgt_subs_unique, tgt_feats_unique)), k
            )
            for ent, raw_triples, verb_list in tgt_samples:
                self.log(f"[TGT] {ent.class_iri} ({ent.labels[0]})", level="debug")
                truncated_raw = raw_triples
                if len(raw_triples) > self.max_log_length:
                    truncated_raw = raw_triples[: self.max_log_length] + [("...")]
                self.log(f"  raw triples: {truncated_raw!r}", level="debug")
                for i, ctx in enumerate(verb_list):
                    txt = ctx
                    if len(txt) > self.max_log_length:
                        txt = txt[: self.max_log_length] + "..."
                    self.log(f"  verbalisation[{i}]: {txt!r}", level="debug")


            # build pandas.Series maps
            counts_map = {
                "source": Series(len(s) for s in src_subs_unique),
                "target": Series(len(s) for s in tgt_subs_unique),
            }
            verblen_map = {
                "source": Series(len(v) for v in src_feats_unique),
                "target": Series(len(v) for v in tgt_feats_unique),
            }

            plot_dir = self.plot_dir / "sanity_check"
            plot_dir.mkdir(parents=True, exist_ok=True)

            alpha = self.plot_params.get("all_alpha", self.plot_params.get("alpha", 0.6)*0.4)
            other_params = self.plot_params.copy()
            other_params.pop("alpha", None)

            self._plot_distributions_core(
                data_map=counts_map,
                plot_dir=plot_dir,
                prefix="sanity_triples_dist",
                xlabel="Triples per context subgraph",
                title="Distribution of Triple Counts",
                single_plot=True,
                alpha=alpha,
                **other_params
            )
            self._plot_distributions_core(
                data_map=verblen_map,
                plot_dir=plot_dir,
                prefix="sanity_verblen_dist",
                xlabel="Verbalisation length (chars)",
                title="Distribution of Verbalisation Lengths",
                single_plot=True,
                alpha=alpha,
                **other_params
            )

        # --- sort by word-count length, using either sum or max of src/tgt ---
        # compute a length key for each example:
        self.log("###Sorting dataset by feature lengths... (for batch efficiency)", level="debug")
        lengths = []
        for src_list, tgt_list in zip(src_feats, tgt_feats):
            # get word‐counts per sample
            src_wc_list = [len(ctx.split()) for ctx in src_list]
            tgt_wc_list = [len(ctx.split()) for ctx in tgt_list]
            if self.batch_length_sort_mode == "sum":
                src_wc = sum(src_wc_list)
                tgt_wc = sum(tgt_wc_list)
            else:  # "max"
                src_wc = max(src_wc_list)
                tgt_wc = max(tgt_wc_list)
            if self.batch_length_sort_mode == "sum":
                lengths.append(src_wc + tgt_wc)
            else:  # "max"
                lengths.append(max(src_wc, tgt_wc))
        # attach, sort descending (biggest first), then drop helper
        dataset["__feat_len"] = lengths
        dataset = dataset.sort_values("__feat_len", ascending=self.smallest_batch_first).reset_index(drop=True)
        dataset.drop(columns="__feat_len", inplace=True)

        self.log("##Entity definitions generated.", level="debug")
        return dataset

    def _verbalize_triples(self, triples: List[Tuple[str, str, str]]) -> str:
        """
        Verbalise a list of (head, relation, tail) triples and join them using the delimiter.
        Results are cached to avoid redundant replace/join operations.
        """
        key = tuple(triples)
        if key not in self._join_cache:
            texts = [
                self.verbalization_templates.get(rel, "$SRC " + rel.replace("_", " ").lower() + " $TGT")
                    .replace('$SRC', head)
                    .replace('$TGT', tail)
                for head, rel, tail in triples
            ]
            self._join_cache[key] = self.delimiter.join(texts)
        return self._join_cache[key]

    def _verbalise_subgraphs(self, subgraphs: List[List[Tuple[str, str, str]]]) -> List[List[str]]:
        """Aggregate each subgraph by verbalising via _verbalize_triples and joining."""
        self.log(f"####Joining subgraphs with delimiter '{self.delimiter}'.", level="debug")
        agregated = [[self._verbalize_triples(triples)] for triples in subgraphs]
        self.log("####Subgraph join complete.", level="debug")
        return agregated
    
    def _separate_subgraphs(
        self,
        subgraphs: List[List[Tuple[str, str, str]]],
        entities:  List[Entity]
    ) -> List[List[str]]:
        """
        Separate subgraphs into individual verbalised triples.
        Each subgraph is verbalised and returned as a list of strings.
        """
        self.log("####Separating subgraphs into individual triples...", level="debug")
        agregated = [[self._verbalize_triples([triple]) for triple in triples] for triples in subgraphs]
        self.log("####Subgraph separation complete.", level="debug")
        return agregated

    def _summarise_subgraphs(
        self,
        subgraphs: List[List[Tuple[str, str, str]]],
        entities:  List[Entity]
    ) -> List[List[str]]:
        # Log summarisation start
        self.log("####Summarising subgraphs with language model...", level="debug")
        prompts = []
        for ent, triples in zip(entities, subgraphs):
            context_str = self._verbalize_triples(triples)
            prompts.append( StaticPrompts.get_summarization(ent.labels[0], context_str) )

        self.log(f"####Generated {len(prompts)} summarisation prompts.", level="debug")
        self.log("####Generating summaries in batches...", level="debug")

        # Deduplicate prompts and call LLM only on new ones
        unique_prompts = list(dict.fromkeys(prompts))
        to_gen = [p for p in unique_prompts if p not in self._summary_cache]
        if to_gen:
            new = self._batch_summarise(to_gen)
            for p,s in zip(to_gen, new):
                self._summary_cache[p] = s
            self.log(f"####Cached {len(to_gen)} new summaries.", level="debug")

        # Reassemble into result
        aggregated = []
        for p in prompts:
            aggregated.append([self._summary_cache[p]])

        self.log("####Subgraph summarisation complete.", level="debug")
        return aggregated

    def _batch_generate(
        self,
        prompts: List[str],
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        name: str
    ) -> List[str]:
        """
        Core batch‐generation loop with global length‐sorting:
        1) sort all prompts by descending word‐count
        2) batch‐tokenize & generate in that order
        3) unsort outputs back to original prompt order
        """
        model.eval()
        total = len(prompts)

        # 1) GLOBAL SORT
        lengths = [len(p.split()) for p in prompts]
        sorted_idxs = sorted(range(total), key=lambda i: lengths[i], reverse=True)
        inv_idxs = [0] * total
        for new_pos, orig_pos in enumerate(sorted_idxs):
            inv_idxs[orig_pos] = new_pos
        sorted_prompts = [prompts[i] for i in sorted_idxs]

        # 2) BATCH PROCESSING
        results_sorted = []
        num_batches = (total + self.batch_size - 1) // self.batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end   = min(start + self.batch_size, total)
            batch = sorted_prompts[start:end]

            self.log(f"{name}: Processing sorted batch {batch_idx+1}/{num_batches} "
                    f"(items {start}-{end-1}, max_len={len(batch[0].split())})",
                    level="debug")

            with torch.no_grad():
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.gen_max_length
                ).to(self.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.gen_max_new_tokens,
                    do_sample=self.do_sample,
                    num_beams=self.num_beams,
                    early_stopping=self.early_stopping,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_results = [extract_answer(txt) for txt in decoded]
            results_sorted.extend(batch_results)

        # 3) UNSORT back to original order
        results = [results_sorted[inv] for inv in inv_idxs]

        self.log(f"{name}: Generated {len(results)}/{total} outputs.", level="debug")
        return results


    def _generate_verbalisation(self, prompts: List[str]) -> List[str]:
        return self._batch_generate(prompts,
                                    self.verbaliser_model,
                                    self.verbaliser_tokenizer,
                                    name="Verbaliser")


    def _batch_summarise(self, prompts: List[str]) -> List[str]:
        return self._batch_generate(prompts,
                                    self.summariser_model,
                                    self.summariser_tokenizer,
                                    name="Summariser")
    
    def _get_paths(self, kind: DatasetMask) -> Dict[str, Path]:
        """
        Return a dict of file paths for offsets and flat memmap arrays for a given split.
        """
        base = self._mem_map_cache_dir
        name = kind
        paths: Dict[str, Path] = {}
        
        for idx in range(4):
            paths[f"off_{idx}"]  = base / f"off_{idx}_{name}.npy"
            paths[f"flat_{idx}"] = base / f"flat_{idx}_{name}.npy"
        return paths
    
    def _build_cache_for_kind(self, kind: DatasetMask, paths: Dict[str, str]) -> None:

        dfk = self.dataframe[self.dataframe[kind]]
        quads = dfk['Features'].tolist()  # [src_lbl, src_ctx, tgt_lbl, tgt_ctx]
        N     = len(quads)
        # split into four parallel lists, each of length N, each entry is a list of K strings
        texts_per_field = [ [ quad[f] for quad in quads ] for f in range(4) ]

        # Log start of cache building
        self.log(f"#### Building cache for split '{kind}' ({N} examples)", level="debug")

        # Pass 1: compute token lengths for each of the 4 fields in chunks
        self.log("## Pass 1: computing token lengths", level="debug")
        lengths = [np.zeros(N*self.K, dtype=np.int64) for _ in range(4)]
        for field_idx, lists_of_K in enumerate(texts_per_field):
            self.log(f"### Field {field_idx}: computing lengths", level="debug")
            # flatten sample×K into a single list
            flat_texts = [
               txt
               for sample_texts in lists_of_K
               for txt in sample_texts
            ]
            for start in range(0, N*self.K, self.cache_chunk_size):
                end = min(start + self.cache_chunk_size, N*self.K)
                encs = self.tokenizer(
                    flat_texts[start:end],
                    padding=False,
                    truncation=True,
                    max_length=self.encoding_max_length
                )['input_ids']
                for i, ids in enumerate(encs, start):
                    lengths[field_idx][i] = len(ids)

        offs = [ np.concatenate([[0], np.cumsum(l, dtype=np.int64)]) for l in lengths ]
        for idx, off in enumerate(offs):
            np.save(paths[f"off_{idx}"], off)
        self.log("## Length pass complete: total tokens per field = " +
                 ", ".join(str(off[-1]) for off in offs), level="debug")
        
        # Allocate memmaps for each field
        flats: List[np.memmap] = []
        for idx, off in enumerate(offs):
            m = np.memmap(
                paths[f"flat_{idx}"],
                mode='w+',
                dtype=np.int32,
                shape=(int(off[-1]),)
            )
            flats.append(m)
        
        # Pass 2: write token IDs into memmaps for all 4 fields
        self.log("## Pass 2: writing token IDs into memmaps", level="debug")
        for field_idx, lists_of_K in enumerate(texts_per_field):
            off = offs[field_idx]
            flat = flats[field_idx]
            flat_texts = [
               txt
               for sample_texts in lists_of_K
               for txt in sample_texts
            ]
            for start in range(0, N*self.K, self.cache_chunk_size):
                end = min(start + self.cache_chunk_size, N*self.K)
                encs = self.tokenizer(
                    flat_texts[start:end],
                    padding=False,
                    truncation=True,
                    max_length=self.encoding_max_length
                )['input_ids']
                for i, ids in enumerate(encs, start):
                    s, e = off[i], off[i+1]
                    flat[s:e] = np.array(ids, dtype=np.int32)

        # Flush and register memmaps
        for m in flats:
            m.flush()
        self._off_fields[kind]    = offs
        self._memmap_fields[kind] = [
            np.memmap(
                paths[f"flat_{idx}"],
                mode='r',
                dtype=np.int32,
                shape=(int(offs[idx][-1]),)
            )
            for idx in range(4)
        ]
        self.log(f"## Cache build for '{kind}' complete", level="debug")