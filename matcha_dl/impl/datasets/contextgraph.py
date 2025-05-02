import torch
from itertools import chain
from torch import Tensor
from torch import device as tdevice
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum
import json

import pandas as pd
DataFrame = pd.DataFrame
import numpy as np

from pathlib import Path

from typing import List, Tuple, Optional, Dict, Any

from matcha_dl.impl.datasets.tabular import TabularDataset
from matcha_dl.core.entities.dataset import StaticPrompts
from matcha_dl.core.entities.ontology import OntologyGraph
from matcha_dl.core.entities.ontology import Entity
from matcha_dl.utils.models import extract_answer
from matcha_dl.utils.logs import capture_stdout
from matcha_dl.core.entities.configs.dataset import AggregationStrategy, BatchLengthSortMode
from matcha_dl.core.entities.dataset import DatasetMask

# TODO Harmonize get_item with TabularDataset

class ContextTabularDataset(TabularDataset):
    def __init__(self,
                 n_hops: int,
                 verbaliser_name: Optional[str],
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
                 gen_max_length: Optional[int] = None,
                 summariser_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
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
        self._memmap_src: Dict[Any, np.memmap] = {}
        self._off_src: Dict[Any, np.ndarray] = {}
        self._memmap_tgt: Dict[Any, np.memmap] = {}
        self._off_tgt: Dict[Any, np.ndarray] = {}
        self._label_cache: Dict[Any, Tensor] = {}

        self._mem_map_cache_dir = self.output_path / "memmap_cache"
        self._mem_map_cache_dir.mkdir(parents=True, exist_ok=True)

        self._meta_mem_map_path = self._mem_map_cache_dir / "meta.json"
        
        self._verb_temp_path = self.output_path / "verbalization_templates.json"

    @property
    def source_graph(self) -> OntologyGraph:
        if self._source_graph is None:
            self.log("Loading source graph from ontology..", level="debug")
            with capture_stdout(self.log, level="debug"):
                self._source_graph = OntologyGraph(self.source.ontology, self.source_reasoner)
            self.log("Source graph loaded.", level="debug")
            self.log(f"Source graph has {len(self._source_graph)} triples.", level="debug")
        return self._source_graph
    
    @property
    def target_graph(self) -> OntologyGraph:
        if self._target_graph is None:
            self.log("Loading target graph from ontology..", level="debug")
            with capture_stdout(self.log, level="debug"):
                self._target_graph = OntologyGraph(self.target.ontology, self.target_reasoner)
            self.log("Target graph loaded.", level="debug")
            self.log(f"Target graph has {len(self._target_graph)} triples.", level="debug")
        return self._target_graph
    
    @property
    def verbalization_templates(self) -> Dict[str, str]:
        if self._verbalization_templates is None:

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
                self.log(f"####Skipping 'prefiltered' candidates", level="debug")
                continue
            
            dfk = self.dataframe[self.dataframe[kind]]

            # Check if DataFrame is empty
            if dfk.empty:
                self.log(f"####Skipping empty DataFrame for kind '{kind}'", level="debug")
                continue
            
            # Loading Labels
            self.log(f"####Loading labels for kind '{kind}'", level="debug")
            self._label_cache[kind] = torch.tensor(dfk['Label'].values, dtype=torch.float32)

            paths = self._get_paths(kind)
            if not rebuild and all(p.exists() for p in paths.values()):
                self.log(f"####Using existing cache files for kind '{kind}'", level="debug")
                off_s = np.load(paths['src_off'], mmap_mode='r')
                off_t = np.load(paths['tgt_off'], mmap_mode='r')
                self._off_src[kind], self._off_tgt[kind] = off_s, off_t
                self._memmap_src[kind] = np.memmap(paths['src_flat'], mode='r', dtype=np.int32, shape=(int(off_s[-1]),))
                self._memmap_tgt[kind] = np.memmap(paths['tgt_flat'], mode='r', dtype=np.int32, shape=(int(off_t[-1]),))
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
        off_s, off_t = self._off_src[kind], self._off_tgt[kind]
        mem_s, mem_t = self._memmap_src[kind], self._memmap_tgt[kind]
        s_ids = mem_s[off_s[idx]:off_s[idx+1]].tolist()
        t_ids = mem_t[off_t[idx]:off_t[idx+1]].tolist()
        return {'input_ids': (s_ids, t_ids)}, self._label_cache[kind][idx]

    def get_features(self, dataset: DataFrame) -> DataFrame:
        return self.generate_definitions(dataset)

    def generate_definitions(self, dataset: DataFrame) -> DataFrame:
        # Log start
        self.log("##Generating Entity definitions...", level="debug")
        self.log("###Loading source and target entities uniquely...", level="debug")

        # Extract IRIs and cache Entity instances
        src_iris = dataset["Src"].tolist()
        tgt_iris = dataset["Tgt"].tolist()
        unique_src = set(src_iris)
        unique_tgt = set(tgt_iris)
        src_map = {iri: Entity(class_iri=iri,
                              ontology=self.source.ontology,
                              reasoner=self.source_reasoner,
                              ontology_graph=self.source_graph)
                   for iri in unique_src}
        tgt_map = {iri: Entity(class_iri=iri,
                              ontology=self.target.ontology,
                              reasoner=self.target_reasoner,
                              ontology_graph=self.target_graph)
                   for iri in unique_tgt}
        source_entities = [src_map[iri] for iri in src_iris]
        target_entities = [tgt_map[iri] for iri in tgt_iris]

        # Generate context subgraphs
        self.log("###Generating context subgraphs for each entity...", level="debug")
        src_subs = [ent.get_context_subgraph(self.n_hops) for ent in source_entities]
        tgt_subs = [ent.get_context_subgraph(self.n_hops) for ent in target_entities]
        self.log(f"###Generated context subgraphs for {len(src_subs)} source and {len(tgt_subs)} target entities.", level="debug")

        # Aggregate via join or summarisation
        if self.aggregation_strategy is AggregationStrategy.JOIN:
            self.log("###Verbalising and joining subgraphs...", level="debug")
            features = self._subgraphs_join([src_subs, tgt_subs])
        else:
            self.log("###Verbalising and summarising subgraphs...", level="debug")
            features = self._subgraphs_summarise(
                [src_subs, tgt_subs],
                [source_entities, target_entities]
            )

        # Attach features back to DataFrame
        src_feats, tgt_feats = features
        dataset["Features"] = [[s, t] for s, t in zip(src_feats, tgt_feats)]

        self.log("###Features verbalised and attached to DataFrame.", level="debug")

        # --- sort by word-count length, using either sum or max of src/tgt ---
        # compute a length key for each example:
        self.log("###Sorting dataset by feature lengths... (for batch efficiency)", level="debug")
        lengths = []
        for s, t in zip(src_feats, tgt_feats):
            src_wc = len(s.split())
            tgt_wc = len(t.split())
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
    
    def _process_subgraphs(self, context_graph_list: List[List[List[Tuple[str]]]], ordered_entities:List[List[Entity]]) -> List[List[str]]:

        """
        Processes the context graph list and ordered entities to generate verbalised sentences.
        Returns: A nested list of shape [2, num_entities] (one aggregated sentence per subgraph).
        """

        if self.aggregation_strategy is AggregationStrategy.JOIN:
            return self._subgraphs_join(context_graph_list)
        else:
            return self._subgraphs_summarise(context_graph_list, ordered_entities)

    def _verbalize_triples(self, triples: List[Tuple[str, str, str]]) -> str:
        """
        Verbalise a list of (head, relation, tail) triples and join them using the delimiter.
        Results are cached to avoid redundant replace/join operations.
        """
        key = tuple(triples)
        if key not in self._join_cache:
            texts = [
                self.verbalization_templates[rel]
                    .replace('$SRC', head)
                    .replace('$TGT', tail)
                for head, rel, tail in triples
            ]
            self._join_cache[key] = self.delimiter.join(texts)
        return self._join_cache[key]
    
    def _subgraphs_join(self, context_graph_list: List[List[List[Tuple[str]]]]) -> List[List[str]]:
        """Aggregate each subgraph by verbalising via _verbalize_triples and joining."""
        self.log("####Joining subgraphs with delimiter...", level="debug")
        sides, num = 2, len(context_graph_list[0])
        aggregated = [[], []]
        for side in range(sides):
            for idx in range(num):
                triples = context_graph_list[side][idx]
                text = self._verbalize_triples(triples)
                aggregated[side].append(text)
        self.log("####Subgraph join complete.", level="debug")
        return aggregated

    def _subgraphs_summarise(
        self,
        context_graph_list: List[List[List[Tuple[str]]]],
        ordered_entity_tuple: List[List[Entity]]
    ) -> List[List[str]]:
        # Log summarisation start
        self.log("####Summarising subgraphs with language model...", level="debug")
        sides = len(context_graph_list)
        num_entities = len(context_graph_list[0])

        # Build prompts using cached verbalisation
        prompts: List[str] = []
        positions: List[Tuple[int, int]] = []
        for side in range(sides):
            for idx in range(num_entities):
                ent = ordered_entity_tuple[side][idx]
                triples = context_graph_list[side][idx]
                context_str = self._verbalize_triples(triples)
                prompt = StaticPrompts.get_summarization(ent.labels[0], context_str)
                prompts.append(prompt)
                positions.append((side, idx))

        self.log(f"####Generated {len(prompts)} summarisation prompts.", level="debug")
        self.log("####Generating summaries in batches...", level="debug")

        # Deduplicate prompts and call LLM only on new ones
        unique_prompts = list(dict.fromkeys(prompts))
        to_generate = [p for p in unique_prompts if p not in self._summary_cache]
        if to_generate:
            new_summaries = self._batch_summarise(to_generate)
            for p, s in zip(to_generate, new_summaries):
                self._summary_cache[p] = s
            self.log(f"####Cached {len(to_generate)} new summaries; cache size: {len(self._summary_cache)}.", level="debug")

        # Reassemble into nested result
        aggregated = [[None] * num_entities for _ in range(sides)]
        for prompt, (side, idx) in zip(prompts, positions):
            aggregated[side][idx] = self._summary_cache[prompt]

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
        Keys: 'src_off', 'tgt_off', 'src_flat', 'tgt_flat'.
        """
        base = self._mem_map_cache_dir
        name = kind
        return {
            'src_off': base / f"src_off_{name}.npy",
            'tgt_off':  base / f"tgt_off_{name}.npy",
            'src_flat': base / f"src_flat_{name}.npy",
            'tgt_flat': base / f"tgt_flat_{name}.npy"
        }
    
    def _build_cache_for_kind(self, kind: DatasetMask, paths: Dict[str, str]) -> None:
        dfk = self.dataframe[self.dataframe[kind]]
        pairs = dfk['Features'].tolist()
        N = len(pairs)
        src_texts = [p[0] for p in pairs]
        tgt_texts = [p[1] for p in pairs]

        # Log start of cache building
        self.log(f"#### Building cache for split '{kind}' ({N} examples)", level="debug")

        # Pass 1: compute lengths
        self.log("## Pass 1: computing token lengths in chunks", level="debug")
        src_l = np.zeros(N, dtype=np.int64)
        tgt_l = np.zeros(N, dtype=np.int64)
        for start in range(0, N, self.cache_chunk_size):
            end = min(start + self.cache_chunk_size, N)
            self.log(f"### Tokenizing lengths [{start}:{end}]", level="debug")
            enc_s = self.tokenizer(src_texts[start:end], padding=False, truncation=True, max_length=self.encoding_max_length)['input_ids']
            enc_t = self.tokenizer(tgt_texts[start:end], padding=False, truncation=True, max_length=self.encoding_max_length)['input_ids']
            for i, ids in enumerate(enc_s, start): src_l[i] = len(ids)
            for i, ids in enumerate(enc_t, start): tgt_l[i] = len(ids)

        off_s = np.concatenate([[0], np.cumsum(src_l, dtype=np.int64)])
        off_t = np.concatenate([[0], np.cumsum(tgt_l, dtype=np.int64)])
        np.save(paths['src_off'], off_s)
        np.save(paths['tgt_off'], off_t)
        self.log(f"## Length pass complete: total src tokens={off_s[-1]}, total tgt tokens={off_t[-1]}", level="debug")

        # Allocate memmaps
        flat_s = np.memmap(paths['src_flat'], mode='w+', dtype=np.int32, shape=(int(off_s[-1]),))
        flat_t = np.memmap(paths['tgt_flat'], mode='w+', dtype=np.int32, shape=(int(off_t[-1]),))

        # Pass 2: write IDs
        self.log("## Pass 2: writing token IDs into memmap", level="debug")
        for start in range(0, N, self.cache_chunk_size):
            end = min(start + self.cache_chunk_size, N)
            self.log(f"### Filling memmap [{start}:{end}]", level="debug")
            enc_s = self.tokenizer(src_texts[start:end], padding=False, truncation=True, max_length=self.encoding_max_length)['input_ids']
            enc_t = self.tokenizer(tgt_texts[start:end], padding=False, truncation=True, max_length=self.encoding_max_length)['input_ids']
            for i, ids in enumerate(enc_s, start):
                s, e = off_s[i], off_s[i+1]
                flat_s[s:e] = np.array(ids, dtype=np.int32)
            for i, ids in enumerate(enc_t, start):
                s, e = off_t[i], off_t[i+1]
                flat_t[s:e] = np.array(ids, dtype=np.int32)

        flat_s.flush()
        flat_t.flush()
        self._off_src[kind], self._off_tgt[kind] = off_s, off_t
        self._memmap_src[kind] = np.memmap(paths['src_flat'], mode='r', dtype=np.int32, shape=(int(off_s[-1]),))
        self._memmap_tgt[kind] = np.memmap(paths['tgt_flat'], mode='r', dtype=np.int32, shape=(int(off_t[-1]),))
        self.log(f"## Cache build for '{kind}' complete", level="debug")