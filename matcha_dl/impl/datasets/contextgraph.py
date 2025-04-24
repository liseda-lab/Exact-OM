import torch
from itertools import chain
from torch import Tensor
from torch import device as tdevice
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum

import pandas as pd
DataFrame = pd.DataFrame
import numpy as np

from typing import List, Tuple, Optional, Dict, Any

from matcha_dl.impl.datasets.tabular import TabularDataset
from matcha_dl.core.entities.dataset import StaticPrompts
from matcha_dl.core.entities.ontology import OntologyGraph
from matcha_dl.core.entities.ontology import Entity
from matcha_dl.utils.models import extract_answer
from matcha_dl.utils.logs import capture_stdout
from matcha_dl.core.entities.configs.dataset import AggregationStrategy
from matcha_dl.core.entities.dataset import DatasetMask

# TODO Harmonize get_item with TabularDataset

class ContextTabularDataset(TabularDataset):
    def __init__(self,
                 n_hops: int,
                 verbaliser_name: str,
                 gen_max_new_tokens: int = 5000,
                 batch_size: int = 32,
                 do_sample: bool = False,
                 num_beams: int = 1,
                 early_stopping: bool = False,
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.JOIN, 
                 delimiter: str = "\n",
                 exclude_missing_dr: bool = False,
                 device: tdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 max_length: Optional[int] = None,
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
        self.max_length: Optional[int] = max_length
        self.gen_max_new_tokens: int = gen_max_new_tokens
        self.batch_size = batch_size
        self.do_sample: bool = do_sample
        self.num_beams: int = num_beams
        self.early_stopping: bool = early_stopping
        self.aggregation_strategy: AggregationStrategy = aggregation_strategy
        self.delimiter: str = delimiter
        self.exclude_missing_dr: bool = exclude_missing_dr
        self.device: tdevice = device
        self.temperature: Optional[float] = temperature
        self.top_p: Optional[float] = top_p
        self.top_k: Optional[int] = top_k

        self.verbaliser_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(verbaliser_name)
        self.verbaliser_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(verbaliser_name)
        self.verbaliser_model.to(self.device)

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
        self._verbalization_templates: Optional[Dict[str, str]] = None
        self._tokenizer: Optional[AutoTokenizer] = None

        self._join_cache: Dict[Tuple[Tuple[str, str, str], ...], str] = {}
        self._summary_cache: Dict[str, str] = {}

        # Raw token-id lists and labels cache
        self._raw_src_ids: Dict[Any, List[List[int]]] = {}
        self._raw_tgt_ids: Dict[Any, List[List[int]]] = {}
        self._label_cache: Dict[Any, Tensor] = {}

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
            self.log("###Generating verbalisation templates from ontology...", level="debug")

            self.log("####Getting example triples from source and target graphs.", level="debug")
            
            examples: Dict[str, List[Tuple[str]]] = self.source_graph.get_example_triples(1, exclude_missing_dr=self.exclude_missing_dr)
            examples.update(self.target_graph.get_example_triples(1, exclude_missing_dr=self.exclude_missing_dr))

            self.log(f"####Found {len(examples)} unique relations", level="debug")
            self.log("####Generating verbalisation prompts for each relation...", level="debug")
            verbalization_prompts: List[str] = [StaticPrompts.get_verbalization(*example[0]) for example in examples.values()]
            self.log("####Generating verbalised templates for each relation...", level="debug")
            verbalized_sentences: List[str] = self._generate_verbalisation(verbalization_prompts)
            verbalized_templates = [sentence.replace(triple[0][0], "$SRC").replace(triple[0][2], "$TGT") for triple, sentence in zip(examples.values(), verbalized_sentences)]
            self._verbalization_templates = {k: v for k, v in zip(examples.keys(), verbalized_templates)}

            self.log("####Verbalisation templates generated.", level="debug")

        return self._verbalization_templates
    
    @property
    def tokenizer(self) -> Optional[AutoTokenizer]:
        return self._tokenizer
    
    @tokenizer.setter
    def tokenizer(self, tokenizer: AutoTokenizer) -> None:

        self.log("###Setting tokenizer for ContextTabularDataset...", level="debug")
        self._tokenizer = tokenizer

        self.log("###Pre-tokenizing and caching features and labels...", level="debug")

        for kind in DatasetMask:
            
            dfk = self.dataframe[self.dataframe[kind]]

            # Check if DataFrame is empty
            if dfk.empty:
                self.log(f"####Skipping empty DataFrame for kind '{kind.name}'", level="debug")
                continue

            pairs: List[List[str]] = dfk["Features"].tolist()
            # Split into separate lists
            src_texts = [p[0] for p in pairs]
            tgt_texts = [p[1] for p in pairs]

            self.log(f"####Tokenizing {len(src_texts)} source and {len(tgt_texts)} target texts for kind '{kind.name}'", level="debug")

            # Tokenize without padding here
            enc_src = self.tokenizer(
                src_texts,
                padding=False,
                truncation=True
            )
            enc_tgt = self.tokenizer(
                tgt_texts,
                padding=False,
                truncation=True
            )

            # Store raw input_ids lists
            self._raw_src_ids[kind] = enc_src['input_ids']
            self._raw_tgt_ids[kind] = enc_tgt['input_ids']

            self.log(f"####Cached {len(self._raw_src_ids[kind])} source and {len(self._raw_tgt_ids[kind])} target raw IDs for kind '{kind.name}'", level="debug")

            # Cache labels as tensor
            labels = torch.tensor(
                dfk["Label"].values,
                dtype=torch.float32
            )
            self._label_cache[kind] = labels

            self.log(f"####Cached labels for kind '{kind.name}' with shape {labels.shape}", level="debug")

    def __len__(self) -> int:
        # dataset length depends on current default_kind
        return len(self._label_cache[self.default_kind])


    def __getitem__(self, idx: int) -> Any:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}.")

        kind = self.default_kind
        src_ids = self._raw_src_ids[kind][idx]
        tgt_ids = self._raw_tgt_ids[kind][idx]
        label   = self._label_cache[kind][idx]

        return { 'input_ids': (src_ids, tgt_ids)}, label

    def get_features(self, dataset: DataFrame) -> DataFrame:
        return self.generate_defenitions(dataset)

    def generate_defenitions(self, dataset: DataFrame) -> DataFrame:
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
        Core batch‐generation loop for any (model, tokenizer) pair.
        Logs progress and returns the list of extracted answers.
        """
        model.eval()
        results: List[str] = []
        total = len(prompts)
        num_batches = (total + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end   = min(start + self.batch_size, total)
            batch = prompts[start:end]

            self.log(f"{name}: Processing batch {batch_idx+1}/{num_batches} (items {start}-{end-1})", level="debug")
            with torch.no_grad():
                # tokenization & to(device)
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)

                # actual generation
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

            # decode & extract
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_results = [extract_answer(text) for text in decoded]
            self.log(f"{name}: Finished batch {batch_idx+1}/{num_batches}", level="debug")

            results.extend(batch_results)

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