
import torch
from itertools import chain
from torch import Tensor
from torch import device as tdevice
from transformers import AutoTokenizer, AutoModelForCausalLM
from enum import Enum

import pandas as pd
DataFrame = pd.DataFrame
import numpy as np

from typing import List, Tuple, Optional, Dict

from matcha_dl.impl.datasets.tabular import TabularDataset
from matcha_dl.core.entities.dataset import StaticPrompts
from matcha_dl.core.entities.ontology import OntologyGraph
from matcha_dl.core.entities.ontology import Entity
from matcha_dl.utils.models import extract_answer
from matcha_dl.core.entities.configs.dataset import AggregationStrategy

class ContextTabularDataset(TabularDataset):
    def __init__(self,
                 n_hops: int,
                 verbaliser_name: str,
                 gen_max_new_tokens: int = 5000,
                 batch_size: int = 32,
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.JOIN, 
                 delimiter: str = "\n",
                 exclude_missing_dr: bool = False,
                 device: tdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 max_length: Optional[int] = None,
                 summariser_name: Optional[str] = None,
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
        self.max_length: int = max_length
        self.gen_max_new_tokens: int = gen_max_new_tokens
        self.batch_size = batch_size
        self.aggregation_strategy: AggregationStrategy = aggregation_strategy
        self.delimiter: str = delimiter
        self.exclude_missing_dr: bool = exclude_missing_dr
        self.device: tdevice = device

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

    @property
    def source_graph(self) -> OntologyGraph:
        if self._source_graph is None:
            self.log("Loading source graph from ontology..", level="debug")
            self._source_graph = OntologyGraph(self.source.ontology, self.source_reasoner)
            self.log("Source graph loaded.", level="debug")
            self.log(f"Source graph has {len(self._source_graph)} triples.", level="debug")
        return self._source_graph
    
    @property
    def target_graph(self) -> OntologyGraph:
        if self._target_graph is None:
            self.log("Loading target graph from ontology..", level="debug")
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
        self._tokenizer = tokenizer


    def pre_process_x(self, data: np.ndarray) -> Tensor:
        if self.tokenizer:
            return self.tokenizer(data.tolist(), padding=True, truncation=True, return_tensors="pt")

        return super().pre_process_x(data)

    def get_features(self, dataset: DataFrame) -> DataFrame:
        return self.generate_defenitions(dataset)

    def generate_defenitions(self, dataset: DataFrame) -> DataFrame:
        
        self.log("##Generating Entity defenitions...", level="debug")

        self.log("###Loading source and target entities...", level="debug")

        # Load the source and target entities
        source_entities = Entity.load_from_list(dataset["Src"], 
                                                ontology=self.source.ontology, 
                                                reasoner=self.source_reasoner, 
                                                ontology_graph=self.source_graph)
        target_entities = Entity.load_from_list(dataset["Tgt"], 
                                                ontology=self.target.ontology, reasoner=self.target_reasoner,
                                                ontology_graph=self.target_graph)
        
        #ordered_entities: List of tuples for each sample: (source_entities, target_entities) lists.
        ordered_entities: List[List[Entity]] = [source_entities, target_entities]

        # context_graph_list: List of context graphs, each with shape [2, num_entities, num_triples, 3].
        #                        The first dimension corresponds to source (index 0) and target (index 1) subgraphs.

        self.log("###Generating context subgraphs for each entity...", level="debug")

        source_context: List[Tuple[str]] = [source.get_context_subgraph(self.n_hops) for source in ordered_entities[0]]
        target_context: List[Tuple[str]] = [target.get_context_subgraph(self.n_hops) for target in ordered_entities[1]]
        context_graph_list: List[List[List[Tuple[str]]]] = [source_context, target_context]

        self.log(f"###Generated context subgraphs for {len(source_context)} source and {len(target_context)} target entities.", level="debug")
        self.log("###Verbalising context subgraphs...", level="debug")

        dataset["Features"] = np.array(self._process_subgraphs(context_graph_list, ordered_entities)).T.tolist()
        self.log("##Entity defenitions generated.", level="debug")

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
    
    def _subgraphs_join(self, context_graph_list: List[List[List[Tuple[str]]]]) -> List[List[str]]:
        """Aggregates each subgraph by verbalising and joining triple strings using a delimiter.
           Returns a nested list of shape [2, num_entities].
        """
        self.log("####Joining subgraphs with delimiter...", level="debug")

        sides = len(context_graph_list)                     # Should be 2 (source and target)
        num_entities = len(context_graph_list[0])
        aggregated = []
        for side in range(sides):
            side_agg = []
            for entity in range(num_entities):
                subgraph = context_graph_list[side][entity]  # list of triples
                triple_verb = [self.verbalization_templates[rel].replace('$SRC', head).replace('$TGT', tail) for head, rel, tail in subgraph]
                joined = self.delimiter.join(triple_verb)
                side_agg.append(joined)
            aggregated.append(side_agg)
        return aggregated

    def _subgraphs_summarise(self, context_graph_list: List[List[List[Tuple[str]]]], ordered_entity_tuple: List[List[Entity]]) -> List[List[str]]:
        """Aggregates each subgraph using a summarisation model.
           ordered_entity_tuple is (source_entities, target_entities) for this sample.
           Returns a nested list of shape [2, num_entities].
        """

        self.log("####Summarising subgraphs with language model...", level="debug")

        sides = len(context_graph_list)
        num_entities = len(context_graph_list[0])
        prompts = []
        # For each side and each entity, build a summarisation prompt.
        for side in range(sides):
            for entity in range(num_entities):
                # Collapse the subgraph into a string using a simple join.
                subgraph = context_graph_list[side][entity]
                triple_strs = [self.verbalization_templates[rel].replace('$SRC', head).replace('$TGT', tail) for head, rel, tail in subgraph]
                context_str = self.delimiter.join(triple_strs)
                # Get the entity name from the ordered_entities tuple.
                entity_name = ordered_entity_tuple[side][entity].labels[0]
                # Build the summarisation prompt.
                prompt = StaticPrompts.get_summarization(entity_name, context_str)
                prompts.append(prompt)
        # Batch generate summarisation.
        self.log(f"####Generated {len(prompts)} summarisation prompts.", level="debug")
        self.log("####Generating summaries in batches...", level="debug")

        self.summariser_model.eval()
        total = len(prompts)
        num_batches = (total + self.batch_size - 1) // self.batch_size
        summarised_sentences: List[str] = []
        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, total)
            batch_prompts = prompts[start:end]
            self.log(f"Summariser: Processing batch {i+1}/{num_batches} (items {start}-{end-1})", level="debug")
            with torch.no_grad():
                inputs = self.summariser_tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                outputs = self.summariser_model.generate(
                    **inputs,
                    max_new_tokens=self.gen_max_new_tokens
                )
            decoded = self.summariser_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_summaries = [extract_answer(resp) for resp in decoded]
            summarised_sentences.extend(batch_summaries)

        self.log(f"####Generated {len(summarised_sentences)} summarised sentences.", level="debug")

        # Reassemble into nested list of shape [2, num_entities]
        aggregated: List[List[str]] = []
        index = 0
        for side in range(sides):
            side_list = []
            for _ in range(num_entities):
                side_list.append(summarised_sentences[index])
                index += 1
            aggregated.append(side_list)
        return aggregated

    def _generate_verbalisation(self, prompts: List[str]) -> List[str]:
        """
        Generates verbalised sentences in batches to optimize memory and log progress.
        """
        self.verbaliser_model.eval()
        verbalised_sentences: List[str] = []
        total = len(prompts)
        num_batches = (total + self.batch_size - 1) // self.batch_size
        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, total)
            batch_prompts = prompts[start:end]
            self.log(f"Verbaliser: Processing batch {i+1}/{num_batches} (items {start}-{end-1})", level="debug")
            with torch.no_grad():
                inputs = self.verbaliser_tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                outputs = self.verbaliser_model.generate(
                    **inputs,
                    max_new_tokens=self.gen_max_new_tokens
                )
            decoded = self.verbaliser_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_answers = [extract_answer(resp) for resp in decoded]
            verbalised_sentences.extend(batch_answers)
        return verbalised_sentences