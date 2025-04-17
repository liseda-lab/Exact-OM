
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

class AggregationStrategy(str, Enum):
    JOIN = "join"
    SUMMARISE = "summarise"

class ContextGraphDataset(TabularDataset):
    def __init__(self,
                 n_hops: int,
                 verbaliser_name: str,
                 max_length: int = 128,
                 gen_max_new_tokens: int = 5000,
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.JOIN, 
                 delimiter: str = "\n",
                 summariser_name: Optional[str] = None,
                 exclude_missing_dr: bool = False,
                 device: tdevice = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 **kwargs):
        """
        Parameters:
            context_graph_list: List of context graphs, each with shape [2, num_entities, num_triples, 3].
                                The first dimension corresponds to source (index 0) and target (index 1) subgraphs.
            verbaliser_name: The model name for triple-level verbalisation.
            max_length: Maximum sequence length for tokenization.
            gen_max_new_tokens: Maximum number of new tokens to generate.
            aggregation_strategy: One of:
                - "join": Collapse all triples in each subgraph by joining with the given delimiter (output shape: [2, num_entities]).
                - "summarise": Collapse using a summarisation model (output shape: [2, num_entities]).  
            delimiter: The string delimiter used in the "join" strategy.
            summariser_name: Model name for summarisation (required if aggregation_strategy=="summarise").
            ordered_entities: List of tuples for each sample: (source_entities, target_entities) lists.
        """

        super().__init__(**kwargs)
        self.n_hops: int = n_hops
        self.max_length: int = max_length
        self.gen_max_new_tokens: int = gen_max_new_tokens
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

        self.source_graph: Optional[OntologyGraph] = None
        self.target_graph: Optional[OntologyGraph] = None
        self.verbalization_templates: Optional[Dict[str, str]] = None

    @property
    def source_graph(self) -> OntologyGraph:
        if self._source_graph is None:
            self.log("Loading source graph from ontology.", level="info")
            self._source_graph = OntologyGraph(self.source.ontology, self.source_reasoner)
        return self._source_graph
    
    @property
    def target_graph(self) -> OntologyGraph:
        if self._target_graph is None:
            self.log("Loading target graph from ontology.", level="info")
            self._target_graph = OntologyGraph(self.target.ontology, self.target_reasoner)
        return self._target_graph
    
    @property
    def verbalization_templates(self) -> Dict[str, str]:
        if self.verbalization_templates is None:
            self.log("Generating verbalisation templates from ontology.", level="debug")
            
            examples: Dict[str, List[Tuple[str]]] = self.source_graph.get_example_triples(1, exclude_missing_dr=self.exclude_missing_dr)
            examples.update(self.target_graph.get_example_triples(1, exclude_missing_dr=self.exclude_missing_dr))

            verbalization_prompts: List[str] = [StaticPrompts.get_verbalization(*example[0]) for example in examples.values()]
            verbalized_sentences: List[str] = self._generate_verbalisation(verbalization_prompts)
            verbalized_templates = [sentence.replace(triple[0], "$SRC").replace(triple[2], "$TGT") for triple, sentence in zip(examples.values(), verbalized_sentences)]
            self.verbalization_templates = {k: v for k, v in zip(examples.keys(), verbalized_templates)}

        return self.verbalization_templates


    def pre_process_x(self, data: np.ndarray) -> Tensor:
        if self.tokenizer:
            return self.tokenizer(data.tolist(), padding=True, truncation=True, return_tensors="pt")
        
        return super().pre_process_x(data)

    def get_features(self, dataset: DataFrame) -> DataFrame:
        return self.generate_prompts(dataset)

    def generate_defenitions(self, dataset: DataFrame) -> DataFrame:
        
        self.log("Generating Entity defenitions...", level="debug")

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

        source_context: List[Tuple[str]] = [source.get_context_subgraph(self.n_hops) for source in ordered_entities]
        target_context: List[Tuple[str]] = [target.get_context_subgraph(self.n_hops) for target in ordered_entities]
        context_graph_list: List[List[List[Tuple[str]]]] = [source_context, target_context]

        dataset["Features"] = list(np.array(self._process_subgraphs(context_graph_list, ordered_entities)).T)
        self.log("Entity defenitions generated.", level="debug")

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
                prompt = StaticPrompts.get_summarisation_prompt(entity_name, context_str)
                prompts.append(prompt)
        # Batch generate summarisation.
        inputs = self.summariser_tokenizer(prompts,
                                           return_tensors="pt",
                                           padding=True,
                                           truncation=True,
                                           max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.summariser_model.generate(**inputs, max_new_tokens=self.gen_max_new_tokens)
        decoded = self.summariser_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        summarised_sentences = [extract_answer(resp) for resp in decoded]
        # Reassemble into nested list of shape [2, num_entities]
        aggregated = []
        index = 0
        for side in range(sides):
            side_list = []
            for _ in range(num_entities):
                side_list.append(summarised_sentences[index])
                index += 1
            aggregated.append(side_list)
        return aggregated

    def _generate_verbalisation(self, prompts):
        """
        Tokenizes and generates triple-level verbalised sentences for a list of prompts.
        """
        inputs = self.verbaliser_tokenizer(prompts,
                                           return_tensors="pt",
                                           padding=True,
                                           truncation=True,
                                           max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.verbaliser_model.generate(**inputs, max_new_tokens=self.gen_max_new_tokens)
        decoded = self.verbaliser_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        verbalised_sentences = [extract_answer(resp) for resp in decoded]
        return verbalised_sentences

