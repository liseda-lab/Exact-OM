
from pathlib import Path

import random
import numpy as np
import pandas as pd

DataFrame = pd.DataFrame
from torch import Tensor
import torch as th

from typing import List, Tuple, Optional

from transformers import AutoTokenizer


from matcha_dl.impl.datasets.tabular import TabularDataset
from matcha_dl.core.entities.configs.dataset import Separator, ComparisonType, ContextType, ContextSemantics, Likelihood
from matcha_dl.core.entities.dataset import StaticPrompts
from matcha_dl.core.entities.ontology import Entity

class PromptDataset(TabularDataset):

    def __init__(
                self,
                example: List[bool],
                positive_examples: List[int],
                negative_examples: List[int],
                task_context: List[bool],
                separator: List[Separator],
                comparison_type: List[ComparisonType],
                label_cardinality: List[int],
                context_type: List[ContextType],
                context_cardinality: List[int],
                context_semantics: List[ContextSemantics],
                likelihood: List[Likelihood],
                **kwargs
        ) -> None:

        super().__init__(**kwargs)
        self._example = example
        self._positive_examples = positive_examples
        self._negative_examples = negative_examples
        self._task_context = task_context
        self._separator = separator
        self._comparison_type = comparison_type
        self._label_cardinality = label_cardinality
        self._context_type = context_type
        self._context_cardinality = context_cardinality
        self._context_semantics = context_semantics
        self._likelihood = likelihood
        self._static_skeletons = None

        self._tokenizer = None
    
    @property
    def example(self) -> List[bool]:
        return self._example
    
    @property
    def positive_examples(self) -> List[int]:
        return self._positive_examples
    
    @property
    def negative_examples(self) -> List[int]:
        return self._negative_examples
    
    @property
    def task_context(self) -> List[bool]:
        return self._task_context
    
    @property
    def separator(self) -> List[Separator]:
        return self._separator
    
    @property
    def comparison_type(self) -> List[ComparisonType]:
        return self._comparison_type
    
    @property
    def label_cardinality(self) -> List[int]:
        return self._label_cardinality
    
    @property
    def context_type(self) -> List[ContextType]:
        return self._context_type
    
    @property
    def context_cardinality(self) -> List[int]:
        return self._context_cardinality
    
    @property
    def context_semantics(self) -> List[ContextSemantics]:
        return self._context_semantics
    
    @property
    def likelihood(self) -> List[Likelihood]:
        return self._likelihood
    
    @property
    def tokenizer(self) -> Optional[AutoTokenizer]:
        return self._tokenizer
    
    @tokenizer.setter
    def tokenizer(self, tokenizer: AutoTokenizer) -> None:
        self._tokenizer = tokenizer

    @property
    def static_skeletons(self) -> List[str]:
        if self._static_skeletons is None:
            self._static_skeletons = self.generate_static_skeletons()
        return self._static_skeletons
    
    def generate_static_skeletons(self) -> List[str]:

        self.log("Generating static skeletons...", level="debug")

        skeletons = []

        for task_context, comparison_type, context_semantics, likelihood in zip(
            self.task_context, self.comparison_type, self.context_semantics, self.likelihood
        ):
            skeleton = StaticPrompts.SKELETON
            skeleton = self.apply_task_context(skeleton, task_context)
            skeleton = self.apply_comparison_type(skeleton, comparison_type)
            skeleton = self.apply_context_semantics(skeleton, context_semantics)
            skeleton = self.apply_instruction_information(skeleton)
            skeleton = self.apply_confidence(skeleton, likelihood)
            skeletons.append(skeleton)

        self.log("Static skeletons generated.", level="debug")
        self.log(f"Skeletons: {skeletons}", level="debug")

        return skeletons

    def pre_process_x(self, data: np.ndarray) -> Tensor:
        if self.tokenizer:
            return self.tokenizer(data, padding=True, truncation=True, return_tensors="pt")
        
        return super().pre_process_x(data)
        
    
    def get_features(self, dataset: DataFrame) -> DataFrame:
        return self.generate_prompts(dataset)


    def generate_prompts(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Generates prompts for the dataset.
        """

        self.log("Generating dynamic prompts...", level="debug")

        source_entities = Entity.load_from_list(dataset["Src"], self.source.ontology)
        target_entities = Entity.load_from_list(dataset["Tgt"], self.target.ontology)
            
        dataset["Features"] = [self.add_dynamic_information(source, target, self.static_skeletons) 
                               for source, target in zip(source_entities, target_entities)]
        
        self.log("Dynamic prompts generated.", level="debug")
        self.log(f"Generated {len(dataset['Features'])*self.static_skeletons} Prompts", level="debug")
        
        return dataset
    
    def add_dynamic_information(self, source: Entity, target: Entity, skeletons: List[str]) -> List[str]:

        dynamic_queries = []

        for static_skeleton, example, positive_examples, negative_examples, label_cardinality, separator, context_type, context_cardinality in zip(
            skeletons, self.example, self.positive_examples, self.negative_examples, self.label_cardinality, self.separator, self.context_type, self.context_cardinality
        ):

            query = self.apply_examples(static_skeleton, example, positive_examples, negative_examples, self.reference, self.negatives)
            query = self.apply_label_information(source, target, query, label_cardinality, separator)
            query = self.apply_context_information(source, target, query, context_type, context_cardinality, separator)

            dynamic_queries.append(query)

        return dynamic_queries

    @staticmethod
    def apply_task_context(skeleton: str, task_context: bool):
        return skeleton.replace('$TC', StaticPrompts.get_task_context(task_context))
    
    @staticmethod
    def apply_comparison_type(skeleton: str, comparison_type: ComparisonType):

        if comparison_type is not None:
            return skeleton.replace('$TYPE', comparison_type.replace("_", " "))

        return skeleton.replace('$TYPE', '')
    
    @staticmethod
    def apply_context_semantics(skeleton: str, context_semantics: ContextSemantics):
        if context_semantics is not None:
            context_semantics = context_semantics.replace("_", " ")
            skeleton = skeleton.replace('$CTX_S', f"{context_semantics} $CTX_S")
            return skeleton.replace('$CTX_T', f"{context_semantics} $CTX_T")
        return skeleton
    
    @staticmethod
    def apply_instruction_information(skeleton: str):
        return skeleton.replace('$I', StaticPrompts.INSTRUCTION)
    
    @staticmethod
    def apply_confidence(skeleton: str, likelihood: Likelihood):
        return skeleton.replace('$CONF', StaticPrompts.get_confidence(likelihood))
    
    @classmethod
    def apply_examples(cls, skeleton: str, example: bool, positive_examples: int, negative_examples: int, reference: DataFrame, negatives: DataFrame) -> str:
        if example:
            examples = ""
            if reference is not None:
                examples = cls._get_examples(reference, positive_examples, random.randint(0, 2**32 - 1), True, examples)

                if negatives is not None:
                    examples = cls._get_examples(negatives, negative_examples, random.randint(0, 2**32 - 1), False, examples)

            return skeleton.replace('$E', examples)

        return skeleton.replace('$E', '')
    
    @classmethod
    def apply_label_information(cls, source: Entity, target: Entity, skeleton: str, label_cardinality: int, separator: Separator) -> str:

        source_labels = cls._format_labels(source.labels, label_cardinality, separator)
        target_labels = cls._format_labels(target.labels, label_cardinality, separator)

        return skeleton.replace('$S', source_labels).replace('$T', target_labels)
    
    @classmethod
    def apply_context_information(cls, source: Entity, target: Entity, skeleton: str, context_type: ContextType, context_cardinality: int, separator: Separator) -> str:

        if context_type is None or context_cardinality <= 0:
            return skeleton.replace('$CTX_S', '').replace('$CTX_T', '')

        source_context = cls._format_labels(getattr(source, context_type), context_cardinality, separator)
        target_context = cls._format_labels(getattr(target, context_type), context_cardinality, separator)

        return skeleton.replace('$CTX_S', source_context).replace('$CTX_T', target_context)

    @staticmethod
    def _format_labels(labels: List[str], cardinality: int, separator: Separator) -> str:

        # Ensure cardinality is a positive integer and labels is not empty
        if cardinality <= 0 or not labels:
            return ""
        
        # Limit the number of labels to the specified cardinality
        if len(labels) > cardinality:
            labels = labels[:cardinality]
        
        # If there's only one label or cardinality is 1, return the label directly
        if len(labels) == 1 or cardinality == 1:
            return labels[0]
        
        # Else format the labels based on the specified separator
        if separator == Separator.comma:
            labels_string = ", ".join(labels[1:])
        elif separator == Separator.paranthesis:
            labels_string = "("+") (".join(labels[1:])+")"
        else:
            labels_string = ""
        
        return f"{labels[0]} {labels_string}"
    
    @staticmethod
    def _get_examples(dataset: DataFrame, num_examples: int, random_state: int, solution: bool, examples: str = "") -> str:
        
        dataset = dataset.sample(num_examples, random_state=random_state).reset_index(drop=True)
        
        return examples + "".join([
            StaticPrompts.get_example(source, target, solution, True)
            if examples == ""
            else StaticPrompts.get_example(source, target, solution, False)
            for source, target in zip(dataset["Src"].tolist(), dataset["Tgt"].tolist()) 
        ])


