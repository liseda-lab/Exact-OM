#dataset_params:
#  ## Validation set to be used to stop training, and in training guidance.
#  ## If Null or not declared, no validation set will be used.
#  validation_set: 0.1
#  # Can also  be list if multiple prompts and then agregation is used majority if no likelihood and aggr param
#  example: [True] # bool                            -» InstructionInformation
#  task_context: [True] # bool                       -» TaskContext
#  separator: ['comma'] # comma, paranthesis         -» LabelInformation
#  label_type: ['Simple'] # Simple, Compound         -» LabelInformation (REDUNDANTE?)
#  label_cardinality: [1] # 3, 5, 100                -» LabelInformation
#  context_type : ['Parent'] # Parent, Child, Top    -» ClassInformation
#  context_cardinality: [1] # 3, 5, 100              -» ClassInformation (not accounted for yet)
#  context_semantics: [part_of] # part_of, kind_of, type_of, subclass_of, with_subclass, with_part, with_type, with_kind -» ClassInformation
#  likelihood: [float] # float, int, cat             -» Confidence
#  critic: [True] # bool                             -» Critic



from pathlib import Path
from typing import Optional

import random
import numpy as np
import pandas as pd

DataFrame = pd.DataFrame

from typing import List, Tuple

from matcha_dl.impl.datasets.tabular import TabularDataset
#from matcha_dl.core.entities.configs.prompt 


class PromptDataset(TabularDataset):

    def __init__(
                self,
                example: List[bool],
                task_context: List[bool],
                separator: List[str],
                label_type: List[str],
                label_cardinality: int,
                context_type: str,
                context_cardinality: int,
                context_semantics: str,
                likelihood: str,
                critic: bool,
                **kwargs
        ) -> None:

        super().__init__(**kwargs)

        self._example = example
        self._task_context = task_context
        self._separator = separator
        self._label_type = label_type
        self._label_cardinality = label_cardinality
        self._context_type = context_type
        self._context_cardinality = context_cardinality
        self._context_semantics = context_semantics
        self._likelihood = likelihood
        self._critic = critic
    
    @property
    def example(self) -> bool:
        return self._example
    
    @property
    def task_context(self) -> bool:
        return self._task_context
    
    @property
    def separator(self) -> str:
        return self._separator
    
    @property
    def label_type(self) -> str:
        return self._label_type
    
    @property
    def label_cardinality(self) -> int:
        return self._label_cardinality
    
    @property
    def context_type(self) -> str:
        return self._context_type
    
    @property
    def context_cardinality(self) -> int:
        return self._context_cardinality
    
    @property
    def context_semantics(self) -> str:
        return self._context_semantics
    
    @property
    def likelihood(self) -> str:
        return self._likelihood
    
    @property
    def critic(self) -> bool:
        return self._critic
    
    def process(self) -> "PromptDataset":

        if self.has_cache():
            self.load()
            return self

        return self


    def _generate_prompts(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Generates prompts for the dataset.
        """
        from matcha_dl.impl.datasets.temporary_prompt_aux import split_instances, generate_static_skeletons, generate_queries, ConfigMock

        config_mock = ConfigMock()

        # Read static and dynamic info from the configuration and format it
        # Both are a list of lists, in which the inner ones are lists of BaseInformation 
        # (the class on which we will do .process). There should be one inner list per "run" 
        # (corresponding to a column in the config), and each run should yield one query at the end.
        static_info = split_instances(config_mock.static_info)
        dynamic_info = split_instances(config_mock.dynamic_info)

        # Generate static skeletons from static info
        # This is a list of lists of strings (skeletons). There should be one skeleton per run.
        static_skeletons = generate_static_skeletons(static_info)

        all_queries = []

        for _, row in dataset.iterrows():

            try:
                source = row["Src"]
                target = row["Tgt"]

                # Use dynamic info to obtain queries from static skeletons
                # This is a list of lists of strings (queries or queries + critic queries). 
                # There should be one query per run.
                generated_queries = generate_queries(source, target, dynamic_info, static_skeletons)

            except AttributeError:
                self.log("Attribute error in query generation", level="error", exc_info=True)
                raise ValueError("Scores for source {} and target {} not found.".format(row["Src"], row["Tgt"]))
            
            all_queries.append(generated_queries)
            
        dataset["Features"] = all_queries

        print(queries)


