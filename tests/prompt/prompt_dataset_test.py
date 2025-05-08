import pandas as pd
import pytest
from matcha_dl.impl.datasets.tabular import TabularDataset 
from matcha_dl.core.entities.dataset import StaticPrompts
from matcha_dl.core.entities.configs.dataset import Separator, ComparisonType, ContextType, ContextSemantics, Likelihood
from matcha_dl.core.entities.ontology import Entity
from matcha_dl.impl.datasets.prompt import PromptDataset 

# Create dummy implementations for StaticPrompts functions/attributes.
StaticPrompts.SKELETON = "SKELETON $TC $TYPE $CTX_S $CTX_T $I $CONF $E $S $T"
StaticPrompts.INSTRUCTION = "INSTRUCTION"

# Monkeypatch functions to return deterministic strings.
@pytest.fixture(autouse=True)
def patch_static_prompts(monkeypatch):
    monkeypatch.setattr(StaticPrompts, "get_task_context", lambda tc: "TASK" if tc else "NO_TASK")
    monkeypatch.setattr(StaticPrompts, "get_confidence", lambda likelihood: "CONF" if likelihood else "NO_CONF")
    monkeypatch.setattr(StaticPrompts, "get_example", lambda src, tgt, solution, first: "EXAMPLE")
    # For consistency, you can also patch any other needed function.

# Create a dummy Entity class to be used by Entity.load_from_list.
class DummyEntity:
    def __init__(self, label):
        self.labels = [label, f"{label}_extra"]
        self.context = [f"{label}_ctx1", f"{label}_ctx2"]

# Monkeypatch Entity.load_from_list to return dummy entities.
@pytest.fixture(autouse=True)
def patch_entity(monkeypatch):
    monkeypatch.setattr(Entity, "load_from_list", lambda lst, ontology: [DummyEntity(x) for x in lst])

@pytest.fixture
def dummy_dataframe():
    # Create a small dataframe with Src and Tgt columns.
    return pd.DataFrame({
        "Src": ["source1", "source2"],
        "Tgt": ["target1", "target2"]
    })

@pytest.fixture
def dummy_prompt_dataset(dummy_dataframe):
    # Create a dummy PromptDataset instance with fixed parameters.
    # For lists that drive static skeleton generation, we use two elements.
    pdataset = PromptDataset(
        example=[True, True],
        positive_examples=[1, 1],
        negative_examples=[0, 0],
        task_context=[True, False],
        separator=[Separator.comma, Separator.comma],
        comparison_type=["COMPARISON_TYPE", "COMPARISON_TYPE"],
        label_cardinality=[2, 2],
        context_type=["context", "context"],  # context attribute name in DummyEntity
        context_cardinality=[2, 2],
        context_semantics=["SEMANTICS", "SEMANTICS"],
        likelihood=[True, False],
        dataframe=dummy_dataframe  # if needed by TabularDataset
    )
    # Set required attributes inherited from TabularDataset.
    pdataset.source = type("DummySource", (), {"ontology": "dummy_ontology"})()
    pdataset.target = type("DummyTarget", (), {"ontology": "dummy_ontology"})()
    pdataset.reference = pd.DataFrame({"Src": ["ref_src"], "Tgt": ["ref_tgt"]})
    pdataset.negatives = pd.DataFrame({"Src": ["neg_src"], "Tgt": ["neg_tgt"]})
    return pdataset

def test_generate_prompts(dummy_prompt_dataset, dummy_dataframe):
    # Call generate_prompts on a copy of the dummy dataframe.
    result_df = dummy_prompt_dataset.generate_prompts(dummy_dataframe.copy())
    
    # Verify that the Features column is added.
    assert "Features" in result_df.columns
    
    # Each row's Features should be a list of dynamic queries, one for each static skeleton.
    # The number of static skeletons is determined by the length of task_context (here, 2).
    for idx, row in result_df.iterrows():
        features = row["Features"]
        assert isinstance(features, list)
        assert len(features) == 2
        
        # Check that each dynamic prompt includes expected substrings from the replacements.
        # For example, it should include the skeleton base, instruction, and confidence strings.
        for prompt in features:
            assert "SKELETON" in prompt
            # Check that task context replacement is applied: one row should use "TASK", the other "NO_TASK"
            assert ("TASK" in prompt) or ("NO_TASK" in prompt)
            assert "INSTRUCTION" in prompt
            # Check that the confidence string is applied.
            assert ("CONF" in prompt) or ("NO_CONF" in prompt)
            # Check that example replacement happened.
            assert "EXAMPLE" in prompt
            # Check that label placeholders were replaced.
            # Since _format_labels returns source.labels[0] + extra labels, expect "source1" or "source1_extra"
            assert "source1" in prompt or "source2" in prompt

