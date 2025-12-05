import torch
import pytest
import pandas as pd

from exact import init_jvm
init_jvm('32G')

# Import the PromptClassifier and AggregationStrategy from your module.
# Adjust the import path as needed.
from exact.impl.models.prompt import PromptClassifier, AggregationStrategy
from exact.core.entities.dataset import StaticPrompts

# --- Define dummy classes to replace the real tokenizer and model ---

class DummyTokenizer:
    def __init__(self, positive=True):
        """
        If positive is True, the dummy tokenizer's batch_decode returns a positive response.
        Otherwise, it returns a negative response.
        """
        self.positive = positive

    def batch_decode(self, output_ids, skip_special_tokens=True):
        # Return a list of dummy responses based on the positive flag.
        # The numeric part (0.85) is within [0,1] and will be interpreted as the base confidence.
        if self.positive:
            return ["0.85 yes"] * output_ids.size(0)
        else:
            return ["0.85 no"] * output_ids.size(0)

    def __call__(self, texts, return_tensors='pt', padding=True, truncation=True):
        # For simplicity, return dummy input_ids and attention_mask.
        batch_size = len(texts)
        seq_len = 10  # arbitrary sequence length for testing
        return {
            'input_ids': torch.ones((batch_size, seq_len), dtype=torch.long),
            'attention_mask': torch.ones((batch_size, seq_len), dtype=torch.long)
        }

class DummyModel:
    def generate(self, input_ids, attention_mask, max_length, do_sample):
        # Return a dummy tensor with the shape [batch_size, max_length].
        batch_size = input_ids.size(0)
        return torch.ones((batch_size, max_length), dtype=torch.long)
    
    def to(self, device):
        # Dummy method to simulate moving the model to a device.
        pass
    

# --- Create a dummy subclass of PromptClassifier to override model/tokenizer loading ---

class DummyPromptClassifier(PromptClassifier):

    def __init_subclass__(cls, **kwargs):
        pass

    def _load_model_and_tokenizer(self):
        # Instead of loading a real model and tokenizer, use our dummy implementations.
        # The dummy tokenizer is provided externally via the attribute "dummy_tokenizer".
        self.tokenizer = None
        self.model = DummyModel()

# --- Define pytest fixtures for positive and negative classifiers ---

@pytest.fixture
def positive_classifier():
    classifier = DummyPromptClassifier(
        model_name="dummy",
        aggregation_strategy=AggregationStrategy.mean,
        use_critic=False  # You can toggle this to test critic behavior separately.
    )
    classifier.tokenizer = DummyTokenizer(positive=True)
    classifier.to("cpu")
    return classifier

@pytest.fixture
def negative_classifier():
    classifier = DummyPromptClassifier(
        model_name="dummy",
        aggregation_strategy=AggregationStrategy.mean,
        use_critic=False
    )
    classifier.tokenizer = DummyTokenizer(positive=False)
    classifier.to("cpu")
    return classifier

# --- Define tests ---

def test_forward_positive(positive_classifier):
    """
    Test the forward method when the responses are positive (include 'yes').
    The expected confidence is 0.85 and after applying a sigmoid it should be > 0.5.
    """
    batch_size = 2
    num_prompts = 3
    seq_len = 10
    batch_prompts = torch.ones((batch_size, num_prompts, seq_len), dtype=torch.long)
    attention_masks = torch.ones((batch_size, num_prompts, seq_len), dtype=torch.long)
    
    output = positive_classifier(batch_prompts, attention_masks)
    # Expect one output per question (batch element)
    assert output.shape == (batch_size,)
    # Sigmoid(0.85) is roughly 0.70, so all outputs should be above 0.5.
    assert torch.all(output > 0.5)

def test_forward_negative(negative_classifier):
    """
    Test the forward method when the responses are negative (include 'no').
    The numeric part is 0.85 but the presence of 'no' will flip it to -0.85,
    so sigmoid(-0.85) should be below 0.5.
    """
    batch_size = 2
    num_prompts = 3
    seq_len = 10
    batch_prompts = torch.ones((batch_size, num_prompts, seq_len), dtype=torch.long)
    attention_masks = torch.ones((batch_size, num_prompts, seq_len), dtype=torch.long)
    
    output = negative_classifier(batch_prompts, attention_masks)
    assert output.shape == (batch_size,)
    # Sigmoid(-0.85) is roughly 0.30, so outputs should be below 0.5.
    assert torch.all(output < 0.5)

def test_vectorized_parse_response_uncertainty(positive_classifier):
    """
    Test that if responses do not contain either positive or negative keywords,
    the parsed confidence is 0.0 (uncertainty).
    """
    # Responses without expected keywords will yield NaN in the keyword check and be set to 0.0.
    responses = ["0.85", "0.5", "0.2"]
    tensor = positive_classifier.vectorized_parse_response(responses)
    assert torch.all(tensor == 0.0)

def test_vectorized_parse_response_keywords(positive_classifier):
    """
    Test that responses containing the expected keywords are parsed correctly.
    """
    responses = ["yes:0.85", "yes:0.7", "yes:0.2", "no:0.2", "yes:not confident", 
                 "yes:yes:0.25", "yes no: 0.25", "0.25", "yes"]
    tensor = positive_classifier.vectorized_parse_response(responses)
    # The extracted numeric values should be used since the positive keyword is found.
    expected = torch.tensor([0.85, 0.7, 0.2, -0.2, 0.25, 0.25, 0.0, 0.0, 1.0], dtype=torch.float32)
    assert torch.allclose(tensor, expected, atol=1e-4)
    assert positive_classifier.unexpected_response_count.multiple_signals == 1
    assert positive_classifier.unexpected_response_count.no_conf == 1
    assert positive_classifier.unexpected_response_count.no_signal == 1

def test_vectorized_parse_critic_responses(positive_classifier):
    """
    Test the critic parsing function.
    """
    critic_responses = ["yes", "no", "uncertain", "unknown", "yes yes", "no yes", "uncertain yes"]
    # Create a fallback tensor to use when no expected keyword is found.
    fallback = torch.tensor([0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85], dtype=torch.float32)
    tensor = positive_classifier.vectorized_parse_critic_responses(critic_responses, fallback)
    # Expected behavior:
    # "yes" -> 1.0, "no" -> -1.0, "uncertain" -> 0.0, "unknown" -> fallback value (0.85)
    expected = torch.tensor([1.0, -1.0, 0.0, 0.85, 1.0, 0.85, 0.85], dtype=torch.float32)
    assert torch.allclose(tensor, expected, atol=1e-4)
    assert positive_classifier.unexpected_response_count.multiple_signals == 2
    assert positive_classifier.unexpected_response_count.no_signal == 1


if __name__ == "__main__":
    pytest.main(["-v", __file__])