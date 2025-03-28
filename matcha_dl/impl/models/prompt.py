import torch
from typing import List, Tuple, Union
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
from enum import Enum

import pandas as pd

from matcha_dl.core.contracts.model import IModel
from matcha_dl.core.entities.dataset import StaticPrompts

class AggregationStrategy(str, Enum):
    max = "max"
    min = "min"
    mean = "mean"

class UnexpectedResponseCount:
    def __init__(self):
        self.multiple_signals = 0
        self.no_conf = 0
        self.no_signal = 0

    def __repr__(self):
        return f"UnexpectedResponseCount(multiple_signals={self.multiple_signals}, no_conf={self.no_conf}, no_signal={self.no_signal})"
    
    def reset(self):
        self.multiple_signals = 0
        self.no_conf = 0
        self.no_signal = 0

class PromptClassifier(IModel):

    positive_pattern = re.compile(r'\b' + re.escape(StaticPrompts.POSITIVE_SOLUTION.lower()) + r'\b')
    negative_pattern = re.compile(r'\b' + re.escape(StaticPrompts.NEGATIVE_SOLUTION.lower()) + r'\b')
    uncertain_pattern = re.compile(r'\b' + re.escape(StaticPrompts.UNCERTAIN_SOLUTION.lower()) + r'\b')
    pattern_very = re.compile(r'\b' + re.escape(StaticPrompts.VERY_POSITIVE_CONFIDENCE.lower()) + r'\b')
    pattern_not  = re.compile(r'\b' + re.escape(StaticPrompts.NEGATIVE_CONFIDENCE.lower()) + r'\b')
    pattern_conf = re.compile(r'\b' + re.escape(StaticPrompts.POSITIVE_CONFIDENCE.lower()) + r'\b')

    def __init__(self, 
                 model_name: str, 
                 max_length: int = 50, 
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.mean,
                 use_critic: bool = False,
                 critic_weight: float = 0.5,
                 critic_uncertainty: bool = False,
                 use_lora: bool = False, 
                 lora_r: int = 16, 
                 lora_alpha: int = 32, 
                 lora_dropout: float = 0.05,
                 apply_sigmoid: bool = True,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.use_critic = use_critic
        self.critic_weight = critic_weight
        self.critic_uncertainty = critic_uncertainty
        self.max_length = max_length
        self.aggregation_strategy = AggregationStrategy(aggregation_strategy)
        self.apply_sigmoid = apply_sigmoid
        self.device = 'cpu'
        self.tokenizer = None
        self.model = None
        self.unexpected_response_count = UnexpectedResponseCount()

        self._load_model_and_tokenizer()

        if use_lora:
            if 't5' in self.model_name.lower():
                config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=["q", "v"]
                )
            else:
                config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=["q_proj", "v_proj"]
                )
            self.model = get_peft_model(self.model, config)

    def __repr__(self):
        return self.model.__repr__()

    def _load_model_and_tokenizer(self):
        if 't5' in self.model_name:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def to(self, device: str):
        self.device = device
        super().to(device)
        self.model.to(device)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.train(mode)

    def eval(self):
        super().eval()
        self.model.eval()

    def aggregate_responses(self, confidences: torch.Tensor) -> torch.Tensor:
        # confidences is of shape [batch_size, num_prompts]
        # return aggregated confidence based on the specified aggregation strategy
        # shape of returned tensor is [batch_size]
        if confidences.shape[-1] == 1:
            return confidences.squeeze(-1)
        elif self.aggregation_strategy == AggregationStrategy.mean:
            return confidences.mean(dim=-1)
        elif self.aggregation_strategy == AggregationStrategy.max:
            return confidences.max(dim=-1).values
        elif self.aggregation_strategy == AggregationStrategy.min:
            return confidences.min(dim=-1).values
    
    def vectorized_parse_response(self, responses: List[str]) -> torch.Tensor:
        """
        Vectorizes the parsing of textual responses.
        Maps responses to confidence values in [-1, 1] based on numeric extraction and keywords.
        """
        # Convert responses to a Pandas Series and clean them
        s = pd.Series(responses).str.lower().str.strip()
        
        # Extract numeric values; errors become NaN
        extracted = s.str.extract(r'(?P<value>\d+(\.\d+)?)')['value']
        numeric_conf = pd.to_numeric(extracted, errors='coerce')
        numeric_conf = numeric_conf.clip(lower=0.0, upper=1.0)
        
        # Use numeric values where available
        conf = numeric_conf.copy()
        # For responses without a numeric value, check keywords
        conf[s.str.contains(self.pattern_very, na=False) & conf.isna()] = 0.9
        conf[s.str.contains(self.pattern_not, na=False) & conf.isna()] = 0.25
        conf[s.str.contains(self.pattern_conf, na=False) & conf.isna()] = 0.75

        # Count responses that do not match any condition
        self.unexpected_response_count.no_conf += int(conf.isna().sum())
        # Fill NaN values with 1.0
        conf = conf.fillna(1.0)
        
        # Determine polarity based on solution keywords
        negative_mask = s.str.contains(self.negative_pattern, na=False)
        positive_mask = s.str.contains(self.positive_pattern, na=False)
        
        # Adjust polarity: if negative, negate
        conf = conf.where(~negative_mask, -conf)
        # Set uncertain responses (those that match neither or match both) to 0.0
        conf = conf.where(positive_mask ^ negative_mask, 0.0)

        # Count unexpected responses
        # those that match both positive and negative
        multiple_mask = positive_mask & negative_mask
        self.unexpected_response_count.multiple_signals += int(multiple_mask.sum())

        # those that do not match any condition
        unexpected_mask = ~(positive_mask | negative_mask)
        self.unexpected_response_count.no_signal += int(unexpected_mask.sum())
        
        return torch.tensor(conf.values, dtype=torch.float32, device=self.device)
    
    def vectorized_parse_critic_responses(self, critic_responses: List[str], fallback: torch.Tensor) -> torch.Tensor:
        """
        Given a list of critic responses and a fallback tensor (the original model confidences),
        return a tensor of critic labels (1.0 for positive, -1.0 for negative, 0.0 for uncertain).
        
        Any response that does not match any condition falls back to the corresponding fallback value.
        """
        # Convert responses to a Pandas Series and clean them
        s = pd.Series(critic_responses).str.lower().str.strip()
        
        # Create boolean masks for each condition
        pos_mask = s.str.contains(self.positive_pattern, na=False)
        neg_mask = s.str.contains(self.negative_pattern, na=False)
        uncertain_mask = s.str.contains(self.uncertain_pattern, na=False)

        # Count unexpected responses 
        
        # Initialize critic labels with zeros
        critic_labels = pd.Series(0.0, index=s.index)
        # Set labels based on the masks
        critic_labels[pos_mask] = 1.0
        critic_labels[neg_mask] = -1.0
        critic_labels[uncertain_mask] = 0.0
        
        # For entries that did not match any condition, fallback to the corresponding predicted confidence
        fallback_np = fallback.cpu().numpy()
        mask_sum = pos_mask.astype(int) + neg_mask.astype(int) + uncertain_mask.astype(int)
        recognized_mask = mask_sum == 1
        critic_labels[~recognized_mask] = fallback_np[~recognized_mask]

        # Count unexpected responses
        # those that match more then one mask
        multiple_mask = mask_sum > 1
        self.unexpected_response_count.multiple_signals += int(multiple_mask.sum())
        #those that do not match any condition
        unrecognized_mark = mask_sum == 0
        self.unexpected_response_count.no_signal += int((unrecognized_mark).sum())
        
        # Return as a torch tensor
        return torch.tensor(critic_labels.values, dtype=torch.float32, device=self.device)

    def get_critics(self, decoded_prompts: List[str], decoded_responses: List[str], predicted_confidences: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        critic_prompts = [
            StaticPrompts.get_critic(prompt, response, self.critic_uncertainty)
            for prompt, response in zip(decoded_prompts, decoded_responses)
        ]
        critic_inputs = self.tokenizer(critic_prompts, return_tensors='pt', padding=True, truncation=True)
        critic_output_ids = self.model.generate(
            input_ids=critic_inputs['input_ids'].to(self.device), 
            attention_mask=critic_inputs['attention_mask'].to(self.device), 
            max_length=self.max_length, 
            do_sample=False
        )
        decoded_critic_responses = self.tokenizer.batch_decode(critic_output_ids, skip_special_tokens=True)
        
        flat_confidences = predicted_confidences.view(-1)
        critic_labels = self.vectorized_parse_critic_responses(decoded_critic_responses, flat_confidences)
        
        signal = torch.sign(flat_confidences)
        updated_confidences = (1 - self.critic_weight) * flat_confidences + self.critic_weight * critic_labels * signal
        return updated_confidences.view(*shape)

    def forward(self, batch_prompts: torch.Tensor, attention_masks: torch.Tensor) -> torch.Tensor:
        """
        Expects:
          - batch_prompts and attention_masks of shape [batch_size, num_prompts, seq_len]
            where each batch element represents an independent question (source-target pair)
            that may have multiple prompt variations.
          
        The model flattens the prompt dimension, processes all prompt variants,
        then aggregates the confidences over the prompt dimension to produce one score per question.
        """

        batch_size, num_prompts, seq_len = batch_prompts.size()
        flat_prompts = batch_prompts.view(batch_size * num_prompts, seq_len)
        flat_masks = attention_masks.view(batch_size * num_prompts, seq_len)

        output_ids = self.model.generate(input_ids=flat_prompts, 
                                         attention_mask=flat_masks, 
                                         max_length=self.max_length, 
                                         do_sample=False)

        decoded_responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        conf_tensor = self.vectorized_parse_response(decoded_responses).view(batch_size, num_prompts)

        if self.use_critic:
            decoded_prompts = self.tokenizer.batch_decode(flat_prompts, skip_special_tokens=True)
            conf_tensor = self.get_critics(decoded_prompts, decoded_responses, conf_tensor, (batch_size, num_prompts))

        aggregated = self.aggregate_responses(conf_tensor)
        return torch.sigmoid(aggregated) if self.apply_sigmoid else aggregated