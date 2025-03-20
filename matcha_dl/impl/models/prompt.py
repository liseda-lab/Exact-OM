import torch
from typing import List, Tuple, Union
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from enum import Enum

from matcha_dl.core.contracts.model import IModel
from matcha_dl.core.entities.dataset import StaticPrompts

class AggregationStrategy(str, Enum):
    max = "max"
    min = "min"
    mean = "mean"

class PromptModel(IModel, torch.nn.Module):
    def __init__(self, 
                 model_name: str, 
                 use_critic: bool = False, 
                 max_length: int = 50, 
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.mean,
                 use_lora: bool = False, 
                 lora_r: int = 16, 
                 lora_alpha: int = 32, 
                 lora_dropout: float = 0.05,
                 **kwargs
        ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.use_critic = use_critic
        self.max_length = max_length
        self.aggregation_strategy = AggregationStrategy(aggregation_strategy)
        self.device = 'cpu'

        if use_lora:
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"]
            )
            self.model = get_peft_model(self.model, config)

    def to(self, device: str):
        self.device = device
        self.model.to(device)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.train(mode)

    def eval(self):
        super().eval()
        self.model.eval()

    def parse_response(self, response: str) -> Tuple[str, float]:
        response = response.lower().strip()
        agreement = 0.0  # Default to 0 if not found
        confidence = None

        match = re.search(r'(?P<value>\d+(\.\d+)?)', response)
        if match:
            confidence = float(match.group('value')) / 10 if float(match.group('value')) > 1 else float(match.group('value'))
        elif 'very confident' in response:
            confidence = 0.9
        elif 'confident' in response:
            confidence = 0.75
        elif 'not confident' in response:
            confidence = 0.25

        if StaticPrompts.POSITIVE_SOLUTION in response:
            agreement = 1.0
        elif StaticPrompts.NEGATIVE_SOLUTION in response:
            agreement = 0.0

        # If confidence is None, use agreement directly
        if confidence is None:
            probability = agreement
        else:
            # If disagreement, invert probability
            probability = confidence if agreement == 1.0 else 1.0 - confidence

        return response, probability

    def aggregate_responses(self, responses: torch.Tensor, strategy: AggregationStrategy) -> torch.Tensor:
        if responses.shape[1] == 1:
            return responses[:, 0, 1]
        
        if strategy == AggregationStrategy.max:
            max_indices = responses[:, :, 1].argmax(dim=1)
            return responses[torch.arange(responses.shape[0]), max_indices, 1]
        elif strategy == AggregationStrategy.min:
            min_indices = responses[:, :, 1].argmin(dim=1)
            return responses[torch.arange(responses.shape[0]), min_indices, 1]
        elif strategy == AggregationStrategy.mean:
            return torch.mean(responses[:, :, 1], dim=1)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def get_critics(self, batch_prompts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        batch_size, num_questions, num_prompts, seq_len = batch_prompts.size()
        critic_prompts = [
                StaticPrompts.CRITIC_SKELETON.replace('$P', prompt).replace('$R', resp)
                for prompt, resp in zip(self.tokenizer.batch_decode(batch_prompts, skip_special_tokens=True), responses.view(-1, 2).tolist())
        ]
        critic_inputs = self.tokenizer(critic_prompts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        critic_output_ids = self.model.generate(input_ids=critic_inputs['input_ids'], attention_mask=critic_inputs['attention_mask'], max_length=self.max_length)
        critic_responses = [self.parse_response(self.tokenizer.decode(out, skip_special_tokens=True)) for out in critic_output_ids]
        critic_responses = torch.tensor(critic_responses, dtype=torch.float32).view(batch_size, num_questions, num_prompts, 2)
        return critic_responses

    def forward(self, batch_prompts: torch.Tensor, attention_masks: torch.Tensor) -> torch.Tensor:
        batch_size, num_questions, num_prompts, seq_len = batch_prompts.size()
        
        batch_prompts = batch_prompts.view(batch_size * num_questions * num_prompts, seq_len)
        attention_masks = attention_masks.view(batch_size * num_questions * num_prompts, seq_len)
        
        output_ids = self.model.generate(input_ids=batch_prompts, attention_mask=attention_masks, max_length=self.max_length)
        decoded_responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        responses = list(map(self.parse_response, decoded_responses))
        
        responses = torch.tensor(responses, dtype=torch.float32).view(batch_size, num_questions, num_prompts, 2)
        
        if self.use_critic:
            responses = self.get_critics(batch_prompts, responses)
        
        return self.aggregate_responses(responses, strategy=self.aggregation_strategy)