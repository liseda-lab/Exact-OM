import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from enum import Enum

from matcha_dl.utils.models import extract_answer
from matcha_dl.core.contracts.model import IModel

class PoolingStrategy(str, Enum):
    mean = "mean"
    max = "max"
    cls = "cls"

class EncoderClassifier(IModel):
    def __init__(self,
                 encoder_name: str,
                 pooling: PoolingStrategy = PoolingStrategy.mean,
                 use_lora: bool = False,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05):
        """
        Parameters:
            encoder_name: The name of the pretrained encoder.
            pooling: One of "mean", "max", or "cls".
            use_lora: Whether to apply LoRA to the encoder (using peft package).
            lora_r: LoRA low-rank dimension.
            lora_alpha: LoRA scaling factor.
            lora_dropout: Dropout probability for LoRA.
        """
        super(EncoderClassifier, self).__init__()
        self.encoder_name = encoder_name
        self.pooling = PoolingStrategy(pooling)
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.device = None
        
        # Load the pre-trained encoder
        self.encoder = AutoModel.from_pretrained(self.encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        
        # Apply LoRA via peft if requested
        if self.use_lora:
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["query", "value"],
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            self.encoder = get_peft_model(self.encoder, lora_config)

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def max_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings = token_embeddings.clone()
        token_embeddings[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(token_embeddings, dim=1)[0]
        return max_embeddings

    def cls_pooling(self, token_embeddings, attention_mask):
        return token_embeddings[:, 0, :]
    
    def to(self, device: device):
        self.device = device
        super().to(device)
        self.encoder.to(device)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Expects:
            input_ids: Tensor of shape [batch_size, 2, seq_len] where the two slices correspond
                       to the tokenized source and target sequences.
        Returns:
            cosine_sim: Cosine similarity for each source-target pair in the batch.
        """
        batch_size, pair_size, seq_len = input_ids.size()  # pair_size should be 2
        # Flatten the input to shape [batch_size*2, seq_len]
        flat_input_ids = input_ids.view(batch_size * pair_size, seq_len).to(self.device)
        attention_mask = attention_mask.view(batch_size * pair_size, seq_len).to(self.device)

        outputs = self.encoder(input_ids=flat_input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # [batch_size*2, seq_len, hidden_dim]

        # Pool the token embeddings using the selected method
        if self.pooling is PoolingStrategy.mean:
            pooled_embeddings = self.mean_pooling(token_embeddings, attention_mask)
        elif self.pooling is PoolingStrategy.max:
            pooled_embeddings = self.max_pooling(token_embeddings, attention_mask)
        elif self.pooling is PoolingStrategy.cls:
            pooled_embeddings = self.cls_pooling(token_embeddings, attention_mask)
        else:
            raise ValueError("Invalid pooling method selected.")

        # Reshape back to [batch_size, 2, hidden_dim]
        pooled_embeddings = pooled_embeddings.view(batch_size, pair_size, -1)
        source_embeddings = pooled_embeddings[:, 0, :]
        target_embeddings = pooled_embeddings[:, 1, :]

        # Compute cosine similarity between source and target embeddings
        cosine_sim = F.cosine_similarity(source_embeddings, target_embeddings, dim=-1)
        return cosine_sim
