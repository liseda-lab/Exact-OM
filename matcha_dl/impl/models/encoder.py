import torch
import torch.nn.functional as F
from torch import device as tdevice
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from enum import Enum

from matcha_dl.core.contracts.model import IModel

class PoolingStrategy(str, Enum):
    mean = "mean"
    max = "max"
    cls = "cls"

class EncoderClassifier(IModel):
    def __init__(
        self,
        encoder_name: str,
        pooling: PoolingStrategy = PoolingStrategy.mean,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        fp16_inference: bool = False
    ):
        """
        Parameters:
            encoder_name: The name of the pretrained encoder.
            pooling: One of "mean", "max", or "cls".
            use_lora: Whether to apply LoRA to the encoder (using peft package).
            lora_r: LoRA low-rank dimension.
            lora_alpha: LoRA scaling factor.
            lora_dropout: Dropout probability for LoRA.
            fp16_inference: Whether to convert model to FP16 for inference.
        """

        super(EncoderClassifier, self).__init__()
        self.encoder_name = encoder_name
        self.pooling = pooling
        self.use_lora = use_lora
        self.fp16_inference = fp16_inference

        # Load base encoder and tokenizer
        self.encoder = AutoModel.from_pretrained(self.encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)

        # Optionally apply LoRA adapters
        if self.use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "value"],
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
            # Freeze all except LoRA parameters
            for name, param in self.encoder.named_parameters():
                param.requires_grad = 'lora_' in name

        # If we want pure FP16 inference, convert weights once
        if self.fp16_inference:
            self.encoder.half()

        self.device = None

    def to(self, device: tdevice):
        self.device = device
        # Move entire model to device
        self.encoder.to(device)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        input_ids: [batch_size, 2, seq_len]
        attention_mask: [batch_size, 2, seq_len]
        returns: [batch_size] cosine similarity scores
        """
        # Flatten pairs: [batch*2, seq_len]
        bs, pair_size, seq_len = input_ids.size()
        flat_ids  = input_ids.reshape(-1, seq_len)
        flat_mask = attention_mask.reshape(-1, seq_len)

        # Encode tokens
        outputs = self.encoder(
            input_ids=flat_ids,
            attention_mask=flat_mask
        ).last_hidden_state  # [bs*2, seq_len, hidden_dim]

        # Prepare mask for pooling: [bs*2, seq_len, 1]
        mask = flat_mask.unsqueeze(-1)

        # Pooling
        if self.pooling == PoolingStrategy.mean:
            # sum then divide by lengths
            sum_emb = (outputs * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_emb / lengths
        elif self.pooling == PoolingStrategy.max:
            # mask then max
            neg_inf = torch.finfo(outputs.dtype).min
            pooled = outputs.masked_fill(mask == 0, neg_inf).max(dim=1)[0]
        else:  # cls pooling
            pooled = outputs[:, 0, :]

        # Restore [batch, 2, hidden_dim]
        pooled = pooled.view(bs, pair_size, -1)
        src_emb = pooled[:, 0, :]
        tgt_emb = pooled[:, 1, :]

        # Cosine similarity
        return F.cosine_similarity(src_emb, tgt_emb, dim=-1)
