import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device as tdevice
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from enum import Enum
from matcha_dl.core.contracts.model import IModel

class PoolingStrategy(str, Enum):
    sum        = "sum"
    max        = "max"
    cls        = "cls"
    sum_max    = "sum_max"
    cls_mean   = "cls_mean"
    attentive  = "attentive"

class AttentivePooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)
    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: [B*2, L, H], mask: [B*2, L]
        scores = self.attn(h).squeeze(-1)                      # [B*2, L]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        alphas = F.softmax(scores, dim=-1).unsqueeze(-1)       # [B*2, L, 1]
        return (h * alphas).sum(dim=1)                         # [B*2, H]

class EncoderClassifier(IModel):
    def __init__(
        self,
        encoder_name: str,
        pooling: PoolingStrategy = PoolingStrategy.sum,
        use_lora: bool        = False,
        lora_r: int           = 16,
        lora_alpha: int       = 32,
        lora_dropout: float   = 0.05,
        fp16_inference: bool  = False,
        use_classifier: bool  = False
    ):
        super().__init__()
        self.pooling         = PoolingStrategy(pooling)
        self.use_lora        = use_lora
        self.fp16_inference  = fp16_inference
        self.use_classifier  = use_classifier

        # base encoder + tokenizer
        self.encoder   = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)

        # optionally LoRA
        if self.use_lora:
            lora_cfg = LoraConfig(
                r            = lora_r,
                lora_alpha   = lora_alpha,
                lora_dropout = lora_dropout,
                target_modules=["query","value"],
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            self.encoder = get_peft_model(self.encoder, lora_cfg)
            for n,p in self.encoder.named_parameters():
                p.requires_grad = 'lora_' in n

        # optional FP16
        if self.fp16_inference:
            self.encoder.half()

        # pooling‐specific modules
        hidden_size = self.encoder.config.hidden_size
        if self.pooling is PoolingStrategy.attentive:
            self.attentive_pool = AttentivePooling(hidden_size)

        # optional supervised head
        if self.use_classifier:
            # embedding size depends on pooling:
            dim = hidden_size
            if self.pooling is PoolingStrategy.sum_max or self.pooling is PoolingStrategy.cls_mean:
                dim = hidden_size * 2
            self.classifier = nn.Linear(dim * 2, 1)  # we concat src & tgt

        self.device = None

    def to(self, device: tdevice):
        self.device = device
        self.encoder.to(device)
        if self.use_classifier:
            self.classifier.to(device)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.train(mode)
        if self.use_classifier:
            self.classifier.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.use_classifier:
            self.classifier.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # input_ids: [B,2,L], mask: [B,2,L]
        B, pair_size, L = input_ids.size()
        flat_ids  = input_ids.view(-1, L)    # [B*2, L]
        flat_mask = attention_mask.view(-1, L)

        # encode
        outputs = self.encoder(
            input_ids     = flat_ids,
            attention_mask= flat_mask
        ).last_hidden_state  # [B*2, L, H]

        mask = flat_mask  # [B*2, L]

        # pooling
        if self.pooling is PoolingStrategy.sum:
            pooled = (outputs * mask.unsqueeze(-1)).sum(dim=1)
        elif self.pooling is PoolingStrategy.max:
            neg_inf = torch.finfo(outputs.dtype).min
            pooled = outputs.masked_fill(mask.unsqueeze(-1)==0, neg_inf).max(dim=1)[0]
        elif self.pooling is PoolingStrategy.sum_max:
            sum_p = (outputs * mask.unsqueeze(-1)).sum(dim=1)
            neg_inf = torch.finfo(outputs.dtype).min
            max_p = outputs.masked_fill(mask.unsqueeze(-1)==0, neg_inf).max(dim=1)[0]
            pooled = torch.cat([sum_p, max_p], dim=-1)
        elif self.pooling is PoolingStrategy.cls_mean:
            cls_p  = outputs[:,0,:]
            mean_p = (outputs * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            pooled = torch.cat([cls_p, mean_p], dim=-1)
        elif self.pooling is PoolingStrategy.cls:
            pooled = outputs[:,0,:]
        elif self.pooling is PoolingStrategy.attentive:
            pooled = self.attentive_pool(outputs, mask)
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling}")

        # reshape → [B,2,D]
        D = pooled.size(-1)
        pooled = pooled.view(B, pair_size, D)
        src, tgt = pooled[:,0,:], pooled[:,1,:]

        # if supervised head: concat & classify
        if self.use_classifier:
            pair = torch.cat([src, tgt], dim=-1)  # [B, 2D]
            logits = self.classifier(pair).squeeze(-1)
            return logits

        # else unsupervised cosine
        return F.cosine_similarity(src, tgt, dim=-1)