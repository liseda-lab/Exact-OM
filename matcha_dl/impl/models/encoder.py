import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device as tdevice
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from enum import Enum
from typing import Optional
from matcha_dl.core.contracts.model import IModel

class PoolingStrategy(str, Enum):
    sum        = "sum"
    max        = "max"
    cls        = "cls"
    sum_max    = "sum_max"
    cls_mean   = "cls_mean"
    attentive  = "attentive"

class FusionStrategy(str, Enum):
    static             = "static"
    gated              = "gated"
    mlp                = "mlp"
    multitask          = "multitask"
    weighted_cosine    = "weighted_cosine"

class AttentivePooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)
    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(h).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        alphas = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (h * alphas).sum(dim=1)

class EncoderClassifier(IModel):
    def __init__(
        self,
        encoder_name: str,
        pooling: PoolingStrategy = PoolingStrategy.sum,
        fusion: FusionStrategy = FusionStrategy.static,
        weight_init: float = 0.5,
        mlp_hidden_dim: Optional[int] = None,
        mlp_use_layernorm: bool = False,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        use_lora: bool        = False,
        lora_r: int           = 16,
        lora_alpha: int       = 32,
        lora_dropout: float   = 0.05,
        fp16_inference: bool  = False,
        use_classifier: bool  = False,
        **kwargs
    ):
        super().__init__()
        self.pooling             = PoolingStrategy(pooling)
        self.fusion_strategy     = FusionStrategy(fusion)
        self.use_lora            = use_lora
        self.fp16_inference      = fp16_inference
        self.use_classifier      = use_classifier
        self.mlp_use_layernorm   = mlp_use_layernorm

        # base encoder + tokenizer
        self.encoder   = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

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

        hidden_size = self.encoder.config.hidden_size
        if self.pooling is PoolingStrategy.attentive:
            self.attentive_pool = AttentivePooling(hidden_size)

        # dropout layers
        self.fusion_dropout     = nn.Dropout(dropout)
        self.classifier_dropout = nn.Dropout(dropout)

        # Fusion-specific params
        if self.fusion_strategy is FusionStrategy.static:
            self.src_weight = nn.Parameter(torch.tensor(weight_init, device="cpu"))
            self.tgt_weight = nn.Parameter(torch.tensor(weight_init, device="cpu"))
        elif self.fusion_strategy is FusionStrategy.gated:
            self.gate_src = nn.Sequential(
                nn.Linear(hidden_size * 2, 1),
                nn.Sigmoid()
            )
            self.gate_tgt = nn.Sequential(
                nn.Linear(hidden_size * 2, 1),
                nn.Sigmoid()
            )
        elif self.fusion_strategy is FusionStrategy.mlp:
            mlp_h = mlp_hidden_dim or hidden_size
            layers = [nn.Linear(hidden_size * 4, mlp_h)]
            if self.mlp_use_layernorm:
                layers.append(nn.LayerNorm(mlp_h))
            layers += [
                nn.ReLU(),
                nn.Dropout(self.fusion_dropout),
                nn.Linear(mlp_h, 1)
            ]
            self.mlp = nn.Sequential(*layers)
        elif self.fusion_strategy is FusionStrategy.multitask:
            self.head_label = nn.Linear(hidden_size, 1)
            self.head_ctx   = nn.Linear(hidden_size, 1)
            self.alpha      = nn.Parameter(torch.tensor(weight_init, device="cpu"))
        elif self.fusion_strategy is FusionStrategy.weighted_cosine:
            self.cos_alpha = nn.Parameter(torch.tensor(weight_init, device="cpu"))

        if self.use_classifier and self.fusion_strategy in (FusionStrategy.static, FusionStrategy.gated):
            dim = hidden_size
            self.classifier = nn.Linear(dim * 2, 1)


        print(f"EncoderClassifier initialized with pooling={self.pooling}, fusion={self.fusion_strategy}, "
              f"lora={self.use_lora}, fp16={self.fp16_inference}, classifier={self.use_classifier}, weight_init={weight_init}")

        
    def _pool(self, outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling is PoolingStrategy.sum:
            return (outputs * mask.unsqueeze(-1)).sum(dim=1)
        elif self.pooling is PoolingStrategy.max:
            neg_inf = torch.finfo(outputs.dtype).min
            return outputs.masked_fill(mask.unsqueeze(-1)==0, neg_inf).max(dim=1)[0]
        elif self.pooling is PoolingStrategy.sum_max:
            sum_p = (outputs * mask.unsqueeze(-1)).sum(dim=1)
            neg_inf = torch.finfo(outputs.dtype).min
            max_p = outputs.masked_fill(mask.unsqueeze(-1)==0, neg_inf).max(dim=1)[0]
            return torch.cat([sum_p, max_p], dim=-1)
        elif self.pooling is PoolingStrategy.cls_mean:
            cls_p  = outputs[:,0,:]
            mean_p = (outputs * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            return torch.cat([cls_p, mean_p], dim=-1)
        elif self.pooling is PoolingStrategy.cls:
            return outputs[:,0,:]
        elif self.pooling is PoolingStrategy.attentive:
            return self.attentive_pool(outputs, mask)
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, T, L = input_ids.size()
        flat_ids  = input_ids.view(-1, L)
        flat_mask = attention_mask.view(-1, L)

        hidden = self.encoder(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
        pooled = self._pool(hidden, flat_mask)
        H = pooled.size(-1)
        pooled = pooled.view(B, T, H)
        e_lbl_src, e_ctx_src, e_lbl_tgt, e_ctx_tgt = pooled[:,0,:], pooled[:,1,:], pooled[:,2,:], pooled[:,3,:]

        # fusion
        if self.fusion_strategy is FusionStrategy.static:
            src = self.src_weight * e_lbl_src + (1 - self.src_weight) * e_ctx_src
            tgt = self.tgt_weight * e_lbl_tgt + (1 - self.tgt_weight) * e_ctx_tgt
        elif self.fusion_strategy is FusionStrategy.gated:
            g_s = self.gate_src(torch.cat([e_lbl_src, e_ctx_src], dim=-1))
            g_t = self.gate_tgt(torch.cat([e_lbl_tgt, e_ctx_tgt], dim=-1))
            src = g_s * e_lbl_src + (1-g_s) * e_ctx_src
            tgt = g_t * e_lbl_tgt + (1-g_t) * e_ctx_tgt
        elif self.fusion_strategy is FusionStrategy.mlp:
            pair = torch.cat([e_lbl_src, e_ctx_src, e_lbl_tgt, e_ctx_tgt], dim=-1)
            out = self.mlp(pair).squeeze(-1)
            return out
        elif self.fusion_strategy is FusionStrategy.multitask:
            lbl_diff = e_lbl_src - e_lbl_tgt
            ctx_diff = e_ctx_src - e_ctx_tgt
            logit_lbl = self.head_label(lbl_diff).squeeze(-1)
            logit_ctx = self.head_ctx(ctx_diff).squeeze(-1)
            alpha = self.alpha
            return alpha * logit_lbl + (1 - alpha) * logit_ctx
        elif self.fusion_strategy is FusionStrategy.weighted_cosine:
            cos_lbl = F.cosine_similarity(e_lbl_src, e_lbl_tgt, dim=-1)
            cos_ctx = F.cosine_similarity(e_ctx_src, e_ctx_tgt, dim=-1)
            alpha = self.cos_alpha
            return alpha * cos_lbl + (1 - alpha) * cos_ctx
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # apply fusion dropout
        src = self.fusion_dropout(src)
        tgt = self.fusion_dropout(tgt)

        # final head / similarity
        if self.use_classifier and hasattr(self, 'classifier'):
            pair = torch.cat([src, tgt], dim=-1)
            pair = self.classifier_dropout(pair)
            return self.classifier(pair).squeeze(-1)
        return F.cosine_similarity(src, tgt, dim=-1)