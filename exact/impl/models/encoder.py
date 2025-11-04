import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from enum import Enum
from typing import Optional, Tuple
from exact.core.contracts.model import IModel

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

class GroupPoolingStrategy(str, Enum):
    concat   = "concat"
    max_pair = "max_pair"

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
        fusion: FusionStrategy  = FusionStrategy.static,
        # dataset-wide maximum numbers of slots
        n_labels: int = 1,
        n_contexts: int = 1,
        label_pooling: GroupPoolingStrategy = GroupPoolingStrategy.max_pair,
        ctx_pooling:   GroupPoolingStrategy = GroupPoolingStrategy.max_pair,
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
        self.pooling         = PoolingStrategy(pooling)
        self.fusion_strategy = FusionStrategy(fusion)
        # max counts for padded input slots
        self.n_labels        = n_labels
        self.n_contexts      = n_contexts
        self.label_pooling   = GroupPoolingStrategy(label_pooling)
        self.ctx_pooling     = GroupPoolingStrategy(ctx_pooling)
        self.use_lora        = use_lora
        self.fp16_inference  = fp16_inference
        self.use_classifier  = use_classifier
        self.mlp_use_layernorm = mlp_use_layernorm

        # encoder & tokenizer
        self.encoder   = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        if freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad = False

        # LoRA adaptation
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

        if self.fp16_inference:
            self.encoder.half()

        hidden_size = self.encoder.config.hidden_size
        if self.pooling is PoolingStrategy.attentive:
            self.attentive_pool = AttentivePooling(hidden_size)

        # projections for concat pooling (using dataset-wide max)
        if self.label_pooling is GroupPoolingStrategy.concat:
            self.label_proj = nn.Linear(n_labels * hidden_size, hidden_size)
        if self.ctx_pooling is GroupPoolingStrategy.concat:
            self.ctx_proj   = nn.Linear(n_contexts * hidden_size, hidden_size)

        # dropout layers
        self.fusion_dropout     = nn.Dropout(dropout)
        self.classifier_dropout = nn.Dropout(dropout)

        # fusion parameters
        if self.fusion_strategy is FusionStrategy.static:
            self.src_weight = nn.Parameter(torch.tensor(weight_init))
            self.tgt_weight = nn.Parameter(torch.tensor(weight_init))
        elif self.fusion_strategy is FusionStrategy.gated:
            self.gate_src = nn.Sequential(nn.Linear(hidden_size * 2, 1), nn.Sigmoid())
            self.gate_tgt = nn.Sequential(nn.Linear(hidden_size * 2, 1), nn.Sigmoid())
        elif self.fusion_strategy is FusionStrategy.mlp:
            mlp_h = mlp_hidden_dim or hidden_size
            layers = [nn.Linear(hidden_size * 4, mlp_h)]
            if self.mlp_use_layernorm: layers.append(nn.LayerNorm(mlp_h))
            layers += [nn.ReLU(), self.fusion_dropout, nn.Linear(mlp_h, 1)]
            self.mlp = nn.Sequential(*layers)
        elif self.fusion_strategy is FusionStrategy.multitask:
            self.head_label = nn.Linear(hidden_size, 1)
            self.head_ctx   = nn.Linear(hidden_size, 1)
            self.alpha      = nn.Parameter(torch.tensor(weight_init))
        elif self.fusion_strategy is FusionStrategy.weighted_cosine:
            self.cos_alpha = nn.Parameter(torch.tensor(weight_init))

        # optional classifier head for static/gated
        if self.use_classifier and self.fusion_strategy in (FusionStrategy.static, FusionStrategy.gated):
            self.classifier = nn.Linear(hidden_size * 2, 1)

        print(
            f"EncoderClassifier init: pooling={self.pooling}, fusion={self.fusion_strategy},"
            f" max_labels={self.n_labels}, max_contexts={self.n_contexts},"
            f" label_pool={self.label_pooling}, ctx_pool={self.ctx_pooling},"
            f" lora={self.use_lora}, fp16={self.fp16_inference}, classifier={self.use_classifier}"
        )

    def _pool(self, outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # same single-sequence pooling as before
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

    def _encode_group(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # ids: (B, K_group, L) -> pooled: (B, K_group, H)
        B, K, L = ids.size()
        hidden = self.encoder(input_ids=ids.view(B*K, L), attention_mask=mask.view(B*K, L)).last_hidden_state
        pooled = self._pool(hidden, mask.view(B*K, L))
        return pooled.view(B, K, -1)
    
    def _pool_group(
        self,
        src_embs: torch.Tensor,
        tgt_embs: torch.Tensor,
        src_valid: torch.Tensor,
        tgt_valid: torch.Tensor,
        pooling: GroupPoolingStrategy,
        proj: Optional[nn.Module]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Ks, H = src_embs.size(); _, Kt, _ = tgt_embs.size()
        if pooling is GroupPoolingStrategy.max_pair:
            cosm = F.cosine_similarity(src_embs.unsqueeze(2), tgt_embs.unsqueeze(1), dim=-1)
            valid_pairs = src_valid.unsqueeze(2) & tgt_valid.unsqueeze(1)
            cosm = cosm.masked_fill(~valid_pairs, float('-inf'))
            flat = cosm.view(B, -1); idx = flat.argmax(dim=1)
            si, ti = idx // Kt, idx % Kt
            return src_embs[torch.arange(B), si], tgt_embs[torch.arange(B), ti]
        elif pooling is GroupPoolingStrategy.concat:
            # zero out padded slots before flattening
            src_masked = src_embs * src_valid.unsqueeze(-1).float()
            tgt_masked = tgt_embs * tgt_valid.unsqueeze(-1).float()
            return proj(src_masked.view(B,-1)), proj(tgt_masked.view(B,-1))
        else:
            raise ValueError(f"Unknown group pooling: {pooling}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids/attention_mask: (B, 4, K_max, L)
        # groups: 0=src_labels,1=src_ctx,2=tgt_labels,3=tgt_ctx
        B, G, K, L = input_ids.size()
        assert G == 4, f"Expected group dim=4, got {G}"
        assert K >= self.n_labels and K >= self.n_contexts, (
            f"Group slot dim K={K} is smaller than max_labels={self.n_labels} "
            f"or max_contexts={self.n_contexts}"
        )

        # slice out token masks
        src_lbl_mask = attention_mask[:,0,:self.n_labels,:]  # (B,K_lbl,L)
        src_ctx_mask = attention_mask[:,1,:self.n_contexts,:]
        tgt_lbl_mask = attention_mask[:,2,:self.n_labels,:]
        tgt_ctx_mask = attention_mask[:,3,:self.n_contexts,:]
        # derive slot-validity from token masks
        src_lbl_valid = (src_lbl_mask.sum(dim=-1) > 0)
        src_ctx_valid = (src_ctx_mask.sum(dim=-1) > 0)
        tgt_lbl_valid = (tgt_lbl_mask.sum(dim=-1) > 0)
        tgt_ctx_valid = (tgt_ctx_mask.sum(dim=-1) > 0)
        # slice ids accordingly
        src_lbl_ids = input_ids[:,0,:self.n_labels,:]
        src_ctx_ids = input_ids[:,1,:self.n_contexts,:]
        tgt_lbl_ids = input_ids[:,2,:self.n_labels,:]
        tgt_ctx_ids = input_ids[:,3,:self.n_contexts,:]
        # encode
        e_lbl_src = self._encode_group(src_lbl_ids, src_lbl_mask)
        e_ctx_src = self._encode_group(src_ctx_ids, src_ctx_mask)
        e_lbl_tgt = self._encode_group(tgt_lbl_ids, tgt_lbl_mask)
        e_ctx_tgt = self._encode_group(tgt_ctx_ids, tgt_ctx_mask)
        # pool
        e_lbl_src, e_lbl_tgt = self._pool_group(
            e_lbl_src, e_lbl_tgt, src_lbl_valid, tgt_lbl_valid,
            pooling=self.label_pooling, proj=getattr(self, 'label_proj', None)
        )
        e_ctx_src, e_ctx_tgt = self._pool_group(
            e_ctx_src, e_ctx_tgt, src_ctx_valid, tgt_ctx_valid,
            pooling=self.ctx_pooling, proj=getattr(self, 'ctx_proj', None)
        )
        # fusion logic follows original patterns
        if self.fusion_strategy is FusionStrategy.static:
            src = self.src_weight * e_lbl_src + (1 - self.src_weight) * e_ctx_src
            tgt = self.tgt_weight * e_lbl_tgt + (1 - self.tgt_weight) * e_ctx_tgt
        elif self.fusion_strategy is FusionStrategy.gated:
            g_s = self.gate_src(torch.cat([e_lbl_src, e_ctx_src], dim=-1))
            g_t = self.gate_tgt(torch.cat([e_lbl_tgt, e_ctx_tgt], dim=-1))
            src = g_s * e_lbl_src + (1 - g_s) * e_ctx_src
            tgt = g_t * e_lbl_tgt + (1 - g_t) * e_ctx_tgt
        elif self.fusion_strategy is FusionStrategy.mlp:
            return self.mlp(torch.cat([e_lbl_src, e_ctx_src, e_lbl_tgt, e_ctx_tgt], dim=-1)).squeeze(-1)
        elif self.fusion_strategy is FusionStrategy.multitask:
            lbl_diff = e_lbl_src - e_lbl_tgt
            ctx_diff = e_ctx_src - e_ctx_tgt
            return self.alpha * self.head_label(lbl_diff).squeeze(-1) + \
                   (1 - self.alpha) * self.head_ctx(ctx_diff).squeeze(-1)
        elif self.fusion_strategy is FusionStrategy.weighted_cosine:
            return self.cos_alpha * F.cosine_similarity(e_lbl_src, e_lbl_tgt, dim=-1) + \
                   (1 - self.cos_alpha) * F.cosine_similarity(e_ctx_src, e_ctx_tgt, dim=-1)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # final classifier or similarity
        src = self.fusion_dropout(src); tgt = self.fusion_dropout(tgt)
        if self.use_classifier and hasattr(self, 'classifier'):
            return self.classifier(self.classifier_dropout(torch.cat([src, tgt], dim=-1))).squeeze(-1)
        return F.cosine_similarity(src, tgt, dim=-1)
