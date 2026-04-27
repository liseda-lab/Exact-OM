import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Iterable, Callable
import re
import time
import json
import hashlib
from pathlib import Path
from collections import OrderedDict
from urllib import error as urlerror

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

from exact.core.contracts.model import IModel
from exact.utils.llm_routing import (
    LLMRouter,
    extract_chat_text,
    extract_first_token_top_logprobs,
    parse_structured_json,
)


class PoolingMethod(str, Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"


class LabelPairPooling(str, Enum):
    MAX = "max"
    MEAN = "mean"


class SemanticScorer(IModel):
    """
    Explainable ontology matching scorer integrating:
      • Lexical embeddings with configurable pooling over label variants
      • Context embeddings over verbalised subgraphs (lists of sentences)
      • Uncertainty-driven, gated LLM (summarise + binary decision probability)
      • Adaptive fusion (w_c) and bounded ambiguity U for LLM weight w_i
      • Optional explanation generation incl. model provenance and triple attributions
    """

    def __init__(
        self,
        # ---- Models ----
        lexical_model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        context_model_name: str = "BAAI/bge-large-en-v1.5",
        llm_model_name: Optional[str] = "Qwen/Qwen2.5-7B-Instruct",

        # ---- Precision / device ----
        fp16_inference: bool = True,
        device: Optional[str] = None,

        # ---- Pooling for encoders ----
        pooling_method: PoolingMethod = PoolingMethod.MEAN,
        label_pair_pooling: LabelPairPooling = LabelPairPooling.MAX,

        # ---- Context sentence delimiter ----
        ctx_sentence_delimiter: Optional[str] = " || ",

        # ---- Token limits ----
        max_input_tokens_lexical: int = 32,
        max_input_tokens_context: int = 256,
        max_total_tokens_llm_summary: int = 512,
        max_total_tokens_llm_decision: int = 384,
        max_total_tokens_llm_rationale: int = 512,
        max_new_tokens_llm: int = 64,
        max_new_tokens_llm_rationale: Optional[int] = None,

        # ---- Ablations / toggles ----
        use_lexical: bool = True,
        use_context: bool = True,
        use_llm: bool = True,

        # ---- Adaptive weighting thresholds ----
        tau: float = 0.5,

        # ---- Adaptive context weighting ----
        gamma: float = 2.0,

        # ---- LLM gating and mixing ----
        beta: float = 0.8,
        tau_LLM: float = 0.35,

        # ---- LLM generation knobs ----
        llm_temperature: float = 0.1,
        llm_top_p: float = 0.9,
        llm_do_sample: bool = False,
        llm_summary_batch_size: Optional[int] = 8,
        llm_decision_batch_size: Optional[int] = 8,
        llm_rationale_batch_size: Optional[int] = 8,
        hosted_decision_labels: Optional[List[str]] = None,
        hosted_decision_logit_bias: float = 20.0,
        force_llm_summaries: bool = False,

        # ---- Explanations / review band ----
        return_explanations: bool = False,
        generate_llm_rationales: bool = False,
        use_llm_calibration: bool = False,
        llm_calibration_a: Optional[float] = None,
        llm_calibration_b: Optional[float] = None,
        llm_calibration_info: Optional[str] = None,
        review_low: float = 0.35,
        review_high: float = 0.75,
        threshold: float = 0.7,

        # ---- Caching ----
        cache_dir: Optional[str] = None,
        cache_namespace: Optional[str] = None,
        dataset_signature: Optional[str] = None,
        persist_cache_to_disk: bool = True,
        max_cached_labels: Optional[int] = None,
        max_cached_contexts: Optional[int] = None,
        max_cached_summaries: Optional[int] = None,
        max_cached_rationales: Optional[int] = None,
        llm_profiles: Optional[Dict[str, Any]] = None,
        llm_routing: Optional[Dict[str, Any]] = None,
        request_seed: Optional[int] = None,

        **kwargs,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = (
            self.device.type if isinstance(self.device, torch.device) else torch.device(self.device).type
        )
        self.fp16 = fp16_inference
        self.pooling_method = PoolingMethod(pooling_method)
        self.label_pair_pooling = LabelPairPooling(label_pair_pooling)

        # attach logger hook
        self.log = getattr(self, "log", lambda *a, **kw: None)

        # store params
        self.max_input_tokens_lexical = max_input_tokens_lexical
        self.max_input_tokens_context = max_input_tokens_context
        self.max_total_tokens_llm_summary = max_total_tokens_llm_summary
        self.max_total_tokens_llm_decision = max_total_tokens_llm_decision
        self.max_total_tokens_llm_rationale = max_total_tokens_llm_rationale
        self.max_new_tokens_llm = max_new_tokens_llm
        self.max_new_tokens_llm_rationale = (
            int(max_new_tokens_llm_rationale)
            if max_new_tokens_llm_rationale is not None
            else int(max_new_tokens_llm)
        )
        self.llm_summary_batch_size = llm_summary_batch_size
        self.llm_decision_batch_size = llm_decision_batch_size
        self.llm_rationale_batch_size = llm_rationale_batch_size
        raw_hosted_labels = list(hosted_decision_labels or ["A", "B"])
        if len(raw_hosted_labels) != 2 or any(not str(label).strip() for label in raw_hosted_labels):
            raise ValueError("hosted_decision_labels must contain exactly two non-empty labels.")
        self.hosted_decision_labels = (str(raw_hosted_labels[0]).strip(), str(raw_hosted_labels[1]).strip())
        if self.hosted_decision_labels[0] == self.hosted_decision_labels[1]:
            raise ValueError("hosted_decision_labels must contain two distinct labels.")
        self.hosted_decision_logit_bias = float(hosted_decision_logit_bias)

        self.use_lexical = use_lexical
        self.use_context = use_context
        self.use_llm = use_llm and (llm_model_name is not None)
        self.force_llm_summaries = bool(force_llm_summaries) and self.use_llm
        self.generate_llm_rationales = generate_llm_rationales and self.use_llm
        self.llm_calibration_a = llm_calibration_a
        self.llm_calibration_b = llm_calibration_b
        self.llm_calibration_info = (llm_calibration_info or "").strip() or None
        self.use_llm_calibration = bool(use_llm_calibration) and self.use_llm
        self._llm_calibration_can_apply = (
            self.use_llm_calibration
            and (self.llm_calibration_a is not None)
            and (self.llm_calibration_b is not None)
        )

        self.tau = tau
        self.gamma = gamma
        self.beta = beta
        self.tau_LLM = tau_LLM

        self.return_explanations = return_explanations
        self.review_low = review_low
        self.review_high = review_high
        self.threshold = threshold
        self.ctx_sentence_delimiter = ctx_sentence_delimiter
        self.dataset_signature = dataset_signature
        self.cache_namespace = (cache_namespace or "default").strip() or "default"
        self.persist_cache_to_disk = bool(persist_cache_to_disk)
        default_cache_dir = Path(cache_dir).expanduser() if cache_dir else (Path.home() / ".cache" / "exact" / "semantic_scorer")
        self.cache_dir = default_cache_dir
        self._cache_tensor_dtype = torch.float16 if self.fp16 else torch.float32
        self.max_cached_labels = max_cached_labels
        self.max_cached_contexts = max_cached_contexts
        self.max_cached_summaries = max_cached_summaries
        self.max_cached_rationales = max_cached_rationales
        self.request_seed = int(request_seed) if request_seed is not None else None
        self._lex_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._ctx_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._summary_cache: "OrderedDict[str, str]" = OrderedDict()
        self._rationale_cache: "OrderedDict[str, str]" = OrderedDict()
        self._cache_dirty = False
        self._cache_file_path = (
            (self.cache_dir / f"{self.cache_namespace}_cache.pt").resolve()
            if self.persist_cache_to_disk else None
        )
        self._log_once_keys = set()
        self._computed_llm_calibration: Optional[Dict[str, Any]] = None
        self._calibration_messages: List[str] = []
        self._calibration_pending_probs: List[float] = []
        self._calibration_pending_labels: List[float] = []
        self._llm_calibration_fitted_once = False
        self.reset_summary_stats()
        self.reset_decision_stats()

        # store model names (for explanations)
        self.lexical_model_name = lexical_model_name
        self.context_model_name = context_model_name
        self.llm_model_name = llm_model_name
        self._llm_router = LLMRouter(llm_profiles=llm_profiles, llm_routing=llm_routing, log=self.log)
        self._local_llm_profile_name = "__semantic_local_llm__"
        self._llm_router.ensure_profile(
            self._local_llm_profile_name,
            {"backend": "local_hf", "model": self.llm_model_name},
        )
        if self._llm_router.routing.default_profile is None:
            self._llm_router.routing.default_profile = self._local_llm_profile_name
        if self._llm_router.routing.fallback_profile is None:
            self._llm_router.routing.fallback_profile = self._local_llm_profile_name
        if self._llm_router.routing.decision_fallback_profile is None:
            self._llm_router.routing.decision_fallback_profile = self._local_llm_profile_name
        self.llm_tok = None
        self.llm = None
        self.yes_token_ids: List[int] = []
        self.no_token_ids: List[int] = []
        self._last_summary_backend_meta: Dict[str, Any] = {}
        self._last_decision_backend_meta: Dict[str, Any] = {}
        self._last_rationale_backend_meta: Dict[str, Any] = {}
        self._hosted_decision_tokenizers: Dict[str, Any] = {}
        self._cache_fingerprint = self._build_cache_fingerprint()

        # ---- Load models ----
        if self.use_lexical:
            self.log("Loading lexical encoder...", "info")
            self.lex_tok = AutoTokenizer.from_pretrained(lexical_model_name)
            self.lex_model = AutoModel.from_pretrained(lexical_model_name).to(self.device)

        if self.use_context:
            self.log("Loading context encoder...", "info")
            self.ctx_tok = AutoTokenizer.from_pretrained(context_model_name)
            self.ctx_model = AutoModel.from_pretrained(context_model_name).to(self.device)

        if self.use_llm:
            self.llm_temperature = llm_temperature
            self.llm_top_p = llm_top_p
            self.llm_do_sample = llm_do_sample

        if self.persist_cache_to_disk:
            self._load_cache_from_disk()
        self.reset_llm_calibration_tracking()

    def _ensure_local_llm(self) -> None:
        if not self.use_llm:
            return
        if self.llm is not None and self.llm_tok is not None:
            return
        if not self.llm_model_name:
            raise ValueError("Local LLM fallback requested but llm_model_name is not configured.")
        self.log("Loading local LLM fallback...", "info")
        self.llm_tok = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float16 if self.fp16 else torch.float32
        ).to(self.device)
        self.yes_token_ids = self._candidate_token_ids([" Yes", "Yes", "yes"])
        self.no_token_ids = self._candidate_token_ids([" No", "No", "no"])

    def _log_once(self, key: str, message: str, level: str = "info") -> None:
        """
        Emit the log message only the first time it is requested.
        Useful for forward() logs that would otherwise repeat every batch.
        """
        if key in self._log_once_keys:
            return
        self._log_once_keys.add(key)
        self.log(message, level)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _pool(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_method == PoolingMethod.MEAN:
            masked = seq * mask.unsqueeze(-1)
            return masked.sum(1) / (mask.sum(1, keepdim=True) + 1e-9)
        if self.pooling_method == PoolingMethod.SUM:
            return (seq * mask.unsqueeze(-1)).sum(1)
        if self.pooling_method == PoolingMethod.MAX:
            masked = seq.masked_fill(~mask.bool().unsqueeze(-1), -1e9)
            return masked.max(1).values
        raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    def _cos_sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(a, b)

    def _select_label_pair(
        self,
        mat: torch.Tensor,
        labs_s: List[str],
        labs_t: List[str],
    ) -> Tuple[torch.Tensor, Tuple[str, str]]:
        if mat.numel() == 0:
            return torch.tensor(0.0, device=mat.device), ("", "")
        if self.label_pair_pooling == LabelPairPooling.MAX:
            score = mat.max()
            idx_flat = torch.argmax(mat)
        elif self.label_pair_pooling == LabelPairPooling.MEAN:
            score = mat.mean()
            idx_flat = torch.argmin((mat - score).abs())
        else:
            raise ValueError(f"Unknown label pair pooling: {self.label_pair_pooling}")
        r = int(idx_flat // mat.shape[1])
        c = int(idx_flat % mat.shape[1])
        pair = (
            labs_s[r] if 0 <= r < len(labs_s) else "",
            labs_t[c] if 0 <= c < len(labs_t) else "",
        )
        return score, pair

    def _candidate_token_ids(self, variants: List[str]) -> List[int]:
        self._ensure_local_llm()
        ids = []
        for v in variants:
            tok = self.llm_tok(v, add_special_tokens=False).input_ids
            if len(tok) >= 1:
                ids.append(tok[0])
        return sorted(list(set(ids)))

    def _join_context(self, triples: List[str]) -> str:
        if not triples:
            return ""
        return self.ctx_sentence_delimiter.join(triples) if self.ctx_sentence_delimiter else " ".join(triples)

    # -------------------------------------------------------------------------
    # Cache helpers
    # -------------------------------------------------------------------------
    def _runtime_fingerprint_payload(
        self,
        generate_llm_rationales_override: Optional[bool] = None,
    ) -> Dict[str, Any]:
        generate_llm_rationales = (
            self.generate_llm_rationales
            if generate_llm_rationales_override is None
            else bool(generate_llm_rationales_override)
        )
        return {
            "lexical_model": self.lexical_model_name if self.use_lexical else None,
            "context_model": self.context_model_name if self.use_context else None,
            "llm_model": self.llm_model_name if self.use_llm else None,
            "llm_router": self._llm_router.fingerprint_payload() if self.use_llm else None,
            "max_input_tokens_lexical": self.max_input_tokens_lexical,
            "max_input_tokens_context": self.max_input_tokens_context,
            "max_total_tokens_llm_summary": self.max_total_tokens_llm_summary,
            "max_total_tokens_llm_decision": self.max_total_tokens_llm_decision,
            "max_total_tokens_llm_rationale": self.max_total_tokens_llm_rationale,
            "max_new_tokens_llm": self.max_new_tokens_llm,
            "max_new_tokens_llm_rationale": self.max_new_tokens_llm_rationale,
            "hosted_decision_labels": list(self.hosted_decision_labels),
            "hosted_decision_logit_bias": self.hosted_decision_logit_bias,
            "request_seed": self.request_seed,
            "pooling_method": self.pooling_method.value,
            "label_pair_pooling": self.label_pair_pooling.value,
            "ctx_sentence_delimiter": self.ctx_sentence_delimiter,
            "dataset_signature": self.dataset_signature,
            "generate_llm_rationales": generate_llm_rationales,
            "use_llm_calibration": self.use_llm_calibration,
            "llm_calibration_a": self.llm_calibration_a,
            "llm_calibration_b": self.llm_calibration_b,
            "llm_calibration_info": self.llm_calibration_info,
            "force_llm_summaries": self.force_llm_summaries,
        }

    def _build_cache_fingerprint(
        self,
        generate_llm_rationales_override: Optional[bool] = None,
    ) -> str:
        payload = self._runtime_fingerprint_payload(
            generate_llm_rationales_override=generate_llm_rationales_override
        )
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def runtime_fingerprint(self) -> str:
        return self._cache_fingerprint

    def runtime_fingerprint_payload(
        self,
        generate_llm_rationales_override: Optional[bool] = None,
    ) -> Dict[str, Any]:
        return self._runtime_fingerprint_payload(
            generate_llm_rationales_override=generate_llm_rationales_override
        )

    def _model_hidden_size(self, model: nn.Module) -> int:
        cfg = getattr(model, "config", None)
        if cfg is None:
            raise ValueError("Encoder lacks a config; cannot infer hidden size for caching.")
        for attr in ("hidden_size", "projection_dim", "text_embed_dim", "word_embed_proj_dim"):
            if hasattr(cfg, attr):
                val = getattr(cfg, attr)
                if isinstance(val, (list, tuple)):
                    if val:
                        return int(val[-1])
                else:
                    return int(val)
        if hasattr(cfg, "hidden_sizes") and getattr(cfg, "hidden_sizes"):
            return int(cfg.hidden_sizes[-1])
        raise ValueError("Unable to infer encoder hidden size for caching.")

    def _cache_store(
        self,
        cache: "OrderedDict[str, Any]",
        key: str,
        value: Any,
        limit: Optional[int],
    ) -> None:
        cache[key] = value
        cache.move_to_end(key)
        if limit and limit > 0 and len(cache) > limit:
            cache.popitem(last=False)
        self._cache_dirty = True

    def _encode_with_cache(
        self,
        texts: List[str],
        tokenizer,
        model,
        max_len: int,
        cache: "OrderedDict[str, torch.Tensor]",
        limit: Optional[int],
    ) -> torch.Tensor:
        if not texts:
            hidden = self._model_hidden_size(model)
            return torch.zeros((0, hidden), device=self.device)

        outputs: List[Optional[torch.Tensor]] = [None] * len(texts)
        missing_keys: List[str] = []
        key_to_indices: Dict[str, List[int]] = {}

        for idx, text in enumerate(texts):
            key = text or ""
            cached = cache.get(key)
            if cached is None:
                key_to_indices.setdefault(key, []).append(idx)
                if key not in missing_keys:
                    missing_keys.append(key)
            else:
                outputs[idx] = cached

        if missing_keys:
            encodings = self._encode_texts(tokenizer, model, missing_keys, max_len)
            encodings = encodings.detach().to("cpu").to(self._cache_tensor_dtype)
            for key, tensor in zip(missing_keys, encodings):
                self._cache_store(cache, key, tensor, limit)
                for idx in key_to_indices[key]:
                    outputs[idx] = tensor

        stacked = torch.stack(outputs, dim=0).to(self.device)
        return stacked

    @staticmethod
    def _summary_key(label: str, context: str) -> str:
        payload = (label or "") + "\u241F" + (context or "")
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _rationale_key(
        src_label: str,
        tgt_label: str,
        src_summary: str,
        tgt_summary: str,
        decision: str,
        decision_context: str = "",
    ) -> str:
        sep = "\u241F"
        parts = [
            src_label or "",
            tgt_label or "",
            src_summary or "",
            tgt_summary or "",
            decision or "",
        ]
        if decision_context:
            parts.append(decision_context)
        payload = sep.join(parts)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _serialize_cache_payload(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "version": 1,
                "fingerprint": self._cache_fingerprint,
                "dataset_signature": self.dataset_signature,
                "namespace": self.cache_namespace,
                "timestamp": time.time(),
            },
            "lexical": [(k, v.cpu().to(self._cache_tensor_dtype)) for k, v in self._lex_cache.items()],
            "context": [(k, v.cpu().to(self._cache_tensor_dtype)) for k, v in self._ctx_cache.items()],
            "summaries": list(self._summary_cache.items()),
            "rationales": list(self._rationale_cache.items()),
        }

    def _load_cache_from_disk(self) -> None:
        if not self.persist_cache_to_disk or not self._cache_file_path:
            return
        if not self._cache_file_path.exists():
            return
        try:
            payload = torch.load(self._cache_file_path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Failed to load SemanticScorer cache at {self._cache_file_path}: {exc}", "warning")
            return

        meta = (payload or {}).get("metadata") or {}
        if meta.get("fingerprint") != self._cache_fingerprint:
            self.log(
                (
                    "Cached embeddings/summaries are invalid for the current configuration. "
                    "Discarding on-disk cache."
                ),
                "warning",
            )
            return

        self._lex_cache.clear()
        for key, tensor in payload.get("lexical", []):
            self._lex_cache[key] = tensor.to(dtype=self._cache_tensor_dtype)
        self._ctx_cache.clear()
        for key, tensor in payload.get("context", []):
            self._ctx_cache[key] = tensor.to(dtype=self._cache_tensor_dtype)
        self._summary_cache.clear()
        for key, text in payload.get("summaries", []):
            self._summary_cache[key] = text
        self._rationale_cache.clear()
        for key, text in payload.get("rationales", []):
            self._rationale_cache[key] = text

        counts = (
            len(self._lex_cache),
            len(self._ctx_cache),
            len(self._summary_cache),
            len(self._rationale_cache),
        )
        self._cache_dirty = False
        model_name = self.__class__.__name__
        self.log(
            (
                f"Loaded {model_name} cache from {self._cache_file_path} "
                f"(lexical={counts[0]}, context={counts[1]}, summaries={counts[2]}, rationales={counts[3]})."
            ),
            "info",
        )

    def persist_caches(self, force: bool = False, reason: str = "manual") -> None:
        if not self.persist_cache_to_disk or not self._cache_file_path:
            return
        if not (force or self._cache_dirty):
            return
        payload = self._serialize_cache_payload()
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, self._cache_file_path)
            model_name = self.__class__.__name__
            self.log(
                (
                    f"Persisted {model_name} cache to {self._cache_file_path} "
                    f"(reason={reason})."
                ),
                "debug",
            )
            self._cache_dirty = False
        except OSError as exc:
            model_name = self.__class__.__name__
            self.log(f"Failed to persist {model_name} cache to {self._cache_file_path}: {exc}", "warning")

    # -------------------------------------------------------------------------
    # Encoders
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def _encode_texts(self, tokenizer, model, texts: List[str], max_len: int) -> torch.Tensor:
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        ).to(self.device)
        with torch.amp.autocast(device_type=self.device_type, enabled=self.fp16):
            out = model(**enc).last_hidden_state
        return self._pool(out, enc["attention_mask"])

    @torch.inference_mode()
    def encode_labels_batch(self, label_texts: List[str]) -> torch.Tensor:
        if not self.use_lexical:
            raise RuntimeError("Lexical encoder is disabled.")
        return self._encode_with_cache(
            label_texts,
            self.lex_tok,
            self.lex_model,
            self.max_input_tokens_lexical,
            self._lex_cache,
            self.max_cached_labels,
        )

    @torch.inference_mode()
    def encode_contexts_batch(self, ctx_texts: List[str]) -> torch.Tensor:
        if not self.use_context:
            raise RuntimeError("Context encoder is disabled.")
        return self._encode_with_cache(
            ctx_texts,
            self.ctx_tok,
            self.ctx_model,
            self.max_input_tokens_context,
            self._ctx_cache,
            self.max_cached_contexts,
        )
    
    # -------------------------------------------------------------------------
    # Generative Models
    # -------------------------------------------------------------------------
    def _summary_prompt(self, label: str, ctx: str) -> Dict[str, str]:
        return {
            "system": "You are an ontology expert entity summariser that returns strict JSON.",
            "user": (
                f"Given the following context subgraph describing the entity '{label}', "
                "provide one concise and informative summary capturing its key characteristics.\n\n"
                f"Context: {ctx}\n\n"
                "Return exactly one JSON object with one key: \"summary\"."
            ),
        }

    def _decision_prompt(self, src_label: str, tgt_label: str, src_summary: str, tgt_summary: str) -> Dict[str, str]:
        return {
            "system": "You are an ontology alignment expert.",
            "user": (
                "Determine whether the following two ontology entities refer to the same concept.\n"
                "Answer with a single token: Yes or No.\n\n"
                f"Source entity: {src_label}\nSummary: {src_summary}\n\n"
                f"Target entity: {tgt_label}\nSummary: {tgt_summary}\n\n"
                "Answer:"
            ),
        }

    def _rationale_prompt(
        self,
        src_label: str,
        tgt_label: str,
        src_summary: str,
        tgt_summary: str,
        decision: str,
        decision_context: str = "",
    ) -> Dict[str, str]:
        context_block = ""
        if decision_context:
            context_block = (
                "\nFinal alignment context\n"
                f"{decision_context}\n"
            )
        return {
            "system": "You are an ontology alignment expert.",
            "user": (
                "Based only on the information below, write two to four sentences explaining "
                "why the two entities should or should not be considered equivalent.\n"
                "Reference specific evidence from the label pair and the summaries. "
                "When final alignment context is present, use it to explain whether the pair "
                "was kept or rejected after candidate-set selection and cardinality filtering. "
                "Do not introduce external knowledge. Return exactly one JSON object with one key: "
                "\"rationale\".\n\n"
                f"Source\nLabel: {src_label}\nSummary: {src_summary}\n\n"
                f"Target\nLabel: {tgt_label}\nSummary: {tgt_summary}\n\n"
                f"Final decision: {decision}\n\n"
                f"{context_block}"
                "Return only JSON."
            ),
        }

    def _render_llm_prompt(self, prompt: Dict[str, str], add_generation_prompt: bool = True) -> str:
        self._ensure_local_llm()
        system = prompt.get("system", "").strip()
        user = prompt.get("user", "").strip()
        tok = self.llm_tok
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if user:
                messages.append({"role": "user", "content": user})
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        segments = [seg for seg in [system, user] if seg]
        return "\n\n".join(segments)

    def _strip_llm_prompt_tokens(self, enc: Dict[str, torch.Tensor], gen: torch.Tensor) -> List[torch.Tensor]:
        prompt_lens = enc["attention_mask"].sum(dim=1)
        outputs: List[torch.Tensor] = []
        for row, plen in zip(gen, prompt_lens):
            plen = int(plen.item())
            if plen < row.shape[0]:
                outputs.append(row[plen:])
            else:
                outputs.append(row[-1:].clone())
        return outputs

    def reset_summary_stats(self) -> None:
        self._summary_stats = {"requested": 0, "usable": 0, "empty": 0}

    def reset_decision_stats(self) -> None:
        self._decision_stats = {
            "requested": 0,
            "hosted_attempted": 0,
            "hosted_scored": 0,
            "probe_failures": 0,
            "scoring_failures": 0,
            "local_fallbacks": 0,
        }
        self._decision_probe_cache: Dict[str, Dict[str, Any]] = {}

    def _record_summary_stats(self, summaries: Iterable[Optional[str]]) -> None:
        if not self.use_llm:
            return
        stats = getattr(self, "_summary_stats", None)
        if stats is None:
            self.reset_summary_stats()
            stats = self._summary_stats
        for text in summaries:
            stats["requested"] += 1
            normalized = ""
            if isinstance(text, str):
                normalized = text.strip()
            elif text is None:
                normalized = ""
            else:
                normalized = str(text).strip()
            if normalized:
                stats["usable"] += 1
            else:
                stats["empty"] += 1

    def llm_summary_stats(self) -> Dict[str, Any]:
        stats = dict(getattr(self, "_summary_stats", {}))
        total = int(stats.get("requested", 0) or 0)
        empty = int(stats.get("empty", 0) or 0)
        usable = int(stats.get("usable", 0) or 0)
        if usable + empty != total:
            usable = max(0, total - empty)
        stats["requested"] = total
        stats["usable"] = usable
        stats["empty"] = empty
        stats["empty_fraction"] = (float(empty) / total) if total else 0.0
        stats["usable_fraction"] = (float(usable) / total) if total else 0.0
        return stats

    def _record_decision_stat(self, key: str, amount: int = 1) -> None:
        stats = getattr(self, "_decision_stats", None)
        if stats is None:
            self.reset_decision_stats()
            stats = self._decision_stats
        stats[key] = int(stats.get(key, 0) or 0) + int(amount)

    def llm_decision_stats(self) -> Dict[str, Any]:
        stats = dict(getattr(self, "_decision_stats", {}))
        for key in [
            "requested",
            "hosted_attempted",
            "hosted_scored",
            "probe_failures",
            "scoring_failures",
            "local_fallbacks",
        ]:
            stats[key] = int(stats.get(key, 0) or 0)
        requested = stats["requested"]
        hosted_attempted = stats["hosted_attempted"]
        hosted_scored = stats["hosted_scored"]
        stats["hosted_success_fraction"] = (
            float(hosted_scored) / hosted_attempted if hosted_attempted else 0.0
        )
        stats["fallback_fraction"] = (
            float(stats["local_fallbacks"]) / requested if requested else 0.0
        )
        return stats

    @staticmethod
    def _clean_summary_text(text: Optional[str]) -> str:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        txt = text.strip()
        txt = re.sub(r"^\s*```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s*```\s*$", "", txt).strip()
        if not txt:
            return ""
        if "Summary:" in txt:
            txt = txt.split("Summary:", 1)[-1].strip()
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if not lines:
            return ""
        if lines[0].lower() in {"json", "```json", "```"}:
            lines = lines[1:]
        if not lines:
            return ""
        line = " ".join(lines)
        prefix_pattern = re.compile(r"^(assistant|assistant:|assistant,|summary:)", re.IGNORECASE)
        line = prefix_pattern.sub("", line).strip(" :,-\t")
        if not line:
            return ""
        sentence_match = re.search(r"^(.+?[.!?](?:['\")\]]|$))", line)
        if sentence_match:
            return sentence_match.group(1).strip()
        return line

    @staticmethod
    def _clean_rationale_text(text: str) -> str:
        txt = text.strip()
        txt = re.sub(r"^\s*```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s*```\s*$", "", txt).strip()
        if not txt:
            return ""
        prefix_pattern = re.compile(r"^(assistant|rationale:|assistant:)", re.IGNORECASE)
        txt = prefix_pattern.sub("", txt).strip()
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if lines and lines[0].lower() in {"json", "```json", "```"}:
            lines = lines[1:]
        if lines:
            txt = " ".join(lines)
        if not txt:
            return ""
        sentence_pattern = re.compile(r".+?[.!?](?=(?:['\")\]]|\s|$))")
        complete_sentences = [match.group(0).strip() for match in sentence_pattern.finditer(txt)]
        if complete_sentences:
            return " ".join(complete_sentences[:4]).strip()
        return txt

    @staticmethod
    def _parse_structured_text(text: str, key: str, cleaner) -> str:
        try:
            return cleaner(parse_structured_json(text, key))
        except Exception:
            return cleaner(text)

    @staticmethod
    def _safe_logsumexp(log_values: List[float]) -> Optional[float]:
        if not log_values:
            return None
        vals = torch.tensor(log_values, dtype=torch.float64)
        return float(torch.logsumexp(vals, dim=0).item())

    def _resolved_backend_metadata(self, resolved) -> Dict[str, Any]:
        return {
            "backend": resolved.backend,
            "profile": resolved.profile_name,
            "model": resolved.model,
            "decision_capable": bool(resolved.decision_capable),
            "fallback_triggered": bool(resolved.fallback_triggered),
            "fallback_reason": resolved.fallback_reason,
            "endpoint": None,
            "provider": None,
            "decision_probe_passed": None,
            "decision_probe_error": None,
            "decision_scoring_mode": None,
        }

    @staticmethod
    def _truncate_debug_text(text: Any, limit: int = 800) -> str:
        value = str(text or "")
        if len(value) <= limit:
            return value
        return value[:limit] + "...<truncated>"

    @staticmethod
    def _text_debug_stats(text: Any) -> Dict[str, int]:
        value = str(text or "")
        return {
            "length": len(value),
            "non_ascii": sum(1 for ch in value if ord(ch) > 127),
            "control": sum(1 for ch in value if ((ord(ch) < 32 and ch not in "\n\r\t") or ord(ch) == 127)),
            "nul": value.count("\x00"),
            "surrogates": sum(1 for ch in value if 0xD800 <= ord(ch) <= 0xDFFF),
        }

    @staticmethod
    def _format_first_token_logprobs(pairs: List[Tuple[str, float]], limit: int = 10) -> str:
        if not pairs:
            return "<empty>"
        formatted = [f"{token!r}:{float(lp):.4f}" for token, lp in pairs[:limit]]
        suffix = "" if len(pairs) <= limit else ", ..."
        return ", ".join(formatted) + suffix

    @staticmethod
    def _logsumexp(values: List[float]) -> float:
        return float(torch.logsumexp(torch.tensor(values, dtype=torch.float64), dim=0).item())

    @staticmethod
    def _candidate_token_ids_for_tokenizer(tokenizer, variants: List[str]) -> List[int]:
        ids: List[int] = []
        for variant in variants:
            token_ids = tokenizer(variant, add_special_tokens=False).input_ids
            if len(token_ids) == 1:
                ids.append(int(token_ids[0]))
        return sorted(set(ids))

    def _run_hosted_chat_prompts(
        self,
        prompts: List[Dict[str, str]],
        profile,
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        concurrency: Optional[int],
    ) -> List[str]:
        if not prompts:
            return []
        workers = concurrency or len(prompts)
        workers = max(1, min(int(workers), len(prompts)))

        def _call(prompt: Dict[str, str]) -> str:
            payload = self._llm_router.hosted.chat_completion(
                profile=profile,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=self.request_seed,
            )
            return extract_chat_text(payload)

        if workers == 1:
            return [_call(prompt) for prompt in prompts]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_call, prompts))

    def _get_hosted_decision_tokenizer(self, profile):
        tokenizer_name = getattr(profile, "tokenizer", None)
        if not tokenizer_name:
            raise ValueError(
                f"OpenRouter decision profile '{profile.name}' must define a tokenizer for hosted decision biasing."
            )
        cached = self._hosted_decision_tokenizers.get(tokenizer_name)
        if cached is not None:
            return cached
        self._log_once(
            f"hosted_decision_tokenizer_load:{tokenizer_name}",
            f"Loading hosted decision tokenizer '{tokenizer_name}' for profile '{profile.name}'.",
            "debug",
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._hosted_decision_tokenizers[tokenizer_name] = tokenizer
        return tokenizer

    def _hosted_decision_label_ids(self, profile) -> Dict[str, List[int]]:
        tokenizer = self._get_hosted_decision_tokenizer(profile)
        label_ids: Dict[str, List[int]] = {}
        for label in self.hosted_decision_labels:
            ids = self._candidate_token_ids_for_tokenizer(tokenizer, [label, f" {label}"])
            if not ids:
                raise ValueError(
                    f"Hosted decision label '{label}' is not a single-token option for tokenizer '{profile.tokenizer}'."
                )
            label_ids[label] = ids
        return label_ids

    def _hosted_decision_logit_bias(self, profile) -> Tuple[Dict[str, float], Dict[str, List[int]]]:
        label_ids = self._hosted_decision_label_ids(profile)
        logit_bias: Dict[str, float] = {}
        for ids in label_ids.values():
            for token_id in ids:
                logit_bias[str(token_id)] = self.hosted_decision_logit_bias
        return logit_bias, label_ids

    def _hosted_decision_prompt(
        self,
        src_label: str,
        tgt_label: str,
        src_summary: str,
        tgt_summary: str,
    ) -> Dict[str, str]:
        positive_label, negative_label = self.hosted_decision_labels
        return {
            "system": "You are a binary classifier for ontology alignment.",
            "user": (
                f"Return exactly one token: {positive_label} or {negative_label}.\n"
                f"{positive_label} = the source and target entities are equivalent.\n"
                f"{negative_label} = the source and target entities are not equivalent.\n\n"
                f"Source entity: {src_label}\nSummary: {src_summary}\n\n"
                f"Target entity: {tgt_label}\nSummary: {tgt_summary}"
            ),
        }

    def _extract_hosted_decision_label_scores(self, payload: Dict[str, Any]) -> Dict[str, float]:
        top_pairs = extract_first_token_top_logprobs(payload)
        grouped: Dict[str, List[float]] = {label: [] for label in self.hosted_decision_labels}
        for token, logprob in top_pairs:
            normalized = token.strip()
            for label in self.hosted_decision_labels:
                if normalized == label:
                    grouped[label].append(float(logprob))
        missing = [label for label, values in grouped.items() if not values]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                f"OpenRouter chat response lacked usable decision-label logprobs for: {missing_text}."
            )
        return {label: self._logsumexp(values) for label, values in grouped.items()}

    def _record_hosted_decision_chat_debug(
        self,
        profile_name: str,
        payload: Dict[str, Any],
        error_message: str,
    ) -> None:
        top_pairs = extract_first_token_top_logprobs(payload)
        self._last_decision_backend_meta["debug"] = {
            "error": error_message,
            "scoring_mode": "chat_logprobs_binary_head",
            "provider": payload.get("provider"),
            "top_logprobs": top_pairs,
            "raw_payload_excerpt": self._truncate_debug_text(
                json.dumps(payload, ensure_ascii=True, sort_keys=True),
                limit=1500,
            ),
        }
        self._log_once(
            "hosted_decision_chat_debug_warning",
            (
                f"Hosted decision debug for profile '{profile_name}': {error_message} "
                f"provider={payload.get('provider')!r} "
                f"first_token_top_logprobs=[{self._format_first_token_logprobs(top_pairs)}]"
            ),
            "warning",
        )
        self._log_once(
            "hosted_decision_chat_debug_payload",
            (
                "Hosted decision raw payload excerpt: "
                f"{self._truncate_debug_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), limit=1500)}"
            ),
            "debug",
        )

    def _record_hosted_decision_request_debug(
        self,
        profile_name: str,
        prompt: Dict[str, str],
        error_message: str,
        *,
        prompt_idx: Optional[int] = None,
        src_label: Optional[str] = None,
        tgt_label: Optional[str] = None,
        request_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        system_text = str(prompt.get("system", ""))
        user_text = str(prompt.get("user", ""))
        prompt_blob = json.dumps(
            {"system": system_text, "user": user_text},
            ensure_ascii=True,
            sort_keys=True,
        )
        prompt_sha1 = hashlib.sha1(prompt_blob.encode("utf-8")).hexdigest()[:12]
        system_stats = self._text_debug_stats(system_text)
        user_stats = self._text_debug_stats(user_text)
        request_excerpt = None
        if request_payload is not None:
            try:
                request_excerpt = self._truncate_debug_text(
                    json.dumps(request_payload, ensure_ascii=True, sort_keys=True),
                    limit=1800,
                )
            except Exception:
                request_excerpt = self._truncate_debug_text(str(request_payload), limit=1800)
        self._last_decision_backend_meta["request_debug"] = {
            "error": error_message,
            "prompt_sha1": prompt_sha1,
            "prompt_idx": prompt_idx,
            "src_label": src_label,
            "tgt_label": tgt_label,
            "system_stats": system_stats,
            "user_stats": user_stats,
            "system_excerpt": self._truncate_debug_text(system_text, limit=400),
            "user_excerpt": self._truncate_debug_text(user_text, limit=1200),
            "request_payload_excerpt": request_excerpt,
        }
        label_note = ""
        if src_label is not None or tgt_label is not None:
            label_note = (
                f" src={self._truncate_debug_text(src_label, limit=120)!r}"
                f" tgt={self._truncate_debug_text(tgt_label, limit=120)!r}"
            )
        idx_note = "" if prompt_idx is None else f" batch_prompt_idx={prompt_idx}"
        self._log_once(
            f"hosted_decision_request_debug_warning:{prompt_sha1}",
            (
                f"Hosted decision request debug for profile '{profile_name}'{idx_note}: "
                f"prompt_sha1={prompt_sha1}{label_note} "
                f"system_stats={system_stats} user_stats={user_stats} "
                f"error={error_message}"
            ),
            "warning",
        )
        self._log_once(
            f"hosted_decision_request_debug_prompt:{prompt_sha1}",
            (
                f"Hosted decision request prompt excerpt [{prompt_sha1}]: "
                f"system={self._truncate_debug_text(system_text, limit=400)!r} "
                f"user={self._truncate_debug_text(user_text, limit=1500)!r}"
            ),
            "debug",
        )
        if request_excerpt:
            self._log_once(
                f"hosted_decision_request_debug_payload:{prompt_sha1}",
                f"Hosted decision request payload excerpt [{prompt_sha1}]: {request_excerpt}",
                "debug",
            )
        return prompt_sha1

    def _probe_hosted_decision_profile(self, profile) -> Dict[str, Any]:
        cached = self._decision_probe_cache.get(profile.name)
        if cached is not None:
            return cached
        positive_label, negative_label = self.hosted_decision_labels
        result = {
            "passed": False,
            "provider": None,
            "error": None,
            "debug": None,
        }
        payload = None
        try:
            logit_bias, label_ids = self._hosted_decision_logit_bias(profile)
            prompt = {
                "system": "You are a binary classifier.",
                "user": (
                    f"Return exactly one token: {positive_label} or {negative_label}.\n"
                    f"{positive_label} = positive\n"
                    f"{negative_label} = negative"
                ),
            }
            payload = self._llm_router.hosted.chat_completion(
                profile=profile,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                logprobs=True,
                top_logprobs=20,
                logit_bias=logit_bias,
                provider={"require_parameters": True},
                seed=self.request_seed,
            )
            self._extract_hosted_decision_label_scores(payload)
            result["passed"] = True
            result["provider"] = payload.get("provider")
            result["label_ids"] = label_ids
        except (RuntimeError, ValueError, KeyError, OSError, urlerror.URLError) as exc:
            result["error"] = str(exc)
            if payload is not None:
                result["debug"] = {
                    "provider": payload.get("provider"),
                    "top_logprobs": extract_first_token_top_logprobs(payload),
                    "raw_payload_excerpt": self._truncate_debug_text(
                        json.dumps(payload, ensure_ascii=True, sort_keys=True),
                        limit=1500,
                    ),
                }
            self._record_decision_stat("probe_failures")
            self._log_once(
                f"hosted_decision_probe_failed:{profile.name}",
                (
                f"OpenRouter decision chat-logprob probe failed for profile '{profile.name}'. "
                    f"Falling back to local decision scoring. Error: {exc}"
                ),
                "warning",
            )
            if result["debug"] is not None:
                self._log_once(
                    f"hosted_decision_probe_failed_debug:{profile.name}",
                    (
                        f"Hosted decision probe debug for profile '{profile.name}': "
                        f"provider={result['debug'].get('provider')!r}, "
                        f"top_logprobs=[{self._format_first_token_logprobs(result['debug'].get('top_logprobs') or [])}]"
                    ),
                    "warning",
                )
                self._log_once(
                    f"hosted_decision_probe_failed_payload:{profile.name}",
                    (
                        "Hosted decision probe raw payload excerpt: "
                        f"{result['debug'].get('raw_payload_excerpt')}"
                    ),
                    "debug",
                )
        self._decision_probe_cache[profile.name] = result
        return result

    def reset_llm_calibration_tracking(self) -> None:
        self._computed_llm_calibration = None
        self._calibration_messages = []
        self._calibration_pending_probs = []
        self._calibration_pending_labels = []
        self._llm_calibration_fitted_once = False

    def _apply_llm_calibration(self, probs: torch.Tensor) -> torch.Tensor:
        if not self._llm_calibration_can_apply:
            return probs
        a = torch.tensor(self.llm_calibration_a, device=probs.device, dtype=probs.dtype)
        b = torch.tensor(self.llm_calibration_b, device=probs.device, dtype=probs.dtype)
        calibrated = torch.sigmoid(a * probs + b)
        return calibrated.clamp(0.0, 1.0)

    def _collect_calibration_samples(
        self,
        idxs: List[int],
        probs: torch.Tensor,
        labels: Optional[List[float]],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if labels is None:
            self._calibration_messages.append("LLM calibration skipped: ground-truth labels not provided.")
            return None
        xs: List[float] = []
        ys: List[float] = []
        for offset, idx in enumerate(idxs):
            if idx >= len(labels):
                continue
            gt = labels[idx]
            if gt is None:
                continue
            xs.append(float(probs[offset].item()))
            ys.append(float(gt))
        if len(xs) < 2:
            self._calibration_messages.append(
                f"LLM calibration skipped: insufficient labelled pairs ({len(xs)})."
            )
            return None
        x_t = torch.tensor(xs, dtype=torch.float64)
        y_t = torch.tensor(ys, dtype=torch.float64).clamp(0.0, 1.0)
        return x_t, y_t

    def _fit_llm_calibration_from_samples(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        max_iters: int = 400,
        lr: float = 1.0,
    ) -> Dict[str, Any]:
        x = probs.clamp(1e-6, 1.0 - 1e-6)
        y = labels
        a = torch.tensor(0.0, dtype=torch.float64)
        b = torch.tensor(0.0, dtype=torch.float64)
        step = lr
        for _ in range(max_iters):
            logits = a * x + b
            sig = torch.sigmoid(logits)
            error = sig - y
            grad_a = (error * x).mean()
            grad_b = error.mean()
            a -= step * grad_a
            b -= step * grad_b
            if max(abs(grad_a.item()), abs(grad_b.item())) < 1e-6:
                break
            step = max(step * 0.99, 0.01)
        logits = a * x + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y).item()
        info = {
            "a": float(a.item()),
            "b": float(b.item()),
            "loss": loss,
            "samples": int(x.numel()),
        }
        return info

    def _llm_calibration_payload(self, batch_samples: int = 0) -> Dict[str, Any]:
        return {
            "enabled": self.use_llm_calibration,
            "configured": self._llm_calibration_can_apply,
            "a": self.llm_calibration_a,
            "b": self.llm_calibration_b,
            "info": self.llm_calibration_info,
            "learned": self._computed_llm_calibration,
            "messages": list(self._calibration_messages),
            "pending_batch_samples": batch_samples,
            "pending_total_samples": len(self._calibration_pending_probs),
        }

    def llm_calibration_state(self) -> Dict[str, Any]:
        return self._llm_calibration_payload(batch_samples=0)

    def finalize_llm_calibration(self) -> Optional[Dict[str, Any]]:
        if (
            not self.use_llm_calibration
            or self._llm_calibration_can_apply
            or self._llm_calibration_fitted_once
        ):
            return None
        if not self._calibration_pending_probs or not self._calibration_pending_labels:
            return None
        probs = torch.tensor(self._calibration_pending_probs, dtype=torch.float64)
        labels = torch.tensor(self._calibration_pending_labels, dtype=torch.float64)
        info = self._fit_llm_calibration_from_samples(probs, labels)
        self._computed_llm_calibration = info
        self._llm_calibration_fitted_once = True
        self._calibration_messages.append(
            (
                "Fitted final LLM calibration coefficients "
                f"(samples={info['samples']}, a={info['a']:.4f}, b={info['b']:.4f}, loss={info['loss']:.4f})."
            )
        )
        return self._llm_calibration_payload(batch_samples=0)

    @torch.inference_mode()
    def generate_summaries_batched(self, labels: List[str], contexts: List[str]) -> List[str]:
        if not self.use_llm:
            return ["" for _ in labels]
        if not labels:
            return []
        summary_backend = self._llm_router.resolve_task("summary")
        self._last_summary_backend_meta = self._resolved_backend_metadata(summary_backend)
        outputs = [""] * len(labels)
        pending: Dict[str, Dict[str, Any]] = {}
        for idx, (label, ctx) in enumerate(zip(labels, contexts)):
            key = self._summary_key(label, ctx)
            cached = self._summary_cache.get(key)
            if cached is not None:
                outputs[idx] = cached
                continue
            entry = pending.setdefault(key, {"label": label, "context": ctx, "indices": []})
            entry["indices"].append(idx)

        if pending:
            pending_keys = list(pending.keys())
            labs = [pending[k]["label"] for k in pending_keys]
            ctxs = [pending[k]["context"] for k in pending_keys]
            if summary_backend.backend == "openrouter":
                self._log_once(
                    "hosted_summary_stage_start",
                    (
                        "Hosted summary stage: "
                        f"{len(labels)} requested, {len(pending_keys)} uncached unique prompts, "
                        f"concurrency={self.llm_summary_batch_size or len(pending_keys)}."
                    ),
                    "debug",
                )
            generated = self._generate_summaries_uncached(labs, ctxs, summary_backend)
            if summary_backend.backend == "openrouter":
                self._log_once(
                    "hosted_summary_stage_done",
                    f"Hosted summary stage completed for {len(pending_keys)} uncached prompts.",
                    "debug",
                )
            for key, summary in zip(pending_keys, generated):
                clean = self._parse_structured_text(summary, "summary", self._clean_summary_text)
                self._cache_store(self._summary_cache, key, clean, self.max_cached_summaries)
                for idx in pending[key]["indices"]:
                    outputs[idx] = clean

        self._record_summary_stats(outputs)
        return outputs

    def _generate_summaries_uncached(self, labels: List[str], contexts: List[str], resolved_backend) -> List[str]:
        if not labels:
            return []
        if resolved_backend.backend == "openrouter":
            profile = self._llm_router.profiles.get(resolved_backend.profile_name or "")
            if profile is None:
                raise RuntimeError("OpenRouter summary profile was resolved but not found.")
            prompts = [self._summary_prompt(label, ctx) for label, ctx in zip(labels, contexts)]
            return self._run_hosted_chat_prompts(
                prompts=prompts,
                profile=profile,
                max_tokens=self.max_new_tokens_llm,
                temperature=self.llm_temperature,
                top_p=self.llm_top_p,
                concurrency=self.llm_summary_batch_size,
            )
        self._ensure_local_llm()
        prompts = [self._render_llm_prompt(self._summary_prompt(l, c)) for l, c in zip(labels, contexts)]
        outputs = [""] * len(prompts)
        chunk = self.llm_summary_batch_size or len(prompts)
        chunk = chunk if chunk > 0 else len(prompts)
        for start in range(0, len(prompts), chunk):
            end = min(start + chunk, len(prompts))
            enc = self.llm_tok(
                prompts[start:end],
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_total_tokens_llm_summary,
            ).to(self.device)
            with torch.amp.autocast(device_type=self.device_type, enabled=self.fp16):
                out = self.llm.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens_llm,
                    temperature=self.llm_temperature,
                    top_p=self.llm_top_p,
                    do_sample=self.llm_do_sample,
                    pad_token_id=self.llm_tok.eos_token_id,
                )
            new_tokens = self._strip_llm_prompt_tokens(enc, out)
            decoded = self.llm_tok.batch_decode(new_tokens, skip_special_tokens=True)
            for offset, text in enumerate(decoded):
                outputs[start + offset] = text
        return outputs

    @torch.inference_mode()
    def generate_rationales_batched(
        self,
        src_labels: List[str],
        tgt_labels: List[str],
        src_summaries: List[str],
        tgt_summaries: List[str],
        decisions: List[str],
        decision_contexts: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        completion_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[str]:
        if not (self.use_llm and self.generate_llm_rationales):
            return ["" for _ in src_labels]
        if not src_labels:
            return []
        if decision_contexts is None:
            decision_contexts = ["" for _ in src_labels]
        rationale_backend = self._llm_router.resolve_task("rationale")
        self._last_rationale_backend_meta = self._resolved_backend_metadata(rationale_backend)
        outputs = [""] * len(src_labels)
        pending: Dict[str, Dict[str, Any]] = {}
        for idx, (s_lab, t_lab, s_sum, t_sum, decision, decision_context) in enumerate(
            zip(src_labels, tgt_labels, src_summaries, tgt_summaries, decisions, decision_contexts)
        ):
            key = self._rationale_key(s_lab, t_lab, s_sum, t_sum, decision, decision_context)
            cached = self._rationale_cache.get(key)
            if cached is not None:
                outputs[idx] = cached
                continue
            entry = pending.setdefault(
                key,
                {
                    "src_label": s_lab,
                    "tgt_label": t_lab,
                    "src_summary": s_sum,
                    "tgt_summary": t_sum,
                    "decision": decision,
                    "decision_context": decision_context,
                    "indices": [],
                },
            )
            entry["indices"].append(idx)

        total_records = len(src_labels)
        uncached_unique_prompts = len(pending)
        uncached_record_count = sum(len(entry["indices"]) for entry in pending.values())
        cached_record_count = total_records - uncached_record_count
        configured_concurrency = self.llm_rationale_batch_size or max(1, uncached_unique_prompts)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "rationale",
                    "event": "start",
                    "total_records": total_records,
                    "cached_records": cached_record_count,
                    "uncached_records": uncached_record_count,
                    "uncached_unique_prompts": uncached_unique_prompts,
                    "backend": rationale_backend.backend,
                    "model": self._last_rationale_backend_meta.get("model"),
                    "concurrency": configured_concurrency,
                }
            )

        if pending:
            pending_keys = list(pending.keys())
            prompts = [
                self._rationale_prompt(
                    pending[key]["src_label"],
                    pending[key]["tgt_label"],
                    pending[key]["src_summary"],
                    pending[key]["tgt_summary"],
                    pending[key]["decision"],
                    pending[key]["decision_context"],
                )
                for key in pending_keys
            ]

            def _on_prompt_completed(prompt_idx: int, raw_text: str) -> None:
                if prompt_idx < 0 or prompt_idx >= len(pending_keys):
                    return
                key = pending_keys[prompt_idx]
                clean = self._parse_structured_text(raw_text, "rationale", self._clean_rationale_text)
                self._cache_store(self._rationale_cache, key, clean, self.max_cached_rationales)
                indices = list(pending[key]["indices"])
                for idx in indices:
                    outputs[idx] = clean
                if completion_callback is not None:
                    completion_callback(
                        {
                            "stage": "rationale",
                            "event": "completion",
                            "indices": indices,
                            "rationale": clean,
                        }
                    )

            if rationale_backend.backend == "openrouter":
                self._log_once(
                    "hosted_rationale_stage_start",
                    (
                        "Hosted rationale stage: "
                        f"{len(src_labels)} requested, {len(pending_keys)} uncached unique prompts, "
                        f"concurrency={self.llm_rationale_batch_size or len(pending_keys)}."
                    ),
                    "debug",
                )
            generated = self._generate_rationales_uncached(
                prompts,
                rationale_backend,
                progress_callback=progress_callback,
                prompt_record_counts=[len(pending[key]["indices"]) for key in pending_keys],
                total_records=total_records,
                cached_records=cached_record_count,
                prompt_completed_callback=_on_prompt_completed,
            )
            if rationale_backend.backend == "openrouter":
                self._log_once(
                    "hosted_rationale_stage_done",
                    f"Hosted rationale stage completed for {len(pending_keys)} uncached prompts.",
                    "debug",
                )
            for key, rationale in zip(pending_keys, generated):
                if all(outputs[idx] for idx in pending[key]["indices"]):
                    continue
                clean = self._parse_structured_text(rationale, "rationale", self._clean_rationale_text)
                self._cache_store(self._rationale_cache, key, clean, self.max_cached_rationales)
                for idx in pending[key]["indices"]:
                    outputs[idx] = clean

        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "rationale",
                    "event": "done",
                    "total_records": total_records,
                    "cached_records": cached_record_count,
                    "uncached_records": uncached_record_count,
                    "uncached_unique_prompts": uncached_unique_prompts,
                    "backend": rationale_backend.backend,
                    "model": self._last_rationale_backend_meta.get("model"),
                    "concurrency": configured_concurrency,
                }
            )

        return outputs

    def _generate_rationales_uncached(
        self,
        prompts: List[Dict[str, str]],
        resolved_backend,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        prompt_record_counts: Optional[List[int]] = None,
        total_records: int = 0,
        cached_records: int = 0,
        prompt_completed_callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[str]:
        if not prompts:
            return []
        prompt_record_counts = list(prompt_record_counts or [1 for _ in prompts])
        total_uncached_records = int(sum(prompt_record_counts))
        completed_records = 0
        completed_unique_prompts = 0

        def _emit_progress(record_delta: int, unique_delta: int) -> None:
            nonlocal completed_records, completed_unique_prompts
            completed_records += int(record_delta)
            completed_unique_prompts += int(unique_delta)
            if progress_callback is None:
                return
            progress_callback(
                {
                    "stage": "rationale",
                    "event": "progress",
                    "total_records": total_records,
                    "cached_records": cached_records,
                    "total_uncached_records": total_uncached_records,
                    "completed_uncached_records": completed_records,
                    "total_unique_prompts": len(prompts),
                    "completed_unique_prompts": completed_unique_prompts,
                }
            )

        if resolved_backend.backend == "openrouter":
            profile = self._llm_router.profiles.get(resolved_backend.profile_name or "")
            if profile is None:
                raise RuntimeError("OpenRouter rationale profile was resolved but not found.")
            workers = self.llm_rationale_batch_size or len(prompts)
            workers = max(1, min(int(workers), len(prompts)))

            def _call(prompt: Dict[str, str]) -> str:
                payload = self._llm_router.hosted.chat_completion(
                    profile=profile,
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    max_tokens=self.max_new_tokens_llm_rationale,
                    temperature=self.llm_temperature,
                    top_p=self.llm_top_p,
                    seed=self.request_seed,
                )
                return extract_chat_text(payload)

            outputs = [""] * len(prompts)
            if workers == 1:
                for idx, prompt in enumerate(prompts):
                    outputs[idx] = _call(prompt)
                    if prompt_completed_callback is not None:
                        prompt_completed_callback(idx, outputs[idx])
                    _emit_progress(prompt_record_counts[idx], 1)
                return outputs
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_idx = {
                    executor.submit(_call, prompt): idx
                    for idx, prompt in enumerate(prompts)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    outputs[idx] = future.result()
                    if prompt_completed_callback is not None:
                        prompt_completed_callback(idx, outputs[idx])
                    _emit_progress(prompt_record_counts[idx], 1)
            return outputs
        self._ensure_local_llm()
        rendered = [self._render_llm_prompt(prompt) for prompt in prompts]
        outputs = [""] * len(rendered)
        chunk = self.llm_rationale_batch_size or len(rendered)
        chunk = chunk if chunk > 0 else len(rendered)
        for start in range(0, len(rendered), chunk):
            end = min(start + chunk, len(rendered))
            enc = self.llm_tok(
                rendered[start:end],
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_total_tokens_llm_rationale,
            ).to(self.device)
            with torch.amp.autocast(device_type=self.device_type, enabled=self.fp16):
                out = self.llm.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens_llm_rationale,
                    temperature=self.llm_temperature,
                    top_p=self.llm_top_p,
                    do_sample=self.llm_do_sample,
                    pad_token_id=self.llm_tok.eos_token_id,
                )
            new_tokens = self._strip_llm_prompt_tokens(enc, out)
            decoded = self.llm_tok.batch_decode(new_tokens, skip_special_tokens=True)
            for offset, text in enumerate(decoded):
                prompt_idx = start + offset
                outputs[prompt_idx] = text
                if prompt_completed_callback is not None:
                    prompt_completed_callback(prompt_idx, text)
            _emit_progress(sum(prompt_record_counts[start:end]), end - start)
        return outputs

    @torch.inference_mode()
    def llm_yesno_probs_batched(
        self,
        src_labels: List[str],
        tgt_labels: List[str],
        src_summaries: List[str],
        tgt_summaries: List[str],
    ) -> torch.Tensor:
        if not self.use_llm:
            return torch.zeros(len(src_labels), device=self.device)
        if not src_labels:
            return torch.zeros(0, device=self.device)
        self._record_decision_stat("requested", len(src_labels))
        decision_backend = self._llm_router.resolve_task("decision", require_logprobs=True)
        self._last_decision_backend_meta = self._resolved_backend_metadata(decision_backend)
        self._last_decision_backend_meta["decision_scoring_mode"] = "chat_logprobs_binary_head"
        if decision_backend.backend == "openrouter":
            profile = self._llm_router.profiles.get(decision_backend.profile_name or "")
            if profile is not None:
                self._last_decision_backend_meta["endpoint"] = "chat/completions"
                self._log_once(
                    "hosted_decision_probe_start",
                    (
                        "Hosted decision probe: checking chat-logprob binary-head support "
                        f"for profile '{profile.name}'."
                    ),
                    "debug",
                )
                probe = self._probe_hosted_decision_profile(profile)
                self._last_decision_backend_meta["decision_probe_passed"] = bool(probe.get("passed"))
                self._last_decision_backend_meta["decision_probe_error"] = probe.get("error")
                if probe.get("provider"):
                    self._last_decision_backend_meta["provider"] = probe.get("provider")
                if probe.get("passed"):
                    self._log_once(
                        "hosted_decision_probe_passed",
                        (
                            f"Hosted decision probe passed for profile '{profile.name}'"
                            + (
                                f" via provider '{probe.get('provider')}'."
                                if probe.get("provider") else "."
                            )
                        ),
                        "debug",
                    )
                if not probe.get("passed"):
                    self._last_decision_backend_meta["fallback_triggered"] = True
                    self._last_decision_backend_meta["fallback_reason"] = "decision_probe_failed"
                    self._last_decision_backend_meta["fallback_error"] = probe.get("error")
                    self._last_decision_backend_meta["backend"] = "local_hf"
                    self._last_decision_backend_meta["model"] = self.llm_model_name
                    self._record_decision_stat("local_fallbacks", len(src_labels))
                else:
                    self._record_decision_stat("hosted_attempted", len(src_labels))
                    try:
                        logit_bias, label_ids = self._hosted_decision_logit_bias(profile)
                        self._last_decision_backend_meta["label_ids"] = label_ids
                        outputs: List[float] = []
                        last_provider = self._last_decision_backend_meta.get("provider")
                        workers = self.llm_decision_batch_size or len(src_labels)
                        workers = max(1, min(int(workers), len(src_labels)))
                        self._log_once(
                            "hosted_decision_stage_start",
                            (
                                "Hosted decision stage: "
                                f"{len(src_labels)} pairs, concurrency={workers}, "
                                f"labels={self.hosted_decision_labels[0]}/{self.hosted_decision_labels[1]}, "
                                f"logit_bias={self.hosted_decision_logit_bias:.2f}."
                            ),
                            "debug",
                        )
                        prompts = [
                            self._hosted_decision_prompt(s_lab, t_lab, s_sum, t_sum)
                            for s_lab, t_lab, s_sum, t_sum in zip(
                                src_labels, tgt_labels, src_summaries, tgt_summaries
                            )
                        ]
                        if prompts:
                            self._log_once(
                                "hosted_decision_first_chunk_start",
                                (
                                    "Hosted decision first concurrent wave started: "
                                    f"{min(workers, len(prompts))} in-flight requests."
                                ),
                                "debug",
                            )

                        def _score_prompt(index_prompt: Tuple[int, Dict[str, str]]) -> Dict[str, Any]:
                            idx, prompt = index_prompt
                            request_payload = {
                                "messages": [
                                    {"role": "system", "content": prompt["system"]},
                                    {"role": "user", "content": prompt["user"]},
                                ],
                                "max_tokens": 1,
                                "temperature": 0.0,
                                "top_p": 1.0,
                                "logprobs": True,
                                "top_logprobs": 20,
                                "logit_bias": logit_bias,
                                "provider": {"require_parameters": True},
                                "seed": self.request_seed,
                                "model": getattr(profile, "model", None),
                            }
                            try:
                                return self._llm_router.hosted.chat_completion(
                                    profile=profile,
                                    messages=request_payload["messages"],
                                    max_tokens=1,
                                    temperature=0.0,
                                    top_p=1.0,
                                    logprobs=True,
                                    top_logprobs=20,
                                    logit_bias=logit_bias,
                                    provider={"require_parameters": True},
                                    seed=self.request_seed,
                                )
                            except Exception as exc:
                                prompt_sha1 = self._record_hosted_decision_request_debug(
                                    profile.name,
                                    prompt,
                                    str(exc),
                                    prompt_idx=idx,
                                    src_label=src_labels[idx] if idx < len(src_labels) else None,
                                    tgt_label=tgt_labels[idx] if idx < len(tgt_labels) else None,
                                    request_payload=request_payload,
                                )
                                raise RuntimeError(
                                    f"{exc} [prompt_sha1={prompt_sha1}, batch_prompt_idx={idx}]"
                                ) from exc

                        if workers == 1:
                            payloads = [_score_prompt((idx, prompt)) for idx, prompt in enumerate(prompts)]
                        else:
                            with ThreadPoolExecutor(max_workers=workers) as executor:
                                payloads = list(executor.map(_score_prompt, enumerate(prompts)))
                        if prompts:
                            self._log_once(
                                "hosted_decision_first_chunk_done",
                                "Hosted decision first concurrent wave completed successfully.",
                                "debug",
                            )
                        positive_label, negative_label = self.hosted_decision_labels
                        for payload in payloads:
                            label_scores = self._extract_hosted_decision_label_scores(payload)
                            if payload.get("provider"):
                                if last_provider is None:
                                    last_provider = payload.get("provider")
                                elif last_provider != payload.get("provider"):
                                    last_provider = "mixed"
                            stacked = torch.tensor(
                                [label_scores[positive_label], label_scores[negative_label]],
                                dtype=torch.float64,
                            )
                            outputs.append(float(torch.softmax(stacked, dim=-1)[0].item()))
                        self._record_decision_stat("hosted_scored", len(outputs))
                        self._last_decision_backend_meta["provider"] = last_provider
                        return torch.tensor(outputs, dtype=torch.float32, device=self.device)
                    except (RuntimeError, ValueError, KeyError, OSError, urlerror.URLError) as exc:
                        self.log(f"Hosted decision backend failed; falling back to local LLM. Error: {exc}", "warning")
                        if 'payloads' in locals() and payloads:
                            self._record_hosted_decision_chat_debug(profile.name, payloads[0], str(exc))
                        self._last_decision_backend_meta["fallback_triggered"] = True
                        self._last_decision_backend_meta["fallback_reason"] = "decision_scoring_failed"
                        self._last_decision_backend_meta["fallback_error"] = str(exc)
                        self._last_decision_backend_meta["backend"] = "local_hf"
                        self._last_decision_backend_meta["model"] = self.llm_model_name
                        self._record_decision_stat("scoring_failures", len(src_labels))
                        self._record_decision_stat("local_fallbacks", len(src_labels))
        elif decision_backend.fallback_triggered:
            self._record_decision_stat("local_fallbacks", len(src_labels))
        self._ensure_local_llm()
        prompts = [
            self._render_llm_prompt(
                self._decision_prompt(s_lab, t_lab, s_sum, t_sum),
                add_generation_prompt=True,
            )
            for s_lab, t_lab, s_sum, t_sum in zip(src_labels, tgt_labels, src_summaries, tgt_summaries)
        ]
        probs = torch.zeros(len(prompts), device=self.device)
        chunk = self.llm_decision_batch_size or len(prompts)
        chunk = chunk if chunk > 0 else len(prompts)
        for start in range(0, len(prompts), chunk):
            end = min(start + chunk, len(prompts))
            enc = self.llm_tok(
                prompts[start:end],
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_total_tokens_llm_decision,
            ).to(self.device)
            with torch.amp.autocast(device_type=self.device_type, enabled=self.fp16):
                out = self.llm(**enc)
                last_logits = out.logits[:, -1, :]
                logprobs = torch.log_softmax(last_logits, dim=-1)
                yes_lp = torch.logsumexp(logprobs[:, self.yes_token_ids], dim=-1)
                no_lp = torch.logsumexp(logprobs[:, self.no_token_ids], dim=-1)
                stacked = torch.stack([yes_lp, no_lp], dim=-1)
                probs[start:end] = torch.softmax(stacked, dim=-1)[:, 0]
        return probs

    def generate_final_rationales_for_records(
        self,
        records: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        completion_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[str]:
        if not (self.use_llm and self.generate_llm_rationales):
            return ["" for _ in records]
        src_labels: List[str] = []
        tgt_labels: List[str] = []
        src_summaries: List[str] = []
        tgt_summaries: List[str] = []
        decisions: List[str] = []
        decision_contexts: List[str] = []
        for rec in records:
            labels = rec.get("selected_labels") or {}
            summaries = rec.get("llm_summaries") or {}
            prediction = rec.get("prediction") or {}
            src_labels.append(str(labels.get("source", "")))
            tgt_labels.append(str(labels.get("target", "")))
            src_summaries.append(str(summaries.get("source", "")))
            tgt_summaries.append(str(summaries.get("target", "")))
            decisions.append(str(prediction.get("rationale_decision_label", "")))
            decision_contexts.append(self._final_alignment_context_for_rationale(rec))
        return self.generate_rationales_batched(
            src_labels=src_labels,
            tgt_labels=tgt_labels,
            src_summaries=src_summaries,
            tgt_summaries=tgt_summaries,
            decisions=decisions,
            decision_contexts=decision_contexts,
            progress_callback=progress_callback,
            completion_callback=completion_callback,
        )

    def _final_alignment_context_for_rationale(self, record: Dict[str, Any]) -> str:
        conf = record.get("confidences") or {}
        pred = record.get("prediction") or {}
        selector_ran = "S_select" in conf or any(str(key).startswith("selector_") for key in pred)
        if not selector_ran:
            return ""

        def _fmt(value: Any) -> str:
            try:
                if value is None:
                    return "unavailable"
                return f"{float(value):.3f}"
            except (TypeError, ValueError):
                return "unavailable"

        def _yes_no(value: Any) -> str:
            return "yes" if bool(value) else "no"

        lines = [
            f"Pairwise score before selector: {_fmt(conf.get('S_pair_final'))}",
            f"Selector score used for final alignment: {_fmt(conf.get('S_select', conf.get('S_final')))}",
            f"Raw selector support before abstention: {_fmt(conf.get('selection_utility'))}",
            f"NO_MATCH abstention risk: {_fmt(conf.get('selection_no_match_prob'))}",
            f"Selector margin: {_fmt(conf.get('selection_margin'))}",
            f"Selector entropy: {_fmt(conf.get('selection_entropy'))}",
            f"Selector evidence agreement: {_fmt(conf.get('selection_evidence_support'))}",
            f"Selector distinctive evidence score: {_fmt(conf.get('selection_distinctive'))}",
            f"Selector abstained: {_yes_no(pred.get('selector_abstained', False))}",
            f"Selector LLM arbitration used: {_yes_no(pred.get('selector_llm_used', False))}",
        ]
        reason = pred.get("selector_reason")
        if reason:
            lines.append(f"Selector reason: {reason}")
        if "threshold_positive" in pred:
            lines.append(f"Passed final score threshold: {_yes_no(pred.get('threshold_positive'))}")
        if "saved_alignment_member" in pred:
            lines.append(f"Kept in saved alignment after cardinality filtering: {_yes_no(pred.get('saved_alignment_member'))}")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Context attribution helper
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def _context_similarity_texts(self, s_text: str, t_text: str) -> float:
        e_cs = self.encode_contexts_batch([s_text])
        e_ct = self.encode_contexts_batch([t_text])
        return float(self._cos_sim(e_cs, e_ct).item())

    @torch.inference_mode()
    def _triple_attributions_by_sentences(
        self, src_triples: List[str], tgt_triples: List[str], s_ctx_orig: float
    ) -> List[Dict[str, Any]]:
        drops, tags = [], []
        for i, s in enumerate(src_triples):
            mod_src = self._join_context(src_triples[:i] + src_triples[i+1:])
            mod_tgt = self._join_context(tgt_triples)
            s_minus = self._context_similarity_texts(mod_src, mod_tgt)
            drops.append(max(0.0, s_ctx_orig - s_minus))
            tags.append(("source", i, s))
        for j, s in enumerate(tgt_triples):
            mod_src = self._join_context(src_triples)
            mod_tgt = self._join_context(tgt_triples[:j] + tgt_triples[j+1:])
            s_minus = self._context_similarity_texts(mod_src, mod_tgt)
            drops.append(max(0.0, s_ctx_orig - s_minus))
            tags.append(("target", j, s))
        denom = sum(abs(d) for d in drops) or 1.0
        imps = [d / denom for d in drops]
        return [
            {"side": side, "index": idx, "sentence": sent, "importance": float(imp)}
            for (side, idx, sent), imp in zip(tags, imps)
        ]

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def forward(
        self,
        src_iris: List[str],
        tgt_iris: List[str],
        src_label_lists: List[List[str]],
        tgt_label_lists: List[List[str]],
        src_contexts: Optional[List[List[str]]] = None,
        tgt_contexts: Optional[List[List[str]]] = None,
        src_ctx_raw: Optional[List[List[str]]] = None,
        tgt_ctx_raw: Optional[List[List[str]]] = None,
        src_ctx_bridges: Optional[List[List[str]]] = None,
        tgt_ctx_bridges: Optional[List[List[str]]] = None,
        label: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        N = len(src_label_lists)
        assert len(tgt_label_lists) == N
        assert len(src_iris) == N and len(tgt_iris) == N

        # Join context triples
        src_contexts = src_contexts or [[] for _ in range(N)]
        tgt_contexts = tgt_contexts or [[] for _ in range(N)]
        src_ctx_raw = src_ctx_raw or [[] for _ in range(N)]
        tgt_ctx_raw = tgt_ctx_raw or [[] for _ in range(N)]
        src_ctx_bridges = src_ctx_bridges or [[] for _ in range(N)]
        tgt_ctx_bridges = tgt_ctx_bridges or [[] for _ in range(N)]
        src_ctx_joined = [self._join_context(c) for c in src_contexts]
        tgt_ctx_joined = [self._join_context(c) for c in tgt_contexts]
        if self.use_context:
            src_ctx_present = torch.tensor(
                [bool(ctx.strip()) for ctx in src_ctx_joined],
                device=self.device,
                dtype=torch.bool,
            )
            tgt_ctx_present = torch.tensor(
                [bool(ctx.strip()) for ctx in tgt_ctx_joined],
                device=self.device,
                dtype=torch.bool,
            )
            ctx_pair_mask = src_ctx_present & tgt_ctx_present
        else:
            ctx_pair_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._computed_llm_calibration = None
        self._calibration_messages = []

        # ---------- LEXICAL + LABEL PAIR POOL ----------
        if self.use_lexical:
            self._log_once("lexical_encoding", "Encoding lexical label variants...", "info")
            flat_src = [lab for labs in src_label_lists for lab in labs]
            flat_tgt = [lab for labs in tgt_label_lists for lab in labs]
            if len(flat_src) == 0 or len(flat_tgt) == 0:
                s_label = torch.zeros(N, device=self.device)
                best_pairs = [("", "") for _ in range(N)]
            else:
                e_src = self.encode_labels_batch(flat_src)
                e_tgt = self.encode_labels_batch(flat_tgt)
                idx = 0; src_emb_slices = []
                for labs in src_label_lists:
                    src_emb_slices.append(e_src[idx: idx + len(labs)] if len(labs) else e_src[0:0]); idx += len(labs)
                idx = 0; tgt_emb_slices = []
                for labs in tgt_label_lists:
                    tgt_emb_slices.append(e_tgt[idx: idx + len(labs)] if len(labs) else e_tgt[0:0]); idx += len(labs)

                sims, best_pairs = [], []
                for labs_s, labs_t, Es, Et in zip(src_label_lists, tgt_label_lists, src_emb_slices, tgt_emb_slices):
                    if Es.shape[0] == 0 or Et.shape[0] == 0:
                        sims.append(torch.tensor(0.0, device=self.device))
                        best_pairs.append(("", ""))
                        continue
                    Esn = torch.nn.functional.normalize(Es, dim=-1)
                    Etn = torch.nn.functional.normalize(Et, dim=-1)
                    mat = Esn @ Etn.T
                    score, pair = self._select_label_pair(mat, labs_s, labs_t)
                    sims.append(score)
                    best_pairs.append(pair)
                s_label = torch.stack(sims)
        else:
            s_label = torch.zeros(N, device=self.device)
            best_pairs = [("", "") for _ in range(N)]

        # ---------- CE REFINEMENT (disabled) ----------
        s_label_star = s_label

        # ---------- CONTEXT ----------
        if self.use_context:
            self._log_once("context_encoding", "Encoding contextual subgraphs...", "info")
            e_cs = self.encode_contexts_batch(src_ctx_joined)
            e_ct = self.encode_contexts_batch(tgt_ctx_joined)
            s_ctx = self._cos_sim(e_cs, e_ct)
        else:
            s_ctx = torch.zeros(N, device=self.device)

        # ---------- ADAPTIVE FUSION ----------
        sigma_label = (s_label_star - self.tau).abs()
        sigma_ctx = (s_ctx - self.tau).abs()
        denom = (sigma_ctx.pow(self.gamma) + sigma_label.pow(self.gamma)).clamp_min(1e-8)
        w_c_adaptive = (sigma_ctx.pow(self.gamma) / denom)

        if not self.use_context and self.use_lexical:
            w_c = torch.zeros_like(w_c_adaptive)
        elif not self.use_lexical and self.use_context:
            w_c = torch.ones_like(w_c_adaptive)
        elif (not self.use_lexical) and (not self.use_context):
            w_c = torch.zeros_like(w_c_adaptive)
        else:
            w_c = w_c_adaptive

        if self.use_context:
            w_c = w_c * ctx_pair_mask.to(w_c.dtype)

        S_lctx = (1.0 - w_c) * s_label_star + w_c * s_ctx

        # ---------- LLM GATING ----------
        if self.use_context:
            ctx_pair_mask_float = ctx_pair_mask.to(s_label_star.dtype)
            m = ctx_pair_mask_float * (0.5 * (s_label_star + s_ctx)) + (1.0 - ctx_pair_mask_float) * s_label_star
        else:
            m = s_label_star
        U = (2.0 * (self.tau - (m - self.tau).abs())).clamp(0.0, 1.0)

        if self.use_llm:
            w_i = (self.beta * U).clamp(0.0, 1.0)
            need_llm = (U >= self.tau_LLM)
        else:
            w_i = torch.zeros_like(U)
            need_llm = torch.zeros_like(U, dtype=torch.bool)

        # ---------- LLM (SUMMARIES + YES/NO) ----------
        src_llm_summaries = [""] * N
        tgt_llm_summaries = [""] * N
        llm_decisions = [""] * N
        llm_rationales = [""] * N
        batch_calibration_samples = 0

        decision_idxs: List[int] = []
        summary_idxs: List[int] = []
        if self.use_llm:
            decision_idxs = torch.nonzero(need_llm).flatten().tolist()
            summary_idxs = list(range(N)) if self.force_llm_summaries else list(decision_idxs)
            self._log_once(
                "first_batch_llm_counts",
                (
                    "First batch LLM gating: "
                    f"batch_size={N}, need_llm={len(decision_idxs)}, "
                    f"summary_requests={len(summary_idxs)}, "
                    f"lexical_only={N - len(decision_idxs)}."
                ),
                "debug",
            )

        if self.use_llm and summary_idxs:
            log_key = "llm_summaries_forced" if self.force_llm_summaries else "llm_triggered"
            log_msg = (
                "LLM summaries forced for all pairs (force_llm_summaries=True)."
                if self.force_llm_summaries
                else "LLM fired for ambiguous/uncertain pairs: generating summaries..."
            )
            self._log_once(log_key, log_msg, "info")

            src_best_summary = [best_pairs[i][0] for i in summary_idxs]
            tgt_best_summary = [best_pairs[i][1] for i in summary_idxs]

            src_sum = self.generate_summaries_batched(src_best_summary, [src_ctx_joined[i] for i in summary_idxs])
            tgt_sum = self.generate_summaries_batched(tgt_best_summary, [tgt_ctx_joined[i] for i in summary_idxs])
            for offset, idx in enumerate(summary_idxs):
                src_llm_summaries[idx] = src_sum[offset]
                tgt_llm_summaries[idx] = tgt_sum[offset]

        p_llm = torch.zeros(N, device=self.device)
        S_final = S_lctx

        if self.use_llm and decision_idxs:
            src_best = [best_pairs[i][0] for i in decision_idxs]
            tgt_best = [best_pairs[i][1] for i in decision_idxs]

            src_sum = [src_llm_summaries[i] for i in decision_idxs]
            tgt_sum = [tgt_llm_summaries[i] for i in decision_idxs]

            p_yes_needed = self.llm_yesno_probs_batched(src_best, tgt_best, src_sum, tgt_sum)
            if self.use_llm_calibration:
                if self._llm_calibration_can_apply:
                    p_yes_needed = self._apply_llm_calibration(p_yes_needed)
                    self._calibration_messages.append("Applied configured LLM calibration coefficients.")
                else:
                    samples = self._collect_calibration_samples(decision_idxs, p_yes_needed, label)
                    if samples is not None:
                        probs_fit, labels_fit = samples
                        count = int(probs_fit.shape[0])
                        self._calibration_pending_probs.extend(
                            probs_fit.detach().cpu().tolist()
                        )
                        self._calibration_pending_labels.extend(
                            labels_fit.detach().cpu().tolist()
                        )
                        batch_calibration_samples += count
                        self._calibration_messages.append(
                            (
                                f"Collected {count} calibration samples this batch "
                                f"(total={len(self._calibration_pending_probs)})."
                            )
                        )
            p_llm[decision_idxs] = p_yes_needed
            decisions_needed = ["Yes" if float(prob) >= 0.5 else "No" for prob in p_yes_needed]
            for offset, idx in enumerate(decision_idxs):
                llm_decisions[idx] = decisions_needed[offset]

            S_final = S_lctx.clone()
            S_final[need_llm] = (1.0 - w_i[need_llm]) * S_lctx[need_llm] + w_i[need_llm] * p_llm[need_llm]

        llm_used_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        if decision_idxs:
            llm_used_mask[decision_idxs] = True
        w_i_effective = w_i * llm_used_mask.to(w_i.dtype)

        # ---------- IMPORTANCES ----------
        I_label = (1.0 - w_c) * (1.0 - w_i_effective)
        I_ctx = w_c * (1.0 - w_i_effective)
        I_llm = w_i_effective

        # ---------- OUTPUT ----------
        result = {
            "s_label": s_label,
            "s_label_star": s_label_star,
            "s_ctx": s_ctx,
            "S_lctx": S_lctx,
            "p_llm": p_llm,
            "S_final": S_final,
            "w_c": w_c,
            "U": U,
            "w_i": w_i,
            "need_llm": need_llm,
            "I_label": I_label,
            "I_ctx": I_ctx,
            "I_llm": I_llm,
            "llm_decisions": llm_decisions,
            "llm_rationales": llm_rationales,
            "llm_calibration": self._llm_calibration_payload(batch_samples=batch_calibration_samples),
            "llm_summary_stats": self.llm_summary_stats(),
            "llm_decision_stats": self.llm_decision_stats(),
            "llm_summaries": {
                "source": src_llm_summaries,
                "target": tgt_llm_summaries,
            },
            "backend_usage": {
                "summary": dict(self._last_summary_backend_meta),
                "decision": dict(self._last_decision_backend_meta),
                "rationale": dict(self._last_rationale_backend_meta),
            },
        }

        if self.return_explanations:
            explanations = []
            for i in range(N):
                triple_attrib = []
                if self.use_context and (src_contexts[i] or tgt_contexts[i]):
                    triple_attrib = self._triple_attributions_by_sentences(
                        src_contexts[i], tgt_contexts[i], float(s_ctx[i])
                    )

                in_review_band = (self.review_low <= float(S_final[i]) <= self.review_high)
                explanations.append({
                    "src_iri": src_iris[i],
                    "tgt_iri": tgt_iris[i],
                    "models": {
                        "lexical_model": self.lexical_model_name if self.use_lexical else None,
                        "context_model": self.context_model_name if self.use_context else None,
                        "llm_model": None,
                        "llm_summary_model": self._last_summary_backend_meta.get("model") if self.use_llm else None,
                        "llm_decision_model": self._last_decision_backend_meta.get("model") if self.use_llm else None,
                        "llm_rationale_model": self._last_rationale_backend_meta.get("model") if self.use_llm else None,
                        "llm_local_fallback_model": self.llm_model_name if self.use_llm else None,
                    },
                    "llm_calibration": self._llm_calibration_payload(batch_samples=0),
                    "confidences": {
                        "s_label": float(s_label[i]),
                        "s_label_star": float(s_label_star[i]),
                        "s_ctx": float(s_ctx[i]),
                        "p_llm": float(p_llm[i]),
                        "S_lctx": float(S_lctx[i]),
                        "S_final": float(S_final[i]),
                    },
                    "weights": {
                        "w_c": float(w_c[i]),
                        "w_i": float(w_i[i]),
                        "U": float(U[i]),
                    },
                    "importances": {
                        "I_label": float(I_label[i]),
                        "I_ctx": float(I_ctx[i]),
                        "I_llm": float(I_llm[i]),
                    },
                    "review_band": {
                        "in_band": bool(in_review_band),
                        "low": self.review_low,
                        "high": self.review_high,
                    },
                    "prediction": {
                        "global_match": bool(S_final[i] >= self.threshold),
                        "ground_truth": label[i] if label is not None and i < len(label) else None,
                        "llm_decision": llm_decisions[i],
                        "llm_rationale": llm_rationales[i],
                        "threshold_positive": bool(S_final[i] >= self.threshold),
                        "saved_alignment_member": False,
                        "rationale_decision_label": "",
                    },
                    "selected_labels": {
                        "source": best_pairs[i][0],
                        "target": best_pairs[i][1],
                    },
                    "backend_usage": {
                        "summary": dict(self._last_summary_backend_meta),
                        "decision": dict(self._last_decision_backend_meta),
                        "rationale": dict(self._last_rationale_backend_meta),
                    },
                    "context_sentences": {
                        "source": list(src_contexts[i]),
                        "target": list(tgt_contexts[i]),
                        "delimiter": self.ctx_sentence_delimiter,
                    },
                    "context_triples": {
                        "source": [list(t) for t in (src_ctx_raw[i] or [])],
                        "target": [list(t) for t in (tgt_ctx_raw[i] or [])],
                        "note": "Context triples used by the model; connectors omitted.",
                    },
                    "connectivity_bridges": {
                        "source": [
                            {
                                "triple": list(t),
                                "used_by_model": False,
                                "info_score": 0.0,
                                "verbalized": False,
                                "reason": "connectivity_bridge",
                            }
                            for t in (src_ctx_bridges[i] or [])
                        ],
                        "target": [
                            {
                                "triple": list(t),
                                "used_by_model": False,
                                "info_score": 0.0,
                                "verbalized": False,
                                "reason": "connectivity_bridge",
                            }
                            for t in (tgt_ctx_bridges[i] or [])
                        ],
                    },
                    "llm_summaries": {
                        "source": src_llm_summaries[i],
                        "target": tgt_llm_summaries[i],
                    },
                    "triple_attributions": triple_attrib,
                })
            result["explanations"] = explanations

        return result
