import math
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Iterable
import re
import time
import json
import hashlib
from pathlib import Path
from collections import OrderedDict

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

from exact.core.contracts.model import IModel


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
      • Uncertainty-driven, gated LLM (summarise + Yes/No logits probability)
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
        self.llm_summary_batch_size = llm_summary_batch_size
        self.llm_decision_batch_size = llm_decision_batch_size
        self.llm_rationale_batch_size = llm_rationale_batch_size

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

        # store model names (for explanations)
        self.lexical_model_name = lexical_model_name
        self.context_model_name = context_model_name
        self.llm_model_name = llm_model_name
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
            self.log("Loading LLM...", "info")
            self.llm_tok = AutoTokenizer.from_pretrained(llm_model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                torch_dtype=torch.float16 if self.fp16 else torch.float32
            ).to(self.device)
            self.llm_temperature = llm_temperature
            self.llm_top_p = llm_top_p
            self.llm_do_sample = llm_do_sample
            self.yes_token_ids = self._candidate_token_ids([" Yes", "Yes", "yes"])
            self.no_token_ids = self._candidate_token_ids([" No", "No", "no"])

        if self.persist_cache_to_disk:
            self._load_cache_from_disk()
        self.reset_llm_calibration_tracking()

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
    def _build_cache_fingerprint(self) -> str:
        payload = {
            "lexical_model": self.lexical_model_name if self.use_lexical else None,
            "context_model": self.context_model_name if self.use_context else None,
            "llm_model": self.llm_model_name if self.use_llm else None,
            "max_input_tokens_lexical": self.max_input_tokens_lexical,
            "max_input_tokens_context": self.max_input_tokens_context,
            "max_total_tokens_llm_summary": self.max_total_tokens_llm_summary,
            "max_total_tokens_llm_decision": self.max_total_tokens_llm_decision,
            "max_total_tokens_llm_rationale": self.max_total_tokens_llm_rationale,
            "pooling_method": self.pooling_method.value,
            "label_pair_pooling": self.label_pair_pooling.value,
            "ctx_sentence_delimiter": self.ctx_sentence_delimiter,
            "dataset_signature": self.dataset_signature,
            "generate_llm_rationales": self.generate_llm_rationales,
            "use_llm_calibration": self.use_llm_calibration,
            "llm_calibration_a": self.llm_calibration_a,
            "llm_calibration_b": self.llm_calibration_b,
            "llm_calibration_info": self.llm_calibration_info,
            "force_llm_summaries": self.force_llm_summaries,
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

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
    ) -> str:
        sep = "\u241F"
        payload = sep.join([
            src_label or "",
            tgt_label or "",
            src_summary or "",
            tgt_summary or "",
            decision or "",
        ])
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
        self.log(
            (
                f"Loaded SemanticScorer cache from {self._cache_file_path} "
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
            self.log(
                (
                    f"Persisted SemanticScorer cache to {self._cache_file_path} "
                    f"(reason={reason})."
                ),
                "debug",
            )
            self._cache_dirty = False
        except OSError as exc:
            self.log(f"Failed to persist SemanticScorer cache to {self._cache_file_path}: {exc}", "warning")

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
            "system": "You are an ontology expert entity summariser.",
            "user": (
                f"Given the following context subgraph describing the entity '{label}', "
                "provide a concise and informative summary capturing its key characteristics.\n\n"
                f"Context: {ctx}\nSummary:"
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
    ) -> Dict[str, str]:
        return {
            "system": "You are an ontology alignment expert.",
            "user": (
                "Based only on the information below, write one or two sentences explaining "
                "why the two entities should or should not be considered equivalent.\n"
                "Reference specific evidence from the label pair and the summaries. "
                "Do not introduce external knowledge.\n\n"
                f"Source\nLabel: {src_label}\nSummary: {src_summary}\n\n"
                f"Target\nLabel: {tgt_label}\nSummary: {tgt_summary}\n\n"
                f"Model decision: {decision}\n\n"
                "Rationale:"
            ),
        }

    def _render_llm_prompt(self, prompt: Dict[str, str], add_generation_prompt: bool = True) -> str:
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

    @staticmethod
    def _clean_summary_text(text: Optional[str]) -> str:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        txt = text.strip()
        if not txt:
            return ""
        if "Summary:" in txt:
            txt = txt.split("Summary:", 1)[-1].strip()
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if not lines:
            return ""
        line = lines[0]
        prefix_pattern = re.compile(r"^(assistant|assistant:|assistant,|summary:)", re.IGNORECASE)
        line = prefix_pattern.sub("", line).strip(" :,-\t")
        sentence_parts = re.split(r"(?<=[.!?])\s+", line)
        cleaned = sentence_parts[0].strip()
        return cleaned or line

    @staticmethod
    def _clean_rationale_text(text: str) -> str:
        txt = text.strip()
        if not txt:
            return ""
        prefix_pattern = re.compile(r"^(assistant|rationale:|assistant:)", re.IGNORECASE)
        txt = prefix_pattern.sub("", txt).strip()
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if lines:
            txt = " ".join(lines)
        sentences = re.split(r"(?<=[.!?])\s+", txt)
        trimmed = " ".join(sentences[:2]).strip()
        return trimmed or txt

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
            generated = self._generate_summaries_uncached(labs, ctxs)
            for key, summary in zip(pending_keys, generated):
                clean = self._clean_summary_text(summary)
                self._cache_store(self._summary_cache, key, clean, self.max_cached_summaries)
                for idx in pending[key]["indices"]:
                    outputs[idx] = clean

        self._record_summary_stats(outputs)
        return outputs

    def _generate_summaries_uncached(self, labels: List[str], contexts: List[str]) -> List[str]:
        if not labels:
            return []
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
    ) -> List[str]:
        if not (self.use_llm and self.generate_llm_rationales):
            return ["" for _ in src_labels]
        if not src_labels:
            return []
        outputs = [""] * len(src_labels)
        pending: Dict[str, Dict[str, Any]] = {}
        for idx, (s_lab, t_lab, s_sum, t_sum, decision) in enumerate(
            zip(src_labels, tgt_labels, src_summaries, tgt_summaries, decisions)
        ):
            key = self._rationale_key(s_lab, t_lab, s_sum, t_sum, decision)
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
                    "indices": [],
                },
            )
            entry["indices"].append(idx)

        if pending:
            pending_keys = list(pending.keys())
            prompts = [
                self._render_llm_prompt(
                    self._rationale_prompt(
                        pending[key]["src_label"],
                        pending[key]["tgt_label"],
                        pending[key]["src_summary"],
                        pending[key]["tgt_summary"],
                        pending[key]["decision"],
                    )
                )
                for key in pending_keys
            ]
            generated = self._generate_rationales_uncached(prompts)
            for key, rationale in zip(pending_keys, generated):
                clean = self._clean_rationale_text(rationale)
                self._cache_store(self._rationale_cache, key, clean, self.max_cached_rationales)
                for idx in pending[key]["indices"]:
                    outputs[idx] = clean

        return outputs

    def _generate_rationales_uncached(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []
        outputs = [""] * len(prompts)
        chunk = self.llm_rationale_batch_size or len(prompts)
        chunk = chunk if chunk > 0 else len(prompts)
        for start in range(0, len(prompts), chunk):
            end = min(start + chunk, len(prompts))
            enc = self.llm_tok(
                prompts[start:end],
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_total_tokens_llm_rationale,
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

            if self.generate_llm_rationales:
                rationales_needed = self.generate_rationales_batched(
                    src_best,
                    tgt_best,
                    src_sum,
                    tgt_sum,
                    decisions_needed,
                )
                for offset, idx in enumerate(decision_idxs):
                    llm_rationales[idx] = rationales_needed[offset]

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
            "llm_summaries": {
                "source": src_llm_summaries,
                "target": tgt_llm_summaries,
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
                        "llm_model": self.llm_model_name if self.use_llm else None,
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
                        "ground_truth": label[i],
                        "llm_decision": llm_decisions[i],
                        "llm_rationale": llm_rationales[i],

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
