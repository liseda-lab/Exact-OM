from __future__ import annotations

import hashlib  # noqa: F401
import json  # noqa: F401
import re  # noqa: F401
import time  # noqa: F401
from collections import OrderedDict  # noqa: F401
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: F401
from enum import Enum
from pathlib import Path  # noqa: F401
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple  # noqa: F401
from urllib import error as urlerror  # noqa: F401

import torch  # noqa: F401
from torch import nn  # noqa: F401
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: F401

from exact.llm.routing import (  # noqa: F401
    LLMRouter,
    extract_chat_text,
    extract_first_token_top_logprobs,
    parse_structured_json,
)
from exact.utils.data import read_table  # noqa: F401


class PoolingMethod(str, Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"


class LabelPairPooling(str, Enum):
    MAX = "max"
    MEAN = "mean"


class ScorerCommonMixin:
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
            self.llm_model_name, torch_dtype=torch.float16 if self.fp16 else torch.float32
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
        return (
            self.ctx_sentence_delimiter.join(triples)
            if self.ctx_sentence_delimiter
            else " ".join(triples)
        )

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
            "llm_calibration_reference": {
                "sha1": self._llm_calibration_reference_fingerprint,
                "pairs": len(self._llm_calibration_reference_pairs),
                "sources": len(self._llm_calibration_reference_sources),
            },
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
        payload = (label or "") + "\u241f" + (context or "")
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
        sep = "\u241f"
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
            "lexical": [
                (k, v.cpu().to(self._cache_tensor_dtype)) for k, v in self._lex_cache.items()
            ],
            "context": [
                (k, v.cpu().to(self._cache_tensor_dtype)) for k, v in self._ctx_cache.items()
            ],
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
            self.log(
                f"Failed to load SemanticScorer cache at {self._cache_file_path}: {exc}", "warning"
            )
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
                (f"Persisted {model_name} cache to {self._cache_file_path} " f"(reason={reason})."),
                "debug",
            )
            self._cache_dirty = False
        except OSError as exc:
            model_name = self.__class__.__name__
            self.log(
                f"Failed to persist {model_name} cache to {self._cache_file_path}: {exc}", "warning"
            )

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

    def _summary_prompt(self, label: str, ctx: str) -> Dict[str, str]:
        return {
            "system": "You are an ontology expert entity summariser that returns strict JSON.",
            "user": (
                f"Given the following context subgraph describing the entity '{label}', "
                "provide one concise and informative summary capturing its key characteristics.\n\n"
                f"Context: {ctx}\n\n"
                'Return exactly one JSON object with one key: "summary".'
            ),
        }

    def _decision_prompt(
        self, src_label: str, tgt_label: str, src_summary: str, tgt_summary: str
    ) -> Dict[str, str]:
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
            context_block = "\nFinal alignment context\n" f"{decision_context}\n"
        return {
            "system": "You are an ontology alignment expert.",
            "user": (
                "Based only on the information below, write two to four sentences explaining "
                "why the two entities should or should not be considered equivalent.\n"
                "Reference specific evidence from the label pair and the summaries. "
                "When final alignment context is present, use it to explain whether the pair "
                "was kept or rejected after candidate-set selection and cardinality filtering. "
                "Do not introduce external knowledge. Return exactly one JSON object with one key: "
                '"rationale".\n\n'
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

    def _strip_llm_prompt_tokens(
        self, enc: Dict[str, torch.Tensor], gen: torch.Tensor
    ) -> List[torch.Tensor]:
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
            "control": sum(
                1 for ch in value if ((ord(ch) < 32 and ch not in "\n\r\t") or ord(ch) == 127)
            ),
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
