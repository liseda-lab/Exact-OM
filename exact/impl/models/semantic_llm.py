from __future__ import annotations

import hashlib  # noqa: F401
import json  # noqa: F401
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: F401
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: F401
from urllib import error as urlerror  # noqa: F401

import torch  # noqa: F401
from torch import nn  # noqa: F401
from transformers import AutoTokenizer  # noqa: F401

from exact.llm.routing import extract_chat_text, extract_first_token_top_logprobs


class SemanticLLMMixin:
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
        src_iris: List[str],
        tgt_iris: List[str],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        xs: List[float] = []
        ys: List[float] = []
        for offset, idx in enumerate(idxs):
            if idx >= len(src_iris) or idx >= len(tgt_iris):
                continue
            src = str(src_iris[idx])
            tgt = str(tgt_iris[idx])
            if src not in self._llm_calibration_reference_sources:
                continue
            xs.append(float(probs[offset].item()))
            ys.append(float((src, tgt) in self._llm_calibration_reference_pairs))
        if len(xs) < 2:
            self._calibration_messages.append(
                f"LLM calibration skipped: insufficient training-reference pairs ({len(xs)})."
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

    def _generate_summaries_uncached(
        self, labels: List[str], contexts: List[str], resolved_backend
    ) -> List[str]:
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
        prompts = [
            self._render_llm_prompt(self._summary_prompt(label_text, context))
            for label_text, context in zip(labels, contexts)
        ]
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
                clean = self._parse_structured_text(
                    raw_text, "rationale", self._clean_rationale_text
                )
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
                clean = self._parse_structured_text(
                    rationale, "rationale", self._clean_rationale_text
                )
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
                    executor.submit(_call, prompt): idx for idx, prompt in enumerate(prompts)
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
                self._last_decision_backend_meta["decision_probe_passed"] = bool(
                    probe.get("passed")
                )
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
                                if probe.get("provider")
                                else "."
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

                        def _score_prompt(
                            index_prompt: Tuple[int, Dict[str, str]],
                        ) -> Dict[str, Any]:
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
                            payloads = [
                                _score_prompt((idx, prompt)) for idx, prompt in enumerate(prompts)
                            ]
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
                        self.log(
                            f"Hosted decision backend failed; falling back to local LLM. Error: {exc}",
                            "warning",
                        )
                        if "payloads" in locals() and payloads:
                            self._record_hosted_decision_chat_debug(
                                profile.name, payloads[0], str(exc)
                            )
                        self._last_decision_backend_meta["fallback_triggered"] = True
                        self._last_decision_backend_meta["fallback_reason"] = (
                            "decision_scoring_failed"
                        )
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
            for s_lab, t_lab, s_sum, t_sum in zip(
                src_labels, tgt_labels, src_summaries, tgt_summaries
            )
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
            lines.append(
                f"Kept in saved alignment after cardinality filtering: {_yes_no(pred.get('saved_alignment_member'))}"
            )
        return "\n".join(lines)

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
            mod_src = self._join_context(src_triples[:i] + src_triples[i + 1 :])
            mod_tgt = self._join_context(tgt_triples)
            s_minus = self._context_similarity_texts(mod_src, mod_tgt)
            drops.append(max(0.0, s_ctx_orig - s_minus))
            tags.append(("source", i, s))
        for j, s in enumerate(tgt_triples):
            mod_src = self._join_context(src_triples)
            mod_tgt = self._join_context(tgt_triples[:j] + tgt_triples[j + 1 :])
            s_minus = self._context_similarity_texts(mod_src, mod_tgt)
            drops.append(max(0.0, s_ctx_orig - s_minus))
            tags.append(("target", j, s))
        denom = sum(abs(d) for d in drops) or 1.0
        imps = [d / denom for d in drops]
        return [
            {"side": side, "index": idx, "sentence": sent, "importance": float(imp)}
            for (side, idx, sent), imp in zip(tags, imps)
        ]

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
                idx = 0
                src_emb_slices = []
                for labs in src_label_lists:
                    src_emb_slices.append(e_src[idx : idx + len(labs)] if len(labs) else e_src[0:0])
                    idx += len(labs)
                idx = 0
                tgt_emb_slices = []
                for labs in tgt_label_lists:
                    tgt_emb_slices.append(e_tgt[idx : idx + len(labs)] if len(labs) else e_tgt[0:0])
                    idx += len(labs)

                sims, best_pairs = [], []
                for labs_s, labs_t, Es, Et in zip(
                    src_label_lists, tgt_label_lists, src_emb_slices, tgt_emb_slices
                ):
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
        w_c_adaptive = sigma_ctx.pow(self.gamma) / denom

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
            m = (
                ctx_pair_mask_float * (0.5 * (s_label_star + s_ctx))
                + (1.0 - ctx_pair_mask_float) * s_label_star
            )
        else:
            m = s_label_star
        U = (2.0 * (self.tau - (m - self.tau).abs())).clamp(0.0, 1.0)

        if self.use_llm:
            w_i = (self.beta * U).clamp(0.0, 1.0)
            need_llm = U >= self.tau_LLM
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

            src_sum = self.generate_summaries_batched(
                src_best_summary, [src_ctx_joined[i] for i in summary_idxs]
            )
            tgt_sum = self.generate_summaries_batched(
                tgt_best_summary, [tgt_ctx_joined[i] for i in summary_idxs]
            )
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
                    self._calibration_messages.append(
                        "Applied configured LLM calibration coefficients."
                    )
                else:
                    samples = self._collect_calibration_samples(
                        decision_idxs, p_yes_needed, src_iris, tgt_iris
                    )
                    if samples is not None:
                        probs_fit, labels_fit = samples
                        count = int(probs_fit.shape[0])
                        self._calibration_pending_probs.extend(probs_fit.detach().cpu().tolist())
                        self._calibration_pending_labels.extend(labels_fit.detach().cpu().tolist())
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
            S_final[need_llm] = (1.0 - w_i[need_llm]) * S_lctx[need_llm] + w_i[need_llm] * p_llm[
                need_llm
            ]

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
            "llm_calibration": self._llm_calibration_payload(
                batch_samples=batch_calibration_samples
            ),
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

                in_review_band = self.review_low <= float(S_final[i]) <= self.review_high
                explanations.append(
                    {
                        "src_iri": src_iris[i],
                        "tgt_iri": tgt_iris[i],
                        "models": {
                            "lexical_model": self.lexical_model_name if self.use_lexical else None,
                            "context_model": self.context_model_name if self.use_context else None,
                            "llm_model": None,
                            "llm_summary_model": (
                                self._last_summary_backend_meta.get("model")
                                if self.use_llm
                                else None
                            ),
                            "llm_decision_model": (
                                self._last_decision_backend_meta.get("model")
                                if self.use_llm
                                else None
                            ),
                            "llm_rationale_model": (
                                self._last_rationale_backend_meta.get("model")
                                if self.use_llm
                                else None
                            ),
                            "llm_local_fallback_model": (
                                self.llm_model_name if self.use_llm else None
                            ),
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
                            "ground_truth": (
                                label[i] if label is not None and i < len(label) else None
                            ),
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
                    }
                )
            result["explanations"] = explanations

        return result
