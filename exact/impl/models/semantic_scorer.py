import math
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import re
import time

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

from exact.core.contracts.model import IModel


class PoolingMethod(str, Enum):
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"


class SemanticScorer(IModel):
    """
    Explainable ontology matching scorer integrating:
      • Lexical embeddings with max-pooling over label variants
      • Ambiguity-triggered cross-encoder refinement (BGE reranker)
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
        cross_encoder_name: Optional[str] = "BAAI/bge-reranker-large",
        llm_model_name: Optional[str] = "Qwen/Qwen2.5-7B-Instruct",

        # ---- Precision / device ----
        fp16_inference: bool = True,
        device: Optional[str] = None,

        # ---- Pooling for encoders ----
        pooling_method: PoolingMethod = PoolingMethod.MEAN,

        # ---- Context sentence delimiter ----
        ctx_sentence_delimiter: Optional[str] = " || ",

        # ---- Token limits ----
        max_input_tokens_lexical: int = 32,
        max_input_tokens_context: int = 256,
        max_input_tokens_ce: int = 64,
        max_total_tokens_llm_summary: int = 512,
        max_total_tokens_llm_decision: int = 384,
        max_new_tokens_llm: int = 64,

        # ---- Ablations / toggles ----
        use_lexical: bool = True,
        use_context: bool = True,
        use_cross_encoder: bool = True,
        use_llm: bool = True,

        # ---- Cross-encoder ambiguity gating ----
        tau: float = 0.5,
        tau_ce: float = 0.3,
        eta: float = 2.0,

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

        # ---- Explanations / review band ----
        return_explanations: bool = False,
        review_low: float = 0.35,
        review_high: float = 0.75,
        threshold: float = 0.7,

        **kwargs,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = (
            self.device.type if isinstance(self.device, torch.device) else torch.device(self.device).type
        )
        self.fp16 = fp16_inference
        self.pooling_method = PoolingMethod(pooling_method)

        # attach logger hook
        self.log = getattr(self, "log", lambda *a, **kw: None)

        # store params
        self.max_input_tokens_lexical = max_input_tokens_lexical
        self.max_input_tokens_context = max_input_tokens_context
        self.max_input_tokens_ce = max_input_tokens_ce
        self.max_total_tokens_llm_summary = max_total_tokens_llm_summary
        self.max_total_tokens_llm_decision = max_total_tokens_llm_decision
        self.max_new_tokens_llm = max_new_tokens_llm
        self.llm_summary_batch_size = llm_summary_batch_size
        self.llm_decision_batch_size = llm_decision_batch_size

        self.use_lexical = use_lexical
        self.use_context = use_context
        self.use_cross_encoder = use_cross_encoder and (cross_encoder_name is not None)
        self.use_llm = use_llm and (llm_model_name is not None)

        self.tau = tau
        self.tau_ce = tau_ce
        self.eta = eta
        self.gamma = gamma
        self.beta = beta
        self.tau_LLM = tau_LLM

        self.return_explanations = return_explanations
        self.review_low = review_low
        self.review_high = review_high
        self.threshold = threshold
        self.ctx_sentence_delimiter = ctx_sentence_delimiter

        # store model names (for explanations)
        self.lexical_model_name = lexical_model_name
        self.context_model_name = context_model_name
        self.cross_encoder_name = cross_encoder_name
        self.llm_model_name = llm_model_name

        # ---- Load models ----
        if self.use_lexical:
            self.log("Loading lexical encoder...", "info")
            self.lex_tok = AutoTokenizer.from_pretrained(lexical_model_name)
            self.lex_model = AutoModel.from_pretrained(lexical_model_name).to(self.device)

        if self.use_context:
            self.log("Loading context encoder...", "info")
            self.ctx_tok = AutoTokenizer.from_pretrained(context_model_name)
            self.ctx_model = AutoModel.from_pretrained(context_model_name).to(self.device)

        if self.use_cross_encoder:
            self.log("Loading cross-encoder...", "info")
            self.ce_tok = AutoTokenizer.from_pretrained(cross_encoder_name)
            self.ce_model = AutoModelForSequenceClassification.from_pretrained(
                cross_encoder_name
            ).to(self.device)

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
        return self._encode_texts(self.lex_tok, self.lex_model, label_texts, self.max_input_tokens_lexical)

    @torch.inference_mode()
    def encode_contexts_batch(self, ctx_texts: List[str]) -> torch.Tensor:
        return self._encode_texts(self.ctx_tok, self.ctx_model, ctx_texts, self.max_input_tokens_context)
    
    @torch.inference_mode()
    def cross_encode_pairs(self, pairs: List[Tuple[str, str]]) -> torch.Tensor:
        """
        Runs the cross-encoder (BGE reranker) on a list of (src_label, tgt_label) pairs
        and returns sigmoid-normalised relevance scores in [0, 1].
        """
        if not self.use_cross_encoder or not pairs:
            return torch.zeros(len(pairs) if pairs else 0, device=self.device)

        batch = self.ce_tok(
            [a for a, _ in pairs], [b for _, b in pairs],
            padding=True, truncation=True, max_length=self.max_input_tokens_ce,
            return_tensors="pt"
        ).to(self.device)

        with torch.cuda.amp.autocast(enabled=self.fp16):
            logits = self.ce_model(**batch).logits.squeeze(-1)  # [B]
        return torch.sigmoid(logits)


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

    @staticmethod
    def _clean_summary_text(text: str) -> str:
        txt = text.strip()
        if not txt:
            return ""
        if "Summary:" in txt:
            txt = txt.split("Summary:", 1)[-1].strip()
        line = txt.splitlines()[0].strip()
        prefix_pattern = re.compile(r"^(assistant|assistant:|assistant,|summary:)", re.IGNORECASE)
        line = prefix_pattern.sub("", line).strip(" :,-\t")
        sentence_parts = re.split(r"(?<=[.!?])\s+", line)
        cleaned = sentence_parts[0].strip()
        return cleaned or line

    @torch.inference_mode()
    def generate_summaries_batched(self, labels: List[str], contexts: List[str]) -> List[str]:
        if not self.use_llm:
            return ["" for _ in labels]
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
                outputs[start + offset] = self._clean_summary_text(text)
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
        label: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        N = len(src_label_lists)
        assert len(tgt_label_lists) == N
        assert len(src_iris) == N and len(tgt_iris) == N

        # Join context triples
        src_contexts = src_contexts or [[] for _ in range(N)]
        tgt_contexts = tgt_contexts or [[] for _ in range(N)]
        src_ctx_joined = [self._join_context(c) for c in src_contexts]
        tgt_ctx_joined = [self._join_context(c) for c in tgt_contexts]

        # ---------- LEXICAL + MAX POOL ----------
        if self.use_lexical:
            self.log("Encoding lexical label variants...", "info")
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
                    v, idx = mat.max(dim=0)[0].max(dim=0)
                    idx_flat = torch.argmax(mat)
                    r = idx_flat // mat.shape[1]
                    c = idx_flat % mat.shape[1]
                    sims.append(mat[r, c])
                    best_pairs.append((labs_s[int(r)], labs_t[int(c)]))
                s_label = torch.stack(sims)
        else:
            s_label = torch.zeros(N, device=self.device)
            best_pairs = [("", "") for _ in range(N)]

        # ---------- CE REFINEMENT ----------
        if self.use_cross_encoder and self.use_lexical:
            self.log("Cross-encoder refinement on ambiguous label pairs...", "info")
            A_label = 1.0 - (s_label - self.tau).abs()
            need_ce = (A_label >= self.tau_ce)
            s_label_star = s_label.clone()

            pairs, pair_idx = [], []
            for i, (labs_s, labs_t) in enumerate(zip(src_label_lists, tgt_label_lists)):
                if not need_ce[i] or not labs_s or not labs_t:
                    continue
                Es = self.encode_labels_batch(labs_s)
                Et = self.encode_labels_batch(labs_t)
                Esn = torch.nn.functional.normalize(Es, dim=-1)
                Etn = torch.nn.functional.normalize(Et, dim=-1)
                mat = Esn @ Etn.T
                idx_max = torch.argmax(mat)
                r = idx_max // mat.shape[1]
                c = idx_max % mat.shape[1]
                pairs.append((labs_s[int(r)], labs_t[int(c)]))
                pair_idx.append(i)
                best_pairs[i] = (labs_s[int(r)], labs_t[int(c)])  # ensure consistency for LLM

            if pairs:
                ce_scores = self.cross_encode_pairs(pairs)
                for j, i in enumerate(pair_idx):
                    alpha = (A_label[i] ** self.eta).clamp(0, 1)
                    s_label_star[i] = (1 - alpha) * s_label[i] + alpha * ce_scores[j]
        else:
            s_label_star = s_label

        # ---------- CONTEXT ----------
        if self.use_context:
            self.log("Encoding contextual subgraphs...", "info")
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

        S_lctx = (1.0 - w_c) * s_label_star + w_c * s_ctx

        # ---------- LLM GATING ----------
        m = 0.5 * (s_label_star + s_ctx)
        U = (2.0 * (self.tau - (m - self.tau).abs())).clamp(0.0, 1.0)

        if self.use_llm:
            w_i = (self.beta * U).clamp(0.0, 1.0)
            need_llm = (U >= self.tau_LLM)
        else:
            w_i = torch.zeros_like(U)
            need_llm = torch.zeros_like(U, dtype=torch.bool)

        # ---------- LLM (SUMMARIES + YES/NO) ----------
        if self.use_llm and need_llm.any():
            self.log("LLM fired for ambiguous/uncertain pairs: generating summaries...", "info")

            idxs = torch.nonzero(need_llm).flatten().tolist()
            src_best = [best_pairs[i][0] for i in idxs]
            tgt_best = [best_pairs[i][1] for i in idxs]

            src_sum = self.generate_summaries_batched(src_best, [src_ctx_joined[i] for i in idxs])
            tgt_sum = self.generate_summaries_batched(tgt_best, [tgt_ctx_joined[i] for i in idxs])

            p_yes_needed = self.llm_yesno_probs_batched(src_best, tgt_best, src_sum, tgt_sum)
            p_llm = torch.zeros(N, device=self.device)
            p_llm[idxs] = p_yes_needed

            S_final = S_lctx.clone()
            S_final[need_llm] = (1.0 - w_i[need_llm]) * S_lctx[need_llm] + w_i[need_llm] * p_llm[need_llm]
        else:
            p_llm = torch.zeros(N, device=self.device)
            S_final = S_lctx

        # ---------- IMPORTANCES ----------
        I_label = (1.0 - w_c) * (1.0 - w_i)
        I_ctx = w_c * (1.0 - w_i)
        I_llm = w_i

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
                        "cross_encoder_model": self.cross_encoder_name if self.use_cross_encoder else None,
                        "llm_model": self.llm_model_name if self.use_llm else None,
                    },
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
                        "ground_truth": label,

                    },
                    "context_sentences": {
                        "source": list(src_contexts[i]),
                        "target": list(tgt_contexts[i]),
                        "delimiter": self.ctx_sentence_delimiter,
                    },
                    "triple_attributions": triple_attrib,
                })
            result["explanations"] = explanations

        return result
