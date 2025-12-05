import difflib
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from exact.core.contracts.model import IModel


class SecondPassReranker(IModel):
    """
    Lightweight reranker that re-scores ambiguous sources after the primary model.
    The module is configured via its own params and can be chained as a second model.
    """

    def __init__(
        self,
        enabled: bool = False,
        top_k: int = 5,
        epsilon: float = 0.03,
        min_ties: int = 3,
        ce_model_name: Optional[str] = None,
        ce_weight: float = 0.5,
        use_symbolic: bool = True,
        symbolic_weight: float = 0.05,
        use_llm: bool = False,
        llm_trigger_epsilon: float = 0.01,
        max_llm_prompts: int = 0,
        max_prompt_candidates: int = 5,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.enabled = enabled
        self.top_k = max(1, int(top_k))
        self.epsilon = float(epsilon)
        self.min_ties = max(2, int(min_ties))
        self.ce_model_name = ce_model_name
        self.ce_weight = float(ce_weight)
        self.use_symbolic = bool(use_symbolic)
        self.symbolic_weight = float(symbolic_weight)
        self.use_llm = bool(use_llm)
        self.llm_trigger_epsilon = float(llm_trigger_epsilon)
        self.max_llm_prompts = int(max_llm_prompts)
        self.max_prompt_candidates = max(1, int(max_prompt_candidates))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._ce_cache: Optional[Tuple[Any, Any]] = None
        self._ce_name: Optional[str] = None
        self._llm_prompts_used = 0

    def _log(self, logger: Optional[Any], msg: str, level: str = "info") -> None:
        if logger is None:
            print(msg)
            return
        log_method = getattr(logger, level, logger.info)
        log_method(msg)

    def forward(
        self,
        candidate_df: pd.DataFrame,
        primary_model: Optional[IModel] = None,
        dataset: Any = None,
        logger: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Runs the second-pass reranking on ambiguous sources.
        Returns a dict with the updated dataframe under 'candidate_df'.
        """
        if not self.enabled or candidate_df.empty:
            return {"candidate_df": candidate_df}

        self._llm_prompts_used = 0
        df = candidate_df.copy()
        flagged: List[Tuple[str, List[int]]] = []
        for src, group in df.groupby("Src", sort=False):
            sub = group.sort_values("S_final", ascending=False).head(self.top_k)
            if self._is_ambiguous(sub["S_final"].tolist()):
                flagged.append((src, sub.index.tolist()))

        if not flagged:
            self._log(logger, "Second pass rerank skipped (no ambiguous sources).", "info")
            return {"candidate_df": df}

        total_sources = df["Src"].nunique()
        start = time.perf_counter()

        ce_time = 0.0
        ce_updates_count = 0
        if self.ce_model_name:
            t_ce = time.perf_counter()
            ce_updates = self._second_pass_ce_scores(flagged, df)
            ce_time = time.perf_counter() - t_ce
            if ce_updates:
                for idx, new_label in ce_updates.items():
                    df.at[idx, "s_label"] = new_label
                df = self._recompute_scores(df)
                ce_updates_count = len(ce_updates)

        symbolic_time = 0.0
        if self.use_symbolic:
            t_sym = time.perf_counter()
            for src, indices in flagged:
                sub = df.loc[indices].sort_values("S_final", ascending=False)
                if not self._is_ambiguous(sub["S_final"].tolist()):
                    continue
                sims = sub.apply(self._symbolic_similarity, axis=1)
                df.loc[sub.index, "s_label"] = sub["s_label"] + self.symbolic_weight * sims
            df = self._recompute_scores(df)
            symbolic_time = time.perf_counter() - t_sym

        llm_time = 0.0
        llm_used = 0
        if self.use_llm and self._allow_more_llm():
            candidates_for_llm: List[Tuple[str, List[int]]] = []
            for src, indices in flagged:
                if self.max_llm_prompts > 0 and len(candidates_for_llm) >= self._remaining_llm_budget():
                    break
                sub = df.loc[indices].sort_values("S_final", ascending=False)
                if self._is_ambiguous(sub["S_final"].tolist(), epsilon=self.llm_trigger_epsilon):
                    candidates_for_llm.append((src, sub.index.tolist()))
            if candidates_for_llm and primary_model is not None:
                t_llm = time.perf_counter()
                llm_choices = self._llm_batch_choose_candidate(
                    candidates_for_llm,
                    df,
                    primary_model=primary_model,
                )
                llm_time = time.perf_counter() - t_llm
                idx_lookup = {src: idxs for src, idxs in candidates_for_llm}
                for src, choice_idx in llm_choices.items():
                    if choice_idx is None:
                        continue
                    rows = idx_lookup.get(src)
                    if not rows or choice_idx >= len(rows):
                        continue
                    row_idx = rows[choice_idx]
                    df.at[row_idx, "s_label"] = df.loc[row_idx, "s_label"] + self.epsilon
                    llm_used += 1
                if llm_choices:
                    df = self._recompute_scores(df)
                    self._llm_prompts_used += len([c for c in llm_choices.values() if c is not None])

        elapsed = time.perf_counter() - start
        self._log(
            logger,
            (
                f"Second pass rerank: {len(flagged)} ambiguous sources of {total_sources}. "
                f"Time {elapsed:.2f}s (CE {ce_time:.2f}s/{ce_updates_count}, "
                f"symbolic {symbolic_time:.2f}s, LLM {llm_time:.2f}s/{llm_used})."
            ),
            "info",
        )
        return {"candidate_df": df}

    def _recompute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        df["S_lctx"] = (1.0 - df["w_c"]) * df["s_label"] + df["w_c"] * df["s_ctx"]
        df["S_final"] = (1.0 - df["w_i"]) * df["S_lctx"] + df["w_i"] * df["p_llm"]
        return df

    def _is_ambiguous(self, scores: List[float], epsilon: Optional[float] = None) -> bool:
        eps = self.epsilon if epsilon is None else epsilon
        if len(scores) < self.min_ties:
            return False
        top = scores[0]
        kth = scores[min(len(scores), self.top_k) - 1]
        if top - kth > eps:
            return False
        ties = sum(1 for s in scores if s >= top - eps)
        return ties >= self.min_ties

    def _second_pass_ce_scores(
        self,
        flagged: List[Tuple[str, List[int]]],
        df: pd.DataFrame,
    ) -> Dict[int, float]:
        if not self.ce_model_name:
            return {}
        tok, model = self._load_second_pass_ce_model()
        if tok is None or model is None:
            return {}
        tasks: List[Tuple[int, str, str]] = []
        for _, indices in flagged:
            sub = df.loc[indices].sort_values("S_final", ascending=False).head(self.top_k)
            for idx, row in sub.iterrows():
                src_txt = row.get("src_label_text") or row.get("Src") or ""
                tgt_txt = row.get("tgt_label_text") or row.get("Tgt") or ""
                tasks.append((idx, src_txt, tgt_txt))
        if not tasks:
            return {}
        results: Dict[int, float] = {}
        model.eval()
        batch_size = 64
        for i in range(0, len(tasks), batch_size):
            chunk = tasks[i : i + batch_size]
            src_batch = [f"(source) {s}" for _, s, _ in chunk]
            tgt_batch = [f"(target) {t}" for _, _, t in chunk]
            inputs = tok(
                src_batch,
                tgt_batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = model(**inputs).logits.squeeze(-1)
            scores = torch.sigmoid(logits).detach().cpu().tolist()
            for (idx, _, _), score in zip(chunk, scores):
                base = df.at[idx, "s_label"]
                results[idx] = (1.0 - self.ce_weight) * base + self.ce_weight * float(score)
        return results

    def _load_second_pass_ce_model(self):
        if self._ce_cache and self._ce_name == self.ce_model_name:
            return self._ce_cache
        try:
            tok = AutoTokenizer.from_pretrained(self.ce_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.ce_model_name).to(self.device)
            self._ce_cache = (tok, model)
            self._ce_name = self.ce_model_name
        except Exception:  # noqa: BLE001
            self._ce_cache = None
            self._ce_name = None
        return self._ce_cache

    @staticmethod
    def _symbolic_similarity(row: pd.Series) -> float:
        src = row.get("src_label_text") or row.get("Src") or ""
        tgt = row.get("tgt_label_text") or row.get("Tgt") or ""
        if not src or not tgt:
            return 0.0
        return difflib.SequenceMatcher(None, src, tgt).ratio()

    def _allow_more_llm(self) -> bool:
        return self.max_llm_prompts != 0 and (
            self.max_llm_prompts < 0 or self._llm_prompts_used < self.max_llm_prompts
        )

    def _remaining_llm_budget(self) -> int:
        if self.max_llm_prompts < 0:
            return 1_000_000_000
        return max(0, self.max_llm_prompts - self._llm_prompts_used)

    def _llm_batch_choose_candidate(
        self,
        specs: List[Tuple[str, List[int]]],
        df: pd.DataFrame,
        primary_model: IModel,
    ) -> Dict[str, Optional[int]]:
        llm = getattr(primary_model, "llm", None)
        tok = getattr(primary_model, "llm_tok", None)
        if llm is None or tok is None:
            return {}
        llm_device = next(llm.parameters()).device
        prompts: List[str] = []
        meta: List[Tuple[str, List[int]]] = []
        for src, indices in specs:
            sub = df.loc[indices].sort_values("S_final", ascending=False).head(self.max_prompt_candidates)
            if sub.empty:
                continue
            src_label = sub["src_label_text"].iloc[0] or sub["Src"].iloc[0]
            src_ctx = sub["src_context_text"].iloc[0]
            prompt_lines = [
                "You are ranking candidate target entities for the given source.",
                f"Source: {src_label}",
            ]
            if src_ctx:
                prompt_lines.append(f"Source context: {src_ctx}")
            prompt_lines.append("Candidates:")
            for idx, (_, row) in enumerate(sub.iterrows(), start=1):
                desc = row["tgt_label_text"] or row["Tgt"]
                ctx = row["tgt_context_text"]
                entry = f"{idx}. {desc}"
                if ctx:
                    entry += f" -- {ctx}"
                prompt_lines.append(entry)
            prompt_lines.append("Reply with the number of the best matching candidate.")
            prompts.append("\n".join(prompt_lines))
            meta.append((src, sub.index.tolist()))

        if not prompts:
            return {}

        inputs = tok(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(llm_device)

        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.0,
                do_sample=False,
            )
        input_lengths = inputs["attention_mask"].sum(dim=1)
        results: Dict[str, Optional[int]] = {}
        for i, (src, idx_list) in enumerate(meta):
            start = int(input_lengths[i].item())
            generated = outputs[i, start:]
            text = tok.decode(generated, skip_special_tokens=True)
            choice = None
            for token in text.split():
                if token.rstrip(".").isdigit():
                    cand_idx = int(token.rstrip(".")) - 1
                    if 0 <= cand_idx < len(idx_list):
                        choice = cand_idx
                        break
            results[src] = choice
        return results
