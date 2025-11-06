from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import random
import json
import torch

from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ast import literal_eval

from exact.core.entities.registry import ComponentType
from exact.core.entities.configs.dataset import DatasetMask, ContextMethod, BestPathMethod
from exact.core.contracts.dataset import IDataset, DataFrame
from exact.core.entities.ontology import OntologyGraph, Entity

def _prompt_verbalize(head: str, rel: str, tail: str) -> str:
    return (
        "You are an ontology expert natural language generator that converts structured data into a coherent sentence.\n"
        "Given the triple below, generate one complete, grammatically correct sentence that clearly expresses the relationship "
        "between the head and the tail as indicated by the relation. You must use the exact wording of HEAD and TAIL.\n"
        f"HEAD: {head}\nRELATION: {rel}\nTAIL: {tail}\nSentence:"
    )

def _prompt_corrective(prev_sentence: str, head: str, tail: str) -> str:
    return (
        "You previously generated the following incorrect sentence that failed to use the exact entity names.\n"
        f"Incorrect: {prev_sentence}\n"
        f"Please regenerate a single sentence that uses EXACTLY these surface forms for the entities:\n"
        f"HEAD must appear as: {head}\nTAIL must appear as: {tail}\n"
        "Provide only the corrected sentence:"
    )


class ContextDataset(IDataset):
    """
      • Loads/caches verbalisation templates with an LLM (unless only_taxonomy=True)
      • Extracts per-entity context subgraphs
      • Verbalises each triple into a sentence
      • Returns items matching SemanticScorer.forward() expectations

    __getitem__ returns:
      {
        'src_iri': str,
        'tgt_iri': str,
        'src_labels': List[str],
        'tgt_labels': List[str],
        'src_ctx_triples': List[str],  # list of sentences (one per triple)
        'tgt_ctx_triples': List[str],
        'label': Optional[int]         # if reference exists, else 0/None
      }
    """

    component_type = ComponentType.DATASET

    def __init__(
        self,
        # Ontology/context extraction
        n_hops: int = 2,
        context_method: ContextMethod = ContextMethod.greedy,            # bfs | greedy
        best_path_method: BestPathMethod = BestPathMethod.dp,            # dp | lagrangian | greedy (when context_method=greedy)
        context_hop_penalty: float = 0.1,                             # α (scaled inside OntologyGraph helper)
        context_token_ratio: float = 1.3,                             # tokens≈words*ratio (if you set a budget externally)
        context_safety: float = 0.8,   
        max_input_tokens_context: int = 256,                          # budget safety
        only_taxonomy: bool = False,                                  # if True, fixed templates for subclass
        all_labels: bool = True,                                     # if True, pass all labels; else use best label only
        # Verbalisation LLM
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbaliser_name: Optional[str] = "Qwen/Qwen2.5-3B-Instruct",
        gen_max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        max_verb_gen_retries: int = 1,
        # Efficiency
        batch_size_verbaliser: int = 32,
        exclude_missing_dr: bool = False,
        # Formatting
        delimiter: str = "\n",
        # Base class args
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_hops = int(n_hops)
        self.context_method = context_method
        self.best_path_method = best_path_method
        self.context_hop_penalty = float(context_hop_penalty)
        self.context_token_ratio = float(context_token_ratio)
        self.context_safety = float(context_safety)
        self.max_input_tokens_context = int(max_input_tokens_context)
        self.only_taxonomy = bool(only_taxonomy)
        self.all_labels = bool(all_labels)
        self.exclude_missing_dr = bool(exclude_missing_dr)
    

        # Verbaliser LLM
        self.device = device
        self.verbaliser_name = verbaliser_name
        self.gen_max_new_tokens = int(gen_max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = top_k
        self.batch_size_verbaliser = int(batch_size_verbaliser)
        self.max_verb_gen_retries = int(max_verb_gen_retries)

        self.delimiter = delimiter

        # Prepared on demand
        self._source_graph: Optional[OntologyGraph] = None
        self._target_graph: Optional[OntologyGraph] = None

        # LLM handle (lazy)
        self._verbaliser_tok: Optional[AutoTokenizer] = None
        self._verbaliser: Optional[AutoModelForCausalLM] = None

        # Templates & caches
        self._verbalization_templates: Optional[Dict[str, str]] = None
        self._verb_temp_path = self.output_path / "verbalization_templates.json"

        # Output dataframe:
        # columns: ["Src", "Tgt", "Label", "SrcLabels", "TgtLabels", "SrcCtx", "TgtCtx"]
        self._df = None  # managed by process()

    # ------------------------------------------------------------------
    # Ontology graphs (cached)
    # ------------------------------------------------------------------
    @property
    def source_graph(self) -> OntologyGraph:
        if self._source_graph is None:
            self.log("Loading source graph (OWL2Vec*)…", level="info")
            self._source_graph = OntologyGraph(self.source.ontology, self.source_reasoner, self.only_taxonomy)
            self.log(f"Source graph edges: {len(self._source_graph)}", level="debug")
        return self._source_graph

    @property
    def target_graph(self) -> OntologyGraph:
        if self._target_graph is None:
            self.log("Loading target graph (OWL2Vec*)…", level="info")
            self._target_graph = OntologyGraph(self.target.ontology, self.target_reasoner, self.only_taxonomy)
            self.log(f"Target graph edges: {len(self._target_graph)}", level="debug")
        return self._target_graph

    # ------------------------------------------------------------------
    # Verbaliser LLM
    # ------------------------------------------------------------------
    def _ensure_verbaliser(self):
        if self.only_taxonomy:
            return # no LLM needed
        if self._verbaliser is None or self._verbaliser_tok is None:
            if self.verbaliser_name is None:
                self.log("verbaliser_name is None but only_taxonomy=False; cannot generate templates.", level="error")
                raise ValueError("Set verbaliser_name or only_taxonomy=True.")
            self.log(f"Loading verbaliser LLM: {self.verbaliser_name}", level="info")
            self._verbaliser_tok = AutoTokenizer.from_pretrained(self.verbaliser_name)
            self._verbaliser = AutoModelForCausalLM.from_pretrained(self.verbaliser_name).to(self.device)

    # ------------------------------------------------------------------
    # Templates (LLM-generated except taxonomy-only)
    # ------------------------------------------------------------------
    @property
    def verbalization_templates(self) -> Dict[str, str]:

        if self.only_taxonomy:
            self.log("Using taxonomy-only templates.", level="debug")
            self._verbalization_templates = {
                "subclassof": "$SRC is a subclass of $TGT",
                "subclass_of": "$SRC is a subclass of $TGT",
                "subClassOf": "$SRC is a subclass of $TGT",
            }
            return self._verbalization_templates
        
        if self._verbalization_templates is not None:
            return self._verbalization_templates

        # Try load cache
        if self._verb_temp_path.exists():
            self.log("Loading verbalisation templates from cache…", level="debug")
            with open(self._verb_temp_path, "r") as f:
                self._verbalization_templates = json.load(f)
            return self._verbalization_templates

        # Else Generate fresh
        self._ensure_verbaliser()
        self.log("Generating verbalisation templates from ontology relations…", level="info")

        examples = self.source_graph.get_example_triples(1, exclude_missing_dr=self.exclude_missing_dr, human_readable=True)
        tgt_examples = self.target_graph.get_example_triples(1, exclude_missing_dr=self.exclude_missing_dr, human_readable=True)
        for k, v in tgt_examples.items():
            if k not in examples:
                examples[k] = v

        keys = list(examples.keys())
        prompts = []
        key2heads = {}
        key2tails = {}
        for key in keys:
            head, rel, tail = examples[key][0]
            prompts.append(_prompt_verbalize(head, rel, tail))
            key2heads[key] = head
            key2tails[key] = tail

        # Batch generate
        sents = self._batch_generate(prompts)

        templates: Dict[str, str] = {}
        to_retry: List[Tuple[str, str]] = []
        for key, sent in zip(keys, sents):
            head = key2heads[key]
            tail = key2tails[key]
            tmpl = sent.replace(head, "$SRC").replace(tail, "$TGT")
            if "$SRC" in tmpl and "$TGT" in tmpl:
                templates[key] = tmpl
            else:
                to_retry.append((key, sent))

        # Retry loop (optional)
        for _ in range(self.max_verb_gen_retries):
            if not to_retry:
                break
            retry_prompts, retry_keys = [], []
            for key, prev in to_retry:
                retry_prompts.append(_prompt_corrective(prev, key2heads[key], key2tails[key]))
                retry_keys.append(key)
            retry_out = self._batch_generate(retry_prompts)
            new_retry: List[Tuple[str, str]] = []
            for key, sent in zip(retry_keys, retry_out):
                head = key2heads[key]
                tail = key2tails[key]
                tmpl = sent.replace(head, "$SRC").replace(tail, "$TGT")
                if "$SRC" in tmpl and "$TGT" in tmpl:
                    templates[key] = tmpl
                else:
                    new_retry.append((key, sent))
            to_retry = new_retry

        # Fallback stubs
        for key, _ in to_retry:
            # Use the relation key as a readable phrase
            templates[key] = "$SRC " + key.replace("_", " ").lower() + " $TGT"

        # Save
        with open(self._verb_temp_path, "w") as f:
            json.dump(templates, f, indent=2)

        self._verbalization_templates = templates
        return self._verbalization_templates

    # ------------------------------------------------------------------
    # LLM batch generation
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _batch_generate(self, prompts: List[str]) -> List[str]:
        self._ensure_verbaliser()
        tok = self._verbaliser_tok
        model = self._verbaliser
        model.eval()

        # Sort by length for efficiency
        lengths = [len(p.split()) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: lengths[i], reverse=True)
        inv = [0] * len(prompts)
        for new_pos, old_pos in enumerate(order):
            inv[old_pos] = new_pos
        sorted_prompts = [prompts[i] for i in order]

        outs_sorted: List[str] = []
        B = self.batch_size_verbaliser
        for i in range(0, len(sorted_prompts), B):
            batch = sorted_prompts[i:i+B]
            enc = tok(
                batch, padding=True, truncation=True,
                return_tensors="pt"
            ).to(self.device)
            gen = model.generate(
                **enc,
                max_new_tokens=self.gen_max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k if self.top_k is not None else 0,
                pad_token_id=tok.eos_token_id,
            )
            dec = tok.batch_decode(gen, skip_special_tokens=True)
            # Heuristic: take last line or the part after "Sentence:"
            cleaned = []
            for txt in dec:
                if "Sentence:" in txt:
                    txt = txt.split("Sentence:", 1)[-1].strip()
                cleaned.append(txt.strip())
            outs_sorted.extend(cleaned)

        # Unsort
        outs = [None] * len(prompts)
        for orig_idx, new_idx in enumerate(inv):
            outs[orig_idx] = outs_sorted[new_idx]
        return outs

    # ------------------------------------------------------------------
    # Verbalise triples with templates
    # ------------------------------------------------------------------
    def _verbalize_triples(self, triples: List[Tuple[str, str, str]]) -> List[str]:
        """
        For each (head, rel, tail), produce one sentence using the relation's template.
        Falls back to a generic '$SRC rel $TGT' when the template key is missing.
        """
        tmpls = self.verbalization_templates  # triggers generation or load
        out: List[str] = []
        for head, rel, tail in triples:
            key = rel  # human-readable key as produced by OntologyGraph.human_readable=True
            tmpl = tmpls.get(key)
            if tmpl is None:
                tmpl = "$SRC " + key.replace("_", " ").lower() + " $TGT"
            out.append(tmpl.replace("$SRC", head).replace("$TGT", tail))
        return out

    # ------------------------------------------------------------------
    # Dataset plumbing
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        if self._df is None:
            raise RuntimeError("Dataset not processed. Call process() first.")
        return len(self._df[self.default_kind])

    def __getitem__(self, idx: int) -> Any:
        """
        Return a dict matching SemanticScorer.forward()’s expectations:
          - src_iri, tgt_iri
          - src_labels: List[str]
          - tgt_labels: List[str]
          - src_ctx_triples: List[str]  (verbalised sentences, each triple separately)
          - tgt_ctx_triples: List[str]
          - label: Optional[int]
        """
        if self._df is None:
            raise RuntimeError("Dataset not processed. Call process() first.")

        dfk = self._df[self._df[self.default_kind]].reset_index(drop=True)
        row = dfk.iloc[idx]

        item = {
            "src_iri": row["Src"],
            "tgt_iri": row["Tgt"],
            "src_labels": row["SrcLabels"],          # List[str]
            "tgt_labels": row["TgtLabels"],          # List[str]
            "src_ctx_triples": row["SrcCtx"],        # List[str] sentences
            "tgt_ctx_triples": row["TgtCtx"],        # List[str] sentences
            "label": row.get("Label", None),
        }
        return item

    # ------------------------------------------------------------------
    # Feature construction for scoring
    # ------------------------------------------------------------------
    def get_features(self, df) -> "DataFrame":
        """
        Given a candidates dataframe with columns ["Src","Tgt","Score", ...],
        attach:
          - SrcLabels: List[str] (all or [best])
          - TgtLabels: List[str]
          - SrcCtx:    List[str] verbalised sentences (each triple separate)
          - TgtCtx:    List[str]
        """
        import pandas as pd

        self.log("Generating labels and context triples…", level="info")

        src_iris: List[str] = df["Src"].tolist()
        tgt_iris: List[str] = df["Tgt"].tolist()

        # Build unique entity maps to avoid recomputation
        usrc = list(dict.fromkeys(src_iris))
        utgt = list(dict.fromkeys(tgt_iris))

        # Labels
        src_lab_map: Dict[str, List[str]] = {}
        for iri in usrc:
            lbls = self.source_graph.get_labels(iri)
            src_lab_map[iri] = lbls[:] if self.all_labels else [lbls[0]] if lbls else [Entity._get_owl_class(iri, self.source.ontology).getIRI().getShortForm()]

        tgt_lab_map: Dict[str, List[str]] = {}
        for iri in utgt:
            lbls = self.target_graph.get_labels(iri)
            tgt_lab_map[iri] = lbls[:] if self.all_labels else [lbls[0]] if lbls else [Entity._get_owl_class(iri, self.target.ontology).getIRI().getShortForm()]

        # Context subgraphs (triples) and verbalisation
        def _ctx(iri: str, graph: OntologyGraph) -> List[str]:

            budget = int(
                (self.max_input_tokens_context * self.context_safety) / self.context_token_ratio
            )
            triples = graph.get_context_subgraph(
                iri,
                n=self.n_hops,
                human_readable=True,
                method=self.context_method,
                best_path_method=self.best_path_method,
                budget=budget,
                hop_penalty=self.context_hop_penalty,
            )
            if not triples:
                return []
            return self._verbalize_triples(triples)

        self.log("Extracting and verbalising source contexts…", level="debug")
        src_ctx_map: Dict[str, List[str]] = {iri: _ctx(iri, self.source_graph) for iri in usrc}

        self.log("Extracting and verbalising target contexts…", level="debug")
        tgt_ctx_map: Dict[str, List[str]] = {iri: _ctx(iri, self.target_graph) for iri in utgt}

        # ---- Context metrics per row ----
        src_ctx_metrics = [self._compute_metrics_for_context_list(xs) for xs in src_ctx_map]
        tgt_ctx_metrics = [self._compute_metrics_for_context_list(xs) for xs in tgt_ctx_map]

        for key in ["n_triples","char_len","word_len","tok_len","is_empty"]:
            df[f"src_ctx_{key}"] = [m[key] for m in src_ctx_metrics]
            df[f"tgt_ctx_{key}"] = [m[key] for m in tgt_ctx_metrics]

        # Quick context emptiness alerts
        empty_src_ctx = int(df["src_ctx_is_empty"].sum())
        empty_tgt_ctx = int(df["tgt_ctx_is_empty"].sum())
        if empty_src_ctx:
            self.log(f"#### Empty source contexts: {empty_src_ctx} ({empty_src_ctx/len(df):.1%})", level="warning")
        if empty_tgt_ctx:
            self.log(f"#### Empty target contexts: {empty_tgt_ctx} ({empty_tgt_ctx/len(df):.1%})", level="warning")

        # ---- Label metrics per row ----
        src_label_metrics = [self._compute_metrics_for_label_list(ls) for ls in src_lab_map]
        tgt_label_metrics = [self._compute_metrics_for_label_list(ls) for ls in tgt_lab_map]

        for key in ["n_labels","char_len","word_len","max_label_words","avg_label_words","is_empty"]:
            df[f"src_lab_{key}"] = [m[key] for m in src_label_metrics]
            df[f"tgt_lab_{key}"] = [m[key] for m in tgt_label_metrics]

        # Quick label emptiness alerts
        empty_src_lab = int(df["src_lab_is_empty"].sum())
        empty_tgt_lab = int(df["tgt_lab_is_empty"].sum())
        if empty_src_lab or empty_tgt_lab:
            self.log(
                f"#### Empty labels — src: {empty_src_lab}, tgt: {empty_tgt_lab}",
                level="warning"
            )


        # Assemble per row
        df = df.copy()
        df["SrcLabels"] = [src_lab_map[iri] for iri in src_iris]
        df["TgtLabels"] = [tgt_lab_map[iri] for iri in tgt_iris]
        df["SrcCtx"]    = [src_ctx_map[iri]   for iri in src_iris]
        df["TgtCtx"]    = [tgt_ctx_map[iri]   for iri in tgt_iris]

        return df
    
    # ------------------------------------------------------------------
    # Feature metrics computation
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
    # Use the same heuristic you rely on elsewhere
        return int(len(text.split()) * self.context_token_ratio)

    def _compute_metrics_for_context_list(self, ctx_list: List[str]) -> Dict[str, float]:
        """ctx_list is a list of verbalised triples (strings)."""
        n_triples = len(ctx_list)
        joined = self.delimiter.join(ctx_list) if ctx_list else ""
        char_len = len(joined)
        word_len = len(joined.split())
        tok_len = self._estimate_tokens(joined)
        return {
            "n_triples": n_triples,
            "char_len": char_len,
            "word_len": word_len,
            "tok_len": tok_len,
            "is_empty": int(n_triples == 0),
        }

    def _compute_metrics_for_label_list(self, labels: List[str]) -> Dict[str, float]:
        """labels is a list of label variants for an entity."""
        n_labels = len(labels)
        joined = " ".join(labels) if labels else ""
        char_len = len(joined)
        word_len = len(joined.split())
        max_label_words = max((len(x.split()) for x in labels), default=0)
        avg_label_words = (sum(len(x.split()) for x in labels) / n_labels) if n_labels > 0 else 0.0
        return {
            "n_labels": n_labels,
            "char_len": char_len,
            "word_len": word_len,
            "max_label_words": max_label_words,
            "avg_label_words": avg_label_words,
            "is_empty": int(n_labels == 0),
        }
    
    def save_feature_metrics(self, df: Optional[DataFrame] = None, filename: str = "feature_metrics.csv") -> Path:
        if df is None:
            df = self.dataframe
        cols = [
            "Src","Tgt",
            # context metrics
            "src_ctx_n_triples","src_ctx_char_len","src_ctx_word_len","src_ctx_tok_len","src_ctx_is_empty",
            "tgt_ctx_n_triples","tgt_ctx_char_len","tgt_ctx_word_len","tgt_ctx_tok_len","tgt_ctx_is_empty",
            # label metrics
            "src_lab_n_labels","src_lab_char_len","src_lab_word_len","src_lab_max_label_words","src_lab_avg_label_words","src_lab_is_empty",
            "tgt_lab_n_labels","tgt_lab_char_len","tgt_lab_word_len","tgt_lab_max_label_words","tgt_lab_avg_label_words","tgt_lab_is_empty",
        ]
        cols = [c for c in cols if c in df.columns]
        out_path = (self.output_path / filename).resolve()
        df[cols].to_csv(out_path, index=False)
        self.log(f"Saved feature metrics CSV to {out_path}", level="info")
        return out_path
    
    def plot_feature_distributions(
            self,
            which: Optional[List[str]] = None,
            bins: int = 30,
            kde: bool = False,
            dpi: int = 300,
            alpha: float = 0.6,
            **kwargs
        ) -> None:
            """
            Provide any of these (examples):
            Context: "src_ctx_n_triples","tgt_ctx_n_triples","src_ctx_tok_len","tgt_ctx_tok_len","src_ctx_word_len","tgt_ctx_word_len","src_ctx_char_len","tgt_ctx_char_len"
            Labels:  "src_lab_n_labels","tgt_lab_n_labels","src_lab_word_len","tgt_lab_word_len","src_lab_char_len","tgt_lab_char_len","src_lab_max_label_words","tgt_lab_max_label_words","src_lab_avg_label_words","tgt_lab_avg_label_words"
            Candidates: "cand_sim"
            """
            if which is None:
                which = [
                    "src_ctx_n_triples","tgt_ctx_n_triples",
                    "src_ctx_tok_len","tgt_ctx_tok_len",
                    "src_lab_n_labels","tgt_lab_n_labels",
                    "src_lab_word_len","tgt_lab_word_len",
                    "src_lab_max_label_words","tgt_lab_max_label_words",
                ]

            df = self.dataframe
            plot_dir = (self.plot_dir / "features").resolve()
            plot_dir.mkdir(parents=True, exist_ok=True)

            for col in which:
                if col not in df.columns:
                    self.log(f"Skipping missing metric '{col}'", level="warning")
                    continue
                plt.figure(figsize=(7,5))
                sns.histplot(df[col], bins=bins, kde=kde, stat="probability", alpha=alpha)
                plt.title(col.replace("_", " ").title())
                plt.xlabel(col)
                plt.ylabel("Probability")
                out = plot_dir / f"{col}.png"
                plt.tight_layout()
                plt.savefig(out, dpi=dpi)
                plt.close()
                self.log(f"Saved plot: {out}", level="debug")

    def log_sanity_examples(self, n: int = 6, max_ctx_show: int = 3, max_label_show: int = 5, **kwargs) -> None:
        df = self.dataframe
        total = len(df)
        if total == 0:
            self.log("Dataset empty; no sanity examples.", level="warning")
            return

        # Prefer edge cases first (empty contexts or empty labels)
        problematic = df[(df.get("src_ctx_is_empty", 0) == 1) | (df.get("tgt_ctx_is_empty", 0) == 1) |
                        (df.get("src_lab_is_empty", 0) == 1) | (df.get("tgt_lab_is_empty", 0) == 1)]
        if len(problematic) < n:
            rest = df.drop(problematic.index)
            if not rest.empty:
                problematic = pd.concat([problematic, rest.sample(min(n-len(problematic), len(rest)))])
        show = problematic.head(n)

        self.log(f"### Sanity examples (n={len(show)})", level="info")
        for i, row in show.iterrows():
            src, tgt = row["Src"], row["Tgt"]
            feats = row["Features"]  # [src_labels, src_ctx, tgt_labels, tgt_ctx]
            src_labels, src_ctx_list, tgt_labels, tgt_ctx_list = feats

            self.log(f"- Pair {i}: Src={src} | Tgt={tgt}", level="info")

            def _show_list(tag, lst, k, truncate_chars=200):
                if not lst:
                    self.log(f"  {tag}: <EMPTY>", level="warning")
                    return
                head = lst[:k]
                for j, line in enumerate(head):
                    trunc = (line[:truncate_chars] + "…") if len(line) > truncate_chars else line
                    self.log(f"  {tag}[{j}]: {trunc!r}", level="debug")
                if len(lst) > k:
                    self.log(f"  {tag}: … (+{len(lst)-k} more)", level="debug")

            # Labels (show up to k)
            _show_list("SRC labels", src_labels, max_label_show)
            _show_list("TGT labels", tgt_labels, max_label_show)
            # Context lines (verbalised triples)
            _show_list("SRC ctx", src_ctx_list, max_ctx_show)
            _show_list("TGT ctx", tgt_ctx_list, max_ctx_show)

            m = {
                "src_ctx_n_triples": row.get("src_ctx_n_triples"),
                "tgt_ctx_n_triples": row.get("tgt_ctx_n_triples"),
                "src_lab_n_labels": row.get("src_lab_n_labels"),
                "tgt_lab_n_labels": row.get("tgt_lab_n_labels"),
            }
            self.log(f"  Metrics: {m}", level="info")
            
    
    def load(self) -> "DataFrame":

        if self._df is not None:
            return self._df
        
        self._df = pd.read_csv(self._df_save_path, converters={
            "SrcLabels": literal_eval,
            "TgtLabels": literal_eval,
            "SrcCtx": literal_eval,
            "TgtCtx": literal_eval,
        })


