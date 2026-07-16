from __future__ import annotations

import hashlib
import json
import re
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exact.core.contracts.dataset import DataFrame
from exact.core.entities.configs.dataset import (
    BestPathMethod,
    ContextMethod,
)
from exact.core.entities.ontology import OntologyGraph
from exact.core.entities.registry import ComponentType
from exact.impl.datasets.base import BaseAlignmentDataset
from exact.llm.routing import LLMRouter, extract_chat_text, parse_structured_json


def _prompt_verbalize(head: str, rel: str, tail: str) -> Dict[str, str]:
    return {
        "system": "You are an ontology expert natural language generator that returns strict JSON.",
        "user": (
            "Given the relation and example triple below, return one JSON object with exactly one key: "
            '"template".\n'
            "The template must be one complete sentence and must contain the literal placeholders $SRC and $TGT "
            "exactly once each. Do not use the concrete entity names in the template.\n\n"
            f"HEAD example: {head}\nRELATION: {rel}\nTAIL example: {tail}\n\n"
            "Return only JSON."
        ),
    }


def _prompt_corrective(prev_sentence: str, head: str, rel: str, tail: str) -> Dict[str, str]:
    return {
        "system": "You are an ontology expert natural language generator that returns strict JSON.",
        "user": (
            "Your previous output did not follow the requested template format.\n"
            f"Previous output: {prev_sentence}\n\n"
            'Regenerate and return one JSON object with exactly one key: "template".\n'
            "The template must contain literal $SRC and $TGT exactly once each and must not contain the example entity names.\n\n"
            f"HEAD example: {head}\nRELATION: {rel}\nTAIL example: {tail}\n\n"
            "Return only JSON."
        ),
    }


class ContextDataset(BaseAlignmentDataset):
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
    _taxonomy_warning_emitted_global = False

    def __init__(
        self,
        # Ontology/context extraction
        n_hops: int = 2,
        context_method: ContextMethod = ContextMethod.greedy,  # bfs | greedy
        best_path_method: BestPathMethod = BestPathMethod.dp,  # dp | lagrangian | greedy (when context_method=greedy)
        context_hop_penalty: float = 0.1,  # α (scaled inside OntologyGraph helper)
        context_token_ratio: float = 1.3,  # tokens≈words*ratio (if you set a budget externally)
        context_safety: float = 0.8,
        max_input_tokens_context: int = 256,  # budget safety
        only_taxonomy: bool = False,  # if True, fixed templates for subclass
        all_labels: bool = True,  # if True, pass all labels; else use best label only
        add_connectivity_bridges: bool = True,  # if True, add explanation-only connectors to keep contexts connected
        bridge_max_hops: Optional[int] = None,  # cap for bridge path search (None = unbounded)
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
        llm_profiles: Optional[Dict[str, Any]] = None,
        llm_routing: Optional[Dict[str, Any]] = None,
        request_seed: Optional[int] = None,
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
        self._only_taxonomy_hint = self.only_taxonomy
        self.all_labels = bool(all_labels)
        self.exclude_missing_dr = bool(exclude_missing_dr)
        self.add_connectivity_bridges = bool(add_connectivity_bridges)
        self.bridge_max_hops = bridge_max_hops

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
        self.request_seed = int(request_seed) if request_seed is not None else None
        self._default_verbaliser_system_prompt = "You are a helpful ontology expert."
        self._taxonomy_warning_emitted = False
        self._taxonomy_template_log_emitted = False
        self._llm_router = LLMRouter(
            llm_profiles=llm_profiles, llm_routing=llm_routing, log=self.log
        )
        self._local_verbaliser_profile_name = "__context_local_verbaliser__"
        self._llm_router.ensure_profile(
            self._local_verbaliser_profile_name,
            {"backend": "local_hf", "model": self.verbaliser_name},
        )
        if (
            self._llm_router.routing.verbaliser_profile is None
            and self._llm_router.routing.default_profile is None
        ):
            self._llm_router.routing.verbaliser_profile = self._local_verbaliser_profile_name
        if self._llm_router.routing.fallback_profile is None:
            self._llm_router.routing.fallback_profile = self._local_verbaliser_profile_name
        self._verbaliser_backend = self._llm_router.resolve_task("verbaliser")

        # Prepared on demand
        self._source_graph: Optional[OntologyGraph] = None
        self._target_graph: Optional[OntologyGraph] = None

        # LLM handle (lazy)
        self._verbaliser_tok: Optional[AutoTokenizer] = None
        self._verbaliser: Optional[AutoModelForCausalLM] = None

        # Templates & caches
        self._verbalization_templates: Optional[Dict[str, str]] = None
        self._verb_temp_path = self.output_path / "verbalization_templates.json"
        self._verb_temp_meta_path = self.output_path / "verbalization_templates.meta.json"
        self.log(
            f"ContextDataset initialised with only_taxonomy={self.only_taxonomy}, all_labels={self.all_labels}",
            level="info",
        )

        # Output dataframe:
        # columns: ["Src", "Tgt", "Label", "SrcLabels", "TgtLabels", "SrcCtx", "TgtCtx"]
        self._df = None  # managed by process()

        def _triple_token_cost(triple):
            sentences = self._verbalize_triples([triple])
            if isinstance(sentences, str):
                sentences = [sentences]
            word_count = sum(len(sentence.split()) for sentence in sentences if sentence)
            return int(word_count * self.context_token_ratio)

        self.context_cost_fn = _triple_token_cost

    def _cache_fingerprint_payload(self) -> Dict[str, Any]:
        payload = super()._cache_fingerprint_payload()
        payload.update(
            {
                "n_hops": self.n_hops,
                "context_method": str(self.context_method),
                "best_path_method": str(self.best_path_method),
                "context_hop_penalty": self.context_hop_penalty,
                "context_token_ratio": self.context_token_ratio,
                "context_safety": self.context_safety,
                "max_input_tokens_context": self.max_input_tokens_context,
                "only_taxonomy": self.only_taxonomy,
                "all_labels": self.all_labels,
                "add_connectivity_bridges": self.add_connectivity_bridges,
                "bridge_max_hops": self.bridge_max_hops,
                "verbaliser_name": self.verbaliser_name,
                "verbaliser_backend": self._verbaliser_backend.backend,
                "verbaliser_profile": self._verbaliser_backend.profile_name,
                "verbaliser_model": self._verbaliser_backend.model,
                "gen_max_new_tokens": self.gen_max_new_tokens,
                "do_sample": self.do_sample,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "batch_size_verbaliser": self.batch_size_verbaliser,
                "max_verb_gen_retries": self.max_verb_gen_retries,
                "delimiter": self.delimiter,
                "llm_router": self._llm_router.fingerprint_payload(),
                "request_seed": self.request_seed,
            }
        )
        return payload

    @property
    def verbalizer_template_fingerprint(self) -> str:
        payload = {
            "dataset_signature": self.dataset_signature,
            "only_taxonomy": self.only_taxonomy,
            "verbaliser_name": self.verbaliser_name,
            "verbaliser_backend": self._verbaliser_backend.backend,
            "verbaliser_profile": self._verbaliser_backend.profile_name,
            "verbaliser_model": self._verbaliser_backend.model,
            "gen_max_new_tokens": self.gen_max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "batch_size_verbaliser": self.batch_size_verbaliser,
            "max_verb_gen_retries": self.max_verb_gen_retries,
            "llm_router": self._llm_router.fingerprint_payload(),
            "request_seed": self.request_seed,
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _template_cache_matches(self) -> bool:
        if not self._verb_temp_path.exists():
            return False
        if not self._verb_temp_meta_path.exists():
            self.log(
                "Existing verbalisation template cache lacks metadata; regenerating templates.",
                level="warning",
            )
            return False
        try:
            meta = json.loads(self._verb_temp_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.log(
                "Verbalisation template cache metadata is unreadable; regenerating templates.",
                level="warning",
            )
            return False
        if meta.get("fingerprint") != self.verbalizer_template_fingerprint:
            self.log(
                "Existing verbalisation template cache is invalid for the current model/configuration; regenerating templates.",
                level="warning",
            )
            return False
        return True

    def _write_template_cache_metadata(self) -> None:
        payload = {
            "fingerprint": self.verbalizer_template_fingerprint,
            "dataset_signature": self.dataset_signature,
            "backend": self._verbaliser_backend.backend,
            "profile": self._verbaliser_backend.profile_name,
            "model": self._verbaliser_backend.model,
        }
        self._verb_temp_meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Ontology graphs (cached)
    # ------------------------------------------------------------------
    @property
    def source_graph(self) -> OntologyGraph:
        if self._source_graph is None:
            if self.only_taxonomy:
                self.log("Loading source graph (taxonomy only)…", level="info")
            else:
                self.log("Loading source graph (OWL2Vec*)…", level="info")
            self._source_graph = OntologyGraph(self.source, only_taxonomy=self.only_taxonomy)
            self.log(f"Source graph edges: {len(self._source_graph)}", level="debug")
        return self._source_graph

    @property
    def target_graph(self) -> OntologyGraph:
        if self._target_graph is None:
            if self.only_taxonomy:
                self.log("Loading target graph (taxonomy only)…", level="info")
            else:
                self.log("Loading target graph (OWL2Vec*)…", level="info")
            self._target_graph = OntologyGraph(self.target, only_taxonomy=self.only_taxonomy)
            self.log(f"Target graph edges: {len(self._target_graph)}", level="debug")
        return self._target_graph

    # ------------------------------------------------------------------
    # Verbaliser LLM
    # ------------------------------------------------------------------
    def _ensure_verbaliser(self):
        if self.only_taxonomy:
            return  # no LLM needed
        if self._verbaliser_backend.backend == "openrouter":
            return
        if self._verbaliser is None or self._verbaliser_tok is None:
            if self.verbaliser_name is None:
                self.log(
                    "verbaliser_name is None but only_taxonomy=False; cannot generate templates.",
                    level="error",
                )
                raise ValueError("Set verbaliser_name or only_taxonomy=True.")
            self.log(f"Loading verbaliser LLM: {self.verbaliser_name}", level="info")
            self._verbaliser_tok = AutoTokenizer.from_pretrained(self.verbaliser_name)
            self._verbaliser = AutoModelForCausalLM.from_pretrained(self.verbaliser_name).to(
                self.device
            )

    # ------------------------------------------------------------------
    # Templates (LLM-generated except taxonomy-only)
    # ------------------------------------------------------------------
    @property
    def verbalization_templates(self) -> Dict[str, str]:

        if self.only_taxonomy:
            if not self._taxonomy_template_log_emitted:
                self.log("Using taxonomy-only templates.", level="debug")
                self._taxonomy_template_log_emitted = True
            self._verbalization_templates = {
                "subclassof": "$SRC is a subclass of $TGT",
                "subclass_of": "$SRC is a subclass of $TGT",
                "subClassOf": "$SRC is a subclass of $TGT",
            }
            return self._verbalization_templates

        if self._verbalization_templates is not None:
            return self._verbalization_templates

        # Try load cache
        if self._template_cache_matches():
            self.log("Loading verbalisation templates from cache…", level="debug")
            with open(self._verb_temp_path, "r") as f:
                payload = json.load(f)
            self._verbalization_templates = (
                payload.get("templates")
                if isinstance(payload, dict) and "templates" in payload
                else payload
            )
            return self._verbalization_templates

        # Else Generate fresh
        self._ensure_verbaliser()
        self.log("Generating verbalisation templates from ontology relations…", level="info")

        examples = self.source_graph.get_example_triples(
            1, exclude_missing_dr=self.exclude_missing_dr, human_readable=True
        )
        tgt_examples = self.target_graph.get_example_triples(
            1, exclude_missing_dr=self.exclude_missing_dr, human_readable=True
        )
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
            tmpl = self._parse_template_output(sent)
            if self._template_is_valid(tmpl):
                templates[key] = tmpl
            else:
                to_retry.append((key, sent))

        # Retry loop (optional)
        for _ in range(self.max_verb_gen_retries):
            if not to_retry:
                break
            retry_prompts, retry_keys = [], []
            for key, prev in to_retry:
                retry_prompts.append(_prompt_corrective(prev, key2heads[key], key, key2tails[key]))
                retry_keys.append(key)
            retry_out = self._batch_generate(retry_prompts)
            new_retry: List[Tuple[str, str]] = []
            for key, sent in zip(retry_keys, retry_out):
                tmpl = self._parse_template_output(sent)
                if self._template_is_valid(tmpl):
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
            json.dump({"templates": templates}, f, indent=2)
        self._write_template_cache_metadata()

        self._verbalization_templates = templates
        return self._verbalization_templates

    # ------------------------------------------------------------------
    # LLM batch generation
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _batch_generate(self, prompts: List[Union[str, Dict[str, str]]]) -> List[str]:
        if self._verbaliser_backend.backend == "openrouter":
            return self._batch_generate_openrouter(prompts)
        self._ensure_verbaliser()
        tok = self._verbaliser_tok
        model = self._verbaliser
        model.eval()

        # Sort by length for efficiency
        rendered_prompts = [self._render_prompt(p) for p in prompts]
        lengths = [len(p.split()) for p in rendered_prompts]
        order = sorted(range(len(prompts)), key=lambda i: lengths[i], reverse=True)
        inv = [0] * len(prompts)
        for new_pos, old_pos in enumerate(order):
            inv[old_pos] = new_pos
        sorted_prompts = [rendered_prompts[i] for i in order]

        outs_sorted: List[str] = []
        B = self.batch_size_verbaliser
        for i in range(0, len(sorted_prompts), B):
            batch = sorted_prompts[i : i + B]
            enc = tok(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            gen = model.generate(
                **enc,
                max_new_tokens=self.gen_max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k if self.top_k is not None else 0,
                pad_token_id=tok.eos_token_id,
            )
            new_tokens = self._strip_prompt_tokens(enc, gen)
            dec = tok.batch_decode(new_tokens, skip_special_tokens=True)
            cleaned = [self._clean_template_text(txt) for txt in dec]
            outs_sorted.extend(cleaned)

        # Unsort
        outs = [None] * len(prompts)
        for orig_idx, new_idx in enumerate(inv):
            outs[orig_idx] = outs_sorted[new_idx]
        return outs

    def _batch_generate_openrouter(self, prompts: List[Union[str, Dict[str, str]]]) -> List[str]:
        profile = self._llm_router.profiles.get(self._verbaliser_backend.profile_name or "")
        if profile is None:
            raise RuntimeError("OpenRouter verbaliser profile was resolved but not found.")
        workers = self.batch_size_verbaliser or len(prompts)
        workers = max(1, min(int(workers), len(prompts))) if prompts else 1
        self.log(
            (
                "Hosted verbaliser stage: "
                f"{len(prompts)} prompts, concurrency={workers}, profile='{profile.name}'."
            ),
            level="debug",
        )

        def _call(prompt: Union[str, Dict[str, str]]) -> str:
            if isinstance(prompt, dict):
                system = prompt.get("system", "").strip()
                user = prompt.get("user", "").strip()
            else:
                system = self._default_verbaliser_system_prompt
                user = str(prompt).strip()
            payload = self._llm_router.hosted.chat_completion(
                profile=profile,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.gen_max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.request_seed,
            )
            return extract_chat_text(payload)

        if workers == 1:
            outputs = [_call(prompt) for prompt in prompts]
            self.log("Hosted verbaliser stage completed.", level="debug")
            return outputs
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outputs = list(executor.map(_call, prompts))
        self.log("Hosted verbaliser stage completed.", level="debug")
        return outputs

    def _render_prompt(self, prompt: Union[str, Dict[str, str]]) -> str:
        if isinstance(prompt, dict):
            system = prompt.get("system", "").strip()
            user = prompt.get("user", "").strip()
        else:
            system = ""
            user = str(prompt).strip()
        if not system:
            system = self._default_verbaliser_system_prompt

        tok = self._verbaliser_tok
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        segments = [seg for seg in [system, user] if seg]
        return "\n\n".join(segments)

    def _strip_prompt_tokens(
        self, enc: Dict[str, torch.Tensor], gen: torch.Tensor
    ) -> List[torch.Tensor]:
        attn = enc["attention_mask"].sum(dim=1)
        outputs: List[torch.Tensor] = []
        for row, plen in zip(gen, attn):
            plen = int(plen.item())
            if plen < row.shape[0]:
                outputs.append(row[plen:])
            else:
                outputs.append(row[-1:].clone())
        return outputs

    @staticmethod
    def _clean_template_text(text: str) -> str:
        txt = text.strip()
        if not txt:
            return ""
        lines = txt.splitlines()
        if not lines:
            return ""
        if "Sentence:" in txt:
            txt = txt.split("Sentence:", 1)[-1].strip()
            lines = txt.splitlines() or lines
        first_line = lines[0].strip()
        prefix_pattern = re.compile(
            r"^(assistant|assistant:|assistant,|ai[- ]generated sentence:)", re.IGNORECASE
        )
        first_line = prefix_pattern.sub("", first_line).strip(" :,-\t")
        if not first_line:
            first_line = txt
        sentence_parts = re.split(r"(?<=[.!?])\s+", first_line)
        cleaned = sentence_parts[0].strip()
        return cleaned or first_line

    @staticmethod
    def _parse_template_output(text: str) -> str:
        try:
            candidate = parse_structured_json(text, "template")
        except Exception:
            candidate = ContextDataset._clean_template_text(text)
        return candidate.strip()

    @staticmethod
    def _template_is_valid(text: str) -> bool:
        if not text:
            return False
        return text.count("$SRC") == 1 and text.count("$TGT") == 1

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
                if (
                    self.only_taxonomy
                    and not self._taxonomy_warning_emitted
                    and not ContextDataset._taxonomy_warning_emitted_global
                ):
                    self.log(
                        "only_taxonomy=True: using generic '$SRC rel $TGT' templates for non-taxonomy relations (suppressing further warnings).",
                        level="warning",
                    )
                    self._taxonomy_warning_emitted = True
                    ContextDataset._taxonomy_warning_emitted_global = True
                tmpl = "$SRC " + key.replace("_", " ").lower() + " $TGT"
            out.append(tmpl.replace("$SRC", head).replace("$TGT", tail))
        return out

    # ------------------------------------------------------------------
    # Dataset plumbing
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._active_dataframe())

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
        dfk = self._active_dataframe()
        row = dfk.iloc[idx]

        feats = row.get("Features")
        if isinstance(feats, (list, tuple)) and len(feats) == 4:
            src_labels, src_ctx_list, tgt_labels, tgt_ctx_list = feats
        else:
            src_labels = row["SrcLabels"]
            src_ctx_list = row["SrcCtx"]
            tgt_labels = row["TgtLabels"]
            tgt_ctx_list = row["TgtCtx"]
        src_ctx_raw = row.get("SrcCtxRaw", [])
        tgt_ctx_raw = row.get("TgtCtxRaw", [])
        src_ctx_bridge = row.get("SrcCtxBridge", [])
        tgt_ctx_bridge = row.get("TgtCtxBridge", [])

        item = {
            "src_iri": row["Src"],
            "tgt_iri": row["Tgt"],
            "src_kind": row.get("SrcKind", "class"),
            "tgt_kind": row.get("TgtKind", row.get("SrcKind", "class")),
            "src_labels": src_labels,  # List[str]
            "tgt_labels": tgt_labels,  # List[str]
            "src_ctx_triples": src_ctx_list,  # List[str] sentences
            "tgt_ctx_triples": tgt_ctx_list,  # List[str] sentences
            "src_ctx_raw_triples": src_ctx_raw,  # List[Tuple[str,str,str]]
            "tgt_ctx_raw_triples": tgt_ctx_raw,
            "src_ctx_bridge_triples": src_ctx_bridge,  # Explanation-only connectors
            "tgt_ctx_bridge_triples": tgt_ctx_bridge,
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
            src_lab_map[iri] = (
                lbls[:]
                if self.all_labels
                else ([lbls[0]] if lbls else [self.source.short_form(iri)])
            )

        tgt_lab_map: Dict[str, List[str]] = {}
        for iri in utgt:
            lbls = self.target_graph.get_labels(iri)
            tgt_lab_map[iri] = (
                lbls[:]
                if self.all_labels
                else ([lbls[0]] if lbls else [self.target.short_form(iri)])
            )

        # Context subgraphs (triples) and verbalisation
        def _ctx(
            iri: str, graph: OntologyGraph
        ) -> Tuple[List[str], List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
            if self.add_connectivity_bridges:
                triples, bridge_triples = graph.get_context_subgraph_with_bridges(
                    iri,
                    n=self.n_hops,
                    human_readable=True,
                    method=self.context_method,
                    best_path_method=self.best_path_method,
                    budget=self.max_input_tokens_context,
                    hop_penalty=self.context_hop_penalty,
                    max_bridge_hops=self.bridge_max_hops,
                )
            else:
                triples = graph.get_context_subgraph(
                    iri,
                    n=self.n_hops,
                    human_readable=True,
                    method=self.context_method,
                    best_path_method=self.best_path_method,
                    budget=self.max_input_tokens_context,
                    hop_penalty=self.context_hop_penalty,
                )
                bridge_triples = []
            if not triples:
                return [], [], []
            return self._verbalize_triples(triples), triples, bridge_triples

        if self.context_method is ContextMethod.greedy:
            self.log("####Using 'greedy' context extraction method.", level="debug")
            if self.best_path_method is None:
                self.log("####No best path source method set, using default 'dp'.", level="debug")
                self.best_path_method = BestPathMethod.dp
            else:
                self.log(
                    f"####Using best path source method: '{self.best_path_method}'.", level="debug"
                )

            # Set cost function for greedy extraction
            self.log(
                "####Setting cost function for context extraction. Computing cost for every triple..",
                level="debug",
            )
            self.source_graph.cost_fn = self.context_cost_fn
            self.target_graph.cost_fn = self.context_cost_fn

        else:
            self.log("####Using 'BFS' context extraction method.", level="debug")

        self.log("Extracting and verbalising source contexts…", level="debug")
        src_ctx_map: Dict[str, List[str]] = {}
        src_ctx_raw_map: Dict[str, List[Tuple[str, str, str]]] = {}
        src_ctx_bridge_map: Dict[str, List[Tuple[str, str, str]]] = {}
        for iri in usrc:
            ctx_sentences, raw_triples, bridge_triples = _ctx(iri, self.source_graph)
            src_ctx_map[iri] = ctx_sentences
            src_ctx_raw_map[iri] = raw_triples
            src_ctx_bridge_map[iri] = bridge_triples

        self.log("Extracting and verbalising target contexts…", level="debug")
        tgt_ctx_map: Dict[str, List[str]] = {}
        tgt_ctx_raw_map: Dict[str, List[Tuple[str, str, str]]] = {}
        tgt_ctx_bridge_map: Dict[str, List[Tuple[str, str, str]]] = {}
        for iri in utgt:
            ctx_sentences, raw_triples, bridge_triples = _ctx(iri, self.target_graph)
            tgt_ctx_map[iri] = ctx_sentences
            tgt_ctx_raw_map[iri] = raw_triples
            tgt_ctx_bridge_map[iri] = bridge_triples

        # ---- Context metrics per row ----
        src_ctx_metrics_map = {
            iri: self._compute_metrics_for_context_list(src_ctx_map[iri]) for iri in usrc
        }
        tgt_ctx_metrics_map = {
            iri: self._compute_metrics_for_context_list(tgt_ctx_map[iri]) for iri in utgt
        }

        for key in ["n_triples", "char_len", "word_len", "tok_len", "is_empty"]:
            df[f"src_ctx_{key}"] = [src_ctx_metrics_map[iri][key] for iri in src_iris]
            df[f"tgt_ctx_{key}"] = [tgt_ctx_metrics_map[iri][key] for iri in tgt_iris]

        # Quick context emptiness alerts
        empty_src_ctx = int(df["src_ctx_is_empty"].sum())
        empty_tgt_ctx = int(df["tgt_ctx_is_empty"].sum())
        if empty_src_ctx:
            self.log(
                f"#### Empty source contexts: {empty_src_ctx} ({empty_src_ctx/len(df):.1%})",
                level="warning",
            )
        if empty_tgt_ctx:
            self.log(
                f"#### Empty target contexts: {empty_tgt_ctx} ({empty_tgt_ctx/len(df):.1%})",
                level="warning",
            )

        # ---- Label metrics per row ----
        src_label_metrics_map = {
            iri: self._compute_metrics_for_label_list(src_lab_map[iri]) for iri in usrc
        }
        tgt_label_metrics_map = {
            iri: self._compute_metrics_for_label_list(tgt_lab_map[iri]) for iri in utgt
        }

        for key in [
            "n_labels",
            "char_len",
            "word_len",
            "max_label_words",
            "avg_label_words",
            "is_empty",
        ]:
            df[f"src_lab_{key}"] = [src_label_metrics_map[iri][key] for iri in src_iris]
            df[f"tgt_lab_{key}"] = [tgt_label_metrics_map[iri][key] for iri in tgt_iris]

        # Quick label emptiness alerts
        empty_src_lab = int(df["src_lab_is_empty"].sum())
        empty_tgt_lab = int(df["tgt_lab_is_empty"].sum())
        if empty_src_lab or empty_tgt_lab:
            self.log(
                f"#### Empty labels — src: {empty_src_lab}, tgt: {empty_tgt_lab}", level="warning"
            )

        # Assemble per row
        df = df.copy()
        src_labels_per_row = [src_lab_map[iri] for iri in src_iris]
        tgt_labels_per_row = [tgt_lab_map[iri] for iri in tgt_iris]
        src_ctx_per_row = [src_ctx_map[iri] for iri in src_iris]
        tgt_ctx_per_row = [tgt_ctx_map[iri] for iri in tgt_iris]
        src_ctx_raw_per_row = [src_ctx_raw_map[iri] for iri in src_iris]
        tgt_ctx_raw_per_row = [tgt_ctx_raw_map[iri] for iri in tgt_iris]
        src_ctx_bridge_per_row = [src_ctx_bridge_map[iri] for iri in src_iris]
        tgt_ctx_bridge_per_row = [tgt_ctx_bridge_map[iri] for iri in tgt_iris]

        df["SrcLabels"] = src_labels_per_row
        df["TgtLabels"] = tgt_labels_per_row
        df["SrcCtx"] = src_ctx_per_row
        df["TgtCtx"] = tgt_ctx_per_row
        df["SrcCtxRaw"] = src_ctx_raw_per_row
        df["TgtCtxRaw"] = tgt_ctx_raw_per_row
        df["SrcCtxBridge"] = src_ctx_bridge_per_row
        df["TgtCtxBridge"] = tgt_ctx_bridge_per_row
        df["Features"] = [
            [src_labels, src_ctx, tgt_labels, tgt_ctx]
            for src_labels, src_ctx, tgt_labels, tgt_ctx in zip(
                src_labels_per_row, src_ctx_per_row, tgt_labels_per_row, tgt_ctx_per_row
            )
        ]

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

    def save_feature_metrics(
        self, df: Optional[DataFrame] = None, filename: str = "feature_metrics.csv"
    ) -> Path:
        if df is None:
            df = self.dataframe
        cols = [
            "Src",
            "Tgt",
            # context metrics
            "src_ctx_n_triples",
            "src_ctx_char_len",
            "src_ctx_word_len",
            "src_ctx_tok_len",
            "src_ctx_is_empty",
            "tgt_ctx_n_triples",
            "tgt_ctx_char_len",
            "tgt_ctx_word_len",
            "tgt_ctx_tok_len",
            "tgt_ctx_is_empty",
            # label metrics
            "src_lab_n_labels",
            "src_lab_char_len",
            "src_lab_word_len",
            "src_lab_max_label_words",
            "src_lab_avg_label_words",
            "src_lab_is_empty",
            "tgt_lab_n_labels",
            "tgt_lab_char_len",
            "tgt_lab_word_len",
            "tgt_lab_max_label_words",
            "tgt_lab_avg_label_words",
            "tgt_lab_is_empty",
        ]
        cols = [c for c in cols if c in df.columns]
        out_path = (self.output_path / filename).resolve()
        df[cols].to_csv(out_path, index=False)
        self.log(f"Saved feature metrics CSV to {out_path}", level="info")
        return out_path

    def emit_feature_metrics_on_build(self) -> bool:
        return False

    def supported_plot_metrics(self) -> List[str]:
        return [
            "src_ctx_n_triples",
            "tgt_ctx_n_triples",
            "src_ctx_tok_len",
            "tgt_ctx_tok_len",
            "src_ctx_word_len",
            "tgt_ctx_word_len",
            "src_ctx_char_len",
            "tgt_ctx_char_len",
            "src_lab_n_labels",
            "tgt_lab_n_labels",
            "src_lab_word_len",
            "tgt_lab_word_len",
            "src_lab_char_len",
            "tgt_lab_char_len",
            "src_lab_max_label_words",
            "tgt_lab_max_label_words",
            "src_lab_avg_label_words",
            "tgt_lab_avg_label_words",
            "cand_sim",
            "cand_sim_src_mean",
            "cand_sim_prob",
            "cand_share_top",
            "cand_share_rest",
            "cand_share_log_ratio",
        ]

    def default_plot_metrics(self) -> List[str]:
        return [
            "src_ctx_n_triples",
            "tgt_ctx_n_triples",
            "src_ctx_tok_len",
            "tgt_ctx_tok_len",
            "src_lab_n_labels",
            "tgt_lab_n_labels",
            "src_lab_word_len",
            "tgt_lab_word_len",
            "src_lab_max_label_words",
            "tgt_lab_max_label_words",
            "cand_sim",
            "cand_sim_src_mean",
        ]

    def _resolve_plot_metrics(self, which: Optional[List[str]], df: DataFrame) -> List[str]:
        supported = list(self.supported_plot_metrics())
        default_metrics = [col for col in self.default_plot_metrics() if col in supported]
        requested = default_metrics if which is None else list(which)
        unsupported = [col for col in requested if col not in supported]
        if unsupported:
            preview = ", ".join(unsupported[:8])
            suffix = "" if len(unsupported) <= 8 else ", ..."
            self.log(
                f"Skipping unsupported dataset plot metrics for {self.__class__.__name__}: {preview}{suffix}",
                level="info",
            )
        selected = [col for col in requested if col in supported and col in df.columns]
        if selected:
            return selected
        fallback = [col for col in default_metrics if col in df.columns]
        if which is not None:
            self.log(
                f"No supported dataset plot metrics requested for {self.__class__.__name__}; using dataset defaults instead.",
                level="warning",
            )
        return fallback

    def plot_feature_distributions(
        self,
        which: Optional[List[str]] = None,
        bins: int = 30,
        kde: bool = False,
        dpi: int = 300,
        alpha: float = 0.6,
        **kwargs,
    ) -> None:
        """
        Provide any of these (examples):
        Context: "src_ctx_n_triples","tgt_ctx_n_triples","src_ctx_tok_len","tgt_ctx_tok_len","src_ctx_word_len","tgt_ctx_word_len","src_ctx_char_len","tgt_ctx_char_len"
        Labels:  "src_lab_n_labels","tgt_lab_n_labels","src_lab_word_len","tgt_lab_word_len","src_lab_char_len","tgt_lab_char_len","src_lab_max_label_words","tgt_lab_max_label_words","src_lab_avg_label_words","tgt_lab_avg_label_words"
        Candidates: "cand_sim"
        """
        df = self.dataframe
        which = self._resolve_plot_metrics(which, df)
        if not which:
            self.log(
                f"No dataset plot metrics available for {self.__class__.__name__}; skipping feature plots.",
                level="warning",
            )
            return
        plot_dir = (self.plot_dir / "features").resolve()
        plot_dir.mkdir(parents=True, exist_ok=True)

        for col in which:
            plt.figure(figsize=(7, 5))
            sns.histplot(df[col], bins=bins, kde=kde, stat="probability", alpha=alpha)
            plt.title(col.replace("_", " ").title())
            plt.xlabel(col)
            plt.ylabel("Probability")
            out = plot_dir / f"{col}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=dpi)
            plt.close()
            self.log(f"Saved plot: {out}", level="debug")

    def log_sanity_examples(
        self, n: int = 6, max_ctx_show: int = 3, max_label_show: int = 5, **kwargs
    ) -> None:
        df = self.dataframe
        total = len(df)
        if total == 0:
            self.log("Dataset empty; no sanity examples.", level="warning")
            return

        # Prefer edge cases first (empty contexts or empty labels)
        problematic = df[
            (df.get("src_ctx_is_empty", 0) == 1)
            | (df.get("tgt_ctx_is_empty", 0) == 1)
            | (df.get("src_lab_is_empty", 0) == 1)
            | (df.get("tgt_lab_is_empty", 0) == 1)
        ]
        if len(problematic) < n:
            rest = df.drop(problematic.index)
            if not rest.empty:
                problematic = pd.concat(
                    [problematic, rest.sample(min(n - len(problematic), len(rest)))]
                )
        show = problematic.head(n)

        def _coerce_seq(value):
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, str):
                try:
                    parsed = literal_eval(value)
                    if isinstance(parsed, (list, tuple)):
                        return list(parsed)
                except (SyntaxError, ValueError):
                    return []
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return []
            return [value]

        def _valid_feats(feats):
            return isinstance(feats, (list, tuple)) and len(feats) == 4

        self.log(f"### Sanity examples (n={len(show)})", level="info")
        for i, row in show.iterrows():
            src, tgt = row["Src"], row["Tgt"]
            feats = row["Features"]  # [src_labels, src_ctx, tgt_labels, tgt_ctx]
            if not _valid_feats(feats):
                feats = [
                    _coerce_seq(row.get("SrcLabels")),
                    _coerce_seq(row.get("SrcCtx")),
                    _coerce_seq(row.get("TgtLabels")),
                    _coerce_seq(row.get("TgtCtx")),
                ]
            if not _valid_feats(feats):
                self.log(
                    f"- Pair {i}: Src={src} | Tgt={tgt} — skipping (missing feature details).",
                    level="warning",
                )
                continue

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

        df = pd.read_csv(self._df_save_path)

        def _parse_column(column: str, fallback_factory):
            missing = 0
            invalid = 0
            parsed = []
            for value in df[column].tolist():
                if pd.isna(value):
                    missing += 1
                    parsed.append(fallback_factory())
                    continue
                text = value if isinstance(value, str) else str(value)
                if not text.strip():
                    missing += 1
                    parsed.append(fallback_factory())
                    continue
                try:
                    parsed_value = literal_eval(text)
                except (SyntaxError, ValueError):
                    invalid += 1
                    parsed.append(fallback_factory())
                    continue
                parsed.append(parsed_value)
            if missing or invalid:
                self.log(
                    f"Filled {missing} empty and {invalid} invalid entries while parsing column '{column}'.",
                    level="warning",
                )
            df[column] = parsed

        for column in ["SrcLabels", "TgtLabels", "SrcCtx", "TgtCtx"]:
            _parse_column(column, lambda: [])

        def _features_fallback():
            return [[], [], [], []]

        _parse_column("Features", _features_fallback)

        def _features_valid(feats):
            return isinstance(feats, (list, tuple)) and len(feats) == 4

        missing_features_mask = ~df["Features"].apply(_features_valid)
        if missing_features_mask.any():
            self.log(
                f"Rebuilding {int(missing_features_mask.sum())} feature rows from individual columns.",
                level="warning",
            )
            rebuilt = [
                [src_labels, src_ctx, tgt_labels, tgt_ctx]
                for src_labels, src_ctx, tgt_labels, tgt_ctx in zip(
                    df.loc[missing_features_mask, "SrcLabels"],
                    df.loc[missing_features_mask, "SrcCtx"],
                    df.loc[missing_features_mask, "TgtLabels"],
                    df.loc[missing_features_mask, "TgtCtx"],
                )
            ]
            df.loc[missing_features_mask, "Features"] = rebuilt

        self._df = df
