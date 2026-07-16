from __future__ import annotations

import hashlib  # noqa: F401
import json  # noqa: F401
import re  # noqa: F401
import time  # noqa: F401
from collections import OrderedDict  # noqa: F401
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple  # noqa: F401
from urllib import error as urlerror  # noqa: F401

import torch  # noqa: F401
from torch import nn  # noqa: F401
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # noqa: F401

from exact.core.contracts.model import IModel
from exact.impl.models.scorer_common import (
    LabelPairPooling,
    PoolingMethod,
    ScorerCommonMixin,
)
from exact.impl.models.semantic_llm import SemanticLLMMixin
from exact.llm.routing import (  # noqa: F401
    LLMRouter,
    extract_chat_text,
    extract_first_token_top_logprobs,
    parse_structured_json,
)
from exact.utils.data import read_table  # noqa: F401


class SemanticScorer(SemanticLLMMixin, ScorerCommonMixin, IModel):
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
        llm_calibration_reference_file_path: Optional[str] = None,
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
            self.device.type
            if isinstance(self.device, torch.device)
            else torch.device(self.device).type
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
        if len(raw_hosted_labels) != 2 or any(
            not str(label).strip() for label in raw_hosted_labels
        ):
            raise ValueError("hosted_decision_labels must contain exactly two non-empty labels.")
        self.hosted_decision_labels = (
            str(raw_hosted_labels[0]).strip(),
            str(raw_hosted_labels[1]).strip(),
        )
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
        self._llm_calibration_reference_pairs: set[tuple[str, str]] = set()
        self._llm_calibration_reference_sources: set[str] = set()
        self._llm_calibration_reference_fingerprint: Optional[str] = None
        if self.use_llm_calibration and not self._llm_calibration_can_apply:
            if not llm_calibration_reference_file_path:
                raise ValueError(
                    "use_llm_calibration requires a training reference file; full/test "
                    "reference labels must not be used for calibration"
                )
            reference_path = Path(llm_calibration_reference_file_path).expanduser().resolve()
            reference = read_table(reference_path)
            if reference.shape[1] < 2:
                raise ValueError(
                    f"LLM calibration training reference must have at least two columns: {reference_path}"
                )
            self._llm_calibration_reference_pairs = {
                (str(src), str(tgt))
                for src, tgt in reference.iloc[:, :2].itertuples(index=False, name=None)
            }
            self._llm_calibration_reference_sources = {
                src for src, _ in self._llm_calibration_reference_pairs
            }
            if not self._llm_calibration_reference_pairs:
                raise ValueError(f"LLM calibration training reference is empty: {reference_path}")
            reference_blob = json.dumps(
                sorted(self._llm_calibration_reference_pairs),
                separators=(",", ":"),
            )
            self._llm_calibration_reference_fingerprint = hashlib.sha1(
                reference_blob.encode("utf-8")
            ).hexdigest()

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
        default_cache_dir = (
            Path(cache_dir).expanduser()
            if cache_dir
            else (Path.home() / ".cache" / "exact" / "semantic_scorer")
        )
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
            if self.persist_cache_to_disk
            else None
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
        self._llm_router = LLMRouter(
            llm_profiles=llm_profiles, llm_routing=llm_routing, log=self.log
        )
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
