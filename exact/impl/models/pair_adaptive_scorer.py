from __future__ import annotations

import math
import hashlib  # noqa: F401
import json  # noqa: F401
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: F401

import torch  # noqa: F401

from exact.impl.models.pair_adaptive_channels import PairAdaptiveChannelsMixin
from exact.impl.models.pair_adaptive_evidence import PairAdaptiveEvidenceMixin
from exact.impl.models.pair_adaptive_experiments import (
    JsonExperimentArtifact,
    config_dict,
)
from exact.impl.models.selector.llm_gate import analytical_oracle_ceiling
from exact.impl.models.semantic_scorer import SemanticScorer
from exact.utils.formatting import clip01, safe_mean  # noqa: F401


class PairAdaptiveSemanticScorer(
    PairAdaptiveEvidenceMixin, PairAdaptiveChannelsMixin, SemanticScorer
):
    """
    Default pair-adaptive scorer.

    Hierarchy is ontology-native and family-aware, non-hierarchical structure is
    projection-backed, attributes are auxiliary/support-only, and the LLM
    consumes one pair brief instead of separate source/target summaries.
    """

    _clip01 = staticmethod(clip01)
    _safe_mean = staticmethod(safe_mean)

    def __init__(
        self,
        max_input_tokens_hier: int = 128,
        max_input_tokens_sim: int = 256,
        max_input_tokens_diff: int = 256,
        max_input_tokens_attr_item: int = 96,
        max_hierarchy_triples_per_family: int = 6,
        max_object_triples: int = 48,
        max_diff_triples: int = 24,
        max_attr_items: int = 12,
        hierarchical_relation_families: Optional[Dict[str, Dict[str, Any]]] = None,
        attribute_property_weights: Optional[Dict[str, float]] = None,
        hierarchy_embedding_weight: float = 0.5,
        hierarchy_support_weight: float = 0.5,
        similarity_embedding_weight: float = 0.5,
        similarity_support_weight: float = 0.5,
        similarity_per_relation_cap: int = 2,
        difference_per_relation_cap: int = 3,
        stability_factor: float = 2.0,
        attribute_information_word_cap: int = 20,
        attribute_score_floor: float = 0.5,
        uncertainty_indecision_scale: float = 2.0,
        uncertainty_disagreement_quality_power: float = 0.5,
        strsim: Optional[Dict[str, Any]] = None,
        attr: Optional[Dict[str, Any]] = None,
        hier: Optional[Dict[str, Any]] = None,
        diff: Optional[Dict[str, Any]] = None,
        lex: Optional[Dict[str, Any]] = None,
        property: Optional[Dict[str, Any]] = None,
        instance: Optional[Dict[str, Any]] = None,
        graph: Optional[Dict[str, Any]] = None,
        fusion_config: Optional[Dict[str, Any]] = None,
        llm_experiment_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.strsim_config = self._experiment_config(
            strsim,
            enabled=False,
            placement="channel",
            abbreviation="off",
            isub_weight=1.0,
            jaro_winkler_weight=1.0,
            token_set_weight=1.0,
        )
        self.attr_config = self._experiment_config(
            attr,
            enabled=False,
            polarity="support_only",
            bank="full",
            signed_property_allowlist=[],
        )
        self.hier_config = self._experiment_config(
            hier,
            enabled=False,
            mode="labels",
            siblings=False,
            depth=2,
            overlap_weight=0.5,
        )
        self.diff_config = self._experiment_config(
            diff,
            enabled=False,
            formulation="normalised",
            dump_components=False,
        )
        self.lex_config = self._experiment_config(
            lex,
            enabled=False,
            quality="margin",
            entropy_top_m=5,
        )
        self.property_config = self._experiment_config(property, enabled=False)
        self.instance_config = self._experiment_config(instance, enabled=False)
        self.graph_config = self._experiment_config(
            graph,
            mode="off",
            artifact=None,
            dump_profile=False,
        )
        self.fusion_config = self._experiment_config(
            fusion_config,
            enabled=False,
            mode="analytic_shipped",
            scope="global",
            sigma_mode="full",
            tau=0.5,
            llm_pivot=0.5,
            gamma=2.0,
            beta=0.8,
            artifact=None,
        )
        raw_llm_config = self._experiment_config(
            llm_experiment_config,
            enabled=False,
            decision={},
            gate={},
            fusion_weight="beta_u",
            fusion_artifact=None,
            exemplars="off",
            exemplar_count=0,
            distill="off",
            distill_artifact=None,
        )
        raw_llm_config["decision"] = self._experiment_config(
            raw_llm_config.get("decision"),
            mode="binary",
            probability="raw_joint",
            listwise_max_candidates=5,
            permutations=2,
            samples_per_permutation=3,
        )
        raw_llm_config["gate"] = self._experiment_config(
            raw_llm_config.get("gate"),
            mode="analytic",
            threshold=0.5,
            quantile_fraction=0.05,
            forced_sample_size=0,
            artifact=None,
        )
        configured_gate_mode = str(raw_llm_config["gate"]["mode"]).strip().lower()
        self._gate_mode_alias: Optional[str] = None
        if configured_gate_mode == "router":
            self._gate_mode_alias = "router"
            raw_llm_config["gate"]["mode"] = "learned"
        self.llm_experiment_config = raw_llm_config

        self.strsim_enabled = bool(self.strsim_config["enabled"])
        self.attr_enabled = bool(self.attr_config["enabled"])
        self.hier_enabled = bool(self.hier_config["enabled"])
        self.diff_enabled = bool(self.diff_config["enabled"])
        self.lex_enabled = bool(self.lex_config["enabled"])
        self.fusion_enabled = bool(self.fusion_config["enabled"])
        self.llm_experiment_enabled = bool(self.llm_experiment_config["enabled"])
        self._fusion_artifact: Optional[JsonExperimentArtifact] = None
        self._gate_artifact: Optional[JsonExperimentArtifact] = None
        self._validate_experiment_configs()

        if self.fusion_enabled:
            kwargs["tau"] = float(self.fusion_config["tau"])
            kwargs["tau_LLM"] = float(self.fusion_config["llm_pivot"])
            kwargs["gamma"] = float(self.fusion_config["gamma"])
            kwargs["beta"] = float(self.fusion_config["beta"])
            fusion_mode = str(self.fusion_config["mode"])
            if fusion_mode != "analytic_shipped":
                if fusion_mode == "learned_adaptive":
                    raise NotImplementedError(
                        "learned_adaptive fusion requires a fitted gating-network runtime"
                    )
                if self.fusion_config["scope"] != "global":
                    raise NotImplementedError(
                        "artifact-backed analytic_fitted/learned_global currently supports "
                        "only fusion scope='global'"
                    )
                self._fusion_artifact = JsonExperimentArtifact.load(
                    self.fusion_config.get("artifact"),
                    expected_mode=fusion_mode,
                    kind="fusion",
                )
                self._validate_fusion_artifact_contract(fusion_mode)
                if fusion_mode == "analytic_fitted":
                    parameters = self._fusion_artifact.scoped_payload(
                        "parameters", scope_key="global"
                    )
                    for name in ("tau", "gamma"):
                        if name in parameters:
                            kwargs[name] = float(parameters[name])

        gate_mode = str(self.llm_experiment_config["gate"]["mode"])
        gate_artifact = self.llm_experiment_config["gate"].get("artifact")
        if self.llm_experiment_enabled and gate_mode in {"quantile", "forced_sample"}:
            if not gate_artifact:
                raise ValueError(
                    f"LLM gate mode {gate_mode!r} requires an immutable selection artifact"
                )
        if self.llm_experiment_enabled and gate_artifact:
            self._gate_artifact = JsonExperimentArtifact.load(
                gate_artifact,
                expected_mode=self._gate_mode_alias or gate_mode,
                kind="LLM gate",
            )
            if gate_mode in {"quantile", "forced_sample"}:
                self._validate_gate_artifact_contract(gate_mode)

        self.attribute_property_weights = dict(
            attribute_property_weights
            or {
                "definition": 1.0,
                "identifier": 0.8,
                "comment": 0.6,
                "other": 0.5,
            }
        )
        self.hierarchy_embedding_weight = float(hierarchy_embedding_weight)
        self.hierarchy_support_weight = float(hierarchy_support_weight)
        self.similarity_embedding_weight = float(similarity_embedding_weight)
        self.similarity_support_weight = float(similarity_support_weight)
        self.similarity_per_relation_cap = int(similarity_per_relation_cap)
        self.difference_per_relation_cap = int(difference_per_relation_cap)
        self.stability_factor = float(stability_factor)
        self.attribute_information_word_cap = int(attribute_information_word_cap)
        self.attribute_score_floor = float(attribute_score_floor)
        self.uncertainty_indecision_scale = float(uncertainty_indecision_scale)
        self.uncertainty_disagreement_quality_power = float(uncertainty_disagreement_quality_power)
        requested_context_cap = int(kwargs.get("max_input_tokens_context", 256))
        kwargs["max_input_tokens_context"] = max(
            requested_context_cap,
            int(max_input_tokens_hier),
            int(max_input_tokens_sim),
            int(max_input_tokens_diff),
            int(max_input_tokens_attr_item),
        )
        super().__init__(**kwargs)
        self.max_input_tokens_hier = int(max_input_tokens_hier)
        self.max_input_tokens_sim = int(max_input_tokens_sim)
        self.max_input_tokens_diff = int(max_input_tokens_diff)
        self.max_input_tokens_attr_item = int(max_input_tokens_attr_item)
        self.max_hierarchy_triples_per_family = int(max_hierarchy_triples_per_family)
        self.max_object_triples = int(max_object_triples)
        self.max_diff_triples = int(max_diff_triples)
        self.max_attr_items = int(max_attr_items)
        self.hierarchical_relation_families = dict(hierarchical_relation_families or {})
        self._attached_dataset = None
        self._exact_anchor_src_to_tgt: Dict[str, set[str]] = {}
        self._exact_anchor_tgt_to_src: Dict[str, set[str]] = {}
        self._hierarchy_ic_cache: Dict[Tuple[str, str], float] = {}

    @staticmethod
    def _experiment_config(value: Any, **defaults: Any) -> Dict[str, Any]:
        merged = dict(defaults)
        merged.update(config_dict(value))
        return merged

    def _validate_experiment_configs(self) -> None:
        if self.strsim_config["placement"] not in {"channel", "folded_into_lexical"}:
            raise ValueError(f"unsupported strsim placement: {self.strsim_config['placement']!r}")
        if self.strsim_config["abbreviation"] not in {"off", "initialism"}:
            raise ValueError(
                f"unsupported strsim abbreviation mode: {self.strsim_config['abbreviation']!r}"
            )
        for key in ("isub_weight", "jaro_winkler_weight", "token_set_weight"):
            if float(self.strsim_config[key]) < 0.0:
                raise ValueError(f"strsim {key} must be non-negative")
        if self.attr_config["polarity"] not in {"support_only", "signed"}:
            raise ValueError(f"unsupported attr polarity: {self.attr_config['polarity']!r}")
        if self.attr_config["bank"] not in {"full", "attrs_labels", "attrs_only"}:
            raise ValueError(f"unsupported attr bank: {self.attr_config['bank']!r}")
        if self.attr_config["polarity"] == "signed" and not self.attr_config.get(
            "signed_property_allowlist"
        ):
            raise ValueError("signed attribute polarity requires a descriptor property allowlist")
        if self.hier_config["mode"] not in {"labels", "labels_overlap"}:
            raise ValueError(f"unsupported hierarchy mode: {self.hier_config['mode']!r}")
        if self.diff_config["formulation"] not in {
            "normalised",
            "absolute",
            "asymmetric",
            "off",
        }:
            raise ValueError(
                f"unsupported difference formulation: {self.diff_config['formulation']!r}"
            )
        if self.lex_config["quality"] not in {
            "margin",
            "entropy",
            "encoder_agreement",
            "constant",
        }:
            raise ValueError(f"unsupported lexical quality: {self.lex_config['quality']!r}")
        if self.fusion_config["sigma_mode"] not in {
            "full",
            "constant_q",
            "no_sharpening",
            "suppression_only",
            "uniform",
        }:
            raise ValueError(f"unsupported fusion sigma mode: {self.fusion_config['sigma_mode']!r}")
        graph_mode = self.graph_config["mode"]
        if graph_mode not in {"off", "inductive", "transductive", "graph_only"}:
            raise ValueError(f"unsupported graph channel mode: {graph_mode!r}")
        if graph_mode != "off":
            raise NotImplementedError(
                "E23 graph scoring needs a normalized graph-feature/artifact contract; "
                "refusing to execute this arm as the graph-off baseline"
            )
        if bool(self.property_config.get("enabled")) or bool(self.instance_config.get("enabled")):
            raise NotImplementedError(
                "property/instance evidence-group switches are not implemented in this scorer"
            )
        if self.llm_experiment_enabled:
            decision_mode = str(self.llm_experiment_config["decision"]["mode"])
            if decision_mode not in {"binary", "listwise", "listwise_sc"}:
                raise ValueError(f"unsupported LLM decision mode: {decision_mode!r}")
            if decision_mode != "binary":
                raise NotImplementedError(
                    "E07 listwise prompt/probability primitives are available, but the "
                    "candidate-group backend runtime is not; refusing to run binary decisions "
                    f"for decision mode {decision_mode!r}"
                )
            if self.llm_experiment_config.get("exemplars", "off") != "off":
                raise NotImplementedError(
                    "E21 k-NN exemplars require a training-only retrieval artifact and "
                    "leakage audit; refusing to run the zero-shot prompt as that arm"
                )
            if self.llm_experiment_config.get("distill", "off") != "off":
                raise NotImplementedError(
                    "E21 distillation requires a fitted student bound to a pinned LLM "
                    "fingerprint; refusing to call the real LLM as that arm"
                )
            gate_mode = str(self.llm_experiment_config["gate"]["mode"])
            if gate_mode not in {
                "off",
                "analytic",
                "quantile",
                "forced_sample",
                "oracle",
                "learned",
            }:
                raise ValueError(f"unsupported LLM gate mode: {gate_mode!r}")
            if gate_mode == "learned":
                raise NotImplementedError(
                    "E21 learned gating requires a counterfactual-training artifact bound to "
                    "the complete pinned LLM fingerprint; the legacy 'router' linear head is "
                    "not a scientifically valid substitute"
                )
        if self.llm_experiment_enabled and self.llm_experiment_config["fusion_weight"] != "beta_u":
            raise NotImplementedError(
                "learned LLM mixing requires an immutable fitted fusion artifact runtime"
            )

    def attach_dataset(self, dataset: Any) -> None:
        super().attach_dataset(dataset)
        self._hierarchy_ic_cache = {}
        self._exact_anchor_src_to_tgt = {}
        self._exact_anchor_tgt_to_src = {}
        exact_matches = getattr(dataset, "exact_matches", None)
        if exact_matches is not None and hasattr(exact_matches, "columns"):
            if {"Src", "Tgt"}.issubset(exact_matches.columns):
                rows = exact_matches[["Src", "Tgt"]].dropna().itertuples(index=False, name=None)
                for src_iri, tgt_iri in rows:
                    src_key = str(src_iri)
                    tgt_key = str(tgt_iri)
                    self._exact_anchor_src_to_tgt.setdefault(src_key, set()).add(tgt_key)
                    self._exact_anchor_tgt_to_src.setdefault(tgt_key, set()).add(src_key)
        dataset_signature = getattr(dataset, "dataset_signature", None)
        if self._fusion_artifact is not None:
            self._fusion_artifact.validate_dataset(dataset_signature, required=True)
        if self._gate_artifact is not None:
            self._gate_artifact.validate_dataset(dataset_signature, required=True)

    def _validate_fusion_artifact_contract(self, mode: str) -> None:
        artifact = self._fusion_artifact
        if artifact is None:
            raise RuntimeError("fusion artifact validation requires a loaded artifact")
        artifact.require_fields(
            "dataset_signature",
            "dataset_lock_sha256",
            "seed",
            "feature_schema",
            "negative_label_policy",
            kind="fusion",
        )
        pool_fingerprint = artifact.payload.get(
            "candidate_pool_fingerprint",
            artifact.payload.get("pool_fingerprint"),
        )
        if not isinstance(pool_fingerprint, str) or not pool_fingerprint.strip():
            raise ValueError("fusion artifact is missing candidate_pool_fingerprint")
        feature_schema = artifact.payload["feature_schema"]
        if not isinstance(feature_schema, list) or not all(
            isinstance(value, str) and value for value in feature_schema
        ):
            raise ValueError("fusion artifact feature_schema must be a non-empty string array")
        try:
            int(artifact.payload["seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError("fusion artifact seed must be an integer") from exc
        if mode == "analytic_fitted":
            parameters = artifact.scoped_payload("parameters", scope_key="global")
            tau = float(parameters.get("tau", self.fusion_config["tau"]))
            gamma = float(parameters.get("gamma", self.fusion_config["gamma"]))
            if not 0.0 <= tau <= 1.0 or gamma < 0.0:
                raise ValueError("analytic_fitted fusion artifact has invalid tau or gamma")
            if (
                "beta" in parameters
                and abs(float(parameters["beta"]) - float(self.fusion_config["beta"])) > 1.0e-12
            ):
                raise ValueError(
                    "analytic_fitted fusion artifact may not refit beta; beta is pinned by E10"
                )
            multipliers = parameters.get("multipliers")
            if not isinstance(multipliers, dict) or not multipliers:
                raise ValueError("analytic_fitted fusion artifact needs channel multipliers")
            fitted_values = [
                float(value) for name, value in multipliers.items() if name != "default"
            ]
            if not fitted_values or any(value < 0.0 for value in fitted_values):
                raise ValueError("analytic_fitted channel multipliers must be non-negative")
            if abs(sum(fitted_values) / len(fitted_values) - 1.0) > 1.0e-6:
                raise ValueError("analytic_fitted channel multipliers must have arithmetic mean 1")
        elif mode == "learned_global":
            weights = artifact.scoped_payload("weights", scope_key="global")
            values = [float(value) for value in weights.values()]
            if not values or any(value < 0.0 for value in values) or max(values) <= 0.0:
                raise ValueError("learned_global fusion weights must be non-negative and non-zero")

    def _validate_gate_artifact_contract(self, mode: str) -> None:
        artifact = self._gate_artifact
        if artifact is None:
            raise RuntimeError("gate artifact validation requires a loaded artifact")
        payload = artifact.payload
        artifact.require_fields(
            "dataset_signature",
            "task_id",
            "entity_kind",
            "pairs",
            kind="LLM gate",
        )
        raw_pairs = payload["pairs"]
        if not isinstance(raw_pairs, list) or any(
            not isinstance(pair, list) or len(pair) != 2 for pair in raw_pairs
        ):
            raise ValueError("LLM gate artifact pairs must be a two-column array")
        selected = [(str(pair[0]), str(pair[1])) for pair in raw_pairs]
        if len(selected) != len(set(selected)):
            raise ValueError("LLM gate artifact pairs must be unique")
        if mode == "forced_sample":
            artifact.require_fields("sample_size", "seed", "selection_rule", kind="LLM gate")
            sample_size = int(payload["sample_size"])
            if sample_size != len(selected):
                raise ValueError("forced_sample artifact sample_size must equal its pair count")
            try:
                int(payload["seed"])
            except (TypeError, ValueError) as exc:
                raise ValueError("forced_sample artifact seed must be an integer") from exc
            if payload["selection_rule"] != "sha256(seed, source_iri, target_iri)":
                raise ValueError("forced_sample artifact has an unsupported selection rule")

            configured_size = int(self.llm_experiment_config["gate"].get("forced_sample_size", 0))
            if configured_size and configured_size != sample_size:
                raise ValueError(
                    "forced_sample artifact pair count does not match configured sample size"
                )
            return

        artifact.require_fields(
            "fraction",
            "row_count",
            "selected_count",
            "threshold",
            "boundary_ids",
            "tie_rule",
            kind="LLM gate",
        )
        fraction = float(payload["fraction"])
        row_count = int(payload["row_count"])
        selected_count = int(payload["selected_count"])
        threshold = float(payload["threshold"])
        if (
            not 0.0 < fraction <= 1.0
            or row_count < 1
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
            or selected_count != math.ceil(fraction * row_count)
            or selected_count != len(selected)
        ):
            raise ValueError("quantile gate artifact has inconsistent fitted counts or cutoff")
        configured_fraction = float(self.llm_experiment_config["gate"]["quantile_fraction"])
        if abs(configured_fraction - fraction) > 1.0e-12:
            raise ValueError("quantile gate artifact fraction does not match configuration")
        if payload["tie_rule"] != "(-U, source_iri, target_iri)":
            raise ValueError("quantile gate artifact has an unsupported tie rule")
        boundary = payload["boundary_ids"]
        if not isinstance(boundary, dict) or boundary.get("selected_last") != raw_pairs[-1]:
            raise ValueError("quantile gate artifact boundary IDs are inconsistent")

    def _fusion_multiplier(self, channel: str) -> float:
        if self._fusion_artifact is None:
            return 1.0
        mode = str(self.fusion_config["mode"])
        field = "parameters" if mode == "analytic_fitted" else "weights"
        payload = self._fusion_artifact.scoped_payload(field, scope_key="global")
        values = payload.get("multipliers", payload) if field == "parameters" else payload
        if not isinstance(values, dict):
            raise ValueError(
                f"fusion artifact {self._fusion_artifact.path} has invalid {field!r} payload"
            )
        fallback = 1.0 if mode == "analytic_fitted" else 0.0
        value = float(values.get(channel, values.get("default", fallback)))
        if value < 0.0:
            raise ValueError(f"fusion weight for {channel!r} must be non-negative")
        return value

    def _sigma_authority(
        self,
        score: torch.Tensor,
        quality: torch.Tensor,
        active: torch.Tensor,
        *,
        channel: str,
    ) -> torch.Tensor:
        """Return channel authority for E19/E26 while preserving the shipped default."""

        active_float = active.to(dtype=score.dtype)
        fusion_mode = str(self.fusion_config["mode"]) if self.fusion_enabled else "analytic_shipped"
        if fusion_mode == "learned_global":
            return active_float * quality * self._fusion_multiplier(channel)

        sigma_mode = str(self.fusion_config["sigma_mode"]) if self.fusion_enabled else "full"
        margin = (score - self.tau).abs()
        if sigma_mode == "full":
            authority = active_float * quality * margin.pow(self.gamma)
        elif sigma_mode == "constant_q":
            authority = active_float * margin.pow(self.gamma)
        elif sigma_mode == "no_sharpening":
            authority = active_float * quality
        elif sigma_mode == "suppression_only":
            authority = active_float
        elif sigma_mode == "uniform":
            authority = torch.ones_like(score)
        else:  # guarded in _validate_experiment_configs
            raise ValueError(f"unsupported fusion sigma mode: {sigma_mode!r}")
        return authority * self._fusion_multiplier(channel)

    def _llm_gate_mask(
        self,
        *,
        U_ind: torch.Tensor,
        U_dis: torch.Tensor,
        U: torch.Tensor,
        S_base: torch.Tensor,
        q_label: torch.Tensor,
        Q_struct: torch.Tensor,
        src_iris: Sequence[str],
        tgt_iris: Sequence[str],
        label: Optional[Sequence[float]],
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Compute a frozen, per-pair gate and an auditable diagnostic row."""

        if not self.llm_experiment_enabled:
            mask = U >= self.tau_LLM
            return mask, []

        gate = self.llm_experiment_config["gate"]
        mode = str(gate["mode"])
        statistic_name = "U"
        statistic = U
        threshold = float(gate["threshold"])
        extra: Dict[str, Any] = {}
        if mode == "off":
            statistic_name = "disabled"
            statistic = torch.zeros_like(U)
            threshold = 1.0
            extra["llm_disabled"] = True
        elif mode == "analytic":
            pass
        elif mode == "quantile":
            assert self._gate_artifact is not None
            payload = self._gate_artifact.payload
            selected = {(str(pair[0]), str(pair[1])) for pair in payload["pairs"]}
            statistic_name = "fitted_quantile_membership"
            statistic = torch.tensor(
                [float((str(src), str(tgt)) in selected) for src, tgt in zip(src_iris, tgt_iris)],
                dtype=torch.float32,
                device=self.device,
            )
            threshold = 0.5
            extra["target_fraction"] = float(payload["fraction"])
            extra["threshold_source"] = "artifact_exact_pairs"
            extra["fitted_cutoff"] = float(payload["threshold"])
            extra["selected_count"] = int(payload["selected_count"])
            extra["boundary_ids"] = dict(payload["boundary_ids"])
        elif mode == "forced_sample":
            assert self._gate_artifact is not None
            pairs = self._gate_artifact.payload.get("pairs")
            if not isinstance(pairs, list):
                raise ValueError("forced_sample artifact must contain a 'pairs' array")
            selected = {
                (str(pair[0]), str(pair[1]))
                for pair in pairs
                if isinstance(pair, list) and len(pair) == 2
            }
            expected_size = int(gate["forced_sample_size"])
            if expected_size and len(selected) != expected_size:
                raise ValueError(
                    "forced_sample artifact pair count does not match configured sample size"
                )
            statistic_name = "frozen_sample_membership"
            statistic = torch.tensor(
                [float((str(src), str(tgt)) in selected) for src, tgt in zip(src_iris, tgt_iris)],
                dtype=torch.float32,
                device=self.device,
            )
            threshold = 0.5
            extra["sample_size"] = len(selected)
        elif mode == "oracle":
            if label is None or len(label) != len(src_iris):
                raise ValueError("oracle LLM gate requires one reference label per scored pair")
            truth = torch.tensor(label, dtype=torch.float32, device=self.device) >= 0.5
            predicted = S_base >= self.threshold
            statistic_name = "baseline_error"
            statistic = (truth != predicted).to(torch.float32)
            threshold = 0.5
            extra["oracle_diagnostic"] = True
        elif mode == "learned":
            assert self._gate_artifact is not None
            payload = self._gate_artifact.payload
            weights = payload.get("weights")
            if not isinstance(weights, dict):
                raise ValueError("learned gate artifact must contain a 'weights' object")
            features = {
                "U_ind": U_ind,
                "U_dis": U_dis,
                "U": U,
                "S_base": S_base,
                "q_label": q_label,
                "Q_struct": Q_struct,
            }
            linear = torch.full_like(U, float(payload.get("bias", 0.0)))
            for name, weight in weights.items():
                if name not in features:
                    raise ValueError(f"learned gate artifact references unknown feature {name!r}")
                linear = linear + float(weight) * features[name]
            statistic_name = "learned_probability"
            statistic = torch.sigmoid(linear)
            threshold = float(payload.get("threshold", threshold))
            extra["artifact"] = dict(self._gate_artifact.provenance)
        else:
            raise ValueError(f"unsupported LLM gate mode: {mode!r}")

        mask = statistic >= threshold
        rows = []
        for idx in range(len(src_iris)):
            rows.append(
                {
                    "mode": mode,
                    "statistic_name": statistic_name,
                    "statistic": float(statistic[idx]),
                    "threshold": threshold,
                    "U_ind": float(U_ind[idx]),
                    "U_dis": float(U_dis[idx]),
                    "U": float(U[idx]),
                    "would_route": bool(mask[idx]),
                    **extra,
                }
            )
        return mask, rows

    def _runtime_fingerprint_payload(
        self,
        generate_llm_rationales_override: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload = super()._runtime_fingerprint_payload(
            generate_llm_rationales_override=generate_llm_rationales_override
        )
        payload["pair_adaptive_channels"] = {
            "hierarchy_embedding_weight": self.hierarchy_embedding_weight,
            "hierarchy_support_weight": self.hierarchy_support_weight,
            "similarity_embedding_weight": self.similarity_embedding_weight,
            "similarity_support_weight": self.similarity_support_weight,
            "similarity_per_relation_cap": self.similarity_per_relation_cap,
            "difference_per_relation_cap": self.difference_per_relation_cap,
            "stability_factor": self.stability_factor,
            "attribute_property_weights": self.attribute_property_weights,
            "attribute_information_word_cap": self.attribute_information_word_cap,
            "attribute_score_floor": self.attribute_score_floor,
            "uncertainty_indecision_scale": self.uncertainty_indecision_scale,
            "uncertainty_disagreement_quality_power": (self.uncertainty_disagreement_quality_power),
            "experiments": {
                "strsim": self.strsim_config,
                "attr": self.attr_config,
                "hier": self.hier_config,
                "diff": self.diff_config,
                "lex": self.lex_config,
                "property": self.property_config,
                "instance": self.instance_config,
                "graph": self.graph_config,
                "fusion": self.fusion_config,
                "fusion_effective": {
                    "tau": self.tau,
                    "gamma": self.gamma,
                    "beta": self.beta,
                },
                "llm": self.llm_experiment_config,
                "fusion_artifact": (
                    dict(self._fusion_artifact.provenance)
                    if self._fusion_artifact is not None
                    else None
                ),
                "gate_artifact": (
                    dict(self._gate_artifact.provenance)
                    if self._gate_artifact is not None
                    else None
                ),
            },
        }
        return payload

    def _brief_prompt(self, src_label: str, tgt_label: str, packet: str) -> Dict[str, str]:
        return {
            "system": "You are an ontology alignment analyst that returns strict JSON.",
            "user": (
                "Condense the pair evidence below into one compact ontology-alignment brief. "
                'Return exactly one JSON object with one key: "summary".\n'
                "The brief must keep these section titles in plain text:\n"
                "Label evidence\nHierarchy evidence\nRelational similarity evidence\n"
                "Distinctive conflicting evidence\nAuxiliary attribute evidence\n\n"
                f"Source entity: {src_label}\n"
                f"Target entity: {tgt_label}\n\n"
                f"Evidence packet:\n{packet}\n\n"
                "Return only JSON."
            ),
        }

    def _generate_briefs_uncached(
        self,
        prompts: List[Dict[str, str]],
        resolved_backend,
    ) -> List[str]:
        if not prompts:
            return []
        if resolved_backend.backend == "openrouter":
            profile = self._llm_router.profiles.get(resolved_backend.profile_name or "")
            if profile is None:
                raise RuntimeError("OpenRouter summary profile was resolved but not found.")
            return self._run_hosted_chat_prompts(
                prompts=prompts,
                profile=profile,
                max_tokens=self.max_new_tokens_llm,
                temperature=self.llm_temperature,
                top_p=self.llm_top_p,
                concurrency=self.llm_summary_batch_size,
            )
        self._ensure_local_llm()
        rendered = [self._render_llm_prompt(prompt) for prompt in prompts]
        outputs = [""] * len(rendered)
        chunk = self.llm_summary_batch_size or len(rendered)
        chunk = chunk if chunk > 0 else len(rendered)
        for start in range(0, len(rendered), chunk):
            end = min(start + chunk, len(rendered))
            enc = self.llm_tok(
                rendered[start:end],
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

    def _decision_prompt(
        self, src_label: str, tgt_label: str, src_summary: str, tgt_summary: str
    ) -> Dict[str, str]:
        pair_brief = src_summary
        return {
            "system": "You are an ontology alignment expert.",
            "user": (
                "Determine whether the following two ontology entities refer to the same concept.\n"
                "Answer with a single token: Yes or No.\n\n"
                f"Source entity: {src_label}\n"
                f"Target entity: {tgt_label}\n\n"
                f"Pair brief:\n{pair_brief}\n\n"
                "Answer:"
            ),
        }

    def _hosted_decision_prompt(
        self,
        src_label: str,
        tgt_label: str,
        src_summary: str,
        tgt_summary: str,
    ) -> Dict[str, str]:
        positive_label, negative_label = self.hosted_decision_labels
        pair_brief = src_summary
        return {
            "system": "You are a binary classifier for ontology alignment.",
            "user": (
                f"Return exactly one token: {positive_label} or {negative_label}.\n"
                f"{positive_label} = the source and target entities are equivalent.\n"
                f"{negative_label} = the source and target entities are not equivalent.\n\n"
                f"Source entity: {src_label}\n"
                f"Target entity: {tgt_label}\n\n"
                f"Pair brief:\n{pair_brief}"
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
        pair_brief = src_summary
        context_block = ""
        if decision_context:
            context_block = f"\nFinal alignment context:\n{decision_context}\n\n"
        return {
            "system": "You are an ontology alignment expert.",
            "user": (
                "Write a concise but specific rationale explaining the final alignment decision. "
                "Use only the pair brief and final alignment context below. Mention supporting and conflicting evidence when available. "
                "When final alignment context is present, explain whether the pair was kept or rejected after candidate-set selection and cardinality filtering. "
                'Return exactly one JSON object with one key: "rationale".\n\n'
                f"Source entity: {src_label}\n"
                f"Target entity: {tgt_label}\n"
                f"Final decision: {decision}\n\n"
                f"Pair brief:\n{pair_brief}\n\n"
                f"{context_block}"
                "Return only JSON."
            ),
        }

    @torch.inference_mode()
    def generate_pair_briefs_batched(
        self,
        src_labels: List[str],
        tgt_labels: List[str],
        evidence_packets: List[str],
    ) -> List[str]:
        if not self.use_llm:
            return list(evidence_packets)
        if not src_labels:
            return []
        summary_backend = self._llm_router.resolve_task("summary")
        self._last_summary_backend_meta = self._resolved_backend_metadata(summary_backend)
        outputs = [""] * len(src_labels)
        pending: Dict[str, Dict[str, Any]] = {}

        short_threshold = int(self.max_total_tokens_llm_summary * 0.75)
        for idx, (src_label, tgt_label, packet) in enumerate(
            zip(src_labels, tgt_labels, evidence_packets)
        ):
            packet = self._normalize_text(packet)
            if not packet:
                outputs[idx] = ""
                continue
            if len(packet.split()) <= short_threshold:
                outputs[idx] = packet
                continue
            key = self._brief_key(src_label, tgt_label, packet)
            cached = self._summary_cache.get(key)
            if cached is not None:
                outputs[idx] = cached
                continue
            entry = pending.setdefault(
                key,
                {
                    "src_label": src_label,
                    "tgt_label": tgt_label,
                    "packet": packet,
                    "indices": [],
                },
            )
            entry["indices"].append(idx)

        if pending:
            pending_keys = list(pending.keys())
            prompts = [
                self._brief_prompt(
                    pending[key]["src_label"],
                    pending[key]["tgt_label"],
                    pending[key]["packet"],
                )
                for key in pending_keys
            ]
            generated = self._generate_briefs_uncached(prompts, summary_backend)
            for key, brief in zip(pending_keys, generated):
                clean = self._parse_structured_text(brief, "summary", self._clean_summary_text)
                self._cache_store(self._summary_cache, key, clean, self.max_cached_summaries)
                for idx in pending[key]["indices"]:
                    outputs[idx] = clean

        self._record_summary_stats(outputs)
        return outputs

    def generate_final_rationales_for_records(
        self,
        records: List[Dict[str, Any]],
        progress_callback=None,
        completion_callback=None,
    ) -> List[str]:
        if not (self.use_llm and self.generate_llm_rationales):
            return ["" for _ in records]
        src_labels: List[str] = []
        tgt_labels: List[str] = []
        pair_briefs: List[str] = []
        decisions: List[str] = []
        decision_contexts: List[str] = []
        for record in records:
            labels = record.get("selected_labels") or {}
            prediction = record.get("prediction") or {}
            src_labels.append(str(labels.get("source", "")))
            tgt_labels.append(str(labels.get("target", "")))
            pair_briefs.append(str(record.get("llm_pair_brief", "")))
            decisions.append(str(prediction.get("rationale_decision_label", "")))
            decision_contexts.append(self._final_alignment_context_for_rationale(record))
        return self.generate_rationales_batched(
            src_labels=src_labels,
            tgt_labels=tgt_labels,
            src_summaries=pair_briefs,
            tgt_summaries=["" for _ in pair_briefs],
            decisions=decisions,
            decision_contexts=decision_contexts,
            progress_callback=progress_callback,
            completion_callback=completion_callback,
        )

    def _build_evidence_packet(
        self,
        src_label: str,
        tgt_label: str,
        hierarchy_payloads: Dict[str, Dict[str, Any]],
        sim_payload: Dict[str, Any],
        diff_payload: Dict[str, Any],
        attr_payload: Dict[str, Any],
    ) -> str:
        lines = [
            "Label evidence",
            f"Source label: {src_label}",
            f"Target label: {tgt_label}",
            "",
            "Hierarchy evidence",
        ]
        active_hier = False
        for family, payload in hierarchy_payloads.items():
            src_sentences = payload.get("src_sentences", [])
            tgt_sentences = payload.get("tgt_sentences", [])
            if not src_sentences and not tgt_sentences:
                continue
            active_hier = True
            lines.append(
                f"{self._family_display_name(family)} support (score={payload.get('score', self.tau):.3f})"
            )
            for sentence in src_sentences[:3]:
                lines.append(f"Source: {sentence}")
            for sentence in tgt_sentences[:3]:
                lines.append(f"Target: {sentence}")
        if not active_hier:
            lines.append("No hierarchy evidence selected.")

        lines.extend(["", "Relational similarity evidence"])
        if sim_payload.get("src_sentences") or sim_payload.get("tgt_sentences"):
            lines.append(f"Similarity score={sim_payload.get('score', self.tau):.3f}")
            for sentence in list(sim_payload.get("src_sentences", []))[:4]:
                lines.append(f"Source: {sentence}")
            for sentence in list(sim_payload.get("tgt_sentences", []))[:4]:
                lines.append(f"Target: {sentence}")
        else:
            lines.append("No non-hierarchical relational similarity evidence selected.")

        lines.extend(["", "Distinctive conflicting evidence"])
        if diff_payload.get("src_sentences") or diff_payload.get("tgt_sentences"):
            lines.append(
                f"Difference compatibility score={diff_payload.get('score', self.tau):.3f}"
            )
            for sentence in list(diff_payload.get("src_sentences", []))[:4]:
                lines.append(f"Source-only signal: {sentence}")
            for sentence in list(diff_payload.get("tgt_sentences", []))[:4]:
                lines.append(f"Target-only signal: {sentence}")
        else:
            lines.append("No distinctive conflicting evidence selected.")

        lines.extend(["", "Auxiliary attribute evidence"])
        if attr_payload.get("src_selected") or attr_payload.get("tgt_selected"):
            lines.append(f"Attribute support score={attr_payload.get('score', self.tau):.3f}")
            for item in list(attr_payload.get("src_selected", []))[:4]:
                lines.append(f"Source attribute: {item.get('text', '')}")
            for item in list(attr_payload.get("tgt_selected", []))[:4]:
                lines.append(f"Target attribute: {item.get('text', '')}")
        else:
            lines.append("No attribute evidence selected.")
        return "\n".join(line for line in lines if line is not None).strip()

    def _uncertainty_components(
        self,
        S_base: torch.Tensor,
        s_label: torch.Tensor,
        S_struct: torch.Tensor,
        q_label: torch.Tensor,
        Q_struct: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute configurable indecision and cross-channel disagreement."""

        indecision = (
            self.uncertainty_indecision_scale * (self.tau - (S_base - self.tau).abs())
        ).clamp(0.0, 1.0)
        disagreement = (
            (q_label * Q_struct).clamp_min(0.0).pow(self.uncertainty_disagreement_quality_power)
            * (s_label - S_struct).abs()
        ).clamp(0.0, 1.0)
        return indecision, disagreement

    def _experiment_diagnostics_for_pair(
        self,
        payload: Dict[str, Any],
        *,
        gate_diagnostic: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        if self.strsim_enabled:
            diagnostics["string_similarity"] = dict(payload["strsim"])
        if self.attr_enabled:
            diagnostics["attribute"] = {
                key: payload["attr"].get(key)
                for key in (
                    "bank",
                    "polarity",
                    "identifier_disagreement",
                    "identifier_comparable_groups",
                    "identifier_groups",
                    "identifier_support",
                )
            }
        if self.hier_enabled:
            diagnostics["hierarchy"] = {
                family: {
                    key: family_payload.get(key)
                    for key in (
                        "label_score",
                        "ancestor_overlap",
                        "anchor_coverage",
                        "sibling_conflict",
                        "sibling_coverage",
                    )
                }
                for family, family_payload in payload["hierarchy"].items()
            }
        if self.diff_enabled and bool(self.diff_config.get("dump_components")):
            diagnostics["difference"] = {
                key: payload["diff"].get(key)
                for key in (
                    "formulation",
                    "n_triples_src",
                    "n_triples_tgt",
                    "unsupported_mass_src",
                    "unsupported_mass_tgt",
                    "unsupported_mean_src",
                    "unsupported_mean_tgt",
                    "diff_absolute",
                    "c_x",
                    "c_y",
                    "diff_pivot_reason",
                )
            }
        if self.lex_enabled or self.fusion_enabled:
            diagnostics["quality_components"] = {
                "label": dict(payload["label_quality"]),
                "hierarchy": {
                    family: {
                        key: family_payload.get(key)
                        for key in (
                            "coverage",
                            "strength",
                            "specificity",
                            "embedding",
                            "anchor_coverage",
                        )
                    }
                    for family, family_payload in payload["hierarchy"].items()
                },
                "similarity": {
                    key: payload["sim"].get(key)
                    for key in ("coverage", "strength", "stability", "embedding")
                },
                "difference": {
                    key: payload["diff"].get(key)
                    for key in ("coverage", "strength", "stability", "conflict")
                },
                "attribute": {
                    key: payload["attr"].get(key)
                    for key in ("coverage", "informativeness", "stability")
                },
                "sigma_mode": (self.fusion_config["sigma_mode"] if self.fusion_enabled else "full"),
            }
        if gate_diagnostic is not None:
            diagnostics["llm_gate"] = dict(gate_diagnostic)
        if self.fusion_enabled:
            diagnostics["fusion"] = {
                "mode": self.fusion_config["mode"],
                "scope": self.fusion_config["scope"],
                "sigma_mode": self.fusion_config["sigma_mode"],
                "effective_parameters": {
                    "tau": self.tau,
                    "gamma": self.gamma,
                    "beta": self.beta,
                },
                "artifact": (
                    dict(self._fusion_artifact.provenance)
                    if self._fusion_artifact is not None
                    else None
                ),
            }
        return diagnostics

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
        n_pairs = len(src_label_lists)
        assert len(tgt_label_lists) == n_pairs
        assert len(src_iris) == n_pairs and len(tgt_iris) == n_pairs
        dataset = self._attached_dataset
        if dataset is None:
            raise RuntimeError("PairAdaptiveSemanticScorer requires an attached dataset.")

        self._log_once(
            "pair_adaptive_inference_mode",
            (
                "Pair-adaptive scoring assembles pair-specific evidence inside inference batches "
                "from per-entity ontology pools (hierarchy, non-hierarchical triples, attributes)."
            ),
            "info",
        )

        src_unique_iris = list(dict.fromkeys(src_iris))
        tgt_unique_iris = list(dict.fromkeys(tgt_iris))
        cache_probe = getattr(dataset, "has_entity_features_cached", None)
        src_cache_hits = sum(
            1 for iri in src_unique_iris if callable(cache_probe) and bool(cache_probe(iri, "src"))
        )
        tgt_cache_hits = sum(
            1 for iri in tgt_unique_iris if callable(cache_probe) and bool(cache_probe(iri, "tgt"))
        )
        src_feature_map = {iri: dataset.get_entity_features(iri, "src") for iri in src_unique_iris}
        tgt_feature_map = {iri: dataset.get_entity_features(iri, "tgt") for iri in tgt_unique_iris}
        self._computed_llm_calibration = None
        self._calibration_messages = []
        self._last_summary_backend_meta = {}
        self._last_decision_backend_meta = {}
        self._last_rationale_backend_meta = {}

        s_label, q_label, best_pairs, label_quality_payloads = self._score_label_channel(
            src_label_lists, tgt_label_lists
        )
        label_active = torch.tensor(
            [
                bool(self.use_lexical and src_labels and tgt_labels)
                for src_labels, tgt_labels in zip(src_label_lists, tgt_label_lists)
            ],
            dtype=torch.bool,
            device=self.device,
        )
        s_strsim = torch.full((n_pairs,), float(self.tau), device=self.device)
        q_strsim = torch.zeros(n_pairs, device=self.device)
        strsim_payloads = [
            {"active": False, "winner": None, "components": {}} for _ in range(n_pairs)
        ]
        strsim_active = torch.zeros(n_pairs, dtype=torch.bool, device=self.device)
        if self.strsim_enabled:
            s_strsim, q_strsim, strsim_payloads = self._score_string_channel(
                src_label_lists, tgt_label_lists
            )
            strsim_active = torch.tensor(
                [bool(payload.get("active")) for payload in strsim_payloads],
                dtype=torch.bool,
                device=self.device,
            )

        needs_inner_lexical_fusion = self.strsim_enabled or (
            self.fusion_enabled
            and self.fusion_config["mode"] in {"analytic_fitted", "learned_global"}
        )
        sig_label_inner = (
            self._sigma_authority(s_label, q_label, label_active, channel="label")
            if needs_inner_lexical_fusion
            else torch.zeros(n_pairs, device=self.device)
        )
        sig_strsim_inner = torch.zeros(n_pairs, device=self.device)
        lex_label_weight = torch.ones(n_pairs, device=self.device)
        lex_strsim_weight = torch.zeros(n_pairs, device=self.device)
        S_lex = s_label
        Q_lex = q_label
        lex_active = label_active
        if self.strsim_enabled and self.strsim_config["placement"] == "folded_into_lexical":
            strsim_wins = strsim_active & (~label_active | (s_strsim > s_label))
            S_lex = torch.where(strsim_wins, s_strsim, s_label)
            Q_lex = torch.where(strsim_wins, q_strsim, q_label)
            lex_active = label_active | strsim_active
            lex_label_weight = (~strsim_wins).to(torch.float32)
            lex_strsim_weight = strsim_wins.to(torch.float32)
        elif self.strsim_enabled:
            sig_strsim_inner = self._sigma_authority(
                s_strsim, q_strsim, strsim_active, channel="strsim"
            )
            lex_sigma_sum = sig_label_inner + sig_strsim_inner
            lex_label_weight = torch.where(
                lex_sigma_sum > 1.0e-8,
                sig_label_inner / lex_sigma_sum.clamp_min(1.0e-8),
                (~(~label_active & strsim_active)).to(torch.float32),
            )
            lex_strsim_weight = torch.where(
                lex_sigma_sum > 1.0e-8,
                sig_strsim_inner / lex_sigma_sum.clamp_min(1.0e-8),
                (~label_active & strsim_active).to(torch.float32),
            )
            S_lex = torch.where(
                lex_sigma_sum > 1.0e-8,
                lex_label_weight * s_label + lex_strsim_weight * s_strsim,
                torch.full_like(s_label, float(self.tau)),
            )
            Q_lex = torch.where(
                lex_sigma_sum > 1.0e-8,
                lex_label_weight * q_label + lex_strsim_weight * q_strsim,
                torch.zeros_like(q_label),
            )
            lex_active = label_active | strsim_active
        s_label_star = S_lex

        family_names = list(self.hierarchical_relation_families.keys() or [])
        if "is_a" not in family_names:
            family_names = ["is_a"] + family_names

        pair_payloads: List[Dict[str, Any]] = []
        struct_channel_scores: Dict[str, List[float]] = {}
        struct_channel_qualities: Dict[str, List[float]] = {}
        struct_channel_active: Dict[str, List[bool]] = {}
        for family in family_names:
            struct_channel_scores[f"hier__{family}"] = []
            struct_channel_qualities[f"hier__{family}"] = []
            struct_channel_active[f"hier__{family}"] = []
        for key in ["sim_obj", "diff", "attr_aux"]:
            struct_channel_scores[key] = []
            struct_channel_qualities[key] = []
            struct_channel_active[key] = []

        for idx, (src_iri, tgt_iri) in enumerate(zip(src_iris, tgt_iris)):
            src_feats = src_feature_map[src_iri]
            tgt_feats = tgt_feature_map[tgt_iri]
            src_best_label, tgt_best_label = best_pairs[idx]
            if not src_best_label:
                src_best_label = src_feats["labels"][0] if src_feats.get("labels") else ""
            if not tgt_best_label:
                tgt_best_label = tgt_feats["labels"][0] if tgt_feats.get("labels") else ""

            hierarchy_payloads: Dict[str, Dict[str, Any]] = {}
            for family in family_names:
                family_payload = self._score_hierarchy_family(
                    family,
                    src_feats.get("hierarchy", {}).get(family, []),
                    tgt_feats.get("hierarchy", {}).get(family, []),
                    src_iri=src_iri,
                    tgt_iri=tgt_iri,
                )
                hierarchy_payloads[family] = family_payload
                struct_channel_scores[f"hier__{family}"].append(float(family_payload["score"]))
                struct_channel_qualities[f"hier__{family}"].append(float(family_payload["quality"]))
                struct_channel_active[f"hier__{family}"].append(
                    bool(family_payload.get("src_selected") or family_payload.get("tgt_selected"))
                )

            sim_payload = self._score_similarity_channel(
                src_feats.get("object_triples", []),
                tgt_feats.get("object_triples", []),
            )
            diff_payload = self._score_difference_channel(
                src_feats.get("object_triples", []),
                tgt_feats.get("object_triples", []),
                support_mat=sim_payload.get("support_matrix"),
            )
            attr_payload = self._score_attribute_channel(
                src_feats.get("attributes", []),
                tgt_feats.get("attributes", []),
                src_feats.get("labels", []),
                tgt_feats.get("labels", []),
                hierarchy_payloads,
                sim_payload,
            )
            struct_channel_scores["sim_obj"].append(float(sim_payload["score"]))
            struct_channel_qualities["sim_obj"].append(float(sim_payload["quality"]))
            struct_channel_scores["diff"].append(float(diff_payload["score"]))
            struct_channel_qualities["diff"].append(float(diff_payload["quality"]))
            struct_channel_scores["attr_aux"].append(float(attr_payload["score"]))
            struct_channel_qualities["attr_aux"].append(float(attr_payload["quality"]))
            struct_channel_active["sim_obj"].append(
                bool(sim_payload.get("src_selected") or sim_payload.get("tgt_selected"))
            )
            struct_channel_active["diff"].append(
                bool(diff_payload.get("src_selected") or diff_payload.get("tgt_selected"))
            )
            struct_channel_active["attr_aux"].append(
                bool(attr_payload.get("src_selected") or attr_payload.get("tgt_selected"))
            )

            packet = self._build_evidence_packet(
                src_best_label,
                tgt_best_label,
                hierarchy_payloads,
                sim_payload,
                diff_payload,
                attr_payload,
            )
            pair_payloads.append(
                {
                    "src_label": src_best_label,
                    "tgt_label": tgt_best_label,
                    "hierarchy": hierarchy_payloads,
                    "sim": sim_payload,
                    "diff": diff_payload,
                    "attr": attr_payload,
                    "label_quality": label_quality_payloads[idx],
                    "strsim": strsim_payloads[idx],
                    "packet": packet,
                }
            )

        src_pool_stats = self._batch_pool_stats(src_feature_map, family_names)
        tgt_pool_stats = self._batch_pool_stats(tgt_feature_map, family_names)
        pair_evidence_stats = self._batch_selected_evidence_stats(pair_payloads, family_names)

        channel_score_tensors = {
            key: torch.tensor(values, dtype=torch.float32, device=self.device)
            for key, values in struct_channel_scores.items()
        }
        channel_quality_tensors = {
            key: torch.tensor(values, dtype=torch.float32, device=self.device)
            for key, values in struct_channel_qualities.items()
        }
        channel_active_tensors = {
            key: torch.tensor(values, dtype=torch.bool, device=self.device)
            for key, values in struct_channel_active.items()
        }

        sigma_tensors: Dict[str, torch.Tensor] = {}
        for key, score_tensor in channel_score_tensors.items():
            quality_tensor = channel_quality_tensors[key]
            sigma_tensors[key] = self._sigma_authority(
                score_tensor,
                quality_tensor,
                channel_active_tensors[key],
                channel=key,
            )

        sigma_sum = torch.zeros(n_pairs, device=self.device)
        for tensor in sigma_tensors.values():
            sigma_sum = sigma_sum + tensor

        struct_weights: Dict[str, torch.Tensor] = {}
        for key, tensor in sigma_tensors.items():
            struct_weights[key] = torch.where(
                sigma_sum > 1e-8,
                tensor / sigma_sum.clamp_min(1e-8),
                torch.zeros_like(tensor),
            )

        S_struct = torch.full((n_pairs,), float(self.tau), device=self.device)
        if struct_weights:
            structural_terms = [
                struct_weights[key] * channel_score_tensors[key] for key in struct_weights
            ]
            structural_sum = torch.stack(structural_terms, dim=0).sum(dim=0)
            S_struct = torch.where(sigma_sum > 1e-8, structural_sum, S_struct)

        Q_struct = torch.zeros(n_pairs, device=self.device)
        if struct_weights:
            quality_terms = [
                struct_weights[key] * channel_quality_tensors[key] for key in struct_weights
            ]
            Q_struct = torch.where(
                sigma_sum > 1e-8,
                torch.stack(quality_terms, dim=0).sum(dim=0),
                torch.zeros_like(sigma_sum),
            )

        hier_keys = [f"hier__{family}" for family in family_names]
        hier_sigma = sum(
            (sigma_tensors[key] for key in hier_keys),
            torch.zeros(n_pairs, device=self.device),
        )
        s_hier = torch.full((n_pairs,), float(self.tau), device=self.device)
        if hier_keys:
            hier_weighted = sum(
                (sigma_tensors[key] * channel_score_tensors[key] for key in hier_keys),
                torch.zeros(n_pairs, device=self.device),
            )
            active_hier = hier_sigma > 1e-8
            if torch.any(active_hier):
                s_hier = s_hier.clone()
                s_hier[active_hier] = hier_weighted[active_hier] / hier_sigma[active_hier]

        s_sim = channel_score_tensors["sim_obj"]
        s_diff = channel_score_tensors["diff"]
        s_attr = channel_score_tensors["attr_aux"]
        q_hier = torch.zeros(n_pairs, device=self.device)
        if hier_keys:
            active_family_count = sum(
                (channel_quality_tensors[key] > 0).to(torch.float32) for key in hier_keys
            )
            family_quality_sum = sum(
                (channel_quality_tensors[key] for key in hier_keys),
                torch.zeros(n_pairs, device=self.device),
            )
            q_hier = torch.where(
                active_family_count > 0,
                family_quality_sum / active_family_count.clamp_min(1.0),
                torch.zeros_like(family_quality_sum),
            )

        fusion_mode = str(self.fusion_config["mode"]) if self.fusion_enabled else "analytic_shipped"
        if fusion_mode in {"analytic_fitted", "learned_global"}:
            if self.strsim_enabled and self.strsim_config["placement"] == "channel":
                sig_lex = sig_label_inner + sig_strsim_inner
            else:
                sig_lex = self._sigma_authority(S_lex, Q_lex, lex_active, channel="label")
            sig_struct = sigma_sum
        else:
            sig_lex = self._sigma_authority(S_lex, Q_lex, lex_active, channel="lex")
            sig_struct = self._sigma_authority(
                S_struct,
                Q_struct,
                torch.stack(list(channel_active_tensors.values()), dim=0).any(dim=0),
                channel="struct",
            )
        authority_active_lex = sig_lex > 1e-8
        authority_active_struct = sig_struct > 1e-8
        both_active = authority_active_lex & authority_active_struct
        only_lex = authority_active_lex & ~authority_active_struct
        only_struct = authority_active_struct & ~authority_active_lex

        w_struct = torch.zeros(n_pairs, device=self.device)
        if torch.any(both_active):
            w_struct = w_struct.clone()
            w_struct[both_active] = sig_struct[both_active] / (
                sig_lex[both_active] + sig_struct[both_active]
            ).clamp_min(1e-8)
        if torch.any(only_struct):
            w_struct = w_struct.clone()
            w_struct[only_struct] = 1.0

        S_base = torch.full((n_pairs,), float(self.tau), device=self.device)
        if torch.any(only_lex):
            S_base = S_base.clone()
            S_base[only_lex] = S_lex[only_lex]
        if torch.any(only_struct):
            S_base = S_base.clone()
            S_base[only_struct] = S_struct[only_struct]
        if torch.any(both_active):
            S_base = S_base.clone()
            S_base[both_active] = (1.0 - w_struct[both_active]) * S_lex[both_active] + w_struct[
                both_active
            ] * S_struct[both_active]

        U_ind, U_dis = self._uncertainty_components(
            S_base,
            S_lex,
            S_struct,
            Q_lex,
            Q_struct,
        )
        U = torch.maximum(U_ind, U_dis)
        would_route, gate_diagnostics = self._llm_gate_mask(
            U_ind=U_ind,
            U_dis=U_dis,
            U=U,
            S_base=S_base,
            q_label=Q_lex,
            Q_struct=Q_struct,
            src_iris=src_iris,
            tgt_iris=tgt_iris,
            label=label,
        )
        oracle_mode = (
            self.llm_experiment_enabled
            and str(self.llm_experiment_config["gate"]["mode"]) == "oracle"
        )
        if self.use_llm and not oracle_mode:
            w_i = (self.beta * U).clamp(0.0, 1.0)
            need_llm = would_route
        else:
            w_i = torch.zeros_like(U)
            need_llm = torch.zeros_like(U, dtype=torch.bool)

        pair_packets = [payload["packet"] for payload in pair_payloads]
        pair_briefs = list(pair_packets)
        llm_decisions = [""] * n_pairs
        llm_rationales = [""] * n_pairs
        batch_calibration_samples = 0

        decision_idxs: List[int] = []
        brief_idxs: List[int] = []
        if self.use_llm and not oracle_mode:
            decision_idxs = torch.nonzero(need_llm).flatten().tolist()
            brief_idxs = list(range(n_pairs)) if self.force_llm_summaries else list(decision_idxs)
        if self.use_llm and not oracle_mode and brief_idxs:
            brief_src = [pair_payloads[i]["src_label"] for i in brief_idxs]
            brief_tgt = [pair_payloads[i]["tgt_label"] for i in brief_idxs]
            brief_packets = [pair_packets[i] for i in brief_idxs]
            generated_briefs = self.generate_pair_briefs_batched(
                brief_src, brief_tgt, brief_packets
            )
            for offset, idx in enumerate(brief_idxs):
                pair_briefs[idx] = generated_briefs[offset]

        p_llm = torch.zeros(n_pairs, device=self.device)
        S_final = S_base.clone()
        oracle_result = None
        if oracle_mode:
            assert label is not None
            oracle_result = analytical_oracle_ceiling(
                S_base.detach().cpu().tolist(),
                label,
                threshold=self.threshold,
            )
            routed = tuple(bool(value) for value in would_route.detach().cpu().tolist())
            if routed != oracle_result.routed:
                raise RuntimeError(
                    "oracle gate diagnostics disagree with analytical ceiling routing"
                )
            S_final = torch.tensor(
                oracle_result.corrected_scores,
                dtype=S_base.dtype,
                device=self.device,
            )
        if self.use_llm and not oracle_mode and decision_idxs:
            src_best = [pair_payloads[i]["src_label"] for i in decision_idxs]
            tgt_best = [pair_payloads[i]["tgt_label"] for i in decision_idxs]
            decision_briefs = [pair_briefs[i] for i in decision_idxs]
            p_yes_needed = self.llm_yesno_probs_batched(
                src_best,
                tgt_best,
                decision_briefs,
                ["" for _ in decision_briefs],
            )
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
                            f"Collected {count} calibration samples this batch (total={len(self._calibration_pending_probs)})."
                        )
            p_llm[decision_idxs] = p_yes_needed
            decisions_needed = ["Yes" if float(prob) >= 0.5 else "No" for prob in p_yes_needed]
            for offset, idx in enumerate(decision_idxs):
                llm_decisions[idx] = decisions_needed[offset]
            S_final[need_llm] = (1.0 - w_i[need_llm]) * S_base[need_llm] + w_i[need_llm] * p_llm[
                need_llm
            ]

        llm_used_mask = torch.zeros(n_pairs, dtype=torch.bool, device=self.device)
        if decision_idxs:
            llm_used_mask[decision_idxs] = True
        w_i_effective = w_i * llm_used_mask.to(w_i.dtype)

        I_lex = (1.0 - w_i_effective) * (1.0 - w_struct)
        I_label = I_lex * lex_label_weight
        I_strsim = I_lex * lex_strsim_weight
        I_struct = (1.0 - w_i_effective) * w_struct
        I_hier = I_struct * sum(
            (struct_weights[key] for key in hier_keys),
            torch.zeros(n_pairs, device=self.device),
        )
        I_sim = I_struct * struct_weights["sim_obj"]
        I_diff = I_struct * struct_weights["diff"]
        I_attr = I_struct * struct_weights["attr_aux"]
        I_ctx = I_struct
        I_llm = w_i_effective
        for idx, diagnostic in enumerate(gate_diagnostics):
            diagnostic.update(
                {
                    "invoked": bool(llm_used_mask[idx]),
                    "mix_weight": float(w_i_effective[idx]),
                    "llm_probability": (float(p_llm[idx]) if bool(llm_used_mask[idx]) else None),
                    "score_before": float(S_base[idx]),
                    "score_after": float(S_final[idx]),
                    "oracle_adjustment": (
                        float(S_final[idx] - S_base[idx]) if oracle_mode else 0.0
                    ),
                }
            )
        w_c = w_struct
        struct_active_pairs = int((sigma_sum > 1e-8).sum().item())
        gate_count_mask = would_route if self.llm_experiment_enabled else need_llm
        llm_gated_pairs = int(gate_count_mask.to(torch.int32).sum().item())
        brief_requested_pairs = int(len(brief_idxs))
        decision_requested_pairs = int(len(decision_idxs))

        result = {
            "s_label": s_label,
            "s_label_star": s_label_star,
            "s_strsim": s_strsim,
            "s_ctx": S_struct,
            "s_hier": s_hier,
            "s_sim": s_sim,
            "s_diff": s_diff,
            "s_attr": s_attr,
            "q_label": q_label,
            "q_lex": Q_lex,
            "q_strsim": q_strsim,
            "q_hier": q_hier,
            "q_sim": channel_quality_tensors["sim_obj"],
            "q_diff": channel_quality_tensors["diff"],
            "q_attr": channel_quality_tensors["attr_aux"],
            "Q_struct": Q_struct,
            "S_base": S_base,
            "S_lctx": S_base,
            "S_struct": S_struct,
            "p_llm": p_llm,
            "S_final": S_final,
            "w_c": w_c,
            "w_struct": w_struct,
            "U": U,
            "U_ind": U_ind,
            "U_dis": U_dis,
            "w_i": w_i,
            "need_llm": need_llm,
            "I_label": I_label,
            "I_lex": I_lex,
            "I_strsim": I_strsim,
            "I_struct": I_struct,
            "I_hier": I_hier,
            "I_sim": I_sim,
            "I_diff": I_diff,
            "I_attr": I_attr,
            "I_ctx": I_ctx,
            "I_llm": I_llm,
            "llm_decisions": llm_decisions,
            "llm_rationales": llm_rationales,
            "llm_pair_briefs": pair_briefs,
            "llm_evidence_packets": pair_packets,
            "llm_calibration": self._llm_calibration_payload(
                batch_samples=batch_calibration_samples
            ),
            "llm_summary_stats": self.llm_summary_stats(),
            "llm_decision_stats": self.llm_decision_stats(),
            "llm_summaries": {
                "source": pair_briefs,
                "target": ["" for _ in pair_briefs],
            },
            "backend_usage": {
                "summary": dict(self._last_summary_backend_meta),
                "decision": dict(self._last_decision_backend_meta),
                "rationale": dict(self._last_rationale_backend_meta),
            },
            "batch_pair_adaptive_stats": {
                "pairs": int(n_pairs),
                "unique_src": int(len(src_unique_iris)),
                "unique_tgt": int(len(tgt_unique_iris)),
                "src_cache_hits": int(src_cache_hits),
                "src_cache_misses": int(len(src_unique_iris) - src_cache_hits),
                "tgt_cache_hits": int(tgt_cache_hits),
                "tgt_cache_misses": int(len(tgt_unique_iris) - tgt_cache_hits),
                "src_pool": src_pool_stats,
                "tgt_pool": tgt_pool_stats,
                "pair_evidence": pair_evidence_stats,
                "struct_active_pairs": int(struct_active_pairs),
                "llm_gated_pairs": int(llm_gated_pairs),
                "brief_requested_pairs": int(brief_requested_pairs),
                "decision_requested_pairs": int(decision_requested_pairs),
            },
        }
        if self.llm_experiment_enabled:
            result["llm_gate_diagnostics"] = gate_diagnostics
        if oracle_mode:
            assert oracle_result is not None
            result["oracle_diagnostic"] = {
                "oracle_only": True,
                "deployable": False,
                "llm_invocations": oracle_result.llm_invocations,
                "routed": list(oracle_result.routed),
                "baseline_predictions": list(oracle_result.baseline_predictions),
                "labels": list(oracle_result.labels),
            }

        if self.return_explanations:
            explanations = []
            for idx, payload in enumerate(pair_payloads):
                family_scores = {
                    family: float(payload["hierarchy"][family]["score"]) for family in family_names
                }
                family_qualities = {
                    family: float(payload["hierarchy"][family]["quality"])
                    for family in family_names
                }
                family_weights = {
                    family: float(struct_weights[f"hier__{family}"][idx]) for family in family_names
                }
                hier_internal_weight = sum(
                    (struct_weights[key][idx] for key in hier_keys),
                    torch.tensor(0.0, device=self.device),
                )
                family_importances = {
                    family: float(I_struct[idx] * struct_weights[f"hier__{family}"][idx])
                    for family in family_names
                }
                family_contribs = {
                    family: float(
                        I_struct[idx]
                        * struct_weights[f"hier__{family}"][idx]
                        * (channel_score_tensors[f"hier__{family}"][idx] - self.tau)
                    )
                    for family in family_names
                }
                experiment_diagnostics = self._experiment_diagnostics_for_pair(
                    payload,
                    gate_diagnostic=(
                        gate_diagnostics[idx] if self.llm_experiment_enabled else None
                    ),
                )
                explanations.append(
                    {
                        "explanation_schema_version": 3,
                        **(
                            {"experiment_diagnostics": experiment_diagnostics}
                            if experiment_diagnostics
                            else {}
                        ),
                        "src_iri": src_iris[idx],
                        "tgt_iri": tgt_iris[idx],
                        "kind": dataset.entity_kind_for(
                            src_iris[idx], "src", warn_unknown=False
                        ).value,
                        "src_kind": dataset.entity_kind_for(
                            src_iris[idx], "src", warn_unknown=False
                        ).value,
                        "tgt_kind": dataset.entity_kind_for(
                            tgt_iris[idx], "tgt", warn_unknown=False
                        ).value,
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
                            "s_label": float(s_label[idx]),
                            "s_label_star": float(s_label_star[idx]),
                            "s_strsim": float(s_strsim[idx]),
                            "s_hier": float(s_hier[idx]),
                            "s_sim": float(s_sim[idx]),
                            "s_diff": float(s_diff[idx]),
                            "s_attr": float(s_attr[idx]),
                            "Q_struct": float(Q_struct[idx]),
                            "S_base": float(S_base[idx]),
                            "S_struct": float(S_struct[idx]),
                            "p_llm": float(p_llm[idx]),
                            "S_lctx": float(S_base[idx]),
                            "S_final": float(S_final[idx]),
                            "family_scores": family_scores,
                        },
                        "qualities": {
                            "q_label": float(q_label[idx]),
                            "q_lex": float(Q_lex[idx]),
                            "q_strsim": float(q_strsim[idx]),
                            "q_hier": float(q_hier[idx]),
                            "q_sim": float(channel_quality_tensors["sim_obj"][idx]),
                            "q_diff": float(channel_quality_tensors["diff"][idx]),
                            "q_attr": float(channel_quality_tensors["attr_aux"][idx]),
                            "family_qualities": family_qualities,
                        },
                        "weights": {
                            "w_label": float(
                                ((1.0 - w_struct[idx]) * lex_label_weight[idx]).item()
                            ),
                            "w_strsim": float(
                                ((1.0 - w_struct[idx]) * lex_strsim_weight[idx]).item()
                            ),
                            "w_struct": float(w_struct[idx]),
                            "w_hier": float(hier_internal_weight.item()),
                            "w_sim": float(struct_weights["sim_obj"][idx]),
                            "w_diff": float(struct_weights["diff"][idx]),
                            "w_attr": float(struct_weights["attr_aux"][idx]),
                            "w_c": float(w_c[idx]),
                            "w_i": float(w_i[idx]),
                            "U": float(U[idx]),
                            "U_ind": float(U_ind[idx]),
                            "U_dis": float(U_dis[idx]),
                            "family_weights": family_weights,
                        },
                        "importances": {
                            "I_label": float(I_label[idx]),
                            "I_lex": float(I_lex[idx]),
                            "I_strsim": float(I_strsim[idx]),
                            "I_struct": float(I_struct[idx]),
                            "I_hier": float(I_hier[idx]),
                            "I_sim": float(I_sim[idx]),
                            "I_diff": float(I_diff[idx]),
                            "I_attr": float(I_attr[idx]),
                            "I_ctx": float(I_ctx[idx]),
                            "I_llm": float(I_llm[idx]),
                            "family_importances": family_importances,
                        },
                        "contributions": {
                            "C_label": float(I_label[idx] * (s_label[idx] - self.tau)),
                            "C_strsim": float(I_strsim[idx] * (s_strsim[idx] - self.tau)),
                            "C_struct": float(I_struct[idx] * (S_struct[idx] - self.tau)),
                            "C_hier": float(
                                sum(
                                    (family_contribs[family] for family in family_names),
                                    0.0,
                                )
                            ),
                            "C_sim": float(
                                I_struct[idx]
                                * struct_weights["sim_obj"][idx]
                                * (s_sim[idx] - self.tau)
                            ),
                            "C_diff": float(
                                I_struct[idx]
                                * struct_weights["diff"][idx]
                                * (s_diff[idx] - self.tau)
                            ),
                            "C_attr": float(
                                I_struct[idx]
                                * struct_weights["attr_aux"][idx]
                                * (s_attr[idx] - self.tau)
                            ),
                            "C_llm": float(I_llm[idx] * (p_llm[idx] - S_base[idx])),
                            "C_oracle": (float(S_final[idx] - S_base[idx]) if oracle_mode else 0.0),
                            "family_contributions": family_contribs,
                        },
                        "prediction": {
                            "global_match": bool(S_final[idx] >= self.threshold),
                            "ground_truth": (
                                label[idx] if label is not None and idx < len(label) else None
                            ),
                            "llm_decision": llm_decisions[idx],
                            "llm_rationale": llm_rationales[idx],
                            "threshold_positive": bool(S_final[idx] >= self.threshold),
                            "saved_alignment_member": False,
                            "rationale_decision_label": "",
                        },
                        "selected_labels": {
                            "source": payload["src_label"],
                            "target": payload["tgt_label"],
                        },
                        "backend_usage": {
                            "summary": dict(self._last_summary_backend_meta),
                            "decision": dict(self._last_decision_backend_meta),
                            "rationale": dict(self._last_rationale_backend_meta),
                        },
                        "context_sentences": {
                            "hierarchy_source": {
                                family: list(payload["hierarchy"][family].get("src_sentences", []))
                                for family in family_names
                            },
                            "hierarchy_target": {
                                family: list(payload["hierarchy"][family].get("tgt_sentences", []))
                                for family in family_names
                            },
                            "similarity_source": list(payload["sim"].get("src_sentences", [])),
                            "similarity_target": list(payload["sim"].get("tgt_sentences", [])),
                            "difference_source": list(payload["diff"].get("src_sentences", [])),
                            "difference_target": list(payload["diff"].get("tgt_sentences", [])),
                        },
                        "context_triples": {
                            "hierarchy_source": {
                                family: [
                                    item["triple"]
                                    for item in payload["hierarchy"][family].get("src_selected", [])
                                ]
                                for family in family_names
                            },
                            "hierarchy_target": {
                                family: [
                                    item["triple"]
                                    for item in payload["hierarchy"][family].get("tgt_selected", [])
                                ]
                                for family in family_names
                            },
                            "similarity_source": [
                                item["triple"] for item in payload["sim"].get("src_selected", [])
                            ],
                            "similarity_target": [
                                item["triple"] for item in payload["sim"].get("tgt_selected", [])
                            ],
                            "difference_source": [
                                item["triple"] for item in payload["diff"].get("src_selected", [])
                            ],
                            "difference_target": [
                                item["triple"] for item in payload["diff"].get("tgt_selected", [])
                            ],
                        },
                        "attributes": {
                            "source": list(payload["attr"].get("src_selected", [])),
                            "target": list(payload["attr"].get("tgt_selected", [])),
                        },
                        "cross_side_provenance": self._build_cross_side_provenance(
                            float(S_lex[idx]),
                            payload["hierarchy"],
                            payload["sim"],
                            payload["diff"],
                            payload["attr"],
                        ),
                        "llm_pair_evidence_packet": pair_packets[idx],
                        "llm_pair_brief": pair_briefs[idx],
                        "llm_summaries": {
                            "source": pair_briefs[idx],
                            "target": "",
                        },
                        "triple_attributions": {
                            "hierarchy": {
                                family: {
                                    "source": list(
                                        payload["hierarchy"][family].get("src_selected", [])
                                    ),
                                    "target": list(
                                        payload["hierarchy"][family].get("tgt_selected", [])
                                    ),
                                }
                                for family in family_names
                            },
                            "similarity": {
                                "source": list(payload["sim"].get("src_selected", [])),
                                "target": list(payload["sim"].get("tgt_selected", [])),
                            },
                            "difference": {
                                "source": list(payload["diff"].get("src_selected", [])),
                                "target": list(payload["diff"].get("tgt_selected", [])),
                            },
                        },
                    }
                )
            result["explanations"] = explanations

        return result
