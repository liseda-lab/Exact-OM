"""Small sequential v2 controls; requirements are inputs, never readiness claims."""

from copy import deepcopy

from exact.experiments.harness import deep_merge

CORE_REQUIREMENTS: dict[str, list[str]] = {}
SAFE_TRAINING = [
    "disjoint_train_candidates",
    "training_reference",
    "verified_negative_label_policy",
]


def core_arms(base):
    """Build overlays while preserving the caller's complete pinned pipeline."""
    arms = {}
    CORE_REQUIREMENTS.clear()
    off = {"llm": {"experiment": {"enabled": True, "gate": {"mode": "off"}}}}
    label_free = {
        "supervision": {
            "mode": "label_free",
            "components": {
                name: "label_free"
                for name in (
                    "retrieval",
                    "fusion",
                    "rerank",
                    "llm",
                    "accept",
                    "calibration",
                    "structure",
                    "relation",
                )
            },
        }
    }

    def add(family, name, overlay=None, role="candidate", requires=(), supervised=False):
        diagnostic = role in {"oracle", "diagnostic"}
        CORE_REQUIREMENTS[f"{family}/{name}"] = list(requires)
        arms.setdefault(family, []).append(
            {
                "id": name,
                "role": role,
                "overlay": overlay or {},
                "deployable": not diagnostic,
                "required_control": role in {"baseline", "control"},
                "supervision_label": (
                    "oracle_diagnostic"
                    if role == "oracle"
                    else "in_pair_supervised" if supervised else "target_label_free"
                ),
            }
        )

    def primary(**params):
        pipeline = deepcopy(base["pipeline"])
        component = next(item for item in pipeline if item["name"] == "PairAdaptiveSemanticScorer")
        component["params"].update(params)
        return {"pipeline": pipeline}

    def controls(overlay):
        return deep_merge(deep_merge(off, label_free), overlay)

    def channel(name, **settings):
        return controls({"matching": {"channels": {name: {"enabled": True, **settings}}}})

    def supervised(*components):
        return {
            "supervision": {
                "mode": "label_free",
                "components": {name: "supervised" for name in components},
            }
        }

    for name in ("production", "replay"):
        add(
            "E00",
            name,
            role="baseline" if name == "production" else "control",
            requires=[
                "operational_checkpoint_relocation_protocol",
                "source_level_outputs",
                "bounded_published_matcher_binding",
            ],
        )
    for mode in (
        "greedy",
        "mutual_best",
        "stable_marriage",
        "assignment_accepted_utility",
        "assignment_legacy",
    ):
        add(
            "E01",
            mode,
            {"matching": {"extraction": {"mode": mode}}},
            role=(
                "baseline"
                if mode == "greedy"
                else "diagnostic" if mode == "assignment_legacy" else "candidate"
            ),
            requires=["frozen_global_scores", "declared_cardinality"],
        )
    for mode in ("none", "platt", "isotonic", "distribution_threshold", "fp_only"):
        calibrated = mode in {"platt", "isotonic", "fp_only"}
        overlay = controls(
            {
                "matching": {
                    "calibration": {
                        "mode": mode if mode in {"platt", "isotonic"} else "none",
                        "threshold_mode": "otsu" if mode == "distribution_threshold" else "fixed",
                    }
                }
            }
        )
        if calibrated:
            overlay = deep_merge(overlay, supervised("calibration"))
        if mode == "fp_only":
            overlay = deep_merge(
                overlay,
                {"selector": {"enabled": True, "tuning": {"count_reference_miss_as": "fp"}}},
            )
        add(
            "E03",
            mode,
            overlay,
            role=(
                "baseline" if mode == "none" else "diagnostic" if mode == "fp_only" else "candidate"
            ),
            supervised=calibrated,
            requires=(SAFE_TRAINING if calibrated else [])
            + (["selected_calibration_recipe"] if mode == "fp_only" else []),
        )
    for name, candidate in (
        ("baseline", {"top_k": 20}),
        (
            "sapbert",
            {
                "top_k": 20,
                "encoder_revision": None,
                (
                    "encoder" if "encoder" in base["candidates"] else "lexical_encoder_name"
                ): "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
            },
        ),
        ("rrf", {"top_k": 20, "fusion": {"mode": "rrf"}}),
        ("adaptive_k", {"top_k": 20, "adaptive_k": {"enabled": True, "k_min": 10, "k_max": 50}}),
        ("combined", {"top_k": 20}),
    ):
        add(
            "E05",
            name,
            controls({"candidates": candidate}),
            role="baseline" if name == "baseline" else "candidate",
            requires=["generated_pool", "matched_mean_k20"]
            + (["development_survivor_combination"] if name == "combined" else []),
        )
    for name in ("current", "string_added", "lexical_only", "string_only"):
        overlay = controls({})
        if name in {"string_added", "string_only"}:
            overlay = deep_merge(overlay, channel("strsim", abbreviation="initialism"))
        if name in {"lexical_only", "string_only"}:
            overlay = deep_merge(
                overlay, primary(use_context=False, use_lexical=name == "lexical_only")
            )
        add(
            "E06",
            name,
            overlay,
            role=(
                "baseline"
                if name == "current"
                else (
                    "control"
                    if name == "lexical_only"
                    else "diagnostic" if name == "string_only" else "candidate"
                )
            ),
        )
    for name, settings in (
        ("current", {"enabled": False}),
        ("unified_bank", {"bank": "attrs_labels"}),
        ("provenance_dedup", {"bank": "attrs_labels", "provenance_dedup": True}),
        (
            "signed_identifiers",
            {"bank": "attrs_labels", "provenance_dedup": True, "polarity": "signed"},
        ),
    ):
        add(
            "E08",
            name,
            channel("attr", **settings),
            role="baseline" if name == "current" else "candidate",
            requires=(
                ["descriptor_identifier_namespace_and_exclusivity_allowlist"]
                if name == "signed_identifiers"
                else []
            ),
        )
    for name, settings in (
        ("current", {"mode": "labels"}),
        ("ancestor_ic", {"mode": "labels_overlap"}),
        ("sibling_context", {"mode": "labels_overlap", "siblings": True}),
        ("hierarchy_removed", {"mode": "off"}),
    ):
        add(
            "E09",
            name,
            channel("hier", **settings),
            role=(
                "baseline"
                if name == "current"
                else "diagnostic" if name == "hierarchy_removed" else "candidate"
            ),
            requires=["hierarchy_rich_case", "frozen_exact_or_trusted_anchors"],
        )
    for name, mode in (
        ("analytic_fixed", "current_fallback"),
        ("distribution_margin", "score_partition"),
        ("reciprocal_consensus", "reciprocal_consensus"),
        ("current_supervised", "current_fallback"),
    ):
        fitted = name == "current_supervised"
        overlay = controls(
            {
                "selector": {
                    "enabled": True,
                    "runtime_enabled": True,
                    "runtime_global_only": False,
                    "label_free_mode": mode,
                }
            }
        )
        if fitted:
            overlay = deep_merge(overlay, supervised("rerank", "accept"))
        add(
            "E15",
            name,
            overlay,
            role="baseline" if name == "analytic_fixed" else "control" if fitted else "candidate",
            supervised=fitted,
            requires=SAFE_TRAINING if fitted else [],
        )
    for name in ("label_free", "in_pair_supervised", "donor_transfer"):
        overlay = controls(
            {"selector": {"enabled": True, "runtime_enabled": True, "runtime_global_only": False}}
        )
        if name == "in_pair_supervised":
            overlay = deep_merge(overlay, supervised("rerank", "accept"))
        add(
            "E16",
            name,
            overlay,
            role="baseline" if name == "label_free" else "candidate",
            supervised=name == "in_pair_supervised",
            requires=(
                SAFE_TRAINING
                if name == "in_pair_supervised"
                else (
                    [
                        "unchanged_selected_donor_artifact_and_threshold",
                        "compatible_recipient_features",
                        "no_recipient_refit",
                    ]
                    if name == "donor_transfer"
                    else []
                )
            ),
        )
        if name == "donor_transfer":
            arms["E16"][-1]["supervision_label"] = "cross_pair_transfer"
    for name in ("baseline", "core_stack", "optional_stack"):
        add(
            "E17",
            name,
            role="baseline" if name == "baseline" else "candidate",
            requires=["frozen_G4_development_composition", "paired_full_population_final_panel"],
        )
    for name, formulation in (
        ("normalised", "normalised"),
        ("diff_off", "off"),
        ("missingness_aware", "missingness_aware"),
        ("absolute", "absolute"),
        ("asymmetric", "asymmetric"),
    ):
        add(
            "E24",
            name,
            channel("diff", formulation=formulation, dump_components=True),
            role=(
                "baseline"
                if name == "normalised"
                else (
                    "control"
                    if name == "diff_off"
                    else "diagnostic" if name in {"absolute", "asymmetric"} else "candidate"
                )
            ),
            requires=["declared_directional_typed_interpretation"] if name == "asymmetric" else [],
        )
    for name, fraction in (
        ("decision_off", None),
        ("analytic", None),
        ("source_top_001", 0.01),
        ("source_top_005", 0.05),
        ("source_top_010", 0.1),
        ("forced_sources", None),
        ("oracle_perfect", None),
        ("oracle_observed", None),
        ("trust_shipped", None),
        ("trust_constant", None),
        ("trust_source", None),
    ):
        gate = {"mode": "off" if name == "decision_off" else "analytic"}
        requirements = ["explicit_openrouter_decision_profile"]
        if fraction:
            gate = {"mode": "source_top_fraction", "quantile_fraction": fraction}
        if name == "forced_sources" or name.startswith("trust_"):
            gate = {"mode": "forced_sample", "forced_sample_size": 200}
            requirements += ["frozen_stratified_source_sample", "forced_source_gate_artifact"]
        if name.startswith("oracle_"):
            gate = {"mode": "oracle_perfect" if name == "oracle_perfect" else "oracle_replay"}
            requirements = ["development_only_fixed_interventions", "oracle_diagnostic_protocol"]
        if name.startswith("trust_"):
            gate = {"mode": "oracle_replay"}
        if name == "oracle_observed" or name.startswith("trust_"):
            requirements += ["cached_real_forced_responses", "no_new_hosted_requests"]
        judge = {
            "enabled": True,
            "gate": gate,
            "fusion_weight": (
                "constant"
                if name == "trust_constant"
                else "source_first" if name == "trust_source" else "beta_u"
            ),
            "constant_weight": 0.5,
        }
        if name == "trust_source":
            judge["decision"] = {"mode": "listwise", "probability": "raw_joint"}
        add(
            "E25",
            name,
            {"llm": {"experiment": judge}},
            role=(
                "baseline"
                if name == "decision_off"
                else (
                    "oracle"
                    if name.startswith("oracle_")
                    else (
                        "diagnostic"
                        if name == "forced_sources" or name.startswith("trust_")
                        else "candidate"
                    )
                )
            ),
            requires=requirements,
        )
    for name in ("full", "constant_q", "no_sharpening", "suppression_only", "uniform"):
        add(
            "E26",
            name,
            controls({"matching": {"fusion": {"enabled": True, "sigma_mode": name}}}),
            role="baseline" if name == "full" else "control",
        )
    for name, quality in (
        ("candidate_margin", "candidate_margin"),
        ("entropy_count_control", "entropy"),
        ("encoder_agreement", "encoder_agreement"),
    ):
        add(
            "E26",
            name,
            channel(
                "lex",
                quality=quality,
                entropy_temperature=1.0,
                entropy_top_m=5,
                deduplicate_labels=True,
            ),
            requires=["selected_E26_decomposition", "complete_frozen_source_pools"],
        )
    for family in ("E00", "E01", "E17", "E25"):
        for arm in arms[family]:
            if arm["role"] != "oracle":
                arm["supervision_label"] = None
    return arms


def core_requirements(base):
    core_arms(base)
    return deepcopy(CORE_REQUIREMENTS)
