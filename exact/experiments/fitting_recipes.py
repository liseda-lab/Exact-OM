"""Bounded v2 fitting arms; data/model prerequisites are bound by the campaign."""

from copy import deepcopy

SAFE_TRAINING = [
    "disjoint_train_candidates",
    "training_reference",
    "verified_negative_label_policy",
]
FITTING_REQUIREMENTS = {}


def _arm(family, identifier, overlay, *, role="candidate", supervised=False, requires=()):
    FITTING_REQUIREMENTS[f"{family}/{identifier}"] = list(requires)
    return {
        "id": identifier,
        "role": role,
        "overlay": overlay,
        "supervision_label": "in_pair_supervised" if supervised else "target_label_free",
    }


def _supervision(*components):
    return {
        "mode": "label_free",
        "components": {
            **{
                key: "label_free"
                for key in (
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
            **{component: "supervised" for component in components},
        },
    }


def _judge(evidence="structured_packet", *, comparative=True, source_first=True):
    return {
        "enabled": True,
        "decision": {
            "mode": "listwise" if comparative else "binary",
            "evidence": evidence,
            "probability": "raw_joint",
            "permutations": 1,
            "listwise_max_candidates": 5,
        },
        "gate": {"mode": "source_top_fraction", "quantile_fraction": 1.0},
        "fusion_weight": "source_first" if source_first and comparative else "beta_u",
    }


def fitting_arms():
    """Return ArmConfig-compatible overlays without overriding bound model profiles."""
    arms = {}
    llm_off = {"llm": {"experiment": {"enabled": True, "gate": {"mode": "off"}}}}
    arms["E04"] = []
    for name, mode in (
        ("nil_off", "off"),
        ("nil_heuristic", "heuristic"),
        ("nil_fitted", "fitted"),
        ("pool_miss", "off"),
        ("listwise_none", "heuristic"),
    ):
        overlay = deepcopy(llm_off)
        overlay.update(
            supervision=_supervision(),
            selector={"enabled": True, "runtime_enabled": True, "runtime_global_only": False},
            matching={"nil": {"mode": mode}},
        )
        requirements = ["frozen_source_universe"]
        if name == "nil_fitted":
            overlay["matching"]["nil"]["label_semantics"] = "natural"
            requirements += [
                "disjoint_train_candidates",
                "training_reference",
                "annotated_nil_training_source_labels",
            ]
        if name == "pool_miss":
            requirements += [
                "bound_pool_miss_development_reference",
                "complete_development_reference",
            ]
        if name == "listwise_none":
            # Bind the actual E07 winner instead of substituting a default prompt.
            overlay["llm"]["experiment"] = {"enabled": True}
            requirements += ["selected_E07_judge", "paired_none_comparison"]
        record = _arm(
            "E04",
            name,
            overlay,
            role=(
                "baseline"
                if name == "nil_off"
                else "diagnostic" if name == "pool_miss" else "candidate"
            ),
            supervised=name == "nil_fitted",
            requires=requirements,
        )
        if name == "pool_miss":
            record["deployable"] = False
        arms["E04"].append(record)
    arms["E07"] = []
    for name, evidence, comparative in (
        ("brief_binary", "generated_brief", False),
        ("brief_256_binary", "generated_brief", False),
        ("packet_binary", "scored_packet", False),
        ("facts_binary", "structured_packet", False),
        ("facts_listwise", "structured_packet", True),
        ("retrieved_listwise", "structured_packet", True),
    ):
        judge = _judge(evidence, comparative=comparative)
        judge["decision"]["brief_max_tokens"] = 256 if name == "brief_256_binary" else 64
        if name == "retrieved_listwise":
            judge["decision"]["max_evidence_packets"] = 2
        overlay = {
            "supervision": _supervision(),
            "selector": {"runtime_enabled": False},
            "llm": {"experiment": judge},
        }
        arms["E07"].append(
            _arm(
                "E07",
                name,
                overlay,
                role="baseline" if name == "brief_binary" else "candidate",
                requires=[
                    "same_frozen_judgment_sources",
                    "explicit_openrouter_decision_profile",
                    "complete_categorical_token_support",
                ]
                + (["openrouter_summary_profile"] if evidence == "generated_brief" else []),
            )
        )

    arms["E10"] = []
    for gamma in (1, 2, 3):
        for tau in (0.4, 0.5):
            name = f"analytic_g{gamma}_t{str(tau).replace('.', '')}"
            overlay = deepcopy(llm_off)
            overlay.update(
                supervision=_supervision(),
                matching={
                    "fusion": {
                        "enabled": True,
                        "mode": "analytic_shipped",
                        "gamma": gamma,
                        "tau": tau,
                        "beta": 0.8,
                    }
                },
            )
            arms["E10"].append(
                _arm(
                    "E10",
                    name,
                    overlay,
                    role="baseline" if (gamma, tau) == (2, 0.5) else "candidate",
                )
            )
    for name, recipe in (
        ("winner_only", "winner_only"),
        ("winner_runnerup", "winner_plus_runnerup"),
    ):
        overlay = deepcopy(llm_off)
        overlay.update(
            supervision=_supervision("accept", "rerank"),
            selector={
                "enabled": True,
                "runtime_enabled": True,
                "runtime_global_only": False,
                "accept_training": recipe,
            },
        )
        arms["E10"].append(
            _arm(
                "E10",
                name,
                overlay,
                supervised=True,
                requires=SAFE_TRAINING + ["selected_E10_analytic_setting"],
            )
        )

    arms["E18"] = []
    for name, objective, model in (
        ("analytic", "analytic", "current_linear"),
        ("current_listwise", "current_listwise", "current_linear"),
        ("pairwise", "pairwise", "current_linear"),
        ("channel_gating", "current_listwise", "channel_gating"),
        ("additive_gam", "current_listwise", "additive_gam"),
    ):
        overlay = deepcopy(llm_off)
        overlay.update(
            supervision=_supervision("rerank", "accept"),
            selector={
                "enabled": True,
                "runtime_enabled": True,
                "runtime_global_only": False,
                "rerank": {"mode": objective, "model": model, "features": "current"},
            },
        )
        arms["E18"].append(
            _arm(
                "E18",
                name,
                overlay,
                role=(
                    "baseline"
                    if name == "analytic"
                    else "diagnostic" if name == "additive_gam" else "candidate"
                ),
                supervised=True,
                requires=SAFE_TRAINING,
            )
        )

    arms["E19"] = []
    for mode in ("analytic_shipped", "analytic_fitted", "learned_global"):
        fitted = mode != "analytic_shipped"
        overlay = deepcopy(llm_off)
        overlay.update(
            supervision=_supervision(*(["fusion"] if fitted else [])),
            matching={"fusion": {"enabled": True, "mode": mode, "beta": 0.8}},
        )
        arms["E19"].append(
            _arm(
                "E19",
                mode,
                overlay,
                role="baseline" if not fitted else "candidate",
                supervised=fitted,
                requires=SAFE_TRAINING if fitted else [],
            )
        )

    arms["E20"] = []
    for name in ("zero_shot", "contrastive", "cross_encoder", "combined"):
        overlay = deepcopy(llm_off)
        overlay["supervision"] = _supervision(*(["retrieval"] if name != "zero_shot" else []))
        overlay["candidates"] = {}
        requirements = ["selected_E05_retriever"]
        if name in {"contrastive", "combined"}:
            overlay["candidates"]["encoder_finetune"] = {"mode": "contrastive"}
            requirements += SAFE_TRAINING + [
                "pinned_encoder_training_recipe_or_fitted_artifact",
                "generated_reporting_pool",
            ]
        if name in {"cross_encoder", "combined"}:
            overlay["candidates"]["cross_encoder"] = {"mode": "on", "top_k": 20}
            requirements += SAFE_TRAINING + [
                "pinned_cross_encoder_training_recipe_or_fitted_artifact",
                "same_frozen_retrieved_pool",
            ]
        if name == "combined":
            requirements += ["development_supported_combination"]
        arms["E20"].append(
            _arm(
                "E20",
                name,
                overlay,
                role="baseline" if name == "zero_shot" else "candidate",
                supervised=name != "zero_shot",
                requires=sorted(set(requirements)),
            )
        )

    arms["E21"] = []
    for name in (
        "frozen_judge",
        "knn_exemplars",
        "benefit_router",
        "student_gold",
        "student_distilled",
    ):
        # The verified E07 consumer supplies decision/evidence/integration settings.
        # Arm overlays carry only the intended intervention.
        judge = {"enabled": True}
        requirements = [
            "selected_E07_judge",
            "judge_benefit_established",
            "fixed_pair_threshold_acceptance",
        ]
        supervised = name != "frozen_judge"
        if supervised:
            requirements += SAFE_TRAINING
        if name == "knn_exemplars":
            judge.update(exemplars="knn", exemplar_count=3)
        elif name == "benefit_router":
            judge.update(gate={"mode": "learned"}, outcome_policy="complete_sources")
            requirements += ["complete_source_outcomes", "no_additional_selector"]
        elif name.startswith("student_"):
            judge.update(
                distill="student",
                student_training="gold_only" if name == "student_gold" else "gold_teacher",
                fusion_weight="beta_u",
            )
            if name == "student_distilled":
                requirements += ["pinned_teacher_and_cost_ledger"]
        overlay = {
            "supervision": _supervision(*(["llm"] if supervised else [])),
            "selector": {"runtime_enabled": False},
            "llm": {"experiment": judge},
        }
        arms["E21"].append(
            _arm(
                "E21",
                name,
                overlay,
                role="baseline" if not supervised else "candidate",
                supervised=supervised,
                requires=requirements,
            )
        )

    arms["E22"] = []
    for name, budget, selection in (
        ("budget_25", 25, "passive"),
        ("budget_100", 100, "passive"),
        ("budget_400", 400, "passive"),
        ("label_free", None, "passive"),
        ("active_100", 100, "uncertainty"),
    ):
        supervised = budget is not None
        overlay = deepcopy(llm_off)
        overlay["supervision"] = {
            **_supervision(*(["rerank", "accept"] if supervised else [])),
            "label_budget": budget,
            "label_selection": selection,
        }
        overlay["selector"] = {
            "enabled": True,
            "runtime_enabled": True,
            "runtime_global_only": False,
            "rerank": {"artifact": None},
        }
        arms["E22"].append(
            _arm(
                "E22",
                name,
                overlay,
                role="baseline" if not supervised else "candidate",
                supervised=supervised,
                requires=["one_selected_component_recipe"] + (SAFE_TRAINING if supervised else []),
            )
        )
    return arms


def fitting_requirements():
    fitting_arms()
    return deepcopy(FITTING_REQUIREMENTS)
