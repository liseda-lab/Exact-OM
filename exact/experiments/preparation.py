"""Materialize the bounded blueprint using explicit local bindings; never run cells."""

from copy import deepcopy
from pathlib import Path
from typing import Any

from exact.core.entities.configs.config import ConfigModel
from exact.core.entities.configs.yaml_io import dump_yaml_document, load_yaml_mapping
from exact.experiments.campaign import CampaignLock, load_campaign
from exact.experiments.core_recipes import core_arms, core_requirements
from exact.experiments.feature_recipes import FEATURE_REQUIREMENTS, feature_arms
from exact.experiments.fitting_recipes import fitting_arms, fitting_requirements
from exact.experiments.harness import deep_merge
from exact.utils.provenance import sha256_file

CASE_BY_FAMILY = {
    "E04": "N0",
    "E09": "D1",
    "E11": "P0",
    "E12": "K0",
    "E13": "R0_case",
    "E14": "T0",
    "E16": "D1",
    "E23": "K0",
    "E24": "D1",
}
POLICY_PATHS = {
    "E05": ["candidates"],
    "E20": [
        "candidates.encoder_finetune",
        "candidates.cross_encoder",
        "supervision.components.retrieval",
    ],
    "E26": ["matching.fusion.enabled", "matching.fusion.sigma_mode", "matching.channels.lex"],
    "E24": ["matching.channels.diff"],
    "E06": ["matching.channels.strsim"],
    "E08": ["matching.channels.attr"],
    "E09": ["matching.channels.hier"],
    "E10": ["matching.fusion.gamma", "matching.fusion.tau", "selector.accept_training"],
    "E19": [
        "matching.fusion." + key
        for key in ("enabled", "mode", "scope", "artifact", "gamma", "tau", "beta")
    ]
    + ["supervision.components.fusion"],
    "E03": ["matching.calibration", "supervision.components.calibration"],
    "E15": ["selector", "supervision.components.rerank", "supervision.components.accept"],
    "E18": ["selector", "supervision.components.rerank", "supervision.components.accept"],
    "E01": ["matching.extraction"],
    "E02": ["matching.anchor_rescoring"],
    "E25": ["llm.experiment.gate"],
    "E07": ["llm.experiment.decision", "llm.experiment.fusion_weight"],
}
PORTS = {
    "E00": ["baseline_evidence", "kind_pool_freeze"],
    "E05": ["E05_initial"],
    "E20": ["pool_freeze"],
    "E13": ["typed_pool_freeze"],
    "E12-retrieval": ["instance_pool_freeze"],
    "E18": ["selected_heads"],
    "E02": ["E02_anchors"],
    "E25": ["E25_initial"],
    "E07": ["E07_judgment_evidence"],
}


def prepare_campaign(
    blueprint: Path, base_config: Path, bindings: dict[str, Any], destination: Path
) -> Path:
    """Write immutable declarations and a complete arm/prerequisite inventory.

    Bindings provide cases, inspected readiness, measured forecasts and explicit
    arm inputs/overlays. Missing prerequisites remain visible and cannot execute.
    No dataset labels, model, credential, or results are accessed by this function.
    """
    base = ConfigModel.load_config(base_config).model_dump(mode="json", by_alias=True)
    design = load_yaml_mapping(blueprint)
    recipes = {**core_arms(base), **fitting_arms(), **feature_arms()}
    requirements = {**core_requirements(base), **fitting_requirements(), **FEATURE_REQUIREMENTS}
    if set(recipes) != {entry["id"] for entry in design["experiments"]}:
        raise ValueError("Every declared family must have bounded implemented recipes")
    cases = deepcopy(bindings["cases"])
    overrides = deepcopy(bindings.get("steps", {}))
    steps = []

    def make_step(identifier, family, arms, case, group, requires, **settings):
        baseline = next((arm["id"] for arm in arms if arm["role"] == "baseline"), arms[0]["id"])
        selectable = [
            arm["id"]
            for arm in arms
            if arm["id"] != baseline and arm.get("deployable", True) and arm["role"] != "oracle"
        ]
        if family == "E06":
            selectable = ["string_added"]
        diagnostic = (
            family in {"E00", "E13"}
            or identifier in {"E25-trust", "E25-oracles", "E25-forced", "E04-pool-miss"}
            or all(arm.get("published_matcher") for arm in arms)
        )
        if not selectable and not diagnostic:
            raise ValueError(f"{identifier}: comparison has no eligible treatment/control")
        retrieval = family == "E05" or identifier == "E12-retrieval"
        guards = []
        if family != "E00":
            guards.append(
                {
                    "id": "recall_noninferiority",
                    "metric": "candidate_recall",
                    "scope": "each_task",
                    "min_delta": -0.005,
                }
            )
        if retrieval:
            guards.append(
                {
                    "id": "matched_mean_k",
                    "metric": "mean_pool_size",
                    "comparison": "matched",
                    "match_tolerance": {"absolute": 1.0, "relative": 0.02, "combine": "max"},
                }
            )
        readiness = {}
        for arm in arms:
            needed = requirements.get(f"{family}/{arm['id']}", [])
            readiness[arm["id"]] = {
                stage: {
                    "status": "implementing",
                    "reason": "Bind inspected code/test evidence and measured real-input probe"
                    + ("; prerequisites: " + ", ".join(needed) if needed else ""),
                }
                for stage in ("screen", "confirm")
            }
        value = {
            "id": identifier,
            "family": family,
            "case": case,
            "budget_group": group,
            "arms": arms,
            "readiness": readiness,
            "requires": requires,
            "produces": PORTS.get(identifier, []),
            "policy_paths": POLICY_PATHS.get(family, []),
            "selection": {
                "decisions": [
                    {
                        "id": "bounded_selection",
                        "baseline": baseline,
                        "candidates": selectable,
                        "metric": "candidate_recall" if retrieval else "F1",
                        "min_delta": (
                            0.005 if retrieval else 0.0 if family in {"E00", "E13"} else 0.003
                        ),
                        "guards": guards,
                        "tie_breaks": [
                            {"kind": "metric", "metric": "wall_seconds", "direction": "min"},
                            {
                                "kind": "arm_order",
                                "order": selectable,
                                "direction": "min",
                                "tolerance": 0.0,
                            },
                        ],
                        "required_controls": [
                            arm["id"] for arm in arms if arm.get("required_control")
                        ],
                    }
                ]
            },
            "design": {
                "primary_comparison": f"{identifier} bounded sequential comparison",
                "primary_endpoint": "candidate_recall" if retrieval else "F1",
                "independent_unit": "source_group",
                "power_status": "descriptive",
                "assumptions": [
                    "Development selection only; no component confirmatory claim.",
                    "Common frozen source populations and role-separated pools.",
                    "Unknown reference entries are not confirmed negatives.",
                ],
                "regression_bound": 0.005,
            },
            **settings,
        }
        if diagnostic:
            value["selection"] = {"decisions": []}
        value["design"]["assumptions"].append(
            "After quality and cost ties, the declared treatment order is the frozen simplicity priority; canonical arm ID breaks any remaining tie."
        )
        if family == "E04" and value["selection"]["decisions"]:
            value["design"]["primary_endpoint"] = "nil.nil_aware.F1"
            decision = value["selection"]["decisions"][0]
            decision["metric"] = "nil.nil_aware.F1"
            decision["guards"].append(
                {
                    "id": "non_nil_mrr_noninferiority",
                    "metric": "nil.non_nil_MRR",
                    "scope": "each_task",
                    "min_delta": -0.005,
                }
            )
        if identifier == "G4":
            value["execution_modes"] = ["global_alignment", "local_ranking"]
            value["design"]["cost_bound"] = 1.2
            value["selection"]["decisions"][0]["guards"] += [
                {
                    "id": "local_mrr_noninferiority",
                    "metric": "local.MRR",
                    "scope": "each_task",
                    "min_delta": -0.005,
                },
                {
                    "id": "default_inference_ceiling",
                    "metric": "inference_seconds",
                    "scope": "each_task",
                    "direction": "min",
                    "min_relative_delta": -0.2,
                },
            ]
            value["selection"]["decisions"][0]["tie_breaks"][0]["metric"] = "inference_seconds"
        if value.get("phase") == "final":
            if identifier == "E17":
                decision = value["selection"]["decisions"][0]
                decision["candidates"] = ["stack_all"]
                decision["tie_breaks"][1]["order"] = ["stack_all"]
            value["design"].update(
                primary_comparison="Frozen stack_all versus R_v2 baseline on H0/H1/H2; label-free control reported separately",
                primary_endpoint="task_macro_global_F1",
                practical_effect=0.003,
                non_inferiority_margin=0.005,
                multiplicity="holm",
            )
            value["design"]["assumptions"] += [
                "Primary gain requires a source-paired 95% CI excluding zero and meeting the frozen practical effect; no secondary endpoint substitution.",
                "Source bootstrap conditions on these ontology pairs and seeds; descriptive status does not establish a broad domain claim.",
            ]
        if family == "E26":
            value["design"]["assumptions"].append(
                "Definitive E26 selection uses the frozen G1 label-free pool; no preliminary change is promoted."
            )
        if "pool_freeze" in requires or family == "E20":
            value["requires"] = list(dict.fromkeys([*requires, "E05_initial"]))
            value["inherits"] = list(dict.fromkeys([*value.get("inherits", []), "E05_initial"]))
            value["design"]["assumptions"].append(
                "G1 publishes separate label-free and supervised retrieval policies; this comparison inherits the label-free E05 pool."
            )
        value = deep_merge(value, overrides.pop(identifier, {}))
        steps.append(value)

    late = []
    for entry in design["experiments"]:
        family = entry["id"]
        if family == "E17":
            continue
        arms = deepcopy(recipes[family])
        if family in {"E05", "E20"}:
            # A survivor-dependent combination is not an executable copy of baseline.
            arms = [arm for arm in arms if arm["id"] != "combined"]
        if family == "E12":
            labels = next(arm for arm in arms if arm["id"] == "labels")
            multi_view = deepcopy(next(arm for arm in arms if arm["id"] == "multi_view"))
            multi_view["overlay"] = deep_merge(
                labels["overlay"], {"candidates": multi_view["overlay"]["candidates"]}
            )
            make_step(
                "E12-retrieval",
                family,
                [deepcopy(labels), multi_view],
                "K0",
                entry["budget_group"],
                ["E00"],
                policy_paths=["candidates.multi_view"],
            )
            arms = [arm for arm in arms if arm["id"] != "multi_view"]
            make_step(
                family,
                family,
                arms,
                "K0",
                entry["budget_group"],
                ["E00", "instance_pool_freeze"],
                phase="expansion",
                source_cap=300,
                inherits=["instance_pool_freeze"],
                policy_paths=["matching.channels.instance"],
            )
            continue
        if family == "E13":
            enrichment = [arm for arm in arms if arm["id"] in {"csv_raw", "csv_materialized"}]
            arms = [arm for arm in arms if arm["id"] in {"owl_parity", "csv_parity"}]
            cases.setdefault(
                "R0_enrichment",
                {
                    "task": "biokg-enrichment-unresolved",
                    "role": "development",
                    "heldout_case": "R1_case",
                    "selection_reason": "Awaiting the independent BioKG release and provenance-bearing Datalog view; parity uses R0_case separately.",
                },
            )
            cases.setdefault(
                "R1_case",
                {
                    "task": "biokg-enrichment-reporting-unresolved",
                    "role": "reporting",
                    "selection_reason": "Prospective distinct reporting case for independent BioKG enrichment; no final input or outcome is accessed during preparation.",
                },
            )
            late.append(
                (
                    "E13-enrichment",
                    family,
                    enrichment,
                    "R0_enrichment",
                    entry["budget_group"],
                    ["E00", "E13"],
                )
            )
        if family == "E25":
            replay = [arm for arm in arms if arm["id"].startswith("trust_")]
            oracles = [arm for arm in arms if arm["id"].startswith("oracle_")]
            forced = next(arm for arm in arms if arm["id"] == "forced_sources")
            control = deepcopy(next(arm for arm in arms if arm["id"] == "decision_off"))
            arms = [
                arm
                for arm in arms
                if arm["id"] != "forced_sources" and not arm["id"].startswith(("trust_", "oracle_"))
            ]
            replay[0]["role"] = "baseline"
            late.extend(
                [
                    (
                        "E25-forced",
                        family,
                        [forced],
                        "D0",
                        "llm",
                        ["E25_initial", "E07_judgment_evidence", "E05_initial"],
                    ),
                    ("E25-oracles", family, [control, *oracles], "D0", "llm", ["E25-forced"]),
                    ("E25-trust", family, replay, "D0", "llm", ["E25-forced"]),
                ]
            )
        if family == "E04":
            control = next(arm for arm in arms if arm["role"] == "baseline")
            for name, identifier, case, dependencies in (
                ("pool_miss", "E04-pool-miss", "D0", ["E00", "pool_freeze", "E03"]),
                ("listwise_none", "E04-listwise", "N0", ["E04", "E07_judgment_evidence"]),
            ):
                treatment = next(arm for arm in arms if arm["id"] == name)
                late.append(
                    (
                        identifier,
                        family,
                        [deepcopy(control), treatment],
                        case,
                        "decisions",
                        dependencies,
                    )
                )
            arms = [arm for arm in arms if arm["id"] not in {"listwise_none", "pool_miss"}]
        dependencies = list(entry["dependencies"])
        if family == "E18":
            dependencies += ["E03", "E19"]
        if family == "E26":
            dependencies += ["pool_freeze"]
        extra = {}
        if family in {"E16", "E22"}:
            extra["inherits"] = ["selected_heads"]
        if family == "E23":
            dependencies += ["instance_pool_freeze"]
            extra["inherits"] = ["instance_pool_freeze"]
        make_step(
            family,
            family,
            arms,
            CASE_BY_FAMILY.get(family, "D0"),
            entry["budget_group"],
            dependencies,
            **extra,
        )
    for args in late:
        settings: dict[str, Any] = {"source_cap": 300 if args[0].startswith("E25-") else 200}
        if args[0] == "E25-forced":
            settings["inherits"] = ["E07_judgment_evidence", "E05_initial"]
        make_step(*args, phase="late", **settings)
        if args[0] == "E13-enrichment" and not cases["R0_enrichment"].get("source"):
            for roles in steps[-1]["readiness"].values():
                for readiness in roles.values():
                    readiness.update(
                        status="blocked_input_resolution",
                        reason="Awaiting independent BioKG release and provenance-bearing Datalog consequences; matched OWL/CSV parity remains independent.",
                    )
    make_step(
        "E22-policy",
        "E22",
        [{"id": "fixed_count", "role": "baseline"}, {"id": "fitted_count", "role": "candidate"}],
        "D1",
        "extensions",
        ["E22", "selected_heads", "pool_freeze"],
        phase="late",
        source_cap=300,
        inherits=["selected_heads"],
        policy_paths=[],
    )
    component_sources = bindings.get(
        "composition_sources",
        [
            "E05",
            "E20",
            "E26",
            "E24",
            "E06",
            "E08",
            "E09",
            "E10",
            "E19",
            "E03",
            "E15",
            "E18",
            "E01",
            "E02",
            "E25",
            "E07",
        ],
    )
    make_step(
        "G4",
        "E17",
        [{"id": "baseline", "role": "baseline"}, {"id": "core", "role": "candidate"}],
        "D0",
        "sentinels",
        list(dict.fromkeys(["E00", *component_sources])),
        phase="freeze",
        additional_cases=["D1"],
        source_cap=1000,
        produces=["G4_frozen_selection"],
    )
    lf = next(arm["overlay"] for arm in recipes["E15"] if arm["id"] == "analytic_fixed")
    make_step(
        "E17",
        "E17",
        [
            {"id": "baseline", "role": "baseline"},
            {"id": "stack_all", "role": "candidate"},
            {
                "id": "label_free",
                "role": "control",
                "overlay": lf,
                "required_control": True,
                "supervision_label": "target_label_free",
            },
        ],
        "H0",
        "final",
        ["G4_frozen_selection"],
        phase="final",
        additional_cases=["H1", "H2"],
        source_cap=None,
        seeds=[17, 29, 43],
        execution_modes=["global_alignment", "local_ranking"],
    )
    if bindings.get("published_matcher"):
        make_step(
            "E17-published",
            "E17",
            [
                {
                    "id": "logmap",
                    "role": "baseline",
                    "published_matcher": bindings["published_matcher"],
                    "supervision_label": "target_label_free",
                }
            ],
            "H0",
            "final",
            ["G4_frozen_selection"],
            phase="final",
            additional_cases=["H1", "H2"],
            source_cap=None,
            seeds=[17, 29, 43],
            execution_modes=["global_alignment"],
        )
    if overrides:
        raise ValueError(f"unknown comparison overrides: {sorted(overrides)}")
    value = {
        "campaign_id": design["campaign_id"],
        "blueprint": {"path": str(blueprint.resolve()), "sha256": sha256_file(blueprint)},
        "base_config": str(base_config.resolve()),
        "cases": cases,
        "steps": steps,
        "freeze_step": "G4",
        "composition_sources": component_sources,
        **bindings.get("campaign", {}),
    }
    lock = CampaignLock.model_validate(value)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "campaign.lock.yaml"
    if path.exists():
        raise FileExistsError(
            "Preparation output exists; preserve its immutable design and use a new directory"
        )
    path.write_text(dump_yaml_document(lock.model_dump(mode="json")))
    load_campaign(path)
    inventory: dict[str, list[dict[str, Any]]] = {}
    for family, arms in recipes.items():
        inventory[family] = []
        for arm in arms:
            comparisons = [
                step["id"]
                for step in steps
                if step["family"] == family
                and any(item["id"] == arm["id"] for item in step["arms"])
            ]
            record = {
                "arm": arm["id"],
                "requires": requirements.get(f"{family}/{arm['id']}", []),
                "comparisons": comparisons,
                "status": "declared",
            }
            if family in {"E05", "E20"} and arm["id"] == "combined":
                record.update(
                    status="conditional_unadmitted",
                    reason="A progressive combination requires supported component results and a separate frozen recipe/budget; it is omitted from core execution, not scored as baseline.",
                )
            inventory[family].append(record)
    (destination / "arm-prerequisites.yaml").write_text(dump_yaml_document(inventory))
    return path
