"""One bounded E22 crossover fit and its independent policy-versus-fixed follow-up."""

from __future__ import annotations

import json
import math
from pathlib import Path

from exact.impl.models.selector.fitting import safe_training_labels
from exact.impl.models.selector.label_budget import (
    fit_count_policy,
    select_label_budget,
)
from exact.utils.data import read_table
from exact.utils.fitted_artifacts import fingerprint, freeze_json
from exact.utils.mappings import candidate_table_views
from exact.utils.provenance import sha256_file

COMPONENTS = (
    "retrieval",
    "fusion",
    "rerank",
    "llm",
    "accept",
    "calibration",
    "structure",
    "relation",
)
BUDGET_ARMS = ("budget_25", "budget_100", "budget_400")


def prepare_training_units(candidates, reference, path, *, binding, budget=100, seed=17):
    """Count a passive training budget without scoring or reading reporting labels."""
    table = read_table(Path(candidates))
    frame, _ = candidate_table_views(table)
    if "confirmed_label" in table:
        confirmed = table.iloc[:, :2].copy()
        confirmed.columns = ["Src", "Tgt"]
        confirmed["confirmed_label"] = table.confirmed_label
        frame = frame.merge(confirmed, on=["Src", "Tgt"], validate="one_to_one")
    ref = read_table(Path(reference))
    pairs = {
        (str(source), str(target))
        for source, target in ref.iloc[:, :2].itertuples(index=False, name=None)
    }
    if set(frame.Src.astype(str)) & set(binding.get("source_ids", [])):
        raise ValueError("Effective-unit preparation overlaps reporting source groups")
    frame, pairs = safe_training_labels(frame, pairs, binding)
    _, units = select_label_budget(frame, pairs, budget=budget, seed=seed)
    return freeze_json(
        path,
        {
            **units,
            "binding": binding,
            "inputs": {
                "candidates_sha256": sha256_file(Path(candidates)),
                "reference_sha256": sha256_file(Path(reference)),
            },
        },
    )


def prepare_count_policy_followup(
    paired_bootstrap_path,
    training_units_by_arm,
    evaluation_units_path,
    destination,
    *,
    components=("rerank", "accept"),
    fixed_minimum=100,
    practical_effect=0.003,
    stage="screen",
    evaluation_role="development",
):
    """Freeze a conservative crossover, then return two existing-runner config overlays.

    The three curve points fit the policy; an independently frozen development
    population evaluates it. Both arms use exactly the same evaluation labels,
    candidate recipe and units. No empirical readiness or benefit is fabricated.
    """
    if stage != "screen" or evaluation_role != "development":
        raise ValueError("E22 policy selection and its follow-up require development roles")
    if set(training_units_by_arm) != set(BUDGET_ARMS):
        raise ValueError("E22 policy needs exactly the three bounded passive budget arms")
    if tuple(components) != ("rerank", "accept"):
        raise ValueError(
            "E22 count-policy follow-up currently supports the joint rerank/accept recipe"
        )
    if fixed_minimum < 0 or not math.isfinite(practical_effect) or practical_effect < 0:
        raise ValueError("E22 fixed count and practical effect must be nonnegative")
    document = json.loads(Path(paired_bootstrap_path).read_text())
    if document.get("schema_version") != 1 or document.get("resampling_unit") != "source_entity":
        raise ValueError("E22 needs source-group paired bootstrap evidence")
    units = {arm: json.loads(Path(path).read_text()) for arm, path in training_units_by_arm.items()}
    donor_binding = units[BUDGET_ARMS[0]]["binding"]
    if any(value["binding"] != donor_binding for value in units.values()):
        raise ValueError("E22 budget curves must share the same frozen feature/pool binding")
    for arm, expected in zip(BUDGET_ARMS, (25, 100, 400)):
        if units[arm].get("requested_groups") != expected:
            raise ValueError("E22 count-policy arm does not match its frozen label budget")
    for key in ("nested_order_sha256", "pool_sha256", "reference_sha256"):
        if len({value.get(key) for value in units.values()}) != 1:
            raise ValueError("E22 budget curves must share the same training pool and label order")
    memberships = [set(units[arm]["selected_sources"]) for arm in BUDGET_ARMS]
    if not memberships[0] <= memberships[1] <= memberships[2]:
        raise ValueError("E22 passive budgets must be nested source groups")
    evaluation_units = json.loads(Path(evaluation_units_path).read_text())
    evaluation_binding = evaluation_units["binding"]
    donor_sources = set(donor_binding.get("source_ids", []))
    evaluation_sources = set(evaluation_binding.get("source_ids", []))
    if not donor_sources or not evaluation_sources or donor_sources & evaluation_sources:
        raise ValueError(
            "E22 policy evaluation requires independent frozen development source groups"
        )
    if not donor_binding.get("entity_kinds") or donor_binding[
        "entity_kinds"
    ] != evaluation_binding.get("entity_kinds"):
        raise ValueError("E22 crossover cannot change the measured entity kind")
    if evaluation_sources & memberships[-1] or donor_sources & set(
        evaluation_units["selected_sources"]
    ):
        raise ValueError(
            "E22 policy learning and evaluation have overlapping training/reporting sources"
        )
    for value in [*units.values(), evaluation_units]:
        if value.get("selection") != "passive" or value.get("seed") != 17:
            raise ValueError("E22 count policy requires the frozen passive seed-17 curve")
        counts = value["component_units"]
        if counts["rerank"] != counts["accept"]:
            raise ValueError(
                "E22 joint selector requires equal effective rank/accept source counts"
            )
    records = []
    for arm in BUDGET_ARMS:
        matches = [
            row
            for row in document["rows"]
            if row.get("experiment_id") == "E22"
            and row.get("baseline_arm") == "label_free"
            and row.get("candidate_arm") == arm
            and row.get("endpoint_scope") == "overall"
            and row.get("metric") == "task_macro_global_F1"
        ]
        if len(matches) != 1 or matches[0].get("status") != "complete":
            raise ValueError(f"E22 missing one complete paired budget comparison: {arm}")
        row = matches[0]
        if not all(
            math.isfinite(float(row[key])) for key in ("delta", "ci_low", "ci_high")
        ) or not float(row["ci_low"]) <= float(row["delta"]) <= float(row["ci_high"]):
            raise ValueError("E22 paired interval is invalid")
        records.append(
            {
                "role": "development",
                "arm": arm,
                "effective_groups": int(units[arm]["component_units"][components[0]]),
                "gain": float(row["delta"]),
                "gain_ci_low": float(row["ci_low"]),
                "gain_ci_high": float(row["ci_high"]),
                "paired_units": int(row["n_units"]),
            }
        )
    payload = fit_count_policy(
        records,
        None,
        component=components[0],
        binding=evaluation_binding,
        practical_effect=practical_effect,
    )
    rule = payload["components"][components[0]]
    payload["components"] = {
        component: {
            **rule,
            "count_definition": evaluation_units["component_definitions"][component],
        }
        for component in components
    }
    payload["selection_provenance"] = {
        "donor_binding": donor_binding,
        "evaluation_role": evaluation_role,
        "paired_bootstrap_sha256": sha256_file(Path(paired_bootstrap_path)),
        "training_units_sha256": {
            arm: sha256_file(Path(path)) for arm, path in training_units_by_arm.items()
        },
        "evaluation_units_sha256": sha256_file(Path(evaluation_units_path)),
        "budget_source_sets_sha256": fingerprint([sorted(items) for items in memberships]),
        "fixed_minimum": fixed_minimum,
    }
    destination = Path(destination)
    policy_path = destination / "count_policy.json"
    freeze_json(policy_path, payload)
    shared = {
        "mode": "label_free",
        "components": {
            name: ("auto" if name in components else "label_free") for name in COMPONENTS
        },
        "label_budget": evaluation_units.get("requested_groups"),
        "label_selection": "passive",
        "artifacts": {"training_units": str(Path(evaluation_units_path).resolve())},
    }
    selector = {
        "enabled": True,
        "runtime_enabled": True,
        "runtime_global_only": False,
        "rerank": {"artifact": None},
    }
    count = int(evaluation_units["component_units"][components[0]])
    fixed_enabled = count >= fixed_minimum
    fitted_enabled = rule["minimum_groups"] is not None and count >= rule["minimum_groups"]
    arms = [
        {
            "id": "fixed_count",
            "role": "baseline",
            "supervision_label": "in_pair_supervised" if fixed_enabled else "target_label_free",
            "overlay": {
                "selector": selector,
                "supervision": {
                    **shared,
                    "auto_policy": {"kind": "threshold", "artifact": None},
                    "min_effective_training_units": {
                        component: {
                            "unit": evaluation_units["component_definitions"][component],
                            "minimum": fixed_minimum,
                        }
                        for component in components
                    },
                },
            },
        },
        {
            "id": "fitted_count",
            "role": "candidate",
            "supervision_label": "in_pair_supervised" if fitted_enabled else "target_label_free",
            "overlay": {
                "selector": selector,
                "supervision": {
                    **shared,
                    "auto_policy": {"kind": "profile_rule", "artifact": str(policy_path.resolve())},
                },
            },
        },
    ]
    return freeze_json(
        destination / "followup.json",
        {
            "schema_version": 1,
            "kind": "E22_policy_followup",
            "policy_path": str(policy_path.resolve()),
            "arms": arms,
            "requires": [
                "same_selected_E22_component_recipe",
                "independent_development_evaluation",
                "complete_paired_fixed_vs_policy_result",
            ],
            "policy_crossover": rule["crossover"],
            "ready_to_claim_benefit": False,
        },
    )


def materialize_followup(source, suite, manifests):
    """Bind the one D1 policy comparison to completed E22 screen artifacts.

    This is a declaration adapter: it reads completed outputs and train-only
    counts, writes immutable inputs, and never scores, fits a head or calls an LLM.
    """
    from dataclasses import replace

    from exact.core.entities.configs.config import ConfigModel
    from exact.experiments.harness import (
        ExperimentSource,
        _inventory_config,
        _paired_bootstrap_rows,
        deep_merge,
        stable_bootstrap_seed,
    )
    from exact.experiments.schema import ExperimentConfig
    from exact.utils.provenance import dataset_signature_for_paths

    settings = source.config.frozen_constants.get("label_policy_followup")
    if not settings:
        return source
    if settings.get("producer", "E22") != "E22":
        raise ValueError("Label-policy follow-up requires the E22 producer")
    stage = source.config.screen
    producer = suite.by_id["E22"]
    if (
        len(stage.tasks) != 1
        or stage.seeds != [17]
        or stage.tasks[0].split_role != "development"
        or len(producer.config.screen.tasks) != 1
        or producer.config.screen.seeds != [17]
    ):
        raise ValueError("E22 policy follow-up requires one development task and seed 17")
    if {arm.id for arm in source.config.arms} != {"fixed_count", "fitted_count"}:
        raise ValueError("E22 policy follow-up declares fixed_count and fitted_count only")
    selected = [
        item
        for item in manifests
        if item.get("experiment_id") == "E22"
        and item.get("stage") == "screen"
        and item.get("arm_id") in {*BUDGET_ARMS, "label_free"}
    ]
    if (
        len(selected) != 4
        or {item.get("arm_id") for item in selected} != {*BUDGET_ARMS, "label_free"}
        or any(item.get("status") != "complete" or item.get("seed") != 17 for item in selected)
    ):
        raise ValueError("E22 policy requires all completed passive and label-free screen cells")
    roots = set()
    units_by_arm = {}
    output_dirs = {}
    for item in selected:
        output = Path(item["fingerprint_payload"]["output_dir"]).resolve()
        if (
            len(output.parents) < 5
            or output.parents[3].name != "runs"
            or output.parents[4].name != "screen"
        ):
            raise ValueError("E22 cell is outside the declared screen/runs layout")
        roots.add(output.parents[4])
        output_dirs[str(output)] = item
        if item["arm_id"] in BUDGET_ARMS:
            matches = sorted((output / "fitting").glob("**/training_units.json"))
            if len(matches) != 1:
                raise ValueError(
                    f"E22 {item['arm_id']} needs one unambiguous completed training-unit artifact"
                )
            units_by_arm[item["arm_id"]] = matches[0]
    if len(roots) != 1:
        raise ValueError("E22 policy cannot mix completed outputs from different stages")
    stage_root = roots.pop()
    records = [
        row
        for row in json.loads((stage_root / "metrics.json").read_text())["rows"]
        if str(Path(row["output_dir"]).resolve()) in output_dirs
    ]
    if len(records) != 4 or any(
        row.get("config_hash")
        != output_dirs[str(Path(row["output_dir"]).resolve())].get("resolved_config_hash")
        or row.get("status") != "complete"
        for row in records
    ):
        raise ValueError("E22 aggregate metrics do not match the completed producer manifests")
    # Stage-wide bootstrap is finalized later; compute just this already-complete
    # producer using the same source-unit inference code and frozen seed.
    destination = stage_root / "policies" / source.config.experiment_id
    evidence_path = destination / "paired_bootstrap.json"
    freeze_json(
        evidence_path,
        {
            "schema_version": 1,
            "resampling_unit": "source_entity",
            "default_resamples": 10000,
            "completed_records_sha256": fingerprint(records),
            "rows": _paired_bootstrap_rows(
                replace(suite, sources=(producer,)),
                records,
                stage="screen",
                resamples=10000,
                seed=stable_bootstrap_seed(suite.suite_id, "screen"),
            ),
        },
    )
    config = _inventory_config(source, stage.tasks[0], "screen")
    data = config.data
    required = [
        data.source,
        data.target,
        data.train_candidates,
        data.refs.get("train"),
        data.source_universe,
    ]
    if any(path is None or not Path(path).is_file() for path in required):
        raise ValueError(
            "E22 follow-up requires explicit ontologies, train pool/reference and frozen reporting sources"
        )
    binding = {
        "dataset_signature": dataset_signature_for_paths(data.source, data.target),
        "source_ids": sorted(set(data.source_universe.read_text().splitlines())),
        "negative_label_policy": config.supervision.negative_label_policy,
        "entity_kinds": [getattr(kind, "value", kind) for kind in config.matching.entity_kinds],
    }
    units_path = destination / "evaluation_training_units.json"
    prepare_training_units(
        data.train_candidates,
        data.refs["train"],
        units_path,
        binding=binding,
        budget=int(settings.get("budget", 100)),
        seed=17,
    )
    result = prepare_count_policy_followup(
        evidence_path,
        units_by_arm,
        units_path,
        destination,
        components=tuple(settings.get("components", ("rerank", "accept"))),
        fixed_minimum=int(settings.get("fixed_minimum", 100)),
        practical_effect=float(settings.get("practical_effect", 0.003)),
    )
    replacements = {arm["id"]: arm for arm in result["arms"]}
    declaration = source.config.model_dump(mode="json")
    declaration["base_config"] = str(source.base_config_path)
    for arm in declaration["arms"]:
        resolved = replacements[arm["id"]]
        arm["overlay"] = deep_merge(arm["overlay"], resolved["overlay"])
        arm["supervision_label"] = resolved["supervision_label"]
        ConfigModel.from_mapping(deep_merge(config.model_dump(mode="python"), arm["overlay"]))
    declaration["frozen_constants"]["resolved_label_policy"] = {
        "source_declaration_sha256": source.raw_hash(),
        "policy_sha256": sha256_file(Path(result["policy_path"])),
        "followup_sha256": sha256_file(destination / "followup.json"),
    }
    resolved = ExperimentConfig.model_validate(declaration)
    path = destination / "resolved-experiment.json"
    freeze_json(path, resolved.model_dump(mode="json"))
    return ExperimentSource(config=resolved, path=path)
