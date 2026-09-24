"""Small verified consumers of completed development selections; no model calls."""

from __future__ import annotations

import json
from pathlib import Path

from exact.core.entities.configs.config import ConfigModel
from exact.utils.fitted_artifacts import fingerprint, freeze_json
from exact.utils.provenance import sha256_file


class PrerequisiteUnavailable(ValueError):
    """A conditional comparison is not an executed negative result."""

    def __init__(self, reason, *, status="blocked_input_resolution", code="selection_prerequisite"):
        super().__init__(reason)
        self.status, self.code = status, code


def _selected_cell(producer, manifests, selections):
    result = selections.get(producer, {})
    decisions = result.get("decisions", [])
    if result.get("status") not in {"selected", "screened_out"} or len(decisions) != 1:
        raise PrerequisiteUnavailable(f"{producer} needs one completed development selection")
    arm = decisions[0].get("selected_arm") or decisions[0].get("baseline")
    return _completed_cell(producer, arm, manifests), result


def _completed_cell(producer, arm, manifests):
    matches = [
        item
        for item in manifests
        if item.get("experiment_id") == producer and item.get("arm_id") == arm
    ]
    if len(matches) != 1:
        raise PrerequisiteUnavailable(f"{producer}/{arm} needs one completed development cell")
    item = matches[0]
    if (
        item.get("stage") != "screen"
        or item.get("split_role") != "development"
        or item.get("status") != "complete"
        or item.get("return_code") != 0
    ):
        raise PrerequisiteUnavailable(f"{producer}/{arm} is not a completed development cell")
    path = Path(item["fingerprint_payload"]["output_dir"]) / "_inputs/resolved.config.yaml"
    config = ConfigModel.load_config(path)
    if config.fingerprint() != item.get("resolved_config_hash"):
        raise ValueError(f"{producer}/{arm}: completed resolved configuration changed")
    if config.data.reference_role != "valid":
        raise ValueError("Selected recipes must use development references")
    return item, config, path


def _freeze(source, suite, record, overlays):
    from exact.experiments.harness import ExperimentSource, deep_merge
    from exact.experiments.schema import ExperimentConfig

    destination = Path(suite.campaign["root"]) / "policies" / source.config.experiment_id
    binding = destination / (fingerprint(record) + ".json")
    freeze_json(binding, record)
    declaration = source.config.model_dump(mode="json", by_alias=True)
    declaration["base_config"] = str(source.base_config_path)
    for arm in declaration["arms"]:
        arm["overlay"] = deep_merge(arm["overlay"], overlays[arm["id"]])
    declaration["frozen_constants"]["staged_selection_binding"] = {
        "path": str(binding.resolve()),
        "sha256": sha256_file(binding),
    }
    resolved = ExperimentConfig.model_validate(declaration)
    path = destination / (fingerprint(declaration) + ".json")
    freeze_json(path, resolved.model_dump(mode="json", by_alias=True))
    return ExperimentSource(config=resolved, path=path)


def _identity(item, config_path):
    return {
        "producer": item["experiment_id"],
        "arm": item["arm_id"],
        "config_sha256": sha256_file(config_path),
        "pool_fingerprint": item.get("candidate_pool_fingerprint"),
        "artifacts": (item.get("recovery") or {}).get("artifacts", {}),
    }


def materialize_analytic_selection(source, suite, manifests, selections):
    """Freeze the six-setting table before either acceptance recipe can fit."""
    from exact.experiments.harness import _metric_value, cell_metrics

    (item, config, config_path), selection = _selected_cell("E10-analytic", manifests, selections)
    table = []
    for gamma in (1, 2, 3):
        for tau in (0.4, 0.5):
            arm = f"analytic_g{gamma}_t{str(tau).replace('.', '')}"
            cell, recipe, path = _completed_cell("E10-analytic", arm, manifests)
            if (recipe.matching.fusion.gamma, recipe.matching.fusion.tau) != (gamma, tau):
                raise ValueError("E10 analytic arm disagrees with its declared constants")
            if cell.get("candidate_pool_fingerprint") != item.get("candidate_pool_fingerprint"):
                raise ValueError("E10 analytic selection changed the frozen candidate pool")
            f1 = _metric_value(cell_metrics(path.parent.parent), "F1")
            if f1 is None:
                raise ValueError("E10 analytic selection requires all six F1 outcomes")
            table.append({**_identity(cell, path), "gamma": gamma, "tau": tau, "F1": f1})
    overlay = {
        "matching": {"fusion": config.matching.fusion.model_dump(mode="json", by_alias=True)}
    }
    record = {
        "schema_version": 1,
        "kind": "selected_analytic_setting",
        "reference_role": "valid",
        "selection_sha256": fingerprint(selection),
        "selected": _identity(item, config_path),
        "constants": overlay["matching"]["fusion"],
        "analytic_table": table,
    }
    return _freeze(source, suite, record, {arm.id: overlay for arm in source.config.arms})


def _score_contract(config):
    """The no-call control and judge must differ only in their judgment treatment."""
    mapping = config.model_dump(mode="json", by_alias=True)
    return {
        key: mapping[key]
        for key in ("pipeline", "matching", "selector", "supervision", "candidates", "dataset")
    }


def _judge_benefit(judge_item, judge_config, judge_path, manifests):
    from exact.experiments.harness import _metric_value, cell_metrics

    baseline, config, path = _completed_cell("E25", "decision_off", manifests)
    if (
        not judge_item.get("candidate_pool_fingerprint")
        or judge_item["candidate_pool_fingerprint"] != baseline.get("candidate_pool_fingerprint")
        or any(
            judge_item.get(key) != baseline.get(key) for key in ("task_id", "seed", "source_cap")
        )
        or _score_contract(judge_config) != _score_contract(config)
    ):
        raise PrerequisiteUnavailable(
            "Judge benefit needs the same population, scores and fixed acceptance as E25 decision_off"
        )
    if config.llm.experiment.gate.mode != "off":
        raise ValueError("Judge benefit control did not disable decision calls")
    scores = [_metric_value(cell_metrics(p.parent.parent), "F1") for p in (path, judge_path)]
    if any(value is None for value in scores):
        raise PrerequisiteUnavailable("Judge benefit needs both completed development F1 outcomes")
    evidence = {
        "baseline": _identity(baseline, path),
        "metric": "F1",
        "minimum_gain": 0.003,
        "baseline_score": scores[0],
        "judge_score": scores[1],
        "gain": scores[1] - scores[0],
    }
    if evidence["gain"] < evidence["minimum_gain"]:
        raise PrerequisiteUnavailable(
            "Selected judge does not establish the declared development benefit over decision_off: "
            + json.dumps(evidence, sort_keys=True),
            status="screened_out",
            code="judge_benefit_not_established",
        )
    return evidence


def _training_prerequisites(config):
    """Inspect only the declared train pool; never promote valid/test judgments to labels."""
    from exact.utils.data import read_table

    data = config.data
    if data.train_candidates is None or not data.refs.get("train"):
        raise PrerequisiteUnavailable("E21 needs a disjoint training pool and a training reference")
    root = Path(data.root or ".")
    paths = [root / data.train_candidates, root / data.refs["train"]]
    if not all(path.is_file() for path in paths):
        raise PrerequisiteUnavailable("E21 training bindings are not materialized")
    if any(
        path.resolve() == (root / value).resolve()
        for path in paths
        for key, value in data.refs.items()
        if key != "train"
    ):
        raise ValueError("E21 training input aliases a development/reporting reference")
    pool = read_table(paths[0])
    source_column = "SrcEntity" if "SrcEntity" in pool else "Src"
    if len(pool.columns) == 3 and any("cand" in str(column).lower() for column in pool.columns):
        # Bio-ML list files keep source IDs first; the middle gold column is not a feature.
        source_column = pool.columns[0]
    if source_column not in pool or pool.empty:
        raise PrerequisiteUnavailable("E21 training pool needs source identities")
    train_sources = set(pool[source_column].astype(str))
    if data.source_universe is None or not (root / data.source_universe).is_file():
        raise PrerequisiteUnavailable("E21 needs its frozen development source universe")
    if train_sources & set((root / data.source_universe).read_text().splitlines()):
        raise ValueError("E21 training sources overlap development sources")
    policy = config.supervision.negative_label_policy
    if policy == "confirmed_negatives":
        if "confirmed_label" not in pool or not pool.confirmed_label.isin([0, 1]).all():
            raise PrerequisiteUnavailable(
                "E21 benefit router needs explicit labels for every training candidate"
            )
    elif policy != "complete_reference":
        raise PrerequisiteUnavailable(
            "E21 cannot derive negatives from positive-unlabelled references"
        )
    return {
        "reference_role": "train",
        "negative_label_policy": policy,
        "inputs": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths],
        "source_ids_sha256": fingerprint(sorted(train_sources)),
        "source_count": len(train_sources),
        "teacher_generation": "actual_train_only_forced_calls_with_request_ledger",
    }


def materialize_selected_judge(source, suite, manifests, selections):
    """Bind the actual E07 winner, checking each consumer's intended contract."""
    from exact.experiments.harness import _inventory_config, deep_merge

    (item, producer, path), selection = _selected_cell("E07", manifests, selections)
    judge = producer.llm.experiment.model_dump(mode="json", by_alias=True)
    identifier = source.config.experiment_id
    if judge["decision"]["mode"] not in {"listwise", "listwise_sc"}:
        raise PrerequisiteUnavailable(
            f"{identifier} needs a selected comparative judge; the E07 winner is binary",
            status="inapplicable",
            code="incompatible_selected_judge",
        )
    if identifier == "E21" and (
        judge["decision"]["evidence"] == "generated_brief"
        or judge["fusion_weight"] != "source_first"
    ):
        raise PrerequisiteUnavailable(
            "E21 benefit routing needs the selected packet judge with source_first integration",
            status="inapplicable",
            code="incompatible_selected_judge",
        )
    record = {
        "schema_version": 1,
        "kind": "selected_judge",
        "reference_role": "valid",
        "selection_sha256": fingerprint(selection),
        "selected": _identity(item, path),
        "judge": judge,
        "training": [],
    }
    if identifier == "E21":
        if (
            producer.selector.runtime_enabled
            or producer.matching.calibration.threshold_mode != "fixed"
        ):
            raise PrerequisiteUnavailable(
                "E21 judge benefit requires fixed pair-threshold acceptance without a selector"
            )
        record["benefit"] = _judge_benefit(item, producer, path, manifests)
    overlays = {}
    for arm in source.config.arms:
        if identifier == "E04-listwise" and arm.id != "listwise_none":
            overlays[arm.id] = {}
            continue
        # Preserve the selected profile/provider and source-first policy. Only the
        # declared learning intervention may change gate/exemplars/student state.
        llm = deep_merge(
            producer.llm.model_dump(mode="json", by_alias=True), arm.overlay.get("llm", {})
        )
        overlays[arm.id] = {"llm": llm}
    for task in source.config.screen.tasks:
        consumer = _inventory_config(source, task, "screen")
        decision_profile = (
            producer.llm.routing.decision_profile or producer.llm.routing.default_profile
        )
        consumer_profile = (
            consumer.llm.routing.decision_profile or consumer.llm.routing.default_profile
        )
        if decision_profile != consumer_profile or producer.llm.profiles.get(
            decision_profile
        ) != consumer.llm.profiles.get(consumer_profile):
            raise PrerequisiteUnavailable(
                "Selected E07 provider/model profile differs from the consumer"
            )
        if identifier == "E21":
            if _score_contract(producer) != _score_contract(
                ConfigModel.from_mapping(
                    deep_merge(
                        consumer.model_dump(mode="json", by_alias=True),
                        source.config.arms[0].overlay,
                    )
                )
            ):
                raise PrerequisiteUnavailable(
                    "E21 must preserve the selected judge's scorer and fixed acceptance recipe"
                )
            record["training"].append(_training_prerequisites(consumer))
    return _freeze(source, suite, record, overlays)
