"""Explicit fixed-judge E04 diagnostic; never substitutes an E07 selection."""

from exact.core.entities.configs.config import ConfigModel
from exact.experiments.staged_selection import (
    PrerequisiteUnavailable,
    _completed_cell,
    _freeze,
    _identity,
    _selected_cell,
)
from exact.utils.fitted_artifacts import fingerprint

DIAGNOSTIC_ID = "E04-listwise-diagnostic"
FIXED_JUDGE = {
    "producer": "E07",
    "arm": "facts_listwise",
    "selection_eligible": False,
    "label_semantics": "benchmark_pool",
}


def materialize_fixed_listwise(source, suite, manifests, selections):
    """Freeze a non-selecting N0 comparison against the completed listwise comparator."""
    from exact.experiments.harness import _inventory_config, deep_merge

    constants = source.config.frozen_constants
    arms = {arm.id: arm for arm in source.config.arms}
    if (
        source.config.experiment_id != DIAGNOSTIC_ID
        or constants.get("fixed_listwise_diagnostic") != FIXED_JUDGE
        or source.config.selection.decisions
        or set(arms) != {"nil_heuristic", "listwise_none"}
        or arms["nil_heuristic"].role != "baseline"
        or arms["listwise_none"].role != "diagnostic"
        or arms["listwise_none"].deployable
        or source.config.screen.seeds != [17]
        or constants.get("campaign_v2", {}).get("policy_paths")
    ):
        raise ValueError("Fixed E04 listwise diagnostic requires its explicit non-selecting scope")
    (winner, _, winner_path), selection = _selected_cell("E07", manifests, selections)
    if winner["arm_id"] != "facts_binary":
        raise PrerequisiteUnavailable(
            "Fixed E04 amendment is bound to the completed facts_binary winner"
        )
    item, producer, path = _completed_cell("E07", "facts_listwise", manifests)
    judge = producer.llm.experiment.model_dump(mode="json", by_alias=True)
    if (
        not judge["enabled"]
        or judge["decision"]["mode"] != "listwise"
        or judge["decision"]["evidence"] != "structured_packet"
        or judge["decision"]["probability"] != "raw_joint"
        or judge["decision"]["output"] != "categorical"
        or judge["decision"]["listwise_max_candidates"] != 5
        or judge["decision"]["permutations"] != 1
        or judge["fusion_weight"] != "source_first"
        or judge["gate"]["mode"] != "source_top_fraction"
        or judge["gate"]["quantile_fraction"] != 1.0
        or judge["distill"] != "off"
        or judge["exemplars"] != "off"
    ):
        raise ValueError(
            "E07 facts_listwise differs from the fixed candidate/NONE diagnostic contract"
        )
    overlays = {
        "nil_heuristic": {},
        "listwise_none": {"llm": producer.llm.model_dump(mode="json", by_alias=True)},
    }
    tasks = source.config.screen.tasks
    if len(tasks) != 1 or tasks[0].id != "N0-global_alignment":
        raise ValueError("Fixed E04 listwise diagnostic is restricted to N0")
    for task in tasks:
        if task.reference_role != "valid" or task.split_role != "development":
            raise ValueError("Fixed E04 listwise diagnostic requires development references")
        annotation = constants.get("evaluation_diagnostics", {}).get(task.id, {})
        if annotation.get("label_semantics") != "benchmark_pool":
            raise ValueError("Fixed E04 listwise diagnostic requires benchmark-pool NIL labels")
        consumer = _inventory_config(source, task, "screen")
        profile = producer.llm.routing.decision_profile or producer.llm.routing.default_profile
        consumer_profile = (
            consumer.llm.routing.decision_profile or consumer.llm.routing.default_profile
        )
        if profile != consumer_profile or producer.llm.profiles.get(
            profile
        ) != consumer.llm.profiles.get(consumer_profile):
            raise PrerequisiteUnavailable("Fixed E07 provider/model profile differs from N0")
        base = consumer.model_dump(mode="json", by_alias=True)
        resolved = {}
        for name, arm in arms.items():
            resolved[name] = ConfigModel.from_mapping(deep_merge(base, arm.overlay))
            if (
                resolved[name].matching.nil.mode != "heuristic"
                or not resolved[name].selector.runtime_enabled
            ):
                raise ValueError(
                    "Both fixed E04 diagnostic arms require heuristic NIL and the same selector"
                )
        baseline = resolved["nil_heuristic"].model_dump(mode="json", by_alias=True)
        treatment = resolved["listwise_none"].model_dump(mode="json", by_alias=True)
        if resolved["nil_heuristic"].llm.experiment.gate.mode != "off":
            raise ValueError("Fixed E04 diagnostic control must disable decision calls")
        if {k: v for k, v in baseline.items() if k != "llm"} != {
            k: v for k, v in treatment.items() if k != "llm"
        }:
            raise ValueError("Fixed E04 diagnostic arms may differ only in their LLM treatment")
    record = {
        "schema_version": 1,
        "kind": "fixed_listwise_diagnostic",
        **FIXED_JUDGE,
        "reference_role": "valid",
        "selection_sha256": fingerprint(selection),
        "actual_E07_winner": _identity(winner, winner_path),
        "fixed_comparator": _identity(item, path),
        "judge": judge,
        "training": [],
        "none_semantics": "none_of_displayed_top_five; not ontology-wide absence",
        "integration": "source_first_choice_then_unchanged_heuristic_NIL_acceptance",
    }
    return _freeze(source, suite, record, overlays)
