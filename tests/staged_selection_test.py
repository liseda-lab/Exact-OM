"""Exercise staged selections with durable synthetic outputs, without model calls."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.core.entities.configs.config import ConfigModel
from exact.core.entities.configs.yaml_io import dump_yaml_document
from exact.experiments import harness
from exact.experiments.campaign import validate_comparison_cells
from exact.experiments.fitting_recipes import _supervision, fitting_arms
from exact.experiments.staged_selection import (
    PrerequisiteUnavailable,
    _training_prerequisites,
    materialize_analytic_selection,
    materialize_selected_judge,
)
from tests.label_policy_test import declaration


@pytest.fixture
def bound(tmp_path):
    paths = {}
    for name, text in {
        "source.owl": "source",
        "target.owl": "target",
        "sources.txt": "dev\n",
        "train.tsv": "Src\tTgt\tconfirmed_label\ntrain-a\tx\t1\ntrain-a\ty\t0\ntrain-b\ty\t1\ntrain-b\tx\t0\n",
        "refs.tsv": "Src\tTgt\ntrain-a\tx\ntrain-b\ty\n",
        "valid.tsv": "Src\tTgt\ndev\tx\n",
    }.items():
        paths[name] = tmp_path / name
        paths[name].write_text(text)
    base = ConfigModel.load_config("exact/default_config.yaml").model_dump(
        mode="json", by_alias=True
    )
    base = harness.deep_merge(
        base,
        {
            "data": {
                "source": str(paths["source.owl"]),
                "target": str(paths["target.owl"]),
                "source_universe": str(paths["sources.txt"]),
                "train_candidates": str(paths["train.tsv"]),
                "refs": {"train": str(paths["refs.tsv"]), "valid": str(paths["valid.tsv"])},
                "reference_role": "valid",
                "execution_mode": "global_alignment",
            },
            "supervision": {**_supervision(), "negative_label_policy": "confirmed_negatives"},
            "selector": {"runtime_enabled": False},
        },
    )
    suite = SimpleNamespace(campaign={"root": str(tmp_path / "runtime")})
    return base, suite


def complete(tmp_path, producer, arm, mapping, *, score=0.6):
    config = ConfigModel.from_mapping(mapping)
    path = tmp_path / producer / arm / "_inputs/resolved.config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(dump_yaml_document(config.model_dump(mode="json", by_alias=True)))
    item = {
        "experiment_id": producer,
        "arm_id": arm,
        "stage": "screen",
        "split_role": "development",
        "status": "complete",
        "return_code": 0,
        "task_id": "development",
        "seed": 17,
        "source_cap": 200,
        "candidate_pool_fingerprint": "same-population",
        "resolved_config_hash": config.fingerprint(),
        "fingerprint_payload": {"output_dir": str(path.parent.parent)},
    }
    (path.parent.parent / "fixture-score.json").write_text(json.dumps({"F1": score}))
    return item


def selected(producer, arm, *, status="selected"):
    return {
        producer: {
            "status": status,
            "decisions": [{"selected_arm": arm if status == "selected" else None, "baseline": arm}],
        }
    }


@pytest.fixture(autouse=True)
def synthetic_metrics(monkeypatch):
    monkeypatch.setattr(
        harness, "cell_metrics", lambda path: json.loads((path / "fixture-score.json").read_text())
    )


def test_analytic_selection_freezes_six_outcomes_before_both_acceptance_fits(tmp_path, bound):
    base, suite = bound
    arms = fitting_arms()["E10"]
    manifests = [
        complete(tmp_path, "E10-analytic", arm["id"], harness.deep_merge(base, arm["overlay"]))
        for arm in arms[:6]
    ]
    consumer = declaration(tmp_path, "E10", arms[6:], base)
    selection = selected("E10-analytic", "analytic_g3_t04")
    result = materialize_analytic_selection(consumer, suite, manifests, selection)
    for arm in result.config.arms:
        assert arm.overlay["matching"]["fusion"]["gamma"] == 3
        assert arm.overlay["matching"]["fusion"]["tau"] == 0.4
    binding = result.config.frozen_constants["staged_selection_binding"]
    assert len(json.loads(Path(binding["path"]).read_text())["analytic_table"]) == 6
    again = materialize_analytic_selection(consumer, suite, manifests, selection)
    assert again.path == result.path and again.raw_hash() == result.raw_hash()
    with pytest.raises(PrerequisiteUnavailable, match="completed development cell"):
        materialize_analytic_selection(consumer, suite, manifests[:-1], selection)


def judge_fixture(
    tmp_path, bound, *, evidence="scored_packet", mode="listwise", gain=0.01, fusion=None
):
    base, suite = bound
    resolved_base = harness.apply_rationale_policy(base, generate_rationales=False)
    judge = {
        "enabled": True,
        "decision": {
            "mode": mode,
            "evidence": evidence,
            "brief_max_tokens": 256,
            "max_evidence_packets": 2,
        },
        "gate": {"mode": "source_top_fraction", "quantile_fraction": 1.0},
        "fusion_weight": fusion or ("beta_u" if mode == "binary" else "source_first"),
    }
    manifests = [
        complete(
            tmp_path,
            "E07",
            "winner",
            harness.deep_merge(resolved_base, {"llm": {"experiment": judge}}),
            score=0.6 + gain,
        ),
        complete(
            tmp_path,
            "E25",
            "decision_off",
            harness.deep_merge(
                resolved_base, {"llm": {"experiment": {"enabled": True, "gate": {"mode": "off"}}}}
            ),
            score=0.6,
        ),
    ]
    source = declaration(tmp_path, "E21", fitting_arms()["E21"], base)
    return source, suite, manifests, selected("E07", "winner")


def test_learning_consumes_selected_prompt_and_freezes_train_only_binding(tmp_path, bound):
    source, suite, manifests, selections = judge_fixture(tmp_path, bound)
    result = materialize_selected_judge(source, suite, manifests, selections)
    for arm in result.config.arms:
        judge = arm.overlay["llm"]["experiment"]
        assert judge["decision"]["evidence"] == "scored_packet"
        assert judge["decision"]["brief_max_tokens"] == 256
        assert judge["decision"]["max_evidence_packets"] == 2
        if arm.id.startswith("student"):
            assert judge["fusion_weight"] == "beta_u"
    payload = json.loads(
        Path(result.config.frozen_constants["staged_selection_binding"]["path"]).read_text()
    )
    assert payload["benefit"]["gain"] == pytest.approx(0.01)
    assert payload["training"][0]["reference_role"] == "train"
    assert payload["training"][0]["source_count"] == 2
    assert all("valid.tsv" not in entry["path"] for entry in payload["training"][0]["inputs"])


def test_learning_contract_resolves_campaign_rationale_policy(tmp_path, bound):
    source, suite, manifests, selections = judge_fixture(tmp_path, bound, mode="binary")
    assert any(entry["params"].get("generate_llm_rationales") for entry in bound[0]["pipeline"])
    assert source.config.generate_rationales is False
    materialize_selected_judge(source, suite, manifests, selections)
    source.config.generate_rationales = True
    with pytest.raises(PrerequisiteUnavailable, match="preserve the selected judge"):
        materialize_selected_judge(source, suite, manifests, selections)


def test_learning_preserves_actual_binary_winner_and_beta_u(tmp_path, bound):
    values = judge_fixture(tmp_path, bound, mode="binary", evidence="structured_packet")
    result = materialize_selected_judge(*values)
    for arm in result.config.arms:
        judge = arm.overlay["llm"]["experiment"]
        assert judge["decision"]["mode"] == "binary"
        assert judge["decision"]["evidence"] == "structured_packet"
        assert judge["fusion_weight"] == "beta_u"
    record = json.loads(
        Path(result.config.frozen_constants["staged_selection_binding"]["path"]).read_text()
    )
    assert record["judge"]["decision"]["mode"] == "binary"
    assert record["benefit"]["gain"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"mode": "binary", "fusion": "constant"}, "inapplicable"),
        ({"evidence": "generated_brief"}, "inapplicable"),
        ({"gain": 0.0}, "screened_out"),
    ],
)
def test_conditional_learning_dispositions_are_not_fabricated_success(
    tmp_path, bound, change, expected
):
    values = judge_fixture(tmp_path, bound, **change)
    with pytest.raises(PrerequisiteUnavailable) as error:
        materialize_selected_judge(*values)
    assert error.value.status == expected


def test_judge_benefit_rejects_changed_pool_and_tampered_config(tmp_path, bound):
    source, suite, manifests, selections = judge_fixture(tmp_path, bound)
    manifests[1]["candidate_pool_fingerprint"] = "other-pool"
    with pytest.raises(PrerequisiteUnavailable, match="same population"):
        materialize_selected_judge(source, suite, manifests, selections)
    path = Path(manifests[0]["fingerprint_payload"]["output_dir"]) / "_inputs/resolved.config.yaml"
    config = ConfigModel.load_config(path)
    config.matching.threshold = 0.123
    path.write_text(dump_yaml_document(config.model_dump(mode="json", by_alias=True)))
    with pytest.raises(ValueError, match="configuration changed"):
        materialize_selected_judge(source, suite, manifests, selections)


def test_training_prerequisite_rejects_overlap_unknown_labels_and_dev_alias(tmp_path, bound):
    base, _ = bound
    config = ConfigModel.from_mapping(base)
    config.data.train_candidates.write_text("Src\tTgt\tconfirmed_label\ndev\tx\t1\n")
    with pytest.raises(ValueError, match="overlap"):
        _training_prerequisites(config)
    config.data.train_candidates.write_text("Src\tTgt\tconfirmed_label\ntrain\tx\t\n")
    with pytest.raises(PrerequisiteUnavailable, match="every training candidate"):
        _training_prerequisites(config)
    config.data.refs["train"] = config.data.refs["valid"]
    with pytest.raises(ValueError, match="aliases"):
        _training_prerequisites(config)


def test_e01_rejects_scorer_or_cardinality_changes_and_local_semantics(bound):
    base, _ = bound
    source = SimpleNamespace(
        config=SimpleNamespace(frozen_constants={"campaign_v2": {"family": "E01"}})
    )
    cells = [
        SimpleNamespace(resolved_config=deepcopy(base), task_id="same", seed=17) for _ in range(2)
    ]
    cells[1].resolved_config["matching"]["extraction"]["mode"] = "mutual_best"
    validate_comparison_cells(cells, source)
    cells[1].resolved_config["matching"]["target_cardinality"] = 2
    with pytest.raises(ValueError, match="only extraction mode"):
        validate_comparison_cells(cells, source)
    cells[1].resolved_config["data"]["execution_mode"] = "local_ranking"
    with pytest.raises(ValueError, match="global alignment"):
        validate_comparison_cells(cells, source)


@pytest.fixture
def amended_e01(bound):
    from exact.experiments.core_recipes import core_arms

    base, _ = bound
    arms = [SimpleNamespace(**arm) for arm in core_arms(base)["E01"]]
    source = SimpleNamespace(
        config=SimpleNamespace(arms=arms, frozen_constants={"campaign_v2": {"family": "E01"}})
    )
    cells = [
        SimpleNamespace(
            resolved_config=harness.deep_merge(base, arm.overlay),
            arm_id=arm.id,
            task_id="same",
            seed=17,
        )
        for arm in arms
    ]
    return cells, source


def test_e01_allows_only_declared_unrestricted_baseline(amended_e01):
    cells, source = amended_e01
    original = deepcopy(cells[0].resolved_config)
    validate_comparison_cells(cells, source)
    assert cells[0].resolved_config == original
    # Merely using threshold mode cannot opt an undeclared/legacy comparison into the amendment.
    source.config.arms = source.config.arms[1:]
    with pytest.raises(ValueError, match="only extraction mode"):
        validate_comparison_cells(cells, source)


@pytest.mark.parametrize(
    "arm,path,value",
    [
        ("threshold_unrestricted", "matching.cardinality", 1),
        ("threshold_unrestricted", "matching.target_cardinality", 1),
        ("threshold_unrestricted", "matching.extraction.mode", "greedy"),
        ("greedy", "matching.cardinality", None),
        ("mutual_best", "matching.target_cardinality", 2),
        ("stable_marriage", "matching.extraction.anchor_conflict_policy", "error"),
        ("assignment_accepted_utility", "matching.threshold", 0.8),
        ("greedy", "selector.runtime_enabled", True),
        ("greedy", "llm.experiment.gate.mode", "all"),
        ("assignment_legacy", "matching.fusion.gamma", 0.123),
    ],
)
def test_e01_amendment_rejects_undeclared_scoring_and_extraction_drift(
    amended_e01, arm, path, value
):
    cells, source = amended_e01
    mapping = next(cell.resolved_config for cell in cells if cell.arm_id == arm)
    parts = path.split(".")
    for part in parts[:-1]:
        mapping = mapping[part]
    mapping[parts[-1]] = value
    with pytest.raises(ValueError, match="E01"):
        validate_comparison_cells(cells, source)


@pytest.mark.parametrize("plan_only", [False, True])
def test_historical_analytic_inputs_bind_without_becoming_current_cells(tmp_path, bound, plan_only):
    base, suite = bound
    arms = fitting_arms()["E10"]
    historical = [
        complete(tmp_path, "E10-analytic", arm["id"], harness.deep_merge(base, arm["overlay"]))
        for arm in arms[:6]
    ]
    consumer = declaration(tmp_path, "E10", arms[6:], base)
    consumer.config.frozen_constants["selected_analytic_setting"] = {"producer": "E10-analytic"}
    selections = selected("E10-analytic", "analytic_g2_t05", status="screened_out")
    current = []
    result, inherited = harness._materialize_campaign_evidence(
        consumer,
        suite,
        current,
        selections,
        {},
        plan_only=plan_only,
        external_manifests=historical,
    )
    assert current == [] and inherited == {}
    assert len(result.config.arms) == 2
    assert all(arm.overlay["matching"]["fusion"]["gamma"] == 2 for arm in result.config.arms)
    assert all(arm.overlay["matching"]["fusion"]["tau"] == 0.5 for arm in result.config.arms)
    with pytest.raises(PrerequisiteUnavailable, match="completed development cell"):
        harness._materialize_campaign_evidence(
            consumer,
            suite,
            [],
            selections,
            {},
            plan_only=plan_only,
            external_manifests=historical[:-1],
        )
    with pytest.raises(PrerequisiteUnavailable, match="completed development selection"):
        harness._materialize_campaign_evidence(
            consumer,
            suite,
            [],
            {},
            {},
            plan_only=plan_only,
            external_manifests=historical,
        )
    config_path = (
        Path(historical[0]["fingerprint_payload"]["output_dir"]) / "_inputs/resolved.config.yaml"
    )
    value = ConfigModel.load_config(config_path).model_dump(mode="json", by_alias=True)
    value["matching"]["threshold"] = 0.123
    config_path.write_text(dump_yaml_document(value))
    with pytest.raises(ValueError, match="resolved configuration changed"):
        harness._materialize_campaign_evidence(
            consumer,
            suite,
            [],
            selections,
            {},
            plan_only=plan_only,
            external_manifests=historical,
        )


def test_historical_config_is_verified_before_new_schema_defaults(tmp_path, bound):
    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.staged_selection import _completed_cell

    base, _ = bound
    item = complete(tmp_path, "E10-analytic", "analytic_g2_t05", base)
    path = Path(item["fingerprint_payload"]["output_dir"]) / "_inputs/resolved.config.yaml"
    raw = load_yaml_mapping(path)
    raw["matching"]["extraction"].pop("anchor_conflict_policy")
    item["resolved_config_hash"] = harness.hash_payload(raw)
    path.write_text(dump_yaml_document(raw))
    _, config, _ = _completed_cell("E10-analytic", "analytic_g2_t05", [item])
    assert config.matching.extraction.anchor_conflict_policy == "error"
    assert config.fingerprint() != item["resolved_config_hash"]
    raw["matching"]["fusion"]["gamma"] = 3
    path.write_text(dump_yaml_document(raw))
    with pytest.raises(ValueError, match="resolved configuration changed"):
        _completed_cell("E10-analytic", "analytic_g2_t05", [item])
