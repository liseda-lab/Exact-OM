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


def judge_fixture(tmp_path, bound, *, evidence="scored_packet", mode="listwise", gain=0.01):
    base, suite = bound
    judge = {
        "enabled": True,
        "decision": {
            "mode": mode,
            "evidence": evidence,
            "brief_max_tokens": 256,
            "max_evidence_packets": 2,
        },
        "gate": {"mode": "source_top_fraction", "quantile_fraction": 1.0},
        "fusion_weight": "source_first",
    }
    manifests = [
        complete(
            tmp_path,
            "E07",
            "winner",
            harness.deep_merge(base, {"llm": {"experiment": judge}}),
            score=0.6 + gain,
        ),
        complete(
            tmp_path,
            "E25",
            "decision_off",
            harness.deep_merge(
                base, {"llm": {"experiment": {"enabled": True, "gate": {"mode": "off"}}}}
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


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"mode": "binary"}, "inapplicable"),
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
