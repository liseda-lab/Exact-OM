import json
from pathlib import Path

import pandas as pd
import pytest

from exact.core.entities.configs.config import ConfigModel
from exact.experiments import harness
from exact.experiments.label_policy import (
    BUDGET_ARMS,
    materialize_followup,
    prepare_count_policy_followup,
    prepare_training_units,
)
from exact.experiments.paper_metrics import SourceConfusion, SourceEvaluation
from exact.experiments.schema import ExperimentConfig
from exact.impl.models.selector.label_budget import select_label_budget
from exact.utils.fitted_artifacts import freeze_json
from exact.utils.provenance import dataset_signature_for_paths


def units(path, budget, *, prefix="train", report="donor-report", signature="donor"):
    frame = pd.DataFrame(
        [(f"{prefix}-{i}", f"t-{i}-{j}", int(j == 0)) for i in range(420) for j in range(2)],
        columns=["Src", "Tgt", "confirmed_label"],
    )
    refs = {(row.Src, row.Tgt) for row in frame.itertuples() if row.confirmed_label}
    _, payload = select_label_budget(frame, refs, budget=budget, seed=17)
    payload["binding"] = {
        "dataset_signature": signature,
        "source_ids": [report],
        "entity_kinds": ["class"],
        "negative_label_policy": "confirmed_negatives",
    }
    freeze_json(path, payload)
    return payload


def evidence(path, *, supported=True):
    rows = []
    for arm, low in zip(
        BUDGET_ARMS, [-0.01, 0.005 if supported else -0.01, 0.006 if supported else -0.01]
    ):
        rows.append(
            {
                "experiment_id": "E22",
                "baseline_arm": "label_free",
                "candidate_arm": arm,
                "endpoint_scope": "overall",
                "metric": "task_macro_global_F1",
                "status": "complete",
                "delta": 0.01,
                "ci_low": low,
                "ci_high": 0.02,
                "n_units": 50,
            }
        )
    freeze_json(path, {"schema_version": 1, "resampling_unit": "source_entity", "rows": rows})


def curve(tmp_path, *, supported=True):
    donor = {arm: tmp_path / (arm + ".json") for arm in BUDGET_ARMS}
    for arm, budget in zip(BUDGET_ARMS, [25, 100, 400]):
        units(donor[arm], budget)
    evaluation = tmp_path / "evaluation.json"
    units(evaluation, 100, prefix="independent", report="recipient-report", signature="recipient")
    bootstrap = tmp_path / "bootstrap.json"
    evidence(bootstrap, supported=supported)
    return donor, evaluation, bootstrap


@pytest.mark.parametrize("supported", [True, False])
def test_policy_resolves_actual_joint_counts_and_preserves_inconclusive_fallback(
    tmp_path, supported
):
    donor, evaluation, bootstrap = curve(tmp_path, supported=supported)
    result = prepare_count_policy_followup(bootstrap, donor, evaluation, tmp_path / "policy")
    base = ConfigModel.load_config("exact/default_config.yaml").model_dump(mode="python")
    base["selector"]["rerank"]["artifact"] = "must-not-reuse.json"
    modes = {}
    for arm in result["arms"]:
        config = ConfigModel.from_mapping(harness.deep_merge(base, arm["overlay"]))
        modes[arm["id"]] = [
            config.supervision.resolve_component(
                component,
                training_available=True,
                profile_binding={"dataset_signature": "recipient"},
            )[0]
            for component in ["rerank", "accept"]
        ]
        assert config.selector.rerank.artifact is None
        assert arm["supervision_label"] == (
            "in_pair_supervised" if "supervised" in modes[arm["id"]] else "target_label_free"
        )
        with pytest.raises(ValueError, match="runtime binding"):
            config.supervision.resolve_component(
                "rerank",
                training_available=True,
                profile_binding={"dataset_signature": "different"},
            )
    assert modes["fixed_count"] == ["supervised"] * 2
    assert modes["fitted_count"] == ["supervised" if supported else "label_free"] * 2
    assert result["ready_to_claim_benefit"] is False
    assert (
        prepare_count_policy_followup(bootstrap, donor, evaluation, tmp_path / "policy") == result
    )
    changed = json.loads(bootstrap.read_text())
    changed["rows"][0]["delta"] = 0.011
    bootstrap.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identity conflict"):
        prepare_count_policy_followup(bootstrap, donor, evaluation, tmp_path / "policy")


def test_policy_rejects_population_leakage_and_unequal_joint_counts(tmp_path):
    donor, evaluation, bootstrap = curve(tmp_path)
    payload = json.loads(evaluation.read_text())
    payload["binding"]["source_ids"] = ["donor-report"]
    evaluation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="independent"):
        prepare_count_policy_followup(bootstrap, donor, evaluation, tmp_path / "p")
    payload["binding"]["source_ids"] = ["recipient-report"]
    payload["component_units"]["accept"] -= 1
    evaluation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="equal effective"):
        prepare_count_policy_followup(bootstrap, donor, evaluation, tmp_path / "p")


def test_train_count_preparation_uses_confirmed_labels_and_no_reporting_labels(tmp_path):
    candidates = tmp_path / "train.tsv"
    candidates.write_text("Src\tTgt\tconfirmed_label\na\tx\t1\na\ty\t0\nb\tz\t\n")
    reference = tmp_path / "reference.tsv"
    reference.write_text("Src\tTgt\n")  # Explicit positive overrides absent repaired reference.
    binding = {
        "dataset_signature": "pair",
        "source_ids": ["report"],
        "negative_label_policy": "confirmed_negatives",
        "entity_kinds": ["class"],
    }
    result = prepare_training_units(candidates, reference, tmp_path / "units.json", binding=binding)
    assert result["selected_sources"] == ["a"]
    assert result["pair_rows"] == 2
    assert result["component_units"]["accept"] == 1
    with pytest.raises(ValueError, match="confirmed negatives|verified complete"):
        prepare_training_units(
            candidates,
            reference,
            tmp_path / "unsafe.json",
            binding={**binding, "negative_label_policy": "unknown"},
        )


def declaration(tmp_path, identifier, arms, overlay, settings=None):
    task = {
        "id": "development",
        "split_role": "development",
        "reference_role": "valid",
        "overlay": overlay,
    }
    config = ExperimentConfig.model_validate(
        {
            "schema_version": 2,
            "experiment_id": identifier,
            "title": "Synthetic count policy",
            "base_config": str(Path("exact/default_config.yaml").resolve()),
            "screen": {"tasks": [task], "seeds": [17]},
            "confirm": {
                "tasks": [
                    {**task, "id": "reporting", "split_role": "reporting", "reference_role": "test"}
                ],
                "seeds": [17, 29, 43],
            },
            "arms": arms,
            "selection": {
                "decisions": [
                    {
                        "id": "curve",
                        "baseline": arms[0]["id"],
                        "candidates": [arm["id"] for arm in arms[1:]],
                        "metric": "F1",
                    }
                ]
            },
            "design": {
                "primary_comparison": "bounded count policy",
                "primary_endpoint": "F1",
                "independent_unit": "source_group",
                "power_status": "descriptive",
                "assumptions": ["synthetic independent populations"],
            },
            "frozen_constants": {"label_policy_followup": settings} if settings else {},
        }
    )
    path = tmp_path / (identifier + ".json")
    freeze_json(path, config.model_dump(mode="json"))
    return harness.ExperimentSource(config, path)


def test_materialize_uses_completed_nested_units_and_real_paired_bootstrap(tmp_path, monkeypatch):
    source_path, target_path = tmp_path / "source.owl", tmp_path / "target.owl"
    source_path.write_text("synthetic source ontology")
    target_path.write_text("synthetic target ontology")
    train, reference, universe = (
        tmp_path / "train.tsv",
        tmp_path / "reference.tsv",
        tmp_path / "sources.txt",
    )
    frame = pd.DataFrame(
        [(f"eval-train-{i}", f"t{i}-{j}", int(j == 0)) for i in range(110) for j in range(2)],
        columns=["Src", "Tgt", "confirmed_label"],
    )
    frame.to_csv(train, sep="\t", index=False)
    frame[frame.confirmed_label == 1][["Src", "Tgt"]].to_csv(reference, sep="\t", index=False)
    universe.write_text("eval-report\n")
    overlay = {
        "data": {
            "source": str(source_path),
            "target": str(target_path),
            "train_candidates": str(train),
            "refs": {"train": str(reference)},
            "source_universe": str(universe),
        },
        "supervision": {"negative_label_policy": "confirmed_negatives"},
    }
    producer = declaration(
        tmp_path,
        "E22",
        [{"id": "label_free", "role": "baseline"}]
        + [{"id": arm, "role": "candidate"} for arm in BUDGET_ARMS],
        overlay,
    )
    followup = declaration(
        tmp_path,
        "E22-policy",
        [{"id": "fixed_count", "role": "baseline"}, {"id": "fitted_count", "role": "candidate"}],
        overlay,
        {"producer": "E22", "budget": 100, "fixed_minimum": 100},
    )
    suite = harness.LoadedSuite(
        "fixture", "R_0", (producer, followup), None, "suite", None, None, {}
    )
    stage_root = tmp_path / "fixture" / "screen"
    manifests, records = [], []
    for arm in ["label_free", *BUDGET_ARMS]:
        output = stage_root / "runs" / "E22" / arm / "development" / "seed-17"
        output.mkdir(parents=True)
        if arm != "label_free":
            units(
                output / "fitting" / "scorer" / "labels-hash" / "training_units.json",
                int(arm.split("_")[1]),
            )
        manifests.append(
            {
                "experiment_id": "E22",
                "stage": "screen",
                "arm_id": arm,
                "task_id": "development",
                "seed": 17,
                "status": "complete",
                "resolved_config_hash": arm,
                "fingerprint_payload": {"output_dir": str(output)},
            }
        )
        records.append(
            {
                **manifests[-1],
                "output_dir": str(output),
                "config_hash": arm,
                "candidate_pool_fingerprint": "same-pool",
                "reference_completeness": "complete",
                "metrics": {"F1": 0.5},
            }
        )
    freeze_json(stage_root / "metrics.json", {"schema_version": 1, "rows": records})

    # Completed evaluator counts are the only mocked boundary; resampling and
    # interval calculation use the real source-group bootstrap implementation.
    def evaluation(record, cache):
        good = record["arm_id"] in {"budget_100", "budget_400"}
        return SourceEvaluation(
            "ref",
            record["arm_id"],
            {
                f"donor-{i}": SourceConfusion(tp=int(good), fp=int(not good), fn=int(not good))
                for i in range(8)
            },
        )

    monkeypatch.setattr(harness, "_overall_evaluation", evaluation)
    bound = materialize_followup(followup, suite, manifests)
    assert bound.path.is_file() and bound.path != followup.path
    policy = json.loads((bound.directory / "count_policy.json").read_text())
    assert policy["components"]["rerank"]["minimum_groups"] == 100
    assert policy["binding"]["dataset_signature"] == dataset_signature_for_paths(
        source_path, target_path
    )
    for arm in bound.config.arms:
        config = ConfigModel.from_mapping(
            harness.deep_merge(
                ConfigModel.load_config("exact/default_config.yaml").model_dump(mode="python"),
                arm.overlay,
            )
        )
        assert (
            config.supervision.resolve_component(
                "accept",
                training_available=True,
                profile_binding={"dataset_signature": policy["binding"]["dataset_signature"]},
            )[0]
            == "supervised"
        )
    assert materialize_followup(followup, suite, manifests).raw_hash() == bound.raw_hash()
    manifests[0]["status"] = "interrupted"
    with pytest.raises(ValueError, match="all completed"):
        materialize_followup(followup, suite, manifests)
