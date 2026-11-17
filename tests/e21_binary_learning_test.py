"""Train-only binary E21 interventions preserve the selected source scoring rule."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from exact.impl.models.selector.llm_learning import (
    fit_llm_artifacts,
    validate_learning_binding,
)


class BinaryTeacher:
    def __init__(self, *, student="off", training="gold_teacher", gate="learned"):
        self.threshold, self.beta, self.request_seed = 0.7, 0.8, 17
        self.hosted_decision_labels = ("Y", "N")
        self.hosted_decision_logit_bias = 10.0
        self._attached_dataset = SimpleNamespace(dataset_signature="report")
        self._llm_router = SimpleNamespace(
            routing=SimpleNamespace(decision_profile="judge", default_profile="judge"),
            profiles={
                "judge": SimpleNamespace(
                    backend="openrouter",
                    model="frozen",
                    revision="v1",
                    tokenizer="tok",
                    tokenizer_revision="v1",
                    api_base="fixture",
                    provider={"only": ["provider"]},
                )
            },
        )
        self.llm_experiment_config = {
            "decision": {
                "mode": "binary",
                "evidence": "structured_packet",
                "listwise_max_candidates": 5,
            },
            "gate": {"mode": gate},
            "fusion_weight": "beta_u",
            "constant_weight": 0.5,
            "distill": student,
            "student_training": training,
            "outcome_policy": "complete_sources",
        }
        self.calls = []
        self.fail_source = None

    def llm_binary_decision_probs(self, sources, targets, labels, target_labels, briefs, scores):
        self.calls.append((sources[0], list(targets)))
        if sources[0] == self.fail_source:
            raise RuntimeError("interrupted source")
        probabilities = {
            target: (0.05 if sources[0] == "overflow" else 0.1 if target == "a" else 0.95)
            for target in targets
        }
        return torch.tensor(list(probabilities.values())), [
            {
                "source": sources[0],
                "valid": True,
                "pair_probabilities": probabilities,
                "calls": [
                    {"target": target, "usage": {"total_tokens": 10}, "response_id": f"r-{target}"}
                    for target in targets
                ],
            }
        ]


def population():
    rows = []
    for source, uncertainty in (("correction", 1.0), ("harm", 1.0), ("fusion", 0.1)):
        for target, score in (("a", 0.9), ("b", 0.8)):
            rows.append({"Src": source, "Tgt": target, "S_base": score, "U": uncertainty})
    for target, score in zip("abcdef", (0.95, 0.9, 0.85, 0.8, 0.75, 0.71)):
        rows.append({"Src": "overflow", "Tgt": target, "S_base": score, "U": 1.0})
    frame = pd.DataFrame(rows)
    frame["q_lex"], frame["Q_struct"] = 0.8, 0.7
    frame["llm_evidence_packet"] = "observed facts"
    frame["src_label_text"], frame["tgt_label_text"] = frame.Src, frame.Tgt
    return frame


REFERENCE = {("correction", "b"), ("harm", "a"), ("fusion", "a"), ("overflow", "f")}
APPLICATION = {
    "source_ids": ["report-source"],
    "dataset_signature": "report",
    "negative_label_policy": "complete_reference",
}


def fit(model, path, *, frame=None, application=None):
    return fit_llm_artifacts(
        model,
        population() if frame is None else frame,
        REFERENCE,
        path,
        config=model.llm_experiment_config,
        application=APPLICATION if application is None else application,
    )


def test_compact_exemplar_rendering_identity_changes_the_frozen_training_artifact(
    tmp_path, monkeypatch
):
    from exact.impl.models.selector.llm_learning import EXEMPLAR_RENDERING

    model = BinaryTeacher(gate="source_top_fraction")
    model.llm_experiment_config["exemplars"] = "knn"
    first = fit(model, tmp_path)
    path = first["exemplar_artifact"]
    assert first["exemplars"]["exemplar_rendering"] == EXEMPLAR_RENDERING
    monkeypatch.setitem(EXEMPLAR_RENDERING, "version", "fixture-new-rendering")
    second = fit(model, tmp_path)
    assert second["exemplar_artifact"] != path
    with open(path) as stream:
        assert json.load(stream)["exemplar_rendering"]["version"] != EXEMPLAR_RENDERING["version"]
    assert not model.calls


def test_binary_counterfactual_uses_beta_uncertainty_threshold_and_full_source_pool(tmp_path):
    model = BinaryTeacher()
    result = fit(model, tmp_path)
    outcomes = {row["source"]: row for row in result["router"]["counterfactuals"]}
    assert outcomes["correction"]["outcome"] == "correction"
    assert outcomes["harm"]["outcome"] == "harm"
    # Raw binary argmax is wrong here; the selected beta*U fusion retains correct a.
    assert outcomes["fusion"]["intervention_choice"] == "a"
    assert outcomes["fusion"]["outcome"] == "no_change"
    # Unjudged overflow is still eligible; forcing listwise choice would lose f.
    assert outcomes["overflow"]["intervention_choice"] == "f"
    assert outcomes["overflow"]["outcome"] == "correction"
    assert dict(model.calls)["overflow"] == list("abcde")
    assert outcomes["correction"]["tokens"] == 20
    assert outcomes["correction"]["target"] == 50
    assert outcomes["harm"]["target"] == -50
    assert outcomes["overflow"]["tokens"] == 50
    assert result["router"]["counterfactual_scoring"]["threshold"] == 0.7
    validate_learning_binding(result["router"], model, ["report-source"])
    model.beta = 0.5
    with pytest.raises(ValueError, match="counterfactual scoring changed"):
        validate_learning_binding(result["router"], model, ["report-source"])
    model.beta = 0.8
    assert fit(model, tmp_path) == result
    assert len(model.calls) == 4  # All paid observations are reusable without new calls.
    model.threshold = 0.95
    changed = fit(model, tmp_path)
    assert len(model.calls) == 8
    assert changed["gate_artifact"] != result["gate_artifact"]
    assert all(row["intervention_choice"] is None for row in changed["router"]["counterfactuals"])


def test_binary_student_matched_gold_folds_and_uses_observed_pair_probabilities(tmp_path):
    gold = BinaryTeacher(student="student", training="gold_only", gate="source_top_fraction")
    teacher = BinaryTeacher(student="student", gate="source_top_fraction")
    gold_result = fit(gold, tmp_path / "gold")["student"]
    distill = fit(teacher, tmp_path / "distilled")["student"]
    assert not gold.calls and len(teacher.calls) == 4
    assert gold_result["folds"] == distill["folds"]
    assert gold_result["feature_schema"] == distill["feature_schema"]
    assert len(gold_result["model"]["weights"]) == len(distill["model"]["weights"])
    gold_targets = {row["index"]: row["target"] for row in gold_result["oof_predictions"]}
    targets = {row["index"]: row["target"] for row in distill["oof_predictions"]}
    assert gold_targets[0] == 0 and targets[0] == 0.05
    assert gold_targets[1] == 1 and targets[1] == 0.975
    assert targets[11] == gold_targets[11] == 1  # Beyond the frozen top-five window.
    validate_learning_binding(distill, teacher, ["report-source"])
    with pytest.raises(ValueError, match="overlaps"):
        validate_learning_binding(distill, teacher, ["harm"])
    teacher.hosted_decision_labels = ("A", "B")
    with pytest.raises(ValueError, match="identity changed"):
        validate_learning_binding(distill, teacher, ["report-source"])


def test_binary_teacher_resumes_finished_source_shards_after_interruption(tmp_path):
    model = BinaryTeacher()
    # Sources are deterministically shuffled; stop after the first completed shard.
    import random

    sources = sorted(population().Src.unique())
    random.Random(model.request_seed).shuffle(sources)
    model.fail_source = sources[1]
    with pytest.raises(RuntimeError, match="interrupted"):
        fit(model, tmp_path)
    assert len(model.calls) == 2
    model.fail_source = None
    fit(model, tmp_path)
    assert [source for source, _ in model.calls].count(sources[0]) == 1
    assert len(model.calls) == 5


def test_binary_teacher_refuses_tampered_source_checkpoint(tmp_path):
    import json

    model = BinaryTeacher(student="student", gate="source_top_fraction")
    fit(model, tmp_path)
    path = next(tmp_path.rglob("teacher.json"))
    payload = json.loads(path.read_text())
    payload["records"].pop()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frozen source population"):
        fit(model, tmp_path)
    assert len(model.calls) == 4


def test_binary_training_safety_rejects_unknown_labels_and_overlap_before_calls(tmp_path):
    for application, message in (
        ({**APPLICATION, "negative_label_policy": "positive_unlabelled"}, "verified complete"),
        ({**APPLICATION, "source_ids": ["harm"]}, "overlap"),
    ):
        model = BinaryTeacher()
        with pytest.raises(ValueError, match=message):
            fit(model, tmp_path, application=application)
        assert not model.calls
    frame = population()
    frame["confirmed_label"] = [int((row.Src, row.Tgt) in REFERENCE) for row in frame.itertuples()]
    frame.loc[0, "confirmed_label"] = float("nan")
    model = BinaryTeacher()
    with pytest.raises(ValueError, match="every candidate"):
        fit(
            model,
            tmp_path,
            frame=frame,
            application={**APPLICATION, "negative_label_policy": "confirmed_negatives"},
        )
    assert not model.calls


@pytest.mark.parametrize("fault", ["probability", "missing_pair", "cost"])
def test_binary_teacher_rejects_unusable_observations_before_freezing(tmp_path, fault):
    model = BinaryTeacher()
    original = model.llm_binary_decision_probs

    def broken(*args):
        probabilities, records = original(*args)
        records = deepcopy(records)
        if fault == "probability":
            records[0]["pair_probabilities"][args[1][0]] = float("nan")
        elif fault == "missing_pair":
            records[0]["pair_probabilities"].pop(args[1][0])
        else:
            records[0]["calls"][0]["usage"]["total_tokens"] = 0
        return probabilities, records

    model.llm_binary_decision_probs = broken
    with pytest.raises(ValueError, match="Binary teacher"):
        fit(model, tmp_path)
    assert not list(tmp_path.rglob("teacher.json"))
    assert not list(tmp_path.rglob("router.json"))
