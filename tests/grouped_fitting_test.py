import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from exact.impl.models.selector import CandidateSetSelector
from exact.impl.models.selector.listwise_llm import build_listwise_call_plan
from exact.impl.models.semantic_llm import SemanticLLMMixin
from exact.impl.trainer.fitting import source_batches


def selector(mode="current_listwise", model="current_linear"):
    return CandidateSetSelector(
        enabled=True,
        global_only=False,
        strategy="calibrated_rank_accept",
        calibration={"max_epochs": 12, "min_positive_sources": 1, "validation_folds": 3},
        experiment_config={"enabled": True, "rerank": {"mode": mode, "model": model}},
        request_seed=17,
    )


def training_rows():
    return pd.DataFrame(
        [
            {
                "Src": f"s{i}",
                "Tgt": f"t{i}-{j}",
                "S_final": score,
                "s_label": score,
                "S_struct": score,
                "s_hier": score,
                "s_sim": score,
                "s_diff": score,
                "s_attr": score,
            }
            for i in range(9)
            for j, score in enumerate((0.9, 0.2))
        ]
    )


@pytest.mark.parametrize(
    "mode,model",
    [
        ("current_listwise", "current_linear"),
        ("pairwise", "current_linear"),
        ("current_listwise", "channel_gating"),
        ("current_listwise", "additive_gam"),
    ],
)
def test_train_only_artifact_replay_and_fold_recovery(tmp_path, mode, model):
    fitted = selector(mode, model)
    frame = training_rows()
    frame.loc[len(frame)] = {"Src": "unknown", "Tgt": "bad", "S_final": 99}
    reference = {(f"s{i}", f"t{i}-0") for i in range(9)}
    artifact = tmp_path / "head.json"
    payload = fitted.fit_training_artifact(
        frame,
        reference,
        artifact,
        application={
            "negative_label_policy": "complete_reference",
            "dataset_signature": "report-v1",
            "source_ids": ["report"],
        },
    )
    assert "unknown" not in payload["fit_provenance"]["training_sources"]
    for fold in payload["folds"]:
        assert not set(fold["train_sources"]) & set(fold["heldout_sources"])
    report = training_rows().iloc[:2].copy()
    report["Src"] = "report"
    report["gold"] = ["hidden", "hidden"]
    result = fitted(candidate_df=report, dataset=SimpleNamespace(dataset_signature="report-v1"))[
        "candidate_df"
    ]
    assert result.selection_winner.sum() == 1
    with pytest.raises(ValueError, match="overlap"):
        fitted(candidate_df=training_rows(), dataset=SimpleNamespace(dataset_signature="report-v1"))
    with pytest.raises(ValueError, match="dataset mismatch"):
        fitted(candidate_df=report, dataset=SimpleNamespace(dataset_signature="other"))
    # Delete only the final artifact: frozen fold states remain usable after interruption.
    artifact.unlink()
    recovered = selector(mode, model)
    recovered._fit_rank_model_original = recovered._fit_rank_model
    calls = []

    def fit(*args, **kwargs):
        calls.append(1)
        return recovered._fit_rank_model_original(*args, **kwargs)

    recovered._fit_rank_model = fit
    rebuilt = recovered.fit_training_artifact(
        frame,
        reference,
        artifact,
        application={
            "negative_label_policy": "complete_reference",
            "dataset_signature": "report-v1",
            "source_ids": ["report"],
        },
    )
    assert len(calls) == 1
    assert rebuilt["rank_model"] == payload["rank_model"]


def test_score_calibration_uses_disjoint_train_rows(tmp_path):
    fitted = selector()
    fitted.matching_calibration["mode"] = "platt"
    payload = fitted.fit_score_calibration_artifact(
        training_rows(),
        {(f"s{i}", f"t{i}-0") for i in range(9)},
        tmp_path / "platt.json",
        application={"negative_label_policy": "complete_reference", "source_ids": ["report"]},
    )
    assert len(payload["oof_predictions"]) == 18
    assert set(row["fold"] for row in payload["oof_predictions"]) == set(range(5))


def test_source_batches_never_split_a_group():
    frame = pd.DataFrame({"Src": ["a"] * 3 + ["b"] * 2 + ["c"]})
    assert list(source_batches(frame, 4)) == [[0, 1, 2], [3, 4, 5]]


class HostedFixture(SemanticLLMMixin):
    def __init__(self, payload):
        self.device = torch.device("cpu")
        self.request_seed = 17
        self.hosted_decision_logit_bias = 50
        self.llm_experiment_config = {
            "decision": {"mode": "listwise", "probability": "raw_joint", "permutations": 1}
        }
        self.calls = []

        def call(**kwargs):
            self.calls.append(kwargs)
            return payload

        backend = SimpleNamespace(backend="openrouter", profile_name="judge")
        self._llm_router = SimpleNamespace(
            resolve_task=lambda *a, **kw: backend,
            profiles={"judge": SimpleNamespace()},
            hosted=SimpleNamespace(chat_completion=call),
        )

    def _resolved_backend_metadata(self, backend):
        return {}

    def _get_hosted_decision_tokenizer(self, profile):
        return None

    def _candidate_token_ids_for_tokenizer(self, tokenizer, tokens):
        return [ord(tokens[0])]

    def _logsumexp(self, values):
        return torch.logsumexp(torch.tensor(values), 0).item()


def test_actual_grouped_call_handles_whitespace_and_missing_probabilities():
    payload = {
        "id": "response",
        "choices": [
            {
                "message": {"content": "A"},
                "logprobs": {
                    "content": [
                        {"token": "\n", "top_logprobs": []},
                        {
                            "token": "A",
                            "top_logprobs": [
                                {"token": key, "logprob": value}
                                for key, value in (("A", -0.1), (" B", -2), ("Z", -3))
                            ],
                        },
                    ]
                },
            }
        ],
    }
    hosted = HostedFixture(payload)
    values, used, records = hosted.llm_grouped_decision_probs(
        ["s", "s"],
        ["t1", "t2"],
        ["S", "S"],
        ["T1", "T2"],
        ["fact-1", "fact-2"],
        [0.9, 0.89],
        [True, False],
    )
    assert len(hosted.calls) == 1
    assert used.tolist() == [True, True]
    assert sum(records[0]["categorical"].values()) == pytest.approx(1)
    payload["choices"][0]["logprobs"]["content"][1]["top_logprobs"].pop()
    _, used, records = hosted.llm_grouped_decision_probs(
        ["s", "s"],
        ["t1", "t2"],
        ["S", "S"],
        ["T1", "T2"],
        ["fact-1", "fact-2"],
        [0.9, 0.89],
        [True, False],
    )
    assert not used.any()
    assert records[0]["error"] == "incomplete_categorical"


def test_primary_listwise_single_call_and_singleton():
    plan = build_listwise_call_plan(
        source_iri="s",
        source_label="S",
        source_summary="",
        candidates=[{"target_iri": "t", "score": 0.9, "brief": "Fact"}],
        mode="listwise",
        request_seed=17,
    )
    assert len(plan.calls) == 1
    assert "{A, Z}" in plan.calls[0].prompt["user"]


def test_fusion_fit_neutral_identity_and_grouped_export(tmp_path):
    from exact.impl.models.selector.fusion_fitting import (
        fit_fusion_artifact,
        fusion_scores,
    )

    frame = training_rows()
    channels = []
    for row in frame.itertuples():
        channels.append(
            {
                name: {
                    "score": row.S_final if name == "label" else 0.7,
                    "quality": 1.0,
                    "active": name != "strsim",
                }
                for name in ("label", "strsim", "hier__is_a", "sim_obj")
            }
        )
    frame["fusion_channels"] = channels
    names = sorted(channels[0]) + ["lex", "struct"]
    tensors = tuple(
        torch.tensor(
            [[row.get(name, {}).get(field, 0) for name in names] for row in channels],
            dtype=torch.float64,
        )
        for field in ("score", "quality", "active")
    )
    scores = fusion_scores(
        tensors,
        names,
        (torch.tensor(0.5), torch.tensor(2.0), torch.ones(len(names))),
        mode="analytic_fitted",
    )
    assert scores[0].item() == pytest.approx(0.86)
    frame["S_base"] = scores.tolist()
    payload = fit_fusion_artifact(
        frame,
        {(f"s{i}", f"t{i}-0") for i in range(9)},
        tmp_path / "fusion.json",
        mode="analytic_fitted",
        application={
            "negative_label_policy": "complete_reference",
            "source_ids": ["report"],
            "dataset_signature": "report-v1",
        },
        epochs=2,
    )
    assert 0.4 <= payload["parameters"]["tau"] <= 0.6
    assert 0.5 <= payload["parameters"]["gamma"] <= 3
    weights = list(payload["parameters"]["multipliers"].values())
    assert sum(weights) / len(weights) == pytest.approx(1)
    for fold in payload["fit_provenance"]["folds"]:
        assert not set(fold["train_sources"]) & set(fold["heldout_sources"])


class TinyDataset:
    dataset_signature = "fixture-pool-v1"
    default_kind = "inference"

    def __init__(self, frame):
        self._df = self.get_features(frame)

    @property
    def dataframe(self):
        return self._df

    def get_features(self, frame):
        frame = frame.copy()
        frame["SrcLabels"] = [[source] for source in frame.Src]
        frame["TgtLabels"] = [[target] for target in frame.Tgt]
        frame["inference"] = True
        return frame

    def _active_dataframe(self):
        return self._df

    def _invalidate_active_dataframe_cache(self):
        pass

    def __getitem__(self, index):
        row = self._df.iloc[index]
        return {
            "src_iri": row.Src,
            "tgt_iri": row.Tgt,
            "src_labels": row.SrcLabels,
            "tgt_labels": row.TgtLabels,
            "label": 999,
        }


class TinyScorer:
    use_llm = True

    def __init__(self):
        self.calls = []
        self.llm_experiment_config = {"gate": {"mode": "off"}}

    def runtime_fingerprint_payload(self):
        return {"model": "fixture-scorer-v1"}

    def forward(self, **kwargs):
        assert not self.use_llm
        assert "label" not in kwargs
        self.calls.append(kwargs)
        scores = torch.tensor(
            [0.9 if target.endswith("0") else 0.2 for target in kwargs["tgt_iris"]]
        )
        return {"S_final": scores, "S_base": scores, "s_label": scores, "U": 1 - scores}

    def _validate_gate_artifact_contract(self, mode):
        assert mode == "source_top_fraction"


def tiny_runner(tmp_path, dataset, model, head):
    from exact.impl.trainer.fitting import TrainingPoolMixin

    class Runner(TrainingPoolMixin):
        def _json_safe_value(self, value):
            return value

        def _build_checkpoint_fingerprint_payload(self):
            return {}

        def _hash_checkpoint_fingerprint_payload(self, value):
            return "fixture"

    runner = Runner()
    runner.output_dir = tmp_path
    runner.dataset = dataset
    runner.model = model
    runner.models = [model, head]
    return runner


def test_real_training_pool_pass_excludes_labels_and_recovers_scores(tmp_path):
    frame = training_rows()
    pool = tmp_path / "training.tsv"
    frame[["Src", "Tgt"]].to_csv(pool, sep="\t", index=False)
    reference = tmp_path / "reference.tsv"
    frame[frame.Tgt.str.endswith("0")][["Src", "Tgt"]].to_csv(reference, sep="\t", index=False)
    head = selector()
    head.training_reference_file_path = str(reference)
    scorer = TinyScorer()
    runner = tiny_runner(
        tmp_path, TinyDataset(pd.DataFrame({"Src": ["report"], "Tgt": ["report-0"]})), scorer, head
    )
    runner.supervision_config = {"negative_label_policy": "complete_reference"}
    runner.training_candidates_file_path = pool
    runner.fit_training_pool(batch_size=5)
    assert scorer.calls
    assert all(source != "report" for call in scorer.calls for source in call["src_iris"])
    assert head.rerank_config["artifact"]
    scorer.calls.clear()
    runner.fit_training_pool(batch_size=5)
    assert scorer.calls == []


def test_population_first_pass_freezes_budget_without_gold_or_llm(tmp_path):
    dataset = TinyDataset(pd.DataFrame({"Src": ["a", "a", "b"], "Tgt": ["t-0", "t-1", "u-1"]}))
    scorer = TinyScorer()
    runner = tiny_runner(tmp_path, dataset, scorer, selector())
    runner.fitting_gate_config = {
        "mode": "source_top_fraction",
        "quantile_fraction": 0.5,
        "artifact": None,
    }
    runner.source_universe = ["a", "b", "empty"]
    runner.prepare_population_gate(batch_size=2)
    artifact = json.loads(open(scorer.llm_experiment_config["gate"]["artifact"]).read())
    assert artifact["selected_count"] == 1
    assert artifact["selected_sources"] == ["b"]
    assert artifact["no_candidate_sources"] == ["empty"]
    assert scorer.use_llm is True


def test_bounded_evidence_acquisition_only_fetches_frozen_entities():
    from exact.impl.models.selector.evidence_acquisition import acquire_plan_evidence

    plan = build_listwise_call_plan(
        source_iri="s",
        source_label="S",
        source_summary="",
        candidates=[{"target_iri": "t", "score": 0.9, "brief": "initial fact"}],
        mode="listwise",
        request_seed=17,
    )
    fetched = []

    def features(iri, side, kind):
        fetched.append((iri, side))
        return {
            "attributes": [{"prop_iri": "definition", "value": "ontology fact", "weight": 0.5}],
            "object_triples": [],
        }

    dataset = SimpleNamespace(
        entity_kind_for=lambda *args, **kwargs: "class", get_entity_features=features
    )
    payload = {"id": "acquisition", "choices": [{"message": {"content": '["S", "A"]'}}]}
    model = HostedFixture(payload)
    model._attached_dataset = dataset
    updated, record = acquire_plan_evidence(model, plan, SimpleNamespace())
    assert fetched == [("s", "src"), ("t", "tgt")]
    assert updated.selected_candidate_ids == plan.selected_candidate_ids
    assert len(record["packets"]) == 2
    assert '"weight"' not in updated.calls[0].prompt["user"]
    payload["choices"][0]["message"]["content"] = '["S", "A", "unseen"]'
    fetched.clear()
    unchanged, record = acquire_plan_evidence(model, plan, SimpleNamespace())
    assert unchanged == plan
    assert fetched == []
    assert record["fallback"] == "original_evidence"


def test_source_first_changes_ranking_before_frozen_acceptance():
    head = selector()
    frame = training_rows().iloc[:2].copy()
    frame["S_pair_final"] = frame.S_final
    frame["llm_source_choice"] = "t0-1"
    features = head._rank_feature_rows(frame, {0: 0, 1: 0}, {0: 0, 1: 0})
    decisions = head._source_decisions(frame, {0: 0.9, 1: 0.2}, features, {}, set())
    assert decisions["s0"]["winner_pair"] == ("s0", "t0-1")
    frame["llm_source_choice"] = "__NONE__"
    decisions = head._source_decisions(frame, {0: 0.9, 1: 0.2}, features, {}, set())
    assert decisions["s0"]["displayed_none"] is True


def test_nested_label_budgets_and_reference_safe_active_selection():
    from exact.impl.models.selector.label_budget import select_label_budget

    frame = training_rows()
    reference = {(f"s{i}", f"t{i}-0") for i in range(9)}
    small_ref, small = select_label_budget(frame, reference, budget=3, seed=17)
    large_ref, large = select_label_budget(frame, reference, budget=6, seed=17)
    assert small_ref < large_ref
    assert small["selected_sources"] == large["selected_sources"][:3]
    assert small["nested_order_sha256"] == large["nested_order_sha256"]
    _, active = select_label_budget(
        frame.assign(reporting_gold="forbidden"),
        reference,
        budget=3,
        seed=17,
        selection="uncertainty",
    )
    assert active["effective_groups"] == 3
    assert active["pair_rows"] == 6


def test_frozen_count_policy_requires_supported_development_crossover(tmp_path):
    from exact.core.entities.configs.experimental import SupervisionConfig
    from exact.impl.models.selector.label_budget import fit_count_policy

    binding = {"dataset_signature": "fixture-v1"}
    policy = tmp_path / "policy.json"
    fit_count_policy(
        [
            {"role": "development", "effective_groups": 25, "gain_ci_low": -0.01},
            {"role": "development", "effective_groups": 100, "gain_ci_low": 0.02},
        ],
        policy,
        component="rerank",
        binding=binding,
        practical_effect=0.01,
    )
    units = tmp_path / "units.json"
    units.write_text(
        json.dumps(
            {
                "binding": binding,
                "component_units": {"rerank": 100},
                "component_definitions": {"rerank": "positive_candidate_source_groups"},
            }
        )
    )
    config = SupervisionConfig(
        auto_policy={"kind": "profile_rule", "artifact": policy},
        artifacts={"training_units": units},
    )
    assert (
        config.resolve_component("rerank", training_available=True, profile_binding=binding)[0]
        == "supervised"
    )
    with pytest.raises(ValueError, match="runtime binding mismatch"):
        config.resolve_component(
            "rerank", training_available=True, profile_binding={"dataset_signature": "wrong"}
        )
    with pytest.raises(ValueError, match="development"):
        fit_count_policy(
            [{"role": "final", "effective_groups": 25, "gain_ci_low": 0.05}],
            tmp_path / "bad.json",
            component="rerank",
            binding=binding,
        )


def test_incomplete_reference_does_not_create_negative_labels(tmp_path):
    from exact.impl.models.selector.fitting import safe_training_labels

    frame = pd.DataFrame(
        {
            "Src": ["s", "s", "s"],
            "Tgt": ["positive", "unknown", "negative"],
            "confirmed_label": [None, None, 0],
        }
    )
    reference = {("s", "positive")}
    with pytest.raises(ValueError, match="positive-unlabelled"):
        safe_training_labels(frame, reference, {"negative_label_policy": "unknown"})
    known, labels = safe_training_labels(
        frame, reference, {"negative_label_policy": "confirmed_negatives"}
    )
    assert list(known.Tgt) == ["positive", "negative"]
    assert labels == reference
    with pytest.raises(ValueError, match="confirmed_label"):
        safe_training_labels(
            frame.drop(columns="confirmed_label"),
            reference,
            {"negative_label_policy": "confirmed_negatives"},
        )
    with pytest.raises(ValueError, match="positive-unlabelled"):
        selector().fit_training_artifact(
            training_rows(),
            {(f"s{i}", f"t{i}-0") for i in range(9)},
            tmp_path / "unsafe.json",
            application={"source_ids": ["report"]},
        )
    assert not (tmp_path / "unsafe.json").exists()


class LearningFixture(HostedFixture):
    def __init__(self, *, student="off", teacher="gold_teacher", gate="analytic", exemplars="off"):
        from exact.core.entities.configs.experimental import LLMExperimentConfig

        payload = {
            "id": "response",
            "usage": {"total_tokens": 100},
            "choices": [
                {
                    "message": {"content": "A"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "A",
                                "top_logprobs": [
                                    {"token": key, "logprob": value}
                                    for key, value in (("A", -0.1), ("B", -2), ("Z", -3))
                                ],
                            }
                        ]
                    },
                }
            ],
        }
        super().__init__(payload)
        self.threshold = 0.5
        self.beta = 0.8
        self._attached_dataset = SimpleNamespace(dataset_signature="report")
        self.llm_experiment_config = LLMExperimentConfig.model_validate(
            {
                "enabled": True,
                "decision": {"mode": "listwise", "evidence": "structured_packet"},
                "gate": {"mode": gate},
                "exemplars": exemplars,
                "exemplar_count": 3 if exemplars == "knn" else 0,
                "distill": student,
                "student_training": teacher,
                "outcome_policy": "complete_sources",
                "fusion_weight": "source_first" if gate == "learned" else "beta_u",
            }
        ).model_dump(mode="python")
        self._llm_router.routing = SimpleNamespace(
            decision_profile="judge", default_profile="judge"
        )
        self._llm_router.profiles["judge"] = SimpleNamespace(
            backend="openrouter",
            model="frozen-judge",
            revision="revision1",
            tokenizer="tokenizer",
            tokenizer_revision="tokenizer1",
            api_base="fixture",
            provider={"only": ["provider"]},
        )


def learning_frame():
    frame = training_rows()
    frame["S_base"] = frame.S_final
    frame["U"] = 0.8
    frame["q_lex"] = 0.9
    frame["Q_struct"] = 0.8
    frame["llm_evidence_packet"] = '{"fact": "ontology fact"}'
    frame["src_label_text"] = frame.Src
    frame["tgt_label_text"] = frame.Tgt
    return frame


def test_gold_and_distilled_students_share_architecture_and_grouped_folds(tmp_path):
    from exact.impl.models.selector.llm_learning import (
        fit_llm_artifacts,
        validate_learning_binding,
    )

    frame = learning_frame()
    reference = {(f"s{i}", f"t{i}-0") for i in range(9)}
    application = {
        "source_ids": ["report-source"],
        "dataset_signature": "report",
        "negative_label_policy": "complete_reference",
    }
    gold = LearningFixture(student="student", teacher="gold_only")
    result = fit_llm_artifacts(
        gold,
        frame,
        reference,
        tmp_path / "gold",
        config=gold.llm_experiment_config,
        application=application,
    )
    assert not gold.calls
    assert len(result["student"]["oof_predictions"]) == len(frame)
    for fold in result["student"]["folds"]:
        assert not set(fold["training_sources"]) & set(fold["heldout_sources"])
    teacher = LearningFixture(student="student")
    distilled = fit_llm_artifacts(
        teacher,
        frame,
        reference,
        tmp_path / "teacher",
        config=teacher.llm_experiment_config,
        application=application,
    )
    assert len(teacher.calls) == 9
    assert len(distilled["student"]["teacher_sources"]) == 9
    assert result["student"]["feature_schema"] == distilled["student"]["feature_schema"]
    assert len(result["student"]["model"]["weights"]) == len(
        distilled["student"]["model"]["weights"]
    )
    replay = fit_llm_artifacts(
        teacher,
        frame,
        reference,
        tmp_path / "teacher",
        config=teacher.llm_experiment_config,
        application=application,
    )
    assert replay == distilled and len(teacher.calls) == 9
    validate_learning_binding(distilled["student"], teacher, ["report-source"])
    with pytest.raises(ValueError, match="overlaps"):
        validate_learning_binding(distilled["student"], teacher, ["s1"])
    teacher._llm_router.profiles["judge"].revision = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        validate_learning_binding(distilled["student"], teacher, ["report-source"])


def test_exemplars_are_train_only_bounded_and_bound_to_teacher(tmp_path):
    from exact.impl.models.selector.llm_learning import (
        exemplar_prompt,
        fit_llm_artifacts,
    )

    model = LearningFixture(exemplars="knn")
    result = fit_llm_artifacts(
        model,
        learning_frame(),
        {(f"s{i}", f"t{i}-0") for i in range(9)},
        tmp_path,
        config=model.llm_experiment_config,
        application={
            "dataset_signature": "report",
            "source_ids": ["report-source"],
            "negative_label_policy": "complete_reference",
        },
    )
    assert not model.calls
    model._exemplar_artifact = result["exemplars"]
    plan = build_listwise_call_plan(
        source_iri="report-source",
        source_label="report source",
        source_summary="fact source",
        candidates=[
            {"target_iri": "a", "brief": "fact A", "score": 0.9},
            {"target_iri": "b", "brief": "fact B", "score": 0.8},
        ],
        mode="listwise",
        request_seed=17,
    )
    enriched, sources = exemplar_prompt(model, plan)
    assert len(sources) == 3 and set(sources) <= {f"s{i}" for i in range(9)}
    assert "Training-only examples" in enriched.calls[0].prompt["user"]
    assert enriched.selected_candidate_ids == plan.selected_candidate_ids


def test_benefit_router_requires_complete_outcomes_and_actual_call_cost(tmp_path):
    from exact.impl.models.selector.llm_learning import fit_llm_artifacts

    model = LearningFixture(gate="learned")
    application = {
        "dataset_signature": "report",
        "source_ids": ["report-source"],
        "negative_label_policy": "complete_reference",
    }
    reference = {(f"s{i}", f"t{i}-0") for i in range(9)}
    config = dict(model.llm_experiment_config, outcome_policy="unknown")
    with pytest.raises(ValueError, match="complete/adjudicated"):
        fit_llm_artifacts(
            model, learning_frame(), reference, tmp_path, config=config, application=application
        )
    assert not model.calls
    result = fit_llm_artifacts(
        model,
        learning_frame(),
        reference,
        tmp_path,
        config=model.llm_experiment_config,
        application=application,
    )
    assert len(model.calls) == 9
    examples = result["router"]["counterfactuals"]
    assert all(row["tokens"] == 100 for row in examples)
    assert {row["outcome"] for row in examples} <= {"correction", "harm", "no_change"}
    assert all(
        row["target"] == {"correction": 10.0, "harm": -10.0, "no_change": 0.0}[row["outcome"]]
        for row in examples
    )
    assert result["router"]["target"] == "correction_minus_harm_per_1000_tokens"


def test_runnerup_acceptance_never_makes_alternative_positive_negative():
    fitted = selector()
    frame = training_rows().iloc[:2].copy()
    frame["S_pair_final"] = frame.S_final
    features = fitted._rank_feature_rows(frame, {}, {})
    utilities = {0: 0.8, 1: 0.7}
    decisions = fitted._source_decisions(
        frame, utilities, features, {}, {("s0", "t0-0"), ("s0", "t0-1")}
    )
    assert all("runnerup_negative" not in decision for decision in decisions.values())
    decisions = fitted._source_decisions(frame, utilities, features, {}, {("s0", "t0-0")})
    assert next(iter(decisions.values()))["runnerup_negative"]["label"] == 0


def test_benefit_router_confirmed_pool_labels_fail_before_calls_when_incomplete(tmp_path):
    from exact.impl.models.selector.llm_learning import fit_llm_artifacts

    reference = {(f"s{i}", f"t{i}-0") for i in range(9)}
    frame = learning_frame()
    frame["confirmed_label"] = [int((row.Src, row.Tgt) in reference) for row in frame.itertuples()]
    application = {
        "dataset_signature": "report",
        "source_ids": ["report-source"],
        "negative_label_policy": "confirmed_negatives",
    }
    model = LearningFixture(gate="learned")
    result = fit_llm_artifacts(
        model,
        frame,
        reference,
        tmp_path / "complete",
        config=model.llm_experiment_config,
        application=application,
    )
    assert len(model.calls) == 9
    assert result["router"]["outcome_scope"] == "frozen_fully_labeled_candidate_pool"
    model = LearningFixture(gate="learned")
    partial = frame.copy()
    partial.loc[1, "confirmed_label"] = float("nan")
    with pytest.raises(ValueError, match="every candidate"):
        fit_llm_artifacts(
            model,
            partial,
            reference,
            tmp_path / "partial",
            config=model.llm_experiment_config,
            application=application,
        )
    assert not model.calls
    # A prior shared fit pass may already have dropped unknown rows; its proof
    # must still prevent a partial original population from appearing complete.
    filtered = partial.drop(index=1)
    coverage = {**application, "fully_labeled_training_sources": [f"s{i}" for i in range(1, 9)]}
    with pytest.raises(ValueError, match="every candidate"):
        fit_llm_artifacts(
            model,
            filtered,
            reference,
            tmp_path / "filtered",
            config=model.llm_experiment_config,
            application=coverage,
        )
    assert not model.calls


def test_training_pool_cannot_use_frozen_reporting_source_with_empty_pool(tmp_path):
    pool = tmp_path / "training.tsv"
    pool.write_text("Src\tTgt\nempty-report\tt0\n")
    reference = tmp_path / "reference.tsv"
    reference.write_text("Src\tTgt\nempty-report\tt0\n")
    head = selector()
    head.training_reference_file_path = str(reference)
    dataset = TinyDataset(pd.DataFrame({"Src": ["report"], "Tgt": ["report-0"]}))
    dataset.eligible_source_iris = ["report", "empty-report"]
    scorer = TinyScorer()
    runner = tiny_runner(tmp_path, dataset, scorer, head)
    runner.supervision_config = {"negative_label_policy": "complete_reference"}
    runner.training_candidates_file_path = pool
    with pytest.raises(ValueError, match="overlaps reporting source groups"):
        runner.fit_training_pool(batch_size=5)
    assert scorer.calls == []
