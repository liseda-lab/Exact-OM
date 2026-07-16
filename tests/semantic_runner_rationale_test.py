import importlib.util
import json
from pathlib import Path

import pandas as pd

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "exact" / "impl" / "trainer" / "semantic_runner.py"
)
_SPEC = importlib.util.spec_from_file_location("semantic_runner_module", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
SemanticAlignmentRunner = _MODULE.SemanticAlignmentRunner

_SCORER_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "exact" / "impl" / "models" / "semantic_scorer.py"
)
_SCORER_SPEC = importlib.util.spec_from_file_location("semantic_scorer_module", _SCORER_MODULE_PATH)
_SCORER_MODULE = importlib.util.module_from_spec(_SCORER_SPEC)
assert _SCORER_SPEC is not None and _SCORER_SPEC.loader is not None
_SCORER_SPEC.loader.exec_module(_SCORER_MODULE)
SemanticScorer = _SCORER_MODULE.SemanticScorer


class _Pred:
    def __init__(self, head: str, tail: str):
        self.head = head
        self.tail = tail


class _DummyModel:
    def __init__(self):
        self._last_rationale_backend_meta = {"backend": "openrouter", "model": "openai/gpt-4o-mini"}
        self.generate_llm_rationales = True

    def generate_final_rationales_for_records(self, records, progress_callback=None):
        return [f"rationale:{rec['prediction']['rationale_decision_label']}" for rec in records]


class _FingerprintModel:
    def __init__(self, generate_llm_rationales):
        self.generate_llm_rationales = generate_llm_rationales

    def runtime_fingerprint_payload(self, generate_llm_rationales_override=None):
        value = self.generate_llm_rationales
        if generate_llm_rationales_override is not None:
            value = bool(generate_llm_rationales_override)
        return {
            "name": "dummy",
            "generate_llm_rationales": value,
            "other_setting": 123,
        }

    def runtime_fingerprint(self):
        raise AssertionError(
            "runner should use runtime_fingerprint_payload for compatibility checks"
        )


def test_global_annotation_uses_saved_alignment_membership():
    runner = object.__new__(SemanticAlignmentRunner)
    runner._results_json = [
        {
            "src_iri": "s1",
            "tgt_iri": "t1",
            "confidences": {"S_final": 0.95},
            "prediction": {},
        },
        {
            "src_iri": "s1",
            "tgt_iri": "t2",
            "confidences": {"S_final": 0.90},
            "prediction": {},
        },
    ]
    runner._annotate_final_prediction_records(
        [_Pred("s1", "t1")], threshold=0.7, local_alignment=False
    )
    assert runner.results_json[0]["prediction"]["saved_alignment_member"] is True
    assert runner.results_json[0]["prediction"]["rationale_decision_label"] == "Match"
    assert runner.results_json[1]["prediction"]["saved_alignment_member"] is False
    assert runner.results_json[1]["prediction"]["rationale_decision_label"] == "No match"


def test_local_annotation_uses_threshold_polarity():
    runner = object.__new__(SemanticAlignmentRunner)
    runner._results_json = [
        {
            "src_iri": "s1",
            "tgt_iri": "t1",
            "confidences": {"S_final": 0.75},
            "prediction": {},
        },
        {
            "src_iri": "s1",
            "tgt_iri": "t2",
            "confidences": {"S_final": 0.20},
            "prediction": {},
        },
    ]
    runner._annotate_final_prediction_records([], threshold=0.7, local_alignment=True)
    assert runner.results_json[0]["prediction"]["rationale_decision_label"] == "Match"
    assert runner.results_json[1]["prediction"]["rationale_decision_label"] == "No match"


def test_generate_final_rationales_updates_records():
    runner = object.__new__(SemanticAlignmentRunner)
    runner._results_json = [
        {
            "prediction": {"rationale_decision_label": "Match", "llm_rationale": ""},
            "backend_usage": {
                "summary": {"model": "openai/gpt-4o-mini"},
                "decision": {"model": "Qwen/Qwen2.5-7B-Instruct"},
            },
            "models": {},
        }
    ]
    runner._model = _DummyModel()
    runner._generate_final_rationales()
    assert runner.results_json[0]["prediction"]["llm_rationale"] == "rationale:Match"
    assert runner.results_json[0]["backend_usage"]["rationale"]["backend"] == "openrouter"
    assert runner.results_json[0]["models"]["llm_summary_model"] == "openai/gpt-4o-mini"
    assert runner.results_json[0]["models"]["llm_decision_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert runner.results_json[0]["models"]["llm_rationale_model"] == "openai/gpt-4o-mini"
    assert runner.results_json[0]["models"]["llm_model"] == "multiple"


def test_compact_candidate_records_support_final_rationales_without_full_audit():
    runner = object.__new__(SemanticAlignmentRunner)
    runner._results_json = []
    runner._model = _DummyModel()
    runner.log = lambda *args, **kwargs: None
    candidate_df = pd.DataFrame(
        [
            {
                "Src": "s1",
                "Tgt": "t1",
                "S_final": 0.91,
                "src_label_text": "source label",
                "tgt_label_text": "target label",
                "llm_pair_brief": "pair evidence",
                "threshold_positive": True,
                "saved_alignment_member": True,
                "rationale_positive": True,
                "rationale_decision_label": "Match",
            }
        ]
    )

    created = runner._ensure_compact_rationale_records(candidate_df)
    runner._generate_final_rationales()

    assert created is True
    assert runner.results_json[0]["selected_labels"]["source"] == "source label"
    assert runner.results_json[0]["prediction"]["llm_rationale"] == "rationale:Match"


def test_selector_metadata_is_available_to_final_rationale_prompt():
    scorer = object.__new__(SemanticScorer)
    record = {
        "confidences": {
            "S_pair_final": 0.86,
            "S_select": 0.0,
            "selection_utility": 0.62,
            "selection_no_match_prob": 0.71,
            "selection_margin": 0.09,
            "selection_entropy": 0.88,
            "selection_evidence_support": 0.83,
            "selection_distinctive": 0.12,
        },
        "prediction": {
            "selector_abstained": True,
            "selector_llm_used": False,
            "selector_reason": "no_match",
            "threshold_positive": False,
            "saved_alignment_member": False,
        },
    }

    context = scorer._final_alignment_context_for_rationale(record)
    prompt = scorer._rationale_prompt(
        "source label",
        "target label",
        "source summary",
        "target summary",
        "No match",
        context,
    )

    assert "Pairwise score before selector: 0.860" in context
    assert "NO_MATCH abstention risk: 0.710" in context
    assert "Selector evidence agreement: 0.830" in context
    assert "Selector abstained: yes" in context
    assert "Kept in saved alignment after cardinality filtering: no" in context
    assert "Final alignment context" in prompt["user"]


def test_checkpoint_with_mismatched_fingerprint_is_ignored(tmp_path):
    runner = object.__new__(SemanticAlignmentRunner)
    runner._dataset = type("DatasetStub", (), {"dataset_signature": "dataset-a"})()
    runner._checkpoint_fingerprint_payload = {
        "dataset_signature": "dataset-a",
        "models": [{"class": "CurrentModel", "fingerprint": "current-fingerprint"}],
    }
    runner._checkpoint_fingerprint = "current-fingerprint"
    runner.log = lambda *args, **kwargs: None
    checkpoint = tmp_path / "inference.json"
    checkpoint.write_text(
        json.dumps(
            {
                "kind": "inference",
                "dataset_signature": "dataset-a",
                "checkpoint_fingerprint": "old-fingerprint",
                "processed_examples": 10,
                "mappings": [{"src": "s", "tgt": "t", "score": 0.9}],
                "results_json": [{"src_iri": "s", "tgt_iri": "t"}],
            }
        ),
        encoding="utf-8",
    )
    mappings, results_json, processed_examples = runner._load_checkpoint_state(
        checkpoint,
        type("Kind", (), {"name": "inference"})(),
    )
    assert mappings == []
    assert results_json == []
    assert processed_examples == 0


def test_checkpoint_mismatch_logs_payload_details_when_available(tmp_path):
    runner = object.__new__(SemanticAlignmentRunner)
    runner._dataset = type("DatasetStub", (), {"dataset_signature": "dataset-a"})()
    runner._checkpoint_fingerprint_payload = {
        "dataset_signature": "dataset-a",
        "models": [
            {
                "class": "PairAdaptiveSemanticScorer",
                "fingerprint": "current-fingerprint",
                "payload": {
                    "generate_llm_rationales": False,
                    "llm_router": {"routing": {"decision_profile": "openrouter_gpt4o_mini"}},
                },
            }
        ],
    }
    runner._checkpoint_fingerprint = "current-fingerprint"
    messages = []
    runner.log = lambda message, level="info": messages.append((level, message))
    checkpoint = tmp_path / "inference.json"
    checkpoint.write_text(
        json.dumps(
            {
                "kind": "inference",
                "dataset_signature": "dataset-a",
                "checkpoint_fingerprint": "old-fingerprint",
                "checkpoint_fingerprint_payload": {
                    "dataset_signature": "dataset-a",
                    "models": [
                        {
                            "class": "SemanticScorer",
                            "fingerprint": "old-fingerprint",
                            "payload": {
                                "generate_llm_rationales": True,
                                "llm_router": {
                                    "routing": {"decision_profile": "local_llm_default"}
                                },
                            },
                        }
                    ],
                },
                "processed_examples": 10,
                "mappings": [{"src": "s", "tgt": "t", "score": 0.9}],
                "results_json": [{"src_iri": "s", "tgt_iri": "t"}],
            }
        ),
        encoding="utf-8",
    )

    mappings, results_json, processed_examples = runner._load_checkpoint_state(
        checkpoint,
        type("Kind", (), {"name": "inference"})(),
    )

    assert mappings == []
    assert results_json == []
    assert processed_examples == 0
    assert len(messages) == 1
    logged = messages[0][1]
    assert "Mismatch details:" in logged
    assert "models[0].class" in logged
    assert "SemanticScorer" in logged
    assert "PairAdaptiveSemanticScorer" in logged
    assert "models[0].payload.generate_llm_rationales" in logged


def test_checkpoint_with_only_rationale_toggle_change_can_resume_when_enabled(tmp_path):
    runner = object.__new__(SemanticAlignmentRunner)
    runner._dataset = type("DatasetStub", (), {"dataset_signature": "dataset-a"})()
    runner._models = [_FingerprintModel(generate_llm_rationales=False)]
    runner._model = runner._models[0]
    runner._checkpoint_fingerprint = runner._build_checkpoint_fingerprint()
    runner.log = lambda *args, **kwargs: None

    checkpoint = tmp_path / "inference.json"
    checkpoint.write_text(
        json.dumps(
            {
                "kind": "inference",
                "dataset_signature": "dataset-a",
                "checkpoint_fingerprint": runner._build_checkpoint_fingerprint(
                    generate_llm_rationales_override=True
                ),
                "processed_examples": 10,
                "mappings": [{"src": "s", "tgt": "t", "score": 0.9}],
                "results_json": [{"src_iri": "s", "tgt_iri": "t"}],
            }
        ),
        encoding="utf-8",
    )

    mappings, results_json, processed_examples = runner._load_checkpoint_state(
        checkpoint,
        type("Kind", (), {"name": "inference"})(),
        allow_rationale_toggle_checkpoint_resume=True,
    )
    assert mappings == [("s", "t", 0.9)]
    assert results_json == [{"src_iri": "s", "tgt_iri": "t"}]
    assert processed_examples == 10


def test_checkpoint_fingerprint_ignores_post_inference_models():
    runner = object.__new__(SemanticAlignmentRunner)
    runner._dataset = type("DatasetStub", (), {"dataset_signature": "dataset-a"})()
    primary = _FingerprintModel(generate_llm_rationales=False)
    extra = object()
    runner._models = [primary, extra]
    runner._model = primary

    payload = runner._build_checkpoint_fingerprint_payload()

    assert len(payload["models"]) == 1
    assert payload["models"][0]["class"] == "_FingerprintModel"


def test_additional_model_checkpoint_resume_can_be_disabled(tmp_path):
    runner = object.__new__(SemanticAlignmentRunner)
    runner._output_dir = tmp_path
    runner._dataset = type("DatasetStub", (), {"dataset_signature": "dataset-a"})()
    runner._model = _FingerprintModel(generate_llm_rationales=False)
    runner._models = [runner._model]
    runner._postprocess_checkpoints_enabled = True
    runner._additional_model_checkpoint_resume_enabled = False
    runner._additional_model_checkpoint_skip_logged = False
    logs = []
    runner.log = lambda msg, level="info", traceback=False: logs.append((level, msg))

    kind = type("Kind", (), {"name": "inference"})()
    path = runner._stage_checkpoint_path(kind, "additional_models", False, 0.7, 1)
    path.write_text(
        json.dumps(
            {
                "stage": "additional_models",
                "complete": True,
                "fingerprint_payload": runner._postprocess_fingerprint_payload(kind, False, 0.7, 1),
                "candidate_records": [{"Src": "s", "Tgt": "t", "S_final": 0.9}],
            }
        ),
        encoding="utf-8",
    )

    restored = runner._load_additional_models_checkpoint(kind, False, 0.7, 1)

    assert restored is None
    assert any("resume_additional_model_checkpoints=False" in message for _, message in logs)


def test_generate_final_rationales_preserves_existing_values_when_disabled():
    runner = object.__new__(SemanticAlignmentRunner)
    runner._results_json = [
        {
            "prediction": {"rationale_decision_label": "Match", "llm_rationale": "keep me"},
            "backend_usage": {},
            "models": {},
        },
        {
            "prediction": {"rationale_decision_label": "No match", "llm_rationale": ""},
            "backend_usage": {},
            "models": {},
        },
    ]
    runner._model = _DummyModel()
    runner._model.generate_llm_rationales = False

    runner._generate_final_rationales()

    assert runner.results_json[0]["prediction"]["llm_rationale"] == "keep me"
    assert runner.results_json[1]["prediction"]["llm_rationale"] == ""
