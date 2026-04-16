import importlib.util
import json
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "exact" / "impl" / "trainer" / "semantic_runner.py"
_SPEC = importlib.util.spec_from_file_location("semantic_runner_module", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
SemanticAlignmentRunner = _MODULE.SemanticAlignmentRunner


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
        raise AssertionError("runner should use runtime_fingerprint_payload for compatibility checks")


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
    runner._annotate_final_prediction_records([_Pred("s1", "t1")], threshold=0.7, local_alignment=False)
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
                                "llm_router": {"routing": {"decision_profile": "local_llm_default"}},
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
