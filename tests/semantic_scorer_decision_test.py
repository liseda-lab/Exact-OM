import importlib.util
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "exact" / "impl" / "models" / "semantic_scorer.py"
)
_SPEC = importlib.util.spec_from_file_location("semantic_scorer_module", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
SemanticScorer = _MODULE.SemanticScorer


class _DummyHosted:
    def chat_completion(
        self,
        profile,
        messages,
        max_tokens,
        temperature=None,
        top_p=None,
        stop=None,
        logprobs=None,
        top_logprobs=None,
        logit_bias=None,
        provider=None,
        seed=None,
        role="unspecified",
    ):
        user_text = messages[-1]["content"]
        if "Source entity: src" not in user_text:
            raise AssertionError(f"Unexpected user prompt: {user_text}")
        return {
            "provider": "unit-test-provider",
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "A",
                                "logprob": -0.2,
                                "top_logprobs": [
                                    {"token": "A", "logprob": -0.2},
                                    {"token": "B", "logprob": -1.4},
                                ],
                            }
                        ]
                    }
                }
            ],
        }


def _hosted_scorer():
    scorer = object.__new__(SemanticScorer)
    scorer.use_llm = True
    scorer.device = torch.device("cpu")
    scorer.llm_model_name = "Qwen/Qwen2.5-7B-Instruct"
    scorer._log_once_keys = set()
    scorer._decision_stats = {
        "requested": 0,
        "hosted_attempted": 0,
        "hosted_scored": 0,
        "probe_failures": 0,
        "scoring_failures": 0,
        "local_fallbacks": 0,
    }
    scorer._decision_probe_cache = {}
    scorer._last_decision_backend_meta = {}
    scorer.log = lambda *args, **kwargs: None
    scorer._llm_router = SimpleNamespace(
        resolve_task=lambda task, require_logprobs=False: SimpleNamespace(
            backend="openrouter",
            profile_name="hosted",
            model="qwen/qwen3.5-122b-a10b",
            decision_capable=True,
            fallback_triggered=False,
            fallback_reason=None,
        ),
        profiles={"hosted": SimpleNamespace(name="hosted", tokenizer="Qwen/Qwen3.5-122B-A10B")},
        hosted=_DummyHosted(),
    )
    scorer.hosted_decision_labels = ("A", "B")
    scorer.hosted_decision_logit_bias = 20.0
    scorer._probe_hosted_decision_profile = lambda profile: {
        "passed": True,
        "provider": "unit-test-provider",
        "error": None,
    }
    scorer._hosted_decision_logit_bias = lambda profile: (
        {"11": 20.0, "12": 20.0},
        {"A": [11], "B": [12]},
    )
    scorer._record_hosted_decision_chat_debug = lambda *args, **kwargs: None
    scorer._ensure_local_llm = lambda: (_ for _ in ()).throw(
        AssertionError("Local fallback was not expected")
    )
    scorer.llm_decision_batch_size = 8
    scorer.request_seed = 123
    return scorer


def test_hosted_chat_binary_head_returns_binary_probability(monkeypatch):
    monkeypatch.setenv("EXACT_EXPERIMENT_MODE", "1")
    scorer = _hosted_scorer()
    scorer.llm_experiment_enabled = False
    probs = scorer.llm_yesno_probs_batched(
        ["src"],
        ["tgt"],
        ["src summary"],
        ["tgt summary"],
    )

    expected = torch.softmax(torch.tensor([-0.2, -1.4], dtype=torch.float64), dim=-1)[0].item()
    assert torch.isclose(probs[0], torch.tensor(expected, dtype=torch.float32))
    assert scorer._last_decision_backend_meta["endpoint"] == "chat/completions"
    assert scorer._last_decision_backend_meta["decision_probe_passed"] is True
    assert scorer._last_decision_backend_meta["provider"] == "unit-test-provider"
    assert (
        scorer._last_decision_backend_meta["decision_scoring_mode"] == "chat_logprobs_binary_head"
    )
    assert scorer.llm_decision_stats()["hosted_attempted"] == 1
    assert scorer.llm_decision_stats()["hosted_scored"] == 1
    assert scorer.llm_decision_stats()["local_fallbacks"] == 0


def test_binary_teacher_retains_ordered_costs_without_changing_prompts(monkeypatch):
    monkeypatch.setenv("EXACT_EXPERIMENT_MODE", "1")
    scorer = _hosted_scorer()
    scorer.llm_experiment_config = {"exemplars": "off"}
    seen = []
    original = scorer._llm_router.hosted.chat_completion

    def hosted(**kwargs):
        seen.append(kwargs["messages"])
        return {**original(**kwargs), "id": "response", "usage": {"total_tokens": 19}}

    scorer._llm_router.hosted.chat_completion = hosted
    baseline = scorer.llm_yesno_probs_batched(["src"], ["tgt"], ["facts"], [""])
    probabilities, records = scorer.llm_binary_decision_probs(
        ["source"], ["target"], ["src"], ["tgt"], ["facts"], [0.6]
    )
    assert torch.equal(baseline, probabilities)
    assert seen[0] == seen[1]
    assert records[0]["pair_probabilities"] == {"target": float(probabilities[0])}
    assert records[0]["calls"][0]["usage"]["total_tokens"] == 19
    assert records[0]["calls"][0]["response_id"] == "response"
    assert records[0]["calls"][0]["target"] == "target"


def test_binary_exemplars_reach_hosted_prompt_and_reject_training_query(monkeypatch):
    from exact.impl.models.selector.llm_learning import teacher_identity

    monkeypatch.setenv("EXACT_EXPERIMENT_MODE", "1")
    scorer = _hosted_scorer()
    scorer.llm_experiment_config = {
        "decision": {"mode": "binary"},
        "exemplars": "knn",
        "exemplar_count": 3,
    }
    scorer._llm_router.routing = SimpleNamespace(
        decision_profile="hosted", default_profile="hosted"
    )
    scorer._llm_router.profiles["hosted"].backend = "openrouter"
    scorer._llm_router.profiles["hosted"].revision = "fixture-revision"
    scorer._attached_dataset = SimpleNamespace(dataset_signature="fixture")
    scorer._exemplar_artifact = {
        "schema_version": 1,
        "training_sources": ["train1", "train2", "train3", "train4"],
        "application": {"dataset_signature": "fixture"},
        "teacher_binding": teacher_identity(scorer),
        "examples": [
            {
                "source": source,
                "features": [0.6, 0.6, 0.0, 1.0],
                "candidates": [
                    {"target": source + "target", "equivalent": True, "evidence": "training fact"}
                ],
            }
            for source in ["train1", "train2", "train3", "train4"]
        ],
    }
    seen = []
    original = scorer._llm_router.hosted.chat_completion

    def hosted(**kwargs):
        seen.append(kwargs["messages"][-1]["content"])
        return original(**kwargs)

    scorer._llm_router.hosted.chat_completion = hosted
    _, records = scorer.llm_binary_decision_probs(
        ["heldout"], ["target"], ["src"], ["tgt"], ["facts"], [0.6]
    )
    assert records[0]["exemplar_sources"] == ["train1", "train2", "train3"]
    assert "Training-only examples" in seen[0]
    assert "training fact" in seen[0] and "train4" not in seen[0]
    with pytest.raises(ValueError, match="overlaps training"):
        scorer.llm_binary_decision_probs(["train1"], ["target"], ["src"], ["tgt"], ["facts"], [0.6])
    assert len(seen) == 1


def test_binary_teacher_rejects_unobserved_probabilities(monkeypatch):
    scorer = _hosted_scorer()
    scorer.llm_experiment_config = {"exemplars": "off"}
    monkeypatch.setattr(
        scorer, "llm_yesno_probs_batched", lambda *args, **kwargs: torch.tensor([0.8])
    )
    with pytest.raises(ValueError, match="observed hosted response"):
        scorer.llm_binary_decision_probs(["s"], ["t"], ["src"], ["tgt"], ["facts"], [0.6])


def _inject_hosted_failure(scorer, failure):
    if failure == "probe":
        scorer._probe_hosted_decision_profile = lambda profile: {
            "passed": False,
            "error": "fixture probe unavailable",
        }
    else:

        def locked(**kwargs):
            raise sqlite3.OperationalError("database is locked")

        scorer._llm_router.hosted.chat_completion = locked
        scorer._record_hosted_decision_request_debug = lambda *args, **kwargs: "fixture"


@pytest.mark.parametrize("failure", ["probe", "request"])
@pytest.mark.parametrize(
    "global_mode,selector_enabled", [(True, False), (False, True), (True, True)]
)
def test_experiment_failure_preserves_hosted_backend_without_local_fallback(
    monkeypatch, failure, global_mode, selector_enabled
):
    monkeypatch.setenv("EXACT_EXPERIMENT_MODE", "1" if global_mode else "0")
    scorer = _hosted_scorer()
    scorer.llm_experiment_enabled = selector_enabled
    _inject_hosted_failure(scorer, failure)

    with pytest.raises(RuntimeError, match="experiment stage paused") as error:
        scorer.llm_yesno_probs_batched(["src"], ["tgt"], ["source"], ["target"])

    if failure == "request":
        assert isinstance(error.value.__cause__.__cause__, sqlite3.OperationalError)
    else:
        assert "fixture probe unavailable" in str(error.value)
    assert scorer._last_decision_backend_meta["backend"] == "openrouter"
    assert scorer._last_decision_backend_meta["fallback_triggered"] is False
    assert scorer.llm_decision_stats()["local_fallbacks"] == 0


@pytest.mark.parametrize("failure", ["probe", "request"])
def test_production_hosted_failure_still_reaches_local_fallback(monkeypatch, failure):
    monkeypatch.delenv("EXACT_EXPERIMENT_MODE", raising=False)
    scorer = _hosted_scorer()
    scorer.llm_experiment_enabled = False
    _inject_hosted_failure(scorer, failure)
    # The existing fake loader raises instead of downloading a local model.
    with pytest.raises(AssertionError, match="Local fallback was not expected"):
        scorer.llm_yesno_probs_batched(["src"], ["tgt"], ["source"], ["target"])
    assert scorer._last_decision_backend_meta["backend"] == "local_hf"
    assert scorer._last_decision_backend_meta["fallback_triggered"] is True
    assert scorer.llm_decision_stats()["local_fallbacks"] == 1


def test_clean_summary_text_drops_fence_only_response():
    assert SemanticScorer._clean_summary_text("```json\n") == ""


def test_clean_rationale_text_keeps_only_complete_sentences():
    text = (
        '```json\n{"rationale":"First sentence. Second sentence. Third sentence without ending"\n'
    )
    cleaned = SemanticScorer._parse_structured_text(
        text, "rationale", SemanticScorer._clean_rationale_text
    )
    assert cleaned == "First sentence. Second sentence."


def test_clean_rationale_text_keeps_up_to_four_complete_sentences():
    text = "First. Second. Third. Fourth. Fifth."
    cleaned = SemanticScorer._clean_rationale_text(text)
    assert cleaned == "First. Second. Third. Fourth."
