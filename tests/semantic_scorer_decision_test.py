import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


def test_hosted_chat_binary_head_returns_binary_probability():
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
