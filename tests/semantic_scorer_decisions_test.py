"""The opt-in final judge uses the real ledger with an entirely offline transport."""

import copy
import json
import sqlite3
from threading import Event

import httpx
import pytest
import torch

from exact.impl.models import semantic_decisions
from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer
from exact.impl.models.semantic_scorer import SemanticScorer
from exact.llm import prompt_budget as budget
from exact.llm import spending_admission as admission
from exact.llm.decisions import DecisionsClient, EQUIVALENCE_INSTRUCTIONS, pair_evidence
from exact.llm.ledger import RequestLedger
from tests.hosted_spending_admission_test import install
from tests.llm_prompt_budget_test import Tokenizer, policy


def configuration(tmp_path):
    return {
        "device": "cpu",
        "use_lexical": False,
        "use_context": False,
        "persist_cache_to_disk": False,
        "cache_dir": str(tmp_path / "cache"),
        "hosted_decision_interface": "openrouter_decisions",
        "llm_profiles": {
            "mini": {"backend": "openrouter", "model": "openai/gpt-4o-mini"},
            "luna_decisions": {
                "backend": "openrouter",
                "model": budget.DECISIONS_MODEL,
                "revision": "openrouter_decisions_noul_v1",
                "api_base": "https://openrouter.ai/api/alpha",
                "provider": copy.deepcopy(budget.DECISIONS_PROVIDER),
                "tokenizer": "pinned/surrogate",
                "tokenizer_revision": "a" * 40,
            },
        },
        "llm_routing": {
            "default_profile": "mini",
            "summary_profile": "mini",
            "rationale_profile": "mini",
            "decision_profile": "luna_decisions",
        },
    }


def forbidden(*args, **kwargs):
    raise AssertionError("No chat, capability probe, model load or local fallback is permitted")


@pytest.fixture
def guarded(tmp_path, monkeypatch):
    for name in (
        "EXACT_EXPERIMENT_MODE", "EXACT_HOSTED_CACHE_ONLY", "OPENROUTER_API_KEY",
        "EXACT_OPENROUTER_REQUEST_CAP", "EXACT_OPENROUTER_TOKEN_CAP",
        "EXACT_OPENROUTER_RETRY_UNKNOWN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EXACT_OPENROUTER_LEDGER_DIR", str(tmp_path / "ledger"))
    install(tmp_path, monkeypatch)
    policy(tmp_path, monkeypatch)
    monkeypatch.setattr(budget, "_load_tokenizer", lambda *_: Tokenizer())
    state = {"calls": [], "clients": [], "refused": False, "status": 200}

    def send(request):
        state["calls"].append(request)
        wire = json.loads(request.content)
        evidence = wire["state"]["evidence"]
        if state.get("wait_for_second"):
            if "Source entity: first\n" in evidence:
                assert state["wait_for_second"].wait(3)
            else:
                state["wait_for_second"].set()
        if state.get("failure"):
            raise state["failure"]
        return httpx.Response(
            state["status"],
            json={
                "id": "reply-" + evidence.splitlines()[0],
                "model": state.get("model", budget.DECISIONS_MODEL + "-20261006"),
                "provider": "OpenAI",
                "answers": {"equivalent": (
                    {"type": "refusal"} if state["refused"] else
                    {"type": "noul", "noul": 0.8 if "first\n" in evidence else 0.3}
                )},
                "usage": {"input_tokens": 7, "output_tokens": 0, "cost": 0.001},
            },
        )

    def client(*args, **kwargs):
        result = DecisionsClient(*args, transport=httpx.MockTransport(send), **kwargs)
        state["clients"].append(result)
        return result

    monkeypatch.setattr(semantic_decisions, "DecisionsClient", client)
    scorer = PairAdaptiveSemanticScorer(**configuration(tmp_path))
    monkeypatch.setattr(scorer._llm_router, "resolve_task", forbidden)
    monkeypatch.setattr(scorer._llm_router.hosted, "chat_completion", forbidden)
    monkeypatch.setattr(scorer._llm_router.hosted, "resolve_api_key", lambda _: "sk-or-fake")
    monkeypatch.setattr(scorer, "_probe_hosted_decision_profile", forbidden)
    monkeypatch.setattr(scorer, "_ensure_local_llm", forbidden)
    return scorer, state


def invoke(scorer, *, observations=None):
    return scorer.llm_yesno_probs_batched(
        ["first"], ["target"], ["retained evidence"], [""], observations=observations,
    )


def test_binary_batches_preserve_evidence_order_float32_and_durable_accounting(guarded):
    scorer, state = guarded
    state["wait_for_second"] = Event()
    observations = []
    probabilities = scorer.llm_yesno_probs_batched(
        ["first", "second"], ["target", "target"], ["facts1", "facts2"], ["", ""],
        observations=observations, prompt_suffixes=["\ntraining examples", ""],
    )
    assert torch.equal(probabilities, torch.tensor([0.8, 0.3], dtype=torch.float32))
    assert [row["probability"] for row in observations] == [0.8, 0.3]
    payloads = [json.loads(call.content) for call in state["calls"]]
    assert {body["state"]["evidence"] for body in payloads} == {
        pair_evidence("first", "target", "facts1") + "\ntraining examples",
        pair_evidence("second", "target", "facts2"),
    }
    assert all(
        body["questions"] == {"equivalent": {"type": "noul", "instructions": EQUIVALENCE_INSTRUCTIONS}}
        and body["provider"] == budget.DECISIONS_PROVIDER
        and "Return exactly one token" not in body["state"]["evidence"]
        and "seed" not in body
        for body in payloads
    )
    assert all(str(call.url) == budget.DECISIONS_ENDPOINT for call in state["calls"])
    assert all(client._client.is_closed for client in state["clients"])
    assert admission.admission_snapshot()["accounted_tokens"] == 114
    assert admission.admission_snapshot()["reserved_tokens"] == 0
    assert admission.admission_snapshot()["reported_cost_usd"] == pytest.approx(1.252)
    stats = scorer.llm_decision_stats()
    assert stats["hosted_scored"] == 2 and stats["local_fallbacks"] == 0
    assert scorer._last_decision_backend_meta["decision_scoring_mode"] == "openrouter_decisions_noul_v1"
    assert scorer._last_decision_backend_meta["seed_supported"] is False
    assert scorer._llm_router.routing.profile_for_task("summary") == "mini"
    assert scorer._llm_router.routing.profile_for_task("rationale") == "mini"


def test_cached_replay_without_credentials_or_new_spending(guarded, monkeypatch):
    scorer, state = guarded
    original = invoke(scorer)
    original_accounting = admission.admission_snapshot()
    monkeypatch.setenv("EXACT_HOSTED_CACHE_ONLY", "1")
    monkeypatch.setattr(scorer._llm_router.hosted, "resolve_api_key", forbidden)
    observations = []
    assert torch.equal(invoke(scorer, observations=observations), original)
    assert observations[0]["cached"] is True
    assert len(state["calls"]) == 1
    assert admission.admission_snapshot() == original_accounting
    with pytest.raises(RuntimeError, match="cached-only"):
        scorer.llm_yesno_probs_batched(["other"], ["target"], ["evidence"], [""])
    assert len(state["calls"]) == 1


def test_concurrent_identical_prompts_share_one_paid_attempt(guarded):
    scorer, state = guarded
    observations = []
    probabilities = scorer.llm_yesno_probs_batched(
        ["first", "first"], ["target", "target"], ["facts", "facts"], ["", ""],
        observations=observations,
    )
    assert torch.equal(probabilities, torch.tensor([0.8, 0.8], dtype=torch.float32))
    assert len(state["calls"]) == 1
    assert sorted(row["cached"] for row in observations) == [False, True]
    assert admission.admission_snapshot()["accounted_tokens"] == 107


def test_disabled_or_empty_batches_do_not_construct_client(guarded):
    scorer, state = guarded
    assert scorer.llm_yesno_probs_batched([], [], [], []).numel() == 0
    scorer.use_llm = False
    assert torch.equal(invoke(scorer), torch.zeros(1))
    assert state["clients"] == [] and state["calls"] == []


@pytest.mark.parametrize("failure", ["refusal", "wrong_model", "http", "unknown"])
def test_unusable_answers_never_fabricate_scores_fallback_or_rebill(guarded, failure):
    scorer, state = guarded
    if failure == "refusal":
        state["refused"] = True
    elif failure == "wrong_model":
        state["model"] = "wrong-model"
    elif failure == "http":
        state["status"] = 502
    else:
        state["failure"] = httpx.ReadTimeout("fixture lost response")
    for _ in range(2):
        observations = []
        with pytest.raises((ValueError, RuntimeError)):
            invoke(scorer, observations=observations)
        assert observations == []
    assert len(state["calls"]) == 1
    assert scorer.llm_decision_stats()["hosted_scored"] == 0
    assert scorer.llm_decision_stats()["local_fallbacks"] == 0
    if failure in {"refusal", "wrong_model"}:
        assert admission.admission_snapshot()["accounted_tokens"] == 107
    assert all(client._client.is_closed for client in state["clients"])


def test_binary_teacher_observations_keep_pair_ids_and_costs(guarded):
    scorer, _ = guarded
    probabilities, rows = scorer.llm_binary_decision_probs(
        ["iri:first"], ["iri:target"], ["first"], ["target"], ["evidence"], [0.5],
    )
    assert rows[0]["pair_probabilities"] == {"iri:target": float(probabilities[0])}
    assert rows[0]["calls"][0]["target"] == "iri:target"
    assert rows[0]["calls"][0]["usage"]["prompt_tokens"] == 7
    assert rows[0]["calls"][0]["request_id"]


@pytest.mark.parametrize("failure", ["length", "target_summary", "suffix", "grouped", "rationale", "ledger"])
def test_invalid_runtime_inputs_fail_before_requests(guarded, monkeypatch, failure):
    scorer, state = guarded
    source, targets, summaries, suffixes = ["first"], ["target"], [""], None
    if failure == "length":
        targets = []
    elif failure == "target_summary":
        summaries = ["must not silently drop this"]
    elif failure == "suffix":
        suffixes = []
    elif failure == "grouped":
        scorer.llm_experiment_config["decision"]["mode"] = "listwise"
    elif failure == "rationale":
        scorer.generate_llm_rationales = True
    else:
        monkeypatch.delenv("EXACT_OPENROUTER_LEDGER_DIR")
    with pytest.raises(ValueError):
        scorer.llm_yesno_probs_batched(source, targets, ["facts"], summaries, prompt_suffixes=suffixes)
    assert state["calls"] == [] and state["clients"] == []


def test_direct_grouped_path_cannot_silently_fall_through_to_chat(guarded):
    scorer, state = guarded
    with pytest.raises(ValueError, match="binary pair judgments only"):
        scorer.llm_listwise_probs_batched([])
    assert state["calls"] == []


@pytest.mark.parametrize("failure", ["prompt", "spending"])
def test_prompt_and_spending_guards_remain_before_transport(guarded, tmp_path, monkeypatch, failure):
    scorer, state = guarded
    if failure == "prompt":
        policy(tmp_path, monkeypatch, max_input_tokens=1)
    else:
        install(tmp_path, monkeypatch, campaign_cap=100)
    with pytest.raises((budget.PromptBudgetError, admission.HostedSpendingError)):
        invoke(scorer)
    assert state["calls"] == []
    ledger = RequestLedger(tmp_path / "ledger")
    with sqlite3.connect(ledger.path) as db:
        assert db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 0


@pytest.mark.parametrize("field,value", [
    ("backend", "local_hf"), ("model", "openai/gpt-4o-mini"),
    ("revision", "main"), ("api_base", "https://openrouter.ai/api/v1"),
    ("provider", {}), ("tokenizer", None), ("tokenizer_revision", "main"),
])
def test_constructor_rejects_wrong_profile_before_model_loading(tmp_path, field, value):
    config = configuration(tmp_path)
    config["llm_profiles"]["luna_decisions"][field] = value
    with pytest.raises(ValueError, match="exact pinned decision profile"):
        PairAdaptiveSemanticScorer(**config)


def test_only_explicit_interface_changes_fingerprint_and_generic_summaries_rejected(tmp_path):
    config = configuration(tmp_path)
    default = copy.deepcopy(config)
    default.pop("hosted_decision_interface")
    legacy = PairAdaptiveSemanticScorer(**default)
    explicit_chat = PairAdaptiveSemanticScorer(**{**default, "hosted_decision_interface": "chat"})
    decisions = PairAdaptiveSemanticScorer(**config)
    assert legacy.runtime_fingerprint() == explicit_chat.runtime_fingerprint()
    assert "hosted_decision_interface" not in legacy.runtime_fingerprint_payload()
    assert decisions.runtime_fingerprint() != legacy.runtime_fingerprint()
    assert decisions.runtime_fingerprint_payload()["hosted_decision_interface"] == "openrouter_decisions"
    chat_prompt = legacy._hosted_decision_prompt("source", "target", "facts", "")["user"]
    assert chat_prompt[chat_prompt.index("Source entity: "):] == pair_evidence("source", "target", "facts")
    with pytest.raises(ValueError, match="pair-adaptive evidence renderer"):
        SemanticScorer(**config)
    with pytest.raises(ValueError, match="Unsupported hosted_decision_interface"):
        PairAdaptiveSemanticScorer(**{**config, "hosted_decision_interface": "unknown"})
