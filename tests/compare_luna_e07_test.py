"""The Luna sensitivity reuses exact E07 evidence without changing the population."""

import fcntl
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path

import httpx
import pytest
import torch
import yaml
import zstandard

from exact.llm import prompt_budget
from exact.llm.decisions import DecisionsClient
from tests.hosted_spending_admission_test import install
from tests.llm_prompt_budget_test import Tokenizer, policy
from tools import compare_luna_e07 as replay


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    install(tmp_path, monkeypatch, campaign_cap=1_000_000, experiment_cap=800_000)
    prompt = policy(tmp_path, monkeypatch)
    monkeypatch.setenv("EXACT_HOSTED_EXPERIMENT_ID", "E07")
    monkeypatch.setattr(prompt_budget, "_load_tokenizer", lambda *_: Tokenizer())
    for name in (
        "EXACT_HOSTED_CACHE_ONLY",
        "EXACT_OPENROUTER_REQUEST_CAP",
        "EXACT_OPENROUTER_TOKEN_CAP",
    ):
        monkeypatch.delenv(name, raising=False)
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    profile = dict(
        model="openai/gpt-4o-mini-2024-07-18",
        revision="2024-07-18",
        tokenizer="pinned/fixture",
        tokenizer_revision="a" * 40,
        api_base="https://openrouter.ai/api/v1",
        provider={"only": ["OpenAI"], "allow_fallbacks": False, "require_parameters": True},
    )
    params = dict(
        hosted_decision_labels=["A", "B"],
        hosted_decision_logit_bias=20.0,
        use_llm_calibration=False,
    )
    config = dict(
        data=dict(
            reference_role="valid", execution_mode="global_alignment", refs={"valid": "public.tsv"}
        ),
        selector=dict(runtime_enabled=False),
        supervision=dict(mode="label_free", components={"llm": "label_free"}),
        pipeline=[dict(params=params)],
        llm=dict(
            profiles={"original": profile},
            routing=dict(decision_profile="original"),
            experiment=dict(
                decision=dict(mode="binary", evidence="structured_packet"), fusion_weight="beta_u"
            ),
        ),
    )
    (source / "_inputs").mkdir()
    (source / "_inputs/resolved.config.yaml").write_text(yaml.safe_dump(config))
    write(
        source / "experiment_manifest.json",
        dict(
            status="complete",
            experiment_id="E07",
            arm_id="facts_binary",
            split_role="development",
            reference_role="valid",
            generate_rationales=False,
            reference_completeness="known_incomplete",
            seed=17,
        ),
    )
    extraction = dict(
        threshold=0.7,
        source_cardinality=1,
        target_cardinality=1,
        local_alignment=False,
        extraction=dict(mode="greedy", anchor_conflict_policy="error"),
    )
    sources, explanations = [], []
    ledger = tmp_path / "history.sqlite3"
    with sqlite3.connect(ledger) as db:
        db.execute("CREATE TABLE requests(request_id TEXT PRIMARY KEY,identity TEXT)")
        db.execute(
            "CREATE TABLE attempts(request_id TEXT,number INTEGER,state TEXT,status INTEGER,raw BLOB,sha256 TEXT)"
        )
        for source_id, targets in (
            ("s0", [("t0", 0.9), ("decoy0", 0.8)]),
            ("s1", [("t1", 0.1), ("decoy1", 0.9)]),
        ):
            candidates = []
            for target, p in targets:
                p = float(torch.tensor(p, dtype=torch.float32))
                base = float(torch.tensor(0.9, dtype=torch.float32))
                final = float(0.5 * torch.tensor(base) + 0.5 * torch.tensor(p))
                diag = dict(mix_weight=0.5, llm_probability=p)
                candidates.append(
                    dict(
                        target=target,
                        S_base=base,
                        S_final=final,
                        p_llm=p,
                        protected_exact=False,
                        llm_gate_invoked=True,
                        llm_gate_diagnostics=json.dumps(diag),
                    )
                )
                record = dict(
                    src_iri=source_id,
                    tgt_iri=target,
                    selected_labels=dict(source=source_id, target=target),
                    llm_pair_brief=f"ontology facts {source_id} {target}",
                    llm_pair_evidence_packet=f"ontology facts {source_id} {target}",
                    confidences=dict(llm_gate_invoked=True),
                    backend_usage=dict(decision=dict(label_ids={"A": [32], "B": [33]})),
                )
                explanations.append(record)
                _, identity = replay.historical_identity(record, profile, params, 17)
                key, encoded = replay.request_identity(identity)
                raw = json.dumps(
                    dict(
                        model=profile["model"],
                        provider="OpenAI",
                        choices=[
                            dict(
                                logprobs=dict(
                                    content=[
                                        dict(
                                            top_logprobs=[
                                                dict(token="A", logprob=math.log(p)),
                                                dict(token="B", logprob=math.log1p(-p)),
                                            ]
                                        )
                                    ]
                                )
                            )
                        ],
                    )
                ).encode()
                db.execute("INSERT INTO requests VALUES(?,?)", (key, encoded))
                db.execute(
                    "INSERT INTO attempts VALUES(?,1,'completed',200,?,?)",
                    (key, raw, hashlib.sha256(raw).hexdigest()),
                )
            sources.append(dict(Src=source_id, candidates=candidates))
    sources.append(
        dict(
            Src="anchor",
            candidates=[
                dict(
                    target="anchor",
                    S_base=None,
                    S_final=1.0,
                    protected_exact=True,
                    llm_gate_invoked=None,
                )
            ],
        )
    )
    write(source / "source_decisions.json", dict(records=sources, policy=extraction))
    write(
        source / "dataset/candidate_pool_sample_manifest.json",
        dict(
            retrieval_config=dict(source_sample=dict(eligible_source_iris=["s0", "s1", "anchor"]))
        ),
    )
    write(source / "explanations/index.json", dict(shards={"0": dict(path="shards/0.jsonl.zst")}))
    shard = source / "explanations/shards/0.jsonl.zst"
    shard.parent.mkdir()
    shard.write_bytes(
        zstandard.ZstdCompressor().compress(
            "".join(json.dumps(x) + "\n" for x in explanations).encode()
        )
    )
    for name, rows in (
        ("alignment/paper.maps_global.tsv", [("s0", "t0"), ("s1", "decoy1"), ("anchor", "anchor")]),
        (
            "evaluation_inputs/full_reference.tsv",
            [("s0", "t0"), ("s1", "t1"), ("anchor", "anchor")],
        ),
    ):
        path = source / name
        path.parent.mkdir()
        path.write_text("Src\tTgt\n" + "".join(f"{a}\t{b}\n" for a, b in rows))
    write(
        source / "evaluation/evaluation_results.json",
        dict(builtin=dict(P=0.667, R=0.667, F1=0.667)),
    )
    route = tmp_path / "endpoints.json"
    write(
        route,
        {
            "data": {
                "endpoints": [
                    {
                        "model_id": replay.MODEL,
                        "tag": "openai",
                        "provider_name": "OpenAI",
                        "pricing": {"prompt": "0.0000001", "completion": "0"},
                    }
                ]
            }
        },
    )
    kwargs = dict(
        source=source,
        ledger=ledger,
        output=output,
        spending_policy=Path(replay.selected_policy()["path"]),
        prompt_policy=Path(prompt["path"]),
        endpoint_metadata=route,
        expected_sources=3,
        expected_decisions=4,
    )
    manifest = replay.prepare(**kwargs)
    calls = []

    def respond(request):
        value = json.loads(request.content)
        calls.append(value)
        p = 0.99 if "Target entity: t" in value["state"]["evidence"] else 0.01
        return httpx.Response(
            200,
            json=dict(
                model=replay.MODEL,
                provider="OpenAI",
                answers={"equivalent": dict(type="noul", noul=p)},
                usage=dict(input_tokens=30, output_tokens=0),
            ),
        )

    client = DecisionsClient(
        output / "decisions",
        tokenizer=profile["tokenizer"],
        tokenizer_revision=profile["tokenizer_revision"],
        api_key="sk-or-fake-fixture",
        transport=httpx.MockTransport(respond),
    )
    yield dict(
        output=output,
        source=source,
        kwargs=kwargs,
        manifest=manifest,
        client=client,
        calls=calls,
        explanations=explanations,
    )
    client.close()


def test_offline_preparation_binds_every_prompt_and_exact_original_mapping(prepared):
    manifest = prepared["manifest"]
    assert manifest["same_evidence_verified"] == 4
    assert manifest["baseline_parity"]["exact_mapping_parity"]
    assert manifest["baseline_parity"]["metrics"]["F1"] == 2 / 3
    assert prepared["calls"] == []
    panel = replay.read(prepared["output"] / "panel.json")
    assert len(panel["candidates"]) == 5 and len(panel["judgments"]) == 4
    assert all("ontology facts" in x["input_text"] for x in panel["judgments"])
    assert manifest["cost_estimate_usd"] == manifest["input_tokens_estimate"] * 0.0000001
    assert manifest["endpoint_metadata"] == replay.bound(prepared["kwargs"]["endpoint_metadata"])


@pytest.mark.parametrize("price", ["NaN", "-1", "0.0000003"])
def test_invalid_or_excessive_captured_route_price_blocks_preparation(prepared, price):
    path = prepared["kwargs"]["endpoint_metadata"]
    data = replay.read(path)
    data["data"]["endpoints"][0]["pricing"]["prompt"] = price
    write(path, data)
    with pytest.raises(ValueError, match="Captured Decisions pricing"):
        replay.prepare(**prepared["kwargs"])
    assert prepared["calls"] == []


def test_real_adapter_execution_changes_only_probabilities_and_reports_paired_outcomes(prepared):
    report = replay.run(prepared["output"], client=prepared["client"])
    assert report["metrics"]["F1"] == 1.0
    assert ["anchor", "anchor"] in report["alignment"]
    assert report["source_paired_f1"]["n_units"] == 3
    assert report["paired"]["corrected"] == 1 and report["paired"]["changed_unknown"] == 1
    assert report["coverage"] == dict(required=4, usable=4, refusals=0)
    assert report["usage"]["roles"]["decision"]["prompt_tokens"] == 120
    assert report["usage"]["roles"]["decision"]["unpriced_attempts"] == 4
    assert all(
        set(wire) == {"model", "state", "questions", "provider"} for wire in prepared["calls"]
    )
    assert len(prepared["calls"]) == 4
    assert replay.run(prepared["output"], client=prepared["client"]) == report
    assert len(prepared["calls"]) == 4


def test_resume_torn_receipt_replays_saved_response_without_paid_duplicate(prepared, monkeypatch):
    original = replay.status

    def interrupted(output, state, completed, total, **extra):
        if state == "running" and completed == 2:
            raise RuntimeError("fixture interrupted before third send")
        original(output, state, completed, total, **extra)

    monkeypatch.setattr(replay, "status", interrupted)
    with pytest.raises(RuntimeError, match="fixture interrupted"):
        replay.run(prepared["output"], client=prepared["client"])
    assert len(prepared["calls"]) == 2
    with (prepared["output"] / "predictions.jsonl").open("ab") as stream:
        stream.write(b'{"torn":')
    monkeypatch.setattr(replay, "status", original)
    report = replay.run(prepared["output"], client=prepared["client"])
    assert report["metrics"]["F1"] == 1.0 and len(prepared["calls"]) == 4
    assert len((prepared["output"] / "predictions.jsonl").read_text().splitlines()) == 4


def test_refusal_is_durable_and_never_becomes_negative_or_dropped(prepared, monkeypatch):
    predict = prepared["client"].predict

    def refuse(*args):
        answer = predict(*args)
        return {**answer, "probability": None, "refused": True, "refusal": "model_refusal"}

    monkeypatch.setattr(prepared["client"], "predict", refuse)
    for _ in range(2):
        with pytest.raises(ValueError, match="Refused"):
            replay.run(prepared["output"], client=prepared["client"])
    assert len(prepared["calls"]) == 1
    assert not (prepared["output"] / "completion.json").exists()


def test_dependency_change_blocks_before_new_calls_and_clears_running_status(prepared):
    replay.status(prepared["output"], "running", 0, 4)
    path = prepared["source"] / "source_decisions.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Frozen input changed"):
        replay.run(prepared["output"], client=prepared["client"])
    assert prepared["calls"] == []
    assert replay.read(prepared["output"] / "status.json")["status"] == "blocked"


def test_changed_last_evidence_packet_cannot_claim_same_prompt(prepared):
    rows = prepared["explanations"]
    rows[-1]["llm_pair_brief"] += " additional fact"
    rows[-1]["llm_pair_evidence_packet"] = rows[-1]["llm_pair_brief"]
    path = prepared["source"] / "explanations/shards/0.jsonl.zst"
    path.write_bytes(
        zstandard.ZstdCompressor().compress("".join(json.dumps(x) + "\n" for x in rows).encode())
    )
    with pytest.raises(ValueError, match="Exact historical prompt/response unavailable"):
        replay.prepare(**prepared["kwargs"])
    assert prepared["calls"] == []


def test_wrong_family_is_blocked_without_creating_requests(prepared, monkeypatch):
    monkeypatch.setenv("EXACT_HOSTED_EXPERIMENT_ID", "E17")
    with pytest.raises(ValueError, match="E07 campaign"):
        replay.run(prepared["output"], client=prepared["client"])
    assert prepared["calls"] == []
    assert replay.read(prepared["output"] / "status.json")["status"] == "blocked"


def test_stop_before_calls_preserves_panel_and_resumes(prepared):
    stop = prepared["output"] / "STOP"
    stop.touch()
    assert replay.run(prepared["output"], client=prepared["client"]) is None
    assert not prepared["calls"] and not (prepared["output"] / "completion.json").exists()
    stop.unlink()
    assert replay.run(prepared["output"], client=prepared["client"])["status"] == "complete"


def test_cli_owner_conflict_never_overwrites_shared_receipts(prepared, monkeypatch):
    output = prepared["output"]
    for name in ("status.json", "exit.json", "completion.json"):
        write(output / name, {"existing_owner": name})
    before = {
        name: (output / name).read_bytes()
        for name in ("status.json", "exit.json", "completion.json")
    }
    monkeypatch.setattr(sys, "argv", ["compare_luna_e07", "run", "--output", str(output)])
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit) as caught:
            replay.main()
        assert caught.value.code == 73
    assert before == {name: (output / name).read_bytes() for name in before}
    assert prepared["calls"] == []


def test_completed_resume_repairs_stale_status_without_new_calls(prepared):
    output = prepared["output"]
    report = replay.run(output, client=prepared["client"])
    completion = (output / "completion.json").read_bytes()
    replay.status(output, "running", 3, 4)
    write(output / "exit.json", {"exit_code": 1})
    assert replay.run(output, client=prepared["client"]) == report
    assert replay.read(output / "status.json")["status"] == "complete"
    assert replay.read(output / "exit.json")["exit_code"] == 0
    assert (output / "completion.json").read_bytes() == completion
    assert len(prepared["calls"]) == 4


def test_completed_resume_rejects_changed_prediction_receipt(prepared):
    output = prepared["output"]
    replay.run(output, client=prepared["client"])
    with (output / "predictions.jsonl").open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        replay.run(output, client=prepared["client"])
    assert replay.read(output / "status.json")["status"] == "blocked"
    assert replay.read(output / "exit.json")["exit_code"] == 1
    assert len(prepared["calls"]) == 4


@pytest.mark.parametrize("with_receipts", [False, True])
def test_missing_initialized_ledger_blocks_before_client_creation(
    prepared, monkeypatch, with_receipts
):
    output = prepared["output"]
    if with_receipts:
        replay.run(output, client=prepared["client"])
        (output / "completion.json").unlink()
        (output / "comparison.json").unlink()
    else:
        (output / "STOP").touch()
        replay.run(output, client=prepared["client"])
        (output / "STOP").unlink()
        assert not (output / "predictions.jsonl").exists()
    assert (output / "ledger-binding.json").exists()
    database = output / "decisions/requests.sqlite3"
    database.unlink()
    from exact.llm import decisions

    def forbidden_client(*args, **kwargs):
        pytest.fail("Missing initialized ledger must be caught before client creation")

    monkeypatch.setattr(decisions, "DecisionsClient", forbidden_client)
    with pytest.raises(ValueError, match="Initialized decision ledger is missing"):
        replay.run(output)
    assert not database.exists()
    assert len(prepared["calls"]) == (4 if with_receipts else 0)
    assert replay.read(output / "status.json")["status"] == "blocked"


def test_saved_receipt_without_completed_attempt_cannot_be_resent(prepared):
    output = prepared["output"]
    replay.run(output, client=prepared["client"])
    (output / "completion.json").unlink()
    (output / "comparison.json").unlink()
    receipt = json.loads((output / "predictions.jsonl").read_text().splitlines()[0])
    with sqlite3.connect(output / "decisions/requests.sqlite3") as database:
        database.execute(
            "DELETE FROM attempts WHERE request_id=?", (receipt["answer"]["request_id"],)
        )
    with pytest.raises(RuntimeError, match="cached-only"):
        replay.run(output, client=prepared["client"])
    assert len(prepared["calls"]) == 4
    assert replay.read(output / "status.json")["status"] == "blocked"
