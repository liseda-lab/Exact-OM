"""Offline qualification of final Elephant routing, caching and score semantics."""

import json
import math
import sqlite3
from pathlib import Path

import pytest

from exact.decisions.pipeline import (
    INSTRUCTIONS,
    REVISION,
    apply,
    file_hash,
    fingerprint,
    prepare,
    read,
    records,
    request,
    run,
)


def candidate(source="s", target="a", score=0.6, uncertainty=0.5, **extra):
    return dict(
        Src=source,
        Tgt=target,
        S_base=score,
        U=uncertainty,
        selected_labels={"source": source, "target": target},
        llm_structured_evidence_packet="structured evidence",
        llm_evidence_packet="local evidence",
        **extra,
    )


def inputs(tmp_path, rows, sources=None):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    universe = tmp_path / "sources.txt"
    universe.write_text("\n".join(sources or sorted({r["Src"] for r in rows})) + "\n")
    return path, universe


class Engine:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def predict(self, payload):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("interrupted")
        assert payload["questions"]["equivalent"]["instructions"] == INSTRUCTIONS
        return {"probability": 0.9, "local_model_call_attempted": True}


def test_global_gap_gate_uses_whole_universe_empty_sources_and_iri_ties(tmp_path):
    rows = [candidate(s, t, 0.7) for s in ("s2", "s1") for t in ("b", "a")]
    candidates, sources = inputs(tmp_path, rows, ["s1", "s2", "empty1", "empty2", "empty3"])
    plan = prepare(candidates, sources, tmp_path / "plan")
    assert read(plan)["selected_sources"] == ["s1"]
    assert [r["pair"] for r in records(plan.parent / "requests.jsonl")] == [
        ["s1", "a"],
        ["s1", "b"],
    ]
    assert prepare(candidates, sources, plan.parent) == plan


def test_corrected_global_scores_anchors_normally_local_preserves_anchor_contract(tmp_path):
    rows = [
        candidate("s", "anchor", 0.6, exact_anchor=True, original_exact_anchor=True),
        candidate("s", "a", 0.55),
    ]
    candidates, sources = inputs(tmp_path, rows)
    global_plan = prepare(candidates, sources, tmp_path / "global")
    local_plan = prepare(candidates, sources, tmp_path / "local", policy="local-train")
    assert read(global_plan)["request_count"] == 2
    assert read(local_plan)["request_count"] == 1
    assert (
        "structured evidence"
        in next(records(global_plan.parent / "requests.jsonl"))["payload"]["state"]["evidence"]
    )
    assert (
        "local evidence"
        in next(records(local_plan.parent / "requests.jsonl"))["payload"]["state"]["evidence"]
    )


def test_request_limit_train_policy_and_duplicate_rejection(tmp_path):
    rows = [candidate("s", f"t{i}", 0.1 * i) for i in range(7)]
    candidates, sources = inputs(tmp_path, rows)
    plan = prepare(candidates, sources, tmp_path / "plan", policy="local-train")
    assert read(plan)["request_count"] == 5
    assert [r["pair"][1] for r in records(plan.parent / "requests.jsonl")] == [
        "t6",
        "t5",
        "t4",
        "t3",
        "t2",
    ]
    candidates.write_text(candidates.read_text() + json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="inputs changed"):
        prepare(candidates, sources, plan.parent, policy="local-train")
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(candidates, sources, tmp_path / "bad")


def test_source_groups_cannot_silently_change_gate(tmp_path):
    candidates, sources = inputs(tmp_path, [candidate("a"), candidate("b"), candidate("a", "z")])
    with pytest.raises(ValueError, match="contiguous"):
        prepare(candidates, sources, tmp_path / "plan")


def test_checkpoint_restarts_only_missing_requests_and_preserves_pairs(tmp_path):
    candidates, sources = inputs(
        tmp_path, [candidate(target="a"), candidate(target="b", score=0.4)]
    )
    plan = prepare(candidates, sources, tmp_path / "plan")
    cache = tmp_path / "cache.sqlite"
    with pytest.raises(RuntimeError, match="interrupted"):
        run(plan, cache, engine=Engine(fail_at=2))
    second = Engine()
    result = run(plan, cache, engine=second)
    assert second.calls == 1 and result["cached_response_reused"] == 1
    assert run(plan, cache, cache_only=True)["local_model_calls"] == 0
    output = tmp_path / "fused.jsonl"
    receipt = apply(candidates, plan, cache, output)
    actual = list(records(output))
    assert actual[0]["S_pair_final"] == pytest.approx(0.6 * 0.6 + 0.4 * 0.9)
    assert actual[0]["llm_weight"] == 0.4  # beta=.8 times U=.5, not constant .5
    assert [r["Tgt"] for r in actual] == ["a", "b"]
    assert receipt["provider_calls"] == 0
    assert actual[0]["cached_response_reused"] and not actual[0]["local_model_call_attempted"]
    assert (
        not actual[0]["live_provider_call_attempted"]
        and not actual[0]["selector_stage_consultation"]
    )
    old = output.read_bytes()
    assert apply(candidates, plan, cache, output) == receipt
    assert old == output.read_bytes()


def test_missing_corrupt_or_changed_cache_never_falls_back(tmp_path):
    candidates, sources = inputs(tmp_path, [candidate()])
    plan = prepare(candidates, sources, tmp_path / "plan")
    cache = tmp_path / "cache.sqlite"
    with pytest.raises(ValueError, match="No cached"):
        run(plan, cache, cache_only=True)
    with pytest.raises(ValueError, match="Incomplete"):
        apply(candidates, plan, cache, tmp_path / "absent.jsonl")
    assert not (tmp_path / "absent.jsonl").exists()
    run(plan, cache, engine=Engine())
    with sqlite3.connect(cache) as db:
        db.execute("UPDATE answers SET value=?", ('{"probability":0.1}',))
    with pytest.raises(ValueError, match="checksum"):
        run(plan, cache, cache_only=True)
    with pytest.raises(ValueError, match="checksum"):
        apply(candidates, plan, cache, tmp_path / "corrupt.jsonl")


def test_zero_effect_is_not_confused_with_no_response_and_gate_never_calls_unselected(tmp_path):
    rows = [candidate("s1", "a", 0.4, 0), candidate("s1", "b", 0.4, 0)]
    rows += [candidate(f"s{i}", "a", 0.99, 0.7) for i in range(2, 6)]
    candidates, sources = inputs(tmp_path, rows)
    plan = prepare(candidates, sources, tmp_path / "plan")
    run(plan, tmp_path / "cache", engine=Engine())
    output = tmp_path / "out"
    apply(candidates, plan, tmp_path / "cache", output)
    result = list(records(output))
    assert result[0]["llm_used"] and not result[0]["effective_pair_score_contribution"]
    assert result[-1]["S_pair_final"] == 0.99 and result[-1]["p_llm"] is None


def test_embedded_request_identity_and_nan_are_rejected(tmp_path):
    row = candidate()
    canonical = request(row, policy="global")
    assert request(dict(row, decision_request=canonical), policy="global") == canonical
    canonical["payload"]["state"]["evidence"] += " changed"
    with pytest.raises(ValueError, match="identity"):
        request(dict(row, decision_request=canonical), policy="global")
    candidates, sources = inputs(tmp_path, [candidate(score=math.nan)])
    with pytest.raises(ValueError, match="finite"):
        prepare(candidates, sources, tmp_path / "plan")


def test_pinned_runtime_is_lazy_and_has_no_server_paths():
    import exact.decisions

    manifest_path = Path(exact.decisions.__file__).with_name("runtime.json")
    manifest = read(manifest_path)
    assert manifest["revision"] == REVISION
    assert manifest["settings"]["readout"] == "slot_lm"
    assert "/home/" not in manifest_path.read_text()


def test_original_gate_accepts_native_anchor_sidecar_without_fabricating_rows(tmp_path):
    rows = [candidate("a", "t", 0.6), candidate("b", "t", 0.8)]
    candidates, sources = inputs(tmp_path, rows)
    anchors = tmp_path / "anchors.json"
    anchors.write_text(json.dumps([{"Src": "b", "Tgt": "anchor", "S_final": 1.0}]))
    without = prepare(candidates, sources, tmp_path / "without", policy="global-original")
    with_anchors = prepare(
        candidates, sources, tmp_path / "with", policy="global-original", anchors_path=anchors
    )
    assert read(without)["selected_sources"] == ["a"]
    assert read(with_anchors)["selected_sources"] == ["b"]
    rec = next(records(with_anchors.parent / "requests.jsonl"))
    assert rec["pair"] == ["b", "t"]
    assert "structured evidence" in rec["payload"]["state"]["evidence"]
    with pytest.raises(ValueError, match="Corrected global"):
        prepare(candidates, sources, tmp_path / "bad", anchors_path=anchors)


def test_source_with_only_tied_anchors_still_counts_in_original_gate(tmp_path):
    candidates, sources = inputs(tmp_path, [candidate("b", "t", 0.9)], ["a", "b"])
    anchors = tmp_path / "anchors.json"
    anchors.write_text(
        json.dumps(
            [{"Src": "a", "Tgt": "x", "S_final": 1.0}, {"Src": "a", "Tgt": "y", "S_final": 1.0}]
        )
    )
    plan = prepare(
        candidates, sources, tmp_path / "plan", policy="local-application", anchors_path=anchors
    )
    assert read(plan)["selected_sources"] == ["a"]
    assert read(plan)["request_count"] == 0


def test_anchor_and_implementation_identity_validation(tmp_path):
    candidates, sources = inputs(tmp_path, [candidate()])
    anchors = tmp_path / "anchors.json"
    anchors.write_text(json.dumps([{"Src": "s", "Tgt": "a", "Score": 0.5}]))
    with pytest.raises(ValueError, match="score of one"):
        prepare(candidates, sources, tmp_path / "bad", policy="local-train", anchors_path=anchors)
    anchors.write_text(json.dumps([{"Src": "s", "Tgt": "a", "Score": 1.0}] * 2))
    with pytest.raises(ValueError, match="Duplicate native anchor"):
        prepare(candidates, sources, tmp_path / "bad", policy="local-train", anchors_path=anchors)
    plan = prepare(candidates, sources, tmp_path / "plan")
    changed = read(plan)
    changed["identity"]["implementation_sha256"] = "0" * 64
    plan.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="implementation"):
        run(plan, tmp_path / "cache", cache_only=True)
