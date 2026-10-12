"""Bounded, CPU-only E07 model-plus-interface sensitivity; no ontology reloads."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
import zstandard

from exact.experiments.difference_replay import _extract
from exact.experiments.evidence_diagnostics import paired_decision_changes
from exact.experiments.paper_metrics import (
    PRFMetrics,
    SourceConfusion,
    SourceEvaluation,
    paired_global_f1_bootstrap,
)
from exact.llm.ledger import request_identity
from exact.llm.prompt_budget import DECISIONS_ENDPOINT as ENDPOINT
from exact.llm.prompt_budget import DECISIONS_MODEL as MODEL
from exact.llm.prompt_budget import load_prompt_budget_policy, validate_prompt_budget
from exact.llm.spending_admission import selected_policy
from exact.utils.fitted_artifacts import fingerprint, freeze_json
from exact.utils.provenance import sha256_file

INSTRUCTIONS = (
    "You are a binary classifier for ontology alignment. "
    "The source and target entities are equivalent."
)
LIMITATION = (
    "Paired development model-plus-interface sensitivity, not a new default or test result. "
    "The reference is known incomplete; unlisted pairs are not verified negatives. "
    "P/R/F1 use the historical benchmark convention; correction/harm uses known labels only."
)


class RunnerBusy(RuntimeError):
    """Another invocation owns the run; its receipts must remain untouched."""


def read(path):
    return json.loads(Path(path).read_text())


def bound(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def verify(binding):
    path = Path(binding["path"])
    if bound(path) != binding:
        raise ValueError(f"Frozen input changed: {path}")
    return path


def pairs(path):
    with Path(path).open(newline="") as stream:
        rows = csv.reader(stream, delimiter="\t")
        next(rows)
        return {tuple(row[:2]) for row in rows}


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Decision requires a numeric probability")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Decision probability must be finite and in [0, 1]")
    return float(value)


def metrics(selected, reference):
    return PRFMetrics.from_counts(
        SourceConfusion(
            len(selected & reference), len(selected - reference), len(reference - selected)
        )
    ).as_dict()


def extract(panel, probabilities):
    """The original float32 fusion and deployed greedy extraction, including anchors."""
    expected = {row["id"] for row in panel["judgments"]}
    if set(probabilities) != expected:
        raise ValueError("Incomplete judgment population; no denominator changes are permitted")
    scores, protected = {}, set()
    for row in panel["candidates"]:
        key = (row["Src"], row["Tgt"])
        score = row["S_final"]
        if row["protected_exact"]:
            protected.add(key)
        if row["judgment_id"] is not None:
            p = probability(probabilities[row["judgment_id"]])
            base, weight, p = (
                torch.tensor(x, dtype=torch.float32) for x in (row["S_base"], row["mix_weight"], p)
            )
            score = float((1 - weight) * base + weight * p)
        scores[key] = score
    return _extract(panel["candidates"], scores, panel["policy"], protected), scores


def historical_identity(record, profile, params, seed):
    """Reconstruct the saved wire request, then require an exact ledger lookup."""
    labels = record["selected_labels"]
    evidence = record["llm_pair_brief"]
    if not evidence or evidence != record["llm_pair_evidence_packet"]:
        raise ValueError("Historical judgment did not use the saved complete facts packet")
    input_text = f"Source entity: {labels['source']}\nTarget entity: {labels['target']}\n\nPair brief:\n{evidence}"
    positive, negative = params["hosted_decision_labels"]
    prompt = (
        f"Return exactly one token: {positive} or {negative}.\n"
        f"{positive} = the source and target entities are equivalent.\n"
        f"{negative} = the source and target entities are not equivalent.\n\n{input_text}"
    )
    bias = {
        str(token): float(params["hosted_decision_logit_bias"])
        for ids in record["backend_usage"]["decision"]["label_ids"].values()
        for token in ids
    }
    return input_text, {
        "schema_version": 1,
        "role": "decision",
        "endpoint": profile["api_base"].rstrip("/") + "/chat/completions",
        "revision": profile["revision"],
        "tokenizer": profile["tokenizer"],
        "tokenizer_revision": profile["tokenizer_revision"],
        "requested_seed": seed,
        "payload": {
            "model": profile["model"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are a binary classifier for ontology alignment.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": True,
            "top_logprobs": 20,
            "logit_bias": bias,
            "seed": seed,
            "provider": {
                **profile["provider"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        },
    }


def verify_historical(db, identity, expected_probability):
    key, encoded = request_identity(identity)
    row = db.execute(
        "SELECT r.identity,a.raw,a.sha256 FROM requests r JOIN attempts a USING(request_id) "
        "WHERE r.request_id=? AND a.state='completed' AND a.status=200 ORDER BY a.number DESC LIMIT 1",
        (key,),
    ).fetchone()
    if row is None or row[0] != encoded or hashlib.sha256(row[1]).hexdigest() != row[2]:
        raise ValueError(f"Exact historical prompt/response unavailable: {key}")
    payload = json.loads(row[1])
    if payload.get("model") != identity["payload"]["model"] or payload.get("provider") != "OpenAI":
        raise ValueError("Historical response has an unexpected model/provider")
    logs = payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    scores = []
    for label in ("A", "B"):
        values = [x["logprob"] for x in logs if x["token"].strip() == label]
        if not values:
            raise ValueError("Historical response lacks binary probabilities")
        scores.append(torch.logsumexp(torch.tensor(values, dtype=torch.float64), dim=0))
    value = float(torch.softmax(torch.stack(scores), dim=0)[0].to(torch.float32))
    if not math.isclose(value, expected_probability, abs_tol=1e-7, rel_tol=1e-6):
        raise ValueError("Historical raw response and saved probability differ")
    return {"request_id": key, "response_sha256": row[2]}


def parity(panel):
    baseline = {row["id"]: row["baseline_probability"] for row in panel["judgments"]}
    selected, scores = extract(panel, baseline)
    expected = {tuple(row) for row in panel["baseline_alignment"]}
    if selected != expected or any(
        abs(scores[(row["Src"], row["Tgt"])] - row["S_final"]) > 1e-7 for row in panel["candidates"]
    ):
        raise ValueError("Original probability replay failed exact mapping/numerical parity")
    computed = metrics(selected, {tuple(row) for row in panel["reference"]})
    if any(abs(round(computed[k], 3) - v) > 1e-9 for k, v in panel["reported_metrics"].items()):
        raise ValueError("Original probability replay failed reported metric parity")
    return {"exact_mapping_parity": True, "all_scores_verified": True, "metrics": computed}


def prepare(
    source,
    ledger,
    output,
    spending_policy,
    prompt_policy,
    endpoint_metadata,
    *,
    expected_sources=200,
    expected_decisions=1000,
):
    """Offline preparation; every judged prompt must match its historical wire identity."""
    from exact.llm.decisions import predicate_identity

    source, output = Path(source).resolve(), Path(output).resolve()
    names = [
        "_inputs/resolved.config.yaml",
        "experiment_manifest.json",
        "source_decisions.json",
        "dataset/candidate_pool_sample_manifest.json",
        "alignment/paper.maps_global.tsv",
        "evaluation_inputs/full_reference.tsv",
        "evaluation/evaluation_results.json",
        "explanations/index.json",
    ]
    dependencies = [bound(source / name) for name in names]
    route_binding = bound(endpoint_metadata)
    endpoints = read(endpoint_metadata)["data"]["endpoints"]
    if (
        len(endpoints) != 1
        or endpoints[0].get("model_id") != MODEL
        or endpoints[0].get("tag") != "openai"
        or endpoints[0].get("provider_name") != "OpenAI"
    ):
        raise ValueError("Captured Decisions endpoint does not match the frozen OpenAI route")
    input_price = float(endpoints[0]["pricing"]["prompt"])
    if (
        not math.isfinite(input_price)
        or not 0 < input_price <= 0.0000002
        or float(endpoints[0]["pricing"]["completion"]) != 0
    ):
        raise ValueError("Captured Decisions pricing is invalid or above the frozen ceiling")
    dependencies.append(route_binding)
    config = yaml.safe_load((source / names[0]).read_text())
    report, decisions = read(source / names[1]), read(source / names[2])
    judge = config["llm"]["experiment"]
    params = config["pipeline"][0]["params"]
    if (
        report.get("status") != "complete"
        or report.get("experiment_id") != "E07"
        or report.get("arm_id") != "facts_binary"
        or report.get("split_role") != "development"
        or report.get("reference_role") != "valid"
        or report.get("generate_rationales") is not False
        or report.get("reference_completeness") != "known_incomplete"
        or config["data"].get("reference_role") != "valid"
        or set(config["data"].get("refs", {})) - {"train", "valid"}
        or config["data"].get("execution_mode") != "global_alignment"
        or config["selector"].get("runtime_enabled") is not False
        or config["supervision"]["mode"] != "label_free"
        or any(v != "label_free" for v in config["supervision"]["components"].values())
        or judge["decision"]["mode"] != "binary"
        or judge["decision"]["evidence"] != "structured_packet"
        or judge["fusion_weight"] != "beta_u"
        or params["use_llm_calibration"] is not False
        or params["hosted_decision_labels"] != ["A", "B"]
    ):
        raise ValueError(
            "Comparison requires the frozen label-free E07 facts_binary development scope"
        )
    policy = decisions["policy"]
    if (
        policy["threshold"] != 0.7
        or policy["source_cardinality"] != 1
        or policy["target_cardinality"] != 1
        or policy["local_alignment"]
        or policy["extraction"]["mode"] != "greedy"
        or policy["extraction"]["anchor_conflict_policy"] != "error"
    ):
        raise ValueError("Unexpected historical extraction policy")
    profile = config["llm"]["profiles"][config["llm"]["routing"]["decision_profile"]]
    spending = selected_policy(bound(spending_policy))
    prompt = load_prompt_budget_policy(bound(prompt_policy), required=True)
    if spending is None or spending["policy"]["schema_version"] != 2:
        raise ValueError("Comparison requires the existing hard-pause spending authority")
    candidates, invoked = [], {}
    for source_row in decisions["records"]:
        for row in source_row["candidates"]:
            pair = (source_row["Src"], row["target"])
            identifier = fingerprint(pair) if row.get("llm_gate_invoked") else None
            diag = json.loads(row["llm_gate_diagnostics"]) if identifier else {}
            item = {
                "Src": pair[0],
                "Tgt": pair[1],
                "S_base": row["S_base"],
                "S_final": row["S_final"],
                "protected_exact": row["protected_exact"],
                "judgment_id": identifier,
                "mix_weight": diag.get("mix_weight", 0.0),
            }
            candidates.append(item)
            if identifier:
                if (
                    pair in invoked
                    or row["protected_exact"]
                    or not math.isclose(diag["llm_probability"], row["p_llm"], abs_tol=1e-7)
                ):
                    raise ValueError("Invalid or duplicate frozen judgment")
                invoked[pair] = {
                    "id": identifier,
                    "baseline_probability": probability(row["p_llm"]),
                }
    if len(decisions["records"]) != expected_sources or len(invoked) != expected_decisions:
        raise ValueError("Frozen E07 population size differs")
    if len({(r["Src"], r["Tgt"]) for r in candidates}) != len(candidates):
        raise ValueError("Duplicate frozen candidates")
    sample = read(source / "dataset/candidate_pool_sample_manifest.json")["retrieval_config"][
        "source_sample"
    ]
    if set(sample["eligible_source_iris"]) != {r["Src"] for r in decisions["records"]}:
        raise ValueError("Source population differs from the frozen sample")
    judgments = {}
    index = read(source / "explanations/index.json")
    ledger = Path(ledger).resolve()
    with sqlite3.connect(ledger.as_uri() + "?mode=ro", uri=True) as db:
        for shard in index["shards"].values():
            path = (source / "explanations" / shard["path"]).resolve()
            if not path.is_relative_to(source / "explanations"):
                raise ValueError("Explanation shard is outside the source run")
            dependencies.append(bound(path))
            with path.open("rb") as stream, zstandard.ZstdDecompressor().stream_reader(
                stream
            ) as reader:
                for line in io.TextIOWrapper(reader):
                    record = json.loads(line)
                    pair = (record["src_iri"], record["tgt_iri"])
                    if pair not in invoked:
                        continue
                    if pair in judgments or not record["confidences"]["llm_gate_invoked"]:
                        raise ValueError("Duplicate/mismatched explanation judgment")
                    item = invoked[pair]
                    input_text, historical = historical_identity(
                        record, profile, params, report["seed"]
                    )
                    history = verify_historical(db, historical, item["baseline_probability"])
                    identity = predicate_identity(
                        input_text,
                        INSTRUCTIONS,
                        tokenizer=profile["tokenizer"],
                        tokenizer_revision=profile["tokenizer_revision"],
                    )
                    estimate = validate_prompt_budget(
                        SimpleNamespace(
                            name="openrouter_luna_decisions",
                            model=MODEL,
                            tokenizer=profile["tokenizer"],
                            tokenizer_revision=profile["tokenizer_revision"],
                        ),
                        identity["payload"],
                        "decisions",
                        role="decision",
                        binding=prompt,
                    )
                    judgments[pair] = {
                        **item,
                        "pair": list(pair),
                        "input_text": input_text,
                        "historical": history,
                        "request_id": request_identity(identity)[0],
                        "prompt_budget": estimate,
                        "reservation_tokens": len(
                            json.dumps(identity["payload"], ensure_ascii=False).encode()
                        )
                        + 256,
                    }
    if judgments.keys() != invoked.keys():
        raise ValueError("Missing explanation evidence for frozen judgment pairs")
    panel = {
        "schema_version": 1,
        "policy": policy,
        "candidates": candidates,
        "judgments": [judgments[k] for k in sorted(judgments)],
        "reference": sorted(pairs(source / "evaluation_inputs/full_reference.tsv")),
        "baseline_alignment": sorted(pairs(source / "alignment/paper.maps_global.tsv")),
        "reported_metrics": read(source / "evaluation/evaluation_results.json")["builtin"],
    }
    if {p[0] for p in panel["reference"]} - set(sample["eligible_source_iris"]):
        raise ValueError("Reference is outside the frozen development population")
    verification = parity(panel)
    for dependency in dependencies:
        verify(dependency)
    freeze_json(output / "panel.json", panel, max_bytes=32 * 1024**2)
    implementation = [
        bound(Path(__file__)),
        *[
            bound(Path(__file__).resolve().parents[1] / name)
            for name in (
                "exact/experiments/difference_replay.py",
                "exact/core/entities/mappings/entity.py",
                "exact/experiments/evidence_diagnostics.py",
                "exact/experiments/paper_metrics.py",
                "exact/llm/decisions.py",
                "exact/llm/prompt_budget.py",
                "exact/llm/ledger.py",
                "exact/llm/spending_admission.py",
            )
        ],
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": "E07",
        "campaign_id": spending["policy"]["campaign_id"],
        "admission_store": spending["policy"]["admission_store"],
        "bootstrap": spending["policy"]["bootstrap"],
        "source_run": str(source),
        "historical_ledger": str(ledger),
        "model": MODEL,
        "endpoint": ENDPOINT,
        "endpoint_metadata": route_binding,
        "input_usd_per_token": input_price,
        "instructions": INSTRUCTIONS,
        "tokenizer": profile["tokenizer"],
        "tokenizer_revision": profile["tokenizer_revision"],
        "spending_policy": bound(spending_policy),
        "prompt_policy": bound(prompt_policy),
        "panel": bound(output / "panel.json"),
        "dependencies": dependencies,
        "implementation": implementation,
        "sources": expected_sources,
        "judgments": len(judgments),
        "same_evidence_verified": len(judgments),
        "input_text_bytes": sum(len(x["input_text"].encode()) for x in judgments.values()),
        "reservation_tokens": sum(x["reservation_tokens"] for x in judgments.values()),
        "input_tokens_estimate": sum(
            x["prompt_budget"]["local_tokenizer_tokens"] for x in judgments.values()
        ),
        "input_admission_tokens": sum(
            x["prompt_budget"]["input_admission_tokens"] for x in judgments.values()
        ),
        "cost_estimate_usd": sum(
            x["prompt_budget"]["local_tokenizer_tokens"] for x in judgments.values()
        )
        * input_price,
        "estimate_basis": "Surrogate tokenizer and hash-bound OpenRouter endpoint input price; output price zero. Not provider-reported usage/cost.",
        "baseline_parity": verification,
        "limitation": LIMITATION,
        "generate_rationales": False,
        "selection_eligible": False,
        "g4_modified": False,
    }
    freeze_json(output / "manifest.json", manifest)
    return manifest


def validate(output):
    manifest = read(output / "manifest.json")
    for item in [
        manifest["panel"],
        manifest["prompt_policy"],
        manifest["spending_policy"],
        *manifest["dependencies"],
        *manifest["implementation"],
    ]:
        verify(item)
    panel = read(output / "panel.json")
    if (
        manifest["model"] != MODEL
        or manifest["endpoint"] != ENDPOINT
        or manifest["instructions"] != INSTRUCTIONS
    ):
        raise ValueError("Frozen decision interface changed")
    parity(panel)
    return manifest, panel


def atomic_record(output, filename, value):
    temporary = output / (filename + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output / filename)
    descriptor = os.open(output, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def status(output, state, completed, total, **extra):
    atomic_record(
        output,
        "status.json",
        {
            "status": state,
            "completed_judgments": completed,
            "total_judgments": total,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        },
    )


def saved_receipts(path, manifest):
    """Recover only a torn final JSONL append; complete malformed records fail closed."""
    result = {}
    if not path.exists():
        return result
    with path.open("r+b") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.truncate(start)
                stream.flush()
                os.fsync(stream.fileno())
                break
            receipt = json.loads(line)
            body = {k: v for k, v in receipt.items() if k != "sha256"}
            if (
                receipt.get("sha256") != fingerprint(body)
                or receipt.get("panel_sha256") != manifest["panel"]["sha256"]
                or receipt["pair_id"] in result
            ):
                raise ValueError("Saved decision receipt changed or duplicated")
            result[receipt["pair_id"]] = receipt
    return result


def comparison(panel, selected):
    baseline = {tuple(x) for x in panel["baseline_alignment"]}
    reference = {tuple(x) for x in panel["reference"]}
    population = {(r["Src"], r["Tgt"]) for r in panel["candidates"]} | reference
    sources = sorted({p[0] for p in population})
    evaluations = []
    for mapping in (baseline, selected):
        counts = {}
        for source in sources:
            emitted = {p for p in mapping if p[0] == source}
            gold = {p for p in reference if p[0] == source}
            counts[source] = SourceConfusion(
                len(emitted & gold), len(emitted - gold), len(gold - emitted)
            )
        evaluations.append(
            {
                ("D0_E03-global_alignment", 17): SourceEvaluation(
                    reference_sha256=fingerprint(panel["reference"]),
                    alignment_sha256=fingerprint(sorted(mapping)),
                    by_source=counts,
                    source_universe_sha256=fingerprint(sources),
                )
            }
        )
    return {
        "alignment": [list(pair) for pair in sorted(selected)],
        "metrics": metrics(selected, reference),
        "baseline_metrics": metrics(baseline, reference),
        "source_paired_f1": paired_global_f1_bootstrap(
            *evaluations, resamples=10_000, seed=17
        ).as_dict(),
        "paired": paired_decision_changes(
            {p: p in baseline for p in population},
            {p: p in selected for p in population},
            {p: True for p in reference},
        ),
    }


def run(output, *, client=None):
    """Resume paid work only under the frozen E07 authority and durable per-pair receipts."""
    output = Path(output).resolve()
    with (output / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerBusy("Another Luna comparison worker owns runner.lock") from exc
        probabilities, receipts, total, owned = {}, {}, 0, client is None
        code = 1
        try:
            if (output / "completion.json").exists():
                report = read(verify(read(output / "completion.json")["report"]))
                verify(report["manifest"])
                verify(report["predictions"])
                if report["status"] != "complete":
                    raise ValueError("Saved completion does not contain a completed report")
                total = report["judgments"]
                status(output, "complete", total, total)
                code = 0
                return report
            manifest, panel = validate(output)
            total = len(panel["judgments"])
            spending = selected_policy()
            prompt = load_prompt_budget_policy(required=True)
            if (
                spending is None
                or {k: spending[k] for k in ("path", "sha256")} != manifest["spending_policy"]
                or {k: prompt[k] for k in ("path", "sha256")} != manifest["prompt_policy"]
                or os.getenv("EXACT_HOSTED_CAMPAIGN_ID") != manifest["campaign_id"]
                or os.getenv("EXACT_HOSTED_EXPERIMENT_ID") != "E07"
            ):
                raise ValueError(
                    "Run is not bound to the prepared E07 campaign admission authority"
                )
            ledger_path = output / "decisions/requests.sqlite3"
            marker = output / "ledger-binding.json"
            ledger_binding = {"panel_sha256": manifest["panel"]["sha256"], "path": str(ledger_path)}
            receipt_path = output / "predictions.jsonl"
            if marker.exists():
                if read(marker) != ledger_binding or not ledger_path.is_file():
                    raise ValueError(
                        "Initialized decision ledger is missing or has a different binding"
                    )
            elif receipt_path.exists():
                raise ValueError("Saved decisions lack their ledger initialization binding")
            if client is None:
                from exact.llm.decisions import DecisionsClient

                client = DecisionsClient(
                    output / "decisions",
                    tokenizer=manifest["tokenizer"],
                    tokenizer_revision=manifest["tokenizer_revision"],
                )
            if client.ledger.path.resolve() != ledger_path or not ledger_path.is_file():
                raise ValueError("Decision client does not own the bound local ledger")
            freeze_json(marker, ledger_binding)
            receipts = saved_receipts(receipt_path, manifest)
            if set(receipts) - {item["id"] for item in panel["judgments"]}:
                raise ValueError("Saved decisions include a different population")
            for item in panel["judgments"]:
                if not ledger_path.is_file():
                    raise ValueError("Initialized decision ledger disappeared")
                if (output / "STOP").exists():
                    status(output, "stopped", len(probabilities), total)
                    code = 75
                    return None
                if item["id"] in receipts:
                    receipt = receipts[item["id"]]
                    original_cache_only = client.cache_only
                    client.cache_only = True
                    try:
                        cached = client.predict(item["input_text"], manifest["instructions"])
                    finally:
                        client.cache_only = original_cache_only
                    if {k: v for k, v in cached.items() if k != "cached"} != {
                        k: v for k, v in receipt["answer"].items() if k != "cached"
                    }:
                        raise ValueError(
                            "Saved decision receipt differs from its durable cached response"
                        )
                else:
                    status(output, "running", len(probabilities), total, active_pair=item["id"])
                    answer = client.predict(item["input_text"], manifest["instructions"])
                    body = {
                        "pair_id": item["id"],
                        "panel_sha256": manifest["panel"]["sha256"],
                        "answer": answer,
                    }
                    receipt = {**body, "sha256": fingerprint(body)}
                    with receipt_path.open("ab") as stream:
                        stream.write(
                            (json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode()
                        )
                        stream.flush()
                        os.fsync(stream.fileno())
                    receipts[item["id"]] = receipt
                answer = receipt["answer"]
                if (
                    receipt["pair_id"] != item["id"]
                    or answer.get("request_id") != item["request_id"]
                    or answer.get("refused") is not False
                ):
                    raise ValueError("Refused or mismatched decision; complete cell required")
                probabilities[item["id"]] = probability(answer.get("probability"))
            selected, _ = extract(panel, probabilities)
            report = {
                "status": "complete",
                "manifest": bound(output / "manifest.json"),
                "predictions": bound(receipt_path),
                **comparison(panel, selected),
                "usage": client.ledger.summary(),
                "cost_estimate_usd": manifest["cost_estimate_usd"],
                "estimate_basis": manifest["estimate_basis"],
                "coverage": {"required": total, "usable": len(probabilities), "refusals": 0},
                "observed_models": sorted({r["answer"]["model"] for r in receipts.values()}),
                "observed_providers": sorted({r["answer"]["provider"] for r in receipts.values()}),
                "limitation": LIMITATION,
                "sources": manifest["sources"],
                "judgments": len(probabilities),
            }
            freeze_json(output / "comparison.json", report)
            freeze_json(
                output / "completion.json",
                {"status": "complete", "report": bound(output / "comparison.json")},
            )
            status(output, "complete", len(probabilities), total)
            code = 0
            return report
        except BaseException as exc:
            status(
                output,
                "blocked",
                len(probabilities),
                total,
                error_type=type(exc).__name__,
                refusals=sum(r["answer"].get("refused") is True for r in receipts.values()),
            )
            raise
        finally:
            if owned and client is not None:
                client.close()
            atomic_record(
                output,
                "exit.json",
                {"exit_code": code, "updated_at": datetime.now(timezone.utc).isoformat()},
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in (
        "source",
        "ledger",
        "output",
        "spending-policy",
        "prompt-policy",
        "endpoint-metadata",
    ):
        prep.add_argument("--" + name, type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        result = prepare(**args)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "sources",
                        "judgments",
                        "same_evidence_verified",
                        "reservation_tokens",
                        "baseline_parity",
                    )
                }
            )
        )
    else:
        try:
            result = run(**args)
        except RunnerBusy as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(73) from exc
        raise SystemExit(0 if result is not None else 75)


if __name__ == "__main__":
    main()
