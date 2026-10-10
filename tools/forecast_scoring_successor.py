#!/usr/bin/env python3
"""Refresh counted-work scenarios; unknown scientific work never becomes a zero ETA."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from exact.runs.store import _atomic_json
from exact.utils.provenance import sha256_file
from tools.storage_guard import tree_bytes


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def read_hosted_history(ledger):
    """Keep incomplete token/cost receipts out of their measured denominators."""
    with sqlite3.connect(f"file:{ledger.resolve()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("""
            SELECT json_extract(r.identity,'$.role') role,a.state,count(*) attempts,
              count(a.elapsed_seconds) measured_attempts,sum(a.elapsed_seconds) service_seconds,
              sum(json_extract(a.usage,'$.prompt_tokens')) prompt_tokens,
              sum(json_extract(a.usage,'$.completion_tokens')) completion_tokens,
              sum(json_extract(a.usage,'$.cost')) reported_usd,sum(a.usage IS NULL) usage_missing,
              count(json_extract(a.usage,'$.cost')) priced_attempts,
              sum(json_extract(a.usage,'$.prompt_tokens') IS NOT NULL
                  AND json_extract(a.usage,'$.completion_tokens') IS NOT NULL) token_attempts,
              sum(CASE WHEN json_extract(a.usage,'$.prompt_tokens') IS NOT NULL
                  AND json_extract(a.usage,'$.completion_tokens') IS NOT NULL
                  THEN json_extract(a.usage,'$.prompt_tokens') + json_extract(a.usage,'$.completion_tokens') END) measured_tokens
            FROM attempts a JOIN requests r USING(request_id) GROUP BY role,a.state
        """)]


def decision_sensitivities(primary_cap, decision):
    def scaled(requests, total, observed):
        return requests * total / observed if total is not None and observed else None

    return [{"route_fraction_assumption": fraction, "primary_decision_requests": primary_cap * fraction,
             "primary_decision_tokens": scaled(primary_cap * fraction, decision.get("measured_tokens"), decision.get("token_attempts", 0)),
             "primary_decision_usd": scaled(primary_cap * fraction, decision.get("reported_usd"), decision.get("priced_attempts", 0)),
             "primary_decision_service_seconds": scaled(primary_cap * fraction, decision.get("service_seconds"), decision.get("measured_attempts", 0)),
             "unknown_token_attempts": decision.get("attempts", 0) - decision.get("token_attempts", 0),
             "unpriced_attempts": decision.get("attempts", 0) - decision.get("priced_attempts", 0),
             "observed_averages_do_not_bound_unknown_exposure": True}
            for fraction in (.001, .01, .1)]


def build(work_counts, measurements, g4, spending_snapshot, storage_snapshot):
    inventory = json.loads(work_counts.read_text())
    cases = inventory["cases"]
    configs = [g4 / (arm + "--D1-global_alignment.resolved.json") for arm in ("baseline", "core")]
    for path in configs:
        config = json.loads(path.read_text())
        if (config["candidates"]["top_k"] != 20 or config["candidates"]["adaptive_k"]["enabled"]
                or config["matching"]["anchor_rescoring"]["mode"] != "off"):
            raise ValueError("Conditional top-20 scenario no longer matches inspected recipes")
    per_case = {name: {"global_cap_per_arm": item["source_population"] * min(20, item["target_population"]),
                      "local_occurrences_per_arm": item["local_candidate_occurrences"],
                      "local_unique_pairs_per_arm": item["local_unique_pairs"]}
                for name, item in cases.items()}
    primary_cap = 2 * sum(item["global_cap_per_arm"] + item["local_occurrences_per_arm"] for item in per_case.values())
    plus_control = primary_cap + per_case["H0"]["global_cap_per_arm"] + per_case["H0"]["local_occurrences_per_arm"]
    samples = []
    for path in measurements:
        data = json.loads(path.read_text())
        cold = data["cold"]
        root = path.parent / "cold/ordinary-evidence"
        physical = tree_bytes(root)
        rows = sum(c["rows"] for c in cold["chunks"])
        rate = cold["pairs_per_second"]
        samples.append({"receipt": binding(path), "gpu": data["gpu"],
                        "source_revision": data.get("source_revision"),
                        "execution_flags": data.get("execution_flags", {}),
                        "scientific_acceptance": "requires_separate_exact_parity_receipt",
                        "rate_is_approved_deployment_forecast": False,
                        "cold_unique_pairs": cold["distinct_computed_pairs"], "cold_pairs_per_second": rate,
                        "minimum_contract_met": cold["minimum_measurement_contract_met"],
                        "warm_rows_per_second": data["warm"]["processed_rows_per_second"],
                        "replay_rows_per_second": data["replay"]["processed_rows_per_second"],
                        "native_preparation_seconds": data["native_preparation_seconds"],
                        "model_loading_seconds": data["model_loading_seconds"],
                        "durable_physical_bytes_per_row": physical / rows if rows else None,
                        "conditional_primary_cap_scoring_days": primary_cap / rate / 86400 if rate else None,
                        "conditional_with_control_cap_scoring_days": plus_control / rate / 86400 if rate else None,
                        "conditional_primary_cap_evidence_bytes": primary_cap * physical / rows if rows else None,
                        "conditional_with_control_cap_evidence_bytes": plus_control * physical / rows if rows else None,
                        "extrapolation_scope": "measured sample rate applied to conditional occurrence cap; not whole-program ETA or bound on slower unseen work"})
    timing_rows = []
    for path in sorted((g4 / "runtime/exact-om-focused-v2/screen/runs/G4").glob("*/*/seed-17/timings.json")):
        data = json.loads(path.read_text())
        for session in data["sessions"]:
            if session.get("ended_at"):
                timing_rows.append({"binding": binding(path), "cell": str(path.parent.parent.relative_to(g4)),
                                    "stages": session["stages"], "scope": "historical inclusive stages, not cold qualification"})
    ledger = g4 / "runtime/exact-om-focused-v2/openrouter/requests.sqlite3"
    hosted = read_hosted_history(ledger)
    spending = json.loads(spending_snapshot.read_text())
    bootstrap = spending["metadata"]["bootstrap"]
    known_tokens = bootstrap["accounted_tokens"] + sum(row["accounted_tokens"] for row in spending["grants_by_family"])
    known_usd = bootstrap["reported_usd"] + sum(row["reported_usd"] or 0 for row in spending["grants_by_family"])
    # Service-time and token sensitivities use actual historical observations;
    # current routes/seed-specific requests must still be counted before admission.
    decision = next((row for row in hosted if row["role"] == "decision" and row["state"] == "completed"), {})
    sensitivities = decision_sensitivities(primary_cap, decision)
    return {"schema_version": 1, "kind": "conditional_resource_forecast_not_execution_admission",
            "count_binding": binding(work_counts), "recipe_bindings": [binding(path) for path in configs],
            "logical_design": {"primary": 12, "label_free_maximum": 2, "published": 3,
                               "bounded_maximum": 54, "additional_component_maximum": 20},
            "conditional_top20_work": {"per_case": per_case, "primary_occurrences_cap": primary_cap,
                "with_H0_label_free_cap": plus_control,
                "actual_unique_numerical_pairs": None, "actual_reuse_fraction": None,
                "conditions": "unchanged top20 recipe; before exact prefilter, overlap and reuse; excludes fitting/bounded/component/LogMap work"},
            "measurements_and_scoring_storage_scenarios": samples, "historical_completed_timings": timing_rows,
            "historical_hosted": {"ledger": str(ledger.resolve()), "scope": "inherited history; do not add to lifetime ledger again",
                                  "roles": hosted, "service_seconds_are_not_wall_time": True},
            "decision_only_sensitivities_not_forecasts": sensitivities,
            "historical_spending": {"snapshot": binding(spending_snapshot), "accounted_tokens": known_tokens,
                "reported_usd": known_usd, "unpriced_historical_attempts": bootstrap["unpriced_attempts"],
                "campaign_tokens_remaining_at_snapshot": 200_000_000 - known_tokens,
                "E17_tokens_remaining_at_snapshot": 25_000_000 - sum(row["accounted_tokens"] for row in spending["grants_by_family"] if row["family"] == "E17"),
                "limits_raised": False, "new_paid_requests_for_implementation": 0},
            "storage_snapshot": binding(storage_snapshot),
            "storage_projection_exclusions": ["candidate tables and indexes", "fitted artifacts", "submission exports", "request ledgers",
                "bounded/component and published outputs", "database/WAL growth within physical caps", "atomic replacement reserve"],
            "remaining_whole_program": {"wall_seconds": None, "gpu_hours": None, "tokens": None, "usd": None, "physical_bytes": None},
            "unresolved_dependencies": ["corrected G4 selection and seed-specific fitted artifact freeze",
                "actual full global retrieval, exact prefilter and cross-mode reuse identities",
                "unique fits, missing texts and bounded/component physical executions",
                "per-case/per-path GPU parity and measured rates", "current-recipe hosted route fractions, tokens, rate limits and retries",
                "native preparation, retrieval, fitting, global extraction, output, recovery and LogMap rates",
                "complete all-root storage growth including atomic replacements"],
            "admissible_as_whole_program_eta": False, "forecast_is_kill_timer": False,
            "target_200_used_as_measurement": False, "live_policies_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("work-counts", "g4", "spending-snapshot", "storage-snapshot", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--measurement", action="append", type=Path, default=[])
    args = parser.parse_args()
    record = build(args.work_counts, args.measurement, args.g4, args.spending_snapshot, args.storage_snapshot)
    _atomic_json(args.output, record)
    print(json.dumps({"output": str(args.output), "primary_occurrence_cap": record["conditional_top20_work"]["primary_occurrences_cap"],
                      "whole_program_eta": None, "limits_changed": False}))


if __name__ == "__main__":
    main()
