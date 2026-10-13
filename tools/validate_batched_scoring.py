#!/usr/bin/env python3
"""One unpaid, bounded fitted-G4 comparison; report drift without claiming equivalence.

All three variants share native ontologies and already materialized public features.
Each owns a fresh scorer and encoder/numerical caches. Nothing here fits, dispatches
scientific work, or promotes an optimization on the strength of fixture predictions.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from exact.experiments.throughput import group_chunks  # noqa: E402
from exact.runs.store import ExplanationStore  # noqa: E402
from tools import benchmark_scoring_throughput as benchmark  # noqa: E402
from tools.compare_scoring_parity import compare_fitted, compare_records  # noqa: E402


VARIANTS = ("legacy", "batched", "batched_graphs")
MODES = ("cold", "warm", "replay", "partial")


def variant_flags(name):
    if name not in VARIANTS:
        raise ValueError("Unknown validation variant")
    return {
        "EXACT_PAIR_CONTEXT_BATCHING": "0" if name == "legacy" else "1",
        "EXACT_PAIR_CONTEXT_SCHEDULE": "staged",
        "EXACT_PAIR_CONTEXT_ENCODER": "batch",
        "EXACT_PAIR_CONTEXT_MATRIX_KERNEL": "mm",
        "EXACT_PAIR_CONTEXT_ARITHMETIC": "device",
        "EXACT_ENCODER_STABLE_TOKENIZER_IDENTITY": "0",
        "EXACT_ENCODER_CUDA_GRAPHS": "1" if name == "batched_graphs" else "0",
        "EXACT_PAIR_CONTEXT_BLOCK_PAIRS": "32",
        "EXACT_PAIR_CONTEXT_TEXT_BATCH": "64",
        "EXACT_ENCODER_GRAPH_MAX_SHAPES": "16",
        "EXACT_ENCODER_GRAPH_MAX_BYTES": str(2 * 1024**3),
        "EXACT_ENCODER_GRAPH_MIN_CALLS": "8",
        "EXACT_EMBEDDING_CACHE_MAX_BYTES": str(1024**3),
        "EXACT_NUMERICAL_CACHE_MAX_BYTES": str(1024**3),
        "EXACT_NUMERICAL_CACHE_MAX_DISK_BYTES": str(1024**3),
        "EXACT_NUMERICAL_CACHE": "1",
    }


def validate_panel(workload):
    queries = workload["queries"]
    pairs = [(query["source"], target) for query in queries for target in query["candidates"]]
    if (
        workload.get("namespace") != "qualification_only"
        or len(pairs) != 5000
        or len(set(pairs)) != 5000
        or len({q["qid"] for q in queries}) != len(queries)
        or len({q["source"] for q in queries}) != len(queries)
        or any(not q["candidates"] for q in queries)
    ):
        raise ValueError(
            "Validation requires exactly 5,000 distinct pairs in complete source groups"
        )
    chunks = list(group_chunks(queries, 200))
    if len(chunks) != 25 or any(sum(len(q["candidates"]) for q in c) != 200 for c in chunks):
        raise ValueError("Validation requires the existing complete-group 25 by 200 fitted panel")
    return {"pairs": len(pairs), "source_groups": len(queries), "chunks": len(chunks)}


def finite(value, path="record"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite output at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite(item, f"{path}[{index}]")


class EvidenceValidator:
    """Validate native evidence membership/citations, allowing different selected facts."""

    def __init__(self):
        self.pools = {}
        self.counts = Counter()

    def pool(self, dataset, scorer, side, iri):
        key = (side, iri)
        if key not in self.pools:
            features = dataset.get_entity_features(iri, "src" if side == "source" else "tgt")
            items = [
                scorer._with_item_id("hierarchy", side, item, family=family)
                for family, rows in features.get("hierarchy", {}).items()
                for item in rows
            ]
            for channel, field in (("similarity", "object_triples"), ("attribute", "attributes")):
                items.extend(
                    scorer._with_item_id(channel, side, item) for item in features.get(field, [])
                )
            self.pools[key] = {item["item_id"]: item for item in items}
        return self.pools[key]

    def __call__(self, dataset, scorer, records):
        for record in records:
            finite(record)
            pools = {
                side: self.pool(dataset, scorer, side, record[iri])
                for side, iri in (("source", "src_iri"), ("target", "tgt_iri"))
            }
            attributions = record["triple_attributions"]
            selections = list(attributions["hierarchy"].values()) + [
                attributions["similarity"],
                attributions["difference"],
                record["attributes"],
            ]
            for selected in selections:
                for side in ("source", "target"):
                    seen = set()
                    for item in selected[side]:
                        item_id = item["item_id"]
                        if item_id in seen or item_id not in pools[side]:
                            raise ValueError(
                                "Selected fact is duplicated or absent from its native entity pool"
                            )
                        seen.add(item_id)
                        native = pools[side][item_id]
                        self.content(item, native)
                        for field in ("source_axiom_refs", "axiom_origins", "derivation"):
                            if item.get(field) != native.get(field):
                                raise ValueError(
                                    f"Selected fact changed its native citation {field}"
                                )
                        self.counts["selected_items"] += 1
                        self.counts["citations"] += len(item.get("source_axiom_refs", []))
                        if item.get("provenance_status") == "provenance_unavailable":
                            self.counts["unavailable_provenance_items"] += 1
            self.links(record["cross_side_provenance"], pools)
            self.counts["records"] += 1

    @staticmethod
    def content(item, native):
        # Attribute display aliases differ from the native projection keys.
        aliases = {"property": "prop", "property_iri": "prop_iri"}
        for field in (
            "subject_iri",
            "rel_iri",
            "object_iri",
            "entity_iri",
            "property",
            "property_iri",
            "text",
            "value",
            "triple",
            "datatype",
            "language",
        ):
            if field not in item:
                continue
            expected = native.get(aliases.get(field, field))
            actual = item[field]
            if field == "triple":
                expected = list(expected or [])
            elif field not in {"datatype", "language"}:
                expected = str(expected or "").strip()
            if actual != expected:
                raise ValueError(f"Selected fact changed its native content {field}")

    @staticmethod
    def links(provenance, pools):
        for link in provenance["lexical"]:
            if (link["source_ref"], link["target_ref"]) != ("__source__", "__target__"):
                raise ValueError("Lexical citation points to an unknown endpoint")
        for links in [*provenance["hierarchy"].values(), provenance["similarity"]]:
            for link in links:
                if any(link[f"{side}_item_id"] not in pools[side] for side in pools):
                    raise ValueError("Cross-side citation points outside the native entity pool")
        for family in ("attributes", "difference"):
            for side, other in (("source", "target"), ("target", "source")):
                for link in provenance[family][side]:
                    if link["item_id"] not in pools[side] or link["anchor_ref"] not in (
                        set(pools[other]) | {"__" + other + "__"}
                    ):
                        raise ValueError("Evidence anchor points outside the native entity pool")


def records_at(root, mode):
    return list(ExplanationStore(root / mode / "ordinary-evidence", read_only=True).iter_all())


def mapping_drift(left, right):
    left_pairs, right_pairs = (set(map(tuple, result["mappings"])) for result in (left, right))
    return {
        "left": len(left_pairs),
        "right": len(right_pairs),
        "removed": sorted(left_pairs - right_pairs),
        "added": sorted(right_pairs - left_pairs),
    }


def drift(left, right, mode):
    left_records, right_records = records_at(left, mode), records_at(right, mode)
    # This strict result is evidence about drift, not an algorithmic acceptance gate.
    fitted = compare_fitted(left, right, ordinary=(left_records, right_records), mode=mode)
    decisions = [
        json.loads((root / mode / "fitted-decisions.json").read_text()) for root in (left, right)
    ]
    if (
        decisions[0]["policy"] != decisions[1]["policy"]
        or decisions[0]["protected_pairs"] != decisions[1]["protected_pairs"]
    ):
        raise ValueError("Variant changed extraction policy or protected exact anchors")
    scores = [
        {(row["Src"], row["Tgt"]): row["S_final"] for row in result["rows"]} for result in decisions
    ]
    return {
        "ordinary": compare_records(left_records, right_records, tolerance=0),
        "fitted": fitted,
        "fitted_final_score_max_absolute_error": max(
            abs(scores[0][pair] - scores[1][pair]) for pair in scores[0]
        ),
        "mappings": mapping_drift(*decisions),
        "interpretation": "observed drift only; no scientific equivalence or quality conclusion",
    }


def cache_check(root):
    warm, replay = records_at(root, "warm"), records_at(root, "replay")
    ordinary = compare_records(warm, replay, tolerance=0)
    # Compare complete saved fitted outputs as well; replay never changes the denominator.
    fitted_equal = all(
        json.loads((root / "warm" / filename).read_text())
        == json.loads((root / "replay" / filename).read_text())
        for filename in ("fitted-decisions.json", "selector-evidence.json", "hosted-fixtures.json")
    )
    return {
        "status": "passed" if ordinary["status"] == "passed" and fitted_equal else "failed",
        "ordinary": ordinary,
        "exact_fitted_and_hosted_replay": fitted_equal,
    }


def validate_saved_mode(root, mode, *, expected_pairs=None, expected_chunks=None):
    """A valid prefix is insufficient: each mode must finish its complete declared bound."""
    expected_pairs = (
        expected_pairs if expected_pairs is not None else (5000 if mode == "cold" else 600)
    )
    expected_chunks = (
        expected_chunks if expected_chunks is not None else (25 if mode == "cold" else 3)
    )
    chunks = json.loads((root / (mode + ".json")).read_text())["chunks"]
    if (
        len(chunks) != expected_chunks
        or sum(row["rows"] + row["exact_prefiltered_rows"] for row in chunks) != expected_pairs
    ):
        raise ValueError("Saved mode did not complete its declared candidate population")
    records = records_at(root, mode)
    result = compare_fitted(root, root, ordinary=(records, records), mode=mode)
    if result["status"] != "passed":
        raise ValueError(f"Invalid saved fitted outputs in {root.name}/{mode}: {result}")
    return result


def timing_summary(measurement):
    cold = measurement["cold"]
    seconds = sum(row["elapsed_seconds"] for row in cold["chunks"])
    pairs = sum(row["rows"] for row in cold["chunks"])
    preparation = (
        measurement["shared_native_preparation_seconds"]
        + measurement["shared_public_feature_preparation"]["seconds"]
    )
    return {
        "model_scored_pairs": pairs,
        "candidate_occurrences": pairs
        + sum(row["exact_prefiltered_rows"] for row in cold["chunks"]),
        "exact_prefiltered_pairs": sum(row["exact_prefiltered_rows"] for row in cold["chunks"]),
        "warm_feature_cold_encoder_seconds": seconds,
        "warm_feature_cold_encoder_pairs_per_second": pairs / seconds,
        "standalone_preparation_inclusive_seconds": preparation
        + seconds
        + measurement["model_loading_seconds"],
        "shared_preparation_seconds": preparation,
        "structural_validation_inclusive_seconds": cold["phases"]["seconds"].get(
            "structural_validation", 0.0
        ),
        "historical_cold_200_gate_established": False,
        "scope": "fixed 5,000-candidate panel; selector/extraction separately timed; no hosted service latency",
    }


def run(args):
    if args.output.exists():
        raise ValueError("Validation output must be new; never overwrite existing receipts")
    args.output.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "status": "running",
        "namespace": "qualification_only",
        "paid_calls": 0,
        "fixture_outputs_promotable": False,
        "review_required": True,
        "variants": {},
        "comparisons": {},
        "algorithmic_invariants": "pending",
        "cache_correctness": "pending",
        "scientific_quality_established": False,
    }
    result_path = args.output / "results.json"
    started = time.perf_counter()
    try:
        report["panel"] = validate_panel(json.loads(args.workload.read_text()))
        if not args.fitted_manifest:
            raise ValueError("Actual fitted G4 manifest required")
        cache_root = benchmark.isolated_cache_root(args.output, args.cache_root)
        shared = None
        for name in VARIANTS:
            os.environ.update(variant_flags(name))
            destination = args.output / name
            validator = EvidenceValidator()
            settings = SimpleNamespace(
                **vars(args),
                diagnostics="natural",
                chunk_pairs=200,
                max_chunks=25,
                replay_max_chunks=3,
                modes=list(MODES),
                pre_materialize_features=True,
            )
            settings.output, settings.cache_root = destination, cache_root / name
            print(json.dumps({"variant": name, "state": "starting"}), flush=True)
            shared = benchmark.run(settings, shared=shared, validate_records=validator)
            measurement = json.loads((destination / "measurement.json").read_text())
            for mode in MODES:
                validate_saved_mode(destination, mode)
            cache = cache_check(destination)
            report["variants"][name] = {
                "measurement": benchmark.binding(destination / "measurement.json"),
                "evidence_invariants": dict(validator.counts),
                "cache": cache,
                "timing": timing_summary(measurement),
                "routing": {mode: measurement[mode]["routing"] for mode in MODES},
                "execution_flags": measurement["execution_flags"],
                "graph": {mode: measurement[mode]["encoder_graph_cumulative"] for mode in MODES},
            }
            benchmark.write(result_path, report)
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for left, right in (("legacy", "batched"), ("batched", "batched_graphs")):
            report["comparisons"][left + "_versus_" + right] = {
                mode: drift(args.output / left, args.output / right, mode) for mode in MODES
            }
        report["algorithmic_invariants"] = "passed"
        graph_stats = report["variants"]["batched_graphs"]["graph"]["cold"]
        report["graph_acceleration_coverage"] = (
            "exercised"
            if graph_stats.get("captures", 0) > 0 and graph_stats.get("replays", 0) > 0
            else "not_exercised; acceleration remains unvalidated"
        )
        report["cache_correctness"] = (
            "passed"
            if all(v["cache"]["status"] == "passed" for v in report["variants"].values())
            else "failed"
        )
        report["status"] = "completed" if report["cache_correctness"] == "passed" else "failed"
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        benchmark.write(result_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "workload", "templates", "fitted-manifest", "output", "cache-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"], "review_required": True}), flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
