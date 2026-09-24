"""Run at most twelve CPU pairs to validate current explanation exports offline.

This operational check performs no training, candidate generation, LLM requests,
or benchmark evaluation. Pair-file labels/scores are never supplied to scoring.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from exact.impl.datasets.pair_adaptive_context import PairAdaptiveContextDataset
from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer
from exact.impl.trainer import SemanticAlignmentRunner
from exact.runs import RunReader
from exact.runs.manifest import refresh_manifest
from exact_inspect.contracts import file_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "pairs", "encoder", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12, choices=range(1, 13))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory for immutable validation results.")
    if not args.encoder.is_dir():
        parser.error("Encoder must be an existing local model snapshot directory.")
    args.output.mkdir(parents=True)
    configuration = {
        "purpose": "bounded_operational_export_validation",
        "max_pairs": args.limit,
        "source_hash": file_hash(args.source),
        "target_hash": file_hash(args.target),
        "pairs_hash": file_hash(args.pairs),
        "lexical_model": str(args.encoder),
        "context_model": str(args.encoder),
        "device": "cpu",
        "ontology_scope": "root",
        "import_policy": "ignore",
        "llm": "not_run",
        "training": "not_run",
        "retrieval": "provided_pool",
        "threshold": 0.5,
        "source_cardinality": 1,
        "target_cardinality": 1,
        "limits": {"hierarchy": 3, "object": 8, "difference": 4, "attribute": 5},
    }
    (args.output / "config.yaml").write_text(
        json.dumps(configuration, sort_keys=True, indent=2) + "\n"
    )
    torch.set_num_threads(2)
    pairs = pd.read_csv(args.pairs, sep="\t", nrows=args.limit)
    pairs = pairs.rename(columns={"SrcEntity": "Src", "TgtEntity": "Tgt"})[["Src", "Tgt"]].copy()
    pairs["SrcKind"], pairs["TgtKind"] = "class", "class"
    pairs["Label"], pairs["cand_sim"], pairs["cand_channels"] = 0, 0.0, "provided_pool"
    dataset = PairAdaptiveContextDataset(
        output_path=args.output,
        source_options={"import_policy": "ignore"},
        target_options={"import_policy": "ignore"},
        only_taxonomy=True,
        verbaliser_name=None,
        projection_include_literals=False,
        cache_ok=False,
        max_hierarchy_triples_per_family=3,
        max_object_triples=8,
        max_diff_triples=4,
        max_attr_items=5,
        num_workers=0,
    )
    dataset.load_ontologies(args.source, args.target)
    dataset._candidates = pairs
    dataset._refresh_candidate_pool_manifest(origin="provided", candidate_file=args.pairs)
    dataset.process()
    parameters = {
        "use_lexical": True,
        "use_context": True,
        "use_llm": False,
        "llm_model_name": None,
        "lexical_model_name": str(args.encoder),
        "context_model_name": str(args.encoder),
        "return_explanations": True,
        "persist_cache_to_disk": False,
        "cache_dir": str(args.output / "model-cache"),
    }
    runner = SemanticAlignmentRunner(
        dataset=dataset,
        model=PairAdaptiveSemanticScorer,
        model_params=parameters,
        device=torch.device("cpu"),
        output_dir=args.output,
    )
    predictions, _ = runner.predict(
        threshold=0.5,
        cardinality=1,
        target_cardinality=1,
        batch_size=4,
        num_workers=0,
        enable_checkpoints=True,
        audit_shard_compression="none",
        mixed_precision=False,
        log_every=100,
    )
    predictions = runner.apply_prefilter(
        predictions, threshold=0.5, cardinality=1, target_cardinality=1
    )
    runner.save_results(
        predictions,
        output_formats=["tsv-global"],
        save_json=True,
        save_csv=False,
        save_stats_csv=False,
    )
    reader = RunReader.open(args.output)
    refresh_manifest(reader.layout, run_id=runner._run_id)
    records = list(reader.iter_explanations())
    counts = Counter()
    for record in records:
        for channel, payload in record.get("triple_attributions", {}).items():
            for group in payload.values() if channel == "hierarchy" else [payload]:
                for side in ("source", "target"):
                    for item in group.get(side, []):
                        counts[channel + ":" + item.get("provenance_status", "missing")] += 1
        for side in ("source", "target"):
            for item in record.get("attributes", {}).get(side, []):
                counts["attributes:" + item.get("provenance_status", "missing")] += 1
    report = {
        "schema_version": 1,
        "fixture_provenance": "actual_current_exact_output",
        "scope": "bounded_operational_export_check",
        "pair_limit": args.limit,
        "source_hash": file_hash(args.source),
        "target_hash": file_hash(args.target),
        "pair_file_hash": file_hash(args.pairs),
        "reference_labels_used": False,
        "encoder_files": {
            p.relative_to(args.encoder).as_posix(): file_hash(p)
            for p in sorted(args.encoder.rglob("*"))
            if p.is_file()
        },
        "candidate_count": len(records),
        "source_count": pairs.Src.nunique(),
        "scores": [record["confidences"]["S_final"] for record in records],
        "mapping_count": len(predictions),
        "provenance_counts": dict(counts),
        "candidate_decision_artifact_hash": file_hash(reader.layout.candidate_decisions_path),
    }
    (args.output / "export-validation.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
