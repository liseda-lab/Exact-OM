"""Bounded offline development trace probe, separate from retained study runs.

Uses three real DOID entities and their nine ordered pairs, an existing local
encoder and the normal current matcher pipeline. No training, evaluation labels,
provider calls, candidate retrieval or final experiment changes are performed.
The self pairs are diagnostic inputs, not biomedical alignment effectiveness data.
"""

import argparse
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path


def run(args):
    if args.output.exists() or not args.encoder.is_dir():
        raise ValueError("Use a new output directory and an existing offline encoder")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    import pandas as pd
    import torch

    from exact.impl.datasets.pair_adaptive_context import PairAdaptiveContextDataset
    from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer
    from exact.impl.trainer import SemanticAlignmentRunner
    from exact.runs import RunReader
    from exact_inspect.contracts import file_hash

    torch.set_num_threads(2)
    args.output.mkdir(parents=True)
    inputs = ["http://purl.obolibrary.org/obo/" + x for x in ("DOID_0080422", "DOID_176", "DOID_13499")]
    pairs = pd.DataFrame([(a, b) for a in inputs for b in inputs], columns=["Src", "Tgt"])
    pair_file = args.output / "diagnostic-pairs.tsv"
    pairs.to_csv(pair_file, sep="\t", index=False)
    pairs["SrcKind"], pairs["TgtKind"] = "class", "class"
    pairs["Label"], pairs["cand_sim"], pairs["cand_channels"] = 0, 0.0, "provided_pool"
    config = {"purpose": "development_review_trace_smoke", "max_pairs": 9, "source_hash": file_hash(args.ontology), "target_hash": file_hash(args.ontology), "pairs_hash": file_hash(pair_file), "source_scope": "declared_DOID_derivative_root_only", "threshold": .9, "cardinality": 1, "filter_exact_matches": True, "drop_exact_match_sources": False, "training": "not_run", "llm": "not_run", "device": "cpu", "threads": 2, "reference_labels_used": False, "encoder": str(args.encoder), "harness_sha256": file_hash(__file__), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "versions": {p: importlib.metadata.version(p) for p in ("torch", "pyowl-core")}}
    (args.output / "config.yaml").write_text(json.dumps(config, indent=2))
    status_path = args.output / "smoke-status.json"
    started = time.monotonic()

    def status(phase, **extra):
        status_path.write_text(json.dumps({"phase": phase, "seconds": time.monotonic() - started, **extra}, indent=2))

    try:
        status("ontology_admission")
        dataset = PairAdaptiveContextDataset(output_path=args.output, source_options={"import_policy": "ignore"}, target_options={"import_policy": "ignore"}, only_taxonomy=True, verbaliser_name=None, projection_include_literals=False, cache_ok=False, filter_exact_matches=True, drop_exact_match_sources=False, max_hierarchy_triples_per_family=3, max_object_triples=8, max_diff_triples=4, max_attr_items=5, num_workers=0)
        dataset.load_ontologies(args.ontology, args.ontology)
        dataset._candidates = pairs
        dataset._refresh_candidate_pool_manifest(origin="provided", candidate_file=pair_file)
        status("exact_matching")
        dataset.get_exact_matches()
        # Keep the real exact matcher output only for these nine provided pairs.
        # This bounds the smoke and does not manufacture any exact-match event.
        dataset._exact_matches = dataset.exact_matches.merge(pairs[["Src", "Tgt", "SrcKind", "TgtKind"]], on=["Src", "Tgt", "SrcKind", "TgtKind"], how="inner")
        dataset.process()
        status("scoring")
        runner = SemanticAlignmentRunner(dataset=dataset, model=PairAdaptiveSemanticScorer, model_params={"use_lexical": True, "use_context": True, "use_llm": False, "llm_model_name": None, "lexical_model_name": str(args.encoder), "context_model_name": str(args.encoder), "return_explanations": True, "persist_cache_to_disk": False, "cache_dir": str(args.output / "model-cache")}, device=torch.device("cpu"), output_dir=args.output)
        predictions, _ = runner.predict(threshold=.9, cardinality=1, target_cardinality=1, batch_size=3, num_workers=0, enable_checkpoints=True, audit_shard_compression="none", mixed_precision=False, log_every=100)
        predictions = runner.apply_prefilter(predictions, threshold=.9, cardinality=1, target_cardinality=1)
        runner.save_results(predictions, output_formats=["tsv-global"], save_json=True, save_csv=False, save_stats_csv=False)
        reader = RunReader.open(args.output)
        records = (reader.candidate_decisions() or {}).get("records", [])
        if len(records) > 9:
            raise ValueError("Smoke exceeded its pair bound; do not admit the output")
        status("complete", pairs=len(records), candidate_decisions_sha256=file_hash(args.output / "candidate_decisions.json"), source_decisions_sha256=file_hash(args.output / "source_decisions.json"), limitation="New diagnostic DOID-to-DOID run; not retained NCIT/DOID study history, final experiment or effectiveness evaluation")
    except Exception as error:
        status("failed", error_type=type(error).__name__, reason=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ontology", "encoder", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
