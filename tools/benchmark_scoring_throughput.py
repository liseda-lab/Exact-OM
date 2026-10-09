#!/usr/bin/env python3
"""Freeze/count public workloads or measure complete numerical scoring, never send requests.

Each run owns a new output/cache namespace. Native preparation is separately timed;
feature gathering, encoding, evidence, routing and durable ordinary records are timed.
This is a qualification consumer of the existing scorer, not a scientific runner.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import ExitStack
from collections import Counter
from functools import wraps
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from exact.experiments.throughput import Measurement, PhaseCounters, freeze_workload, group_chunks, identity, stress_coverage
from exact.utils.provenance import sha256_file


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def verified(value):
    path = Path(value["path"])
    if sha256_file(path) != value["sha256"]:
        raise ValueError(f"Input binding changed: {path}")
    return path


def write(path, value):
    from exact.runs.store import _atomic_json
    _atomic_json(Path(path), value)


def prepare(args):
    panel = json.loads(args.public_inputs.read_text())
    cases = panel.get("cases", panel)
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for case, item in sorted(cases.items()):
        query_record = json.loads(verified(item["local_queries"]).read_text())
        if query_record.get("labels_exposed") is not False:
            raise ValueError("Qualification requires a gold-stripped public query binding")
        query_path = verified(query_record["outputs"]["queries"])
        queries = [json.loads(line) for line in query_path.read_text().splitlines() if line]
        workload = freeze_workload(queries, pair=case, minimum_pairs=args.minimum_pairs)
        workload["inputs"] = {"queries": binding(query_path),
                              "public_inputs": binding(args.public_inputs)}
        write(args.output / f"{case}.json", workload)
        sources = json.loads(verified(item["source_population"]).read_text())
        targets = json.loads(verified(item["target_population"]).read_text())
        pair_set = {(q["source"], t) for q in queries for t in q["candidates"]}
        inventory[case] = {
            "source_population": sources["count"], "target_population": targets["count"],
            "local_original_queries": len(queries), "local_sources": len({q["source"] for q in queries}),
            "local_candidate_occurrences": sum(len(q["candidates"]) for q in queries),
            "local_unique_pairs": len(pair_set),
            "global_candidates": {"count": None, "interval": [0, sources["count"] * targets["count"]],
                                  "unknown_dependency": "frozen fitted retrieval over full native populations"},
            "global_local_overlap": None,
            "overlap_unknown_dependency": "global retrieval not executed; local pools never substitute",
            "workload": binding(args.output / f"{case}.json"),
        }
    write(args.output / "work-counts.json", {
        "schema_version": 1, "primary_exact_logical_cells": 12,
        "label_free_control_cells": [0, 2], "published_deterministic_cells": 3,
        "bounded_stochastic_logical_cells": [36, 54], "component_additional_cells": [0, 20],
        "cases": inventory, "private_references_read": False,
        "unique_fits": {"count": None, "unknown_dependency": "corrected selection and per-seed fitting freeze"},
        "unique_texts": {"count": None, "unknown_dependency": "natural selected evidence for frozen candidates"},
        "stochastic_requests": {"count": None, "unknown_dependency": "frozen gate routes and request seed identity"},
    })


def run(args):
    # Hugging Face reads these flags at import time, before model construction.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OPENROUTER_API_KEY="")
    import resource
    import torch
    import pandas as pd
    from exact.core.entities.configs.config import ConfigModel
    from exact.impl.datasets.pair_adaptive_context import PairAdaptiveContextDataset
    from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer
    from exact.runs.store import ExplanationStore

    if args.output.exists():
        raise ValueError("Each measurement requires a new isolated output/cache directory")
    args.output.mkdir(parents=True)
    # Prevent all accidental network dispatch, including a newly added code path.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OPENROUTER_API_KEY="",
                      EXACT_EXPERIMENT_ROLE="throughput_fixture", EXACT_EXPERIMENT_SEED="17",
                      EXACT_EMBEDDING_CACHE_DIR=str(args.output / "cache/embeddings"),
                      EXACT_NUMERICAL_CACHE_ROOT=str(args.output / "cache"),
                      EXACT_EXPERIMENT_SHARED_CACHE_ROOT=str(args.output / "cache"),
                      EXACT_EVIDENCE_PREFETCH="0")
    import socket
    def no_network(*_a, **_k):
        raise RuntimeError("Network forbidden in throughput qualification")
    socket.socket.connect = no_network
    socket.create_connection = no_network
    config = ConfigModel.load_config(args.config)
    config.resolve_dependencies()
    workload = json.loads(args.workload.read_text())
    if workload["namespace"] != "qualification_only":
        raise ValueError("Qualification namespace required")
    for value in workload.get("inputs", {}).values():
        verified(value)
    namespace = "qualification-" + workload["identity"]
    os.environ["EXACT_PRIMITIVE_NAMESPACE"] = namespace
    runtime_path = args.output / "recovery-runtime.json"
    write(runtime_path, {
        "root": str(args.output / "cache"), "primitive_namespace": namespace,
        "identity": {
            "parameters": config.model_dump(mode="json"),
            "inputs": {"source": binding(config.data.source), "target": binding(config.data.target),
                       "workload": binding(args.workload)},
            "implementation": {"scorer": binding(ROOT / "exact/impl/models/pair_adaptive_scorer.py"),
                               "channels": binding(ROOT / "exact/impl/models/pair_adaptive_channels.py")},
            "dependencies": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
            "role": "qualification_fixture", "entity_kind": "class", "seed": 17,
        },
    })
    os.environ["EXACT_EXPERIMENT_RUNTIME"] = str(runtime_path)
    groups = list(group_chunks(workload["queries"], args.chunk_pairs))
    if args.max_chunks:
        groups = groups[:args.max_chunks]
    setup_start = time.perf_counter()
    dataset = PairAdaptiveContextDataset(
        output_path=args.output / "native", device=args.device,
        source_options=config.io.source_options, target_options=config.io.target_options,
        **config.dataset_params.model_dump(),
    )
    dataset.load_ontologies(config.data.source, config.data.target)
    # Reuse the exact existing templates, never regenerate them through a model.
    templates = json.loads(args.templates.read_text())
    dataset._verbalization_templates = templates.get("templates", templates)
    # Native graph creation belongs to preparation; per-entity feature extraction stays cold.
    _ = dataset.source_graph, dataset.target_graph
    dataset._candidates = pd.DataFrame(
        [(q["source"], t) for group in groups for q in group for t in q["candidates"]],
        columns=["Src", "Tgt"],
    )
    dataset.get_exact_matches()
    dataset.restrict_benchmark_exact_matches()
    exact_pairs = dataset._exact_mapping_pairs()
    native_seconds = time.perf_counter() - setup_start
    model_start = time.perf_counter()
    primary = config.get_model_sequence()[0]
    params = {**primary.params, **config.matching.channels.model_dump(mode="python"),
              "fusion_config": config.matching.fusion.model_dump(mode="python"),
              "llm_experiment_config": config.llm.experiment.model_dump(mode="python"),
              **config.alignment_params.model_dump(exclude_none=True),
              "use_llm": False, "device": args.device, "persist_cache_to_disk": False,
              "generate_llm_rationales": False, "return_explanations": True, "request_seed": 17}
    scorer = PairAdaptiveSemanticScorer(**params)
    scorer.attach_dataset(dataset)
    scorer.use_llm = bool(primary.params.get("use_llm", False))
    fixture_calls = {"brief_pairs": 0, "decision_pairs": 0}
    def briefs(sources, targets, packets):
        fixture_calls["brief_pairs"] += len(packets)
        return ["qualification fixture brief" for _ in packets]
    def judgments(sources, targets, *_):
        fixture_calls["decision_pairs"] += len(targets)
        return torch.full((len(targets),), 0.5, device=args.device)
    scorer.generate_pair_briefs_batched = briefs
    scorer.llm_yesno_probs_batched = judgments
    # Complex hosted interfaces require compatible recorded fixtures, not invented group choices.
    if scorer.llm_experiment_config["decision"]["mode"] != "binary" or scorer.llm_experiment_config.get("exemplars") == "knn":
        raise ValueError("This path requires a recorded compatible group/exemplar response fixture")
    model_seconds = time.perf_counter() - model_start
    from exact.experiments import numerical_cache
    from exact.experiments.encoder_store import encoder_cache_stats
    from exact.impl.models import pair_adaptive_batch
    encoded_texts = set()
    encoding = {"calls": 0, "rows": 0, "tokens": 0, "batch_sizes": Counter(), "counter_seconds": 0.0}
    original_encode = scorer._encode_texts
    @wraps(original_encode)
    def counted_encode(tokenizer, model, texts, max_len):
        started = time.perf_counter()
        encoded_texts.update((id(model), max_len, text) for text in texts)
        encoding["counter_seconds"] += time.perf_counter() - started
        return original_encode(tokenizer, model, texts, max_len)
    scorer._encode_texts = counted_encode
    routing = {"would_route": 0, "pairs": 0}
    original_gate = scorer._llm_gate_mask
    @wraps(original_gate)
    def counted_gate(**kwargs):
        mask, rows = original_gate(**kwargs)
        routing["would_route"] += int(mask.detach().cpu().sum())
        routing["pairs"] += mask.numel()
        return mask, rows
    scorer._llm_gate_mask = counted_gate
    def encoded_batch(_model, _args, kwargs):
        started = time.perf_counter()
        ids = kwargs["input_ids"]
        rows = len(ids)
        encoding["calls"] += 1
        encoding["rows"] += rows
        encoding["batch_sizes"][rows] += 1
        encoding["tokens"] += int(kwargs["attention_mask"].detach().cpu().sum())
        encoding["counter_seconds"] += time.perf_counter() - started
    hooks = [model.register_forward_pre_hook(encoded_batch, with_kwargs=True)
             for model in (getattr(scorer, "lex_model", None), getattr(scorer, "ctx_model", None))
             if model is not None]
    methods = {"_experiment_entity_features": "feature_gathering", "_encode_label_matrix": "raw_matrices",
               "_encode_context_matrix": "raw_matrices", "_object_support_matrix": "raw_matrices",
               "_select_diverse_indices": "evidence_selection",
               "_context_similarity_from_sentences": "combined_context_encoding",
               "_uncertainty_components": "fusion_uncertainty", "_llm_gate_mask": "routing",
               "encode_labels_batch": "encoder_labels", "encode_contexts_batch": "encoder_contexts"}
    receipts = {}
    for mode in ("cold", "warm", "replay"):
        phases = PhaseCounters()
        encoded_texts.clear()
        encoding.update(calls=0, rows=0, tokens=0, batch_sizes=Counter(), counter_seconds=0.0)
        routing.update(would_route=0, pairs=0)
        if mode == "warm":
            # Warm vectors, cold numerical primitives; replay is a separate pass.
            cache = getattr(scorer, "_numerical_cache", None)
            if cache is not None:
                cache.close()
                del scorer._numerical_cache
            os.environ["EXACT_NUMERICAL_CACHE_ROOT"] = str(args.output / "warm-pair-cache")
        store = ExplanationStore(args.output / mode / "ordinary-evidence")
        measure = Measurement(mode=mode, workload=workload, device=args.device)
        for number, group in enumerate(groups):
            pairs = [(q["source"], t) for q in group for t in q["candidates"]
                     if (q["source"], t) not in exact_pairs]
            sources = [s for s, _ in pairs]
            targets = [t for _, t in pairs]
            if not targets:
                continue
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            # Each original local query owns its complete pool. Repeated-source
            # queries are separate calls, never the union of other query pools.
            dataset._df = pd.DataFrame({"Src": sources, "Tgt": targets,
                                       dataset.default_kind: [True] * len(sources)})
            dataset._invalidate_active_dataframe_cache()
            with ExitStack() as instrumentation:
                instrumentation.enter_context(phases.instrument(scorer, methods))
                instrumentation.enter_context(phases.instrument(pair_adaptive_batch, {
                    "_batch_matrices": "raw_matrices_batched", "_batch_context_similarity": "combined_context_encoding_batched"}))
                instrumentation.enter_context(phases.instrument(numerical_cache, {"_scope": "numerical_identity"}))
                source_labels = [dataset.source_graph.get_labels(s) for s in sources]
                target_labels = [dataset.target_graph.get_labels(t) for t in targets]
                result = scorer(src_iris=sources, tgt_iris=targets,
                                src_label_lists=source_labels, tgt_label_lists=target_labels)
            with phases.phase("durable_output"):
                records = result["explanations"]
                if len(records) != len(sources) or tuple(result["S_final"].shape) != (len(sources),):
                    raise ValueError("Incomplete numerical result cannot count as completed pairs")
                if not bool(torch.isfinite(result["S_final"]).all()):
                    raise ValueError("Nonfinite numerical result cannot count as completed pairs")
                query_ids = {q["source"]: q["qid"] for q in group}
                for row, s, t in zip(records, sources, targets):
                    row.update(src_iri=s, tgt_iri=t, original_query_id=query_ids[s], qualification_fixture=True)
                store.append(records)
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            measure.commit_chunk(elapsed=elapsed,
                computed=[(s, t, workload["identity"]) for s, t in zip(sources, targets)] if mode == "cold" else [],
                available=len(sources) if mode != "cold" else 0, durable=True,
                rows=len(sources), source_group_sizes=[len(q["candidates"]) for q in group],
                output_bytes=store.stored_bytes,
                exact_prefiltered_rows=sum((q["source"], t) in exact_pairs for q in group for t in q["candidates"]),
                scores=result["S_final"].detach().cpu().tolist(),
                stress_coverage=stress_coverage(records, [len(q["candidates"]) for q in group]),
                query_ids=[q["qid"] for q in group])
            write(args.output / f"{mode}.json", {**measure.receipt(), "phases": phases.receipt(),
                "encoder": {**encoding, "distinct_texts": len(encoded_texts)},
                "routing": {**routing, "fraction": routing["would_route"] / routing["pairs"] if routing["pairs"] else None,
                            "hosted_queue_backlog": 0, "scope": "offline_fixture_only"},
                "numerical_scope_rebuilds_cumulative": getattr(scorer, "_numerical_scope_rebuilds", 0),
                "encoder_cache_cumulative": encoder_cache_stats(),
                "numerical_cache_cumulative": scorer._numerical_cache.stats() if hasattr(scorer, "_numerical_cache") else None})
            print(json.dumps({"mode": mode, "chunk": number, "rows": len(sources), "seconds": elapsed}), flush=True)
        receipts[mode] = {**measure.receipt(), "phases": phases.receipt(),
                          "encoder": {**encoding, "distinct_texts": len(encoded_texts)}}
    write(args.output / "measurement.json", {
        "schema_version": 1, "namespace": "qualification_only", "paid_calls": 0,
        "fixture_calls": fixture_calls, "cold": receipts["cold"], "warm": receipts["warm"],
        "replay": receipts["replay"],
        "native_preparation_seconds": native_seconds, "model_loading_seconds": model_seconds,
        "phases": phases.receipt(), "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "torch": torch.__version__, "host": platform.node(), "slurm_job": os.getenv("SLURM_JOB_ID"),
        "allocated_cpus": os.getenv("SLURM_CPUS_PER_TASK"),
        "python": sys.version, "platform": platform.platform(),
        "precision": {"fp16": scorer.fp16, "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
        "slurm_step": os.getenv("SLURM_STEP_ID"), "config": binding(args.config),
        "workload": binding(args.workload), "templates": binding(args.templates),
        "source_revision": ((ROOT / "source-revision.txt").read_text().strip()
                            if (ROOT / "source-revision.txt").is_file()
                            else subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()),
        "hosted_service_time": None, "end_to_end_scientific_time": None,
        "fixture_outputs_promotable": False,
        "cache_root": str(args.output / "cache"),
        "execution_flags": {key: os.getenv(key) for key in ("EXACT_PAIR_CONTEXT_BATCHING", "EXACT_PAIR_CONTEXT_BLOCK_PAIRS", "EXACT_PAIR_CONTEXT_TEXT_BATCH")},
        "instrumentation": "opt-in inclusive host timers and encoder counters; measured counter_seconds included in wall time",
    })
    for hook in hooks:
        hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--public-inputs", type=Path, required=True)
    prep.add_argument("--minimum-pairs", type=int, default=5000)
    prep.add_argument("--output", type=Path, required=True)
    measure = sub.add_parser("run")
    for field in ("config", "workload", "templates", "output"):
        measure.add_argument("--" + field, type=Path, required=True)
    measure.add_argument("--device", default="cuda")
    measure.add_argument("--chunk-pairs", type=int, default=512)
    measure.add_argument("--max-chunks", type=int)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
