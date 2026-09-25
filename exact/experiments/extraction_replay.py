"""Replay an E01 extraction treatment from a bound development score packet.

No ontology, embedding model, or hosted model is constructed. The ordinary
campaign runner still owns admission, subprocess measurement, recovery and
selection. Only the explicitly amended extraction/cardinality controls vary.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast

import pandas as pd
import zstandard as zstd

from exact.core.entities.configs.yaml_io import load_yaml_mapping
from exact.core.entities.mappings import EntityMapping
from exact.runs.decisions import observe_extraction
from exact.utils.provenance import sha256_file

_MAX_FILE_BYTES = 128 * 1024 * 1024
_DOWNSTREAM = {"threshold", "cardinality", "extraction", "relation_typing", "repair"}
_DATA_FILES = (
    "dataset/dataset.csv",
    "dataset/dataset.meta.json",
    "dataset/candidate_pool_manifest.json",
    "dataset/candidate_pool_sample_manifest.json",
    "dataset/sampled_inputs/full_reference.tsv",
    "dataset/sampled_inputs/training_reference.tsv",
)


def _json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError(f"Replay input exceeds the bounded file limit: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Replay artifact must be an object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def scoring_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the declared extraction treatment and output-only settings."""
    result = copy.deepcopy(dict(config))
    result.pop("output", None)
    for key in ("extraction", "cardinality", "target_cardinality"):
        result.get("matching", {}).pop(key, None)
    for key in ("logging_level", "use_file_cache"):
        result.get("run", {}).pop(key, None)
    return result


def _bound(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError(f"Missing or oversized replay input: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _verified(binding: dict[str, Any]) -> Path:
    path = Path(binding["path"])
    if _bound(path) != binding:
        raise ValueError(f"Frozen replay input changed: {path}")
    return path


def _scope(config: Mapping[str, Any]) -> None:
    data, matching = config["data"], config["matching"]
    if data.get("reference_role") not in {"valid", "validation", "development", "dev"}:
        raise ValueError("Extraction replay is restricted to development references")
    if set(data.get("refs", {})) - {
        "train",
        "training",
        "valid",
        "validation",
        "dev",
        "development",
    }:
        raise ValueError("Extraction replay refuses private/full/test reference bindings")
    if data.get("execution_mode") != "global_alignment":
        raise ValueError("Extraction replay requires global alignment")
    if matching.get("relation_prediction", "none") != "none":
        raise ValueError("Extraction replay cannot change relation typing")


def build_packet(
    source_run: Path,
    destination: Path,
    *,
    evidence_run: Path | None = None,
    expected_sources: int = 300,
    expected_scored_pairs: int = 5842,
    expected_protected_pairs: int = 158,
) -> dict[str, Any]:
    """Bind a completed scoring checkpoint, even if its extraction failed."""
    source_run = source_run.resolve()
    evidence_run = (evidence_run or source_run).resolve()
    config = load_yaml_mapping(source_run / "_inputs/resolved.config.yaml")
    _scope(config)
    metadata_paths = sorted((source_run / "checkpoints").glob("*_additional_models_*.json"))
    if len(metadata_paths) != 1:
        raise ValueError("Replay needs exactly one completed additional-models checkpoint")
    metadata_path = metadata_paths[0]
    metadata = _json(metadata_path)
    if (
        metadata.get("complete") is not True
        or metadata.get("candidate_records_count") != expected_scored_pairs
    ):
        raise ValueError("Replay scoring checkpoint is incomplete or has an unexpected population")
    score_path = metadata_path.with_suffix(".jsonl.zst")
    evidence_score = evidence_run / "checkpoints" / score_path.name
    if sha256_file(score_path) != sha256_file(evidence_score):
        raise ValueError("Replay score and explanation populations differ")
    index_path = evidence_run / "explanations/index.json"
    index = _json(index_path)
    if index.get("total_records") != expected_scored_pairs or index.get("overlays"):
        raise ValueError("Replay requires a complete compacted explanation index")
    files = {
        "config": _bound(source_run / "_inputs/resolved.config.yaml"),
        "source_manifest": _bound(source_run / "experiment_manifest.json"),
        "score_metadata": _bound(metadata_path),
        "scores": _bound(score_path),
        "evidence_config": _bound(evidence_run / "_inputs/resolved.config.yaml"),
        "evidence_manifest": _bound(evidence_run / "experiment_manifest.json"),
        "upstream_stats": _bound(evidence_run / "stats/run_stats.json"),
        "explanation_index": _bound(index_path),
        **{name: _bound(evidence_run / name) for name in _DATA_FILES},
    }
    for key, shard in index["shards"].items():
        path = (index_path.parent / shard["path"]).resolve()
        path.relative_to(index_path.parent)
        files[f"explanation_shard/{key}"] = _bound(path)
    universe = config["data"].get("source_universe")
    if not universe:
        raise ValueError("Replay needs the original declared development source universe")
    files["source_universe"] = _bound(Path(config["data"].get("root") or ".") / universe)
    packet = {
        "schema_version": 1,
        "kind": "e01_development_extraction_replay",
        "files": files,
        "expected_sources": expected_sources,
        "expected_scored_pairs": expected_scored_pairs,
        "expected_protected_pairs": expected_protected_pairs,
    }
    validate_packet(packet, config)
    if destination.exists():
        raise FileExistsError(f"Replay packets are immutable: {destination}")
    _write(destination, packet)
    return packet


def validate_packet(packet: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Path]:
    if (
        packet.get("schema_version") != 1
        or packet.get("kind") != "e01_development_extraction_replay"
    ):
        raise ValueError("Unsupported extraction replay packet")
    paths = {key: _verified(value) for key, value in packet["files"].items()}
    _scope(config)
    baseline = load_yaml_mapping(paths["config"])
    evidence = load_yaml_mapping(paths["evidence_config"])
    _scope(baseline)
    _scope(evidence)
    if any(
        scoring_configuration(value) != scoring_configuration(baseline)
        for value in (config, evidence)
    ):
        raise ValueError("Extraction replay refused a changed scoring configuration")
    manifests = [_json(paths[name]) for name in ("source_manifest", "evidence_manifest")]
    for manifest in manifests:
        if manifest.get("experiment_id") != "E01" or manifest.get("split_role") != "development":
            raise ValueError("Replay source is not an E01 development cell")
        if manifest.get("source_cap") != packet["expected_sources"]:
            raise ValueError("Replay source population does not match its declared cap")
    if manifests[0].get("runtime_versions") != manifests[1].get("runtime_versions"):
        raise ValueError("Replay source native/runtime identities differ")
    if manifests[1].get("status") != "complete":
        raise ValueError("Replay evidence source did not complete")
    pool = _json(paths["dataset/candidate_pool_sample_manifest.json"])
    expected = (packet["expected_sources"], packet["expected_scored_pairs"])
    summary = pool["gold_free_summary"]
    if (summary["source_entities"], summary["candidate_pairs"]) != expected:
        raise ValueError("Replay candidate pool population differs")
    return paths


def _frames(paths: dict[str, Path], packet: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    with zstd.open(paths["scores"], "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if len(rows) >= packet["expected_scored_pairs"]:
                raise ValueError("Replay score checkpoint contains too many rows")
            row.pop("ground_truth", None)
            row.pop("Label", None)
            for key in ("saved_alignment_member", "threshold_positive", "extraction_selected"):
                row.pop(key, None)
            decision = row.get("candidate_decision") or {}
            decision["events"] = [
                event for event in decision.get("events", []) if event["stage"] not in _DOWNSTREAM
            ]
            row["candidate_decision"] = decision
            rows.append(row)
    if len(rows) != packet["expected_scored_pairs"]:
        raise ValueError("Replay score checkpoint is truncated")
    scores = pd.DataFrame(rows)
    if scores[["Src", "Tgt"]].duplicated().any() or not scores.S_final.map(math.isfinite).all():
        raise ValueError("Replay score rows must have unique pairs and finite unchanged scores")
    dataset = pd.read_csv(
        paths["dataset/dataset.csv"],
        usecols=lambda name: name in {"Src", "Tgt", "SrcKind", "TgtKind", "prefiltered", "Scores"}
        or name.startswith("cand_"),
    )
    exact = dataset.loc[dataset.prefiltered.fillna(False)].copy()
    if len(exact) != packet["expected_protected_pairs"] or not (exact.Scores == 1.0).all():
        raise ValueError("Replay protected lexical inventory changed")
    if set(map(tuple, scores[["Src", "Tgt"]].values)) & set(
        map(tuple, exact[["Src", "Tgt"]].values)
    ):
        raise ValueError("Replay scored and protected populations overlap")
    universe = set(paths["source_universe"].read_text().splitlines())
    sources = set(dataset.Src.astype(str))
    if len(sources) != packet["expected_sources"] or not sources <= universe:
        raise ValueError("Replay source sample changed")
    if set(
        map(tuple, dataset.loc[~dataset.prefiltered.fillna(False), ["Src", "Tgt"]].values)
    ) != set(map(tuple, scores[["Src", "Tgt"]].values)):
        raise ValueError("Replay scores and saved candidate dataset differ")
    return scores, dataset


def _explanations(paths: dict[str, Path]):
    # Reading the mutable store may recover/truncate old uncommitted tails.
    # Consume only the explicitly bound compacted shards without opening it.
    for name, path in sorted(paths.items()):
        if name.startswith("explanation_shard/"):
            with zstd.open(path, "rt") as stream:
                for line in stream:
                    yield json.loads(line)


def run_replay(wrapper: Path, packet_path: Path) -> None:
    from exact.core.actions.evaluation import run_evaluation
    from exact.impl.extraction import extract_global_alignment
    from exact.impl.trainer.audit_io import AuditIOMixin
    from exact.impl.trainer.overlays import OverlaysMixin
    from exact.impl.trainer.runner import SemanticAlignmentRunner
    from exact.io.writers import write
    from exact.runs.layout import RunLayout
    from exact.runs.manifest import refresh_manifest
    from exact.runs.store import ExplanationStore

    started = time.perf_counter()
    job = load_yaml_mapping(wrapper)["job"]
    config = load_yaml_mapping(Path(job["config_file"]))
    packet = _json(packet_path)
    paths = validate_packet(packet, config)
    matching = config["matching"]
    extraction = matching["extraction"]
    if extraction.get("anchor_conflict_policy") != "compete":
        raise ValueError("Amended E01 replay requires the explicit compete anchor policy")
    if any(
        component.get("params", {}).get("generate_llm_rationales")
        for component in config.get("pipeline", [])
    ):
        raise ValueError("Extraction replay never generates rationales")
    scores, dataset = _frames(paths, packet)
    layout = RunLayout.create(Path(job["output_dir"]))
    if layout.mapping_path("global").exists():
        raise FileExistsError("Replay output already contains a final alignment")
    for name in _DATA_FILES:
        destination = layout.root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths[name], destination)
    exact = dataset.loc[dataset.prefiltered.fillna(False)].copy()
    exact["S_final"] = exact.Scores
    protected = set(map(tuple, exact[["Src", "Tgt"]].values))
    frame = pd.concat([scores, exact], ignore_index=True)
    mappings = EntityMapping.read_table_mappings(
        frame.rename(columns={"S_final": "Score"})[["Src", "Tgt", "Score", "SrcKind", "TgtKind"]]
    )
    extraction_started = time.perf_counter()
    result = extract_global_alignment(
        mappings,
        mode=extraction["mode"],
        threshold=matching["threshold"],
        protected_pairs=protected,
        source_cardinality=matching.get("cardinality"),
        target_cardinality=matching.get("target_cardinality"),
        assignment_component_cap=extraction.get("assignment_component_cap", 500),
        anchor_conflict_policy="compete",
    )
    extraction_seconds = time.perf_counter() - extraction_started
    policy = {
        "threshold": matching["threshold"],
        "threshold_origin": "configured",
        "source_cardinality": matching.get("cardinality"),
        "target_cardinality": matching.get("target_cardinality"),
        "local_alignment": False,
        "relation_prediction": "none",
        "extraction": extraction,
    }
    observe_extraction(
        frame,
        result.mappings,
        threshold=matching["threshold"],
        config=policy,
        implementation="exact.impl.extraction.extract_global_alignment",
    )
    typed = pd.DataFrame(
        [
            {
                "SrcEntity": item.head,
                "TgtEntity": item.tail,
                "Score": item.score,
                "Relation": "=",
                "SrcKind": "class",
                "TgtKind": "class",
            }
            for item in result.mappings
        ],
        columns=["SrcEntity", "TgtEntity", "Score", "Relation", "SrcKind", "TgtKind"],
    )
    layout.alignment_dir.mkdir(parents=True, exist_ok=True)
    write("tsv-global", typed, layout.alignment_dir, filename="maps_global.tsv")
    typed.to_csv(layout.alignment_dir / "paper.maps_global.tsv", sep="\t", index=False)
    facade = SimpleNamespace(
        _final_candidate_frame=frame,
        dataset=SimpleNamespace(
            dataframe=dataset,
            eligible_source_iris=sorted(set(dataset.Src)),
            dataset_signature=_json(paths["dataset/dataset.meta.json"])["dataset_signature"],
            candidate_pool_manifest=_json(paths["dataset/candidate_pool_sample_manifest.json"]),
        ),
        model=SimpleNamespace(),
        results_json=[],
        output_dir=layout.root,
        _decision_policy=policy,
        _extraction_diagnostics=result.diagnostics,
        _json_safe_value=SemanticAlignmentRunner._json_safe_value,
    )
    AuditIOMixin._write_source_decisions(cast(Any, facade), result.mappings, typed)
    selected = {(item.head, item.tail) for item in result.mappings}
    decisions = {
        (row["Src"], row["Tgt"]): row["candidate_decision"] for row in frame.to_dict("records")
    }
    records = []
    numeric_fields = {
        "src_iri",
        "tgt_iri",
        "src_kind",
        "tgt_kind",
        "kind",
        "confidences",
        "qualities",
        "weights",
        "importances",
        "contributions",
        "reconstruction",
    }
    score_lookup = dict(zip(zip(scores.Src, scores.Tgt), scores.S_final))
    reconstruction_errors = []
    packet_hash = sha256_file(packet_path)
    seen = set()
    for original in _explanations(paths):
        pair = (original["src_iri"], original["tgt_iri"])
        if pair not in score_lookup or original["confidences"]["S_final"] != score_lookup[pair]:
            raise ValueError("Replay explanation scores disagree with frozen checkpoint")
        seen.add(pair)
        record = {key: value for key, value in original.items() if key in numeric_fields}
        record["candidate_decision"] = decisions[pair]
        record["replay_provenance"] = {
            "packet_sha256": packet_hash,
            "detail": "frozen_numeric_evidence",
        }
        reconstructed = sum(
            record["contributions"].get(name, 0.0)
            for name in ("C_label", "C_strsim", "C_struct", "C_llm", "C_oracle")
        )
        reconstruction_errors.append(
            abs(
                reconstructed
                - (record["reconstruction"]["score"] - record["reconstruction"]["baseline"])
            )
        )
        records.append(record)
    if len(records) != len(scores) or seen != set(score_lookup):
        raise ValueError("Replay explanation population changed")
    if any(not math.isfinite(value) or value > 1e-6 for value in reconstruction_errors):
        raise ValueError("Frozen numeric explanation reconstruction failed")
    facade.results_json = records
    OverlaysMixin._annotate_final_prediction_records(
        cast(Any, facade), result.mappings, matching["threshold"], False
    )
    ExplanationStore(layout.explanations_dir).append(records)
    frame["saved_alignment_member"] = [
        (source, target) in selected for source, target in zip(frame.Src, frame.Tgt)
    ]
    layout.stats_dir.mkdir(parents=True, exist_ok=True)
    frame[["Src", "Tgt", "S_final", "saved_alignment_member"]].to_csv(
        layout.summary_metrics_path, sep="\t", index=False
    )
    upstream = _json(paths["upstream_stats"])
    # Only immutable upstream population measurements survive. Decision, timing,
    # model-usage and execution measurements are regenerated for this replay.
    stats = {
        key: upstream[key]
        for key in (
            "source_sampling",
            "candidate_recall",
            "candidate_recall_after_exact",
            "candidate_recall_diagnostics",
            "mean_pool_size",
            "gold_rank_p90",
            "gold_rank_median",
            "ontology_stack",
        )
        if key in upstream
    }
    accepted_sources = len({item.head for item in result.mappings})
    stats.update(
        n_mappings=len(result.mappings),
        n_llm=0,
        frac_llm=0.0,
        coverage=accepted_sources / packet["expected_sources"],
        abstention_rate=1 - accepted_sources / packet["expected_sources"],
        decision_source_counts={
            "declared": packet["expected_sources"],
            "observed": len(set(dataset.Src)),
            "accepted": accepted_sources,
            "unscored": 0,
        },
        extraction=result.diagnostics,
        observed_execution={
            "device": "cpu",
            "device_type": "cpu",
            "mode": "extraction_only_replay",
        },
        metric_applicability={"coverage": True, "abstention_rate": True},
        explanation_reconstruction={
            "status": "complete",
            "result_rows": len(records),
            "reconstructed_rows": len(records),
            "score_stage": "pair_pre_selector",
            "tolerance": 1e-6,
            "failed_rows": sum(value > 1e-6 for value in reconstruction_errors),
            "max_abs_error": max(reconstruction_errors, default=0.0),
        },
        extraction_replay={
            "packet": _bound(packet_path),
            "new_scoring_calls": 0,
            "new_hosted_requests": 0,
            "new_hosted_tokens": 0,
            "upstream_measurements": paths["upstream_stats"].as_posix(),
        },
    )
    _write(layout.run_stats_path, stats)
    evaluation = config.get("evaluation", {})
    run_evaluation(
        layout.mapping_path("global"),
        layout.evaluation_dir,
        error_on_fail=True,
        K=evaluation.get("k"),
        backends=evaluation.get("backends"),
        backend_options={"bioml": evaluation.get("bioml") or {}},
        train_reference_file_path=layout.root / "dataset/sampled_inputs/training_reference.tsv",
        full_reference_file_path=layout.root / "dataset/sampled_inputs/full_reference.tsv",
        run_stats_path=layout.run_stats_path,
    )
    _write(
        layout.timings_path,
        {
            "schema_version": 1,
            "execution_scope": "extraction_only_replay",
            "upstream_reused": True,
            "sessions": [
                {
                    "stages": [
                        {
                            "stage": "Extraction.Replay",
                            "seconds": extraction_seconds,
                            "cache_status": "fresh",
                        },
                        {
                            "stage": "Extraction.ReplayTotal",
                            "seconds": time.perf_counter() - started,
                            "cache_status": "fresh",
                        },
                    ]
                }
            ],
        },
    )
    refresh_manifest(layout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    args = parser.parse_args()
    run_replay(args.run_config, args.packet)


if __name__ == "__main__":
    main()
