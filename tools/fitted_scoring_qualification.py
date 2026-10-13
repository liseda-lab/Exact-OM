"""Qualification-only adapter for an already fitted, stopped G4 global run.

Reads fitted heads and label-free candidate columns, never training/reference inputs.
Synthetic hosted replies exercise the selected numerical recipe; they are not results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from exact.experiments.throughput import freeze_workload, identity
from exact.runs.store import _atomic_json
from exact.utils.provenance import sha256_file

CANDIDATE_COLUMNS = (
    "Src",
    "Tgt",
    "SrcKind",
    "TgtKind",
    "cand_sim",
    "cand_sim_semantic",
    "cand_sim_lexical",
    "cand_channels",
    "cand_sim_retrieval",
    "cand_sim_cross_encoder",
    "cand_sim_src_mean",
    "cand_sim_prob",
    "cand_share_top",
    "cand_share_rest",
    "cand_share_log_ratio",
)

# Same public metadata families merged by Trainer._merge_dataset_candidate_columns.
CANDIDATE_FEATURE_PREFIXES = (
    "cand_",
    "src_hier",
    "tgt_hier",
    "src_obj",
    "tgt_obj",
    "src_attr",
    "tgt_attr",
    "src_lab",
    "tgt_lab",
)


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def verify(entry):
    path = Path(entry["path"])
    if bind(path) != entry:
        raise ValueError(f"Changed fitted qualification dependency: {path}")
    return path


def candidate_frame(path):
    """Do not materialize Label, stored features, or historical model scores."""
    import pandas as pd

    frame = pd.read_csv(
        path,
        usecols=lambda name: name in CANDIDATE_COLUMNS
        or name.startswith(CANDIDATE_FEATURE_PREFIXES),
    )
    if not {"Src", "Tgt", "cand_sim"} <= set(frame):
        raise ValueError("Saved candidates lack required inference metadata")
    if frame[["Src", "Tgt"]].isna().any().any() or frame.duplicated(["Src", "Tgt"]).any():
        raise ValueError("Missing or duplicate candidate identity")
    return frame


def records_without_nan(frame):
    # pandas.to_json defaults to ten-digit rounding, which can change close ranks.
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def complete_queries(frame, index):
    """Exclude the final incomplete source rather than truncating its candidate pool."""
    queries = []
    for source, group in frame.groupby("Src", sort=False):
        count = int(index["sources"].get(source, {}).get("records", 0))
        if count > len(group):
            raise ValueError("Explanation index exceeds its saved candidate group")
        if count == len(group):
            queries.append(
                {
                    "qid": identity(["G4-fitted-global", source]),
                    "source": source,
                    "candidates": group.Tgt.tolist(),
                }
            )
    if not queries:
        raise ValueError("No complete checkpointed candidate groups")
    return queries


def prepare(args):
    from exact.core.entities.configs.config import ConfigModel

    run = args.fitted_run.resolve()
    if args.minimum_pairs < 1:
        raise ValueError("Qualification minimum pairs must be positive")
    if args.output.resolve().is_relative_to(run):
        raise ValueError("Qualification must not write inside the scientific run")
    checkpoint_path = args.checkpoint or max(
        (run / "checkpoints").glob("inference_*.json"), key=lambda p: p.stat().st_mtime_ns
    )
    checkpoint = json.loads(checkpoint_path.read_text())
    config_path = run / "config.yaml"
    config = ConfigModel.load_config(config_path)
    mapping = config.model_dump(mode="json", by_alias=True)
    if config.data.execution_mode != "global_alignment":
        raise ValueError("Fitted qualification currently supports the selected global G4 recipe")
    models = checkpoint["checkpoint_fingerprint_payload"]["models"]
    if len(models) != 1 or models[0]["class"] != "PairAdaptiveSemanticScorer":
        raise ValueError("Unsupported checkpoint model chain")
    experiments = models[0]["payload"]["pair_adaptive_channels"]["experiments"]
    fusion = experiments["fusion_artifact"]
    if bind(fusion["path"])["sha256"] != fusion["sha256"]:
        raise ValueError("Checkpoint-bound fusion head changed")
    fit_root = Path(fusion["path"]).parent
    heads = {
        "fusion": bind(fit_root / "fusion.json"),
        "selector": bind(fit_root / "selector.json"),
        "calibration": bind(run / "fitting/score_calibrator.json"),
        "retrieval": bind(run / "fitting/cross_encoder/retrieval_artifact.json"),
    }
    selector = json.loads(verify(heads["selector"]).read_text())
    if (
        selector.get("kind") != "fitted_selector"
        or selector["fit_provenance"]["application"]["dataset_signature"]
        != checkpoint["dataset_signature"]
    ):
        raise ValueError("Fitted selector belongs to another ontology pair")
    if (
        config.matching.fusion.mode != "analytic_fitted"
        or config.matching.calibration.mode != "platt"
        or config.matching.nil.mode != "off"
        or config.matching.channels.graph.mode != "off"
        or config.matching.relation_prediction != "none"
        or config.llm.experiment.decision.mode != "binary"
        or config.llm.experiment.exemplars != "off"
        or config.llm.experiment.distill != "off"
    ):
        raise ValueError("Unsupported fitted recipe; do not silently approximate additional heads")
    data_path = run / "dataset/dataset.csv"
    index_path = run / "explanations/index.json"
    frame = candidate_frame(data_path)
    index = json.loads(index_path.read_text())
    if (
        index["total_records"] != checkpoint["processed_examples"]
        or index["total_records"] != checkpoint["explanation_records_count"]
    ):
        raise ValueError("Explanation index does not match the durable inference checkpoint")
    queries = complete_queries(frame, index)
    workload = freeze_workload(queries, pair="G4-core-D0-fitted", minimum_pairs=args.minimum_pairs)
    sources = {q["source"] for q in workload["queries"]}
    workload["queries"] = [q for q in queries if q["source"] in sources]
    workload["selection"] = "fixed_source_hash_complete_groups_in_original_source_order"
    workload.pop("identity")
    workload["identity"] = identity(workload)
    if sources & set(selector["fit_provenance"]["training_sources"]):
        raise ValueError("Qualification sources overlap selector training sources")
    calibration = json.loads(verify(heads["calibration"]).read_text())
    calibration_provenance = calibration["fit_provenance"]
    if (
        calibration.get("kind") != "score_calibrator"
        or calibration_provenance["application"]["dataset_signature"]
        != checkpoint["dataset_signature"]
        or sources & set(calibration_provenance["training_sources"])
    ):
        raise ValueError("Calibration belongs to another pair or overlaps qualification sources")
    selected = frame[frame.Src.isin(sources)]
    if workload["unique_candidate_pairs"] < args.minimum_pairs:
        raise ValueError("Not enough complete checkpointed pairs for the requested panel")
    mapping["matching"]["fusion"]["artifact"] = heads["fusion"]["path"]
    mapping["matching"]["calibration"]["artifact"] = heads["calibration"]["path"]
    mapping["selector"]["rerank"]["artifact"] = heads["selector"]["path"]
    mapping["candidates"]["cross_encoder"]["artifact"] = heads["retrieval"]["path"]
    for name in ("cross_encoder", "encoder_finetune"):
        mapping["candidates"][name]["training"] = None
    # This config is consumed only by the qualification harness, never Trainer.
    for name in ("track", "task", "descriptor", "train_candidates", "reference_role", "candidates"):
        mapping["data"][name] = None
    mapping["data"]["refs"] = {}
    mapping["supervision"]["transfer_artifact"] = None
    mapping["supervision"]["inference_artifact"] = None
    for row in mapping["pipeline"]:
        if any(value for name, value in row["params"].items() if "reference" in name):
            raise ValueError("Unexpected reference-dependent primary/selector parameter")
        row["params"].pop("training_reference_file_path", None)
        row["params"]["persist_cache_to_disk"] = False
    pool_path = run / "dataset/candidate_pool_manifest.json"
    retrieval_digest = json.loads(pool_path.read_text())["models"]["cross_encoder"]["sha256"]
    if not isinstance(retrieval_digest, str) or len(retrieval_digest) != 64:
        raise ValueError("Saved pool lacks a fitted retrieval weight digest")
    dependencies = {
        "selected_config": bind(config_path),
        "checkpoint": bind(checkpoint_path),
        "candidates": bind(data_path),
        "explanation_index": bind(index_path),
        "pool_manifest": bind(pool_path),
        "templates": bind(run / "dataset/verbalization_templates.json"),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    _atomic_json(args.output / "config.json", mapping)
    # JSON converts missing optional candidate features to null, never a NaN receipt.
    _atomic_json(args.output / "candidates.json", records_without_nan(selected))
    workload["inputs"] = {"candidates": bind(args.output / "candidates.json")}
    _atomic_json(args.output / "workload.json", workload)
    manifest = {
        "schema_version": 1,
        "kind": "fitted_g4_qualification",
        "namespace": "qualification_only",
        "dataset_signature": checkpoint["dataset_signature"],
        "dependencies": dependencies,
        "heads": heads,
        "config": bind(args.output / "config.json"),
        "workload": bind(args.output / "workload.json"),
        "candidate_metadata": bind(args.output / "candidates.json"),
        "checkpoint_processed_pairs": checkpoint["processed_examples"],
        "expected_fusion": experiments["fusion_effective"],
        "expected_retrieval_sha256": retrieval_digest,
        "hosted_responses": "deterministic_synthetic_interface_fixtures",
        "raw_training_or_reference_inputs_read": False,
        "candidate_labels_materialized": False,
        "promotable_scientific_result": False,
        "selector_scope": "complete bounded cohort; not whole-G4 mapping reproduction",
        "retrieval_scope": "reuse saved candidates; validate fitted weights, no retrieval or training",
        "fit_binding_evidence": {
            "fusion": "saved primary checkpoint digest and effective parameters",
            "retrieval": "saved candidate-pool model digest",
            "selector": "sibling of checkpoint-bound fusion; saved recipe and application provenance",
            "calibration": "existing fitting/score_calibrator.json; saved application provenance",
            "limitation": "old checkpoint predates selector/calibrator deployment receipts; no historical output parity claim",
        },
    }
    _atomic_json(args.output / "fitted-manifest.json", manifest)
    return manifest


def load(path, *, config, workload, templates):
    """Validate immutable input bindings before model/native construction."""
    manifest = json.loads(Path(path).read_text())
    if (
        manifest.get("kind") != "fitted_g4_qualification"
        or manifest.get("namespace") != "qualification_only"
    ):
        raise ValueError("Expected fitted qualification manifest")
    for key, actual in (("config", config), ("workload", workload)):
        if manifest[key] != bind(actual):
            raise ValueError(f"Qualification {key} differs from the admitted fitted panel")
    if manifest["dependencies"]["templates"] != bind(templates):
        raise ValueError("Qualification templates differ")
    for group in (manifest["dependencies"], manifest["heads"]):
        for entry in group.values():
            verify(entry)
    verify(manifest["candidate_metadata"])
    return manifest


def validate_primary(scorer, manifest, diagnostics):
    """Only the explicitly requested synthetic-diagnostic omission may differ."""
    checkpoint = json.loads(Path(manifest["dependencies"]["checkpoint"]["path"]).read_text())
    expected = checkpoint["checkpoint_fingerprint_payload"]["models"][0]["payload"]
    if diagnostics == "natural":
        expected["pair_adaptive_channels"]["experiments"]["diff"][
            "controlled_perturbations"
        ] = False
    actual = scorer._runtime_fingerprint_payload()
    # JSON normalizes Path/tuple values already represented as strings/lists on disk.
    actual = json.loads(json.dumps(actual, default=str))
    if actual != expected:
        keys = [
            key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key)
        ]
        raise ValueError(
            "Primary numerical/prompt recipe differs from stopped checkpoint: "
            + ", ".join(sorted(keys))
        )
    return actual


class HostedFixtures:
    """Capture exact interface inputs and deterministic replies without any transport."""

    def __init__(self, device, scorer=None):
        self.device = device
        self.scorer = scorer
        self.records = []

    def briefs(self, sources, targets, packets):
        if not len(sources) == len(targets) == len(packets):
            raise ValueError("Incomplete brief inputs")
        replies = []
        for source, target, packet in zip(sources, targets, packets):
            row = {"role": "brief", "source": source, "target": target, "packet": packet}
            if self.scorer is not None:
                packet = self.scorer._normalize_text(packet)
                row["packet"] = packet
                row["prompt"] = self.scorer._brief_prompt(source, target, packet)
            passthrough = not packet or (
                self.scorer is not None
                and not self.scorer.llm_experiment_enabled
                and len(packet.split()) <= int(self.scorer.max_total_tokens_llm_summary * 0.75)
            )
            row["would_generate"] = not passthrough
            row["reply"] = packet if passthrough else "qualification fixture brief " + identity(row)
            self.records.append(row)
            replies.append(row["reply"])
        return replies

    def judgments(self, sources, targets, briefs, target_briefs, **kwargs):
        import torch

        if kwargs or not len(sources) == len(targets) == len(briefs) == len(target_briefs):
            raise ValueError("Incomplete decision inputs")
        replies = []
        for source, target, brief, target_brief in zip(sources, targets, briefs, target_briefs):
            row = {
                "role": "decision",
                "source": source,
                "target": target,
                "brief": brief,
                "target_brief": target_brief,
            }
            if self.scorer is not None:
                row["prompt"] = self.scorer._hosted_decision_prompt(
                    source, target, brief, target_brief
                )
            row["probability"] = int(identity(row)[:8], 16) / (2**32 - 1)
            self.records.append(row)
            replies.append(row["probability"])
        return torch.tensor(replies, dtype=torch.float32, device=self.device)


def selector_rows(metadata, sources, targets, result):
    """Match Trainer's scalar scorer-output projection, retaining retrieval metadata."""
    import torch

    lookup = {(row["Src"], row["Tgt"]): row for row in metadata}
    rows = [dict(lookup[(s, t)]) for s, t in zip(sources, targets)]
    for name, value in result.items():
        if isinstance(value, torch.Tensor) and tuple(value.shape) == (len(rows),):
            values = value.detach().cpu().tolist()
            if any(not math.isfinite(v) for v in values):
                raise ValueError(f"Nonfinite fitted scorer output: {name}")
            for row, scalar in zip(rows, values):
                row[name] = scalar
    return rows


def apply_selector(config, dataset, scorer, rows, records, exact_pairs):
    """Run the real fitted selector and extraction once over the entire bounded panel."""
    import pandas as pd
    from exact.experiments.difference_replay import _extract
    from exact.impl.models.selector.selector import CandidateSetSelector

    specs = config.get_model_sequence()[1:]
    if len(specs) != 1 or specs[0].name.__name__ != "CandidateSetSelector":
        raise ValueError("Qualification expects exactly the selected fitted selector")
    params = dict(specs[0].params)
    if config.selector.runtime_enabled is not None:
        params["enabled"] = config.selector.runtime_enabled
    if config.selector.runtime_global_only is not None:
        params["global_only"] = config.selector.runtime_global_only
    params.update(
        experiment_config=config.selector.model_dump(mode="python"),
        matching_calibration=config.matching.calibration.model_dump(mode="python"),
        nil_config=config.matching.nil.model_dump(mode="python"),
        request_seed=config.seed,
    )
    if params.get("training_reference_file_path") or params.get("llm", {}).get("enabled"):
        raise ValueError("Fitted qualification must not fit or invoke a selector LLM")
    selector = CandidateSetSelector(**params)
    result = selector(
        pd.DataFrame(rows),
        primary_model=scorer,
        dataset=dataset,
        results_json=records,
        local_alignment=False,
        threshold=config.matching.threshold,
        cardinality=config.matching.cardinality,
        target_cardinality=config.matching.target_cardinality,
    )
    frame = result["candidate_df"]
    output = records_without_nan(frame)
    for row in output:
        for field in ("selector_explanation", "candidate_decision"):
            if isinstance(row.get(field), str):
                row[field] = json.loads(row[field])
    if not frame.S_final.map(math.isfinite).all():
        raise ValueError("Fitted selector emitted nonfinite scores")
    sources = set(frame.Src)
    protected = {(s, t) for s, t in exact_pairs if s in sources}
    extraction_rows = output + [{"Src": s, "Tgt": t, "S_final": 1.0} for s, t in sorted(protected)]
    scores = {(row["Src"], row["Tgt"]): row["S_final"] for row in extraction_rows}
    policy = {
        "threshold": config.matching.threshold,
        "source_cardinality": config.matching.cardinality,
        "target_cardinality": config.matching.target_cardinality,
        "extraction": config.matching.extraction.model_dump(mode="python"),
    }
    mappings = _extract(extraction_rows, scores, policy, protected)
    return {
        "namespace": "qualification_only",
        "hosted_quality_established": False,
        "selector_scope": "bounded_cohort",
        "rows": output,
        "mappings": sorted(map(list, mappings)),
        "policy": policy,
        "protected_pairs": sorted(map(list, protected)),
    }
