"""Read-only recovery proof for matched E08 probes with cache-dependent metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from exact.experiments.recovery import ArtifactStore
from exact.utils.provenance import sha256_file

_ARMS = {"current", "unified_bank", "provenance_dedup"}
_KEYS = ["Src", "Tgt", "SrcKind", "TgtKind"]
_SCORES = [
    "cand_sim",
    "cand_sim_semantic",
    "cand_sim_lexical",
    "cand_sim_retrieval",
    "cand_sim_cross_encoder",
    "cand_channels",
]


def _read(path):
    return json.loads(Path(path).read_text())


def _binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def _verify(path, expected):
    if sha256_file(path) != expected["sha256"]:
        raise ValueError(f"Historical population evidence changed: {path}")


def _population(frame, groups):
    # These are pure summaries: no ontology/model loading or scientific execution.
    from exact.impl.datasets.base import BaseAlignmentDataset

    per_kind, counts = {}, []
    for kind in sorted({kind for _, kind in groups}):
        part = frame.loc[frame.SrcKind == kind]
        population, sizes = BaseAlignmentDataset._candidate_kind_manifest(
            BaseAlignmentDataset,
            part,
            expected_sources=sum(k == kind for _, k in groups),
        )
        per_kind[kind] = population
        counts.extend(sizes)
    return {
        "gold_free_summary": {
            **BaseAlignmentDataset._candidate_count_summary(counts),
            "candidate_pairs": len(frame),
            "covered_sources": int(frame.Src.nunique()),
        },
        "per_kind": per_kind,
    }


def audit_matched_measurements(rows):
    """Bind complete old probes and prove equal actual unlabeled populations.

    Accept the historical raw-versus-processed manifest discrepancy only when
    each saved dataset is bound by its original completed artifact, all ordered
    populations/scores/masks match, and every manifest describes either the
    common raw pool or the independently reconstructed effective population.
    Originals and their completion receipts are never rewritten.
    """
    if len(rows) != 3 or {row["name"] for row in rows} != _ARMS:
        raise ValueError("Recovery requires every matched E08 treatment")
    bindings, evidence, signatures = [], [], []
    common_population = common_raw = common_sample = None
    for row in rows:
        output = Path(row["output_dir"]).resolve()
        measurement = output.parent / "measurement.json"
        if _read(measurement) != row:
            raise ValueError("Measurement differs from its original complete receipt")
        if (
            row["status"] != "passed"
            or row["execution_status"] != "complete"
            or row["prefix"]
            or any(row["new_usage"].values())
            or row["processed_pairs"] != row["dataset_rows"]
            or row["worker_calls"] != 1
            or row["worker_measurement"]["return_code"] != 0
            or row["generate_rationales"]
            or not row["no_private_test_references"]
        ):
            raise ValueError("Recovery requires complete single-call cached-only probes")
        required = {
            output / "recovery-runtime.json",
            output / "experiment_manifest.json",
            output / "validation-worker.json",
            output / "dataset/candidate_pool_sample_manifest.json",
        }
        verified = set()
        for record in row["bindings"]:
            path = Path(record["path"]).resolve()
            _verify(path, record)
            verified.add(path)
            bindings.append(_binding(path))
        if not required.issubset(verified):
            raise ValueError("Original measurement lacks required population bindings")
        bindings.append(_binding(measurement))
        runtime = _read(output / "recovery-runtime.json")
        identity = runtime["identity"]
        artifact_id = ArtifactStore._identity(identity)
        root = Path(runtime["root"]).resolve()
        if root != output.parent or artifact_id != identity["artifact_id"]:
            raise ValueError("Historical runtime identity/root changed")
        stage_path = root / "artifacts/stages" / f"{artifact_id}.json"
        stage = _read(stage_path)
        if stage["status"] != "complete" or stage["identity"] != identity:
            raise ValueError("Historical probe has no compatible complete artifact")
        bindings.append(_binding(stage_path))
        for name in (
            "dataset/dataset.csv",
            "dataset/candidate_pool_manifest.json",
            "dataset/candidate_pool_sample_manifest.json",
        ):
            _verify(output / name, stage["outputs"][name])
            bindings.append(_binding(output / name))
        raw = _read(output / "dataset/candidate_pool_manifest.json")
        sampled = _read(output / "dataset/candidate_pool_sample_manifest.json")
        sample = sampled["retrieval_config"]["source_sample"]
        groups = [tuple(x) for x in sample["source_kind_groups"]]
        if (
            len(groups) != len(set(groups))
            or sample["selected_groups"] != len(groups)
            or sample["cap"] != row["source_cap"]
            or sample["seed"] != row["seed"]
            or sample["eligible_source_iris"] != sorted({src for src, _ in groups})
            or sample["sha256"]
            != hashlib.sha256(
                "\n".join(f"{s}\t{k}" for s, k in sorted(groups)).encode()
            ).hexdigest()
        ):
            raise ValueError("Historical frozen source population changed")
        frame = pd.read_csv(
            output / "dataset/dataset.csv",
            usecols=lambda column: column in _KEYS + _SCORES + ["inference", "prefiltered"],
        )
        if not set(_KEYS + ["cand_sim", "inference", "prefiltered"]).issubset(frame.columns):
            raise ValueError("Historical dataset lacks candidate identities or masks")
        if frame[_KEYS].isna().any().any() or frame.duplicated(_KEYS).any():
            raise ValueError("Historical dataset contains invalid or duplicate pairs")
        if not set(zip(frame.Src, frame.SrcKind)).issubset(set(groups)):
            raise ValueError("Historical dataset escaped the frozen source population")
        effective = frame.loc[frame.cand_sim.notna()].reset_index(drop=True)
        exact = frame.loc[frame.cand_sim.isna()]
        if (
            len(effective) != row["dataset_rows"]
            or not effective.inference.eq(True).all()
            or not effective.prefiltered.eq(False).all()
            or not exact.inference.eq(False).all()
            or not exact.prefiltered.eq(True).all()
        ):
            raise ValueError("Historical processed population/count/masks disagree")
        population = _population(effective, groups)
        raw_population = {key: raw[key] for key in ("gold_free_summary", "per_kind")}
        recorded = {key: sampled[key] for key in ("gold_free_summary", "per_kind")}
        if recorded not in (population, raw_population):
            raise ValueError("Historical manifest has an unexplained population difference")
        if raw_population["gold_free_summary"]["candidate_pairs"] != len(frame):
            raise ValueError("Historical raw population does not match saved dataset")
        run_manifest_path = output / "run_manifest.json"
        run_manifest = _read(run_manifest_path)
        decisions_path = output / "candidate_decisions.json"
        decision_records = [
            entry
            for entry in run_manifest["artifacts"]
            if entry["path"] == "candidate_decisions.json"
        ]
        if len(decision_records) != 1:
            raise ValueError("Missing original candidate decision digest")
        _verify(decisions_path, decision_records[0])
        decisions = _read(decisions_path)
        decision_pairs = [
            (r["source"]["iri"], r["target"]["iri"], r["source"]["kind"], r["target"]["kind"])
            for r in decisions["records"]
        ]
        pairs = list(frame[_KEYS].itertuples(index=False, name=None))
        if (
            decisions["reference_labels_used"]
            or len(decision_pairs) != len(set(decision_pairs))
            or set(decision_pairs) != set(pairs)
            or decisions["candidate_pool"] != sampled
        ):
            raise ValueError("Historical candidate decisions disagree with saved population")
        bindings.extend([_binding(decisions_path), _binding(run_manifest_path)])
        signature = hashlib.sha256(
            frame.to_json(orient="split", double_precision=15).encode()
        ).hexdigest()
        signatures.append(signature)
        if common_population is None:
            common_population, common_raw, common_sample = population, raw, sample
        elif (population, raw, sample) != (common_population, common_raw, common_sample):
            raise ValueError("E08 treatments changed actual populations or retrieval declarations")
        evidence.append(
            {
                "name": row["name"],
                "raw_rows": len(frame),
                "effective_rows": len(effective),
                "exact_prefilter_rows": len(exact),
                "ordered_unlabeled_rows_sha256": signature,
                "historical_manifest_scope": "effective" if recorded == population else "raw",
            }
        )
    if len(set(signatures)) != 1:
        raise ValueError("E08 treatments changed candidate identities, order, scores or masks")
    return {"candidate_population": common_population, "bindings": bindings, "evidence": evidence}
