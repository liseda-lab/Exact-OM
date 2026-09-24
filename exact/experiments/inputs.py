"""Small, deterministic input transformations for the research campaign."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from exact.utils.provenance import sha256_file


def nested_sources(sources: Iterable[str], cap: int | None, *, seed: int = 17) -> list[str]:
    """Choose nested source samples independently of predictions and labels."""
    ranked = sorted(
        set(map(str, sources)),
        key=lambda iri: (hashlib.sha256(f"{seed}\x1f{iri}\x1fclass".encode()).digest(), iri),
    )
    if cap is not None and cap < 1:
        raise ValueError("source cap must be positive")
    return ranked if cap is None else ranked[:cap]


def prepare_pool(
    path: Path, destination: Path, *, role: str, expose_labels: bool
) -> dict[str, Any]:
    """Separate a benchmark pool's gold column from canonical unlabeled candidates.

    Final data preparation strips labels without emitting their values. Development
    label extraction is explicit and never turns a missing gold field into a negative.
    """
    if role not in {"train", "valid", "test", "internal_check"}:
        raise ValueError("unknown research role")
    if expose_labels and role == "test":
        raise ValueError("final labels require the evaluation access path after G4")
    columns = ["SrcEntity", "TgtCandidates"] + (["TgtEntity"] if expose_labels else [])
    frame = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, usecols=lambda name: name in columns
    )
    required = {"SrcEntity", "TgtCandidates"}
    if not required.issubset(frame.columns):
        raise ValueError("expected benchmark SrcEntity/TgtCandidates columns")
    if "TgtEntity" not in frame:
        frame["TgtEntity"] = ""
    rows, labels, queries = [], [], []
    pooled: dict[str, set[str]] = {}
    for query_id, row in enumerate(frame.itertuples(index=False)):
        candidates = ast.literal_eval(row.TgtCandidates)
        if not isinstance(candidates, (list, tuple)) or not all(
            isinstance(x, str) for x in candidates
        ):
            raise ValueError("candidate pool must contain a list of IRI strings")
        if not row.SrcEntity or not candidates or len(candidates) != len(set(candidates)):
            raise ValueError("queries need a source and nonempty unique candidates")
        queries.append({"qid": query_id, "source": row.SrcEntity, "candidates": list(candidates)})
        pooled.setdefault(row.SrcEntity, set()).update(candidates)
        if expose_labels and row.TgtEntity:
            labels.append((row.SrcEntity, row.TgtEntity, "=", 1.0))
    rows = [(source, "", repr(sorted(candidates))) for source, candidates in sorted(pooled.items())]
    destination.mkdir(parents=True, exist_ok=True)
    pool_path = destination / f"{role}.candidates.tsv"
    pd.DataFrame(sorted(set(rows)), columns=["SrcEntity", "TgtEntity", "TgtCandidates"]).to_csv(
        pool_path, sep="\t", index=False
    )
    universe_path = destination / f"{role}.sources.txt"
    universe_path.write_text("\n".join(sorted(set(frame.SrcEntity))) + "\n")
    query_path = destination / f"{role}.queries.jsonl"
    query_path.write_text("".join(json.dumps(row) + "\n" for row in queries))
    public_path = destination / f"{role}.public.queries.tsv"
    frame[["SrcEntity", "TgtCandidates"]].to_csv(public_path, sep="\t", index=False)
    outputs = {
        "candidates": pool_path,
        "source_universe": universe_path,
        "queries": query_path,
        "public_queries": public_path,
    }
    if expose_labels:
        reference_path = destination / f"{role}.reference.tsv"
        pd.DataFrame(
            sorted(set(labels)), columns=["SrcEntity", "TgtEntity", "Relation", "Score"]
        ).to_csv(reference_path, sep="\t", index=False)
        outputs["reference"] = reference_path
    record = {
        "schema_version": 3,
        "role": role,
        "input_sha256": sha256_file(path),
        "transformation": "query_preserving_pool_gold_separation_v2",
        "sources": frame.SrcEntity.nunique(),
        "original_query_rows": len(frame),
        "query_grouping": "original_rows_preserved_with_positional_qid",
        "candidate_inventory_grouping": "union_candidates_per_source",
        "union_score_reuse": "query_independent_components_only",
        "labels_exposed": expose_labels,
        "outputs": {
            name: {"path": str(file.resolve()), "sha256": sha256_file(file)}
            for name, file in outputs.items()
        },
    }
    (destination / f"{role}.inputs.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def research_partitions(
    reference: pd.DataFrame, *, seed: int = 17, group_column: str = "SrcEntity"
) -> dict[str, pd.DataFrame]:
    """Derive predeclared 60/20/20 development partitions by independent group."""
    if group_column not in reference:
        raise ValueError(f"reference needs grouping column {group_column}")
    groups = nested_sources(reference[group_column].astype(str), None, seed=seed)
    if len(groups) < 5:
        raise ValueError("too few independent groups for research train/valid/internal-check")
    train_end, valid_end = int(0.6 * len(groups)), int(0.8 * len(groups))
    bounds = {
        "train": groups[:train_end],
        "valid": groups[train_end:valid_end],
        "internal_check": groups[valid_end:],
    }
    return {
        role: reference.loc[reference[group_column].astype(str).isin(selected)].copy()
        for role, selected in bounds.items()
    }


BIOML_LOCAL_CONTRACT = {
    "url": "https://bio-ml.oaei-ml.org/tasks/local/",
    "repository": "https://github.com/liseda-lab/OAEI-Bio-ML",
    "revision": "ec436a97f49875227dedf634faa5280b1376bda9",
    "path": "tasks/local/ranking_task_index.md",
    "sha256": "882c59e3788bd93a9d529a24389179ebef1feccfd4fd083d7ea40c8981e32de5",
    "interpretation": "provided local-ranking distractors are benchmark negatives; no ontology-wide completeness claim",
}


def prepare_confirmed_bioml_training(
    path: Path,
    destination: Path,
    *,
    dataset_revision: str,
    reporting_sources: Iterable[str] = (),
) -> dict[str, Any]:
    """Convert the official gold-bearing train pool under its explicit distractor contract.

    This is deliberately separate from generic pool preparation: unlabeled or
    generated candidates never acquire negative labels through this function.
    All known alternative train positives override every query's distractors.
    """
    from exact.utils.fitted_artifacts import freeze_json

    path = Path(path)
    if path.name != "local.train.cands.tsv":
        raise ValueError("Confirmed BioML conversion accepts only local.train.cands.tsv")
    if len(dataset_revision) != 40 or any(c not in "0123456789abcdef" for c in dataset_revision):
        raise ValueError("An immutable official dataset revision is required")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if not {"SrcEntity", "TgtEntity", "TgtCandidates"}.issubset(frame.columns):
        raise ValueError("Official train pools need source, gold and candidate columns")
    reporting = set(map(str, reporting_sources))
    if set(frame.SrcEntity) & reporting:
        raise ValueError("Official train pools overlap frozen reporting source groups")
    positives: set[tuple[str, str]] = set()
    candidates: set[tuple[str, str]] = set()
    for row in frame.itertuples(index=False):
        pool = ast.literal_eval(row.TgtCandidates)
        if (
            not row.SrcEntity
            or not row.TgtEntity
            or not isinstance(pool, (list, tuple))
            or not pool
            or not all(isinstance(target, str) and target for target in pool)
            or row.TgtEntity not in pool
        ):
            raise ValueError(
                "Every train query needs a nonempty gold target inside its candidate pool"
            )
        positives.add((row.SrcEntity, row.TgtEntity))
        candidates.update((row.SrcEntity, target) for target in pool)
    if not positives:
        raise ValueError("Official training pool has no labeled queries")
    rows = [
        (source, target, int((source, target) in positives))
        for source, target in sorted(candidates)
    ]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "candidates": destination / "train.confirmed.candidates.tsv",
        "reference": destination / "train.local.reference.tsv",
    }
    tables = {
        "candidates": pd.DataFrame(rows, columns=["Src", "Tgt", "confirmed_label"]),
        "reference": pd.DataFrame(
            [(source, target, "=", 1.0) for source, target in sorted(positives)],
            columns=["SrcEntity", "TgtEntity", "Relation", "Score"],
        ),
    }
    for name, table in tables.items():
        encoded = table.to_csv(sep="\t", index=False)
        file = outputs[name]
        if file.exists() and file.read_text() != encoded:
            raise ValueError(f"Confirmed training input identity conflict: {file}")
        if not file.exists():
            temporary = file.with_suffix(file.suffix + ".partial")
            temporary.write_text(encoded)
            temporary.replace(file)
    record = {
        "schema_version": 1,
        "transformation": "official_bioml_train_distractors_v1",
        "role": "train",
        "dataset": "OAEI-ML/bio-ml",
        "dataset_revision": dataset_revision,
        "input_sha256": sha256_file(path),
        "official_contract": BIOML_LOCAL_CONTRACT,
        "reference_basis": "standard_unrepaired_local_equivalence",
        "reference_completeness": "known_incomplete",
        "negative_policy": "confirmed_only",
        "negative_scope": "provided_training_candidate_pairs_only",
        "positive_policy": "union_all_gold_alternatives_per_source_before_assigning_negatives",
        "sources": int(frame.SrcEntity.nunique()),
        "original_query_rows": len(frame),
        "positive_pairs": len(positives),
        "confirmed_negative_pairs": len(candidates - positives),
        "reporting_source_overlap": 0,
        "reporting_sources_checked": len(reporting),
        "outputs": {
            name: {"path": str(file.resolve()), "sha256": sha256_file(file)}
            for name, file in outputs.items()
        },
    }
    freeze_json(destination / "train.confirmed.inputs.json", record)
    return record
