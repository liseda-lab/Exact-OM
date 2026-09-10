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
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"SrcEntity", "TgtCandidates"}
    if not required.issubset(frame.columns):
        raise ValueError("expected benchmark SrcEntity/TgtCandidates columns")
    if "TgtEntity" not in frame:
        frame["TgtEntity"] = ""
    rows, labels = [], []
    pooled: dict[str, set[str]] = {}
    for row in frame.itertuples(index=False):
        candidates = ast.literal_eval(row.TgtCandidates)
        if not isinstance(candidates, (list, tuple)) or not all(
            isinstance(x, str) for x in candidates
        ):
            raise ValueError("candidate pool must contain a list of IRI strings")
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
    outputs = {"candidates": pool_path, "source_universe": universe_path}
    if expose_labels:
        reference_path = destination / f"{role}.reference.tsv"
        pd.DataFrame(
            sorted(set(labels)), columns=["SrcEntity", "TgtEntity", "Relation", "Score"]
        ).to_csv(reference_path, sep="\t", index=False)
        outputs["reference"] = reference_path
    record = {
        "schema_version": 2,
        "role": role,
        "input_sha256": sha256_file(path),
        "transformation": "canonical_pool_gold_separation_v1",
        "sources": frame.SrcEntity.nunique(),
        "original_query_rows": len(frame),
        "query_grouping": "union_candidates_per_source",
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
