"""Pinned BioLLM rejection benchmarks; labels apply only to supplied candidate pools.

Prepare acquired inputs with ``python -m exact.experiments.biollm_inputs POOL SOURCE
TARGET OUTPUT --task ncit-doid --role development``. The SNOMED-FMA task permits
only ``--role reporting``; it cannot create training splits through this converter.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from exact.experiments.inputs import nested_sources
from exact.utils.fitted_artifacts import freeze_json
from exact.utils.provenance import sha256_file

REVISION = "1972457c9be2664c627f822b5f5086bcbd63e274"
CONTRACT = {
    "url": "https://krr-oxford.github.io/DeepOnto/bio-ml/#oaei-bio-llm-2023",
    "documentation_revision": "35c91c0bb86423ef65c8f003a92f7630e22a582c",
    "documentation_sha256": "29a44ef546e5f7c102e68d068722c73d1f23babd4791bcd3445db0bdf44d2b1e",
    "ontology_release": "https://doi.org/10.5281/zenodo.13119437",
    "candidate_repository": "https://github.com/KRR-Oxford/LLMap-Prelim",
    "candidate_revision": REVISION,
    "scope": "benchmark-defined rejection and pair labels within the original supplied candidate pools; no verified ontology-wide absence",
}
INPUTS = {
    "ncit-doid": {
        "candidate_path": "data/ncit2doid/test_cands.tsv",
        "pool_sha256": "316737e95d7a8b75cf73e085d26185e5739422d78538a72ad5e145cc5c449205",
        "source_sha256": "379a37f47c0c8e7c30397769358cca955140d16b2797a1cc75da4b1fc2b354eb",
        "target_sha256": "76f41cce3616ad1a9ba6353f469e96bde7addba5d43e541651a3ab703f9ba2bc",
        "archive_sha256": "701602faf7fd0427a029ef718ada000d15aaa13b5e810ff3797538307d7aa0a6",
    },
    "snomed-fma.body": {
        "candidate_path": "data/snomed2fma/test_cands.tsv",
        "pool_sha256": "962348afc0c5eb96059595cdb6da96d5a0ec8fa95d03b20f3b6b848a5d0cd611",
        "source_sha256": "3a69eea251e40b4c09bd1f22ad9e1ef1229540264b4ed1121b29a432ca77e7a7",
        "target_sha256": "d6013f4b710b6163c67ff25a4c7e8e80ab5c541b4a51a73b49edd32a75bfa143",
        "archive_sha256": "5f9e5fd337233bcd4472a881a13a7f6eb7fe5d38a33104762de6b54843364919",
    },
}


def prepare_biollm_nil(
    pool: Path,
    source: Path,
    target: Path,
    destination: Path,
    *,
    task: str,
    role: str,
    seed: int = 17,
) -> dict[str, Any]:
    """Separate benchmark labels and freeze stratified source partitions on pinned ontologies."""
    if task not in INPUTS or role not in {"development", "reporting"}:
        raise ValueError("Expected a pinned BioLLM task and development/reporting role")
    if role != {"ncit-doid": "development", "snomed-fma.body": "reporting"}[task]:
        raise ValueError("Designated BioLLM reporting data cannot become training/development data")
    inputs = {}
    for name, path in {"pool": pool, "source": source, "target": target}.items():
        observed = sha256_file(path)
        if observed != INPUTS[task][f"{name}_sha256"]:
            raise ValueError(
                f"BioLLM {name} differs from its pinned legacy input; no label transplant"
            )
        inputs[name] = {"path": str(path.resolve()), "sha256": observed}
    frame = pd.read_csv(pool, sep="\t", dtype=str, keep_default_na=False)
    if set(frame) != {"SrcEntity", "TgtEntity", "TgtCandidates"}:
        raise ValueError("BioLLM needs the original explicit gold-bearing candidate schema")
    if len(frame) != 100 or frame.SrcEntity.duplicated().any() or (frame.SrcEntity == "").any():
        raise ValueError("BioLLM requires 100 distinct source groups")
    rows: dict[str, dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        candidates = ast.literal_eval(row.TgtCandidates)
        if (
            not isinstance(candidates, (list, tuple))
            or not 1 <= len(candidates) <= 100
            or not all(isinstance(value, str) and value for value in candidates)
            or len(set(candidates)) != len(candidates)
        ):
            raise ValueError("BioLLM candidate pools must retain their unique original members")
        nil = row.TgtEntity == "UnMatched"
        if not nil and (not row.TgtEntity or row.TgtEntity not in candidates):
            raise ValueError("Matched BioLLM source needs its explicit positive inside the pool")
        rows[row.SrcEntity] = {
            "gold": None if nil else row.TgtEntity,
            "status": "benchmark_nil" if nil else "in_pool",
            "candidates": sorted(candidates),
        }
    if Counter(row["status"] for row in rows.values()) != {"benchmark_nil": 50, "in_pool": 50}:
        raise ValueError("Pinned BioLLM has 50 explicit matched and 50 UnMatched source groups")
    if role == "development":
        partitions: dict[str, list[str]] = {
            name: [] for name in ["train", "valid", "internal_check"]
        }
        for status in ["benchmark_nil", "in_pool"]:
            ordered = nested_sources(
                (src for src, row in rows.items() if row["status"] == status), None, seed=seed
            )
            for name, subset in zip(partitions, [ordered[:30], ordered[30:40], ordered[40:]]):
                partitions[name].extend(subset)
    else:
        partitions = {"test": list(rows)}
    partitions = {name: sorted(members) for name, members in partitions.items()}
    destination.mkdir(parents=True, exist_ok=True)
    pending: dict[Path, str] = {}
    split_records: dict[str, dict[str, Any]] = {}
    for split, members in partitions.items():
        confirmed = [
            (src, tgt, int(tgt == rows[src]["gold"]))
            for src in members
            for tgt in rows[src]["candidates"]
        ]
        reference = [(src, rows[src]["gold"], "=", 1.0) for src in members if rows[src]["gold"]]
        tables = {
            "candidates": pd.DataFrame([(s, t) for s, t, _ in confirmed], columns=["Src", "Tgt"]),
            "confirmed_candidates": pd.DataFrame(
                confirmed, columns=["Src", "Tgt", "confirmed_label"]
            ),
            "reference": pd.DataFrame(
                reference, columns=["SrcEntity", "TgtEntity", "Relation", "Score"]
            ),
            "source_labels": pd.DataFrame(
                [(src, rows[src]["status"]) for src in members], columns=["Src", "Status"]
            ),
        }
        files = {name: destination / f"{split}.{name}.tsv" for name in tables}
        pending.update(
            {files[name]: table.to_csv(sep="\t", index=False) for name, table in tables.items()}
        )
        files["source_universe"] = destination / f"{split}.sources.txt"
        pending[files["source_universe"]] = "\n".join(members) + "\n"
        split_records[split] = {
            "sources": members,
            "source_status_counts": dict(Counter(rows[src]["status"] for src in members)),
            "candidate_count_histogram": dict(
                Counter(str(len(rows[src]["candidates"])) for src in members)
            ),
            "pairs": len(confirmed),
            "positive_pairs": len(reference),
            "exposure": (
                "training_only"
                if split == "train"
                else (
                    "development_evaluation_only"
                    if role == "development"
                    else "public_heldout_reporting_only"
                )
            ),
            "outputs": files,
        }
    for path, text in pending.items():
        if path.exists() and path.read_text() != text:
            raise ValueError(f"Immutable BioLLM preparation conflict: {path}")
    for path, text in pending.items():
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_text(text)
            temporary.replace(path)
    for record in split_records.values():
        record["outputs"] = {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in record["outputs"].items()
        }
    manifest = {
        "schema_version": 1,
        "transformation": (
            "source_stratified_60_20_20_v1"
            if role == "development"
            else "unmodified_public_reporting_population_v1"
        ),
        "task": task,
        "role": role,
        "seed": seed,
        "inputs": inputs,
        "official_contract": CONTRACT,
        "pinned_task": INPUTS[task],
        "original_reference_role": "public_benchmark_test",
        "label_semantics": "benchmark_pool",
        "reference_completeness": "known_incomplete",
        "negative_policy": "confirmed_only",
        "unknown_outside_supplied_pool": True,
        "natural_ontology_nil_claim": False,
        "splits": split_records,
    }
    freeze_json(destination / "inputs.json", manifest)
    return manifest


def main() -> None:
    """Prepare already acquired, hash-verified legacy inputs without model execution."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["pool", "source", "target", "destination"]:
        parser.add_argument(name, type=Path)
    parser.add_argument("--task", choices=sorted(INPUTS), required=True)
    parser.add_argument("--role", choices=["development", "reporting"], required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    record = prepare_biollm_nil(**vars(args))
    print(
        json.dumps(
            {"task": record["task"], "role": record["role"], "splits": list(record["splits"])}
        )
    )


if __name__ == "__main__":
    main()
