#!/usr/bin/env python3
"""Bind official BioML train-pool distractors as explicit labels; never run a model."""

import argparse
import json
from pathlib import Path

from exact.core.entities.configs.yaml_io import dump_yaml_document, load_yaml_mapping
from exact.experiments.inputs import prepare_confirmed_bioml_training
from exact.utils.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    args = parser.parse_args()
    bindings = dict(load_yaml_mapping(args.bindings))
    input_lock = json.loads((args.dataset_root / "inputs.lock.json").read_text())
    by_task = {}
    for case in bindings["cases"].values():
        pair = str(case["task"]).upper()
        if pair in {"NCIT-DOID", "SNOMED-FMA", "SNOMED-NCIT"} and "train" in case.get(
            "candidates", {}
        ):
            by_task.setdefault(pair, []).append(case)
    for pair, cases in sorted(by_task.items()):
        input_path = args.dataset_root / pair / "local.train.cands.tsv"
        binding = input_lock[f"{pair}/local.train.cands.tsv"]
        if binding["revision"] != args.dataset_revision or binding["sha256"] != sha256_file(
            input_path
        ):
            raise ValueError(f"Official dataset lock mismatch: {input_path}")
        reporting_sources = set()
        for case in cases:
            universe = case.get("source_universe")
            if universe:
                path = Path(universe["path"])
                path = path if path.is_absolute() else args.bindings.parent / path
                reporting_sources.update(path.read_text().splitlines())
        record = prepare_confirmed_bioml_training(
            args.dataset_root / pair / "local.train.cands.tsv",
            args.dataset_root / pair / "prepared",
            dataset_revision=args.dataset_revision,
            reporting_sources=reporting_sources,
        )
        for case in cases:
            case["candidates"]["train"] = record["outputs"]["candidates"]
            case.setdefault("references", {})["train"] = record["outputs"]["reference"]
            case.setdefault("local_references", {})["train"] = record["outputs"]["reference"]
            case["negative_policy"] = "confirmed_only"
            case.setdefault("overlay", {}).setdefault("candidates", {}).setdefault(
                "encoder_finetune", {}
            )["negative_policy"] = "confirmed_negative"
            case["reference_completeness"] = "known_incomplete"
            case["selection_reason"] = (
                f"Official BioML revision {args.dataset_revision}; training-only benchmark distractor labels "
                "from the pinned local-ranking contract, alternatives unioned per source; "
                "ontology-wide reference remains incomplete. Provenance: "
                + str((args.dataset_root / pair / "prepared/train.confirmed.inputs.json").resolve())
            )
        print(
            f"{pair}: {record['sources']} train sources, {record['positive_pairs']} positives, "
            f"{record['confirmed_negative_pairs']} declared distractors; zero reporting overlap"
        )
    temporary = args.bindings.with_suffix(args.bindings.suffix + ".partial")
    temporary.write_text(dump_yaml_document(bindings))
    temporary.replace(args.bindings)


if __name__ == "__main__":
    main()
