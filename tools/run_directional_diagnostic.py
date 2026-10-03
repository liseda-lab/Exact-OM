#!/usr/bin/env python3
"""Prepare/run the public-development E24 known-pair diagnostic, without fitting."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exact.experiments.directional_diagnostics import (  # noqa: E402
    score_directions,
    summarize,
)
from exact.experiments.inputs import nested_sources  # noqa: E402
from exact.experiments.runtime import _code_identity  # noqa: E402
from exact.utils.provenance import sha256_file, sha256_path  # noqa: E402


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def verified(binding):
    path = Path(binding["path"]).resolve()
    if sha256_path(path) != binding["sha256"]:
        raise ValueError(f"E24 input changed: {path}")
    return path


def encoder_bindings(config_path):
    from exact.core.entities.configs.config import ConfigModel

    config = ConfigModel.load_config(config_path)
    primary = config.get_model_sequence()[0]
    if primary.name != "PairAdaptiveSemanticScorer":
        raise ValueError("E24 requires the production pair-adaptive scorer")
    defaults = {
        "lexical": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "context": "BAAI/bge-large-en-v1.5",
    }
    result = {}
    for kind, default in defaults.items():
        name = str(primary.params.get(f"{kind}_model_name", default))
        revision = primary.params.get(f"{kind}_model_revision")
        if Path(name).is_dir():
            result[kind] = {"path": str(Path(name).resolve()), "sha256": sha256_path(Path(name))}
        elif not re.fullmatch(r"[0-9a-fA-F]{40}", str(revision or "")):
            raise ValueError(f"E24 {kind} encoder requires a frozen 40-hex revision or local model")
        else:
            result[kind] = {"model": name, "revision": revision}
    return result


def prepare_recipe(bindings_path, config_path, output, *, case_id="T0", device="cuda:0"):
    bindings = yaml.safe_load(Path(bindings_path).read_text())
    case = bindings["cases"][case_id]
    if case["role"] != "development" or case["kind"] != "class":
        raise ValueError("E24 requires a public development class case")
    if not {"typed_reference", "csv_graph"}.issubset(case["capabilities"]):
        raise ValueError("E24 requires the bound public typed CSV view")
    inputs = {key: case[key] for key in ("source", "target")}
    inputs.update({role: case["references"][role] for role in ("train", "valid")})
    for binding in inputs.values():
        verified(binding)
    return {
        "schema_version": 1,
        "kind": "e24_directional_known_pairs",
        "encoders": encoder_bindings(config_path),
        "case": case_id,
        "reference_scope": "public_development_train_valid_only",
        "inputs": inputs,
        "config": {
            "path": str(Path(config_path).resolve()),
            "sha256": sha256_file(Path(config_path)),
        },
        "output": str(Path(output).resolve()),
        "source_cap": 300,
        "seed": 17,
        "device": device,
        "decision_rule": "higher_of_both_fixed_hypotheses_ties_or_unsupported_abstain",
        "control": "fixed_less_on_same_decided_pairs",
        "hosted_calls": False,
        "rationales": False,
        "selection_eligible": False,
    }


def load_pairs(path, cap, seed):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = ["SrcEntity", "TgtEntity", "Relation"]
    if not set(required).issubset(frame.columns):
        raise ValueError("typed reference must provide source, target, and relation")
    frame = frame[required].drop_duplicates()
    if not set(frame.Relation).issubset({"=", "<", ">"}):
        raise ValueError("unknown typed reference relation")
    if frame.duplicated(["SrcEntity", "TgtEntity"]).any():
        raise ValueError("ambiguous multi-relation pair in diagnostic reference")
    sources = set(nested_sources(frame.SrcEntity, cap, seed=seed))
    frame = frame[frame.SrcEntity.isin(sources)].sort_values(required)
    return frame, sorted(sources)


def build_components(config_path, inputs, output, device):
    """Use the production dataset and scorer; never attach any reference labels."""
    from exact.core.entities.configs.config import ConfigModel
    from exact.impl.datasets.pair_adaptive_context import PairAdaptiveContextDataset
    from exact.impl.models.pair_adaptive_scorer import PairAdaptiveSemanticScorer

    config = ConfigModel.load_config(config_path)
    primary = config.get_model_sequence()[0]
    if primary.name != "PairAdaptiveSemanticScorer":
        raise ValueError("E24 requires the production pair-adaptive scorer")
    dataset_params = config.dataset_params.model_dump()
    dataset_params.update(
        verbalization_mode="deterministic", verbaliser_name=None, filter_exact_matches=False
    )
    dataset = PairAdaptiveContextDataset(
        output_path=output,
        input_format="csv-kg",
        source_options={},
        target_options={},
        entity_kinds=["class"],
        device=device,
        cache_ok=True,
        **dataset_params,
    )
    dataset.load_ontologies(inputs["source"], inputs["target"])
    params = {
        **primary.params,
        **config.matching.channels.model_dump(mode="python"),
        "use_lexical": True,
        "use_context": True,
        "use_llm": False,
        "llm_model_name": None,
        "generate_llm_rationales": False,
        "use_llm_calibration": False,
        "persist_cache_to_disk": False,
        "device": device,
        "request_seed": 17,
        "graph": {"mode": "off"},
        "fusion_config": {"enabled": False},
        "llm_experiment_config": {"enabled": False},
        "diff": {
            **config.matching.channels.diff.model_dump(),
            "enabled": True,
            "formulation": "asymmetric",
            "relation_interpretation": "<",
        },
    }
    scorer = PairAdaptiveSemanticScorer(**params)
    scorer.attach_dataset(dataset)
    scorer.eval()
    return dataset, scorer


def run_diagnostic(recipe_path, *, component_factory=None):
    """Resume only checksummed components from exactly this implementation/input identity."""
    import torch

    started = time.monotonic()
    recipe = json.loads(Path(recipe_path).read_text())
    required = {
        "schema_version": 1,
        "kind": "e24_directional_known_pairs",
        "reference_scope": "public_development_train_valid_only",
        "source_cap": 300,
        "seed": 17,
        "hosted_calls": False,
        "rationales": False,
        "selection_eligible": False,
        "decision_rule": "higher_of_both_fixed_hypotheses_ties_or_unsupported_abstain",
        "control": "fixed_less_on_same_decided_pairs",
    }
    if any(recipe.get(key) != value for key, value in required.items()):
        raise ValueError("E24 recipe differs from the fixed diagnostic protocol")
    if set(recipe["inputs"]) != {"source", "target", "train", "valid"}:
        raise ValueError("E24 permits only source/target and public train/valid bindings")
    inputs = {name: verified(binding) for name, binding in recipe["inputs"].items()}
    config_path = verified(recipe["config"])
    if encoder_bindings(config_path) != recipe["encoders"]:
        raise ValueError("E24 encoder identity changed")
    output = Path(recipe["output"])
    output.mkdir(parents=True, exist_ok=True)
    code_root = Path(__file__).resolve().parents[1]
    identity = {
        "recipe": {key: value for key, value in recipe.items() if key != "output"},
        "implementation": _code_identity(code_root, evaluation=False)["sha256"],
        "producer": sha256_file(Path(__file__)),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "pandas",
                "pyowl-core",
                "pyowl2vec-star-projector",
            )
        },
        "cuda": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "gpu": (
            torch.cuda.get_device_name(recipe["device"])
            if str(recipe["device"]).startswith("cuda")
            else None
        ),
    }
    identity_hash = digest(identity)
    splits = {
        role: load_pairs(inputs[role], recipe["source_cap"], recipe["seed"])
        for role in ("train", "valid")
    }
    train_pairs = set(map(tuple, splits["train"][0][["SrcEntity", "TgtEntity"]].to_numpy()))
    valid_pairs = set(map(tuple, splits["valid"][0][["SrcEntity", "TgtEntity"]].to_numpy()))
    if train_pairs & valid_pairs:
        raise ValueError("train and valid known-pair inputs overlap")
    components = None
    results = {}
    computed = reused = 0
    with sqlite3.connect(output / "components.sqlite") as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL)")
        existing = connection.execute("SELECT identity FROM metadata").fetchall()
        if existing and existing != [(identity_hash,)]:
            raise ValueError("E24 checkpoint identity differs; use a new output directory")
        if not existing:
            connection.execute("INSERT INTO metadata VALUES (?)", (identity_hash,))
        connection.execute(
            "CREATE TABLE IF NOT EXISTS components (pair TEXT PRIMARY KEY, payload TEXT NOT NULL, sha256 TEXT NOT NULL)"
        )
        connection.commit()
        for role, (frame, sources) in splits.items():
            rows = []
            for source, target, relation in frame.itertuples(index=False, name=None):
                key = json.dumps([source, target])
                cached = connection.execute(
                    "SELECT payload, sha256 FROM components WHERE pair=?", (key,)
                ).fetchone()
                if cached:
                    value = json.loads(cached[0])
                    if digest(value) != cached[1]:
                        raise ValueError("E24 cached component checksum mismatch")
                    reused += 1
                else:
                    if components is None:
                        components = (component_factory or build_components)(
                            config_path, inputs, output, recipe["device"]
                        )
                    dataset, scorer = components
                    source_facts = dataset.get_entity_features(source, "src")["object_triples"]
                    target_facts = dataset.get_entity_features(target, "tgt")["object_triples"]
                    with torch.inference_mode():
                        value = score_directions(scorer, source_facts, target_facts)
                    connection.execute(
                        "INSERT INTO components VALUES (?, ?, ?)",
                        (key, json.dumps(value, sort_keys=True), digest(value)),
                    )
                    connection.commit()
                    computed += 1
                    if computed == 1 or computed % 25 == 0:
                        print(f"E24 {role}: {computed} pairs computed, {reused} reused", flush=True)
                rows.append({"source": source, "target": target, "relation": relation, **value})
            results[role] = {"source_groups": sources, "metrics": summarize(rows), "rows": rows}
    # Catch changed input bytes before publishing any completed diagnostic.
    for binding in [*recipe["inputs"].values(), recipe["config"]]:
        verified(binding)
    result = {
        "schema_version": 1,
        "experiment": "E24-asymmetric",
        "status": "complete",
        "scope": "known_public_development_pairs_only_not_end_to_end_alignment",
        "selection_eligible": False,
        "identity": identity,
        "identity_sha256": identity_hash,
        "execution_wall_seconds": time.monotonic() - started,
        "computed_pairs": computed,
        "reused_pairs": reused,
        "splits": results,
    }
    write(output / "directional-diagnostic.json", result)
    write(
        output / "completion.json",
        {
            "status": "complete",
            "diagnostic": {
                "path": str((output / "directional-diagnostic.json").resolve()),
                "sha256": sha256_file(output / "directional-diagnostic.json"),
            },
            "selection_eligible": False,
            "identity_sha256": identity_hash,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--bindings", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--recipe", type=Path, required=True)
    prepare.add_argument("--device", default="cuda:0")
    run = commands.add_parser("run")
    run.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        write(
            args.recipe, prepare_recipe(args.bindings, args.config, args.output, device=args.device)
        )
    else:
        result = run_diagnostic(args.recipe)
        print(
            json.dumps({key: result[key] for key in ("status", "computed_pairs", "reused_pairs")})
        )


if __name__ == "__main__":
    main()
