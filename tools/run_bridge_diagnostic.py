#!/usr/bin/env python3
"""Compare named graph closure with native OWL on frozen public development pairs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exact.core.entities.kinds import EntityKind  # noqa: E402
from exact.experiments.relation_metrics import relation_metrics  # noqa: E402
from exact.io.relation_bridge import native_bridge  # noqa: E402
from exact.io.relations import _semantic_entailment  # noqa: E402
from exact.io.sources.csv_kg import create_source  # noqa: E402
from exact.utils.provenance import sha256_path  # noqa: E402
from tools.run_directional_diagnostic import load_pairs, verified, write  # noqa: E402


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_path(path)}


def prepare_recipe(bindings, originals, output):
    """Bind the pinned NCIT/DOID originals; labels are confined to public development."""
    import yaml

    case = yaml.safe_load(Path(bindings).read_text())["cases"]["T0"]
    if case["role"] != "development" or case["kind"] != "class":
        raise ValueError("E14 requires the public typed development class case")
    originals = Path(originals)
    imports = json.loads((originals / "DOID-import-map.json").read_text())
    recipe = {
        "schema_version": 1,
        "kind": "e14_native_known_pairs",
        "reference_scope": "public_development_train_valid_only",
        "inputs": {
            **{side: case[side] for side in ("source", "target")},
            **{role: case["references"][role] for role in ("train", "valid")},
        },
        "owl": {
            "source": bind(originals / "NCIT-26.04d-Thesaurus.owl"),
            "target": bind(originals / "DOID-a3447b9f-doid.owl"),
        },
        "imports": {iri: bind(path) for iri, path in imports.items()},
        "output": str(Path(output).resolve()),
        "source_cap": 300,
        "seed": 17,
        "anchors": "all_public_train_equivalences_query_bridge_excluded",
        "selection_eligible": False,
        "hosted_calls": False,
        "rationales": False,
        "max_memory_bytes": 80 * 1024**3,
        "max_compile_work": 2**64 - 1,
        "max_native_symbol_index_bytes": 80 * 1024**3,
        "timeout_seconds": None,
        "comparison_scope": "known_valid_pairs_graph_view_vs_full_OWL_not_logical_parity_or_end_to_end_F1",
    }
    # Do not infer originals from a release name: these are the investigated pinned bytes.
    expected = {
        "source": "1b2cbe682d0d9a55b893be4a5c6ac4114f46a9f554f24208d6dbe5dd56b7fb84",
        "target": "d23d1a034075abd3d9c23f19ee79d3f7dfce29244b1e38078d35c0a952f35485",
    }
    if {side: value["sha256"] for side, value in recipe["owl"].items()} != expected:
        raise ValueError("E14 originals do not match the investigated BioKG release")
    for value in recipe["inputs"].values():
        verified(value)
    return recipe


def iri_map(directory):
    frame = pd.read_csv(directory / "properties.csv", usecols=["node_id", "iri"], dtype=str)
    if frame.isna().any().any() or frame.node_id.duplicated().any() or frame.iri.duplicated().any():
        raise ValueError("Public identifier mapping must be unambiguous")
    return dict(zip(frame.node_id, frame.iri))


def remap(frame, source_ids, target_ids):
    result = frame.copy()
    for column, mapping in (("SrcEntity", source_ids), ("TgtEntity", target_ids)):
        result[column] = result[column].map(mapping)
        if result[column].isna().any():
            raise ValueError("Public pair has no witnessed original OWL IRI")
    return result


def anchor_records(frame):
    return [
        {
            "src": row.SrcEntity,
            "tgt": row.TgtEntity,
            "score": 1.0,
            "src_kind": EntityKind.CLASS,
            "tgt_kind": EntityKind.CLASS,
            "origin": "public_train",
        }
        for row in frame.itertuples(index=False)
    ]


def run_diagnostic(recipe_path):
    recipe = json.loads(Path(recipe_path).read_text())
    fixed = {
        "schema_version": 1,
        "kind": "e14_native_known_pairs",
        "source_cap": 300,
        "seed": 17,
        "reference_scope": "public_development_train_valid_only",
        "anchors": "all_public_train_equivalences_query_bridge_excluded",
        "selection_eligible": False,
        "hosted_calls": False,
        "rationales": False,
        "timeout_seconds": None,
        "comparison_scope": "known_valid_pairs_graph_view_vs_full_OWL_not_logical_parity_or_end_to_end_F1",
    }
    if any(recipe.get(key) != value for key, value in fixed.items()):
        raise ValueError("E14 recipe differs from approved diagnostic protocol")
    if set(recipe["inputs"]) != {"source", "target", "train", "valid"}:
        raise ValueError("E14 inputs must be public train/valid only")
    paths = {name: verified(value) for name, value in recipe["inputs"].items()}
    reasoning_inputs, reasoning_imports = recipe["owl"], recipe["imports"]
    preparation = recipe.get("reasoning_preparation")
    excluded_datatypes = set()
    if preparation:
        from tools.prepare_bridge_ontology import verify_preparation

        reasoning_inputs, reasoning_imports = verify_preparation(
            preparation, recipe["owl"], recipe["imports"]
        )
        excluded_datatypes = set(
            json.loads(verified(preparation).read_text())["excluded_datatypes"]
        )
    owl = {name: verified(value) for name, value in reasoning_inputs.items()}
    imports = {name: verified(value) for name, value in reasoning_imports.items()}
    source_ids, target_ids = (iri_map(paths[side]) for side in ("source", "target"))
    gold, sources = load_pairs(paths["valid"], recipe["source_cap"], recipe["seed"])
    training = pd.read_csv(paths["train"], sep="\t", dtype=str)
    if set(training.SrcEntity) & set(gold.SrcEntity):
        raise ValueError("E14 train anchors and development query sources overlap")
    anchors = training.loc[training.Relation.eq("="), ["SrcEntity", "TgtEntity"]].drop_duplicates()
    pairs = gold[["SrcEntity", "TgtEntity"]].assign(Score=0.0)
    native_pairs = remap(pairs, source_ids, target_ids)
    native_anchors = remap(anchors, source_ids, target_ids)
    endpoints = {
        value
        for frame in (native_pairs, native_anchors)
        for column in ("SrcEntity", "TgtEntity")
        for value in frame[column]
    }
    if endpoints & excluded_datatypes:
        raise ValueError("Excluded metadata datatype cannot be a native bridge/query class")
    output = Path(recipe["output"])
    output.mkdir(parents=True, exist_ok=True)
    write(
        output / "population.json",
        {
            "sources": sources,
            "reference_role": "valid",
            "anchor_role": "train",
            "anchor_count": len(anchors),
            "query_count": len(pairs),
            "recipe": bind(recipe_path),
        },
    )
    native = native_bridge(
        native_pairs,
        owl["source"],
        owl["target"],
        anchor_rows=anchor_records(native_anchors),
        timeout_seconds=recipe["timeout_seconds"],
        max_memory_bytes=recipe["max_memory_bytes"],
        max_compile_work=recipe.get("max_compile_work"),
        max_native_symbol_index_bytes=recipe.get("max_native_symbol_index_bytes"),
        import_map=imports,
        checkpoint_path=output / "native-checkpoint.json",
        preparation_identity=preparation,
    )
    native_attrs = native.attrs.copy()
    native = remap(
        native, {v: k for k, v in source_ids.items()}, {v: k for k, v in target_ids.items()}
    )
    for collection in (recipe["inputs"], recipe["owl"], recipe["imports"]):
        for value in collection.values():
            verified(value)
    if preparation:
        verify_preparation(preparation, recipe["owl"], recipe["imports"])
    # One explicit anchor inventory is shared by both methods. No exact-label or
    # scored-candidate anchor discovery is allowed in this controlled comparison.
    graph = _semantic_entailment(
        pairs,
        create_source(paths["source"]),
        create_source(paths["target"]),
        anchors=None,
        anchor_threshold=1.0,
        anchor_margin=0.0,
        relation_threshold=0.5,
        timeout_seconds=float("inf"),
        frozen_anchor_rows=anchor_records(anchors),
    )
    for collection in (recipe["inputs"], recipe["owl"], recipe["imports"]):
        for value in collection.values():
            verified(value)
    if preparation:
        verify_preparation(preparation, recipe["owl"], recipe["imports"])
    native.to_csv(output / "native.tsv", sep="\t", index=False)
    graph.to_csv(output / "graph.tsv", sep="\t", index=False)
    result = {
        **fixed,
        "recipe": bind(recipe_path),
        "anchor_count": len(anchors),
        "query_count": len(pairs),
        "graph": relation_metrics(gold, graph),
        "native": relation_metrics(gold, native),
        "native_coherence": native_attrs["coherence_audit"],
        "native_abstentions": native_attrs["relation_abstentions"],
        "graph_abstentions": graph.attrs.get("relation_abstentions", []),
        "predictions": {name: bind(output / (name + ".tsv")) for name in ("native", "graph")},
    }
    if preparation:
        result.update(
            reasoning_preparation=preparation,
            reasoning_inputs=reasoning_inputs,
            reasoning_imports=reasoning_imports,
            admission_scope="full_original_logical_content_with_recorded_metadata_exclusions",
        )
    write(output / "bridge-diagnostic.json", result)
    write(
        output / "completion.json",
        {
            "status": "complete",
            "selection_eligible": False,
            "diagnostic": bind(output / "bridge-diagnostic.json"),
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    for name in ("bindings", "originals", "output", "recipe"):
        prepare.add_argument("--" + name, type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        write(args.recipe, prepare_recipe(args.bindings, args.originals, args.output))
    else:
        run_diagnostic(args.recipe)


if __name__ == "__main__":
    main()
