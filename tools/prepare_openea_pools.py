#!/usr/bin/env python3
"""Generate E23's independent pools with the existing retrieval recipe; no scoring or fits."""

import argparse
import json
import tempfile
from pathlib import Path

from exact.experiments.openea import ARCHIVE_SHA256, CASE_ID, SPLIT, TASK, verify_prepared
from exact.experiments.openea_pools import _binding, _verified, prepare_pools
from exact.core.entities.configs.yaml_io import load_yaml_mapping
from exact.experiments.harness import deep_merge
from exact.experiments.preparation import prepare_campaign
from exact.utils.fitted_artifacts import fingerprint, freeze_json


def _campaign(blueprint, config, bindings, destination, *, identity):
    """Publish metadata atomically so a resumed preparation can reuse it safely."""
    receipt_name = "openea-preparation.json"
    if destination.exists():
        receipt = json.loads((destination / receipt_name).read_text())
        if receipt["identity"] != identity or any(
            _binding(destination / name)["sha256"] != digest
            for name, digest in receipt["files"].items()
        ):
            raise ValueError("E23 prepared campaign identity changed")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".openea-campaign-", dir=destination.parent) as tmp:
            staged = Path(tmp) / "campaign"
            prepare_campaign(blueprint, config, bindings, staged)
            freeze_json(
                staged / receipt_name,
                {
                    "identity": identity,
                    "files": {
                        str(path.relative_to(staged)): _binding(path)["sha256"]
                        for path in sorted(staged.rglob("*"))
                        if path.is_file()
                    },
                },
            )
            staged.rename(destination)
    return destination / "campaign.lock.yaml"


def run_diagnostic(recipe_path):
    """Prepare the approved own-case pools and emit the ordinary E23 batch group."""
    recipe_path = Path(recipe_path)
    recipe_binding = _binding(recipe_path)
    recipe = json.loads(recipe_path.read_text())
    fixed = {
        "schema_version": 1,
        "kind": "e23_pool_preparation",
        "case": CASE_ID,
        "source_cap": 300,
        "training_source_cap": 2000,
        "seed": 17,
        "selection_eligible": False,
        "hosted_calls": False,
        "rationales": False,
    }
    if set(recipe) != {*fixed, "inputs", "output", "campaign_output", "device"} or any(
        type(recipe.get(key)) is not type(value) or recipe.get(key) != value
        for key, value in fixed.items()
    ):
        raise ValueError("E23 preparation recipe differs from the approved protocol")
    if recipe["device"] not in {"cpu", "cuda:0"}:
        raise ValueError("E23 preparation requires the approved single-device lane")
    if set(recipe["inputs"]) != {"config", "preparation", "campaign_bindings", "blueprint"}:
        raise ValueError("E23 preparation permits only its four frozen input bindings")
    for value in recipe["inputs"].values():
        if set(value) != {"path", "sha256"} or not Path(value["path"]).is_absolute():
            raise ValueError("E23 preparation requires absolute checksummed input bindings")
    paths = {name: _verified(value) for name, value in recipe["inputs"].items()}
    if paths["preparation"].name != "preparation.json":
        raise ValueError("E23 preparation must bind the prepared dataset manifest")
    prepared = paths["preparation"].parent
    metadata = verify_prepared(prepared)
    expected = {
        "dataset": "OpenEA_v2.0",
        "task": TASK,
        "split": SPLIT,
        "training_cap": 2000,
        "source_cap": 300,
        "seed": 17,
        "schema_version": 2,
    }
    if metadata["recipe"] != expected or metadata["archive"]["sha256"] != ARCHIVE_SHA256:
        raise ValueError("E23 prepared dataset differs from the approved public split")
    original = dict(load_yaml_mapping(paths["campaign_bindings"]))
    if original.get("e23_natural_case") != CASE_ID or set(
        original.get("steps", {}).get("E23", {})
    ) - {"readiness", "estimate"}:
        raise ValueError("E23 preparation cannot override the approved case or four-arm recipe")
    output, campaign_output = Path(recipe["output"]), Path(recipe["campaign_output"])
    if not output.is_absolute() or not campaign_output.is_absolute() or output == campaign_output:
        raise ValueError("E23 preparation requires distinct absolute output directories")
    fragment = prepare_pools(paths["config"], prepared, output, device=recipe["device"])
    merged = deep_merge(original, fragment)
    if any(
        merged["cases"][key] != value for key, value in original["cases"].items() if key != CASE_ID
    ):
        raise ValueError("E23 preparation changed an unrelated case")
    # Long encoder work must not publish outputs under input hashes that changed.
    for value in [recipe_binding, *recipe["inputs"].values()]:
        _verified(value)
    verify_prepared(prepared)
    freeze_json(output / "campaign-bindings.json", merged)
    campaign = _campaign(
        paths["blueprint"],
        paths["config"],
        merged,
        campaign_output,
        identity=fingerprint(
            {"recipe": recipe_binding, "producer": _binding(Path(__file__)), "bindings": merged}
        ),
    )
    lock = dict(load_yaml_mapping(campaign))
    step = next(row for row in lock["steps"] if row["id"] == "E23")
    if step["case"] != CASE_ID or [arm["id"] for arm in step["arms"]] != [
        "natural_graph_off",
        "natural_inductive",
        "graph_only",
        "graph_shuffled",
    ]:
        raise ValueError("E23 preparation changed the approved four-arm comparison")
    group = {
        "kind": "metadata_recipe_bindings_not_admitted_campaign",
        "cases": {CASE_ID: lock["cases"][CASE_ID]},
        "rows": [{"step": "E23", "case_ids": [CASE_ID], "declaration": step}],
        "base_config": recipe["inputs"]["config"],
        "source_design": _binding(campaign),
    }
    freeze_json(output / "group.json", group)
    receipt = {
        **fixed,
        "status": "complete",
        "recipe": recipe_binding,
        "producer": _binding(Path(__file__)),
        "group": _binding(output / "group.json"),
        "campaign": _binding(campaign),
        "pool_provenance": _binding(output / "pools.json"),
        "campaign_bindings": _binding(output / "campaign-bindings.json"),
    }
    freeze_json(output / "preparation-receipt.json", receipt)
    freeze_json(
        output / "completion.json",
        {
            "status": "complete",
            "selection_eligible": False,
            "diagnostic": _binding(output / "preparation-receipt.json"),
        },
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--campaign-bindings", type=Path)
    parser.add_argument("--campaign-output", type=Path)
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "specs/experiments/campaign-v2.yaml",
    )
    args = parser.parse_args()
    if bool(args.campaign_bindings) != bool(args.campaign_output):
        parser.error("--campaign-bindings and --campaign-output must be supplied together")
    fragment = prepare_pools(args.config, args.prepared, args.output, device=args.device)
    print(args.output / "bindings-fragment.json")
    if args.campaign_bindings:
        bindings = deep_merge(dict(load_yaml_mapping(args.campaign_bindings)), fragment)
        freeze_json(args.output / "campaign-bindings.json", bindings)
        print(prepare_campaign(args.blueprint, args.config, bindings, args.campaign_output))


if __name__ == "__main__":
    main()
