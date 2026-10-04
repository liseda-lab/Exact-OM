"""Freeze E23 OpenEA pools using the existing configured retrieval implementation."""

from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

from exact.experiments.openea import CASE_ID, label_training_candidates, verify_prepared
from exact.experiments.recovery import implementation_identity
from exact.utils.fitted_artifacts import fingerprint, freeze_json
from exact.utils.provenance import sha256_file


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _verified(binding: dict[str, str]) -> Path:
    path = Path(binding["path"])
    if sha256_file(path) != binding["sha256"]:
        raise ValueError("OpenEA pool input changed: " + str(path))
    return path


def training_support(frame: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    """Audit the existing graph recipe's source folds; never repair their labels or rows."""
    labeled = frame.loc[frame.confirmed_label.isin([0, 1])]
    groups = sorted(set(labeled.Src.astype(str)))
    random.Random(int(seed)).shuffle(groups)

    def counts(part: pd.DataFrame) -> dict[str, int]:
        return {
            "source_groups": int(part.Src.nunique()),
            "positive": int(part.confirmed_label.eq(1).sum()),
            "negative": int(part.confirmed_label.eq(0).sum()),
        }

    problems = []
    total = counts(labeled)
    if len(groups) < 2 or not total["positive"] or not total["negative"]:
        problems.append("Need at least two labeled source groups and both permitted label classes")
    folds = []
    for fold in range(min(3, len(groups))):
        heldout = set(groups[fold :: min(3, len(groups))])
        mask = labeled.Src.astype(str).isin(heldout)
        record: dict[str, Any] = {
            "fold": fold,
            "training": counts(labeled.loc[~mask]),
            "heldout": counts(labeled.loc[mask]),
        }
        for role in ("training", "heldout"):
            if not record[role]["positive"] or not record[role]["negative"]:
                problems.append(
                    f"Graph fold {fold} {role} lacks positive or permitted negative examples"
                )
        folds.append(record)
    return {
        "usable": not problems,
        "problems": problems,
        "seed": seed,
        "recipe": "inductive_graph_statistics_logistic_l2_0.01_v1",
        "total": total,
        "unknown_pairs": int(frame.confirmed_label.isna().sum()),
        "source_groups_without_labels": int(frame.Src.nunique()) - len(groups),
        "folds": folds,
    }


def _generate(config, case, sources, output, *, cap, seed, device):
    """Use exactly the dataset's normal generated-candidate path, without a reference."""
    import torch
    from exact.core.entities.registry import ComponentRegistry, ComponentType

    config.resolve_dependencies()
    ComponentRegistry.get(ComponentType.SEED_SETTER, "SeedSetter")(seed)
    factory = config.dataset_runtime
    assert factory is not None
    params = config.dataset_params.model_dump(mode="python")
    exact_policy = config.matching.anchor_rescoring.exact_policy
    if exact_policy is not None:
        params["filter_exact_matches"] = exact_policy == "hard"
    candidates = config.candidates.model_dump(mode="python")
    dataset = factory(
        output_path=output,
        logger=logging.getLogger("exact.openea"),
        cache_ok=True,
        device=torch.device(device),
        request_seed=seed,
        candidate_generation_params=candidates,
        input_format="csv-kg",
        source_options={},
        target_options={},
        entity_kinds=["individual"],
        **params,
    )
    dataset.load_ontologies(Path(case["source"]["path"]), Path(case["target"]["path"]))
    dataset.freeze_source_universe(sources, cap=cap, seed=seed)
    dataset.load_candidates(None, device=torch.device(device), **candidates)
    frame, _ = dataset.candidate_recall_frames()
    if frame is None:
        raise ValueError("OpenEA retrieval did not publish its candidate frame")
    return frame.copy(), list(dataset.eligible_source_iris), dataset.candidate_pool_manifest


def prepare_pools(config_path: Path, prepared: Path, destination: Path, *, device="cuda:0"):
    """Generate once per role; resume completed roles and bind the same pools to all arms."""
    from exact.core.entities.configs.config import ConfigModel

    prepared, destination = Path(prepared).resolve(), Path(destination).resolve()
    metadata = verify_prepared(prepared)
    if metadata["recipe"]["schema_version"] != 2:
        raise ValueError("OpenEA pools require source-kind-aware preparation revision 2")
    fragment = json.loads((prepared / "bindings-fragment.json").read_text())
    case = fragment["cases"][CASE_ID]
    scope = json.loads(_verified(fragment["e23_training_label_scope"]).read_text())
    config = ConfigModel.load_config(config_path)
    if config.candidates.encoder_finetune.mode != "off":
        raise ValueError("OpenEA pools require the frozen label-free retrieval recipe")
    generator = {
        "config": _binding(config_path),
        "preparation": _binding(prepared / "preparation.json"),
        "source": case["source"],
        "target": case["target"],
        "training_label_scope": fragment["e23_training_label_scope"],
        "candidates": config.candidates.model_dump(mode="json"),
        "dataset": config.dataset_params.model_dump(mode="json"),
        "exact_policy": config.matching.anchor_rescoring.exact_policy,
        "entity_kinds": ["individual"],
        "seed": metadata["recipe"]["seed"],
        "source_cap": metadata["recipe"]["source_cap"],
        "training_cap": metadata["recipe"]["training_cap"],
        "implementation": implementation_identity(
            Path(__file__).resolve().parents[2],
            ["exact/experiments/openea_pools.py", "exact/impl/datasets/__init__.py"],
        )["sha256"],
    }
    destination.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for role in ("train", "valid"):
        source_binding = scope["training_sources" if role == "train" else "validation_sources"]
        sources = _verified(source_binding).read_text().splitlines()
        cap = metadata["recipe"]["training_cap" if role == "train" else "source_cap"]
        identity = fingerprint(
            {"generator": generator, "role": role, "sources": source_binding, "cap": cap}
        )
        receipt_path = destination / f"{role}.pool.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["identity"] != identity:
                raise ValueError("Immutable OpenEA retrieval recipe changed")
            _verified(receipt["pool"])
            _verified(receipt["sources"])
        else:
            # Labels cannot enter this boundary: _generate receives public KGs
            # and the frozen source universe, never train/validation references.
            knowledge = {key: case[key] for key in ("source", "target")}
            frame, selected, candidate_manifest = _generate(
                config,
                knowledge,
                sources,
                destination / "retrieval" / role,
                cap=cap,
                seed=generator["seed"],
                device=device,
            )
            if not set(frame.Src.astype(str)) <= set(selected) <= set(sources):
                raise ValueError("Retrieved OpenEA pool escaped its frozen source universe")
            if frame[["Src", "Tgt"]].duplicated().any():
                raise ValueError("Generated OpenEA pool contains duplicate pairs")
            if role == "train":
                train = scope["official_train_links"]
                frame = label_training_candidates(
                    frame, _verified(train), expected_sha256=train["sha256"]
                )
            path = destination / f"{role}.candidates.tsv"
            temporary = path.with_suffix(".partial")
            frame.to_csv(temporary, sep="\t", index=False)
            temporary.replace(path)
            source_path = destination / f"{role}.sources.txt"
            source_path.write_text("\n".join(sorted(selected)) + "\n")
            receipt = {
                "identity": identity,
                "pool": _binding(path),
                "sources": _binding(source_path),
                "candidate_manifest": candidate_manifest,
                "pairs": len(frame),
                "source_groups": len(selected),
                "gold_insertions": 0,
                "confirmed_positive": (
                    int(frame.confirmed_label.eq(1).sum()) if role == "train" else None
                ),
                "confirmed_negative": (
                    int(frame.confirmed_label.eq(0).sum()) if role == "train" else None
                ),
                "unknown": int(frame.confirmed_label.isna().sum()) if role == "train" else None,
            }
            if role == "train":
                receipt["training_support"] = training_support(frame, seed=generator["seed"])
            freeze_json(receipt_path, receipt)
        if role == "train" and not receipt.get("training_support", {}).get("usable"):
            raise ValueError(
                "OpenEA retrieved training pool lacks usable source-fold supervision; "
                f"inspect {receipt_path}. Retained rows are unchanged; do not regenerate "
                "with different seeds, insert gold, or invent negatives."
            )
        receipts[role] = receipt
    if set(Path(receipts["train"]["sources"]["path"]).read_text().splitlines()) & set(
        Path(receipts["valid"]["sources"]["path"]).read_text().splitlines()
    ):
        raise ValueError("Generated OpenEA train/validation source overlap")
    provenance = {"schema_version": 1, "generator": generator, "roles": receipts}
    proof = destination / "pools.json"
    freeze_json(proof, provenance)
    amended = copy.deepcopy(fragment)
    case = amended["cases"][CASE_ID]
    case["candidates"]["train"] = receipts["train"]["pool"]
    case["frozen_global_candidates"]["valid"] = receipts["valid"]["pool"]
    case["source_universe"] = receipts["valid"]["sources"]
    # These controls already ran in _generate. Provided-pool loading must not
    # apply retrieval/finetuning/reranking a second time or reject the frozen pool.
    case["overlay"]["candidates"] = {
        "fusion": {"mode": "max"},
        "adaptive_k": {"enabled": False},
        "encoder_finetune": {"mode": "off"},
        "cross_encoder": {"mode": "off"},
        "multi_view": {"mode": "labels"},
    }
    case["transformation"] += " Frozen retrieval provenance SHA256=" + sha256_file(proof)
    amended["e23_pool_provenance"] = _binding(proof)
    freeze_json(destination / "bindings-fragment.json", amended)
    return amended


def validate_pool_bindings(fragment: dict[str, Any]) -> dict[str, Any] | None:
    """Check the E23-only provenance chain without running retrieval or reading gold."""
    case = fragment["cases"][fragment["e23_natural_case"]]
    scope_binding = fragment.get("e23_training_label_scope")
    if not scope_binding:
        raise ValueError("OpenEA E23 requires its verified training-label scope")
    scope = json.loads(_verified(scope_binding).read_text())
    if scope.get("kind") != "openea_bijective_training_endpoints" or scope[
        "training_reference"
    ] != case["references"].get("train"):
        raise ValueError("OpenEA E23 training scope/reference mismatch")
    proof = fragment.get("e23_pool_provenance")
    if proof is None:
        if case.get("candidates", {}).get("train") or case.get("frozen_global_candidates", {}).get(
            "valid"
        ):
            raise ValueError("OpenEA pool bindings lack retrieval provenance")
        return None
    value: dict[str, Any] = json.loads(_verified(proof).read_text())
    if value["generator"]["training_label_scope"] != scope_binding:
        raise ValueError("OpenEA frozen pool training scope changed")
    if any(value["generator"][side] != case[side] for side in ("source", "target")):
        raise ValueError("OpenEA frozen pool KG identity changed")
    if not value["roles"]["train"].get("training_support", {}).get("usable"):
        raise ValueError("OpenEA frozen pool lacks usable source-fold supervision")
    if (
        value["roles"]["train"]["pool"] != case["candidates"].get("train")
        or value["roles"]["valid"]["pool"] != case["frozen_global_candidates"].get("valid")
        or value["roles"]["valid"]["sources"] != case["source_universe"]
    ):
        raise ValueError("OpenEA frozen pools differ from their case binding")
    for role in ("train", "valid"):
        _verified(value["roles"][role]["pool"])
        _verified(value["roles"][role]["sources"])
    return value
