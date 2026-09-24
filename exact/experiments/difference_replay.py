"""E24 controlled difference interventions through frozen analytic decisions."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import torch

from exact.core.entities.mappings import EntityMapping
from exact.experiments.evidence_diagnostics import paired_decision_changes
from exact.impl.extraction import extract_global_alignment
from exact.utils.fitted_artifacts import freeze_json
from exact.utils.provenance import sha256_file


def replay_analytic_fusion(scorer, channels):
    """Use the scorer's authority policy, preserving nested lexical/structural fusion."""
    values = {
        row["name"]: tuple(
            torch.tensor([row[key]], device="cpu", dtype=torch.float32)
            for key in ("score", "quality", "active")
        )
        for row in channels
    }
    label, string = values.pop("label"), values.pop("strsim")
    lex = label
    if scorer.strsim_enabled:
        if scorer.strsim_config["placement"] == "folded_into_lexical":
            wins = string[2].bool() & (~label[2].bool() | (string[0] > label[0]))
            lex = (
                torch.where(wins, string[0], label[0]),
                torch.where(wins, string[1], label[1]),
                label[2].bool() | string[2].bool(),
            )
        else:
            authority = [
                scorer._sigma_authority(*[s, q, a.bool()], channel=name)
                for name, (s, q, a) in (("label", label), ("strsim", string))
            ]
            total = authority[0] + authority[1]
            lex = (
                torch.where(
                    total > 1e-8,
                    (authority[0] / total.clamp_min(1e-8)) * label[0]
                    + (authority[1] / total.clamp_min(1e-8)) * string[0],
                    torch.full_like(total, scorer.tau),
                ),
                torch.where(
                    total > 1e-8,
                    (authority[0] / total.clamp_min(1e-8)) * label[1]
                    + (authority[1] / total.clamp_min(1e-8)) * string[1],
                    torch.zeros_like(total),
                ),
                label[2].bool() | string[2].bool(),
            )
    authorities = [
        scorer._sigma_authority(s, q, a.bool(), channel=name) for name, (s, q, a) in values.items()
    ]
    total = sum(authorities, torch.zeros(1, device="cpu"))
    weights = [
        torch.where(total > 1e-8, item / total.clamp_min(1e-8), torch.zeros_like(total))
        for item in authorities
    ]
    structural_score = torch.where(
        total > 1e-8,
        torch.stack([weight * item[0] for weight, item in zip(weights, values.values())]).sum(0),
        torch.full_like(total, scorer.tau),
    )
    structural_quality = torch.where(
        total > 1e-8,
        torch.stack([weight * item[1] for weight, item in zip(weights, values.values())]).sum(0),
        torch.zeros_like(total),
    )
    structural_active = torch.stack([item[2].bool() for item in values.values()]).any(0)
    lex_authority = scorer._sigma_authority(lex[0], lex[1], lex[2].bool(), channel="lex")
    structural_authority = scorer._sigma_authority(
        structural_score, structural_quality, structural_active, channel="struct"
    )
    both = (lex_authority > 1e-8) & (structural_authority > 1e-8)
    weight = structural_authority / (lex_authority + structural_authority).clamp_min(1e-8)
    score = torch.full_like(total, scorer.tau)
    score = torch.where((lex_authority > 1e-8) & ~both, lex[0], score)
    score = torch.where((structural_authority > 1e-8) & ~both, structural_score, score)
    score = torch.where(both, (1 - weight) * lex[0] + weight * structural_score, score)
    indecision, disagreement = scorer._uncertainty_components(
        score, lex[0], structural_score, lex[1], structural_quality
    )
    return {
        "S_base": float(score.item()),
        "U": float(torch.maximum(indecision, disagreement).item()),
    }


def _extract(rows, scores, policy, protected):
    mappings = [
        EntityMapping(
            str(row["Src"]), str(row["Tgt"]), score=scores[(str(row["Src"]), str(row["Tgt"]))]
        )
        for row in rows
    ]
    mode = policy.get("extraction", {}).get("mode", "greedy")
    threshold = policy["threshold"]
    if mode != "greedy":
        return {
            (str(item.head), str(item.tail))
            for item in extract_global_alignment(
                mappings,
                mode=mode,
                threshold=threshold,
                protected_pairs=protected,
                source_cardinality=policy.get("source_cardinality"),
                target_cardinality=policy.get("target_cardinality"),
                assignment_component_cap=policy.get("extraction", {}).get(
                    "assignment_component_cap", 500
                ),
            ).mappings
        }
    mappings = (
        EntityMapping.filter_entity_mappings_by_score(mappings, threshold)
        if threshold is not None
        else mappings
    )
    for axis, method in (
        ("source", EntityMapping.filter_top_n_entity_mappings),
        ("target", EntityMapping.filter_top_n_target_entity_mappings),
    ):
        if policy.get(axis + "_cardinality") is not None:
            mappings = method(mappings, policy[axis + "_cardinality"], protected_pairs=protected)
    return {(str(item.head), str(item.tail)) for item in mappings}


def write_difference_replay(runner, frame, explanations, emitted, protected, policy):
    """Freeze diagnostic mappings without reading any reference labels."""
    scorer = runner.model
    if not getattr(scorer, "diff_config", {}).get("controlled_perturbations"):
        return None
    if (
        policy.get("local_alignment")
        or policy.get("threshold_origin") not in {None, "configured"}
        or any(getattr(model, "enabled", True) for model in runner.models[1:])
        or getattr(scorer, "_fusion_artifact", None) is not None
        or (
            getattr(scorer, "use_llm", False)
            and scorer.llm_experiment_config.get("gate", {}).get("mode") != "off"
        )
    ):
        raise ValueError(
            "E24 fixed-decision replay does not support active extra heads, adaptive thresholds or decision LLMs"
        )
    rows = frame.to_dict("records")
    population = {(str(row["Src"]), str(row["Tgt"])) for row in rows}
    score_sets, uncertainty, inventory = {}, {}, {}
    names = None
    for row in rows:
        pair = (str(row["Src"]), str(row["Tgt"]))
        if pair in protected:
            continue
        diagnostics = explanations.get(pair, {}).get("experiment_diagnostics", {})
        raw, controlled = diagnostics.get("difference_replay"), diagnostics.get(
            "difference", {}
        ).get("controlled_perturbations")
        if not raw or not controlled:
            raise ValueError(
                "E24 replay requires raw channels and controlled variants for every scored pair"
            )
        current_names = [item["name"] for item in controlled["variants"]]
        if names is not None and names != current_names:
            raise ValueError("E24 replay perturbation inventories differ")
        names = current_names
        inventory[json.dumps(pair)] = controlled["inventory_sha256"]
        for variant in controlled["variants"]:
            channels = deepcopy(raw["channels"])
            diff = next(item for item in channels if item["name"] == "diff")
            diff.update({key: variant[key] for key in ("score", "quality", "active")})
            replay = replay_analytic_fusion(scorer, channels)
            if variant["name"] == "natural" and any(
                not math.isfinite(replay[key])
                or not math.isfinite(raw[key])
                or abs(replay[key] - raw[key]) > 1e-6
                for key in ("S_base", "U")
            ):
                raise ValueError(
                    "E24 natural numerical replay failed before perturbation decisions"
                )
            score_sets.setdefault(variant["name"], {})[pair] = replay["S_base"]
            uncertainty.setdefault(variant["name"], {})[json.dumps(pair)] = replay["U"]
    if not names:
        return None
    emitted = set(emitted)
    variants = {}
    for name in names:
        scores = score_sets[name]
        scores.update(
            {
                (str(row["Src"]), str(row["Tgt"])): float(row["S_final"])
                for row in rows
                if (str(row["Src"]), str(row["Tgt"])) in protected
            }
        )
        selected = _extract(rows, scores, policy, protected)
        if name == "natural" and selected != emitted:
            raise ValueError(
                "E24 natural emitted mapping replay failed before perturbation decisions"
            )
        variants[name] = {
            "emitted": sorted(selected),
            "uncertainty": uncertainty[name],
            "changed_sources": sorted({pair[0] for pair in selected ^ emitted}),
        }
    payload = {
        "schema_version": 1,
        "scope": "frozen_analytic_pair_threshold_and_extraction",
        "natural_parity": True,
        "reference_labels_used": False,
        "population": sorted(population),
        "source_universe": sorted(
            set(getattr(runner.dataset, "eligible_source_iris", []) or [])
            | {pair[0] for pair in population}
        ),
        "protected": sorted(protected),
        "policy": policy,
        "inventories": inventory,
        "variants": variants,
    }
    path = runner.output_dir / "difference_replay.json"
    freeze_json(path, payload)
    return path


def evaluate_difference_replay(cell):
    path = cell.output_dir / "difference_replay.json"
    if not path.is_file():
        return None
    if cell.split_role != "development" or cell.reference_role != "valid":
        raise ValueError("Controlled E24 outcomes are development diagnostics only")
    replay = json.loads(path.read_text())
    data = cell.resolved_config["data"]
    reference_path = Path(data.get("root") or ".") / data["refs"][cell.reference_role]
    from exact.utils.data import read_table

    reference = read_table(reference_path)
    pairs = [
        tuple(map(str, row)) for row in reference.iloc[:, :2].itertuples(index=False, name=None)
    ]
    population = {tuple(pair) for pair in replay["population"]}
    if "confirmed_label" in reference:
        if not reference.confirmed_label.isin([0, 1]).all():
            raise ValueError("Explicit diagnostic candidate labels must be 0 or 1")
        known = {
            pair: bool(label)
            for pair, label in zip(pairs, reference.confirmed_label)
            if pair in population
        }
        positives = {pair for pair, label in known.items() if label}
    else:
        positives = set(pairs)
        known = {pair: True for pair in population & positives}
    if cell.negative_label_policy == "complete_reference":
        known.update({pair: pair in positives for pair in population})
    natural_selected = {tuple(row) for row in replay["variants"]["natural"]["emitted"]}
    baseline = {pair: pair in natural_selected for pair in population}
    reports = {}
    for name, variant in replay["variants"].items():
        selected = {tuple(row) for row in variant["emitted"]}
        reports[name] = paired_decision_changes(
            baseline, {pair: pair in selected for pair in population}, known
        )
    result = {
        "schema_version": 1,
        "diagnostic_only": True,
        "natural_parity": True,
        "replay_sha256": sha256_file(path),
        "reference_sha256": sha256_file(reference_path),
        "negative_label_policy": cell.negative_label_policy,
        "variants": reports,
    }
    destination = cell.output_dir / "diagnostics/difference-decisions.json"
    freeze_json(destination, result)
    return {"path": str(destination), "sha256": sha256_file(destination)}
