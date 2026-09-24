"""Deterministic E24 evidence perturbations and paired, label-safe diagnostics.

These inventories mutate bounded scorer-visible evidence, never ontology files or
reference labels. A channel-only replay is not an end-to-end decision experiment.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Callable, Mapping

import torch


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def difference_perturbation_inventory(source, target, *, pair_id: str) -> dict:
    """Freeze separate side-specific operations, including explicit no-op cases."""
    original = {"source": deepcopy(list(source)), "target": deepcopy(list(target))}
    variants: list[dict[str, Any]] = [
        {"name": "natural", "operation": "control", **deepcopy(original)}
    ]
    for side in ("source", "target"):
        for operation in (
            "deletion",
            "duplication",
            "irrelevant_insertion",
            "equivalent_serialization",
        ):
            changed = deepcopy(original)
            items = changed[side]
            selected = min(range(len(items)), key=lambda i: (_hash(items[i]), i)) if items else None
            applicable = selected is not None or operation == "irrelevant_insertion"
            if operation == "deletion" and selected is not None:
                items.pop(selected)
            elif operation == "duplication" and selected is not None:
                items.append(deepcopy(items[selected]))
            elif operation == "irrelevant_insertion":
                # A pair-specific namespace avoids accidental overlap with genuine
                # predicates or a declared semantic incompatibility rule.
                iri = "urn:exact:diagnostic:irrelevant:" + _hash([pair_id, side, original])
                items.append(
                    {
                        "triple": [iri + ":subject", iri, iri + ":object"],
                        "subject_iri": iri + ":subject",
                        "rel_iri": iri,
                        "object_iri": iri + ":object",
                        "score": 0.2,
                        "provenance_status": "controlled_diagnostic_insertion",
                    }
                )
            elif operation == "equivalent_serialization":
                # JSON member order and triple tuple/list are semantically inert;
                # literal spelling, language, and IRI identity are left intact.
                changed[side] = [dict(reversed(list(item.items()))) for item in reversed(items)]
                for item in changed[side]:
                    item["triple"] = list(item.get("triple", ()))
            variants.append(
                {
                    "name": f"{operation}_{side}",
                    "operation": operation,
                    "side": side,
                    "selected_index": selected,
                    "applicable": applicable,
                    **changed,
                }
            )
    variants.append(
        {
            "name": "side_reversal",
            "operation": "side_reversal",
            "source": deepcopy(original["target"]),
            "target": deepcopy(original["source"]),
        }
    )
    result = {
        "schema_version": 1,
        "pair_id": pair_id,
        "scope": "bounded_difference_channel_evidence",
        "original_sha256": _hash(original),
        "variants": variants,
    }
    result["sha256"] = _hash(result)
    return result


def replay_difference_diagnostics(
    scorer, inventory: Mapping, *, decision: Callable | None = None, support_matrix=None
) -> dict:
    """Replay the actual channel; optional caller supplies a frozen decision rule.

    The caller accounts for this scoring and persists this record at the same
    durable boundary as the corresponding unperturbed pair. It must never feed
    perturbation outcomes back into the private final-reference evaluation.
    """
    body = {key: value for key, value in inventory.items() if key != "sha256"}
    if inventory.get("sha256") != _hash(body):
        raise ValueError("difference perturbation inventory hash mismatch")
    direction = scorer.diff_config.get("relation_interpretation")
    if scorer.diff_config.get("formulation") == "asymmetric" and direction not in {"<", ">"}:
        raise ValueError("asymmetric diagnostic requires a separately bound typed interpretation")
    original = inventory["variants"][0]
    matrix = support_matrix.detach().cpu() if support_matrix is not None else None
    if matrix is not None and tuple(matrix.shape) != (
        len(original["source"]),
        len(original["target"]),
    ):
        raise ValueError("difference support matrix does not match original inventory")
    rows = []
    for variant in inventory["variants"]:
        # Reversal swaps the typed relation too. The declaration is restored even
        # if an encoder or scorer raises, so later pairs cannot inherit it.
        try:
            if variant["operation"] == "side_reversal" and direction in {"<", ">"}:
                scorer.diff_config["relation_interpretation"] = ">" if direction == "<" else "<"
            replay_matrix = matrix
            if matrix is not None:
                if variant["operation"] == "side_reversal":
                    replay_matrix = matrix.T
                elif variant["operation"] != "control":
                    side = variant["side"]
                    positions = list(range(len(original[side])))
                    selected = variant["selected_index"]
                    if variant["operation"] == "deletion" and selected is not None:
                        positions.pop(selected)
                    elif variant["operation"] == "duplication" and selected is not None:
                        positions.append(selected)
                    elif variant["operation"] == "equivalent_serialization":
                        positions.reverse()
                    replay_matrix = (
                        matrix[positions, :] if side == "source" else matrix[:, positions]
                    )
                    if variant["operation"] == "irrelevant_insertion":
                        shape = (1, matrix.shape[1]) if side == "source" else (matrix.shape[0], 1)
                        replay_matrix = torch.cat(
                            [replay_matrix, torch.zeros(shape, dtype=matrix.dtype)],
                            dim=0 if side == "source" else 1,
                        )
            payload = scorer._score_difference_channel(
                variant["source"], variant["target"], support_mat=replay_matrix, verbalize=False
            )
        finally:
            scorer.diff_config["relation_interpretation"] = direction
        row = {
            "name": variant["name"],
            "applicable": variant.get("applicable", True),
            "source_count": len(variant["source"]),
            "target_count": len(variant["target"]),
            "active": bool(payload.get("src_selected") or payload.get("tgt_selected")),
            **{
                key: payload.get(key)
                for key in (
                    "score",
                    "quality",
                    "formulation",
                    "diff_pivot_reason",
                    "evidence_states",
                    "unsupported_mass_src",
                    "unsupported_mass_tgt",
                    "c_x",
                    "c_y",
                )
            },
        }
        if decision is not None:
            row["decision"] = decision(payload, variant)
        rows.append(row)
    baseline = rows[0]
    for row in rows:
        row["score_delta"] = row["score"] - baseline["score"]
        row["quality_delta"] = row["quality"] - baseline["quality"]
    return {
        "schema_version": 1,
        "pair_id": inventory["pair_id"],
        "inventory_sha256": inventory["sha256"],
        "scope": inventory["scope"],
        "original": {"source": original["source"], "target": original["target"]},
        "support_semantics": (
            "frozen_matrix_with_zero_support_irrelevant_insertion"
            if matrix is not None
            else "recomputed_support"
        ),
        "support_matrix": matrix.tolist() if matrix is not None else None,
        "perturbation_operations": [
            {key: row[key] for key in ("name", "operation", "side", "selected_index") if key in row}
            for row in inventory["variants"]
        ],
        "decision_scope": "caller_declared_frozen_rule" if decision else "channel_only",
        "formulation": scorer.diff_config.get("formulation", "normalised"),
        "semantics_sha256": _hash(scorer.diff_config),
        "variants": rows,
    }


def paired_decision_changes(baseline: Mapping, treatment: Mapping, known_labels: Mapping) -> dict:
    """Count correction/harm only where an explicit known binary label exists.

    Keys are caller-owned pair/source IDs, values are bool decisions. Unlabelled
    candidate pairs remain in the denominator and are never negative by absence.
    """
    if set(baseline) != set(treatment):
        raise ValueError("paired decision diagnostics require identical populations")
    if any(type(label) is not bool for label in known_labels.values()):
        raise ValueError("diagnostic labels must be explicit binary labels")
    if any(type(value) is not bool for rows in (baseline, treatment) for value in rows.values()):
        raise ValueError("diagnostic decisions must be binary")
    corrected = harmed = changed_unknown = known = 0
    for key, before in baseline.items():
        after = treatment[key]
        if key not in known_labels:
            changed_unknown += before != after
            continue
        known += 1
        corrected += before != known_labels[key] and after == known_labels[key]
        harmed += before == known_labels[key] and after != known_labels[key]
    return {
        "population": len(baseline),
        "known_label_count": known,
        "unknown_label_count": len(baseline) - known,
        "corrected": corrected,
        "harmed": harmed,
        "correction_minus_harm": corrected - harmed,
        "changed_unknown": changed_unknown,
    }
