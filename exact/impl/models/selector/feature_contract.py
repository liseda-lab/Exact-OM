"""Feature meanings shared by TRAIN fitting and frozen selector application."""

import numpy as np
import pandas as pd

EVIDENCE_GAP_CHANNELS = ("s_label", "S_struct", "s_hier", "s_sim", "s_attr")
RETRIEVAL_FEATURES = ("cand_sim", "cand_sim_prob", "cand_share_log_ratio")
SPECIFICITY_FEATURES = tuple(
    f"{side}_{name}"
    for side in ("src", "tgt")
    for name in ("obj_ic_mean", "hier_total_count", "obj_count", "attr_count")
)
OPTIONAL_FEATURES = RETRIEVAL_FEATURES + SPECIFICITY_FEATURES
FEATURE_CONTRACT_VERSION = 1


def restore_specificity_metadata(frame, dataset):
    """Hydrate legacy TRAIN rows without touching their saved numerical scores."""
    missing = [name for name in SPECIFICITY_FEATURES if name not in frame]
    if not missing or not hasattr(dataset, "_pair_adaptive_metric_columns"):
        return frame
    keys = [name for name in ("Src", "Tgt", "SrcKind", "TgtKind") if name in frame]
    metadata = dataset.get_features(frame[keys])
    if len(metadata) != len(frame) or not metadata[keys].reset_index(drop=True).equals(
        frame[keys].reset_index(drop=True)
    ):
        raise ValueError("Pair-adaptive TRAIN metadata changed pair identity or order")
    result = frame.copy()
    for name in missing:
        if name not in metadata:
            raise ValueError(f"Pair-adaptive TRAIN metadata missing {name}")
        result[name] = metadata[name].to_numpy()
    return result


def feature_contract(frame):
    """Freeze supplied metadata; an unavailable retrieval score is never a pair score."""
    available = []
    for name in OPTIONAL_FEATURES:
        if name not in frame:
            continue
        if not np.isfinite(pd.to_numeric(frame[name], errors="coerce")).all():
            raise ValueError(f"Selector feature {name} must contain finite numeric values")
        available.append(name)
    return {
        "version": FEATURE_CONTRACT_VERSION,
        "gap_channels": list(EVIDENCE_GAP_CHANNELS),
        "available_metadata": available,
        "unavailable_metadata": [name for name in OPTIONAL_FEATURES if name not in available],
    }


def apply_feature_contract(frame, contract):
    """Mask untrained metadata and fail if a trained dependency is missing."""
    available = contract.get("available_metadata", [])
    if (
        contract.get("version") != FEATURE_CONTRACT_VERSION
        or contract.get("gap_channels") != list(EVIDENCE_GAP_CHANNELS)
        or len(set(available)) != len(available)
        or not set(available) <= set(OPTIONAL_FEATURES)
        or contract.get("unavailable_metadata")
        != [name for name in OPTIONAL_FEATURES if name not in available]
    ):
        raise ValueError("Fitted selector feature contract is incompatible; refit required")
    result = frame.copy()
    for name in OPTIONAL_FEATURES:
        if name not in available:
            result[name] = 0.0
        elif (
            name not in frame or not np.isfinite(pd.to_numeric(frame[name], errors="coerce")).all()
        ):
            raise ValueError(f"Fitted selector requires finite feature {name}")
    return result
