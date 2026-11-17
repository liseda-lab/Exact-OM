"""Apply immutable experimental judgments without rerouting or making API calls."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch


def replay_identity():
    path = os.getenv("EXACT_COMPONENT_INFERENCE_JUDGMENTS")
    sha = os.getenv("EXACT_COMPONENT_INFERENCE_JUDGMENTS_SHA256")
    if bool(path) != bool(sha):
        raise ValueError("Inference judgments require a frozen manifest and checksum")
    return {"sha256": sha, "model_id": os.getenv("EXACT_COMPONENT_JUDGE_MODEL")} if path else None


def apply_judgments(model, sources, targets, base, uncertainty):
    """Return masks/probabilities/weights on the unchanged complete candidate pool."""
    identity = replay_identity()
    if identity is None or getattr(model, "_numerical_scoring_role", None) == "train":
        return None
    dataset = getattr(model, "_attached_dataset", None)
    frame = dataset._active_dataframe()
    # Native datasets replace/invalidate the active frame when their pool changes.
    # Recheck once per such population, not once per scoring batch.
    population = (identity, id(dataset), id(frame), len(frame))
    cached = getattr(model, "_component_judgments", None)
    if cached is None or cached[0] != population:
        raw = Path(os.environ["EXACT_COMPONENT_INFERENCE_JUDGMENTS"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != identity["sha256"]:
            raise ValueError("Frozen inference judgment manifest changed")
        payload = json.loads(raw)
        if (
            payload.get("schema") != "exact-component-inference-judgments-v1"
            or not identity["model_id"]
            or payload.get("model_id") != identity["model_id"]
            or payload.get("dataset_signature") != getattr(dataset, "dataset_signature", None)
            or payload.get("provenance", {}).get("role") != "valid"
        ):
            raise ValueError("Inference judgments do not match this model/population")
        rows = payload["predictions"]
        probabilities = {(r["Src"], r["Tgt"]): r["probability"] for r in rows}
        expected = {(r["Src"], r["Tgt"]) for r in payload["expected_pairs"]}
        if (
            len(probabilities) != len(rows)
            or set(probabilities) != expected
            or len(expected) != len(payload["expected_pairs"])
        ):
            raise ValueError("Inference judgment support is incomplete or duplicated")
        if any(
            isinstance(p, bool)
            or not isinstance(p, (float, int))
            or not math.isfinite(p)
            or not 0 <= p <= 1
            for p in probabilities.values()
        ):
            raise ValueError("Invalid cached probability")
        if (
            payload["policy"] not in {"beta_u", "constant_0_5"}
            or not payload.get("candidate_pool_sha256")
            or not payload.get("request_bank_sha256")
        ):
            raise ValueError("Inference replay lacks frozen policy and candidate provenance")
        # Dataset identity alone is insufficient: changed retrieval must not reuse
        # an old shortlist. Check the actual full candidate inventory once.
        pairs = sorted(
            {(str(s), str(t)) for s, t in frame[["Src", "Tgt"]].itertuples(index=False, name=None)}
        )
        from exact.utils.fitted_artifacts import fingerprint

        if fingerprint(pairs) != payload["candidate_pool_sha256"] or not expected <= set(pairs):
            raise ValueError("Frozen judgment bank belongs to a different candidate pool")
        cached = (population, payload, probabilities)
        model._component_judgments = cached
    _, payload, probabilities = cached
    keys = list(zip(map(str, sources), map(str, targets)))
    mask = torch.tensor(
        [key in probabilities for key in keys], device=base.device, dtype=torch.bool
    )
    values = torch.tensor(
        [probabilities.get(key, 0.0) for key in keys], device=base.device, dtype=torch.float32
    )
    weights = (
        torch.full_like(uncertainty, 0.5)
        if payload["policy"] == "constant_0_5"
        else (model.beta * uncertainty).clamp(0, 1)
    )
    return mask, values, weights, payload
