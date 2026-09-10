"""Bounded same-family fusion fitting over frozen raw channel evidence."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from .fitting import fingerprint, freeze_json, safe_training_labels


def fusion_scores(
    channels, names, parameters, *, mode, strsim_placement="off", return_components=False
):
    scores, qualities, active = channels
    tau, gamma, weights = parameters

    def authority(score, quality, mask, weight):
        if mode == "learned_global":
            return quality * mask * weight
        margin = (score - tau).abs()
        return (
            quality
            * mask
            * torch.where(margin > 0, margin.clamp_min(1e-8).pow(gamma), 0.0)
            * weight
        )

    def combine(indices):
        idx = torch.tensor(indices)
        mass = authority(scores[:, idx], qualities[:, idx], active[:, idx], weights[idx])
        total = mass.sum(1)
        normalized = mass / total[:, None].clamp_min(1e-8)
        return (
            torch.where(total > 1e-8, (scores[:, idx] * normalized).sum(1), tau),
            (qualities[:, idx] * normalized).sum(1),
            active[:, idx].any(1),
            total,
        )

    lexical = [names.index("label")]
    if strsim_placement == "channel":
        lexical.append(names.index("strsim"))
    lex_score, lex_quality, lex_active, lex_mass = combine(lexical)
    if strsim_placement == "folded_into_lexical":
        index = names.index("strsim")
        wins = (scores[:, index] > lex_score) & active[:, index].bool()
        lex_score = torch.where(wins, scores[:, index], lex_score)
        lex_quality = torch.where(wins, qualities[:, index], lex_quality)
        lex_active = lex_active | active[:, index].bool()
    if strsim_placement == "off":
        index = names.index("label")
        lex_score, lex_quality, lex_active = (
            scores[:, index],
            qualities[:, index],
            active[:, index].bool(),
        )
    structural = [
        i for i, name in enumerate(names) if name not in {"label", "strsim", "lex", "struct"}
    ]
    struct_score, struct_quality, struct_active, struct_mass = combine(structural)
    if mode == "analytic_fitted":
        lex_weight = weights[names.index("label" if strsim_placement == "off" else "lex")]
        lex_mass = authority(lex_score, lex_quality, lex_active, lex_weight)
        struct_mass = authority(
            struct_score, struct_quality, struct_active, weights[names.index("struct")]
        )
    if mode == "learned_global" and strsim_placement != "channel":
        lex_mass = authority(lex_score, lex_quality, lex_active, weights[names.index("label")])
    lex_mass = torch.where(lex_mass > 1e-8, lex_mass, 0.0)
    struct_mass = torch.where(struct_mass > 1e-8, struct_mass, 0.0)
    fraction = struct_mass / (lex_mass + struct_mass).clamp_min(1e-8)
    score = torch.where(
        lex_mass + struct_mass > 0, (1 - fraction) * lex_score + fraction * struct_score, tau
    )
    if return_components:
        hierarchy = [i for i, name in enumerate(names) if name.startswith("hier__")]
        hierarchy_score = combine(hierarchy)[0] if hierarchy else torch.full_like(score, tau)
        return {
            "S_base": score,
            "S_final": score,
            "S_struct": struct_score,
            "Q_struct": struct_quality,
            "s_label_star": lex_score,
            "q_lex": lex_quality,
            "s_hier": hierarchy_score,
        }
    return score


def fit_fusion_artifact(
    frame, reference_pairs, path, *, mode, application, seed=17, strsim_placement="off", epochs=80
):
    frame, reference = safe_training_labels(frame, reference_pairs, application)
    if frame.empty or set(frame.Src.astype(str)) & set(application["source_ids"]):
        raise ValueError("Fusion fitting requires disjoint labeled training source groups")
    names = sorted(frame.iloc[0].fusion_channels) + ["lex", "struct"]
    channels = tuple(
        torch.tensor(
            [
                [row.get(name, {}).get(field, 0.0) for name in names]
                for row in frame.fusion_channels
            ],
            dtype=torch.float64,
        )
        for field in ("score", "quality", "active")
    )
    labels = torch.tensor(
        [float((row.Src, row.Tgt) in reference) for row in frame.itertuples()], dtype=torch.float64
    )
    source_ids = sorted(frame.Src.astype(str).unique())
    random.Random(seed).shuffle(source_ids)
    folds = min(5, len(source_ids))
    if folds < 2:
        raise ValueError("Grouped fusion fitting needs at least two training sources")
    if mode == "analytic_fitted":
        neutral = fusion_scores(
            channels,
            names,
            (torch.tensor(0.5), torch.tensor(2.0), torch.ones(len(names))),
            mode=mode,
            strsim_placement=strsim_placement,
        )
        if not torch.allclose(
            neutral, torch.tensor(frame.S_base.tolist(), dtype=torch.float64), atol=2e-6, rtol=2e-6
        ):
            raise ValueError("Neutral fitted fusion does not replay the shipped training scores")

    def fit(indices, regularization):
        with torch.enable_grad():
            parameters = torch.zeros(2 + len(names), dtype=torch.float64, requires_grad=True)
            optimizer = torch.optim.Adam([parameters], lr=0.03)
            for _ in range(epochs):
                optimizer.zero_grad()
                tau = (
                    0.5 + 0.1 * torch.tanh(parameters[0])
                    if mode == "analytic_fitted"
                    else torch.tensor(0.5)
                )
                gamma = (
                    0.5 + 2.5 * torch.sigmoid(parameters[1] + 0.4054651081081644)
                    if mode == "analytic_fitted"
                    else torch.tensor(2.0)
                )
                weights = torch.softmax(parameters[2:], 0) * len(names)
                prediction = fusion_scores(
                    tuple(channel[indices] for channel in channels),
                    names,
                    (tau, gamma, weights),
                    mode=mode,
                    strsim_placement=strsim_placement,
                )
                loss = (
                    prediction - labels[indices]
                ).square().mean() + regularization * parameters.square().mean()
                loss.backward()
                optimizer.step()
            return tau.detach(), gamma.detach(), weights.detach()

    losses, split_records, oof_by_penalty = [], [], {}
    identity = fingerprint(
        {
            "application": application,
            "rows": frame[["Src", "Tgt", "fusion_channels"]].to_dict("records"),
            "reference": sorted(reference),
            "epochs": epochs,
            "mode": mode,
            "placement": strsim_placement,
            "seed": seed,
        }
    )
    for regularization in (0.001, 0.01):
        squared_error = 0.0
        heldout_records = []
        for fold in range(folds):
            heldout = set(source_ids[fold::folds])
            train = [i for i, source in enumerate(frame.Src) if source not in heldout]
            test = [i for i, source in enumerate(frame.Src) if source in heldout]
            checkpoint = (
                Path(path).parent
                / (Path(path).name + ".folds")
                / f"{identity}-{regularization}-{fold}.json"
            )
            if checkpoint.exists():
                record = json.loads(checkpoint.read_text())
            else:
                parameters = fit(train, regularization)
                components = fusion_scores(
                    tuple(channel[test] for channel in channels),
                    names,
                    parameters,
                    mode=mode,
                    strsim_placement=strsim_placement,
                    return_components=True,
                )
                record = {
                    "fit_identity": identity,
                    "heldout_sources": sorted(heldout),
                    "rows": [
                        {
                            "Src": str(frame.iloc[index].Src),
                            "Tgt": str(frame.iloc[index].Tgt),
                            **{name: float(values[offset]) for name, values in components.items()},
                        }
                        for offset, index in enumerate(test)
                    ],
                }
                freeze_json(checkpoint, record)
            if record["fit_identity"] != identity or record["heldout_sources"] != sorted(heldout):
                raise ValueError("Fusion fold checkpoint source/recipe mismatch")
            predictions = torch.tensor(
                [row["S_final"] for row in record["rows"]], dtype=torch.float64
            )
            heldout_records.extend(record["rows"])
            squared_error += float((predictions - labels[test]).square().sum())
            if regularization == 0.001:
                split_records.append(
                    {
                        "fold": fold,
                        "train_sources": sorted(set(source_ids) - heldout),
                        "heldout_sources": sorted(heldout),
                    }
                )
        losses.append((squared_error / len(frame), regularization))
        oof_by_penalty[regularization] = heldout_records
    _, regularization = min(losses)
    tau, gamma, weights = fit(list(range(len(frame))), regularization)
    payload = {
        "schema_version": 1,
        "mode": mode,
        "dataset_signature": application["dataset_signature"],
        "dataset_lock_sha256": fingerprint(application),
        "candidate_pool_fingerprint": fingerprint(frame[["Src", "Tgt"]].values.tolist()),
        "seed": seed,
        "feature_schema": names,
        "negative_label_policy": application["negative_label_policy"],
        "fit_provenance": {
            "application": application,
            "folds": split_records,
            "regularization": regularization,
            "oof_brier": dict((str(penalty), error) for error, penalty in losses),
            "training_channels_sha256": fingerprint(frame.fusion_channels.tolist()),
        },
    }
    payload["oof_predictions"] = oof_by_penalty[regularization]
    if mode == "analytic_fitted":
        payload["parameters"] = {
            "tau": float(tau),
            "gamma": float(gamma),
            "multipliers": dict(zip(names, weights.tolist())),
        }
    elif mode == "learned_global":
        payload["weights"] = dict(zip(names, weights.tolist()))
    else:
        raise ValueError("Only analytic_fitted and learned_global have frozen global providers")
    return freeze_json(path, payload)
