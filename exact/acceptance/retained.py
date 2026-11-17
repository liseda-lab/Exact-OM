"""Compatibility inference for the retained calibrated rank/winner acceptance head.

The matrix and winner arithmetic are extracted from the validated final model;
no campaign loader or refitting code is imported.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .io import Candidates, digest, export, read_frame, write_json
from .features import RAW


class Native:
    """Frozen numerical operations used by the linear winner head."""

    def __init__(self, temperature, eps):
        self.temperature, self.eps = temperature, eps

    def _softmax(self, values, temperature):
        if not values:
            return []
        scaled = [float(value) / max(float(temperature), self.eps) for value in values]
        largest = max(scaled)
        exps = [math.exp(value - largest) for value in scaled]
        total = sum(exps)
        return [value / total for value in exps]

    def _normalized_entropy(self, probabilities):
        if len(probabilities) <= 1:
            return 0.0
        return -sum(float(p) * math.log(max(float(p), self.eps)) for p in probabilities) / math.log(
            len(probabilities)
        )

    def _linear_score(self, row, model):
        result = float(model.get("bias", 0.0))
        for value, mean, scale, weight in zip(row, model["mean"], model["scale"], model["weights"]):
            result += (float(value) - float(mean)) / max(float(scale), self.eps) * float(weight)
        return result

    @staticmethod
    def _sigmoid(value):
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)


class Matrix:
    def __init__(self, frame, global_scores, global_targets, native=None):
        self.native = native
        self.frame = frame.reset_index(drop=True).copy()
        if self.frame[["Src", "Tgt"]].duplicated().any():
            raise ValueError("Duplicate generated pair")
        self.sources, self.starts, self.lengths = [], [], []
        self.source_indices = {}
        for source, group in self.frame.groupby("Src", sort=False):
            positions = group.index.to_numpy()
            if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
                raise ValueError("Source rows must be contiguous")
            self.sources.append(source)
            self.starts.append(int(positions[0]))
            self.lengths.append(len(positions))
            self.source_indices[source] = positions
        self.starts = np.asarray(self.starts)
        self.lengths = np.asarray(self.lengths)
        self.codes = np.repeat(np.arange(len(self.sources)), self.lengths)
        k = max(self.lengths, default=1)
        self.mask = np.arange(k)[None, :] < self.lengths[:, None]
        self.indices = np.minimum(self.starts[:, None] + np.arange(k), max(0, len(frame) - 1))
        self.global_scores = np.asarray(global_scores, dtype=np.float64)
        self.target_codes, self.targets = pd.factorize(global_targets, sort=False)
        target_map = {s: i for i, s in enumerate(self.targets)}
        self.local_target_codes = np.asarray([target_map[t] for t in self.frame.Tgt])
        self.raw = self.frame.S_pair_final.to_numpy(dtype=np.float64)
        self.static = np.zeros((len(frame), 17), dtype=np.float64)
        for i, name in enumerate(
            [
                "s_label",
                "S_struct",
                "s_hier",
                "s_sim",
                "s_diff",
                "s_attr",
                "cand_sim",
                "cand_sim_prob",
                "cand_share_log_ratio",
            ],
            1,
        ):
            self.static[:, i] = self.frame[name].to_numpy(dtype=np.float64)
        gaps = np.zeros(len(frame))
        for name in ["s_label", "S_struct", "s_hier", "s_sim", "s_attr"]:
            values = self.frame[name].to_numpy(dtype=np.float64)
            gaps += np.maximum(0.0, self.gap(values)) / 5.0
        self.static[:, 12] = gaps
        self.static[:, 15] = self.frame.repair_distinctive.to_numpy(dtype=np.float64)

        def specificity(side):
            score = 0.6 * np.clip(self.frame[side + "_obj_ic_mean"].to_numpy(dtype=float), 0.0, 1.0)
            for col, cap, weight in [
                ("hier_total_count", 20.0, 0.2),
                ("obj_count", 50.0, 0.1),
                ("attr_count", 20.0, 0.1),
            ]:
                score += weight * np.minimum(
                    1.0,
                    np.log1p(np.maximum(0.0, self.frame[side + "_" + col].to_numpy(dtype=float)))
                    / math.log1p(cap),
                )
            return score

        granularity = abs(specificity("src") - specificity("tgt"))
        contradiction = 1.0 - np.clip(self.frame.s_diff.to_numpy(dtype=float), 0.0, 1.0)
        self.static[:, 16] = np.clip(1.0 - (contradiction + granularity) / 2.0, 0.0, 1.0)
        if not np.isfinite(self.static).all() or not np.isfinite(self.raw).all():
            raise ValueError("Nonfinite primary evidence")

    def padded(self, values, fill=-np.inf):
        out = np.asarray(values)[self.indices].copy()
        out[~self.mask] = fill
        return out

    def gap(self, values):
        padded = self.padded(values)
        order = np.argsort(-padded, axis=1, kind="stable")
        best = padded[np.arange(len(self.sources)), order[:, 0]]
        second = padded[np.arange(len(self.sources)), order[:, 1]] if padded.shape[1] > 1 else best
        second = np.where(self.lengths > 1, second, best)
        other = np.repeat(best, self.lengths)
        first_indices = self.starts + order[:, 0]
        other[first_indices] = second
        result = np.asarray(values) - other
        result[np.repeat(self.lengths == 1, self.lengths)] = 0.0
        return result

    def features(self, calibration):
        eps = self.native.eps
        slope, intercept = calibration["slope"], calibration["intercept"]
        p = expit(slope * self.raw + intercept)
        x = self.static.copy()
        clipped = np.clip(p, eps, 1.0 - eps)
        x[:, 0] = np.log(clipped / (1.0 - clipped))
        means = np.add.reduceat(p, self.starts) / self.lengths
        delta = p - means[self.codes]
        std = np.sqrt(np.add.reduceat(delta * delta, self.starts) / self.lengths)
        x[:, 10] = np.divide(
            delta, std[self.codes], out=np.zeros_like(delta), where=std[self.codes] >= eps
        )
        x[:, 11] = self.gap(p)
        order = np.argsort(-self.padded(p), axis=1, kind="stable")
        ranks = np.argsort(order, axis=1, kind="stable")[self.mask]
        x[:, 13] = 1.0 - ranks / np.maximum(1, self.lengths[self.codes] - 1)
        # Whole eligible population, never just the labeled fitting cohort.
        gp = np.clip(expit(slope * self.global_scores + intercept), eps, 1.0 - eps)
        logits = np.log(gp / (1.0 - gp)) / max(self.native.temperature, eps)
        maxima = np.full(len(self.targets), -np.inf)
        np.maximum.at(maxima, self.target_codes, logits)
        denom = np.bincount(
            self.target_codes,
            weights=np.exp(logits - maxima[self.target_codes]),
            minlength=len(self.targets),
        )
        local_logits = x[:, 0] / max(self.native.temperature, eps)
        x[:, 14] = (
            np.exp(local_logits - maxima[self.local_target_codes]) / denom[self.local_target_codes]
        )
        if not np.isfinite(x).all():
            raise ValueError("Invalid recomputed features")
        return x, p


def decisions(matrix, features, probabilities, rank_model, sources, references, background_weight):
    native = matrix.native
    utilities = (
        (features - np.asarray(rank_model["mean"])) / np.asarray(rank_model["scale"])
    ) @ np.asarray(rank_model["weights"]) + rank_model["bias"]
    results = {}
    ref_sources = {s for s, t in references}
    for source in sorted(sources):
        indices = matrix.source_indices.get(source)
        known = source in ref_sources
        if indices is None:
            results[source] = dict(
                source=source,
                pair=None,
                has_reference=known,
                label=False,
                sample_weight=1.0 if known else background_weight,
                p_match=0.0,
                top_pair_score=0.0,
                accept_features=None,
                empty=True,
            )
            continue
        values = utilities[indices]
        order = np.argsort(-values, kind="stable")
        winner = int(indices[order[0]])
        second = int(indices[order[1]]) if len(indices) > 1 else None
        rank_probs = np.asarray(native._softmax(values.tolist(), native.temperature))
        top_prob = rank_probs[order[0]]
        pair = (source, str(matrix.frame.at[winner, "Tgt"]))
        row = matrix.frame.loc[winner]
        accept_features = [
            float(utilities[winner]),
            float(top_prob),
            float(probabilities[winner]),
            float(utilities[winner] - (utilities[second] if second is not None else 0.0)),
            float(top_prob - (rank_probs[order[1]] if second is not None else 0.0)),
            native._normalized_entropy(rank_probs.tolist()),
            float(row.repair_distinctive),
            float(row.s_label),
            float(row.S_struct),
            float(row.s_diff),
        ]
        results[source] = dict(
            source=source,
            pair=list(pair),
            has_reference=known,
            label=pair in references,
            sample_weight=1.0 if known else background_weight,
            top_pair_score=float(probabilities[winner]),
            raw_pair_score=float(row.S_pair_final),
            accept_features=accept_features,
            exact_anchor=bool(row.original_exact_anchor),
            empty=False,
            source_label=str(row.src_label_text),
            target_label=str(row.tgt_label_text),
        )
    return results


def assign(native, records, model):
    for record in records.values():
        record["p_match"] = (
            native._sigmoid(native._linear_score(record["accept_features"], model))
            if record["accept_features"] is not None
            else 0.0
        )


def select(records, threshold, floor=0.0, control=False):
    eligible = [
        r
        for r in records.values()
        if r["pair"] is not None
        and (r["top_pair_score"] if control else r["p_match"]) >= threshold
        and r["top_pair_score"] >= floor
    ]
    eligible.sort(
        key=lambda r: (
            -(r["top_pair_score"] if control else r["p_match"]),
            r["pair"][0],
            r["pair"][1],
        )
    )
    targets, result = set(), []
    for record in eligible:
        if record["pair"][1] not in targets:
            result.append(record)
            targets.add(record["pair"][1])
    return result


RANK_FEATURES = [
    "logit_pair",
    "s_label",
    "S_struct",
    "s_hier",
    "s_sim",
    "s_diff",
    "s_attr",
    "cand_sim",
    "cand_sim_prob",
    "cand_share_log_ratio",
    "z_pair",
    "pair_gap",
    "channel_gap",
    "pair_rank_score",
    "reciprocity",
    "distinctive",
    "safety",
]
ACCEPT_FEATURES = [
    "top_utility",
    "top_rank_prob",
    "top_pair_score",
    "utility_margin",
    "rank_prob_margin",
    "rank_entropy",
    "top_distinctive",
    "top_label",
    "top_struct",
    "top_diff",
]


def validate(head):
    """Reject a superficially similar but incompatible legacy head."""
    if head.get("rank_features") != RANK_FEATURES or head.get("accept_features") != ACCEPT_FEATURES:
        raise ValueError("Retained head feature schema differs")
    if (
        head.get("exact_anchor_bypass") is not False
        or head.get("synthetic_runnerup_negatives") != 0
    ):
        raise ValueError("Retained head must use real winners with no anchor bypass")
    if head.get("global_reciprocity") is not True:
        raise ValueError("Retained head requires whole-population reciprocity")
    for name, count in [("rank_model", 17), ("accept_model", 10)]:
        model = head[name]
        if any(len(model[k]) != count for k in ["mean", "scale", "weights"]):
            raise ValueError("Invalid retained model dimensions")
        if (
            not np.isfinite(
                [*model["mean"], *model["scale"], *model["weights"], model["bias"]]
            ).all()
            or min(model["scale"]) <= 0
        ):
            raise ValueError("Invalid retained model coefficients")
    if head["rank_model"].get("model_type", "current_linear") != "current_linear":
        raise ValueError("Only the qualified linear retained rank model is supported")
    calibration = head["pair_calibration"]
    if (
        calibration.get("mode") != "platt"
        or not np.isfinite([calibration["slope"], calibration["intercept"]]).all()
    ):
        raise ValueError("Retained head requires finite Platt calibration")
    if not 0 <= head["accept_threshold"] <= 1 or not 0 <= head["absolute_pair_floor"] <= 1:
        raise ValueError("Invalid retained operating policy")


def import_head(path, directory, *, temperature, eps=1e-6):
    """Remove campaign paths from a frozen numeric JSON head without refitting."""
    if not np.isfinite([temperature, eps]).all() or temperature <= 0 or not 0 < eps < 0.5:
        raise ValueError("Invalid numerical temperature or epsilon")
    directory = Path(directory)
    if (directory / "retained.json").exists():
        raise ValueError("Output already contains a retained head")
    head = json.loads(Path(path).read_text())
    validate(head)
    keys = [
        "rank_features",
        "accept_features",
        "rank_model",
        "accept_model",
        "pair_calibration",
        "accept_threshold",
        "absolute_pair_floor",
        "exact_anchor_bypass",
        "global_reciprocity",
        "synthetic_runnerup_negatives",
    ]
    write_json(
        directory / "retained.json",
        dict(
            schema=1,
            kind="retained_winner_acceptance",
            head={k: head[k] for k in keys},
            imported_head_sha256=digest(path),
            temperature=temperature,
            eps=eps,
            refitted=False,
        ),
    )
    return directory / "retained.json"


def predict_frame(frame, head, *, temperature=0.75, eps=1e-6):
    """Apply genuine-winner acceptance with calibrated score floor and global conflicts."""
    validate(head)
    missing = set(["Src", "Tgt", *RAW]) - set(frame.columns)
    if missing:
        raise ValueError("Missing candidate columns: " + ", ".join(sorted(missing)))
    if (
        frame[["Src", "Tgt"]].isna().any().any()
        or not np.isfinite(frame[RAW].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Invalid retained candidate evidence")
    frame = frame.copy()
    frame["Src"] = frame.Src.astype(str)
    frame["Tgt"] = frame.Tgt.astype(str)
    if (frame.Src == "").any() or (frame.Tgt == "").any():
        raise ValueError("Empty candidate endpoint")
    if frame[["Src", "Tgt"]].duplicated().any():
        raise ValueError("Duplicate global candidate pair")
    frame = frame.sort_values("Src", kind="stable").reset_index(drop=True)
    frame["src_label_text"] = ""
    frame["tgt_label_text"] = ""
    if not len(frame):
        return [], {}
    native = Native(temperature, eps)
    matrix = Matrix(frame, frame.S_pair_final.to_numpy(), frame.Tgt.to_numpy(), native)
    features, probabilities = matrix.features(head["pair_calibration"])
    outcomes = decisions(
        matrix, features, probabilities, head["rank_model"], set(frame.Src), set(), 0.0
    )
    assign(native, outcomes, head["accept_model"])
    return select(outcomes, head["accept_threshold"], head["absolute_pair_floor"]), outcomes


def apply(candidate_path, directory, output):
    """Apply a portable retained head; fitting labels are neither accepted nor loaded."""
    path = Path(directory) / "retained.json"
    info = json.loads(path.read_text())
    if info.get("schema") != 1 or info.get("kind") != "retained_winner_acceptance":
        raise ValueError("Invalid retained head artifact")
    frame = read_frame(candidate_path, columns=["Src", "Tgt", *RAW])
    chosen, _ = predict_frame(frame, info["head"], temperature=info["temperature"], eps=info["eps"])
    sources = sorted({r["pair"][0] for r in chosen})
    targets = sorted({r["pair"][1] for r in chosen})
    sm, tm = {s: i for i, s in enumerate(sources)}, {t: i for i, t in enumerate(targets)}
    data = Candidates(
        np.empty((len(chosen), 0)),
        np.array([sm[r["pair"][0]] for r in chosen], dtype=int),
        np.array([tm[r["pair"][1]] for r in chosen], dtype=int),
        sources,
        targets,
    )
    return export(
        output,
        data,
        np.arange(len(chosen)),
        np.array([r["p_match"] for r in chosen]),
        dict(
            name="retained_full",
            mode="winner",
            threshold=info["head"]["accept_threshold"],
            pair_floor=info["head"]["absolute_pair_floor"],
            model_manifest_sha256=digest(path),
            candidates_sha256=digest(candidate_path),
            complete_candidate_pairs=len(frame),
        ),
    )
