"""Small grouped, train-only fitting/export boundary for experiment consumers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from exact.utils.fitted_artifacts import fingerprint, freeze_json

from .calibration_helpers import fit_isotonic_calibrator, fit_platt_calibrator


def safe_training_labels(frame, reference_pairs, application):
    """Exclude unknown pairs; a known positive never establishes other negatives."""
    policy = application.get("negative_label_policy", "unknown")
    if policy not in {"complete_reference", "confirmed_negatives"}:
        raise ValueError(
            "Supervised fitting requires a verified complete reference or explicit confirmed negatives; positive-unlabelled references cannot supply negative labels"
        )
    reference = {(str(source), str(target)) for source, target in reference_pairs}
    frame = frame.copy()
    pairs = list(zip(frame.Src.astype(str), frame.Tgt.astype(str)))
    if policy == "complete_reference":
        known_sources = {source for source, _ in reference}
        frame = frame[frame.Src.astype(str).isin(known_sources)].copy()
    else:
        if "confirmed_label" not in frame:
            raise ValueError(
                "confirmed_negatives requires an explicit confirmed_label column (0/1; missing means unknown)"
            )
        labels = pd.to_numeric(frame.confirmed_label, errors="coerce")
        if ((labels.notna()) & ~labels.isin([0, 1])).any():
            raise ValueError("confirmed_label values must be 0, 1, or missing")
        if any(pair in reference and label == 0 for pair, label in zip(pairs, labels)):
            raise ValueError("A reference positive conflicts with a confirmed negative")
        reference |= {pair for pair, label in zip(pairs, labels) if label == 1}
        selected = [pair in reference or label == 0 for pair, label in zip(pairs, labels)]
        frame = frame.loc[selected].copy()
        frame["confirmed_label"] = [
            int((str(row.Src), str(row.Tgt)) in reference) for row in frame.itertuples()
        ]
    return frame.reset_index(drop=True), reference


class FittedSelectorMixin:
    def fit_training_artifact(
        self, training_df, reference_pairs, path, *, application, logger=None
    ):
        """Fit folds on train sources only; select acceptance from held-out predictions.

        Unknown source groups are excluded, including from feature normalization.
        No-positive candidate groups participate only in the acceptance pool-miss
        outcome when their positive reference exists outside the displayed pool.
        """
        application = {
            "entity_kinds": sorted(
                set(training_df.get("SrcKind", pd.Series(["class"])).astype(str))
            ),
            **application,
        }
        frame, ref = safe_training_labels(training_df, reference_pairs, application)
        if "S_pair_final" not in frame:
            frame["S_pair_final"] = frame["S_final"]
        if frame.empty:
            raise ValueError("Training pool has no labeled source groups")
        source_ids = sorted(frame.Src.astype(str).unique())
        if set(source_ids) & set(application.get("source_ids", [])):
            raise ValueError("Training and reporting source groups overlap")
        if len(source_ids) < 2:
            raise ValueError("Grouped out-of-fold fitting needs at least two source groups")
        # Whitelist exact scorer fields; reference/candidate labels never enter features.
        evidence = {
            (str(row.Src), str(row.Tgt)): {
                "selector_evidence_items": getattr(row, "selector_evidence_items", [])
            }
            for row in frame.itertuples()
        }
        distinctive = self._distinctive_scores(frame, evidence, logger=logger)
        features = self._rank_feature_rows(frame, distinctive, self._reciprocity_scores(frame))
        feature_rows = [
            [str(frame.at[index, "Src"]), str(frame.at[index, "Tgt"]), row]
            for index, row in features.items()
        ]
        provenance = {
            "fitting_recipe_version": 2,
            "implementation": self._implementation_fingerprint(),
            "negative_label_policy": application["negative_label_policy"],
            "accept_training": self.experiment_config.get("accept_training", "winner_only"),
            "seed": self.request_seed,
            "training_sources": source_ids,
            "training_features_sha256": fingerprint(feature_rows),
            "training_reference_sha256": fingerprint(sorted(ref)),
            "rank_features": self.RANK_FEATURE_NAMES,
            "accept_features": self.ACCEPT_FEATURE_NAMES,
            "score_calibration": self.matching_calibration["mode"],
            "recipe": {
                "calibration": self.calibration,
                "rerank": {
                    key: value for key, value in self.rerank_config.items() if key != "artifact"
                },
            },
            "application": application,
        }
        identity = fingerprint(provenance)
        path = Path(path)
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("fit_identity") != identity:
                raise ValueError("Fitted selector artifact does not match training inputs/recipe")
            self.rerank_config["artifact"] = str(path)
            return payload
        shuffled = list(source_ids)
        random.Random(self.request_seed).shuffle(shuffled)
        fold_count = min(len(shuffled), max(2, int(self.calibration["validation_folds"])))
        oof = {}
        fold_records = []
        for fold in range(fold_count):
            heldout = set(shuffled[fold::fold_count])
            train = frame[~frame.Src.astype(str).isin(heldout)]
            validation = frame[frame.Src.astype(str).isin(heldout)]
            checkpoint = path.parent / (path.name + ".folds") / f"{identity}-{fold}.json"
            if checkpoint.exists():
                record = json.loads(checkpoint.read_text())
            else:
                groups, _ = self._rank_training_groups(train, ref)
                model = self._fit_rank_model(features, groups, logger=logger)
                if model is None:
                    raise ValueError(f"Fold {fold} has no positive training candidate groups")
                held_features = self._rank_feature_rows(
                    validation, distinctive, self._reciprocity_scores(validation)
                )
                utilities = {
                    index: self._score_rank_model(row, model)
                    for index, row in held_features.items()
                }
                decisions = self._source_decisions(
                    validation, utilities, held_features, distinctive, ref
                )
                train_features = self._rank_feature_rows(
                    train, distinctive, self._reciprocity_scores(train)
                )
                train_utilities = {
                    index: self._score_rank_model(row, model)
                    for index, row in train_features.items()
                }
                train_decisions = self._source_decisions(
                    train, train_utilities, train_features, distinctive, ref
                )
                fold_accept = self._fit_accept_model(train_decisions, logger=logger)
                if fold_accept is None:
                    raise ValueError("Acceptance fold has no positive training winners")
                self._assign_p_match(decisions, fold_accept)
                record = {
                    "fit_identity": identity,
                    "fold": fold,
                    "train_sources": sorted(set(source_ids) - heldout),
                    "heldout_sources": sorted(heldout),
                    "rank_model": model,
                    "decisions": decisions,
                }
                freeze_json(checkpoint, record)
            if record["fit_identity"] != identity or set(record["heldout_sources"]) != heldout:
                raise ValueError("OOF checkpoint does not match the frozen source split")
            oof.update(record["decisions"])
            fold_records.append(
                {key: record[key] for key in ("fold", "train_sources", "heldout_sources")}
            )
        accept = self._fit_accept_model(oof, logger=logger)
        if accept is None:
            raise ValueError("No positive held-out winners for acceptance fitting")
        # Both ranker and acceptance for each held-out source were fitted without
        # that source's labels. The threshold never uses an in-sample winner.
        threshold, metrics = self._tune_accept_threshold(oof)
        groups, _ = self._rank_training_groups(frame, ref)
        rank = self._fit_rank_model(features, groups, logger=logger)
        payload = {
            "schema_version": 1,
            "kind": "fitted_selector",
            "fit_identity": identity,
            "fit_provenance": provenance,
            "rank_model": rank,
            "accept_model": accept,
            "accept_threshold": threshold,
            "oof_metrics": metrics,
            "folds": fold_records,
            "explanation_schema": (
                "feature_additive"
                if self.rerank_config["model"] != "channel_gating"
                else "normalized_channel_contributions"
            ),
        }
        freeze_json(path, payload)
        self.rerank_config["artifact"] = str(path)
        return payload

    def fit_score_calibration_artifact(self, training_df, reference_pairs, path, *, application):
        application = {
            "entity_kinds": sorted(
                set(training_df.get("SrcKind", pd.Series(["class"])).astype(str))
            ),
            **application,
        }
        frame, ref = safe_training_labels(training_df, reference_pairs, application)
        if frame.empty or set(frame.Src.astype(str)) & set(application.get("source_ids", [])):
            raise ValueError("Score calibration requires disjoint labeled training source groups")
        scores = [float(value) for value in frame.S_final]
        labels = [int((str(row.Src), str(row.Tgt)) in ref) for row in frame.itertuples()]
        method = self.matching_calibration["mode"]
        fit = fit_platt_calibrator if method == "platt" else fit_isotonic_calibrator
        calibrator = fit(scores, labels)
        sources = sorted(set(frame.Src.astype(str)))
        random.Random(self.request_seed).shuffle(sources)
        oof = []
        for fold in range(min(5, len(sources))):
            heldout = set(sources[fold :: min(5, len(sources))])
            train_indices = [
                i for i, source in enumerate(frame.Src.astype(str)) if source not in heldout
            ]
            test_indices = [
                i for i, source in enumerate(frame.Src.astype(str)) if source in heldout
            ]
            if not train_indices:
                raise ValueError("Score calibration OOF requires at least two source groups")
            model = fit([scores[i] for i in train_indices], [labels[i] for i in train_indices])
            predictions = model.predict([scores[i] for i in test_indices])
            oof.extend(
                {
                    "source": str(frame.iloc[i].Src),
                    "target": str(frame.iloc[i].Tgt),
                    "label": labels[i],
                    "probability": value,
                    "fold": fold,
                }
                for i, value in zip(test_indices, predictions)
            )
        payload = {
            "schema_version": 1,
            "kind": "score_calibrator",
            "calibrator": calibrator.to_dict(),
            "fit_provenance": {
                "negative_label_policy": application["negative_label_policy"],
                "training_sources": sorted(sources),
                "training_rows_sha256": fingerprint([scores, labels]),
                "application": application,
                "seed": self.request_seed,
            },
            "oof_predictions": oof,
        }
        return freeze_json(path, payload)

    def _apply_fitted_selector(self, df, distinctive, reciprocity, dataset, score_threshold=None):
        payload = json.loads(Path(self.rerank_config["artifact"]).read_text())
        if payload.get("kind") != "fitted_selector" or payload.get("schema_version") != 1:
            raise ValueError("Invalid fitted selector artifact")
        provenance = payload["fit_provenance"]
        if (
            provenance["rank_features"] != self.RANK_FEATURE_NAMES
            or provenance["accept_features"] != self.ACCEPT_FEATURE_NAMES
        ):
            raise ValueError("Fitted selector feature schema mismatch")
        if provenance["score_calibration"] != self.matching_calibration["mode"]:
            raise ValueError("Fitted selector score calibration mismatch")
        current_recipe = {
            key: value for key, value in self.rerank_config.items() if key != "artifact"
        }
        if current_recipe != provenance["recipe"]["rerank"]:
            raise ValueError("Fitted selector model/objective mismatch")
        from exact.utils.artifact_transfer import validate_transferred_artifact

        transferred = validate_transferred_artifact(
            dataset,
            self.rerank_config["artifact"],
            kind="selector",
            features={"rank": self.RANK_FEATURE_NAMES, "accept": self.ACCEPT_FEATURE_NAMES},
            score_threshold=score_threshold,
        )
        expected = provenance["application"].get("dataset_signature")
        if not transferred and expected and expected != getattr(dataset, "dataset_signature", None):
            raise ValueError("Fitted selector application dataset mismatch")
        from exact.utils.frozen_inference import frozen_application

        if set(df.Src.astype(str)) & set(provenance["training_sources"]) and not frozen_application(
            dataset, payload
        ):
            raise ValueError("Reporting sources overlap fitted selector training sources")
        features = self._rank_feature_rows(df, distinctive, reciprocity)
        utilities = {
            index: self._score_rank_model(row, payload["rank_model"])
            for index, row in features.items()
        }
        decisions = self._source_decisions(df, utilities, features, distinctive, set())
        self._assign_p_match(decisions, payload["accept_model"])
        threshold = float(payload["accept_threshold"])
        for decision in decisions.values():
            winner = decision["winner_idx"]
            accepted = decision["p_match"] >= threshold and not decision.get(
                "displayed_none", False
            )
            for index in decision["indices"]:
                is_winner = index == winner
                probability = float(decision["p_match"])
                candidate_probability = probability
                if not is_winner:
                    candidate_probability *= decision["rank_probs"][index] / max(
                        decision["rank_probs"][winner], self.eps
                    )
                    candidate_probability = min(
                        candidate_probability, max(0.0, probability - self.eps)
                    )
                row_score = self._final_selector_score(
                    p_match=candidate_probability,
                    accept_threshold=threshold,
                    score_threshold=(
                        float(score_threshold) if score_threshold is not None else threshold
                    ),
                )
                df.at[index, "S_select"] = (
                    row_score if accepted and (is_winner or self.emit_candidate_scores) else 0.0
                )
                df.at[index, "P_select"] = df.at[index, "S_select"]
                df.at[index, "P_match"] = candidate_probability
                df.at[index, "P_rank"] = decision["rank_probs"][index]
                df.at[index, "selection_source_p_match"] = probability
                df.at[index, "selection_winner"] = is_winner
                df.at[index, "selection_abstained"] = not accepted
                df.at[index, "selection_accept_threshold"] = threshold
                df.at[index, "selection_utility"] = utilities[index]
                rank_model = payload["rank_model"]
                model_type = rank_model.get("model_type", "current_linear")
                if model_type == "analytic":
                    contributions = {"bounded_pair_score": utilities[index]}
                    rank_bias = 0.0
                    basis = list(features[index])
                else:
                    basis = list(self._rank_basis(features[index], model_type))
                    values = [
                        value * weight
                        for value, weight in zip(
                            self._standardize(basis, rank_model["mean"], rank_model["scale"]),
                            rank_model["weights"],
                        )
                    ]
                    if model_type == "additive_gam":
                        values = [sum(values[i : i + 2]) for i in range(0, len(values), 2)]
                    names = (
                        [self.RANK_FEATURE_NAMES[i] for i in (1, 3, 4, 5, 6)]
                        if model_type == "channel_gating"
                        else self.RANK_FEATURE_NAMES
                    )
                    contributions = dict(zip(names, values))
                    rank_bias = rank_model["bias"]
                explanation = {
                    "schema_version": 2,
                    "schema": payload["explanation_schema"],
                    "stage": "fitted_selector_before_nil_and_extraction",
                    "fit_identity": payload["fit_identity"],
                    "rank": {
                        "feature_names": self.RANK_FEATURE_NAMES,
                        "features": dict(zip(self.RANK_FEATURE_NAMES, features[index])),
                        "basis": basis,
                        "model": rank_model,
                    },
                    "bias": rank_bias,
                    "contributions": contributions,
                    "logit": utilities[index],
                    "accept_logit": self._linear_score(
                        decision["accept_features"], payload["accept_model"]
                    ),
                    "source_decision_target": str(df.at[winner, "Tgt"]),
                    "output": {
                        "is_winner": is_winner,
                        "accepted": bool(accepted),
                        "P_match": candidate_probability,
                        "P_rank": decision["rank_probs"][index],
                        "S_select": float(df.at[index, "S_select"]),
                    },
                }
                if is_winner:
                    accept_model = payload["accept_model"]
                    accept_values = self._standardize(
                        decision["accept_features"], accept_model["mean"], accept_model["scale"]
                    )
                    explanation["source_decision"] = {
                        "source": decision["source"],
                        "source_kind": decision["source_kind"],
                        "winner": str(df.at[winner, "Tgt"]),
                        "candidates": [
                            {
                                "target": str(df.at[item, "Tgt"]),
                                "target_kind": (
                                    str(df.at[item, "TgtKind"]) if "TgtKind" in df else "class"
                                ),
                                "utility": utilities[item],
                                "rank_probability": decision["rank_probs"][item],
                            }
                            for item in decision["indices"]
                        ],
                        "rank_temperature": self.temperature,
                        "tie_break": "stable_input_order",
                        "llm_source_choice": (
                            str(df.at[winner, "llm_source_choice"])
                            if "llm_source_choice" in df
                            else ""
                        ),
                        "displayed_none": bool(decision.get("displayed_none", False)),
                        "accept": {
                            "feature_names": self.ACCEPT_FEATURE_NAMES,
                            "features": dict(
                                zip(self.ACCEPT_FEATURE_NAMES, decision["accept_features"])
                            ),
                            "model": accept_model,
                            "contributions": dict(
                                zip(
                                    self.ACCEPT_FEATURE_NAMES,
                                    [
                                        value * weight
                                        for value, weight in zip(
                                            accept_values, accept_model["weights"]
                                        )
                                    ],
                                )
                            ),
                            "logit": explanation["accept_logit"],
                            "probability": probability,
                        },
                        "accept_threshold": threshold,
                        "score_threshold": (
                            float(score_threshold) if score_threshold is not None else threshold
                        ),
                        "score_mode": self.score_mode,
                        "emit_candidate_scores": self.emit_candidate_scores,
                        "eps": self.eps,
                    }
                df.at[index, "selector_explanation"] = json.dumps(explanation, sort_keys=True)
                df.at[index, "selection_entropy"] = decision["rank_entropy"]
                df.at[index, "selection_margin"] = decision["utility_margin"]
                df.at[index, "selection_no_match_prob"] = 1.0 - probability
                df.at[index, "selection_reason"] = "fitted_accept" if accepted else "fitted_abstain"
        self._calibration_meta = {
            "fit_identity": payload["fit_identity"],
            "fit_provenance": provenance,
            "accept_threshold": threshold,
            "oof_metrics": payload["oof_metrics"],
            "explanation_schema": payload["explanation_schema"],
        }
        return df
