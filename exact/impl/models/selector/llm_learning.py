"""Small train-only exemplars, counterfactual benefit router, and student heads."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .fitting import fingerprint, freeze_json, safe_training_labels

PAIR_FEATURES = ["S_base", "U", "q_lex", "Q_struct"]
SOURCE_FEATURES = ["top_score", "top_two_margin", "candidate_entropy", "displayed_count"]


def source_features(scores):
    values = sorted((float(value) for value in scores), reverse=True)[:5]
    if not values:
        raise ValueError("No-candidate sources cannot train a judge/router")
    masses = [math.exp((value - values[0]) / 0.1) for value in values]
    probabilities = [value / sum(masses) for value in masses]
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
    return [
        values[0],
        values[0] - values[1] if len(values) > 1 else values[0],
        entropy,
        float(len(values)),
    ]


def teacher_identity(model):
    router = model._llm_router
    name = router.routing.decision_profile or router.routing.default_profile
    profile = router.profiles.get(name)
    if profile is None or profile.backend != "openrouter":
        raise ValueError("Teacher binding requires an explicit OpenRouter decision profile")
    return {
        "profile": name,
        **{
            key: getattr(profile, key, None)
            for key in (
                "model",
                "revision",
                "tokenizer",
                "tokenizer_revision",
                "api_base",
                "provider",
            )
        },
        "decision": dict(model.llm_experiment_config["decision"]),
        "seed": model.request_seed,
        "prompt_version": "listwise-v2-raw-categorical",
        "weight_revision_limitation": (
            "hosted_revision_not_immutable" if not profile.revision else None
        ),
    }


def fit_linear(features, targets, *, logistic):
    x = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor(targets, dtype=torch.float64)
    if len(x) < 1 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Head fitting requires finite labeled examples")
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    x = (x - mean) / scale
    with torch.enable_grad():
        weights = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([weights, bias], lr=0.05)
        for _ in range(100):
            optimizer.zero_grad()
            logits = x @ weights + bias
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
                if logistic
                else (logits - y).square().mean()
            )
            (loss + 0.001 * weights.square().mean()).backward()
            optimizer.step()
    return {
        "weights": weights.detach().tolist(),
        "bias": float(bias.detach()),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "logistic": logistic,
    }


def predict_head(features, model):
    result = []
    for row in features:
        if len(row) != len(model["weights"]):
            raise ValueError("Learned head feature schema mismatch")
        logit = model["bias"] + sum(
            (float(value) - mean) / scale * weight
            for value, mean, scale, weight in zip(
                row, model["mean"], model["scale"], model["weights"]
            )
        )
        result.append(
            1 / (1 + math.exp(-max(-700, min(700, logit)))) if model["logistic"] else logit
        )
    return result


def grouped_linear_fit(features, targets, sources, *, logistic, directory, seed):
    import random

    groups = sorted(set(sources))
    if len(groups) < 2:
        raise ValueError("Grouped learned-head fitting needs at least two source groups")
    random.Random(seed).shuffle(groups)
    folds, predictions = [], []
    for fold in range(min(5, len(groups))):
        heldout = set(groups[fold :: min(5, len(groups))])
        train = [i for i, source in enumerate(sources) if source not in heldout]
        test = [i for i, source in enumerate(sources) if source in heldout]
        path = Path(directory) / f"fold-{fold}.json"
        if path.exists():
            record = json.loads(path.read_text())
        else:
            fitted = fit_linear(
                [features[i] for i in train], [targets[i] for i in train], logistic=logistic
            )
            scores = predict_head([features[i] for i in test], fitted)
            record = {
                "fold": fold,
                "training_sources": sorted(set(groups) - heldout),
                "heldout_sources": sorted(heldout),
                "model": fitted,
                "predictions": [
                    {"index": i, "source": sources[i], "target": targets[i], "score": score}
                    for i, score in zip(test, scores)
                ],
            }
            freeze_json(path, record)
        folds.append({key: record[key] for key in ("fold", "training_sources", "heldout_sources")})
        predictions.extend(record["predictions"])
    return {
        "model": fit_linear(features, targets, logistic=logistic),
        "folds": folds,
        "oof_predictions": predictions,
    }


def read_learning_artifact(path, kind):
    if not path:
        raise ValueError(f"{kind} requires an immutable training artifact")
    if not Path(path).is_file():
        raise ValueError(f"{kind} requires an immutable training artifact: {path}")
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 1 or payload.get("kind") != kind:
        raise ValueError(f"Invalid {kind} artifact")
    return payload


def exemplar_prompt(model, plan):
    from dataclasses import replace

    payload = model._exemplar_artifact
    validate_learning_binding(payload, model, [plan.source_iri])
    profile = source_features(plan.candidate_scores)
    ordered = sorted(
        payload["examples"],
        key=lambda row: (
            sum((a - b) ** 2 for a, b in zip(row["features"], profile)),
            row["source"],
        ),
    )
    selected = ordered[: min(3, int(model.llm_experiment_config.get("exemplar_count", 3)))]
    text = "\nTraining-only examples (their gold labels concern other sources):\n" + json.dumps(
        [{key: row[key] for key in ("source", "candidates")} for row in selected], sort_keys=True
    )
    calls = tuple(
        replace(call, prompt={**call.prompt, "user": call.prompt["user"] + text})
        for call in plan.calls
    )
    return replace(plan, calls=calls), [row["source"] for row in selected]


def validate_learning_binding(payload, model, source_ids):
    if payload.get("schema_version") != 1:
        raise ValueError("Invalid learned LLM artifact schema")
    if set(str(value) for value in source_ids) & set(payload["training_sources"]):
        raise ValueError("Learned LLM inference overlaps training sources")
    expected = payload.get("teacher_binding")
    if expected is not None and expected != teacher_identity(model):
        raise ValueError(
            "Teacher/provider/prompt identity changed; learned LLM artifact is incompatible"
        )
    signature = getattr(model._attached_dataset, "dataset_signature", None)
    if payload["application"].get("dataset_signature") != signature:
        raise ValueError("Learned LLM application dataset mismatch")


def fit_llm_artifacts(model, frame, reference_pairs, directory, *, config, application):
    """Teacher requests are executed only when this explicit runtime fit is called."""
    original_counts = frame.groupby("Src").size()
    frame, reference = safe_training_labels(frame, reference_pairs, application)
    if config.get("distill") == "student" and config.get("fusion_weight") == "source_first":
        raise ValueError("The pair student requires beta_u or constant integration")
    if config["gate"]["mode"] == "learned" and (
        config.get("fusion_weight") != "source_first"
        or config.get("decision", {}).get("evidence", "structured_packet") == "generated_brief"
    ):
        raise ValueError(
            "Benefit router requires scored or structured packets and frozen source_first integration"
        )
    if (
        config["gate"]["mode"] == "learned"
        and application["negative_label_policy"] == "confirmed_negatives"
    ):
        labeled_counts = frame.groupby("Src").size()
        complete = {
            str(source)
            for source, count in original_counts.items()
            if int(labeled_counts.get(source, 0)) == int(count)
        }
        complete &= set(application.get("fully_labeled_training_sources", complete))
        if set(original_counts.index.astype(str)) - complete:
            raise ValueError(
                "Benefit router requires every candidate of each training source to have an explicit confirmed label"
            )
    sources = sorted(set(frame.Src.astype(str)))
    binding = teacher_identity(model)
    recipe = {
        "application": application,
        "teacher": binding,
        "features": frame[["Src", "Tgt", *PAIR_FEATURES]].to_dict("records"),
        "references": sorted(reference),
        "student": config.get("student_training", "gold_teacher"),
        "outcomes": config.get("outcome_policy", "unknown"),
        "teacher_source_cap": config.get("teacher_source_cap", 200),
        "threshold": float(model.threshold),
        "evidence": (
            frame.get("llm_evidence_packet", []).tolist() if "llm_evidence_packet" in frame else []
        ),
    }
    directory = Path(directory) / ("llm-" + fingerprint(recipe))
    common = {
        "schema_version": 1,
        "training_sources": sources,
        "application": application,
        "negative_label_policy": application["negative_label_policy"],
        "seed": model.request_seed,
        "teacher_binding": binding,
    }
    result = {}
    if config.get("exemplars") == "knn":
        examples = []
        for source, group in frame.groupby("Src", sort=True):
            group = group.sort_values(["S_base", "Tgt"], ascending=[False, True]).head(5)
            examples.append(
                {
                    "source": source,
                    "features": source_features(group.S_base),
                    "candidates": [
                        {
                            "target": row.Tgt,
                            "evidence": row.llm_evidence_packet,
                            "equivalent": (row.Src, row.Tgt) in reference,
                        }
                        for row in group.itertuples()
                    ],
                }
            )
        payload = {
            **common,
            "kind": "llm_exemplars",
            "feature_schema": SOURCE_FEATURES,
            "examples": examples,
        }
        result["exemplars"] = freeze_json(directory / "exemplars.json", payload)
        result["exemplar_artifact"] = str(directory / "exemplars.json")
    needs_teacher = config["gate"]["mode"] == "learned" or (
        config.get("distill") == "student"
        and config.get("student_training", "gold_teacher") == "gold_teacher"
    )
    teacher_records = []
    if needs_teacher:
        if config["decision"]["mode"] == "binary":
            raise ValueError("E21 source counterfactuals require the frozen comparative judge")
        if (
            config["gate"]["mode"] == "learned"
            and config.get("outcome_policy") != "complete_sources"
        ):
            raise ValueError("Benefit router needs complete/adjudicated source outcomes")
        limit = int(config.get("teacher_source_cap", 200))
        teacher_path = directory / "teacher.json"
        if teacher_path.exists():
            teacher_records = json.loads(teacher_path.read_text())["records"]
        else:
            import random

            teacher_sources = sorted(frame.Src.astype(str).unique())
            random.Random(model.request_seed).shuffle(teacher_sources)
            for source in teacher_sources[:limit]:
                group = frame[frame.Src.astype(str) == source]
                group = group.sort_values(["S_base", "Tgt"], ascending=[False, True]).head(5)
                shard = directory / (fingerprint(source) + ".json")
                if shard.exists():
                    teacher_records.extend(json.loads(shard.read_text())["records"])
                    continue
                briefs = group.llm_evidence_packet.astype(str).tolist()
                if config["decision"].get("evidence") == "generated_brief":
                    briefs = model.generate_pair_briefs_batched(
                        group.src_label_text.tolist(), group.tgt_label_text.tolist(), briefs
                    )
                _, _, records = model.llm_grouped_decision_probs(
                    group.Src.tolist(),
                    group.Tgt.tolist(),
                    group.src_label_text.tolist(),
                    group.tgt_label_text.tolist(),
                    briefs,
                    group.S_base.tolist(),
                    [True] * len(group),
                )
                freeze_json(shard, {"records": records})
                teacher_records.extend(records)
            freeze_json(teacher_path, {"teacher_binding": binding, "records": teacher_records})
    teacher_by_source = {
        record["source"]: record for record in teacher_records if record.get("valid")
    }
    if config.get("distill") == "student":
        targets = []
        for row in frame.itertuples():
            gold = float((row.Src, row.Tgt) in reference)
            teacher = teacher_by_source.get(row.Src, {}).get("pair_probabilities", {}).get(row.Tgt)
            targets.append(0.5 * (gold + teacher) if teacher is not None else gold)
        payload = {
            **common,
            "kind": "llm_student",
            "feature_schema": PAIR_FEATURES,
            **grouped_linear_fit(
                frame[PAIR_FEATURES].values.tolist(),
                targets,
                frame.Src.tolist(),
                logistic=True,
                directory=directory / "student-folds",
                seed=model.request_seed,
            ),
            "training_recipe": config.get("student_training", "gold_teacher"),
            "teacher_sources": sorted(teacher_by_source),
        }
        result["student"] = freeze_json(directory / "student.json", payload)
        result["distill_artifact"] = str(directory / "student.json")
    if config["gate"]["mode"] == "learned":
        examples = []
        for source, group in frame.groupby("Src", sort=True):
            teacher = teacher_by_source.get(source)
            if teacher is None:
                continue
            tokens = sum(
                float(call.get("usage", {}).get("total_tokens", 0)) for call in teacher["calls"]
            ) + float(
                (teacher.get("evidence_acquisition") or {}).get("usage", {}).get("total_tokens", 0)
            )
            if tokens <= 0:
                continue
            base = group.sort_values(["S_base", "Tgt"], ascending=[False, True]).iloc[0]
            base_correct = (str(source), str(base.Tgt)) in reference and float(
                base.S_base
            ) >= model.threshold
            choice = teacher["choice"]
            chosen = group[group.Tgt == choice]
            judged_correct = (
                (str(source), choice) in reference
                and not chosen.empty
                and float(chosen.iloc[0].S_base) >= model.threshold
            )
            benefit = int(judged_correct) - int(base_correct)
            examples.append(
                {
                    "source": source,
                    "features": source_features(group.S_base),
                    "outcome": (
                        "correction" if benefit > 0 else "harm" if benefit < 0 else "no_change"
                    ),
                    "tokens": tokens,
                    "target": 1000 * benefit / tokens,
                }
            )
        if len(examples) < 2:
            raise ValueError(
                "Benefit router needs at least two complete costed counterfactual sources"
            )
        payload = {
            **common,
            "kind": "llm_gate",
            "mode": "learned",
            "feature_schema": SOURCE_FEATURES,
            **grouped_linear_fit(
                [row["features"] for row in examples],
                [row["target"] for row in examples],
                [row["source"] for row in examples],
                logistic=False,
                directory=directory / "router-folds",
                seed=model.request_seed,
            ),
            "threshold": 0.0,
            "target": "correction_minus_harm_per_1000_tokens",
            "outcome_policy": config["outcome_policy"],
            "outcome_scope": "frozen_fully_labeled_candidate_pool",
            "counterfactuals": examples,
        }
        result["router"] = freeze_json(directory / "router.json", payload)
        result["gate_artifact"] = str(directory / "router.json")
    return result
