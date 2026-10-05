"""Small train-only exemplars, counterfactual benefit router, and student heads."""

from __future__ import annotations

import json
import math
from itertools import zip_longest
from pathlib import Path

import torch

from exact.llm.prompt_budget import PromptBudgetError

from .fitting import fingerprint, freeze_json, safe_training_labels
from .oracle_replay import _outcome

PAIR_FEATURES = ["S_base", "U", "q_lex", "Q_struct"]
SOURCE_FEATURES = ["top_score", "top_two_margin", "candidate_entropy", "displayed_count"]
EXEMPLAR_RENDERING = {
    "version": "compact-whole-facts-v1",
    "max_input_tokens": 2500,
    "max_utf8_bytes": 12000,
    "selection": "fair-first-fact-then-candidate-round-robin-packet-order",
}


def _exemplar_evidence(candidate):
    """Project semantic facts, excluding repeated transport/provenance fields."""
    packet = candidate["evidence"]
    packet = json.loads(packet) if isinstance(packet, str) else packet
    if not isinstance(packet, dict) or not isinstance(packet.get("facts"), list):
        raise ValueError("Compact exemplars require structured evidence packets")
    if any(not isinstance(packet.get(key), str) for key in ("source_label", "target_label")):
        raise ValueError("Compact exemplars require source and target labels")
    facts, seen = [], set()
    for raw in packet["facts"]:
        if not isinstance(raw, dict) or raw.get("side") not in {"source", "target"}:
            raise ValueError("Compact exemplar fact has no valid evidence side")
        fact = {key: raw[key] for key in ("side", "group") if key in raw}
        if "triple" in raw:
            if not isinstance(raw["triple"], list) or len(raw["triple"]) != 3:
                raise ValueError("Compact exemplar triples must retain all three terms")
            fact["triple"] = raw["triple"]
        elif isinstance(raw.get("text"), str) and raw.get("property_iri"):
            fact.update(property=raw["property_iri"], text=raw["text"])
        else:
            raise ValueError("Compact exemplar fact has no complete semantic representation")
        fact.update(
            (key, raw[key])
            for key in ("datatype", "language", "state", "contradiction_semantics")
            if raw.get(key) not in (None, "")
        )
        identity = json.dumps(fact, ensure_ascii=False, sort_keys=True)
        if identity not in seen:
            facts.append(fact)
            seen.add(identity)
    return packet, facts


def _compact_exemplars(selected, tokenizer):
    """Keep every selected identity/label and budget only complete evidence facts."""
    header = (
        "\nTraining-only examples (their gold labels concern other sources). "
        "Facts are a budgeted subset; omitted or absent facts are unknown, not contradictions.\n"
    )
    examples, slots = [], []
    for row in selected:
        example = {"source": row["source"], "candidates": []}
        for candidate in row["candidates"]:
            packet, facts = _exemplar_evidence(candidate)
            example.setdefault("source_label", packet["source_label"])
            if not isinstance(candidate["equivalent"], bool):
                raise ValueError("Exemplar gold labels must be Boolean")
            rendered = {
                "target": candidate["target"],
                "target_label": packet["target_label"],
                "equivalent": candidate["equivalent"],
                "facts": [],
                "omitted_facts": len(facts),
            }
            if packet["source_label"] != example["source_label"]:
                rendered["source_label"] = packet["source_label"]
            example["candidates"].append(rendered)
            slots.append((rendered, facts))
        examples.append(example)

    def render():
        return header + json.dumps(examples, ensure_ascii=False, separators=(",", ":"))

    def measure(text):
        return len(tokenizer.encode(text, add_special_tokens=False)), len(text.encode("utf-8"))

    def fits(counts):
        return (
            counts[0] <= EXEMPLAR_RENDERING["max_input_tokens"]
            and counts[1] <= EXEMPLAR_RENDERING["max_utf8_bytes"]
        )

    counts = measure(render())
    if not fits(counts):
        raise PromptBudgetError("Compact exemplar identities and labels exceed the input budget")
    # Count facts individually first: avoid repeatedly tokenizing the complete suffix.
    # The final measurement below accounts for tokenizer boundary interactions exactly.
    added, remaining = [], []
    evidence_slots = sum(bool(facts) for _, facts in slots)
    fair_tokens = (EXEMPLAR_RENDERING["max_input_tokens"] - counts[0]) // max(evidence_slots, 1)
    fair_bytes = (EXEMPLAR_RENDERING["max_utf8_bytes"] - counts[1]) // max(evidence_slots, 1)
    for rendered, facts in slots:
        pending = []
        for fact in facts:
            encoded = json.dumps(fact, ensure_ascii=False, separators=(",", ":"))
            size = len(encoded.encode("utf-8")) + 1
            if size > EXEMPLAR_RENDERING["max_utf8_bytes"]:
                continue
            tokens = measure(encoded)[0] + 8
            if not rendered["facts"] and tokens <= fair_tokens and size <= fair_bytes:
                rendered["facts"].append(fact)
                rendered["omitted_facts"] -= 1
                added.append(rendered)
                counts = (counts[0] + tokens, counts[1] + size)
            else:
                pending.append((fact, tokens, size))
        remaining.append(pending)
    if any(facts and not row["facts"] for row, facts in slots):
        raise PromptBudgetError(
            "Compact exemplar budget cannot retain evidence for every candidate"
        )
    for facts_at_rank in zip_longest(*remaining):
        for (rendered, _), item in zip(slots, facts_at_rank):
            if item is None:
                continue
            fact, tokens, size = item
            prospective = (counts[0] + tokens, counts[1] + size)
            if fits(prospective):
                rendered["facts"].append(fact)
                rendered["omitted_facts"] -= 1
                added.append(rendered)
                counts = prospective
    text = render()
    while not fits(measure(text)) and added:
        rendered = added.pop()
        rendered["facts"].pop()
        rendered["omitted_facts"] += 1
        text = render()
    if not fits(measure(text)) or any(facts and not row["facts"] for row, facts in slots):
        raise PromptBudgetError(
            "Compact exemplar budget cannot retain evidence for every candidate"
        )
    return text


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
    binary = model.llm_experiment_config["decision"]["mode"] == "binary"
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
        "prompt_version": "binary-chat-logprobs-v1" if binary else "listwise-v2-raw-categorical",
        **(
            {
                "labels": list(model.hosted_decision_labels),
                "logit_bias": float(model.hosted_decision_logit_bias),
                "max_tokens": 1,
            }
            if binary
            else {}
        ),
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


def exemplar_context(model, source_iri, candidate_scores):
    """Shared source-level retrieval for the selected binary or comparative prompt."""
    payload = model._exemplar_artifact
    validate_learning_binding(payload, model, [source_iri])
    if payload.get("exemplar_rendering") != EXEMPLAR_RENDERING:
        raise ValueError("Exemplar rendering identity changed; prepare a new training artifact")
    profile = source_features(candidate_scores)
    ordered = sorted(
        [row for row in payload["examples"] if row["source"] != source_iri],
        key=lambda row: (
            sum((a - b) ** 2 for a, b in zip(row["features"], profile)),
            row["source"],
        ),
    )
    selected = ordered[: min(3, int(model.llm_experiment_config.get("exemplar_count", 3)))]
    router = model._llm_router
    profile_name = router.routing.decision_profile or router.routing.default_profile
    tokenizer = model._get_hosted_decision_tokenizer(router.profiles[profile_name])
    text = _compact_exemplars(selected, tokenizer)
    return text, [row["source"] for row in selected]


def exemplar_prompt(model, plan):
    from dataclasses import replace

    text, sources = exemplar_context(model, plan.source_iri, plan.candidate_scores)
    calls = tuple(
        replace(call, prompt={**call.prompt, "user": call.prompt["user"] + text})
        for call in plan.calls
    )
    return replace(plan, calls=calls), sources


def validate_learning_binding(payload, model, source_ids):
    if payload.get("schema_version") != 1:
        raise ValueError("Invalid learned LLM artifact schema")
    from exact.utils.frozen_inference import frozen_application

    if set(str(value) for value in source_ids) & set(
        payload["training_sources"]
    ) and not frozen_application(model._attached_dataset, payload):
        raise ValueError("Learned LLM inference overlaps training sources")
    expected = payload.get("teacher_binding")
    if expected is not None and expected != teacher_identity(model):
        raise ValueError(
            "Teacher/provider/prompt identity changed; learned LLM artifact is incompatible"
        )
    if payload.get("kind") == "llm_gate" and payload.get("counterfactual_scoring") != {
        "threshold": float(model.threshold),
        "beta": float(model.beta),
        "fusion_weight": model.llm_experiment_config["fusion_weight"],
        "constant_weight": float(model.llm_experiment_config.get("constant_weight", 0.5)),
    }:
        raise ValueError("Learned router counterfactual scoring changed")
    signature = getattr(model._attached_dataset, "dataset_signature", None)
    if payload["application"].get("dataset_signature") != signature:
        raise ValueError("Learned LLM application dataset mismatch")


def _validate_binary_teacher(records, source, targets):
    """An actual binary source intervention must cover every displayed pair."""
    if len(records) != 1 or records[0].get("source") != source or not records[0].get("valid"):
        raise ValueError("Binary teacher requires one valid observed source intervention")
    record = records[0]
    probabilities = record.get("pair_probabilities", {})
    if set(probabilities) != targets or any(
        not math.isfinite(float(value)) or not 0 <= float(value) <= 1
        for value in probabilities.values()
    ):
        raise ValueError("Binary teacher probabilities do not cover the frozen displayed pool")
    if (
        len(record.get("calls", [])) != len(targets)
        or {call.get("target") for call in record.get("calls", [])} != targets
    ):
        raise ValueError("Binary teacher requires recorded usage for every displayed pair")
    if any(
        not math.isfinite(float(call.get("usage", {}).get("total_tokens", 0)))
        or float(call.get("usage", {}).get("total_tokens", 0)) <= 0
        for call in record["calls"]
    ):
        raise ValueError("Binary teacher requires positive observed call token costs")


def fit_llm_artifacts(model, frame, reference_pairs, directory, *, config, application):
    """Teacher requests are executed only when this explicit runtime fit is called."""
    original_counts = frame.groupby("Src").size()
    frame, reference = safe_training_labels(frame, reference_pairs, application)
    binary = config["decision"]["mode"] == "binary"
    if binary and getattr(model, "use_llm_calibration", False):
        raise ValueError("Binary E21 learning requires the frozen uncalibrated judge probabilities")
    limit_candidates = int(config["decision"].get("listwise_max_candidates", 5))
    scoring = {
        "threshold": float(model.threshold),
        "beta": float(model.beta),
        "fusion_weight": config.get("fusion_weight", "beta_u"),
        "constant_weight": float(config.get("constant_weight", 0.5)),
    }
    if config.get("distill") == "student" and config.get("fusion_weight") == "source_first":
        raise ValueError("The pair student requires beta_u or constant integration")
    if config["gate"]["mode"] == "learned" and (
        config.get("fusion_weight") != ("beta_u" if binary else "source_first")
        or config.get("decision", {}).get("evidence", "structured_packet") == "generated_brief"
    ):
        raise ValueError(
            "Benefit router requires packets and frozen binary beta_u or comparative source_first integration"
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
    if set(sources) & set(map(str, application.get("source_ids", []))):
        raise ValueError("Training and reporting source groups overlap")
    binding = teacher_identity(model)
    recipe = {
        "learning_version": 2,
        "application": application,
        "teacher": binding,
        "features": frame[["Src", "Tgt", *PAIR_FEATURES]].to_dict("records"),
        "labels": frame[["src_label_text", "tgt_label_text"]].to_dict("records"),
        "references": sorted(reference),
        "student": config.get("student_training", "gold_teacher"),
        "outcomes": config.get("outcome_policy", "unknown"),
        "teacher_source_cap": config.get("teacher_source_cap", 200),
        "counterfactual_scoring": scoring,
        "evidence": (
            frame.get("llm_evidence_packet", []).tolist() if "llm_evidence_packet" in frame else []
        ),
    }
    if config.get("exemplars") == "knn":
        recipe["exemplar_rendering"] = dict(EXEMPLAR_RENDERING)
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
            group = group.sort_values(["S_base", "Tgt"], ascending=[False, True]).head(
                limit_candidates
            )
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
            "exemplar_rendering": dict(EXEMPLAR_RENDERING),
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
        if (
            config["gate"]["mode"] == "learned"
            and config.get("outcome_policy") != "complete_sources"
        ):
            raise ValueError("Benefit router needs complete/adjudicated source outcomes")
        limit = int(config.get("teacher_source_cap", 200))
        import random

        teacher_sources = sorted(frame.Src.astype(str).unique())
        random.Random(model.request_seed).shuffle(teacher_sources)
        teacher_sources = teacher_sources[:limit]
        teacher_path = directory / "teacher.json"
        if teacher_path.exists():
            teacher_payload = json.loads(teacher_path.read_text())
            if teacher_payload["teacher_binding"] != binding:
                raise ValueError("Cached teacher identity changed")
            teacher_records = teacher_payload["records"]
        else:
            for source in teacher_sources:
                group = frame[frame.Src.astype(str) == source]
                group = group.sort_values(["S_base", "Tgt"], ascending=[False, True]).head(
                    limit_candidates
                )
                shard = directory / (fingerprint(source) + ".json")
                if shard.exists():
                    records = json.loads(shard.read_text())["records"]
                    if binary:
                        _validate_binary_teacher(records, str(source), set(group.Tgt.astype(str)))
                    teacher_records.extend(records)
                    continue
                briefs = group.llm_evidence_packet.astype(str).tolist()
                if config["decision"].get("evidence") == "generated_brief":
                    briefs = model.generate_pair_briefs_batched(
                        group.src_label_text.tolist(), group.tgt_label_text.tolist(), briefs
                    )
                args = (
                    group.Src.tolist(),
                    group.Tgt.tolist(),
                    group.src_label_text.tolist(),
                    group.tgt_label_text.tolist(),
                    briefs,
                    group.S_base.tolist(),
                )
                if binary:
                    _, records = model.llm_binary_decision_probs(*args)
                else:
                    _, _, records = model.llm_grouped_decision_probs(*args, [True] * len(group))
                if binary:
                    _validate_binary_teacher(records, str(source), set(group.Tgt.astype(str)))
                freeze_json(shard, {"records": records})
                teacher_records.extend(records)
            freeze_json(teacher_path, {"teacher_binding": binding, "records": teacher_records})
        if binary:
            if sorted(record["source"] for record in teacher_records) != sorted(teacher_sources):
                raise ValueError("Binary teacher cache changed its frozen source population")
            for record in teacher_records:
                group = frame[frame.Src.astype(str) == record["source"]]
                displayed = group.sort_values(["S_base", "Tgt"], ascending=[False, True]).head(
                    limit_candidates
                )
                _validate_binary_teacher([record], record["source"], set(displayed.Tgt.astype(str)))
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
            outcome = _outcome(
                group, teacher["pair_probabilities"], teacher.get("choice"), reference, **scoring
            )
            benefit = outcome["net_correction"]
            examples.append(
                {
                    **outcome,
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
            "counterfactual_scoring": scoring,
            "counterfactuals": examples,
        }
        result["router"] = freeze_json(directory / "router.json", payload)
        result["gate_artifact"] = str(directory / "router.json")
    return result
