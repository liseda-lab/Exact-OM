"""Materialize E25 no-call policies from one immutable forced-response run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from exact.impl.models.selector.oracle_replay import (
    NONE,
    _payload,
    observed_response_oracle,
    perfect_intervention,
)
from exact.utils.fitted_artifacts import fingerprint, freeze_json
from exact.utils.provenance import sha256_file


def build_oracle_artifacts(
    forced_trace_path: str | Path,
    reference_rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    reference_role: str,
    negative_label_policy: str,
    budget: int = 200,
    confirmed_negatives: Iterable[tuple[str, str]] = (),
    nil_sources: Iterable[str] = (),
) -> dict[str, Any]:
    """Freeze observed/perfect oracle and three trust arms from cached evidence.

    Trust uses every valid response in the original forced sample, including
    harmful calls. Only the observed oracle selects using outcomes. Unsupported
    source-choice or outcome-label inputs produce explicit unavailable entries.
    No models, router, or hosted client are instantiated.
    """
    if reference_role not in {
        "development",
        "dev",
        "validation",
        "valid",
        "diagnostic",
        "oracle",
        "research_development",
    }:
        raise ValueError("E25 cached policy production requires development/diagnostic references")
    if not 0 <= int(budget) <= 200:
        raise ValueError("E25 oracle budget must be between zero and 200 source interventions")
    if negative_label_policy not in {
        "complete_reference",
        "confirmed_negatives",
        "positive_unlabelled",
        "unknown",
        "not_applicable",
    }:
        raise ValueError("Unknown negative-label policy for E25 cached replay")
    path = Path(forced_trace_path)
    trace = json.loads(path.read_text())
    if (
        trace.get("schema_version") != 2
        or trace.get("source_universe_status") != "declared"
        or trace.get("stage") != "after_cardinality_and_relation_typing"
    ):
        raise ValueError("E25 requires the complete frozen source decision trace")
    llm = trace["policy"].get("llm", {})
    sample = llm.get("forced_sample")
    if llm.get("gate", {}).get("mode") != "forced_sample" or not sample:
        raise ValueError("E25 producer requires the actual forced-source experiment output")
    selected = set(map(str, sample["selected_sources"]))
    if len(selected) != len(sample["selected_sources"]) or not 1 <= len(selected) <= 200:
        raise ValueError("Invalid frozen forced-source sample")
    source_rows = {str(row["Src"]): row for row in trace["records"]}
    if len(source_rows) != len(trace["records"]) or len(set(trace["source_universe"])) != len(
        trace["source_universe"]
    ):
        raise ValueError("Forced source trace contains duplicate source identities")
    if set(source_rows) != set(trace["source_universe"]) or not selected <= set(source_rows):
        raise ValueError("Forced-source IDs must belong to the complete frozen source universe")
    references = set()
    for row in reference_rows:
        source, target = row.get("SrcEntity", row.get("Src")), row.get("TgtEntity", row.get("Tgt"))
        if source is None or target is None:
            raise ValueError("E25 positive reference rows require source and target IDs")
        references.add((str(source), str(target)))
    negatives = {(str(source), str(target)) for source, target in confirmed_negatives}
    nil = sorted(set(map(str, nil_sources)))
    if references & negatives or set(nil) & {source for source, _ in references}:
        raise ValueError("Contradictory positive and negative/NIL labels")
    population_rows, responses, invalid = [], [], []
    unscored_protected_pairs = []
    for source, record in sorted(source_rows.items()):
        candidates = record.get("candidates", [])
        for row in candidates:
            score_values = [row.get("S_base"), row.get("U")]
            if row.get("protected_exact") and any(value is None for value in score_values):
                unscored_protected_pairs.append([source, str(row["target"])])
                continue
            if any(value is None or not math.isfinite(float(value)) for value in score_values):
                raise ValueError("Forced response trace lacks finite frozen S_base/U values")
            target = str(row["target"])
            population_rows.append(
                {
                    "Src": source,
                    "Tgt": target,
                    "S_base": float(score_values[0]),
                    "U": float(score_values[1]),
                    "confirmed_label": (
                        1
                        if (source, target) in references
                        else 0 if (source, target) in negatives or source in nil else None
                    ),
                }
            )
        if source not in selected:
            continue
        grouped = [
            row["llm_grouped_decision"] for row in candidates if row.get("llm_grouped_decision")
        ]
        if grouped:
            parsed = [json.loads(value) if isinstance(value, str) else value for value in grouped]
            if any(item != parsed[0] for item in parsed):
                raise ValueError("Forced trace contains conflicting source response records")
            response = dict(parsed[0])
            if str(response.get("source")) != source:
                raise ValueError("Cached comparative response has a different source identity")
        else:
            values = {
                str(row["target"]): float(row["p_llm"])
                for row in candidates
                if row.get("llm_gate_invoked") and row.get("p_llm") is not None
            }
            response = {
                "source": source,
                "valid": bool(values),
                "pair_probabilities": values,
                "choice": None,
                "response_kind": "cached_binary_used_probabilities",
            }
        values = response.get("pair_probabilities", {})
        if not response.get("valid") or not values:
            invalid.append({"source": source, "reason": "no_valid_cached_response"})
            continue
        if not set(values) <= {str(row["target"]) for row in candidates} or any(
            not math.isfinite(float(value)) or not 0 <= float(value) <= 1
            for value in values.values()
        ):
            raise ValueError(
                "Cached response probabilities differ from the frozen displayed candidates"
            )
        responses.append(response)
    population = pd.DataFrame(
        population_rows, columns=["Src", "Tgt", "S_base", "U", "confirmed_label"]
    )
    if population.duplicated(["Src", "Tgt"]).any():
        raise ValueError("Forced source trace contains duplicate candidate pairs")
    teacher = {
        "dataset_signature": trace.get("dataset_signature"),
        "forced_trace_sha256": sha256_file(path),
        "forced_population_fingerprint": sample["population_fingerprint"],
        "response_sha256": fingerprint(responses),
        "reference_role": reference_role,
        "reference_sha256": fingerprint(sorted(references)),
        "confirmed_negatives_sha256": fingerprint(sorted(negatives)),
        "nil_sources": nil,
        "sampled_sources": sorted(selected),
        "unscored_protected_pairs": sorted(unscored_protected_pairs),
    }
    threshold_value = llm.get("pair_threshold")
    if threshold_value is None:
        threshold_value = trace["policy"].get("threshold")
    threshold = 0.5 if threshold_value is None else float(threshold_value)
    beta = float(llm.get("beta", 0.8))
    identity = fingerprint(
        {
            "teacher": teacher,
            "budget": budget,
            "policy": negative_label_policy,
            "threshold": threshold,
            "beta": beta,
        }
    )
    destination = Path(output_dir) / identity
    artifacts, unavailable = {}, {}
    valid_sources = [row["source"] for row in responses]
    probabilities = {row["source"]: row["pair_probabilities"] for row in responses}
    choices = {row["source"]: row.get("choice") for row in responses}
    for name, weight in (
        ("trust_shipped", "beta_u"),
        ("trust_constant", "constant"),
        ("trust_source", "source_first"),
    ):
        if weight == "source_first" and any(
            choices[source] not in {NONE, *probabilities[source]} for source in valid_sources
        ):
            unavailable[name] = (
                "requires cached canonical comparative choices for the same forced source sample"
            )
            continue
        payload = _payload(
            population,
            valid_sources,
            probabilities,
            choices,
            [],
            invalid,
            mode="oracle_replay",
            protocol="forced_cached_trust",
            budget=len(valid_sources),
            negative_label_policy=negative_label_policy,
            threshold=threshold,
            beta=beta,
            fusion_weight=weight,
            constant_weight=0.5,
            teacher_binding=teacher,
        )
        payload.update(label_independent_selection=True, forced_sample_sources=sorted(selected))
        artifact = destination / (name + ".json")
        freeze_json(artifact, payload)
        artifacts[name] = str(artifact)
    if negative_label_policy not in {"complete_reference", "confirmed_negatives"}:
        for name in ("oracle_observed", "oracle_perfect"):
            unavailable[name] = (
                "outcome-selected oracle requires complete or explicitly confirmed development labels"
            )
    else:
        observed = observed_response_oracle(
            population,
            responses,
            references,
            budget=int(budget),
            negative_label_policy=negative_label_policy,
            threshold=threshold,
            beta=beta,
            fusion_weight="beta_u",
            constant_weight=0.5,
            teacher_binding=teacher,
            nil_sources=nil,
        )
        perfect = perfect_intervention(
            population,
            references,
            observed["selected_sources"],
            displayed_candidates=observed["decision_probs"],
            negative_label_policy=negative_label_policy,
            threshold=threshold,
            beta=beta,
            fusion_weight="beta_u",
            constant_weight=0.5,
            teacher_binding=teacher,
            nil_sources=nil,
        )
        for name, payload in (("oracle_observed", observed), ("oracle_perfect", perfect)):
            artifact = destination / (name + ".json")
            freeze_json(artifact, payload)
            artifacts[name] = str(artifact)
    manifest = {
        "schema_version": 1,
        "kind": "e25_cached_policy_artifacts",
        "identity": identity,
        "teacher_binding": teacher,
        "artifacts": artifacts,
        "unavailable": unavailable,
        "valid_response_sources": valid_sources,
        "invalid_responses": invalid,
        "no_llm_invocations": True,
    }
    freeze_json(destination / "manifest.json", manifest)
    return manifest


def materialize_followup(source, suite, manifests, selections):
    """Bind cached E25 diagnostics to the exact completed forced-judgment recipe."""
    from exact.core.entities.configs.config import ConfigModel
    from exact.experiments.harness import (
        ExperimentSource,
        _inventory_config,
        deep_merge,
        hash_payload,
    )
    from exact.experiments.schema import ExperimentConfig
    from exact.utils.data import read_table

    if source.config.experiment_id not in {"E25-trust", "E25-oracles"}:
        return source
    complete = [
        item
        for item in manifests
        if item.get("experiment_id") == "E25-forced"
        and item.get("stage") == "screen"
        and item.get("arm_id") == "forced_sources"
    ]
    if len(complete) != 1 or complete[0].get("status") != "complete":
        raise ValueError("E25 cached policies require exactly one completed forced-source cell")
    item = complete[0]
    output = Path(item["fingerprint_payload"]["output_dir"]).resolve()
    config_path = output / "_inputs/resolved.config.yaml"
    producer = ConfigModel.load_config(config_path).model_dump(mode="json", by_alias=True)
    if hash_payload(producer) != item.get("resolved_config_hash"):
        raise ValueError("Forced-source resolved configuration changed")
    role = producer["data"]["reference_role"]
    if role not in {
        "development",
        "dev",
        "validation",
        "valid",
        "diagnostic",
        "oracle",
        "research_development",
    }:
        raise ValueError("Cached E25 producer must use a development reference role")
    trace_path = output / "source_decisions.json"
    # The shared stage recovery already validates completed manifests; require the
    # trace's published checksum as well, so stale files cannot supply new policies.
    recovery = item.get("recovery") or {}
    artifact_id = (recovery.get("artifacts") or {}).get("extraction")
    if suite.campaign and not artifact_id:
        raise ValueError("Cached E25 requires a verified completed extraction artifact")
    if artifact_id and suite.campaign:
        from exact.experiments.recovery import ArtifactStore

        store = ArtifactStore(Path(suite.campaign["root"]))
        record = store.verify(artifact_id)
        expected = record["outputs"].get("source_decisions.json", {}).get("sha256")
        if expected != sha256_file(trace_path):
            raise ValueError("Forced-source trace differs from its immutable extraction artifact")
    stage = source.config.screen
    if (
        len(stage.tasks) != 1
        or stage.seeds != [item["seed"]]
        or stage.source_cap != item.get("source_cap")
        or stage.tasks[0].split_role != "development"
        or stage.tasks[0].id != item.get("task_id")
    ):
        raise ValueError(
            "Cached E25 replay must preserve the forced run's source cap, seed and development scope"
        )
    consumer = _inventory_config(source, stage.tasks[0], "screen").model_dump(
        mode="json", by_alias=True
    )
    from exact.utils.provenance import sha256_path

    for key in ("source", "target", "source_universe", "candidates"):
        values = []
        for config in (producer, consumer):
            data = config["data"]
            path = data.get(key)
            values.append(sha256_path(Path(data.get("root") or ".") / path) if path else None)
        if values[0] != values[1]:
            raise ValueError(f"Cached E25 changed its frozen {key} input")
    root = Path(producer["data"].get("root") or ".")
    refs = producer["data"].get("refs") or {}
    reference = root / refs[role]
    frame = read_table(reference)
    trace = json.loads(trace_path.read_text())
    sources = set(trace["source_universe"])
    source_column = "SrcEntity" if "SrcEntity" in frame else "Src"
    frame = frame.loc[frame[source_column].astype(str).isin(sources)]
    # A training-pool negative policy never licenses missing development labels.
    # Only explicit labels in the reporting pool can support the outcome oracle.
    negatives = []
    pool = producer["data"].get("candidates")
    policy = "complete_reference" if item.get("reference_completeness") == "complete" else "unknown"
    if pool:
        candidate_frame = read_table(root / pool)
        if "confirmed_label" in candidate_frame:
            pairs = candidate_frame.loc[candidate_frame["confirmed_label"].eq(0)]
            src = "SrcEntity" if "SrcEntity" in pairs else "Src"
            tgt = "TgtEntity" if "TgtEntity" in pairs else "Tgt"
            negatives = list(pairs[[src, tgt]].itertuples(index=False, name=None))
            policy = "confirmed_negatives"
    campaign_root = Path(suite.campaign["root"]) if suite.campaign else source.directory
    destination = campaign_root / "policies" / source.config.experiment_id
    artifacts = build_oracle_artifacts(
        trace_path,
        frame.to_dict("records"),
        destination,
        reference_role=role,
        negative_label_policy=policy,
        confirmed_negatives=negatives,
        budget=200,
    )
    declaration = source.config.model_dump(mode="json")
    declaration["base_config"] = str(config_path)
    for arm in declaration["arms"]:
        name = arm["id"]
        if name in artifacts["unavailable"]:
            arm["stages"] = []
            continue
        if name == "decision_off":
            arm["overlay"] = {"llm": {"experiment": {"gate": {"mode": "off", "artifact": None}}}}
            continue
        path = Path(artifacts["artifacts"][name])
        payload = json.loads(path.read_text())
        fixed = payload["fixed_fusion"]
        arm["overlay"] = {
            "llm": {
                "experiment": {
                    "enabled": True,
                    "gate": {"mode": payload["mode"], "artifact": str(path)},
                    "fusion_weight": fixed["fusion_weight"],
                    "constant_weight": fixed["constant_weight"],
                }
            }
        }
        ConfigModel.from_mapping(deep_merge(producer, arm["overlay"]))
    declaration["frozen_constants"]["resolved_oracle_policy"] = {
        "producer_config_sha256": sha256_file(config_path),
        "trace_sha256": sha256_file(trace_path),
        "manifest_sha256": sha256_file(destination / artifacts["identity"] / "manifest.json"),
        "unavailable": artifacts["unavailable"],
        "no_new_hosted_requests": True,
    }
    resolved = ExperimentConfig.model_validate(declaration)
    path = destination / "resolved-experiment.json"
    freeze_json(path, resolved.model_dump(mode="json"))
    return ExperimentSource(config=resolved, path=path)
