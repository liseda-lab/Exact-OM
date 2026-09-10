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
    for source, record in sorted(source_rows.items()):
        candidates = record.get("candidates", [])
        for row in candidates:
            score_values = [row.get("S_base"), row.get("U")]
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
