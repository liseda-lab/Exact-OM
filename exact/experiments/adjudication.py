"""Blinded adjudication and weighted precision for incomplete references."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

_LABELS = {"match": 1, "nonmatch": 0, "uncertain": None}
_STRATIFICATION_FIELDS = ("task_id", "entity_kind", "disagreement_group", "score_band")
_FORBIDDEN_BLIND_KEYS = {
    "arm",
    "arm_id",
    "baseline_arm",
    "candidate_arm",
    "hypothesis",
    "score",
    "score_band",
    "scores",
    "raw_score",
    "baseline_score",
    "candidate_score",
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _nonempty(value: Any, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    return normalized


def adjudication_sample_size(
    population_size: int,
    *,
    target_ci_width: float,
    confidence: float = 0.95,
    anticipated_precision: float = 0.5,
    minimum_when_available: int = 50,
) -> dict[str, Any]:
    """Pre-register a finite-population binomial-precision sample size."""

    if isinstance(population_size, bool) or int(population_size) < 1:
        raise ValueError("adjudication population_size must be positive")
    population = int(population_size)
    width = float(target_ci_width)
    if not math.isfinite(width) or not 0.0 < width < 1.0:
        raise ValueError("target_ci_width must be between zero and one")
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be between zero and one")
    anticipated = float(anticipated_precision)
    if not math.isfinite(anticipated) or not 0.0 < anticipated < 1.0:
        raise ValueError("anticipated_precision must be between zero and one")
    if isinstance(minimum_when_available, bool) or int(minimum_when_available) < 1:
        raise ValueError("minimum_when_available must be positive")
    minimum = int(minimum_when_available)
    z_value = NormalDist().inv_cdf(0.5 + confidence_value / 2.0)
    margin = width / 2.0
    infinite_n = z_value * z_value * anticipated * (1.0 - anticipated) / (margin * margin)
    finite_n = infinite_n / (1.0 + ((infinite_n - 1.0) / population))
    calculated = max(1, int(math.ceil(finite_n)))
    selected = min(population, max(calculated, min(minimum, population)))
    achieved_margin = z_value * math.sqrt(
        anticipated
        * (1.0 - anticipated)
        * (population - selected)
        / (selected * max(1, population - 1))
    )
    return {
        "schema_version": 1,
        "method": "normal_binomial_with_finite_population_correction",
        "population_size": population,
        "sample_size": selected,
        "target_ci_width": width,
        "ci_width_definition": "two_sided_total_width",
        "confidence": confidence_value,
        "anticipated_precision": anticipated,
        "minimum_when_available": minimum,
        "approximate_achieved_ci_width": 2.0 * achieved_margin,
    }


def _forbidden_blind_paths(
    value: Any,
    *,
    path: str = "",
    allow_root_score_band: bool = False,
) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.strip().lower()
            item_path = f"{path}.{key}" if path else key
            allowed = allow_root_score_band and not path and normalized == "score_band"
            if normalized in _FORBIDDEN_BLIND_KEYS and not allowed:
                found.append(item_path)
            found.extend(
                _forbidden_blind_paths(
                    item,
                    path=item_path,
                    allow_root_score_band=False,
                )
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            found.extend(_forbidden_blind_paths(item, path=item_path))
    return found


def _case_id(source: str, target: str, relation: str) -> str:
    identity = {"source_id": source, "target_id": target, "relation": relation}
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()[:20]


def verify_sample_manifest(sample: Mapping[str, Any]) -> str:
    """Verify the sample hash, case identities, uniqueness, and recursive blinding."""

    if sample.get("schema_version") != 1:
        raise ValueError("adjudication sample schema_version must be 1")
    if sample.get("status") != "awaiting_annotation" or sample.get("blinded") is not True:
        raise ValueError("adjudication sample must be awaiting annotation and blinded")
    declared = str(sample.get("sample_hash") or "").strip().lower()
    if len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
        raise ValueError("adjudication sample has no valid sample_hash")
    unhashed = dict(sample)
    unhashed.pop("sample_hash", None)
    observed = hashlib.sha256(_canonical(unhashed).encode("utf-8")).hexdigest()
    if observed != declared:
        raise ValueError(
            f"adjudication sample hash mismatch: declared={declared}, observed={observed}"
        )
    cases = sample.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("adjudication sample has no cases")
    population_size = sample.get("population_size")
    sample_size = sample.get("sample_size")
    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, int)
        or population_size < len(cases)
    ):
        raise ValueError("adjudication sample has invalid population_size")
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, int)
        or sample_size != len(cases)
    ):
        raise ValueError("adjudication sample_size does not match its cases")
    population_hash = str(sample.get("population_hash") or "").strip().lower()
    if len(population_hash) != 64 or any(
        character not in "0123456789abcdef" for character in population_hash
    ):
        raise ValueError("adjudication sample has no valid population_hash")
    sampling = sample.get("sampling")
    if not isinstance(sampling, Mapping) or sampling.get("method") != (
        "deterministic_stratified_without_replacement"
    ):
        raise ValueError("adjudication sample has invalid sampling metadata")
    if sampling.get("stratification_fields") != list(_STRATIFICATION_FIELDS):
        raise ValueError("adjudication sample changed its required stratification fields")
    strata = sampling.get("strata")
    if not isinstance(strata, list) or not strata:
        raise ValueError("adjudication sample has no stratum summaries")
    stratum_ids: set[str] = set()
    stratum_probabilities: dict[str, float] = {}
    declared_stratum_samples: dict[str, int] = {}
    stratum_population = 0
    stratum_sample = 0
    for stratum in strata:
        if not isinstance(stratum, Mapping):
            raise ValueError("adjudication stratum summaries must be mappings")
        stratum_id = _nonempty(stratum.get("stratum_id"), "stratum_id")
        if stratum_id in stratum_ids:
            raise ValueError(f"duplicate adjudication stratum_id {stratum_id!r}")
        stratum_ids.add(stratum_id)
        raw_population = stratum.get("population_size")
        raw_sample = stratum.get("sample_size")
        if (
            isinstance(raw_population, bool)
            or not isinstance(raw_population, int)
            or raw_population < 1
            or isinstance(raw_sample, bool)
            or not isinstance(raw_sample, int)
            or not 0 <= raw_sample <= raw_population
        ):
            raise ValueError(f"adjudication stratum {stratum_id!r} has invalid counts")
        probability = float(stratum.get("inclusion_probability") or 0.0)
        expected_probability = raw_sample / raw_population
        if not math.isfinite(probability) or not math.isclose(
            probability,
            expected_probability,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"adjudication stratum {stratum_id!r} has invalid inclusion_probability"
            )
        stratum_probabilities[stratum_id] = probability
        declared_stratum_samples[stratum_id] = raw_sample
        stratum_population += raw_population
        stratum_sample += raw_sample
    if stratum_population != population_size or stratum_sample != sample_size:
        raise ValueError("adjudication stratum totals do not match the sample manifest")
    sample_design = sample.get("sample_design")
    if sample_design is not None:
        if not isinstance(sample_design, Mapping):
            raise ValueError("adjudication sample_design must be a mapping")
        if sample_design.get("population_size") != population_size:
            raise ValueError("adjudication sample_design population_size mismatch")
        if sample_design.get("sample_size") != sample_size:
            raise ValueError("adjudication sample_design sample_size mismatch")
    seen: set[str] = set()
    observed_stratum_samples: dict[str, int] = defaultdict(int)
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("adjudication sample cases must be mappings")
        case_id = _nonempty(case.get("case_id"), "sample case_id")
        if case_id in seen:
            raise ValueError(f"duplicate adjudication case_id {case_id!r}")
        seen.add(case_id)
        expected = _case_id(
            _nonempty(case.get("source_id"), "sample source_id"),
            _nonempty(case.get("target_id"), "sample target_id"),
            _nonempty(case.get("relation"), "sample relation"),
        )
        if case_id != expected:
            raise ValueError(f"adjudication case identity hash mismatch for {case_id!r}")
        case_stratum = _nonempty(case.get("stratum_id"), "sample stratum_id")
        if case_stratum not in stratum_ids:
            raise ValueError(f"adjudication case {case_id!r} has an unknown stratum_id")
        probability = float(case.get("inclusion_probability") or 0.0)
        if not math.isfinite(probability) or not math.isclose(
            probability,
            stratum_probabilities[case_stratum],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"adjudication case {case_id!r} has invalid inclusion_probability")
        observed_stratum_samples[case_stratum] += 1
        forbidden = _forbidden_blind_paths(case)
        if forbidden:
            raise ValueError(f"adjudication sample is not recursively blinded: {forbidden}")
    if dict(observed_stratum_samples) != {
        key: value for key, value in declared_stratum_samples.items() if value
    }:
        raise ValueError("adjudication case counts do not match stratum summaries")
    return declared


def _stratified_allocation(
    counts: Mapping[tuple[str, str, str, str], int],
    sample_size: int,
) -> dict[tuple[str, str, str, str], int]:
    strata = sorted(counts)
    population = sum(counts.values())
    target = min(int(sample_size), population)
    allocation = {stratum: 0 for stratum in strata}
    if target >= len(strata):
        allocation = {stratum: 1 for stratum in strata}
    quotas = {stratum: target * counts[stratum] / population for stratum in strata}
    while sum(allocation.values()) < target:
        eligible = [stratum for stratum in strata if allocation[stratum] < counts[stratum]]
        chosen = max(
            eligible,
            key=lambda stratum: (
                quotas[stratum] - allocation[stratum],
                counts[stratum] - allocation[stratum],
                tuple(reversed(stratum)),
            ),
        )
        allocation[chosen] += 1
    return allocation


def build_blinded_sample(
    disagreements: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    seed: int,
    sample_design: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select a deterministic required-strata sample without exposing strata."""

    if sample_size < 1:
        raise ValueError("adjudication sample_size must be positive")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in disagreements:
        source = _nonempty(raw.get("source_id") or raw.get("source"), "source_id")
        target = _nonempty(raw.get("target_id") or raw.get("target"), "target_id")
        relation = _nonempty(raw.get("relation") or "equivalence", "relation")
        key = (source, target, relation)
        if key in seen:
            raise ValueError(f"duplicate adjudication mapping {key!r}")
        seen.add(key)
        forbidden = _forbidden_blind_paths(raw, allow_root_score_band=True)
        if forbidden:
            raise ValueError(
                f"blinded adjudication input exposes arm/score/hypothesis fields: {forbidden}"
            )
        task_id = _nonempty(raw.get("task_id"), "task_id")
        entity_kind = _nonempty(raw.get("entity_kind"), "entity_kind")
        disagreement_group = _nonempty(raw.get("disagreement_group"), "disagreement_group").lower()
        if disagreement_group not in {"arm_only", "both"}:
            raise ValueError("disagreement_group must be arm_only or both")
        score_band = _nonempty(raw.get("score_band"), "score_band")
        candidates.append(
            {
                "source_id": source,
                "target_id": target,
                "relation": relation,
                "evidence": raw.get("evidence") or {},
                "_stratum": (task_id, entity_kind, disagreement_group, score_band),
            }
        )
    if not candidates:
        raise ValueError("adjudication export has no disagreements")
    selected_count = min(int(sample_size), len(candidates))
    if len(candidates) >= 50 and selected_count < 50:
        raise ValueError("adjudication sample must contain at least 50 cases when available")
    normalized_design: dict[str, Any] | None = None
    if sample_design is not None:
        normalized_design = json.loads(_canonical(sample_design))
        if normalized_design.get("population_size") != len(candidates):
            raise ValueError("sample_design population_size does not match disagreements")
        if normalized_design.get("sample_size") != selected_count:
            raise ValueError("sample_design sample_size does not match the selected sample")
    candidates.sort(
        key=lambda item: (
            item["source_id"],
            item["target_id"],
            item["relation"],
            item["_stratum"],
        )
    )
    by_stratum: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_stratum[candidate["_stratum"]].append(candidate)
    counts = {stratum: len(items) for stratum, items in by_stratum.items()}
    allocation = _stratified_allocation(counts, selected_count)
    selected: list[tuple[dict[str, Any], str, float]] = []
    strata_summary: list[dict[str, Any]] = []
    for stratum in sorted(by_stratum):
        items = list(by_stratum[stratum])
        seed_digest = hashlib.sha256(f"{int(seed)}\0{_canonical(stratum)}".encode("utf-8")).digest()
        random.Random(int.from_bytes(seed_digest[:8], "big")).shuffle(items)
        count = allocation[stratum]
        stratum_id = hashlib.sha256(_canonical(stratum).encode("utf-8")).hexdigest()[:16]
        probability = count / len(items)
        selected.extend((item, stratum_id, probability) for item in items[:count])
        strata_summary.append(
            {
                "stratum_id": stratum_id,
                "population_size": len(items),
                "sample_size": count,
                "inclusion_probability": probability,
            }
        )
    cases: list[dict[str, Any]] = []
    for item, stratum_id, inclusion_probability in selected:
        identity = {key: item[key] for key in ("source_id", "target_id", "relation")}
        cases.append(
            {
                "case_id": _case_id(
                    identity["source_id"],
                    identity["target_id"],
                    identity["relation"],
                ),
                **identity,
                "evidence": item["evidence"],
                "stratum_id": stratum_id,
                "inclusion_probability": inclusion_probability,
            }
        )
    cases.sort(key=lambda item: item["case_id"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "awaiting_annotation",
        "blinded": True,
        "seed": int(seed),
        "population_size": len(candidates),
        "sample_size": selected_count,
        "population_hash": hashlib.sha256(_canonical(candidates).encode("utf-8")).hexdigest(),
        "sampling": {
            "method": "deterministic_stratified_without_replacement",
            "stratification_fields": list(_STRATIFICATION_FIELDS),
            "strata": strata_summary,
        },
        "cases": cases,
    }
    if normalized_design is not None:
        payload["sample_design"] = normalized_design
    payload["sample_hash"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return payload


def write_blinded_sample(sample: Mapping[str, Any], path: Path) -> tuple[Path, Path]:
    """Write an annotation TSV plus an immutable, arm-free sample manifest."""

    if sample.get("blinded") is not True or sample.get("status") != "awaiting_annotation":
        raise ValueError("only awaiting, blinded samples may be exported")
    sample_hash = verify_sample_manifest(sample)
    cases = sample.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("blinded sample has no cases")
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            delimiter="\t",
            fieldnames=(
                "sample_hash",
                "case_id",
                "source_id",
                "target_id",
                "relation",
                "evidence_json",
                "annotator_id",
                "role",
                "label",
                "notes",
            ),
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "sample_hash": sample_hash,
                    "case_id": case["case_id"],
                    "source_id": case["source_id"],
                    "target_id": case["target_id"],
                    "relation": case["relation"],
                    "evidence_json": _canonical(case.get("evidence") or {}),
                }
            )
    os.replace(temporary, output)
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    manifest_tmp = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    manifest_tmp.write_text(_canonical(sample) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest)
    return output, manifest


def import_annotations(path: Path, sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate annotation rows against exactly the exported blinded cases."""

    sample_hash = verify_sample_manifest(sample)
    cases = sample.get("cases")
    expected = (
        {
            str(case.get("case_id")): case
            for case in cases
            if isinstance(case, Mapping) and case.get("case_id")
        }
        if isinstance(cases, list)
        else {}
    )
    if not expected:
        raise ValueError("annotation sample contains no expected cases")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row_number, raw in enumerate(reader, start=2):
            row_sample_hash = _nonempty(
                raw.get("sample_hash"), f"sample_hash at row {row_number}"
            ).lower()
            if row_sample_hash != sample_hash:
                raise ValueError(
                    f"annotation sample_hash mismatch at row {row_number}: "
                    f"{row_sample_hash} != {sample_hash}"
                )
            case_id = _nonempty(raw.get("case_id"), f"case_id at row {row_number}")
            annotator = _nonempty(raw.get("annotator_id"), f"annotator_id at row {row_number}")
            label = _nonempty(raw.get("label"), f"label at row {row_number}").lower()
            if case_id not in expected:
                raise ValueError(f"unknown adjudication case_id {case_id!r}")
            if label not in _LABELS:
                raise ValueError(f"invalid adjudication label {label!r}")
            role = str(raw.get("role") or "annotator").strip().lower()
            if role not in {"annotator", "adjudicator"}:
                raise ValueError(f"invalid adjudication role {role!r}")
            if role == "adjudicator" and label == "uncertain":
                raise ValueError("adjudicator labels must resolve to match or nonmatch")
            expected_case = expected[case_id]
            for field in ("source_id", "target_id", "relation"):
                supplied = str(raw.get(field) or "").strip()
                if supplied and supplied != str(expected_case.get(field)):
                    raise ValueError(f"annotation {field} changed for case_id {case_id!r}")
            key = (case_id, annotator)
            if key in seen:
                raise ValueError(f"duplicate annotation by {annotator!r} for {case_id!r}")
            seen.add(key)
            rows.append(
                {
                    "case_id": case_id,
                    "annotator_id": annotator,
                    "role": role,
                    "label": label,
                    "notes": str(raw.get("notes") or ""),
                    "sample_hash": sample_hash,
                }
            )
    if not rows:
        raise ValueError("annotation import contains no completed rows")
    return rows


def _inter_annotator_agreement(
    cases: Sequence[Mapping[str, Any]],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    pairs: list[tuple[str, str]] = []
    uncertain_ratings = 0
    cases_with_two = 0
    for case in cases:
        case_id = str(case["case_id"])
        primary = sorted(
            (
                (str(row.get("annotator_id")), str(row.get("label")).lower())
                for row in grouped.get(case_id, ())
                if str(row.get("role") or "annotator").lower() == "annotator"
            ),
            key=lambda item: item[0],
        )
        uncertain_ratings += sum(label == "uncertain" for _, label in primary)
        if len(primary) >= 2:
            cases_with_two += 1
        for left_index in range(len(primary)):
            for right_index in range(left_index + 1, len(primary)):
                pairs.append((primary[left_index][1], primary[right_index][1]))
    if not pairs:
        return {
            "method": "pairwise_nominal_kappa",
            "status": "insufficient",
            "cases_with_two_primary": cases_with_two,
            "rating_pairs": 0,
            "exact_agreement": None,
            "kappa": None,
            "uncertain_primary_ratings": uncertain_ratings,
        }
    observed = sum(left == right for left, right in pairs) / len(pairs)
    marginals: dict[str, int] = defaultdict(int)
    for left, right in pairs:
        marginals[left] += 1
        marginals[right] += 1
    total = 2 * len(pairs)
    expected = sum((count / total) ** 2 for count in marginals.values())
    if math.isclose(expected, 1.0):
        kappa = 1.0 if math.isclose(observed, 1.0) else 0.0
    else:
        kappa = (observed - expected) / (1.0 - expected)
    return {
        "method": "pairwise_nominal_kappa",
        "status": "complete",
        "cases_with_two_primary": cases_with_two,
        "rating_pairs": len(pairs),
        "exact_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
        "uncertain_primary_ratings": uncertain_ratings,
        "label_pair_counts": {
            f"{left}|{right}": sum(pair == (left, right) for pair in pairs)
            for left, right in sorted(set(pairs))
        },
    }


def merge_adjudications(
    sample: Mapping[str, Any],
    annotations: Sequence[Mapping[str, Any]],
    *,
    minimum_annotators: int = 2,
) -> dict[str, Any]:
    """Merge unanimous annotations or an explicit adjudicator decision."""

    if minimum_annotators < 2:
        raise ValueError("adjudication requires at least two primary annotators")
    sample_hash = verify_sample_manifest(sample)
    cases = sample.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("adjudication sample has no cases")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    expected_case_ids = {str(case["case_id"]) for case in cases}
    seen_annotations: set[tuple[str, str]] = set()
    for row in annotations:
        case_id = _nonempty(row.get("case_id"), "annotation case_id")
        annotator_id = _nonempty(row.get("annotator_id"), "annotation annotator_id")
        if case_id not in expected_case_ids:
            raise ValueError(f"unknown adjudication case_id {case_id!r}")
        if str(row.get("sample_hash") or "").lower() != sample_hash:
            raise ValueError(f"annotation sample_hash mismatch for {case_id!r}")
        key = (case_id, annotator_id)
        if key in seen_annotations:
            raise ValueError(f"duplicate annotation by {annotator_id!r} for {case_id!r}")
        seen_annotations.add(key)
        role = str(row.get("role") or "annotator").lower()
        label = str(row.get("label") or "").lower()
        if role not in {"annotator", "adjudicator"} or label not in _LABELS:
            raise ValueError(f"invalid annotation role/label for {case_id!r}")
        if role == "adjudicator" and label == "uncertain":
            raise ValueError("adjudicator labels must resolve to match or nonmatch")
        grouped[case_id].append(row)
    merged: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        rows = grouped.get(case_id, [])
        primary_rows = [
            row for row in rows if str(row.get("role") or "annotator").lower() == "annotator"
        ]
        adjudicator_rows = [
            row for row in rows if str(row.get("role") or "").lower() == "adjudicator"
        ]
        if len(adjudicator_rows) > 1:
            raise ValueError(f"multiple adjudicator decisions for {case_id}")
        if adjudicator_rows and len(primary_rows) < minimum_annotators:
            raise ValueError(
                f"adjudicator decision for {case_id} requires two primary annotations first"
            )
        labels = [str(row.get("label")).lower() for row in primary_rows]
        definitive = [label for label in labels if label in {"match", "nonmatch"}]
        unanimous = (
            len(primary_rows) >= minimum_annotators
            and len(definitive) == len(primary_rows)
            and len(set(definitive)) == 1
        )
        if adjudicator_rows and unanimous:
            raise ValueError(
                f"adjudicator decision for {case_id} is invalid after unanimous primaries"
            )
        resolution: str | None = definitive[0] if unanimous else None
        if resolution is None and adjudicator_rows:
            resolution = str(adjudicator_rows[0].get("label")).lower()
        merged.append(
            {
                **dict(case),
                "annotation_count": len(rows),
                "primary_annotation_count": len(primary_rows),
                "adjudicator_count": len(adjudicator_rows),
                "resolution": resolution,
                "is_match": _LABELS[resolution] if resolution is not None else None,
                "status": "complete" if resolution is not None else "pending",
            }
        )
    status = "complete" if all(case["status"] == "complete" for case in merged) else "pending"
    return {
        "schema_version": 1,
        "sample_hash": sample_hash,
        "sample_hash_verified": True,
        "status": status,
        "inter_annotator_agreement": _inter_annotator_agreement(cases, grouped),
        "cases": merged,
    }


def _weighted_precision(cases: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    numerator = 0.0
    denominator = 0.0
    squared_weight_sum = 0.0
    for case in cases:
        probability = float(case.get("inclusion_probability") or 0.0)
        outcome = case.get("is_match")
        if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("invalid adjudication inclusion_probability")
        if outcome not in {0, 1}:
            raise ValueError("weighted precision encountered an unresolved label")
        weight = 1.0 / probability
        numerator += weight * int(outcome)
        denominator += weight
        squared_weight_sum += weight * weight
    if denominator <= 0.0:
        raise ValueError("weighted precision denominator is zero")
    effective_n = denominator * denominator / squared_weight_sum
    return numerator / denominator, numerator, effective_n


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("bootstrap produced no estimates")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def inverse_probability_weighted_precision(
    merged: Mapping[str, Any],
    *,
    resamples: int = 10_000,
    seed: int = 17,
) -> dict[str, Any]:
    """Estimate IPW precision with a deterministic stratified bootstrap CI."""

    if merged.get("status") != "complete":
        raise ValueError("weighted precision requires completed adjudication")
    if merged.get("sample_hash_verified") is not True:
        raise ValueError("weighted precision requires a verified sample manifest")
    if isinstance(resamples, bool) or int(resamples) < 1:
        raise ValueError("bootstrap resamples must be positive")
    cases = merged.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("weighted precision has no adjudicated cases")
    estimate, numerator, effective_n = _weighted_precision(cases)
    denominator = sum(1.0 / float(case["inclusion_probability"]) for case in cases)
    raw_precision = sum(int(case["is_match"]) for case in cases) / len(cases)
    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        stratum_id = _nonempty(case.get("stratum_id"), "adjudication stratum_id")
        by_stratum[stratum_id].append(case)
    rng = random.Random(int(seed))
    bootstrap: list[float] = []
    for _ in range(int(resamples)):
        replicate: list[Mapping[str, Any]] = []
        for stratum_id in sorted(by_stratum):
            stratum_cases = by_stratum[stratum_id]
            replicate.extend(rng.choices(stratum_cases, k=len(stratum_cases)))
        replicate_estimate, _, _ = _weighted_precision(replicate)
        bootstrap.append(replicate_estimate)
    ci_low = _percentile(bootstrap, 0.025)
    ci_high = _percentile(bootstrap, 0.975)
    return {
        "schema_version": 1,
        "status": "complete",
        "estimator": "hajek_inverse_probability_weighted_precision",
        "precision": estimate,
        "raw_precision": raw_precision,
        "weighted_matches": numerator,
        "weighted_predictions": denominator,
        "effective_sample_size": effective_n,
        "sample_size": len(cases),
        "sample_hash": merged.get("sample_hash"),
        "confidence_interval": {
            "method": "stratified_percentile_bootstrap",
            "confidence": 0.95,
            "low": ci_low,
            "high": ci_high,
            "resamples": int(resamples),
            "seed": int(seed),
            "independent_unit": "adjudicated_mapping",
        },
        "inter_annotator_agreement": merged.get("inter_annotator_agreement"),
    }


__all__ = [
    "adjudication_sample_size",
    "build_blinded_sample",
    "import_annotations",
    "inverse_probability_weighted_precision",
    "merge_adjudications",
    "verify_sample_manifest",
    "write_blinded_sample",
]
