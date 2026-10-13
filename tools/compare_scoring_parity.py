#!/usr/bin/env python3
"""Compare complete ordinary evidence and discrete decisions on a bounded GPU prefix."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from exact.runs.store import ExplanationStore, _atomic_json  # noqa: E402
from exact.utils.provenance import sha256_file  # noqa: E402


SYNTHETIC_DIAGNOSTIC_PATHS = (
    "experiment_diagnostics.difference.controlled_perturbations",
    "experiment_diagnostics.difference_replay",
)


def ordinary_records(records):
    """Exclude only T07's synthetic replays; retain every natural evidence field."""
    result = []
    for record in records:
        row = dict(record)
        if isinstance(row.get("experiment_diagnostics"), dict):
            diagnostics = row["experiment_diagnostics"] = dict(row["experiment_diagnostics"])
            diagnostics.pop("difference_replay", None)
            if isinstance(diagnostics.get("difference"), dict):
                difference = diagnostics["difference"] = dict(diagnostics["difference"])
                difference.pop("controlled_perturbations", None)
        result.append(row)
    return result


def compare_records(left, right, *, tolerance=1e-5, ordinary_only=False):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Parity tolerance must be finite and nonnegative")
    if ordinary_only:
        left, right = ordinary_records(left), ordinary_records(right)
    numeric = []
    discrete = []
    max_error = 0.0
    compared_floats = 0

    def check(a, b, path):
        nonlocal max_error, compared_floats
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                discrete.append(path + ": keys")
            for key in a.keys() & b.keys():
                check(a[key], b[key], path + "." + str(key))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                discrete.append(path + ": length")
            for i, (av, bv) in enumerate(zip(a, b)):
                check(av, bv, path + f"[{i}]")
        elif isinstance(a, float) and isinstance(b, (int, float)) and not isinstance(b, bool):
            compared_floats += 1
            error = abs(a - b)
            if not math.isfinite(a) or not math.isfinite(b):
                # Equal NaNs/infinities do not establish usable numerical output.
                numeric.append(path + ": nonfinite value")
            else:
                max_error = max(max_error, error)
                if error > tolerance:
                    numeric.append(path)
        elif type(a) is not type(b) or a != b:
            # Strings include exact selected facts and the hosted evidence packet.
            # Booleans, integer positions and ordered lists are never approximate.
            discrete.append(path)

    if len(left) != len(right):
        discrete.append("ordinary record count")
    if not left or not right:
        discrete.append("no ordinary records to compare")
    for index, (a, b) in enumerate(zip(left, right)):
        check(a, b, f"pair[{index}]")
    for field in ("S_final", "S_base", "S_struct", "s_label"):
        groups = [defaultdict(list), defaultdict(list)]
        for number, records in enumerate((left, right)):
            for row in records:
                value = row.get("confidences", {}).get(field)
                if value is not None:
                    groups[number][(row.get("original_query_id"), row["src_iri"])].append(
                        (float(value), row["tgt_iri"])
                    )
        for source in groups[0].keys() | groups[1].keys():
            rankings = [
                [target for _, target in sorted(g[source], key=lambda pair: (-pair[0], pair[1]))]
                for g in groups
            ]
            if rankings[0] != rankings[1]:
                discrete.append(f"{field} ranking:{source}")
    return {
        "compared_rows": len(left),
        "compared_floats": compared_floats,
        "excluded_diagnostic_paths": list(SYNTHETIC_DIAGNOSTIC_PATHS) if ordinary_only else [],
        "absolute_tolerance": tolerance,
        "max_absolute_error": max_error,
        "numeric_mismatch_count": len(numeric),
        "discrete_mismatch_count": len(discrete),
        "numeric_mismatches_sample": numeric[:100],
        "discrete_mismatches_sample": discrete[:100],
        "natural_decisions_equal": not discrete,
        "status": "passed" if not discrete and not numeric else "failed",
    }


def compare_fitted(left_path, right_path, *, ordinary, ordinary_only=False, mode="cold"):
    """Compare the same bounded fitted cohort, including prompts and extraction."""
    measurements = [
        json.loads((root / "measurement.json").read_text()) for root in (left_path, right_path)
    ]
    for measurement in measurements:
        binding = measurement.get("fitted_manifest")
        if (
            not binding
            or measurement.get("namespace") != "qualification_only"
            or measurement.get("paid_calls") != 0
            or measurement.get("fixture_outputs_promotable") is not False
        ):
            raise ValueError("Fitted parity requires isolated unpaid fixture measurements")
        if sha256_file(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("Changed fitted qualification manifest")
    if measurements[0]["fitted_manifest"] != measurements[1]["fitted_manifest"]:
        raise ValueError("Fitted parity requires the same admitted fitted recipe")

    decisions, evidence, prompts = [], [], []
    for root in (left_path, right_path):
        decisions.append(json.loads((root / mode / "fitted-decisions.json").read_text()))
        evidence.append(json.loads((root / mode / "selector-evidence.json").read_text()))
        prompts.append(json.loads((root / mode / "hosted-fixtures.json").read_text()))
    rows = []
    for result, records, selected in zip(decisions, ordinary, evidence):
        if (
            result.get("namespace") != "qualification_only"
            or result.get("selector_scope") != "bounded_cohort"
            or result.get("hosted_quality_established") is not False
        ):
            raise ValueError("Scientific outputs cannot enter fitted fixture parity")
        required = {"rows", "policy", "protected_pairs", "mappings"}
        if (
            not required <= result.keys()
            or not isinstance(result["policy"], dict)
            or not {"threshold", "source_cardinality", "target_cardinality", "extraction"}
            <= result["policy"].keys()
        ):
            raise ValueError("Incomplete fitted selector or extraction result")
        for name in ("mappings", "protected_pairs"):
            value = result[name]
            if (
                not isinstance(value, list)
                or any(
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or any(not isinstance(iri, str) or not iri for iri in pair)
                    for pair in value
                )
                or len(set(map(tuple, value))) != len(value)
            ):
                raise ValueError("Incomplete fitted mapping or protected-pair identities")
        cohort = [(row["src_iri"], row["tgt_iri"]) for row in records]
        if (
            not cohort
            or len(set(cohort)) != len(cohort)
            or [(row["Src"], row["Tgt"]) for row in result["rows"]] != cohort
            or [(row["src_iri"], row["tgt_iri"]) for row in selected] != cohort
            or [row.get("original_query_id") for row in selected]
            != [row["original_query_id"] for row in records]
        ):
            raise ValueError(
                "Fitted rows and selector evidence must cover the complete ordinary cohort"
            )
        if any(
            isinstance(row.get("S_final"), bool)
            or not isinstance(row.get("S_final"), (int, float))
            or not math.isfinite(row["S_final"])
            for row in result["rows"]
        ):
            raise ValueError("Fitted rows require finite final scores")
        rows.append(
            [
                {
                    **row,
                    "src_iri": row["Src"],
                    "tgt_iri": row["Tgt"],
                    "confidences": {
                        field: row[field]
                        for field in ("S_final", "S_base", "S_struct", "s_label")
                        if field in row
                    },
                }
                for row in result["rows"]
            ]
        )
    row_parity = compare_records(*rows)
    evidence_parity = compare_records(*evidence, ordinary_only=ordinary_only)
    # Policy, protected exact anchors and final mapping membership are discrete.
    metadata_equal = {key: value for key, value in decisions[0].items() if key != "rows"} == {
        key: value for key, value in decisions[1].items() if key != "rows"
    }
    prompts_equal = prompts[0] == prompts[1]
    passed = (
        row_parity["status"] == evidence_parity["status"] == "passed"
        and metadata_equal
        and prompts_equal
    )
    return {
        "status": "passed" if passed else "failed",
        "scope": "same_bounded_cohort_with_synthetic_hosted_responses",
        "rows": row_parity,
        "selector_evidence": evidence_parity,
        "extraction_policy_anchors_and_mappings_equal": metadata_equal,
        "exact_hosted_fixture_inputs_and_responses_equal": prompts_equal,
        "hosted_fixture_records": [len(value) for value in prompts],
        "historical_paid_prediction_parity_established": False,
    }


def compare(left_path, right_path, output, *, ordinary_only=False, fitted=False, mode="cold"):
    if mode not in {"cold", "warm", "replay", "partial"}:
        raise ValueError("Unknown cache comparison mode")
    left_receipt = json.loads((left_path / (mode + ".json")).read_text())
    right_receipt = json.loads((right_path / (mode + ".json")).read_text())
    count = sum(chunk["rows"] for chunk in left_receipt["chunks"])
    left = list(ExplanationStore(left_path / mode / "ordinary-evidence", read_only=True).iter_all())
    candidate_records = ExplanationStore(
        right_path / mode / "ordinary-evidence", read_only=True
    ).iter_all()
    right = list(candidate_records if fitted else itertools.islice(candidate_records, count))
    if len(left) != count:
        raise ValueError("Legacy numerator differs from durably stored ordinary records")
    if fitted and len(right) != sum(chunk["rows"] for chunk in right_receipt["chunks"]):
        raise ValueError("Candidate numerator differs from its complete durable ordinary store")
    prefix = right_receipt["chunks"][: len(left_receipt["chunks"])]
    boundaries = [c["query_ids"] for c in left_receipt["chunks"]] == [
        c["query_ids"] for c in prefix
    ]
    for records, root, chunks in (
        (left, left_path, left_receipt["chunks"]),
        (right, right_path, prefix),
    ):
        # Older immutable qualification snapshots did not stamp query IDs. Rebind
        # them using the original hashed workload and exact chunk boundaries.
        if records and any("original_query_id" not in row for row in records):
            if fitted or any("original_query_id" in row for row in records):
                raise ValueError(
                    "Incomplete query identities cannot be repaired during parity checks"
                )
            measured = root / "measurement.json"
            workload_binding = (
                json.loads(measured.read_text())["workload"]
                if measured.exists()
                else json.loads((root / "recovery-runtime.json").read_text())["identity"]["inputs"][
                    "workload"
                ]
            )
            workload_path = Path(workload_binding["path"])
            if sha256_file(workload_path) != workload_binding["sha256"]:
                raise ValueError("Changed qualification workload binding")
            queries = {q["qid"]: q for q in json.loads(workload_path.read_text())["queries"]}
            start = 0
            for chunk in chunks:
                by_source = {queries[qid]["source"]: qid for qid in chunk["query_ids"]}
                if len(by_source) != len(chunk["query_ids"]):
                    raise ValueError("Ambiguous repeated-source chunk")
                for row in records[start : start + chunk["rows"]]:
                    row["original_query_id"] = by_source[row["src_iri"]]
                start += chunk["rows"]
    receipt = {
        **compare_records(left, right, ordinary_only=ordinary_only),
        "schema_version": 1,
        "namespace": "qualification_only",
        "query_boundaries_equal": boundaries,
        "compared_mode": mode,
        "baseline": {
            "path": str(left_path),
            mode + "_sha256": sha256_file(left_path / (mode + ".json")),
        },
        "candidate": {
            "path": str(right_path),
            mode + "_sha256": sha256_file(right_path / (mode + ".json")),
        },
        "hosted_quality_established": False,
        "fitted_selector_mapping_parity": "pending",
    }
    if not boundaries:
        receipt["status"] = "failed"
    if fitted:
        if (
            not boundaries
            or len(left_receipt["chunks"]) != len(right_receipt["chunks"])
            or count != sum(chunk["rows"] for chunk in right_receipt["chunks"])
        ):
            raise ValueError("Fitted selector parity requires equal complete cohorts, not a prefix")
        receipt["fitted"] = compare_fitted(
            left_path,
            right_path,
            ordinary=(left, right),
            ordinary_only=ordinary_only,
            mode=mode,
        )
        receipt["fitted_selector_mapping_parity"] = receipt["fitted"]["status"]
        if receipt["fitted"]["status"] != "passed":
            receipt["status"] = "failed"
    _atomic_json(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--ordinary-only",
        action="store_true",
        help="Exclude only the two declared synthetic-diagnostic payload paths",
    )
    parser.add_argument(
        "--fitted",
        action="store_true",
        help="Also require fitted selectors, mappings and exact hosted fixture parity",
    )
    parser.add_argument("--mode", choices=("cold", "warm", "replay", "partial"), default="cold")
    args = parser.parse_args()
    result = compare(
        args.baseline,
        args.candidate,
        args.output,
        ordinary_only=args.ordinary_only,
        fitted=args.fitted,
        mode=args.mode,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "compared_rows",
                    "max_absolute_error",
                    "numeric_mismatch_count",
                    "discrete_mismatch_count",
                )
            },
            indent=2,
        )
    )
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
