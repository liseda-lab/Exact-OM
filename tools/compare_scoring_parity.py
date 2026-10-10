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

from exact.runs.store import ExplanationStore, _atomic_json
from exact.utils.provenance import sha256_file


def compare_records(left, right, *, tolerance=1e-5):
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
                if not (math.isnan(a) and math.isnan(b)) and a != b:
                    numeric.append(path + ": nonfinite mismatch")
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
                    groups[number][(row.get("original_query_id"), row["src_iri"])].append((float(value), row["tgt_iri"]))
        for source in groups[0].keys() | groups[1].keys():
            rankings = [[target for _, target in sorted(g[source], key=lambda pair: (-pair[0], pair[1]))]
                        for g in groups]
            if rankings[0] != rankings[1]:
                discrete.append(f"{field} ranking:{source}")
    return {"compared_rows": len(left), "compared_floats": compared_floats,
            "absolute_tolerance": tolerance, "max_absolute_error": max_error,
            "numeric_mismatch_count": len(numeric), "discrete_mismatch_count": len(discrete),
            "numeric_mismatches_sample": numeric[:100], "discrete_mismatches_sample": discrete[:100],
            "natural_decisions_equal": not discrete,
            "status": "passed" if not discrete and not numeric else "failed"}


def compare(left_path, right_path, output):
    left_receipt = json.loads((left_path / "cold.json").read_text())
    right_receipt = json.loads((right_path / "cold.json").read_text())
    count = sum(chunk["rows"] for chunk in left_receipt["chunks"])
    left = list(ExplanationStore(left_path / "cold/ordinary-evidence", read_only=True).iter_all())
    right = list(itertools.islice(ExplanationStore(right_path / "cold/ordinary-evidence", read_only=True).iter_all(), count))
    if len(left) != count:
        raise ValueError("Legacy numerator differs from durably stored ordinary records")
    prefix = right_receipt["chunks"][:len(left_receipt["chunks"])]
    boundaries = ([c["query_ids"] for c in left_receipt["chunks"]]
                  == [c["query_ids"] for c in prefix])
    for records, root, chunks in ((left, left_path, left_receipt["chunks"]), (right, right_path, prefix)):
        # Older immutable qualification snapshots did not stamp query IDs. Rebind
        # them using the original hashed workload and exact chunk boundaries.
        if records and any("original_query_id" not in row for row in records):
            measured = root / "measurement.json"
            workload_binding = (json.loads(measured.read_text())["workload"] if measured.exists()
                                else json.loads((root / "recovery-runtime.json").read_text())["identity"]["inputs"]["workload"])
            workload_path = Path(workload_binding["path"])
            if sha256_file(workload_path) != workload_binding["sha256"]:
                raise ValueError("Changed qualification workload binding")
            queries = {q["qid"]: q for q in json.loads(workload_path.read_text())["queries"]}
            start = 0
            for chunk in chunks:
                by_source = {queries[qid]["source"]: qid for qid in chunk["query_ids"]}
                if len(by_source) != len(chunk["query_ids"]):
                    raise ValueError("Ambiguous repeated-source chunk")
                for row in records[start:start + chunk["rows"]]:
                    row["original_query_id"] = by_source[row["src_iri"]]
                start += chunk["rows"]
    receipt = {**compare_records(left, right), "schema_version": 1,
               "namespace": "qualification_only", "query_boundaries_equal": boundaries,
               "baseline": {"path": str(left_path), "cold_sha256": sha256_file(left_path / "cold.json")},
               "candidate": {"path": str(right_path), "cold_sha256": sha256_file(right_path / "cold.json")},
               "hosted_quality_established": False, "fitted_selector_mapping_parity": "pending"}
    if not boundaries:
        receipt["status"] = "failed"
    _atomic_json(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.output)
    print(json.dumps({k: result[k] for k in ("status", "compared_rows", "max_absolute_error", "numeric_mismatch_count", "discrete_mismatch_count")}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
