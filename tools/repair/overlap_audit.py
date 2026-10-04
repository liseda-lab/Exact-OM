"""Close the controlled overlap stage from receipts, without new native calls."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

from exact.repair.records import canonical_hash


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Evidence:
    def __init__(self):
        self.files = {}

    def verify(self, ref):
        path = Path(ref["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == ref["sha256"], "Changed evidence: " + str(path))
        require(self.files.get(str(path), digest) == digest, "Conflicting evidence binding")
        self.files[str(path)] = digest
        return path

    def read(self, ref):
        return json.loads(self.verify(ref).read_text())

    def checkpoint(self, ref, identity):
        value = self.read(ref)
        require(value.get("identity") == identity, "Checkpoint dependencies differ")
        payload = {k: v for k, v in value.items() if k != "content_hash"}
        require(canonical_hash(payload) == value.get("content_hash"), "Checkpoint content differs")
        return value


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def receipt(evidence, item, ledger):
    run = item["run"]
    complete = evidence.read(item["completion"])
    step = evidence.read(item["step"])
    for value in (complete, step):
        require(value["step_id"] == run["step_id"], "Slurm receipt owner differs")
        require(value["dispatch_nonce"] == run["dispatch_nonce"], "Dispatch nonce differs")
    require(complete["job_id"] == run["logical_id"], "Logical job differs")
    require(int(evidence.verify(item["exit"]).read_text()) == complete["exit_code"],
            "Exit receipt differs")
    active = not run.get("superseded_by")
    require(not active or (complete["status"] == "complete" and complete["exit_code"] == 0),
            "Active overlap descendant is incomplete")
    batch_ref = item["batch"]
    require(complete["batch"] == batch_ref["path"], "Batch receipt differs")
    batch = evidence.read(batch_ref)
    require(evidence.verify(item["batch_hash"]).read_text().strip() == batch_ref["sha256"],
            "Batch manifest digest differs")
    for path, sha in batch["frozen_files"].items():
        evidence.verify(dict(path=path, sha256=sha))
    runtime = evidence.read(item["runtime"])
    charge = ledger["attempts"][str(Path(run["completion_path"]).parent)]
    require(charge["logical_id"] == run["logical_id"] and charge["status"] == "settled",
            "Missing settled attempt charge")
    require(charge["elapsed_seconds"] >= complete["elapsed_seconds"] and
            charge["cpu_seconds"] >= complete["cpu_seconds"], "Attempt cost was lost")
    if charge.get("slurm_reconciliation"):
        evidence.verify(charge["slurm_reconciliation"]["evidence"])
        evidence.verify(charge["slurm_reconciliation"]["original_wrapper_completion"])
    if active:
        for relative, sha in evidence.read(item["outputs"]).items():
            evidence.verify(dict(path=str(Path(complete["work"]) / relative), sha256=sha))
    return dict(run=run, completion=complete, batch=batch, runtime=runtime, charge=charge)


def validate_rows(evidence, schedule, report, evaluations):
    expected = {row["id"]: row for row in schedule["rows"]}
    require(len(expected) == len(schedule["rows"]) == 36, "Schedule denominator differs")
    rows, seen = [], set()
    for ref in report["evaluation_reports"]:
        source = evaluations[ref["path"]]
        saved_report = evidence.read(ref)
        require(saved_report["runtime"] == source["runtime"], "Evaluation runtime differs")
        effective = evidence.read(saved_report["schedule"])
        require(effective == dict(schedule, witness_report=report["witness_report"]),
                "Qualified evaluation schedule differs")
        fresh_path = str(Path(source["batch"]["code"]) / "tools/repair/fresh_evaluation.py")
        identity = canonical_hash((saved_report["schedule"]["sha256"],
                                   source["batch"]["frozen_files"][fresh_path],
                                   saved_report["runtime"]))
        evidence.checkpoint(ref, identity)
        require(saved_report["status"] == "complete", "Incomplete evaluation report")
        for row_ref in saved_report["rows"]:
            saved = evidence.read(row_ref)
            row = saved["row"]
            require(row["id"] not in seen and expected.get(row["id"]) == row,
                    "Duplicate or unexpected evaluation row")
            seen.add(row["id"])
            evidence.checkpoint(row_ref, canonical_hash((identity, row)))
            require(saved["cleanup_complete"], "Unresolved evaluation cleanup")
            for payload in saved.get("payloads", []):
                evidence.verify(payload)
            result = evidence.read(saved["result"]) if saved.get("result") else {}
            require(not result or result["row_id"] == row["id"], "Result row differs")
            item = schedule["cases"][row["case_index"]]
            rows.append(dict(row_id=row["id"], arm_id=row["arm_id"],
                             condition=item["condition"], control=item["control"],
                             status=saved["status"],
                             semantic_benefit=result.get("semantic_benefit"),
                             logical_status=result.get("logical_status", "UNKNOWN"),
                             elapsed_seconds=saved["elapsed_seconds"], resources=saved["resources"],
                             receipt=row_ref, scientific_status=result.get("status", saved["status"]),
                             detail=result.get("detail", saved.get("detail", ""))))
    require(seen == set(expected), "Incomplete evaluation denominator")
    original = [{k: v for k, v in row.items() if k not in ("scientific_status", "detail")}
                for row in rows]
    require(original == report["rows"], "Final report differs from actual row payloads")
    require(report["scheduled"] == report["recorded"] == len(rows), "Report denominator differs")
    require(report["outcomes"] == dict(Counter(row["status"] for row in rows)),
            "Report process statuses differ")
    return rows


def validate_pairs(schedule, report, rows):
    pairs = []
    for arm in schedule["arms"]:
        for control in ("corrupted", "coherent"):
            pair = {r["condition"]: r for r in rows
                    if r["arm_id"] == arm["id"] and r["control"] == control}
            require(set(pair) == {"nonshared", "shared"}, "Incomplete paired denominator")
            known = all(r["logical_status"] == "VERIFIED_FEASIBLE" and
                        r["semantic_benefit"] is not None for r in pair.values())
            pairs.append(dict(arm_id=arm["id"], control=control, available=known,
                              shared_minus_nonshared_benefit=(pair["shared"]["semantic_benefit"] -
                                  pair["nonshared"]["semantic_benefit"]) if known else None,
                              shared_minus_nonshared_seconds=pair["shared"]["elapsed_seconds"] -
                                  pair["nonshared"]["elapsed_seconds"],
                              rows={c: pair[c]["row_id"] for c in ("nonshared", "shared")}))
    require(pairs == report["pairs"], "Paired results differ from scientific availability")
    return pairs


def validate_witness(evidence, schedule, report, source):
    witness = evidence.read(report["witness_report"])
    path = str(Path(source["batch"]["code"]) / "tools/repair/conflict_overlap.py")
    identity = canonical_hash((report["schedule"]["sha256"],
                               source["batch"]["frozen_files"][path], source["runtime"]))
    evidence.checkpoint(report["witness_report"], identity)
    require(witness["schedule"] == report["schedule"], "Witness schedule differs")
    cases = {case["case_id"]: case for case in schedule["cases"]}
    seen, summary = set(), []
    for ref in witness["rows"]:
        saved = evidence.read(ref)
        require(saved["case_id"] not in seen and saved["case_id"] in cases,
                "Duplicate or unexpected witness")
        seen.add(saved["case_id"])
        case = cases[saved["case_id"]]
        evidence.checkpoint(ref, canonical_hash((identity, case)))
        require(saved["cleanup_complete"] and saved["status"] == "complete",
                "Witness process incomplete")
        result = saved["result"]
        require(result["qualified"] and result["input_hash"] == case["input_hash"],
                "Witness is not qualified for this input")
        checks = result["checks"]
        expected = ({tuple(s) for n in range(5) for s in itertools.combinations(range(4), n)}
                    if case["control"] == "corrupted" else {(0, 1)})
        require(len(checks) == len(expected) and {tuple(c["subset"]) for c in checks} == expected,
                "Witness subset denominator differs")
        for check in checks:
            evidence.verify(check["proof"])
            require(check["qualified"] and check["status"] in
                    {"VERIFIED_FEASIBLE", "VERIFIED_INFEASIBLE"}, "Unknown native witness")
        bad = [set(c["subset"]) for c in checks if c["status"] == "VERIFIED_INFEASIBLE"]
        minimal = sorted(sorted(s) for s in bad if not any(t < s for t in bad))
        require(minimal == result["actual_minimal_supports"] ==
                result["expected_minimal_supports"] == case["expected_minimal_supports"],
                "Native minimal supports differ")
        summary.append(dict(case_id=case["case_id"], condition=case["condition"],
                            control=case["control"], native_checks=len(checks), minimal_supports=minimal))
    require(seen == set(cases) and len(cases) == witness["scheduled"] == witness["qualified"] == 4,
            "Witness case denominator differs")
    return summary


def audit(manifest_path):
    evidence = Evidence()
    manifest = evidence.read(binding(manifest_path))
    report = evidence.read(manifest["report"])
    schedule = evidence.read(report["schedule"])
    ledger = evidence.read(manifest["ledger_snapshot"])
    require(ledger["limit_worker_seconds"] is None, "Expanded authorization missing")
    sources = [receipt(evidence, item, ledger) for item in manifest["attempts"]]
    runs = {source["run"]["id"]: source for source in sources}
    require(len(runs) == len(sources), "Duplicate attempt")
    for source in sources:
        run = source["run"]
        if run.get("superseded_by"):
            child = runs[run["superseded_by"]]["run"]
            require(child["recovery_of"] == run["id"] and child["repair_attempt"] >= 1,
                    "Recovery lineage lost")
    active = [s for s in sources if not s["run"].get("superseded_by")]
    require(len(active) == 5, "Expected witness, three evaluations and final report")
    result_sources = {str(Path(s["completion"]["work"]) / s["run"]["result_relative"]): s
                      for s in active}
    require(manifest["report"]["path"] in result_sources, "Final report lacks active receipt")
    witnesses = validate_witness(evidence, schedule, report,
                                 result_sources[report["witness_report"]["path"]])
    rows = validate_rows(evidence, schedule, report, result_sources)
    pairs = validate_pairs(schedule, report, rows)
    structure = evidence.read(schedule["structural_audit"])
    require(structure["independent_parent_count"] == report["independent_parent_count"] == 1 and
            structure["heldout_claim"] is False, "Diagnostic ancestry scope differs")
    require({c["group_id"] for c in schedule["cases"]} == {structure["source_parent"]} and
            {c["split"] for c in schedule["cases"]} == {structure["inherited_split"]},
            "Inherited parent or split differs")
    require(report["status"] == "complete" and report["study_complete"] is True,
            "Original final report incomplete")
    costs = {str(Path(s["run"]["completion_path"]).parent): s["charge"] for s in sources}
    return dict(schema="exact-repair/overlap-scope-completion/v1", status="complete",
                stage_id=manifest["stage_id"], study_complete=True,
                scope="controlled_overlap_diagnostic_only", campaign_complete=False,
                original_report=manifest["report"], manifest=binding(manifest_path),
                scheduled_rows=36, recorded_rows=len(rows), independent_parent_count=1,
                logical_statuses=dict(Counter(r["logical_status"] for r in rows)),
                scientific_statuses=dict(Counter(r["scientific_status"] for r in rows)),
                process_statuses=dict(Counter(r["status"] for r in rows)), rows=rows, pairs=pairs,
                available_pairs=sum(p["available"] for p in pairs), scheduled_pairs=len(pairs),
                native_witnesses=witnesses, structural_audit=schedule["structural_audit"],
                claim_scope=report["claim_scope"], uncertainty=report["uncertainty"],
                symbolic_limitation=schedule["symbolic_limitation"],
                gate_passage=False, heldout_claim=False, scientific_rows_rerun=0,
                attempt_costs=costs, worker_seconds=sum(c["elapsed_seconds"] for c in costs.values()),
                measured_cpu_seconds=sum(c["cpu_seconds"] for c in costs.values()),
                cost_scope="All overlap witness/evaluation/report attempts including failed originals; "
                           "shared preparation overhead remains in cumulative ledger. This audit worker "
                           "is separately charged by its batch receipt after this report is written.",
                cumulative_ledger_snapshot=manifest["ledger_snapshot"],
                attempt_lineage=[s["run"] for s in sources],
                verified_files=evidence.files, verified_file_count=len(evidence.files), api_spend_usd=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    result = audit(args.manifest)
    from exact.repair.api import write_artifact

    write_artifact(Path(args.output) / "report.json", result)


if __name__ == "__main__":
    main()
