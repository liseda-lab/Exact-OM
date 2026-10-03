"""Explicit row repair for the historical routing adapter; immutable prior evidence."""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import math
from collections import Counter
from pathlib import Path

from exact.repair.records import canonical_hash, read_record
from tools.repair.batch import read, sha
from tools.repair.expanded_profile import checkpoint
from tools.repair.historical_audit import _one, checked_content
from tools.repair.historical_regression import binding, verify_binding

CAUSE = 'historical_native_active_original_routing'
ERROR = 'RuntimeError: generator raised StopIteration'


def repaired_input(original, previous):
    """Admit only confirmed pre-search adapter errors; debit previous wall time."""
    problem = read_record(original)
    if (previous['status'] != 'error' or previous['detail'] != ERROR
            or previous['stage'] != 'migrated_inventory'
            or previous.get('result') is not None or not previous['cleanup_complete']):
        raise ValueError('Only confirmed pre-search routing errors are repairable here')
    used = previous['elapsed_seconds']
    if not math.isfinite(used) or not 0 <= used < problem.budgets.total_seconds:
        raise ValueError('Prior row consumed its original scientific budget')
    matches = [tuple(c for c in obj.candidates if set(c.axioms) == set(obj.original_axioms))
               for obj in problem.objects]
    if not all(matches) or not any(all(c.active_expressions for c in cs) for cs in matches):
        raise ValueError('Input does not witness the active-original routing defect')
    return dataclasses.replace(problem, budgets=dataclasses.replace(
        problem.budgets, total_seconds=problem.budgets.total_seconds - used))


def validate_plan(plan):
    if plan['repair_cause'] != CAUSE or plan['same_cause_repair_attempt'] not in (1, 2):
        raise ValueError('Unknown cause or exhausted same-cause repair limit')
    schedule = verify_binding(plan['native_schedule'])
    generation = verify_binding(schedule['generation_plan'])
    cases = {r['case_id']: r for r in generation['rows']}
    if len(cases) != 90 or plan['original_native_denominator'] != 270:
        raise ValueError('Original denominator changed')
    seen = set()
    admitted = []
    for entry in plan['rows']:
        previous = verify_binding(entry['previous_row'])
        checked_content(entry['previous_row']['path'])
        key = (previous['case_id'], previous['stage'])
        if key in seen or key != (entry['case_id'], entry['stage']):
            raise ValueError('Duplicate or misidentified repair row')
        seen.add(key)
        directory = Path(entry['previous_row']['path']).parents[1]
        stem = Path(entry['previous_row']['path']).stem
        if (directory/'inflight'/f'{stem}.json').exists() or (directory/'payloads'/stem).exists():
            raise ValueError('Prior native work needs ledger/owner reconciliation')
        original = cases[entry['case_id']]['migrated_case']['case']['problem']
        problem = repaired_input(original, previous)
        if problem.to_dict() != entry['remaining_budget_input']:
            raise ValueError('Recovery input or remaining budget changed')
        admitted.append((entry, previous, problem))
    if len(admitted) != plan['expected_repair_rows'] or not admitted:
        raise ValueError('Recovery denominator changed')
    return schedule, admitted


def run(plan_path, output):
    from exact.repair.study import runtime_manifest

    plan = read(plan_path)
    schedule, admitted = validate_plan(plan)
    runtime = runtime_manifest()
    identity = canonical_hash((sha(plan_path), sha(__file__),
                               sha(Path(__file__).with_name('historical_audit.py')), runtime))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for entry, previous, problem in admitted:
            key = canonical_hash((entry['case_id'], entry['stage']))
            metadata = dict(case_id=entry['case_id'], stage=entry['stage'], family=previous['family'],
                formerly_failed=previous['formerly_failed'], generation_receipt=None,
                repair_of=entry['previous_row'], repair_cause=CAUSE,
                same_cause_repair_attempt=plan['same_cause_repair_attempt'],
                original_total_seconds=problem.budgets.total_seconds + previous['elapsed_seconds'],
                prior_elapsed_seconds=previous['elapsed_seconds'],
                remaining_total_seconds=problem.budgets.total_seconds,
                scope='Explicit software repair; previous costs retained; no scientific budget reset')
            row = _one(output, identity, key, problem.to_dict(), schedule, metadata)
            rows.append(dict(**binding(output/'rows'/f'{key}.json'), **metadata, status=row['status']))
        return checkpoint(output/'report.json', identity,
            schema='exact-repair/historical-native-row-repair/v1', status='complete',
            plan=binding(plan_path), runtime=runtime, rows=rows,
            scheduled_rows=len(admitted), recorded_rows=len(rows),
            counts=dict(Counter(r['status'] for r in rows)), study_complete=False,
            original_native_denominator=270, original_results_immutable=True,
            next_stage='xr21-expanded-historical-audit-001',
            gates={g:'not_established_requires_final_receipt_review' for g in ('G0','G1','G2')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)


if __name__ == '__main__':
    main()
