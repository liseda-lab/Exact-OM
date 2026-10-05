"""Fresh-evaluation intended-parent qualification, separate from arm comparisons.

The resource allowance is inherited from development qualification, never chosen
from held-out arm outcomes. Original results, queries and budgets remain fixed.
"""
from __future__ import annotations

import argparse
import fcntl
from collections import Counter
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, read_record
from tools.repair import development_intended as native
from tools.repair.batch import read, sha
from tools.repair.expanded_profile import checkpoint, parent_fingerprints
from tools.repair.historical_regression import binding, verify_binding
from tools.repair.prepare import case_from_dict

SCHEMA = 'exact-repair/fresh-evaluation-intended-resource/v1'
BUDGET = dict(wall_seconds=600, cpu_seconds=1200, memory_mb=8192)
query_scope = native.query_scope


def qualified_case(row):
    """Reject training-test substitution, changed identities and renamed parents."""
    record = verify_binding(row['evaluator'])
    case = case_from_dict(record)
    observable = read_record(verify_binding(row['observable']))
    if (row['status'] != 'materialized' or row['split'] != 'fresh_evaluation'
            or case.split != 'test' or case.schema_revision != 'v3'
            or case.case_id != row['case_id'] or record['hash'] != row['case_hash']
            or case.structural_parent != row['structural_parent']
            or case.problem.content_hash != row['input_hash'] or observable != case.problem
            or case.control != row['control'] or case.family != row['family']
            or set(parent_fingerprints(case)) != set(row['fingerprints'])
            or case.final_candidate_removals):
        raise ValueError('Frozen fresh-evaluation case, split or fingerprint differs')
    return row, record, case


def validate_schedule(plan):
    if (plan.get('schema') != SCHEMA or plan.get('split') != 'fresh_evaluation'
            or plan.get('expected_cases') != 64 or plan.get('expected_parents') != 32
            or plan.get('model_fitting') is not False
            or plan.get('supervision_admitted') is not False
            or plan.get('test_feedback_for_selection') is not False
            or plan.get('original_results_replaced') is not False
            or plan.get('scientific_comparison_budgets_changed') is not False
            or plan.get('budget') != BUDGET):
        raise ValueError('Fresh qualification boundary or resource budget differs')
    primary = verify_binding(plan['primary_schedule'])
    completion = verify_binding(plan['corpus_completion'])
    release = verify_binding(plan['fresh_manifest'])
    if (primary['schema'] != 'exact-repair/fresh-evaluation/v1'
            or primary['corpus_completion'] != plan['corpus_completion']
            or completion['status'] != 'complete'
            or completion['releases']['fresh_evaluation']['manifest'] != plan['fresh_manifest']
            or release['split'] != 'fresh_evaluation' or release['scheduled'] != 64
            or plan['cases'] != primary['cases'] or plan['cases'] != release['rows']
            or len(plan['cases']) != 64 or len({r['case_id'] for r in plan['cases']}) != 64
            or len({r['group_id'] for r in plan['cases']}) != 32):
        raise ValueError('Frozen fresh schedule or release identity differs')
    cases = [qualified_case(row) for row in plan['cases']]
    if (len({c.structural_parent for _, _, c in cases}) != 32
            or plan['query_scope'] != [dict(case_id=c.case_id, probes=query_scope(c))
                                      for _, _, c in cases]):
        raise ValueError('Frozen parents or typed queries differ')
    for parent in {c.structural_parent for _, _, c in cases}:
        group = [c for _, _, c in cases if c.structural_parent == parent]
        if Counter(c.control for c in group) != {'corrupted': 1, 'coherent': 1}:
            raise ValueError('Fresh parent clean/corrupted pairing differs')
    return cases


def case_worker(record, directory):
    case = case_from_dict(record)
    if case.split != 'test' or case.schema_revision != 'v3':
        raise ValueError('Fresh intended worker only admits the validated v3 test cases')
    return native.check_intended(case, directory)


def one_case(row, record, case, output, identity, budget):
    return native.one_case(row, record, case, output, identity, budget,
                           worker=case_worker, schema=SCHEMA)


def run(plan_path, output, start, stop):
    from exact.repair.study import runtime_manifest

    plan = read(plan_path)
    cases = validate_schedule(plan)
    if not 0 <= start < stop <= len(cases):
        raise ValueError('Invalid fresh intended resource slice')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = canonical_hash((binding(plan_path), sha(__file__), sha(native.__file__),
        sha(Path(__file__).with_name('acquisition.py')), runtime_manifest()))
    with (output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for row, record, case in cases[start:stop]:
            write_artifact(output / 'progress.json', dict(completed_rows=len(rows), case_id=case.case_id))
            rows.append(one_case(row, record, case, output, identity, plan['budget']))
        saved = [verify_binding(r) for r in rows]
        return checkpoint(output / 'report.json', identity, schema=SCHEMA, status='complete',
            plan=binding(plan_path), scheduled_rows=stop-start, recorded_rows=len(rows), rows=rows,
            row_slice=[start, stop], study_scheduled_cases=64, study_parent_groups=32,
            counts=dict(Counter(r['status'] for r in saved)),
            query_denominator=sum(r['query_denominator'] for r in saved),
            heldout_cases_opened=True, supervision_admitted=False, gates_passed=False,
            original_results_replaced=False, fitting_eligible=False,
            strongest_symbolic_comparison_qualified=False,
            next_stage='xr21-expanded-evaluation-001', limitations=[
                'Separately charged intended-only resource experiment; original comparison budgets unchanged.',
                'Typed nonvacuity is derived where not explicit; no unwanted probes were added.',
                'Qualification does not alter original selected-assignment labels, scores or unknowns.',
                'Scope is the frozen fresh intended theory, policy, queries and actual native backend.',
                'No fitting, query augmentation, test-based selection, G0-G2 or learning-efficiency claim.',
                'Native receipt review, required semantic-control review and parent-group final report remain.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--stop', type=int, required=True)
    args = parser.parse_args()
    run(args.plan, args.output, args.start, args.stop)


if __name__ == '__main__':
    main()
