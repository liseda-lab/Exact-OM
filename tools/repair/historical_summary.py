"""Validate and close a historical engineering audit without promoting research gates."""
from __future__ import annotations

import argparse
import dataclasses
from collections import Counter, defaultdict
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, read_record
from tools.repair.batch import read, sha
from tools.repair.historical_audit import checked_content, prepare, validate_attempt
from tools.repair.historical_audit_recovery import validate_plan
from tools.repair.historical_regression import binding, verify_binding


def revised_view(original, replacements):
    """Replace only explicitly bound errors and retain the original denominator."""
    indexed = {(r['case_id'], r['stage']): r for r in original}
    if len(indexed) != len(original):
        raise ValueError('Duplicate original historical row')
    seen = set()
    for row in replacements:
        key = (row['case_id'], row['stage'])
        if key in seen or key not in indexed:
            raise ValueError('Duplicate or unscheduled replacement row')
        seen.add(key)
        previous = indexed[key]
        if previous['status'] != 'error' or row['repair_of'] != previous['receipt']:
            raise ValueError('Replacement does not bind an original error')
        if not row['cleanup_complete'] or row['same_cause_repair_attempt'] not in (1, 2):
            raise ValueError('Unqualified replacement row')
        if abs(row['prior_elapsed_seconds'] + row['remaining_total_seconds']
               - row['original_total_seconds']) > 1e-8:
            raise ValueError('Replacement reset the scientific time budget')
        indexed[key] = row
    return list(indexed.values())


def event_backends(value):
    """Inspect actual native verification events, never admitted-route names."""
    result = set()
    if isinstance(value, dict):
        if value.get('$record') == 'VerificationEventV3' and value.get('backend'):
            result.add(value['backend'])
        for child in value.values():
            result.update(event_backends(child))
    elif isinstance(value, list):
        for child in value:
            result.update(event_backends(child))
    return result


def counts(rows):
    return dict(sorted(Counter(row['status'] for row in rows).items()))


def audit(plan):
    schedule = verify_binding(plan['schedule'])
    if prepare(schedule['generation_plan']['path'], schedule['attempts'],
               schedule['requirement_audit']['path']) != schedule:
        raise ValueError('Historical generation schedule changed')
    original = verify_binding(plan['original_audit'])
    original_rows = original['native_rows']
    generation_rows = original['generation_rows']
    if (len(schedule['cases']), len(generation_rows), len(original_rows)) != (90, 180, 270):
        raise ValueError('Historical denominator changed')
    if sum(c['formerly_failed'] for c in schedule['cases']) != 79:
        raise ValueError('Formerly failed case denominator changed')
    scheduled = {c['case_id']: c for c in schedule['cases']}
    expected = {(c, stage) for c in scheduled for stage in schedule['stages']}
    if {(r['case_id'], r['stage']) for r in original_rows} != expected:
        raise ValueError('Historical stage coverage changed')
    expected_generation = {(c, mode) for c in scheduled for mode in ('cold', 'warm')}
    if {(r['case_id'], r['mode']) for r in generation_rows} != expected_generation:
        raise ValueError('Historical generation mode coverage changed')
    for row in generation_rows:
        if row['receipt'] != scheduled[row['case_id']]['modes'][row['mode']]['receipt']:
            raise ValueError('Generation receipt schedule mismatch')
        checked_content(row['receipt']['path'])
    attempts = [*schedule['attempts'], *original['attempts'], plan['repair_attempt']]
    ledger = verify_binding(plan['ledger_snapshot'])
    charges = []
    validated_rows = {}
    for attempt in attempts:
        report, batch = validate_attempt(attempt)
        completion = verify_binding(attempt['completion'])
        directory = Path(attempt['completion']['path']).parent
        step = read(directory / 'step.json')
        if (step['step_id'], step['dispatch_nonce']) != (attempt['step_id'], attempt['nonce']):
            raise ValueError('Historical step ownership changed')
        if int((directory / 'exit_code').read_text()) != 0:
            raise ValueError('Historical process did not complete')
        if report['runtime'] != read(Path(batch['code']).parent / 'runtime.json'):
            raise ValueError('Historical runtime differs from its frozen source')
        work = Path(completion['work'])
        outputs = verify_binding(attempt['outputs'])
        if set(outputs) != {str(p.relative_to(work)) for p in work.rglob('*')
                           if p.is_file() and p.suffix != '.lock'}:
            raise ValueError('Historical output manifest is incomplete')
        if list(work.glob('**/inflight/*.json')):
            raise ValueError('Historical in-flight row lacks reconciliation')
        charge = ledger['attempts'][str(directory)]
        if (charge['status'] != 'settled'
                or charge['elapsed_seconds'] != completion['elapsed_seconds']
                or charge['cpu_seconds'] != completion['cpu_seconds']):
            raise ValueError('Historical cost ledger differs from completion')
        charges.append(dict(attempt=binding(directory / 'completion.json'), **charge))
        if attempt in original['attempts']:
            if report['plan'] != plan['schedule']:
                raise ValueError('Native original schedule mismatch')
            for row_ref in report['rows']:
                row = verify_binding(row_ref)
                checked_content(row_ref['path'])
                if row_ref['path'] in validated_rows:
                    raise ValueError('Duplicate original native receipt')
                validated_rows[row_ref['path']] = row
    for row in original_rows:
        actual = validated_rows.get(row['receipt']['path'])
        if actual is None or binding(row['receipt']['path']) != row['receipt']:
            raise ValueError('Original row absent from completed attempt')
        if any(actual[k] != row[k] for k in ('case_id', 'stage', 'status', 'family',
                                            'formerly_failed', 'elapsed_seconds', 'result')):
            raise ValueError('Original audit does not match its native row')
        if not actual['cleanup_complete']:
            raise ValueError('Original row cleanup incomplete')
    repair, batch = validate_attempt(plan['repair_attempt'])
    recovery_plan = verify_binding(repair['plan'])
    _, admitted = validate_plan(recovery_plan)
    if repair['recorded_rows'] != repair['scheduled_rows'] or len(repair['rows']) != len(admitted):
        raise ValueError('Repair report denominator mismatch')
    repairs = []
    by_key = {(entry['case_id'], entry['stage']): (entry, prior, problem)
              for entry, prior, problem in admitted}
    for ref in repair['rows']:
        row = verify_binding(ref)
        checked_content(ref['path'])
        entry, prior, problem = by_key[(row['case_id'], row['stage'])]
        if row['repair_of'] != entry['previous_row'] or row['prior_elapsed_seconds'] != prior['elapsed_seconds']:
            raise ValueError('Repair ancestry mismatch')
        if row['remaining_total_seconds'] != problem.budgets.total_seconds:
            raise ValueError('Repair budget differs from its admitted plan')
        expected_identity = canonical_hash((repair['identity'], canonical_hash((row['case_id'], row['stage'])), problem.to_dict()))
        if row['identity'] != expected_identity:
            raise ValueError('Repair row identity mismatch')
        if row.get('result'):
            payload = verify_binding(row['result']['payload'])
            effective = dataclasses.replace(problem, budgets=dataclasses.replace(problem.budgets,
                memory_mb=min(problem.budgets.memory_mb or schedule['memory_mb'], schedule['memory_mb'])))
            if payload['input_hash'] != effective.content_hash or payload['pool_hash'] != canonical_hash(problem.objects):
                raise ValueError('Repair payload input mismatch')
            result = read_record(payload['result'])
            if (result.logical_status, result.search_status) != (row['result']['logical_status'], row['result']['search_status']):
                raise ValueError('Repair result status mismatch')
        repairs.append(dict(row, receipt=binding(ref['path'])))
    revised = revised_view(original_rows, repairs)
    families = defaultdict(list)
    backends = Counter()
    search_states = Counter()
    for row in revised:
        families[row['family']].append(row)
        result = row.get('result')
        if result:
            payload = verify_binding(result['payload'])
            read_record(payload['result'])
            if result.get('semantic_benefit') is not None:
                raise ValueError('Historical diagnostic cannot become semantic supervision')
            search_states[result['logical_status'] + '/' + result['search_status']] += 1
        directory = Path(row['receipt']['path']).parents[1]
        search = directory / 'payloads' / Path(row['receipt']['path']).stem / 'search-ledger.json'
        if search.exists():
            evidence = read(search)
            read_record(evidence)
            backends.update(event_backends(evidence))
    requirements = verify_binding(plan['requirement_review'])
    for ref in requirements['current_bindings']:
        if sha(Path(__file__).resolve().parents[2] / ref['file']) != ref['sha256']:
            raise ValueError('Source changed after requirement review: ' + ref['file'])
    return dict(schema='exact-repair/historical-final-audit/v1', status='complete',
        study_complete=True, scope='Historical engineering regression only',
        denominators=original['denominators'], generation_counts=original['generation_counts'],
        original_native_counts=counts(original_rows), revised_native_counts=counts(revised),
        replacement_counts=counts(repairs), revised_search_states=dict(search_states),
        family_native_counts={family: counts(rows) for family, rows in sorted(families.items())},
        formerly_failed_native_counts=counts([r for r in revised if r['formerly_failed']]),
        actual_backend_row_counts=dict(backends),
        backend_scope='Count each backend once per revised row with actual VerificationEventV3 evidence; route admission alone is excluded',
        original_generation_rows=generation_rows, original_native_rows=original_rows,
        replacement_rows=repairs, revised_native_rows=revised,
        requirements=plan['requirement_review'], requirement_counts=requirements['counts'],
        attempt_charges=charges, worker_seconds=sum(x['elapsed_seconds'] for x in charges),
        campaign_costs_snapshot=plan['ledger_snapshot'], costs_reset=False,
        gates={g: 'not_established' for g in ('G0', 'G1', 'G2')},
        limitations=['90 historically exposed development cases; no held-out scientific evaluation',
            'Original and revised views both retained; six replacements debit failed elapsed time',
            'Fixed-budget timeouts and missing generated pools remain in every denominator',
            'Separate stage timings are not a matched end-to-end comparison',
            'Negative edit-cost objective; no learned/teacher semantic superiority or learning efficiency',
            'Current-source requirement qualification and genuine training-label acquisition remain necessary'],
        next_stage='xr21-expanded-training-001',
        training_eligibility=dict(fitting='not_yet_qualified',
            independent_preparation='authorized',
            required=['Source-compatible requirement tests and explicit native scope decision',
                'Dependency-bound generated label acquisition retaining unknown/partial labels and inner native outcomes',
                'Generated-pool development checkpoint selection and circuit backward evidence'],
            forbidden=['Profiling checkpoint warm start', 'Using historical or test outcomes for fitting or settings']))


def run(plan_path, output):
    result = audit(read(plan_path))
    result['plan'] = binding(plan_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_artifact(output / 'report.json', result)
    write_artifact(output / 'completion.json', dict(schema='exact-repair/historical-final-completion/v1',
        status='complete', report=binding(output / 'report.json'), plan=binding(plan_path),
        study_complete=True, gates=result['gates']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.plan, args.output)


if __name__ == '__main__':
    main()
