"""Staged native historical diagnostics; never teacher labels or gate passage.

The original generation rows remain immutable. Missing generated pools consume
scheduled denominator rows, and native selection retains each saved case budget.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import time
from collections import Counter
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, canonical_json
from tools.repair.batch import read, sha
from tools.repair.expanded_profile import checked_checkpoint, checkpoint
from tools.repair.historical_regression import binding, verify_binding
from tools.repair.prepare import case_from_dict


def checked_content(path):
    value = read(path)
    if value.get('content_hash') != canonical_hash({k: v for k, v in value.items() if k != 'content_hash'}):
        raise ValueError('Historical checkpoint content changed: ' + str(path))
    return value


def validate_attempt(item):
    completion = verify_binding(item['completion'])
    outputs = verify_binding(item['outputs'])
    if (completion.get('status'), completion.get('exit_code'), completion.get('step_id'),
        completion.get('dispatch_nonce')) != ('complete', 0, item['step_id'], item['nonce']):
        raise ValueError('Historical completion ownership mismatch')
    batch = verify_binding(item['batch'])
    if completion['batch'] != item['batch']['path']:
        raise ValueError('Historical batch ownership mismatch')
    if sha(item['batch']['path']) != Path(item['batch']['path']).with_name('batch.sha256').read_text().strip():
        raise ValueError('Historical batch manifest changed')
    for path, digest in batch['frozen_files'].items():
        if sha(path) != digest:
            raise ValueError('Historical frozen dependency changed: ' + path)
    work = Path(completion['work'])
    for relative, digest in outputs.items():
        if sha(work / relative) != digest:
            raise ValueError('Historical completed output changed: ' + relative)
    report_path = Path(item['report']['path'])
    if outputs[str(report_path.relative_to(work))] != item['report']['sha256']:
        raise ValueError('Historical report missing from output manifest')
    verify_binding(item['report'])
    return checked_content(report_path), batch


def generation_classification(row):
    if row['call_status'] == 'complete':
        return (row.get('result') or {}).get('status', 'missing_result')
    detail = row.get('detail', '')
    if 'CircuitBudgetExceeded' in detail or 'deadline exhausted' in detail:
        return 'resource_bounded_original_limits'
    if 'cannot infer complete original relation for endpoint retrieval' in detail:
        return 'unsupported_original_mapping_bundle'
    return 'requires_software_or_resource_review'


def prepare(plan_path, attempts, requirement_path):
    plan = read(plan_path)
    rows = {}
    evidence = []
    for item in attempts:
        report, batch = validate_attempt(item)
        if report['plan'] != binding(plan_path) or report['recorded_rows'] != report['scheduled_rows']:
            raise ValueError('Historical report schedule mismatch')
        if report['runtime'] != read(Path(batch['code']).parent / 'runtime.json'):
            raise ValueError('Historical runtime mismatch')
        evidence.append(item)
        for entry in report['rows']:
            verify_binding(entry)
            row = checked_content(entry['path'])
            key = (row['case_id'], row['mode'])
            if key in rows or not row['cleanup_complete']:
                raise ValueError('Duplicate historical row or incomplete cleanup')
            rows[key] = dict(receipt=binding(entry['path']), classification=generation_classification(row))
    expected = {(r['case_id'], m) for r in plan['rows'] for m in plan['modes']}
    if set(rows) != expected or len(plan['rows']) != 90 or len(rows) != 180:
        raise ValueError('Historical 90-case / 180-mode denominator changed')
    if sum(r['historical']['status'] == 'generation_error' for r in plan['rows']) != 79:
        raise ValueError('Historical 79 failures changed')
    scheduled = []
    for item in plan['rows']:
        case = case_from_dict(item['migrated_case'])
        if case.split != 'development' or case.schema_revision != 'v3':
            raise ValueError('Historical audit requires migrated development inputs')
        for mode in plan['modes']:
            row = verify_binding(rows[(item['case_id'], mode)]['receipt'])
            if row['migration'] != item['migration'] or row['historical'] != item['historical']:
                raise ValueError('Historical row case identity differs')
        scheduled.append(dict(case_id=item['case_id'], family=item['family'], parent=item['parent'],
            formerly_failed=item['historical']['status'] == 'generation_error',
            modes={mode: rows[(item['case_id'], mode)] for mode in plan['modes']},
            original_budgets=json.loads(canonical_json(case.problem.budgets))))
    protocol = verify_binding(plan['sources'][2])
    return dict(schema='exact-repair/historical-native-audit/v1', generation_plan=binding(plan_path),
        attempts=evidence, requirement_audit=binding(requirement_path), cases=scheduled,
        expected_cases=90, expected_generation_rows=180, expected_native_rows=270,
        stages=['migrated_inventory', 'generated_cold', 'generated_warm'],
        profile=protocol['preferences']['cost_weights'], scale=protocol['solver']['quantization_scale'],
        memory_mb=plan['per_case_memory_mb'],
        objective='negative declared edit cost; no teacher/learned utility; rich existing inventory',
        budget_policy='Use original saved total/solver/verification/check/solve/retry limits; only cap missing memory at original 8192 MiB. Each staged diagnostic has a separate recorded budget; never sum as matched end-to-end timing.',
        generation_counts=dict(Counter(r['classification'] for r in rows.values())),
        claims='Engineering backend/policy/solver diagnostics only; no model evaluation, semantic benefit, learning efficiency or G0-G2 passage',
        study_complete=False)


def native_probe(problem_record, profile, scale, memory_mb, directory):
    from exact.repair.kernel import materialize, repair
    from exact.repair.owl import qualified_routes, snapshot_from_axioms
    from exact.repair.records import make_objective, read_record

    problem = read_record(problem_record)
    if problem.schema_version != 'exact-repair/records/v3':
        raise ValueError('Native historical probe requires explicit v3 input')
    original_budgets = problem.budgets
    problem = dataclasses.replace(problem, budgets=dataclasses.replace(original_budgets,
        memory_mb=min(original_budgets.memory_mb or memory_mb, memory_mb)))
    # Complex unchanged bundles may still require active-expression checks.
    # Retain those expressions when qualifying routes; they are not edits.
    current = tuple(next(i for i, c in enumerate(obj.candidates)
                         if set(c.axioms) == set(obj.original_axioms))
                    for obj in problem.objects)
    axioms, active = materialize(problem, current)
    routes = qualified_routes(snapshot_from_axioms(axioms),
                              (*problem.policy.required, *problem.policy.prohibited), active)
    objective = make_objective(problem.objects, profile=tuple(sorted(profile.items())), scale=scale)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = repair(problem, objective, preserve_verified_input=False, diagnose=True,
                    ledger_path=directory/'search-ledger.json')
    path = directory/'native-result.json'
    write_artifact(path, dict(input_hash=problem.content_hash, pool_hash=canonical_hash(problem.objects),
        original_budgets=json.loads(canonical_json(original_budgets)),
        effective_budgets=json.loads(canonical_json(problem.budgets)), routes=routes,
        objective=objective.to_dict(), result=result.to_dict()))
    return dict(status='recorded', logical_status=result.logical_status, search_status=result.search_status,
        verification_scope=result.verification_scope, checks=result.checks, solves=result.solves,
        routes=routes, payload=binding(path), selected=result.assignment is not None,
        semantic_benefit=None, semantic_quality_status='not_measured_no_teacher_relabeling')


def _one(output, identity, key, problem_record, plan, metadata):
    from exact.repair.workers import bounded_call
    from exact.repair.records import read_record

    receipt = output/'rows'/(key+'.json')
    row_identity = canonical_hash((identity, key, problem_record))
    previous = checked_checkpoint(receipt, row_identity)
    if previous is not None:
        payload = (previous.get('result') or {}).get('payload')
        if payload:
            verify_binding(payload)
        return previous
    marker = output/'inflight'/(key+'.json')
    if marker.exists():
        raise RuntimeError('Interrupted native probe requires ownership and budget reconciliation: ' + str(marker))
    if problem_record is None:
        return checkpoint(receipt, row_identity, **metadata, status='unavailable_generated_pool',
                          result=None, elapsed_seconds=0, cleanup_complete=True)
    problem = read_record(problem_record)
    write_artifact(marker, dict(identity=row_identity, started_epoch=time.time(),
        seconds=problem.budgets.total_seconds, metadata=metadata,
        continuation='Do not silently repeat; reconcile saved search ledger, ownership and prior costs'))
    started = time.monotonic()
    result = bounded_call(native_probe, problem_record, plan['profile'], plan['scale'], plan['memory_mb'],
        str(output/'payloads'/key), timeout=problem.budgets.total_seconds, memory_mb=plan['memory_mb'])
    row = checkpoint(receipt, row_identity, **metadata, status=result.status, detail=result.detail,
        result=result.value, resources=dict(result.resource_usage), cleanup_complete=result.cleanup_complete,
        elapsed_seconds=time.monotonic()-started)
    if not result.cleanup_complete:
        raise RuntimeError('Native probe cleanup incomplete; preserve in-flight guard')
    marker.unlink()
    return row


def run(plan_path, output, start, stop):
    from exact.repair.study import runtime_manifest

    plan = read(plan_path)
    generation = verify_binding(plan['generation_plan'])
    verify_binding(plan['requirement_audit'])
    if not 0 <= start < stop <= len(plan['cases']):
        raise ValueError('Invalid native audit shard')
    runtime = runtime_manifest()
    identity = canonical_hash((sha(plan_path), sha(__file__), runtime))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        by_id = {r['case_id']: r for r in generation['rows']}
        rows = []
        for item in plan['cases'][start:stop]:
            original = by_id[item['case_id']]['migrated_case']['case']['problem']
            for stage in plan['stages']:
                source = None
                problem = original
                if stage.startswith('generated_'):
                    mode = stage.removeprefix('generated_')
                    source = item['modes'][mode]['receipt']
                    generated = verify_binding(source)
                    problem = (generated.get('result') or {}).get('generated_input')
                key = canonical_hash((item['case_id'], stage))
                metadata = dict(case_id=item['case_id'], family=item['family'], stage=stage,
                    formerly_failed=item['formerly_failed'], generation_receipt=source)
                write_artifact(output/'progress.json', dict(completed_rows=len(rows), **metadata))
                row = _one(output, identity, key, problem, plan, metadata)
                if not row['cleanup_complete']:
                    raise RuntimeError('Prior cleanup incomplete')
                rows.append(dict(**binding(output/'rows'/(key+'.json')), **metadata, status=row['status']))
        return checkpoint(output/f'report-{start:03d}-{stop:03d}.json', identity,
            schema='exact-repair/historical-native-shard/v1', status='complete',
            plan=binding(plan_path), runtime=runtime, rows=rows,
            scheduled_rows=(stop-start)*len(plan['stages']), recorded_rows=len(rows),
            counts=dict(Counter(r['status'] for r in rows)), study_complete=False,
            next_stage='xr21-expanded-historical-audit-001',
            gates={g:'not_established_requires_final_receipt_review' for g in ('G0','G1','G2')})


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
