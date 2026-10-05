"""Authenticate fixed-inventory diagnostics without replay or new semantic claims."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, read_record
from tools.repair import common_inventory as science
from tools.repair.common_inventory_recovery import reconciled_unknown, slice_bounds
from tools.repair.fresh_evaluation_audit import references, validate_attempt, lineage_row, row_summary
from tools.repair.overlap_audit import Evidence, binding, require
from tools.repair.scaling_audit import checkpoint


def scientific_identity(schedule_ref, batch, runtime):
    return canonical_hash((schedule_ref['sha256'], *[
        batch['frozen_files'][str(Path(batch['code']) / ('tools/repair/' + name))]
        for name in ('common_inventory.py', 'fresh_evaluation.py')], runtime))


def query_scope(evidence, schedule):
    scope = []
    for item in schedule['cases']:
        case = evidence.read(item['evaluator'])['case']
        require(case['case_id'] == item['case_id'] and case['split'] == 'test'
                and case['structural_parent'] == item['structural_parent'],
                'Fresh evaluator identity or split differs')
        require(case['problem'] == evidence.read(item['observable']),
                'Observable and evaluator inputs differ')
        scope.append(dict(case_id=item['case_id'], family=item['family'],
            parent=item['structural_parent'], probe_count=len(case['probes']),
            desired=sum(p['desired'] for p in case['probes']),
            unwanted=sum(not p['desired'] for p in case['probes']),
            nonvacuity_declared=sum(p.get('nonvacuity') is not None for p in case['probes']),
            intended_parent_qualification='not_established_by_receipt_audit'))
    return scope


def unknown_row(evidence, ref, saved, recovery, identity):
    plan = evidence.read(recovery['plan'])
    proof = evidence.read(recovery['evidence'])
    references(evidence, proof)
    require(plan['evidence'] == recovery['evidence'], 'Recovery proof differs')
    original = checkpoint(evidence, saved['original_receipt'])
    guard = checkpoint(evidence, proof['guard'])
    evidence.checkpoint(saved['original_receipt'], canonical_hash((identity, saved['row'])))
    evidence.checkpoint(proof['guard'], canonical_hash((identity, saved['row'])))
    expected = reconciled_unknown(original, guard, saved['original_receipt'],
        recovery['evidence'], saved['ownership'], plan['original_step'])
    require({k: v for k, v in saved.items() if k not in ('identity', 'content_hash')} == expected,
            'Interrupted diagnostic row was changed or promoted')
    evidence.checkpoint(ref, canonical_hash((recovery['plan'], identity)))


def audit_rows(evidence, schedule, entries):
    expected = {r['id']: r for r in schedule['rows']}
    seen, rows = set(), []
    for item, source in entries:
        report = checkpoint(evidence, item['report'])
        require(report['status'] == 'complete' and evidence.read(report['schedule']) == schedule,
                'Incomplete or mismatched diagnostic report')
        start, stop = item['row_slice']
        require(report['slice'] == [start, stop] and
                report['scheduled'] == report['recorded'] == len(report['rows']) == stop-start,
                'Diagnostic slice denominator differs')
        wanted = schedule['rows'][start:stop]
        require([r['row_id'] for r in report['rows']] == [r['id'] for r in wanted],
                'Diagnostic order or slice differs')
        batch = evidence.read(item['batch'])
        runtime = source['runtime']
        recovery = report.get('cleanup_recovery') or report.get('manifest_recovery')
        manifest_recovery = bool(report.get('manifest_recovery'))
        plan = None
        if recovery:
            plan = evidence.read(recovery['plan'])
            references(evidence, plan)
            require(slice_bounds(plan)[:2] == (start, stop) and plan['schedule'] == report['schedule'],
                    'Recovery slice or schedule differs')
            batch = evidence.read(plan['frozen_batch'])
            runtime = evidence.read(binding(Path(plan['frozen_batch']['path']).with_name('runtime.json')))
            require(runtime == plan['runtime'], 'Recovery changed scientific runtime')
        identity = scientific_identity(report['schedule'], batch, runtime)
        require(report['runtime'] == runtime and report['identity'] == (
            canonical_hash((recovery['plan'], identity)) if recovery else identity),
            'Diagnostic source or runtime identity differs')
        if plan:
            require(plan['scientific_identity'] == identity, 'Recovery scientific identity differs')
            proof = evidence.read(recovery['evidence'])
            require(proof['completion'] == binding(Path(item['run']['completion_path']).parent.parent /
                    item['run']['recovery_of'].rsplit('-', 1)[1] / 'completion.json'),
                    'Recovery does not bind its original attempt')
            prefix = {p['index']: p['receipt'] for p in proof['rows']}
            require(set(prefix) == set(range(start, plan['failed_index'] + (0 if manifest_recovery else 1))),
                    'Recovery lost original spent prefix')
        counts = Counter()
        for index, ref in enumerate(report['rows'], start):
            saved = lineage_row(evidence, ref, expected[ref['row_id']])
            row = saved['row']
            require(row['id'] not in seen and saved['cleanup_complete'],
                    'Duplicate diagnostic row or unresolved cleanup')
            seen.add(row['id'])
            if saved['status'] == 'unknown_after_cleanup_reconciliation':
                require(plan is not None and index == plan['failed_index'],
                        'Unknown diagnostic row lacks its recovery')
                require(saved['original_receipt'] == prefix[index], 'Unknown original receipt differs')
                unknown_row(evidence, ref, saved, recovery, identity)
            elif saved['status'] == 'recovered_result_after_manifest_failure':
                require(manifest_recovery and index == plan['failed_index'], 'Retained result lacks manifest recovery')
                manifest_unknown(evidence, ref, saved, recovery, identity)
            else:
                row_identity = canonical_hash((recovery['plan'], identity)) if manifest_recovery and index > plan['failed_index'] else identity
                evidence.checkpoint(ref, canonical_hash((row_identity, row)))
                require(not (Path(ref['path']).parent.parent / 'inflight' / (row['id']+'.json')).exists(),
                        'Finished diagnostic row retains an ownership guard')
                if plan and index < plan['failed_index']:
                    require(all(ref[k] == prefix[index][k] for k in ('path', 'sha256')),
                            'Recovery replaced an already spent row')
                if plan and index > plan['failed_index']:
                    require(Path(ref['path']).parent.parent == Path(item['report']['path']).parent / 'continuation',
                            'Recovery suffix is not the separately recorded continuation')
            summary = row_summary(evidence, saved, schedule['cases'][row['case_index']], schedule)
            require(ref['status'] == summary['scientific_status'], 'Diagnostic reference status differs')
            counts[summary['scientific_status']] += 1
            summary.update(receipt=ref, source_run_id=item['run']['id'],
                           value_diagnostic=value_diagnostic(evidence, saved, schedule),
                           outer_telemetry_available=saved.get('outer_telemetry_available', True))
            rows.append(summary)
        require(dict(counts) == report['outcomes'], 'Diagnostic outcome counts differ')
    require(seen == set(expected), 'Missing diagnostic denominator rows')
    return rows


def manifest_unknown(evidence, ref, saved, recovery, identity):
    """Reconstruct the retained UNKNOWN exactly, including absent outer telemetry."""
    plan = evidence.read(recovery['plan'])
    proof = evidence.read(recovery['evidence'])
    references(evidence, proof)
    require(plan['evidence'] == recovery['evidence'], 'Manifest recovery evidence differs')
    result = evidence.read(proof['result'])
    require(result['row_id'] == saved['row']['id'] and result['status'] == 'verification_timeout'
            and result['logical_status'] == 'UNKNOWN' and result['semantic_benefit'] is None
            and result['semantic_status'] == 'not_evaluated', 'Retained timeout was promoted')
    owner = saved['ownership']
    require(owner['step_id'] == plan['original_step'] and owner['surviving_processes'] == 0
            and owner['cgroup_absent'] and owner['signals_sent'] == 0, 'Manifest owner unresolved')
    guard = evidence.checkpoint(proof['guard'], canonical_hash((identity, saved['row'])))
    require(guard['row'] == saved['row'], 'Original manifest guard differs')
    expected = dict(row=saved['row'], status='recovered_result_after_manifest_failure',
        detail='Original published UNKNOWN timeout retained without replay; outer call telemetry was not persisted.',
        result=proof['result'], cleanup_complete=True, resources={},
        elapsed_seconds=result['elapsed_seconds'], payloads=proof['payloads'],
        unpublished_payloads=[proof['temporary']], original_guard=proof['guard'],
        recovery_evidence=plan['evidence'], ownership=owner, additional_scientific_seconds=0,
        outer_telemetry_available=False, prior_costs_reset=False)
    require({k:v for k,v in saved.items() if k not in ('identity','content_hash')} == expected,
            'Retained manifest result, cost or telemetry changed')
    evidence.checkpoint(ref, canonical_hash((recovery['plan'], identity)))


def value_diagnostic(evidence, saved, schedule):
    """Check saved coefficients against the selected assignment, never an optimum label."""
    row = saved['row']
    arm = next(a for a in schedule['arms'] if a['id'] == row['arm_id'])
    item = schedule['cases'][row['case_index']]
    result = evidence.read(saved['result']) if saved.get('result') else {}
    refs = {Path(p['path']).name:p for p in saved.get('payloads', [])}
    diagnostic = dict(status='unavailable', selected_objective_utility=None,
        selected_utility=None, selected_utility_error=None, common_inventory_regret=None,
        regret_status='unavailable_no_complete_external_teacher', semantic_claim_qualified=False,
        native_label_call_telemetry='not_recorded_by_original_diagnostic_worker')
    if not result:
        return diagnostic
    require(result['study_kind'] == 'common_inventory_diagnostic'
            and result['common_inventory_regret'] is None
            and result['regret_status'] == diagnostic['regret_status'], 'Diagnostic promoted to semantic regret')
    require(not result.get('generation_statuses', []), 'Fixed inventory acquired generation')
    if 'pool.json' in refs:
        pool = evidence.read(refs['pool.json'])
        scoring = evidence.read(refs['inventory-scoring.json'])
        original = read_record(evidence.read(item['observable']))
        scored = read_record(pool['input'])
        require(not pool['proposal_reports'] and not any(scoring[k] for k in
            ('generation','fitting','evaluator_opened')), 'Inventory scoring boundary differs')
        require(scoring['original_inventory_hash'] == canonical_hash(original.objects)
                and scoring['scored_inventory_hash'] == canonical_hash(scored.objects)
                and result['inventory_hash'] == canonical_hash(scored.objects), 'Inventory identities differ')
        if arm['id'] != 'deletion':
            require(original.objects == scored.objects, 'Common inventory candidates changed')
        else:
            from tools.repair.fresh_evaluation import control_objective
            expected, _ = control_objective(original, 'deletion', evidence.read(arm['protocol']))
            require(expected.objects == scored.objects, 'Deletion language ablation differs')
        if arm['kind'] == 'learned':
            require(pool['model_hash'] == arm['model_hash'] and
                    scoring['coverage']['model_hash'] == arm['model_hash'], 'Frozen model differs')
        if result.get('selected_assignment') is not None:
            repair = read_record(evidence.read(refs['repair.json']))
            assignment = tuple(result['selected_assignment'])
            require(repair.logical_status == 'VERIFIED_FEASIBLE' and repair.verification is not None
                    and repair.verification.authorizes and repair.assignment == assignment,
                    'Selected value lacks native authorizing assignment')
            objective = read_record(pool['objective'])
            require(objective.pool_hash == canonical_hash(scored.objects)
                    and repair.objective_hash == objective.content_hash
                    and repair.verification.assignment_hash == canonical_hash(assignment)
                    and repair.verification.policy_hash == scored.policy.content_hash,
                    'Objective, pool or native verification identity differs')
            predicted = objective.score(assignment) / objective.scale
            require(result['selected_objective_utility'] == predicted, 'Selected objective value differs')
            diagnostic.update(status='selected_without_usable_label', selected_objective_utility=predicted)
    if result.get('semantic_status') == 'known':
        require(diagnostic['status'] == 'selected_without_usable_label', 'Known diagnostic lacks objective evidence')
        selected = result['selected_utility']
        expected_error = diagnostic['selected_objective_utility'] - selected if arm['kind'] == 'learned' else None
        require(result['selected_utility_error'] == expected_error, 'Selected-only value error differs')
        diagnostic.update(status='selected_known', selected_utility=selected, selected_utility_error=expected_error)
    else:
        require(result.get('selected_utility_error') is None, 'Unknown label acquired a value error')
    return diagnostic


def summaries(rows):
    errors = [r['value_diagnostic']['selected_utility_error'] for r in rows
              if r['value_diagnostic']['selected_utility_error'] is not None]
    return dict(scheduled_rows=len(rows),
        scientific_statuses=dict(Counter(r['scientific_status'] for r in rows)),
        semantic_statuses=dict(Counter(r['semantic_status'] for r in rows)),
        value_statuses=dict(Counter(r['value_diagnostic']['status'] for r in rows)),
        selected_error_count=len(errors), selected_error_mean=sum(errors)/len(errors) if errors else None,
        selected_absolute_error_mean=sum(abs(v) for v in errors)/len(errors) if errors else None,
        selected_error_scope='Conditional on usable selected labels; not population calibration or semantic regret')


def audit(manifest_path, output):
    from tools.repair.report_campaign import totals

    evidence = Evidence()
    manifest = evidence.read(binding(manifest_path))
    require(manifest['schema'] == 'exact-repair/common-inventory-audit-manifest/v1'
            and manifest['supervision_admitted'] is False and manifest['scientific_rows_replayed'] == 0
            and manifest['expected_rows'] == 576, 'Inventory audit boundary differs')
    schedule = evidence.read(manifest['schedule'])
    science.validate_schedule(schedule)
    references(evidence, schedule)
    scope = query_scope(evidence, schedule)
    ledger = evidence.read(manifest['ledger_snapshot'])
    require(ledger['limit_worker_seconds'] is None, 'Campaign time amendment missing')
    snapshot = evidence.read(manifest['registry_snapshot'])
    all_runs = {r['id']:r for r in snapshot['runs']}
    active = set(manifest['active_run_ids'])
    require(len(active) == len(manifest['active_run_ids']) == 37, 'Active diagnostic jobs differ')
    sources, entries, validations = [], [], []
    for item in manifest['attempts']:
        run = item['run']
        require(all_runs[run['id']] == run, 'Attempt differs from registry snapshot')
        source = validate_attempt(evidence, item, ledger, run['id'] in active)
        references(evidence, item['commands'])
        outputs = evidence.read(item['outputs']) if item.get('outputs') else {}
        for relative, digest in outputs.items():
            evidence.verify(dict(path=str(Path(source['completion']['work'])/relative), sha256=digest))
        if run['id'] in active:
            if 'row_slice' in item:
                relative = run['result_relative']
                require(item['report']['path'] == str(Path(source['completion']['work'])/relative)
                        and outputs[relative] == item['report']['sha256'], 'Report lacks worker output binding')
                entries.append((item, source))
            else:
                require(outputs['validation.xml'] == item['validation']['sha256'], 'Validation lacks output binding')
                root = ET.parse(evidence.verify(item['validation'])).getroot()
                counts = {k:sum(int(s.get(k,0)) for s in root.iter('testsuite'))
                          for k in ('tests','failures','errors','skipped')}
                require(counts == dict(tests=26,failures=0,errors=0,skipped=0), 'Native validation differs')
                validations.append(dict(receipt=item['validation'], counts=counts))
        sources.append(source)
    by_id = {s['run']['id']:s for s in sources}
    require(len(by_id) == len(sources) == 39 and active <= set(by_id)
            and len(entries) == 36 and len(validations) == 1, 'Attempt denominator differs')
    for source in sources:
        run = source['run']
        if run.get('superseded_by'):
            require(by_id[run['superseded_by']]['run'].get('recovery_of') == run['id'], 'Recovery lineage lost')
        else:
            require(run['id'] in active, 'Attempt omitted from active denominator')
    rows = audit_rows(evidence, schedule, entries)
    costs = {str(Path(s['run']['completion_path']).parent):s['charge'] for s in sources}
    result = dict(schema='exact-repair/common-inventory-audit/v1', status='complete',
        manifest=binding(manifest_path), recorded_rows=len(rows),
        learned_rows=384, control_rows=192, cases=64, parent_groups=32, arms=9,
        rows=rows, attempts=sources, attempt_costs=costs, validation=validations,
        worker_seconds=sum(c['elapsed_seconds'] for c in costs.values()),
        cumulative_costs=totals(ledger['attempts']), ledger_snapshot=manifest['ledger_snapshot'],
        **summaries(rows),
        by_arm={a['id']:summaries([r for r in rows if r['row']['arm_id']==a['id']]) for a in schedule['arms']},
        by_family={f:summaries([r for r in rows if r['family']==f]) for f in sorted({r['family'] for r in rows})},
        by_family_exposure={f:summaries([r for r in rows if r['family_exposure']==f]) for f in sorted({r['family_exposure'] for r in rows})},
        semantic_query_scope=scope,
        semantic_query_totals={k:sum(s[k] for s in scope) for k in
            ('probe_count','desired','unwanted','nonvacuity_declared')},
        scientific_errors=[dict(row_id=r['row']['id'], errors=r['software_errors']) for r in rows if r['software_errors']],
        study_complete=False, campaign_complete=False, gates_passed=False,
        supervision_admitted=False, scientific_rows_replayed=0, native_calls=0,
        strongest_symbolic_comparison_qualified=False, common_inventory_regret=None,
        regret_status='unavailable_no_complete_external_teacher',
        required_followups=['intended_parent_and_query_scope_qualification',
            'required_semantic_control_review', 'failure_inclusive_parent_group_final_report'],
        limitations=[schedule['control_limitation'],
            'Selected-only conditional value errors are not whole-inventory calibration or primary generated-pool quality.',
            'Receipt integrity does not establish native intended-parent truth, G0-G2 or learning efficiency.',
            'Original workers did not persist per-obligation native label call traces; saved assignment/label/query evidence is checked.',
            'Typed nonvacuity is derived by the query implementation when undeclared; absence of declarations is not absence of checks.',
            'Unknown, unavailable and recovered rows remain in denominators; no spent row was replayed.',
            'All 39 validation/original/replacement attempt costs are retained; maintenance and this audit are charged separately.'],
        followup_stage=manifest['followup_stage'], api_spend_usd=0, verified_file_count=len(evidence.files))
    write_artifact(Path(output)/'report.json', result)
    write_artifact(Path(output)/'verified-files.json', evidence.files)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.manifest, args.output)
