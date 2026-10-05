"""Audit development decoding receipts and eligibility gaps without new science.

Disposable models and selected diagnostic labels never become fitting inputs.
The complete denominator and original recovery costs remain visible.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair import development_decode as decode
from tools.repair.development_decode_recovery import reconciled_unknown, slice_bounds
from tools.repair.fresh_evaluation_audit import (
    references, validate_attempt, lineage_row, row_summary,
)
from tools.repair.overlap_audit import Evidence, binding, require
from tools.repair.scaling_audit import checkpoint


def scientific_identity(schedule_ref, batch, runtime):
    return canonical_hash((schedule_ref['sha256'], *[
        batch['frozen_files'][str(Path(batch['code']) / ('tools/repair/' + name))]
        for name in ('development_decode.py', 'fresh_evaluation.py')], runtime))


def query_scope(evidence, schedule):
    scope = []
    for item in schedule['cases']:
        case = evidence.read(item['evaluator'])['case']
        require(case['case_id'] == item['case_id'] and case['split'] == 'development'
                and case['structural_parent'] == item['structural_parent'],
                'Development evaluator identity or split differs')
        require(case['problem'] == evidence.read(item['observable']),
                'Observable and evaluator inputs differ')
        scope.append(dict(case_id=item['case_id'], family=item['family'],
            parent=item['structural_parent'], probe_count=len(case['probes']),
            desired=sum(p['desired'] for p in case['probes']),
            unwanted=sum(not p['desired'] for p in case['probes']),
            nonvacuity_declared=sum(p.get('nonvacuity') is not None for p in case['probes']),
            intended_parent_qualification='not_established_by_receipt_audit'))
    return scope


def native_evidence(evidence, saved, item):
    """Retain call errors and scope, including missing native evidence explicitly."""
    from exact.repair.records import read_record
    from tools.repair.prepare import case_from_dict

    refs = {Path(p['path']).name: p for p in saved.get('payloads', [])
            if Path(p['path']).parent.name == 'native-label'}
    call = evidence.read(refs['call.json']) if 'call.json' in refs else None
    checks = [dict(receipt=ref, result=evidence.read(ref)) for name, ref in sorted(refs.items())
              if name.startswith('check-') and not name.endswith('.started.json')]
    starts = [dict(receipt=ref, result=evidence.read(ref)) for name, ref in sorted(refs.items())
              if name.endswith('.started.json')]
    result = evidence.read(saved['result']) if saved.get('result') else {}
    if call:
        require(call['case_id'] == item['case_id'] and call['split'] == 'development'
                and call['parent'] == item['structural_parent'], 'Native label scope differs')
        case = case_from_dict(evidence.read(item['evaluator']))
        require(call['query_hash'] == canonical_hash(case.probes), 'Native query identity differs')
        pool_ref = next((p for p in saved['payloads'] if Path(p['path']).name == 'pool.json'), None)
        require(pool_ref is not None, 'Native label lacks generated pool')
        pool = read_record(evidence.read(pool_ref)['input'])
        require(call['input_hash'] == pool.content_hash and call['policy_hash'] == pool.policy.content_hash,
                'Native label input or policy differs from generated pool')
    if result.get('semantic_status') == 'known':
        require(call is not None and call['status'] == 'complete' and call['cleanup_complete'],
                'Known score lacks completed native label call')
        label_ref = next(p for p in saved['payloads'] if Path(p['path']).name == 'selected-label.json')
        label = evidence.read(label_ref)
        require(call['assignment'] == label['assignment'] and len(checks) >= 1,
                'Known native label assignment or check evidence differs')
    return dict(call=call, call_receipt=refs.get('call.json'), checks=checks, starts=starts,
                status=call['status'] if call else 'unavailable_no_label_call',
                supervision_admitted=False)


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
        recovery = report.get('cleanup_recovery')
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
            require(set(prefix) == set(range(start, plan['failed_index']+1)),
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
            else:
                evidence.checkpoint(ref, canonical_hash((identity, row)))
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
                           native_label=native_evidence(evidence, saved, schedule['cases'][row['case_index']]))
            rows.append(summary)
        require(dict(counts) == report['outcomes'], 'Diagnostic outcome counts differ')
    require(seen == set(expected), 'Missing diagnostic denominator rows')
    return rows


def audit(manifest_path, output):
    from tools.repair.report_campaign import totals
    from tools.repair.training_audit import attempt_report

    evidence = Evidence()
    manifest = evidence.read(binding(manifest_path))
    require(manifest['schema'] == 'exact-repair/development-decode-audit-manifest/v1'
            and manifest['heldout_cases_opened'] is False and manifest['supervision_admitted'] is False
            and manifest['scientific_rows_replayed'] == 0 and manifest['expected_rows'] == 288,
            'Diagnostic audit boundary differs')
    schedule = evidence.read(manifest['schedule'])
    decode.validate_schedule(schedule)
    references(evidence, schedule)
    scope = query_scope(evidence, schedule)
    ledger = evidence.read(manifest['ledger_snapshot'])
    require(ledger['limit_worker_seconds'] is None, 'Campaign time amendment missing')
    snapshot = evidence.read(manifest['registry_snapshot'])
    all_runs = {r['id']: r for r in snapshot['runs']}
    active = set(manifest['active_run_ids'])
    require(len(active) == len(manifest['active_run_ids']) == 19, 'Active diagnostic jobs differ')
    sources, entries, validation = [], [], []
    for item in manifest['attempts']:
        run = item['run']
        require(all_runs[run['id']] == run, 'Attempt differs from registry snapshot')
        source = validate_attempt(evidence, item, ledger, run['id'] in active)
        references(evidence, item['commands'])
        if item.get('outputs'):
            outputs = evidence.read(item['outputs'])
            for relative, digest in outputs.items():
                evidence.verify(dict(path=str(Path(source['completion']['work'])/relative), sha256=digest))
        if run['id'] in active:
            relative = run['result_relative']
            require(item['report']['path'] == str(Path(source['completion']['work'])/relative)
                    and outputs[relative] == item['report']['sha256'], 'Report lacks worker output binding')
            if 'row_slice' in item:
                entries.append((item, source))
            else:
                value = evidence.read(item['report'])
                require(value['scheduled_rows'] == 288 and value['scientific_rows_executed'] == 0
                        and value['schedule'] == manifest['schedule']
                        and len(value['initialized_models']) == 6, 'Model validation receipt differs')
                validation.append(item['report'])
        sources.append(source)
    by_id = {s['run']['id']: s for s in sources}
    require(len(by_id) == len(sources) and active <= set(by_id)
            and len(entries) == 18 and len(validation) == 1, 'Attempt denominator differs')
    for source in sources:
        run = source['run']
        if run.get('superseded_by'):
            require(by_id[run['superseded_by']]['run'].get('recovery_of') == run['id'],
                    'Attempt supersession ancestry lost')
        else:
            require(run['id'] in active, 'Completed attempt omitted from active denominator')
    rows = audit_rows(evidence, schedule, entries)
    review = evidence.read(manifest['prerequisite_review'])
    # Reauthenticate existing prerequisites; no relabeling or test payload access.
    label, _, _ = attempt_report(review['label_audit'])
    qualification, _, _ = attempt_report(review['qualification'])
    requirements = evidence.read(qualification['requirement_review'])
    require(label['scheduled_rows'] == 160 and label['supervision_admitted'] is False,
            'Original supervision boundary differs')
    costs = {str(Path(s['run']['completion_path']).parent): s['charge'] for s in sources}
    result = dict(schema='exact-repair/development-decode-audit/v1', status='complete',
        manifest=binding(manifest_path), scheduled_rows=288, recorded_rows=len(rows),
        cases=32, parent_groups=16, arms=9, rows=rows, attempts=sources, attempt_costs=costs,
        worker_seconds=sum(c['elapsed_seconds'] for c in costs.values()),
        cumulative_costs=totals(ledger['attempts']), ledger_snapshot=manifest['ledger_snapshot'],
        scientific_statuses=dict(Counter(r['scientific_status'] for r in rows)),
        semantic_statuses=dict(Counter(r['semantic_status'] for r in rows)),
        native_call_statuses=dict(Counter(r['native_label']['status'] for r in rows)),
        by_arm={arm['id']: dict(scheduled_rows=32,
            scientific_statuses=dict(Counter(r['scientific_status'] for r in rows if r['row']['arm_id']==arm['id'])),
            semantic_statuses=dict(Counter(r['semantic_status'] for r in rows if r['row']['arm_id']==arm['id'])))
            for arm in schedule['arms']},
        by_family={family: dict(scheduled_rows=sum(r['family']==family for r in rows),
            scientific_statuses=dict(Counter(r['scientific_status'] for r in rows if r['family']==family)))
            for family in sorted({r['family'] for r in rows})},
        semantic_query_scope=scope,
        semantic_query_totals={k:sum(s[k] for s in scope) for k in
            ('probe_count','desired','unwanted','nonvacuity_declared')},
        original_label_summaries=label['summaries'], requirement_review=qualification['requirement_review'],
        fixture_requirement_counts={k:requirements[k] for k in ('required','source_compatible')},
        scientific_errors=[dict(row_id=r['row']['id'], errors=r['software_errors']) for r in rows if r['software_errors']],
        study_complete=False, campaign_complete=False, gates_passed=False, fitting_eligible=False,
        supervision_admitted=False, heldout_cases_opened=False, scientific_rows_replayed=0,
        checkpoint_selection='unavailable_untrained_diagnostic_models',
        required_followups=['native_backend_intended_parent_and_query_eligibility',
            'actual_supervised_generated_acquisition', 'optimizer_minibatch_capacity',
            'development_checkpoint_selection', '18_strict_full_schema_fitting_protocols_and_resumable_jobs',
            '64_heldout_cases_with_matched_semantic_controls', 'failure_inclusive_training_final_report'],
        limitations=['Selected untrained diagnostic scores are not admitted supervision or checkpoint selection.',
            'Receipt integrity and mapped fixture passage do not establish G0-G2 or native intended-parent truth.',
            'Unknowns, partial generation, unsupported bundles and all failed attempts retain their denominators.',
            'The retained-axiom symbolic heuristic is not the required strongest semantic control.',
            'Original fixed-inventory usable labels cover only overlap; no spent case was relabeled.',
            'Worker costs include validation and both failed originals; maintenance and audit costs are separate.'],
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
