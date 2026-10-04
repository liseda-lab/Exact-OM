"""Audit frozen fresh-evaluation attempts without replay or scientific claims.

Primary generated-pool receipts, their recovery ancestry and fixed denominators
are independent of the still-required inventory diagnostics and final report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair.overlap_audit import Evidence, binding, require
from tools.repair.scaling_audit import checkpoint


def references(evidence, value):
    """Verify explicit file bindings, including retained partial search ledgers."""
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and 'sha256' in value:
            evidence.verify(value)
        else:
            for child in value.values():
                references(evidence, child)
    elif isinstance(value, list):
        for child in value:
            references(evidence, child)


def validate_attempt(evidence, item, ledger, active):
    run = item['run']
    complete, step = evidence.read(item['completion']), evidence.read(item['step'])
    for receipt in (complete, step):
        require(receipt['step_id'] == run['step_id'] and
                receipt['dispatch_nonce'] == run['dispatch_nonce'], 'Attempt ownership differs')
    # Schema integration retained the old registry logical_id for supersession,
    # while the frozen worker and ledger use the explicit schema job ID.
    require(complete['job_id'] == run['id'].rsplit('-', 1)[0], 'Frozen worker ID differs')
    if complete['job_id'] != run['logical_id']:
        require(run.get('original_run_id', '').rsplit('-', 1)[0] == run['logical_id'] or
                run.get('recovery_of', '').rsplit('-', 1)[0] == complete['job_id'],
                'Unexplained registry logical ID alias')
    require(int(evidence.verify(item['exit']).read_text()) == complete['exit_code'],
            'Exit receipt differs')
    require(not active or (complete['status'] == 'complete' and complete['exit_code'] == 0),
            'Active shard is not complete')
    batch = evidence.read(item['batch'])
    require(complete['batch'] == item['batch']['path'] and
            any(j['id'] == complete['job_id'] for j in batch['jobs']), 'Frozen job absent from batch')
    require(evidence.verify(item['batch_hash']).read_text().strip() == item['batch']['sha256'],
            'Batch digest differs')
    for path, digest in batch['frozen_files'].items():
        if path not in evidence.files:
            evidence.verify(dict(path=path, sha256=digest))
        else:
            require(evidence.files[path] == digest, 'Conflicting frozen source binding')
    runtime = evidence.read(item['runtime'])
    charge = ledger['attempts'][str(Path(run['completion_path']).parent)]
    require(charge['status'] == 'settled' and charge['logical_id'] == complete['job_id'],
            'Attempt cost missing or belongs to another worker')
    require(charge['elapsed_seconds'] >= complete['elapsed_seconds'] and
            (charge.get('cpu_seconds') or 0) >= complete['cpu_seconds'], 'Attempt cost lost')
    references(evidence, charge)
    return dict(run=run, completion=complete, charge=charge, runtime=runtime,
                source_commit=batch['commit'], batch=item['batch'])


def validate_schedule(evidence, manifest):
    schedule = evidence.read(manifest['schedule'])
    require(schedule['schema'] == 'exact-repair/fresh-evaluation/v1', 'Wrong evaluation schema')
    require(len(schedule['cases']) == 64 and len(schedule['arms']) == 9 and
            len(schedule['rows']) == 576, 'Fresh evaluation denominator differs')
    require(sum(a['kind'] == 'learned' for a in schedule['arms']) == 6,
            'Frozen model denominator differs')
    require({a['id'] for a in schedule['arms'] if a['kind'] == 'control'} ==
            {'symbolic_rich_action', 'uniform', 'deletion'}, 'Control denominator differs')
    require(schedule['profile_timeouts_retained'] == 4 and
            schedule['test_feedback_for_selection'] is False and schedule['warm_start'] is False,
            'Historical limitations or frozen selection changed')
    references(evidence, schedule)
    expected = []
    for arm in schedule['arms']:
        protocol = evidence.read(arm['protocol'])
        resources = protocol['resources']
        require(all(resources[k] == v for k, v in dict(case_wall_seconds=300,
            case_cpu_seconds=600, case_rss_mb=8192, generation_seconds=60).items()),
            'Scientific comparison budgets changed')
        for index, case in enumerate(schedule['cases']):
            row = dict(arm_id=arm['id'], case_index=index, case_id=case['case_id'],
                       seconds=300, cpu_seconds=600, memory_mb=8192)
            row['id'] = canonical_hash(row)
            expected.append(row)
    require(schedule['rows'] == expected, 'Frozen schedule order or identities differ')
    groups = {c['group_id'] for c in schedule['cases']}
    require(len(groups) == 32, 'Parent denominator differs')
    for group in groups:
        require(Counter(c['control'] for c in schedule['cases'] if c['group_id'] == group) ==
                {'coherent': 1, 'corrupted': 1}, 'Parent variants differ')
    audit = evidence.read(manifest['corpus_audit'])
    references(evidence, audit)
    require(audit['fresh_evaluation_cases'] == 64 and audit['training_cases'] == 224 and
            audit['exposed_parent_exclusion'] and audit['structural_fingerprints_recomputed'] and
            audit['no_label_caches'], 'Original corpus admission differs')
    require(audit['model_hashes'] == {a['id']: a['model']['sha256']
            for a in schedule['arms'] if a['kind'] == 'learned'}, 'Frozen models changed')
    # Receipt validation does not repeat a native target/clean-parent experiment.
    scope = []
    for item in schedule['cases']:
        if item['status'] != 'materialized':
            scope.append(dict(case_id=item['case_id'], status=item['status']))
            continue
        case = evidence.read(item['evaluator'])['case']
        require(case['case_id'] == item['case_id'] and case['split'] == 'test' and
                case['structural_parent'] == item['structural_parent'],
                'Evaluator identity or split differs')
        require(case['problem'] == evidence.read(item['observable']), 'Observable/evaluator input differs')
        scope.append(dict(case_id=item['case_id'], group_id=item['group_id'],
            family=item['family'], family_exposure=item['family_exposure'],
            probe_count=len(case['probes']), desired=sum(p['desired'] for p in case['probes']),
            unwanted=sum(not p['desired'] for p in case['probes']),
            nonvacuity_declared=sum(p.get('nonvacuity') is not None for p in case['probes']),
            intended_theory_axioms=len(case['intended_theory']),
            native_intended_parent_query_qualification='not_established_by_this_receipt_audit'))
    return schedule, scope


def schema_identity(evidence, plan_ref, source, runtime):
    plan = evidence.read(plan_ref)
    require(evidence.read(plan['schedule'])['schema'] == 'exact-repair/fresh-evaluation/v1',
            'Recovery plan references another study')
    return canonical_hash((plan_ref['sha256'], source['frozen_files'][
        str(Path(source['code']) / 'tools/repair/schema_recovery.py')], source['frozen_files'][
        str(Path(source['code']) / 'tools/repair/fresh_evaluation.py')], runtime))


def lineage_row(evidence, ref, row, seen=None):
    """Check original rows recursively; never overwrite their error classifications."""
    seen = set() if seen is None else seen
    require(ref['path'] not in seen, 'Cyclic row recovery ancestry')
    seen.add(ref['path'])
    saved = checkpoint(evidence, ref)
    require(saved['row'] == row, 'Recovery ancestry changes scheduled row')
    references(evidence, saved)
    for key in ('reuse_of', 'recovery_of', 'original_receipt'):
        if saved.get(key):
            lineage_row(evidence, saved[key], row, seen.copy())
    return saved


def validate_budget(evidence, saved, arm):
    if not saved.get('remaining_budget'):
        return
    from tools.repair.schema_recovery import remaining_budget, validate_remaining_budget

    row = saved['row']
    resources = evidence.read(arm['protocol'])['resources']
    budget = saved['remaining_budget']
    validate_remaining_budget(budget, row, resources)
    if saved.get('recovery_of'):
        previous = checkpoint(evidence, saved['recovery_of'])
        payload = evidence.read(previous['result'])
        require(budget == remaining_budget(previous, payload, row, resources),
                'Recovery restored spent scientific budget')
    else:
        require(budget == dict(wall_seconds=row['seconds'], generation_seconds=resources['generation_seconds'],
            cpu_seconds=row['cpu_seconds'], prior_wall_seconds=0, prior_generation_seconds=0,
            prior_cpu_seconds=0), 'Fresh row starts with an unexplained budget')


def validate_unknown(evidence, saved, recovery, ref):
    from tools.repair.schema_cleanup_recovery import reconciled_unknown

    require(recovery is not None, 'Unknown cleanup row lacks recovery plan')
    plan = evidence.read(recovery['plan'])
    proof = evidence.read(recovery['evidence'])
    references(evidence, proof)
    original = checkpoint(evidence, saved['original_receipt'])
    guard = checkpoint(evidence, proof['guard'])
    expected = reconciled_unknown(original, guard, saved['original_receipt'],
        recovery['evidence'], saved['ownership'], plan['original_step'])
    require({k: v for k, v in saved.items() if k not in ('identity', 'content_hash')} == expected,
            'Reconciled unknown was changed or promoted')
    identity = canonical_hash((recovery['plan'], plan['schema_identity']))
    evidence.checkpoint(ref, identity)
    require(saved['result'] is None and saved['additional_elapsed_seconds'] == 0,
            'Interrupted row was replayed or promoted')


def row_summary(evidence, saved, item, schedule):
    result = evidence.read(saved['result']) if saved.get('result') else {}
    if result:
        require(result['row_id'] == saved['row']['id'] and
                result['schedule_hash'] == canonical_hash(schedule) and
                result['case_id'] == item['case_id'], 'Result scope differs')
    from exact.experiments.science_health import software_failure

    errors = [dict(status=status, detail=detail) for status, detail in (
        (saved['status'], saved.get('detail', '')),
        (result.get('status'), result.get('detail', '')),
        (result.get('semantic_status'), result.get('semantic_detail', '')))
        if software_failure(status, detail)]
    pool_refs = [p for p in saved.get('payloads', []) if Path(p['path']).name == 'pool.json']
    require(len(pool_refs) <= 1, 'Ambiguous generated pool')
    pool = evidence.read(pool_refs[0]) if pool_refs else None
    generation = []
    for wrapped in pool.get('proposal_reports', []) if pool else []:
        payload = wrapped.get('payload', wrapped)
        generation.append({key: payload.get(key) for key in (
            'object_id', 'generation_status', 'distribution_scope', 'attempted_draws',
            'valid_draws', 'unique_draws', 'support_admission', 'support_omissions',
            'compilation_seconds', 'sampling_seconds', 'grammar_hash', 'retrieval_hash')})
    if result.get('semantic_status') == 'known':
        from exact.repair.learning import SemanticTargetSpec
        from exact.repair.records import read_record
        from tools.repair.prepare import case_from_dict

        payloads = {Path(p['path']).name: p for p in saved['payloads']}
        require('selected-label.json' in payloads and 'repair.json' in payloads,
                'Known semantic score lacks label or native verification')
        label = evidence.read(payloads['selected-label.json'])
        repair = read_record(evidence.read(payloads['repair.json']))
        require(result['logical_status'] == repair.logical_status == 'VERIFIED_FEASIBLE' and
                repair.verification is not None and repair.verification.authorizes and
                label['feasible'] is True and list(repair.assignment) == label['assignment'],
                'Known label lacks authorizing assignment verification')
        require(label['benefit'] == result['semantic_benefit'] and label['cost'] == result['edit_cost'],
                'Semantic result differs from selected label')
        if 'selected_utility' in result:
            require(result['selected_utility'] == label['benefit'] - label['cost'],
                    'Selected utility differs from label')
        case = case_from_dict(evidence.read(item['evaluator']))
        arm = next(a for a in schedule['arms'] if a['id'] == saved['row']['arm_id'])
        weights = evidence.read(arm['protocol'])['teacher']['family_weights']
        target = SemanticTargetSpec(canonical_hash(case.probes), weights['desired'], weights['unwanted'])
        require(result['semantic_target_hash'] == target.content_hash, 'Semantic target identity differs')
        require([(p['probe_id'], p['family'], p['desired']) for p in label['semantic_vector']] ==
                [(p.probe_id, p.family, p.desired) for p in case.probes], 'Semantic query denominator differs')
    return dict(row=saved['row'], group_id=item['group_id'], family=item['family'],
        family_exposure=item['family_exposure'], control=item['control'],
        process_status=saved['status'], scientific_status=result.get('status', saved['status']),
        logical_status=result.get('logical_status', 'UNKNOWN'),
        semantic_status=result.get('semantic_status', 'unavailable'),
        semantic_benefit=result.get('semantic_benefit'), edit_cost=result.get('edit_cost'),
        selected_utility=result.get('selected_utility'),
        semantic_claim_qualified=False, generation_statuses=result.get('generation_statuses', []),
        candidate_counts=result.get('candidate_counts'),
        pool=pool_refs[0] if pool_refs else None, generation_reports=generation,
        inventory_hash=result.get('inventory_hash'),
        elapsed_seconds=saved['elapsed_seconds'], resources=saved['resources'],
        first_verified_seconds=result.get('first_verified_seconds'), checks=result.get('checks'),
        master_solves=result.get('master_solves'), remaining_budget=saved.get('remaining_budget'),
        recovery_action=saved.get('recovery_action'), result=saved.get('result'),
        detail=result.get('detail', saved.get('detail', '')), software_errors=errors)


def audit_rows(evidence, schedule, entries):
    expected = {r['id']: r for r in schedule['rows']}
    seen, rows = set(), []
    for item, source in entries:
        report = checkpoint(evidence, item['report'])
        require(report['status'] == 'complete' and evidence.read(report['schedule']) == schedule,
                'Incomplete or mismatched shard report')
        start, stop = item['row_slice']
        wanted = schedule['rows'][start:stop]
        require(report['scheduled'] == report['recorded'] == len(report['rows']) == len(wanted),
                'Shard denominator differs')
        require([r['row_id'] for r in report['rows']] == [r['id'] for r in wanted],
                'Shard order or slice differs')
        batch = evidence.read(item['batch'])
        recovery = report.get('cleanup_recovery')
        runtime = source['runtime']
        if recovery:
            plan = evidence.read(recovery['plan'])
            require(plan['schema_plan'] == report['plan'] and plan['evidence'] == recovery['evidence'],
                    'Cleanup plan differs')
            batch = evidence.read(plan['frozen_batch'])
            runtime = plan['runtime']
            require(runtime == evidence.read(binding(Path(plan['frozen_batch']['path']).with_name('runtime.json'))),
                    'Cleanup scientific runtime differs')
        identity = schema_identity(evidence, report['plan'], batch, runtime)
        require(report['identity'] == (canonical_hash((recovery['plan'], identity))
                if recovery else identity), 'Shard source/runtime identity differs')
        plan = evidence.read(report['plan'])
        require(evidence.read(plan['schedule']) == schedule, 'Recovery changed schedule')
        admission = evidence.read(plan['preflight'])
        admissions = {r['row_id']: r for r in admission['rows']}
        require(set(admissions) == set(expected), 'Schema admission denominator differs')
        counts = Counter()
        for ref in report['rows']:
            saved = lineage_row(evidence, ref, expected[ref['row_id']])
            row = saved['row']
            require(row['id'] not in seen and saved['cleanup_complete'],
                    'Duplicate row or unresolved cleanup')
            seen.add(row['id'])
            entry = next(e for e in plan['rows'] if e['row_id'] == row['id'])
            if saved['status'] == 'unknown_after_cleanup_reconciliation':
                validate_unknown(evidence, saved, recovery, ref)
            else:
                evidence.checkpoint(ref, canonical_hash((identity, row, entry, admissions[row['id']])))
                guard = Path(ref['path']).parent.parent / 'inflight' / (row['id'] + '.json')
                require(not guard.exists(), 'Finished row retains unresolved ownership guard')
            require(ref['status'] == saved['status'], 'Row reference status differs')
            arm = next(a for a in schedule['arms'] if a['id'] == row['arm_id'])
            validate_budget(evidence, saved, arm)
            counts[saved['status']] += 1
            summary = row_summary(evidence, saved, schedule['cases'][row['case_index']], schedule)
            summary.update(receipt=ref, source_run_id=item['run']['id'])
            rows.append(summary)
        require(dict(counts) == report['outcomes'], 'Shard outcome counts differ')
    require(seen == set(expected), 'Missing primary denominator rows')
    return rows


def audit(manifest_path, output):
    from tools.repair.report_campaign import totals

    evidence = Evidence()
    manifest = evidence.read(binding(manifest_path))
    schedule, semantic_scope = validate_schedule(evidence, manifest)
    references(evidence, evidence.read(manifest['primary_validation']))
    ledger = evidence.read(manifest['ledger_snapshot'])
    require(ledger['limit_worker_seconds'] is None, 'Campaign time amendment missing')
    active = set(manifest['active_run_ids'])
    require(len(active) == len(manifest['active_run_ids']) == 36, 'Active shard denominator differs')
    sources, entries = [], []
    for item in manifest['attempts']:
        source = validate_attempt(evidence, item, ledger, item['run']['id'] in active)
        references(evidence, item['commands'])
        # Preserve and check original completed output inventories too.
        if item.get('outputs'):
            outputs = evidence.read(item['outputs'])
            for relative, digest in outputs.items():
                evidence.verify(dict(path=str(Path(source['completion']['work']) / relative), sha256=digest))
        if item['run']['id'] in active:
            relative = item['run']['result_relative']
            require(str(Path(source['completion']['work']) / relative) == item['report']['path'] and
                    outputs[relative] == item['report']['sha256'], 'Report lacks worker output receipt')
            entries.append((item, source))
        sources.append(source)
    by_id = {s['run']['id']: s for s in sources}
    require(len(by_id) == len(sources) and active <= set(by_id), 'Missing or duplicate attempt')
    snapshot = evidence.read(manifest['registry_snapshot'])
    all_runs = {r['id']: r for r in snapshot['runs']}
    for source in sources:
        run = source['run']
        require(all_runs[run['id']] == run, 'Attempt differs from registry snapshot')
        if run.get('superseded_by'):
            child = by_id[run['superseded_by']]['run']
            require(child.get('recovery_of') == run['id'], 'Attempt supersession ancestry lost')
    rows = audit_rows(evidence, schedule, entries)
    errors = [dict(row_id=r['row']['id'], errors=r['software_errors']) for r in rows if r['software_errors']]
    costs = {str(Path(s['run']['completion_path']).parent): s['charge'] for s in sources}
    result = dict(schema='exact-repair/fresh-evaluation-primary-audit/v1',
        manifest=binding(manifest_path), status='complete', stage_id=manifest['stage_id'],
        study_complete=False, campaign_complete=False, gates_passed=False,
        scheduled_rows=576, recorded_rows=len(rows), learned_rows=384, control_rows=192,
        cases=64, parent_groups=32, profile_timeouts_retained=4,
        rows=rows, attempts=sources, attempt_costs=costs,
        worker_seconds=sum(c['elapsed_seconds'] for c in costs.values()),
        measured_cpu_seconds=sum(c.get('cpu_seconds') or 0 for c in costs.values()),
        ledger_snapshot=manifest['ledger_snapshot'], cumulative_costs=totals(ledger['attempts']),
        cost_scope='All captured original/validation/schema attempts, including failed attempts. '
                   'Maintenance and this audit worker are charged separately; row reuse is not charged twice.',
        process_statuses=dict(Counter(r['process_status'] for r in rows)),
        scientific_statuses=dict(Counter(r['scientific_status'] for r in rows)),
        logical_statuses=dict(Counter(r['logical_status'] for r in rows)),
        semantic_statuses=dict(Counter(r['semantic_status'] for r in rows)),
        scientific_errors=errors, semantic_query_scope=semantic_scope,
        semantic_query_totals={key: sum(s.get(key, 0) for s in semantic_scope)
            for key in ('probe_count', 'desired', 'unwanted', 'nonvacuity_declared')},
        symbolic_control_scope=schedule['control_definition'],
        strongest_symbolic_comparison_qualified=False,
        required_followups=['common_inventory_value_and_selection_diagnostics',
            'intended_parent_and_query_scope_qualification', 'required_semantic_control_review',
            'failure_inclusive_parent_group_final_report'],
        limitations=[schedule['symbolic_limitation'], schedule['gate_status'], schedule['coverage'],
            'Native intended-parent/query truth is not established by receipt integrity; '
            'stored semantic scores remain descriptive and unqualified.',
            'Semantic scope is limited to the declared query basis. Absent unwanted or '
            'explicit nonvacuity queries cannot support claims about those dimensions.',
            'Process completion does not imply usable labels, complete generation or semantic success.',
            'Schema-unavailable, partial, unsupported and cleanup-unknown rows remain in all denominators.'],
        followup_stage=manifest['followup_stage'], scientific_rows_rerun=0,
        api_spend_usd=0, verified_file_count=len(evidence.files))
    write_artifact(Path(output) / 'report.json', result)
    write_artifact(Path(output) / 'verified-files.json', evidence.files)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.manifest, args.output)
