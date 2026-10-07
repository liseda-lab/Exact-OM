"""Authenticate all frozen robustness receipts without scientific replay.

This audit supplies row evidence for the separately registered paired report.
It does not close robustness, qualify research gates, or acquire labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, read_record
from tools.repair import fresh_evaluation_audit as fresh
from tools.repair.overlap_audit import Evidence, binding, require
from tools.repair.scaling_audit import checkpoint
from tools.repair.report_campaign import totals

SCHEMA = 'exact-repair/robustness-audit-manifest/v1'
CONDITIONS = ('baseline', 'score_noise_020', 'evidence_missing_050',
              'final_pool_050', 'retrieval_symbols_025')


class AuditEvidence(Evidence):
    """Verify each immutable file once; bound the parsed-document cache."""
    def __init__(self):
        super().__init__()
        self.documents = OrderedDict()

    def verify(self, ref):
        if ref['path'] in self.files:
            require(self.files[ref['path']] == ref['sha256'], 'Conflicting evidence binding')
            return Path(ref['path'])
        return super().verify(ref)

    def read(self, ref):
        path = self.verify(ref)
        key = (str(path), ref['sha256'])
        if key not in self.documents:
            self.documents[key] = json.loads(path.read_text())
            if len(self.documents) > 32:
                self.documents.popitem(last=False)
        self.documents.move_to_end(key)
        return self.documents[key]


def validate_schedule(evidence, manifest):
    schedule = evidence.read(manifest['schedule'])
    base = evidence.read(schedule['base_schedule'])
    require(schedule['schema'] == 'exact-repair/fresh-evaluation/v1' and
            len(base['cases']) == 64 and len(schedule['cases']) == 320 and
            len(schedule['rows']) == 2880 and len(schedule['arms']) == 9,
            'Robustness denominator differs')
    require(schedule['arms'] == base['arms'] and
            sum(a['kind'] == 'learned' for a in schedule['arms']) == 6,
            'Frozen model/control identities changed')
    require([c[0] for c in schedule['conditions']] == list(CONDITIONS) and
            schedule['profile_timeouts_retained'] == 4 and
            not schedule['test_feedback_for_selection'] and not schedule['warm_start'],
            'Frozen intervention or selection policy changed')
    fresh.references(evidence, schedule)
    parents = {}
    for i, original in enumerate(base['cases']):
        parents.setdefault(original['group_id'], []).append(original['control'])
        variants = schedule['cases'][5*i:5*i+5]
        require([v['condition'] for v in variants] == list(CONDITIONS), 'Condition denominator differs')
        for item in variants:
            require(item['base_case_id'] == original['case_id'] and
                    item['case_id'] == original['case_id'] + ':robustness:' + item['condition'],
                    'Variant ancestry differs')
            for key in ('group_id','family','family_exposure','control','split','structural_parent',
                        'source_parent','fingerprints','evaluator','status'):
                require(item[key] == original[key], 'Inherited case identity differs: ' + key)
            if item['status'] == 'materialized':
                control = item['intervention']
                require(control['condition'] == item['condition'] and control['group_id'] == item['group_id']
                        and control['seed'] == 20261003 and control['original_input_hash'] == original['input_hash']
                        and control['effective_input_hash'] == item['input_hash'], 'Intervention binding differs')
                require(read_record(evidence.read(item['observable'])).content_hash == item['input_hash'],
                        'Effective input identity differs')
    require(len(parents) == 32 and all(Counter(v) == {'coherent':1,'corrupted':1}
                                     for v in parents.values()), 'Parent pairing differs')
    expected = []
    for arm in schedule['arms']:
        resources = evidence.read(arm['protocol'])['resources']
        require(all(resources[k] == v for k,v in dict(case_wall_seconds=300,
            case_cpu_seconds=600,case_rss_mb=8192,generation_seconds=60).items()),
            'Scientific budgets changed')
        for i, case in enumerate(schedule['cases']):
            row = dict(arm_id=arm['id'],case_index=i,case_id=case['case_id'],
                       seconds=300,cpu_seconds=600,memory_mb=8192)
            row['id'] = canonical_hash(row); expected.append(row)
    require(schedule['rows'] == expected, 'Row schedule changed')
    return schedule


def recovery_plans(evidence, report):
    """Bind every descendant, including the two distinct slice-24 unknowns."""
    plans = []
    for key in ('cleanup_recovery','omission_recovery','removal_recovery'):
        if key not in report:
            continue
        rec = report[key]
        ref = rec['plan']
        seen = set()
        while ref:
            require(ref['path'] not in seen, 'Cyclic recovery plan')
            seen.add(ref['path'])
            plan = evidence.read(ref)
            fresh.references(evidence, plan)
            require(plan['schema_plan'] == report['plan'] and not plan['prior_costs_reset'],
                    'Recovery changes science plan or costs')
            require(plan['evidence'] == rec['evidence'] or len(plans) > 0,
                    'Recovery evidence differs')
            plans.append((ref, plan))
            ref = plan.get('previous_cleanup_plan')
    return plans


def validate_unknown(evidence, saved, ref, plans):
    from tools.repair import schema_cleanup_recovery as cleanup
    from tools.repair import schema_omission_recovery as omission
    from tools.repair import schema_removal_recovery as removal
    functions = {
        'unknown_after_cleanup_reconciliation': ('cleanup_reconciliation',cleanup.reconciled_unknown),
        'unknown_after_omission_failure': ('omission_reconciliation',omission.reconciled_unknown),
        'unknown_after_removal_failure': ('removal_reconciliation',removal.reconciled_unknown)}
    key, reconstruct = functions[saved['status']]
    matches = [(p,v) for p,v in plans if v['evidence'] == saved[key]]
    require(len(matches) == 1, 'Unknown row lacks its exact recovery plan')
    plan_ref, plan = matches[0]
    proof = evidence.read(plan['evidence']); fresh.references(evidence, proof)
    original = checkpoint(evidence, saved['original_receipt'])
    args = [original]
    if key == 'cleanup_reconciliation':
        args.append(checkpoint(evidence, proof['guard']))
    args.extend([saved['original_receipt'], plan['evidence'], saved['ownership'], plan['original_step']])
    expected = reconstruct(*args)
    require({k:v for k,v in saved.items() if k not in ('identity','content_hash')} == expected,
            'Unknown row changed or promoted')
    evidence.checkpoint(ref, canonical_hash((plan_ref,plan['schema_identity'])))
    require(saved['result'] is None and saved['additional_elapsed_seconds'] == 0,
            'Spent unknown row replayed')


def scientific_identity(evidence, plan_ref, batch, runtime):
    return fresh.schema_identity(evidence, plan_ref, batch, runtime)


def identity_sources(evidence, item, source, report, plans):
    sources = []
    if not plans:
        batch = evidence.read(item['batch'])
        identity = scientific_identity(evidence,report['plan'],batch,source['runtime'])
        require(report['identity'] == identity, 'Shard source/runtime identity differs')
        return [(identity,batch['commit'])]
    root_ref, root = plans[0]
    require(report['identity'] == canonical_hash((root_ref,root['schema_identity'])),
            'Recovery report identity differs')
    for _, plan in plans:
        batch = evidence.read(plan['frozen_batch'])
        runtime = plan.get('original_runtime',plan['runtime'])
        bound_runtime = evidence.read(binding(Path(plan['frozen_batch']['path']).with_name('runtime.json')))
        require(runtime == bound_runtime, 'Original scientific runtime differs')
        identity = scientific_identity(evidence,report['plan'],batch,runtime)
        require(identity == plan['schema_identity'], 'Original scientific identity differs')
        sources.append((identity,batch['commit']))
        if plan.get('corrected_source_root'):
            root = Path(plan['corrected_source_root'])
            revised = evidence.read(binding(root.parent/'batch.json'))
            revision = evidence.read(plan['source_revision'])
            for relative,digest in revision['files'].items():
                evidence.verify(dict(path=str(root/relative),sha256=digest))
            corrected_runtime = evidence.read(binding(root.parent/'runtime.json'))
            require(corrected_runtime == plan['runtime'], 'Corrected runtime differs')
            sources.append((scientific_identity(evidence,report['plan'],revised,corrected_runtime),revised['commit']))
    return list(dict.fromkeys(sources))


def intervention_scope(evidence, saved, item, summary):
    from tools.repair.robustness import audit_final_pool
    control = item.get('intervention',{})
    scope = dict(condition=item['condition'],applicable=control.get('applicable'),
        removal_target_count=sum(map(len,control.get('final_candidate_removals',{}).values())),
        omitted_symbol_count=len(control.get('omitted_generation_symbols',[])),
        final_exclusion_verified=None,status='unavailable_no_published_pool',
        effective_menu_scope='Declared output-symbol omission; asserted evidence remains visible. '
                             'Full retrieved-menu snapshots were not persisted separately.')
    if summary['pool']:
        pool = evidence.read(summary['pool'])
        checked = audit_final_pool(read_record(pool['input']),item)
        result = evidence.read(saved['result']) if saved.get('result') else {}
        if result:
            require(result.get('intervention_audit') == checked, 'Persisted final exclusion audit differs')
        else:
            require(saved['status'].startswith('unknown_after_'),
                    'Published pool lacks a result or reconciled unknown receipt')
        scope.update(checked,status='published_final_pool_exclusion_verified',
            outer_exclusion_receipt='matched' if result else 'unavailable_retained_unknown')
    return scope



def execution_provenance(evidence, saved, receipt_commit):
    """A schema view does not change the source that produced a reused result."""
    if not saved.get('reuse_of'):
        return dict(kind='schema_execution_or_explicit_unavailability',source_commit=receipt_commit)
    require(saved.get('reuse_source') and saved.get('additional_elapsed_seconds') == 0,
            'Reused row lacks original owner or spends new work')
    prior = checkpoint(evidence,saved['reuse_of']); owner = saved['reuse_source']
    complete = evidence.read(owner['completion']); outputs = evidence.read(owner['outputs'])
    require(complete['step_id'] == owner['step_id'] and complete['dispatch_nonce'] == owner['nonce'],
            'Reused scientific owner differs')
    relative = str(Path(saved['reuse_of']['path']).relative_to(complete['work']))
    require(outputs[relative] == saved['reuse_of']['sha256'], 'Original worker does not bind reused row')
    for key in ('row','status','resources','elapsed_seconds','cleanup_complete'):
        require(saved[key] == prior[key], 'Reused row changed original outcome or effort')
    action = saved['recovery_action']
    if action == 'reused_original_unchanged':
        require(saved['result'] == prior['result'] and saved['payloads'] == prior['payloads'],
                'Unchanged reuse changed scientific payloads')
    else:
        require(action == 'reclassified_original_unsupported_without_rerun', 'Unexpected reuse revision')
        original = evidence.read(prior['result']); revised = evidence.read(saved['result'])
        require(original['status'] == 'generation_error' and original['detail'] ==
                'ValueError: cannot infer complete original relation for endpoint retrieval',
                'Unexpected original support error')
        require(revised == dict(original,status='unsupported_original_mapping_bundle',
                original_result=prior['result'],classification_revision='explicit-input-support/v1'),
                'Support reclassification changed scientific result')
    batch = evidence.read(binding(complete['batch']))
    return dict(kind=action,source_commit=batch['commit'],original_owner=owner,
                original_receipt=saved['reuse_of'],additional_elapsed_seconds=0)

def audit_rows(evidence, schedule, entries):
    expected = {r['id']:r for r in schedule['rows']}; seen = set(); rows = []
    for item, source in entries:
        report = checkpoint(evidence,item['report'])
        require(report['status'] == 'complete' and evidence.read(report['schedule']) == schedule,
                'Incomplete or mismatched shard')
        start,stop = item['row_slice']; wanted = schedule['rows'][start:stop]
        require(report['scheduled'] == report['recorded'] == len(report['rows']) == len(wanted)
                and [r['row_id'] for r in report['rows']] == [r['id'] for r in wanted],
                'Shard denominator or ordering differs')
        plans = recovery_plans(evidence,report)
        identities = identity_sources(evidence,item,source,report,plans)
        plan = evidence.read(report['plan']); admission = evidence.read(plan['preflight'])
        admissions = {r['row_id']:r for r in admission['rows']}
        require(set(admissions) == set(expected), 'Schema admission denominator differs')
        counts = Counter()
        for ref in report['rows']:
            saved = fresh.lineage_row(evidence,ref,expected[ref['row_id']]); row = saved['row']
            require(row['id'] not in seen and saved['cleanup_complete'], 'Duplicate or unclean row')
            seen.add(row['id']); entry = plan['rows'][row['case_index'] +
                next(i for i,a in enumerate(schedule['arms']) if a['id']==row['arm_id'])*len(schedule['cases'])]
            require(entry['row_id'] == row['id'], 'Schema plan order differs')
            regimes = []
            if saved['status'].startswith('unknown_after_'):
                validate_unknown(evidence,saved,ref,plans)
                regimes = ['retained_unknown_without_replay']
            else:
                regimes = [commit for identity,commit in identities if
                    saved['identity'] == canonical_hash((identity,row,entry,admissions[row['id']]))]
                require(len(regimes) == 1, 'Row source/runtime identity differs')
                guard = Path(ref['path']).parent.parent/'inflight'/(row['id']+'.json')
                require(not guard.exists(), 'Unresolved row ownership guard')
            require(ref['status'] == saved['status'], 'Row reference status differs')
            arm = next(a for a in schedule['arms'] if a['id'] == row['arm_id'])
            fresh.validate_budget(evidence,saved,arm)
            case = schedule['cases'][row['case_index']]
            summary = fresh.row_summary(evidence,saved,case,schedule)
            execution = execution_provenance(evidence,saved,regimes[0])
            summary.update(receipt=ref,source_run_id=item['run']['id'],
                receipt_source_regime=regimes[0],source_regime=execution['source_commit'],execution_source=execution,
                retained_error={k:saved[k] for k in ('original_status','original_detail','original_result','original_receipt') if k in saved},
                base_case_id=case['base_case_id'],condition=case['condition'],
                intervention_scope=intervention_scope(evidence,saved,case,summary))
            rows.append(summary); counts[saved['status']] += 1
        require(dict(counts) == report['outcomes'], 'Shard outcomes differ')
        print(f'Authenticated {len(rows)}/{len(expected)} rows: {item["run"]["id"]}',flush=True)
    require(seen == set(expected), 'Missing robustness denominator')
    return rows


def audit(manifest_path, output):
    evidence = AuditEvidence(); manifest = evidence.read(binding(manifest_path))
    require(manifest['schema'] == SCHEMA and manifest['scientific_rows_replayed'] == 0
            and manifest['native_calls'] == 0, 'Audit boundary differs')
    amendment = evidence.read(manifest['scope_amendment'])
    require(not amendment['original_scientific_budgets_changed'], 'Scope amendment differs')
    schedule = validate_schedule(evidence,manifest)
    ledger = evidence.read(manifest['ledger_snapshot']); snapshot = evidence.read(manifest['registry_snapshot'])
    require(ledger['limit_worker_seconds'] is None, 'Campaign authorization changed')
    active = set(manifest['active_run_ids'])
    require(len(active) == len(manifest['active_run_ids']) == 180, 'Active shard denominator differs')
    all_runs = {r['id']:r for r in snapshot['runs']}; sources = []; entries = []
    for item in manifest['attempts']:
        run = item['run']; require(all_runs[run['id']] == run, 'Attempt registry ownership differs')
        source = fresh.validate_attempt(evidence,item,ledger,run['id'] in active)
        fresh.references(evidence,item.get('commands',[]))
        if item.get('outputs'):
            outputs = evidence.read(item['outputs'])
            for relative,digest in outputs.items():
                evidence.verify(dict(path=str(Path(source['completion']['work'])/relative),sha256=digest))
        if run['id'] in active:
            require(item.get('outputs') and outputs[run['result_relative']] == item['report']['sha256']
                    and str(Path(source['completion']['work'])/run['result_relative']) == item['report']['path'],
                    'Report lacks worker output binding')
            entries.append((item,source))
        sources.append(source)
        if len(sources) % 20 == 0:
            print(f'Authenticated {len(sources)}/{len(manifest["attempts"])} attempts',flush=True)
    by_id = {s['run']['id']:s for s in sources}
    require(len(by_id) == len(sources) and active <= set(by_id), 'Missing or duplicate attempt')
    for source in sources:
        run = source['run']
        if run.get('superseded_by'):
            child = by_id[run['superseded_by']]['run']
            require(child.get('recovery_of') == run['id'], 'Replacement ancestry lost')
    rows = audit_rows(evidence,schedule,entries)
    costs = {str(Path(s['run']['completion_path']).parent):s['charge'] for s in sources}
    scope = []
    for case in evidence.read(schedule['base_schedule'])['cases']:
        raw = evidence.read(case['evaluator'])['case']
        scope.append(dict(case_id=case['case_id'],group_id=case['group_id'],probe_count=len(raw['probes']),
            desired=sum(p['desired'] for p in raw['probes']),unwanted=sum(not p['desired'] for p in raw['probes']),
            nonvacuity_declared=sum(p.get('nonvacuity') is not None for p in raw['probes']),
            typed_nonvacuity='Derived by existing query implementation when undeclared',
            intended_qualification='Separate completed fresh scope audit; no new native qualification here'))
    result = dict(schema='exact-repair/robustness-receipt-audit/v1',manifest=binding(manifest_path),
        status='complete',study_complete=False,campaign_complete=False,gates_passed=False,
        scheduled_rows=2880,recorded_rows=len(rows),learned_rows=1920,control_rows=960,
        case_variants=320,base_cases=64,parent_groups=32,profile_timeouts_retained=4,
        rows=rows,attempts=sources,attempt_costs=costs,attempt_cost_totals=totals(costs),
        cumulative_costs=totals(ledger['attempts']),ledger_snapshot=manifest['ledger_snapshot'],
        process_statuses=dict(Counter(r['process_status'] for r in rows)),
        scientific_statuses=dict(Counter(r['scientific_status'] for r in rows)),
        semantic_statuses=dict(Counter(r['semantic_status'] for r in rows)),
        source_regimes=dict(Counter(r['source_regime'] for r in rows)),
        intervention_coverage={c:dict(scheduled_rows=sum(r['condition']==c for r in rows),
            applicable_rows=sum(r['condition']==c and bool(r['intervention_scope']['applicable']) for r in rows),
            pool_verified_rows=sum(r['condition']==c and r['intervention_scope']['final_exclusion_verified'] is True for r in rows))
            for c in CONDITIONS},
        semantic_query_scope=scope,related_scope_and_overlap=manifest['related_scope_and_overlap'],
        strongest_symbolic_comparison_qualified=False,scientific_rows_replayed=0,native_calls=0,
        cost_scope='Original validation, primary, schema and replacement attempts retained. Maintenance, '
                   'overlap and this audit charged separately; row reuse is not a new charge.',
        limitations=[schedule['symbolic_limitation'],schedule['gate_status'],schedule['coverage'],
            'Process completion does not establish usable labels or complete generation.',
            'Final exclusion is independently checked only for published pools; absent pools retain unknown scope.',
            'Full effective retrieval menus were not separately persisted; final vocabulary exclusion does not prove menu completeness.',
            'Stored selected labels lack per-obligation native traces. No unwanted probes; typed nonvacuity is retained.',
            'No new common-inventory robustness experiment; existing fresh diagnostic is separate. Any remaining requirement is a deferred design gap.'],
        required_followups=['authenticate_this_worker_receipt','bind_completed_overlap_and_fresh_scope_audits',
            'failure_inclusive_parent_paired_report_2000_bootstrap_seed_20261003',
            'carry_corrections_and_deferred_controls_to_preliminary_report'],
        followup_stage='xr21-expanded-robustness-001',api_spend_usd=0,verified_file_count=len(evidence.files))
    write_artifact(Path(output)/'report.json',result)
    write_artifact(Path(output)/'verified-files.json',evidence.files)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path); parser.add_argument('output',type=Path)
    args = parser.parse_args(); audit(args.manifest,args.output)
