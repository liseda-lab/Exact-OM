"""Receipt-bound preliminary fresh report; no inference, fitting or row replay."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
import statistics

from exact.repair.api import write_artifact
from tools.repair.fresh_evaluation_audit import validate_attempt
from tools.repair.overlap_audit import Evidence, binding, require
from tools.repair.report_campaign import totals

SCHEMA = 'exact-repair/evaluation-final-manifest/v1'
BOOTSTRAP = dict(resamples=2000, seed=20261003, confidence=0.95, unit='parent_group')
CONTROL_IDS = ('symbolic_rich_action', 'uniform', 'deletion')


def known(row):
    return (row['semantic_status'] == 'known' and
            row['logical_status'] == 'VERIFIED_FEASIBLE' and
            row['semantic_benefit'] is not None and row['edit_cost'] is not None)


def generated_known(row):
    # A selected fixed-inventory label alone is never generated-pool evidence.
    return known(row) and bool(row.get('pool')) and bool(row.get('generation_reports'))


def interval(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    def quantile(p):
        x = (len(values)-1)*p
        lo = int(x)
        hi = min(lo+1, len(values)-1)
        return values[lo] + (values[hi]-values[lo])*(x-lo)
    return [quantile(.025), quantile(.975)]


def bootstrap_clusters(clusters, seed=20261003, resamples=2000):
    """Cluster resampling preserves both variants and all missing rows in each draw.

    Missing quality is never zero-imputed. Conditional quality uses available
    values; its observed and bootstrap denominators accompany every interval.
    """
    require(bool(clusters) and all(clusters.values()), 'Empty parent clusters')
    groups = [clusters[k] for k in sorted(clusters)]
    def estimate(chosen):
        values = [v for group in chosen for v in group]
        usable = [v for v in values if v is not None]
        return (statistics.fmean(usable) if usable else None, len(usable), len(values))
    point, usable, scheduled = estimate(groups)
    usable_parents = sum(any(v is not None for v in g) for g in groups)
    rng = random.Random(seed)
    draws = [estimate([groups[rng.randrange(len(groups))] for _ in groups])
             for _ in range(resamples)]
    return dict(estimate=point, scheduled_rows=scheduled, usable_rows=usable,
        missing_rows=scheduled-usable, parent_groups=len(groups),
        parents_with_usable_values=usable_parents,
        descriptive_95_interval=interval([d[0] for d in draws]) if usable_parents>1 else None,
        interval_unavailable_reason='fewer_than_two_usable_parents' if usable_parents<2 else None,
        valid_bootstrap_draws=sum(d[0] is not None for d in draws),
        bootstrap_usable_rows_range=[min(d[1] for d in draws), max(d[1] for d in draws)],
        bootstrap_scheduled_rows_range=[min(d[2] for d in draws), max(d[2] for d in draws)],
        missing_policy='Retained in scheduled denominator; conditional quality is not imputed',
        uncertainty_scope='Descriptive parent-group bootstrap, no confirmatory inference')


def metric(rows, value):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row['group_id']].append(value(row))
    return bootstrap_clusters(clusters)


def summary(rows, primary):
    eligible = generated_known if primary else known
    quality = [r['semantic_benefit'] for r in rows if eligible(r)]
    errors = [r['value_diagnostic']['selected_utility_error'] for r in rows
              if r.get('value_diagnostic', {}).get('selected_utility_error') is not None]
    return dict(scheduled_rows=len(rows), recorded_rows=len(rows),
        parent_groups=len({r['group_id'] for r in rows}),
        process_statuses=dict(Counter(r['process_status'] for r in rows)),
        scientific_statuses=dict(Counter(r['scientific_status'] for r in rows)),
        logical_statuses=dict(Counter(r['logical_status'] for r in rows)),
        semantic_statuses=dict(Counter(r['semantic_status'] for r in rows)),
        native_verified_selected_scores=sum(known(r) for r in rows),
        quality_usable_rows=len(quality), quality_unavailable_rows=len(rows)-len(quality),
        conditional_quality_mean=statistics.fmean(quality) if quality else None,
        generated_pool_rows=sum(bool(r.get('generation_reports')) for r in rows) if primary else None,
        generation_object_statuses=dict(Counter(g['generation_status'] for r in rows
                                               for g in r.get('generation_reports', []))),
        generation_support_omissions=sum(bool(g.get('support_omissions')) for r in rows
                                         for g in r.get('generation_reports', [])),
        row_wall_seconds=sum(r['elapsed_seconds'] for r in rows),
        software_error_rows=sum(bool(r.get('software_errors')) for r in rows),
        selected_value_error_count=len(errors),
        conditional_selected_absolute_error=statistics.fmean(map(abs, errors)) if errors else None)


def arm_report(rows, primary):
    eligible = generated_known if primary else known
    return dict(**summary(rows, primary), metrics={
        'verified_quality_coverage': metric(rows, lambda r: int(eligible(r))),
        'conditional_selected_benefit': metric(rows, lambda r: r['semantic_benefit'] if eligible(r) else None),
        'conditional_selected_utility': metric(rows, lambda r:
            r['semantic_benefit']-r['edit_cost'] if eligible(r) else None),
        'row_wall_seconds': metric(rows, lambda r: r['elapsed_seconds'])})


def paired_comparison(rows, learned, control, primary):
    indexed = {(r['row']['arm_id'], r['row']['case_id']):r for r in rows}
    require(len(indexed) == len(rows), 'Duplicate arm/case row')
    cases = {r['row']['case_id'] for r in rows if r['row']['arm_id']==learned}
    require(cases == {r['row']['case_id'] for r in rows if r['row']['arm_id']==control},
            'Missing paired control row')
    clusters = {k:defaultdict(list) for k in ('coverage_difference', 'conditional_benefit_difference',
                                           'conditional_utility_difference', 'wall_seconds_difference')}
    eligible = generated_known if primary else known
    for case in sorted(cases):
        left, right = indexed[learned, case], indexed[control, case]
        require((left['group_id'],left['control'],left['family_exposure']) ==
                (right['group_id'],right['control'],right['family_exposure']), 'Paired parent differs')
        parent = left['group_id']; both = eligible(left) and eligible(right)
        clusters['coverage_difference'][parent].append(int(eligible(left))-int(eligible(right)))
        clusters['conditional_benefit_difference'][parent].append(
            left['semantic_benefit']-right['semantic_benefit'] if both else None)
        clusters['conditional_utility_difference'][parent].append(
            left['semantic_benefit']-left['edit_cost']-right['semantic_benefit']+right['edit_cost'] if both else None)
        clusters['wall_seconds_difference'][parent].append(left['elapsed_seconds']-right['elapsed_seconds'])
    return dict(learned_arm=learned, historical_control=control, direction='learned minus control',
        scheduled_pairs=len(cases), metrics={k:bootstrap_clusters(v) for k,v in clusters.items()},
        strongest_symbolic_comparison_qualified=False)


def validate_rows(rows, schedule):
    expected = {r['id']:r for r in schedule['rows']}
    require(len(rows)==len(expected)==576, 'Row denominator differs')
    require({r['row']['id'] for r in rows} == set(expected), 'Missing or duplicate row')
    for row in rows:
        require(row['row'] == expected[row['row']['id']], 'Row schedule differs')
        case = schedule['cases'][row['row']['case_index']]
        require((row['row']['case_id'],row['group_id'],row['family'],row['family_exposure'],row['control']) ==
                (case['case_id'],case['group_id'],case['family'],case['family_exposure'],case['control']),
                'Case ancestry or stratum differs')
    groups = defaultdict(list)
    for case in schedule['cases']:
        groups[case['group_id']].append(case)
    require(len(groups)==32 and len(schedule['cases'])==64, 'Parent denominator differs')
    require(all(Counter(c['control'] for c in group)=={'coherent':1,'corrupted':1}
                for group in groups.values()), 'Paired variants differ')


def authenticate_audit(evidence, item, ledger, snapshot):
    run = item['run']
    require(next(r for r in snapshot['runs'] if r['id']==run['id']) == run,
            'Audit owner changed')
    source = validate_attempt(evidence, item, ledger, True)
    outputs = evidence.read(item['outputs'])
    work = Path(source['completion']['work'])
    for key in ('report','verified_files'):
        relative = str(Path(item[key]['path']).relative_to(work))
        require(outputs.get(relative)==item[key]['sha256'], 'Audit lacks worker output binding')
    for relative, digest in outputs.items():
        evidence.verify(dict(path=str(work/relative),sha256=digest))
    files = evidence.read(item['verified_files'])
    for path, digest in files.items():
        if path not in evidence.files:
            evidence.verify(dict(path=path,sha256=digest))
        else:
            require(evidence.files[path]==digest,'Conflicting prior audit evidence')
    report = evidence.read(item['report'])
    require(report['status']=='complete' and not report['gates_passed'] and not report['study_complete'],
            'Audit boundary differs')
    return report, source


def attempt_costs(reports, owners, ledger, evidence=None):
    """Union immutable attempt charges: reused rows never create a second charge."""
    evidence = evidence or Evidence()
    costs = {}; attempts = []
    for report in reports.values():
        for source in report['attempts']:
            runid = source.get('run_id') or source['run']['id']
            complete = source['completion']
            # Scope audit binds completions by file/hash; primary embeds their data.
            if 'path' in complete:
                key = str(Path(complete['path']).parent)
                complete = evidence.read(complete)
            else:
                key = next((k for k,v in ledger['attempts'].items()
                        if v == source['charge'] and k.endswith('/'+runid.rsplit('-',1)[1])
                        and v['logical_id']==complete['job_id']), None)
            require(key is not None and ledger['attempts'].get(key)==source['charge'],
                    'Original charge missing or changed')
            if key not in costs:
                costs[key] = source['charge']
                attempts.append(dict(run_id=runid,status=complete['status'],
                    exit_code=complete['exit_code'],step_id=complete['step_id'],ledger_key=key,
                    superseded_by=source.get('run',{}).get('superseded_by'),
                    recovery_of=source.get('run',{}).get('recovery_of')))
    for source in owners:
        key = str(Path(source['run']['completion_path']).parent)
        require(key not in costs, 'Audit worker counted twice')
        costs[key] = source['charge']
        c = source['completion']
        attempts.append(dict(run_id=source['run']['id'],status=c['status'],exit_code=c['exit_code'],
                             step_id=c['step_id'],ledger_key=key))
    return costs, attempts


def report(manifest_path, output):
    evidence = Evidence(); manifest = evidence.read(binding(manifest_path))
    require(manifest['schema']==SCHEMA and manifest['bootstrap']==BOOTSTRAP and
            manifest['scientific_rows_replayed']==manifest['native_calls']==0 and
            manifest['supervision_admitted'] is False, 'Final report boundary differs')
    amendment = evidence.read(manifest['scope_amendment'])
    require(amendment['schema']=='exact-repair/preliminary-scope-amendment/v1' and
            not amendment['original_scientific_budgets_changed'], 'Preliminary scope amendment differs')
    evidence.read(manifest['program'])
    require(evidence.read(manifest['authorization'])['limit_worker_seconds'] is None, 'Time amendment missing')
    ledger = evidence.read(manifest['ledger_snapshot'])
    require(ledger['limit_worker_seconds'] is None, 'Campaign limit changed')
    snapshot = evidence.read(manifest['registry_snapshot'])
    reports = {}; owners = []
    for item in manifest['audits']:
        require(item['kind'] not in reports, 'Duplicate audit branch')
        reports[item['kind']], owner = authenticate_audit(evidence,item,ledger,snapshot)
        owners.append(owner)
    require(set(reports)=={'primary','inventory','scope'}, 'Missing audited branch')
    primary, inventory, scope = (reports[k] for k in ('primary','inventory','scope'))
    schedules = {k:evidence.read(evidence.read(reports[k]['manifest'])['schedule'])
                 for k in ('primary','inventory')}
    require(schedules['primary']['cases']==schedules['inventory']['cases'], 'Diagnostic case identities changed')
    require(schedules['primary']['arms']==schedules['inventory']['arms'], 'Diagnostic models/controls changed')
    require({r['case_id'] for r in scope['rows']} == {c['case_id'] for c in schedules['primary']['cases']}
            and len(scope['rows'])==64, 'Intended qualification denominator differs')
    costs, attempts = attempt_costs(reports,owners,ledger,evidence)
    branches = {}
    for name in ('primary','inventory'):
        rows = reports[name]['rows']; schedule = schedules[name]; is_primary = name=='primary'
        validate_rows(rows,schedule)
        arms = sorted({r['row']['arm_id'] for r in rows})
        require(len(arms)==9 and set(CONTROL_IDS)<=set(arms), 'Arm denominator differs')
        branches[name] = dict(**summary(rows,is_primary), rows=rows,
            metric_scope=('Generated-pool native-verified selected quality; partial/limited generation stays explicit'
                if is_primary else 'Fixed-inventory selected-only diagnostic; not primary quality or external regret'),
            by_arm={a:arm_report([r for r in rows if r['row']['arm_id']==a],is_primary) for a in arms},
            strata={field:{str(value):dict(**summary([r for r in rows if r[field]==value],is_primary),
                by_arm={a:arm_report([r for r in rows if r[field]==value and r['row']['arm_id']==a],is_primary)
                        for a in arms}) for value in sorted({r[field] for r in rows})}
                    for field in ('family','family_exposure','control')},
            paired_comparisons=[paired_comparison(rows,a,c,is_primary)
                for a in arms if a not in CONTROL_IDS for c in CONTROL_IDS],
            scientific_errors=reports[name]['scientific_errors'])
    corrections = [dict(item,local_disposition='deferred_correction_or_design_gap',
        authority=manifest['scope_amendment']) for item in scope['control_review']['required_comparisons']
        if item['status'] not in ('diagnostics_already_audited_with_limits','disabled_by_authorization')]
    corrections += [dict(id='model_schema_compatibility',local_disposition='correction_required_before_cluster',
        evidence=manifest['audits'][0]['report'],rows=branches['primary']['scientific_statuses'].get('unavailable_model_schema',0)),
        dict(id='native_generation_and_teardown',local_disposition='correction_and_resource_design_review',
        evidence=[x['report'] for x in manifest['audits']],
        detail='Generation/verification timeouts, unsupported bundles and retained cleanup unknowns are not successes. '
               'Do not replay spent rows or enlarge original comparison budgets.'),
        dict(id='query_and_selected_label_scope',local_disposition='design_and_instrumentation_review',
        detail='No unwanted probes; typed nonvacuity exists. Intended qualification does not relabel selected repairs. '
               'Original selected-label workers lack per-obligation native traces; retain this evidence limit.')]
    result = dict(schema='exact-repair/evaluation-final-report/v1',status='complete',
        manifest=binding(manifest_path),scope='local_preliminary_fresh_evaluation',
        local_reporting_complete=True,study_complete=False,campaign_complete=False,gates_passed=False,
        learning_efficiency_qualified=False,strongest_symbolic_comparison_qualified=False,
        branches=branches,bootstrap=BOOTSTRAP,scope_audit=scope,
        deferred_obligations=corrections,scope_amendment=manifest['scope_amendment'],
        attempts=attempts,attempt_costs=costs,attempt_totals=totals(costs),
        cumulative_costs=totals(ledger['attempts']),ledger_snapshot=manifest['ledger_snapshot'],
        cost_scope='Union of all primary, diagnostic, qualification and audit attempt charges, including failures. '
                   'All campaign maintenance remains in cumulative costs; this report worker and its preparation '
                   'settle separately. Row effort is descriptive and is not an additional ledger charge.',
        profile_timeouts_retained=primary['profile_timeouts_retained'],
        common_inventory_regret=None,regret_status=inventory['regret_status'],
        scientific_rows_replayed=0,new_native_calls=0,supervision_admitted=False,api_spend_usd=0,
        original_results_replaced=False,
        limitations=['Sparse conditional scores do not estimate unconditional semantic quality.',
            'Every unknown, unavailable, unsupported, partial and recovered row remains in its scheduled denominator.',
            'Family and seen/unseen strata describe this frozen construction cohort, not all ontologies.',
            'SAMPLED generation and support coverage do not imply exhaustive grammar or completed research gates.',
            'Historical heuristic controls do not establish strongest-symbolic superiority; no confirmatory tests.',
            'The preliminary amendment defers missing controls and larger training obligations; these are not completed.',
            'Selected-only fixed-inventory errors are conditional diagnostics, not population calibration or regret.'],
        remaining_action='Authenticate this worker completion, close only the local evaluation preparation slot, '
                         'and carry correction/design gaps into the registered consolidated preliminary report. '
                         'Continue all other eligible registered branches and preserve deduplicated idle notification.')
    output = Path(output);write_artifact(output/'report.json',result)
    write_artifact(output/'verified-files.json',evidence.files)
    lines = ['# Preliminary fresh-evaluation results','',
        'This report closes local evaluation reporting. It does not complete the larger scientific study or establish research gates.','',
        '| Evidence | Rows | Native-verified selected scores | Quality-usable rows |',
        '| --- | ---: | ---: | ---: |']
    for name,b in branches.items():
        lines.append(f"| {name} | {b['scheduled_rows']} | {b['native_verified_selected_scores']} | {b['quality_usable_rows']} |")
    lines += ['', 'All 64 cases, 32 paired parents, six model arms and three historical controls remain in each branch. '
              'The JSON report contains parent-group descriptive intervals, full failure denominators, per-arm family/exposure strata, '
              'paired control differences and the complete correction list.','',
              'Primary generated-pool quality and fixed-inventory selected-only diagnostics are separate. '
              'Unknown outcomes are not zero-imputed. Missing semantic/reference controls remain deferred design gaps.','',
              f"The report binds {len(attempts)} settled attempts, including failed ancestors. "
              'Preparation and the report worker remain separately charged in the cumulative campaign ledger.','']
    (output/'report.md').write_text('\n'.join(lines))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();report(args.manifest,args.output)
