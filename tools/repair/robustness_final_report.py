"""Receipt-bound preliminary robustness report; no inference or scientific replay."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

from exact.repair.api import write_artifact
from tools.repair.evaluation_final_report import (
    BOOTSTRAP, CONTROL_IDS, arm_report, authenticate_audit, bootstrap_clusters,
    generated_known, summary,
)
from tools.repair.fresh_evaluation_audit import validate_attempt
from tools.repair.overlap_audit import Evidence, binding, require
from tools.repair.report_campaign import totals
from tools.repair.robustness_audit import CONDITIONS

SCHEMA = 'exact-repair/robustness-final-manifest/v1'


def validate_rows(rows, schedule):
    expected = {r['id']: r for r in schedule['rows']}
    require(len(rows) == len(expected) == 2880, 'Robustness denominator differs')
    require({r['row']['id'] for r in rows} == set(expected), 'Missing or duplicate row')
    require(len(schedule['cases']) == 320 and len(schedule['arms']) == 9,
            'Case or arm denominator differs')
    require(sum(a['kind'] == 'learned' for a in schedule['arms']) == 6 and
            {a['id'] for a in schedule['arms'] if a['kind'] == 'control'} == set(CONTROL_IDS),
            'Model/control denominator differs')
    parents = defaultdict(list)
    for case in schedule['cases']:
        parents[case['group_id']].append((case['condition'], case['control']))
    require(len(parents) == 32 and all(Counter(v) == Counter(
        (c, control) for c in CONDITIONS for control in ('coherent', 'corrupted'))
        for v in parents.values()), 'Parent/condition pairing differs')
    for row in rows:
        require(row['row'] == expected[row['row']['id']], 'Row schedule differs')
        case = schedule['cases'][row['row']['case_index']]
        fields = ('base_case_id', 'group_id', 'family', 'family_exposure', 'control', 'condition')
        require(all(row[k] == case[k] for k in fields), 'Case ancestry or stratum differs')
        require(row['row']['case_id'] == case['case_id'], 'Case identity differs')
    return sorted(a['id'] for a in schedule['arms'])


def paired_intervention(rows, arm, condition):
    """Keep both variants in their inherited parent; never impute missing quality."""
    require(condition in CONDITIONS and condition != 'baseline', 'Unknown intervention')
    selected = [r for r in rows if r['row']['arm_id'] == arm and
                r['condition'] in ('baseline', condition)]
    indexed = {(r['condition'], r['base_case_id']): r for r in selected}
    require(len(indexed) == len(selected), 'Duplicate paired row')
    cases = {r['base_case_id'] for r in selected if r['condition'] == 'baseline'}
    require(cases and cases == {r['base_case_id'] for r in selected if r['condition'] == condition},
            'Missing intervention pair')
    clusters = {k: defaultdict(list) for k in ('coverage_difference', 'conditional_benefit_difference',
        'conditional_utility_difference', 'wall_seconds_difference', 'measured_cpu_seconds_difference')}
    regimes = Counter(); applicable = 0; noops = 0; same_source = 0; both_known = 0
    for case in sorted(cases):
        left, right = indexed[condition, case], indexed['baseline', case]
        require(all(left[k] == right[k] for k in ('group_id', 'family', 'family_exposure', 'control')),
                'Paired ancestry differs')
        parent = left['group_id']; both = generated_known(left) and generated_known(right)
        both_known += both
        regimes[left['source_regime'] + ' minus ' + right['source_regime']] += 1
        same_source += left['source_regime'] == right['source_regime']
        applicable += bool(left['intervention_scope']['applicable'])
        noops += not bool(left['intervention_scope']['applicable'])
        clusters['coverage_difference'][parent].append(int(generated_known(left))-int(generated_known(right)))
        clusters['conditional_benefit_difference'][parent].append(
            left['semantic_benefit']-right['semantic_benefit'] if both else None)
        clusters['conditional_utility_difference'][parent].append(
            left['semantic_benefit']-left['edit_cost']-right['semantic_benefit']+right['edit_cost'] if both else None)
        clusters['wall_seconds_difference'][parent].append(left['elapsed_seconds']-right['elapsed_seconds'])
        cpu = [r.get('resources', {}).get('cpu_seconds') for r in (left, right)]
        clusters['measured_cpu_seconds_difference'][parent].append(
            cpu[0]-cpu[1] if all(v is not None for v in cpu) else None)
    return dict(arm_id=arm, condition=condition, direction='intervention minus charged baseline',
        scheduled_pairs=len(cases), quality_usable_pairs=both_known, applicable_pairs=applicable,
        no_op_pairs=noops, same_scientific_source_pairs=same_source,
        different_scientific_source_pairs=len(cases)-same_source, source_pairs=dict(regimes),
        metrics={k: bootstrap_clusters(v) for k, v in clusters.items()},
        interpretation='Descriptive paired executed outcomes; source revisions can confound intervention effects. '
                       'No-op and unavailable rows remain included; no causal or confirmatory superiority claim.')


def union_costs(reports, owners, ledger):
    """Use immutable attempt keys, including overlap, with explicit conflict checks."""
    costs = {}
    def add(key, charge):
        require(ledger['attempts'].get(key) == charge and charge['status'] == 'settled',
                'Missing or changed attempt charge')
        require(key not in costs or costs[key] == charge, 'Conflicting shared attempt charge')
        costs[key] = charge
    for report in reports.values():
        for key, charge in report.get('attempt_costs', {}).items():
            add(key, charge)
        # The scope audit carries charges inside its attempt records.
        for source in report.get('attempts', []):
            completion_path = (source['run']['completion_path'] if 'run' in source
                               else source['completion']['path'])
            add(str(Path(completion_path).parent), source['charge'])
    for source in owners:
        add(str(Path(source['run']['completion_path']).parent), source['charge'])
    return costs


def authenticate_overlap(evidence, item, ledger, snapshot):
    require(next(r for r in snapshot['runs'] if r['id'] == item['run']['id']) == item['run'],
            'Overlap owner changed')
    owner = validate_attempt(evidence, item, ledger, True)
    outputs = evidence.read(item['outputs']); work = Path(owner['completion']['work'])
    require(outputs.get(str(Path(item['report']['path']).relative_to(work))) == item['report']['sha256'],
            'Overlap report lacks worker output binding')
    for relative, digest in outputs.items():
        evidence.verify(dict(path=str(work/relative), sha256=digest))
    report = evidence.read(item['report'])
    require(report['status'] == 'complete' and report['study_complete'] and not report['gate_passage'] and
            report['scheduled_rows'] == report['recorded_rows'] == 36 and
            report['independent_parent_count'] == 1, 'Overlap scope differs')
    for path, digest in report['verified_files'].items():
        if path not in evidence.files:
            evidence.verify(dict(path=path, sha256=digest))
        else:
            require(evidence.files[path] == digest, 'Conflicting overlap evidence')
    return report, owner


def validate_tests(evidence, item, owner, minimum):
    outputs = evidence.read(item['outputs'])
    require('validation.xml' in outputs, 'Audit validation absent')
    path = evidence.verify(dict(path=str(Path(owner['completion']['work'])/'validation.xml'),
                                sha256=outputs['validation.xml']))
    counts = {k: sum(int(s.get(k, 0)) for s in ET.parse(path).getroot().iter('testsuite'))
              for k in ('tests', 'failures', 'errors', 'skipped')}
    require(counts['tests'] >= minimum and all(counts[k] == 0 for k in ('failures','errors','skipped')),
            'Audit integrity tests did not pass')
    return counts


def report(manifest_path, output):
    evidence = Evidence(); manifest = evidence.read(binding(manifest_path))
    require(manifest['schema'] == SCHEMA and manifest['bootstrap'] == BOOTSTRAP and
            manifest['scientific_rows_replayed'] == manifest['native_calls'] == 0 and
            manifest['supervision_admitted'] is False, 'Final report boundary differs')
    amendment = evidence.read(manifest['scope_amendment'])
    require(amendment['schema'] == 'exact-repair/preliminary-scope-amendment/v1' and
            not amendment['original_scientific_budgets_changed'], 'Scope amendment differs')
    evidence.read(manifest['program'])
    require(evidence.read(manifest['authorization'])['limit_worker_seconds'] is None,
            'Time amendment missing')
    ledger = evidence.read(manifest['ledger_snapshot']); snapshot = evidence.read(manifest['registry_snapshot'])
    require(ledger['limit_worker_seconds'] is None, 'Campaign limit changed')
    reports = {}; owners = []; validation = {}
    for item in manifest['audits']:
        require(item['kind'] not in reports, 'Duplicate audit branch')
        authenticate = authenticate_overlap if item['kind'] == 'overlap' else authenticate_audit
        reports[item['kind']], owner = authenticate(evidence, item, ledger, snapshot)
        owners.append(owner)
        if item['kind'] == 'robustness':
            validation[item['kind']] = validate_tests(evidence, item, owner, 44)
        print('Authenticated '+item['kind']+' worker and '+str(len(evidence.files))+' files', flush=True)
    require(set(reports) == {'robustness', 'scope', 'overlap'}, 'Missing completed branch')
    robust, scope, overlap = (reports[k] for k in ('robustness','scope','overlap'))
    schedule = evidence.read(manifest['schedule'])
    require(evidence.read(robust['manifest'])['schedule'] == manifest['schedule'], 'Audit schedule differs')
    rows = robust['rows']; arms = validate_rows(rows, schedule)
    require(len(scope['rows']) == 64 and {r['case_id'] for r in scope['rows']} ==
            {r['base_case_id'] for r in rows}, 'Intended scope denominator differs')
    costs = union_costs(reports, owners, ledger)
    by_condition = {}
    for condition in CONDITIONS:
        subset = [r for r in rows if r['condition'] == condition]
        by_condition[condition] = dict(**summary(subset, True),
            by_arm={a: arm_report([r for r in subset if r['row']['arm_id'] == a], True) for a in arms})
    pairs = [paired_intervention(rows, a, c) for a in arms for c in CONDITIONS if c != 'baseline']
    strata = {field: {str(value): {c: dict(**summary([r for r in rows if r[field] == value and
        r['condition'] == c], True), by_arm={a: summary([r for r in rows if r[field] == value and
        r['condition'] == c and r['row']['arm_id'] == a], True) for a in arms}) for c in CONDITIONS}
        for value in sorted({r[field] for r in rows})} for field in ('family','family_exposure','control')}
    corrections = [dict(item, local_disposition='deferred_correction_or_design_gap',
        authority=manifest['scope_amendment']) for item in scope['control_review']['required_comparisons']
        if item['status'] not in ('diagnostics_already_audited_with_limits','disabled_by_authorization')]
    corrections += [
        dict(id='robustness_common_inventory', status='deferred_design_gap',
             detail='Fresh fixed-inventory diagnostics do not replace robustness generated-pool comparisons; '
                    'no additional robustness diagnostic was run under the preliminary scope.'),
        dict(id='model_schema', status='correction_required_before_cluster',
             rows=robust['scientific_statuses'].get('unavailable_model_schema',0)),
        dict(id='native_lifecycle', status='unresolved_root_cause',
             detail='Four cleanup rows remain unknown; both slice24 errors and all ancestors are retained.'),
        dict(id='omission_and_final_removal', status='focused_corrections_executed_original_unknowns_retained',
             detail='One omission and one removal failure remain unknown. Executed source regimes differ across pairs.'),
        dict(id='query_and_menu_evidence', status='design_and_instrumentation_review',
             detail='No unwanted probes, no full effective-menu snapshots, and no per-obligation selected-label traces. '
                    'Intended qualification does not relabel selected repairs.'),
        dict(id='generation_resource_scope', status='resource_and_support_design_review',
             detail='Timeouts, unsupported bundles and sampled pools do not qualify exhaustive language coverage.')]
    result = dict(schema='exact-repair/robustness-final-report/v1', status='complete', manifest=binding(manifest_path),
        scope='local_preliminary_robustness', local_reporting_complete=True, study_complete=False,
        campaign_complete=False, gates_passed=False, learning_efficiency_qualified=False,
        strongest_symbolic_comparison_qualified=False, **summary(rows, True),
        base_cases=64, case_variants=320, learned_rows=1920, control_rows=960,
        rows=rows, by_condition=by_condition, paired_interventions=pairs, strata=strata, bootstrap=BOOTSTRAP,
        intervention_coverage=robust['intervention_coverage'], source_regimes=robust['source_regimes'],
        receipt_source_regimes=dict(Counter(r['receipt_source_regime'] for r in rows)),
        source_strata={s: summary([r for r in rows if r['source_regime'] == s], True)
                       for s in sorted({r['source_regime'] for r in rows})},
        scope_audit=scope, overlap=overlap, audited_branch_reports={i['kind']: i['report'] for i in manifest['audits']},
        prior_attempt_lineage={k: v.get('attempts', v.get('attempt_lineage', [])) for k,v in reports.items()},
        audit_workers=owners, validation=validation, attempt_costs=costs, attempt_totals=totals(costs),
        cumulative_costs=totals(ledger['attempts']), ledger_snapshot=manifest['ledger_snapshot'],
        maintenance_costs={k:v for k,v in ledger['attempts'].items() if
            'robustness' in v['logical_id'] and ('maintenance' in k or 'preparation' in v['logical_id'])},
        cost_scope='Union of prior robustness, overlap, scope and audit attempts, including shared validation once. '
                   'All maintenance remains in cumulative costs. This preparation and final worker settle separately. '
                   'Row effort excludes shared materialization overhead and is descriptive, not a second charge.',
        profile_timeouts_retained=robust['profile_timeouts_retained'], deferred_obligations=corrections,
        scope_amendment=manifest['scope_amendment'], scientific_rows_replayed=0, new_native_calls=0,
        supervision_admitted=False, api_spend_usd=0, original_results_replaced=False,
        limitations=robust['limitations'] + [
            'Conditional scores do not estimate unconditional semantic quality; missing quality is not zero-imputed.',
            'Source revisions can confound perturbation effects; source pairs accompany each paired estimate.',
            'Overlap is a separate single exposed diagnostic ancestry, with 36 rows and no population interval.',
            'Family/exposure strata describe only these frozen constructions. No confirmatory or learning claim.'],
        remaining_action='Authenticate the actual final worker receipt and close only local robustness reporting. '
            'Carry every correction, deferral, original attempt and cost into the registered consolidated preliminary '
            'report. Preserve remaining_work_status pending until that scope accounting is complete; existing controller '
            'owns the eventual deduplicated idle notification.', verified_file_count=len(evidence.files))
    output = Path(output); write_artifact(output/'report.json', result)
    write_artifact(output/'verified-files.json', evidence.files)
    lines = ['# Preliminary robustness results', '',
        'All 2,880 evidence/omission rows, 320 variants, 64 base cases and 32 parent groups remain represented. '
        'This closes local reporting only; missing controls and larger-study obligations remain deferred.', '',
        '| Condition | Scheduled | Quality usable | Unavailable quality |', '| --- | ---: | ---: | ---: |']
    for c, value in by_condition.items():
        lines.append(f"| {c} | {value['scheduled_rows']} | {value['quality_usable_rows']} | {value['quality_unavailable_rows']} |")
    lines += ['', 'The JSON contains 36 within-arm perturbation/baseline comparisons with 2,000 parent-group '
        'bootstrap draws (seed 20261003), descriptive 95% intervals, usable/scheduled denominators and source pairs. '
        'Missing quality is not imputed. Source corrections can confound intervention differences.', '',
        'The separately audited overlap branch retains 36 rows from one exposed diagnostic ancestry; it has no '
        'population interval. Intended-parent qualification does not relabel selected repairs. No gate, '
        'strongest-symbolic, learning-efficiency or confirmatory superiority claim follows.', '',
        f"The immutable union contains {len(costs)} prior settled attempt charges. Preparation and the final worker "
        'settle separately; all campaign maintenance and historical costs remain recorded.', '']
    (output/'report.md').write_text('\n'.join(lines))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(); report(args.manifest, args.output)
