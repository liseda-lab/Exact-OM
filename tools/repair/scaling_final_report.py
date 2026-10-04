"""Receipt-only final resource report; preserves separate scientific denominators."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair.overlap_audit import Evidence, binding, require
from tools.repair.scaling_audit import attempt, audit_rows, checkpoint, comparisons

METHODS = ('grammar_circuit', 'semantic_circuit', 'semantic_enumeration_decoder')


def quality_known(row):
    return (row['semantic_status'] == 'known' and row['logical_status'] == 'VERIFIED_FEASIBLE'
            and row['semantic_benefit'] is not None and row['fallback_original_inventory'] is False)


def summary(rows):
    """Never silently filter failures or equate process completion with coverage."""
    return dict(scheduled=len(rows), recorded=len(rows),
        process_statuses=dict(Counter(r['process_status'] for r in rows)),
        logical_statuses=dict(Counter(r['logical_status'] for r in rows)),
        generation_statuses=dict(Counter(r['generation_status'] for r in rows)),
        semantic_statuses=dict(Counter(r['semantic_status'] for r in rows)),
        generation_complete=sum(r['generation_complete'] for r in rows),
        generated_pool_selected_quality_known=sum(quality_known(r) for r in rows),
        complete_generation_quality_known=sum(quality_known(r) and r['generation_complete'] for r in rows),
        fallback_rows=sum(r['fallback_original_inventory'] is True for r in rows),
        row_wall_seconds=sum(r['elapsed_seconds'] for r in rows),
        quality_policy='Native-verified selected quality from a generated pool, including partial generation; '
                       'complete_generation_quality_known additionally requires completed generation.',
        effort_scope='Recorded row effort, including failures; not an additional ledger charge.')


def validate_schedule(schedule):
    require(len({r['id'] for r in schedule['rows']}) == len(schedule['rows']), 'Duplicate schedule row')
    grouped = defaultdict(list)
    for row in schedule['rows']:
        require((row['seconds'], row['cpu_seconds'], row['memory_mb']) == (300, 600, 8192),
                'Scientific row budget changed')
        grouped[row['case_index']].append((row['method'], row['cache_mode']))
    expected = {(method, mode) for method in METHODS for mode in ('cold', 'warm')}
    require(set(grouped) == set(range(len(schedule['cases']))), 'Case denominator differs')
    require(all(len(v) == 6 and set(v) == expected for v in grouped.values()),
            'Matched method/cache denominator differs')
    require(schedule['generation_seconds'] == 60, 'Generation allowance changed')


def incidence_pairs(schedule, rows):
    grouped = defaultdict(dict)
    for row in rows:
        case = schedule['cases'][row['row']['case_index']]
        key = (case['parent'], case['corrupted_mapping_count'], case['control'],
               row['row']['method'], row['row']['cache_mode'])
        condition = case['condition']
        require(condition in ('shared', 'nonshared') and condition not in grouped[key],
                'Duplicate or unexpected incidence condition')
        grouped[key][condition] = row
    pairs = []
    for key, pair in sorted(grouped.items()):
        require(set(pair) == {'shared', 'nonshared'}, 'Missing incidence pair member')
        shared, nonshared = pair['shared'], pair['nonshared']
        known = quality_known(shared) and quality_known(nonshared)
        pairs.append(dict(parent=key[0], corrupted_mapping_count=key[1], control=key[2],
            method=key[3], cache_mode=key[4], quality_available=known,
            both_generation_complete=shared['generation_complete'] and nonshared['generation_complete'],
            shared_minus_nonshared_benefit=(shared['semantic_benefit'] - nonshared['semantic_benefit'])
                if known else None,
            shared_minus_nonshared_seconds=shared['elapsed_seconds'] - nonshared['elapsed_seconds'],
            rows={c: r['row']['id'] for c, r in pair.items()}))
    return pairs


def semantic_evidence(evidence, row, item):
    """Bind known scores to the saved native-authorized selection and query list."""
    from exact.repair.records import read_record
    saved = checkpoint(evidence, row['receipt'])
    result = evidence.read(row['result']) if row.get('result') else {}
    scope = {key: result.get(key) for key in ('native_scope', 'verification_scope',
             'selection_status', 'semantic_detail', 'generation_detail', 'stage_seconds')}
    if row['semantic_status'] != 'known':
        return dict(known_score_verified=False, **scope)
    payloads = {Path(ref['path']).name: ref for ref in saved.get('payloads', [])}
    require('selected-label.json' in payloads and 'repair.json' in payloads,
            'Known score lacks saved selection evidence')
    label = evidence.read(payloads['selected-label.json'])
    repair = read_record(evidence.read(payloads['repair.json']))
    require(repair.logical_status == row['logical_status'] == 'VERIFIED_FEASIBLE' and
            repair.verification is not None and repair.verification.authorizes and
            label['feasible'] is True and list(repair.assignment) == label['assignment'],
            'Known score lacks native-authorized assignment')
    require(label['benefit'] == row['semantic_benefit'] and label['cost'] == row['edit_cost'],
            'Known score differs from selected label')
    queries = evidence.read(item['case'])['case']['probes']
    require([(p['probe_id'], p['family'], p['desired']) for p in label['semantic_vector']] ==
            [(p['probe_id'], p['family'], p['desired']) for p in queries],
            'Known score query denominator differs')
    return dict(known_score_verified=True, selected_label=payloads['selected-label.json'],
                repair=payloads['repair.json'], queries=len(queries), **scope)


def source_identity(evidence, ref, source, module):
    report = checkpoint(evidence, ref)
    batch = evidence.read(source['batch'])
    require(report['runtime'] == source['runtime'], 'Report runtime differs from frozen source')
    source_hash = batch['frozen_files'][str(Path(batch['code']) / module)]
    return report, source_hash


def incidence_qualification(evidence, manifest, sources):
    ref = manifest['incidence_qualification']
    report, module_hash = source_identity(evidence, ref, sources[manifest['qualification_run']],
                                         'tools/repair/scaling_incidence_evaluation.py')
    source = sources[manifest['qualification_run']]
    batch = evidence.read(source['batch'])
    scaling_hash = batch['frozen_files'][str(Path(batch['code'])/'tools/repair/scaling.py')]
    expected_identity = canonical_hash((report['schedule'], report['witness'], module_hash,
                                       scaling_hash, report['runtime']))
    require(report['identity'] == expected_identity and report['admitted'] and
            report['status'] == 'complete', 'Qualification source or admission differs')
    schedule = evidence.read(manifest['incidence_schedule'])
    require(report['schedule'] == manifest['incidence_schedule'], 'Qualification schedule differs')
    require(len(report['adapters']) == len(schedule['cases']) == 12, 'Adapter denominator differs')
    expected = {(i, m) for i in range(12) for m in METHODS}
    seen, statuses, complete = set(), Counter(), 0
    for ref in report['probes']:
        probe = checkpoint(evidence, ref)
        key = (probe['case_index'], probe['method'])
        require(key in expected and key not in seen, 'Probe denominator differs')
        require(probe['identity'] == canonical_hash((expected_identity, canonical_hash(key))),
                'Probe source identity differs')
        require(probe['cleanup_complete'], 'Probe cleanup unresolved')
        seen.add(key); statuses[probe['status']] += 1
        for payload in probe.get('payloads', []):
            evidence.verify(payload)
        pool = evidence.read(probe['pool']) if probe.get('pool') else None
        complete += bool(pool and pool['completed_objects'] == pool['scheduled_objects'] and
                         all(p['status'] == 'complete' for p in pool['reports']))
    require(seen == expected and report['scheduled'] == 36 and report['evaluation_rows'] == 72,
            'Qualification denominator missing')
    require(dict(statuses) == report['statuses'] and complete == report['complete_generation_probes'],
            'Qualification coverage differs')
    witness = checkpoint(evidence, report['witness'])
    require(witness['schedule'] == report['schedule'] and witness['scheduled'] ==
            witness['qualified'] == len(witness['rows']) == 12, 'Witness denominator differs')
    checks = 0
    for item, ref in zip(schedule['cases'], witness['rows'], strict=True):
        row = checkpoint(evidence, ref); result = row['result']
        require(row['case_id'] == item['case_id'] and row['cleanup_complete'] and
                row['status'] == 'complete' and result['qualified'] and
                result['graph_minimal_supports'] == item['expected_minimal_supports'],
                'Witness case differs')
        for check in result['native_checks']:
            evidence.verify(check['proof'])
            require(check['qualified'] and check['status'] == check['expected'], 'Unqualified witness')
            checks += 1
    return dict(scheduled_probes=36, statuses=dict(statuses), complete_generation_probes=complete,
                witness_cases=12, native_checks=checks, report=manifest['incidence_qualification'],
                witness=report['witness'], claim='Restricted witnesses and bounded generation coverage only')


def validate_lineage(sources):
    require(len({s['run']['id'] for s in sources}) == len(sources), 'Duplicate attempt charge')
    indexed = {s['run']['id']: s for s in sources}
    for source in sources:
        run = source['run']
        if run.get('superseded_by'):
            require(run['superseded_by'] in indexed, 'Missing replacement descendant')
            child = indexed[run['superseded_by']]['run']
            # Historical primary descriptors also use supersedes; preserve either recorded link.
            require(child.get('recovery_of', child.get('supersedes')) == run['id'],
                    'Replacement lineage differs')
    return indexed


def audit(manifest_path, output):
    evidence = Evidence(); manifest = evidence.read(binding(manifest_path))
    ledger = evidence.read(manifest['ledger_snapshot'])
    evidence.read(manifest['program'])
    require(evidence.read(manifest['authorization'])['limit_worker_seconds'] is None,
            'Authorization amendment missing')
    require(ledger['limit_worker_seconds'] is None, 'Time amendment missing')
    sources = [attempt(evidence, item, ledger, not item['run'].get('superseded_by'))
               for item in manifest['attempts']]
    indexed = validate_lineage(sources)
    result_sources = {str(Path(s['completion']['work'])/s['run']['result_relative']): s
                      for s in sources if s['run'].get('result_relative') and not s['run'].get('superseded_by')}
    for name in ('primary_audit', 'endpoint_report', 'incidence_qualification'):
        require(manifest[name]['path'] in result_sources, 'Unowned report: ' + name)
    primary = evidence.read(manifest['primary_audit'])
    endpoint = evidence.read(manifest['endpoint_report'])
    primary_manifest = evidence.read(primary['manifest'])
    endpoint_manifest = evidence.read(endpoint['manifest'])
    branches = {}
    for name, spec, expected in (('primary', primary_manifest, 240),
                                ('endpoint_amendment', endpoint_manifest, 24),
                                ('incidence', dict(schedule=manifest['incidence_schedule'],
                                                  reports=manifest['incidence_reports']), 72)):
        schedule = evidence.read(spec['schedule'])
        require(len(schedule['rows']) == expected, 'Scientific denominator differs')
        validate_schedule(schedule)
        for case in schedule['cases']:
            value = evidence.read(case['case'])
            require(value['case']['split'] == 'development', 'Non-development scaling case')
        rows = audit_rows(evidence, schedule, spec['reports'])
        if name != 'incidence':
            original = primary if name == 'primary' else endpoint
            require(rows == original['rows'], 'Published aggregate differs from original receipts')
        else:
            for ref in spec['reports']:
                require(ref['path'] in result_sources, 'Incidence shard has no terminal owner')
                report, source_hash = source_identity(evidence, ref, result_sources[ref['path']],
                                                       'tools/repair/scaling.py')
                identity = canonical_hash((report['schedule'], source_hash, report['runtime']))
                require(report['identity'] == identity, 'Incidence scientific source differs')
                for row_ref in report['rows']:
                    saved = checkpoint(evidence, row_ref)
                    require(saved['identity'] == canonical_hash((identity, saved['row'])),
                            'Incidence row scientific identity differs')
            ancestry = evidence.read(schedule['ancestry_audit'])
            require(len({c['parent'] for c in schedule['cases']}) == 1 and
                    ancestry['independent_new_parents'] == 0 and ancestry['inherited_split'] == 'development' and
                    {c['parent'] for c in schedule['cases']} == {ancestry['inherited_parent']}, 'Incidence ancestry inflated')
        for row in rows:
            row['semantic_evidence'] = semantic_evidence(evidence, row,
                schedule['cases'][row['row']['case_index']])
        languages, caches = comparisons(rows)
        for pair in languages:
            # An empty intersection of partial objects is never proof of matched language.
            pair['qualified_complete_language_match'] = bool(pair['both_generation_complete'] and
                pair['compared_objects'] and not pair['missing_objects'] and not pair['mismatches'])
        groups = defaultdict(list)
        for row in rows:
            groups[(row['parent'],row['config']['id'],row['control'],
                    row['row']['method'],row['row']['cache_mode'])].append(row)
        branches[name] = dict(schedule=spec['schedule'], summary=summary(rows), rows=rows,
            language_comparisons=languages, cache_comparisons=caches,
            grouped=[dict(parent=k[0], configuration=k[1], control=k[2], method=k[3],
                          cache_mode=k[4], **summary(v)) for k,v in sorted(groups.items())])
        if name != 'incidence':
            branches[name]['original_report'] = manifest['primary_audit' if name == 'primary' else 'endpoint_report']
            branches[name]['original_report_quality_count'] = original['generated_quality_known']
            branches[name]['quality_count_note'] = 'Original count preserved; revised selected-quality count explicitly includes partial generated pools.'
        if name == 'incidence':
            pairs = incidence_pairs(schedule, rows)
            require(len(pairs) == 36, 'Incidence paired denominator differs')
            branches[name].update(pairs=pairs, ancestry= schedule['ancestry_audit'],
                independent_parent_count=1, clean_controls_load_matched_to_corrupted=False)
    parent_sets = {name: {r['parent'] for r in branch['rows']} for name, branch in branches.items()}
    require(len(parent_sets['primary']) == 2 and parent_sets['endpoint_amendment'] == parent_sets['primary']
            and parent_sets['incidence'] <= parent_sets['primary'], 'Inherited development ancestry differs')
    qualification = incidence_qualification(evidence, manifest, indexed)
    issue = evidence.read(manifest['primary_issue'])
    errors = [r for r in branches['primary']['rows'] if r['generation_status'] == 'error']
    require(len(errors) == issue['receipt_validation']['affected_receipts_sha256_verified'] == 120,
            'Original error denominator differs')
    require(all(r['control'] == 'coherent' and r['fallback_original_inventory'] is True for r in errors),
            'Original fallback scope differs')
    charges = {str(Path(s['run']['completion_path']).parent): s['charge'] for s in sources}
    scaling_costs = dict(attempts=len(charges), worker_seconds=sum(c['elapsed_seconds'] for c in charges.values()),
        allocated_cpu_seconds=sum(c['elapsed_seconds']*c['resources']['cpus'] for c in charges.values()),
        allocated_gpu_seconds=sum(c['elapsed_seconds']*c['resources']['gpus'] for c in charges.values()),
        measured_cpu_seconds=sum(c.get('cpu_seconds') or 0 for c in charges.values()),
        allocated_memory_mb_seconds=sum(c['elapsed_seconds']*c['resources']['memory_mb'] for c in charges.values()))
    result = dict(schema='exact-repair/scaling-final-report/v1', status='complete',
        manifest=binding(manifest_path), branches=branches, scheduled=336, recorded=336,
        qualification=qualification, attempts=sources, scaling_attempt_costs=scaling_costs,
        exposed_development_parent_count=2, independent_new_parents=0,
        cost_snapshot=manifest['ledger_snapshot'], cumulative_costs_at_snapshot=ledger['cumulative'],
        cost_scope='All recorded scaling validation/science/recovery attempts once; cumulative campaign '
                   'ledger also retains maintenance and active reservations. Final reporting settles separately.',
        primary_endpoint_disposition=dict(status='accounted_generation_unavailable', affected_rows=120,
            audit=manifest['primary_issue'], original_results_preserved=True, original_rows_replayed=0,
            retrospective_repair=False, endpoint_amendment_is_separate=True,
            unresolved_capability='Complete composite-bundle endpoint substitution; original comparison unestablished'),
        depth3_decision=primary['depth3_decision'],
        scaling_stage_accounted=True, campaign_complete=False, gates_passed=False, api_spend_usd=0,
        limitations=['Two historically exposed primary development parents; incidence is one inherited ancestry, '
                     'not additional independent parents. No held-out or population inference.',
                     'All process failures, generation errors, fallback, partial, unknown and unavailable rows '
                     'remain in their separate 240/72/24 denominators. Do not pool branch effects.',
                     'Reported semantic values retain the original query and native scope; receipt verification '
                     'does not expand intended-parent/query qualification or establish general OWL support.',
                     'No strongest-symbolic, G0-G2 or learning-efficiency claim; no fitting or new native calls.',
                     'Clean incidence controls contain half as many mapping objects as corrupt cases. '
                     'Shared/nonshared comparisons are within control and size; one ancestry supports no confidence interval.',
                     'Candidate caps can be nonbinding; inspect per-object sampled_unique, supplied, pool_size, '
                     'effective_menu, family status and context truncation retained with every row.'],
        verified_files=len(evidence.files))
    output=Path(output)
    write_artifact(output/'report.json', result)
    write_artifact(output/'verified-files.json', evidence.files)
    lines=['# XR-2.1 bounded scaling report', '',
           'All 336 scheduled rows are retained in three separate development experiments.', '',
           '| Branch | Rows | Complete generation | Generated quality known | Native unknown |',
           '|---|---:|---:|---:|---:|']
    for name, branch in branches.items():
        s=branch['summary'];lines.append(f"| {name} | {s['scheduled']} | {s['generation_complete']} | "
            f"{s['generated_pool_selected_quality_known']} | {s['logical_statuses'].get('UNKNOWN',0)} |")
    lines += ['', 'The original 120 composite-control endpoint errors remain generation-unavailable; '
              'the 24-row endpoint-disabled amendment does not repair the original comparisons.', '',
              'Incidence qualification retained 36 probes and 12 witnessed cases. Paired rows, generation '
              'telemetry, cache/language checks, source/nonce receipts and all costs are in report.json.', '',
              *[f'- {x}' for x in result['limitations']]]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();audit(args.manifest,args.output)
