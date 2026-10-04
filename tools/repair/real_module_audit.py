"""Close the real-projection stage using frozen evidence, with no new scientific calls."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from exact.experiments.science_health import software_failure
from exact.repair.records import canonical_hash
from tools.repair.overlap_audit import Evidence, binding, receipt, require
from tools.repair.report_campaign import totals


def validate_scope(schedule, preparation, report):
    pair = preparation['pair']
    require(preparation['independent_pair_count'] == schedule['independent_pair_count'] ==
            report['independent_pair_count'] == 1, 'Projection siblings are not independent pairs')
    require(pair['split'] == report['inherited_split'] == 'train', 'Inherited training split differs')
    require(len(schedule['cases']) == 6 and len(schedule['arms']) == 9,
            'Case/arm denominator differs')
    require(len({a['id'] for a in schedule['arms']}) == 9, 'Duplicate arm')
    for case in schedule['cases']:
        require(case['group_id'] == case['structural_parent'] == pair['group_id'] and
                case['split'] == pair['split'] and
                case['family_exposure'] == 'historical_train_pair_exploratory',
                'Inherited pair, exposure or split differs')
    require(report['scope'] == preparation['scope'] == schedule['scope'] and
            preparation['source_coherence_qualified'] is False and
            schedule['source_coherence_qualified'] is False and report['gates_passed'] is False,
            'Projection evidence cannot qualify full-source reasoning or research gates')
    require(report['matcher_inputs_consumed'] == preparation['matcher_inputs_consumed'] == 0 and
            report['labels_used_for_fitting'] == preparation['labels_used_for_fitting'] == 0,
            'Matcher or fitting scope changed')
    require(report['control_definition'] == schedule['control_definition'], 'Control scope differs')


def validate_qualification(saved):
    require(saved['cleanup_complete'], 'Unresolved qualification cleanup')
    require(saved['status'] not in {'error', 'worker_error'},
            'Native software error: ' + saved.get('detail', ''))
    value = saved.get('value') or {}
    native = value.get('report') or {}
    support = native.get('support') or {}
    known = (saved['status'] == 'complete' and
             native.get('logical_status') == 'VERIFIED_FEASIBLE' and
             native.get('verification_scope') == 'complete_supported_fragment' and
             support.get('input_supported') is True and support.get('complete_imports') is True and
             bool(native.get('obligations')) and
             all(o['complete'] and o['verdict'] == o['expected'] for o in native['obligations']))
    require(bool(value.get('authorizes')) == known, 'Qualification authorization differs from native evidence')
    return known


def validate_preparation(evidence, manifest, source, preparation, schedule, report):
    plan = evidence.read(manifest['preparation_plan'])
    job = next(j for j in source['batch']['jobs'] if j['id'] == source['run']['logical_id'])
    require(any(manifest['preparation_plan']['path'] in c for c in job['commands']),
            'Preparation plan lacks executed source binding')
    dependencies = {name: source['batch']['frozen_files'][str(Path(source['batch']['code']) /
                    'tools/repair' / name)] for name in
                    ('real_modules.py', 'expanded_corpus.py', 'expanded_profile.py',
                     'prepare.py', 'fresh_evaluation.py', 'corpus.py', 'batch.py')}
    identity = canonical_hash((plan, dependencies, source['runtime']))
    complete = evidence.checkpoint(manifest['preparation_completion'], identity)
    require(complete['dependencies'] == dependencies and complete['runtime'] == source['runtime'] and
            preparation['identity'] == identity and complete['schedule'] == report['schedule'],
            'Preparation source, runtime or schedule differs')
    refs = {Path(ref['path']).name: ref for ref in complete['artifacts']}
    for ref in complete['artifacts']:
        evidence.verify(ref)
    provenance, splits = evidence.read(plan['provenance']), evidence.read(plan['splits'])
    pair = next(p for p in provenance['pairs'] if p['id'] == plan['pair_id'])
    split = next(p for p in splits['pairs'] if p['id'] == plan['pair_id'])
    require(pair == preparation['pair'] and plan['splits'] == preparation['split_inheritance'] ==
            schedule['split_schedule'], 'Pair provenance or split source differs')
    for key in ('split', 'group_id', 'cohort', 'ontology_assets', 'ontology_names'):
        require(pair[key] == split[key], 'Frozen ontology-pair split differs')
    assets = []
    for identifier in pair['ontology_assets']:
        record = next(a for a in provenance['assets'] if a['asset']['id'] == identifier)
        asset, lineage = record['asset'], record['lineage']
        require(asset['role'] == 'ontology' and asset['requires_real_matcher_inputs'] is False and
                asset['revision'] == 'c454644a334ab43754bc1070c0fb7fdd56a90a1d' and
                lineage['license_status'] == 'captured_public_terms' and
                lineage['release_links'] and lineage['license_claims'], 'Unqualified source release/license')
        evidence.verify(asset)
        for link in lineage['release_links'] + lineage['license_claims']:
            evidence.verify(dict(path=link['source'], sha256=link['sha256']))
        assets.append(asset)
        extraction = refs[canonical_hash(asset) + '.json']
        saved = evidence.checkpoint(extraction, canonical_hash((identity, asset)))
        require(saved['cleanup_complete'] and saved['status'] == 'complete', 'Extraction incomplete')
    require(assets == preparation['assets'], 'Pinned source assets differ')
    qualifications, calls = [], []
    for ordinal in range(plan['parents']):
        results = []
        for scope in ('source', 'target', 'union', 'intended'):
            ref = refs[f'{ordinal}-{scope}.json']
            saved = evidence.checkpoint(ref, canonical_hash((identity, ordinal, scope)))
            known = validate_qualification(saved)
            results.append(known)
            calls.append(dict(parent=ordinal, scope=scope, receipt=ref, qualified=known,
                              status=saved['status'], detail=saved.get('detail', ''),
                              native_report=(saved.get('value') or {}).get('report')))
        qualified = all(results)
        qualifications.append(dict(parent=ordinal, qualified=qualified,
                                   status='qualified' if qualified else 'parent_qualification_unknown_or_failed'))
    require(qualifications == preparation['qualifications'] == complete['qualifications'] ==
            report['qualifications'], 'Qualification summary differs')
    require(plan['parents'] == report['projection_parents'] == 2 and
            complete['scheduled_cases'] == 6 and complete['scheduled_rows'] == 54,
            'Preparation denominator differs')
    require(evidence.read(plan['model_schedule'])['arms'] == schedule['arms'], 'Pilot arms changed')
    for arm in schedule['arms']:
        for key in ('model', 'protocol', 'completion', 'training_report'):
            if key in arm:
                evidence.verify(arm[key])
    for case in schedule['cases']:
        for key in ('observable', 'evaluator'):
            if key in case:
                evidence.verify(case[key])
    evidence.read(plan['program'])
    require(evidence.read(plan['authorization'])['limit_worker_seconds'] is None,
            'Unlimited authorization missing')
    return calls


def validate_row_outcome(saved, result):
    for status, detail in ((saved.get("status"), saved.get("detail", "")),
                           (result.get("status"), result.get("detail", "")),
                           (result.get("semantic_status"), result.get("semantic_detail", ""))):
        require(not software_failure(status, detail),
                "Scientific software error: " + (detail or str(status)))
    return result.get("status", saved["status"])


def validate_rows(evidence, schedule, report, sources, plan):
    expected = {r['id']: r for r in schedule['rows']}
    require(len(expected) == len(schedule['rows']) == 54, 'Schedule denominator differs')
    require({(r['arm_id'], r['case_index']) for r in expected.values()} ==
            {(a['id'], i) for a in schedule['arms'] for i in range(6)}, 'Arm/case cross product differs')
    rows, seen = [], set()
    for execution, shard in zip(report['executions'], plan['shards'], strict=True):
        source = sources[execution['run_id']]
        require(execution['completion'] == binding(source['run']['completion_path']) and
                execution['step_id'] == source['run']['step_id'] and
                execution['nonce'] == source['run']['dispatch_nonce'], 'Report execution owner differs')
        ref = execution['report']
        saved_report = evidence.read(ref)
        require(saved_report['runtime'] == source['runtime'] and saved_report['schedule'] ==
                report['schedule'] and saved_report['slice'] == shard['slice'], 'Shard provenance differs')
        code = str(Path(source['batch']['code']) / 'tools/repair/fresh_evaluation.py')
        identity = canonical_hash((report['schedule']['sha256'],
                                   source['batch']['frozen_files'][code], source['runtime']))
        evidence.checkpoint(ref, identity)
        require(saved_report['status'] == 'complete' and saved_report['scheduled'] ==
                saved_report['recorded'] == len(saved_report['rows']), 'Incomplete shard')
        actual_ids = []
        for row_ref in saved_report['rows']:
            saved = evidence.read(row_ref)
            row = saved['row']
            require(row['id'] not in seen and expected.get(row['id']) == row and
                    row_ref['row_id'] == row['id'], 'Duplicate or unexpected row')
            require((row['seconds'], row['cpu_seconds'], row['memory_mb']) == (300, 600, 8192),
                    'Scientific comparison budget changed')
            evidence.checkpoint(row_ref, canonical_hash((identity, row)))
            require(saved['cleanup_complete'] and saved['status'] == row_ref['status'],
                    'Unresolved row cleanup/status')
            for payload in saved.get('payloads', []):
                evidence.verify(payload)
            result = evidence.read(saved['result']) if saved.get('result') else {}
            require(not result or (result['row_id'] == row['id'] and
                    result['schedule_hash'] == canonical_hash(schedule)), 'Result provenance differs')
            outcome = validate_row_outcome(saved, result)
            guard = Path(row_ref['path']).parent.parent / 'inflight' / (row['id'] + '.json')
            require(not guard.exists(), 'Unresolved inflight row ownership')
            case = schedule['cases'][row['case_index']]
            rows.append(dict(row_id=row['id'], arm_id=row['arm_id'], case_id=case['case_id'],
                             condition=case['condition'], projection_parent=row['case_index'] // 3,
                             inherited_pair=case['group_id'], split=case['split'],
                             source_status=case['status'], status=saved['status'], outcome=outcome,
                             logical_status=result.get('logical_status', 'UNKNOWN'),
                             semantic_status=result.get('semantic_status', 'unavailable'),
                             semantic_benefit=result.get('semantic_benefit'), edit_cost=result.get('edit_cost'),
                             elapsed_seconds=saved['elapsed_seconds'], resources=saved['resources'],
                             receipt=row_ref, result=saved.get('result')))
            seen.add(row['id'])
            actual_ids.append(row['id'])
        start, stop = shard['slice']
        require(actual_ids == [r['id'] for r in schedule['rows'][start:stop]], 'Shard row coverage differs')
    require(seen == set(expected), 'Incomplete row denominator')
    require(rows == report['rows'] and report['scheduled_rows'] == report['recorded_rows'] == 54,
            'Original report differs from row evidence')
    require(report['outcomes'] == dict(Counter(r['outcome'] for r in rows)), 'Outcome counts differ')
    summaries = []
    for arm in schedule['arms']:
        selected = [r for r in rows if r['arm_id'] == arm['id']]
        known = [r['semantic_benefit'] for r in selected if r['logical_status'] == 'VERIFIED_FEASIBLE'
                 and r['semantic_status'] == 'known' and r['semantic_benefit'] is not None]
        summaries.append(dict(arm_id=arm['id'], scheduled=6, recorded=len(selected),
                              outcomes=dict(Counter(r['outcome'] for r in selected)),
                              known_semantic_rows=len(known),
                              mean_semantic_benefit=sum(known) / len(known) if known else None))
    require(summaries == report['arm_summaries'], 'Failure-inclusive arm summaries differ')
    return rows


def audit(manifest_path):
    evidence = Evidence()
    manifest = evidence.read(binding(manifest_path))
    report = evidence.read(manifest['report'])
    plan = evidence.read(report['plan'])
    schedule = evidence.read(report['schedule'])
    preparation = evidence.read(report['preparation'])
    ledger = evidence.read(manifest['ledger_snapshot'])
    require(ledger['limit_worker_seconds'] is None, 'Campaign authorization missing')
    sources = [receipt(evidence, item, ledger) for item in manifest['attempts']]
    by_id = {s['run']['id']: s for s in sources}
    require(len(by_id) == len(sources) == 8, 'Expected preparation, six evaluations and report')
    require(all(not s['run'].get('superseded_by') for s in sources), 'Unreviewed replacement lineage')
    paths = {str(Path(s['completion']['work']) / s['run']['result_relative']): s for s in sources}
    require(manifest['report']['path'] in paths, 'Final report lacks active completion')
    validate_scope(schedule, preparation, report)
    calls = validate_preparation(evidence, manifest, paths[manifest['preparation_completion']['path']],
                                 preparation, schedule, report)
    rows = validate_rows(evidence, schedule, report, by_id, plan)
    require(report['status'] == 'complete' and report['study_complete'], 'Original report incomplete')
    costs = {str(Path(s['run']['completion_path']).parent): s['charge'] for s in sources}
    related = {k: v for k, v in ledger['attempts'].items() if v['logical_id'] in plan['cost_logical_ids']}
    require(costs == related and sorted(costs) == report['charged_attempts'], 'Attempt accounting lost')
    evidence.read(report['ledger_snapshot'])
    return dict(schema='exact-repair/real-projection-scope-completion/v1', status='complete',
                stage_id=manifest['stage_id'], study_complete=True, campaign_complete=False,
                original_report=manifest['report'], manifest=binding(manifest_path),
                scheduled_rows=54, recorded_rows=len(rows), scheduled_cases=6, projection_parents=2,
                independent_pair_count=1, inherited_split='train', scope=report['scope'],
                interpretation=report['interpretation'], control_definition=report['control_definition'],
                scientific_outcomes=report['outcomes'],
                logical_statuses=dict(Counter(r['logical_status'] for r in rows)),
                rows=rows, arm_summaries=report['arm_summaries'], qualification_calls=calls,
                gate_passage=False, heldout_claim=False, source_coherence_qualified=False,
                scientific_rows_rerun=0, matcher_inputs_consumed=0, labels_used_for_fitting=0,
                settled_study_costs=totals(costs), attempt_costs=costs,
                original_cost_snapshot=report['ledger_snapshot'],
                cumulative_ledger_snapshot=manifest['ledger_snapshot'],
                cost_scope='All eight settled preparation/evaluation/report attempts. Original report '
                           'snapshot preserved. Audit worker and maintenance are separately charged; '
                           'nested native/row diagnostics are not charged twice.',
                attempt_lineage=[s['run'] for s in sources], verified_files=evidence.files,
                verified_file_count=len(evidence.files), api_spend_usd=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('output')
    args = parser.parse_args()
    from exact.repair.api import write_artifact
    write_artifact(Path(args.output) / 'report.json', audit(args.manifest))


if __name__ == '__main__':
    main()
