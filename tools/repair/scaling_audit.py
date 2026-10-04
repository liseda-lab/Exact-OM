"""Failure-inclusive audit of published scaling receipts; no scientific replay."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from tools.repair.overlap_audit import Evidence, binding, require


def checkpoint(evidence, ref):
    value = evidence.read(ref)
    require(canonical_hash({k: v for k, v in value.items() if k != 'content_hash'})
            == value.get('content_hash'), 'Checkpoint content changed')
    return value


def attempt(evidence, item, ledger, active):
    run = item['run']
    complete, step = evidence.read(item['completion']), evidence.read(item['step'])
    for receipt in (complete, step):
        require(receipt['step_id'] == run['step_id'] and
                receipt['dispatch_nonce'] == run['dispatch_nonce'], 'Attempt ownership differs')
    require(complete['job_id'] == run['logical_id'], 'Logical job differs')
    require(int(evidence.verify(item['exit']).read_text()) == complete['exit_code'],
            'Exit receipt differs')
    require(not active or (complete['status'] == 'complete' and complete['exit_code'] == 0),
            'Active shard is not complete')
    batch = evidence.read(item['batch'])
    require(complete['batch'] == item['batch']['path'], 'Batch path differs')
    require(evidence.verify(item['batch_hash']).read_text().strip() == item['batch']['sha256'],
            'Batch digest differs')
    for path, digest in batch['frozen_files'].items():
        if path not in evidence.files:
            evidence.verify(dict(path=path, sha256=digest))
        else:
            require(evidence.files[path] == digest, 'Conflicting frozen source binding')
    runtime = evidence.read(item['runtime'])
    charge = ledger['attempts'][str(Path(run['completion_path']).parent)]
    require(charge['status'] == 'settled' and charge['logical_id'] == run['logical_id'],
            'Attempt cost missing')
    require(charge['elapsed_seconds'] >= complete['elapsed_seconds'] and
            (charge.get('cpu_seconds') or 0) >= complete['cpu_seconds'], 'Attempt cost lost')
    if active:
        for relative, digest in evidence.read(item['outputs']).items():
            evidence.verify(dict(path=str(Path(complete['work']) / relative), sha256=digest))
    return dict(run=run, completion=complete, charge=charge, runtime=runtime,
                source_commit=batch['commit'], batch=item['batch'])


def audit_rows(evidence, schedule, reports):
    expected = {r['id']: r for r in schedule['rows']}
    require(len(expected) == len(schedule['rows']), 'Duplicate scheduled row')
    seen, rows = set(), []
    for report_ref in reports:
        report = checkpoint(evidence, report_ref)
        require(report['status'] == 'complete' and evidence.read(report['schedule']) == schedule,
                'Incomplete or mismatched shard schedule')
        require(report['recorded_rows'] == report['scheduled_rows'] == len(report['rows']),
                'Shard denominator differs')
        for ref in report['rows']:
            saved = checkpoint(evidence, ref)
            row = saved['row']
            require(row['id'] not in seen and expected.get(row['id']) == row,
                    'Duplicate or unexpected scaling row')
            require(saved['cleanup_complete'], 'Unresolved row cleanup')
            seen.add(row['id'])
            for payload in saved.get('payloads', []):
                evidence.verify(payload)
            result = evidence.read(saved['result']) if saved.get('result') else {}
            require(not result or result['row'] == row, 'Published result row differs')
            item = schedule['cases'][row['case_index']]
            pool = evidence.read(result['pool']) if result.get('pool') else None
            rows.append(dict(row=row, receipt=ref, result=saved.get('result'),
                parent=item['parent'], control=item['control'], config=item['config'],
                process_status=saved['status'], detail=saved.get('detail', ''),
                elapsed_seconds=saved['elapsed_seconds'], resources=saved['resources'],
                logical_status=result.get('logical_status', 'UNKNOWN'),
                semantic_status=result.get('semantic_status', 'unavailable'),
                semantic_benefit=result.get('semantic_benefit'), edit_cost=result.get('edit_cost'),
                generation_status=result.get('generation_status', 'unavailable'),
                generation_detail=result.get('generation_detail', ''),
                generation_complete=result.get('generation_complete', False),
                fallback_original_inventory=result.get('fallback_original_inventory'),
                cache_source=result.get('cache_source'),
                first_verified_seconds=result.get('first_verified_seconds'),
                checks=result.get('checks'), solves=result.get('solves'),
                lower_bound=result.get('lower_bound'), upper_bound=result.get('upper_bound'),
                generation_objects=pool['reports'] if pool else [],
                generation_scheduled_objects=pool['scheduled_objects'] if pool else None))
    require(seen == set(expected), 'Missing scaling denominator rows')
    return rows


def comparisons(rows):
    languages, caches = [], []
    indexed = {(r['row']['case_index'], r['row']['method'], r['row']['cache_mode']): r for r in rows}
    for case in sorted({r['row']['case_index'] for r in rows}):
        for mode in ('cold', 'warm'):
            pair = [indexed[(case, method, mode)] for method in
                    ('semantic_circuit', 'semantic_enumeration_decoder')]
            objects = [{r['object_id']: r for r in p['generation_objects']} for p in pair]
            shared = set(objects[0]) & set(objects[1])
            languages.append(dict(case_index=case, cache_mode=mode,
                compared_objects=len(shared), missing_objects=sorted(set(objects[0]) ^ set(objects[1])),
                mismatches=[o for o in sorted(shared) if any(objects[0][o].get(k) != objects[1][o].get(k)
                    for k in ('language_hash', 'context_proofs', 'effective_menu'))],
                both_generation_complete=all(p['generation_complete'] for p in pair)))
        for method in ('grammar_circuit', 'semantic_circuit', 'semantic_enumeration_decoder'):
            cold, warm = (indexed[(case, method, mode)] for mode in ('cold', 'warm'))
            # Recovery rows can retain a result in the original payload directory.
            expected = str(Path(cold['result']['path']).parent / 'compiler-cache') if cold['result'] else None
            caches.append(dict(case_index=case, method=method, cold=cold['row']['id'],
                warm=warm['row']['id'], source=warm['cache_source'], expected_source=expected,
                provenance_status=('unavailable' if not warm['result'] or expected is None else
                    'matched' if warm['cache_source'] == expected else 'mismatch'),
                cold_seconds=cold['elapsed_seconds'], warm_seconds=warm['elapsed_seconds'],
                paired_quality_available=all(r['semantic_status'] == 'known' and
                    r['logical_status'] == 'VERIFIED_FEASIBLE' for r in (cold, warm))))
    return languages, caches


def audit(manifest_path, output):
    evidence = Evidence()
    manifest = evidence.read(binding(manifest_path))
    schedule, ledger = evidence.read(manifest['schedule']), evidence.read(manifest['ledger_snapshot'])
    require(ledger['limit_worker_seconds'] is None, 'Time amendment missing')
    require(len(schedule['rows']) == 240 and len(manifest['active_run_ids']) == 20,
            'Primary scaling scope differs')
    sources = [attempt(evidence, item, ledger, item['run']['id'] in manifest['active_run_ids'])
               for item in manifest['attempts']]
    require(set(manifest['active_run_ids']) <= {s['run']['id'] for s in sources}, 'Missing active receipts')
    for item in schedule['cases']:
        case = evidence.read(item['case'])
        require(case['case']['split'] == 'development', 'Scaling opened a non-development case')
    rows = audit_rows(evidence, schedule, manifest['reports'])
    languages, caches = comparisons(rows)
    depth2 = [r for r in rows if r['config']['depth'] == 2]
    errors = Counter(r['generation_detail'] for r in rows if r['generation_status'] == 'error')
    depth_decision = dict(status='not_activated_development_viability_not_established',
        optional=True, scheduled_depth2_rows=len(depth2),
        generation_complete=sum(r['generation_complete'] for r in depth2),
        rationale='Depth2 has incomplete generation and primary clean-control endpoint errors. '
                  'Larger grammar is not justified by these development observations. '
                  'No test outcomes or scientific budgets were changed.', test_outcomes_used=False)
    result = dict(schema='exact-repair/scaling-primary-audit/v1', manifest=binding(manifest_path),
        scheduled=240, recorded=len(rows), attempts=sources, rows=rows,
        process_statuses=dict(Counter(r['process_status'] for r in rows)),
        logical_statuses=dict(Counter(r['logical_status'] for r in rows)),
        generation_statuses=dict(Counter(r['generation_status'] for r in rows)),
        generation_errors=dict(errors),
        semantic_known=sum(r['semantic_status'] == 'known' for r in rows),
        generated_quality_known=sum(r['semantic_status'] == 'known' and not r['fallback_original_inventory']
                                    for r in rows),
        language_comparisons=languages, cache_comparisons=caches, depth3_decision=depth_decision,
        cost_snapshot=manifest['ledger_snapshot'], cumulative_costs_at_snapshot=ledger['cumulative'],
        verified_files=len(evidence.files), status='complete', study_complete=False, gates_passed=False,
        limitations=['A completed row can contain a generation error or UNKNOWN native outcome.',
            'Endpoint-retrieval errors in the original clean controls need explicit repair or terminal accounting; '
            'original fallback results remain immutable and cannot qualify generated-pool comparisons.',
            'Two exposed development parents; padding is benign load. No held-out or strongest-symbolic claim.',
            'Candidate caps32/64 can be nonbinding under32 draws; inspect actual sampled_unique and pool_size.',
            'Support-incidence native witnesses/evaluation and failure-inclusive final report remain pending.'],
        api_spend_usd=0)
    write_artifact(Path(output) / 'report.json', result)
    write_artifact(Path(output) / 'verified-files.json', evidence.files)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.manifest, args.output)
