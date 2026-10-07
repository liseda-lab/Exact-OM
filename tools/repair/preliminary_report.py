"""Consolidated preliminary scope, authenticated receipts only; no scientific replay."""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from exact.repair.api import write_artifact
from tools.repair.fresh_evaluation_audit import references
from tools.repair.overlap_audit import Evidence, binding, require
from tools.repair.report_campaign import totals
from tools.repair.training_readiness_report import authenticate_attempt

SCHEMA = 'exact-repair/preliminary-report-manifest/v1'
BRANCHES = {'qualification', 'requirements', 'historical', 'profile', 'numerical',
            'corpus', 'evaluation', 'training', 'robustness', 'scaling', 'real'}


class StreamEvidence(Evidence):
    """Hash each immutable file once per report, without loading native payloads."""
    def verify(self, ref):
        path = Path(ref['path'])
        if str(path) in self.files:
            require(self.files[str(path)] == ref['sha256'], 'Conflicting evidence binding')
            return path
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        require(digest == ref['sha256'], 'Changed evidence: ' + str(path))
        self.files[str(path)] = digest
        return path


def junit(path):
    result = {k: sum(int(s.get(k, 0)) for s in ET.parse(path).getroot().iter('testsuite'))
              for k in ('tests', 'failures', 'errors', 'skipped')}
    require(result['tests'] > 0 and not any(result[k] for k in ('failures', 'errors', 'skipped')),
            'Final report validation did not pass')
    return result


def authenticate(evidence, item, ledger, registry):
    source = authenticate_attempt(evidence, item, ledger, registry)
    report = evidence.read(item['report'])
    require(report['status'] == 'complete', 'Incomplete branch report')
    work = Path(source['completion']['work'])
    outputs = evidence.read(item['outputs'])
    # Published audit indexes bind the original rows, descendants and partial payloads.
    for relative in outputs:
        if Path(relative).name == 'verified-files.json':
            index = evidence.read(dict(path=str(work/relative), sha256=outputs[relative]))
            for path, digest in index.items():
                evidence.verify(dict(path=path, sha256=digest))
    if isinstance(report.get('verified_files'), dict):
        for path, digest in report['verified_files'].items():
            evidence.verify(dict(path=path, sha256=digest))
    references(evidence, report)
    for key in ('attempt_costs',):
        for identity, charge in report.get(key, {}).items():
            require(ledger['attempts'].get(identity) == charge, 'Prior branch cost changed')
    if item.get('validation_minimum'):
        require('validation.xml' in outputs, 'Final worker validation missing')
        counts = junit(work/'validation.xml')
        require(counts['tests'] >= item['validation_minimum'], 'Final validation denominator differs')
        source['validation'] = counts
    for attempt in report.get('attempts', []):
        if isinstance(attempt, dict) and 'run' in attempt and 'charge' in attempt:
            identity = str(Path(attempt['run']['completion_path']).parent)
            require(ledger['attempts'].get(identity) == attempt['charge'], 'Prior attempt cost changed')
    return report, source


def validate_denominators(reports):
    require(set(reports) == BRANCHES, 'Missing or duplicate study branch')
    historical = reports['historical']['denominators']
    require(historical == dict(cases=90, formerly_failed_cases=79, generation_rows=180,
                               native_rows=270), 'Historical denominator differs')
    corpus = reports['corpus']
    require((corpus['scheduled_cases'], corpus['training_target_cases'],
             corpus['fresh_evaluation_target_cases']) == (288, 224, 64), 'Corpus denominator differs')
    require(corpus['teacher_queries'] == 0 and not corpus['test_outcomes_opened'],
            'Corpus scope differs')
    for name, branch in reports['evaluation']['branches'].items():
        require(name in ('primary', 'inventory') and branch['scheduled_rows'] ==
                branch['recorded_rows'] == len(branch['rows']) == 576 and
                branch['parent_groups'] == 32, 'Fresh denominator differs')
        require(branch['quality_usable_rows'] + branch['quality_unavailable_rows'] == 576,
                'Fresh missing quality was dropped')
    require(set(reports['evaluation']['branches']) == {'primary', 'inventory'}, 'Missing fresh branch')
    train = reports['training']; collect = train['collection']['summary']
    require(train['original_fixed_inventory']['scheduled_cases'] == 160 and
            train['decode_diagnostic']['scheduled_rows'] == 288 and
            train['intended_qualification']['scheduled_cases'] == 32 and
            collect['scheduled_cases'] == 32 and collect['parent_groups'] == 16 and
            collect['scheduled_slots'] == 256, 'Training readiness denominator differs')
    robust = reports['robustness']
    require(robust['scheduled_rows'] == robust['recorded_rows'] == len(robust['rows']) == 2880 and
            robust['base_cases'] == 64 and robust['case_variants'] == 320 and
            robust['parent_groups'] == 32 and len(robust['paired_interventions']) == 36,
            'Robustness denominator differs')
    require(robust['quality_usable_rows'] + robust['quality_unavailable_rows'] == 2880 and
            robust['overlap']['scheduled_rows'] == robust['overlap']['recorded_rows'] == 36,
            'Robustness missing or overlap rows dropped')
    require(reports['scaling']['scheduled'] == reports['scaling']['recorded'] == 336,
            'Scaling denominator differs')
    require(set(reports['scaling']['branches']) == {'primary', 'incidence', 'endpoint_amendment'},
            'Missing scaling branch')
    for name, expected in (('primary', 240), ('incidence', 72), ('endpoint_amendment', 24)):
        branch = reports['scaling']['branches'][name]
        require(branch['summary']['scheduled'] == branch['summary']['recorded'] ==
                len(branch['rows']) == expected, 'Scaling branch denominator differs')
    real = reports['real']
    require(real['scheduled_rows'] == real['recorded_rows'] == 54 and
            real['scheduled_cases'] == 6 and real['independent_pair_count'] == 1,
            'Real projection denominator differs')
    for name in ('evaluation', 'training', 'robustness'):
        value = reports[name]
        require(value['local_reporting_complete'] and not any(value[k] for k in
            ('study_complete', 'campaign_complete', 'gates_passed', 'supervision_admitted',
             'scientific_rows_replayed', 'new_native_calls')), 'Preliminary claim promoted')
    require(not train['fitting_eligible'] and not train['heldout_cases_opened'], 'Training scope promoted')


def authenticate_history(evidence, registry, ledger, receipts):
    runs = {r['id']: r for r in registry['runs']}
    require(len(runs) == len(registry['runs']) == len(receipts), 'Attempt history denominator differs')
    require({r['id'] for r in receipts} == set(runs), 'Attempt receipt missing or duplicated')
    result = []
    for item in receipts:
        run = runs[item['id']]
        require(item['completion']['path'] == run['completion_path'], 'Attempt path differs')
        receipt = evidence.read(item['completion'])
        require(all(receipt.get(k) == run.get(k) for k in ('step_id', 'dispatch_nonce')),
                'Historical attempt nonce/step differs')
        active = run.get('enabled', True) and not run.get('superseded_by')
        require(not active or (receipt['status'] == 'complete' and receipt['exit_code'] == 0),
                'Unaccounted active failed attempt')
        charge = ledger['attempts'][str(Path(run['completion_path']).parent)]
        require(charge['status'] == 'settled' and charge['logical_id'] == receipt['job_id'],
                'Historical attempt charge absent or unsettled')
        result.append(dict(run=run, receipt=item['completion'], status=receipt['status'], charge=charge))
    return result


def reconcile_costs(ledger, pilot, smoke):
    require(ledger['limit_worker_seconds'] is None, 'Campaign time amendment lost')
    require(all(ledger['attempts'].get(k) == v for k, v in pilot['attempts'].items()),
            'Inherited pilot costs changed')
    require(not (set(smoke['attempts']) & set(ledger['attempts'])), 'Historical smoke double charge')
    current = totals(ledger['attempts'])
    require(math.isclose(current['worker_seconds'] + current['reserved_worker_seconds'],
                         ledger['cumulative']['worker_seconds']), 'Cumulative costs do not reconcile')
    return dict(campaign=current, historical_pilot_included=totals(pilot['attempts']),
        historical_smoke_separate=totals(smoke['attempts']),
        combined=totals({**ledger['attempts'], **smoke['attempts']}), costs_reset=False,
        limit_worker_seconds=None, api_spend_usd=0,
        scope='Immutable snapshot includes failed attempts and maintenance. This preparation and '
              'final worker settle separately; final closure must publish a later cumulative snapshot. '
              'Stage totals overlap and must never be summed; row effort is descriptive, not another charge.')


def compact(value):
    """Keep scientific summaries; complete rows remain in the hash-bound source reports."""
    omitted = {'rows', 'attempt_costs', 'scaling_attempt_costs', 'attempts', 'attempt_lineage',
               'prior_attempt_lineage', 'audit_workers', 'verified_files', 'scope_audit',
               'maintenance_costs', 'original_generation_rows', 'original_native_rows',
               'revised_native_rows', 'replacement_rows', 'attempt_charges'}
    if isinstance(value, dict):
        return {k: compact(v) for k,v in value.items() if k not in omitted}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def corrections(reports, refs):
    result = []
    for item in reports['training']['corrections']:
        priority = 1 if item['kind'] == 'implementation' else 2
        result.append(dict(item, priority=priority, source=refs['training']))
    extras = [
        ('model_schema_robustness', 'implementation', 1,
         'Pilot checkpoints lack some generated-pool schema types; unavailable rows remain in all denominators.',
         'Run all six full-schema encoders against unseen development relation/node types and compatible checkpoints.', 'robustness'),
        ('omission_and_activation_removal', 'implementation', 1,
         'Focused omission and canonical-bundle removal corrections executed; original failures and source regimes retained.',
         'Test final exclusion after every producer, unchanged activation-free protection, and activation-sensitive bundles.', 'robustness'),
        ('effective_menu_and_native_traces', 'instrumentation', 2,
         'Full effective retrieved-menu snapshots and per-obligation selected-label traces were not persisted.',
         'Persist hashed effective menus and complete native obligations, including partial/unknown calls, in development.', 'robustness'),
        ('generation_caps', 'resource_exhaustion', 2,
         'Native/generation deadlines and node caps constrain coverage; increased resource experiments stay separately named.',
         'Profile development cold/warm generation and full row CPU/wall/RSS; freeze matched scientific budgets before test.', 'historical'),
        ('composite_endpoint', 'unsupported_capability', 1,
         '120 original composite endpoint-admission comparisons remain unavailable; separate endpoint-disabled amendment is bounded.',
         'Qualify the intended rich action language with clean and corrupt composite bundles and matched resource controls.', 'scaling'),
        ('stronger_controls', 'experimental_design', 2,
         'Historical retained-axiom and zero-benefit controls do not qualify strongest semantic comparisons.',
         'Freeze observable-only semantic, no-repair, score-greedy, fixed-cost deletion and matched generator/decoder controls on development.', 'evaluation'),
        ('real_projection_scope', 'experimental_design', 3,
         'Two projections share one exposed training ontology pair; full-source and held-out claims remain unavailable.',
         'Freeze separately licensed releases and ontology/pair splits before enrichment; qualify native support and intended queries.', 'real')]
    for id, kind, priority, detail, acceptance, source in extras:
        result.append(dict(id=id, kind=kind, priority=priority, status='review_before_cluster',
                           evidence=detail, acceptance=acceptance, source=refs[source]))
    return sorted(result, key=lambda row: (row['priority'], row['id']))


def report(manifest_path, output):
    output = Path(output); evidence = StreamEvidence()
    manifest = evidence.read(binding(manifest_path))
    require(manifest['schema'] == SCHEMA and manifest['scientific_rows_replayed'] ==
            manifest['new_native_calls'] == 0, 'Wrong consolidated manifest or replay boundary')
    registry = evidence.read(manifest['registry_snapshot']); ledger = evidence.read(manifest['ledger_snapshot'])
    amendment = evidence.read(manifest['scope_amendment']); program = evidence.read(manifest['program'])
    require(amendment['original_program'] == manifest['program'] and
            not amendment['original_scientific_budgets_changed'], 'Scope amendment differs')
    require(evidence.read(manifest['authorization'])['limit_worker_seconds'] is None, 'Authorization differs')
    storage = evidence.read(manifest['storage_policy'])
    reports = {}; owners = []; refs = {}
    for item in manifest['branches']:
        kind = item['kind']; require(kind not in reports, 'Duplicate study branch')
        reports[kind], owner = authenticate(evidence, item, ledger, registry)
        refs[kind] = item['report']; owners.append(owner)
        print('Authenticated '+kind+': '+str(len(evidence.files))+' immutable files', flush=True)
    validate_denominators(reports)
    lineage = evidence.read(manifest['budget_lineage'])
    inherited = {k: evidence.read(dict(path=v['snapshot'], sha256=v['sha256']))
                 for k,v in lineage['inherited'].items() if k.endswith('ledger.json')}
    cost = reconcile_costs(ledger, inherited['pilot-ledger.json'], inherited['smoke-ledger.json'])
    history = authenticate_history(evidence, registry, ledger, manifest['attempt_receipts'])
    # Full registry history and ledger are output unchanged, including disabled originals.
    require(len({r['id'] for r in registry['runs']}) == len(registry['runs']), 'Duplicate registered attempt')
    unfinished = [r['id'] for r in registry['runs'] if r.get('enabled', True) and
                  not r.get('superseded_by') and not Path(r['completion_path']).exists()]
    require(not unfinished, 'Unaccounted active registered attempt: '+str(unfinished))
    qualification = reports['training']['prerequisites']['qualification'][0]['data']
    requirement_review = evidence.read(qualification['requirement_review'])
    proposal = dict(reports['training']['proposed_cluster_design'])
    require(proposal['status'] == 'proposal_not_frozen_or_approved' and proposal['storage'] == storage,
            'Cluster proposal approval or storage scope differs')
    proposal.update(program_targets=program['studies'], resource_policy=dict(
        hardware_profile='Current development: RTX 2060 SUPER 8 GiB, 8 CPUs, 61952 MiB RAM',
        admission=dict(cpus=6, gpus=1, memory_mb=49152),
        usual_workers=[dict(cpus=4,gpus=1,memory_mb=32768),dict(cpus=2,gpus=0,memory_mb=16384)],
        supervisor=dict(cpus=1,gpus=0,memory_mb=8192),
        scientific_comparison=dict(wall_seconds=300,cpu_seconds=600,memory_mb=8192,generation_seconds=60),
        operational_slices='Profile development optimizer/RNG, minibatches and decoder; finite resumable slices, no campaign ceiling.'),
        reproducibility=['Freeze corrected source/runtime, arm schema, split and alias-family identities, release hashes, '
            'training-only acquisition, query basis, controls and development selection before held-out opening.',
            'Never reuse exposed or renamed sibling parents as independent held-out parents; no silent pilot warm start.',
            'Separate resource experiments from matched comparisons; retain every failed/unknown row and charge.',
            'Use staged robustness and bounded scaling, not an optional-feature full factorial.'])
    stage_map = dict(qualification=['qualification','requirements'], historical_regression=['historical'],
        profile_and_corpus=['profile','numerical','corpus'], fresh_evaluation=['evaluation'],
        larger_training_readiness=['training'], robustness=['robustness'], scaling=['scaling'], real_modules=['real'])
    scope = dict(schema='exact-repair/preliminary-scope-readiness/v1', status='ready_for_completion_receipt_review',
        local_scientific_dependencies_accounted=True, local_scope_complete=False,
        remaining_obligation='Authenticate this final worker, all outputs and validation; settle preparation/report costs; '
            'publish final cumulative accounting and scope-completion receipt under registry lock before terminalizing.',
        stages={k:dict(status='local_preliminary_accounted',reports=[refs[n] for n in names]) for k,names in stage_map.items()},
        deferred_work=[dict(status='deferred_by_user',obligation=x) for x in amendment['deferred_work']],
        full_expanded_program_complete=False, gates=dict(G0='not_established',G1='not_established',G2='not_established'),
        fitting_eligible=False, learning_efficiency_qualified=False, strongest_symbolic_comparison_qualified=False,
        production_matcher='deferred_until_user_release', llm_labels='disabled', api_spend_usd=0,
        scope_amendment=manifest['scope_amendment'], notification='Existing deterministic notify_when_idle only; no manual email.')
    result = dict(schema='exact-repair/preliminary-cross-study-report/v1', status='complete',
        scope='local_preliminary_only', manifest=binding(manifest_path), source_reports=refs,
        stages={k:compact(v) for k,v in reports.items()}, branch_workers=owners,
        corrections=corrections(reports,refs), deferred_work=scope['deferred_work'],
        deferred_control_obligations={k:reports[k]['deferred_obligations'] for k in ('evaluation','training','robustness')},
        proposed_cluster_design=proposal, scope_readiness=scope, cumulative_costs=cost,
        all_attempts=len(registry['runs']), ledger_entries=len(ledger['attempts']),
        current_requirement_review=requirement_review, historical_requirement_counts=reports['historical']['requirement_counts'],
        costs_snapshot=manifest['ledger_snapshot'], scope_amendment=manifest['scope_amendment'],
        verified_file_count=len(evidence.files), scientific_rows_replayed=0, new_native_calls=0,
        new_fitting_runs_executed=0, api_spend_usd=0, full_program_complete=False,
        gates_passed=False, campaign_complete=False, local_report_complete=True)
    write_artifact(output/'report.json',result);write_artifact(output/'scope-readiness.json',scope)
    write_artifact(output/'attempt-history.json',history)
    write_artifact(output/'costs.json',dict(summary=cost, ledger=ledger, historical=inherited))
    write_artifact(output/'verified-files.json',evidence.files)
    render(result,output/'REPORT.md')
    return result


def render(value, path):
    r=value['stages']; lines=['# Preliminary XR-2.1 study report','',
        'Local preliminary diagnostics are accounted for. Full training and final cluster execution remain deferred. '
        'G0–G2, strongest-symbolic comparisons and learning efficiency are not established. '
        'Final controller closure requires this worker receipt and settled cumulative costs.','',
        '| Stage | Scheduled denominator | Scope |','|---|---:|---|',
        '| Historical engineering regression | 90 cases; 180 generation / 270 native rows | Exposed development |',
        '| Fresh frozen-model evaluation | 576 generated + 576 inventory rows | 64 cases / 32 parents |',
        '| Training readiness | 160 acquisition cases; 192 gradient probes; 288 decode rows; 32 intended checks; 256 collection slots | No fitting |',
        '| Robustness | 2,880 rows + 36 separate overlap rows | 320 variants / 32 parents; overlap one exposed ancestry |',
        '| Scaling | 240 + 72 + 24 rows | Bounded development diagnostics |',
        '| Real projections | 54 rows / 6 cases | One exposed ontology pair |','',
        'Native fixture qualification is source-bound implementation evidence. The historical requirement matrix, '
        'current qualification, generation coverage and unresolved failures remain separate in report.json. '
        'Corpus releases retain 224 training-program and 64 fresh-evaluation cases; no held-out training payload is opened here.','']
    for key in ('primary','inventory'):
        b=r['evaluation']['branches'][key]
        lines.append(f"Fresh {key}: {b['quality_usable_rows']}/{b['scheduled_rows']} usable selected-quality rows; "
                     f"{b['quality_unavailable_rows']} unavailable. Scientific statuses: {b['scientific_statuses']}.")
    robust=r['robustness'];train=r['training']['collection']['summary']
    lines += ['',f"Robustness: {robust['quality_usable_rows']}/2880 usable generated-pool quality rows; "
        f"{robust['quality_unavailable_rows']} unavailable. Scientific statuses: {robust['scientific_statuses']}.",
        '',f"Development collection: {train['usable_unique_labels']} usable unique labels, {train['unknown_slots']} unknown "
        f"slots out of {train['scheduled_slots']}; development labels are excluded from fitting.",'',
        'All no-op, partial, unknown, unsupported, timeout and original failure rows remain in the linked source reports. '
        'Parent-paired intervals are descriptive. Executed source revisions can confound robustness differences. '
        'Fixed-inventory diagnostics do not establish generated-pool quality or external regret.','',
        '## Corrections before cluster review','']
    for item in value['corrections']:
        lines += [f"- P{item['priority']} {item['id']} ({item['kind']}; {item['status']}): {item['evidence']} "
                  f"Acceptance: {item['acceptance']}"]
    lines += ['', '## Proposed final cluster study', '',
        'Proposal only: six encoder/pairwise arms × seeds 13/37/73; up to 50 epochs, evaluation every five epochs '
        'and patience 10. Qualify TRAIN-only generated supervision and optimizer/minibatch resources first. '
        'Freeze development generated-quality/effort selection, full schema, releases, ancestry splits, query basis '
        'and observable-only semantic/reference/matched generator controls before held-out execution. '
        'Use independent bounded robustness, scaling and licensed real-module branches; no full factorial.', '',
        'Keep code/configuration in home; active data, logs, checkpoints and caches on parallel storage; completed '
        'bundles move to archive only after dependency checks and checksum-verified relocation receipts. '
        'Actual mount paths and final resource profiles await review. No migration or new cluster launch occurred.', '',
        '## Cumulative accounting', '',
        f"Snapshot combined paid worker-hours: {value['cumulative_costs']['combined']['worker_seconds']/3600:.3f}; "
        f"reserved worker-hours: {value['cumulative_costs']['combined']['reserved_worker_seconds']/3600:.3f}. "
        'All failures and maintenance are retained. Historical smoke is separate from the inherited pilot; '
        'shared branch costs are charged once. Preparation/final-worker settlement and closure add a later snapshot. '
        'No campaign time ceiling; API spend is zero.', '',
        'Machine-readable report.json contains all stage summaries, source report hashes, correction acceptance checks, '
        'proposal, denominators and limitations. attempt-history.json preserves the complete registry ancestry; '
        'costs.json preserves every ledger entry. Existing deterministic idle notification requests the next decision '
        'only after final receipt authentication and local-scope closure.']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('output');a=p.parse_args()
    result=report(a.manifest,a.output)
    print('Consolidated '+str(len(result['source_reports']))+' branches; '+str(result['verified_file_count'])+' files')


if __name__=='__main__': main()
