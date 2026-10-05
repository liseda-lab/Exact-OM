"""Receipt-only preliminary readiness report; no fitting or new native calls."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, read_record
from exact.repair.learning import SemanticTargetSpec
from tools.repair import development_collection as collection
from tools.repair.development_collection_recovery import authenticate_label
from tools.repair.fresh_evaluation_audit import references, validate_attempt
from tools.repair.overlap_audit import Evidence, binding, require
from tools.repair.prepare import case_from_dict, case_to_dict, cache_from_dict
from tools.repair.report_campaign import totals
from tools.repair.scaling_audit import checkpoint
from tools.repair.training_audit import label_masks

SCHEMA = 'exact-repair/preliminary-training-readiness-manifest/v1'


def slot_labels(evidence, sample, slots):
    """Duplicates retain slots and may reuse only the exact same assignment label."""
    require(len(sample['rows']) == len(slots) == 8, 'Slot denominator differs')
    unique = {}; labels = []
    for expected, slot in zip(sample['rows'], slots):
        require({k: slot[k] for k in expected} == expected, 'Sampler slot changed')
        assignment = tuple(expected['assignment'])
        label = evidence.read(slot['label'])
        require(label['assignment'] == expected['assignment'], 'Label assignment differs')
        require(slot['reused_exact_assignment'] == (assignment in unique), 'Duplicate flag differs')
        if assignment in unique:
            require(unique[assignment] == slot['label'], 'Duplicate slot changed label')
        else:
            unique[assignment] = slot['label']
        labels.append(label)
    require(len(unique) == sample['unique_assignments'], 'Unique denominator differs')
    return unique, labels


def summarize(rows):
    return dict(scheduled_cases=len(rows), parent_groups=len({r['parent'] for r in rows}),
        scheduled_slots=8*len(rows), recorded_slots=sum(r['recorded_slots'] for r in rows),
        unknown_slots=sum(r['unknown_slots'] for r in rows),
        unique_labels=sum(r['unique_labels'] for r in rows),
        usable_unique_labels=sum(r['usable_labels'] for r in rows),
        cases_with_usable_labels=sum(r['usable_labels'] > 0 for r in rows),
        parents_with_usable_labels=len({r['parent'] for r in rows if r['usable_labels'] > 0}),
        duplicate_slots=sum(r['duplicate_slots'] for r in rows),
        statuses=dict(Counter(r['status'] for r in rows)),
        native_call_statuses=dict(Counter(c['status'] for r in rows for c in r['label_calls'])),
        row_wall_seconds=sum(r['elapsed_seconds'] for r in rows))


def authenticate_attempt(evidence, item, ledger, snapshot):
    run = item['run']
    require(next(r for r in snapshot['runs'] if r['id'] == run['id']) == run,
            'Frozen registry owner differs')
    active = not bool(run.get('superseded_by'))
    source = validate_attempt(evidence, item, ledger, active)
    if item.get('outputs'):
        outputs = evidence.read(item['outputs'])
        work = Path(source['completion']['work'])
        for relative, digest in outputs.items():
            path = str(work / relative)
            if path not in evidence.files:
                evidence.verify(dict(path=path, sha256=digest))
            else:
                require(evidence.files[path] == digest, 'Output binding conflict')
        if item.get('report'):
            require(outputs[str(Path(item['report']['path']).relative_to(work))] == item['report']['sha256'],
                    'Report not bound by worker')
    return source


def scientific_source(evidence, item, source, plan_ref):
    batch = evidence.read(item['batch']); runtime = source['runtime']
    adapter = item['run'].get('transport_recovery_plan')
    if adapter:
        adapter = evidence.read(adapter)
        references(evidence, adapter)
        batch = evidence.read(adapter['frozen_batch'])
        runtime = evidence.read(binding(Path(adapter['frozen_batch']['path']).with_name('runtime.json')))
        require(adapter['runtime'] == runtime and adapter['schedule'] == plan_ref,
                'Transport changed scientific runtime or schedule')
    digest = batch['frozen_files'][str(Path(batch['code'])/'tools/repair/development_collection.py')]
    original = canonical_hash((plan_ref, digest, runtime))
    return original, adapter


def collection_case(evidence, ref, row, identity, plan):
    saved = evidence.checkpoint(ref, canonical_hash((identity, row)))
    require((saved['case_id'], saved['parent'], saved['family'], saved['split']) ==
            (row['case_id'], row['structural_parent'], row['family'], 'development'),
            'Collection case or split differs')
    require(saved['budget'] == collection.BUDGET and saved['cleanup_complete'] and
            not any(saved[k] for k in ('fitting_labels_admitted', 'heldout_cases_opened', 'original_results_replaced')),
            'Collection boundary or budget differs')
    references(evidence, saved['payloads'])
    payloads = {p['path']:p for p in saved['payloads']}
    directory = Path(ref['path']).parent
    def payload(name):
        return evidence.read(payloads[str(directory/name)])
    result = saved['result']; generation = payload('generation-call.json')
    require(generation['cleanup_complete'] and generation['deadline_seconds'] == 300,
            'Generation allowance or cleanup differs')
    require(result['requested_slots'] == 8 and result['fitting_labels_admitted'] is False,
            'Collection requested slots differ')
    labels = []; calls = []; native = []; scope = None
    native_statuses=Counter();backends=Counter();obligations=Counter()
    for payload_ref in saved['payloads']:
        path=Path(payload_ref['path'])
        if path.parent.name=='native' and path.stem.removeprefix('check-').isdigit():
            check=evidence.read(payload_ref)
            native_statuses[check['logical_status']]+=1
            backends[str((check['support']['reasoner'],check['support']['backend']))]+=1
            for obligation in check['obligations']:
                obligations[str((obligation['kind'],(obligation['verdict'] == obligation['expected'] if obligation['complete'] else None),obligation['complete']))]+=1
    if saved['status'] == 'sampled':
        require(generation['status'] == 'complete', 'Sample without completed generation')
        original = read_record(evidence.read(row['observable']))
        pool = payload('pool.json'); problem = read_record(pool['input'])
        require(original.content_hash == row['input_hash'], 'Original input changed')
        sample = collection.sample_assignments(problem, original, plan['seed'])
        require(sample == payload('sample.json'), 'Frozen sampling changed')
        case = case_from_dict(evidence.read(row['evaluator']))
        require(case.split == 'development' and case.structural_parent == row['structural_parent']
                and case.problem.content_hash == original.content_hash, 'Evaluator identity differs')
        case = replace(case, problem=problem)
        generated = payload('generated-case.json')
        require(canonical_hash(generated) == canonical_hash(case_to_dict(case)), 'Generated evaluator changed')
        unique, labels = slot_labels(evidence, sample, result['slots'])
        protocol = evidence.read(plan['protocol']); profile = sorted(protocol['objective']['edit_weights'].items())
        for assignment, label_ref in unique.items():
            target = Path(label_ref['path']).parent
            call = evidence.read(payloads[str(target/'call.json')])
            require(call['cleanup_complete'] and call['deadline_seconds'] == 300,
                    'Label allowance or cleanup differs')
            label = evidence.read(label_ref)
            proof_ref = payloads.get(str(target/'transport-recovery.json'))
            native_ref = label_ref; native_payloads = saved['payloads']
            if proof_ref:
                proof = evidence.read(proof_ref); references(evidence, proof)
                require(proof['additional_scientific_seconds'] == 0 and proof['prior_costs_reset'] is False,
                        'Transport recovery replayed science')
                native_ref = proof['original_label']
                require(evidence.read(native_ref) == label, 'Retained label changed')
                # Original evidence is authenticated by the slice recovery manifest.
                native_payloads = plan['_recovery_payloads']
            if call['status'] == 'complete':
                check = authenticate_label(generated, assignment, profile, native_ref, native_payloads)
                native.append(check)
            else:
                require(call['status'] in ('timeout','memory_limit','cpu_limit') and
                        label['feasible'] is None and label['benefit'] is None,
                        'Unknown call was promoted to known label')
            calls.append(dict(label=label_ref, status=call['status'], detail=call['detail'],
                original_transport_error=proof['original_call'] if proof_ref else None,
                native_checks=sum(1 for p in native_payloads if Path(p['path']).parent == Path(native_ref['path']).parent/'native'
                    and Path(p['path']).stem.removeprefix('check-').isdigit())))
        cache_data = evidence.read(result['cache']);cache = cache_from_dict(cache_data)
        require(not cache.complete and cache.stop_reason == 'predeclared_development_sample',
                'Sample was promoted to exact teacher')
        require(canonical_hash(cache_data['labels']) == canonical_hash([evidence.read(ref) for ref in unique.values()]),
                'Cache labels differ')
        hashes = dict(cache.hashes)
        expected = dict(input=problem.content_hash, patch=canonical_hash(problem.objects),
            policy=problem.policy.content_hash, query=canonical_hash(case.probes),
            inventory=canonical_hash(tuple(o.candidates for o in problem.objects)),
            profile=canonical_hash(tuple(tuple(p) for p in profile)), sampler=canonical_hash(sample),
            development_plan=canonical_hash({k:v for k,v in plan.items() if not k.startswith('_')}),
            backend=canonical_hash(('pyhermit', plan['_runtime']['dependencies']['pyhermit'], 'python')),
            teacher_weights=canonical_hash((1.0,1.0)),
            semantic_target=SemanticTargetSpec(canonical_hash(case.probes)).content_hash)
        require(all(hashes[k] == v for k,v in expected.items()), 'Cache dependency differs')
        masks = label_masks(cache, case, dict(desired_family_weight=1.0,false_positive_weight=1.0))
        require(masks == result['masks'] and masks['usable'] == result['usable_labels'], 'Usable mask differs')
        unknown = sum(l['feasible'] is None or (l['feasible'] and l['benefit'] is None) for l in labels)
        require(result['recorded_slots'] == 8 and result['unique_labels'] == len(unique)
                and result['unknown_slots'] == unknown and result['exact_teacher'] is False,
                'Collection label denominator differs')
        require(result['query_hash'] == canonical_hash(case.probes) and result['query_denominator'] == len(case.probes),
                'Query identity differs')
        scope = dict(query_hash=result['query_hash'],queries=len(case.probes),
            desired=sum(p.desired for p in case.probes),unwanted=sum(not p.desired for p in case.probes),
            typed_nonvacuity_conditions=sum(len(p.conditions()) for p in case.probes),
            native_backend=dict(reasoner='hermit',backend='python',version=plan['_runtime']['dependencies']['pyhermit']),
            generation_reports=pool['proposal_reports'],generation_scope=result['generation_scope'])
    else:
        require(result['recorded_slots'] == result['unique_labels'] == result['usable_labels'] == 0
                and result['unknown_slots'] == 8, 'Unavailable generation lost denominator')
        require((saved['status'] == 'generation_timeout' and generation['status'] == 'timeout') or
                (saved['status'] == 'unsupported_original_mapping_bundle' and generation['status'] == 'error'
                 and generation['detail'] == collection.UNSUPPORTED), 'Unaccounted generation error')
    return dict(case_id=row['case_id'],parent=row['structural_parent'],family=row['family'],
        receipt=ref,status=saved['status'],elapsed_seconds=saved['elapsed_seconds'],
        recorded_slots=result['recorded_slots'],unknown_slots=result['unknown_slots'],
        unique_labels=result['unique_labels'],usable_labels=result['usable_labels'],
        duplicate_slots=result['recorded_slots']-result['unique_labels'],label_calls=calls,
        native_authentication=native,scope=scope,generation_call=generation,supervision_admitted=False,
        native_trace_summary=dict(check_statuses=dict(native_statuses),backends=dict(backends),
            obligations=dict(obligations),scope='This row payload; copied transport label native evidence remains bound by native_authentication original receipt'))


def corrections(summary):
    return [
        dict(id='native_lifecycle',kind='implementation',status='unresolved_root_cause',
             evidence='Two decode cleanup unknowns and original failed attempts retained',
             acceptance='Reproduce nested cleanup under finite deadlines; prove descendant cleanup and receipt persistence without replay.'),
        dict(id='label_transport',kind='implementation',status='focused_correction_validated',
             evidence='Collection14 retained a published label exceeding16MiB; slices15/16 use hash receipts',
             acceptance='File/hash transport must round-trip payloads beyond16MiB without truncation; retain original native obligations.'),
        dict(id='generation_bundle_support',kind='unsupported_capability',status='unresolved',
             evidence=summary['statuses'],
             acceptance='Qualify composite endpoint retrieval for the declared language or predeclare unavailable scope; test clean and corrupt bundles.'),
        dict(id='label_and_query_coverage',kind='scientific_readiness',status='not_qualified_for_fitting',
             evidence=dict(usable_unique_labels=summary['usable_unique_labels'],scheduled_slots=summary['scheduled_slots']),
             acceptance='Qualify TRAIN-only generated supervision by family/parent, known policy and query masks, novelty and sampler; never use development labels for fitting.'),
        dict(id='resource_and_optimizer',kind='resource_qualification',status='unresolved',
             evidence='Single-case gradient diagnostics did not execute optimizer updates or qualify minibatch/Adam state',
             acceptance='Profile full-schema optimizer/RNG checkpoints, minibatches and generated decoding on development before final slices; preserve scientific budgets.'),
        dict(id='semantic_controls_and_query_design',kind='experimental_design',status='deferred_by_user',
             evidence='No unwanted probes in original basis; stronger observable semantic and matched generator controls remain absent',
             acceptance='Predeclare query basis including unwanted/nonvacuity obligations and matched observable-only rich-action/reference controls before held-out opening.'),
        dict(id='graph_schema',kind='implementation',status='declaration_qualified_fitting_unqualified',
             evidence='Full declared graph schema replaces data-inferred pilot contracts; no trained successor yet',
             acceptance='Bind identical full-schema contracts to every encoder/pairwise arm, checkpoint and generated pool.')]


def report(manifest_path, output):
    evidence = Evidence();m=evidence.read(binding(manifest_path))
    require(m['schema']==SCHEMA and m['native_calls']==m['scientific_rows_replayed']==0,
            'Readiness boundary differs')
    snapshot=evidence.read(m['registry_snapshot']);ledger=evidence.read(m['ledger_snapshot'])
    require(ledger['limit_worker_seconds'] is None,'Campaign ceiling changed')
    amendment=evidence.read(m['scope_amendment']);storage=evidence.read(m['storage_policy'])
    require(amendment['schema']=='exact-repair/preliminary-scope-amendment/v1','Scope amendment differs')
    plan=evidence.read(m['collection_plan']);collection.validate_schedule(plan)
    references(evidence,plan)
    sources=[];reports={};items={}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for item in m['attempts']:
        write_artifact(output/'progress.json',dict(stage='attempts',completed=len(sources),run_id=item['run']['id']))
        source=authenticate_attempt(evidence,item,ledger,snapshot);sources.append(source)
        items[item['run']['id']]=item
        if item.get('report'):
            reports[item['run']['id']]=evidence.read(item['report'])
    require(len(items)==len(m['attempts']),'Duplicate attempts')
    for source in sources:
        run=source['run']
        if run.get('superseded_by'):
            require(run['superseded_by'] in items,'Missing recovery descendant')
    byid={s['run']['id']:s for s in sources}
    rows=[]
    for runid in m['collection_runs']:
        write_artifact(output/'progress.json',dict(stage='collection',completed_cases=len(rows),run_id=runid))
        item=items[runid];source=byid[runid];saved=checkpoint(evidence,item['report'])
        require(saved['status']=='complete' and saved['plan']==m['collection_plan'] and
                saved['scheduled_rows']==saved['recorded_rows']==2 and saved['scheduled_assignment_slots']==16,
                'Collection slice denominator differs')
        original,adapter=scientific_source(evidence,item,source,m['collection_plan'])
        identity=original;recovery_payloads=[]
        if saved.get('recovery_plan'):
            require(adapter is not None and saved['recovery_plan']==item['run']['transport_recovery_plan'],
                    'Recovery plan differs')
            proof=evidence.read(adapter['evidence']);references(evidence,proof)
            provenance=evidence.read(saved['recovery_provenance']);references(evidence,provenance)
            require(provenance['scientific_calls_replayed']==0 and provenance['ownership']['signals_sent']==0
                    and provenance['ownership']['surviving_processes']==0,'Recovery ownership differs')
            require(proof['finished_case']==saved['rows'][0] and adapter['scientific_identity']==original,
                    'Recovery changed completed prefix')
            identity=canonical_hash((saved['recovery_plan'],original));recovery_payloads=proof['payloads']
        require(saved['identity']==identity,'Collection source identity differs')
        start,stop=item['run']['row_slice']
        for index,ref in enumerate(saved['rows'],start):
            write_artifact(output/'progress.json',dict(stage='case',completed_cases=len(rows),case_index=index,run_id=runid))
            row_identity=original if saved.get('recovery_plan') and index==start else identity
            rows.append(collection_case(evidence,ref,plan['cases'][index],row_identity,
                dict(plan,_recovery_payloads=recovery_payloads,_runtime=(adapter['runtime'] if adapter else source['runtime']))))
        require(stop-start==2 and saved['usable_labels']==sum(r['usable_labels'] for r in rows[-2:]),
                'Slice summary differs')
    require([r['case_id'] for r in rows]==[r['case_id'] for r in plan['cases']] and
            len(rows)==32 and len({r['parent'] for r in rows})==16,'Collection case/parent denominator differs')
    branches={}
    for kind,ids in m['prerequisite_runs'].items():
        branches[kind]=[dict(run_id=id,report=items[id]['report'],data=reports[id]) for id in ids]
    decode=branches['decode_audit'][0]['data'];label=branches['labels'][0]['data']
    require(decode['scheduled_rows']==decode['recorded_rows']==288 and label['scheduled_rows']==160,
            'Prerequisite denominator differs')
    for p,digest in evidence.read(m['decode_verified_files']).items():
        if p not in evidence.files:evidence.verify(dict(path=p,sha256=digest))
    references(evidence,label['rows'])
    for row in label['rows']:
        saved=evidence.read(row['receipt']);references(evidence,saved.get('payloads',[]))
    intended=[]
    for entry in branches['intended']:
        for ref in entry['data']['rows']:
            saved=checkpoint(evidence,ref);references(evidence,saved['payloads']);intended.append(saved)
    require(len(intended)==32 and all(r['status']=='qualified' for r in intended),
            'Intended eligibility differs')
    interpretation=evidence.read(m['probe_interpretation']);references(evidence,interpretation)
    release=evidence.read(m['release_audit'])
    original_obligations=evidence.read(m['original_larger_study_obligations'])
    query_scope=[]
    for row in plan['cases']:
        case=case_from_dict(evidence.read(row['evaluator']))
        require(case.split=='development' and case.case_id==row['case_id'], 'Query split differs')
        query_scope.append(dict(case_id=case.case_id,parent=case.structural_parent,
            query_hash=canonical_hash(case.probes),queries=len(case.probes),
            desired=sum(p.desired for p in case.probes),unwanted=sum(not p.desired for p in case.probes),
            typed_nonvacuity_conditions=sum(len(p.conditions()) for p in case.probes)))
    costs={str(Path(s['run']['completion_path']).parent):s['charge'] for s in sources}
    # Original acquisition and decoding costs stay separate and are unioned by attempt path.
    for key,cost in decode['attempt_costs'].items():
        require(ledger['attempts'][key]==cost,'Prior decode charge changed');costs[key]=cost
    summary=summarize(rows);gaps=corrections(summary)
    result=dict(schema='exact-repair/preliminary-training-readiness/v1',status='complete',
        manifest=binding(manifest_path),local_reporting_complete=True,study_complete=False,campaign_complete=False,
        fitting_eligible=False,gates_passed=False,supervision_admitted=False,heldout_cases_opened=False,
        scientific_rows_replayed=0,new_native_calls=0,api_spend_usd=0,
        collection=dict(summary=summary,rows=rows,
            by_family={f:summarize([r for r in rows if r['family']==f]) for f in sorted({r['family'] for r in rows})},
            by_parent={p:summarize([r for r in rows if r['parent']==p]) for p in sorted({r['parent'] for r in rows})}),
        semantic_query_scope=query_scope,release_audit=m['release_audit'],
        original_larger_study_obligations=m['original_larger_study_obligations'],
        original_fixed_inventory=dict(scheduled_cases=160,summaries=label['summaries'],rows=label['rows']),
        decode_diagnostic=dict(scheduled_rows=288,semantic_statuses=decode['semantic_statuses'],
            scientific_statuses=decode['scientific_statuses'],by_family=decode['by_family']),
        intended_qualification=dict(scheduled_cases=32,qualified_cases=32,scope='Original development intended theory/policy/query on recorded backend only'),
        prerequisites=branches,probe_interpretation=m['probe_interpretation'],
        attempts=sources,attempt_costs=costs,attempt_totals=totals(costs),
        cumulative_costs=totals(ledger['attempts']),ledger_snapshot=m['ledger_snapshot'],
        cost_scope='Union of listed prerequisite,collection and prior decode attempts; cumulative ledger includes all original acquisition,maintenance and other branches. This report and its preparation are charged separately.',
        corrections=gaps,scope_amendment=m['scope_amendment'],
        deferred_obligations=[dict(status='deferred_by_user',obligation=x) for x in amendment['deferred_work']],
        proposed_cluster_design=dict(status='proposal_not_frozen_or_approved',
            arms=6,seeds=[13,37,73],fitting_runs=18,encoders=['hgt','rgcn','none'],pairwise=[False,True],
            max_epochs=50,patience=10,evaluate_every=5,
            checkpoint_selection='development generated-pool verified quality/effort; freeze criterion before fitting',
            heldout_cases=64,heldout_payloads_opened=False,warm_start=False,
            acquisition='TRAIN-only after native/query/sampler/coverage qualification; separate sample-conditioned and exact labels',
            controls='Observable-only rich-action semantic,uniform,deletion,no-repair,score-greedy and matched generator/decoder controls',
            slices='Development profiling and compatible optimizer/RNG checkpoints; no campaign time cap',
            storage=storage),
        limitations=['Development generated labels never enter fitting or exact-teacher claims.',
            'Receipt integrity and45mapped fixtures do not establish research gates or learning efficiency.',
            'Unknown native calls retain partial evidence and full scheduled slots; duplicates are not independent samples.',
            'Original query basis has no unwanted probes; absence of explicit nonvacuity does not remove typed checks.',
            'Gradient probes do not establish optimizer/minibatch capacity or convergence.',
            'Held-out identities remain reserved; final cluster design requires user review.'],
        remaining_action='Authenticate actual final readiness worker then close only local training preparation; carry corrections/deferrals to consolidated preliminary report.',
        verified_file_count=len(evidence.files))
    write_artifact(output/'progress.json',dict(stage='complete',completed_cases=len(rows)))
    write_artifact(output/'report.json',result);write_artifact(output/'verified-files.json',evidence.files)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('manifest',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();report(args.manifest,args.output)
