"""Second and final cleanup recovery, preserving the first replacement's evidence.

The separately frozen adapter runs only untouched rows in the original science
export. Both failed rows stay unknown; their guards, payloads and costs survive.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib.util
from pathlib import Path
import sys
import subprocess


def prior_adapter(plan):
    from tools.repair.expanded_corpus import bound
    previous = bound(plan['previous_cleanup_plan'])
    bound_runner = previous['runner']
    from tools.repair.batch import sha
    if sha(bound_runner['path']) != bound_runner['sha256']:
        raise ValueError('Previous cleanup runner changed')
    spec = importlib.util.spec_from_file_location('frozen_previous_cleanup', bound_runner['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return previous, module


def validate_previous_export(reference):
    """The wrapper export and inner science export have distinct runtime hashes."""
    from tools.repair.expanded_corpus import bound
    batch = bound(reference)
    subprocess.run([batch['python'], '-c',
        'import sys; from tools.repair.batch import checked_batch; checked_batch(sys.argv[1])',
        reference['path']], cwd=batch['code'], check=True, capture_output=True, text=True, timeout=120)
    return batch


def validate_chain(plan, evidence, identity):
    from exact.repair.records import canonical_hash
    from tools.repair.expanded_corpus import bound
    from tools.repair.expanded_profile import checked_checkpoint
    from tools.repair import schema_recovery as science

    previous, adapter = prior_adapter(plan)
    start, stop, failed = adapter.slice_bounds(plan)
    pstart, pstop, pfailed = adapter.slice_bounds(previous)
    if (plan['repair_attempt'] != 2 or plan['max_repairs'] != 2
            or plan['unsuccessful_prior_repairs'] != 1
            or 'previous_cleanup_plan' in previous
            or (start, stop) != (pstart, pstop) or failed <= pfailed
            or any(plan[k] != previous[k] for k in
                ('source_root', 'frozen_batch', 'schema_plan', 'runtime', 'schema_identity'))):
        raise ValueError('Only the second same-source cleanup repair is admissible')
    prior_evidence = bound(previous['evidence'])
    prefix = adapter.validate_original(previous, prior_evidence, identity)
    adapter.confirm_owner_gone(previous['original_step'])
    completion, owner, command = (bound(evidence[k]) for k in ('completion','ownership','command'))
    replacement_batch = validate_previous_export(plan['previous_batch'])
    old_output = Path(previous['output'])
    expected_argv = [replacement_batch['python'], previous['runner']['path'],
        plan['previous_cleanup_plan']['path'], str(old_output), '--code', previous['source_root']]
    jobs = [j for j in replacement_batch['jobs'] if j['id'] == completion['job_id']]
    if (completion['status'] != 'failed' or completion['exit_code'] != 1
            or completion['error']['message'] != adapter.COMMAND_ERROR
            or completion['step_id'] != plan['original_step']
            or completion['step_id'] == previous['original_step']
            or completion['batch'] != plan['previous_batch']['path']
            or owner != dict(step_id=completion['step_id'],dispatch_nonce=completion['dispatch_nonce'])
            or Path(evidence['command']['path']) != Path(evidence['completion']['path']).parent/'command-1.json'
            or command != dict(argv=expected_argv,cwd=replacement_batch['code'])
            or len(jobs) != 1 or jobs[0]['commands'][1] != ['{python}', *expected_argv[1:]]
            or old_output.parent != Path(completion['work'])
            or Path(plan['output']).parent != old_output.parent
            or Path(plan['output']) in (old_output, Path(completion['work'])/'evaluation')):
        raise ValueError('Previous replacement command, owner, batch or output differs')
    # Verify the already-published unknown against the full original proof.
    revised = old_output/'reconciled-unknown.json'
    if evidence['previous_unknown']['path'] != str(revised):
        raise ValueError('Original reconciled unknown path differs')
    saved = bound(evidence['previous_unknown'])
    expected_identity = canonical_hash((plan['previous_cleanup_plan'], identity))
    if checked_checkpoint(revised,expected_identity) != saved:
        raise ValueError('Original reconciled unknown identity differs')
    expected = adapter.reconciled_unknown(prefix[pfailed][1],bound(prior_evidence['guard']),
        prefix[pfailed][0],previous['evidence'],saved['ownership'],previous['original_step'])
    if {k:v for k,v in saved.items() if k not in ('identity','content_hash')} != expected:
        raise ValueError('Original reconciled unknown changed')
    prefix[pfailed] = (evidence['previous_unknown'], saved)
    schema_plan = bound(plan['schema_plan']); schedule = bound(schema_plan['schedule'])
    admission = {r['row_id']:r for r in bound(schema_plan['preflight'])['rows']}
    work = old_output/'continuation'
    for item in evidence['rows']:
        index, ref = item['index'], item['receipt']
        if index not in range(pfailed+1,failed+1) or index in prefix:
            raise ValueError('Replacement retained prefix differs')
        row, entry = schedule['rows'][index], schema_plan['rows'][index]
        expected = canonical_hash((identity,row,entry,admission[row['id']]))
        path = work/'rows'/(row['id']+'.json'); saved=bound(ref)
        if (ref['path'] != str(path) or saved != checked_checkpoint(path,expected)
                or saved['row'] != row or not saved['cleanup_complete']):
            raise ValueError('Replacement row identity or outer cleanup differs')
        science.fresh.validate_payloads(saved)
        guard = work/'inflight'/(row['id']+'.json')
        if index == failed:
            if (str(guard) != evidence['guard']['path']
                    or checked_checkpoint(guard,expected) != bound(evidence['guard'])):
                raise ValueError('Replacement failed guard differs')
        else:
            if guard.exists():
                raise ValueError('Replacement finished row has an unresolved guard')
            science.fresh.raise_on_software_failure(saved)
        prefix[index]=(ref,saved)
    if set(prefix) != set(range(start,failed+1)):
        raise ValueError('Every row from both attempts must be retained')
    for index in range(failed+1,stop):
        row_id = schedule['rows'][index]['id']
        if any(p.exists() for p in (work/'rows'/(row_id+'.json'),
                work/'inflight'/(row_id+'.json'),work/'payloads'/row_id)):
            raise ValueError('Continuation row already spent work in the replacement')
    owner_now=adapter.confirm_owner_gone(plan['original_step'])
    # Qualify the new unknown before starting any native work.
    adapter.reconciled_unknown(prefix[failed][1],bound(evidence['guard']),prefix[failed][0],
        plan['evidence'],owner_now,plan['original_step'])
    return prefix,owner_now,adapter


def run(plan_path, output):
    from exact.repair.records import canonical_hash
    from exact.repair.study import runtime_manifest
    from tools.repair.batch import read,sha,checked_batch
    from tools.repair.expanded_corpus import binding,bound
    from tools.repair.expanded_profile import checked_checkpoint,checkpoint
    from tools.repair import schema_recovery as science

    plan,output=read(plan_path),Path(output).resolve()
    if (binding(__file__) != plan['runner'] or str(output) != plan['output']
            or Path(science.__file__).resolve().parents[2] != Path(plan['source_root']).resolve()
            or runtime_manifest() != plan['runtime']):
        raise ValueError('Frozen runner, source, runtime or output changed')
    identity=canonical_hash((plan['schema_plan']['sha256'],sha(science.__file__),sha(science.fresh.__file__),plan['runtime']))
    if identity != plan['schema_identity']:
        raise ValueError('Original scientific identity changed')
    frozen=checked_batch(plan['frozen_batch']['path'])
    if frozen != bound(plan['frozen_batch']) or Path(frozen['code']).resolve() != Path(plan['source_root']).resolve():
        raise ValueError('Original science export changed')
    evidence=bound(plan['evidence'])
    output.mkdir(parents=True,exist_ok=True)
    with (output/'recovery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prefix,ownership,adapter=validate_chain(plan,evidence,identity)
        start,stop,failed=adapter.slice_bounds(plan)
        recovery_identity=canonical_hash((binding(plan_path),identity))
        revised=output/'reconciled-unknown.json'
        expected=adapter.reconciled_unknown(prefix[failed][1],bound(evidence['guard']),
            prefix[failed][0],plan['evidence'],ownership,plan['original_step'])
        saved=checked_checkpoint(revised,recovery_identity)
        if saved:
            expected['ownership']=saved['ownership']
            if {k:v for k,v in saved.items() if k not in ('identity','content_hash')} != expected:
                raise ValueError('Second reconciled unknown changed')
        else:
            saved=checkpoint(revised,recovery_identity,**expected)
        continuation=science.run(Path(plan['schema_plan']['path']),output/'continuation',failed+1,stop)
        if continuation['scheduled'] != stop-failed-1 or continuation['recorded'] != stop-failed-1:
            raise ValueError('Continuation denominator changed')
        refs=[dict(**prefix[i][0],row_id=prefix[i][1]['row']['id'],status=prefix[i][1]['status']) for i in range(start,failed)]
        refs += [dict(**binding(revised),row_id=saved['row']['id'],status=saved['status'])]
        refs += continuation['rows']
        schema_plan=bound(plan['schema_plan']);schedule=bound(schema_plan['schedule'])
        if [r['row_id'] for r in refs] != [r['id'] for r in schedule['rows'][start:stop]]:
            raise ValueError('Full original denominator changed')
        return checkpoint(output/'report.json',recovery_identity,
            schema='exact-repair/schema-recovery/v1',status='complete',plan=plan['schema_plan'],
            schedule=schema_plan['schedule'],rows=refs,scheduled=stop-start,recorded=len(refs),
            outcomes=dict(Counter(r['status'] for r in refs)),gates_passed=False,study_complete=False,
            prior_costs_reset=False,checkpoint_parameters_changed=False,
            followup=schedule.get('followup','xr21-expanded-evaluation-001'),
            cleanup_recovery=dict(plan=binding(plan_path),evidence=plan['evidence'],
                previous_cleanup_plan=plan['previous_cleanup_plan'],repair_attempt=2,max_repairs=2,
                retained_finished_rows=failed-start-1,retained_unknown_rows_without_replay=2,
                continued_rows=stop-failed-1,original_errors_retained=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--code',type=Path,required=True);args=parser.parse_args()
    sys.path.insert(0,str(args.code.resolve()))
    run(args.plan,args.output)
