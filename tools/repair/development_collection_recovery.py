"""Recover published collection labels and continue only untouched native calls.

This adapter runs against the original --code export. All original artifacts,
spent calls and unknown labels remain immutable; only result transport changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import fcntl
from pathlib import Path
import shutil
import sys

CAUSE = 'ValueError: worker result exceeds the transport frame limit'


def label_reference(*args):
    from tools.repair import development_collection as science
    from tools.repair.historical_regression import binding
    science.label_assignment(*args)
    return binding(Path(args[-1]) / 'label.json')


def checked_native(value, snapshot, monitored=(), required=(), prohibited=(), activated=()):
    import pyowl_core as owl
    from exact.repair.owl import CheckReport, ObligationResult, SupportReport, _class, _query_id
    classes = sorted({_class(v) for v in monitored} - {owl.OWL_NOTHING}, key=lambda v: v.iri.value)
    queries = [('consistency', None, True)]
    queries += [('class_satisfiability', v, True) for v in classes]
    queries += [('required_entailment', v, True) for v in required]
    queries += [('prohibited_entailment', v, False) for v in prohibited]
    queries += [('active_satisfiability', v, True)
                for v in sorted(set(activated), key=lambda v: v.canonical_bytes())]
    expected = [(k, _query_id(v), e) for k, v, e in queries]
    actual = [(o['kind'], o['query_id'], o['expected']) for o in value['obligations']]
    support = value['support']
    if (value['theory_hash'] != snapshot.logical_fingerprint.hex or actual != expected
            or (support['reasoner'], support['backend']) != ('hermit', 'python')
            or support['query_support'] != [[o['kind'], o['query_id'], o['complete']]
                                           for o in value['obligations']]):
        raise ValueError('Saved native theory, query or support identity differs')
    obligations = tuple(ObligationResult(**o) for o in value['obligations'])
    feasible = support['input_supported'] and all(o.satisfied is True for o in obligations)
    infeasible = support['input_supported'] and any(o.satisfied is False for o in obligations)
    if value['logical_status'] != ('VERIFIED_FEASIBLE' if feasible else
                                  'VERIFIED_INFEASIBLE' if infeasible else 'UNKNOWN'):
        raise ValueError('Saved native aggregate differs')
    return CheckReport(value['logical_status'], value['verification_scope'], value['theory_hash'],
                       obligations, SupportReport(**support))


def authenticate_label(record, assignment, profile, label_ref, payloads):
    """Recompute the label from saved native reports, without a reasoner call."""
    from exact.repair.learning import SemanticTargetSpec
    from exact.repair.records import canonical_hash
    from tools.repair import acquisition, train
    from tools.repair.prepare import case_from_dict
    from tools.repair.historical_regression import verify_binding
    directory = Path(label_ref['path']).parent
    reports = {p['path']: p for p in payloads}
    class Replay:
        def __init__(self, unused):
            self.sequence = 0
        def check_theory(self, snapshot, monitored=(), **kwargs):
            path = directory / 'native' / f'check-{self.sequence}.json'
            start = verify_binding(reports[str(path.with_name(f'check-{self.sequence}.started.json'))])
            if (start['reasoner'], start['backend']) != ('hermit', 'python'):
                raise ValueError('Native start backend differs')
            self.sequence += 1
            return checked_native(verify_binding(reports[str(path)]), snapshot, monitored, **kwargs)
    replay = Replay(None)
    original = acquisition.RecordingVerifier
    try:
        acquisition.RecordingVerifier = lambda directory: replay
        case = case_from_dict(record)
        expected = train._assignment_label(case, tuple(assignment), tuple(map(tuple, profile)),
            semantic_target=SemanticTargetSpec(canonical_hash(case.probes)), evidence_directory=directory/'native')
    finally:
        acquisition.RecordingVerifier = original
    completed = [p for p in (directory / 'native').glob('check-*.json')
                 if p.stem.removeprefix('check-').isdigit()]
    if (len(completed) != replay.sequence
            or canonical_hash(asdict(expected)) != canonical_hash(verify_binding(label_ref))):
        raise ValueError('Published label differs from saved native evidence')
    return dict(native_calls=replay.sequence, label=label_ref, reasoner_calls_repeated=0)


def validate_original(plan):
    from exact.repair.records import canonical_hash, read_record
    from tools.repair import development_collection as science
    from tools.repair.batch import checked_batch, sha
    from tools.repair.expanded_profile import checked_checkpoint
    from tools.repair.historical_regression import verify_binding as bound
    from tools.repair.prepare import case_from_dict, case_to_dict
    evidence = bound(plan['evidence'])
    for ref in evidence['payloads']:
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('Original payload changed')
    batch = checked_batch(plan['frozen_batch']['path'])
    if batch != bound(plan['frozen_batch']) or batch['code'] != plan['source_root']:
        raise ValueError('Original frozen batch changed')
    schedule = bound(plan['schedule'])
    science.validate_schedule(schedule)
    completion, step, command = (bound(evidence[k]) for k in ('completion', 'step', 'command'))
    start, stop = plan['slice']
    if stop - start != 2 or plan['failed_index'] != start + 1 or plan['prior_costs_reset'] is not False:
        raise ValueError('Recovery requires one finished case and one partially labeled case')
    job = next(j for j in batch['jobs'] if j['id'] == completion['job_id'])
    argv = [[a.replace('{python}', batch['python']).replace('{work}', completion['work'])
             for a in cmd] for cmd in job['commands']]
    if (completion['status'] != 'failed' or completion['exit_code'] != 1
            or completion['error']['message'] != 'command 0 exited 1: RuntimeError: Acquisition software failure: ' + CAUSE
            or completion['step_id'] != plan['original_step']
            or step != dict(step_id=completion['step_id'], dispatch_nonce=completion['dispatch_nonce'])
            or completion['batch'] != plan['frozen_batch']['path']
            or command != dict(argv=argv[0], cwd=plan['source_root']) or len(argv) != 1
            or Path(argv[0][3]).resolve() != Path(plan['schedule']['path'])
            or argv[0][5:] != ['--start', str(start), '--stop', str(stop)]):
        raise ValueError('Original nonce, command, error or slice differs')
    original_output = Path(completion['work']) / 'collection'
    identity = canonical_hash((plan['schedule'],
                               sha(science.__file__), plan['runtime']))
    if identity != plan['scientific_identity']:
        raise ValueError('Original scientific identity changed')
    first, failed = schedule['cases'][start:stop]
    def directory(row):
        return original_output / 'cases' / canonical_hash((row['case_id'], row['case_hash']))
    first_dir, failed_dir = directory(first), directory(failed)
    finished = bound(evidence['finished_case'])
    if (Path(evidence['finished_case']['path']) != first_dir/'completion.json'
            or finished != checked_checkpoint(first_dir/'completion.json', canonical_hash((identity, first)))
            or (first_dir/'inflight.json').exists() or not finished['cleanup_complete']):
        raise ValueError('Finished case identity or ownership changed')
    for ref in finished['payloads']:
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('Finished payload changed')
    guard = bound(evidence['guard'])
    if (Path(evidence['guard']['path']) != failed_dir/'inflight.json'
            or guard['identity'] != canonical_hash((identity, failed)) or guard['case_id'] != failed['case_id']
            or (failed_dir/'completion.json').exists() or (failed_dir/'label-progress.json').exists()):
        raise ValueError('Interrupted case ownership or progress differs')
    generation = bound(evidence['generation_call'])
    if (generation['status'] != 'complete' or not generation['cleanup_complete']
            or generation['deadline_seconds'] != schedule['budget']['generation_seconds']):
        raise ValueError('Original generation did not complete')
    pool, sample, record = (bound(evidence[k]) for k in ('pool','sample','generated_case'))
    original = read_record(bound(failed['observable']))
    problem = read_record(pool['input'])
    case = replace(case_from_dict(bound(failed['evaluator'])), problem=problem)
    if (sample != science.sample_assignments(problem, original, schedule['seed'])
            or canonical_hash(record) != canonical_hash(case_to_dict(case)) or case.split != 'development'):
        raise ValueError('Generated inventory, sampler or evaluator changed')
    label = bound(evidence['label'])
    call = bound(evidence['label_call'])
    if (call['status'] != 'error' or call['detail'] != CAUSE or not call['cleanup_complete']
            or call['deadline_seconds'] != schedule['budget']['assignment_seconds']
            or label['assignment'] != sample['rows'][0]['assignment']
            or Path(evidence['label']['path']).parent != failed_dir/'labels'/canonical_hash(tuple(label['assignment']))
            or Path(evidence['label_call']['path']) != Path(evidence['label']['path']).with_name('call.json')):
        raise ValueError('Only the first published transport-failed label can be recovered')
    untouched = sorted({tuple(s['assignment']) for s in sample['rows']} - {tuple(label['assignment'])})
    if [list(a) for a in untouched] != plan['untouched_assignments']:
        raise ValueError('Untouched assignment denominator differs')
    if any((failed_dir/'labels'/canonical_hash(a)).exists() for a in untouched):
        raise ValueError('Continuation assignment already spent work')
    if any(Path(evidence[k]['path']) != failed_dir/name for k,name in [
            ('generation_call','generation-call.json'),('pool','pool.json'),
            ('sample','sample.json'),('generated_case','generated-case.json')]):
        raise ValueError('Original artifact path differs')
    profile = sorted(bound(schedule['protocol'])['objective']['edit_weights'].items())
    audit = authenticate_label(record, label['assignment'], profile, evidence['label'], evidence['payloads'])
    return evidence, schedule, failed, record, profile, audit


class ReuseCalls:
    """Rebuild receipts in a separate directory; never repeat a spent call."""
    def __init__(self, science, evidence, record, profile, untouched, execute):
        self.science, self.evidence, self.record, self.profile = science, evidence, record, profile
        self.untouched = {tuple(a) for a in untouched}
        self.execute, self.started = execute, set()

    def __call__(self, function, *args, **limits):
        from exact.repair.api import write_artifact
        from exact.repair.records import canonical_hash
        from exact.repair.workers import CallResult
        from tools.repair.historical_regression import binding, verify_binding as bound
        if function is self.science.generate:
            original = bound(self.evidence['generation_call'])
            destination = Path(args[-1])
            shutil.copyfile(self.evidence['pool']['path'], destination/'pool.json')
            write_artifact(destination/'generation-recovery.json', dict(original_call=self.evidence['generation_call'],
                original_pool=self.evidence['pool'], additional_scientific_seconds=0, prior_costs_reset=False))
            return CallResult('complete', binding(destination/'pool.json'),
                detail='Retained original generation without replay', resource_usage=tuple(original['resource_usage'].items()))
        if function is not self.science.label_assignment or canonical_hash(args[0]) != canonical_hash(self.record) or list(args[2]) != self.profile:
            raise ValueError('Unexpected scientific call during recovery')
        assignment = tuple(args[1])
        original_label = bound(self.evidence['label'])
        if assignment == tuple(original_label['assignment']):
            destination = Path(args[-1]); destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.evidence['label']['path'], destination/'label.json')
            write_artifact(destination/'transport-recovery.json', dict(original_label=self.evidence['label'],
                original_call=self.evidence['label_call'], additional_scientific_seconds=0, prior_costs_reset=False))
            call = bound(self.evidence['label_call'])
            return CallResult('complete', binding(destination/'label.json'),
                detail='Published native label authenticated; original transport error retained',
                resource_usage=tuple(call['resource_usage'].items()))
        if assignment not in self.untouched or assignment in self.started:
            raise ValueError('Refusing repeated or unregistered assignment')
        if limits != dict(timeout=300, cpu_seconds=600, memory_mb=8192):
            raise ValueError('Scientific call budget changed')
        self.started.add(assignment)
        return self.execute(label_reference, *args, **limits)


def run(plan_path, output, *, owner_check=None, preflight=False):
    from exact.repair.api import write_artifact
    from exact.repair.records import canonical_hash
    from exact.repair.study import runtime_manifest
    from tools.repair import development_collection as science
    from tools.repair.batch import read
    from tools.repair.expanded_profile import checkpoint
    from tools.repair.historical_regression import binding, verify_binding as bound
    from tools.repair.schema_cleanup_recovery import confirm_owner_gone
    plan, output = read(plan_path), Path(output).resolve()
    if (binding(__file__) != plan['runner'] or str(output) != plan['output']
            or Path(science.__file__).resolve().parents[2] != Path(plan['source_root']).resolve()
            or runtime_manifest() != plan['runtime']):
        raise ValueError('Frozen adapter, source, runtime or output changed')
    if plan.get('mode') == 'unstarted_slice':
        return run_unstarted(plan, plan_path, output, preflight=preflight)
    evidence, schedule, row, record, profile, audit = validate_original(plan)
    owner = (owner_check or confirm_owner_gone)(plan['original_step'])
    if (owner['step_id'] != plan['original_step'] or owner['surviving_processes'] != 0
            or not owner['cgroup_absent'] or owner['signals_sent'] != 0):
        raise ValueError('Original ownership remains unresolved')
    if preflight:
        return dict(status='validated', native_label_audit=audit, ownership=owner,
            retained_finished_cases=1, retained_published_labels=1,
            untouched_assignments=plan['untouched_assignments'], scientific_calls_replayed=0)
    original_work = Path(bound(evidence['completion'])['work']).resolve()
    if output.parent != original_work or output == original_work/'collection':
        raise ValueError('Recovery requires separate output in original logical work')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = canonical_hash((binding(plan_path), plan['scientific_identity']))
        write_artifact(output/'recovery-provenance.json', dict(plan=binding(plan_path), evidence=plan['evidence'],
            native_label_audit=audit, ownership=owner, scientific_calls_replayed=0))
        original = science.bounded_call
        try:
            science.bounded_call = ReuseCalls(science, evidence, record, profile,
                                              plan['untouched_assignments'], original)
            revised = science.one_case(row, output, identity, schedule)
        finally:
            science.bounded_call = original
        rows = [evidence['finished_case'], revised]
        saved = [bound(r) for r in rows]
        return checkpoint(output/'report.json', identity, schema=science.SCHEMA, status='complete',
            plan=plan['schedule'], recovery_plan=binding(plan_path), original_error_retained=True,
            scheduled_rows=2, recorded_rows=2, rows=rows, counts=dict(Counter(r['status'] for r in saved)),
            study_scheduled_cases=32, study_parent_groups=16, study_assignment_slots=256,
            scheduled_assignment_slots=16, usable_labels=sum(r['result']['usable_labels'] for r in saved),
            scientific_calls_replayed=0, prior_costs_reset=False, fitting_labels_admitted=False,
            fitting_eligible=False, gates_passed=False, heldout_cases_opened=False,
            next_stage='xr21-expanded-training-001', recovery_provenance=binding(output/'recovery-provenance.json'),
            limitations=['Development labels are sample-conditioned, not fitting data or exact teachers.',
                'All unknowns, duplicate slots, original errors and costs retained.',
                'Training acquisition, capacity, eighteen protocols, selection, held-out controls and final report remain.'])


def run_unstarted(plan, plan_path, output, *, preflight=False):
    from tools.repair import development_collection as science
    from tools.repair.batch import checked_batch
    from tools.repair.historical_regression import binding, verify_binding as bound
    from exact.repair.api import write_artifact
    batch = checked_batch(plan['frozen_batch']['path'])
    if batch != bound(plan['frozen_batch']) or batch['code'] != plan['source_root']:
        raise ValueError('Original frozen batch changed')
    schedule = bound(plan['schedule'])
    science.validate_schedule(schedule)
    start, stop = plan['slice']
    if not 28 <= start < stop <= 32 or stop-start != 2:
        raise ValueError('Transport update covers only the two untouched final slices')
    if preflight:
        return dict(status='validated', slice=plan['slice'], scientific_budgets_changed=False)
    original = science.bounded_call
    def transport(function, *args, **limits):
        result = original(label_reference if function is science.label_assignment else function, *args, **limits)
        if function is science.label_assignment and result.status == 'complete':
            if result.value != binding(Path(args[-1])/'label.json'):
                raise ValueError('Transported label receipt differs')
        return result
    try:
        science.bounded_call = transport
        report = science.run(plan['schedule']['path'], output, start, stop)
    finally:
        science.bounded_call = original
    write_artifact(output/'transport-provenance.json', dict(adapter_plan=binding(plan_path),
        original_source=plan['source_root'], scientific_budgets_changed=False,
        scientific_calls_replayed=0, report=binding(output/'report.json')))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    result = run(args.plan, args.output, preflight=args.preflight)
    if args.preflight:
        import json
        print(json.dumps(result))
