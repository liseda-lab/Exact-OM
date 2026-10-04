"""Qualify and evaluate the frozen development incidence schedule without tuning.

Qualification probes have fresh disposable caches and separately charged time.
Partial generation is measured coverage, never adapter or semantic completeness.
"""
from __future__ import annotations

import argparse
import fcntl
import time
from collections import Counter
from pathlib import Path

from exact.repair.records import canonical_hash, read_record
from tools.repair import scaling
from tools.repair.batch import sha
from tools.repair.expanded_corpus import binding, bound
from tools.repair.expanded_profile import checkpoint, checked_checkpoint
from tools.repair.prepare import case_from_dict


def adapter_contract(item):
    """Inspect observed inputs only, preserving all supplied controls and theory."""
    from exact.repair.candidates import materialize_retrieved_endpoints
    from exact.repair.grammar import mapping_grammar
    from exact.repair.retrieval import RetrievalConfig, retrieve_vocabulary

    case = case_from_dict(bound(item['case']))
    if case.split != 'development':
        raise ValueError('Incidence qualification requires development inputs')
    original, config = case.problem, item['config']
    retrieval = retrieve_vocabulary(original, config=RetrievalConfig(
        classes_per_side=config['menu'], properties_per_side=2, endpoints_per_side=2))
    problem = materialize_retrieved_endpoints(original, retrieval)
    if read_record(problem.to_dict()) != problem:
        raise ValueError('Materialized input does not round trip')
    objects = []
    for old, obj in zip(original.objects, problem.objects, strict=True):
        if (old.object_id != obj.object_id or old.original_axioms != obj.original_axioms
                or not {c.candidate_id for c in old.candidates} <= {c.candidate_id for c in obj.candidates}):
            raise ValueError('Generation adapter changed an observed object or supplied control')
        menu = retrieval.for_object(obj.object_id)
        encoding = mapping_grammar(obj, menu.classes, menu.properties,
            source_classes=menu.source_classes, target_classes=menu.target_classes,
            source_properties=menu.source_properties, target_properties=menu.target_properties,
            max_depth=config['depth'], max_constructors=config['constructors'],
            constraint_identity=canonical_hash((original.policy, menu)))
        objects.append(dict(object_id=obj.object_id, language_hash=encoding.content_hash,
            variables=encoding.variable_count, supplied_controls=len(old.candidates)))
    return dict(case_id=case.case_id, input_hash=original.content_hash,
        retrieval_hash=canonical_hash(retrieval), objects=objects, admitted=True)


def witness_admission(schedule_path, witness_path):
    from tools.repair.scaling_audit import Evidence, checkpoint as checked
    evidence = Evidence()
    schedule = evidence.read(binding(schedule_path))
    report = checked(evidence, binding(witness_path))
    if report['schedule'] != binding(schedule_path) or len(report['rows']) != len(schedule['cases']):
        raise ValueError('Witness schedule/denominator differs')
    for item, ref in zip(schedule['cases'], report['rows'], strict=True):
        row = checked(evidence, ref)
        result = row.get('result') or {}
        if (row['case_id'] != item['case_id'] or not row['cleanup_complete']
                or row['status'] != 'complete' or not result.get('qualified')
                or result.get('input_hash') != case_from_dict(bound(item['case'])).problem.content_hash
                or result['graph_minimal_supports'] != item['expected_minimal_supports']):
            raise ValueError('Unqualified witness: retain its evaluation denominator pending review')
        for check in result['native_checks']:
            evidence.read(check['proof'])
            if not check['qualified'] or check['status'] != check['expected']:
                raise ValueError('Native witness check is not qualified')
    return schedule


def generation_contract(saved, item):
    from exact.experiments.science_health import software_failure
    if not saved['cleanup_complete'] or software_failure(saved['status'], saved.get('detail', '')):
        raise RuntimeError('Incidence generator failure: ' + saved.get('detail', ''))
    for ref in saved.get('payloads', []):
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('Qualification generation payload changed')
    pool = bound(saved['pool']) if saved.get('pool') else None
    if saved['status'] == 'complete' and pool is None:
        raise ValueError('Completed generation has no pool')
    if pool:
        original = case_from_dict(bound(item['case'])).problem
        generated = read_record(pool['input'])
        if (generated.fixed_axioms != original.fixed_axioms or generated.policy != original.policy
                or len(generated.objects) != len(original.objects)):
            raise ValueError('Generation changed the observed theory')
        for old, obj in zip(original.objects, generated.objects, strict=True):
            if (old.object_id != obj.object_id or old.original_axioms != obj.original_axioms
                    or not {c.candidate_id for c in old.candidates} <= {c.candidate_id for c in obj.candidates}):
                raise ValueError('Generation lost an observed object or supplied control')
        for obj in pool['reports']:
            if obj.get('error') and not obj['error'].startswith('TimeoutError:'):
                raise RuntimeError('Incidence generator failure: ' + obj['error'])
    return bool(pool and pool['completed_objects'] == pool['scheduled_objects']
                and all(r['status'] == 'complete' for r in pool['reports']))


def qualification_identity(schedule_path, witness_path):
    from exact.repair.study import runtime_manifest
    runtime = runtime_manifest()
    return canonical_hash((binding(schedule_path), binding(witness_path),
                           sha(__file__), sha(scaling.__file__), runtime)), runtime


def qualify(schedule_path, witness_path, output):
    from exact.repair.workers import bounded_call
    from tools.repair.fresh_evaluation import ensure_cleanup
    schedule = witness_admission(schedule_path, witness_path)
    identity, runtime = qualification_identity(schedule_path, witness_path)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    adapters, probes = [], []
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, item in enumerate(schedule['cases']):
            # Pure input/language qualification; no labels or outcomes are used.
            adapters.append(adapter_contract(item))
            case = case_from_dict(bound(item['case']))
            for method in scaling.METHODS:
                key = canonical_hash((index, method))
                row_identity = canonical_hash((identity, key))
                path, guard = output/'rows'/(key+'.json'), output/'inflight'/(key+'.json')
                saved = checked_checkpoint(path, row_identity)
                if saved is None:
                    if guard.exists():
                        raise RuntimeError('Qualification inflight owner requires reconciliation')
                    checkpoint(guard, row_identity, started_epoch=time.time())
                    payload = output/'payloads'/key
                    started = time.monotonic()
                    call = bounded_call(scaling.generate, case.problem.to_dict(), item['config'],
                        method, schedule, str(payload), str(payload/'cache'),
                        timeout=schedule['generation_seconds'], cpu_seconds=600, memory_mb=8192)
                    pool = payload/'pool-progress.json'
                    saved = checkpoint(path, row_identity, case_index=index, method=method,
                        status=call.status, detail=call.detail, cleanup_complete=call.cleanup_complete,
                        resources=dict(call.resource_usage), elapsed_seconds=time.monotonic()-started,
                        pool=binding(pool) if pool.exists() else None,
                        payloads=[binding(p) for p in sorted(payload.rglob('*'))
                                  if p.is_file() and p.suffix != '.lock'])
                    ensure_cleanup(call)
                    guard.unlink()
                if guard.exists():
                    raise RuntimeError('Qualification cleanup requires reconciliation')
                complete = generation_contract(saved, item)
                probes.append(dict(binding(path), generation_complete=complete,
                    status=saved['status'], case_index=index, method=method))
                checkpoint(output/'progress.json', identity, recorded=len(probes),
                    scheduled=len(schedule['cases'])*len(scaling.METHODS))
        return checkpoint(output/'report.json', identity, status='complete', admitted=True,
            schedule=binding(schedule_path), witness=binding(witness_path), runtime=runtime,
            adapters=adapters, probes=probes, scheduled=len(probes),
            statuses=dict(Counter(p['status'] for p in probes)),
            complete_generation_probes=sum(p['generation_complete'] for p in probes),
            evaluation_rows=len(schedule['rows']), gates_passed=False, study_complete=False,
            scope='Input/language compatibility and failure-visible bounded generation. '
                  'Timeout/partial generation retained; admission does not assert complete generation, '
                  'semantic quality, independent parents or research gates. Probe caches never reused.')


def evaluate(schedule_path, witness_path, qualification_path, output, start, stop):
    schedule = witness_admission(schedule_path, witness_path)
    identity, _ = qualification_identity(schedule_path, witness_path)
    report = checked_checkpoint(Path(qualification_path), identity)
    if (report is None or report.get('status') != 'complete' or not report.get('admitted')
            or len(report['adapters']) != len(schedule['cases'])
            or len(report['probes']) != len(schedule['cases'])*len(scaling.METHODS)):
        raise ValueError('Completed source-bound generation qualification required')
    expected = {(i, m) for i in range(len(schedule['cases'])) for m in scaling.METHODS}
    seen = set()
    for ref in report['probes']:
        saved = bound(ref)
        pair = (saved['case_index'], saved['method'])
        if pair not in expected or pair in seen:
            raise ValueError('Qualification probe denominator differs')
        seen.add(pair)
        generation_contract(saved, schedule['cases'][pair[0]])
    if seen != expected:
        raise ValueError('Missing qualification probe')
    return scaling.run(schedule_path, output, start, stop, strict_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('qualify', 'evaluate'))
    parser.add_argument('schedule', type=Path)
    parser.add_argument('witness', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--start', type=int)
    parser.add_argument('--stop', type=int)
    args = parser.parse_args()
    if args.mode == 'qualify':
        qualify(args.schedule, args.witness, args.output)
    else:
        evaluate(args.schedule, args.witness, args.qualification, args.output, args.start, args.stop)
