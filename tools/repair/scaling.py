"""Development resource sweeps with matched finite languages and charged caches.

Padding varies benign mapping load, not conflict incidence. Every variant keeps
its profiled parent and development split. No trained model or held-out outcome
is used to choose bounds. Full semantic quality and qualification remain audits.
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import fcntl
import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import pyowl_core as owl

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, canonical_json, read_record, replace_inventory
from tools.repair.batch import read, sha
from tools.repair.expanded_corpus import binding, bound, immutable
from tools.repair.expanded_profile import checkpoint, checked_checkpoint, parent_fingerprints
from tools.repair.prepare import case_from_dict, case_to_dict

METHODS = ('grammar_circuit', 'semantic_circuit', 'semantic_enumeration_decoder')


def configurations():
    base = dict(mapping_count=4, depth=1, constructors=1, menu=2, candidate_cap=16)
    return [dict(id='baseline', **base)] + [
        dict(base, id=f'{key}-{value}', **{key: value})
        for key, values in (('mapping_count', (8, 16)), ('depth', (2,)),
                            ('constructors', (2,)), ('menu', (4, 8)),
                            ('candidate_cap', (32, 64))) for value in values
    ] + [dict(base, id='depth2-constructors2', depth=2, constructors=2)]


def extend_case(case, count):
    """Distinct benign mappings add load without inventing independent parents."""
    from exact.repair.candidates import mapping_candidates
    from exact.repair.records import RevisionObjectV3, promote_input_v3, freeze_public_policy

    if case.split != 'development' or case.schema_revision != 'v3':
        raise ValueError('Scaling accepts native development parents only')
    if any(o.kind != 'mapping' for o in case.problem.objects) or len(case.problem.objects) > count:
        raise ValueError('Base must contain no more than the requested mapping count')
    objects = list(case.problem.objects)
    sources, targets = list(case.problem.source_axioms), list(case.problem.target_axioms)
    added_axioms = []
    for index in range(len(objects), count):
        prefix = 'urn:exact:scaling:' + canonical_hash(case.structural_parent) + f':padding{index}'
        source, target = (owl.Class(owl.IRI(prefix + side)) for side in ('_s', '_t'))
        object_id = f'scaling-padding-{index}'
        candidates = mapping_candidates(object_id, source, target, '<')
        original = (owl.SubClassOf(source, target),)
        objects.append(RevisionObjectV3(object_id, 'mapping', original, candidates,
                                       source_entity=source, target_entity=target, authorship='generated'))
        sources.append(owl.Declaration(source)); targets.append(owl.Declaration(target))
        added_axioms.extend(original)
    problem = replace_inventory(case.problem, tuple(objects))
    problem = dc.replace(problem, source_axioms=tuple(sources), target_axioms=tuple(targets))
    choices = list(case.intended_assignment)
    for obj in problem.objects[len(choices):]:
        choices.append(next(i for i, c in enumerate(obj.candidates) if set(c.axioms) == set(obj.original_axioms)))
    return dc.replace(case, case_id=case.case_id + f':scaling-m{count}', problem=problem,
        intended_assignment=tuple(choices), intended_theory=(*case.intended_theory, *added_axioms),
        variation=(*case.variation, ('scaling_mapping_count', count), ('padding', 'benign_disconnected')))


def prepare(corpus_attempt, profile_schedule, protocol_binding, output):
    from tools.repair.fresh_evaluation import validate_corpus

    report, _ = validate_corpus(corpus_attempt)
    profile = bound(profile_schedule)
    bound(protocol_binding)
    # Predeclared family and both profile parents/controls; never choose on timing or quality.
    selected = [r for r in profile['cases'] if r['family'] == 'papers']
    if len(selected) != 4 or Counter(r['control'] for r in selected) != {'coherent': 2, 'corrupted': 2}:
        raise ValueError('Expected both papers profile parents and their controls')
    cases, rows = [], []
    for item in selected:
        original = case_from_dict(bound(item['case']))
        for config in configurations():
            case = extend_case(original, config['mapping_count'])
            if set(parent_fingerprints(case)) != set(parent_fingerprints(original)):
                raise ValueError('Padding altered the connected parent fingerprint')
            record = case_to_dict(case)
            case_binding = immutable(Path(output)/'cases'/(canonical_hash(record)+'.json'), record)
            index = len(cases)
            cases.append(dict(case=case_binding, parent=case.structural_parent, split=case.split,
                base_case=item['case'], control=case.control, config=config,
                fingerprints=parent_fingerprints(case), case_id=case.case_id))
            for method in METHODS:
                pair = canonical_hash((index, method))
                for mode in ('cold', 'warm'):
                    row = dict(case_index=index, method=method, cache_mode=mode, pair_id=pair,
                               seconds=300, cpu_seconds=600, memory_mb=8192)
                    row['id'] = canonical_hash(row)
                    rows.append(row)
    return dict(schema='exact-repair/scaling/v1', corpus_attempt=corpus_attempt,
        corpus_completion=corpus_attempt['report'], releases=report['releases'],
        profile_schedule=profile_schedule, protocol=protocol_binding, cases=cases, rows=rows,
        scheduled_rows=len(rows), generation_seconds=60, seed=13, draws_per_object=32,
        compiler_nodes=100000, compiler_call_seconds=10, compiler_object_seconds=20,
        compiler_rss_mb=4096, max_expressions=10000, context_checks=128,
        test_outcomes_opened=False, warm_start=False, api_spend_usd=0,
        cache_policy='Cold has an empty per-pair cache; warm copies only its charged cold predecessor cache. Copy and loading count inside warm deadline. No implicit warmup.',
        quality_scope='Native verified core probe benefit and whole-plan edit cost; padding adds benign mappings, not additional quality targets or independent parents.',
        objective='Identical negative declared edit cost, complete native verifier/cuts; no neural fitting or semantic-optimality claim.',
        comparison='Semantic circuit and enumeration use identical immutable-context proofs and language, draws and candidate caps. Grammar circuit omits contextual pruning. Every method retains supplied elementary controls.',
        conditional_depth3='Separate development viability review after these receipts; never automatically promoted, and never based on held-out outcomes.',
        remaining=['receipt_and_matched_language_audit', 'depth3_development_viability_decision',
                   'support_incidence_scaling_design', 'failure_inclusive_resource_report'],
        study_complete=False, gates_passed=False)


def generate(problem_record, config, method, schedule, directory, cache):
    """Publish incremental pools so a generation timeout retains completed work."""
    import torch
    from exact.repair.candidates import materialize_retrieved_endpoints, deduplicate_candidates
    from exact.repair.circuit import FactoredConditionedMixture
    from exact.repair.grammar import mapping_grammar, compile_families, with_immutable_context
    from exact.repair.proposals import enumerate_grammar
    from exact.repair.retrieval import RetrievalConfig, retrieve_vocabulary

    torch.set_num_threads(1)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    original = read_record(problem_record)
    retrieval = retrieve_vocabulary(original, config=RetrievalConfig(
        classes_per_side=config['menu'], properties_per_side=2, endpoints_per_side=2))
    problem = materialize_retrieved_endpoints(original, retrieval)
    objects = list(problem.objects)
    reports = []
    for index, obj in enumerate(problem.objects):
        menu = retrieval.for_object(obj.object_id)
        encoding = mapping_grammar(obj, menu.classes, menu.properties,
            source_classes=menu.source_classes, target_classes=menu.target_classes,
            source_properties=menu.source_properties, target_properties=menu.target_properties,
            max_depth=config['depth'], max_constructors=config['constructors'],
            constraint_identity=canonical_hash((original.policy, menu)))
        if method != 'grammar_circuit':
            encoding = with_immutable_context(encoding, original.fixed_axioms, original.policy,
                                              max_checks=schedule['context_checks'])
        details = dict(object_id=obj.object_id, language_hash=encoding.content_hash,
            variables=encoding.variable_count, context_checks=encoding.contextual_checks,
            context_truncated=encoding.contextual_truncated,
            context_proofs=[p.content_hash for p in encoding.context_proofs],
            effective_menu=json.loads(canonical_json(menu)), status='complete')
        sampled = []
        try:
            remaining = schedule['generation_seconds'] - (time.monotonic()-started) - 1
            if remaining <= 0:
                raise TimeoutError('Aggregate generation allowance exhausted')
            if method == 'semantic_enumeration_decoder':
                enumeration = enumerate_grammar(encoding, max_expressions=schedule['max_expressions'],
                                                deadline=time.monotonic()+remaining)
                candidates = enumeration.candidates
                weights = [len(encoding.candidate_assignments(c)) for c in candidates]
                sampled = random.Random(schedule['seed']).choices(candidates, weights=weights,
                            k=schedule['draws_per_object']) if candidates else []
                details['enumerated_candidates'] = len(candidates)
            else:
                compiled = compile_families(encoding, max_nodes=schedule['compiler_nodes'],
                    cache_directory=str(cache), circuit_limits=dict(
                        call_seconds=schedule['compiler_call_seconds'],
                        aggregate_seconds=min(schedule['compiler_object_seconds'], remaining),
                        rss_mb=schedule['compiler_rss_mb']))
                distribution = FactoredConditionedMixture(compiled, torch.zeros((4, encoding.variable_count)))
                samples = distribution.sample(schedule['draws_per_object'], seed=schedule['seed'])
                sampled = [encoding.decode(sample.assignment) for sample in samples]
                details['families'] = [dict(name=f.name, status=f.status,
                    telemetry=dict(f.circuit.telemetry) if f.circuit else None) for f in compiled.families]
                details['compile_seconds'] = compiled.compilation_seconds
                if any(f.status not in {'resolved','empty_language'} for f in compiled.families):
                    details['status'] = 'partial'
        except (ValueError, TimeoutError, RuntimeError) as error:
            details.update(status='unresolved', error=type(error).__name__+': '+str(error))
        # Cap newly sampled candidates; supplied controls are always retained and counted.
        sampled = sorted(deduplicate_candidates(sampled), key=lambda c:c.candidate_id)[:config['candidate_cap']]
        candidates = deduplicate_candidates((*obj.candidates, *sampled))
        objects[index] = dc.replace(obj, candidates=candidates)
        details.update(sampled_unique=len(sampled), supplied=len(obj.candidates), pool_size=len(candidates))
        reports.append(details)
        write_artifact(directory/'pool-progress.json', dict(input=replace_inventory(problem, tuple(objects)).to_dict(),
            reports=reports, completed_objects=len(reports), scheduled_objects=len(objects),
            retrieval_hash=canonical_hash(retrieval)))
    return binding(directory/'pool-progress.json')


def evaluate(schedule, row, output, cache_source=None):
    from exact.repair.workers import bounded_call
    from exact.repair.kernel import repair
    from exact.repair.records import make_objective
    from exact.repair.learning import SemanticTargetSpec
    from tools.repair.train import _assignment_label
    from tools.repair.fresh_evaluation import ensure_cleanup

    started = time.monotonic()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    item = schedule['cases'][row['case_index']]
    case = case_from_dict(bound(item['case']))
    protocol = bound(schedule['protocol'])
    cache = output/'compiler-cache'
    if cache_source is not None and Path(cache_source).exists():
        shutil.copytree(cache_source, cache)
    generation = bounded_call(generate, case.problem.to_dict(), item['config'], row['method'],
        schedule, str(output), str(cache), timeout=min(schedule['generation_seconds'],
        max(.001,row['seconds']-(time.monotonic()-started)-1)),
        cpu_seconds=row['cpu_seconds'], memory_mb=row['memory_mb'])
    ensure_cleanup(generation)
    pool_path = output/'pool-progress.json'
    pool = read(pool_path) if pool_path.exists() else None
    problem = read_record(pool['input']) if pool else case.problem
    result = dict(generation_status=generation.status, generation_detail=generation.detail,
        generation_resources=dict(generation.resource_usage), pool=binding(pool_path) if pool else None,
        generation_complete=bool(pool and pool['completed_objects']==pool['scheduled_objects']
            and all(r['status']=='complete' for r in pool['reports'])),
        fallback_original_inventory=pool is None, logical_status='UNKNOWN', semantic_status='not_evaluated',
        semantic_benefit=None, cache_source=cache_source, native_scope='complete backend on recorded pool')
    remaining = row['seconds']-(time.monotonic()-started)-65
    if remaining > 0:
        profile = tuple(sorted(protocol['objective']['edit_weights'].items()))
        selection = protocol['selection']
        problem = dc.replace(problem, budgets=dc.replace(problem.budgets, total_seconds=remaining,
            memory_mb=row['memory_mb'], verification_seconds=protocol['resources']['verification_seconds'],
            max_checks=selection['max_candidate_checks'], max_solves=selection['max_master_solves'],
            retries=selection['retry_budget']))
        objective = make_objective(problem.objects, profile=profile, scale=protocol['objective']['integer_scale'])
        solved = bounded_call(repair, problem, objective, preserve_verified_input=False, diagnose=True,
            ledger_path=output/'search-ledger.json', timeout=remaining,
            memory_mb=row['memory_mb'], cpu_seconds=row['cpu_seconds'])
        ensure_cleanup(solved)
        result.update(selection_status=solved.status, selection_detail=solved.detail,
                      selection_resources=dict(solved.resource_usage))
        if solved.status == 'complete':
            value = solved.value
            write_artifact(output/'repair.json', value.to_dict())
            result.update(logical_status=value.logical_status, certification=value.search_status,
                checks=value.checks, solves=value.solves, first_verified_seconds=value.first_verified_seconds,
                lower_bound=value.lower_bound, upper_bound=value.upper_bound,
                stage_seconds=value.stage_seconds, verification_scope=value.verification_scope)
            remaining = row['seconds']-(time.monotonic()-started)-3
            if value.assignment is not None and value.logical_status=='VERIFIED_FEASIBLE' and remaining>0:
                if value.verification is None or not value.verification.authorizes:
                    raise ValueError('Missing native verification authorization')
                weights = protocol['teacher']['family_weights']
                target = SemanticTargetSpec(canonical_hash(case.probes), weights['desired'], weights['unwanted'])
                labeled = bounded_call(_assignment_label, dc.replace(case, problem=problem), value.assignment,
                    profile, semantic_target=target, timeout=remaining,
                    memory_mb=row['memory_mb'], cpu_seconds=row['cpu_seconds'])
                ensure_cleanup(labeled)
                result['semantic_status'] = labeled.status
                if labeled.status=='complete':
                    label = labeled.value
                    write_artifact(output/'selected-label.json', json.loads(canonical_json(label)))
                    result.update(semantic_status='known' if label.usable else 'unknown',
                        semantic_benefit=label.benefit, edit_cost=label.cost)
    result.update(row=row, parent=item['parent'], config=item['config'],
                  elapsed_seconds=time.monotonic()-started)
    write_artifact(output/'result.json', result)
    return binding(output/'result.json')


def validate_payloads(saved):
    """Check scaling's full row contract and every immutable payload on resume."""
    for item in saved.get('payloads', []):
        if sha(item['path']) != item['sha256']:
            raise ValueError('Completed scaling payload changed')
    if saved.get('result'):
        result = bound(saved['result'])
        if result.get('row') != saved['row']:
            raise ValueError('Scaling payload row identity differs')


def run(schedule_path, output, start, stop):
    from exact.repair.workers import bounded_call
    from exact.repair.study import runtime_manifest

    schedule = read(schedule_path)
    if schedule['schema']!='exact-repair/scaling/v1' or not 0<=start<stop<=len(schedule['rows']):
        raise ValueError('Invalid scaling schedule or slice')
    if start % 2 or stop % 2:
        raise ValueError('Cold/warm pairs must share an operational slice')
    bound(schedule['protocol'])
    runtime = runtime_manifest()
    identity = canonical_hash((binding(schedule_path), sha(__file__), runtime))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = []
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index in range(start,stop):
            row = schedule['rows'][index]; key=row['id']
            path = output/'rows'/(key+'.json'); guard=output/'inflight'/(key+'.json')
            row_identity=canonical_hash((identity,row))
            saved=checked_checkpoint(path,row_identity)
            if saved is None:
                if guard.exists():
                    raise RuntimeError('Interrupted row requires owner/ledger/cleanup reconciliation')
                cache_source=None
                if row['cache_mode']=='warm':
                    cold=schedule['rows'][index-1]
                    assert cold['pair_id']==row['pair_id'] and cold['cache_mode']=='cold'
                    receipt=checked_checkpoint(output/'rows'/(cold['id']+'.json'), canonical_hash((identity,cold)))
                    if receipt is None or not receipt['cleanup_complete']:
                        raise RuntimeError('Warm row requires completed charged cold predecessor')
                    validate_payloads(receipt)
                    cache_source=str(output/'payloads'/cold['id']/'compiler-cache')
                checkpoint(guard,row_identity,row=row,started_epoch=time.time())
                payload=output/'payloads'/key
                started=time.monotonic()
                outcome=bounded_call(evaluate,schedule,row,str(payload),cache_source,
                    timeout=row['seconds'],cpu_seconds=row['cpu_seconds'],memory_mb=row['memory_mb'])
                saved=checkpoint(path,row_identity,row=row,status=outcome.status,detail=outcome.detail,
                    result=outcome.value if outcome.status=='complete' else None,
                    cleanup_complete=outcome.cleanup_complete,resources=dict(outcome.resource_usage),
                    elapsed_seconds=time.monotonic()-started,
                    payloads=[binding(p) for p in sorted(payload.rglob('*')) if p.is_file() and p.suffix!='.lock'])
                if not outcome.cleanup_complete or 'cleanup incomplete' in outcome.detail:
                    raise RuntimeError('Worker cleanup incomplete; keep inflight guard')
                guard.unlink()
            validate_payloads(saved)
            if guard.exists() or not saved['cleanup_complete']:
                raise RuntimeError('Saved row requires cleanup reconciliation')
            rows.append(dict(binding(path),status=saved['status']))
            write_artifact(output/'progress.json',dict(recorded=len(rows),scheduled=stop-start,last_row=key))
        return checkpoint(output/'report.json',identity,schema='exact-repair/scaling-shard/v1',
            schedule=binding(schedule_path),runtime=runtime,status='complete',rows=rows,
            scheduled_rows=stop-start,recorded_rows=len(rows),counts=dict(Counter(r['status'] for r in rows)),
            study_complete=False,gates_passed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('schedule',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--start',type=int,required=True);p.add_argument('--stop',type=int,required=True)
    a=p.parse_args();run(a.schedule,a.output,a.start,a.stop)


if __name__=='__main__':
    main()
