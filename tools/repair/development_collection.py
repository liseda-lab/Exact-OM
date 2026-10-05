"""Bounded development generated-pool sampling and external label acquisition.

This resource experiment profiles acquisition before training-side collection.
Development caches are never fitting inputs or exact whole-inventory teachers.
"""
from __future__ import annotations

import argparse
import fcntl
import math
import random
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.learning import RepairLabel, SemanticTargetSpec
from exact.repair.records import canonical_hash, canonical_json, read_record
from exact.repair.workers import bounded_call
from tools.repair.acquisition import raise_on_software_failure
from tools.repair.batch import read, sha
from tools.repair.development_intended import validate_schedule as intended_schedule
from tools.repair.expanded_profile import checked_checkpoint, checkpoint
from tools.repair.historical_regression import binding, verify_binding
from tools.repair.prepare import case_from_dict, case_to_dict, publish_label_cache

SCHEMA = 'exact-repair/development-generated-collection/v1'
BUDGET = dict(generation_seconds=300, assignment_seconds=300,
              call_cpu_seconds=600, memory_mb=8192, assignments=8)
UNSUPPORTED = 'ValueError: cannot infer complete original relation for endpoint retrieval'


def validate_schedule(plan):
    if (plan.get('schema') != SCHEMA or plan.get('split') != 'development'
            or plan.get('expected_cases') != 32 or plan.get('expected_parents') != 16
            or plan.get('heldout_use') is not False or plan.get('model_fitting') is not False
            or plan.get('fitting_labels_admitted') is not False or plan.get('budget') != BUDGET):
        raise ValueError('Development collection boundary or budget differs')
    intended = verify_binding(plan['intended_plan'])
    cases = intended_schedule(intended)
    if plan['cases'] != [r for r, _, _ in cases]:
        raise ValueError('Collection changed development identities')
    eligibility = verify_binding(plan['eligibility'])
    if (eligibility['case_ids'] != [c.case_id for _, _, c in cases]
            or eligibility['qualified_cases'] != 32 or eligibility['heldout_cases_opened']):
        raise ValueError('Collection requires bound intended-parent eligibility')
    for ref in eligibility['row_receipts']:
        saved = verify_binding(ref)
        if (saved['status'] != 'qualified' or not saved['cleanup_complete']
                or saved['result']['intended_target_satisfied'] is not True):
            raise ValueError('Intended resource qualification changed')
        for payload in saved['payloads']:
            if sha(payload['path']) != payload['sha256']:
                raise ValueError('Intended native evidence changed')
    verify_binding(plan['protocol'])
    return cases


def generate(observable, protocol, directory):
    """Only an observable input is passed into the proposal process."""
    import json
    from tools.repair.fresh_evaluation import generate_control
    problem = read_record(observable)
    frozen = generate_control(problem, 'uniform', protocol, directory)
    payload = dict(input=frozen.problem.to_dict(),
        proposal_reports=json.loads(canonical_json(frozen.proposal_reports)),
        generator='grammar_uniform', model_role='zero_weights_discarded_objective',
        graph_hash=frozen.graph_hash)
    path = Path(directory) / 'pool.json'
    write_artifact(path, payload)
    return binding(path)


def sample_assignments(problem, original, seed):
    """Eight fixed slots, including controls, uniform draws and a quartet.

    Duplicates keep their slots and share one exact-assignment label. Uniform
    probabilities are draw probabilities, not deduplicated inclusion weights.
    Quartet alternatives prefer genuinely new axiom/activation bundles.
    """
    if [o.object_id for o in problem.objects] != [o.object_id for o in original.objects]:
        raise ValueError('Generated object order changed')
    counts = [len(o.candidates) for o in problem.objects]
    keep, delete, alternatives, novel = [], [], [], []
    for index, (obj, old) in enumerate(zip(problem.objects, original.objects)):
        unchanged = [i for i, c in enumerate(obj.candidates)
                     if set(c.axioms) == set(obj.original_axioms) and not c.active_expressions]
        if not unchanged:
            raise ValueError('Generated pool lacks unchanged control')
        keep.append(unchanged[0])
        empty = [i for i, c in enumerate(obj.candidates) if not c.axioms and not c.active_expressions]
        delete.append(empty[0] if empty and obj.eligible and not obj.locked else unchanged[0])
        old_bundles = {canonical_hash((c.axioms, c.active_expressions)) for c in old.candidates}
        fresh = [i for i, c in enumerate(obj.candidates)
                 if canonical_hash((c.axioms, c.active_expressions)) not in old_bundles]
        novel.append(len(fresh))
        others = fresh or [i for i in range(len(obj.candidates)) if i != keep[-1]]
        if others and obj.eligible and not obj.locked:
            alternatives.append((index, others[0]))
    rng = random.Random(seed)
    rows = [dict(origin='unchanged_control', assignment=keep, draw_probability=None),
            dict(origin='delete_control', assignment=delete, draw_probability=None)]
    for _ in range(2):
        rows.append(dict(origin='uniform', assignment=[rng.randrange(n) for n in counts],
                         draw_probability=1 / math.prod(counts)))
    for first, second in ((False, False), (True, False), (False, True), (True, True)):
        assignment = list(keep)
        for use, alternative in zip((first, second), alternatives[:2]):
            if use:
                assignment[alternative[0]] = alternative[1]
        rows.append(dict(origin='counterfactual_quartet', assignment=assignment,
                         quartet_member=[int(first), int(second)],
                         quartet_available=len(alternatives) >= 2, draw_probability=None))
    seen = {}
    for index, row in enumerate(rows):
        key = tuple(row['assignment'])
        row.update(slot=index, duplicate_of=seen.get(key),
            assignment_map=[[obj.object_id, obj.candidates[i].candidate_id]
                            for obj, i in zip(problem.objects, key)])
        seen.setdefault(key, index)
    return dict(rows=rows, novel_candidates_by_object=novel,
                requested_slots=8, unique_assignments=len(seen),
                inventory_size=math.prod(counts), exact_teacher=False)


def label_assignment(record, assignment, profile, directory):
    """External native policy/semantic label; no model score is a target."""
    from tools.repair.train import _assignment_label
    case = case_from_dict(record)
    if case.split != 'development' or case.schema_revision != 'v3':
        raise ValueError('Collection labels require development v3')
    target = SemanticTargetSpec(canonical_hash(case.probes))
    label = _assignment_label(case, tuple(assignment), tuple(tuple(p) for p in profile),
                              semantic_target=target, evidence_directory=Path(directory) / 'native')
    write_artifact(Path(directory) / 'label.json', asdict(label))
    return binding(Path(directory) / 'label.json')


def save_call(path, result, seconds):
    saved = dict(status=result.status, detail=result.detail,
                 cleanup_complete=result.cleanup_complete,
                 resource_usage=dict(result.resource_usage), deadline_seconds=seconds)
    write_artifact(path, saved)
    if not result.cleanup_complete:
        raise RuntimeError('Collection cleanup incomplete; preserve owner guard')
    return saved


def one_case(row, output, identity, plan):
    directory = Path(output) / 'cases' / canonical_hash((row['case_id'], row['case_hash']))
    receipt, guard = directory / 'completion.json', directory / 'inflight.json'
    expected = canonical_hash((identity, row))
    saved = checked_checkpoint(receipt, expected)
    if saved:
        for ref in saved['payloads']:
            if sha(ref['path']) != ref['sha256']:
                raise ValueError('Collection payload changed')
        if not saved['cleanup_complete']:
            raise RuntimeError('Collection ownership reconciliation required')
        if guard.exists():
            if read(guard)['identity'] != expected:
                raise ValueError('Collection owner changed')
            guard.unlink()
        return binding(receipt)
    if guard.exists():
        raise RuntimeError('Interrupted collection needs owner and spent-budget reconciliation')
    write_artifact(guard, dict(identity=expected, case_id=row['case_id'], started_epoch=time.time()))
    began = time.monotonic()
    budget = plan['budget']
    observable = verify_binding(row['observable'])
    original = read_record(observable)
    if original.content_hash != row['input_hash']:
        raise ValueError('Observable identity changed')
    protocol = verify_binding(plan['protocol'])
    generation = bounded_call(generate, observable, protocol, str(directory),
        timeout=budget['generation_seconds'], cpu_seconds=budget['call_cpu_seconds'],
        memory_mb=budget['memory_mb'])
    call = save_call(directory / 'generation-call.json', generation, budget['generation_seconds'])
    status = 'generation_' + generation.status
    result = dict(requested_slots=8, recorded_slots=0, unique_labels=0, usable_labels=0,
                  unknown_slots=8, fitting_labels_admitted=False)
    if generation.status == 'error' and generation.detail == UNSUPPORTED:
        status = 'unsupported_original_mapping_bundle'
    else:
        raise_on_software_failure(call)
    if generation.status == 'complete':
        pool = verify_binding(generation.value)
        problem = read_record(pool['input'])
        sample = sample_assignments(problem, original, plan['seed'])
        write_artifact(directory / 'sample.json', sample)
        # Selection is frozen before the evaluator record is opened here.
        case = case_from_dict(verify_binding(row['evaluator']))
        if (case.split != 'development' or case.problem.content_hash != original.content_hash
                or case.case_id != row['case_id'] or case.structural_parent != row['structural_parent']):
            raise ValueError('Evaluator input or inherited split changed')
        case = replace(case, problem=problem)
        record = case_to_dict(case)
        write_artifact(directory / 'generated-case.json', record)
        profile = sorted(protocol['objective']['edit_weights'].items())
        labels, slot_rows, by_assignment = [], [], {}
        for selected in sample['rows']:
            assignment = tuple(selected['assignment'])
            if assignment in by_assignment:
                slot_rows.append(dict(**selected, label=by_assignment[assignment], reused_exact_assignment=True))
                continue
            target = directory / 'labels' / canonical_hash(assignment)
            native = bounded_call(label_assignment, record, assignment, profile, str(target),
                timeout=budget['assignment_seconds'], cpu_seconds=budget['call_cpu_seconds'],
                memory_mb=budget['memory_mb'])
            label_call = save_call(target / 'call.json', native, budget['assignment_seconds'])
            raise_on_software_failure(label_call)
            if native.status == 'complete':
                if native.value != binding(target / 'label.json'):
                    raise ValueError('Label transport receipt differs from published artifact')
            if native.status != 'complete':
                from exact.repair.records import candidate_cost
                cost = sum(candidate_cost(o, o.candidates[i], tuple(tuple(p) for p in profile))
                           for o, i in zip(problem.objects, assignment))
                write_artifact(target / 'label.json', asdict(RepairLabel(assignment, None, None, cost)))
            label_ref = binding(target / 'label.json')
            labels.append(verify_binding(label_ref))
            by_assignment[assignment] = label_ref
            slot_rows.append(dict(**selected, label=label_ref, reused_exact_assignment=False))
            write_artifact(directory / 'label-progress.json', dict(rows=slot_rows, completed_slots=len(slot_rows)))
        from importlib.metadata import version
        from tools.repair.prepare import cache_from_dict
        target = SemanticTargetSpec(canonical_hash(case.probes))
        hashes = dict(input=problem.content_hash, patch=canonical_hash(problem.objects),
            policy=problem.policy.content_hash, query=canonical_hash(case.probes),
            inventory=canonical_hash(tuple(o.candidates for o in problem.objects)),
            backend=canonical_hash(('pyhermit', version('pyhermit'), 'python')),
            profile=canonical_hash(tuple(tuple(p) for p in profile)),
            teacher_weights=canonical_hash((1.0, 1.0)), semantic_target=target.content_hash,
            sampler=canonical_hash(sample), development_plan=canonical_hash(plan))
        cache = cache_from_dict(dict(schema='exact-repair/teacher-cache/v3',
            candidate_counts=[len(o.candidates) for o in problem.objects], labels=labels,
            complete=False, stop_reason='predeclared_development_sample', hashes=sorted(hashes.items()),
            elapsed_seconds=time.monotonic()-began))
        artifact = publish_label_cache(cache, directory / 'cache')
        from tools.repair.training_audit import label_masks
        masks = label_masks(cache, case, dict(desired_family_weight=1.0, false_positive_weight=1.0))
        slot_labels = [verify_binding(s['label']) for s in slot_rows]
        result.update(recorded_slots=8, unique_labels=len(labels), usable_labels=masks['usable'],
            unknown_slots=sum(l['feasible'] is None or (l['feasible'] and l['benefit'] is None)
                              for l in slot_labels),
            cache=artifact, masks=masks, slots=slot_rows, novel_candidates_by_object=sample['novel_candidates_by_object'],
            inventory_size=sample['inventory_size'], exact_teacher=False,
            generation_scope=('partial' if any(r.get('distribution_scope') != 'declared_language'
                for r in pool['proposal_reports']) else 'declared_language'),
            query_denominator=len(case.probes), query_hash=canonical_hash(case.probes))
        status = 'sampled'
    payloads = [binding(p) for p in sorted(directory.rglob('*')) if p.is_file()
                and p.name not in {'completion.json', 'inflight.json'}
                and not p.name.startswith('.') and not p.name.endswith('.tmp')]
    checkpoint(receipt, expected, schema=SCHEMA, status=status, case_id=row['case_id'],
        parent=row['structural_parent'], split='development', family=row['family'],
        cleanup_complete=True, result=result, payloads=payloads, budget=budget,
        elapsed_seconds=time.monotonic()-began, fitting_labels_admitted=False,
        original_results_replaced=False, heldout_cases_opened=False)
    guard.unlink()
    return binding(receipt)


def run(plan_path, output, start, stop):
    from exact.repair.study import runtime_manifest
    plan = read(plan_path)
    cases = validate_schedule(plan)
    if not 0 <= start < stop <= len(cases):
        raise ValueError('Invalid collection slice')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = canonical_hash((binding(plan_path), sha(__file__), runtime_manifest()))
    with (output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for row, _, _ in cases[start:stop]:
            write_artifact(output / 'progress.json', dict(completed_rows=len(rows), case_id=row['case_id']))
            rows.append(one_case(row, output, identity, plan))
        saved = [verify_binding(r) for r in rows]
        return checkpoint(output / 'report.json', identity, schema=SCHEMA, status='complete',
            plan=binding(plan_path), scheduled_rows=stop-start, recorded_rows=len(rows), rows=rows,
            counts=dict(Counter(r['status'] for r in saved)), study_scheduled_cases=32,
            study_parent_groups=16, study_assignment_slots=256, scheduled_assignment_slots=8*(stop-start),
            usable_labels=sum(r['result']['usable_labels'] for r in saved),
            fitting_labels_admitted=False, fitting_eligible=False, gates_passed=False,
            heldout_cases_opened=False, next_stage='xr21-expanded-training-001',
            limitations=['Development labels cannot enter fitting; training-side acquisition remains.',
                'Sample-conditioned labels are not exact whole-inventory teachers.',
                'Unknowns, duplicates and generation failures retain all scheduled slots.',
                'Uniform grammar proposal is not a strongest semantic search control.',
                '18 protocols, optimizer capacity, selection, held-out controls and reporting remain.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--stop', type=int, required=True)
    args = parser.parse_args()
    run(args.plan, args.output, args.start, args.stop)


if __name__ == '__main__':
    main()
