"""Development support-incidence scaling with restricted-logic native witnesses.

One ancestry group, 4/8/16 mappings, shared/nonshared two-edge conflicts and
coherent controls. This projection adds no independent or held-out parents.
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import fcntl
import itertools
import time
from pathlib import Path

import pyowl_core as owl

from exact.repair.candidates import make_candidate, replacement_cost_features
from exact.repair.learning import TeacherProbe
from exact.repair.records import PolicyV2, RepairInputV2, RevisionObjectV2, canonical_hash, promote_input_v3, replace_inventory
from tools.repair.corpus import GeneratedCase, coherent_control
from tools.repair.expanded_corpus import binding, bound, immutable
from tools.repair.expanded_profile import checkpoint, checked_checkpoint, parent_fingerprints
from tools.repair.prepare import case_from_dict, case_to_dict


def construct(parent, count, condition):
    if parent.split != 'development' or count not in (4, 8, 16) or condition not in ('shared', 'nonshared'):
        raise ValueError('Expected a development parent and declared incidence setting')
    anchor = sorted(parent.problem.policy.monitored_classes, key=owl.structural_hexdigest)[0]
    prefix = 'urn:exact:scaling-incidence:' + canonical_hash(parent.structural_parent) + ':'
    size = count // 2
    xs = [owl.Class(owl.IRI(prefix + 'X' + str(i))) for i in range(size)]
    ends = [owl.Class(owl.IRI(prefix + 'A' + str(i))) for i in range(size)]
    edges = tuple(edge for i in range(size) for edge in
                  ((anchor, xs[i]), (xs[0] if condition == 'shared' else xs[i], ends[i])))
    fixed = tuple(owl.DisjointClasses((anchor, end)) for end in ends)
    objects = []
    for i, (left, right) in enumerate(edges):
        name, axioms = f'incidence-{i}', (owl.SubClassOf(left, right),)
        objects.append(RevisionObjectV2(name, 'mapping', axioms,
            (make_candidate(name, axioms, ('keep',)), make_candidate(name, (), ('delete',),
                cost_features=replacement_cost_features(axioms, (), kind='mapping', authorship='generated'))),
            source='source', authorship='generated', source_entity=left, target_entity=right))
    problem = promote_input_v3(RepairInputV2(fixed, tuple(objects), PolicyV2((anchor, *xs, *ends)),
        source_identity='development-incidence-source', target_identity='development-incidence-target',
        matcher_identity='synthetic-controlled-no-matcher', source_axioms=fixed,
        evidence=tuple((o.object_id, {'score': 0.8}) for o in objects)))
    intended = tuple(i % 2 for i in range(count))
    case = GeneratedCase(parent.case_id + f':incidence:{count}:{condition}', parent.structural_parent,
        'scaling_conflict_incidence', 'development', problem,
        tuple(TeacherProbe(f'prefix-{i}', owl.SubClassOf(anchor, x), 'path') for i, x in enumerate(xs)),
        intended, 20261003, False,
        intended_theory=(*fixed, *(ax for obj, choice in zip(objects, intended) for ax in obj.candidates[choice].axioms)),
        variation=(('condition', condition), ('corrupted_mapping_count', count)), schema_revision='v3')
    clean = coherent_control(case)
    retained = [i for i, obj in enumerate(clean.problem.objects) if obj.original_axioms]
    clean = dc.replace(clean, problem=replace_inventory(clean.problem,
        tuple(clean.problem.objects[i] for i in retained)),
        intended_assignment=tuple(clean.intended_assignment[i] for i in retained))
    return case, clean


def graph_conflicts(case):
    """Enumerate subset-minimal path witnesses in this named-subclass fragment.

    Reject any other axiom language. No native or general OWL completeness claim
    follows from this finite graph derivation.
    """
    fixed = case.problem.fixed_axioms
    if not all(isinstance(ax, owl.DisjointClasses) and len(ax.expressions) == 2
               and all(isinstance(x, owl.Class) for x in ax.expressions) for ax in fixed):
        raise ValueError('Unsupported immutable witness language')
    edges = []
    for obj in case.problem.objects:
        if len(obj.original_axioms) != 1 or not isinstance(obj.original_axioms[0], owl.SubClassOf):
            raise ValueError('Unsupported editable witness language')
        ax = obj.original_axioms[0]
        if not isinstance(ax.sub_class, owl.Class) or not isinstance(ax.super_class, owl.Class):
            raise ValueError('Witness graph requires named classes')
        edges.append((ax.sub_class, ax.super_class))
    nodes = sorted(set(case.problem.policy.monitored_classes), key=owl.structural_hexdigest)
    indices = {node: i for i, node in enumerate(nodes)}
    if any(node not in indices for edge in edges for node in edge):
        raise ValueError('Witness graph requires all named classes monitored')
    edges = [(indices[left], indices[right]) for left, right in edges]
    pairs = [sum(1 << indices[x] for x in ax.expressions) for ax in fixed]
    # Breadth-first edge-subset enumeration stops at already minimal conflicts.
    # For this declared family every minimal support is a pair, so supersets
    # containing a known conflict never require a graph closure.
    minimal, masks = [], []
    for size in range(len(edges) + 1):
        for subset in itertools.combinations(range(len(edges)), size):
            selected = sum(1 << i for i in subset)
            if any(old & selected == old for old in masks):
                continue
            reach = [1 << i for i in range(len(nodes))]
            for index in subset:
                left, right = edges[index]
                reach[left] |= 1 << right
            changed = True
            while changed:
                changed = False
                for index in subset:
                    left, right = edges[index]
                    expanded = reach[left] | reach[right]
                    if expanded != reach[left]:
                        reach[left] = expanded
                        changed = True
            if any(pair & targets == pair for pair in pairs for targets in reach):
                minimal.append(subset)
                masks.append(selected)
    return [list(s) for s in minimal]


def witness_subsets(case, supports):
    full = tuple(range(len(case.problem.objects)))
    intended = tuple(i for i, choice in enumerate(case.intended_assignment)
                     if case.problem.objects[i].candidates[choice].axioms)
    subsets = {(), full, intended}
    for support in supports:
        for size in range(len(support) + 1):
            subsets.update(itertools.combinations(support, size))
    return sorted(subsets, key=lambda s: (len(s), s))


def native_witness(item, directory):
    from exact.repair.owl import OwlVerifier, snapshot_from_axioms
    case = case_from_dict(bound(item['case']))
    supports = graph_conflicts(case)
    if supports != item['expected_minimal_supports']:
        raise ValueError('Constructed graph support scope differs')
    verifier, checks = OwlVerifier('hermit', backend='python'), []
    for subset in witness_subsets(case, supports):
        axioms = (*case.problem.fixed_axioms,
                  *(ax for i in subset for ax in case.problem.objects[i].original_axioms))
        report = verifier.check_theory(snapshot_from_axioms(axioms), case.problem.policy.monitored_classes)
        expected = 'VERIFIED_INFEASIBLE' if any(set(s) <= set(subset) for s in supports) else 'VERIFIED_FEASIBLE'
        proof = immutable(Path(directory) / ('subset-' + ('-'.join(map(str, subset)) or 'empty') + '.json'), report.to_dict())
        checks.append(dict(subset=subset, expected=expected, status=report.logical_status, proof=proof,
            qualified=report.complete and report.support.complete_imports and report.logical_status == expected))
    return dict(case_id=case.case_id, input_hash=case.problem.content_hash,
        graph_minimal_supports=supports, native_checks=checks,
        qualified=all(r['qualified'] for r in checks),
        scope='Exact finite named-subclass graph supports; native checks of each support, all proper subsets, '
              'empty/full and intended theories. Not exhaustive native enumeration or general OWL qualification.')


def prepare(primary_ref, output, exposure_refs=()):
    primary, output = bound(primary_ref), Path(output)
    sources = sorted((i for i in primary['cases'] if i['control'] == 'corrupted'), key=lambda i: i['case_id'])
    parent = case_from_dict(bound(sources[0]['base_case']))
    cases, rows, fingerprints = [], [], set(parent_fingerprints(parent))
    for count in (4, 8, 16):
        for condition in ('nonshared', 'shared'):
            for case in construct(parent, count, condition):
                fp = parent_fingerprints(case)
                fingerprints.update(fp)
                config = dict(id=f'incidence-{count}-{condition}', mapping_count=len(case.problem.objects),
                              depth=1, constructors=1, menu=2, candidate_cap=16)
                supports = ([[0 if condition == 'shared' else 2*i, 2*i+1] for i in range(count//2)]
                            if case.control == 'corrupted' else [])
                item = dict(case=immutable(output/'cases'/(canonical_hash(case.case_id)+'.json'), case_to_dict(case)),
                    parent=parent.structural_parent, case_id=case.case_id, split='development',
                    control=case.control, config=config, condition=condition, corrupted_mapping_count=count,
                    fingerprints=fp, expected_minimal_supports=supports)
                index = len(cases)
                cases.append(item)
                for method in ('grammar_circuit', 'semantic_circuit', 'semantic_enumeration_decoder'):
                    pair = canonical_hash(('scaling-incidence-v1', index, method))
                    for mode in ('cold', 'warm'):
                        row = dict(case_index=index, method=method, cache_mode=mode, pair_id=pair,
                                   seconds=300, cpu_seconds=600, memory_mb=8192)
                        row['id'] = canonical_hash(row)
                        rows.append(row)
    # Conservative ancestry closure over existing profile/corpus and overlap
    # observable schedules. Aliases remain exposed even if previously named test.
    candidates = []
    for ref in exposure_refs:
        source = bound(ref)
        for item in source.get('cases', []):
            candidates.append(dict(source=ref, case_id=item.get('case_id', item.get('key')),
                split=item.get('split', 'historically_exposed'), fingerprints=item.get('fingerprints', [])))
        for key in ('selected', 'exposed'):
            for item in source.get(key, []):
                candidates.append(dict(source=ref, case_id=item.get('key'),
                    split=item.get('split', 'historically_exposed'), fingerprints=item['fingerprints']))
    matched, changed = [], True
    while changed:
        changed = False
        for i, item in enumerate(candidates):
            if i not in matched and fingerprints.intersection(item['fingerprints']):
                matched.append(i); fingerprints.update(item['fingerprints']); changed = True
    audit = immutable(output/'ancestry.json', dict(source_parent=sources[0]['base_case'],
        inherited_parent=parent.structural_parent, inherited_split='development',
        projection_alias_group='all-support-incidence-variants', independent_new_parents=0,
        ancestry_union_fingerprints=sorted(fingerprints), matching_exposed_cases=[candidates[i] for i in matched],
        exposure_sources=list(exposure_refs), test_outcomes_opened=False,
        heldout_claim=False, policy='One exposed development projection family; all names, sizes, conditions '
        'and clean/corrupt siblings share one diagnostic group. Any matching prior test motif is exposed '
        'and ineligible for future held-out claims. Existing split artifacts are immutable.'))
    schedule = {k: primary[k] for k in ('schema', 'protocol', 'generation_seconds', 'seed',
        'draws_per_object', 'compiler_nodes', 'compiler_call_seconds', 'compiler_object_seconds',
        'compiler_rss_mb', 'max_expressions', 'context_checks', 'program', 'authorization')}
    schedule.update(cases=cases, rows=rows, scheduled_rows=len(rows), source_schedule=primary_ref,
        ancestry_audit=audit, study_kind='development_support_incidence', test_outcomes_opened=False,
        claim_scope='Restricted controlled diagnostic; no held-out, G0-G2, learning or superiority claim.',
        design='4/8/16 nonempty corrupted mappings;2/4/8 pair supports. Equal mapping/symbol/disjointness '
        'counts between shared/nonshared arms. Clean controls retain only nonempty intended prefix mappings '
        'and have half the mappings; compare clean controls within condition/count, not as load-matched corrupt arms.',
        witness_seconds=300, witness_cpu_seconds=600, witness_memory_mb=8192,
        evaluation_admission='Native witness report and generation adapter qualification required before evaluation; '
        'unavailable witnesses retain all72 evaluation denominator rows.',
        operational_slice_seconds=4200, api_spend_usd=0, study_complete=False, gates_passed=False)
    immutable(output/'schedule.json', schedule)
    return schedule


def run_witness(schedule_path, output):
    from exact.experiments.science_health import software_failure
    from exact.repair.study import runtime_manifest
    from exact.repair.workers import bounded_call
    from tools.repair.fresh_evaluation import ensure_cleanup
    schedule, output = bound(binding(schedule_path)), Path(output)
    runtime = runtime_manifest()
    identity = canonical_hash((binding(schedule_path), binding(__file__), runtime))
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for item in schedule['cases']:
            key = canonical_hash(item)
            row_path, guard = output/'rows'/(key+'.json'), output/'inflight'/(key+'.json')
            row_identity = canonical_hash((identity, item))
            saved = checked_checkpoint(row_path, row_identity)
            if saved is None:
                if guard.exists():
                    raise RuntimeError('Witness inflight owner requires reconciliation; no silent repeat')
                checkpoint(guard, row_identity, case_id=item['case_id'], started_epoch=time.time())
                started = time.monotonic()
                call = bounded_call(native_witness, item, str(output/'proofs'/key),
                    timeout=schedule['witness_seconds'], cpu_seconds=schedule['witness_cpu_seconds'],
                    memory_mb=schedule['witness_memory_mb'])
                saved = checkpoint(row_path, row_identity, case_id=item['case_id'],
                    status=call.status, detail=call.detail, result=call.value if call.status == 'complete' else None,
                    cleanup_complete=call.cleanup_complete, resources=dict(call.resource_usage),
                    elapsed_seconds=time.monotonic()-started)
                ensure_cleanup(call)
                guard.unlink()
            if guard.exists() or not saved['cleanup_complete']:
                raise RuntimeError('Witness cleanup unresolved')
            if software_failure(saved['status'], saved.get('detail', '')):
                raise RuntimeError('Native incidence witness software failure: '+saved.get('detail', ''))
            for check in (saved.get('result') or {}).get('native_checks', []):
                bound(check['proof'])
            rows.append(binding(row_path))
        return checkpoint(output/'report.json', identity, schedule=binding(schedule_path),
            runtime=runtime, status='complete', rows=rows, scheduled=len(schedule['cases']),
            qualified=sum(bool((bound(r).get('result') or {}).get('qualified')) for r in rows),
            evaluation_scheduled=len(schedule['rows']), study_complete=False, gates_passed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('schedule', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run_witness(args.schedule, args.output)
