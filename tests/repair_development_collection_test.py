"""Observable-only sampling, honest label masks and no-replay boundaries."""
from dataclasses import asdict, replace

import pytest

from exact.repair.api import write_artifact
from exact.repair.learning import RepairLabel
from exact.repair.records import canonical_hash
from exact.repair.workers import CallResult
from tools.repair import development_collection as collection
from tools.repair.expanded_profile import parent_case
from tools.repair.historical_regression import binding
from tools.repair.prepare import case_to_dict, read_label_cache


def inputs(tmp_path):
    case = parent_case('overlap', 2, 13)
    record = case_to_dict(case)
    write_artifact(tmp_path / 'observable.json', case.problem.to_dict())
    write_artifact(tmp_path / 'evaluator.json', record)
    write_artifact(tmp_path / 'protocol.json', dict(objective=dict(edit_weights={})))
    row = dict(case_id=case.case_id, case_hash=record['hash'], input_hash=case.problem.content_hash,
               structural_parent=case.structural_parent, family=case.family,
               observable=binding(tmp_path / 'observable.json'), evaluator=binding(tmp_path / 'evaluator.json'))
    plan = dict(budget=collection.BUDGET, seed=13, protocol=binding(tmp_path / 'protocol.json'))
    return row, plan, case


@pytest.mark.parametrize('change', [dict(split='train'), dict(split='test'),
    dict(expected_cases=31), dict(expected_parents=15), dict(heldout_use=True),
    dict(model_fitting=True), dict(fitting_labels_admitted=True), dict(budget={})])
def test_boundary_rejects_before_files(change):
    plan = dict(schema=collection.SCHEMA, split='development', expected_cases=32,
                expected_parents=16, heldout_use=False, model_fitting=False,
                fitting_labels_admitted=False, budget=collection.BUDGET)
    with pytest.raises(ValueError, match='boundary'):
        collection.validate_schedule({**plan, **change})


def test_sample_preserves_control_uniform_quartet_duplicates_and_probabilities():
    case = parent_case('overlap', 2, 13)
    sample = collection.sample_assignments(case.problem, case.problem, 13)
    assert sample == collection.sample_assignments(case.problem, case.problem, 13)
    rows = sample['rows']
    assert len(rows) == 8 and rows[4]['duplicate_of'] == 0
    assert [r['origin'] for r in rows] == ['unchanged_control', 'delete_control'] + ['uniform']*2 + ['counterfactual_quartet']*4
    assert all(r['draw_probability'] == 1 / sample['inventory_size'] for r in rows[2:4])
    assert all(r['draw_probability'] is None for r in rows[:2]+rows[4:])
    assert sample['unique_assignments'] == len({tuple(r['assignment']) for r in rows})
    assert not sample['exact_teacher']


def test_generation_timeout_keeps_eight_unknown_slots_and_never_replays(tmp_path, monkeypatch):
    row, plan, case = inputs(tmp_path)
    calls = []
    def execute(fn, observable, protocol, directory, **kwargs):
        calls.append(kwargs)
        assert observable == case.problem.to_dict()
        assert 'intended_theory' not in str(observable.keys())
        return CallResult('timeout', detail='generation deadline')
    monkeypatch.setattr(collection, 'bounded_call', execute)
    ref = collection.one_case(row, tmp_path / 'work', 'identity', plan)
    saved = collection.verify_binding(ref)
    assert saved['status'] == 'generation_timeout'
    assert saved['result']['unknown_slots'] == saved['result']['requested_slots'] == 8
    assert collection.one_case(row, tmp_path / 'work', 'identity', plan) == ref
    assert len(calls) == 1
    assert calls[0] == dict(timeout=300, cpu_seconds=600, memory_mb=8192)
    payload = saved['payloads'][0]
    write_artifact(payload['path'], dict(changed=True))
    with pytest.raises(ValueError, match='payload changed'):
        collection.one_case(row, tmp_path / 'work', 'identity', plan)


def test_new_bundle_coverage_and_quartet_use_original_inventory_identity():
    case = parent_case('overlap', 2, 13)
    original = replace(case.problem, objects=tuple(replace(o, candidates=o.candidates[:1])
                                                  for o in case.problem.objects))
    sample = collection.sample_assignments(case.problem, original, 13)
    assert sum(sample['novel_candidates_by_object']) > 0
    assert sample['rows'][5]['assignment'] != sample['rows'][4]['assignment']
    assert sample['rows'][6]['assignment'] != sample['rows'][4]['assignment']
    assert all(row['quartet_available'] for row in sample['rows'][4:])


def test_generator_worker_accepts_only_observable_record(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.repair import fresh_evaluation
    case = parent_case('overlap', 2, 13)
    received = []
    def freeze(problem, arm, protocol, directory):
        received.append(problem)
        assert arm == 'uniform'
        return SimpleNamespace(problem=problem, proposal_reports=(), graph_hash='fixture')
    monkeypatch.setattr(fresh_evaluation, 'generate_control', freeze)
    ref = collection.generate(case.problem.to_dict(), {}, tmp_path)
    assert collection.verify_binding(ref)['input'] == case.problem.to_dict()
    assert received == [case.problem]
    with pytest.raises(ValueError):
        collection.generate(case_to_dict(case), {}, tmp_path)


@pytest.mark.parametrize('detail,cleanup,raises', [(collection.UNSUPPORTED, True, False),
    ('ValueError: actual software defect', True, True), ('deadline', False, True)])
def test_unsupported_separate_from_software_and_cleanup(tmp_path, monkeypatch, detail, cleanup, raises):
    row, plan, _ = inputs(tmp_path)
    calls = []
    def execute(*a, **k):
        calls.append(1)
        return CallResult('error', detail=detail, cleanup_complete=cleanup)
    monkeypatch.setattr(collection, 'bounded_call', execute)
    if raises:
        with pytest.raises(RuntimeError):
            collection.one_case(row, tmp_path / 'work', 'identity', plan)
        with pytest.raises(RuntimeError, match='owner and spent-budget'):
            collection.one_case(row, tmp_path / 'work', 'identity', plan)
    else:
        ref = collection.one_case(row, tmp_path / 'work', 'identity', plan)
        assert collection.verify_binding(ref)['status'] == 'unsupported_original_mapping_bundle'
    assert len(calls) == 1


def test_sampled_timeout_labels_retain_masks_and_exact_duplicate_identity(tmp_path, monkeypatch):
    row, plan, case = inputs(tmp_path)
    calls = []
    def execute(fn, *args, **kwargs):
        calls.append(fn)
        if fn is collection.generate:
            directory = collection.Path(args[2])
            write_artifact(directory / 'pool.json', dict(input=case.problem.to_dict(), proposal_reports=[]))
            return CallResult('complete', binding(directory / 'pool.json'))
        return CallResult('timeout', detail='native timeout')
    monkeypatch.setattr(collection, 'bounded_call', execute)
    ref = collection.one_case(row, tmp_path / 'work', 'identity', plan)
    saved = collection.verify_binding(ref)
    result = saved['result']
    assert result['recorded_slots'] == result['unknown_slots'] == 8
    assert result['usable_labels'] == result['masks']['risk'] == 0
    assert len(calls) == result['unique_labels'] + 1
    assert result['slots'][0]['label'] == result['slots'][4]['label']
    artifact = result['cache']
    cache = read_label_cache(artifact, collection.Path(artifact['path']).parent, case)
    assert not cache.complete and not any(label.feasible is not None for label in cache.labels)
    assert dict(cache.hashes)['input'] == case.problem.content_hash
    assert not saved['fitting_labels_admitted']
    assert collection.one_case(row, tmp_path / 'work', 'identity', plan) == ref


@pytest.mark.parametrize('split', ['train', 'test'])
def test_native_label_worker_refuses_other_splits(tmp_path, split):
    _, _, case = inputs(tmp_path)
    with pytest.raises(ValueError, match='development'):
        collection.label_assignment(case_to_dict(replace(case, split=split)), (0,)*len(case.problem.objects), (), tmp_path)


@pytest.mark.slow
def test_native_assignment_label_evidence_is_external_and_query_complete(tmp_path):
    from tools.repair.corpus import generate_corpus
    case = generate_corpus(split_counts={'train': 0, 'development': 1, 'test': 0},
        siblings_per_parent=1, families=('overlap',), revision='v3')[0]
    # All-delete is checked natively, never assumed feasible.
    sample = collection.sample_assignments(case.problem, case.problem, 13)
    assignment = sample['rows'][1]['assignment']
    result = collection.bounded_call(collection.label_assignment, case_to_dict(case), assignment,
        (), str(tmp_path), timeout=120, cpu_seconds=240, memory_mb=8192)
    assert result.status == 'complete', result.detail
    assert result.cleanup_complete
    assert (tmp_path / 'label.json').exists()
    assert list((tmp_path / 'native').glob('check-*.json'))
    if result.value['feasible']:
        assert len(result.value['semantic_vector']) == len(case.probes)
        assert result.value['benefit'] is not None


@pytest.mark.slow
def test_native_generated_inventory_gets_fresh_bound_labels(tmp_path):
    import json
    from tools.repair.corpus import generate_corpus
    case = generate_corpus(split_counts={'train': 0, 'development': 1, 'test': 0},
        siblings_per_parent=1, families=('range',), revision='v3')[0]
    protocol = json.loads(collection.Path('specs/exact-repair/protocol/xr21-review2-conformance.json')
                          .read_text().replace('UNFROZEN', 'collection-native-fixture'))
    protocol['identity']['execution_authorized'] = True
    write_artifact(tmp_path / 'protocol.json', protocol)
    record = case_to_dict(case)
    write_artifact(tmp_path / 'observable.json', case.problem.to_dict())
    write_artifact(tmp_path / 'evaluator.json', record)
    row = dict(case_id=case.case_id, case_hash=record['hash'], input_hash=case.problem.content_hash,
        structural_parent=case.structural_parent, family=case.family,
        observable=binding(tmp_path / 'observable.json'), evaluator=binding(tmp_path / 'evaluator.json'))
    plan = dict(budget=collection.BUDGET, seed=13, protocol=binding(tmp_path / 'protocol.json'))
    ref = collection.one_case(row, tmp_path / 'work', 'native-fixture', plan)
    saved = collection.verify_binding(ref)
    assert saved['status'] == 'sampled', saved
    result = saved['result']
    assert result['recorded_slots'] == result['requested_slots'] == 8
    generated = collection.case_from_dict(collection.read(collection.Path(ref['path']).parent / 'generated-case.json'))
    cache = read_label_cache(result['cache'], collection.Path(result['cache']['path']).parent, generated)
    assert not cache.complete and result['usable_labels'] > 0
    assert dict(cache.hashes)['query'] == canonical_hash(case.probes)
    assert dict(cache.hashes)['inventory'] == canonical_hash(tuple(o.candidates for o in generated.problem.objects))
    assert collection.one_case(row, tmp_path / 'work', 'native-fixture', plan) == ref
