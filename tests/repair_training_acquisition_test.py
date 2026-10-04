from dataclasses import replace
from types import SimpleNamespace

import pytest

from exact.repair.api import write_artifact
from tools.repair import acquisition, train
from tools.repair.expanded_profile import parent_case
from tools.repair.historical_regression import binding
from tools.repair.prepare import case_to_dict


def schedule(tmp_path, split='train', count=128):
    # Serialization/admission fixtures, not independent scientific parents.
    base = parent_case('overlap', 2, 13)
    rows = []
    for index in range(count):
        case = replace(base, split=split, case_id=f'fixture-{index}')
        record = case_to_dict(case)
        path = tmp_path / f'case-{index}.json'
        write_artifact(path, record)
        rows.append(dict(case_id=case.case_id, structural_parent=case.structural_parent,
                         split=split, status='materialized', evaluator=binding(path),
                         case_hash=record['hash'], input_hash=case.problem.content_hash))
    manifest = tmp_path / 'manifest.json'
    write_artifact(manifest, dict(rows=rows))
    completion = tmp_path / 'completion.json'
    write_artifact(completion, dict(status='complete', releases={split: dict(manifest=binding(manifest))}))
    training = split == 'train'
    return dict(schema=f'exact-repair/{"training" if training else "development"}-acquisition-plan/v1',
                split=split, expected_cases=count, heldout_use=False, model_fitting=False,
                corpus_completion=binding(completion),
                **{('training_manifest' if training else 'development_manifest'): binding(manifest)})


@pytest.mark.parametrize('split,count', [('train', 128), ('development', 32)])
def test_admits_complete_bound_split_only(tmp_path, split, count):
    plan = schedule(tmp_path, split, count)
    cases = acquisition.validate_schedule(plan)
    assert len(cases) == count and {case.split for _, _, case in cases} == {split}
    # A relabelled plan cannot admit the other release, even with correct counts.
    with pytest.raises(ValueError, match='split'):
        acquisition.validate_schedule(dict(plan, split='development' if split == 'train' else 'train'))


@pytest.mark.parametrize('change', [dict(split='test'), dict(heldout_use=True),
                                   dict(model_fitting=True), dict(expected_cases=127)])
def test_rejects_heldout_or_incomplete_plan_before_opening_inputs(change):
    plan = dict(schema='exact-repair/training-acquisition-plan/v1', split='train',
                expected_cases=128, heldout_use=False, model_fitting=False)
    with pytest.raises(ValueError):
        acquisition.validate_schedule({**plan, **change})


def test_rejects_wrong_case_split_even_with_updated_bindings(tmp_path):
    plan = schedule(tmp_path)
    path = tmp_path / 'case-0.json'
    case = replace(parent_case('overlap', 2, 13), split='test', case_id='fixture-0')
    record = case_to_dict(case)
    write_artifact(path, record)
    manifest = tmp_path / 'manifest.json'
    content = acquisition.read(manifest)
    content['rows'][0].update(evaluator=binding(path), case_hash=record['hash'])
    write_artifact(manifest, content)
    plan['training_manifest'] = binding(manifest)
    completion = tmp_path / 'completion.json'
    write_artifact(completion, dict(status='complete', releases={'train': dict(manifest=binding(manifest))}))
    plan['corpus_completion'] = binding(completion)
    with pytest.raises(ValueError, match='split or case identity'):
        acquisition.validate_schedule(plan)


def test_worker_requires_explicit_train_admission_and_never_admits_test(tmp_path, monkeypatch):
    case = replace(parent_case('overlap', 2, 13), split='train')
    calls = []
    def label(case, **settings):
        calls.append(case.split)
        return SimpleNamespace(complete=False, coverage={'usable': 0}, stop_reason='deadline')
    monkeypatch.setattr(train, 'label_case', label)
    monkeypatch.setattr(acquisition, 'publish_label_cache', lambda *args: {'fixture': True})
    with pytest.raises(ValueError):
        acquisition.case_worker(case_to_dict(case), {}, tmp_path)
    result = acquisition.case_worker(case_to_dict(case), {}, tmp_path, 'train')
    assert calls == ['train'] and result['status'] == 'partial'
    with pytest.raises(ValueError):
        acquisition.case_worker(case_to_dict(replace(case, split='test')), {}, tmp_path, 'test')
    assert calls == ['train']


def test_training_timeout_is_retained_on_resume_without_spending_again(tmp_path, monkeypatch):
    from exact.repair import study
    plan = schedule(tmp_path)
    plan.update(teacher={}, worker_seconds=315, cpu_seconds=600, memory_mb=8192)
    path = tmp_path / 'plan.json'
    write_artifact(path, plan)
    monkeypatch.setattr(study, 'runtime_manifest', lambda: {'fixture': 'bound-runtime'})
    calls = []
    def call(function, *args, **kwargs):
        calls.append((args[-1], kwargs))
        return SimpleNamespace(status='timeout', value=None, detail='deadline',
                               cleanup_complete=True, resource_usage=(('wall_seconds', 315),))
    monkeypatch.setattr(acquisition, 'bounded_call', call)
    first = acquisition.run(path, tmp_path / 'out', 0, 1)
    second = acquisition.run(path, tmp_path / 'out', 0, 1)
    assert first == second and len(calls) == 1 and calls[0][0] == 'train'
    assert first['schema'] == 'exact-repair/training-acquisition/v1'
    assert first['counts'] == {'timeout': 1} and not first['supervision_eligible']
    receipt = acquisition.read(first['rows'][0]['path'])
    assert receipt['split'] == 'train' and receipt['result'] is None
    assert receipt['resources']['wall_seconds'] == 315
