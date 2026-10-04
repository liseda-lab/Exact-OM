"""Reject false qualification, split leakage and missing real-projection outcomes."""

import json

import pytest

from exact.repair.records import canonical_hash
from tools.repair import real_module_audit as audit


def save(path, value):
    path.write_text(json.dumps(value))
    return audit.binding(path)


def qualified():
    return dict(cleanup_complete=True, status='complete', value=dict(authorizes=True, report=dict(
        logical_status='VERIFIED_FEASIBLE', verification_scope='complete_supported_fragment',
        support=dict(input_supported=True, complete_imports=True),
        obligations=[dict(complete=True, verdict=True, expected=True)])))


@pytest.mark.parametrize('corruption', ['unknown', 'incomplete_query', 'unsupported', 'imports'])
def test_native_completion_is_not_qualification(corruption):
    saved = qualified()
    assert audit.validate_qualification(saved)
    native = saved['value']['report']
    if corruption == 'unknown':
        native['logical_status'] = 'UNKNOWN'
    elif corruption == 'incomplete_query':
        native['obligations'][0]['complete'] = False
    elif corruption == 'unsupported':
        native['support']['input_supported'] = False
    else:
        native['support']['complete_imports'] = False
    with pytest.raises(ValueError, match='native evidence'):
        audit.validate_qualification(saved)
    saved['value']['authorizes'] = False
    assert audit.validate_qualification(saved) is False


def test_native_error_detail_is_not_silenced():
    saved = dict(status='worker_error', cleanup_complete=True, detail='native query exploded')
    with pytest.raises(ValueError, match='native query exploded'):
        audit.validate_qualification(saved)


@pytest.mark.parametrize('corruption', ['split', 'sibling', 'full_source'])
def test_projections_do_not_create_heldout_pairs_or_full_source_support(corruption):
    preparation = dict(pair=dict(group_id='pair', split='train'), independent_pair_count=1,
                       scope='projection', source_coherence_qualified=False,
                       matcher_inputs_consumed=0, labels_used_for_fitting=0)
    schedule = dict(independent_pair_count=1, scope='projection', source_coherence_qualified=False,
                    control_definition={}, arms=[dict(id=str(i)) for i in range(9)],
                    cases=[dict(group_id='pair', structural_parent='pair', split='train',
                                family_exposure='historical_train_pair_exploratory') for _ in range(6)])
    report = dict(independent_pair_count=1, inherited_split='train', scope='projection',
                  gates_passed=False, matcher_inputs_consumed=0, labels_used_for_fitting=0,
                  control_definition={})
    audit.validate_scope(schedule, preparation, report)
    if corruption == 'split':
        schedule['cases'][0]['split'] = 'test'
    elif corruption == 'sibling':
        report['independent_pair_count'] = 2
    else:
        preparation['source_coherence_qualified'] = True
    with pytest.raises(ValueError, match='split|independent|full-source'):
        audit.validate_scope(schedule, preparation, report)


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'budget'])
def test_all_unavailable_rows_must_stay_in_fixed_budget_denominator(tmp_path, corruption):
    cases = [dict(case_id=str(i), condition='coherent', group_id='pair', split='train',
                  status='qualification_unknown') for i in range(6)]
    arms = [dict(id=str(i)) for i in range(9)]
    rows = [dict(id=f'{arm}:{case}', arm_id=str(arm), case_index=case,
                 seconds=300, cpu_seconds=600, memory_mb=8192)
            for arm in range(9) for case in range(6)]
    if corruption == 'budget':
        rows[0]['seconds'] = 600
    schedule = dict(rows=rows, cases=cases, arms=arms)
    schedule_ref = save(tmp_path / 'schedule.json', schedule)
    identity = canonical_hash((schedule_ref['sha256'], 'source-hash', {}))
    selected = rows[:-1] if corruption == 'missing' else rows
    if corruption == 'duplicate':
        selected = rows[:-1] + [rows[0]]
    refs = []
    for i, row in enumerate(selected):
        value = dict(identity=canonical_hash((identity, row)), row=row, cleanup_complete=True,
                     status='unavailable', elapsed_seconds=0, resources={})
        ref = save(tmp_path / f'row-{i}.json', dict(value, content_hash=canonical_hash(value)))
        refs.append(dict(ref, row_id=row['id'], status='unavailable'))
    value = dict(identity=identity, runtime={}, schedule=schedule_ref, slice=[0, 54],
                 status='complete', scheduled=len(refs), recorded=len(refs), rows=refs)
    ref = save(tmp_path / 'evaluation.json', dict(value, content_hash=canonical_hash(value)))
    completion = save(tmp_path / 'completion.json', {})
    run = dict(completion_path=completion['path'], step_id='14408.9', dispatch_nonce='nonce')
    source_path = str(tmp_path / 'tools/repair/fresh_evaluation.py')
    sources = dict(evaluation=dict(run=run, runtime={}, batch=dict(code=str(tmp_path),
                                                                frozen_files={source_path: 'source-hash'})))
    report = dict(schedule=schedule_ref, executions=[dict(run_id='evaluation', completion=completion,
                  step_id=run['step_id'], nonce=run['dispatch_nonce'], report=ref)])
    with pytest.raises(ValueError, match='coverage|Duplicate|budget'):
        audit.validate_rows(audit.Evidence(), schedule, report, sources, dict(shards=[dict(slice=[0, 54])]))


def test_nested_semantic_software_failure_retains_detail():
    saved = dict(status='complete')
    result = dict(status='evaluated', semantic_status='error', semantic_detail='ValueError: broken adapter')
    with pytest.raises(ValueError, match='broken adapter'):
        audit.validate_row_outcome(saved, result)
    assert audit.validate_row_outcome(saved, dict(status='verification_timeout')) == 'verification_timeout'
