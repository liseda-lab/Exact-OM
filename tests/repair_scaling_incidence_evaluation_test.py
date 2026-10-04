"""Input compatibility, failure visibility and no replay at the incidence boundary."""
import pytest

from exact.repair.api import write_artifact
from exact.repair.workers import CallResult
from tools.repair import scaling, scaling_incidence as incidence
from tools.repair import scaling_incidence_evaluation as evaluation
from tools.repair.expanded_corpus import binding
from tools.repair.expanded_profile import parent_case
from tools.repair.prepare import case_to_dict


@pytest.mark.parametrize('count', [4, 8, 16])
@pytest.mark.parametrize('condition', ['shared', 'nonshared'])
def test_actual_incidence_inputs_keep_controls_and_grammar(tmp_path, count, condition):
    for case in incidence.construct(parent_case('papers', 13, 13), count, condition):
        path = tmp_path/(case.control+'.json')
        write_artifact(path, case_to_dict(case))
        result = evaluation.adapter_contract(dict(case=binding(path),
            config=dict(menu=2, depth=1, constructors=1)))
        assert result['admitted'] and len(result['objects']) == len(case.problem.objects)
        assert all(o['supplied_controls'] >= 1 and o['variables'] > 0 for o in result['objects'])


def test_nested_generation_error_is_durable_and_never_replayed(tmp_path, monkeypatch):
    import exact.repair.workers
    protocol = tmp_path/'protocol.json';write_artifact(protocol, {})
    rows = [dict(id=mode, pair_id='pair', cache_mode=mode, seconds=300,
                 cpu_seconds=600, memory_mb=8192) for mode in ('cold', 'warm')]
    schedule = tmp_path/'schedule.json'
    write_artifact(schedule, dict(schema='exact-repair/scaling/v1', protocol=binding(protocol), rows=rows))
    calls = []
    def call(fn, schedule, row, output, *a, **kw):
        calls.append(row['id'])
        path = tmp_path/'generation-error.json'
        write_artifact(path, dict(row=row, generation_status='error', generation_detail='ValueError: bad input'))
        return CallResult('complete', value=binding(path), cleanup_complete=True)
    monkeypatch.setattr(exact.repair.workers, 'bounded_call', call)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='bad input'):
            scaling.run(schedule, tmp_path/'work', 0, 2, strict_errors=True)
    assert calls == ['cold']
    assert (tmp_path/'work/rows/cold.json').exists()


def test_partial_timeout_is_coverage_but_completed_missing_pool_is_error():
    saved = dict(status='timeout', cleanup_complete=True, detail='deadline', pool=None)
    assert evaluation.generation_contract(saved, {}) is False
    with pytest.raises(ValueError, match='no pool'):
        evaluation.generation_contract(dict(saved, status='complete'), {})
    with pytest.raises(RuntimeError, match='adapter broken'):
        evaluation.generation_contract(dict(saved, status='error', detail='adapter broken'), {})


def test_generation_python_error_inside_pool_cannot_pass(tmp_path):
    case = incidence.construct(parent_case('papers', 13, 13), 4, 'shared')[0]
    case_path = tmp_path/'case.json';write_artifact(case_path, case_to_dict(case))
    pool = tmp_path/'pool.json'
    write_artifact(pool, dict(input=case.problem.to_dict(), completed_objects=1, scheduled_objects=4,
        reports=[dict(error='ValueError: broken grammar')]))
    saved = dict(status='complete', cleanup_complete=True, pool=binding(pool))
    with pytest.raises(RuntimeError, match='broken grammar'):
        evaluation.generation_contract(saved, dict(case=binding(case_path)))
    write_artifact(pool, dict(input=case.problem.to_dict(), completed_objects=1, scheduled_objects=4,
        reports=[dict(status='unresolved', error='TimeoutError: exhausted')]))
    assert evaluation.generation_contract(dict(saved, pool=binding(pool)), dict(case=binding(case_path))) is False


def test_qualification_probe_timeout_resumes_without_budget_reset(tmp_path, monkeypatch):
    import exact.repair.workers
    case = incidence.construct(parent_case('papers', 13, 13), 4, 'shared')[0]
    path = tmp_path/'case.json';write_artifact(path, case_to_dict(case))
    schedule = dict(cases=[dict(case=binding(path), config=dict(menu=2, depth=1, constructors=1))],
                    rows=[{}]*6, generation_seconds=60)
    schedule_path = tmp_path/'schedule.json';write_artifact(schedule_path, schedule)
    witness_path = tmp_path/'witness.json';write_artifact(witness_path, {})
    monkeypatch.setattr(evaluation, 'witness_admission', lambda *a: schedule)
    monkeypatch.setattr(evaluation, 'qualification_identity', lambda *a: ('frozen-fixture', {}))
    calls = []
    def timeout(*a, **kw):
        calls.append(kw)
        return CallResult('timeout', cleanup_complete=True)
    monkeypatch.setattr(exact.repair.workers, 'bounded_call', timeout)
    result = evaluation.qualify(schedule_path, witness_path, tmp_path/'work')
    assert result['admitted'] and result['complete_generation_probes'] == 0
    assert result['scheduled'] == 3 and result['evaluation_rows'] == 6
    assert evaluation.qualify(schedule_path, witness_path, tmp_path/'work') == result
    assert calls == [dict(timeout=60, cpu_seconds=600, memory_mb=8192)]*3


def test_evaluation_rejects_missing_qualification_without_running(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, 'witness_admission', lambda *a: dict(cases=[{}]))
    monkeypatch.setattr(evaluation, 'qualification_identity', lambda *a: ('identity', {}))
    with pytest.raises(ValueError, match='qualification required'):
        evaluation.evaluate(None, None, tmp_path/'absent.json', tmp_path/'work', 0, 2)
