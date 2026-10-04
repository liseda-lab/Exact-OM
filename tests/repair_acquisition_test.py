from dataclasses import replace
from types import SimpleNamespace

import pytest

from exact.repair.owl import OwlVerifier
from tools.repair import train
from tools.repair.acquisition import RecordingVerifier, case_worker
from tools.repair.batch import read
from tools.repair.expanded_profile import parent_case
from tools.repair.prepare import case_to_dict


def outcome(status, value=None, detail=''):
    return SimpleNamespace(status=status, value=value, detail=detail,
                           cleanup_complete=True, resource_usage=(('wall_seconds', 0.01),))


@pytest.mark.parametrize('status', ['timeout', 'unsupported', 'unknown'])
def test_acquisition_keeps_intended_unknown_evidence_without_labels(tmp_path, monkeypatch, status):
    case = parent_case('overlap', 2, 13)
    monkeypatch.setattr(train, 'bounded_call', lambda *a, **kw: outcome(status, detail='no qualified result'))
    with pytest.raises(ValueError, match='intended parent'):
        train.label_case(case, evidence_directory=tmp_path)
    receipt = read(tmp_path / 'intended/call.json')
    assert receipt['status'] == status and receipt['cleanup_complete']
    assert receipt['input_hash'] == case.problem.content_hash
    assert not list(tmp_path.glob('*/cache*'))


@pytest.mark.parametrize('status', ['timeout', 'unsupported', 'unknown'])
def test_acquisition_preserves_unknown_assignment_mask_and_failure_reason(tmp_path, monkeypatch, status):
    case = parent_case('overlap', 2, 13)
    def call(function, *args, **options):
        return outcome('complete', True) if function is train._verify_intended else outcome(status, detail='backend_unavailable')
    monkeypatch.setattr(train, 'bounded_call', call)
    cache = train.label_case(case, max_assignments=1, evidence_directory=tmp_path)
    assert not cache.complete and cache.coverage['unknown_policy'] == 1
    assert cache.labels[0].benefit is None
    receipts = [read(p) for p in tmp_path.glob('*/call.json')]
    assert any(r['detail'] == 'backend_unavailable' and r['assignment'] is not None for r in receipts)


def test_recording_verifier_retains_completed_check_before_later_failure(tmp_path, monkeypatch):
    calls = []
    def check(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('later semantic call failed')
        return SimpleNamespace(to_dict=lambda: {'logical_status': 'UNKNOWN', 'obligations': [{'reason': 'unsupported'}]})
    monkeypatch.setattr(OwlVerifier, 'check_theory', check)
    verifier = RecordingVerifier(tmp_path)
    verifier.check_theory(None)
    with pytest.raises(RuntimeError):
        verifier.check_theory(None)
    assert read(tmp_path / 'check-0.json')['obligations'][0]['reason'] == 'unsupported'
    assert (tmp_path / 'check-1.started.json').is_file()
    assert not (tmp_path / 'check-1.json').exists()


def test_diagnostic_acquisition_rejects_heldout_case(tmp_path):
    case = replace(parent_case('overlap', 2, 13), split='test')
    with pytest.raises(ValueError, match='development'):
        case_worker(case_to_dict(case), {}, tmp_path)


def test_acquisition_resume_preserves_finished_error_and_rejects_changed_plan(tmp_path, monkeypatch):
    from tools.repair import acquisition
    from exact.repair.api import write_artifact
    from exact.repair import study
    case = parent_case('overlap', 2, 13)
    record = case_to_dict(case)
    row = dict(case_id=case.case_id, case_hash=record['hash'])
    plan = tmp_path / 'plan.json'
    settings = dict(teacher={}, worker_seconds=10, cpu_seconds=20, memory_mb=512)
    write_artifact(plan, settings)
    monkeypatch.setattr(acquisition, 'validate_schedule', lambda p: [(row, record, case)])
    monkeypatch.setattr(study, 'runtime_manifest', lambda: {'fixture': 'same-source'})
    calls = []
    def call(*args, **kwargs):
        calls.append(1)
        return outcome('timeout', detail='outer deadline')
    monkeypatch.setattr(acquisition, 'bounded_call', call)
    first = acquisition.run(plan, tmp_path / 'output', 0, 1)
    resumed = acquisition.run(plan, tmp_path / 'output', 0, 1)
    assert first == resumed and len(calls) == 1
    assert resumed['counts'] == {'timeout': 1}
    write_artifact(plan, dict(settings, cpu_seconds=30))
    with pytest.raises(ValueError, match='dependencies changed'):
        acquisition.run(plan, tmp_path / 'output', 0, 1)
    assert len(calls) == 1


@pytest.mark.parametrize('detail,expected_status', [
    ('RuntimeError: native bridge failed', 'error'),
    ('ValueError: Generated intended parent has not been verified feasible and query-complete',
     'unknown_intended_parent'),
    ('RuntimeError: Generated intended parent has not been verified feasible and query-complete', 'error'),
])
def test_acquisition_run_persists_and_stops_on_unexpected_error(tmp_path, monkeypatch, detail, expected_status):
    from tools.repair import acquisition
    from exact.repair.api import write_artifact
    from exact.repair import study

    cases = []
    for seed in (13, 37):
        case = parent_case('overlap', 2, seed)
        record = case_to_dict(case)
        cases.append((dict(case_id=case.case_id, case_hash=record['hash']), record, case))
    plan, output = tmp_path / 'plan.json', tmp_path / 'output'
    write_artifact(plan, dict(teacher={}, worker_seconds=10, cpu_seconds=20, memory_mb=512))
    monkeypatch.setattr(acquisition, 'validate_schedule', lambda p: cases)
    monkeypatch.setattr(study, 'runtime_manifest', lambda: {'fixture': 'same-source'})
    calls = []
    def call(*args, **kwargs):
        calls.append(args)
        return outcome('error', detail=detail)
    monkeypatch.setattr(acquisition, 'bounded_call', call)
    if expected_status == 'error':
        for _ in range(2):
            with pytest.raises(RuntimeError, match='Acquisition software failure'):
                acquisition.run(plan, output, 0, 2)
        assert len(calls) == 1  # Resume preserves the error, never spends again.
        assert not (output / 'report.json').exists()
        receipts = list(output.glob('cases/*/completion.json'))
        assert len(receipts) == 1
    else:
        report = acquisition.run(plan, output, 0, 2)
        assert acquisition.run(plan, output, 0, 2) == report
        assert len(calls) == 2
        assert report['counts'] == {'unknown_intended_parent': 2}
        receipts = list(output.glob('cases/*/completion.json'))
    assert not list(output.glob('cases/*/inflight.json'))
    for receipt in receipts:
        saved = read(receipt)
        assert saved['status'] == expected_status
        assert saved['detail'] == detail and saved['outer_status'] == 'error'
        assert saved['cleanup_complete'] and saved['resources']['wall_seconds'] == 0.01
        assert not saved['supervision_eligible']


@pytest.mark.parametrize('stage', ['intended', 'assignment'])
def test_acquisition_inner_error_reaches_run_without_becoming_unknown(tmp_path, monkeypatch, stage):
    from tools.repair import acquisition
    from exact.repair.api import write_artifact
    from exact.repair import study

    cases = []
    for seed in (13, 37):
        case = parent_case('overlap', 2, seed)
        record = case_to_dict(case)
        cases.append((dict(case_id=case.case_id, case_hash=record['hash']), record, case))
    plan, output = tmp_path / 'plan.json', tmp_path / 'output'
    write_artifact(plan, dict(teacher={'max_assignments': 1}, worker_seconds=10,
                             cpu_seconds=20, memory_mb=512))
    monkeypatch.setattr(acquisition, 'validate_schedule', lambda p: cases)
    monkeypatch.setattr(study, 'runtime_manifest', lambda: {'fixture': 'same-source'})
    inner_calls, outer_calls = [], []
    def inner(function, *args, **kwargs):
        inner_calls.append(function)
        if function is train._verify_intended and stage == 'assignment':
            return outcome('complete', True)
        return outcome('error', detail='RuntimeError: native adapter crashed')
    def outer(function, *args, **kwargs):
        outer_calls.append(function)
        try:
            return outcome('complete', function(*args))
        except Exception as error:
            return outcome('error', detail=f'{type(error).__name__}: {error}')
    monkeypatch.setattr(train, 'bounded_call', inner)
    monkeypatch.setattr(acquisition, 'bounded_call', outer)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='native adapter crashed'):
            acquisition.run(plan, output, 0, 2)
    assert len(outer_calls) == 1  # Persisted error halts resume without spending again.
    assert len(inner_calls) == (1 if stage == 'intended' else 2)
    receipts = list(output.glob('cases/*/completion.json'))
    assert len(receipts) == 1 and not (output / 'report.json').exists()
    saved = read(receipts[0])
    assert saved['status'] == saved['outer_status'] == 'error'
    assert 'native adapter crashed' in saved['detail']
    native = [read(ref['path']) for ref in saved['native_evidence']]
    failed = [r for r in native if r['status'] == 'error']
    assert len(failed) == 1
    assert failed[0]['detail'] == 'RuntimeError: native adapter crashed'
    assert failed[0]['cleanup_complete'] and failed[0]['resources']['wall_seconds'] == 0.01
    assert (failed[0]['assignment'] is None) == (stage == 'intended')
    assert not list(output.glob('cases/*/inflight.json'))
