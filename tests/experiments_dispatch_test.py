"""Prepared handoffs require receipts and retain ownership across controller crashes."""
import fcntl
import hashlib
import json
import os
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest

from exact.experiments import dispatch


def save(path, value):
    path.write_text(json.dumps(value))


def load(path):
    return json.loads(path.read_text())


@pytest.fixture
def queue(tmp_path, monkeypatch):
    calls = []

    def spawn(argv, **kwargs):
        # Ownership must be durable before the existing external server launches.
        state = load(tmp_path / 'dispatch-state.json')
        assert any(row['may_have_started'] for row in state.values())
        assert argv[:3] == ['/usr/bin/tmux', '-N', '-S']
        assert 'new-session' in argv
        calls.append(argv)
        return SimpleNamespace(returncode=0, stderr='')

    monkeypatch.setattr(dispatch.subprocess, 'run', spawn)
    monkeypatch.setattr(dispatch, '_tmux_server', lambda launch:
                        (['/usr/bin/tmux', '-N', '-S', launch['tmux_socket']], 12345))
    worker = tmp_path / 'worker.sh'
    worker.write_text('#!/bin/bash\nexit 0\n')
    batch = {
        'id': 'E18', 'resources': {'gpus': 1, 'cpus': 4, 'memory_gib': 80},
        'launch': {
            'argv': ['/usr/bin/srun', '--jobid=14372', '--overlap', '--time=0', '/bin/bash', str(worker)],
            'nonce': 'test-dispatch-nonce-001',
            'tmux_socket': str(tmp_path / 'tmux.sock'),
            'bindings': [{'path': str(worker), 'sha256': hashlib.sha256(worker.read_bytes()).hexdigest()}],
            'step_path': str(tmp_path / 'step.json'),
            'launcher_log': str(tmp_path / 'launcher.log'),
            'run': {'id': 'E18', 'status_path': str(tmp_path / 'status.json'),
                    'exit_path': str(tmp_path / 'exit'), 'completion_path': str(tmp_path / 'complete.json')},
        },
    }
    registry = {'runs': [], 'pending_batches': [batch],
                'capacity': {'gpus': 1, 'cpus': 6, 'memory_gib': 125}}
    save(tmp_path / 'registry.json', registry)
    return registry, batch, calls


def tick(tmp_path, steps=None):
    return dispatch.dispatch_ready(tmp_path, '14372', steps or {'14372.0': 'RUNNING'}, supervisor_step='35')


def receipt(batch, step='14372.36'):
    save(Path(batch['launch']['step_path']), {'step_id': step, 'dispatch_nonce': batch['launch']['nonce']})


def test_launch_registers_verified_step_without_duplicate_spawn(tmp_path, queue):
    registry, batch, calls = queue
    assert tick(tmp_path)['status'] == 'starting'
    assert load(tmp_path / 'registry.json')['runs'] == []
    assert tick(tmp_path)['status'] == 'no_ready_launch'
    receipt(batch)
    result = tick(tmp_path, {'14372.0': 'RUNNING', '14372.36': 'RUNNING'})
    assert result == {'status': 'registered', 'batch_id': 'E18', 'step_id': '14372.36'}
    current = load(tmp_path / 'registry.json')
    assert current['pending_batches'] == []
    assert current['runs'][0]['dispatch_nonce'] == batch['launch']['nonce']
    assert len(calls) == 1
    assert tick(tmp_path)['status'] == 'no_ready_launch'


def test_restart_reconciles_existing_worker_and_registration_crash(tmp_path, queue):
    _, batch, calls = queue
    tick(tmp_path)
    # A new supervisor can only reconcile durable receipts, not inherit a handle.
    receipt(batch)
    assert tick(tmp_path, {'14372.0': 'RUNNING', '14372.36': 'RUNNING'})['status'] == 'registered'
    state = load(tmp_path / 'dispatch-state.json')
    state['E18']['status'] = 'starting'  # Crash after registry write, before state write.
    save(tmp_path / 'dispatch-state.json', state)
    tick(tmp_path)
    assert load(tmp_path / 'dispatch-state.json')['E18']['status'] == 'registered'
    assert len(calls) == 1


def test_short_worker_requires_matching_terminal_receipt(tmp_path, queue):
    _, batch, calls = queue
    tick(tmp_path)
    receipt(batch)
    save(tmp_path / 'complete.json', {'status': 'complete'})
    (tmp_path / 'exit').write_text('0')
    assert tick(tmp_path)['status'] != 'registered'
    save(tmp_path / 'complete.json', {'status': 'complete', 'exit_code': 0,
         'step_id': '14372.36', 'dispatch_nonce': batch['launch']['nonce']})
    assert tick(tmp_path)['status'] == 'registered'
    assert len(calls) == 1


@pytest.mark.parametrize('change', ['worker', 'allocation', 'inline', 'nonce'])
def test_unreviewed_or_invalid_descriptor_never_spawns(tmp_path, queue, change):
    registry, batch, calls = queue
    launch = batch['launch']
    if change == 'worker':
        Path(launch['argv'][-1]).write_text('changed')
    elif change == 'allocation':
        launch['argv'][1] = '--jobid=99999'
    elif change == 'inline':
        launch['argv'][-2:] = ['/bin/bash', '-c']
    else:
        launch['nonce'] = 'short'
    save(tmp_path / 'registry.json', registry)
    assert tick(tmp_path)['status'] == 'failed'
    assert not calls
    assert dispatch.dispatch_incidents(tmp_path)[0]['kind'] == 'dispatch_failed'


def test_stale_or_forged_step_receipt_is_not_registered(tmp_path, queue):
    _, batch, calls = queue
    tick(tmp_path)
    receipt(batch, '99999.4')
    tick(tmp_path)
    assert not load(tmp_path / 'registry.json')['runs']
    assert 'nonce' in dispatch.dispatch_incidents(tmp_path)[0]['reason']
    assert len(calls) == 1


def test_receipt_timeout_is_actionable_and_never_blindly_retried(tmp_path, queue, monkeypatch):
    _, _, calls = queue
    monkeypatch.setattr(dispatch.time, 'time', lambda: 100)
    tick(tmp_path)
    monkeypatch.setattr(dispatch.time, 'time', lambda: 191)
    tick(tmp_path)
    tick(tmp_path)
    assert len(calls) == 1
    assert '90s' in dispatch.dispatch_incidents(tmp_path)[0]['reason']


def test_pending_dependency_resolves_after_parent_registered_and_complete(tmp_path, queue):
    registry, batch, calls = queue
    child = json.loads(json.dumps(batch))
    child.update(id='E16', depends_on=['E18'])
    child['launch']['run']['id'] = 'E16'
    child['launch']['nonce'] = 'test-dispatch-nonce-002'
    child['launch']['step_path'] = str(tmp_path / 'child-step.json')
    registry['pending_batches'].append(child)
    save(tmp_path / 'registry.json', registry)
    assert tick(tmp_path)['batch_id'] == 'E18'
    receipt(batch)
    tick(tmp_path, {'14372.0': 'RUNNING', '14372.36': 'RUNNING'})
    assert tick(tmp_path, {'14372.0': 'RUNNING', '14372.36': 'RUNNING'})['status'] == 'no_ready_launch'
    save(tmp_path / 'complete.json', {'status': 'complete', 'passed': True})
    (tmp_path / 'exit').write_text('0')
    assert tick(tmp_path)['batch_id'] == 'E16'
    assert len(calls) == 2


def test_gpu_reservation_does_not_block_available_cpu_preparation(tmp_path, queue):
    registry, batch, calls = queue
    prep = json.loads(json.dumps(batch))
    prep.update(id='prepare', resources={'gpus': 0, 'cpus': 1, 'memory_gib': 2})
    prep['launch']['run']['id'] = 'prepare'
    prep['launch']['nonce'] = 'test-dispatch-prep-001'
    prep['launch']['step_path'] = str(tmp_path / 'prep-step.json')
    registry['pending_batches'].append(prep)
    save(tmp_path / 'registry.json', registry)
    assert tick(tmp_path)['batch_id'] == 'E18'
    assert tick(tmp_path)['batch_id'] == 'prepare'
    assert len(calls) == 2


@pytest.mark.parametrize('guard', ['pause', 'stop', 'unknown_step', 'lock'])
def test_admission_guards(tmp_path, queue, guard):
    _, _, calls = queue
    if guard in {'pause', 'stop'}:
        (tmp_path / guard.upper()).touch()
        assert tick(tmp_path)['status'] == 'paused'
    elif guard == 'unknown_step':
        assert tick(tmp_path, {'14372.0': 'RUNNING', '14372.77': 'RUNNING'})['status'] == 'unregistered_steps_present'
    else:
        with (tmp_path / 'registry.json.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert tick(tmp_path)['status'] == 'registry_busy'
    assert not calls


def test_no_allocation_cannot_submit(tmp_path, queue):
    with pytest.raises(ValueError, match='allocation unavailable'):
        dispatch.dispatch_ready(tmp_path, '14372', {})
    assert not queue[2]


def test_late_receipt_recovers_uncertain_launch_without_resubmitting(tmp_path, queue, monkeypatch):
    _, batch, calls = queue
    monkeypatch.setattr(dispatch.time, 'time', lambda: 100)
    tick(tmp_path)
    monkeypatch.setattr(dispatch.time, 'time', lambda: 191)
    tick(tmp_path)
    assert dispatch.dispatch_incidents(tmp_path)
    receipt(batch)
    assert tick(tmp_path, {'14372.0': 'RUNNING', '14372.36': 'RUNNING'})['status'] == 'registered'
    assert not dispatch.dispatch_incidents(tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize('server_group,accepted', [
    ('0::/system.slice/slurmstepd.scope/allocation/step_extern/user/task_0', True),
    ('0::/system.slice/slurmstepd.scope/allocation/step_35/user/task_0', False),
    ('0::/system.slice/slurmstepd.scope/other/step_extern/user/task_0', False),
    ('0::/user.slice/user-1001.slice/session-4.scope', False),
])
def test_tmux_server_must_be_in_same_allocation_extern(monkeypatch, server_group, accepted):
    pid = os.getpid()
    commands = []
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if str(path) == '/proc/self/cgroup':
            return '0::/system.slice/slurmstepd.scope/allocation/step_35/user/task_0\n'
        if str(path) == f'/proc/{pid}/cgroup':
            return server_group + '\n'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_text)

    def query(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(pid))

    monkeypatch.setattr(dispatch.subprocess, 'run', query)
    if accepted:
        assert dispatch._tmux_server({'tmux_socket': '/tmp/existing.sock'})[1] == pid
    else:
        with pytest.raises(ValueError, match='extern cgroup'):
            dispatch._tmux_server({'tmux_socket': '/tmp/existing.sock'})
    assert commands == [['/usr/bin/tmux', '-N', '-S', '/tmp/existing.sock', 'display-message', '-p', '#{pid}']]


def test_missing_tmux_server_cannot_autocreate(monkeypatch):
    calls = []

    def missing(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1, stdout='')

    monkeypatch.setattr(dispatch.subprocess, 'run', missing)
    with pytest.raises(ValueError, match='never create'):
        dispatch._tmux_server({'tmux_socket': '/tmp/missing.sock'})
    assert len(calls) == 1 and '-N' in calls[0]


def test_wrapper_preserves_literal_argv_and_records_launcher_exit(tmp_path):
    sentinel = tmp_path / 'should-not-exist'
    literal = f'$(touch {sentinel}) `touch {sentinel}` spaces'
    worker = tmp_path / 'literal-worker.sh'
    worker.write_text('printf "%s\\n" "$1"\nexit 7\n')
    wrapper, exit_path = dispatch._detached_wrapper({
        'argv': ['/bin/bash', str(worker), literal],
        'launcher_log': str(tmp_path / 'a log.txt'),
    })
    result = subprocess.run(['/bin/bash', str(wrapper)], capture_output=True)
    assert result.returncode == 7
    assert exit_path.read_text().strip() == '7'
    assert (tmp_path / 'a log.txt').read_text().strip() == literal
    assert not sentinel.exists()


def test_launcher_exit_becomes_failure_without_repeating_submission(tmp_path, queue):
    tick(tmp_path)
    state = load(tmp_path / 'dispatch-state.json')
    Path(state['E18']['launcher_exit_path']).write_text('1\n')
    tick(tmp_path)
    assert 'Launcher exited' in dispatch.dispatch_incidents(tmp_path)[0]['reason']
    assert len(queue[2]) == 1
