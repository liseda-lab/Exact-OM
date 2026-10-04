"""Development-only teacher acquisition with per-call native evidence and safe resume."""
from __future__ import annotations

import argparse
import fcntl
import time
from collections import Counter
from pathlib import Path

from exact.repair.api import write_artifact
from exact.repair.owl import OwlVerifier
from exact.repair.records import canonical_hash
from exact.repair.workers import bounded_call
from tools.repair.batch import read, sha
from tools.repair.expanded_profile import checked_checkpoint, checkpoint
from tools.repair.historical_regression import binding, verify_binding
from tools.repair.prepare import case_from_dict, publish_label_cache, read_label_cache


_INTENDED_PARENT_UNVERIFIED = (
    "ValueError: Generated intended parent has not been verified feasible and query-complete"
)


def raise_on_software_failure(saved):
    from exact.experiments.science_health import software_failure

    if software_failure(saved["status"], saved.get("detail", "")):
        raise RuntimeError("Acquisition software failure: " + saved.get("detail", saved["status"]))


class RecordingVerifier(OwlVerifier):
    """Persist each actual native check, including unknown reasons and support scope."""
    def __init__(self, directory):
        super().__init__('hermit', backend='python')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sequence = 0

    def check_theory(self, *args, **kwargs):
        sequence = self.sequence
        self.sequence += 1
        write_artifact(self.directory / f'check-{sequence}.started.json',
                       dict(started_epoch=time.time(), reasoner=self.reasoner, backend=self.backend))
        report = super().check_theory(*args, **kwargs)
        write_artifact(self.directory / f'check-{sequence}.json', report.to_dict())
        return report


def record_call(directory, result, seconds, case, assignment):
    write_artifact(Path(directory) / 'call.json', dict(
        schema='exact-repair/acquisition-call/v1', case_id=case.case_id,
        input_hash=case.problem.content_hash, policy_hash=case.problem.policy.content_hash,
        query_hash=canonical_hash(case.probes), split=case.split,
        parent=case.structural_parent, assignment=assignment,
        status=result.status, detail=result.detail, cleanup_complete=result.cleanup_complete,
        deadline_seconds=seconds, resources=dict(result.resource_usage)))
    if not result.cleanup_complete:
        raise RuntimeError('Acquisition child cleanup incomplete')


def case_worker(record, settings, directory):
    from tools.repair.train import label_case

    case = case_from_dict(record)
    if case.split != 'development' or case.schema_revision != 'v3':
        raise ValueError('Diagnostic acquisition admits only frozen v3 development cases')
    cache = label_case(case, **settings, evidence_directory=Path(directory) / 'native')
    return dict(artifact=publish_label_cache(cache, Path(directory) / 'cache'),
                status='complete' if cache.complete else 'partial', coverage=cache.coverage,
                stop_reason=cache.stop_reason)


def validate_schedule(plan):
    completion = verify_binding(plan['corpus_completion'])
    if completion['status'] != 'complete' or completion['releases']['development']['manifest'] != plan['development_manifest']:
        raise ValueError('Acquisition corpus completion or release mismatch')
    manifest = verify_binding(plan['development_manifest'])
    rows = manifest['rows']
    if len(rows) != 32 or len({r['case_id'] for r in rows}) != 32:
        raise ValueError('Acquisition must preserve all 32 development cases')
    cases = []
    for row in rows:
        record = verify_binding(row['evaluator'])
        case = case_from_dict(record)
        if (row['status'] != 'materialized' or row['split'] != 'development'
                or case.split != 'development' or case.schema_revision != 'v3'
                or case.case_id != row['case_id'] or case.structural_parent != row['structural_parent']
                or record['hash'] != row['case_hash'] or case.problem.content_hash != row['input_hash']):
            raise ValueError('Acquisition crossed frozen development case identity')
        cases.append((row, record, case))
    return cases


def run(plan_path, output, start, stop):
    from exact.repair.study import runtime_manifest

    plan = read(plan_path)
    cases = validate_schedule(plan)
    if not 0 <= start < stop <= len(cases):
        raise ValueError('Invalid development acquisition slice')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = canonical_hash((sha(plan_path), sha(__file__),
        sha(Path(__file__).with_name('train.py')), runtime_manifest()))
    with (output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for row, record, case in cases[start:stop]:
            key = canonical_hash((row['case_id'], row['case_hash']))
            directory = output / 'cases' / key
            receipt = directory / 'completion.json'
            expected = canonical_hash((identity, row))
            marker = directory / 'inflight.json'
            saved = checked_checkpoint(receipt, expected)
            if saved:
                for ref in saved['native_evidence']:
                    verify_binding(ref)
                if saved.get('result'):
                    read_label_cache(saved['result']['artifact'], directory / 'cache', case)
                if marker.exists():
                    if read(marker).get('identity') != expected or not saved['cleanup_complete']:
                        raise ValueError('Completed acquisition marker identity or cleanup differs')
                    marker.unlink()
                raise_on_software_failure(saved)
                rows.append(binding(receipt))
                continue
            if marker.exists():
                raise RuntimeError('Interrupted acquisition needs owner/budget reconciliation: ' + str(marker))
            write_artifact(marker, dict(identity=expected, started_epoch=time.time(), case_id=case.case_id))
            write_artifact(output / 'progress.json', dict(completed_rows=len(rows), case_id=case.case_id))
            began = time.monotonic()
            result = bounded_call(case_worker, record, plan['teacher'], str(directory),
                timeout=plan['worker_seconds'], cpu_seconds=plan['cpu_seconds'], memory_mb=plan['memory_mb'])
            if not result.cleanup_complete:
                raise RuntimeError('Acquisition outer worker cleanup incomplete')
            native = [binding(p) for p in sorted((directory / 'native').glob('**/*.json'))]
            value = result.value if result.status == 'complete' else None
            if value:
                read_label_cache(value['artifact'], directory / 'cache', case)
            saved = checkpoint(receipt, expected, case_id=case.case_id, family=case.family,
                parent=case.structural_parent, split=case.split,
                status=(value['status'] if value else 'unknown_intended_parent'
                        if result.status == 'error' and result.detail == _INTENDED_PARENT_UNVERIFIED
                        else result.status), result=value,
                outer_status=result.status, detail=result.detail, cleanup_complete=True,
                resources=dict(result.resource_usage), elapsed_seconds=time.monotonic()-began,
                native_evidence=native, supervision_eligible=False,
                scope='Separate development acquisition diagnostic; adoption requires dependency and gate review')
            marker.unlink()
            raise_on_software_failure(saved)
            rows.append(binding(receipt))
        return checkpoint(output / 'report.json', identity,
            schema='exact-repair/development-acquisition/v1', status='complete',
            plan=binding(plan_path), scheduled_rows=stop-start, recorded_rows=len(rows), rows=rows,
            counts=dict(Counter(read(r['path'])['status'] for r in rows)),
            supervision_eligible=False, heldout_cases_opened=False,
            next_stage='xr21-expanded-training-001')


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
