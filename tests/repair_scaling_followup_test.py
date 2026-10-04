"""Scientific denominator, restricted witness scope and interruption contracts."""
import dataclasses as dc
from pathlib import Path

import pyowl_core as owl
import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash, read_record
from exact.repair.workers import CallResult
from tools.repair import scaling_audit as audit, scaling_incidence as incidence
from tools.repair.expanded_corpus import binding, bound
from tools.repair.expanded_profile import parent_case, checkpoint
from tools.repair.prepare import case_to_dict, case_from_dict


@pytest.mark.parametrize('count', [4, 8, 16])
@pytest.mark.parametrize('condition', ['shared', 'nonshared'])
def test_exact_graph_supports_and_nonempty_clean_controls(count, condition):
    parent = parent_case('papers', 13, 13)
    corrupt, clean = incidence.construct(parent, count, condition)
    supports = [[0 if condition == 'shared' else 2*i, 2*i+1] for i in range(count//2)]
    assert incidence.graph_conflicts(corrupt) == supports
    assert incidence.graph_conflicts(clean) == []
    assert len(corrupt.problem.objects) == count
    assert len(clean.problem.objects) == count//2
    for case in (corrupt, clean):
        assert case.split == 'development' and case.structural_parent == parent.structural_parent
        assert case_from_dict(case_to_dict(case)) == case
        assert all(obj.original_axioms for obj in case.problem.objects)
        assert read_record(case.problem.to_dict()) == case.problem
    checks = incidence.witness_subsets(corrupt, supports)
    assert () in checks and tuple(range(count)) in checks and tuple(range(0, count, 2)) in checks
    assert all(tuple(s) in checks and (s[0],) in checks and (s[1],) in checks for s in supports)


def test_scope_rejects_non_development_and_complex_axioms():
    with pytest.raises(ValueError, match='development'):
        incidence.construct(parent_case('papers', 1, 13, split='test'), 4, 'shared')
    case = incidence.construct(parent_case('papers', 13, 13), 4, 'shared')[0]
    bad = dc.replace(case, problem=dc.replace(case.problem, fixed_axioms=(owl.Declaration(
        case.problem.policy.monitored_classes[0]),)))
    with pytest.raises(ValueError, match='language'):
        incidence.graph_conflicts(bad)


def test_witness_timeout_is_retained_and_never_replayed(tmp_path, monkeypatch):
    import exact.repair.workers
    item = dict(case_id='case', case={'not_opened': True})
    schedule = tmp_path/'schedule.json'
    write_artifact(schedule, dict(cases=[item], rows=[{}]*6,
        witness_seconds=300, witness_cpu_seconds=600, witness_memory_mb=8192))
    calls = []
    def timeout(*args, **kwargs):
        calls.append(kwargs)
        return CallResult('timeout', cleanup_complete=True)
    monkeypatch.setattr(exact.repair.workers, 'bounded_call', timeout)
    report = incidence.run_witness(schedule, tmp_path/'output')
    assert report['qualified'] == 0 and report['scheduled'] == 1 and report['evaluation_scheduled'] == 6
    assert incidence.run_witness(schedule, tmp_path/'output') == report and len(calls) == 1
    assert calls[0] == dict(timeout=300, cpu_seconds=600, memory_mb=8192)


def test_witness_software_error_remains_visible(tmp_path, monkeypatch):
    import exact.repair.workers
    schedule = tmp_path/'schedule.json'
    write_artifact(schedule, dict(cases=[dict(case_id='case')], rows=[],
        witness_seconds=300, witness_cpu_seconds=600, witness_memory_mb=8192))
    monkeypatch.setattr(exact.repair.workers, 'bounded_call', lambda *a, **k:
        CallResult('error', detail='ValueError: incorrect adapter', cleanup_complete=True))
    with pytest.raises(RuntimeError, match='incorrect adapter'):
        incidence.run_witness(schedule, tmp_path/'output')
    assert len(list((tmp_path/'output/rows').glob('*.json'))) == 1
    with pytest.raises(RuntimeError, match='incorrect adapter'):
        incidence.run_witness(schedule, tmp_path/'output')


def audit_fixture(tmp_path):
    rows, refs = [], []
    case = dict(parent='development-parent', control='coherent', config={'depth': 2})
    for method in ('grammar_circuit', 'semantic_circuit', 'semantic_enumeration_decoder'):
        for mode in ('cold', 'warm'):
            row = dict(id=method+'-'+mode, case_index=0, method=method, cache_mode=mode)
            rows.append(row)
            result_path = tmp_path/row['id']/'result.json'
            write_artifact(result_path, dict(row=row, logical_status='VERIFIED_FEASIBLE',
                generation_status='error', generation_detail='endpoint unavailable',
                fallback_original_inventory=True, generation_complete=False,
                cache_source=str(tmp_path/(method+'-cold')/'compiler-cache') if mode=='warm' else None))
            path = tmp_path/(row['id']+'.json')
            checkpoint(path, 'fixture', row=row, cleanup_complete=True, status='complete',
                result=binding(result_path), payloads=[binding(result_path)], elapsed_seconds=1, resources={})
            refs.append(binding(path))
    schedule = tmp_path/'schedule.json'
    write_artifact(schedule, dict(rows=rows, cases=[case]))
    report = tmp_path/'report.json'
    checkpoint(report, 'fixture', schedule=binding(schedule), status='complete',
        scheduled_rows=6, recorded_rows=6, rows=refs)
    return schedule, report, refs


def test_audit_keeps_generation_errors_despite_successful_fallback(tmp_path):
    schedule, report, _ = audit_fixture(tmp_path)
    rows = audit.audit_rows(audit.Evidence(), bound(binding(schedule)), [binding(report)])
    assert len(rows) == 6 and all(r['generation_status'] == 'error' for r in rows)
    assert all(r['logical_status'] == 'VERIFIED_FEASIBLE' for r in rows)
    languages, caches = audit.comparisons(rows)
    assert all(r['compared_objects'] == 0 and not r['both_generation_complete'] for r in languages)
    assert all(r['provenance_status'] == 'matched' and not r['paired_quality_available'] for r in caches)


def test_audit_rejects_duplicate_missing_and_tampered_rows(tmp_path):
    schedule, report, refs = audit_fixture(tmp_path)
    expected = bound(binding(schedule))
    with pytest.raises(ValueError, match='Duplicate'):
        audit.audit_rows(audit.Evidence(), expected, [binding(report), binding(report)])
    checkpoint(report, 'fixture', schedule=binding(schedule), status='complete',
        scheduled_rows=5, recorded_rows=5, rows=refs[:5])
    with pytest.raises(ValueError, match='Missing'):
        audit.audit_rows(audit.Evidence(), expected, [binding(report)])
    saved = bound(refs[0])
    Path(saved['result']['path']).write_text('{}')
    with pytest.raises(ValueError, match='Changed evidence'):
        audit.audit_rows(audit.Evidence(), expected, [binding(report)])


def test_native_minimal_witness_and_clean_control(tmp_path):
    """Native qualification runs only in its explicitly selected Slurm batch."""
    for case in incidence.construct(parent_case('papers', 13, 13), 4, 'shared'):
        path = tmp_path/(case.control+'.json')
        write_artifact(path, case_to_dict(case))
        item = dict(case=binding(path), expected_minimal_supports=([[0,1],[0,3]] if case.control=='corrupted' else []))
        result = incidence.native_witness(item, tmp_path/case.control)
        assert result['qualified']
        assert result['graph_minimal_supports'] == item['expected_minimal_supports']
        assert any(c['status'] == 'VERIFIED_FEASIBLE' for c in result['native_checks'])
