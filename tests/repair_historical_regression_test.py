"""Historical source preservation, explicit scope and failure-inclusive recovery."""
import dataclasses
import json
from pathlib import Path

import pytest

from exact.repair.api import write_artifact
from exact.repair.records import canonical_hash
from exact.repair.workers import CallResult
from tools.repair import historical_regression as historical
from tools.repair.corpus import _case
from tools.repair.prepare import case_from_dict, case_to_dict


def archived_case(split='development'):
    return case_to_dict(_case('papers:path-1', 'papers', split, 1, 0, 13))


def test_migration_retains_axioms_candidates_intent_and_old_bytes():
    original = archived_case()
    before = json.dumps(original, sort_keys=True)
    migrated, receipt = historical.migrate(original)
    a, b = case_from_dict(original), case_from_dict(migrated)
    assert json.dumps(original, sort_keys=True) == before
    assert b.schema_revision == 'v3' and b.problem.migration_source_hash == a.problem.content_hash
    assert (a.case_id, a.structural_parent, a.split, a.probes, a.intended_assignment) == (
        b.case_id, b.structural_parent, b.split, b.probes, b.intended_assignment)
    assert a.problem.fixed_axioms == b.problem.fixed_axioms
    assert a.problem.budgets == b.problem.budgets
    assert [[c.candidate_id for c in o.candidates] for o in a.problem.objects] == [
        [c.candidate_id for c in o.candidates] for o in b.problem.objects]
    assert receipt['source_case_hash'] == original['hash']
    assert receipt['public_policy_signature_after'] >= receipt['public_policy_signature_before']


@pytest.mark.parametrize('split', ['train', 'test'])
def test_non_development_inputs_rejected(split):
    with pytest.raises(ValueError, match='development'):
        historical.migrate(archived_case(split))


def tiny_plan(tmp_path):
    original = archived_case()
    migrated, receipt = historical.migrate(original)
    qualification = tmp_path/'qualification.json'
    write_artifact(qualification, {'status':'complete'})
    plan = dict(sources=[], qualification=historical.binding(qualification), modes=['cold','warm'],
        seed=13, per_case_seconds=30, per_case_memory_mb=2048, generation={}, model={},
        next_stage='audit', rows=[dict(case_id='papers:path-1', family='papers', parent='papers:path-1',
        migrated_case=migrated, migration=receipt, historical={'status':'generation_error'})])
    path = tmp_path/'plan.json'
    write_artifact(path, plan)
    return path


def test_failed_cold_and_partial_warm_are_retained_and_resume_without_calls(tmp_path, monkeypatch):
    import exact.repair.workers
    import exact.repair.study
    monkeypatch.setattr(exact.repair.study, 'runtime_manifest', lambda: {'version':'test'})
    calls=[]
    def bounded(*args, **kwargs):
        calls.append(args)
        if len(calls)==1:
            return CallResult('timeout', detail='forced', cleanup_complete=True)
        return CallResult('complete', value={'status':'partial'}, cleanup_complete=True)
    monkeypatch.setattr(exact.repair.workers, 'bounded_call', bounded)
    plan = tiny_plan(tmp_path)
    first = historical.run(plan, tmp_path/'out', 0, 1)
    assert first['recorded_rows']==first['scheduled_rows']==2
    assert first['counts']=={'timeout':1, 'partial':1}
    assert first['study_complete'] is False
    assert all(row['formerly_failed'] for row in first['rows'])
    assert calls[0][4]==calls[1][4]  # warm uses this cold population, even if incomplete
    second = historical.run(plan, tmp_path/'out', 0, 1)
    assert len(calls)==2 and first==second
    value=json.loads(plan.read_text()); value['seed']=37; write_artifact(plan,value)
    with pytest.raises(ValueError, match='dependencies'):
        historical.run(plan,tmp_path/'out',0,1)


def test_changed_evidence_rejected_before_execution(tmp_path):
    path=tmp_path/'evidence.json'; write_artifact(path,{'version':1})
    bound=historical.binding(path); write_artifact(path,{'version':2})
    with pytest.raises(ValueError,match='evidence changed'):
        historical.verify_binding(bound)


def test_native_production_uniform_replay_serializes_partial_or_generated_pool(tmp_path):
    migrated,_=historical.migrate(archived_case())
    settings=dict(retrieval_config=dict(classes_per_side=2, properties_per_side=1, endpoints_per_side=2),
        draws_per_object=2,candidate_cap=16,max_depth=1,max_constructors=1,
        compile_seconds=10,max_circuit_nodes=100000,max_graph_nodes=4096,max_graph_edges=32768,
        max_explanations=64,max_text_tokens=128,pair_factor_limit_per_object=16,
        quantization_scale=10000,contextual_filtering=False,factored=True,
        proposal_arm='grammar_uniform',vtree_type='balanced')
    model=dict(encoder='none',hidden_dim=8,heads=2,layers=0,dropout=0,revision='v3',pairwise=False,plan_risk=False)
    from exact.repair.workers import bounded_call
    outcome=bounded_call(historical.generate,migrated,settings,model,str(tmp_path/'cache'),13,
                         timeout=90,memory_mb=8192)
    assert outcome.status=='complete', outcome.detail
    result=outcome.value
    assert result['status'] in {'partial','generated'}
    assert result['reports'] and result['generated_input']['schema'].endswith('/v3')
    assert all(row['arm']=='grammar_uniform' for row in result['reports'])
    assert all(row['attempted_draws']==2 for row in result['reports'])
    assert all('family_statuses' in row for row in result['reports'])
