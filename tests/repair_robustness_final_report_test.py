"""Protect perturbation pairing, all denominators, source drift and attempt charges."""
from copy import deepcopy

import pytest

from tools.repair.robustness_final_report import (
    CONDITIONS, paired_intervention, union_costs, validate_rows,
)


def row(case='a', parent='p1', condition='baseline', usable=True):
    return dict(row=dict(arm_id='learned', case_id=case+condition), base_case_id=case,
        group_id=parent, family='f', family_exposure='seen_family', control='corrupted',
        condition=condition, source_regime='source-a', intervention_scope=dict(applicable=True),
        semantic_status='known' if usable else 'unavailable',
        logical_status='VERIFIED_FEASIBLE' if usable else 'UNKNOWN',
        semantic_benefit=2 if usable else None, edit_cost=.25 if usable else None,
        pool={'path':'saved'}, generation_reports=[dict(generation_status='SAMPLED')],
        elapsed_seconds=10, resources=dict(cpu_seconds=8))


def paired():
    return [row(), row(condition='final_pool_050'), row('b','p2'),
            row('b','p2','final_pool_050',False)]


def test_conditional_quality_keeps_unknown_pair_in_denominator():
    result=paired_intervention(paired(),'learned','final_pool_050')
    assert result['scheduled_pairs']==2 and result['quality_usable_pairs']==1
    q=result['metrics']['conditional_benefit_difference']
    assert q['scheduled_rows']==2 and q['usable_rows']==1 and q['missing_rows']==1
    assert q['estimate']==0 and q['descriptive_95_interval'] is None
    assert result['metrics']['coverage_difference']['estimate']==-.5
    assert result['metrics']['wall_seconds_difference']['usable_rows']==2


def test_noops_retained_and_source_changes_explicit():
    rows=paired(); rows[1]['intervention_scope']['applicable']=False
    rows[1]['source_regime']='corrected-source'
    result=paired_intervention(rows,'learned','final_pool_050')
    assert result['no_op_pairs']==result['applicable_pairs']==1
    assert result['scheduled_pairs']==2
    assert result['same_scientific_source_pairs']==result['different_scientific_source_pairs']==1
    assert result['source_pairs']['corrected-source minus source-a']==1


def test_missing_cpu_is_not_imputed_zero():
    rows=paired(); rows[1]['resources']={}
    q=paired_intervention(rows,'learned','final_pool_050')['metrics']['measured_cpu_seconds_difference']
    assert q['scheduled_rows']==2 and q['missing_rows']==1 and q['estimate']==0


@pytest.mark.parametrize('mutation', ['missing','duplicate','parent','family','exposure','control'])
def test_invalid_pairs_rejected(mutation):
    rows=paired()
    if mutation=='missing': rows.pop()
    elif mutation=='duplicate': rows.append(deepcopy(rows[0]))
    else: rows[1][dict(parent='group_id',family='family',exposure='family_exposure',control='control')[mutation]]='changed'
    with pytest.raises(ValueError): paired_intervention(rows,'learned','final_pool_050')


def test_direction_is_intervention_minus_baseline():
    rows=paired(); rows[1].update(semantic_benefit=5,elapsed_seconds=7,edit_cost=.5)
    r=paired_intervention(rows,'learned','final_pool_050')['metrics']
    assert r['conditional_benefit_difference']['estimate']==3
    assert r['conditional_utility_difference']['estimate']==2.75
    assert r['wall_seconds_difference']['estimate']==-1.5


def test_supplied_inventory_score_does_not_become_generated_quality():
    rows=paired(); rows[1]['generation_reports']=[]
    result=paired_intervention(rows,'learned','final_pool_050')
    assert result['quality_usable_pairs']==0
    assert result['metrics']['conditional_benefit_difference']['estimate'] is None


def full_schedule():
    cases=[dict(case_id=f'{i}:{c}',base_case_id=f'c{i}',group_id=f'p{i//2}',family='f',
                family_exposure='seen_family',control='coherent' if i%2 else 'corrupted',condition=c)
           for i in range(64) for c in CONDITIONS]
    arms=[dict(id=f'learned{i}',kind='learned') for i in range(6)] + [
        dict(id=c,kind='control') for c in ('symbolic_rich_action','uniform','deletion')]
    entries=[dict(id=f'{a["id"]}:{i}',arm_id=a['id'],case_id=c['case_id'],case_index=i)
             for a in arms for i,c in enumerate(cases)]
    rows=[dict(row=e,**{k:v for k,v in cases[e['case_index']].items() if k!='case_id'}) for e in entries]
    return rows,dict(rows=entries,cases=cases,arms=arms)


def test_full_frozen_denominator():
    rows,schedule=full_schedule();assert len(validate_rows(rows,schedule))==9
    with pytest.raises(ValueError,match='denominator'): validate_rows(rows[:-1],schedule)
    rows[-1]=deepcopy(rows[0])
    with pytest.raises(ValueError,match='Missing or duplicate'):validate_rows(rows,schedule)


def test_renamed_parent_does_not_create_independence():
    rows,schedule=full_schedule();rows[0]['group_id']='renamed'
    with pytest.raises(ValueError,match='ancestry'):validate_rows(rows,schedule)


def test_complete_parent_condition_crossing_required():
    rows,schedule=full_schedule();schedule['cases'][0]['condition']='other'
    with pytest.raises(ValueError,match='pairing'):validate_rows(rows,schedule)


def test_cost_union_preserves_shared_and_failed_attempts_once():
    charge=dict(status='settled',elapsed_seconds=10,logical_id='failed')
    key='/attempts/failed/001';ledger=dict(attempts={key:charge})
    source=dict(run=dict(completion_path=key+'/completion.json'),charge=charge)
    reports={'robustness':dict(attempt_costs={key:charge},attempts=[source]),
             'overlap':dict(attempt_costs={key:charge}),
             'scope':dict(attempts=[dict(completion=dict(path=key+'/completion.json'),charge=charge)])}
    assert union_costs(reports,[source],ledger)=={key:charge}


@pytest.mark.parametrize('alteration',['missing','reduced','reserved'])
def test_changed_original_cost_rejected(alteration):
    charge=dict(status='settled',elapsed_seconds=10);key='/attempt/1'
    ledger=dict(attempts={key:deepcopy(charge)})
    if alteration=='missing': ledger['attempts']={}
    elif alteration=='reduced': ledger['attempts'][key]['elapsed_seconds']=1
    else: ledger['attempts'][key]['status']='reserved'
    with pytest.raises(ValueError,match='charge'):
        union_costs({'robustness':dict(attempt_costs={key:charge})},[],ledger)
