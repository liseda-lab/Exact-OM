"""Protect failure denominators, paired uncertainty and receipt-only reporting."""
from copy import deepcopy

import pytest

from tools.repair.evaluation_final_report import (
    attempt_costs, bootstrap_clusters, generated_known, interval, known,
    paired_comparison, summary, validate_rows,
)


def row(arm='learned', case='c', group='g', usable=True):
    return dict(row=dict(arm_id=arm, case_id=case), group_id=group, family='f',
        family_exposure='seen_family', control='corrupted',
        semantic_status='known' if usable else 'unavailable',
        logical_status='VERIFIED_FEASIBLE' if usable else 'UNKNOWN',
        semantic_benefit=2 if usable else None,edit_cost=.25 if usable else None,
        pool={'path':'saved'},generation_reports=[dict(generation_status='SAMPLED')],
        elapsed_seconds=10,process_status='complete',scientific_status='evaluated',software_errors=[])


def test_unknowns_remain_in_bootstrap_denominator():
    value=bootstrap_clusters({'p1':[1,None],'p2':[None,None]},resamples=500)
    assert value['scheduled_rows']==4 and value['usable_rows']==1
    assert value['missing_rows']==3 and value['estimate']==1
    assert value['bootstrap_scheduled_rows_range']==[4,4]
    assert value['bootstrap_usable_rows_range']==[0,2]
    assert value['valid_bootstrap_draws']<500
    assert value['descriptive_95_interval'] is None
    assert value['interval_unavailable_reason']=='fewer_than_two_usable_parents'


def test_parent_variants_are_resampled_together():
    value=bootstrap_clusters({'p1':[0,2],'p2':[10,12]},resamples=1000)
    assert value['estimate']==6
    # Row-wise resampling would allow endpoints 0 and 12. Cluster means are 1 and 11.
    assert value['descriptive_95_interval']==[1,11]


def test_all_unknown_is_null_not_zero():
    value=bootstrap_clusters({'a':[None,None],'b':[None,None]},resamples=20)
    assert value['estimate'] is None and value['descriptive_95_interval'] is None
    assert value['valid_bootstrap_draws']==0 and value['missing_rows']==4


def test_single_parent_cannot_establish_independent_parent_interval():
    value=bootstrap_clusters({'a':[1,3]},resamples=20)
    assert value['estimate']==2 and value['descriptive_95_interval'] is None


def test_bootstrap_is_reproducible():
    clusters={'a':[None,2],'b':[1,3],'c':[10,None]}
    assert bootstrap_clusters(clusters,resamples=30)==bootstrap_clusters(clusters,resamples=30)


@pytest.mark.parametrize('clusters',[{}, {'a':[]}])
def test_empty_parent_denominator_rejected(clusters):
    with pytest.raises(ValueError,match='Empty parent'):
        bootstrap_clusters(clusters,resamples=10)


def test_known_label_requires_native_feasibility():
    r=row();r['logical_status']='UNKNOWN'
    assert not known(r)


@pytest.mark.parametrize('field',['pool','generation_reports'])
def test_fixed_inventory_score_is_not_generated_quality(field):
    r=row();r[field]=None if field=='pool' else []
    assert known(r) and not generated_known(r)


def test_summary_keeps_failed_and_unavailable_rows():
    rows=[row(),row(case='other',usable=False)]
    rows[1].update(scientific_status='unavailable_model_schema',process_status='failed')
    value=summary(rows,True)
    assert value['scheduled_rows']==2 and value['quality_usable_rows']==1
    assert value['quality_unavailable_rows']==1 and value['row_wall_seconds']==20
    assert value['scientific_statuses']['unavailable_model_schema']==1


def test_paired_quality_requires_both_labels_but_effort_keeps_all_pairs():
    rows=[row('learned','a','p1'),row('uniform','a','p1',False),
          row('learned','b','p2'),row('uniform','b','p2')]
    result=paired_comparison(rows,'learned','uniform',True)
    assert result['scheduled_pairs']==2
    quality=result['metrics']['conditional_benefit_difference']
    assert quality['usable_rows']==1 and quality['missing_rows']==1
    assert result['metrics']['wall_seconds_difference']['usable_rows']==2


def test_paired_comparison_rejects_missing_and_duplicate_rows():
    rows=[row(),row('uniform')]
    with pytest.raises(ValueError,match='Missing paired'):
        paired_comparison(rows[:1],'learned','uniform',True)
    with pytest.raises(ValueError,match='Duplicate'):
        paired_comparison(rows+[rows[0]],'learned','uniform',True)


def test_paired_comparison_rejects_false_parent_independence():
    rows=[row(),row('uniform',group='renamed')]
    with pytest.raises(ValueError,match='Paired parent'):
        paired_comparison(rows,'learned','uniform',True)


def full_schedule():
    cases=[dict(case_id=f'c{i}',group_id=f'p{i//2}',family='f',family_exposure='seen_family',
                control='coherent' if i%2 else 'corrupted') for i in range(64)]
    entries=[dict(id=f'{a}:{i}',arm_id=a,case_id=c['case_id'],case_index=i)
             for a in range(9) for i,c in enumerate(cases)]
    rows=[dict(row=e,**{k:v for k,v in cases[e['case_index']].items() if k!='case_id'}) for e in entries]
    return rows,dict(rows=entries,cases=cases)


def test_all_576_schedule_rows_and_32_parents_required():
    rows,schedule=full_schedule();validate_rows(rows,schedule)
    with pytest.raises(ValueError,match='denominator'):
        validate_rows(rows[:-1],schedule)
    rows[-1]=deepcopy(rows[0])
    with pytest.raises(ValueError,match='Missing or duplicate'):
        validate_rows(rows,schedule)


def test_renamed_parent_is_not_a_new_independent_parent():
    rows,schedule=full_schedule();rows[0]['group_id']='renamed'
    with pytest.raises(ValueError,match='ancestry'):
        validate_rows(rows,schedule)


def test_failed_attempt_charges_are_union_not_double_counted():
    charge=dict(logical_id='old',status='settled',elapsed_seconds=10)
    source=dict(run=dict(id='old-001',superseded_by='old-002'),charge=charge,
                completion=dict(job_id='old',status='failed',exit_code=1,step_id='14408.1'))
    ledger=dict(attempts={'/attempts/old/001':charge})
    costs,attempts=attempt_costs({'primary':dict(attempts=[source]),
                                'duplicate_reference':dict(attempts=[source])},[],ledger)
    assert len(costs)==len(attempts)==1
    assert attempts[0]['status']=='failed' and attempts[0]['superseded_by']=='old-002'


def test_missing_or_reduced_historical_charge_rejected():
    source=dict(run=dict(id='old-001'),charge=dict(logical_id='old',elapsed_seconds=10),
                completion=dict(job_id='old'))
    with pytest.raises(ValueError,match='charge missing'):
        attempt_costs({'primary':dict(attempts=[source])},[],dict(attempts={}))


def test_interval_excludes_unknown_draws_only_with_explicit_denominators():
    assert interval([None,None]) is None
    assert interval([None,3,3])==[3,3]
