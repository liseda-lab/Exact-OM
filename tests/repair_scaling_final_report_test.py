"""Final-report failures must remain visible in quality and paired denominators."""
import copy
import pytest

from tools.repair.scaling_final_report import incidence_pairs, summary, validate_lineage, semantic_evidence
from tools.repair.overlap_audit import Evidence, binding
from tools.repair.expanded_profile import checkpoint


def rows():
    base = dict(semantic_status='known', logical_status='VERIFIED_FEASIBLE', semantic_benefit=2.0,
        fallback_original_inventory=False, generation_complete=True, process_status='complete',
        generation_status='complete', elapsed_seconds=1.0)
    return [dict(copy.deepcopy(base), row=dict(id=c, case_index=i, method='grammar_circuit', cache_mode='cold'))
            for i,c in enumerate(('shared','nonshared'))]


def schedule():
    return dict(cases=[dict(parent='one-ancestry', corrupted_mapping_count=8, control='corrupted', condition=c)
                       for c in ('shared','nonshared')])


@pytest.mark.parametrize('changes', [dict(semantic_status='unavailable',semantic_benefit=None),
    dict(logical_status='UNKNOWN'), dict(fallback_original_inventory=True)])
def test_missing_or_fallback_quality_keeps_pair_and_denominator(changes):
    data=rows();data[1].update(changes)
    report=summary(data);pairs=incidence_pairs(schedule(),data)
    assert report['scheduled']==report['recorded']==2 and report['generated_pool_selected_quality_known']==1
    assert len(pairs)==1 and not pairs[0]['quality_available']
    assert pairs[0]['shared_minus_nonshared_benefit'] is None
    assert len(pairs[0]['rows'])==2


def test_partial_generation_known_quality_is_separate_from_complete_coverage():
    data=rows();data[1].update(generation_status='timeout',generation_complete=False)
    report=summary(data);pair=incidence_pairs(schedule(),data)[0]
    assert report['generated_pool_selected_quality_known']==2 and report['complete_generation_quality_known']==1
    assert report['generation_statuses']==dict(complete=1,timeout=1)
    assert pair['quality_available'] and not pair['both_generation_complete']


@pytest.mark.parametrize('data', [lambda:rows()[:1], lambda:rows()+rows()[:1]])
def test_missing_duplicate_pair_is_rejected(data):
    with pytest.raises(ValueError,match='incidence'):
        incidence_pairs(schedule(),data())


def test_clean_and_corrupt_pairs_cannot_be_combined():
    spec=schedule();spec['cases'][1]['control']='coherent'
    with pytest.raises(ValueError,match='Missing incidence pair'):
        incidence_pairs(spec,rows())


def test_lineage_accepts_healthy_continuation_without_repair_attempt():
    sources=[dict(run=dict(id='old',superseded_by='new')),
             dict(run=dict(id='new',recovery_of='old',repair_attempt=0,continuation_attempt=1))]
    assert set(validate_lineage(sources))=={'old','new'}


@pytest.mark.parametrize('sources', [
    [dict(run=dict(id='old',superseded_by='missing'))],
    [dict(run=dict(id='old',superseded_by='new')),dict(run=dict(id='new',recovery_of='other'))],
    [dict(run=dict(id='same')),dict(run=dict(id='same'))]])
def test_missing_or_corrupt_attempt_lineage_is_rejected(sources):
    with pytest.raises(ValueError): validate_lineage(sources)


def test_known_score_without_proof_is_rejected_but_unknown_is_retained(tmp_path):
    path=tmp_path/'row.json';checkpoint(path,'fixture',payloads=[])
    row=dict(receipt=binding(path),result=None,semantic_status='known')
    with pytest.raises(ValueError,match='saved selection evidence'):
        semantic_evidence(Evidence(),row,{})
    row['semantic_status']='unavailable'
    assert semantic_evidence(Evidence(),row,{})['known_score_verified'] is False


def test_matched_scientific_budgets_and_method_denominator():
    from tools.repair.scaling_final_report import validate_schedule, METHODS
    spec=dict(cases=[{}],generation_seconds=60,rows=[dict(id=m+c,method=m,cache_mode=c,
        case_index=0,seconds=300,cpu_seconds=600,memory_mb=8192) for m in METHODS for c in ('cold','warm')])
    validate_schedule(spec)
    bad=copy.deepcopy(spec);bad['rows'][1]['seconds']=301
    with pytest.raises(ValueError,match='budget'):validate_schedule(bad)
    bad=copy.deepcopy(spec);bad['rows'].pop()
    with pytest.raises(ValueError,match='denominator'):validate_schedule(bad)
