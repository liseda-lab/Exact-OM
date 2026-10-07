"""Consolidated closure must preserve unknowns, deferred claims and inherited costs."""
from copy import deepcopy
import pytest
from tools.repair.preliminary_report import (
    BRANCHES, StreamEvidence, compact, reconcile_costs, validate_denominators,
)
from tools.repair.overlap_audit import binding


def reports():
    r={k:{} for k in BRANCHES}
    r['historical']['denominators']=dict(cases=90,formerly_failed_cases=79,generation_rows=180,native_rows=270)
    r['corpus'].update(scheduled_cases=288,training_target_cases=224,fresh_evaluation_target_cases=64,
                       teacher_queries=0,test_outcomes_opened=False)
    for k in ('evaluation','training','robustness'):
        r[k].update(local_reporting_complete=True,study_complete=False,campaign_complete=False,
            gates_passed=False,supervision_admitted=False,scientific_rows_replayed=0,new_native_calls=0)
    b=dict(scheduled_rows=576,recorded_rows=576,rows=[{}]*576,parent_groups=32,
           quality_usable_rows=0,quality_unavailable_rows=576)
    r['evaluation']['branches']={k:deepcopy(b) for k in ('primary','inventory')}
    r['training'].update(original_fixed_inventory=dict(scheduled_cases=160),
        decode_diagnostic=dict(scheduled_rows=288),intended_qualification=dict(scheduled_cases=32),
        collection=dict(summary=dict(scheduled_cases=32,parent_groups=16,scheduled_slots=256)),
        fitting_eligible=False,heldout_cases_opened=False)
    r['robustness'].update(scheduled_rows=2880,recorded_rows=2880,rows=[{}]*2880,base_cases=64,
        case_variants=320,parent_groups=32,paired_interventions=[{}]*36,quality_usable_rows=0,
        quality_unavailable_rows=2880,overlap=dict(scheduled_rows=36,recorded_rows=36))
    r['scaling'].update(scheduled=336,recorded=336,branches={k:dict(summary=dict(scheduled=n,recorded=n),rows=[{}]*n)
        for k,n in [('primary',240),('incidence',72),('endpoint_amendment',24)]})
    r['real'].update(scheduled_rows=54,recorded_rows=54,scheduled_cases=6,independent_pair_count=1)
    return r


def test_all_unknown_quality_is_accounted_without_becoming_success():
    validate_denominators(reports())


@pytest.mark.parametrize('branch', sorted(BRANCHES))
def test_every_registered_scientific_branch_required(branch):
    r=reports();del r[branch]
    with pytest.raises(ValueError,match='branch'):validate_denominators(r)


@pytest.mark.parametrize('mutation', ['drop_unknown','overlap','independence','fit','gate','heldout','replay'])
def test_scope_cannot_be_promoted_or_rows_dropped(mutation):
    r=reports()
    if mutation=='drop_unknown':r['robustness']['quality_unavailable_rows']=2879
    if mutation=='overlap':r['robustness']['overlap']['recorded_rows']=35
    if mutation=='independence':r['real']['independent_pair_count']=2
    if mutation=='fit':r['training']['fitting_eligible']=True
    if mutation=='gate':r['evaluation']['gates_passed']=True
    if mutation=='heldout':r['training']['heldout_cases_opened']=True
    if mutation=='replay':r['robustness']['scientific_rows_replayed']=1
    with pytest.raises(ValueError):validate_denominators(r)


def cost(seconds,status='settled'):
    value=dict(status=status,reserved_seconds=seconds,resources=dict(cpus=2,gpus=0,memory_mb=16384))
    if status=='settled':value['elapsed_seconds']=seconds
    return value


def ledger():
    return dict(limit_worker_seconds=None,attempts=dict(pilot=cost(5),failed=cost(7),
        maintenance=cost(10,'reserved')),cumulative=dict(worker_seconds=22))


def test_costs_preserve_failed_attempt_reservation_and_inherited_once():
    result=reconcile_costs(ledger(),dict(attempts=dict(pilot=cost(5))),dict(attempts=dict(smoke=cost(3))))
    assert result['combined']['worker_seconds']==15
    assert result['combined']['reserved_worker_seconds']==10
    assert result['combined']['attempts']==4


@pytest.mark.parametrize('mutation',['pilot','duplicate','total','limit'])
def test_cost_resets_or_double_count_rejected(mutation):
    l=ledger();pilot=dict(attempts=dict(pilot=cost(5)));smoke=dict(attempts=dict(smoke=cost(3)))
    if mutation=='pilot':l['attempts']['pilot']=cost(4)
    if mutation=='duplicate':smoke['attempts']['pilot']=cost(5)
    if mutation=='total':l['cumulative']['worker_seconds']=12
    if mutation=='limit':l['limit_worker_seconds']=172800
    with pytest.raises(ValueError):reconcile_costs(l,pilot,smoke)


def test_compact_keeps_missing_status_and_deferrals_without_duplicating_rows():
    source=dict(rows=[dict(status='unknown')],scientific_statuses=dict(unknown=1),
                deferred_obligations=['18 models'],gates_passed=False)
    result=compact(source)
    assert 'rows' not in result and result['scientific_statuses']==dict(unknown=1)
    assert result['deferred_obligations']==['18 models'] and result['gates_passed'] is False


def test_streamed_receipts_reject_changed_or_conflicting_hashes(tmp_path):
    p=tmp_path/'native';p.write_bytes(b'x'*1024);ref=binding(p);e=StreamEvidence()
    assert e.verify(ref)==p
    with pytest.raises(ValueError,match='Conflicting'):e.verify(dict(ref,sha256='0'*64))
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed'):StreamEvidence().verify(ref)


def test_full_history_keeps_disabled_failed_ancestor_and_authenticates_nonce(tmp_path):
    import json
    from tools.repair.preliminary_report import authenticate_history
    p=tmp_path/'failed'/'completion.json';p.parent.mkdir()
    p.write_text(json.dumps(dict(step_id='14408.12',dispatch_nonce='n',status='failed',exit_code=1,job_id='x')))
    run=dict(id='x-001',completion_path=str(p),step_id='14408.12',dispatch_nonce='n',enabled=False,superseded_by='x-002')
    charge=dict(status='settled',logical_id='x')
    args=(dict(runs=[run]),dict(attempts={str(p.parent):charge}),[dict(id=run['id'],completion=binding(p))])
    result=authenticate_history(StreamEvidence(),*args)
    assert result[0]['status']=='failed' and result[0]['run']['superseded_by']=='x-002'
    changed=deepcopy(args);changed[0]['runs'][0]['dispatch_nonce']='wrong'
    with pytest.raises(ValueError,match='nonce'):authenticate_history(StreamEvidence(),*changed)
    changed=deepcopy(args);changed[0]['runs'][0].update(enabled=True,superseded_by=None)
    with pytest.raises(ValueError,match='failed'):authenticate_history(StreamEvidence(),*changed)
    with pytest.raises(ValueError,match='denominator'):authenticate_history(StreamEvidence(),args[0],args[1],[])
