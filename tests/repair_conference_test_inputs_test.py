"""Closed TEST guards, source provenance and terminal no-replay semantics."""
import zipfile

import pytest

from exact.repair.workers import CallResult
from tools.repair import conference_test_inputs as inputs
from tools.repair.corrective_conference import immutable, binding
from tests.repair_corrective_conference_test import alignment


def frozen_plan(tmp_path):
    rows = [dict(id=f'pair-{i}', split='test' if i < 7 else 'train') for i in range(21)]
    declaration = immutable(tmp_path/'declaration.json', dict(rows=rows))
    cases = [dict(case_id=f'pair-{i}', cohort='conference', split='test', structural_parent=f'p-{i}') for i in range(7)]
    schedule = immutable(tmp_path/'schedule.json', dict(conference_declaration=declaration, cases=cases, models_freeze_epoch=100))
    models = {f'hgt-pair-{c}-s{s}': None for s in (13,37,73) for c in ('symbolic','symbolic_plus_llm')}
    freeze = immutable(tmp_path/'freeze.json', dict(models=models, schedule=schedule, test_outcomes_opened=False, final_freeze=True))
    return dict(schema=inputs.SCHEMA, execution_authorized=True, schedule=schedule, model_freeze_receipt=freeze,
                not_before_epoch=100, case_ids=[c['case_id'] for c in cases], case_seconds=120)


def test_closed_guard_and_six_explicit_unavailable_models(tmp_path):
    plan = frozen_plan(tmp_path)
    with pytest.raises(ValueError, match='remains closed'):
        inputs.validate(plan, 99)
    rows, cases = inputs.validate(plan, 100)
    assert len(rows) == len(cases) == 7
    with pytest.raises(ValueError, match='denominator'):
        inputs.validate(dict(plan, case_ids=plan['case_ids'][:-1]), 100)


def test_constructor_preserves_whole_ontologies_and_never_reads_reference(tmp_path):
    ontologies = []
    for side, name in [('left', 'A'), ('right', 'B')]:
        p = tmp_path/f'{side}.owl'
        p.write_text(f'Ontology(<urn:{side}> Declaration(Class(<urn:{side}:{name}>)))')
        ontologies.append(binding(p))
    archive = tmp_path/'matcher.zip'
    with zipfile.ZipFile(archive, 'w') as out:
        out.writestr('LogMap-left-right.rdf', alignment())
    row = dict(id='pair', split='test', ontology_names=['left','right'], ontology_bindings=ontologies,
               matcher_archive=binding(archive), matcher_member='LogMap-left-right.rdf',
               reference_bindings=[dict(path=str(tmp_path/'NEVER-EXISTS-REFERENCE'), sha256='closed')])
    case = dict(case_id='pair', structural_parent='parent', split='test')
    result = inputs.materialize(row, case, tmp_path/'output')
    assert result['status'] == 'public_input_constructed'
    public = inputs.bound(result['public_input'])
    assert public['metadata']['split'] == 'test'
    assert len(public['observable_mapping_scores']) == 1
    assert result['complete_source_axiom_counts'] == [1, 1]
    assert not result['reference_rows_opened'] and not result['logical_qualification_claimed']
    with zipfile.ZipFile(archive, 'w') as out:
        out.writestr('LogMap-left-right.rdf', alignment(left='urn:LEFT:A'))
    row['matcher_archive'] = binding(archive)
    mismatch = inputs.materialize(row, case, tmp_path/'mismatch')
    assert mismatch['status'] == 'unavailable_source_namespace_mismatch'
    assert mismatch['public_input'] is None


def test_timeout_is_terminal_and_includes_worker_budget(tmp_path, monkeypatch):
    calls=[]
    monkeypatch.setattr(inputs.time, 'time', lambda: 100)
    monkeypatch.setattr(inputs, 'bounded_call', lambda *a, **k: calls.append(k) or CallResult('timeout'))
    case=dict(case_id='case')
    ref = inputs.one_case({}, case, tmp_path, 'identity', 120, 130)
    assert inputs.bound(ref)['status'] == 'timeout'
    assert calls[0]['timeout'] == 28 and calls[0]['cpu_seconds'] == 84
    assert inputs.one_case({}, case, tmp_path, 'identity', 120, 900) == ref
    assert len(calls) == 1


def test_interrupted_or_cleanup_failed_row_cannot_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs.time, 'time', lambda: 100)
    monkeypatch.setattr(inputs, 'bounded_call', lambda *a, **k: CallResult('timeout', cleanup_complete=False))
    case=dict(case_id='case')
    with pytest.raises(RuntimeError, match='implementation/cleanup'):
        inputs.one_case({}, case, tmp_path, 'identity', 120, 130)
    monkeypatch.setattr(inputs, 'bounded_call', lambda *a, **k: pytest.fail('replayed'))
    with pytest.raises(RuntimeError, match='Prior'):
        inputs.one_case({}, case, tmp_path, 'identity', 120, 900)


def public_producer(tmp_path):
    from exact.repair.records import canonical_hash
    from exact.repair.api import write_artifact
    plan = frozen_plan(tmp_path)
    plan.update(source_commit='source', dispatch_nonce='producer-nonce')
    manifest = immutable(tmp_path/'producer.json', plan)
    work, attempt = tmp_path/'work', tmp_path/'attempt'
    report_rows=[]
    for index, case_id in enumerate(plan['case_ids']):
        public = immutable(work/f'public-{index}.json', dict(input='fixture')) if index == 0 else None
        saved = dict(status='complete', cleanup_complete=True, result=dict(public_input=public))
        ref = immutable(work/f'row-{index}.json', saved)
        report_rows.append(dict(case_id=case_id, completion=ref, **saved))
    immutable(work/'report.json', dict(manifest_identity=canonical_hash(plan), expected_cases=7, rows=report_rows))
    outputs={str(p.relative_to(work)): binding(p)['sha256'] for p in work.glob('*.json')}
    batch=immutable(tmp_path/'batch.json', dict(frozen_files={manifest['path']:manifest['sha256']}, jobs=[dict(id='producer', commands=[['python',manifest['path']]])]))
    (tmp_path/'batch.sha256').write_text(batch['sha256'])
    immutable(attempt/'completion.json', dict(status='complete', exit_code=0, job_id='producer', work=str(work), batch=batch['path'], dispatch_nonce='producer-nonce', step_id='123.4'))
    immutable(attempt/'step.json', dict(dispatch_nonce='producer-nonce',step_id='123.4'))
    immutable(attempt/'outputs.json', outputs)
    cases=inputs.bound(plan['schedule'])['cases']
    consumer=dict(public_inputs={}, cases={c['case_id']:c for c in cases}, schedule=plan['schedule'], model_freeze_receipt=plan['model_freeze_receipt'], source_commit='source', public_input_producer=dict(manifest=manifest, attempt=str(attempt),work=str(work),job_id='producer',dispatch_nonce='producer-nonce'))
    return consumer, work, attempt


def test_public_handoff_authenticates_actual_output_and_retains_missing_inputs(tmp_path):
    from tools.repair.evaluation_ownership import external_public_inputs
    consumer, work, attempt=public_producer(tmp_path)
    result=external_public_inputs(consumer)
    assert set(result)=={'pair-0'}
    (work/'public-0.json').write_text('{}')
    with pytest.raises(ValueError, match='dependency changed'):
        external_public_inputs(consumer)


@pytest.mark.parametrize('tamper',['nonce','output','batch','freeze'])
def test_public_handoff_rejects_unbound_producer(tmp_path,tamper):
    from exact.repair.api import write_artifact
    from tools.repair.evaluation_ownership import external_public_inputs
    consumer, work, attempt=public_producer(tmp_path)
    if tamper=='nonce':
        consumer['public_input_producer']['dispatch_nonce']='other'
    elif tamper=='output':
        outputs=inputs.read(attempt/'outputs.json');outputs.pop('public-0.json')
        write_artifact(attempt/'outputs.json',outputs)
    elif tamper=='batch':
        (tmp_path/'batch.sha256').write_text('wrong')
    else:
        consumer['model_freeze_receipt']=None
    with pytest.raises(ValueError):
        external_public_inputs(consumer)
