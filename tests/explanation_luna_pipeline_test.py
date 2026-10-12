"""Small synthetic publication and interruption checks; no hosted model calls."""
import json

import pytest
from fastapi.testclient import TestClient

from exact_inspect.artifacts import atomic_json, publish_bundle, validate_bundle
from exact_inspect.context import OntologyContext
from exact_inspect.context_export import export_policy_context
from exact_inspect.contracts import EntityRef, VisibilityPolicy, canonical_hash, file_hash
from exact_inspect.description_review import build_review_request
from exact_inspect.descriptions import prepare_description_packet
from exact_inspect.generation import entity_packet
from exact_inspect.readable_descriptions import enrich
from exact_inspect.service import create_prepared_app
from exact_inspect.study.api import create_study_app
from exact_inspect.study.builder import build_explanation_resource
from exact_inspect.study.models import Publish
from exact_inspect.study.workspace import build_workspace_scope
from tests.explanation_review_publication_test import review_fixture
from tools.publish_luna_review import case_scope, publish
from tools.review_luna_bundle import freeze, snapshot, source_result, wire


@pytest.fixture
def staged(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    raw = review_fixture(base)
    # This deliberately small fixture publishes only a profile; the existing
    # tutorial still retains its complete comparison and evidence resources.
    raw['definition']['components']=['original_context','entity_description','hierarchy']
    policy = VisibilityPolicy.model_validate(raw['definition']['visibility_policy'])
    contexts = {}
    for name in ('source', 'target'):
        context = export_policy_context(OntologyContext(base / (name+'-v2-context')),
                                        base/'package'/name, policy)
        contexts[context.ontology_version_id] = context
    entity = EntityRef.model_validate(raw['definition']['cases'][0]['candidates'][0]['entity'])
    context = contexts[entity.ontology_version_id]
    facts = context.facts(entity, category='definitions', policy=policy)['items']
    packet = prepare_description_packet(entity_packet(entity, facts,
        context_hash=context.manifest['artifacts']['context.sqlite'], policy=policy), context, policy)
    packet = enrich(packet, contexts, policy)
    packet.selection['description_prose'] = True
    response = {'model':'openai/gpt-6-luna','provider':'OpenAI',
        'choices':[{'finish_reason':'stop','message':{'content':json.dumps({
            'claims':[{'text':'Invented test concept target 1', 'fact_ids':[facts[0]['fact_id']], 'category':'scope'}],
            'relation':None,'limitations':[]})}}]}
    identity = {'packet':packet.model_dump(mode='json'), 'payload':wire(packet)}
    job = {'job_id':canonical_hash(identity),'identity':identity}
    source = source_result(job,response)
    job['old_id'] = source['explanation_id']
    atomic_json(base/'package/original.json',source)
    resource = build_explanation_resource(contexts,[entity],policy=policy,packets=[packet],explanations=[source])
    scopes = [s for s in raw['definition']['workspace_scopes'] if s['kind']=='tutorial']
    for case in raw['definition']['cases']:
        rel = 'original-'+case['case_id']+'.json'
        atomic_json(base/rel, resource.model_dump(mode='json'))
        case['explanation_refs'] = [rel]
        raw['definition']['assets'].append({'asset_id':rel,'path':rel,
            'sha256':file_hash(base/rel)[7:],'size_bytes':(base/rel).stat().st_size,
            'kind':'explanation','media_type':'application/json','policy_hash':policy.policy_hash[7:]})
        scopes.append(build_workspace_scope(base,scope_id='original-'+case['case_id'],kind='case',
            case=case,contexts=contexts,policy=policy,explanations=[source],pair_metadata={},
            components=['context','hierarchy','profiles'],
            reuse_prepared_contexts=True).model_dump(mode='json'))
    raw['definition']['workspace_scopes'] = scopes
    # Public provenance and counters are checked as well as the served prose.
    from tests.explanation_review_publication_test import with_catalogue
    with_catalogue(raw)
    raw['definition']['review_bundle']['delivered_layers']={'recovered_evidence':canonical_hash('evidence')}
    Publish.model_validate(raw)
    manifest = publish_bundle(base/'package',policy=policy,
        ontologies={v:str(c.path.relative_to(base/'package')) for v,c in contexts.items()},
        explanations={source['explanation_id']:'original.json'},
        provenance={'review_bundle':raw['definition']['review_bundle']})
    atomic_json(base/'publication.json',raw)
    atomic_json(base/'catalogue.json',{'cases':[{'case_id':c['case_id']} for c in raw['definition']['cases']],
        'generation_attempt':7,'generation_summary':{'historical':'preserved'}})
    atomic_json(base/'composition.json',{'synthetic':True})
    root = tmp_path/'revision'
    plan = {'base':str(base),'jobs':[job],'created_utc':'2026-10-12T00:00:00Z',
        'publication_sha256':file_hash(base/'publication.json'),
        'package_sha256':file_hash(manifest),'composition_sha256':file_hash(base/'composition.json'),
        'base_file_stats':{str(p):snapshot(p) for p in base.rglob('*') if p.is_file()}}
    freeze(root/'plan.json',plan)
    freeze(root/'generation-complete.json',{'plan_sha256':canonical_hash(plan)})
    gen = root/'generation'/job['job_id'][7:]
    freeze(gen/'source.json',source)
    freeze(gen/'response.json',{'response':response,'sha256':canonical_hash(response)})
    request = build_review_request(source,response)
    freeze(root/'review-requests'/(job['job_id'][7:]+'.json'),request)
    receipt={'contract_version':'exact-description-review/1',**request['bindings'],
        'reviewer_id':'synthetic-fixture','reviewer_kind':'independent_agent',
        'review_provenance':'Synthetic mechanical regression only, not an actual independent semantic review.',
        'decisions':[{'ordinal':0,**request['claim_inventory'][0], 'decision':'approved',
            'support_explanation':'This fixture exactly quotes its single supplied original definition.'}]}
    return base,root,raw,job,receipt,contexts,policy


@pytest.mark.parametrize('approve',[True,False])
def test_complete_pipeline_rejects_missing_review_and_preserves_old_publication(staged, approve, monkeypatch):
    base,root,old,job,review,contexts,policy=staged
    with pytest.raises(ValueError,match='separate complete review'):
        publish(root)
    if not approve:
        review['decisions'][0]['decision']='rejected'
        review['decisions'][0]['support_explanation']='Synthetic rejected-draft scenario must retain the original safely.'
    freeze(root/'reviews'/(job['job_id'][7:]+'.json'),review)
    if approve:
        import tools.publish_luna_review as publisher
        def interrupted(path,value):
            if path.parent.name=='assembly':
                raise RuntimeError('injected interruption after index install')
            return freeze(path,value)
        with monkeypatch.context() as patch:
            patch.setattr(publisher,'freeze',interrupted)
            with pytest.raises(RuntimeError,match='injected interruption'):
                publish(root)
    receipt=publish(root)
    assert receipt['summary']['luna_reviewed_artifacts']==int(approve)
    assert receipt['summary']['retained_predecessor_artifacts']==int(not approve)
    assert publish(root)==receipt
    bundle=validate_bundle(base/'package/package-luna-20261012.json')
    before=validate_bundle(base/'package/package.json')
    assert before.explanations=={job['old_id']:'original.json'}
    assert bundle.provenance['review_bundle']['delivered_layers']['recovered_evidence']==canonical_hash('evidence')
    assert bundle.provenance['review_bundle']['delivered_layers']['luna_reviewed_prose']==canonical_hash(receipt['binding'])
    catalogue=json.loads((base/'catalogue-luna-20261012.json').read_bytes())
    assert catalogue['predecessor_generation']['generation_attempt']==7
    assert 'generation_attempt' not in catalogue
    selected=receipt['supersessions'][job['old_id']]['replacement']
    with TestClient(create_prepared_app(base/'package/package-luna-20261012.json')) as client:
        result=client.get('/api/v1/explanations/'+selected)
        assert result.status_code==200,result.text
        assert result.json()['support_status']==('independently_reviewed' if approve else 'fallback')
    newer=json.loads((base/'publication-luna-20261012.json').read_bytes())
    app=create_study_app(database_url='sqlite:///'+str(root/'test.sqlite'),
        signing_secret='s'*48,researcher_token='r'*48,origin='https://localhost',assets_dir=base,allow_test_sqlite=True)
    with TestClient(app,base_url='https://localhost') as client:
        headers={'Authorization':'Bearer '+'r'*48}
        for pub in [old,newer,old,newer]:
            response=client.post('/api/v1/admin/studies',json=pub,headers=headers)
            assert response.status_code==200,response.text
    assert json.loads((base/'publication.json').read_bytes())==old


def test_interrupted_small_index_is_reused_only_when_identical(staged):
    base,root,old,job,review,contexts,policy=staged
    case=old['definition']['cases'][0]
    args=dict(scope_id='interrupted',kind='case',case=case,contexts=contexts,policy=policy,pair_metadata={})
    scope=case_scope(base,**args)
    target=base/scope.index_path
    before=snapshot(target)
    assert case_scope(base,**args)==scope
    assert snapshot(target)==before
    target.write_bytes(b'corrupt private test artifact')
    with pytest.raises(ValueError,match='Interrupted case index differs'):
        case_scope(base,**args)
    assert target.read_bytes()==b'corrupt private test artifact'
    assert not list((base/'workspaces').glob('*/workspace.sqlite-journal'))


def test_admission_hash_ignores_read_access_time_but_rejects_content_changes(tmp_path,monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from tools import review_environment as environment
    target=tmp_path/'immutable-input'
    target.write_bytes(b'original')
    original_stat=Path.stat
    original_hash=environment.file_hash
    read_happened=False
    def stat(path,*args,**kwargs):
        value=original_stat(path,*args,**kwargs)
        if path==target and read_happened:
            fields={name:getattr(value,name) for name in dir(value) if name.startswith('st_')}
            fields.update(st_atime=value.st_atime+10,st_atime_ns=value.st_atime_ns+10_000_000_000)
            return SimpleNamespace(**fields)
        return value
    def read_once(path):
        nonlocal read_happened
        digest=original_hash(path)
        read_happened=True
        return digest
    monkeypatch.setattr(Path,'stat',stat)
    monkeypatch.setattr(environment,'file_hash',read_once)
    hasher=environment.AdmissionHashes()
    assert hasher(target)==original_hash(target)
    assert hasher(target)==original_hash(target)
    assert hasher.bytes_read==len(b'original')
    def altered_during_read(path):
        digest=original_hash(path)
        path.write_bytes(b'modified')
        return digest
    monkeypatch.setattr(environment,'file_hash',altered_during_read)
    with pytest.raises(environment.ReviewError,match='changed during admission'):
        environment.AdmissionHashes()(target)
