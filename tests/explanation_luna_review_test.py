"""Synthetic review tests; these do not approve real provider claims."""
import copy
import json

import pyowl_core as core
import pytest
from fastapi.testclient import TestClient

from exact_inspect.artifacts import atomic_json, publish_bundle
from exact_inspect.context import build_context_package
from exact_inspect.contracts import EntityRef, VisibilityPolicy, canonical_hash
from exact_inspect.description_review import admit_independent_review, build_review_request
from exact_inspect.descriptions import prepare_description_packet, validate_prepared_support
from exact_inspect.generation import entity_packet
from exact_inspect.readable_descriptions import enrich
from exact_inspect.service import create_prepared_app
from exact_inspect.study.builder import build_explanation_resource
from exact_inspect.study.resources import ExplanationResource
from exact_inspect.study.workspace import _admit_explanation
from tools.review_luna_bundle import source_result, wire


@pytest.fixture
def reviewed_labels(tmp_path):
    context = build_context_package(core.load_snapshot(b'''Ontology(
      Declaration(Class(<urn:A>)) Declaration(Class(<urn:B>))
      AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:A> "Focal")
      AnnotationAssertion(<http://www.w3.org/2000/01/rdf-schema#label> <urn:B> "Parent")
      SubClassOf(<urn:A> <urn:B>))'''), tmp_path/'context')
    entity = EntityRef(ontology_version_id=context.ontology_version_id, iri='urn:A', kind='class')
    contexts = {context.ontology_version_id: context}
    policy = VisibilityPolicy(ontology_ids=(context.ontology_version_id,))
    facts = sum((context.facts(entity, category=c, policy=policy)['items'] for c in ['labels','hierarchy']), [])
    packet = prepare_description_packet(entity_packet(entity, facts, context_hash=context.manifest['artifacts']['context.sqlite'], policy=policy), context, policy)
    packet = enrich(packet, contexts, policy)
    packet.selection['description_prose'] = True
    label = next(f for f in packet.selection['referenced_labels'] if f['subject']['iri']=='urn:B')
    ids = list(dict.fromkeys([f['fact_id'] for f in packet.facts]+[label['fact_id']]))
    raw = {'model':'openai/gpt-6-luna','provider':'OpenAI','choices':[{'finish_reason':'stop','message':{'content':json.dumps({'claims':[{'text':'Focal is a subclass of Parent.','fact_ids':ids,'category':'scope'}],'limitations':[],'relation':None})}}]}
    identity = {'packet':packet.model_dump(mode='json'),'payload':wire(packet)}
    job = {'job_id':canonical_hash(identity),'identity':identity}
    source = source_result(job, raw)
    request = build_review_request(source, raw)
    receipt = {'contract_version':'exact-description-review/1',**request['bindings'],
      'reviewer_id':'synthetic-label-test','reviewer_kind':'independent_agent',
      'review_provenance':'Synthetic test receipt; no real model output or semantic acceptance is represented.',
      'decisions':[{'ordinal':0,**request['claim_inventory'][0],'decision':'approved',
                    'support_explanation':'Synthetic assertion follows the outgoing edge with the two exact typed labels.'}]}
    result = admit_independent_review(source,raw,receipt)
    return contexts, policy, packet, source, raw, receipt, result


def test_reviewed_endpoint_labels_cross_main_frozen_and_scoped_boundaries(reviewed_labels,tmp_path):
    contexts,policy,packet,source,raw,receipt,result = reviewed_labels
    validate_prepared_support(result,contexts,policy)
    resource=build_explanation_resource(contexts,packet.entities,policy=policy,packets=[packet],explanations=[result])
    assert resource.entity_profiles[0].grounding=='independent_review'
    assert any(f.subject.iri=='urn:B' for f in resource.referenced_labels)
    case={'source':packet.entities[0].model_dump(),'candidates':[]}
    assert _admit_explanation(result,case,policy,contexts)['explanation_id']==result['explanation_id']
    atomic_json(tmp_path/'reviewed.json',result)
    manifest=publish_bundle(tmp_path,policy=policy,ontologies={packet.entities[0].ontology_version_id:'context'},explanations={result['explanation_id']:'reviewed.json'})
    with TestClient(create_prepared_app(manifest)) as client:
        response=client.get('/api/v1/explanations/'+result['explanation_id'])
        assert response.status_code==200,response.text
        assert response.json()['claims'][0]['fact_ids']==result['claims'][0]['fact_ids']


@pytest.mark.parametrize('field,value',[('kind','object_property'),('iri','urn:outside')])
def test_review_request_rejects_unbound_typed_label(reviewed_labels,field,value):
    _,_,_,source,raw,_,_=reviewed_labels
    altered=copy.deepcopy(source)
    labels=altered['support_packet']['selection']['referenced_labels']
    next(x for x in labels if x['subject']['iri']=='urn:B')['subject'][field]=value
    altered['manifest']['packet_hash']=canonical_hash(altered['support_packet'])
    with pytest.raises(ValueError,match='exact typed reference'):
        build_review_request(altered,raw)


def test_frozen_review_rejects_changed_reference_label_even_with_rebound_receipt(reviewed_labels):
    contexts,policy,packet,source,raw,receipt,result=reviewed_labels
    resource=build_explanation_resource(contexts,packet.entities,policy=policy,packets=[packet],explanations=[result]).model_dump(mode='json')
    # Change only the admitted reference label, leaving the signed support packet intact.
    next(f for f in resource['referenced_labels'] if f['subject']['iri']=='urn:B')['value']['lexical_form']='Invented Parent'
    with pytest.raises(ValueError,match='Reviewed packet value'):
        ExplanationResource.model_validate(resource)


def test_provider_never_becomes_approved_without_receipt(reviewed_labels):
    *_,source,raw,receipt,result=reviewed_labels
    assert source['support_status']=='fallback'
    assert result['support_status']=='independently_reviewed'
