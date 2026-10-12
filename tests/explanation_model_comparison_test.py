"""Negative tests for the isolated model diagnostic's admission boundaries."""
import json

import pytest

from exact_inspect.contracts import canonical_hash
from exact_inspect.generation import FactPacket
from tools.review_model_comparison import MODELS, corrected_wire, payload, save, validate_output


@pytest.fixture
def packet():
    entity = {'ontology_version_id': 'sha256:' + 'a' * 64, 'iri': 'urn:focal', 'kind': 'class'}
    parent = {**entity, 'iri': 'urn:parent'}
    label = {'fact_id': 'parent-label', 'subject': parent, 'category': 'labels',
             'value': {'term_type': 'literal', 'lexical_form': 'Parent'}}
    return FactPacket(task='entity_profile', entities=[entity], context_hashes=[], policy_hash='p',
        missingness=['No definition'], facts=[{'fact_id':'edge', 'subject':entity, 'category':'hierarchy',
        'value': {'ast': {'type':'SubClassOf', 'sub_class':{'type':'Class','iri':{'value':'urn:focal'}},
                         'super_class':{'type':'Class','iri':{'value':'urn:parent'}}}}}],
        selection={'render_revision':'typed-readable/3', 'referenced_labels':[label]})


def response(claims, **updates):
    value = {'claims':claims, 'limitations':[], 'relation':None, **updates}
    return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]}


def test_corrected_labels_reach_both_models_without_changing_input(packet):
    original = canonical_hash(packet.model_dump(mode='json'))
    wire = corrected_wire(packet)
    assert wire['referenced_labels'][0]['value']['lexical_form'] == 'Parent'
    a, b = [payload({'messages':[{'role':'user','content':json.dumps(wire)}]}, m) for m in MODELS]
    assert a['messages'] == b['messages']
    assert a['response_format'] == b['response_format']
    assert 'temperature' not in b and b['reasoning']['effort'] == 'medium'
    assert a['provider'] == b['provider'] == {'only':['OpenAI'], 'allow_fallbacks':False,'require_parameters':True}
    assert canonical_hash(packet.model_dump(mode='json')) == original


def test_wrong_typed_reference_label_is_rejected(packet):
    packet.selection['referenced_labels'][0]['subject']['kind'] = 'object_property'
    with pytest.raises(ValueError, match='exact typed reference'):
        corrected_wire(packet)


@pytest.mark.parametrize('ids', [[], ['outside-packet']])
def test_invalid_citations_are_not_admitted(packet, ids):
    _, errors = validate_output(response([{'text':'A statement','fact_ids':ids,'category':'scope'}]), packet)
    assert 'invalid_citation_membership' in errors


def test_reference_label_is_a_valid_citation_but_no_semantic_approval_is_returned(packet):
    value, errors = validate_output(response([{'text':'Unreviewed prose','fact_ids':['edge','parent-label'],'category':'scope'}]), packet)
    assert not errors
    assert 'approved' not in value


@pytest.mark.parametrize('updates,error', [({'relation':'equivalent'},'claim_or_relation_contract'), ({'limitations':['Invented absence']},'invented_missingness'), ({'unapproved_field':True},'invalid_json_or_schema')])
def test_unadmitted_relation_missingness_and_fields_fail(packet, updates, error):
    _, errors = validate_output(response([], **updates), packet)
    assert error in errors


def test_truncated_output_is_not_accepted(packet):
    raw = response([])
    raw['choices'][0]['finish_reason'] = 'length'
    assert 'non_stop_finish' in validate_output(raw, packet)[1]


def test_wire_byte_cap_precedes_dispatch():
    with pytest.raises(ValueError, match='byte cap'):
        payload({'messages':[{'role':'user','content':'x' * 65536}]}, MODELS[0])


def test_frozen_study_files_cannot_be_silently_replaced(tmp_path):
    path = tmp_path/'review.json'
    save(path, {'approved':False})
    save(path, {'approved':False})
    with pytest.raises(ValueError, match='Refusing to replace'):
        save(path, {'approved':True})
