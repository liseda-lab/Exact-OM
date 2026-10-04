"""Output omissions cover ontology controls rebuilt from visible fixed premises."""
import dataclasses

import pyowl_core as owl
import pytest

from exact.repair.candidates import ontology_candidates, make_candidate
from exact.repair.grammar import mapping_grammar
from exact.repair.records import PolicyV2, RepairInputV2, RevisionObjectV2, promote_input_v3


def fixture(kind):
    narrow, omitted, allowed = [owl.Class(owl.IRI('urn:omission:' + n))
                                for n in ('Narrow', 'Omitted', 'Allowed')]
    role = owl.ObjectProperty(owl.IRI('urn:omission:r'))
    if kind == 'domain':
        original = owl.ObjectPropertyDomain(role, narrow)
    elif kind == 'range':
        original = owl.ObjectPropertyRange(role, narrow)
    else:
        original = owl.SubClassOf(narrow, owl.ObjectSomeValuesFrom(role, narrow))
    obj = RevisionObjectV2('o', 'ontology_axiom', (original,), (make_candidate('o', (original,), ('keep',)),), occurrence_id='fixture-occurrence')
    fixed = (owl.SubClassOf(narrow, omitted), owl.SubClassOf(narrow, allowed))
    return obj, fixed, narrow, omitted, allowed, role


@pytest.mark.parametrize('kind', ['domain', 'range', 'existential'])
def test_rebuilt_fixed_controls_respect_omission_but_keep_visible_premises(kind):
    obj, fixed, narrow, omitted, allowed, role = fixture(kind)
    # Supplied controls can already be filtered; the grammar regenerates them.
    raw = mapping_grammar(obj, (narrow, allowed), (role,), fixed_axioms=fixed)
    changed = mapping_grammar(obj, (narrow, allowed), (role,), fixed_axioms=fixed,
                              omitted_generation_symbols=(omitted.iri.value,))
    symbols = lambda c: {e for a in (*c.axioms, *c.active_expressions) for e in owl.signature(a)}
    assert any(t.fixed and omitted in symbols(t.fixed) for t in raw.templates)
    assert all(not t.fixed or omitted not in symbols(t.fixed) for t in changed.templates)
    assert any(t.fixed and allowed in symbols(t.fixed) for t in changed.templates)
    assert {'keep', 'delete'} <= {t.action for t in changed.templates}
    # Original editable vocabulary is exempt, and no-omission behavior is stable.
    assert mapping_grammar(obj, (narrow, allowed), (role,), fixed_axioms=fixed,
                           omitted_generation_symbols=(narrow.iri.value,)) == raw


@pytest.mark.parametrize('proposal', ['grammar_uniform', 'bounded_enumeration'])
@pytest.mark.parametrize('kind', ['domain', 'range'])
def test_frozen_pool_excludes_rebuilt_ontology_controls(tmp_path, kind, proposal):
    from exact.repair.graph import EffectivePreparation
    from exact.repair.model import RepairModel
    from exact.repair.pipeline import freeze_neural_round
    from exact.repair.retrieval import retrieve_vocabulary
    from tools.repair.robustness import audit_final_pool

    obj, fixed, narrow, omitted, allowed, role = fixture(kind)
    obj = dataclasses.replace(obj, candidates=ontology_candidates(obj, fixed_axioms=fixed))
    problem = promote_input_v3(RepairInputV2(fixed, (obj,), PolicyV2((narrow,))))
    prep = EffectivePreparation(4096, 32768, 64, 128, revision='v3')
    graph = prep.graph(problem, retrieve_vocabulary(problem))
    model = RepairModel(graph.metadata, encoder='none', hidden_dim=8, heads=2,
                        layers=0, dropout=0, revision='v3', plan_risk=False)
    frozen = freeze_neural_round(problem, model, proposal_arm=proposal, draws_per_object=8,
        max_depth=1, max_constructors=1, candidate_cap=16, contextual_filtering=False,
        omitted_generation_symbols=(omitted.iri.value,), compiler_cache_directory=str(tmp_path))
    audit = audit_final_pool(frozen.problem, {'intervention': dict(condition='fixture',
        applicable=True, final_candidate_removals={}, omitted_generation_symbols=[omitted.iri.value])})
    assert audit['final_exclusion_verified']
    assert frozen.problem.fixed_axioms == problem.fixed_axioms
    assert any(allowed in owl.signature(a) for c in frozen.problem.objects[0].candidates for a in c.axioms)
