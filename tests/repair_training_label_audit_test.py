"""Admission failures must preserve scientific denominators and unknown masks."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from exact.repair.learning import ProbeOutcome, RepairLabel, TeacherCache
from tools.repair.training_audit import label_masks, validate_boundary, attempt_report
from tools.repair.historical_regression import binding
from exact.repair.api import write_artifact


def fixture():
    probes = tuple(SimpleNamespace(probe_id=str(i), family='f', desired=True) for i in range(5))
    outcomes = tuple(ProbeOutcome(str(i), 'f', True, True, True, True) for i in range(5))
    case = SimpleNamespace(probes=probes)
    settings = dict(desired_family_weight=1.0, false_positive_weight=1.0)
    return case, outcomes, settings


def cache(labels, complete=False):
    return TeacherCache((3,), tuple(labels), complete, 'fixture', (), 0, 'exact-repair/teacher-cache/v3')


def test_partial_unknowns_preserve_full_assignment_and_query_denominators():
    case, outcomes, settings = fixture()
    unknown = outcomes[:-1] + (replace(outcomes[-1], entailed=None, credit=None),)
    saved = cache([RepairLabel((0,), True, 1, 0, outcomes),
                   RepairLabel((1,), True, None, 0, unknown)])
    counts = label_masks(saved, case, settings)
    assert counts['requested'] == 3 and counts['visited'] == 2 and counts['unvisited'] == 1
    assert counts['usable'] == 1 and counts['unknown_queries'] == 1 and counts['risk'] == 2
    assert counts['exact_proposal_case'] == 0 and counts['sample_conditioned_proposal_case'] == 1
    assert counts['anchored_value_candidates'] == 0


@pytest.mark.parametrize('mutation', ['omitted', 'unknown', 'duplicate', 'wrong_scalar', 'wrong_family'])
def test_invalid_semantic_scalar_is_rejected(mutation):
    case, outcomes, settings = fixture()
    benefit = 1
    if mutation == 'omitted': outcomes = outcomes[:-1]
    if mutation == 'unknown': outcomes = outcomes[:-1] + (replace(outcomes[-1], entailed=None, credit=None),)
    if mutation == 'duplicate': outcomes += outcomes[-1:]
    if mutation == 'wrong_scalar': benefit = 0.8
    if mutation == 'wrong_family': outcomes = (replace(outcomes[0], family='different'),) + outcomes[1:]
    with pytest.raises(ValueError):
        label_masks(cache([RepairLabel((0,), True, benefit, 0, outcomes)]), case, settings)


def test_infeasible_unknown_and_unvisited_rows_never_become_semantic_labels():
    case, _, settings = fixture()
    counts = label_masks(cache([RepairLabel((0,), False, None, 0),
                               RepairLabel((1,), None, None, 0)]), case, settings)
    assert counts['risk'] == 1 and counts['infeasible'] == 1 and counts['unknown_policy'] == 1
    assert counts['usable'] == counts['exact_proposal_case'] == counts['sample_conditioned_proposal_case'] == 0
    assert counts['unvisited'] == 1


def test_complete_cache_does_not_create_exact_teacher_without_feasible_semantics():
    case, _, settings = fixture()
    counts = label_masks(cache([RepairLabel((i,), False, None, 0) for i in range(3)], True), case, settings)
    assert counts['exact_proposal_case'] == 0 and counts['risk'] == 3


@pytest.mark.parametrize('field,value', [('heldout_use', True), ('model_fitting', True),
    ('expected_cases', {'train': 128, 'development': 31}),
    ('acquisition_plans', {'train': {}, 'development': {}, 'test': {}})])
def test_split_boundary_rejected_before_any_payload_is_opened(field, value):
    plan = dict(schema='exact-repair/training-label-audit-plan/v1',
                acquisition_plans={'train': {}, 'development': {}},
                expected_cases={'train': 128, 'development': 32}, heldout_use=False, model_fitting=False)
    plan[field] = value
    with pytest.raises(ValueError, match='split boundary'):
        validate_boundary(plan)


def test_foreign_nonce_rejected_before_following_output_references(tmp_path):
    path = tmp_path/'completion.json'
    write_artifact(path, dict(status='complete', exit_code=0, step_id='14408.3', dispatch_nonce='foreign'))
    with pytest.raises(ValueError, match='ownership'):
        attempt_report(dict(completion=binding(path), step_id='14408.3', nonce='expected'))


def test_passing_junit_cannot_qualify_changed_implementation(tmp_path):
    from tools.repair.training_audit import requirement_review
    source, frozen = tmp_path/'current', tmp_path/'frozen'
    for root in (source, frozen):
        (root/'tests').mkdir(parents=True)
        (root/'tests/case.py').write_text('unchanged test')
        (root/'impl.py').write_text('old implementation')
    (source/'impl.py').write_text('changed implementation')
    matrix = tmp_path/'matrix.json'
    write_artifact(matrix, dict(requirements=[dict(requirement='fixture', implementation=['impl.py'],
                                                  tests=['tests/case.py::test_example'])]))
    xml, log = tmp_path/'junit.xml', tmp_path/'log.txt'
    xml.write_text('<testsuites><testsuite><testcase classname="tests.case" name="test_example" /></testsuite></testsuites>')
    log.write_text('passed')
    report = dict(schema='exact-repair/native-qualification/v1',
                  groups=[dict(junit=binding(xml), log=binding(log))])
    item = dict(run_id='qualified-source', report={'fixture': True})
    result = requirement_review([binding(matrix)], [(item, report, dict(code=str(frozen)))], source)
    assert result['source_compatible'] == 0
    (source/'impl.py').write_text('old implementation')
    assert requirement_review([binding(matrix)], [(item, report, dict(code=str(frozen)))], source)['source_compatible'] == 1


@pytest.mark.parametrize('outcome', ['', '<failure/>', '<error/>', '<skipped/>'])
@pytest.mark.parametrize('selector', ['tests/case.py', 'tests/case.py::test_example'])
def test_requirement_selectors_keep_all_parameterized_outcomes(tmp_path, outcome, selector):
    from tools.repair.training_audit import requirement_review
    source = tmp_path/'source'
    (source/'tests').mkdir(parents=True)
    (source/'tests/case.py').write_text('source-bound tests')
    (source/'impl.py').write_text('source-bound implementation')
    matrix = tmp_path/'matrix.json'
    write_artifact(matrix, dict(requirements=[dict(requirement='fixture', implementation=['impl.py'],
                                                  tests=[selector])]))
    xml, log = tmp_path/'junit.xml', tmp_path/'log.txt'
    xml.write_text('<testsuites><testsuite>'
                  '<testcase classname="tests.case" name="test_example[0]" />'
                  '<testcase classname="tests.case" name="test_example[1]">' + outcome + '</testcase>'
                  '<testcase classname="tests.case_extra" name="test_example"><failure/></testcase>'
                  '</testsuite></testsuites>')
    log.write_text('qualification log')
    report = dict(schema='exact-repair/native-qualification/v1',
                  groups=[dict(junit=binding(xml), log=binding(log))])
    evidence = [(dict(run_id='qualified-source', report=binding(xml)), report, dict(code=str(source)))]
    result = requirement_review([binding(matrix)], evidence, source)
    assert result['required'] == 1 and result['source_compatible'] == int(not outcome)
    observed = result['requirements'][0]['tests'][0]['evidence'][0]['observed']
    assert set(observed) == {'tests.case::test_example[0]', 'tests.case::test_example[1]'}


@pytest.mark.parametrize('mode', ['class', 'absent', 'changed_test', 'changed_impl', 'duplicate_failure'])
def test_file_requirement_needs_matching_source_and_unambiguous_passes(tmp_path, mode):
    from tools.repair.training_audit import requirement_review
    source, frozen = tmp_path/'source', tmp_path/'frozen'
    for root in (source, frozen):
        (root/'tests').mkdir(parents=True)
        (root/'tests/case.py').write_text('tests')
        (root/'impl.py').write_text('implementation')
    if mode == 'changed_test': (source/'tests/case.py').write_text('new tests')
    if mode == 'changed_impl': (source/'impl.py').write_text('new implementation')
    matrix = tmp_path/'matrix.json'
    write_artifact(matrix, dict(requirements=[dict(requirement='fixture', implementation=['impl.py'],
                                                  acceptance_tests=['tests/case.py'])]))
    classname = 'tests.case_extra' if mode == 'absent' else 'tests.case.TestExample'
    case = '<testcase classname="' + classname + '" name="test_example"'
    contents = case + ' />'
    if mode == 'duplicate_failure': contents = case + '><failure/></testcase>' + contents
    xml, log = tmp_path/'junit.xml', tmp_path/'log.txt'
    xml.write_text('<testsuites><testsuite>' + contents + '</testsuite></testsuites>')
    log.write_text('qualification log')
    report = dict(schema='exact-repair/native-qualification/v1',
                  groups=[dict(junit=binding(xml), log=binding(log))])
    evidence = [(dict(run_id='qualified-source', report=binding(xml)), report, dict(code=str(frozen)))]
    result = requirement_review([binding(matrix)], evidence, source)
    assert result['source_compatible'] == int(mode == 'class')
