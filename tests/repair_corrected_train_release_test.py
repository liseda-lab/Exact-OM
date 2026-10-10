from dataclasses import replace

import pytest

from exact.repair.learning import RepairLabel, TeacherCache
from tools.repair.corrected_train_release import substitute_unknowns


def caches():
    old = TeacherCache((3,), (RepairLabel((0,), True, 1, 0),
                             RepairLabel((1,), None, None, 0),
                             RepairLabel((2,), None, None, 0)), False, "old",
                       (("input", "same"), ("policy", "fixed")), 10)
    correction = TeacherCache((3,), (RepairLabel((1,), True, 2, 0),), False,
                              "single_assignment_transport", (("input", "same"),), 2)
    return old, correction


def test_only_unknown_is_substituted_with_history_and_full_dependencies_retained():
    old, correction = caches()
    result = substitute_unknowns(old, [correction])
    assert result.labels == (old.labels[0], correction.labels[0], old.labels[2])
    assert old.labels[1].feasible is None
    assert result.hashes == old.hashes and result.candidate_counts == old.candidate_counts
    assert result.elapsed_seconds == 12 and not result.complete


@pytest.mark.parametrize("change", [
    dict(labels=(RepairLabel((0,), True, 2, 0),)),
    dict(hashes=(("input", "different"),)),
    dict(hashes=(("input", "same"), ("policy", "other"))),
    dict(candidate_counts=(4,)),
    dict(schema="exact-repair/teacher-cache/v3"),
])
def test_refuses_known_label_replay_or_changed_scientific_transport(change):
    old, correction = caches()
    with pytest.raises(ValueError):
        substitute_unknowns(old, [replace(correction, **change)])


def test_repeated_assignment_is_not_counted_twice_and_unknown_stays_unknown():
    old, correction = caches()
    with pytest.raises(ValueError):
        substitute_unknowns(old, [correction, correction])
    unknown = replace(correction, labels=(RepairLabel((1,), None, None, 0),))
    assert substitute_unknowns(old, [unknown]).labels == old.labels


def test_infeasibility_is_a_decided_risk_label_not_a_usable_semantic_label():
    old, correction = caches()
    result = substitute_unknowns(old, [replace(correction, labels=(RepairLabel((1,), False, None, 0),))])
    assert result.labels[1].feasible is False
    assert not result.labels[1].usable
