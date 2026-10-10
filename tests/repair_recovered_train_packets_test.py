import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from exact.repair.learning import RepairLabel, TeacherCache
from exact.repair.semantic_fidelity import validate_comparison
from tests.repair_semantic_fidelity_test import packet, judgment
from tools.repair.recovered_train_packets import clarify, select_pairs


def test_task_clarification_preserves_full_evidence_plans_and_rubric():
    old = packet()
    new = clarify(old)
    assert new.content_hash != old.content_hash
    assert replace(new, task=old.task) == old
    assert "Use tie with supported equal scores" in new.task
    for split in ("development", "test"):
        with pytest.raises(ValueError, match="TRAIN-only"):
            clarify(packet(split))


def test_fixed_calibration_fixture_keeps_ties_preferences_and_unknown_masks_strict():
    p = clarify(packet())
    for decision, a, b in (("A", .75, .25), ("B", .25, .75), ("tie", .75, .75)):
        response = judgment(p, decision=decision, a=a, b=b)
        assert validate_comparison(json.dumps(response), p).global_target_eligible
    response = judgment(p, decision="abstain")
    response["abstention_reason"] = "Insufficient evidence"
    for criterion in response["criteria"]:
        criterion.update(status="unknown", preference="abstain", a_score=None, b_score=None)
    assert not validate_comparison(json.dumps(response), p).global_target_eligible
    response = judgment(p)
    response["criteria"][0]["symbolic_claims"] = [dict(evidence_id="D1", value="true")]
    with pytest.raises(ValueError, match="unsupported symbolic"):
        validate_comparison(json.dumps(response), p)


def test_pair_preselection_preserves_existing_pair_and_required_swap_without_feedback():
    case = SimpleNamespace(split="train", case_id="case", problem=SimpleNamespace(objects=[
        SimpleNamespace(object_id="object", candidates=[SimpleNamespace(candidate_id=str(i)) for i in range(4)])]))
    cache = TeacherCache((4,), tuple(RepairLabel((i,), True, i, 0) for i in range(4)),
                         True, "complete", (), 1)
    rows = [dict(case_id="case", id="old", comparison_id="old", comparison_index=0,
                 pair=dict(assignments=[[0], [1]]), packet={"path": "old"}, swapped=False)]
    rows += [dict(case_id="case", id="new" + str(swapped), comparison_id="new", comparison_index=1,
                  pair=None, packet=None, swapped=swapped) for swapped in (False, True)]
    selected = select_pairs(case, cache, rows)
    assert len(selected) == 2 and selected[0]["pair"] == selected[1]["pair"]
    assert {tuple(a) for a in selected[0]["pair"]["assignments"]} != {(0,), (1,)}
    changed_scores = replace(cache, labels=tuple(replace(label, benefit=100-label.benefit) for label in cache.labels))
    assert select_pairs(case, changed_scores, rows) == selected
    assert rows[1]["pair"] is None
    assert select_pairs(case, None, rows) == []
