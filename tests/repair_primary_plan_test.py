import pytest

from tools.repair.primary_plan import phase_plan


def test_pair_order_uses_completed_fit_and_terminal_dev_dependencies():
    first = phase_plan(13, "symbolic")
    second = phase_plan(13, "symbolic_plus_llm")
    assert second[0]["depends_on"] == [first[2]["id"]]
    for plan in (first, second):
        assert all(not p["execution_authorized"] and not p["descriptor_prepared"] for p in plan)
        assert [p["phase"] for p in plan] == ["fit", "development", "fit", "development"]
        assert all(plan[i]["depends_on"] == [plan[i - 1]["id"]] for i in range(1, 4))
    third = phase_plan(73, "symbolic")
    assert set(third[0]["depends_on"]) == {
        phase_plan(37, condition)[-1]["id"] for condition in ("symbolic", "symbolic_plus_llm")
    }
    for seed in (37, 73):
        assert all(
            p["release_gate"] == "first_pair_technical_review" for p in phase_plan(seed, "symbolic")
        )


def test_plan_rejects_unapproved_seed_or_condition():
    with pytest.raises(ValueError):
        phase_plan(99, "symbolic")
    with pytest.raises(ValueError):
        phase_plan(13, "new_condition")
