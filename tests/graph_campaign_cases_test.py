"""E23 ablations retain case/kind and never invent graph training negatives."""

from copy import deepcopy
from pathlib import Path

import pytest

from exact.experiments.campaign import load_campaign
from exact.experiments.preparation import prepare_campaign
from tests.campaign_preparation_test import _cases


def prepare(tmp_path, cases, **extra):
    root = Path(__file__).resolve().parents[1]
    path = prepare_campaign(
        root / "specs/experiments/campaign-v2.yaml",
        root / "exact/default_config.yaml",
        {"cases": cases, **extra},
        tmp_path / "prepared",
    )
    lock, _ = load_campaign(path)
    return {step.id: step for step in lock.steps}


def test_graph_ablations_are_matched_class_pairs_and_kg_stays_separate(tmp_path):
    cases = _cases()
    cases["D1"]["negative_policy"] = "confirmed_only"
    cases["K0"].update(kind="individual", negative_policy="positive_unlabelled")
    before = deepcopy(cases)
    steps = prepare(tmp_path, cases)
    assert cases == before
    for fraction in (0, 50, 100):
        step = steps[f"E23-rich-{fraction}"]
        assert step.case == "D1" and step.additional_cases == []
        assert len(step.arms) == 2
        graph = [arm.overlay["matching"]["channels"]["graph"] for arm in step.arms]
        assert graph[0]["hierarchy_removal"] == graph[1]["hierarchy_removal"] == fraction / 100
        assert {row["mode"] for row in graph} == {"off", "inductive"}
        assert step.selection.decisions[0].baseline == f"rich_{fraction}_off"
        assert step.selection.decisions[0].candidates == [f"rich_{fraction}_inductive"]
        assert all(
            record.status != "inapplicable"
            for roles in step.readiness.values()
            for record in roles.values()
        )
    natural = steps["E23"]
    assert natural.case == "K0"
    assert len(natural.arms) == 4
    for arm in natural.arms:
        if arm.id == "natural_graph_off":
            assert natural.readiness[arm.id]["screen"].status == "implementing"
        else:
            record = natural.readiness[arm.id]["screen"]
            assert record.status == "inapplicable"
            assert "unlisted pairs cannot supply" in record.reason
    assert all(
        arm.supervision_label == "in_pair_supervised"
        for arm in natural.arms
        if arm.id != "natural_graph_off"
    )


@pytest.mark.parametrize(
    "case,kind,role",
    [
        ("absent", "class", "development"),
        ("D1", "individual", "development"),
        ("D1", "class", "reporting"),
    ],
)
def test_graph_ablations_reject_missing_wrong_kind_or_final_case(tmp_path, case, kind, role):
    cases = _cases()
    cases["D1"].update(kind=kind, role=role)
    with pytest.raises(ValueError, match="E23 rich-case ablation requires"):
        prepare(tmp_path, cases, graph_rich_case=case)


def test_optional_native_bridge_is_separate_and_keeps_same_case_control(tmp_path):
    from exact.utils.provenance import sha256_file

    cases = _cases()
    unbound = prepare(tmp_path / "unbound", cases)
    assert len(unbound["E14"].arms) == 5
    assert unbound["E14-bridge"].readiness["bridge_parity"]["screen"].status == "inapplicable"
    fixtures = Path(__file__).parent / "fixtures/ontologies"
    for side, name in (("source", "mini_src.owl"), ("target", "mini_tgt.owl")):
        path = fixtures / name
        cases["D0"][side] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    bound = prepare(tmp_path / "bound", cases, e14_bridge_case="D0")
    bridge = bound["E14-bridge"]
    assert bridge.case == "D0" and bridge.additional_cases == []
    assert [arm.id for arm in bridge.arms] == ["graph_entailment", "bridge_parity"]
    assert bridge.arms[0].role == "baseline" and bridge.arms[0].required_control
    assert bridge.selection.decisions == []
