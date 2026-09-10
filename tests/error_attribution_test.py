import pytest

from exact.experiments.error_attribution import attribute_source_errors, _cardinality_ceiling


def candidate(target, score=0.9, **kwargs):
    return {
        "target": target,
        "S_final": score,
        "S_base": score,
        "threshold_positive": score >= 0.7,
        "emitted": False,
        **kwargs,
    }


def fixture():
    rows = [
        {"Src": "retrieval", "candidates": []},
        {
            "Src": "ranking",
            "candidates": [candidate("gold", 0.2), candidate("wrong", emitted=True)],
            "pre_typing_targets": ["wrong"],
            "emitted_targets": ["wrong"],
        },
        {"Src": "acceptance", "candidates": [candidate("gold", 0.1, P_rank=1.0)]},
        {
            "Src": "anchor",
            "candidates": [candidate("wrong", protected_exact=True, emitted=True)],
            "pre_typing_targets": ["wrong"],
            "emitted_targets": ["wrong"],
        },
        {
            "Src": "collision",
            "candidates": [candidate("shared", reason="cardinality_or_extraction")],
        },
        {
            "Src": "typing",
            "candidates": [candidate("gold", reason="relation_abstention")],
            "pre_typing_targets": ["gold"],
        },
        {
            "Src": "correct",
            "candidates": [candidate("shared", emitted=True, relation="=")],
            "pre_typing_targets": ["shared"],
            "emitted_targets": ["shared"],
        },
        {
            "Src": "nil",
            "candidates": [candidate("wrong", emitted=True)],
            "pre_typing_targets": ["wrong"],
            "emitted_targets": ["wrong"],
        },
    ]
    trace = {
        "schema_version": 2,
        "stage": "after_cardinality_and_relation_typing",
        "source_universe_status": "declared",
        "source_universe": [row["Src"] for row in rows],
        "policy": {"source_cardinality": 1, "target_cardinality": 1},
        "records": rows,
    }
    reference = [
        {
            "SrcEntity": row["Src"],
            "TgtEntity": "shared" if row["Src"] in {"collision", "correct"} else "gold",
        }
        for row in rows
        if row["Src"] != "nil"
    ]
    return trace, reference


def test_six_stage_attributions_and_distinct_reachable_oracle_populations():
    trace, reference = fixture()
    result = attribute_source_errors(
        trace, reference, reference_role="development", negative_label_policy="complete_reference"
    )
    rows = {row["Src"]: row for row in result["records"]}
    for source, stage in {
        "retrieval": "candidate_loss",
        "ranking": "in_pool_ranking",
        "acceptance": "acceptance_nil",
        "anchor": "exact_anchors",
        "collision": "collisions",
        "typing": "typing",
    }.items():
        assert rows[source]["flags"][stage]
    assert rows["nil"]["known_nil"]
    assert rows["nil"]["flags"]["acceptance_nil"]
    assert result["observed"]["tp"] == 1
    assert result["observed"]["fp_confirmed"] == 3
    assert result["observed"]["fn_known_positive"] == 6
    assert rows["correct"]["metrics"]["tp"] == 1
    assert rows["correct"]["metrics"]["f1"] == 1
    assert rows["ranking"]["metrics"]["tp"] == 0
    ceilings = result["oracle_ceilings"]
    assert {name: row["metrics"]["tp"] for name, row in ceilings.items()} == {
        "retrieval": 7,
        "ranking": 5,
        "acceptance_nil": 4,
        "exact_anchors": 2,
        "collisions": 2,
        "typing": 2,
    }
    assert ["typing", "gold", "="] in ceilings["typing"]["correct_typed_pairs"]
    assert result["diagnostic_only"] and not result["deployable"]


def test_unknown_pairs_and_absent_reference_sources_are_never_false_negatives_or_nil_labels():
    trace, reference = fixture()
    result = attribute_source_errors(
        trace, reference, reference_role="validation", negative_label_policy="positive_unlabelled"
    )
    assert result["observed"]["fp_confirmed"] == 0
    assert result["observed"]["unassessed_predictions"] == 3
    assert result["observed"]["precision"] is None
    assert result["observed"]["f1"] is None
    rows = {row["Src"]: row for row in result["records"]}
    assert not rows["nil"]["known_nil"] and rows["nil"]["reference_status"] == "unassessed"
    assert not rows["anchor"]["flags"]["exact_anchors"]
    assert ["anchor", "wrong"] in result["oracle_ceilings"]["exact_anchors"]["reachable_pairs"]
    confirmed = attribute_source_errors(
        trace,
        reference,
        reference_role="diagnostic",
        negative_label_policy="confirmed_negatives",
        confirmed_negatives=[("anchor", "wrong")],
        nil_sources=["nil"],
    )
    assert confirmed["observed"]["fp_confirmed"] == 2
    assert confirmed["observed"]["unassessed_predictions"] == 1


def test_final_gold_and_incomplete_source_trace_fail_closed():
    trace, reference = fixture()
    for role in ("test", "reporting", "final"):
        with pytest.raises(ValueError, match="development/diagnostic"):
            attribute_source_errors(
                trace, reference, reference_role=role, negative_label_policy="unknown"
            )
    trace["records"].pop()
    with pytest.raises(ValueError, match="every eligible source"):
        attribute_source_errors(
            trace, reference, reference_role="development", negative_label_policy="unknown"
        )


def test_collision_oracle_is_exact_for_assignment_trap_and_general_capacities():
    pairs = {("a", "x"), ("a", "y"), ("b", "x")}
    assert len(_cardinality_ceiling(pairs, 1, 1)) == 2
    assert _cardinality_ceiling(pairs, 2, 2) == pairs
    assert _cardinality_ceiling(pairs, None, None) == pairs


def test_typing_oracle_distinguishes_pair_correctness_from_canonical_relation():
    trace, reference = fixture()
    correct = next(row for row in trace["records"] if row["Src"] == "correct")
    correct["candidates"][0]["relation"] = "<"
    report = attribute_source_errors(
        trace, reference, reference_role="diagnostic", negative_label_policy="complete_reference"
    )
    assert report["observed"]["tp"] == 1
    assert report["observed_typed"]["tp"] == 0
    assert report["oracle_ceilings"]["typing"]["typed_metrics"]["tp"] == 2
    assert next(row for row in report["records"] if row["Src"] == "correct")["flags"]["typing"]
