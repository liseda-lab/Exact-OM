"""Family/parent diversity gates both primary arms before GPU or optimizer access."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.repair_semantic_fidelity_test import packet
from tools.repair.training_readiness import FAMILIES, assess, require_ready


def corpus(parents=4, families=FAMILIES):
    declarations, labels = [], {}
    for family in families:
        for parent in range(parents):
            for control in ("coherent", "corrupted"):
                key, group = f"{family}-{parent}-{control}", f"{family}-{parent}"
                p = replace(packet(), case_id=key, parent_group_id=group)
                declarations.append(
                    dict(
                        case_id=key,
                        structural_parent=group,
                        split="train",
                        family=family,
                        control=control,
                    )
                )
                labels[key] = [(p, SimpleNamespace(global_target_eligible=True, decision="A"))]
    return declarations, labels


def test_full_diversity_passes_but_many_correlated_labels_cannot_replace_parents():
    declarations, labels = corpus()
    assert assess(declarations, labels)["ready"]
    declarations, labels = corpus(parents=3)
    result = assess(declarations, labels)
    assert not result["ready"] and len(result["failures"]) == 8
    declarations, labels = corpus(parents=40, families=("overlap",))
    assert not assess(declarations, labels)["ready"]


def test_missing_control_ties_and_wrong_split_never_establish_readiness():
    declarations, labels = corpus()
    for key in list(labels):
        if "corrupted" in key:
            del labels[key]
    assert not assess(declarations, labels)["ready"]
    declarations, labels = corpus()
    for values in labels.values():
        values[0][1].decision = "tie"
    assert not assess(declarations, labels)["ready"]
    declarations[0]["split"] = "development"
    with pytest.raises(ValueError, match="TRAIN"):
        assess(declarations, labels)


def test_duplicate_plan_pair_cannot_inflate_counts():
    declarations, labels = corpus()
    key = declarations[0]["case_id"]
    labels[key] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        assess(declarations, labels)


def test_missing_gate_fails_without_any_artifact_or_training_access():
    with pytest.raises(ValueError, match="readiness receipt"):
        require_ready(None, {}, [])
