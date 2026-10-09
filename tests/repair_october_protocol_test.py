"""Corrective launch contracts preserve old identities and bind new semantics."""

import copy
import json
from pathlib import Path

import pytest

from exact.repair.protocol import RepairProtocolV3

SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "specs/exact-repair/protocol/xr21-review2-conformance.json"
)


def original():
    return json.loads(SAMPLE.read_text())


def combined():
    value = original()
    value["identity"].update(
        implementation_revision="exact-repair/preliminary-corrections/v1",
        label_schema="exact-repair/symbolic-plus-fidelity/v1",
    )
    value["losses"].update(
        target_basis="symbolic_plus_llm",
        loss_contract="provenance_additive/v1",
        weak_anchor_weight=0.2,
        weak_comparison_weight=0.2,
    )
    value["llm_labels"].update(
        enabled=True,
        execution_authorized=True,
        teacher_profile="teacher",
        evaluator_profile="independent-assessor",
        annotation_manifest="frozen/labels.json",
        development_use_policy="development_selection/v1",
        plan_rating_aggregation=dict(
            revision="per_criterion_median/v1", minimum_ratings=1, max_criterion_range=0.25
        ),
    )
    value["training"].update(
        development_epochs=[5, 10],
        max_epochs=10,
        patience=10,
        patience_enabled=False,
        max_full_development_evaluations=2,
        development_case_ids=["dev-clean", "dev-corrupt"],
    )
    return value


def test_legacy_protocol_does_not_acquire_new_semantics_or_hash_fields():
    source = original()
    result = RepairProtocolV3.model_validate(source).model_dump(by_alias=True)
    for section, names in {
        "generation": ["execution_schedule", "elementary_seconds"],
        "losses": ["loss_contract", "weak_anchor_weight", "weak_comparison_weight"],
        "llm_labels": ["development_use_policy", "plan_rating_aggregation"],
        "training": [
            "development_epochs",
            "development_case_ids",
            "patience_enabled",
            "max_full_development_evaluations",
        ],
    }.items():
        assert not set(names) & set(result[section])


def test_corrective_supervision_and_development_contract_round_trip():
    source = combined()
    result = RepairProtocolV3.model_validate(source)
    assert result.losses.target_basis == "symbolic_plus_llm"
    assert result.llm_labels.development_use_policy == "development_selection/v1"
    assert result.training.development_case_ids == ["dev-clean", "dev-corrupt"]
    assert (
        RepairProtocolV3.model_validate(result.model_dump(by_alias=True)).resolved_hash
        == result.resolved_hash
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing_contract",
        "zero_weak",
        "missing_aggregation",
        "duplicate_cases",
        "too_many_passes",
        "late_pass",
    ],
)
def test_corrective_manifest_rejects_ambiguous_or_unbounded_semantics(change):
    source = copy.deepcopy(combined())
    if change == "missing_contract":
        source["losses"].pop("loss_contract")
    elif change == "zero_weak":
        source["losses"].update(weak_anchor_weight=0.0, weak_comparison_weight=0.0)
    elif change == "missing_aggregation":
        source["llm_labels"].pop("plan_rating_aggregation")
    elif change == "duplicate_cases":
        source["training"]["development_case_ids"] = ["same", "same"]
    elif change == "too_many_passes":
        source["training"]["development_epochs"] = [1, 5, 10]
    else:
        source["training"]["development_epochs"] = [5, 11]
    with pytest.raises(ValueError):
        RepairProtocolV3.model_validate(source)
