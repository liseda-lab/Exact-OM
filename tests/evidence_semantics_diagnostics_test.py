"""Production annotation binding and controlled E24 replay, without encoders or APIs."""

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from exact.experiments.evidence_diagnostics import (
    difference_perturbation_inventory,
    paired_decision_changes,
    replay_difference_diagnostics,
)
from exact.impl.annotation_semantics import (
    annotate_literal,
    deduplicate_annotations,
    validate_annotation_semantics,
)
from exact.impl.datasets.pair_adaptive_context import PairAdaptiveContextDataset
from tests.pair_adaptive_experiments_test import _scorer

RULES = {
    prop: {
        "category": "definition",
        "fact_group": "definition",
        "evidence_id": "fixture:equivalent-definition-properties",
    }
    for prop in ("urn:def:one", "urn:def:two")
}
IDENTIFIER = {
    "urn:id": {
        "category": "identifier",
        "identifier_namespace": "fixture",
        "normalization": "strip_casefold",
        "value_prefix": "ID:",
        "exclusive_values": True,
        "evidence_id": "fixture:exclusive-identifiers",
    }
}


def literal(prop, value="shared", **extra):
    return {"prop_iri": prop, "prop": prop, "value": value, "text": prop + ": " + value, **extra}


def test_descriptor_controls_equivalence_and_preserves_literal_identity():
    rules = validate_annotation_semantics(RULES)
    rows = [annotate_literal(literal(prop, language="en"), rules) for prop in RULES]
    rows += [
        annotate_literal(literal("urn:def:two", language="pt"), rules),
        literal("urn:unknown", language="en"),
    ]
    reduced = deduplicate_annotations(rows)
    assert len(reduced) == 3
    shared = next(
        row for row in reduced if row.get("annotation_fact_group") and row["language"] == "en"
    )
    assert {row["prop_iri"] for row in shared["grouped_semantic_features"]} == set(RULES)
    assert shared["text"] == "definition: shared"
    assert rows == [annotate_literal(literal(prop, language="en"), rules) for prop in RULES] + [
        annotate_literal(literal("urn:def:two", language="pt"), rules),
        literal("urn:unknown", language="en"),
    ]


def test_identifier_prefix_exclusivity_and_unknowns():
    rules = validate_annotation_semantics(IDENTIFIER)
    row = annotate_literal(literal("urn:id", " ID:A1 "), rules)
    assert row["identifier_normalized"] == "a1"
    assert row["identifier_exclusive"]
    assert "identifier_namespace" not in annotate_literal(literal("urn:id", "OTHER:A1"), rules)
    scorer = _scorer(
        attr={"enabled": True, "polarity": "signed", "signed_property_allowlist": ["urn:id"]}
    )
    other = annotate_literal(literal("urn:id", "ID:B2"), rules)
    assert scorer._signed_identifier_disagreement([row], [other]) == (1.0, 1, ["fixture"])
    assert scorer._signed_identifier_disagreement([row], []) == (0.0, 0, [])
    assert scorer._signed_identifier_disagreement([row, other], [other]) == (0.0, 0, [])
    nonexclusive = {**other, "identifier_exclusive": False}
    assert scorer._signed_identifier_disagreement([row], [nonexclusive]) == (0.0, 0, [])
    with pytest.raises(ValueError, match="evidence_id"):
        validate_annotation_semantics(
            {"urn:x": {"category": "identifier", "identifier_namespace": "x"}}
        )
    with pytest.raises(ValueError, match="boolean"):
        validate_annotation_semantics({"urn:x": {"exclusive_values": "false"}})


@dataclass
class Attribute:
    property_iri: str
    value: str
    lang: str | None = "en"
    datatype: str | None = None
    is_literal: bool = True


def test_production_dataset_deduplicates_before_cap_and_changes_cache_identity(tmp_path):
    dataset = PairAdaptiveContextDataset(
        output_path=tmp_path,
        cache_ok=False,
        verbaliser_name=None,
        projection_include_literals=False,
        max_attr_items=2,
        annotation_semantics=RULES,
        annotation_provenance_dedup=True,
    )
    dataset._source = SimpleNamespace(
        attributes=lambda iri: [
            Attribute("urn:def:one", "shared"),
            Attribute("urn:def:two", "shared"),
            Attribute("urn:def:two", "unique"),
        ]
    )
    graph = SimpleNamespace(get_labels=lambda iri: [iri])
    values = dataset._annotation_bundle("urn:entity", graph, "src")
    assert {row["value"] for row in values} == {"shared", "unique"}
    shared = next(row for row in values if row["value"] == "shared")
    assert len(shared["grouped_semantic_features"]) == 2
    # The schema/config is part of the prepared evidence cache, independently of vector caches.
    assert dataset.annotation_semantics == RULES
    assert dataset.annotation_provenance_dedup


def fact(obj, score=0.2):
    return {
        "triple": ["entity", "role", obj],
        "subject_iri": "urn:entity",
        "rel_iri": "urn:role",
        "object_iri": obj,
        "score": score,
    }


def difference(formulation, **extra):
    scorer = _scorer(diff={"enabled": True, "formulation": formulation, **extra})
    scorer.use_context = True
    scorer._object_support_matrix = lambda left, right: torch.zeros((len(left), len(right)))
    return scorer


def test_one_sided_empty_distinguishes_authority_from_scale():
    src = [fact("urn:a")]
    normalized = difference("normalised")._score_difference_channel(src, [])
    absolute = difference("absolute")._score_difference_channel(src, [])
    aware = difference("missingness_aware")._score_difference_channel(src, [])
    assert normalized["score"] == pytest.approx(0.5)
    assert absolute["score"] == pytest.approx(0.9)
    assert aware["quality"] == 0.0  # Zero signed authority, not a confident negative.
    assert aware["evidence_states"][0]["state"] == "unobserved"


def test_pinned_perturbations_replay_actual_channel_and_do_not_mutate_inputs():
    source, target = [fact("urn:a")], [fact("urn:b")]
    before = deepcopy((source, target))
    inventory = difference_perturbation_inventory(source, target, pair_id="D1:s:t")
    assert inventory == difference_perturbation_inventory(source, target, pair_id="D1:s:t")
    assert len(inventory["variants"]) == 10
    assert (source, target) == before
    scorer = difference("missingness_aware")
    report = replay_difference_diagnostics(scorer, inventory)
    assert report["decision_scope"] == "channel_only"
    assert all(row["quality"] == 0.0 for row in report["variants"])
    inventory["variants"][0]["source"][0]["score"] = 0.9
    with pytest.raises(ValueError, match="hash mismatch"):
        replay_difference_diagnostics(scorer, inventory)


def test_asymmetric_reversal_swaps_interpretation_and_restores_it():
    scorer = difference("asymmetric", relation_interpretation="<")
    report = replay_difference_diagnostics(
        scorer, difference_perturbation_inventory([fact("urn:a")], [], pair_id="typed")
    )
    assert report["variants"][0]["score"] == report["variants"][-1]["score"]
    assert scorer.diff_config["relation_interpretation"] == "<"
    with pytest.raises(ValueError, match="typed interpretation"):
        replay_difference_diagnostics(
            difference("asymmetric"), difference_perturbation_inventory([], [], pair_id="bad")
        )


def test_disjoint_fillers_are_not_exclusive_for_multivalued_properties():
    rule = {
        "property_iri": "urn:role",
        "source_object": "urn:a",
        "target_object": "urn:b",
        "semantic_rule": "disjoint_objects",
        "evidence_id": "fixture:disjoint",
    }
    scorer = difference("missingness_aware", incompatibilities=[rule])
    with pytest.raises(ValueError, match="single-valued"):
        scorer._score_difference_channel([fact("urn:a")], [fact("urn:b")])
    rule["single_valued"] = True
    scorer.diff_config["incompatibilities"] = [rule]
    assert scorer._score_difference_channel([fact("urn:a")], [fact("urn:b")])["score"] == 0.0


def test_correction_harm_requires_known_labels_and_same_denominator():
    result = paired_decision_changes(
        {"a": False, "b": True, "u": True},
        {"a": True, "b": False, "u": False},
        {"a": True, "b": True},
    )
    assert result == {
        "population": 3,
        "known_label_count": 2,
        "unknown_label_count": 1,
        "corrected": 1,
        "harmed": 1,
        "correction_minus_harm": 0,
        "changed_unknown": 1,
    }
    with pytest.raises(ValueError, match="identical populations"):
        paired_decision_changes({"a": True}, {}, {})


def test_frozen_matrix_diagnostics_never_encode_or_verbalize(monkeypatch):
    scorer = difference("normalised")

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostic replay must reuse frozen supports without hosted calls")

    monkeypatch.setattr(scorer, "_object_support_matrix", forbidden)
    monkeypatch.setattr(scorer, "_verbalize_object_items", forbidden)
    inventory = difference_perturbation_inventory(
        [fact("urn:a")], [fact("urn:b")], pair_id="D1:s:t"
    )
    result = replay_difference_diagnostics(scorer, inventory, support_matrix=torch.tensor([[0.75]]))
    assert result["support_semantics"] == "frozen_matrix_with_zero_support_irrelevant_insertion"
    assert len(result["variants"]) == 10
    serialized = next(
        row for row in result["variants"] if row["name"] == "equivalent_serialization_source"
    )
    assert serialized["score_delta"] == 0.0
    assert result["original"]["source"] == inventory["variants"][0]["source"]


def test_evidence_recipe_bindings_dispose_unsupported_semantics_and_retain_controls(tmp_path):
    from pathlib import Path

    from exact.experiments.campaign import load_campaign
    from exact.experiments.preparation import prepare_campaign
    from tests.campaign_preparation_test import _cases

    root = Path(__file__).resolve().parents[1]
    path = prepare_campaign(
        root / "specs/experiments/campaign-v2.yaml",
        root / "exact/default_config.yaml",
        {"cases": _cases()},
        tmp_path / "prepared",
    )
    campaign, _ = load_campaign(path)
    steps = {step.id: step for step in campaign.steps}
    assert len(steps["E08"].arms) == 3
    assert all(
        row.status == "inapplicable"
        for roles in steps["E08-identifiers"].readiness.values()
        for row in roles.values()
    )
    core_control = next(arm for arm in steps["E08"].arms if arm.id == "provenance_dedup")
    assert core_control.overlay == steps["E08-identifiers"].arms[0].overlay
    assert steps["E24"].additional_cases == ["D0"]
    assert len(steps["E24"].arms) == 4
    assert steps["E24-asymmetric"].selection.decisions == []
    assert steps["E24-asymmetric"].readiness["asymmetric"]["screen"].status == "inapplicable"
    bound = prepare_campaign(
        root / "specs/experiments/campaign-v2.yaml",
        root / "exact/default_config.yaml",
        {
            "cases": _cases(),
            "annotation_semantics": IDENTIFIER,
            "e24_typed_diagnostic": {"case": "T0", "relation": "<"},
        },
        tmp_path / "bound",
    )
    campaign, _ = load_campaign(bound)
    steps = {step.id: step for step in campaign.steps}
    assert steps["E08-identifiers"].arms[1].overlay["matching"]["channels"]["attr"][
        "signed_property_allowlist"
    ] == ["urn:id"]
    assert steps["E24-asymmetric"].case == "T0"
    assert (
        steps["E24-asymmetric"]
        .arms[0]
        .overlay["matching"]["channels"]["diff"]["relation_interpretation"]
        == "<"
    )


def test_production_pair_diagnostics_are_persistable_and_leave_scores_unchanged(monkeypatch):
    import json

    from tests.pair_adaptive_experiments_test import _TinyDataset

    outputs = []
    for diagnostic in (False, True):
        scorer = _scorer(
            return_explanations=True,
            diff={
                "enabled": True,
                "formulation": "missingness_aware",
                "dump_components": True,
                "controlled_perturbations": diagnostic,
            },
        )
        dataset = _TinyDataset()
        original_features = dataset.get_entity_features
        dataset.get_entity_features = lambda iri, side: {
            **original_features(iri, side),
            "object_triples": [fact("urn:a" if side == "src" else "urn:b")],
        }
        scorer.attach_dataset(dataset)
        scorer.use_context = True
        monkeypatch.setattr(
            scorer,
            "_object_support_matrix",
            lambda left, right: torch.full((len(left), len(right)), 0.75),
        )
        monkeypatch.setattr(scorer, "_context_similarity_from_sentences", lambda *args: 0.8)
        result = scorer(
            src_iris=["s"],
            tgt_iris=["t"],
            src_label_lists=[["source"]],
            tgt_label_lists=[["target"]],
        )
        outputs.append(result)
    torch.testing.assert_close(outputs[0]["S_final"], outputs[1]["S_final"], rtol=0, atol=0)
    # The ordinary explanation artifact is already committed by runtime_checkpoint.
    encoded = json.dumps(outputs[1]["explanations"])
    assert "controlled_perturbations" in encoded
    assert "inventory_sha256" in encoded
    assert "frozen_matrix_with_zero_support_irrelevant_insertion" in encoded
