"""Bounded v2 feature screens; external data/artifacts are declared separately."""

from copy import deepcopy

FEATURE_REQUIREMENTS = {
    "E02/second_pass_trusted": ["disjoint_training_anchor_file"],
    "E02/second_pass_predicted": ["immutable_development_selected_anchor_rule"],
    "E02/anchor_noise_001": [
        "same_declared_training_anchor_population",
        "development_diagnostic_only",
    ],
    "E02/anchor_noise_005": [
        "same_declared_training_anchor_population",
        "development_diagnostic_only",
    ],
    "E11/property_labels": ["property_source_universe_and_kind_reference"],
    "E12/labels": ["individual_source_universe_and_reference"],
    "E12/multi_view": ["generated_candidate_pool_before_instance_kind_freeze"],
    "E13/owl_parity": ["matched_owl_input_pair"],
    "E13/csv_parity": ["matched_csv_input_pair", "conversion_evidence_parity_manifest"],
    "E13/csv_raw": ["independent_public_csv_input_pair"],
    "E13/csv_materialized": ["same_csv_pair_with_provenance_bearing_datalog_consequences"],
    "E14/learned_three_way": ["disjoint_typed_training_file", "all_three_relation_labels"],
    "E14/semantic_then_learned": ["disjoint_typed_training_file", "all_three_relation_labels"],
    "E14/bridge_parity": ["supported_owl_profile_and_valid_bridge_reasoner_handoff"],
}


def _arm(name, role, overlay=None, **kwargs):
    overlay = deepcopy(overlay or {})
    components = {
        name: "label_free"
        for name in (
            "retrieval",
            "fusion",
            "rerank",
            "llm",
            "accept",
            "calibration",
            "structure",
            "relation",
        )
    }
    if name in {"second_pass_trusted", "anchor_noise_001", "anchor_noise_005"}:
        kwargs.setdefault("supervision_label", "in_pair_supervised")
        components["structure"] = "supervised"
    elif kwargs.get("supervision_label") == "in_pair_supervised":
        component = (
            "relation"
            if overlay.get("matching", {}).get("relation_prediction")
            in {"learned_three_way", "semantic_then_learned"}
            else "structure"
        )
        components[component] = "supervised"
    kwargs.setdefault("supervision_label", "target_label_free")
    overlay["supervision"] = {"mode": "label_free", "components": components}
    return {"id": name, "role": role, "overlay": overlay, **kwargs}


def _channels(kind, **settings):
    kinds = ["object_property", "data_property"] if kind == "property" else ["individual"]
    return {"matching": {"entity_kinds": kinds, "channels": {kind: {"enabled": True, **settings}}}}


def feature_arms():
    """Return strict ArmConfig-shaped records without inferring case readiness."""
    arms = {
        "E02": [
            _arm(
                "single_pass_hard",
                "baseline",
                {"matching": {"anchor_rescoring": {"mode": "off", "exact_policy": "hard"}}},
            ),
            _arm(
                "single_pass_soft",
                "candidate",
                {"matching": {"anchor_rescoring": {"mode": "off", "exact_policy": "soft"}}},
            ),
            _arm(
                "second_pass_trusted",
                "candidate",
                {
                    "matching": {
                        "anchor_rescoring": {
                            "mode": "one_pass",
                            "source": "trusted",
                            "exact_policy": "soft",
                        }
                    }
                },
            ),
            _arm(
                "second_pass_predicted",
                "candidate",
                {
                    "matching": {
                        "anchor_rescoring": {
                            "mode": "one_pass",
                            "source": "predicted",
                            "exact_policy": "soft",
                        }
                    }
                },
            ),
            *[
                _arm(
                    name,
                    "diagnostic",
                    {
                        "matching": {
                            "anchor_rescoring": {
                                "mode": "one_pass",
                                "source": "trusted",
                                "exact_policy": "soft",
                                "corruption_fraction": fraction,
                                "diagnostic": True,
                            }
                        }
                    },
                    deployable=False,
                    stages=["screen"],
                )
                for name, fraction in (("anchor_noise_001", 0.01), ("anchor_noise_005", 0.05))
            ],
        ],
        "E11": [
            _arm(
                "property_labels",
                "baseline",
                _channels(
                    "property",
                    annotations=False,
                    signature=False,
                    hierarchy=False,
                    characteristics=False,
                    usage=False,
                ),
            ),
            _arm(
                "property_annotations",
                "candidate",
                _channels(
                    "property", signature=False, hierarchy=False, characteristics=False, usage=False
                ),
            ),
            _arm("property_signature", "candidate", _channels("property", usage=False)),
            _arm("property_usage", "candidate", _channels("property")),
        ],
        "E12": [
            _arm(
                "labels",
                "baseline",
                _channels("instance", types=False, literals=False, relations=False),
            ),
            _arm("types_literals", "candidate", _channels("instance", relations=False)),
            _arm("relations", "candidate", _channels("instance")),
            _arm(
                "relations_shuffled",
                "control",
                _channels("instance", relations_shuffled=True),
                required_control=True,
            ),
            _arm(
                "multi_view",
                "candidate",
                {
                    **_channels("instance"),
                    "candidates": {"multi_view": {"mode": "labels_relations"}},
                },
            ),
        ],
        "E13": [
            _arm(
                name,
                "baseline" if name in {"owl_parity", "csv_raw"} else "control",
                required_control=True,
            )
            for name in ("owl_parity", "csv_parity", "csv_raw", "csv_materialized")
        ],
        "E14": [
            _arm(
                "all_equivalent",
                "baseline",
                {"matching": {"relation_prediction": "none"}},
                required_control=True,
            ),
            _arm(
                "hierarchy_heuristic",
                "control",
                {"matching": {"relation_prediction": "hierarchy_heuristic"}},
            ),
            _arm(
                "graph_entailment",
                "candidate",
                {"matching": {"relation_prediction": "semantic_entailment"}},
            ),
            _arm(
                "learned_three_way",
                "candidate",
                {"matching": {"relation_prediction": "learned_three_way"}},
                supervision_label="in_pair_supervised",
            ),
            _arm(
                "semantic_then_learned",
                "candidate",
                {"matching": {"relation_prediction": "semantic_then_learned"}},
                supervision_label="in_pair_supervised",
            ),
            _arm(
                "bridge_parity",
                "diagnostic",
                {
                    "matching": {
                        "relation_prediction": "semantic_entailment",
                        "relation_semantic_backend": "bridge_reasoner",
                    }
                },
                deployable=False,
            ),
        ],
        "E23": [],
    }
    for fraction in (0.0, 0.5, 1.0):
        for mode in ("off", "inductive"):
            name = f"rich_{int(fraction * 100)}_{mode}"
            arms["E23"].append(
                _arm(
                    name,
                    "control" if mode == "off" else "candidate",
                    {
                        "matching": {
                            "channels": {"graph": {"mode": mode, "hierarchy_removal": fraction}}
                        }
                    },
                    supervision_label=(
                        "target_label_free" if mode == "off" else "in_pair_supervised"
                    ),
                )
            )
            FEATURE_REQUIREMENTS["E23/" + name] = ["same_tbox_rich_pair_and_frozen_pool"] + (
                [] if mode == "off" else ["disjoint_training_pool_and_permitted_negatives"]
            )
    for name, mode, shuffled, role in (
        ("natural_graph_off", "off", False, "baseline"),
        ("natural_inductive", "inductive", False, "candidate"),
        ("graph_only", "graph_only", False, "diagnostic"),
        ("graph_shuffled", "inductive", True, "control"),
    ):
        arms["E23"].append(
            _arm(
                name,
                role,
                {"matching": {"channels": {"graph": {"mode": mode, "shuffled": shuffled}}}},
                supervision_label="target_label_free" if mode == "off" else "in_pair_supervised",
            )
        )
        FEATURE_REQUIREMENTS["E23/" + name] = ["same_natural_kg_pair_and_frozen_pool"] + (
            [] if mode == "off" else ["disjoint_training_pool_and_permitted_negatives"]
        )
    return deepcopy(arms)
