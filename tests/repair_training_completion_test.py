"""Bounded corpus, protocol and decoded-training conformance; no campaign runs."""

import json
from dataclasses import replace
from pathlib import Path

import pyowl_core as owl
import pytest

from exact.repair.learning import RepairLabel, enumerate_teacher
from exact.repair.records import ObjectiveV2, candidate_cost, canonical_hash
from tools.repair.corpus import generate_corpus
from tools.repair.prepare import (
    case_from_dict,
    case_to_dict,
    generated_from_protocol,
    load_preparation,
    load_protocol,
    prepare_real_manifest,
    save_preparation,
)
from tools.repair.train import (
    DEFAULT_PROFILE,
    _verify_intended,
    decoded_development,
    train_cases,
)


def cache_for(case, *, unknown=False):
    hashes = {
        "input": case.problem.content_hash,
        "patch": canonical_hash(case.problem.objects),
        "policy": case.problem.policy.content_hash,
        "query": canonical_hash(case.probes),
        "inventory": canonical_hash(tuple(obj.candidates for obj in case.problem.objects)),
        "backend": "conformance-fixture",
        "profile": canonical_hash(DEFAULT_PROFILE),
    }

    def label(assignment):
        cost = sum(
            candidate_cost(obj, obj.candidates[choice], DEFAULT_PROFILE)
            for obj, choice in zip(case.problem.objects, assignment)
        )
        if assignment == (0,):
            return RepairLabel(assignment, None if unknown else False, None, cost)
        return RepairLabel(assignment, True, float(assignment == case.intended_assignment), cost)

    return enumerate_teacher(
        tuple(len(obj.candidates) for obj in case.problem.objects), label, hashes=hashes
    )


def test_connected_parent_variation_preserves_feasible_intentions_and_holdouts():
    cases = generate_corpus(
        parents_per_family=3,
        siblings_per_parent=1,
        families=("papers", "interaction_complementary", "mixed"),
        heldout_families=("mixed",),
    )
    for case in cases:
        assert _verify_intended(case), case.case_id
        if case.family == "mixed":
            assert case.split == "test"
        if dict(case.variation)["path_length"] > 1:
            assert any(
                "Core" in node.iri.value
                for axiom in case.problem.fixed_axioms
                for node in owl.walk(axiom)
                if isinstance(node, owl.Class)
            )
    assert len({case.problem.content_hash for case in cases}) == len(cases)


def test_protocol_names_exact_group_counts_and_ambiguity_stay_evaluator_only():
    protocol = load_protocol(Path("specs/exact-repair/protocol/smoke.json"))
    cases = generated_from_protocol(protocol)
    expected = len(protocol["data"]["families"]) * 3 * 2
    assert len(cases) == expected
    for family in protocol["data"]["families"]:
        parents = {case.structural_parent: case.split for case in cases if case.family == family}
        assert len(parents) == 3
        assert set(parents.values()) == (
            {"test"} if family == "mixed" else {"train", "development", "test"}
        )
    ambiguous = next(
        case
        for case in generate_corpus(
            parents_per_family=1,
            siblings_per_parent=1,
            families=("range",),
            ambiguity_controls=True,
        )
        if case.control == "ambiguous"
    )
    assert ambiguous.ambiguity_group and not ambiguous.intended_assignment
    assert "ambiguous" not in str(dict(ambiguous.problem.evidence))
    with pytest.raises(ValueError, match="Ambiguous"):
        from tools.repair.train import label_case

        label_case(ambiguous)


def test_teacher_case_roundtrip_keeps_hidden_store_separate_and_detects_tampering(tmp_path):
    case = generate_corpus(parents_per_family=1, siblings_per_parent=1, families=("range",))[0]
    encoded = case_to_dict(case)
    decoded = case_from_dict(json.loads(json.dumps(encoded)))
    assert decoded == case
    cache = cache_for(case)
    path = tmp_path / "preparation.json"
    save_preparation(path, (case,), {"scope": "test"}, {case.case_id: cache})
    restored, caches, report = load_preparation(path)
    assert restored == (case,) and caches[case.case_id] == cache
    assert report["scope"] == "test"
    tampered = json.loads(path.read_text())
    tampered["caches"][case.case_id]["labels"][0]["cost"] = 999
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="hash"):
        load_preparation(path)
    encoded["case"]["split"] = "test-tamper"
    with pytest.raises(ValueError, match="hash"):
        case_from_dict(encoded)


def test_decoding_reports_unknown_model_choice_and_rejects_only_proved_infeasibility():
    case = generate_corpus(parents_per_family=1, siblings_per_parent=1, families=("range",))[0]
    objective = ObjectiveV2(((100, 10, 0),))
    unknown = decoded_development(objective, cache_for(case, unknown=True))
    assert unknown["status"] == "unknown_selected_assignment"
    assert unknown["assignment"] is None and unknown["exact_regret"] is None
    decided = decoded_development(objective, cache_for(case))
    assert decided["status"] == "verified"
    assert decided["assignment"] == (1,)
    assert decided["infeasible_checks"] == 1
    # Utility difference uses explicit candidate costs once.
    costs = {label.assignment: label.cost for label in cache_for(case).labels}
    assert decided["exact_regret"] == pytest.approx(1.0 - costs[(2,)] + costs[(1,)])


def test_training_selects_periodic_decoded_checkpoint_and_masks_partial_train_cache():
    pytest.importorskip("torch_geometric")
    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 1},
        siblings_per_parent=1,
        families=("range",),
    )
    train = next(case for case in cases if case.split == "train")
    dev = next(case for case in cases if case.split == "development")
    _, report = train_cases(
        [(train, cache_for(train, unknown=True))],
        [(dev, cache_for(dev))],
        epochs=1,
        hidden_dim=8,
        heads=2,
        layers=0,
        encoder="none",
        mixtures=1,
        max_depth=1,
        max_constructors=1,
        decode_seconds=15,
        development_draws_per_object=2,
        deadline_seconds=60,
    )
    assert report["checkpoint_criterion"] == "development_decoded_teacher_regret_on_complete_subset"
    assert report["selected_epoch"] == 1
    assert any(row["status"] == "supervised" for row in report["proposal_coverage"].values())
    assert report["proposal_arm"] == "grammar_mixture"
    assert report["history"][0]["decoded_complete_coverage"] == 1
    assert report["history"][0]["generated"][dev.case_id]["status"] == "generated"
    assert 0 <= report["history"][0]["useful_candidate_coverage"] <= 1
    assert report["coverage"][train.case_id]["unknown_policy"] == 1


def test_local_real_pair_preparation_and_whole_ontology_holdout(tmp_path):
    source = tmp_path / "source.ofn"
    target = tmp_path / "target.ofn"
    source.write_text("Ontology(<urn:source> Declaration(Class(<urn:source:A>)))")
    target.write_text("Ontology(<urn:target> Declaration(Class(<urn:target:B>)))")
    alignment = [{"Src": "urn:source:A", "Tgt": "urn:target:B", "Relation": "=", "object_id": "m"}]
    (tmp_path / "alignment.json").write_text(json.dumps(alignment))
    axiom = owl.SubClassOf(owl.Class(owl.IRI("urn:source:A")), owl.Class(owl.IRI("urn:target:B")))
    entry = {
        "case_id": "local-train",
        "source": source.name,
        "target": target.name,
        "clean_alignment": "alignment.json",
        "observed_alignment": "alignment.json",
        "split": "train",
        "supervision": "user_declared",
        "probes": [
            {
                "probe_id": "intended",
                "axiom_hex": owl.canonical_bytes(axiom).hex(),
                "family": "mapping",
            }
        ],
    }
    manifest = tmp_path / "real.json"
    value = {"schema": "exact-repair/local-real-pairs/v2", "cases": [entry]}
    manifest.write_text(json.dumps(value))
    cases, report = prepare_real_manifest(manifest, call_seconds=10, deadline_seconds=30)
    assert report["produced"] == 1, report
    assert cases[0].origin == "training_side_real" and _verify_intended(cases[0])
    value["heldout_ontologies"] = [source.name]
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="holdout"):
        prepare_real_manifest(manifest)


def test_protocol_cycle_and_parent_split_leakage_are_rejected(tmp_path):
    path = tmp_path / "cycle.json"
    path.write_text(json.dumps({"extends": "cycle.json"}))
    with pytest.raises(ValueError, match="Cyclic"):
        load_protocol(path)
    case = generate_corpus(parents_per_family=1, siblings_per_parent=1, families=("range",))[0]
    save_preparation(
        path,
        (
            case,
            replace(
                case,
                case_id="other",
                split="development" if case.split != "development" else "train",
            ),
        ),
        {},
    )
    with pytest.raises(ValueError, match="parent"):
        load_preparation(path)


def test_protocol_mechanism_intentions_satisfy_their_declared_desired_probes():
    from tools.repair.train import _assignment_label

    protocol = load_protocol(Path("specs/exact-repair/protocol/smoke.json"))
    cases = generate_corpus(
        parents_per_family=1, siblings_per_parent=1, families=protocol["data"]["families"]
    )
    for case in cases:
        label = _assignment_label(case, case.intended_assignment, DEFAULT_PROFILE)
        assert label.feasible is True, case.family
        assert label.benefit is not None and label.benefit >= 1.0, case.family
        assert all(outcome.credit for outcome in label.semantic_vector), case.family


def test_evidence_quality_controls_do_not_use_intended_correctness():
    options = dict(parents_per_family=1, siblings_per_parent=1, families=("papers",), score_noise=0)
    informative = generate_corpus(**options, misleading_label_fraction=0)[0]
    misleading = generate_corpus(**options, misleading_label_fraction=1)[0]
    left = dict(informative.problem.evidence)["object-0"]
    right = dict(misleading.problem.evidence)["object-0"]
    assert left["channels"]["lexical"] > 0
    assert right["channels"]["lexical"] == 0
    assert informative.intended_assignment == misleading.intended_assignment
    assert "correctness" not in right and "family" not in right
