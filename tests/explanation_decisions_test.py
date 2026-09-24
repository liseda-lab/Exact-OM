"""Decision observability contracts without models, remote calls, or experiments."""

import json
from types import SimpleNamespace

import pandas as pd
import pyowl_core as owl
import pytest

from exact.core.entities.mappings import EntityMapping
from exact.impl.models.pair_adaptive_evidence import PairAdaptiveEvidenceMixin
from exact.impl.trainer.overlays import OverlaysMixin
from exact.io.sources.feature_provenance import RDFS, FeatureProvenance
from exact.runs import ExplanationStore
from exact.runs.decisions import (
    append_event,
    event,
    observe_extraction,
    observe_selection,
    write_candidate_decisions,
)
from exact_inspect.contracts import DomainError, canonical_hash, encode_cursor
from exact_inspect.decisions import DecisionStore, import_run, selected_evidence


def _record(target="urn:t", kind="class"):
    return {
        "src_iri": "urn:s",
        "tgt_iri": target,
        "src_kind": kind,
        "tgt_kind": kind,
        "confidences": {"S_final": 0.9},
        "prediction": {"ground_truth": True, "saved_alignment_member": True},
    }


def _run(tmp_path):
    run = tmp_path / "run"
    store = ExplanationStore.create(run)
    store.append([_record(), _record("urn:other")])
    from exact.runs import RunLayout

    layout = RunLayout.open(run)
    pd.DataFrame(
        [
            {
                "SrcEntity": "urn:s",
                "TgtEntity": "urn:t",
                "Score": 0.9,
                "Relation": "=",
                "SrcKind": "class",
                "TgtKind": "class",
            }
        ]
    ).to_csv(layout.mapping_path("global"), sep="\t", index=False)
    return run, store


def test_import_joins_persisted_mapping_and_never_invents_legacy_events(tmp_path):
    run, _ = _run(tmp_path)
    manifest = import_run(
        run,
        tmp_path / "package",
        source_ontology_version_id="source-v1",
        target_ontology_version_id="target-v1",
    )
    store = DecisionStore(tmp_path / "package")
    rows = store.candidates("urn:s")["items"]
    rejected = next(row for row in rows if row["target"]["iri"] == "urn:other")
    assert rejected["saved_alignment_member"] is False
    assert rejected["membership_provenance"]["status"] == "derived_from_saved_artifacts"
    trace = store.pair(rejected["pair_id"])
    assert all(item["status"] == "not_recorded" for item in trace["events"])
    assert trace["relation_basis"] == "unavailable"
    assert "ground_truth" not in json.dumps([rows, trace, manifest])


def test_pages_bind_to_source_policy_and_run_revision(tmp_path):
    run, _ = _run(tmp_path)
    import_run(
        run,
        tmp_path / "package",
        source_ontology_version_id="source-v1",
        target_ontology_version_id="target-v1",
    )
    store = DecisionStore(tmp_path / "package")
    first = store.candidates("urn:s", limit=1, policy_hash=canonical_hash("a"))
    second = store.candidates(
        "urn:s", limit=1, cursor=first["next_cursor"], policy_hash=canonical_hash("a")
    )
    assert first["total_count"] == 2
    assert first["items"][0]["pair_id"] != second["items"][0]["pair_id"]
    with pytest.raises(DomainError, match="another query"):
        store.candidates("urn:s", cursor=first["next_cursor"], policy_hash=canonical_hash("b"))
    with pytest.raises(DomainError):
        store.candidates("urn:s", limit=101)
    for malformed_key in ({"offset": 1}, [1], 1, True):
        cursor = encode_cursor(first["scope"], malformed_key)
        with pytest.raises(DomainError) as error:
            store.candidates("urn:s", cursor=cursor, policy_hash=canonical_hash("a"))
        assert error.value.envelope.code == "invalid_cursor"


def test_nil_and_events_survive_overlay_compaction_and_typed_punning(tmp_path):
    store = ExplanationStore.create(tmp_path / "run")
    store.append([_record(), _record(kind="individual")])
    row = pd.Series(
        {
            "Src": "urn:s",
            "Tgt": "urn:t",
            "SrcKind": "class",
            "TgtKind": "class",
            "S_final": 0.9,
            "P_nil": 0.2,
            "Q_nil": 0.1,
            "Q_match": 0.9,
            "nil_rank": 2,
            "candidate_joint_rank": 1,
            "selection_nil_winner": False,
            "nil_absence_semantics": "no_answer_in_candidate_pool",
        }
    )
    row["candidate_decision"] = append_event(
        row, event("pair_scoring", reason="observed", implementation="test")
    )
    mixin = OverlaysMixin()
    store.append_overlay([mixin._overlay_record_from_candidate_row(row)])
    store.compact()
    records = ExplanationStore(store.directory).get("urn:s")
    class_record = next(item for item in records if item["src_kind"] == "class")
    assert class_record["candidate_decision"]["values"]["nil_rank"] == 2
    assert class_record["candidate_decision"]["values"]["P_nil"] == 0.2
    assert "candidate_decision" not in next(
        item for item in records if item["src_kind"] == "individual"
    )


def test_actual_selector_and_extraction_boundaries_preserve_scores_and_competitors():
    frame = pd.DataFrame(
        [
            {"Src": "urn:s", "Tgt": "urn:t", "S_final": 0.9, "selection_winner": True},
            {"Src": "urn:s", "Tgt": "urn:loss", "S_final": 0.8, "selection_winner": False},
            {"Src": "urn:s2", "Tgt": "urn:t", "S_final": 0.7, "selection_winner": True},
            {"Src": "urn:s3", "Tgt": "urn:low", "S_final": 0.2, "selection_winner": True},
        ]
    )
    before = frame.copy()
    observe_selection(frame, implementation="selector", config={"threshold": 0.5})
    observe_extraction(
        frame,
        [EntityMapping("urn:s", "urn:t", score=0.9)],
        threshold=0.5,
        config={"source_cardinality": 1, "target_cardinality": 1},
        implementation="greedy",
    )
    pd.testing.assert_frame_equal(frame[before.columns], before)
    winner = frame.iloc[0].candidate_decision
    loss = frame.iloc[1].candidate_decision
    conflict = frame.iloc[2].candidate_decision
    assert loss["events"][0]["outcome"] == "not_selected"
    assert winner["pair_key"] in loss["events"][0]["competitor_pair_keys"]
    assert winner["pair_key"] in conflict["events"][-1]["competitor_pair_keys"]
    assert (
        next(e for e in frame.iloc[3].candidate_decision["events"] if e["stage"] == "threshold")[
            "reason_code"
        ]
        == "below_threshold"
    )


def test_current_artifact_exports_nil_convention_and_disabled_repair(tmp_path):
    run, _ = _run(tmp_path)
    frame = pd.DataFrame(
        [
            {
                "Src": "urn:s",
                "Tgt": "urn:t",
                "S_final": 0.9,
                "P_nil": 0.2,
                "nil_rank": 2,
                "candidate_joint_rank": 1,
                "selection_winner": True,
            }
        ]
    )
    frame["candidate_decision"] = [
        append_event(
            frame.iloc[0],
            event(
                "retrieval",
                implementation="dataset",
                reason="provided_candidate_pool",
                status="not_run",
                values={
                    "cand_rank": 3,
                    "cand_ordering": "saved_candidate_pool_order",
                    "cand_tie_rule": "original_saved_order",
                    "cand_rank_provenance": "derived_from_saved_artifacts",
                },
            ),
        )
    ]
    frame["nil_ranking_scale"] = "joint_accept_probability"
    observe_selection(
        frame, implementation="selector", config={"nil": {"tie_rule": "candidate_iri_before_nil"}}
    )
    mappings = pd.DataFrame(
        [{"SrcEntity": "urn:s", "TgtEntity": "urn:t", "Relation": "=", "relation_confidence": 1.0}]
    )
    write_candidate_decisions(run, frame, mappings, policy={"relation_prediction": "none"})
    import_run(
        run,
        tmp_path / "package",
        source_ontology_version_id="source-v1",
        target_ontology_version_id="target-v1",
    )
    store = DecisionStore(tmp_path / "package")
    candidate = next(
        row for row in store.candidates("urn:s")["items"] if row["target"]["iri"] == "urn:t"
    )
    trace = store.pair(candidate["pair_id"])
    assert trace["relation_basis"] == "default_convention"
    assert candidate["nil"]["P_nil"] == 0.2
    assert candidate["ordinal_ranks"] == {
        "candidate_joint_rank": 1,
        "nil_rank": 2,
        "joint_ordering": "joint_accept_probability",
        "joint_tie_rule": "candidate_iri_before_nil",
        "retrieval_rank": 3,
        "retrieval_ordering": "saved_candidate_pool_order",
        "retrieval_tie_rule": "original_saved_order",
        "retrieval_provenance": "derived_from_saved_artifacts",
    }
    assert not {"nil_rank", "candidate_joint_rank", "cand_rank"}.intersection(
        score["name"] for score in candidate["scores"]
    )
    assert next(e for e in trace["events"] if e["stage"] == "repair")["status"] == "not_run"
    assert not any(
        score["name"] == "relation_confidence" for e in trace["events"] for score in e["scores"]
    )


def test_semantic_evidence_identity_ignores_display_labels_and_distinguishes_iris():
    scorer = PairAdaptiveEvidenceMixin()
    first = {
        "subject_iri": "urn:s",
        "object_iri": "urn:o",
        "rel_iri": RDFS + "subClassOf",
        "triple": ["label", "parent", "label"],
    }
    renamed = {**first, "triple": ["new", "parent", "new"]}
    other = {**first, "object_iri": "urn:other"}
    first["axiom_origins"] = [{"original_syntax": "Different display syntax"}]
    renamed["axiom_origins"] = [{"original_syntax": "Another display syntax"}]
    assert scorer._stable_item_id("hierarchy", "source", first) == scorer._stable_item_id(
        "hierarchy", "source", renamed
    )
    assert scorer._stable_item_id("hierarchy", "source", first) != scorer._stable_item_id(
        "hierarchy", "source", other
    )
    assert scorer._stable_item_id("hierarchy", "source", {"triple": ["s", "p", "o"]}).startswith(
        "legacy:"
    )


def test_exact_axiom_origins_preserve_multiple_support_and_restriction_role():
    a, b = owl.Class(owl.IRI("urn:a")), owl.Class(owl.IRI("urn:b"))
    prop = owl.ObjectProperty(owl.IRI("urn:p"))
    axioms = [
        owl.SubClassOf(a, b),
        owl.EquivalentClasses(frozenset([a, b])),
        owl.SubClassOf(a, owl.ObjectSomeValuesFrom(prop, b)),
    ]
    source = SimpleNamespace(owl_snapshot=lambda: SimpleNamespace(iter_axioms=lambda: iter(axioms)))
    index = FeatureProvenance(source)
    hierarchy = index.enrich(
        {"subject_iri": "urn:a", "object_iri": "urn:b"}, kind="class", family="is_a"
    )
    assert len(hierarchy["source_axiom_refs"]) == 2
    restriction = index.enrich(
        {"subject_iri": "urn:a", "object_iri": "urn:b", "rel_iri": "urn:p"}, kind="class"
    )
    assert restriction["interpretation"] == "projected"
    assert restriction["derivation"]["rules"] == ["existential_named_filler_projection"]
    assert restriction["axiom_origins"][0]["original_syntax"]
    closure = index.enrich(
        {"subject_iri": "urn:a", "object_iri": "urn:b", "type_closure": True},
        kind="individual",
        family="is_a",
    )
    assert closure["type_closure"] is True
    assert closure["provenance_status"] == "provenance_unavailable"


def test_legacy_evidence_remains_unresolved_without_iris():
    record = {
        "triple_attributions": {
            "hierarchy": {
                "is_a": {"source": [{"triple": ["Same", "is_a", "Same"], "item_id": "old"}]}
            }
        }
    }
    entity = {"ontology_version_id": "v1", "iri": "urn:s", "kind": "class"}
    row = selected_evidence(record, entity, entity)[0]
    assert row["feature_id"] is None
    assert row["fact_ids"] == []
    assert row["historical_item_alias"] == "legacy:old"
    assert row["status"] == "not_exported"


def test_original_literal_identity_survives_facade_datatype_and_display_normalization():
    term = owl.Literal(
        "  exact lexical text  ", owl.Datatype(owl.IRI("http://www.w3.org/2001/XMLSchema#string"))
    )
    axiom = owl.AnnotationAssertion(
        owl.AnnotationProperty(owl.IRI("urn:definition")), owl.IRI("urn:a"), term
    )
    source = SimpleNamespace(
        owl_snapshot=lambda: SimpleNamespace(iter_axioms=lambda: iter([axiom]))
    )
    item = FeatureProvenance(source).enrich(
        {
            "entity_iri": "urn:a",
            "prop_iri": "urn:definition",
            "value": "exact lexical text",
            "literal_lexical_form": "  exact lexical text  ",
            "datatype": None,
            "language": None,
        },
        kind="class",
    )
    assert item["value"] == "exact lexical text"
    assert item["provenance_status"] == "available"
    assert item["literal_terms"] == [
        {
            "lexical_form": "  exact lexical text  ",
            "datatype": "http://www.w3.org/2001/XMLSchema#string",
            "language": None,
        }
    ]


def test_event_captures_an_immutable_copy_of_mutable_configuration():
    config = {"constraints": {"source_cardinality": 1}}
    observation = event("extraction", reason="selected", implementation="test", config=config)
    config["constraints"]["source_cardinality"] = 9
    assert observation["config"]["constraints"]["source_cardinality"] == 1


def test_empty_source_pool_and_explicit_nil_semantics_survive_import(tmp_path):
    run, _ = _run(tmp_path)
    (run / "source_decisions.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "records": [
                    {
                        "Src": "urn:empty",
                        "SrcKind": "class",
                        "candidate_count": 0,
                        "empty_candidate_pool": True,
                        "absence_semantics": "unknown",
                        "action": "abstain",
                        "ontology_nil_probability": None,
                    }
                ],
            }
        )
    )
    import_run(
        run,
        tmp_path / "package",
        source_ontology_version_id="s-v1",
        target_ontology_version_id="t-v1",
    )
    store = DecisionStore(tmp_path / "package")
    empty = next(row for row in store.sources()["items"] if row["entity"]["iri"] == "urn:empty")
    assert empty["source_decision"]["absence_semantics"] == "unknown"
    assert empty["source_decision"]["ontology_nil_probability"] is None
    assert store.candidates("urn:empty")["total_count"] == 0


def test_dataset_boundary_retains_prefiltered_candidates_without_reference_labels(tmp_path):
    from exact.runs.decisions import observe_dataset

    candidates = pd.DataFrame(
        [
            {"Src": "urn:s", "Tgt": "urn:exact", "cand_sim": 1.0, "Label": 1},
            {"Src": "urn:s", "Tgt": "urn:removed", "cand_sim": 0.8, "Label": 0},
        ]
    )
    dataset = SimpleNamespace(
        _candidates=candidates,
        dataframe=candidates.iloc[:1].copy(),
        candidate_pool_manifest={"origin": "provided", "fingerprint": "pool-v1"},
        filter_exact_matches=True,
        exact_matches=candidates.iloc[:1].copy(),
        drop_exact_match_sources=True,
    )
    observe_dataset(dataset)
    candidates.loc[0, "cand_sim"] = 0.0
    path = write_candidate_decisions(
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        policy={"relation_prediction": "none"},
        initial=dataset._candidate_stage_observations,
    )
    records = json.loads(path.read_text())["records"]
    assert len(records) == 2
    assert records[0]["values"]["cand_sim"] == 1.0
    assert records[1]["values"]["cand_rank"] == 2
    removed = {event["stage"]: event for event in records[1]["events"]}
    assert removed["prefilter"]["outcome"] == "rejected"
    assert removed["pair_scoring"]["status"] == "not_run"
    assert removed["retrieval"]["reason_code"] == "provided_candidate_pool"
    assert "Label" not in path.read_text()
    observe_dataset(dataset, restored=True)
    assert dataset._candidate_stage_observations == {}


def test_projector_subclass_alias_resolves_originals_without_changing_model_input():
    from exact.runs.decisions import feature_terms

    a, b = owl.Class(owl.IRI("urn:a")), owl.Class(owl.IRI("urn:b"))
    axiom = owl.SubClassOf(a, b)
    source = SimpleNamespace(
        owl_snapshot=lambda: SimpleNamespace(iter_axioms=lambda: iter([axiom]))
    )
    raw = {
        "subject_iri": "urn:a",
        "rel_iri": "http://subclassof",
        "object_iri": "urn:b",
        "triple": ["A", "//subclassof", "B"],
    }
    enriched = FeatureProvenance(source).enrich(raw, kind="class")
    assert all(enriched[key] == value for key, value in raw.items())
    assert enriched["source_axiom_refs"] == ["sha256:" + owl.structural_hexdigest(axiom)]
    assert feature_terms(enriched)["rel_iri"] == RDFS + "subClassOf"
    assert enriched["derivation"]["rules"] == ["owl2vecstar_subclass_predicate_alias"]


def test_import_rejects_same_iris_bound_to_different_ontology_bytes(tmp_path):
    run, _ = _run(tmp_path)
    source_hash, target_hash = canonical_hash("source"), canonical_hash("target")
    write_candidate_decisions(
        run,
        pd.DataFrame([{"Src": "urn:s", "Tgt": "urn:t", "S_final": 0.9}]),
        pd.DataFrame(),
        policy={"relation_prediction": "none"},
        pool={
            "inputs": {"source": {"sha256": source_hash[7:]}, "target": {"sha256": target_hash[7:]}}
        },
    )
    with pytest.raises(DomainError, match="ontology bytes differ"):
        import_run(
            run,
            tmp_path / "rejected",
            source_ontology_version_id="s-v1",
            target_ontology_version_id="t-v1",
            source_root_sha256=canonical_hash("wrong"),
            target_root_sha256=target_hash,
        )
    assert not (tmp_path / "rejected" / "decisions.sqlite").exists()
    manifest = import_run(
        run,
        tmp_path / "verified",
        source_ontology_version_id="s-v1",
        target_ontology_version_id="t-v1",
        source_root_sha256=source_hash,
        target_root_sha256=target_hash,
    )
    assert manifest["ontology_input_binding"]["status"] == "verified"


def test_legacy_run_without_root_digests_never_claims_verified_context(tmp_path):
    run, _ = _run(tmp_path)
    manifest = import_run(
        run,
        tmp_path / "legacy",
        source_ontology_version_id="s-v1",
        target_ontology_version_id="t-v1",
        source_root_sha256=canonical_hash("s"),
        target_root_sha256=canonical_hash("t"),
    )
    assert manifest["ontology_input_binding"]["status"] == "unverified"
    assert manifest["ontology_input_binding"]["source"]["status"] == "legacy_unverified"


def test_import_requires_every_consumed_artifact_in_declared_inventory(tmp_path):
    from exact_inspect.contracts import file_hash

    run, _ = _run(tmp_path)
    expected = {str(p.resolve()): file_hash(p) for p in run.rglob("*") if p.is_file()}
    mapping = next(run.glob("alignment/maps_global.tsv"))
    omitted = {path: sha for path, sha in expected.items() if path != str(mapping.resolve())}
    with pytest.raises(DomainError) as error:
        import_run(
            run,
            tmp_path / "rejected",
            source_ontology_version_id="s-v1",
            target_ontology_version_id="t-v1",
            expected_artifacts=omitted,
        )
    assert error.value.envelope.code == "unverified_run_input"
    assert not (tmp_path / "rejected" / "decisions.sqlite").exists()
    manifest = import_run(
        run,
        tmp_path / "accepted",
        source_ontology_version_id="s-v1",
        target_ontology_version_id="t-v1",
        expected_artifacts=expected,
    )
    assert manifest["input_binding"]["status"] == "verified"

    mapping.write_text(mapping.read_text().replace("0.9", "0.8"))
    with pytest.raises(DomainError) as error:
        import_run(
            run,
            tmp_path / "changed",
            source_ontology_version_id="s-v1",
            target_ontology_version_id="t-v1",
            expected_artifacts=expected,
        )
    assert error.value.envelope.code == "unverified_run_input"
    assert not (tmp_path / "changed" / "decisions.sqlite").exists()


def test_candidate_summary_scores_retain_producing_stage_and_unknown_legacy_scope():
    from exact_inspect.decisions import _summary_scores

    values = {"cand_sim": 0.7, "S_base": 0.8, "S_final": 0.9, "P_match": 0.9, "old": 0.1}
    events = [
        {"stage": "retrieval", "values": {"cand_sim": 0.7}},
        {"stage": "pair_scoring", "values": {"S_base": 0.8, "S_final": 0.8}},
        {"stage": "selection", "values": {"P_match": 0.9, "S_final": 0.9}},
        {"stage": "extraction", "values": {"S_final": 0.9}},
    ]
    scores = {score["name"]: score for score in _summary_scores(values, events)}
    assert {name: score["stage"] for name, score in scores.items()} == {
        "cand_sim": "retrieval",
        "S_base": "pair_scoring",
        "S_final": "selection",
        "P_match": "selection",
        "old": "saved_artifact",
    }
    assert {name: score["value"] for name, score in scores.items()} == values
