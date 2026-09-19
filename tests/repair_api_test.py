"""Standalone/sequential handoff, feature preservation, occurrence and artifact checks."""

import json
import subprocess
import sys

import pyowl_core as owl
import pytest

from exact.repair.api import ontology_occurrences, prepare_repair, write_artifact
from exact.repair.candidates import make_candidate
from exact.repair.kernel import materialize, replay_safety
from exact.repair.owl import snapshot_from_axioms
from exact.repair.records import RepairInputV2, RevisionObjectV2, read_record


def cls(name):
    return owl.Class(owl.IRI(f"urn:api:{name}"))


def test_full_features_preserved_and_matching_never_required():
    source = snapshot_from_axioms((owl.SubClassOf(cls("S"), cls("Parent")),))
    target = snapshot_from_axioms((owl.Declaration(cls("T")),))
    rows = [
        {
            "Src": "urn:api:S",
            "Tgt": "urn:api:T",
            "Score": 7.5,
            "object_id": "m",
            "channels": {"lexical": [0.2, 0.3], "structural": -0.5},
        }
    ]
    features = {
        "m": {
            "decomposition": {"attributes": ["a", "b"]},
            "embedding": [1, 2, 3],
            "explanations": [{"text": "observed"}],
            "reference_alignment": ["must not leak"],
        }
    }
    problem = prepare_repair(source, target, rows, evidence=features)
    evidence = dict(problem.evidence)
    assert evidence["m"]["matching_features"]["embedding"] == (1, 2, 3)
    assert evidence["m"]["mapping"]["channels"]["structural"] == -0.5
    assert "reference_alignment" not in evidence["m"]["matching_features"]
    assert evidence["evidence_omissions"] == ("m.reference_alignment",)
    assert set(problem.policy.monitored_classes) == {cls("S"), cls("T"), cls("Parent")}
    assert (
        read_record(json.loads(json.dumps(problem.to_dict()))).content_hash == problem.content_hash
    )


def test_scores_can_be_missing_but_not_nonfinite():
    view = snapshot_from_axioms(())
    problem = prepare_repair(view, view, [{"Src": "urn:api:S", "Tgt": "urn:api:T"}])
    assert dict(problem.evidence)[problem.objects[0].object_id]["score_missing"]
    with pytest.raises(ValueError, match="finite"):
        prepare_repair(
            view, view, [{"Src": "urn:api:S", "Tgt": "urn:api:T", "Score": float("nan")}]
        )


def test_duplicate_occurrence_patch_preserves_fixed_copy():
    axiom = owl.SubClassOf(cls("S"), cls("T"))
    view = snapshot_from_axioms((axiom,))
    occurrences = [row for row in ontology_occurrences(view, view) if row[2] == axiom]
    assert len(occurrences) == 2
    identity, origin, original = occurrences[0]
    candidate = make_candidate("o", (), ("delete",))
    obj = RevisionObjectV2(
        "o", "ontology_axiom", (original,), (candidate,), occurrence_id=identity, source=origin
    )
    problem = prepare_repair(view, view, [], ontology_objects=(obj,))
    assert axiom in materialize(problem, (0,))[0]
    assert axiom in view.iter_axioms()


def test_punning_keeps_typed_property_correspondence():
    view = snapshot_from_axioms(())
    rows = [
        {"Src": "urn:api:S", "Tgt": "urn:api:T", "Kind": kind, "Score": 0.5}
        for kind in ("class", "object_property")
    ]
    problem = prepare_repair(view, view, rows)
    property_obj = problem.objects[1]
    assert all(isinstance(ax, owl.SubObjectPropertyOf) for ax in property_obj.original_axioms)
    assert {tag for c in property_obj.candidates for tag in c.action_tags} == {"keep", "delete"}


def test_record_hash_legacy_rejection_and_atomic_write(tmp_path):
    view = snapshot_from_axioms(())
    problem = prepare_repair(view, view, [{"Src": "urn:api:S", "Tgt": "urn:api:T"}])
    destination = tmp_path / "repair.json"
    write_artifact(destination, problem.to_dict())
    assert isinstance(read_record(json.loads(destination.read_text())), RepairInputV2)
    payload = problem.to_dict()
    payload["schema"] = "exact-repair/records/v1"
    with pytest.raises(ValueError, match="migration"):
        read_record(payload)
    payload = problem.to_dict()
    payload["record"]["matcher_identity"] = "tampered"
    with pytest.raises(ValueError, match="hash"):
        read_record(payload)


def test_import_is_lazy():
    code = "import exact.repair, sys; assert not any(m in sys.modules for m in ('torch','pysat','pysdd','pyhermit','pyelk'))"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


def test_standalone_cli_and_safety_replay(tmp_path):
    source = tmp_path / "source.ofn"
    target = tmp_path / "target.ofn"
    source.write_text("Ontology(<urn:s> Declaration(Class(<urn:S>)))")
    target.write_text("Ontology(<urn:t> Declaration(Class(<urn:T>)))")
    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps([{"Src": "urn:S", "Tgt": "urn:T", "Score": 0.9}]))
    output = tmp_path / "repair.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "exact.delivery.cli.main",
            "repair",
            "--source",
            str(source),
            "--target",
            str(target),
            "--alignment",
            str(alignment),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(output.read_text())
    problem, repaired = read_record(payload["input"]), read_record(payload["result"])
    assert repaired.logical_status == "VERIFIED_FEASIBLE"
    assert replay_safety(problem, repaired, timeout=10)
    assert source.read_text() == "Ontology(<urn:s> Declaration(Class(<urn:S>)))"


def test_sequential_run_reuses_complete_available_feature_artifacts(tmp_path):
    from exact.repair.api import matching_run_evidence
    from exact.runs import ExplanationStore, RunLayout

    layout = RunLayout.create(tmp_path / "matching")
    layout.mapping_path("global").write_text(
        "SrcEntity\tTgtEntity\tScore\nurn:api:S\turn:api:T\t0.8\n"
    )
    layout.mapping_path("local").write_text(
        "SrcEntity\tTgtEntity\tScore\nurn:api:S\turn:api:Alternative\t0.7\n"
    )
    store = ExplanationStore(layout.explanations_dir, run_id="matching")
    store.append(
        [
            {
                "src_iri": "urn:api:S",
                "tgt_iri": "urn:api:T",
                "decomposition": {"attributes": ["name", "parent"]},
                "confidences": {"lexical": 0.8, "structural": 0.5},
            }
        ]
    )
    layout.checkpoints_dir.mkdir(exist_ok=True)
    (layout.checkpoints_dir / "inference_features.json").write_text(
        json.dumps({"channels": {"full_embedding": [1, 2, 3]}, "gold_label": True})
    )
    (layout.checkpoints_dir / "test_labels.json").write_text('{"do_not_read":true}')
    mappings, evidence = matching_run_evidence(layout.root)
    assert mappings.iloc[0]["Score"] == 0.8
    assert evidence["explanations"][0]["decomposition"]["attributes"] == ["name", "parent"]
    artifact = evidence["inference_artifacts"][0]
    assert artifact["content"] == {"channels": {"full_embedding": [1, 2, 3]}}
    assert artifact["omitted"] == ("gold_label",)
    assert len(evidence["inference_artifacts"]) == 1
    view = snapshot_from_axioms(())
    problem = prepare_repair(view, view, mappings, matcher_identity="exact", evidence=evidence)
    assert dict(problem.evidence)["alternative_candidates"][0]["TgtEntity"] == "urn:api:Alternative"


def test_mapping_locks_and_declared_kinds_are_preserved():
    from exact.core.entities.kinds import EntityKind

    view = snapshot_from_axioms(())
    row = {"Src": "urn:api:S", "Tgt": "urn:api:T", "Kind": EntityKind.CLASS, "locked": True}
    problem = prepare_repair(view, view, [row])
    assert problem.objects[0].source_entity == cls("S")
    assert problem.objects[0].target_entity == cls("T")
    assert problem.objects[0].locked
    assert len(problem.objects[0].candidates) == 1
    with pytest.raises(ValueError, match="cross-kind"):
        prepare_repair(view, view, [{**row, "TgtKind": "object_property"}])


def test_optional_row_fields_keep_missingness_and_defaults():
    view = snapshot_from_axioms(())
    problem = prepare_repair(
        view,
        view,
        [
            {
                "Src": "urn:api:A",
                "Tgt": "urn:api:B",
                "Score": None,
                "locked": True,
                "object_id": "fixed",
            },
            {"Src": "urn:api:C", "Tgt": "urn:api:D", "Score": 0.8},
        ],
    )
    assert problem.objects[0].locked and not problem.objects[1].locked
    assert problem.objects[1].object_id != "nan"
    assert dict(problem.evidence)["fixed"]["score_missing"]


def test_in_memory_array_and_tensor_channels_are_frozen_without_gradients():
    import numpy as np
    import torch

    from exact.repair.api import observable_evidence

    embedding = torch.tensor([[1.0, 2.0]], requires_grad=True)
    observed, omitted = observable_evidence(
        {"embedding": embedding, "scores": np.array([0.1, 0.2])}
    )
    assert observed == {"embedding": [[1.0, 2.0]], "scores": [0.1, 0.2]}
    assert embedding.requires_grad and not omitted
    with pytest.raises(ValueError, match="nonfinite"):
        observable_evidence({"embedding": np.array([float("inf")])})


def test_snapshot_document_identities_are_preserved(tmp_path):
    path = tmp_path / "source.ofn"
    path.write_text("Ontology(<urn:source> Declaration(Class(<urn:S>)))")
    source = owl.load_snapshot(str(path))
    problem = prepare_repair(source, source, [])
    assert len(problem.source_documents) == 1
    key, location, digest = problem.source_documents[0]
    assert key and location.endswith("source.ofn") and len(digest) == 64
    assert problem.target_documents == problem.source_documents
