"""A metadata-derived input must pass real native admission under its own identity."""

import json
from pathlib import Path

import pytest

from tools import validate_native_owl
from tools.prepared_batch import binding, write


def _prepared(tmp_path):
    from tools.prepare_bridge_ontology import EXPECTED, prepare

    inputs = {}
    for side, entity in (("source", "a"), ("target", "b")):
        metadata = ""
        if side == "source":
            iri = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#textArea"
            metadata = (
                f'<rdfs:Datatype rdf:about="{iri}"/>'
                '<owl:AnnotationProperty rdf:about="urn:definition">'
                f'<rdfs:range rdf:resource="{iri}"/></owl:AnnotationProperty>'
            )
        path = tmp_path / (side + ".owl")
        path.write_text(
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
            'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" '
            'xmlns:owl="http://www.w3.org/2002/07/owl#" '
            f'xml:base="https://example.org/{side}">'
            f'<owl:Ontology rdf:about="https://example.org/{side}"/>'
            f'{metadata}<owl:Class rdf:about="urn:{entity}"/></rdf:RDF>'
        )
        inputs[side] = binding(path)
    expected = {iri: {"declaration": 0, "annotation_range": 0} for iri in EXPECTED}
    expected[next(iri for iri in expected if iri.endswith("#textArea"))] = {
        "declaration": 1,
        "annotation_range": 1,
    }
    manifest = prepare(inputs, {}, tmp_path / "reasoning-inputs", expected_counts=expected)
    recipe = {
        "kind": "e14_native_admission",
        "inputs": inputs,
        "imports": {},
        "reasoning_preparation": manifest,
        "output": str(tmp_path / "admission"),
        "max_memory_bytes": 64 * 1024**2,
        "query_classes": {"source": "urn:a", "target": "urn:b"},
    }
    path = tmp_path / "recipe.json"
    write(path, recipe)
    return path, recipe


def test_prepared_input_passes_strict_native_admission_and_resumes(tmp_path):
    path, recipe = _prepared(tmp_path)
    result = validate_native_owl.run_diagnostic(path)
    assert result["status"] == "passed"
    assert result["inputs"] == recipe["inputs"]
    assert result["reasoning_preparation"] == recipe["reasoning_preparation"]
    assert result["reasoning_inputs"]["source"] != recipe["inputs"]["source"]
    assert "metadata_exclusions" in result["admission_scope"]
    assert validate_native_owl.run_diagnostic(path)["reused_ontologies"] == ["source", "target"]
    # A previously admitted derived document cannot masquerade as original-file admission.
    recipe.pop("reasoning_preparation")
    write(path, recipe)
    with pytest.raises(ValueError, match="admission identity changed"):
        validate_native_owl.run_diagnostic(path)


def test_derived_input_changed_during_native_load_cannot_publish_pass(tmp_path, monkeypatch):
    import pyowl_core as core

    path, recipe = _prepared(tmp_path)
    original = core.load_snapshot
    calls = []

    def load(document, **kwargs):
        view = original(document, **kwargs)
        if not calls:
            with Path(document).open("a") as stream:
                stream.write("\n<!-- changed after native load -->")
        calls.append(document)
        return view

    monkeypatch.setattr(core, "load_snapshot", load)
    with pytest.raises(ValueError):
        validate_native_owl.run_diagnostic(path)
    assert calls
    output = Path(recipe["output"])
    assert not (output / "completion.json").exists()
    assert json.loads((output / "admission.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("artifact", ["original", "manifest"])
def test_preparation_drift_blocks_before_native_load(tmp_path, monkeypatch, artifact):
    import pyowl_core as core

    path, recipe = _prepared(tmp_path)
    item = recipe["inputs"]["source"] if artifact == "original" else recipe["reasoning_preparation"]
    with Path(item["path"]).open("a") as stream:
        stream.write("\n")

    def forbidden(*args, **kwargs):
        pytest.fail("An invalid preparation must be rejected before native loading")

    monkeypatch.setattr(core, "load_snapshot", forbidden)
    with pytest.raises(ValueError):
        validate_native_owl.run_diagnostic(path)
    assert not (Path(recipe["output"]) / "completion.json").exists()
