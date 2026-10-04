"""The native comparison freezes the same public training anchors for both views."""

import json

import pandas as pd
import pytest

from tools import run_bridge_diagnostic as diagnostic

_METADATA_DATATYPE = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#textArea"


def _fixture(tmp_path):
    from tools.run_directional_diagnostic import write

    inputs = {}
    for side, rows, edge in (
        ("source", [("a", "urn:a"), ("c", "urn:c")], ("c", "a")),
        ("target", [("b", "urn:b"), ("d", "urn:d")], ("b", "d")),
    ):
        path = tmp_path / side
        path.mkdir()
        pd.DataFrame(rows, columns=["node_id", "iri"]).to_csv(path / "properties.csv", index=False)
        pd.DataFrame([(name, "class") for name, _ in rows], columns=["entity", "kind"]).to_csv(
            path / "entities.csv", index=False
        )
        pd.DataFrame([(edge[0], "subclass_of", edge[1])], columns=["src", "rel", "dst"]).to_csv(
            path / "triples.csv", index=False
        )
        (path / "kg.yaml").write_text(
            "descriptor_version: 1\nentities_file: entities.csv\n"
            "triples_files: [triples.csv]\nhierarchy_relations: [subclass_of]\n"
        )
        inputs[side] = diagnostic.bind(path)
    for role, row in [("train", ("a", "b", "=")), ("valid", ("c", "d", "<"))]:
        path = tmp_path / (role + ".tsv")
        pd.DataFrame([row], columns=["SrcEntity", "TgtEntity", "Relation"]).to_csv(
            path, sep="\t", index=False
        )
        inputs[role] = diagnostic.bind(path)
    owl = {}
    for side, declarations, edge in [
        ("source", ["a", "c"], ("c", "a")),
        ("target", ["b", "d"], ("b", "d")),
    ]:
        path = tmp_path / (side + ".ofn")
        path.write_text(
            "Ontology("
            + " ".join(f"Declaration(Class(<urn:{name}>))" for name in declarations)
            + f" SubClassOf(<urn:{edge[0]}> <urn:{edge[1]}>))"
        )
        owl[side] = diagnostic.bind(path)
    recipe = {
        "schema_version": 1,
        "kind": "e14_native_known_pairs",
        "inputs": inputs,
        "owl": owl,
        "imports": {},
        "output": str(tmp_path / "result"),
        "source_cap": 300,
        "seed": 17,
        "reference_scope": "public_development_train_valid_only",
        "anchors": "all_public_train_equivalences_query_bridge_excluded",
        "selection_eligible": False,
        "hosted_calls": False,
        "rationales": False,
        "max_memory_bytes": 64 * 1024**2,
        "timeout_seconds": None,
        "comparison_scope": "known_valid_pairs_graph_view_vs_full_OWL_not_logical_parity_or_end_to_end_F1",
    }
    path = tmp_path / "recipe.json"
    write(path, recipe)
    return path, recipe


def test_bridge_comparison_uses_same_anchors_and_recovers_native_queries(tmp_path):
    path, recipe = _fixture(tmp_path)
    result = diagnostic.run_diagnostic(path)
    assert result["anchor_count"] == result["query_count"] == 1
    assert result["graph"]["by_relation"]["<"]["tp"] == 1
    assert result["native"]["by_relation"]["<"]["tp"] == 1
    assert result["selection_eligible"] is False
    second = diagnostic.run_diagnostic(path)
    assert second["native_coherence"]["compiled_anchor_worlds"] == 0
    receipt = json.loads((tmp_path / "result/completion.json").read_text())
    assert receipt["selection_eligible"] is False


def test_bridge_comparison_rejects_test_reference_binding(tmp_path):
    path, recipe = _fixture(tmp_path)
    recipe["inputs"]["test"] = recipe["inputs"]["valid"]
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="public train/valid only"):
        diagnostic.run_diagnostic(path)


def test_bridge_index_allowance_reaches_config_and_invalidates_checkpoint(tmp_path, monkeypatch):
    import pyhermit

    path, recipe = _fixture(tmp_path)
    recipe["max_native_symbol_index_bytes"] = 128 * 1024**2
    path.write_text(json.dumps(recipe))
    original = pyhermit.ReasonerConfig
    captured = []

    def config(**options):
        captured.append(options.pop("max_native_symbol_index_bytes"))
        return original(**options)

    monkeypatch.setattr(pyhermit, "ReasonerConfig", config)
    result = diagnostic.run_diagnostic(path)
    assert result["native"]["by_relation"]["<"]["tp"] == 1
    assert captured and set(captured) == {128 * 1024**2}
    recipe["max_native_symbol_index_bytes"] *= 2
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="checkpoint inputs or implementation changed"):
        diagnostic.run_diagnostic(path)


def test_bridge_comparison_detects_inputs_changing_during_native_work(tmp_path, monkeypatch):
    path, recipe = _fixture(tmp_path)
    original = diagnostic.native_bridge

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        target = tmp_path / "target/properties.csv"
        target.write_text(target.read_text() + "\n")
        return result

    monkeypatch.setattr(diagnostic, "native_bridge", changed)
    with pytest.raises(ValueError, match="input changed"):
        diagnostic.run_diagnostic(path)
    assert not (tmp_path / "result/completion.json").exists()


def test_native_admission_does_not_read_alignment_references(tmp_path):
    from tools.validate_native_owl import run_diagnostic

    _, recipe = _fixture(tmp_path)
    admission = {
        "kind": "e14_native_admission",
        "inputs": recipe["owl"],
        "imports": {},
        "output": str(tmp_path / "admission"),
        "max_memory_bytes": 64 * 1024**2,
        "query_classes": {"source": "urn:a", "target": "urn:b"},
    }
    for role in ("train", "valid"):
        (tmp_path / (role + ".tsv")).unlink()
    path = tmp_path / "admission-recipe.json"
    path.write_text(json.dumps(admission))
    result = run_diagnostic(path)
    assert result["status"] == "passed"
    assert {report["status"] for report in result["ontologies"].values()} == {"passed"}
    assert json.loads((tmp_path / "admission/completion.json").read_text())["status"] == "complete"
    second = run_diagnostic(path)
    assert second["reused_ontologies"] == ["source", "target"]
    admission["inputs"]["test"] = admission["inputs"]["source"]
    path.write_text(json.dumps(admission))
    with pytest.raises(ValueError, match="only original OWL inputs"):
        run_diagnostic(path)


def _prepared_fixture(tmp_path):
    from tools.prepare_bridge_ontology import EXPECTED, prepare

    path, recipe = _fixture(tmp_path)
    for side, declarations, edge in [
        ("source", ["a", "c"], ("c", "a")),
        ("target", ["b", "d"], ("b", "d")),
    ]:
        ontology = tmp_path / (side + ".owl")
        metadata = ""
        if side == "source":
            for datatype, counts in EXPECTED.items():
                metadata += f'<rdfs:Datatype rdf:about="{datatype}"/>'
                for index in range(counts["annotation_range"]):
                    metadata += (
                        f'<owl:AnnotationProperty rdf:about="urn:description:{datatype}:{index}">'
                        f'<rdfs:range rdf:resource="{datatype}"/></owl:AnnotationProperty>'
                    )
        ontology.write_text(
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
            'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" '
            'xmlns:owl="http://www.w3.org/2002/07/owl#">'
            + metadata
            + "".join(f'<owl:Class rdf:about="urn:{name}"/>' for name in declarations)
            + f'<owl:Class rdf:about="urn:{edge[0]}">'
            f'<rdfs:subClassOf rdf:resource="urn:{edge[1]}"/></owl:Class></rdf:RDF>'
        )
        recipe["owl"][side] = diagnostic.bind(ontology)
    recipe["reasoning_preparation"] = prepare(
        recipe["owl"], recipe["imports"], tmp_path / "preparation"
    )
    path.write_text(json.dumps(recipe))
    return path, recipe


def test_prepared_native_bridge_preserves_full_logical_chain_and_checkpoint(tmp_path):
    path, recipe = _prepared_fixture(tmp_path)
    result = diagnostic.run_diagnostic(path)
    assert result["native"]["by_relation"]["<"]["tp"] == 1
    assert result["graph"]["by_relation"]["<"]["tp"] == 1
    assert result["reasoning_preparation"] == recipe["reasoning_preparation"]
    assert result["reasoning_inputs"]["source"] != recipe["owl"]["source"]
    assert result["native_coherence"]["compiled_anchor_worlds"] == 1
    assert result["native_coherence"]["ontology_consistency"] == "consistent"
    assert diagnostic.run_diagnostic(path)["native_coherence"]["compiled_anchor_worlds"] == 0


def test_prepared_bridge_rechecks_derived_bytes_after_native_work(tmp_path, monkeypatch):
    path, recipe = _prepared_fixture(tmp_path)
    original = diagnostic.native_bridge

    def changed(frame, source, target, **options):
        result = original(frame, source, target, **options)
        source.write_text(source.read_text() + "\n")
        return result

    monkeypatch.setattr(diagnostic, "native_bridge", changed)
    with pytest.raises(ValueError, match="input changed"):
        diagnostic.run_diagnostic(path)
    assert not (tmp_path / "result/completion.json").exists()


@pytest.mark.parametrize("entity", ["a", "c"])
def test_prepared_bridge_rejects_removed_datatype_as_anchor_or_query_class(
    tmp_path, monkeypatch, entity
):
    path, recipe = _prepared_fixture(tmp_path)
    properties = tmp_path / "source/properties.csv"
    frame = pd.read_csv(properties)
    frame.loc[frame.node_id.eq(entity), "iri"] = _METADATA_DATATYPE
    frame.to_csv(properties, index=False)
    recipe["inputs"]["source"] = diagnostic.bind(properties.parent)
    path.write_text(json.dumps(recipe))

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsafe bridge must be rejected before native reasoning")

    monkeypatch.setattr(diagnostic, "native_bridge", forbidden)
    with pytest.raises(ValueError, match="datatype cannot be a native bridge/query class"):
        diagnostic.run_diagnostic(path)
    assert not (tmp_path / "result/completion.json").exists()
