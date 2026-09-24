from exact.core.entities.kinds import EntityKind
from exact.experiments.evidence_inventory import evidence_inventory, export_matched_csv
from exact.impl.datasets.pair_adaptive_context import PairAdaptiveContextDataset
from exact.io.sources.csv_kg import CsvKgSource
from tests import kind_evidence_controls_test

dataset = kind_evidence_controls_test.dataset
SRC = kind_evidence_controls_test.SRC


def _scoring_features(value):
    provenance_fields = {
        "source_axiom_refs",
        "axiom_origins",
        "derivation",
        "provenance_status",
        "interpretation",
        "literal_terms",
    }
    if isinstance(value, dict):
        return {
            key: _scoring_features(item)
            for key, item in value.items()
            if key not in provenance_fields
        }
    if isinstance(value, list):
        return [_scoring_features(item) for item in value]
    return value


def test_real_owl_to_csv_preserves_typed_evidence_and_literal_identity(dataset, tmp_path):
    destination = tmp_path / "matched-source"
    manifest = export_matched_csv(dataset.source, destination)
    assert manifest["comparison"]["evidence_parity"]
    csv_source = CsvKgSource.from_path(destination)
    heart_annotations = csv_source.annotations(SRC + "Heart")
    assert any(value.lang == "pt" for value in heart_annotations)
    assert any(value.datatype for value in csv_source.annotations(SRC + "DeprecatedConcept"))
    assert csv_source.entities(EntityKind.OBJECT_PROPERTY) == tuple(
        dataset.source.entities(EntityKind.OBJECT_PROPERTY)
    )
    restored = PairAdaptiveContextDataset(
        output_path=tmp_path / "features",
        cache_ok=False,
        verbaliser_name=None,
        projection_include_literals=True,
        entity_kinds=["class", "object_property", "data_property", "individual"],
    )
    restored._source = restored._target = csv_source
    for kind in (
        EntityKind.CLASS,
        EntityKind.OBJECT_PROPERTY,
        EntityKind.DATA_PROPERTY,
        EntityKind.INDIVIDUAL,
    ):
        for iri in dataset.source.entities(kind):
            owl_features = dataset.get_entity_features(iri, "src", kind)
            csv_features = restored.get_entity_features(iri, "src", kind)
            # The matched CSV retains scoring terms, but does not carry original
            # OWL axiom syntax. Its new audit metadata must disclose this loss.
            assert _scoring_features(csv_features) == _scoring_features(owl_features)
            for item in csv_features["attributes"]:
                assert item["provenance_status"] == "provenance_unavailable"
                assert item["source_axiom_refs"] == []
    assert restored._property_experiment_extras("src") == dataset._property_experiment_extras("src")
    assert evidence_inventory(csv_source)["sha256"] == manifest["source_inventory"]["sha256"]


def test_public_biokg_table_shape_preserves_candidate_node_ids_and_drops_anchors(tmp_path):
    import csv

    from exact.experiments.evidence_inventory import import_biokg_public_graph

    package = tmp_path / "public"
    (package / "graph").mkdir(parents=True)
    with (package / "graph/properties.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "node_id",
                "ontology",
                "iri",
                "local_id",
                "preferred_label",
                "synonyms",
                "definition",
                "semantic_category",
                "source_version",
            ]
        )
        writer.writerow(
            [
                "NCIT:C001",
                "NCIT",
                "https://example.org/NCIT/C001",
                "C001",
                "Lung cancer",
                "Pulmonary cancer",
                "A concept.",
                "disease",
                "mini",
            ]
        )
        writer.writerow(
            [
                "DOID:D001",
                "DOID",
                "https://example.org/DOID/D001",
                "D001",
                "Lung cancer",
                "Cancer of lung",
                "",
                "disease",
                "mini",
            ]
        )
    with (package / "graph/triples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "triple_id",
                "head_id",
                "relation",
                "tail_id",
                "head_ontology",
                "tail_ontology",
                "source",
                "provenance",
                "is_inferred",
                "is_anchor",
                "release_layer",
            ]
        )
        writer.writerow(
            [
                "T1",
                "NCIT:C001",
                "subclass_of",
                "NCIT:ROOT",
                "NCIT",
                "NCIT",
                "mini",
                "asserted",
                "false",
                "false",
                "public",
            ]
        )
        writer.writerow(
            [
                "T2",
                "NCIT:C001",
                "equivalent",
                "DOID:D001",
                "NCIT",
                "DOID",
                "mini",
                "anchor",
                "false",
                "true",
                "public",
            ]
        )
    destination = tmp_path / "source"
    manifest = import_biokg_public_graph(package, destination, ontology="NCIT")
    source = CsvKgSource.from_path(destination)
    assert source.entities(EntityKind.CLASS) == ("NCIT:C001", "NCIT:ROOT")
    assert source.labels("NCIT:C001") == ["Lung cancer", "Pulmonary cancer"]
    assert source.direct_parents("NCIT:C001") == ["NCIT:ROOT"]
    assert manifest["alignment_anchors_imported"] == 0 and manifest["retained_edges"] == 1
