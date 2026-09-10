"""Canonical evidence parity and information-loss inventories for E13."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from typing import Any, Iterable

from exact.core.contracts.knowledge import KnowledgeSource
from exact.core.entities.kinds import EntityKind
from exact.io.sources.evidence import property_schema_evidence


def evidence_inventory(
    source: KnowledgeSource, *, kinds: Iterable[EntityKind] = tuple(EntityKind)
) -> dict[str, Any]:
    """Record exposed evidence, preserving literal identities and typed entities."""
    entities = sorted(
        (EntityKind(kind).value, str(iri)) for kind in kinds for iri in source.entities(kind)
    )
    iris = sorted({iri for _, iri in entities})
    facts: dict[str, list[Any]] = {
        "entities": entities,
        "labels": sorted((iri, label) for iri in iris for label in set(source.labels(iri))),
        "annotations": sorted(
            {
                json.dumps({"entity": iri, **asdict(value)}, sort_keys=True, ensure_ascii=False)
                for iri in iris
                for value in source.annotations(iri)
            }
        ),
        "attributes": sorted(
            {
                json.dumps({"entity": iri, **asdict(value)}, sort_keys=True, ensure_ascii=False)
                for iri in iris
                for value in source.attributes(iri)
            }
        ),
        "hierarchy": sorted(
            {
                (kind, iri, parent)
                for kind, iri in entities
                for parent in source.direct_parents(iri, EntityKind(kind))
            }
        ),
        "domains": sorted(
            (iri, domain)
            for kind, iri in entities
            if kind.endswith("property")
            for domain in source.property_domains(iri)
        ),
        "ranges": sorted(
            (iri, range_)
            for kind, iri in entities
            if kind.endswith("property")
            for range_ in source.property_ranges(iri)
        ),
        "excluded": sorted(source.excluded_from_alignment()),
        "property_axioms": property_schema_evidence(source),
        "projected_iri_edges": sorted(
            {edge.astuple() for edge in source.projection_edges(include_literals=False)}
        ),
        "projected_edges": sorted(
            {edge.astuple() for edge in source.projection_edges(include_literals=True)}
        ),
    }
    encoded = json.dumps(facts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    snapshot = getattr(source, "owl_snapshot", None)
    axiom_types = (
        Counter(type(axiom).__name__ for axiom in snapshot().iter_axioms())
        if callable(snapshot)
        else None
    )
    return {
        "schema_version": 1,
        "fact_semantics": "set",
        "facts": facts,
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "counts": {name: len(rows) for name, rows in facts.items()},
        "owl_axiom_counts": dict(sorted(axiom_types.items())) if axiom_types is not None else None,
    }


def compare_evidence_inventories(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Expose missing and extra facts instead of crediting information gain to syntax."""
    differences = {}
    for section in sorted(set(left["facts"]) | set(right["facts"])):
        lhs = {
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in left["facts"].get(section, [])
        }
        rhs = {
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in right["facts"].get(section, [])
        }
        differences[section] = {
            "left_only": [json.loads(row) for row in sorted(lhs - rhs)],
            "right_only": [json.loads(row) for row in sorted(rhs - lhs)],
        }
    return {
        "schema_version": 1,
        "left_sha256": left["sha256"],
        "right_sha256": right["sha256"],
        "evidence_parity": all(
            not row["left_only"] and not row["right_only"] for row in differences.values()
        ),
        "differences": differences,
        "owl_axiom_counts": {
            "left": left.get("owl_axiom_counts"),
            "right": right.get("owl_axiom_counts"),
        },
        "owl_semantic_equivalence": "not_established_by_projection_parity",
    }


def export_matched_csv(source: KnowledgeSource, destination) -> dict[str, Any]:
    """Export the scorer-visible OWL information into the same CSV-KG adapter.

    Projected triples alone are the graph-only information-loss arm. The metadata
    table makes the matched-information arm retain typed entities and literal
    datatype/language, including facts that are not projected as graph edges.
    """
    import csv
    from pathlib import Path

    import yaml

    from exact.io.sources.csv_kg import CsvKgSource

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Matched CSV export needs an empty destination")
    inventory = evidence_inventory(source)
    with (destination / "triples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["src", "rel", "dst"])
        writer.writerows(inventory["facts"]["projected_iri_edges"])
    with (destination / "evidence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["section", "record"])
        for section, records in inventory["facts"].items():
            for row in records:
                writer.writerow([section, json.dumps(row, sort_keys=True, ensure_ascii=False)])
    descriptor = {
        "descriptor_version": 1,
        "triples_files": ["triples.csv"],
        "evidence_file": "evidence.csv",
    }
    (destination / "kg.yaml").write_text(yaml.safe_dump(descriptor), encoding="utf-8")
    comparison = compare_evidence_inventories(
        inventory, evidence_inventory(CsvKgSource.from_path(destination))
    )
    manifest = {
        "schema_version": 1,
        "representation": "matched_information_csv",
        "source_origin": str(source.origin),
        "source_inventory": inventory,
        "comparison": comparison,
        "files": {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in ("kg.yaml", "triples.csv", "evidence.csv")
        },
    }
    (destination / "conversion.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not comparison["evidence_parity"]:
        raise ValueError("Matched CSV conversion failed evidence parity; inspect conversion.json")
    return manifest


def import_biokg_public_graph(package, destination, *, ontology: str) -> dict[str, Any]:
    """Normalize one ontology from the public table package; never import alignment links."""
    import csv
    from pathlib import Path

    import yaml

    from exact.core.entities.graph import AnnotationValue, Edge
    from exact.io.sources.csv_kg import CsvKgSource

    package, destination = Path(package), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("BioKG public graph import needs an empty destination")
    with (package / "graph/properties.csv").open(encoding="utf-8", newline="") as stream:
        properties = [row for row in csv.DictReader(stream) if row["ontology"] == ontology]
    with (package / "graph/triples.csv").open(encoding="utf-8", newline="") as stream:
        raw = list(csv.DictReader(stream))
    selected = [
        row
        for row in raw
        if row["head_ontology"] == ontology
        and row["tail_ontology"] == ontology
        and row.get("is_anchor", "false").lower() != "true"
        and row.get("release_layer", "public") == "public"
    ]
    edges = sorted(
        {Edge(row["head_id"], row["relation"], row["tail_id"]) for row in selected},
        key=Edge.astuple,
    )
    if not properties or not edges:
        raise ValueError(
            "Public graph package has no eligible entities/edges for the requested ontology"
        )
    entities = {iri for edge in edges for iri in (edge.src, edge.dst)} | {
        row["node_id"] for row in properties
    }
    kinds = {row["node_id"]: row.get("entity_kind") or "class" for row in properties}
    annotations, labels = [], []
    for row in properties:
        iri = row["node_id"]
        values = [("http://www.w3.org/2000/01/rdf-schema#label", row.get("preferred_label", ""))]
        raw_synonyms = row.get("synonyms", "")
        if raw_synonyms.startswith("["):
            synonyms = json.loads(raw_synonyms)
        else:
            synonyms = raw_synonyms.split("|")
        values.extend(
            ("http://www.geneontology.org/formats/oboInOwl#hasExactSynonym", text)
            for text in synonyms
            if text
        )
        values.append(("http://purl.obolibrary.org/obo/IAO_0000115", row.get("definition", "")))
        for relation, value in values:
            if value:
                annotations.append(
                    json.dumps(
                        {"entity": iri, **asdict(AnnotationValue(relation, value, True))},
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                )
                if not relation.endswith("IAO_0000115"):
                    labels.append((iri, value))
    facts: dict[str, list[Any]] = {
        "entities": sorted((kinds.get(iri, "class"), iri) for iri in entities),
        "labels": sorted(set(labels)),
        "annotations": sorted(set(annotations)),
        "attributes": sorted(
            value
            for value in set(annotations)
            if "IAO_0000115" in json.loads(value)["property_iri"]
        ),
        "hierarchy": sorted(
            (kinds.get(edge.src, "class"), edge.src, edge.dst)
            for edge in edges
            if edge.rel in {"subclass_of", "subClassOf", "http://subclassof"}
        ),
        "domains": [],
        "ranges": [],
        "excluded": [],
        "property_axioms": [],
        "projected_iri_edges": [edge.astuple() for edge in edges],
        "projected_edges": [edge.astuple() for edge in edges],
    }
    with (destination / "triples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["src", "rel", "dst"])
        writer.writerows(facts["projected_edges"])
    with (destination / "evidence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["section", "record"])
        for section, rows in facts.items():
            for row in rows:
                writer.writerow([section, json.dumps(row, sort_keys=True, ensure_ascii=False)])
    (destination / "kg.yaml").write_text(
        yaml.safe_dump(
            {
                "descriptor_version": 1,
                "triples_files": ["triples.csv"],
                "evidence_file": "evidence.csv",
            }
        )
    )
    source = CsvKgSource.from_path(destination)
    manifest = {
        "schema_version": 1,
        "mode": "public_biokg_tables",
        "ontology": ontology,
        "input_files": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in ("graph/triples.csv", "graph/properties.csv")
        },
        "raw_edges": len(raw),
        "retained_edges": len(edges),
        "alignment_anchors_imported": 0,
        "ontology_rows": len(properties),
        "entity_identifier": "node_id",
        "source_versions": sorted({row.get("source_version", "") for row in properties}),
        "evidence_sha256": evidence_inventory(source)["sha256"],
    }
    (destination / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
