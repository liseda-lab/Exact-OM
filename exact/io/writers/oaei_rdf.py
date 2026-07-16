"""OAEI Alignment Format writer and small round-trip reader."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from exact.io.writers._frames import validated_mapping_frame
from exact.io.writers.base import WriterOptionsError

ALIGN_NS = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XSD_FLOAT = "http://www.w3.org/2001/XMLSchema#float"

ET.register_namespace("", ALIGN_NS)
ET.register_namespace("rdf", RDF_NS)


def _tag(local: str) -> str:
    return f"{{{ALIGN_NS}}}{local}"


def _add_ontology(parent: ET.Element, side: str, location: str) -> None:
    container = ET.SubElement(parent, _tag(side))
    ontology = ET.SubElement(
        container,
        _tag("Ontology"),
        {f"{{{RDF_NS}}}about": location},
    )
    ET.SubElement(ontology, _tag("location")).text = location


class OaeiRdfWriter:
    """Write mappings in the OAEI RDF/XML Alignment Format."""

    name = "oaei-rdf"
    default_filename = "align.rdf"

    def write(
        self,
        mappings: Any,
        path: Path,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Path:
        normalized = dict(options or {})
        unknown = sorted(set(normalized) - {"source_uri", "target_uri"})
        if unknown:
            raise WriterOptionsError(f"Unknown OAEI RDF writer option(s): {', '.join(unknown)}")
        source_uri = str(normalized.get("source_uri", "urn:exact:source"))
        target_uri = str(normalized.get("target_uri", "urn:exact:target"))
        if not source_uri or not target_uri:
            raise WriterOptionsError("source_uri and target_uri cannot be empty")
        frame = validated_mapping_frame(mappings)
        relations = frame["Relation"].map(str)
        invalid = sorted(set(relations) - {"=", "<", ">"})
        if invalid:
            raise WriterOptionsError(
                f"OAEI RDF relations must be '=', '<', or '>'; found: {', '.join(invalid)}"
            )

        root = ET.Element(f"{{{RDF_NS}}}RDF")
        alignment = ET.SubElement(root, _tag("Alignment"))
        ET.SubElement(alignment, _tag("xml")).text = "yes"
        ET.SubElement(alignment, _tag("level")).text = "0"
        ET.SubElement(alignment, _tag("type")).text = "??"
        _add_ontology(alignment, "onto1", source_uri)
        _add_ontology(alignment, "onto2", target_uri)
        for row in frame.itertuples(index=False):
            mapping = ET.SubElement(alignment, _tag("map"))
            cell = ET.SubElement(mapping, _tag("Cell"))
            ET.SubElement(
                cell,
                _tag("entity1"),
                {f"{{{RDF_NS}}}resource": str(row.SrcEntity)},
            )
            ET.SubElement(
                cell,
                _tag("entity2"),
                {f"{{{RDF_NS}}}resource": str(row.TgtEntity)},
            )
            measure = ET.SubElement(
                cell,
                _tag("measure"),
                {f"{{{RDF_NS}}}datatype": XSD_FLOAT},
            )
            measure.text = format(float(row.Score), ".17g")
            ET.SubElement(cell, _tag("relation")).text = str(row.Relation)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)
        return Path(path)


def read_alignment(path: Path) -> pd.DataFrame:
    """Read the mapping cells emitted by :class:`OaeiRdfWriter`."""

    root = ET.parse(path).getroot()
    records: list[dict[str, Any]] = []
    for cell in root.findall(f".//{_tag('Cell')}"):
        entity1 = cell.find(_tag("entity1"))
        entity2 = cell.find(_tag("entity2"))
        measure = cell.find(_tag("measure"))
        relation = cell.find(_tag("relation"))
        if entity1 is None or entity2 is None or measure is None:
            raise WriterOptionsError(f"Malformed Alignment Cell in {path}")
        source = entity1.get(f"{{{RDF_NS}}}resource")
        target = entity2.get(f"{{{RDF_NS}}}resource")
        if not source or not target or measure.text is None:
            raise WriterOptionsError(f"Incomplete Alignment Cell in {path}")
        records.append(
            {
                "SrcEntity": source,
                "TgtEntity": target,
                "Score": float(measure.text),
                "Relation": relation.text if relation is not None and relation.text else "=",
            }
        )
    return pd.DataFrame(
        records,
        columns=["SrcEntity", "TgtEntity", "Score", "Relation"],
    )


WRITER = OaeiRdfWriter()

__all__ = [
    "ALIGN_NS",
    "OaeiRdfWriter",
    "RDF_NS",
    "WRITER",
    "read_alignment",
]
