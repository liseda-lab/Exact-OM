"""Small registered transforms for declarative dataset descriptors."""

from __future__ import annotations

import ast
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .provider import DescriptorError, IntegrityError


@dataclass(frozen=True)
class TransformResult:
    """Materialized path and flags emitted by a descriptor transform."""

    path: Path
    extras: Mapping[str, Any] = field(default_factory=dict)


Transform = Callable[[Path, Path], TransformResult]
_TRANSFORMS: dict[str, Transform] = {}


def register_transform(name: str, transform: Transform, *, replace: bool = False) -> None:
    """Register a named descriptor transform."""

    if not replace and name in _TRANSFORMS:
        raise ValueError(f"Track transform {name!r} is already registered")
    _TRANSFORMS[name] = transform


def transform_names() -> tuple[str, ...]:
    """Return registered transform names in deterministic order."""

    return tuple(sorted(_TRANSFORMS))


def apply_transform(name: str, source: Path, destination: Path) -> TransformResult:
    """Apply a registered transform or fail with an actionable descriptor error."""

    try:
        transform = _TRANSFORMS[name]
    except KeyError as exc:
        available = ", ".join(transform_names()) or "none"
        raise DescriptorError(f"Unknown track transform {name!r}; available: {available}") from exc
    return transform(Path(source), Path(destination))


def read_alignment(
    path: Path,
) -> tuple[str | None, str | None, Sequence[tuple[str, str, str]]]:
    """Parse an OAEI Alignment RDF/XML file.

    Optional ontology locations and every complete ``Cell`` are returned. Some
    official tracks (including Anatomy) omit the location metadata. Malformed
    cells are ignored to preserve the historic retrieval-script behavior.
    """

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise IntegrityError(f"Could not parse alignment RDF {path}: {exc}") from exc

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    locations = [
        element.text.strip()
        for element in root.iter()
        if local_name(element.tag) == "location" and element.text and element.text.strip()
    ]
    cells: list[tuple[str, str, str]] = []
    for cell in (element for element in root.iter() if local_name(element.tag) == "Cell"):
        children = {local_name(child.tag): child for child in cell}
        entity1 = children.get("entity1")
        entity2 = children.get("entity2")
        measure = children.get("measure")
        if entity1 is None or entity2 is None or measure is None or measure.text is None:
            continue
        source = next(
            (value for key, value in entity1.attrib.items() if local_name(key) == "resource"), None
        )
        target = next(
            (value for key, value in entity2.attrib.items() if local_name(key) == "resource"), None
        )
        if source and target:
            cells.append((source, target, measure.text.strip()))
    source_location = locations[0] if locations else None
    target_location = locations[1] if len(locations) > 1 else None
    return source_location, target_location, cells


def alignment_rdf_to_tsv(source: Path, destination: Path) -> TransformResult:
    """Convert OAEI Alignment RDF/XML cells to Exact's three-column TSV."""

    _, _, cells = read_alignment(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["SrcEntity", "TgtEntity", "Score"])
        writer.writerows(cells)
    return TransformResult(destination)


def _first(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _candidate_identifier(value: Any) -> tuple[str | None, bool, bool]:
    if value is None:
        return None, True, False
    if isinstance(value, str):
        text = value.strip()
        is_nil = text.lower() in {"nil", "none", "null", "__nil__"} or text.endswith("/NIL")
        return text, is_nil, False
    if isinstance(value, Mapping):
        identifier = _first(value, ("iri", "id", "entity", "target", "candidate", "value"))
        is_nil = bool(value.get("nil", False))
        is_gold = bool(value.get("gold", value.get("correct", value.get("is_reference", False))))
        mapping_text = str(identifier).strip() if identifier is not None else None
        if mapping_text and (
            mapping_text.lower() in {"nil", "none", "null", "__nil__"}
            or mapping_text.endswith("/NIL")
        ):
            is_nil = True
        return mapping_text, is_nil, is_gold
    return str(value), False, False


def _candidate_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [value]
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    raise IntegrityError("Candidate pool must be a JSON list (or a serialized list/tuple)")


def pools_jsonl_to_cands_tsv(source: Path, destination: Path) -> TransformResult:
    """Convert DISO pool records to candidate TSV, dropping the NIL option."""

    rows: list[tuple[str, str, str]] = []
    nil_seen = False
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(record, Mapping):
                raise IntegrityError(f"Expected an object at {source}:{line_number}")
            source_id = _first(
                record, ("SrcEntity", "src", "source", "source_id", "query", "query_id")
            )
            target_id = _first(
                record, ("TgtEntity", "tgt", "target", "target_id", "reference", "gold")
            )
            raw_candidates = _first(record, ("TgtCandidates", "candidates", "pool", "targets"))
            if isinstance(source_id, Mapping):
                source_id = _first(source_id, ("iri", "id", "entity", "value"))
            if source_id is None or raw_candidates is None:
                raise IntegrityError(
                    f"Pool record at {source}:{line_number} needs a source and candidates"
                )
            candidates: list[str] = []
            inferred_target: str | None = None
            for raw_candidate in _candidate_values(raw_candidates):
                identifier, is_nil, is_gold = _candidate_identifier(raw_candidate)
                if is_nil:
                    nil_seen = True
                    continue
                if identifier is not None:
                    candidates.append(identifier)
                    if is_gold:
                        inferred_target = identifier
            if target_id is None:
                target_id = inferred_target or ""
            rows.append((str(source_id), str(target_id), repr(tuple(candidates))))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["SrcEntity", "TgtEntity", "TgtCandidates"])
        writer.writerows(rows)
    return TransformResult(destination, {"nil": nil_seen, "nil_candidates": nil_seen})


def flatten_refs_equiv(source: Path, destination: Path) -> TransformResult:
    """Copy the contents of ``refs_equiv`` into a flat destination directory."""

    if not source.is_dir():
        raise IntegrityError(f"flatten_refs_equiv expects a directory, got {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.rglob("*")):
        if child.is_symlink():
            raise IntegrityError(f"Reference directory contains a symbolic link: {child}")
        if child.is_file():
            relative = child.relative_to(source)
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, output)
    return TransformResult(destination)


register_transform("alignment_rdf_to_tsv", alignment_rdf_to_tsv)
register_transform("pools_jsonl_to_cands_tsv", pools_jsonl_to_cands_tsv)
register_transform("flatten_refs_equiv", flatten_refs_equiv)
