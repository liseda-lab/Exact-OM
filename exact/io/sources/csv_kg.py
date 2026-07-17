"""BioKG-style CSV/TSV knowledge graph source."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency is part of the core lock.
    raise ImportError(
        'CSV-KG descriptors require PyYAML. Reinstall with `pip install "exact-om"`.'
    ) from exc

from exact.core.contracts.knowledge import KnowledgeSource
from exact.core.entities.graph import AnnotationValue, Edge
from exact.core.entities.kinds import EntityKind
from exact.io._common import short_form
from exact.io._hierarchy import HierarchyIndex
from exact.io.sources import SourceOptionsError
from exact.io.sources.datalog import read_facts

DESCRIPTOR_VERSION = 1


def _sequence(value: Any, *, option: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise SourceOptionsError(f"{option} must be a string or sequence of strings")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise SourceOptionsError(f"{option} cannot contain empty values")
    return result


def _safe_relative(value: Any, *, option: str) -> str:
    normalized = str(value).replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise SourceOptionsError(f"{option} must be a safe relative path, got {value!r}")
    return path.as_posix()


def _delimiter(path: str, configured: Any, *, option: str) -> str:
    if configured is None:
        return "\t" if Path(path).suffix.lower() == ".tsv" else ","
    value = str(configured)
    if value == "\\t":
        value = "\t"
    if len(value) != 1:
        raise SourceOptionsError(f"{option} must be one character")
    return value


@dataclass(frozen=True)
class TripleFileSpec:
    """Column and delimiter declaration for one triples table."""

    path: str
    src_col: str = "src"
    rel_col: str = "rel"
    dst_col: str = "dst"
    delimiter: str = ","

    @classmethod
    def parse(cls, value: Any, *, location: str) -> "TripleFileSpec":
        """Parse one compact or mapping-based triples-file declaration."""

        if isinstance(value, str):
            path = _safe_relative(value, option=f"{location}.path")
            return cls(path=path, delimiter=_delimiter(path, None, option=location))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
            if len(value) != 4:
                raise SourceOptionsError(
                    f"{location} tuple form must be [path, src_col, rel_col, dst_col]"
                )
            path = _safe_relative(value[0], option=f"{location}.path")
            return cls(
                path=path,
                src_col=str(value[1]),
                rel_col=str(value[2]),
                dst_col=str(value[3]),
                delimiter=_delimiter(path, None, option=location),
            )
        if not isinstance(value, Mapping):
            raise SourceOptionsError(f"{location} must be a path, tuple, or mapping")
        unknown = sorted(set(value) - {"delimiter", "dst_col", "path", "rel_col", "src_col"})
        if unknown:
            raise SourceOptionsError(f"Unknown key(s) at {location}: {', '.join(unknown)}")
        if "path" not in value:
            raise SourceOptionsError(f"{location}.path is required")
        path = _safe_relative(value["path"], option=f"{location}.path")
        return cls(
            path=path,
            src_col=str(value.get("src_col", "src")),
            rel_col=str(value.get("rel_col", "rel")),
            dst_col=str(value.get("dst_col", "dst")),
            delimiter=_delimiter(path, value.get("delimiter"), option=f"{location}.delimiter"),
        )


@dataclass(frozen=True)
class LabelsFileSpec:
    """Column and delimiter declaration for an entity-label table."""

    path: str
    entity_col: str = "entity"
    label_col: str = "label"
    lang_col: str | None = None
    delimiter: str = ","

    @classmethod
    def parse(cls, value: Any, *, location: str = "labels_file") -> "LabelsFileSpec":
        """Parse a labels-file declaration and normalize its delimiter."""

        if isinstance(value, str):
            path = _safe_relative(value, option=f"{location}.path")
            return cls(path=path, delimiter=_delimiter(path, None, option=location))
        if not isinstance(value, Mapping):
            raise SourceOptionsError(f"{location} must be a path or mapping")
        unknown = sorted(set(value) - {"delimiter", "entity_col", "label_col", "lang_col", "path"})
        if unknown:
            raise SourceOptionsError(f"Unknown key(s) at {location}: {', '.join(unknown)}")
        if "path" not in value:
            raise SourceOptionsError(f"{location}.path is required")
        path = _safe_relative(value["path"], option=f"{location}.path")
        lang_col = value.get("lang_col")
        return cls(
            path=path,
            entity_col=str(value.get("entity_col", "entity")),
            label_col=str(value.get("label_col", "label")),
            lang_col=str(lang_col) if lang_col is not None else None,
            delimiter=_delimiter(path, value.get("delimiter"), option=f"{location}.delimiter"),
        )


@dataclass(frozen=True)
class CsvKgDescriptor:
    """Validated in-memory representation of a ``kg.yaml`` descriptor."""

    triples_files: tuple[TripleFileSpec, ...]
    labels_file: LabelsFileSpec | None = None
    label_relation: str | None = None
    hierarchy_relations: tuple[str, ...] = ()
    attribute_relations: tuple[str, ...] = ()
    datalog_files: tuple[str, ...] = ()
    class_relation: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CsvKgDescriptor":
        """Validate and normalize a raw descriptor mapping."""

        allowed = {
            "attribute_relations",
            "class_relation",
            "datalog_files",
            "description",
            "descriptor_version",
            "hierarchy_relations",
            "label_relation",
            "labels_file",
            "triples_files",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise SourceOptionsError(f"Unknown CSV-KG descriptor key(s): {', '.join(unknown)}")
        version = value.get("descriptor_version", DESCRIPTOR_VERSION)
        if version != DESCRIPTOR_VERSION:
            raise SourceOptionsError(
                f"Unsupported CSV-KG descriptor_version {version!r}; expected {DESCRIPTOR_VERSION}"
            )
        raw_triples = value.get("triples_files")
        if not isinstance(raw_triples, Sequence) or isinstance(
            raw_triples, (str, bytes, bytearray)
        ):
            raise SourceOptionsError("triples_files must be a non-empty sequence")
        triples = tuple(
            TripleFileSpec.parse(item, location=f"triples_files[{index}]")
            for index, item in enumerate(raw_triples)
        )
        if not triples:
            raise SourceOptionsError("triples_files must contain at least one table")
        labels = value.get("labels_file")
        label_relation = value.get("label_relation")
        class_relation = value.get("class_relation")
        return cls(
            triples_files=triples,
            labels_file=LabelsFileSpec.parse(labels) if labels is not None else None,
            label_relation=str(label_relation) if label_relation is not None else None,
            hierarchy_relations=_sequence(
                value.get("hierarchy_relations"), option="hierarchy_relations"
            ),
            attribute_relations=_sequence(
                value.get("attribute_relations"), option="attribute_relations"
            ),
            datalog_files=tuple(
                _safe_relative(item, option="datalog_files")
                for item in _sequence(value.get("datalog_files"), option="datalog_files")
            ),
            class_relation=str(class_relation) if class_relation is not None else None,
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SourceOptionsError(f"Invalid CSV-KG descriptor {path}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise SourceOptionsError(f"CSV-KG descriptor {path} must contain a mapping")
    return {str(key): item for key, item in loaded.items()}


def _resolve(root: Path, relative: str, *, option: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:  # pragma: no cover - _safe_relative rejects this first.
        raise SourceOptionsError(f"{option} escapes the CSV-KG directory") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"CSV-KG file declared by {option} does not exist: {candidate}")
    return candidate


def _column(
    fieldnames: Sequence[str] | None,
    requested: str,
    aliases: Sequence[str],
    *,
    location: str,
) -> str:
    available = tuple(fieldnames or ())
    if requested in available:
        return requested
    for alias in aliases:
        if alias in available:
            return alias
    raise SourceOptionsError(
        f"Missing column {requested!r} in {location}; available columns: "
        f"{', '.join(available) or 'none'}"
    )


class CsvKgSource(KnowledgeSource):
    """An in-memory source for descriptor-driven CSV/TSV knowledge graphs."""

    def __init__(self, root: Path, descriptor: CsvKgDescriptor) -> None:
        self._origin = Path(root)
        self.descriptor = descriptor
        raw_edges: set[Edge] = set()
        for index, triple_spec in enumerate(descriptor.triples_files):
            path = _resolve(
                self._origin,
                triple_spec.path,
                option=f"triples_files[{index}].path",
            )
            with path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream, delimiter=triple_spec.delimiter)
                src_col = _column(
                    reader.fieldnames,
                    triple_spec.src_col,
                    ("subject", "head", "SrcEntity"),
                    location=str(path),
                )
                rel_col = _column(
                    reader.fieldnames,
                    triple_spec.rel_col,
                    ("relation", "predicate"),
                    location=str(path),
                )
                dst_col = _column(
                    reader.fieldnames,
                    triple_spec.dst_col,
                    ("object", "tail", "TgtEntity"),
                    location=str(path),
                )
                for line_number, row in enumerate(reader, start=2):
                    src = str(row.get(src_col, "")).strip()
                    rel = str(row.get(rel_col, "")).strip()
                    dst = str(row.get(dst_col, "")).strip()
                    if not src or not rel or not dst:
                        raise SourceOptionsError(f"Empty triple component in {path}:{line_number}")
                    raw_edges.add(Edge(src, rel, dst))
        for index, relative in enumerate(descriptor.datalog_files):
            path = _resolve(self._origin, relative, option=f"datalog_files[{index}]")
            raw_edges.update(read_facts(path))

        label_relations = (
            frozenset({descriptor.label_relation})
            if descriptor.label_relation is not None
            else frozenset()
        )
        attribute_relations = frozenset(descriptor.attribute_relations)
        literal_relations = label_relations | attribute_relations
        hierarchy_relations = frozenset(descriptor.hierarchy_relations)

        annotations: dict[str, set[AnnotationValue]] = defaultdict(set)
        labels: dict[str, set[str]] = defaultdict(set)
        entities: set[str] = set()
        relations: set[str] = set()
        relation_targets: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        hierarchy_edges: set[tuple[str, str]] = set()
        iri_edges: set[Edge] = set()
        literal_edges: set[Edge] = set()
        class_entities: set[str] = set()

        for edge in raw_edges:
            entities.add(edge.src)
            relations.add(edge.rel)
            if edge.rel in literal_relations:
                value = AnnotationValue(edge.rel, edge.dst, True)
                annotations[edge.src].add(value)
                literal_edges.add(edge)
                if edge.rel in label_relations:
                    labels[edge.src].add(edge.dst)
                continue
            entities.add(edge.dst)
            iri_edges.add(edge)
            relation_targets[edge.rel][edge.src].add(edge.dst)
            if edge.rel in hierarchy_relations:
                hierarchy_edges.add((edge.src, edge.dst))
            if descriptor.class_relation is not None and edge.rel == descriptor.class_relation:
                class_entities.update((edge.src, edge.dst))

        label_property = descriptor.label_relation or "label"
        if descriptor.labels_file is not None:
            label_spec = descriptor.labels_file
            path = _resolve(self._origin, label_spec.path, option="labels_file.path")
            with path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream, delimiter=label_spec.delimiter)
                entity_col = _column(
                    reader.fieldnames,
                    label_spec.entity_col,
                    ("id", "iri", "entity_id"),
                    location=str(path),
                )
                label_col = _column(
                    reader.fieldnames,
                    label_spec.label_col,
                    ("name", "text"),
                    location=str(path),
                )
                if label_spec.lang_col is not None:
                    lang_col = _column(
                        reader.fieldnames,
                        label_spec.lang_col,
                        (),
                        location=str(path),
                    )
                else:
                    lang_col = None
                for line_number, row in enumerate(reader, start=2):
                    iri = str(row.get(entity_col, "")).strip()
                    label = str(row.get(label_col, "")).strip()
                    if not iri or not label:
                        raise SourceOptionsError(f"Empty label value in {path}:{line_number}")
                    lang = str(row.get(lang_col, "")).strip() if lang_col else None
                    entities.add(iri)
                    labels[iri].add(label)
                    annotations[iri].add(
                        AnnotationValue(label_property, label, True, lang=lang or None)
                    )

        self._signature: dict[EntityKind, tuple[str, ...]] = {
            EntityKind.CLASS: tuple(sorted(class_entities)),
            EntityKind.OBJECT_PROPERTY: tuple(sorted(relations - literal_relations)),
            EntityKind.DATA_PROPERTY: tuple(sorted(attribute_relations)),
            EntityKind.ANNOTATION_PROPERTY: tuple(
                sorted(
                    label_relations
                    | ({label_property} if descriptor.labels_file is not None else set())
                )
            ),
            EntityKind.INDIVIDUAL: tuple(sorted(entities - class_entities)),
        }
        self.hierarchy = HierarchyIndex(entities, hierarchy_edges, filter_owl_bounds=False)
        self._labels = {iri: tuple(sorted(values)) for iri, values in labels.items()}
        self._annotations = {
            iri: tuple(
                sorted(
                    values,
                    key=lambda value: (
                        value.property_iri,
                        value.value,
                        value.lang or "",
                    ),
                )
            )
            for iri, values in annotations.items()
        }
        self._attributes = {
            iri: tuple(
                value
                for value in values
                if value.property_iri not in label_relations
                and value.property_iri != label_property
            )
            for iri, values in self._annotations.items()
        }
        self._relation_targets = {
            relation: {subject: tuple(sorted(targets)) for subject, targets in by_subject.items()}
            for relation, by_subject in relation_targets.items()
        }
        self._iri_edges = tuple(sorted(iri_edges, key=Edge.astuple))
        self._literal_edges = tuple(sorted(literal_edges, key=Edge.astuple))

    @classmethod
    def from_path(cls, path: Path, *, options: Mapping[str, Any] | None = None) -> "CsvKgSource":
        """Load a file/directory and merge ``kg.yaml`` with explicit options."""

        source_path = Path(path)
        if source_path.is_dir():
            root = source_path
            descriptor_path = root / "kg.yaml"
            raw = _read_yaml(descriptor_path) if descriptor_path.is_file() else {}
        elif source_path.is_file():
            root = source_path.parent
            raw = {"triples_files": [source_path.name]}
        else:
            raise FileNotFoundError(f"CSV-KG source does not exist: {source_path}")
        raw.update(dict(options or {}))
        if "triples_files" not in raw:
            raise SourceOptionsError(
                f"CSV-KG directory {root} needs kg.yaml or source_options.triples_files"
            )
        return cls(root, CsvKgDescriptor.from_mapping(raw))

    @property
    def origin(self) -> Path | None:
        return self._origin

    def entities(self, kind: EntityKind = EntityKind.CLASS) -> Sequence[str]:
        """Return the indexed identifiers for ``kind``."""

        try:
            return self._signature[EntityKind(kind)]
        except ValueError as exc:
            raise ValueError(f"Unknown entity kind: {kind!r}") from exc

    def labels(self, iri: str) -> list[str]:
        """Return labels loaded for ``iri`` in stable order."""

        return list(self._labels.get(str(iri), ()))

    def annotations(
        self, iri: str, properties: Sequence[str] | None = None
    ) -> list[AnnotationValue]:
        """Return annotations, optionally filtered by relation identifier."""

        values = self._annotations.get(str(iri), ())
        if properties is None:
            return list(values)
        selected = frozenset(str(item) for item in properties)
        return [value for value in values if value.property_iri in selected]

    def attributes(self, iri: str) -> list[AnnotationValue]:
        """Return non-label literal values for ``iri``."""

        return list(self._attributes.get(str(iri), ()))

    def direct_parents(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        """Return direct hierarchy parents for class-like entities."""

        if EntityKind(kind) in {EntityKind.CLASS, EntityKind.INDIVIDUAL}:
            return self.hierarchy.direct_parents(str(iri))
        return []

    def direct_children(self, iri: str, kind: EntityKind = EntityKind.CLASS) -> list[str]:
        """Return direct hierarchy children for class-like entities."""

        if EntityKind(kind) in {EntityKind.CLASS, EntityKind.INDIVIDUAL}:
            return self.hierarchy.direct_children(str(iri))
        return []

    def hierarchy_bundle(
        self, iri: str, families: Mapping[str, Sequence[str]]
    ) -> dict[str, list[str]]:
        """Collect configured hierarchy-relation targets by family."""

        result: dict[str, list[str]] = {}
        for family, relations in families.items():
            if family == "is_a":
                result[family] = self.direct_parents(iri)
                continue
            targets: set[str] = set()
            for relation in relations:
                targets.update(self._relation_targets.get(str(relation), {}).get(str(iri), ()))
            result[family] = sorted(targets)
        return result

    def projection_edges(
        self, *, method: str = "owl2vecstar", include_literals: bool = False
    ) -> list[Edge]:
        """Return graph edges, optionally including literal-valued rows."""

        del method
        edges = self._iri_edges + (self._literal_edges if include_literals else ())
        return sorted(edges, key=Edge.astuple)

    def property_domains(self, prop_iri: str) -> list[str]:
        """Return no domains because CSV descriptors do not declare them."""

        del prop_iri
        return []

    def property_ranges(self, prop_iri: str) -> list[str]:
        """Return no ranges because CSV descriptors do not declare them."""

        del prop_iri
        return []

    def excluded_from_alignment(self) -> frozenset[str]:
        """Return identifiers excluded by source metadata, if any."""

        return frozenset()

    def short_form(self, iri: str) -> str:
        """Return a compact display form for ``iri``."""

        return short_form(iri)

    def ancestors(self, iri: str) -> set[str]:
        """Return all configured hierarchy ancestors."""

        return self.hierarchy.ancestors(str(iri))

    def descendants(self, iri: str) -> set[str]:
        """Return all configured hierarchy descendants."""

        return self.hierarchy.descendants(str(iri))


def create_source(path: Path, *, options: Mapping[str, Any] | None = None) -> CsvKgSource:
    """Create a descriptor-driven CSV-KG source."""

    return CsvKgSource.from_path(path, options=options)


__all__ = [
    "CsvKgDescriptor",
    "CsvKgSource",
    "LabelsFileSpec",
    "TripleFileSpec",
    "create_source",
]
