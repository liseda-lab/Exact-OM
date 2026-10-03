"""E23-only OpenEA preparation: public graphs, official train/valid split, no test labels."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from exact.experiments.inputs import nested_sources
from exact.utils.provenance import sha256_file, sha256_path

ARCHIVE_URL = "https://ndownloader.figshare.com/files/34234391"
ARCHIVE_SHA256 = "37adcaf4a7ed33530a39b637da7329e891456bfc89557d8b1ac307c67eb8f5bc"
ARCHIVE_BYTES = 248800649
ARTICLE_URL = "https://figshare.com/articles/dataset/OpenEA_dataset_v1_1/19258760/3"
TASK = "D_W_15K_V1"
SPLIT = "721_5fold/1"
CASE_ID = "K0_OpenEA"
PREFIX = f"OpenEA_dataset_v2.0/{TASK}/"
PUBLIC_FILES = (
    "rel_triples_1",
    "rel_triples_2",
    "attr_triples_1",
    "attr_triples_2",
    f"{SPLIT}/train_links",
    f"{SPLIT}/valid_links",
)
OFFICIAL_COUNTS = (15000, 3000, 1500)
LABEL_PROPERTIES = frozenset(
    [
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://www.w3.org/2004/02/skos/core#prefLabel",
        "http://www.w3.org/2004/02/skos/core#altLabel",
        "http://xmlns.com/foaf/0.1/name",
        "http://xmlns.com/foaf/0.1/givenName",
        *(
            "http://dbpedia.org/ontology/" + name
            for name in ("name", "birthName", "longName", "otherName", "formerName", "teamName")
        ),
    ]
)


def download_archive(cache: Path) -> Path:
    """Download the pinned official archive once; keep the complete archive node-local."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "OpenEA_dataset_v2.0.zip"
    if path.exists():
        if sha256_file(path) != ARCHIVE_SHA256:
            raise ValueError("Existing OpenEA archive differs from its pinned SHA256")
        return path
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            digest, count = hashlib.sha256(), 0
            with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > ARCHIVE_BYTES:
                        raise ValueError("OpenEA download exceeds its pinned size")
                    digest.update(block)
                    stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
            if count != ARCHIVE_BYTES or digest.hexdigest() != ARCHIVE_SHA256:
                raise ValueError("OpenEA download differs from its pinned identity")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def _rows(path: Path, columns: int, *, allow_empty_literal: bool = False):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            values = line.rstrip("\r\n").split("\t", columns - 1)
            required = values[:-1] if allow_empty_literal else values
            if len(values) != columns or any(not value.strip() for value in required):
                raise ValueError(f"Invalid OpenEA row: {path.name}:{number}")
            yield tuple(values)


def _links(path: Path) -> dict[str, str]:
    rows = list(_rows(path, 2))
    links = dict(rows)
    if not rows or len(links) != len(rows) or len(set(links.values())) != len(rows):
        raise ValueError("Official OpenEA training/validation links must each be bijective")
    return links


def label_training_candidates(
    frame: pd.DataFrame, train_links: Path, *, expected_sha256: str
) -> pd.DataFrame:
    """Label an existing retrieval pool only within verified official training endpoints.

    Cross-pairs are negatives under OpenEA's bijective benchmark alignment, not
    verified real-world non-equivalence. Unlisted endpoints remain unknown.
    """
    if sha256_file(Path(train_links)) != expected_sha256:
        raise ValueError("OpenEA training-link identity changed")
    if not {"Src", "Tgt"} <= set(frame) or frame[["Src", "Tgt"]].isna().any().any():
        raise ValueError("Training candidates require nonempty Src/Tgt columns")
    links = _links(Path(train_links))
    targets = set(links.values())
    labels = [
        int(links[source] == target) if source in links and target in targets else float("nan")
        for source, target in zip(frame.Src.astype(str), frame.Tgt.astype(str))
    ]
    result = frame.copy()
    # Ordinary NaN matches read_table/CSV and the existing fitter's unknown-label
    # semantics; nullable pd.NA would make its scalar comparisons ambiguous.
    result["confirmed_label"] = labels
    return result


def _table(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t" if path.suffix == ".tsv" else ",")
        writer.writerow(columns)
        writer.writerows(rows)


def _graph(raw: Path, destination: Path, side: int) -> dict[str, Any]:
    edges = list(_rows(raw / f"rel_triples_{side}", 3))
    original_attributes = list(_rows(raw / f"attr_triples_{side}", 3, allow_empty_literal=True))
    # The original bytes remain in official/. Empty values carry no lexical
    # evidence and cannot be represented as nonempty CSV-KG literal edges.
    attributes = [row for row in original_attributes if row[2].strip()]
    entities = {value for source, _, target in edges for value in (source, target)}
    entities.update(source for source, _, _ in original_attributes)
    predicates = {predicate for _, predicate, _ in attributes}
    if predicates & {predicate for _, predicate, _ in edges}:
        raise ValueError("OpenEA predicate used as both relation and attribute")
    labels = sorted({(s, v) for s, p, v in attributes if p in LABEL_PROPERTIES})
    _table(destination / "relations.csv", ["src", "rel", "dst"], edges)
    _table(destination / "attributes.csv", ["src", "rel", "dst"], attributes)
    _table(
        destination / "entities.csv",
        ["entity", "kind"],
        ((s, "individual") for s in sorted(entities)),
    )
    _table(destination / "labels.csv", ["entity", "label"], labels)
    # JSON is valid YAML. Names are observed attributes; encoded IDs are not decoded.
    descriptor = {
        "descriptor_version": 1,
        "description": "OpenEA v2.0 public within-KG triples; no alignment links inserted.",
        "entities_file": "entities.csv",
        "labels_file": "labels.csv",
        "triples_files": ["relations.csv", "attributes.csv"],
        "attribute_relations": sorted(predicates),
        "hierarchy_relations": [],
    }
    (destination / "kg.yaml").write_text(json.dumps(descriptor, indent=2) + "\n")
    return {
        "entities": entities,
        "relations": len(edges),
        "attributes": len(attributes),
        "empty_attributes_omitted": len(original_attributes) - len(attributes),
        "labeled_entities": len({s for s, _ in labels}),
    }


def verify_prepared(destination: Path) -> dict[str, Any]:
    """Verify every published preparation artifact before reuse."""
    destination = Path(destination)
    manifest: dict[str, Any] = json.loads((destination / "preparation.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256_file(destination / name) != digest:
            raise ValueError(f"Prepared OpenEA input changed: {name}")
    return manifest


def prepare_openea(
    archive: Path,
    destination: Path,
    *,
    training_cap: int = 2000,
    source_cap: int = 300,
    seed: int = 17,
) -> dict[str, Any]:
    """Publish a separate E23 case; never open test_links, ent_links or other folds."""
    archive, destination = Path(archive), Path(destination).resolve()
    if training_cap < 2 or source_cap < 1:
        raise ValueError(
            "OpenEA preparation requires at least two training and one validation source"
        )
    recipe = {
        "dataset": "OpenEA_v2.0",
        "task": TASK,
        "split": SPLIT,
        "training_cap": training_cap,
        "source_cap": source_cap,
        "seed": seed,
        "schema_version": 1,
    }
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("OpenEA archive differs from the approved v2.0 release")
    if destination.exists():
        manifest = verify_prepared(destination)
        if manifest["recipe"] != recipe or manifest["archive"]["sha256"] != ARCHIVE_SHA256:
            raise ValueError("Immutable OpenEA preparation recipe changed")
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".openea-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        raw = stage / "official"
        with zipfile.ZipFile(archive) as package:
            names = package.namelist()
            for relative in PUBLIC_FILES:
                member = PREFIX + relative
                if names.count(member) != 1:
                    raise ValueError(f"Missing or duplicated approved archive member: {relative}")
                path = raw / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with package.open(member) as src, path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        graphs = [
            _graph(raw, stage / "views" / name, side)
            for side, name in [(1, "source"), (2, "target")]
        ]
        train_path, valid_path = raw / SPLIT / "train_links", raw / SPLIT / "valid_links"
        train, valid = _links(train_path), _links(valid_path)
        if set(train) & set(valid) or set(train.values()) & set(valid.values()):
            raise ValueError("OpenEA train/validation endpoints overlap")
        if graphs[0]["entities"] & graphs[1]["entities"]:
            raise ValueError("OpenEA within-KG views have cross-graph entity overlap")
        if any(
            not set(links) <= graphs[0]["entities"]
            or not set(links.values()) <= graphs[1]["entities"]
            for links in (train, valid)
        ):
            raise ValueError("Alignment endpoint absent from public KG")
        entities_count, train_count, valid_count = OFFICIAL_COUNTS
        if (
            [len(g["entities"]) for g in graphs] != [entities_count, entities_count]
            or len(train) != train_count
            or len(valid) != valid_count
        ):
            raise ValueError("OpenEA official population counts differ from the pinned case")
        training = nested_sources(train, training_cap, seed=seed)
        validation = nested_sources(valid, None, seed=seed)
        for role, links in [("train", train), ("valid", valid)]:
            _table(
                stage / f"{role}.reference.tsv",
                ["SrcEntity", "TgtEntity", "Relation", "Score"],
                ((s, t, "=", 1.0) for s, t in sorted(links.items())),
            )
        for role, values in [("train", training), ("valid", validation)]:
            (stage / f"{role}.sources.txt").write_text("\n".join(values) + "\n")

        def binding(relative: str) -> dict[str, str]:
            return {"path": str(destination / relative), "sha256": sha256_path(stage / relative)}

        scope = {
            "kind": "openea_bijective_training_endpoints",
            "official_train_links": binding(f"official/{SPLIT}/train_links"),
            "training_sources": binding("train.sources.txt"),
            "training_reference": binding("train.reference.tsv"),
            "validation_sources": binding("valid.sources.txt"),
            "negative_scope": "cross_pairs_between_distinct_official_training_matches_only",
            "unknown_scope": "any_source_or_target_outside_official_training_endpoints",
            "benchmark_assumption": "Official OpenEA alignment is bijective; negatives are benchmark-scoped, not ontology-wide NIL or independently verified real-world non-equivalence.",
        }
        (stage / "training-label-scope.json").write_text(json.dumps(scope, indent=2) + "\n")
        case = {
            "task": "openea-v2-D_W_15K_V1-fold1",
            "kind": "individual",
            "role": "development",
            "source": binding("views/source"),
            "target": binding("views/target"),
            "source_universe": binding("valid.sources.txt"),
            "references": {role: binding(f"{role}.reference.tsv") for role in ("train", "valid")},
            "candidates": {},
            "frozen_global_candidates": {},
            "capabilities": [
                "instance_equivalence",
                "literal_evidence",
                "relation_evidence",
                "csv_graph",
                "train_reference",
                "scoped_bijective_training_negatives",
            ],
            "reference_completeness": "known_incomplete",
            "negative_policy": "confirmed_only",
            "selection_reason": "Prospective E23-only graph-rich, TBox-poor public benchmark with official disjoint train/validation and bijective training endpoints; no outcomes inspected.",
            "transformation": "Public relation/attribute CSV conversion; empty literal rows omitted and counted, original bytes retained. Explicit name attributes only; no URI decoding, alignment graph edges or new labels. Existing retrieval must generate and freeze separate pools for this case; the verified official train-endpoint scope labels only that generated training pool.",
            "overlay": {"io": {"input_format": "csv-kg"}},
        }
        bindings = {
            "cases": {CASE_ID: case},
            "e23_natural_case": CASE_ID,
            "e23_training_label_scope": binding("training-label-scope.json"),
        }
        (stage / "bindings-fragment.json").write_text(json.dumps(bindings, indent=2) + "\n")
        manifest = {
            "recipe": recipe,
            "archive": {
                "url": ARCHIVE_URL,
                "article": ARTICLE_URL,
                "sha256": ARCHIVE_SHA256,
                "bytes": archive.stat().st_size,
            },
            "opened_archive_members": [PREFIX + name for name in PUBLIC_FILES],
            "heldout_alignment_contents_read": False,
            "graphs": [
                {
                    **{k: v for k, v in graph.items() if k != "entities"},
                    "entities": len(graph["entities"]),
                }
                for graph in graphs
            ],
            "train_links": len(train),
            "valid_links": len(valid),
            "selected_training_sources": len(training),
            "training_label_scope": scope,
            "files": {
                str(path.relative_to(stage)): sha256_file(path)
                for path in sorted(stage.rglob("*"))
                if path.is_file()
            },
        }
        (stage / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(destination)
    return manifest
