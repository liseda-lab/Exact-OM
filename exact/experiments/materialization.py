"""Native, independently sourced BioKG enrichment; Python only prepares/serializes facts.

Computes native closure of the raw graph with the legacy rule program and empty
auxiliary property relations; this is neither full legacy nor unrestricted OWL reasoning. Public
alignment anchors are excluded. Every derived tuple binds the exact native program,
input inventory and executable; this is reproducible lineage, not a minimal proof tree.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import yaml

from exact.experiments.evidence_inventory import evidence_inventory
from exact.io.sources.csv_kg import CsvKgSource
from exact.utils.provenance import sha256_file

_ARITIES = {
    "subclass": 2,
    "equiv": 2,
    "edge": 3,
    "subprop": 2,
    "inverse": 2,
    "transitive": 1,
    "chain2": 3,
    "domain": 2,
    "range": 2,
}
_FACT = re.compile(r"([a-z_][a-z_0-9]*)\((.*)\)\.")
_OUTPUTS = {"inferred_subclass": 2, "inferred_edge": 3}
_RESULT_FIELDS = {
    "status",
    "identity",
    "derived_edges",
    "added_edges",
    "enriched_evidence_sha256",
    "native_receipt",
    "consequences",
}


def _request_identity(manifest):
    request = {key: value for key, value in manifest.items() if key not in _RESULT_FIELDS}
    return hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _binding(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _fact(line, number):
    """Parse quoted native-export terms losslessly; never evaluate Python or rules."""
    match = _FACT.fullmatch(line.strip())
    if not match or match[1] not in _ARITIES:
        raise ValueError(f"Unsupported legacy fact at line {number}")
    args = json.loads("[" + match[2] + "]")
    if len(args) != _ARITIES[match[1]] or any(not isinstance(v, str) for v in args):
        raise ValueError(f"Invalid legacy fact arguments at line {number}")
    return match[1], tuple(args)


def _serialize(predicate, args):
    predicate = "ontology_range" if predicate == "range" else predicate
    return predicate + "(" + ", ".join(json.dumps(v, ensure_ascii=False) for v in args) + ").\n"


def prepare_legacy_materialization(package, raw, destination, *, ontology, executable, jobs=2):
    """Freeze a native executable program from one anchor-free public ontology view.

    Neither private paths nor reference files are inputs. Node/edge facts are reconstructed
    from the same public triples as the raw view. Legacy auxiliary property axioms lack
    ontology provenance, so their declared relations remain empty. No auxiliary shortcuts
    or cross-ontology anchors enter this explicitly bounded structural closure.
    """
    package, raw, destination = map(Path, (package, raw, destination))
    engine = Path(shutil.which(str(executable)) or executable).resolve()
    if not engine.is_file() or not 1 <= jobs <= 2:
        raise ValueError("A native Souffle executable and one or two CPU workers are required")
    destination.mkdir(parents=True, exist_ok=False)
    raw_source = CsvKgSource.from_path(raw)
    inventory = evidence_inventory(raw_source)
    conversion = json.loads((raw / "conversion.json").read_text())
    if conversion.get("mode") != "public_biokg_tables" or conversion["ontology"] != ontology:
        raise ValueError("Expected the anchor-free public BioKG conversion for this ontology")
    for name, digest in conversion["input_files"].items():
        if sha256_file(package / name) != digest:
            raise ValueError("Raw view and materialization must use the same release")
    if conversion["evidence_sha256"] != inventory["sha256"]:
        raise ValueError("Raw evidence changed after conversion")
    rules_path = package / "graph/rules.dl"
    rules = rules_path.read_text()
    # No external includes, input/output directives or functors may introduce hidden evidence.
    declarations = {}
    for line in rules.splitlines():
        value = line.strip()
        if not value or value.startswith("//"):
            continue
        if value.startswith(".decl "):
            match = re.fullmatch(r"\.decl (\w+)\(([^()]*)\)", value)
            if not match or any(
                not re.fullmatch(r"\w+\s*:\s*symbol", v.strip()) for v in match[2].split(",")
            ):
                raise ValueError("Only the declared legacy symbol-valued fragment is supported")
            declarations[match[1]] = len(match[2].split(","))
        elif (
            value.startswith(".")
            or any(token in value for token in ("@", "!", "#"))
            or ":-" not in value
        ):
            raise ValueError("Legacy rules must be self-contained positive Datalog")
    if declarations != {**_ARITIES, **_OUTPUTS}:
        raise ValueError("Unsupported rule schema; bind and implement the new fragment explicitly")
    edges = {tuple(row) for row in inventory["facts"]["projected_edges"]}
    facts = set()
    for head, relation, tail in edges:
        if relation == "subclass_of":
            facts.add(("subclass", (head, tail)))
        elif relation == "equivalent_class":
            facts.add(("equiv", (head, tail)))
        else:
            facts.add(("edge", (head, relation, tail)))
    excluded = Counter()
    facts_path = package / "graph/facts.dl"
    # Counts are an exclusion inventory, not inputs to materialization. In
    # particular, equiv contains public reference-derived alignment anchors and
    # the auxiliary property names have no per-ontology ownership evidence.
    with facts_path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            name, _ = _fact(line, number)
            excluded[name] += 1
    program = destination / "program.dl"
    with program.open("w") as stream:
        # Souffle 2.5 reserves range as a functor; rename the predicate only.
        stream.write(re.sub(r"\brange(?=\s*\()", "ontology_range", rules).rstrip() + "\n")
        for name, args in sorted(facts):
            stream.write(_serialize(name, args))
        for name in _OUTPUTS:
            stream.write(f".output {name}\n")
    version = subprocess.run(
        [str(engine), "--version"], check=True, text=True, capture_output=True
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "kind": "native_legacy_datalog",
        "status": "prepared",
        "ontology": ontology,
        "raw": _binding(raw / "conversion.json"),
        "raw_evidence_sha256": inventory["sha256"],
        "raw_files": {
            name: _binding(raw / name) for name in ("kg.yaml", "triples.csv", "evidence.csv")
        },
        "rules": _binding(rules_path),
        "released_facts": _binding(facts_path),
        "program": _binding(program),
        "engine": {**_binding(engine), "version": version},
        "jobs": jobs,
        "implementation": _binding(Path(__file__)),
        "input_fact_count": len(facts),
        "retained_auxiliary_facts": 0,
        "excluded_released_facts": sum(excluded.values()),
        "excluded_released_facts_by_predicate": dict(sorted(excluded.items())),
        "semantic_fragment": "raw_graph_closure_with_empty_auxiliary_relations",
        "full_legacy_materialization": False,
        "alignment_anchors_imported": 0,
        "term_encoding": "public_node_id_and_predicate_identity",
        "predicate_renames": {"range": "ontology_range"},
        "input_role": "public_ontology_facts_only",
        "provenance_scope": "rule_program_and_input_inventory",
    }
    manifest["identity"] = _request_identity(manifest)
    _write_json(destination / "materialization.json", manifest)
    return manifest


def verify_legacy_materialization(
    destination, *, raw=None, package=None, ontology=None, executable=None
):
    """Verify a frozen request, including CLI reuse, without running native closure."""
    destination = Path(destination)
    manifest = json.loads((destination / "materialization.json").read_text())
    if manifest.get("identity") != _request_identity(manifest):
        raise ValueError("Materialization request identity changed")
    if manifest.get("status") not in {"prepared", "complete"}:
        raise ValueError("Unknown materialization status")
    if (
        manifest.get("input_role") != "public_ontology_facts_only"
        or manifest.get("semantic_fragment") != "raw_graph_closure_with_empty_auxiliary_relations"
        or manifest.get("alignment_anchors_imported") != 0
        or manifest.get("retained_auxiliary_facts") != 0
    ):
        raise ValueError("Unsupported materialization evidence contract")
    for key in ("engine", "program", "raw", "rules", "released_facts", "implementation"):
        item = manifest[key]
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("Materialization input changed: " + key)
    expected_package = Path(manifest["rules"]["path"]).parent.parent.resolve()
    requested = {
        "ontology": (ontology, manifest["ontology"]),
        "package": (Path(package).resolve() if package is not None else None, expected_package),
        "raw": (
            Path(raw).resolve() if raw is not None else None,
            Path(manifest["raw"]["path"]).parent.resolve(),
        ),
        "engine": (
            (
                Path(shutil.which(str(executable)) or executable).resolve()
                if executable is not None
                else None
            ),
            Path(manifest["engine"]["path"]).resolve(),
        ),
    }
    for name, (current, frozen) in requested.items():
        if current is not None and current != frozen:
            raise ValueError(
                "Requested materialization " + name + " changed; use a new output directory"
            )
    raw_files = manifest.get("raw_files", {})
    if set(raw_files) != {"kg.yaml", "triples.csv", "evidence.csv"}:
        raise ValueError("Materialization lacks complete frozen raw-file bindings")
    for name, item in raw_files.items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("Materialization raw input changed: " + name)
    conversion = json.loads(Path(manifest["raw"]["path"]).read_text())
    for name, digest in conversion["input_files"].items():
        if sha256_file(expected_package / name) != digest:
            raise ValueError("Materialization package input changed: " + name)
    return manifest


def materialize_legacy(raw, destination):
    """Run native closure, then serialize consequences without changing base evidence.

    Native outputs are a durable boundary. Re-entry verifies them and only repeats the
    cheap conversion; an incomplete native run is rerun, never accepted as a partial closure.
    """
    raw, destination = map(Path, (raw, destination))
    path = destination / "materialization.json"
    manifest = verify_legacy_materialization(destination, raw=raw)
    source = CsvKgSource.from_path(raw)
    inventory = evidence_inventory(source)
    if inventory["sha256"] != manifest["raw_evidence_sha256"]:
        raise ValueError("Raw evidence changed")
    native = destination / "native"
    receipt = destination / "native-complete.json"
    native.mkdir(exist_ok=True)
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if (
            saved["identity"] != manifest["identity"]
            or set(saved["outputs"]) != set(_OUTPUTS)
            or any(
                Path(value["path"]).resolve() != (native / (name + ".csv")).resolve()
                or sha256_file(Path(value["path"])) != value["sha256"]
                for name, value in saved["outputs"].items()
            )
        ):
            raise ValueError("Native closure checkpoint changed")
    else:
        # Never accept stale partial output from an earlier failed engine call.
        for name in _OUTPUTS:
            (native / (name + ".csv")).unlink(missing_ok=True)
        command = [
            manifest["engine"]["path"],
            "-j",
            str(manifest["jobs"]),
            "-D",
            str(native.resolve()),
            manifest["program"]["path"],
        ]
        with (destination / "native.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        saved = {
            "identity": manifest["identity"],
            "outputs": {name: _binding(native / (name + ".csv")) for name in _OUTPUTS},
        }
    facts = inventory["facts"]
    entities = {row[1] for row in facts["entities"]}
    derived = set()
    for name, arity in _OUTPUTS.items():
        with (native / (name + ".csv")).open(newline="") as stream:
            for row in csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(row) != arity:
                    raise ValueError("Unexpected native output arity")
                head, relation, tail = (row[0], "subclass_of", row[1]) if arity == 2 else row
                if head not in entities or tail not in entities:
                    raise ValueError(
                        "Materialization introduced entities outside the frozen population"
                    )
                derived.add((head, relation, tail))
    # Commit the native boundary only after checking every tuple's arity/population.
    if not receipt.exists():
        for name in _OUTPUTS:
            with (native / (name + ".csv")).open("rb") as stream:
                os.fsync(stream.fileno())
        _write_json(receipt, saved)
    if manifest["status"] == "complete":
        for key in ("native_receipt", "consequences"):
            item = manifest[key]
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise ValueError("Completed materialization artifact changed: " + key)
        restored = evidence_inventory(CsvKgSource.from_path(destination / "enriched"))
        if restored["sha256"] != manifest["enriched_evidence_sha256"]:
            raise ValueError("Completed enriched evidence changed")
        return manifest
    base = {tuple(row) for row in facts["projected_edges"]}
    added = sorted(derived - base)
    facts["projected_edges"] = sorted(base | derived)
    facts["projected_iri_edges"] = sorted(
        {tuple(row) for row in facts["projected_iri_edges"]} | derived
    )
    kinds = {iri: kind for kind, iri in facts["entities"]}
    facts["hierarchy"] = sorted(
        {tuple(row) for row in facts["hierarchy"]}
        | {
            (kinds[head], head, tail)
            for head, relation, tail in derived
            if relation == "subclass_of"
        }
    )
    enriched = destination / "enriched"
    enriched.mkdir(exist_ok=True)
    with (enriched / "triples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["src", "rel", "dst"])
        writer.writerows(facts["projected_edges"])
    with (enriched / "evidence.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["section", "record"])
        for section, rows in facts.items():
            for row in rows:
                writer.writerow([section, json.dumps(row, sort_keys=True, ensure_ascii=False)])
    (enriched / "kg.yaml").write_text(
        yaml.safe_dump(
            {
                "descriptor_version": 1,
                "triples_files": ["triples.csv"],
                "evidence_file": "evidence.csv",
            }
        )
    )
    with (enriched / "consequences.jsonl").open("w") as stream:
        for row in added:
            stream.write(
                json.dumps({"triple": row, "materialization": manifest["identity"]}) + "\n"
            )
    result = evidence_inventory(CsvKgSource.from_path(enriched))
    for key in (
        "entities",
        "labels",
        "annotations",
        "attributes",
        "domains",
        "ranges",
        "excluded",
        "property_axioms",
    ):
        if result["facts"][key] != inventory["facts"][key]:
            raise ValueError("Enrichment changed nonstructural base evidence: " + key)
    manifest.update(
        status="complete",
        derived_edges=len(derived),
        added_edges=len(added),
        enriched_evidence_sha256=result["sha256"],
        native_receipt=_binding(receipt),
        consequences=_binding(enriched / "consequences.jsonl"),
    )
    _write_json(path, manifest)
    return manifest
