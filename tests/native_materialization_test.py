import csv
import json
import os
from pathlib import Path

import pytest

from exact.experiments.evidence_inventory import (
    evidence_inventory,
    import_biokg_public_graph,
)
from exact.experiments.materialization import (
    materialize_legacy,
    prepare_legacy_materialization,
)
from exact.io.sources.csv_kg import CsvKgSource


def _package(tmp_path):
    graph = tmp_path / "public" / "graph"
    graph.mkdir(parents=True)
    names = ["A:a", "A:b", "A:c", "B:other"]
    with (graph / "properties.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ontology", "node_id", "preferred_label", "entity_kind"])
        writer.writerows([(name[0], name, name, "class") for name in names])
    with (graph / "triples.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "head_id",
                "relation",
                "tail_id",
                "head_ontology",
                "tail_ontology",
                "is_anchor",
                "release_layer",
            ]
        )
        writer.writerows(
            [
                ["A:a", "subclass_of", "A:b", "A", "A", "false", "public"],
                ["A:b", "subclass_of", "A:c", "A", "A", "false", "public"],
                ["A:c", "anchor_equivalent", "B:other", "A", "B", "true", "public"],
                # Even a same-ontology reference-verified anchor must be excluded.
                ["A:c", "anchor_equivalent", "A:a", "A", "A", "true", "public"],
            ]
        )
    from exact.experiments.materialization import _ARITIES, _OUTPUTS

    rules = "\n".join(
        ".decl " + name + "(" + ", ".join(f"x{i}:symbol" for i in range(arity)) + ")"
        for name, arity in {**_ARITIES, **_OUTPUTS}.items()
    )
    rules += "\ninferred_subclass(x,y) :- subclass(x,y).\ninferred_subclass(x,z) :- inferred_subclass(x,y), inferred_subclass(y,z).\ninferred_edge(x,r,y) :- edge(x,r,y).\n"
    (graph / "rules.dl").write_text(rules)
    (graph / "facts.dl").write_text('equiv("A:c", "B:other").\nequiv("A:c", "A:a").\n')
    return graph.parent


def test_native_materialization_retains_population_and_excludes_reference_anchors(tmp_path):
    executable = Path(
        os.environ.get("EXACT_TEST_SOUFFLE", "data/tools/souffle-2.5/usr/bin/souffle")
    )
    if not executable.is_file():
        pytest.skip("Set EXACT_TEST_SOUFFLE to a native Souffle binary")
    package = _package(tmp_path)
    raw, destination = tmp_path / "raw", tmp_path / "closure"
    import_biokg_public_graph(package, raw, ontology="A")
    before = evidence_inventory(CsvKgSource.from_path(raw))
    prepared = prepare_legacy_materialization(
        package, raw, destination, ontology="A", executable=executable
    )
    assert prepared["alignment_anchors_imported"] == 0
    assert prepared["input_fact_count"] == 2
    result = materialize_legacy(raw, destination)
    assert result["added_edges"] == 1
    enriched = CsvKgSource.from_path(destination / "enriched")
    assert ("A:a", "subclass_of", "A:c") in {edge.astuple() for edge in enriched.projection_edges()}
    assert enriched.direct_parents("A:c") == []
    assert evidence_inventory(enriched)["facts"]["entities"] == before["facts"]["entities"]
    assert materialize_legacy(raw, destination) == result  # durable native checkpoint reused
    (destination / "native/inferred_subclass.csv").write_text("corruption")
    with pytest.raises(ValueError, match="checkpoint changed"):
        materialize_legacy(raw, destination)


def test_materialization_rejects_external_evidence_and_rule_directives(tmp_path):
    package = _package(tmp_path)
    raw = tmp_path / "raw"
    import_biokg_public_graph(package, raw, ontology="A")
    (package / "graph/rules.dl").write_text('.include "private.dl"\n')
    with pytest.raises(ValueError, match="self-contained"):
        prepare_legacy_materialization(
            package, raw, tmp_path / "bad", ontology="A", executable="/bin/true"
        )


def test_unscoped_auxiliary_axioms_never_enter_the_native_program(tmp_path):
    package = _package(tmp_path)
    graph = package / "graph"
    # Both property names occur locally: that alone cannot prove these axioms
    # came from this ontology instead of another ontology in the public release.
    with (graph / "triples.csv").open("a") as stream:
        csv.writer(stream).writerows(
            [
                ["A:a", "same_predicate", "A:b", "A", "A", "false", "public"],
                ["A:b", "other_predicate", "A:c", "A", "A", "false", "public"],
            ]
        )
    with (graph / "facts.dl").open("a") as stream:
        stream.write('subprop("same_predicate", "other_predicate").\n')
        stream.write('domain("same_predicate", "A:c").\n')
    raw = tmp_path / "raw"
    import_biokg_public_graph(package, raw, ontology="A")
    destination = tmp_path / "closure"
    result = prepare_legacy_materialization(
        package, raw, destination, ontology="A", executable="/bin/true"
    )
    program = (destination / "program.dl").read_text()
    assert 'subprop("same_predicate"' not in program
    assert 'domain("same_predicate"' not in program
    assert result["retained_auxiliary_facts"] == 0
    assert result["excluded_released_facts_by_predicate"] == {"equiv": 2, "subprop": 1, "domain": 1}
    assert result["full_legacy_materialization"] is False
    assert result["semantic_fragment"] == "raw_graph_closure_with_empty_auxiliary_relations"


@pytest.mark.parametrize("field,value", [("jobs", 1), ("input_role", "private"), ("ontology", "B")])
def test_resume_rejects_changed_materialization_request(tmp_path, field, value):
    package = _package(tmp_path)
    raw, destination = tmp_path / "raw", tmp_path / "closure"
    import_biokg_public_graph(package, raw, ontology="A")
    prepare_legacy_materialization(package, raw, destination, ontology="A", executable="/bin/true")
    path = destination / "materialization.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="request identity changed"):
        materialize_legacy(raw, destination)


def test_stale_native_files_cannot_be_accepted_after_an_empty_successful_process(tmp_path):
    package = _package(tmp_path)
    raw, destination = tmp_path / "raw", tmp_path / "closure"
    import_biokg_public_graph(package, raw, ontology="A")
    prepare_legacy_materialization(package, raw, destination, ontology="A", executable="/bin/true")
    native = destination / "native"
    native.mkdir()
    (native / "inferred_subclass.csv").write_text("A:a\tA:c\n")
    (native / "inferred_edge.csv").write_text("")
    with pytest.raises(FileNotFoundError):
        materialize_legacy(raw, destination)
    assert not (destination / "native-complete.json").exists()


def _cli(monkeypatch, package, output, *, ontology="A", executable="/bin/true"):
    import sys

    from tools.materialize_biokg import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_biokg",
            "--package",
            str(package),
            "--ontology",
            ontology,
            "--output",
            str(output),
            "--souffle",
            str(executable),
            "--prepare-only",
        ],
    )
    main()


def test_cli_prepare_only_reuse_verifies_request_without_running_closure(tmp_path, monkeypatch):
    import tools.materialize_biokg as cli

    package, output = _package(tmp_path), tmp_path / "output"
    _cli(monkeypatch, package, output)
    manifest_path = output / "materialized/materialization.json"
    before = manifest_path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("valid prepare-only reuse must not reprepare or run closure")

    monkeypatch.setattr(cli, "materialize_legacy", forbidden)
    monkeypatch.setattr(cli, "prepare_legacy_materialization", forbidden)
    _cli(monkeypatch, package, output)
    assert manifest_path.read_bytes() == before
    assert not (output / "materialized/native").exists()


@pytest.mark.parametrize("changed", ["ontology", "engine", "package"])
def test_cli_prepare_only_rejects_changed_requested_bindings(tmp_path, monkeypatch, changed):
    import shutil

    package, output = _package(tmp_path), tmp_path / "output"
    _cli(monkeypatch, package, output)
    kwargs = {}
    if changed == "ontology":
        kwargs["ontology"] = "B"
    elif changed == "engine":
        kwargs["executable"] = "/bin/false"
    else:
        copied = tmp_path / "copy-of-same-package"
        shutil.copytree(package, copied)
        package = copied
    with pytest.raises(ValueError, match="Requested materialization " + changed + " changed"):
        _cli(monkeypatch, package, output, **kwargs)
    assert not (output / "materialized/native").exists()


@pytest.mark.parametrize(
    "relative", ["materialized/program.dl", "raw/triples.csv", "raw/evidence.csv", "raw/kg.yaml"]
)
def test_cli_prepare_only_rejects_changed_frozen_files(tmp_path, monkeypatch, relative):
    package, output = _package(tmp_path), tmp_path / "output"
    _cli(monkeypatch, package, output)
    path = output / relative
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Materialization .* changed"):
        _cli(monkeypatch, package, output)
    assert not (output / "materialized/native").exists()
