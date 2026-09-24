import ast
import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from exact.experiments.inputs import prepare_pool
from exact.experiments.public_inference import prepare_public_inference
from exact.experiments.submission import export_submission
from exact.runs.layout import RunLayout


def _pool(tmp_path):
    path = tmp_path / "public.tsv"
    pd.DataFrame(
        [
            ("urn:s", "['urn:t1', 'urn:t2']"),
            ("urn:other", "['urn:t1']"),
            ("urn:s", "['urn:t2', 'urn:t3', 'urn:t4']"),
        ],
        columns=["SrcEntity", "TgtCandidates"],
    ).to_csv(path, sep="\t", index=False)
    return path


def test_preparation_retains_original_query_rows_and_no_gold(tmp_path):
    path = _pool(tmp_path)
    frame = pd.read_csv(path, sep="\t")
    frame["TgtEntity"] = "urn:private-never-output"
    frame.to_csv(path, sep="\t", index=False)
    record = prepare_pool(path, tmp_path / "prepared", role="test", expose_labels=False)
    queries = [
        json.loads(line)
        for line in Path(record["outputs"]["queries"]["path"]).read_text().splitlines()
    ]
    assert [row["qid"] for row in queries] == [0, 1, 2]
    assert [len(row["candidates"]) for row in queries] == [2, 1, 3]
    assert queries[0]["source"] == queries[2]["source"]
    for output in record["outputs"].values():
        assert "private-never-output" not in Path(output["path"]).read_text()
    assert record["union_score_reuse"] == "query_independent_components_only"


def test_query_shards_isolate_pool_dependent_scores_and_export_original_order(tmp_path):
    from exact.core.entities.configs.config import ConfigModel

    pool = _pool(tmp_path)
    selected = tmp_path / "selected.yaml"
    selected.write_text(
        "config_version: 2\ndata:\n  refs:\n    test: /unreadable/private-gold.tsv\nrun:\n  source_cap: 1\n"
    )
    manifest = prepare_public_inference(
        selected,
        tmp_path / "prepared",
        source=pool,
        target=pool,
        track="bioml-local",
        public_candidates=pool,
    )
    plan = json.loads(manifest.read_text())
    assert [run["query_indices"] for run in plan["runs"]] == [[0, 1], [2]]
    for run in plan["runs"]:
        config = ConfigModel.load_config(Path(run["config"]["path"]))
        assert config.data.refs == {} and config.run.source_cap is None
        assert all(not row.params.get("generate_llm_rationales") for row in config.pipeline)
        layout = RunLayout.create(Path(run["run_dir"]))
        layout.config_path.write_text(Path(run["config"]["path"]).read_text())
        queries = [plan["queries"][i] for i in run["query_indices"]]
        layout.source_decisions_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "stage": "after_cardinality_and_relation_typing",
                    "source_universe_status": "declared",
                    "source_universe": [q["source"] for q in queries],
                    "records": [
                        {
                            "Src": q["source"],
                            "candidates": [
                                {"target": t, "S_final": i} for i, t in enumerate(q["candidates"])
                            ],
                        }
                        for q in queries
                    ],
                }
            )
        )
    output = tmp_path / "submission.tsv"
    export_submission(
        Path(plan["runs"][0]["run_dir"]),
        output,
        "bioml-local",
        public_candidates=pool,
        query_runs=manifest,
    )
    rows = list(csv.DictReader(output.open(), delimiter="\t"))
    assert [row["SrcEntity"] for row in rows] == ["urn:s", "urn:other", "urn:s"]
    assert [ast.literal_eval(row["TgtCandidates"]) for row in rows] == [
        ["urn:t2", "urn:t1"],
        ["urn:t1"],
        ["urn:t4", "urn:t3", "urn:t2"],
    ]
    with pytest.raises(ValueError, match="Different original pools"):
        export_submission(
            Path(plan["runs"][0]["run_dir"]),
            tmp_path / "bad.tsv",
            "bioml-local",
            public_candidates=pool,
        )


def test_global_rejects_query_list_without_native_signature_provenance(tmp_path):
    layout = RunLayout.create(tmp_path / "run")
    sources = tmp_path / "test.sources.txt"
    sources.write_text("urn:s\n")
    with pytest.raises(ValueError, match="native population manifest"):
        export_submission(
            layout.root, tmp_path / "out.rdf", "bioml-global", source_universe=sources
        )


def test_native_population_reuse_and_incomplete_import_guard(tmp_path, monkeypatch):
    from exact.experiments.public_inference import (
        prepare_population,
        validate_population,
    )

    ontology = tmp_path / "source.ofn"
    ontology.write_text("Ontology(Declaration(Class(<urn:s>)))")
    path = tmp_path / "sources.txt"
    original = prepare_population(ontology, path, entity_kinds=["class"])

    def forbidden(*args, **kwargs):
        raise AssertionError("native population should be reused")

    monkeypatch.setattr("exact.ontology.load_ontology", forbidden)
    assert prepare_population(ontology, path, entity_kinds=["class"]) == original
    with pytest.raises(ValueError, match="different inputs/policy"):
        prepare_population(ontology, path, entity_kinds=["individual"])
    manifest = path.with_suffix(".txt.manifest.json")
    original["ontology_core"]["closure"]["complete"] = False
    manifest.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="full native"):
        validate_population(manifest)
