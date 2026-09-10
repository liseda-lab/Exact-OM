import csv
import json

import pytest

from exact.experiments.submission import NIL_IRI, export_submission
from exact.io.writers.oaei_rdf import read_alignment
from exact.runs.layout import RunLayout


def _run(tmp_path, targets, *, sources=("urn:s1",), field="S_final"):
    layout = RunLayout.create(tmp_path / "run")
    records = [
        {
            "Src": source,
            "candidates": [
                {"target": target, field: (i + 1) / len(targets)}
                for i, target in enumerate(targets)
            ],
        }
        for source in sources
    ]
    trace = {
        "schema_version": 2,
        "stage": "after_cardinality_and_relation_typing",
        "source_universe": list(sources),
        "source_universe_status": "declared",
        "records": records,
    }
    layout.source_decisions_path.write_text(json.dumps(trace))
    # Paths are deliberately nonexistent: export must never load configuration labels.
    layout.config_path.write_text("data:\n  refs:\n    test: /unreadable/private-test.tsv\n")
    return layout, trace


def _pool(tmp_path, targets, track="bioml-local", *, qid=7, source="urn:s1"):
    path = tmp_path / ("pools.jsonl" if track == "diso-ranking" else "pools.tsv")
    if track == "diso-ranking":
        path.write_text(
            json.dumps({"qid": qid, "source": source, "type": "CLS", "candidates": targets}) + "\n"
        )
    else:
        with path.open("w") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["SrcEntity", "TgtCandidates"])
            writer.writerow([source, repr(targets)])
    return path


def test_bioml_local_exports_all_saved_scores_without_references(tmp_path):
    targets = [f"urn:t{i:03}" for i in range(100)]
    layout, _ = _run(tmp_path, targets)
    pool = _pool(tmp_path, targets)
    output = tmp_path / "submission.tsv"
    export_submission(layout.root, output, "bioml-local", public_candidates=pool)
    with output.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 100 and list(rows[0]) == ["SrcEntity", "TgtCandidate", "Score"]
    assert rows[0] == {"SrcEntity": "urn:s1", "TgtCandidate": "urn:t099", "Score": "1"}
    original = output.read_bytes()
    export_submission(layout.root, output, "bioml-local", public_candidates=pool)
    assert output.read_bytes() == original
    manifest = json.loads(output.with_suffix(".tsv.manifest.json").read_text())
    assert manifest["reference_labels_used"] is False and manifest["models_invoked"] is False
    assert manifest["query_count"] == 1


@pytest.mark.parametrize("defect", ["missing", "extra", "duplicate", "nonfinite", "capped_source"])
def test_incomplete_or_ambiguous_scores_fail_without_output(tmp_path, defect):
    targets = [f"urn:t{i}" for i in range(100)]
    layout, trace = _run(tmp_path, targets)
    candidates = trace["records"][0]["candidates"]
    if defect == "missing":
        candidates.pop()
    elif defect == "extra":
        candidates.append({"target": "urn:extra", "S_final": 0.2})
    elif defect == "duplicate":
        candidates.append(candidates[0])
    elif defect == "nonfinite":
        candidates[0]["S_final"] = float("nan")
    else:
        trace["source_universe"].append("urn:empty")
        trace["records"].append({"Src": "urn:empty", "candidates": []})
    layout.source_decisions_path.write_text(json.dumps(trace))
    output = tmp_path / "submission.tsv"
    with pytest.raises(ValueError):
        export_submission(
            layout.root, output, "bioml-local", public_candidates=_pool(tmp_path, targets)
        )
    assert not output.exists()


def test_diso_preserves_query_ids_and_saved_joint_nil_scale(tmp_path):
    targets = [f"urn:t{i:03}" for i in range(49)]
    layout, trace = _run(tmp_path, targets, field="Q_match")
    for candidate in trace["records"][0]["candidates"]:
        candidate["Q_match"] *= 0.1
        candidate["Q_nil"] = 0.7
    trace["records"][0]["ontology_nil_probability"] = 0.7
    layout.source_decisions_path.write_text(json.dumps(trace))
    pool = _pool(tmp_path, targets + [NIL_IRI], "diso-ranking", qid=83)
    # Same source may represent several official queries; never collapse qid.
    with pool.open("a") as stream:
        stream.write(
            json.dumps({"qid": 2, "source": "urn:s1", "candidates": targets + [NIL_IRI]}) + "\n"
        )
    output = tmp_path / "submission.jsonl"
    export_submission(
        layout.root, output, "diso-ranking", public_candidates=pool, score_field="Q_match"
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["qid"] for row in rows] == [83, 2]
    assert all(row["ranking"][0] == NIL_IRI and len(row["ranking"]) == 50 for row in rows)
    assert set(rows[0]["ranking"]) == set(targets + [NIL_IRI])


@pytest.mark.parametrize(
    "defect",
    [
        "missing_qid",
        "duplicate_qid",
        "duplicate_candidate",
        "missing_nil",
        "missing_nil_score",
        "answer_field",
    ],
)
def test_diso_rejects_lost_query_identity_or_incomplete_pool(tmp_path, defect):
    targets = [f"urn:t{i}" for i in range(49)] + [NIL_IRI]
    layout, trace = _run(tmp_path, targets)
    pool = _pool(tmp_path, targets, "diso-ranking")
    row = json.loads(pool.read_text())
    if defect == "missing_qid":
        row.pop("qid")
    elif defect == "duplicate_candidate":
        row["candidates"][0] = row["candidates"][1]
    elif defect == "missing_nil":
        row["candidates"][-1] = "urn:other"
    elif defect == "missing_nil_score":
        trace["records"][0]["candidates"].pop()
        layout.source_decisions_path.write_text(json.dumps(trace))
    elif defect == "answer_field":
        row["gold"] = "urn:never-read"
    pool.write_text(json.dumps(row) + "\n")
    if defect == "duplicate_qid":
        with pool.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        export_submission(
            layout.root, tmp_path / "out.jsonl", "diso-ranking", public_candidates=pool
        )


def test_global_requires_full_population_and_reuses_predictions(tmp_path):
    layout, trace = _run(tmp_path, ["urn:t1"], sources=("urn:s1", "urn:empty"))
    trace["records"][1]["candidates"] = []
    layout.source_decisions_path.write_text(json.dumps(trace))
    layout.mapping_path("global").write_text("SrcEntity\tTgtEntity\tScore\nurn:s1\turn:t1\t0.8\n")
    population = tmp_path / "sources.txt"
    population.write_text("urn:s1\nurn:empty\n")
    output = tmp_path / "out.rdf"
    kwargs = {"source_universe": population, "source_uri": "urn:source", "target_uri": "urn:target"}
    export_submission(layout.root, output, "bioml-global", **kwargs)
    frame = read_alignment(output)
    assert frame.to_dict("records") == [
        {"SrcEntity": "urn:s1", "TgtEntity": "urn:t1", "Score": 0.8, "Relation": "="}
    ]
    population.write_text("urn:s1\nurn:empty\nurn:unprocessed\n")
    with pytest.raises(ValueError, match="full declared source population"):
        export_submission(layout.root, tmp_path / "invalid.rdf", "oaei-kg-global", **kwargs)


def test_export_never_replaces_different_file_or_accepts_gold_pool(tmp_path):
    targets = [f"urn:t{i}" for i in range(100)]
    layout, _ = _run(tmp_path, targets)
    pool = _pool(tmp_path, targets)
    output = tmp_path / "out.tsv"
    output.write_text("existing user submission")
    with pytest.raises(FileExistsError):
        export_submission(layout.root, output, "bioml-local", public_candidates=pool)
    assert output.read_text() == "existing user submission"
    pool.write_text("SrcEntity\tTgtEntity\tTgtCandidates\n")
    with pytest.raises(ValueError, match="gold-stripped"):
        export_submission(
            layout.root, tmp_path / "invalid.tsv", "bioml-local", public_candidates=pool
        )
    with pytest.raises(ValueError, match="immutable run"):
        export_submission(
            layout.root, layout.root / "submission.tsv", "bioml-local", public_candidates=pool
        )


def test_historical_explanations_export_without_source_trace(tmp_path):
    targets = [f"urn:t{i:03}" for i in range(100)]
    layout, _ = _run(tmp_path, targets)
    layout.source_decisions_path.unlink()
    layout.full_explanations_path.write_text(
        json.dumps(
            [
                {"src_iri": "urn:s1", "tgt_iri": target, "confidences": {"P_rank": 0.01}}
                for target in reversed(targets)
            ]
        )
    )
    output = tmp_path / "ranking.tsv"
    export_submission(
        layout.root,
        output,
        "bioml-local",
        public_candidates=_pool(tmp_path, targets),
        score_field="P_rank",
    )
    with output.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["TgtCandidate"] for row in rows] == targets


def test_global_uses_final_audit_when_compatibility_output_is_local(tmp_path):
    layout, _ = _run(tmp_path, ["urn:t1"])
    (layout.alignment_dir / "paper.maps_global.tsv").write_text(
        "SrcEntity\tTgtEntity\tScore\tRelation\nurn:s1\turn:t1\t0.7\t=\n"
    )
    population = tmp_path / "sources.txt"
    population.write_text("urn:s1\n")
    output = tmp_path / "final.rdf"
    export_submission(
        layout.root,
        output,
        "oaei-kg-global",
        source_universe=population,
        source_uri="urn:source",
        target_uri="urn:target",
    )
    assert read_alignment(output).iloc[0]["Score"] == 0.7
