import json
from types import SimpleNamespace

import pandas as pd
import pytest

from exact.experiments.relation_metrics import (
    relation_metrics,
    write_relation_diagnostics,
)
from exact.io.relations import predict_relations
from exact.ontology import load_ontology


def _frame(rows):
    return pd.DataFrame(rows, columns=["SrcEntity", "TgtEntity", "Relation"]).assign(Score=0.8)


def test_relation_macro_counts_wrong_types_and_missing_pairs():
    gold = _frame([("s1", "t1", "="), ("s2", "t2", "<"), ("s3", "t3", ">")])
    prediction = _frame([("s1", "t1", "="), ("s2", "t2", ">")])
    result = relation_metrics(gold, prediction)
    assert result["macro_F1"] == pytest.approx(1 / 3)
    assert result["by_relation"]["<"]["fn"] == 1
    assert result["by_relation"][">"]["fp"] == 1
    assert result["by_relation"][">"]["fn"] == 1


def test_oracle_pairs_never_become_anchors_or_training_examples(tmp_path, monkeypatch):
    gold = _frame([("s1", "t1", "="), ("s2", "t2", "<"), ("s3", "t3", ">")])
    accepted = gold.iloc[:1].copy()
    accepted.to_csv(tmp_path / "paper.maps_global.tsv", sep="\t", index=False)
    candidates = gold.iloc[:2].drop(columns="Relation")
    trainer = SimpleNamespace(
        dataset=SimpleNamespace(reference=gold, source=None, target=None),
        results_df=candidates,
        alignment_dir=tmp_path,
        relation_evaluation_role="valid",
    )
    observed = []

    def typer(frame, source, target, **options):
        assert "Relation" not in frame and set(frame.Score) == {0.0}
        assert len(options["anchor_candidates"]) == 2
        observed.append(frame.copy())
        return gold

    monkeypatch.setattr("exact.experiments.relation_metrics.predict_relations", typer)
    write_relation_diagnostics(trainer, accepted, options={"mode": "none"})
    metrics = json.loads((tmp_path / "relation_metrics.json").read_text())
    assert len(observed) == 1
    assert metrics["oracle_pairs"]["macro_F1"] == 1
    assert metrics["full_pipeline"]["macro_F1"] == pytest.approx(1 / 3)
    trainer.relation_evaluation_role = "test"
    write_relation_diagnostics(trainer, accepted, options={"mode": "none"})
    assert len(observed) == 1


def test_native_bridge_entails_without_queried_anchor_and_times_out(tmp_path, monkeypatch):
    a, b = tmp_path / "a.ofn", tmp_path / "b.ofn"
    a.write_text(
        "Ontology(Declaration(Class(<urn:a>)) Declaration(Class(<urn:b>)) SubClassOf(<urn:a> <urn:b>) "
        "Declaration(Class(<urn:unsat>)) SubClassOf(<urn:unsat> <http://www.w3.org/2002/07/owl#Nothing>))"
    )
    b.write_text(
        "Ontology(Declaration(Class(<urn:c>)) Declaration(Class(<urn:d>)) SubClassOf(<urn:c> <urn:d>))"
    )
    source, target = load_ontology(a), load_ontology(b)
    frame = pd.DataFrame(
        [("urn:a", "urn:d", 0.8), ("urn:b", "urn:c", 0.9)], columns=["Src", "Tgt", "Score"]
    )
    anchors = pd.DataFrame([("urn:b", "urn:c", 1.0)], columns=["Src", "Tgt", "Score"])
    kwargs = dict(
        mode="semantic_entailment",
        semantic_backend="bridge_reasoner",
        anchors=anchors,
        timeout_seconds=10,
    )
    result = predict_relations(frame, source, target, **kwargs)
    assert result[["SrcEntity", "TgtEntity", "Relation"]].to_dict("records") == [
        {"SrcEntity": "urn:a", "TgtEntity": "urn:d", "Relation": "<"}
    ]
    assert result.attrs["coherence_audit"]["ontology_consistency"] == "consistent"
    assert result.attrs["coherence_audit"]["logical_unsatisfiability"] == "unknown"
    assert result.attrs["coherence_audit"]["native_import_document_cache_hits"] >= 2
    assert result.attrs["relation_abstentions"] == [
        {"source": "urn:b", "target": "urn:c", "reason": "not_entailed"}
    ]

    def timeout(*args, **kwargs):
        raise TimeoutError("native deadline")

    monkeypatch.setattr("exact.ontology.reasoning._create_hermit", timeout)
    result = predict_relations(frame, source, target, **kwargs)
    assert result.empty
    assert {item["reason"] for item in result.attrs["relation_abstentions"]} == {
        "reasoning_timeout"
    }


def test_harness_loads_verified_typed_endpoints(tmp_path, monkeypatch):
    from exact.experiments.harness import cell_metrics
    from exact.utils.provenance import sha256_file

    alignment = tmp_path / "alignment"
    alignment.mkdir()
    payload = {
        "schema_version": 1,
        "evaluation_only": True,
        "reference_role": "valid",
        "oracle_pairs": {"macro_F1": 0.75},
        "full_pipeline": {"macro_F1": 0.6},
    }
    for key, name in (
        ("reference", "relations.reference.tsv"),
        ("oracle", "relations.oracle.tsv"),
        ("alignment", "paper.maps_global.tsv"),
    ):
        path = alignment / name
        path.write_text("saved artifact")
        payload[key + "_sha256"] = sha256_file(path)
    (alignment / "relation_metrics.json").write_text(json.dumps(payload))
    monkeypatch.setattr("exact.experiments.harness.extract_evaluation_metrics", lambda path: {})
    assert cell_metrics(tmp_path) == {
        "relation.oracle_pairs.macro_F1": 0.75,
        "relation.full_pipeline.macro_F1": 0.6,
    }
    (alignment / "relations.oracle.tsv").write_text("changed")
    with pytest.raises(ValueError, match="identity changed"):
        cell_metrics(tmp_path)
