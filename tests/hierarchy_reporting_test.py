"""Selected hierarchy reports retain empty sources and use authoritative outcomes."""

import hashlib
import json

import pytest

from exact.experiments.reporting import hierarchy_metric_rows, metric_reports
from exact.impl.evaluators.builtin import BuiltinEvaluator
from exact.runs.store import ExplanationStore
from exact.utils.provenance import file_provenance


def _explanation(source, target, *, edges=(), lexical=0.2, hierarchy=0.1):
    return {
        "src_iri": source,
        "tgt_iri": target,
        "contributions": {"C_label": lexical, "C_strsim": 0, "C_hier": hierarchy},
        "triple_attributions": {
            "hierarchy": {
                "is_a": {
                    "source": [
                        {"subject_iri": child, "object_iri": parent} for child, parent in edges
                    ]
                }
            }
        },
    }


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / "run"
    store = ExplanationStore.create(run)
    alignment = run / "alignment" / "maps_global.tsv"
    alignment.parent.mkdir(exist_ok=True)
    alignment.write_text("SrcEntity\tTgtEntity\tScore\ns1\tt1\t0.9\ns2\twrong\t0.9\n")
    reference = tmp_path / "reference.tsv"
    reference.write_text("SrcEntity\tTgtEntity\ns1\tt1\ns2\tt2\n")
    metrics = BuiltinEvaluator.global_eval(alignment, reference)
    report = run / "evaluation" / "evaluation_results.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "builtin": {key: metrics[key] for key in ("P", "R", "F1")},
                "meta": {
                    "refs": {
                        "alignment": file_provenance(alignment),
                        "full_reference": file_provenance(reference),
                    }
                },
            }
        )
    )
    groups = [[source, "class"] for source in ("empty", "s1", "s2")]
    sample = {
        "source_kind_groups": groups,
        "eligible_source_iris": [source for source, _ in groups],
        "sha256": hashlib.sha256(
            "\n".join(f"{source}\t{kind}" for source, kind in groups).encode()
        ).hexdigest(),
    }
    manifest = run / "dataset" / "candidate_pool_sample_manifest.json"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_text(json.dumps({"retrieval_config": {"source_sample": sample}}))
    return (
        {
            "experiment_id": "E09",
            "arm_id": "current",
            "task_id": "fixture",
            "seed": 17,
            "status": "complete",
            "output_dir": str(run),
            "metrics": {},
        },
        store,
        manifest,
    )


def _lookup(rows, suffix):
    return next(row for row in rows if row["metric"] == "selected_hierarchy.is_a." + suffix)


def test_reports_use_selected_graph_and_keep_unobserved_sources(saved_run):
    record, store, _ = saved_run
    edges = (("s1", "parent"), ("parent", "ancestor"), ("sibling", "parent"))
    store.append(
        [
            _explanation("s1", "t1", edges=edges, lexical=0.2),
            _explanation("s1", "t2", edges=edges, lexical=0.6),
            _explanation("s2", "t2", lexical=0.9, hierarchy=0),
        ]
    )
    rows, macros = metric_reports([record])
    assert _lookup(rows, "depth.2_plus.source_count")["value"] == 1
    assert _lookup(rows, "branching.2_plus.source_count")["value"] == 1
    assert _lookup(rows, "coverage.present.source_fraction")["value"] == pytest.approx(1 / 3)
    assert _lookup(rows, "coverage.present.F1")["value"] == 1
    assert _lookup(rows, "coverage.absent.F1")["value"] == 0
    assert _lookup(rows, "coverage.unobserved.source_count")["value"] == 1
    assert _lookup(rows, "coverage.present.lexical_contribution")["value"] == pytest.approx(0.4)
    assert (
        _lookup(rows, "coverage.unobserved.hierarchy_contribution")["availability"] == "unavailable"
    )
    diagnostics = [row for row in rows if row["metric_family"] == "hierarchy_diagnostic"]
    assert all(row["frozen_source_count"] == 3 and row["diagnostic_only"] for row in diagnostics)
    assert all(
        row["evidence_scope"] == "selected_explanation_evidence_not_full_ontology"
        for row in diagnostics
    )
    assert any(row["metric"] == "selected_hierarchy.is_a.depth.2_plus.F1" for row in macros)


def test_cycles_disconnected_edges_and_source_balancing(saved_run):
    record, store, _ = saved_run
    store.append(
        [
            _explanation("s1", "t1", edges=(("s1", "p"), ("p", "s1")), lexical=0.2),
            _explanation("s1", "t2", edges=(("s1", "p"),), lexical=0.6),
            _explanation("s2", "t2", edges=(("unrelated", "ancestor"),), lexical=0.9),
        ]
    )
    rows = hierarchy_metric_rows(record)
    assert _lookup(rows, "depth.1.source_count")["value"] == 1
    assert _lookup(rows, "depth.unreachable.source_count")["value"] == 1
    assert _lookup(rows, "coverage.present.lexical_contribution")["value"] == pytest.approx(0.65)
    assert _lookup(rows, "coverage.present.F1")["value"] == pytest.approx(0.5)
    assert (
        sum(
            _lookup(rows, f"coverage.{bucket}.source_count")["value"]
            for bucket in ("present", "absent", "unobserved")
        )
        == 3
    )


@pytest.mark.parametrize(
    "fault, error",
    [
        ("duplicate", "Duplicate"),
        ("outside", "outside frozen"),
        ("missing_iri", "actual subject/object"),
        ("changed_population", "integrity"),
        ("missing_population", "verified frozen"),
    ],
)
def test_reports_reject_unbound_or_ambiguous_evidence(saved_run, fault, error):
    record, store, manifest = saved_run
    explanation = _explanation("s1", "t1", edges=(("s1", "parent"),))
    if fault == "outside":
        explanation["src_iri"] = "outside"
    elif fault == "missing_iri":
        del explanation["triple_attributions"]["hierarchy"]["is_a"]["source"][0]["subject_iri"]
    elif fault == "changed_population":
        payload = json.loads(manifest.read_text())
        payload["retrieval_config"]["source_sample"]["eligible_source_iris"].remove("empty")
        manifest.write_text(json.dumps(payload))
    elif fault == "missing_population":
        manifest.unlink()
    store.append([explanation, explanation] if fault == "duplicate" else [explanation])
    with pytest.raises(ValueError, match=error):
        hierarchy_metric_rows(record)
