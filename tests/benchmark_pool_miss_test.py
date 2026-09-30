"""Development gold removal is complete within a pinned benchmark pool only."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from exact.core.entities.configs.config import ConfigModel
from exact.core.entities.configs.experimental import NilConfig, PoolMissCandidateLabels
from exact.experiments.nil_evaluation import evaluate_source_labels
from exact.experiments.public_inference import prepare_public_inference
from exact.impl.models.selector.nil_head import (
    prepare_pool_miss_diagnostic,
    remove_development_positives,
    restrict_benchmark_exact_matches,
)
from exact.utils.frozen_inference import _check_config
from exact.utils.provenance import sha256_file


def benchmark(tmp_path):
    rows = [
        (f"s{i}", f"t{j}", int(i < 10 and j == 0))
        for i in range(20)
        for j in range(70 if i == 19 else 100)
    ]
    labels = pd.DataFrame(rows, columns=["Src", "Tgt", "confirmed_label"])
    reference = {(s, t) for s, t, value in rows if value}
    label_path = tmp_path / "valid.confirmed.tsv"
    labels.to_csv(label_path, sep="\t", index=False)
    reference_path = tmp_path / "valid.reference.tsv"
    pd.DataFrame(sorted(reference), columns=["Src", "Tgt"]).to_csv(
        reference_path, sep="\t", index=False
    )
    binding = {"path": str(label_path), "sha256": sha256_file(label_path)}
    return labels, reference, reference_path, binding


class Dataset:
    def __init__(self, frame):
        self._candidates = frame.copy()
        self.eligible_source_iris = tuple(sorted(set(frame.Src)))
        self._exact_matches = frame.iloc[:2].copy()
        self._active_candidate_config = {}

    @property
    def candidates(self):
        return self._candidates

    def _refresh_candidate_pool_manifest(self, **kwargs):
        self.manifest_origin = kwargs["origin"]


def test_pinned_pool_intervention_removes_only_confirmed_positives_and_preserves_all_sources(
    tmp_path,
):
    labels, reference, reference_path, binding = benchmark(tmp_path)
    dataset = Dataset(labels[["Src", "Tgt"]])
    source_universe = dataset.eligible_source_iris
    report = prepare_pool_miss_diagnostic(
        dataset,
        reference_path,
        negative_label_policy="confirmed_negatives",
        seed=17,
        label_semantics="benchmark_pool",
        candidate_labels=binding,
    )
    assert len(labels) == 1970 and len(dataset.candidates) == 1960
    assert set(zip(dataset.candidates.Src, dataset.candidates.Tgt)) == {
        (s, t) for s, t, value in labels.itertuples(index=False, name=None) if value == 0
    }
    assert len(source_universe) == 20 and dataset.eligible_source_iris == source_universe
    assert set(map(tuple, report["removed_pairs"])) == reference
    assert report["absence_semantics"] == "synthetic_benchmark_pool_miss"
    assert not report["ontology_nil_claim"] and report["candidate_labels"] == binding
    assert set(zip(dataset._exact_matches.Src, dataset._exact_matches.Tgt)) == {("s0", "t1")}
    assert dataset.manifest_origin == "development_gold_removal"


@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate", "reference", "nonbinary"])
def test_benchmark_gold_removal_rejects_unverified_pool_or_reference(tmp_path, damage):
    labels, reference, _, _ = benchmark(tmp_path)
    frame = labels[["Src", "Tgt"]].copy()
    if damage == "missing":
        labels = labels.iloc[1:]
    elif damage == "extra":
        frame.loc[len(frame)] = ["s0", "unassessed"]
    elif damage == "duplicate":
        frame.loc[len(frame)] = frame.iloc[0]
    elif damage == "reference":
        reference.remove(("s0", "t0"))
    else:
        labels.loc[0, "confirmed_label"] = 2
    with pytest.raises(ValueError):
        remove_development_positives(
            frame,
            reference,
            role="development",
            negative_label_policy="confirmed_negatives",
            label_semantics="benchmark_pool",
            candidate_labels=labels,
        )


def test_changed_label_binding_and_source_universe_fail_before_dataset_mutation(tmp_path):
    labels, _, reference_path, binding = benchmark(tmp_path)
    dataset = Dataset(labels[["Src", "Tgt"]])
    before = dataset.candidates.copy()
    kwargs = dict(
        negative_label_policy="confirmed_negatives",
        seed=17,
        label_semantics="benchmark_pool",
        candidate_labels={**binding, "sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="changed after binding"):
        prepare_pool_miss_diagnostic(dataset, reference_path, **kwargs)
    kwargs["candidate_labels"] = binding
    dataset.eligible_source_iris += ("missing-source",)
    with pytest.raises(ValueError, match="complete source universe"):
        prepare_pool_miss_diagnostic(dataset, reference_path, **kwargs)
    pd.testing.assert_frame_equal(dataset.candidates, before)


def test_old_incomplete_reference_and_non_development_interventions_remain_forbidden(tmp_path):
    labels, reference, _, _ = benchmark(tmp_path)
    for options in (
        {"role": "development", "negative_label_policy": "confirmed_negatives"},
        {
            "role": "reporting",
            "negative_label_policy": "confirmed_negatives",
            "label_semantics": "benchmark_pool",
            "candidate_labels": labels,
        },
    ):
        with pytest.raises(ValueError, match="development"):
            remove_development_positives(labels[["Src", "Tgt"]], reference, **options)
    kept, _ = remove_development_positives(
        labels[["Src", "Tgt"]],
        reference,
        role="development",
        negative_label_policy="complete_reference",
    )
    assert len(kept) == 1960


def diagnostic_cell(tmp_path):
    labels, _, reference, binding = benchmark(tmp_path)
    source_labels = tmp_path / "valid.source_labels.tsv"
    pd.DataFrame(
        [(f"s{i}", "in_pool" if i < 10 else "benchmark_nil") for i in range(20)],
        columns=["Src", "Status"],
    ).to_csv(source_labels, sep="\t", index=False)
    records = [
        {
            "Src": source,
            "candidates": [{"target": r.Tgt, "S_final": 0.1} for r in group.itertuples()],
            "emitted_targets": [],
            "absence_semantics": "unknown",
        }
        for source, group in labels[labels.confirmed_label == 0].groupby("Src")
    ]
    trace = {
        "schema_version": 2,
        "stage": "after_cardinality_and_relation_typing",
        "source_universe_status": "declared",
        "source_universe": [row["Src"] for row in records],
        "records": records,
    }
    (tmp_path / "source_decisions.json").write_text(json.dumps(trace))
    return SimpleNamespace(
        output_dir=tmp_path,
        split_role="development",
        reference_role="valid",
        reference_completeness="known_incomplete",
        diagnostics={
            "role": "development",
            "reference_role": "valid",
            "label_semantics": "benchmark_pool",
            "evaluation_source_labels": {
                "path": str(source_labels),
                "sha256": sha256_file(source_labels),
            },
            "evaluation_candidate_labels": binding,
        },
        resolved_config={
            "data": {"refs": {"valid": str(reference)}},
            "matching": {
                "nil": {
                    "label_semantics": "benchmark_pool",
                    "pool_miss_development_reference": str(reference),
                    "pool_miss_candidate_labels": binding,
                }
            },
        },
    )


def test_evaluator_keeps_benchmark_nil_and_synthetic_misses_separate(tmp_path):
    cell = diagnostic_cell(tmp_path)
    result = evaluate_source_labels(cell)
    metrics = result["metrics"]
    assert metrics["labeled_sources"] == 20 and metrics["unknown_sources_excluded"] == 0
    assert metrics["synthetic_benchmark_pool_miss"] == {
        "sources": 10,
        "abstained_sources": 10,
        "emitted_sources": 0,
        "removed_positive_pairs": 10,
        "remaining_pairs": 1960,
        "ontology_nil_claim": False,
    }
    assert metrics["non_nil_sources"] == 10 and metrics["non_nil_MRR"] == 0
    assert "natural_nil" not in metrics
    report = json.loads(Path(result["artifact"]["path"]).read_text())
    assert report["evaluated_source_status_counts"] == {"pool_miss": 10, "benchmark_nil": 10}
    assert (
        report["reference_completeness"] == "known_incomplete" and not report["ontology_nil_claim"]
    )
    assert report["pool_status_unresolved_sources"] == []


@pytest.mark.parametrize(
    "damage",
    ["missing_candidate", "unassessed", "positive_retained", "source", "binding", "emitted"],
)
def test_synthetic_evaluation_rejects_incomplete_or_changed_intervention(tmp_path, damage):
    cell = diagnostic_cell(tmp_path)
    path = tmp_path / "source_decisions.json"
    trace = json.loads(path.read_text())
    if damage == "missing_candidate":
        trace["records"][0]["candidates"].pop()
    elif damage == "unassessed":
        trace["records"][0]["candidates"].append({"target": "outside", "S_final": 0.1})
    elif damage == "positive_retained":
        trace["records"][0]["candidates"].append({"target": "t0", "S_final": 0.9})
    elif damage == "source":
        source = trace["records"].pop()["Src"]
        trace["source_universe"].remove(source)
    elif damage == "emitted":
        trace["records"][0]["emitted_targets"] = ["outside"]
    else:
        cell.resolved_config["matching"]["nil"]["pool_miss_candidate_labels"] = None
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="Benchmark pool-miss"):
        evaluate_source_labels(cell)


def test_diagnostic_bindings_are_explicit_and_removed_from_public_inference(tmp_path):
    _, _, reference, binding = benchmark(tmp_path)
    kwargs = {
        "label_semantics": "benchmark_pool",
        "pool_miss_development_reference": reference,
        "pool_miss_candidate_labels": binding,
    }
    assert NilConfig(**kwargs).pool_miss_candidate_labels.sha256 == binding["sha256"]
    for overrides in (
        {"label_semantics": "natural"},
        {"pool_miss_candidate_labels": None},
        {"pool_miss_development_reference": None},
    ):
        with pytest.raises(ValueError):
            NilConfig(**{**kwargs, **overrides})
    pool = tmp_path / "public.tsv"
    pool.write_text("SrcEntity\tTgtCandidates\nurn:s\t['urn:t']\n")
    selected = tmp_path / "selected.json"
    selected.write_text(
        json.dumps(
            {
                "config_version": 2,
                "matching": {"nil": {**kwargs, "pool_miss_development_reference": str(reference)}},
            }
        )
    )
    plan = json.loads(
        prepare_public_inference(
            selected,
            tmp_path / "public",
            source=pool,
            target=pool,
            track="bioml-local",
            public_candidates=pool,
        ).read_text()
    )
    deployed = ConfigModel.load_config(Path(plan["runs"][0]["config"]["path"]))
    assert deployed.matching.nil.pool_miss_development_reference is None
    assert deployed.matching.nil.pool_miss_candidate_labels is None
    # Even an orphan inserted without revalidating the nested model cannot deploy.
    deployed.matching.nil.pool_miss_candidate_labels = PoolMissCandidateLabels(**binding)
    with pytest.raises(ValueError, match="interventions cannot be deployed"):
        _check_config(deployed)


def test_exact_prefilter_and_actual_trace_writer_preserve_the_supplied_pool(tmp_path):
    from exact.core.entities.configs.dataset import DatasetMask
    from exact.impl.trainer.audit_io import AuditIOMixin

    labels, reference, reference_path, binding = benchmark(tmp_path)
    dataset = Dataset(labels[["Src", "Tgt"]])
    dataset._exact_matches = pd.DataFrame(
        [("s0", "t0"), ("s0", "t1"), ("s0", "outside")], columns=["Src", "Tgt"]
    )
    restrict_benchmark_exact_matches(dataset)
    assert set(zip(dataset._exact_matches.Src, dataset._exact_matches.Tgt)) == {
        ("s0", "t0"),
        ("s0", "t1"),
    }
    prepare_pool_miss_diagnostic(
        dataset,
        reference_path,
        negative_label_policy="confirmed_negatives",
        seed=17,
        label_semantics="benchmark_pool",
        candidate_labels=binding,
    )
    scored = dataset.candidates.copy()
    scored["S_final"] = 0.1
    scored[DatasetMask.prefiltered] = False
    scored.loc[(scored.Src == "s0") & (scored.Tgt == "t1"), DatasetMask.prefiltered] = True
    scored["Scores"] = 1.0
    dataset.dataframe = scored
    runner = AuditIOMixin()
    runner.dataset, runner.output_dir = dataset, tmp_path
    runner._final_candidate_frame = scored[~scored[DatasetMask.prefiltered]].copy()
    runner.results_json = []
    runner.model = SimpleNamespace()
    runner._json_safe_value = lambda value: value
    path = runner._write_source_decisions([], pd.DataFrame())
    trace = json.loads(path.read_text())
    traced = {
        (row["Src"], candidate["target"])
        for row in trace["records"]
        for candidate in row["candidates"]
    }
    assert len(trace["records"]) == 20 and len(traced) == 1960
    assert not traced & reference
    assert ("s0", "t1") in traced and ("s0", "outside") not in traced
