import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from exact.analysis.candidate_recall import (
    absent_gold_dataframe,
    analyze_candidate_recall,
    flatten_candidate_recall,
    write_absent_gold_tsv,
)
from exact.core.actions.alignment import _candidate_recall_run_stats


def test_candidate_recall_excludes_train_and_counts_exact_oracle():
    candidates = pd.DataFrame(
        [
            {"Src": "s1", "Tgt": "t1", "cand_sim": 0.9},
            {"Src": "s1", "Tgt": "t_alt", "cand_sim": 0.8},
            {"Src": "s2", "Tgt": "t_wrong", "cand_sim": 0.7},
            {"Src": "s2", "Tgt": "t2", "cand_sim": 0.6},
            {"Src": "s_train", "Tgt": "t_train", "cand_sim": 1.0},
        ]
    )
    reference_pairs = {
        ("s1", "t1"),
        ("s2", "t2"),
        ("s3", "t3"),
        ("s4", "t4"),
        ("s_train", "t_train"),
    }
    train_pairs = {("s_train", "t_train")}
    exact_pairs = {("s4", "t4"), ("s_train", "t_train")}

    analysis = analyze_candidate_recall(
        candidates, reference_pairs, train_pairs=train_pairs, exact_pairs=exact_pairs
    )

    assert analysis["counts"]["reference_pairs"] == 4
    assert analysis["counts"]["candidate_pairs"] == 4
    assert analysis["counts"]["generated_hits"] == 2
    assert analysis["counts"]["oracle_hits"] == 3
    assert analysis["counts"]["absent_gold_pairs"] == 2
    assert analysis["counts"]["absent_gold_pairs_after_exact"] == 1
    assert analysis["metrics"]["generated_candidate_recall"] == 0.5
    assert analysis["metrics"]["exact_prefilter_oracle_recall"] == 0.75
    assert analysis["gold_rank"]["present_pairs"] == 2
    assert analysis["gold_rank"]["rank_median"] == 1.5
    assert analysis["gold_rank"]["rank_p90"] == 1.9
    assert absent_gold_dataframe(analysis).to_dict("records") == [
        {"Src": "s3", "Tgt": "t3"},
        {"Src": "s4", "Tgt": "t4"},
    ]
    assert absent_gold_dataframe(analysis, after_exact=True).to_dict("records") == [
        {"Src": "s3", "Tgt": "t3"},
    ]


def test_candidate_recall_writes_absent_gold_and_flattens(tmp_path: Path):
    analysis = analyze_candidate_recall(
        candidates=[("s1", "t1")],
        reference_pairs=[("s1", "t1"), ("s2", "t2")],
    )
    out = tmp_path / "absent.tsv"
    out_after_exact = tmp_path / "absent_after_exact.tsv"

    write_absent_gold_tsv(out, analysis)
    write_absent_gold_tsv(out_after_exact, analysis, after_exact=True)
    flat = flatten_candidate_recall(40, analysis)

    assert out.read_text(encoding="utf-8").splitlines() == ["Src\tTgt", "s2\tt2"]
    assert out_after_exact.read_text(encoding="utf-8").splitlines() == ["Src\tTgt", "s2\tt2"]
    assert flat["top_k"] == 40
    assert flat["reference_pairs"] == 2
    assert flat["generated_candidate_recall"] == 0.5


def test_candidate_recall_run_stats_uses_materialized_pool_and_exact_prefilter(
    tmp_path: Path,
):
    training = tmp_path / "train.tsv"
    pd.DataFrame([{"SrcEntity": "s_train", "TgtEntity": "t_train"}]).to_csv(
        training, sep="\t", index=False
    )
    dataset = SimpleNamespace(
        reference=pd.DataFrame(
            [
                {"Src": "s1", "Tgt": "t1"},
                {"Src": "s2", "Tgt": "t2"},
                {"Src": "s_train", "Tgt": "t_train"},
            ]
        ),
        candidates=pd.DataFrame(
            [
                {"Src": "s1", "Tgt": "t1", "cand_sim": 0.9},
                {"Src": "s2", "Tgt": "t_wrong", "cand_sim": 0.8},
                {"Src": "s_train", "Tgt": "t_train", "cand_sim": 1.0},
            ]
        ),
        exact_matches=pd.DataFrame([{"Src": "s2", "Tgt": "t2"}]),
        candidate_pool_manifest={"gold_free_summary": {"mean_pool_size": 7.5}},
    )

    stats = _candidate_recall_run_stats(
        dataset,
        training_reference_path=training,
    )

    assert stats["metric_applicability"]["candidate_recall"] is True
    assert stats["candidate_recall"] == 0.5
    assert stats["candidate_recall_after_exact"] == 1.0
    assert stats["mean_pool_size"] == 7.5
    assert stats["gold_rank_p90"] == 1.0
    assert stats["gold_rank_median"] == 1.0
    diagnostics = stats["candidate_recall_diagnostics"]
    assert diagnostics["status"] == "available"
    assert diagnostics["counts"]["reference_pairs"] == 2
    assert diagnostics["counts"]["generated_hits"] == 1
    assert diagnostics["counts"]["oracle_hits"] == 2
    assert diagnostics["gold_rank"]["present_pairs"] == 1
    assert diagnostics["gold_rank"]["rank_median"] == 1.0


def test_candidate_recall_run_stats_marks_empty_reference_not_applicable():
    dataset = SimpleNamespace(
        reference=pd.DataFrame(columns=["Src", "Tgt"]),
        candidates=pd.DataFrame(columns=["Src", "Tgt", "cand_sim"]),
        exact_matches=pd.DataFrame(columns=["Src", "Tgt"]),
        candidate_pool_manifest={"gold_free_summary": {"mean_pool_size": 0.0}},
    )

    stats = _candidate_recall_run_stats(dataset, training_reference_path=None)

    assert stats["metric_applicability"]["candidate_recall"] is False
    assert stats["candidate_recall_diagnostics"] == {
        "status": "not_applicable",
        "reason": "stage reference is absent or empty",
    }


@pytest.fixture
def cached_reporting_dataset(tmp_path):
    from exact.impl.datasets.base import BaseAlignmentDataset

    class CandidateRecallCacheDataset(BaseAlignmentDataset):
        def __getitem__(self, index):
            raise IndexError(index)

        def __len__(self):
            return len(self.dataframe)

        def get_features(self, frame):
            return frame

        def plot_feature_distributions(self, *args, **kwargs):
            pass

        def log_sanity_examples(self, *args, **kwargs):
            pass

    def make(output_path=None):
        dataset = CandidateRecallCacheDataset(
            output_path=output_path or tmp_path, entity_kinds=["class", "object_property"]
        )
        dataset._source = dataset._target = SimpleNamespace(entities=lambda kind: ())
        dataset._eligible_source_groups = {("s", "class"), ("p", "object_property")}
        return dataset

    dataset = make()
    columns = ["Src", "Tgt", "SrcKind", "TgtKind", "cand_sim"]
    dataset._candidates = pd.DataFrame(
        [
            ("s", "wrong", "class", "class", 0.9),
            ("s", "gold", "class", "class", 0.7),
            ("p", "property_gold", "object_property", "object_property", 0.8),
            ("outside", "outside_gold", "class", "class", 1.0),
            ("s", "wrong_kind", "object_property", "object_property", 1.0),
        ],
        columns=columns,
    )
    dataset._exact_matches = pd.DataFrame(
        [
            ("s", "gold", "class", "class"),
            ("s", "inserted_exact", "class", "class"),
        ],
        columns=columns[:4],
    )
    reference = pd.concat(
        [
            dataset._candidates.iloc[[1, 2, 3, 4]][columns[:4]],
            dataset._exact_matches.iloc[[1]],
        ],
        ignore_index=True,
    )
    dataset._reference = reference.copy()
    dataset._df = pd.concat(
        [
            dataset._candidates.drop(index=1).assign(inference=True, prefiltered=False),
            dataset._exact_matches.assign(inference=False, prefiltered=True),
        ],
        ignore_index=True,
    )
    dataset._candidate_pool_sizes = {
        "class": {"source_entities": 2},
        "object_property": {"source_entities": 2},
    }
    dataset._refresh_candidate_pool_manifest(origin="generated")
    dataset.save()
    return dataset, make, reference


def test_cached_recall_preserves_raw_ranks_exact_insertions_and_source_kinds(
    cached_reporting_dataset,
):
    cold, make, reference = cached_reporting_dataset
    cache_bytes = cold._df_save_path.read_bytes()
    cold.restrict_sources(cap=2, seed=17)
    expected = _candidate_recall_run_stats(cold, training_reference_path=None)
    assert expected["candidate_recall"] == 2 / 3
    assert expected["candidate_recall_after_exact"] == 1.0
    assert expected["gold_rank_median"] == 1.5
    assert expected["mean_pool_size"] == 1.5

    warm = make()
    assert warm.has_cache()
    warm.load()
    warm._reference = reference.copy()
    warm.restrict_sources(cap=2, seed=17)
    before = warm.dataframe.copy(deep=True)
    assert _candidate_recall_run_stats(warm, training_reference_path=None) == expected
    assert warm.candidates is None and warm.exact_matches is None
    pd.testing.assert_frame_equal(warm.dataframe, before)
    assert warm._df_save_path.read_bytes() == cache_bytes
    sidecar = pd.read_csv(warm.output_path / "candidate_recall.tsv", sep="\t")
    assert "Label" not in sidecar and "Features" not in sidecar


@pytest.mark.parametrize("change", ["legacy", "missing", "tampered"])
def test_cached_recall_rejects_unavailable_or_changed_raw_pool(cached_reporting_dataset, change):
    cold, make, reference = cached_reporting_dataset
    sidecar = cold.output_path / "candidate_recall.tsv"
    if change == "legacy":
        metadata = json.loads(cold._cache_meta_path.read_text())
        metadata.pop("candidate_recall_sha256")
        cold._cache_meta_path.write_text(json.dumps(metadata))
    elif change == "missing":
        sidecar.unlink()
    else:
        sidecar.write_text(sidecar.read_text() + "tampered\n")
    warm = make()
    warm._reference = reference
    if change == "legacy":
        stats = _candidate_recall_run_stats(warm, training_reference_path=None)
        assert stats["metric_applicability"]["candidate_recall"] is False
        assert "raw candidate pool" in stats["candidate_recall_diagnostics"]["reason"]
    else:
        with pytest.raises(ValueError, match="candidate-recall pool"):
            _candidate_recall_run_stats(warm, training_reference_path=None)


def test_shared_prepared_cache_uses_normal_native_gate(
    cached_reporting_dataset, tmp_path, monkeypatch
):
    from exact.impl.datasets import prepared_cache

    cold, make, reference = cached_reporting_dataset
    monkeypatch.setenv("EXACT_DATASET_CACHE_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("EXACT_EXPERIMENT_ROLE", "development")
    prepared_cache.publish(cold.output_path, cold.cache_fingerprint)
    warm = make(tmp_path / "new-arm")
    assert warm.has_cache()
    warm.load()
    warm._reference = reference
    assert _candidate_recall_run_stats(warm, training_reference_path=None)["candidate_recall"] > 0

    # A different candidate policy cannot consume the shared prepared rows.
    changed = make(tmp_path / "changed-arm")
    changed._candidate_generation_params["top_k"] = 99
    assert not changed.has_cache()

    # Even an integrity-valid shared snapshot must pass the existing schema gate.
    metadata = json.loads(cold._cache_meta_path.read_text())
    metadata["cache_schema_version"] = 1
    cold._cache_meta_path.write_text(json.dumps(metadata))
    monkeypatch.setenv("EXACT_DATASET_CACHE_DIR", str(tmp_path / "legacy-shared"))
    prepared_cache.publish(cold.output_path, cold.cache_fingerprint)
    with pytest.raises(ValueError, match="native/schema compatibility"):
        make(tmp_path / "legacy-arm").has_cache()
