"""Frozen populations survive pool misses, exact matches and grouped preparation."""

import ast
from types import SimpleNamespace

import pandas as pd
import pytest

from exact.core.entities.kinds import EntityKind
from exact.experiments.budget import BudgetLedger
from exact.experiments.inputs import prepare_pool, research_partitions
from exact.impl.datasets.base import BaseAlignmentDataset


class _PopulationDataset(BaseAlignmentDataset):
    def log_sanity_examples(self, *args, **kwargs):
        pass

    def plot_feature_distributions(self, *args, **kwargs):
        pass

    def __getitem__(self, index):
        raise IndexError(index)

    def __len__(self):
        return len(self.dataframe)

    def get_features(self, frame):
        return frame.copy()


def test_frozen_sources_include_pool_misses_and_exact_sources(tmp_path):
    dataset = _PopulationDataset(output_path=tmp_path)
    dataset._source_entity_kind_index = {source: EntityKind.CLASS for source in ("s1", "s2", "s3")}
    dataset._source = SimpleNamespace(
        entities=lambda kind: ("s1", "s2", "s3") if kind == EntityKind.CLASS else ()
    )
    dataset._target = SimpleNamespace(
        entities=lambda kind: ("t1", "t2", "t3") if kind == EntityKind.CLASS else ()
    )
    dataset.freeze_source_universe(["s3", "s1", "s2"], cap=None, seed=17)
    frame = pd.DataFrame(
        [("s1", "t1", "class", "class", 0.8)],
        columns=["Src", "Tgt", "SrcKind", "TgtKind", "cand_sim"],
    )
    dataset._df = frame.copy()
    dataset._candidates = frame.copy()
    dataset._reference = pd.DataFrame(
        [("s1", "t1"), ("s2", "t2"), ("s3", "t3")], columns=["Src", "Tgt"]
    )
    dataset._exact_matches = pd.DataFrame([("s3", "t3")], columns=["Src", "Tgt"])
    assert dataset.restrict_sources(cap=3, seed=17) == {"s1", "s2", "s3"}
    assert len(dataset.reference) == 3
    assert len(dataset.exact_matches) == 1
    sample = dataset.candidate_pool_manifest["retrieval_config"]["source_sample"]
    assert sample["eligible_source_iris"] == ["s1", "s2", "s3"]
    assert len(dataset.dataframe) == 1


def test_source_cap_is_nested_and_independent_of_candidate_rows(tmp_path):
    samples = []
    for cap, sources in ((2, ["s1"]), (3, ["s3", "s2"])):
        dataset = _PopulationDataset(output_path=tmp_path / str(cap))
        dataset._source_entity_kind_index = {s: EntityKind.CLASS for s in ("s1", "s2", "s3", "s4")}
        dataset.freeze_source_universe(["s4", "s3", "s2", "s1"], cap=cap, seed=17)
        dataset._df = pd.DataFrame([(s, "t") for s in sources], columns=["Src", "Tgt"])
        samples.append(dataset.restrict_sources(cap=cap, seed=17))
    assert len(samples[0]) == 2
    assert len(samples[1]) == 3
    assert samples[0] < samples[1]


def test_pool_preparation_unions_queries_strips_gold_and_retains_empty_sources(tmp_path):
    raw = tmp_path / "raw.tsv"
    pd.DataFrame(
        [("s", "t1", "['t1','t2']"), ("s", "t3", "['t3','t1']"), ("empty", "", "[]")],
        columns=["SrcEntity", "TgtEntity", "TgtCandidates"],
    ).to_csv(raw, sep="\t", index=False)
    record = prepare_pool(raw, tmp_path / "prepared", role="valid", expose_labels=True)
    pool = pd.read_csv(record["outputs"]["candidates"]["path"], sep="\t", keep_default_na=False)
    assert len(pool) == 2 and set(pool.TgtEntity) == {""}
    assert ast.literal_eval(pool.loc[pool.SrcEntity == "s", "TgtCandidates"].iloc[0]) == [
        "t1",
        "t2",
        "t3",
    ]
    assert record["sources"] == 2 and record["original_query_rows"] == 3
    with pytest.raises(ValueError, match="final labels"):
        prepare_pool(raw, tmp_path / "final", role="test", expose_labels=True)


def test_research_split_keeps_all_rows_for_a_source_together():
    frame = pd.DataFrame(
        [(str(i), str(j)) for i in range(10) for j in range(3)], columns=["SrcEntity", "TgtEntity"]
    )
    partitions = research_partitions(frame)
    assert [len(partitions[x]) for x in ("train", "valid", "internal_check")] == [18, 6, 6]
    assert not set(partitions["train"].SrcEntity) & set(partitions["valid"].SrcEntity)
    assert not set(partitions["valid"].SrcEntity) & set(partitions["internal_check"].SrcEntity)


def test_budget_keeps_retry_spend_reserve_and_overlapping_node_time(tmp_path):
    limits = {
        "envelopes_hours": {"channels": 1, "final": 1},
        "requests_cap": 10,
        "tokens_cap": 100,
        "final_requests_reserved": 4,
        "final_tokens_reserved": 40,
    }
    ledger = BudgetLedger(tmp_path / "budget.json", limits)
    ledger.admit("one", group="channels", seconds=100, requests=3, tokens=30)
    ledger.admit("two", group="channels", seconds=100, requests=3, tokens=30)
    with pytest.raises(ValueError, match="protected final"):
        ledger.admit("over", group="channels", seconds=1, requests=1)
    ledger.finish("one", start=10, end=110, status="failed", requests=3, tokens=30, actual_usd=None)
    ledger.finish(
        "two", start=20, end=120, status="complete", requests=3, tokens=30, actual_usd=0.1
    )
    state = ledger.snapshot()
    assert state["node_seconds"] == 110
    assert state["work"]["one"]["status"] == "failed"
    assert "over" not in state["work"]
    with pytest.raises(ValueError, match="new attempt"):
        ledger.admit("one", group="channels", seconds=1)


def test_public_final_pool_without_gold_column_preserves_empty_source(tmp_path):
    from exact.experiments.inputs import prepare_pool
    import pandas as pd

    pool = tmp_path / "public.tsv"
    pool.write_text("SrcEntity\tTgtCandidates\ns1\t['t1']\ns2\t[]\n")
    record = prepare_pool(pool, tmp_path / "prepared", role="test", expose_labels=False)
    output = pd.read_csv(record["outputs"]["candidates"]["path"], sep="\t", keep_default_na=False)
    assert output.SrcEntity.tolist() == ["s1", "s2"]
    assert output.TgtEntity.tolist() == ["", ""]
    assert output.TgtCandidates.tolist() == ["['t1']", "[]"]
    assert not (tmp_path / "prepared/test.reference.tsv").exists()
