import pytest

from exact.experiments.throughput import Measurement, PhaseCounters, forecast, freeze_workload, group_chunks


def test_workload_preserves_original_query_context_and_empty_groups():
    queries = [
        {"qid": "q0", "source": "a", "candidates": ["x", "y"]},
        {"qid": "q1", "source": "a", "candidates": ["y", "z"]},
        {"qid": "q2", "source": "b", "candidates": []},
    ]
    manifest = freeze_workload(queries, pair="H0", minimum_pairs=5000)
    assert manifest["unique_candidate_pairs"] == 3
    assert manifest["candidate_occurrences"] == 4
    assert sorted(manifest["queries"], key=lambda q: q["qid"]) == queries
    chunks = list(group_chunks(manifest["queries"], 1))
    assert sum(len(q["candidates"]) for chunk in chunks for q in chunk) == 4
    assert all(len({q["source"] for q in c}) == len(c) for c in chunks)


def test_numerator_excludes_replayed_pairs_and_requires_durability():
    workload = freeze_workload([], pair="H0")
    measurement = Measurement(mode="cold", workload=workload, device="test-double")
    with pytest.raises(ValueError, match="durably"):
        measurement.commit_chunk(elapsed=1, computed=[("s", "t", "evidence")], durable=False)
    measurement.commit_chunk(elapsed=1, computed=[("s", "t", "evidence")]*100, durable=True)
    measurement.commit_chunk(elapsed=1, computed=[("s", "t", "evidence")], durable=True)
    result = measurement.receipt()
    assert result["distinct_computed_pairs"] == 1
    assert result["pairs_per_second"] == 0.5
    assert result["throughput_target_status"] == "unqualified"


def test_forecast_never_substitutes_target_or_zero_for_unknown_work():
    result = forecast({"local": {"count": [100, 200], "rate_per_second": [10, 20]},
                       "hosted": {"count": 3, "rate_per_second": None}})
    assert result["known_seconds"] == [5, 20]
    assert result["remaining_seconds"] is None
    assert result["unknown_phases"] == ["hosted"]


def test_profiler_restores_inherited_method_after_failure():
    class Worker:
        def work(self):
            raise RuntimeError("scorer failed")
    worker, phases = Worker(), PhaseCounters()
    with pytest.raises(RuntimeError, match="scorer failed"):
        with phases.instrument(worker, {"work": "raw_matrices"}):
            worker.work()
    assert "work" not in worker.__dict__
    assert phases.calls == {"raw_matrices": 1}
