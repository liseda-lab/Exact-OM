import pytest

from exact.impl.retrieval import (
    matched_mean_pool_size,
    retrieval_diagnostic_record,
    select_e05_candidate,
)


def _row(
    arm: str,
    task: str,
    recall: float,
    *,
    pool: float = 20.0,
    p90: float = 8.0,
    median: float = 3.0,
    runtime: float = 10.0,
    split_role: str = "development",
) -> dict:
    return {
        "arm_id": arm,
        "task_id": task,
        "split_role": split_role,
        "candidate_recall": recall,
        "mean_pool_size": pool,
        "gold_rank_p90": p90,
        "gold_rank_median": median,
        "runtime_seconds": runtime,
        "candidate_pool_fingerprint": ("a" * 63 + str((sum(map(ord, arm + task)) % 10))),
    }


def test_retrieval_diagnostic_record_joins_gold_and_gold_free_artifacts() -> None:
    record = retrieval_diagnostic_record(
        arm_id="rrf",
        task_id="fixture",
        analysis={
            "metrics": {"generated_candidate_recall": 0.75},
            "gold_rank": {"rank_p90": 4.5, "rank_median": 2.0},
        },
        pool_manifest={
            "fingerprint": "b" * 64,
            "gold_free_summary": {"mean_pool_size": 19.5},
        },
        runtime_seconds=3.25,
        seed=17,
    )

    assert record == {
        "arm_id": "rrf",
        "task_id": "fixture",
        "split_role": "development",
        "seed": 17,
        "candidate_recall": 0.75,
        "mean_pool_size": 19.5,
        "gold_rank_p90": 4.5,
        "gold_rank_median": 2.0,
        "runtime_seconds": 3.25,
        "candidate_pool_fingerprint": "b" * 64,
    }


def test_e05_selection_enforces_matched_pools_recall_guards_and_rank_order() -> None:
    rows = [
        _row("baseline", "task-a", 0.90),
        _row("baseline", "task-b", 0.80),
        _row("better_rank", "task-a", 0.91, p90=5.0),
        _row("better_rank", "task-b", 0.80, p90=5.0),
        _row("worse_rank", "task-a", 0.91, p90=7.0),
        _row("worse_rank", "task-b", 0.80, p90=7.0),
        _row("task_regression", "task-a", 0.94),
        _row("task_regression", "task-b", 0.794),
        _row("unmatched_pool", "task-a", 0.95, pool=22.0),
        _row("unmatched_pool", "task-b", 0.95, pool=22.0),
    ]

    result = select_e05_candidate(
        rows,
        baseline_arm="baseline",
        candidate_arms=[
            "better_rank",
            "worse_rank",
            "task_regression",
            "unmatched_pool",
        ],
    )

    assert result["status"] == "selected"
    assert result["selected_arm"] == "better_rank"
    reports = {row["arm_id"]: row for row in result["candidates"]}
    assert reports["better_rank"]["eligible"] is True
    assert reports["worse_rank"]["eligible"] is True
    assert {reason["code"] for reason in reports["task_regression"]["exclusion_reasons"]} == {
        "task_recall_regression"
    }
    assert {reason["code"] for reason in reports["unmatched_pool"]["exclusion_reasons"]} == {
        "mean_pool_size_not_matched"
    }


def test_e05_selection_screens_out_and_never_reads_reporting_rows() -> None:
    rows = [
        _row("baseline", "task", 0.80),
        _row("candidate", "task", 0.804),
    ]
    result = select_e05_candidate(
        rows,
        baseline_arm="baseline",
        candidate_arms=["candidate"],
    )
    assert result["status"] == "screened_out"
    assert result["selected_arm"] is None

    rows[-1]["split_role"] = "reporting"
    with pytest.raises(ValueError, match="cannot consume reporting"):
        select_e05_candidate(
            rows,
            baseline_arm="baseline",
            candidate_arms=["candidate"],
        )


def test_matched_pool_tolerance_uses_larger_of_one_candidate_or_two_percent() -> None:
    assert matched_mean_pool_size(20, 21)["matched"] is True
    assert matched_mean_pool_size(20, 21.01)["matched"] is False
    check = matched_mean_pool_size(100, 102)
    assert check["matched"] is True
    assert check["allowed_absolute_difference"] == 2.0
