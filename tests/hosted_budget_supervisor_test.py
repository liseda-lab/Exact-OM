"""Shared spending milestones and approval pauses never launch repair models."""

import copy

import pytest

from exact.experiments import notifications
from exact.llm import spending_admission
from tests.experiments_supervisor_notifications_test import monitor  # noqa: F401


@pytest.fixture
def observed(monitor, monkeypatch):  # noqa: F811
    cli, policy, state, observation = monitor
    policy["hosted_spending_policy"] = {"path": "fixture", "sha256": "fixture"}
    observation.update(status="healthy", incidents=[])
    snapshot = {
        "snapshot_valid": True,
        "campaign_id": "campaign",
        "accounted_tokens": 96_190_067,
        "historical_tokens": 96_190_067,
        "reserved_tokens": 0,
        "reported_cost_usd": 14.75,
        "unpriced_attempts": 2,
        "notification_tokens": 10_000_000,
        "experiment_warning_tokens": 10_000_000,
        "experiment_tokens_cap": 25_000_000,
        "campaign_tokens_cap": 200_000_000,
        "experiments": {
            "E21": {"historical_tokens": 72_193_688, "accounted_tokens": 72_193_688},
            "E12": {"historical_tokens": 0, "accounted_tokens": 0},
        },
        "paused": False,
        "paused_experiments": [],
        "pauses": [],
    }
    monkeypatch.setattr(
        spending_admission, "admission_snapshot", lambda *args: copy.deepcopy(snapshot)
    )
    monkeypatch.setattr(
        notifications, "_deliver", lambda *args: pytest.fail("No external email in tests")
    )
    monkeypatch.setattr(
        cli, "run_agent", lambda *args: pytest.fail("Budget events must not start repair")
    )
    return cli, policy, state, observation, snapshot


def test_historical_excess_is_acknowledged_and_new_intervals_warn_once(observed, tmp_path):
    cli, policy, state, _, snapshot = observed
    first = cli.check(tmp_path, policy, state, act=True)
    assert first["status"] == "healthy" and not (tmp_path / "alerts").exists()
    acknowledged = cli.read(tmp_path / "hosted-spending-acknowledged.json")
    assert acknowledged == {
        "campaign_id": "campaign",
        "campaign_tokens": 90_000_000,
        "experiments": ["E21"],
    }
    snapshot["accounted_tokens"] = 106_190_067
    snapshot["experiments"]["E12"]["accounted_tokens"] = 10_000_000
    snapshot["experiment_tokens_caps"] = {"E12": 30_000_000}
    cli.check(tmp_path, policy, state, act=True)
    alerts = [cli.read(path) for path in (tmp_path / "alerts").glob("*.json")]
    assert len(alerts) == 2
    assert all(row["delivery"] == "pending" for row in alerts)
    assert any("30,000,000" in row["reason"] for row in alerts)
    assert all("$14.750000" in row["summary"] and "2 attempts" in row["summary"] for row in alerts)
    for _ in range(2):
        cli.check(tmp_path, policy, cli.read(tmp_path / "state.json"), act=True)
    assert len(list((tmp_path / "alerts").glob("*.json"))) == 2
    snapshot["accounted_tokens"] = 110_000_000
    cli.check(tmp_path, policy, state, act=True)
    assert len(list((tmp_path / "alerts").glob("*.json"))) == 3
    assert not state["agent_runs"] and not state["incidents"]


def test_existing_100m_notice_is_not_duplicated_by_new_interval_policy(observed, tmp_path):
    cli, policy, state, _, snapshot = observed
    policy["hosted_spending_milestone"] = {"id": "old-notice", "notification_tokens": 100_000_000}
    cli.notify_intervention(
        tmp_path,
        {"id": "hosted-token-milestone:old-notice"},
        "hosted_token_milestone",
        "Previously requested milestone",
        config=policy["notifications"],
        defer=True,
    )
    snapshot["accounted_tokens"] = 100_000_000
    cli.check(tmp_path, policy, state, act=True)
    assert len(list((tmp_path / "alerts").glob("*.json"))) == 1


def test_read_only_observation_does_not_acknowledge_or_queue(observed, tmp_path):
    cli, policy, state, _, snapshot = observed
    snapshot["accounted_tokens"] = 100_000_000
    cli.check(tmp_path, policy, state, act=False)
    assert not (tmp_path / "hosted-spending-acknowledged.json").exists()
    assert not (tmp_path / "alerts").exists()


def test_failed_outbox_write_does_not_acknowledge_unsaved_warning(observed, tmp_path, monkeypatch):
    cli, policy, _, _, snapshot = observed
    snapshot["accounted_tokens"] = 100_000_000
    monkeypatch.setattr(
        cli, "notify_intervention", lambda *args, **kwargs: {"path": str(tmp_path / "missing")}
    )
    result = cli.observe_hosted_limits(tmp_path, policy, act=True)
    assert result["status"] == "unavailable"
    assert not (tmp_path / "hosted-spending-acknowledged.json").exists()


def pause(snapshot, *, scope="experiment"):
    snapshot["pauses"] = [
        {
            "id": "durable-denial",
            "scope": scope,
            "experiment_id": "E21" if scope == "experiment" else None,
            "accounted_tokens": 72_193_688 if scope == "experiment" else 199_999_000,
            "requested_tokens": 4000,
            "limit": 25_000_000 if scope == "experiment" else 200_000_000,
            "active": True,
        }
    ]
    snapshot["paused"] = scope == "campaign"
    snapshot["paused_experiments"] = ["E21"] if scope == "experiment" else []


def test_notification_failure_cannot_turn_an_observed_budget_pause_into_repair(
    observed, tmp_path, monkeypatch
):
    cli, policy, state, observation, snapshot = observed
    pause(snapshot)
    snapshot["accounted_tokens"] = 100_000_000
    cli.write(
        tmp_path / "registry.json",
        {
            "runs": [
                {
                    "id": "failed-run",
                    "hosted_scope": {"campaign_id": "campaign", "experiment_id": "E21"},
                }
            ]
        },
    )
    observation.update(
        status="needs_attention",
        incidents=[
            {
                "id": "generic",
                "kind": "run_failed",
                "run_ids": ["failed-run"],
                "reason": "worker failed",
            }
        ],
    )
    monkeypatch.setattr(
        cli,
        "notify_intervention",
        lambda *args, **kwargs: {
            "path": str(tmp_path / "unwritten"),
            "delivery": "failed",
        },
    )
    result = cli.check(tmp_path, policy, state, act=True)
    assert result["hosted_spending"]["status"] == "unavailable"
    assert result["hosted_spending"]["snapshot_valid"]
    assert result["action"] == "requires_user" and not state["agent_runs"]
    assert [row["kind"] for row in observation["incidents"]] == ["hosted_spending_pause"]


def test_actual_pause_replaces_generic_failure_and_never_consumes_repair_attempt(
    observed, tmp_path
):
    cli, policy, state, observation, snapshot = observed
    pause(snapshot)
    root = tmp_path / "random-recovery"
    cli.write(root / "hosted-scope.json", {"campaign_id": "campaign", "experiment_id": "E21"})
    cli.write(
        tmp_path / "registry.json",
        {"runs": [{"id": "random-recovery", "status_path": str(root / "status.json")}]},
    )
    observation.update(
        status="needs_attention",
        incidents=[
            {
                "id": "generic-failure",
                "kind": "run_failed",
                "run_ids": ["random-recovery"],
                "reason": "worker failed",
            }
        ],
    )
    for _ in range(2):
        result = cli.check(tmp_path, policy, state, act=True)
        assert result["action"] == "requires_user"
    assert [row["kind"] for row in observation["incidents"]] == ["hosted_spending_pause"]
    assert len(list((tmp_path / "alerts").glob("*.json"))) == 1
    assert not state["agent_runs"]
    assert all(row["attempts"] == 0 for row in state["incidents"].values())


def test_family_pause_preserves_unrelated_incidents(observed):
    cli, _, _, observation, snapshot = observed
    pause(snapshot)
    snapshot["status"] = "observed"
    registry = {
        "runs": [{"id": "r", "hosted_scope": {"campaign_id": "campaign", "experiment_id": "E21"}}]
    }
    observation["incidents"] = [
        {"id": "failure", "kind": "run_failed", "run_ids": ["r"]},
        {"id": "other", "kind": "run_failed", "run_ids": ["native-E14"]},
    ]
    cli.prioritize_hosted_pauses(snapshot, registry, observation)
    assert [row["id"] for row in observation["incidents"]] == [
        "hosted-spend-pause:durable-denial",
        "other",
    ]


def test_pending_dispatch_pause_remains_approval_only_after_error_text_is_overwritten(observed):
    cli, _, _, observation, snapshot = observed
    pause(snapshot)
    snapshot["status"] = "observed"
    registry = {
        "runs": [],
        "pending_batches": [
            {
                "id": "pending-batch",
                "launch": {
                    "run": {
                        "id": "not-yet-registered",
                        "hosted_scope": {
                            "campaign_id": "campaign",
                            "experiment_id": "E21",
                        },
                    }
                },
            }
        ],
    }
    observation["incidents"] = [
        {
            "id": "dispatch-expired",
            "kind": "dispatch_failed",
            "run_ids": ["pending-batch"],
            "reason": "Launch has no verified receipt after 90s",
        }
    ]
    cli.prioritize_hosted_pauses(snapshot, registry, observation)
    assert [row["kind"] for row in observation["incidents"]] == ["hosted_spending_pause"]
    incident = observation["incidents"][0]
    assert set(incident["run_ids"]) == {"pending-batch", "not-yet-registered"}
    assert cli.eligible({"incidents": {incident["id"]: {}}}, incident, {}, 0) == (
        False,
        "requires_user",
    )


def test_dispatch_accepts_scoped_shared_env_and_blocks_only_actual_paid_pause(
    observed, tmp_path, monkeypatch
):
    cli, policy, _, _, snapshot = observed
    monkeypatch.setattr(cli.storage_guard, "validate_launch", lambda *args: None)
    monkeypatch.setattr(cli.hosted_prompt_guard, "validate_launch", lambda *args: None)
    environment = tmp_path / "environment.json"
    cli.write(environment, {"EXACT_HOSTED_CAMPAIGN_ID": "campaign"})
    receipt = tmp_path / "receipt.json"
    cli.write(receipt, {"mode": "guarded_hosted", "environment": {"path": str(environment)}})
    launch = {
        "hosted_prompt_guard": {"path": str(receipt)},
        "run": {"hosted_scope": {"campaign_id": "campaign", "experiment_id": "E12"}},
    }
    pause(snapshot)
    cli.validate_prepared_launch(launch, policy, tmp_path)
    launch["run"]["hosted_scope"]["experiment_id"] = "E21"
    with pytest.raises(spending_admission.HostedSpendPause):
        cli.validate_prepared_launch(launch, policy, tmp_path)
    pause(snapshot, scope="campaign")
    launch["run"]["hosted_scope"]["experiment_id"] = "E12"
    with pytest.raises(spending_admission.HostedSpendPause):
        cli.validate_prepared_launch(launch, policy, tmp_path)
    cli.write(receipt, {"mode": "no_new_hosted_calls"})
    cli.validate_prepared_launch(launch, policy, tmp_path)
