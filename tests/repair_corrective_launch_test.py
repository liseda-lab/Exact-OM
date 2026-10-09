"""First-wave planning binds every gate without starting processes or paid calls."""

import json
from pathlib import Path

import pytest

from tools.repair import corrective_launch as launch
from tools.repair.batch import sha


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def campaign(tmp_path):
    save(
        tmp_path / "campaign.json",
        dict(
            primary_freeze_epoch=1791997200,
            measurements_deadline_epoch=1792191600,
            capacity=dict(cpus=14, gpus=2, memory_mb=101297),
            gpu_devices={"one": {"gres": "gpu:rtx5090:1"}, "two": {"gres": "gpu:rtx2080ti:1"}},
        ),
    )
    save(
        tmp_path / "authorization.json",
        dict(
            launch_authorized=True,
            monetary_ceiling_usd=35,
            calibration_ceiling_usd=2,
            allocation="14451",
        ),
    )
    save(
        tmp_path / "ledger.json",
        dict(
            limit_worker_seconds=None,
            attempts={},
            stage_limits=dict(calibration=dict(elapsed_seconds=14400)),
        ),
    )
    for path in (
        "planning-contract.json",
        "implementation-source.json",
        "inputs/calibration-preparation.json",
        "protocols/hgt-pair-symbolic-s13.json",
    ):
        save(tmp_path / path, {})
    packet = tmp_path / "packet.json"
    save(packet, {"packet": "fixed"})
    reference = dict(path=str(packet), sha256=sha(packet))
    save(
        tmp_path / "annotations/calibration/manifest.json",
        dict(
            phase="calibration",
            authorized=True,
            cost_ceiling_usd=35,
            deadline_epoch=1791990000,
            slots=[dict(id=str(index), packet=reference) for index in range(32)],
        ),
    )
    rows = [
        dict(
            id="pair-" + str(index),
            split="train",
            ontology_names=["a", "b" + str(index)],
            status="ready_for_whole_source_qualification",
            matcher=reference,
            ontology_bindings=[reference, reference],
        )
        for index in range(13)
    ]
    rows += [
        dict(id="closed-" + str(index), split="test", ontology_names=["ekaw", str(index)])
        for index in range(8)
    ]
    save(
        tmp_path / "inputs/conference/manifest.json", dict(scheduled=21, train_count=13, rows=rows)
    )
    for index in range(3):
        save(
            tmp_path / f"calibration/study-{index}.json",
            dict(cohort="exposed_development", heldout_outcomes_opened=False),
        )
    return tmp_path


def test_first_queue_is_bound_paired_and_budgeted_without_reservation(campaign):
    before = (campaign / "ledger.json").read_bytes()
    value = launch.plan(campaign)
    assert len(value["jobs"]) == 7
    devices = [row for row in value["jobs"] if row["resources"]["gpus"]]
    assert {tuple(row["gpu_devices"]) for row in devices} == {("one",), ("two",)}
    assert all(row["budget_stages"] == [] and row["seconds"] == 900 for row in devices)
    hosted = next(row for row in value["jobs"] if row["id"] == "calibration-hosted")
    assert hosted["seconds"] == 3300 and hosted["resources"]["cpus"] == 1
    conference = next(row for row in value["jobs"] if row["id"] == "calibration-conference-train")
    assert len(conference["commands"]) == 13
    assert all("closed-" not in str(command) for command in conference["commands"])
    descriptors = [
        {key: row[key] for key in ("id", "resources", "gpu_devices", "priority")}
        for row in value["jobs"]
    ]
    for row in descriptors:
        row["resources"] = {key: row["resources"][key] for key in ("cpus", "gpus", "memory_mb")}
    registry = launch.registry_for(value, descriptors)
    followup = registry["pending_batches"][-1]
    assert set(followup["depends_on"]) == {row["id"] for row in value["jobs"]}
    assert followup["preparation_only"] and not followup["needs_user"]
    assert registry["campaign_context"]["preserve_steps"] == ["14451.0", "14451.4", "14451.extern"]
    assert (campaign / "ledger.json").read_bytes() == before
    assert not (campaign / "supervisor").exists()


def test_invalid_old_ledger_shape_is_not_silently_reset(campaign):
    path = campaign / "ledger.json"
    value = json.loads(path.read_text())
    value["attempts"] = []
    save(path, value)
    with pytest.raises(ValueError, match="empty mapping"):
        launch.plan(campaign)
    assert json.loads(path.read_text())["attempts"] == []


def test_changed_paid_packet_is_rejected_before_freezing(campaign):
    (campaign / "packet.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        launch.plan(campaign)
    assert not (campaign / "batches").exists()


def test_unknown_preparation_dependency_fails_closed():
    with pytest.raises(ValueError, match="prerequisite"):
        launch.validate_registry(
            dict(
                runs=[],
                pending_batches=[
                    dict(
                        id="next",
                        resources=dict(cpus=0, gpus=0, memory_mb=0),
                        depends_on=["missing"],
                    )
                ],
                capacity=dict(cpus=14, gpus=2, memory_mb=101297),
            ),
            "14451",
        )
