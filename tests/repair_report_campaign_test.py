"""Report boundaries: immutable evidence, complete denominators and cumulative costs."""

import json

import pytest

from tools.repair.report_campaign import (
    binding,
    check_receipt,
    checked,
    costs,
    matched_rows,
    output_binding,
    summary,
    totals,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def attempt(seconds, *, status="settled", cpus=2, gpus=0):
    value = {
        "status": status,
        "reserved_seconds": 100,
        "logical_id": "job",
        "resources": {"cpus": cpus, "gpus": gpus, "memory_mb": 16},
    }
    if status == "settled":
        value.update(elapsed_seconds=seconds, cpu_seconds=seconds / 2, peak_rss_mb=4)
    return value


@pytest.fixture
def campaign(tmp_path):
    pilot = {"old/attempt": attempt(30, cpus=4, gpus=1)}
    smoke = {"smoke/attempt": attempt(5)}
    p = save(tmp_path / "history/pilot.json", {"attempts": pilot})
    s = save(tmp_path / "history/smoke.json", {"attempts": smoke})
    inherited = {
        name: {"snapshot": str(path), "sha256": binding(path)["sha256"]}
        for name, path in [("pilot-ledger.json", p), ("smoke-ledger.json", s)]
    }
    save(
        tmp_path / "budget-lineage.json",
        {
            "inherited": inherited,
            "historical_pilot_worker_seconds": 30,
            "incremental_limit_worker_seconds": 200,
        },
    )
    rows = {
        **pilot,
        str(tmp_path / "attempts/job/001"): attempt(10),
        str(tmp_path / "attempts/job/002"): attempt(0, status="reserved"),
    }
    save(
        tmp_path / "resource-ledger.json",
        {"attempts": rows, "limit_worker_seconds": 230, "cumulative": {"worker_seconds": 140}},
    )
    return tmp_path


def test_costs_include_failures_and_smoke_once_without_counting_reservation_as_spend(campaign):
    value = costs(campaign)
    assert value["combined"]["worker_seconds"] == 45
    assert value["xr21_incremental"]["worker_seconds"] == 10
    assert value["combined"]["reserved_worker_seconds"] == 100
    assert value["unreserved_worker_seconds"] == 90
    assert value["combined"]["allocated_gpu_seconds"] == 30
    assert value["combined"]["measured_cpu_seconds"] == 22.5
    assert value["jobs"]["job"]["attempts"] == 2
    assert len(value["historical_attempts"]["pilot"]) == 1


@pytest.mark.parametrize("change", ["history", "ceiling", "cumulative"])
def test_costs_reject_reset_or_inconsistent_ledger(campaign, change):
    path = campaign / "resource-ledger.json"
    value = json.loads(path.read_text())
    if change == "history":
        value["attempts"]["old/attempt"]["elapsed_seconds"] = 0
    elif change == "ceiling":
        value["limit_worker_seconds"] += 1
    else:
        value["cumulative"]["worker_seconds"] -= 1
    save(path, value)
    with pytest.raises(ValueError):
        costs(campaign)


def test_unknowns_remain_in_denominator_and_do_not_become_zero():
    value = summary(
        [
            {
                "status": "evaluated",
                "logical_status": "VERIFIED_FEASIBLE",
                "measurement": {"semantic_benefit": 0.0},
            },
            {"status": "unavailable", "logical_status": "UNKNOWN"},
        ]
    )
    assert value["scheduled"] == 2
    assert value["known_semantic_count"] == 1
    assert value["semantic_mean_all_scheduled"] is None
    assert value["semantic_mean_known_subset"] == 0
    assert value["logical_statuses"]["UNKNOWN"] == 1


@pytest.mark.parametrize("rows", [[{"id": 1}], [{"id": 1}, {"id": 1}], [{"id": 1}, {"id": 3}]])
def test_missing_duplicate_and_unscheduled_rows_rejected(rows):
    with pytest.raises(ValueError, match="result rows"):
        matched_rows([{"id": 1}, {"id": 2}], rows, key=lambda r: r["id"])


def test_matching_restores_frozen_schedule_order():
    rows = [{"id": 2, "status": "unknown"}, {"id": 1, "status": "unavailable"}]
    assert matched_rows([{"id": 1}, {"id": 2}], rows, key=lambda r: r["id"]) == rows[::-1]


def test_receipt_nonce_and_published_output_required(tmp_path):
    report = save(tmp_path / "work/report.json", {"rows": []})
    completion = save(
        tmp_path / "attempt/completion.json",
        {
            "status": "complete",
            "exit_code": 0,
            "step_id": "1.2",
            "dispatch_nonce": "a",
            "work": str(report.parent),
        },
    )
    save(completion.with_name("outputs.json"), {"report.json": binding(report)["sha256"]})
    run = {"completion_path": str(completion), "step_id": "1.2", "dispatch_nonce": "a"}
    assert checked(output_binding(run, "report.json")) == {"rows": []}
    with pytest.raises(ValueError, match="identity"):
        check_receipt({**run, "dispatch_nonce": "b"})
    report.write_text("{}")
    with pytest.raises(ValueError, match="output"):
        output_binding(run, "report.json")


def test_reservation_and_settlement_are_separate():
    value = totals({"a": attempt(12), "b": attempt(0, status="reserved")})
    assert value["worker_seconds"] == 12
    assert value["reserved_worker_seconds"] == 100
    assert value["allocated_cpu_seconds"] == 24


@pytest.fixture
def report_fixture(tmp_path, campaign):
    from tools.repair.report_campaign import SCHEMA

    runs, arms, planned, rows, inputs = [], [], [], [], []

    def file(name, value):
        path = save(tmp_path / name, value)
        item = binding(path)
        inputs.append(item)
        return item

    def run(name, status="complete"):
        completion = file(
            f"attempts/{name}/001/completion.json",
            {
                "status": status,
                "exit_code": int(status != "complete"),
                "step_id": f"1.{len(runs)}",
                "dispatch_nonce": name,
                "work": str(tmp_path / "work" / name),
                "elapsed_seconds": 1,
            },
        )
        result = {
            "id": name + "-001",
            "logical_id": name,
            "completion_path": completion["path"],
            "step_id": f"1.{len(runs)}",
            "dispatch_nonce": name,
            "enabled": status == "complete",
        }
        runs.append(result)
        return result, completion

    for index in range(6):
        name = f"arm-{index}"
        available = index >= 2
        r, completion = run(name, "complete" if available else "failed")
        arm = {
            "id": name,
            "run_id": r["id"],
            "status": "available" if available else "unavailable",
            "completion": completion,
        }
        if available:
            arm["training_report"] = file(
                f"work/{name}/training/report.json", {"status": "complete", "epochs": 10}
            )
        arms.append(arm)
        row = {"id": name, "arm_id": name, "case_id": "case-1", "kind": "generated_pool"}
        planned.append(row)
        result = {
            **row,
            "status": "evaluated" if available else "unavailable",
            "logical_status": "VERIFIED_FEASIBLE" if available else "UNKNOWN",
            "schedule_hash": "schedule",
        }
        if available:
            result["artifact"] = file(
                f"work/eval/{name}.json",
                {
                    "row_id": name,
                    "schedule_hash": "schedule",
                    "status": "evaluated",
                    "logical_status": "VERIFIED_FEASIBLE",
                    "semantic_benefit": 1.0,
                },
            )
        rows.append(result)
    evaluation_run, _ = run("eval")
    controls_run, _ = run("controls")
    schedule = file(
        "schedule.json",
        {
            "arms": arms,
            "rows": planned,
            "cases": [{"case": {"case_id": "case-1", "structural_parent": "parent-1"}}],
            "circuit_scope": "diagnostic",
            "decoder": "enumerator",
            "source_preparation": {},
            "original_cache_hashes": {},
            "original_label_seconds": 0,
            "original_label_cpu_seconds": 0,
            "claims": "none",
        },
    )
    controls_schedule = file(
        "control-schedule.json",
        {"arms": [{"arm_id": "symbolic"}], "cases": [{"case_id": "case-1"}]},
    )
    controls = file(
        "control-report.json",
        {
            "schedule_sha256": controls_schedule["sha256"],
            "rows": [
                {
                    "case_id": "case-1",
                    "arm_id": "symbolic",
                    "status": "deferred",
                    "logical_status": "UNKNOWN",
                }
            ],
        },
    )
    manifest = {
        "schema": SCHEMA,
        "campaign": str(campaign),
        "plan": file("plan.json", {"case_counts": {"test": 1}}),
        "registry": file("registry-input.json", {"runs": runs}),
        "evaluation_schedule": schedule,
        "evaluation": file("eval-report.json", {"rows": rows}),
        "controls": controls,
        "controls_schedule": controls_schedule,
        "preparation_gate": file("gate.json", {"status": "complete"}),
        "dependencies": [evaluation_run, controls_run],
        "planned_arm_ids": [a["id"] for a in arms],
        "deferred_scope": ["test controls"],
        "inputs": inputs,
    }
    path = save(tmp_path / "manifest.json", manifest)
    return path, tmp_path / "report-output", runs


def test_report_keeps_unavailable_arms_and_deferrals(report_fixture):
    from tools.repair.report_campaign import report

    manifest, output, _ = report_fixture
    result = report(manifest, output)
    assert len(result["arms"]) == 6
    assert result["primary_generated_pool"]["scheduled"] == 6
    assert result["primary_generated_pool"]["statuses"]["unavailable"] == 2
    assert result["primary_generated_pool"]["semantic_mean_all_scheduled"] is None
    assert result["common_inventory_diagnostic"]["statuses"]["deferred"] == 1
    assert result["gates"]["G0"] == "not_established"
    assert result["finite_scope_status"] == "accounted_pending_report_receipt_and_final_costs"


def test_closure_requires_settled_report_and_preserves_immutable_output(report_fixture, campaign):
    from tools.repair.report_campaign import close, report

    manifest, output, runs = report_fixture
    report(manifest, output)
    report_path = output / "report.json"
    original = report_path.read_bytes()
    completion = save(
        campaign / "attempts/report/001/completion.json",
        {
            "status": "complete",
            "exit_code": 0,
            "step_id": "1.99",
            "dispatch_nonce": "report",
            "work": str(output),
            "elapsed_seconds": 2,
        },
    )
    save(completion.with_name("outputs.json"), {"report.json": binding(report_path)["sha256"]})
    runs.append(
        {
            "id": "xr21-t2-report-001",
            "logical_id": "report",
            "completion_path": str(completion),
            "step_id": "1.99",
            "dispatch_nonce": "report",
        }
    )
    save(campaign / "supervisor/registry.json", {"runs": runs, "pending_batches": []})
    with pytest.raises(ValueError, match="unsettled"):
        close(campaign, campaign / "final")
    ledger_path = campaign / "resource-ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["attempts"][str(campaign / "attempts/job/002")] = attempt(3)
    ledger["attempts"][str(completion.parent)] = attempt(2)
    ledger["cumulative"]["worker_seconds"] = 45
    save(ledger_path, ledger)
    result = close(campaign, campaign / "final")
    assert result["remaining_work_status"] == "terminal"
    final_costs = checked(result["costs"])
    assert final_costs["xr21_incremental"]["worker_seconds"] == 15
    assert final_costs["combined"]["worker_seconds"] == 50
    assert final_costs["combined"]["reserved_worker_seconds"] == 0
    assert report_path.read_bytes() == original
