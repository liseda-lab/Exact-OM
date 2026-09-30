"""Corrective REV-01/02 transport cardinality and responsive-event regressions."""

import time

from exact.repair.workers import bounded_call, emit_event


def many_events(count):
    for index in range(count):
        emit_event(("coverage", index))
    return count


def one_event():
    emit_event(("coverage", 0))
    return "done"


def stalled_handler(event):
    time.sleep(2)
    return True


def test_rev01_large_event_count_uses_bounded_durable_stream(record_property):
    outcome = bounded_call(many_events, 10001, timeout=30)
    assert outcome.status == "complete", outcome.detail
    record_property("transported_events", len(outcome.events))
    for key, value in outcome.resource_usage:
        record_property(key, value)
    assert outcome.value == 10001 and len(outcome.events) == 10001
    assert list(outcome.events)[-1] == ("coverage", 10000)
    assert dict(outcome.resource_usage)["peak_buffered_event_bytes"] <= 16 * 1024 * 1024


def test_rev02_blocked_validation_keeps_deadline_responsive():
    started = time.monotonic()
    outcome = bounded_call(one_event, timeout=0.8, event_handler=stalled_handler)
    assert time.monotonic() - started < 1.4  # deadline + cleanup + Linux scheduling allowance
    assert outcome.status == "timeout" and not outcome.events


def controlled_verify(problem, assignment):
    """Actual kernel producer, controlled complete adapter, no native scale claim."""
    import dataclasses

    from exact.repair import kernel
    from exact.repair import owl as adapter
    from exact.repair.records import canonical_hash

    class Controlled:
        def __init__(self, **kwargs):
            pass

        def check_theory(self, snapshot, monitored, *, on_complete, **kwargs):
            support = adapter.SupportReport(
                "hermit",
                "0.2.1",
                "controlled",
                "fixture",
                True,
                True,
                ("Class", "Declaration"),
                ("consistency", "class_satisfiability"),
            )
            rows = [adapter.ObligationResult("consistency", "consistency", True, complete=True)]
            rows.extend(
                adapter.ObligationResult(
                    "class_satisfiability",
                    adapter._query_id(c),
                    True,
                    complete=True,
                    class_iri=c.iri.value,
                )
                for c in monitored
            )
            for row in rows:
                on_complete(row, support)
            return adapter.CheckReport(
                "VERIFIED_FEASIBLE",
                "complete_supported_fragment",
                snapshot.logical_fingerprint.hex,
                tuple(rows),
                dataclasses.replace(
                    support,
                    query_support=tuple((row.kind, row.query_id, row.complete) for row in rows),
                ),
            )

    adapter.OwlVerifier = Controlled
    return kernel.verify_theory((), (), problem.policy, canonical_hash(assignment))


def _problem(count=1):
    import pyowl_core as owl

    from exact.repair.records import PolicyV3, RepairInputV3

    return RepairInputV3(
        (), (), PolicyV3(tuple(owl.Class(owl.IRI(f"urn:review:public:{i}")) for i in range(count)))
    )


def test_rev01_real_producer_large_coverage_and_compact_final_report(record_property):
    import pickle

    from exact.repair.kernel import _valid_report, _VerificationStream
    from exact.repair.records import CompactVerificationReportV3

    problem = _problem(10001)
    outcome = bounded_call(
        controlled_verify, problem, (), timeout=60, event_handler=_VerificationStream(problem, ())
    )
    assert outcome.status == "complete", outcome.detail
    assert isinstance(outcome.value, CompactVerificationReportV3)
    assert outcome.value.authorizes and _valid_report(problem, (), outcome.value)
    assert outcome.value.coverage_count == 10002 and len(outcome.events) == 10002
    record_property("final_report_bytes", len(pickle.dumps(outcome.value)))
    record_property("completed_obligations", outcome.value.coverage_count)
    record_property("adapter_scope", "controlled complete adapter; actual kernel producer")
    for key, value in outcome.resource_usage:
        record_property(key, value)
    assert len(pickle.dumps(outcome.value)) < 65536
    assert dict(outcome.resource_usage)["peak_buffered_event_bytes"] < 16 * 1024 * 1024


def malformed_stream(problem, kind):
    import dataclasses

    from exact.repair import kernel, workers
    from exact.repair.records import (
        ObligationV2,
        VerificationEventV3,
        VerificationReportV3,
        canonical_hash,
    )

    names = kernel.expected_queries((), problem.policy)
    capability = {
        "reasoner": "hermit",
        "package_version": "0.2.1",
        "backend": "controlled",
        "input_supported": True,
        "complete_imports": True,
    }
    events = [
        VerificationEventV3(
            canonical_hash(()),
            canonical_hash(((), ())),
            problem.policy.content_hash,
            i,
            ObligationV2(name, "pass", True),
            "hermit/0.2.1:controlled",
            canonical_hash(capability),
            capability=tuple(capability.items()),
        )
        for i, name in enumerate(names)
    ]
    if kind in {"conflict", "hang", "broken"}:
        events[0] = dataclasses.replace(events[0], obligation=ObligationV2(names[0], "fail", True))
    if kind == "malformed":
        events[0] = dataclasses.replace(events[0], theory_hash="wrong")
    if kind == "out_of_order":
        events[0] = dataclasses.replace(events[0], sequence=1)
    emit_event(events[0])
    if kind == "hang":
        time.sleep(30)
    if kind == "broken":
        workers._EVENT_CHANNEL.send_bytes(b"broken frame")
        time.sleep(30)
    if kind == "duplicate":
        emit_event(dataclasses.replace(events[0], sequence=1))
    elif kind != "missing":
        emit_event(events[1])
    return VerificationReportV3(
        canonical_hash(()),
        canonical_hash(((), ())),
        problem.policy.content_hash,
        "VERIFIED_FEASIBLE",
        "complete_supported_fragment",
        tuple(ObligationV2(name, "pass", True) for name in names),
        expected_obligations=names,
    )


def test_rev01_invalid_coverage_never_manufactures_acceptance():
    from exact.repair.kernel import _VerificationStream

    for kind in ("missing", "duplicate", "malformed", "out_of_order", "conflict"):
        problem = _problem()
        result = bounded_call(
            malformed_stream,
            problem,
            kind,
            timeout=5,
            event_handler=_VerificationStream(problem, ()),
        )
        assert not getattr(result.value, "authorizes", False), kind
        if kind == "conflict":
            assert result.value.verdict == "VERIFIED_INFEASIBLE"
            assert "integrity discrepancy" in result.value.detail


def test_rev01_committed_failure_survives_hang_and_broken_frame():
    from exact.repair.kernel import _VerificationStream

    for kind in ("hang", "broken"):
        problem = _problem()
        result = bounded_call(
            malformed_stream,
            problem,
            kind,
            timeout=0.8,
            event_handler=_VerificationStream(problem, ()),
        )
        assert result.status in {"timeout", "error"}
        assert result.event_failure.obligation.verdict == "fail"
        assert len(result.events) == 1


def test_rev02_stalled_persistence_is_killable_and_uncommitted_rows_are_invisible(
    monkeypatch, tmp_path
):
    import hashlib
    import os
    import pickle

    from exact.repair import workers

    original = workers._commit_event_batch
    marker = tmp_path / "transaction-started"

    def blocked(directory, prior, values):
        if prior.batches == 0:
            return original(directory, prior, values)
        payload = pickle.dumps((prior.batches, prior.digest, values), protocol=5)
        conn = workers._journal_connection(directory, os.getpid())
        conn.execute(
            "INSERT INTO events VALUES (?,?,?,?)",
            (
                prior.batches,
                payload,
                hashlib.sha256(payload).hexdigest(),
                prior.event_count + len(values),
            ),
        )
        marker.write_text("uncommitted")
        time.sleep(30)

    monkeypatch.setattr(workers, "_commit_event_batch", blocked)
    started = time.monotonic()
    result = bounded_call(many_events, 2, timeout=0.8)
    assert time.monotonic() - started < 1.4
    assert result.status == "timeout" and marker.exists()
    assert list(result.events) == [("coverage", 0)]
    assert list(workers.committed_events(result.events.directory)) == [("coverage", 0)]


def test_rev02_resource_monitor_runs_during_blocked_callback(monkeypatch, tmp_path):
    from exact.repair import workers

    marker = tmp_path / "validation-started"

    def blocking(event):
        marker.write_text("ready")
        time.sleep(30)
        return True

    original = workers._resident_tree_bytes
    monkeypatch.setattr(
        workers, "_resident_tree_bytes", lambda pid: 10**12 if marker.exists() else original(pid)
    )
    result = bounded_call(one_event, timeout=4, memory_mb=100000, event_handler=blocking)
    assert marker.exists() and result.status == "memory_limit"


def busy_handler(event):
    value = 1
    while True:
        value = (value * 17 + 11) % 65537


def test_rev02_cpu_budget_includes_event_validator_work():
    result = bounded_call(one_event, timeout=5, cpu_seconds=0.5, event_handler=busy_handler)
    assert result.status == "cpu_limit" and result.cleanup_complete
    assert dict(result.resource_usage)["cpu_seconds"] >= 0.5
    assert not result.events


def stuck_ledger_write(path, record):
    time.sleep(30)


def test_rev02_search_ledger_write_cannot_bypass_deadline(monkeypatch, tmp_path):
    import dataclasses

    from exact.repair import kernel
    from exact.repair.records import BudgetsV2, ObjectiveV3

    original = kernel.bounded_call

    def intercept(function, *args, **kwargs):
        if function is kernel._atomic_record:
            function = stuck_ledger_write
        return original(function, *args, **kwargs)

    monkeypatch.setattr(kernel, "bounded_call", intercept)
    monkeypatch.setattr(kernel, "_NATIVE_BOUNDED_CALL", intercept)
    problem = dataclasses.replace(
        _problem(), budgets=BudgetsV2(total_seconds=2, verification_seconds=0.4, solver_seconds=0.4)
    )
    started = time.monotonic()
    result = kernel.repair(
        problem, ObjectiveV3(()), diagnose=False, ledger_path=tmp_path / "search.json"
    )
    assert time.monotonic() - started < 1.2
    assert result.assignment is None
    assert any("ledger persistence: timeout" in failure for failure in result.failures)
    assert not (tmp_path / "search.json").exists()


def test_rev02_durable_completed_report_recovers_without_reverification(tmp_path):
    import dataclasses
    import json

    from exact.repair import kernel
    from exact.repair.records import (
        BudgetsV2,
        ObjectiveV3,
        PendingAssignmentV2,
        RecoverySearchLedgerV3,
    )

    problem = dataclasses.replace(
        _problem(), budgets=BudgetsV2(total_seconds=20, verification_seconds=5, max_checks=1)
    )
    directory = str(tmp_path / "events")
    outcome = bounded_call(
        controlled_verify,
        problem,
        (),
        timeout=5,
        event_handler=kernel._VerificationStream(problem, ()),
        event_directory=directory,
    )
    assert outcome.value.authorizes and outcome.event_journal == directory
    objective = ObjectiveV3(())
    pending = PendingAssignmentV2(
        (), 0, kernel._unknown(problem, (), "interrupted before publication")
    )
    saved = RecoverySearchLedgerV3(
        problem.content_hash,
        objective.content_hash,
        problem.policy.content_hash,
        pending=(pending,),
        elapsed_seconds=1,
        checks=1,
        upper_bound=0,
        work_upper_bound=0,
        event_journals=(((), directory),),
    )
    path = tmp_path / "search.json"
    path.write_text(json.dumps(saved.to_dict()))
    result = kernel.repair(problem, objective, diagnose=False, ledger_path=path, resume=True)
    assert result.assignment == () and result.verification.authorizes
    assert result.checks == 1 and result.elapsed_seconds > saved.elapsed_seconds
    assert result.ledger.pending == ()
    assert result.ledger.event_journals == saved.event_journals


def test_rev02_replay_rejects_tampered_committed_result(tmp_path):
    import sqlite3

    from exact.repair import kernel

    problem = _problem()
    directory = str(tmp_path / "events")
    result = bounded_call(
        controlled_verify,
        problem,
        (),
        timeout=5,
        event_handler=kernel._VerificationStream(problem, ()),
        event_directory=directory,
    )
    assert result.value.authorizes
    with sqlite3.connect(tmp_path / "events" / "events.sqlite") as connection:
        connection.execute("UPDATE completion SET digest=? WHERE id=1", ("0" * 64,))
    replay = bounded_call(kernel._replay_journal, problem, (), directory, timeout=5)
    assert replay.status == "error" and "invalid committed final result" in replay.detail
