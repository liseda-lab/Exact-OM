"""Real loopback CLI startup and HTTP probes include admission and always stop the child."""

import json
from pathlib import Path

from exact_inspect.contracts import file_hash
from tests.explanation_preparation_test import prepared  # noqa: F401,F811
from tools.verify_explanation_runtime import verify_runtime


def test_fresh_cli_http_reads_measure_child_and_preserve_prepared_bytes(
    prepared, tmp_path  # noqa: F811
):
    _, package, _, _ = prepared
    before = {str(path): file_hash(path) for path in package.parent.rglob("*") if path.is_file()}
    output = tmp_path / "http-verification"
    report = verify_runtime(
        package,
        output,
        hardware_profile="synthetic-test",
        storage_profile="temporary-test",
        startup_timeout=30,
        request_timeout=5,
    )
    assert report["failure"] is None, (report, (output / "server.log").read_text())
    assert report["checks"]["bounded_reads"] == "passed"
    assert report["checks"]["no_runtime_imports"] == "passed"
    assert report["measurements"]["cold_health_seconds"] > 0
    assert report["measurements"]["cold_entity_queries"][0]["status"] == 200
    assert report["measurements"]["cold_entity_queries"][0]["route"] == "/api/v1/entity-context"
    selected = report["query_plan_selection"]
    assert selected["profiles_inspected"] == 2
    assert {row["entity"]["iri"] for row in selected["entities"]} == {"urn:A", "urn:B"}
    assert all("prepared_entity_profile" in row["origins"] for row in selected["entities"])
    assert any(
        sample["route"] == "/api/v1/entity-context" and sample["query"]["iri"] == "urn:B"
        for sample in report["samples"]
    )  # The prepared sparse class is exercised.
    assert report["measurements"]["query_count"] == len(report["samples"]) == 100
    assert report["measurements"]["readers"] == 4
    assert {sample["index"] for sample in report["samples"]} == set(range(100))
    assert any(sample["route"] == "/api/v1/entity-context" for sample in report["samples"])
    assert any(sample["route"].startswith("/api/v1/explanations/") for sample in report["samples"])
    assert report["child"]["host"] == "127.0.0.1"
    assert "exact_inspect.cli" in report["command"]
    assert report["child"]["termination"] == "terminated"
    if Path("/proc/self/status").exists():
        assert report["measurements"]["peak_rss_bytes"] > 0
        assert not Path(f'/proc/{report["child"]["pid"]}').exists()
    assert json.loads((output / "http-runtime.json").read_text()) == report
    assert before == {path: file_hash(Path(path)) for path in before}


def test_failed_server_start_still_publishes_receipt_and_reaps_process(tmp_path):
    report = verify_runtime(tmp_path / "missing.json", tmp_path / "failed", startup_timeout=10)
    assert report["status"] == "failed"
    assert report["checks"]["cold_service_health"] == "failed"
    assert report["failure"]["phase"] == "startup"
    assert report["child"]["termination"] == "already_exited"
    assert report["child"]["returncode"] != 0
    assert (tmp_path / "failed" / "http-runtime.json").is_file()
    if Path("/proc/self/status").exists():
        assert not Path(f'/proc/{report["child"]["pid"]}').exists()
