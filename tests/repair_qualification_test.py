"""Native qualification never reuses missing or changed positive evidence."""

import pytest

from exact.repair.api import write_artifact
from tools.repair.batch import sha
from tools.repair.qualification import checked_result


def test_native_suite_runner_resumes_without_repeating_passed_work(tmp_path, monkeypatch):
    from tools.repair import qualification

    source = tmp_path / "source"
    tools = source / "tools/repair"
    tools.mkdir(parents=True)
    monkeypatch.setattr(qualification, "__file__", str(tools / "qualification.py"))
    test = source / "test_native.py"
    test.write_text(
        "from pathlib import Path\n"
        "def test_native():\n"
        "    path = Path(__file__).with_name('executions')\n"
        "    path.write_text(path.read_text() + 'x' if path.exists() else 'x')\n"
        "    assert 1 + 1 == 2\n"
    )
    manifest = tmp_path / "manifest.json"
    write_artifact(manifest, {"groups": [{"id": "native", "tests": ["test_native.py"]}]})
    output = tmp_path / "results"
    first = qualification.run(manifest, output)
    assert qualification.run(manifest, output) == first
    assert (source / "executions").read_text() == "x"
    assert first["groups"][0]["counts"]["tests"] == 1
    assert all(value != "passed" for value in first["gates"].values())
    test.write_text(test.read_text() + "\n# Changed dependency\n")
    with pytest.raises(ValueError, match="dependencies changed"):
        qualification.run(manifest, output)


def test_qualification_resume_is_bound_to_dependencies_and_evidence(tmp_path):
    log, junit, receipt = (tmp_path / name for name in ("run.log", "run.xml", "receipt.json"))
    log.write_text("one passed")
    junit.write_text('<testsuites><testsuite tests="1" failures="0"/></testsuites>')
    value = dict(
        identity="source-and-tests",
        status="complete",
        log=dict(path=str(log), sha256=sha(log)),
        junit=dict(path=str(junit), sha256=sha(junit)),
    )
    write_artifact(receipt, value)
    assert checked_result(receipt, "source-and-tests") == value
    with pytest.raises(ValueError, match="dependencies changed"):
        checked_result(receipt, "different-code")
    value["status"] = "failed"
    write_artifact(receipt, value)
    assert checked_result(receipt, "source-and-tests") is None
    value["status"] = "complete"
    write_artifact(receipt, value)
    junit.write_text("corrupt")
    with pytest.raises(ValueError, match="evidence changed"):
        checked_result(receipt, "source-and-tests")
