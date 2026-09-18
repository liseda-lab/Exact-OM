"""The campaign CLI loads credentials before starting expensive work."""

import os
import subprocess
import sys

import pytest

from exact.experiments import campaign
from tools import run_experiment


def test_key_file_reaches_campaign_workers_without_entering_arguments_or_output(
    tmp_path, monkeypatch, capsys
):
    secret = "fixture-openrouter-key"
    key_file = tmp_path / "api_key"
    key_file.write_text(f"  {secret}\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "old-key")
    calls = []

    def execute(*args, **kwargs):
        calls.append((args, kwargs))
        assert os.environ["OPENROUTER_API_KEY"] == secret
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import os; assert os.environ['OPENROUTER_API_KEY'] == 'fixture-openrouter-key'",
            ],
            check=True,
        )
        return 0

    monkeypatch.setattr(campaign, "execute_campaign", execute)
    assert (
        run_experiment.main(
            ["--campaign", "lock.yaml", "--stage", "screen", "--api-key-file", str(key_file)]
        )
        == 0
    )
    assert len(calls) == 1
    assert secret not in repr(calls)
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert sorted(path.name for path in tmp_path.iterdir()) == ["api_key"]


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_missing_or_empty_key_file_fails_before_campaign_work(
    content, tmp_path, monkeypatch, capsys
):
    key_file = tmp_path / "api_key"
    if content is not None:
        key_file.write_text(content, encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "existing-key")

    def unexpected(*args, **kwargs):
        pytest.fail("Campaign work must not start with an invalid credential file")

    monkeypatch.setattr(campaign, "execute_campaign", unexpected)
    output = tmp_path / "output"
    assert (
        run_experiment.main(
            [
                "--campaign",
                "lock.yaml",
                "--stage",
                "screen",
                "--api-key-file",
                str(key_file),
                "--output-root",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()
    assert os.environ["OPENROUTER_API_KEY"] == "existing-key"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "run_experiment:" in captured.err
    assert "existing-key" not in captured.err


def test_existing_worker_environment_is_preserved_without_key_file(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "inherited-key")

    def execute(*args, **kwargs):
        assert os.environ["OPENROUTER_API_KEY"] == "inherited-key"
        return 0

    monkeypatch.setattr(campaign, "execute_campaign", execute)
    assert run_experiment.main(["--campaign", "lock.yaml", "--stage", "screen"]) == 0
