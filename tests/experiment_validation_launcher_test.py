"""Bounded validation preserves role separation and refuses unaffordable follow-ups."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from exact.core.entities.configs.config import ConfigModel
from exact.experiments.campaign import CaseBinding, InputBinding
from exact.utils.provenance import sha256_file
from tools.run_experiment_validation import (
    bounded_training,
    forecast,
    forward_stop,
    validation_config,
)


def case_fixture(tmp_path):
    def binding(name, text):
        path = tmp_path / name
        path.write_text(text)
        return InputBinding(path=path, sha256=sha256_file(path))

    return CaseBinding(
        task="fixture",
        role="development",
        selection_reason="fixed before outcomes",
        source=binding("source.owl", "source"),
        target=binding("target.owl", "target"),
        source_universe=binding("sources.txt", "\n".join(f"dev:{i}" for i in range(320)) + "\n"),
        references={
            "train": binding(
                "train.ref.tsv",
                "SrcEntity\tTgtEntity\n" + "".join(f"train:{i}\tt:{i}\n" for i in range(70)),
            ),
            "valid": binding("valid.ref.tsv", "SrcEntity\tTgtEntity\ndev:0\tt:0\n"),
        },
        candidates={
            "train": binding(
                "train.cands.tsv",
                "Src\tTgt\tconfirmed_label\n"
                + "".join(f"train:{i}\tt:{i}\t1\ntrain:{i}\twrong:{i}\t0\n" for i in range(70)),
            ),
            "valid": binding(
                "valid.cands.tsv", "SrcEntity\tTgtEntity\tTgtCandidates\ndev:0\t\t['t:0']\n"
            ),
        },
        negative_policy="confirmed_only",
    )


def test_training_bound_precedes_features_and_retains_whole_groups(tmp_path):
    case = case_fixture(tmp_path)
    first = bounded_training(case, tmp_path, tmp_path / "output")
    second = bounded_training(case, tmp_path, tmp_path / "output")
    assert first == second
    pool = pd.read_csv(first["pool"], sep="\t")
    assert pool.Src.nunique() == 64 and len(pool) == 128
    assert set(pool.groupby("Src").size()) == {2}
    assert set(pool.confirmed_label) == {0, 1}
    assert not set(first["sources"]) & set(case.source_universe.path.read_text().splitlines())


def test_probe_is_explicit_variant_and_production_keeps_frozen_encoders_and_llm(tmp_path):
    case = case_fixture(tmp_path)
    train = bounded_training(case, tmp_path, tmp_path / "output")
    baseline = ConfigModel().model_dump(mode="json", by_alias=True)
    baseline["data"]["refs"]["test"] = "/never-open-private-test.tsv"
    probe, _ = validation_config(
        baseline,
        case,
        tmp_path,
        train,
        mode="global_alignment",
        cap=64,
        hosted=False,
        fit=False,
        evaluate=False,
    )
    assert probe["data"]["refs"] == {}
    assert probe["data"]["train_candidates"] is None
    assert probe["dataset"]["verbalization_mode"] == "deterministic"
    assert probe["pipeline"][0]["params"]["use_llm"] is False
    production, _ = validation_config(
        baseline,
        case,
        tmp_path,
        train,
        mode="local_ranking",
        cap=300,
        hosted=True,
        fit=True,
        evaluate=True,
    )
    assert production["pipeline"] == baseline["pipeline"]
    assert production["dataset"]["verbalization_mode"] == baseline["dataset"]["verbalization_mode"]
    assert set(production["data"]["refs"]) == {"train", "valid"}
    assert production["data"]["train_candidates"] == train["pool"]
    assert production["data"]["execution_mode"] == "local_ranking"
    assert production["data"]["candidate_provenance"] == "benchmark_supplied"
    assert "/never-open-private-test.tsv" not in json.dumps(production)
    for key in (
        "lexical_model_name",
        "lexical_model_revision",
        "context_model_name",
        "context_model_revision",
    ):
        assert probe["pipeline"][0]["params"][key] == baseline["pipeline"][0]["params"][key]


def test_forecast_uses_real_warm_work_and_requires_headroom_for_hosted_calls():
    rows = [
        {"id": name, "wall_seconds": seconds}
        for name, seconds in (("cold64", 100), ("warm64", 50), ("fit64", 30))
    ]
    rows.append(
        {
            "id": "hosted20",
            "wall_seconds": 10,
            "new_usage": {"attempts": 60, "prompt_tokens": 10000, "completion_tokens": 1000},
        }
    )
    args = {
        "elapsed": 190,
        "max_seconds": 41400,
        "requests": 60,
        "tokens": 11000,
        "limits": {"requests": 2000, "tokens": 3200000},
    }
    estimate = forecast(rows, **args)
    assert estimate["remaining_requests"] == 2700
    assert estimate["fits_limits"] is False
    rows[-1]["new_usage"]["attempts"] = 2
    assert forecast(rows, **args)["fits_limits"] is True


@pytest.mark.parametrize("fit", [False, True])
@pytest.mark.parametrize("mode", ["global_alignment", "local_ranking"])
def test_no_evaluation_resolves_inputs_without_reference_fallback(tmp_path, monkeypatch, fit, mode):
    from exact.core.actions import alignment

    case = case_fixture(tmp_path)
    training = bounded_training(case, tmp_path, tmp_path / "output")
    baseline = ConfigModel().model_dump(mode="json", by_alias=True)
    baseline["data"].update(
        track="must-not-materialize",
        task="private-task",
        descriptor="/must-not-open-descriptor.yaml",
        reference_role="test",
        refs={"test": "/must-not-open-private-gold.tsv"},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("No evaluation stage may resolve an inherited dataset track")

    monkeypatch.setattr(alignment, "get_track", forbidden)
    monkeypatch.setattr(alignment, "provider_from_descriptor", forbidden)
    config, _ = validation_config(
        baseline,
        case,
        tmp_path,
        training,
        mode=mode,
        cap=64,
        hosted=False,
        fit=fit,
        evaluate=False,
    )
    resolved = alignment.resolve_alignment_inputs(
        configs=ConfigModel.from_mapping(config, warn_v1=False)
    )
    assert resolved.full_reference is None
    assert resolved.training_reference == (Path(training["reference"]) if fit else None)
    assert resolved.track_provenance is None
    assert resolved.candidates == (
        case.candidates["valid"].path if mode == "local_ranking" else None
    )


def test_stop_interrupts_worker_before_trainer_checkpoint_poll(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import signal,time; "
        "signal.signal(signal.SIGINT, lambda *_: exit(130)); "
        "print('ready', flush=True); time.sleep(30)",
    ]
    process = subprocess.Popen(command, start_new_session=True, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        stop = tmp_path / "STOP"
        assert forward_stop(process, stop, False) is False
        stop.write_text("user requested stop\n")
        assert forward_stop(process, stop, False) is True
        assert process.wait(timeout=3) == 130
        assert forward_stop(process, stop, True) is True
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
