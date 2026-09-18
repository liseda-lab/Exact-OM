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
    from exact.experiments.rationale_policy import apply_rationale_policy

    assert production["pipeline"] == apply_rationale_policy(baseline)["pipeline"]
    assert production["pipeline"][0]["params"]["generate_llm_rationales"] is False
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


def test_saved_cold_measurement_alone_blocks_unaffordable_continuation():
    from tools.run_experiment_validation import cold_budget_bound

    bound = cold_budget_bound(
        {"wall_seconds": 25927.17662280702, "new_worker_calls": 0},
        elapsed=26003.846488725016,
        limits={"seconds": 41400, "soft_seconds": 39600},
    )
    assert bound["fits_limits"] is False
    assert bound["remaining_seconds_lower_bound"] == pytest.approx(155563.05973684211)
    assert "warm64" in bound["unmeasured"]


def test_resumed_validation_retains_budget_and_never_restarts_cold(tmp_path, monkeypatch):
    from argparse import Namespace
    from types import SimpleNamespace

    from exact.core.entities.configs import yaml_io
    from exact.experiments import campaign, harness
    from tools import run_experiment_validation as launcher
    from tools import validation_resume

    case = case_fixture(tmp_path)
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text("immutable campaign")
    lock = SimpleNamespace(
        cases={"D0": case},
        base_config=tmp_path / "baseline.yaml",
        openrouter_profile={},
        baseline_id="fixture",
        model_lock=case.source,
    )
    monkeypatch.setattr(campaign, "load_campaign", lambda *_: (lock, None))
    monkeypatch.setattr(campaign, "validate_baseline", lambda *_: None)
    monkeypatch.setattr(campaign, "openrouter_only", lambda base, *_: base)
    monkeypatch.setattr(yaml_io, "load_yaml_mapping", lambda *_: {})
    monkeypatch.setattr(harness, "_bind_model_lock_revisions", lambda config, *a, **kw: config)
    monkeypatch.setattr(launcher, "bounded_training", lambda *a: {})
    monkeypatch.setattr(launcher, "validation_config", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(launcher, "node_info", lambda *_: {})
    monkeypatch.setattr(launcher.signal, "signal", lambda *a: None)
    totals = {"attempts": 0, "prompt_tokens": 0, "completion_tokens": 0}
    monkeypatch.setattr(launcher, "ledger_totals", lambda *_: totals)
    prior = 26003.846488725016
    row = {
        "id": "cold64",
        "status": "complete",
        "wall_seconds": 25927.17662280702,
        "new_worker_calls": 0,
        "imported": True,
        "output_dir": str(tmp_path / "saved/run"),
    }
    monkeypatch.setattr(validation_resume, "adopt_cold_probe", lambda *a, **kw: (row, prior))
    clock = iter((100.0, 110.0, 130.0))
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(clock))

    def forbidden(*args, **kwargs):
        raise AssertionError("No worker may run for this unaffordable continuation")

    monkeypatch.setattr(harness, "execute_cell", forbidden)
    args = Namespace(
        api_key_file=None,
        output_root=tmp_path / "resumed",
        campaign=campaign_path,
        max_seconds=41400,
        ram_gb=56,
        requests_cap=2000,
        tokens_cap=3200000,
        resume_from=tmp_path / "previous",
        execute=True,
        scratch_root=None,
        skip_hosted=False,
    )
    assert launcher.execute(args) == 2
    report = json.loads((args.output_root / "report.json").read_text())
    assert report["status"] == "blocked_budget"
    assert report["elapsed_seconds"] == pytest.approx(prior + 30)
    assert report["stages"] == [row]
    assert report["hosted_usage"] == totals
    assert report["resume"]["previous_elapsed_seconds"] == prior


@pytest.mark.parametrize("requested", [False, True])
def test_g0_narratives_require_explicit_opt_in(tmp_path, requested):
    case = case_fixture(tmp_path)
    training = bounded_training(case, tmp_path, tmp_path / "output")
    base = ConfigModel().model_dump(mode="json", by_alias=True)
    base["pipeline"][0]["params"]["generate_llm_rationales"] = True
    resolved, _ = validation_config(
        base,
        case,
        tmp_path,
        training,
        mode="global_alignment",
        cap=20,
        hosted=True,
        fit=False,
        evaluate=False,
        generate_rationales=requested,
    )
    assert resolved["pipeline"][0]["params"]["generate_llm_rationales"] is requested
    assert base["pipeline"][0]["params"]["generate_llm_rationales"] is True


def test_unlimited_wall_still_enforces_hosted_allowances():
    from tools.run_experiment_validation import cold_budget_bound

    limits = {"seconds": None, "soft_seconds": None, "requests": 100000, "tokens": 32000000}
    rows = [{"id": name, "wall_seconds": 1e9} for name in ("cold64", "warm64", "fit64")]
    assert cold_budget_bound(rows[0], elapsed=1e9, limits=limits)["fits_limits"]
    rows.append(
        {
            "id": "hosted20",
            "wall_seconds": 1e9,
            "new_usage": {"attempts": 133, "prompt_tokens": 17455, "completion_tokens": 2324},
        }
    )
    kwargs = dict(elapsed=1e9, max_seconds=None, requests=655, tokens=396987, limits=limits)
    estimate = forecast(rows, **kwargs)
    assert estimate["fits_limits"]
    assert estimate["remaining_seconds"] > 1e9
    assert json.loads(json.dumps(estimate))["limits"]["soft_seconds"] is None
    limits["requests"] = 2000
    assert not forecast(rows, **kwargs)["fits_limits"]
    limits["requests"] = 100000
    limits["tokens"] = 400000
    assert not forecast(rows, **kwargs)["fits_limits"]


@pytest.mark.parametrize("extra,expected", [([], 43200), (["--no-time-limit"], None)])
def test_cli_explicit_unlimited_wall_and_revised_hosted_caps(monkeypatch, extra, expected):
    from tools import run_experiment_validation as launcher

    captured = {}

    def execute(args):
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(launcher, "execute", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validation",
            "--campaign",
            "campaign.yaml",
            "--output-root",
            "new",
            "--requests-cap",
            "100000",
            "--tokens-cap",
            "32000000",
            *extra,
        ],
    )
    assert launcher.main() == 0
    assert captured["max_seconds"] is expected or captured["max_seconds"] == expected
    assert captured["requests_cap"] == 100000
    assert captured["tokens_cap"] == 32000000
    assert captured["generate_rationales"] is False


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "roundoff",
        "resumed_globals",
        "reboot_replay",
        "reboot_worker",
        "reboot_api",
        "reboot_missing_cache",
        "changed_mapping",
        "changed_identity",
        "changed_relation",
        "changed_metrics",
        "changed_local_mapping",
        "global_request",
        "local_request",
        "changed_cache_score",
    ],
)
def test_full_validation_recovery_flow_rejects_changed_or_paid_replay(tmp_path, monkeypatch, fault):
    """Exercise every launcher stage with deterministic, offline cell results."""
    from argparse import Namespace
    from types import SimpleNamespace

    from exact.core.entities.configs import yaml_io
    from exact.experiments import campaign, harness
    from tools import run_experiment_validation as launcher

    case = case_fixture(tmp_path)
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text("immutable development campaign")
    lock = SimpleNamespace(
        cases={"D0": case},
        base_config=tmp_path / "baseline.yaml",
        openrouter_profile={},
        baseline_id="fixture",
        model_lock=case.source,
    )
    base = ConfigModel().model_dump(mode="json", by_alias=True)
    base["data"]["refs"]["test"] = "/never-open-private-test.tsv"
    base["pipeline"][0]["params"]["generate_llm_rationales"] = True
    monkeypatch.setattr(campaign, "load_campaign", lambda *_: (lock, None))
    monkeypatch.setattr(campaign, "validate_baseline", lambda *_: None)
    monkeypatch.setattr(campaign, "openrouter_only", lambda base, *_: base)
    monkeypatch.setattr(yaml_io, "load_yaml_mapping", lambda *_: base)
    monkeypatch.setattr(harness, "_bind_model_lock_revisions", lambda config, *a, **kw: config)
    monkeypatch.setattr(launcher, "node_info", lambda *_: {})
    monkeypatch.setattr(launcher.signal, "signal", lambda *a: None)
    usage = {"attempts": 0, "prompt_tokens": 0, "completion_tokens": 0}
    monkeypatch.setattr(launcher, "ledger_totals", lambda *_: dict(usage))
    cells = {}
    reboot_attempt = False

    def execute_cell(cell, suite, *, workdir, resume):
        name = cell.output_dir.parent.name
        cells[name] = cell
        if reboot_attempt:
            assert name == "local300-replay"
            if fault == "reboot_worker":
                harness._run_subprocess([], cwd=tmp_path, stdout_path=None, stderr_path=None)
            elif fault == "reboot_api":
                from exact.llm.routing import OpenRouterClient

                OpenRouterClient.__init__(object())
            elif fault == "reboot_missing_cache":
                from exact.experiments.runtime import CellRecovery

                CellRecovery.prepare(SimpleNamespace(reuse={"inputs", "extraction"}))
        assert cell.resolved_config["pipeline"][0]["params"]["generate_llm_rationales"] is False
        assert "/never-open-private-test.tsv" not in json.dumps(cell.resolved_config)
        assert cell.split_role == "development"
        cell.output_dir.mkdir(parents=True)
        if cell.resolved_config["data"].get("train_candidates"):
            fit = cell.output_dir / "fitting/selector/training_units.json"
            fit.parent.mkdir(parents=True)
            fit.write_text('{"effective_groups": 64}')
        reconstructed = name in {"global300-resume", "global300-replay"}
        if "300" in name:
            stats = cell.output_dir / "stats/run_stats.json"
            stats.parent.mkdir()
            stats.write_text(
                json.dumps({"observed_execution": {"device_type": "cuda", "device": "cuda:0"}})
            )
            evaluation = cell.output_dir / "evaluation/evaluation_results.json"
            evaluation.parent.mkdir()
            metric = 0.81 if fault == "changed_metrics" and reconstructed else 0.8
            role = "reference_candidates" if name.startswith("local") else "full_reference"
            evaluation.write_text(
                json.dumps(
                    {
                        "builtin": {"P": metric, "R": metric, "F1": metric},
                        "meta": {"refs": {role: {"path": "fixture"}}},
                    }
                )
            )
        if name.startswith("global300"):
            alignment = cell.output_dir / "alignment/maps_global.tsv"
            alignment.parent.mkdir()
            score = "0.8" if fault == "changed_mapping" and reconstructed else "0.9"
            if (fault == "roundoff" and reconstructed) or (
                fault == "changed_cache_score" and name == "global300-replay"
            ):
                score = "0.9000000000000004"
            target = "other" if fault == "changed_identity" and reconstructed else "t:0"
            relation = "<" if fault == "changed_relation" and reconstructed else "="
            alignment.write_text(
                f"SrcEntity\tTgtEntity\tScore\tRelation\ndev:0\t{target}\t{score}\t{relation}\n"
            )
        if name.startswith("local300"):
            alignment = cell.output_dir / "alignment/maps_local.tsv"
            alignment.parent.mkdir()
            score = (
                "0.8" if fault == "changed_local_mapping" and name == "local300-replay" else "0.9"
            )
            alignment.write_text(
                f"SrcEntity\tTgtEntity\tTgtCandidates\ndev:0\t\t[('t:0', {score})]\n"
            )
            (alignment.parent / "paper.maps_global.tsv").write_text(
                "SrcEntity\tTgtEntity\tScore\tRelation\ndev:0\tt:0\t0.9\t=\n"
            )
        if (fault, name) in {
            ("global_request", "global300-replay"),
            ("local_request", "local300-replay"),
        }:
            usage["attempts"] += 1
        return {"status": "interrupted" if cell.recovery["stop_after_checkpoint"] else "complete"}

    monkeypatch.setattr(harness, "execute_cell", execute_cell)
    args = Namespace(
        api_key_file=None,
        output_root=tmp_path / "g0",
        campaign=campaign_path,
        max_seconds=None,
        ram_gb=48,
        requests_cap=100000,
        tokens_cap=32000000,
        resume_from=None,
        resume_probes_from=None,
        resume_failed_from=None,
        resume_interrupted_from=None,
        execute=True,
        scratch_root=None,
        skip_hosted=False,
        generate_rationales=False,
    )
    reboot = str(fault).startswith("reboot_")
    passes = reboot or fault in {None, "roundoff", "resumed_globals"}
    assert launcher.execute(args) == (0 if passes else 2)
    report = json.loads((args.output_root / "report.json").read_text())
    assert report["status"] == ("passed" if passes else "failed")
    assert report["limits"]["soft_seconds"] is None
    assert report["generate_rationales"] is False
    for name, parent in (
        ("global300-resume", "global300-stop"),
        ("global300-replay", "global300-resume"),
    ):
        assert cells[name].recovery["resume_from"] == str(cells[parent].output_dir.parent)
    if passes or fault in {"changed_local_mapping", "local_request"}:
        assert cells["local300"].resolved_config["data"]["execution_mode"] == "local_ranking"
        assert cells["local300-replay"].recovery["resume_from"] == str(
            cells["local300"].output_dir.parent
        )
        assert all(cells[name].source_cap == 300 for name in cells if "300" in name)
    if passes:
        assert len(cells) == 10
        assert all(row["new_usage"]["attempts"] == 0 for row in report["stages"])
        checks = json.loads((args.output_root / "replay-checks.json").read_text())
        assert checks == report["replay_checks"]
        assert checks["global300-resume"]["max_score_delta"] < 1e-5
        assert checks["global300-cache"]["completed_cache_bytes_verified"]
        assert checks["local300-cache"]["completed_cache_bytes_verified"]
    if reboot or fault == "resumed_globals":
        from copy import deepcopy
        from tools import validation_resume
        from exact.experiments.runtime import CellRecovery
        from exact.llm.routing import OpenRouterClient

        original_hooks = (harness._run_subprocess, CellRecovery.prepare, OpenRouterClient.__init__)
        saved_count = 9 if reboot else 8
        imported = {row["id"]: deepcopy(row) for row in report["stages"][:saved_count]}
        for row in imported.values():
            row.update(imported=True, new_worker_calls=0, new_usage=dict(usage))
            row["measurement_usage"] = dict(usage)
        imported["cold64"]["adoption_evidence"] = {
            "prior_hosted_usage": dict(usage),
            "prior_ledger_usage": dict(usage),
        }
        if reboot:
            imported["cold64"]["adoption_evidence"]["elapsed_accounting"] = {
                "status": "lower_bound",
                "elapsed_seconds_lower_bound": report["elapsed_seconds"],
                "final_elapsed_seconds": None,
            }
        monkeypatch.setattr(
            validation_resume,
            "adopt_interrupted_validation" if reboot else "adopt_failed_validation",
            lambda *a, **kw: (imported, report["elapsed_seconds"], {}),
        )
        if reboot:
            args.resume_interrupted_from = args.output_root
        else:
            args.resume_failed_from = args.output_root
        args.output_root = tmp_path / "remaining"
        cells.clear()
        reboot_attempt = reboot
        rejects_work = reboot and fault != "reboot_replay"
        assert launcher.execute(args) == (2 if rejects_work else 0)
        resumed = json.loads((args.output_root / "report.json").read_text())
        assert original_hooks == (
            harness._run_subprocess,
            CellRecovery.prepare,
            OpenRouterClient.__init__,
        )
        assert list(cells) == (["local300-replay"] if reboot else ["local300", "local300-replay"])
        assert resumed["stages"][:saved_count] == list(imported.values())
        assert resumed["elapsed_seconds"] >= report["elapsed_seconds"]
        if reboot:
            assert resumed["replay_only"] is True
            assert resumed["elapsed_accounting"]["status"] == "lower_bound"
            assert (
                resumed["elapsed_accounting"]["elapsed_seconds_lower_bound"]
                == resumed["elapsed_seconds"]
            )
            assert resumed["elapsed_accounting"]["final_elapsed_seconds"] is None
            assert resumed["resume"]["scope"] == (
                "Finalize the local cache replay without model workers or hosted requests"
            )
            if rejects_work:
                assert "Replay-only finalization" in resumed["reason"]
            else:
                assert resumed["stages"][-1]["id"] == "local300-replay"
                assert resumed["stages"][-1]["new_worker_calls"] == 0
                assert resumed["stages"][-1]["new_usage"]["attempts"] == 0
        else:
            assert [row["id"] for row in resumed["stages"][-2:]] == list(cells)
            assert resumed["resume"]["scope"] == (
                "Reuse completed probes and global checks; run remaining local checks"
            )
    elif not passes:
        message = {
            "changed_mapping": "score drift",
            "changed_identity": "canonical IDs",
            "changed_relation": "relation labels",
            "changed_metrics": "metric drift",
            "changed_local_mapping": "cache replay changed mapping bytes",
            "changed_cache_score": "cache replay changed mapping bytes",
        }.get(fault, "performed new model work")
        assert message in report["reason"]


def test_forecast_retains_imported_hosted_measurement_without_new_usage():
    rows = [{"id": name, "wall_seconds": 1} for name in ("cold64", "warm64", "fit64")]
    rows.append(
        {
            "id": "hosted20",
            "wall_seconds": 1,
            "new_usage": {"attempts": 0, "prompt_tokens": 0, "completion_tokens": 0},
            "measurement_usage": {"attempts": 10, "prompt_tokens": 1000, "completion_tokens": 100},
        }
    )
    estimate = forecast(
        rows,
        elapsed=100,
        max_seconds=None,
        requests=10,
        tokens=1100,
        limits={"requests": 400, "tokens": 32000000},
    )
    assert estimate["remaining_requests"] == 450
    assert estimate["remaining_tokens"] == 49500
    assert estimate["fits_limits"] is False
