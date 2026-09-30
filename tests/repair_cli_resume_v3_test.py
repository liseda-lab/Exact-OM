"""Learned search replay preserves its inert risk snapshot and scheduling epoch."""

import copy
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from exact.repair.api import write_artifact
from exact.repair.graph import GraphExplanation
from exact.repair.pipeline import FrozenPlanRisk, freeze_neural_round
from exact.repair.records import make_objective, read_record
from tests.repair_generation_v3_test import neural_fixture


@pytest.fixture
def learned_round(tmp_path):
    problem, model = neural_fixture(False, revision="v3")
    frozen = freeze_neural_round(
        problem,
        model,
        draws_per_object=0,
        max_depth=0,
        max_constructors=0,
        candidate_cap=32,
        compiler_cache_directory=str(tmp_path / "cache"),
    )
    return problem, model, frozen


def test_strict_risk_json_roundtrip_and_snapshot_failures(learned_round, tmp_path):
    _, _, frozen = learned_round
    with pytest.raises(ValueError, match="graph admission"):
        dataclasses.replace(
            frozen.risk_scorer,
            supports=(
                GraphExplanation(
                    "e",
                    ("m",),
                    frozen.problem.fixed_axioms,
                    frozen.problem.policy.monitored_classes[0],
                    support_status="sufficient",
                ),
            ),
        )
    risk = frozen.risk_scorer

    def restore(payload, **changes):
        return FrozenPlanRisk.from_dict(
            payload,
            problem=frozen.problem,
            objective=dataclasses.replace(frozen.objective, **changes),
        )

    payload = json.loads(json.dumps(risk.to_dict()))
    restored = restore(payload)
    assert restored == risk
    assert restored.to_dict() == payload
    assert restored.risk_identity == risk.risk_identity
    assert restored((0,)) == risk((0,))
    changed = copy.deepcopy(payload)
    changed["record"]["checkpoint_bytes"] += 1
    with pytest.raises(ValueError, match="content hash"):
        restore(changed)
    changed = copy.deepcopy(payload)
    changed["record"]["graph"]["nodes"][0]["$record"] = "os.system"
    with pytest.raises(ValueError, match="unknown or malformed"):
        restore(changed)
    with pytest.raises(ValueError, match="bindings"):
        restore(payload, model_hash="0" * 64)
    bad_model = dataclasses.replace(risk, model_hash="0" * 64)
    with pytest.raises(ValueError, match="model identity"):
        restore(bad_model.to_dict(), model_hash="0" * 64)
    missing = dataclasses.replace(risk, checkpoint_path=str(tmp_path / "missing.pt"))
    with pytest.raises(ValueError, match="artifact is missing"):
        restore(missing.to_dict())
    artifact = Path(risk.checkpoint_path)
    original = artifact.read_bytes()
    artifact.write_bytes(original[:-1] + bytes((original[-1] ^ 1,)))
    with pytest.raises(ValueError, match="integrity"):
        restore(payload)
    artifact.write_bytes(original + b"x")
    with pytest.raises(ValueError, match="size mismatch"):
        restore(payload)
    artifact.write_bytes(original)


def test_cli_learned_search_resume_restores_risk_and_nondefault_settings(learned_round, tmp_path):
    problem, model, _ = learned_round
    checkpoint, source = tmp_path / "source-model.pt", tmp_path / "input.json"
    first_path, resumed_path, ledger_path = (
        tmp_path / name for name in ("first.json", "resumed.json", "ledger.json")
    )
    torch.save(
        {
            "metadata": model.metadata,
            "config": model.config,
            "model_schema": "exact-repair/model/v3",
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    write_artifact(
        source, {"input": problem.to_dict(), "objective": make_objective(problem.objects).to_dict()}
    )

    def cli(*options):
        return subprocess.run(
            [sys.executable, "-m", "exact.delivery.cli.repair", *map(str, options)],
            text=True,
            capture_output=True,
            timeout=80,
        )

    first_run = cli(
        "--problem",
        source,
        "--model",
        checkpoint,
        "--output",
        first_path,
        "--search-ledger",
        ledger_path,
        "--seconds",
        60,
        "--stage-seconds",
        20,
        "--proposal-seconds",
        25,
        "--grammar-depth",
        0,
        "--constructors",
        0,
        "--draws",
        0,
        "--shortlist-size",
        3,
        "--utility-window",
        7,
        "--shortlist-seconds",
        2.25,
        "--compiler-cache",
        tmp_path / "cli-cache",
    )
    assert first_run.returncode == 0, first_run.stdout + first_run.stderr
    first = json.loads(first_path.read_text())
    assert first["risk_descriptor"] is not None
    prior = read_record(first["result"]).ledger
    assert prior.risk_identity != "none"
    assert (prior.shortlist_size, prior.utility_window) == (3, 7)
    resumed = cli(
        "--problem",
        first_path,
        "--output",
        resumed_path,
        "--search-ledger",
        ledger_path,
        "--resume-search",
        "--stage-seconds",
        20,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    replayed = json.loads(resumed_path.read_text())
    assert replayed["risk_descriptor"] == first["risk_descriptor"]
    assert replayed["search_options"] == first["search_options"]
    current = read_record(replayed["result"]).ledger
    assert current.risk_identity == prior.risk_identity
    assert current.input_hash == prior.input_hash
    assert current.objective_hash == prior.objective_hash
    assert current.elapsed_seconds >= prior.elapsed_seconds
    assert current.solves >= prior.solves and current.checks >= prior.checks

    def rejected(source_path, output_name, *extra):
        output = tmp_path / output_name
        completed = cli(
            "--problem",
            source_path,
            "--output",
            output,
            "--search-ledger",
            ledger_path,
            "--resume-search",
            "--stage-seconds",
            20,
            *extra
        )
        assert completed.returncode == 2, completed.stdout + completed.stderr
        failed = json.loads(output.read_text())
        assert failed["stage"] == "preparation"
        return failed["failure"]

    assert "frozen search setting" in rejected(first_path, "drift.json", "--shortlist-size", 1)
    absent = {**first, "risk_descriptor": None}
    absent_path = tmp_path / "absent.json"
    write_artifact(absent_path, absent)
    assert "requires its frozen risk descriptor" in rejected(absent_path, "absent-failure.json")
    Path(first["risk_descriptor"]["record"]["checkpoint_path"]).unlink()
    assert "artifact is missing" in rejected(first_path, "missing-failure.json")
