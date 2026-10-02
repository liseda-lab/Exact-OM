"""Cross-arm supervision reuse and dispatcher receipt qualification."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from exact.experiments.dispatch import _step
from exact.repair.api import write_artifact
from exact.repair.protocol import RepairProtocolV3, training_projection_v3
from exact.repair.records import canonical_hash
from tests.repair_training_completion_test import cache_for
from tools.repair import batch
from tools.repair.campaign_handoff import (
    dispatched_worker,
    preparation_gate,
    reuse_preparation,
)
from tools.repair.corpus import generate_corpus
from tools.repair.prepare import load_preparation, save_preparation


def prepared(tmp_path):
    path = Path(__file__).parents[1] / "specs/exact-repair/protocol/xr21-review2-conformance.json"
    protocol = json.loads(path.read_text().replace("UNFROZEN", "unit-fixture"))
    protocol["identity"]["execution_authorized"] = True
    projection = training_projection_v3(RepairProtocolV3.model_validate(protocol))
    cases = generate_corpus(
        split_counts={"train": 1, "development": 1, "test": 1},
        siblings_per_parent=1,
        families=("range",),
    )
    caches = {c.case_id: cache_for(c) for c in cases if c.split != "test"}
    source = tmp_path / "source.json"
    save_preparation(
        source,
        cases,
        dict(
            protocol=projection,
            protocol_hash=canonical_hash(projection),
            label_seconds=19.0,
            label_cpu_seconds=7.0,
        ),
        caches,
    )
    return protocol, source, caches


def test_architecture_reuse_preserves_exact_cache_provenance_and_cost(tmp_path):
    protocol, source, caches = prepared(tmp_path)
    protocol["identity"]["run_id"] = "rgcn-unary"
    protocol["model"].update(backbone="rgcn", pair_benefit=False)
    protocol["circuit"]["cache_directory"] = "separate-arm-cache"
    target = tmp_path / "target.json"
    write_artifact(target, protocol)
    destination = tmp_path / "derived.json"
    provenance = reuse_preparation(source, target, destination)
    _, actual, report = load_preparation(destination)
    assert actual == caches
    assert report["label_seconds"] == 19.0 and report["label_cpu_seconds"] == 7.0
    assert report["protocol"]["model"]["backbone"] == "rgcn"
    assert not report["protocol"]["model"]["pair_benefit"]
    assert provenance["cache_hashes"] == {
        key: canonical_hash(cache) for key, cache in caches.items()
    }
    assert reuse_preparation(source, target, destination) == provenance


@pytest.mark.parametrize(
    "section,key",
    [
        ("policy", "hard_query_manifest"),
        ("teacher", "consequence_manifest"),
        ("input", "alignment_manifest"),
    ],
)
def test_changed_supervision_dependencies_reject_before_reuse(tmp_path, section, key):
    protocol, source, _ = prepared(tmp_path)
    protocol[section][key] = "changed-dependency"
    target = tmp_path / "target.json"
    write_artifact(target, protocol)
    destination = tmp_path / "derived.json"
    with pytest.raises(ValueError, match="Label dependencies"):
        reuse_preparation(source, target, destination)
    assert not destination.exists()


def test_preparation_gate_keeps_missing_cases_in_denominator(tmp_path):
    _, source, _ = prepared(tmp_path)
    cases, caches, report = load_preparation(source)
    del caches[next(c.case_id for c in cases if c.split == "development")]
    save_preparation(source, cases, report, caches)
    output = tmp_path / "gate.json"
    with pytest.raises(ValueError, match="gate incomplete"):
        preparation_gate(source, output)
    evidence = json.loads(output.read_text())
    assert evidence["scheduled_label_cases"] == 2
    assert evidence["complete_label_cases"] == 1
    assert evidence["case_counts"]["test"] == 1
    assert len(evidence["incomplete_cases"]) == 1


def test_dispatched_completion_retains_step_nonce(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "14408")
    monkeypatch.setenv("SLURM_STEP_ID", "9")
    nonce = "fixture-dispatch-0001"

    def worker(path, job, attempt):
        evidence = dict(status="complete", exit_code=0, step_id="14408.9")
        write_artifact(attempt / "status.json", evidence)
        write_artifact(attempt / "completion.json", evidence)
        (attempt / "exit_code").write_text("0\n")
        return 0

    monkeypatch.setattr(batch, "run", worker)
    assert dispatched_worker("batch.json", "job", tmp_path, nonce) == 0
    launch = dict(
        step_path=str(tmp_path / "step.json"),
        nonce=nonce,
        run=dict(
            completion_path=str(tmp_path / "completion.json"), exit_path=str(tmp_path / "exit_code")
        ),
    )
    assert _step(launch, "14408", {}) == "14408.9"
    altered = copy.deepcopy(launch)
    altered["nonce"] = "different-dispatch-0002"
    with pytest.raises(ValueError, match="nonce"):
        _step(altered, "14408", {})


def test_batch_freeze_binds_external_input_files(tmp_path):
    repo = tmp_path / "repository"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "protocol.json").write_text("{}")
    subprocess.run(["git", "-C", str(repo), "add", "protocol.json"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    source = tmp_path / "input.json"
    source.write_text('{"input": "original"}')
    spec = tmp_path / "spec.json"
    write_artifact(
        spec,
        dict(
            repository=str(repo),
            protocol_source="protocol.json",
            python=sys.executable,
            input_files=[str(source)],
        ),
    )
    path = batch.freeze(spec, tmp_path / "frozen")
    assert str(source) in json.loads(path.read_text())["frozen_files"]
    source.write_text('{"input": "changed"}')
    with pytest.raises(ValueError, match="Frozen batch file changed"):
        batch.checked_batch(path)
