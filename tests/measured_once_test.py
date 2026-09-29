"""Measured executions become scientific evidence without another worker or fit."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.core.entities.configs.yaml_io import dump_yaml_document
from exact.experiments import harness
from exact.experiments.recovery import ArtifactStore
from exact.experiments.runtime import CellRecovery
from exact.experiments.schema import ResourceConfig
from tools import measured_once as once
from tools.qualify_cached_family import binding, preserve_training_progress


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def example(tmp_path, monkeypatch):
    workdir = tmp_path / "source"
    (workdir / "exact").mkdir(parents=True)
    (workdir / "exact/score.py").write_text("SCORER = 1\n")
    runtime = tmp_path / "runtime/campaign"
    suite = SimpleNamespace(
        sources=[],
        baseline_manifest=None,
        baseline_manifest_hash=None,
        specification={},
        dataset_lock=None,
        dataset_lock_hash=None,
    )
    source = SimpleNamespace(config=SimpleNamespace(experiment_id="E19", depends_on=[]))
    suite.sources = [source]
    from exact.core.entities.configs.config import ConfigModel
    from exact.experiments.rationale_policy import apply_rationale_policy

    config = apply_rationale_policy(ConfigModel().model_dump(mode="json", by_alias=True))
    config["data"]["execution_mode"] = "global_alignment"
    cell = harness.RunCell(
        "campaign",
        "E19",
        "screen",
        "fixed",
        "baseline",
        "D0-global_alignment",
        "development",
        "valid",
        "known_incomplete",
        17,
        300,
        ResourceConfig(kind="cpu"),
        runtime / "screen/runs/E19/fixed/D0-global_alignment/seed-17",
        config,
        "config",
        "experiment",
        "design",
        None,
        "label_free",
        {},
        "confirmed_negatives",
    )

    def provenance(cell, suite, *, workdir):
        return {
            "fingerprint": "provenance",
            "fingerprint_payload": {
                "inputs": {},
                "output_dir": str(cell.output_dir),
                "ontology_artifacts": {},
            },
            "packages": {"torch": "pinned"},
        }

    monkeypatch.setattr(harness, "_provenance_payload", provenance)
    monkeypatch.setattr(harness, "inherited_selection_overlay", lambda *_: {})
    cells = [cell]
    monkeypatch.setattr(harness, "build_cells", lambda *a, **kw: cells)
    from exact.experiments import campaign

    monkeypatch.setattr(campaign, "materialize_campaign", lambda *a, **kw: suite)
    origin = tmp_path / "measured"
    original = replace(
        cell,
        suite_id="qualification",
        experiment_id="QUAL-D0",
        arm_id="D0--fixed",
        output_dir=origin / "run",
        recovery={"root": str(origin)},
    )
    recovery = CellRecovery(original, provenance(original, suite, workdir=workdir), workdir)
    recovery.prepare()
    output = original.output_dir
    write(
        output / "dataset/candidate_pool_sample_manifest.json",
        {
            "fingerprint": "pool",
            "gold_free_summary": {"candidate_pairs": 2},
            "per_kind": {"class": {"candidate_pairs": 2}},
        },
    )
    write(output / "source_decisions.json", {"s": "t"})
    (output / "alignment").mkdir()
    (output / "alignment/paper.maps_global.tsv").write_text("s\tt\n")
    write(output / "fitting/fusion.json", {"weights": [1, 2]})
    write(
        output / "evaluation/metrics.json",
        {"f1": 0.85, "alignment": binding(output / "alignment/paper.maps_global.tsv")},
    )
    measurement = {
        "schema_version": 1,
        "status": "measured",
        "artifact_id": recovery.identities["extraction"]["artifact_id"],
        "wall_seconds": 3600,
        "peak_memory_kb": 100,
        "origin_attempt": recovery.attempt["attempt_id"],
    }
    write(output / "stats/execution_measurement.json", measurement)
    saved = recovery.finish({"status": "complete", "extraction_complete": True})
    manifest = {"status": "complete", "recovery": saved, "execution_measurement": measurement}
    write(original.manifest_path, manifest)
    config_path = origin / "config.yaml"
    config_path.write_text(dump_yaml_document(config))
    row = {
        "name": "D0--fixed",
        "case_id": "D0",
        "status": "passed",
        "execution_status": "complete",
        "prefix": False,
        "processed_pairs": 2,
        "dataset_rows": 2,
        "seed": 17,
        "source_cap": 300,
        "generate_rationales": False,
        "new_usage": {"attempts": 0},
        "config": binding(config_path),
        "output_dir": str(output),
        "budget_work_id": "original",
        "bindings": [
            binding(output / name)
            for name in (
                "experiment_manifest.json",
                "recovery-runtime.json",
                "dataset/candidate_pool_sample_manifest.json",
                "alignment/paper.maps_global.tsv",
                "source_decisions.json",
            )
        ],
    }
    write(
        runtime / "budget.json",
        {
            "schema_version": 2,
            "limits": {
                "envelopes_hours": {"reserve": 60},
                "node_hours_cap": 336,
                "requests_cap": 50000,
                "tokens_cap": 32000000,
            },
            "intervals": [[1, 3601]],
            "work": {
                "original": {
                    "status": "complete",
                    "seconds": 3600,
                    "start": 1,
                    "end": 3601,
                    "group": "reserve",
                    "requests": 0,
                    "tokens": 0,
                }
            },
        },
    )
    return SimpleNamespace(
        runtime=runtime,
        suite=suite,
        source=source,
        cell=cell,
        cells=cells,
        row=row,
        workdir=workdir,
        origin=origin,
        output=output,
        saved=saved,
        measurement=measurement,
    )


def promote(example):
    return once.promote_measured_cells(
        Path("campaign.yaml"), example.runtime, [example.row], "E19", {}, example.workdir
    )


def test_completed_import_reuses_predictions_fit_metrics_and_original_time(example, monkeypatch):
    before = (example.runtime / "budget.json").read_bytes()
    source_manifest = (example.output / "experiment_manifest.json").read_bytes()
    receipt = promote(example)
    assert receipt["cells"][0]["source_attempt"] == example.saved["attempt_id"]
    after = json.loads((example.runtime / "budget.json").read_text())
    assert after["work"]["original"] == json.loads(before)["work"]["original"]
    assert len(after["work"]) == 2
    assert next(row for key, row in after["work"].items() if key != "original")["seconds"] >= 0
    assert (example.output / "experiment_manifest.json").read_bytes() == source_manifest
    monkeypatch.setattr(
        harness,
        "_post_run_provenance",
        lambda *a, **kw: {
            "candidate_pool_fingerprint": "pool",
            "candidate_pool_manifest_provenance": {"sha256": "hash"},
        },
    )
    import exact.runs.manifest

    monkeypatch.setattr(exact.runs.manifest, "refresh_manifest", lambda *_: None)
    from exact.llm.routing import OpenRouterClient

    monkeypatch.setattr(
        OpenRouterClient, "_generation", lambda *a, **kw: pytest.fail("unexpected hosted call")
    )
    cell = replace(example.cell, recovery={"root": str(example.runtime)})
    with once.require_reuse_only():
        result = harness.execute_cell(cell, example.suite, workdir=example.workdir, resume=True)
    assert result["status"] == "complete", result.get("failure")
    assert result["execution_measurement"] == example.measurement
    assert set(result["recovery"]["reused_stages"]) == {"inputs", "extraction", "evaluation"}
    assert (cell.output_dir / "alignment/paper.maps_global.tsv").read_bytes() == (
        example.output / "alignment/paper.maps_global.tsv"
    ).read_bytes()
    assert (cell.output_dir / "fitting/fusion.json").read_bytes() == (
        example.output / "fitting/fusion.json"
    ).read_bytes()
    metrics = json.loads((cell.output_dir / "evaluation/metrics.json").read_text())
    original_metrics = json.loads((example.output / "evaluation/metrics.json").read_text())
    assert metrics["f1"] == original_metrics["f1"]
    assert metrics["alignment"]["sha256"] == original_metrics["alignment"]["sha256"]
    assert metrics["alignment"]["path"] == str(cell.output_dir / "alignment/paper.maps_global.tsv")
    attempt = json.loads(
        (
            example.runtime / "attempts" / result["recovery"]["attempt_id"] / "attempt.json"
        ).read_text()
    )
    assert attempt["parent_attempt"] == example.saved["attempt_id"]


@pytest.mark.parametrize(
    "change",
    [
        "seed",
        "population",
        "config",
        "source",
        "blob",
        "accounting",
        "incomplete",
        "role",
        "reference",
    ],
)
def test_promotion_rejects_incompatible_or_incomplete_execution(example, change):
    if change == "seed":
        example.cells[0] = replace(example.cell, seed=29)
    elif change == "population":
        example.cells[0] = replace(example.cell, source_cap=1000)
    elif change == "config":
        example.cells[0] = replace(example.cell, resolved_config={"changed": True})
    elif change == "source":
        (example.workdir / "exact/score.py").write_text("SCORER = 2\n")
    elif change == "blob":
        store = ArtifactStore(example.origin)
        artifact = store.verify(example.saved["artifacts"]["extraction"])
        store._blob(artifact["outputs"]["fitting/fusion.json"]["sha256"]).write_bytes(b"tampered")
    elif change == "accounting":
        state = json.loads((example.runtime / "budget.json").read_text())
        state["work"]["original"]["status"] = "reserved"
        write(example.runtime / "budget.json", state)
    elif change == "incomplete":
        example.row["processed_pairs"] = 1
    elif change == "reference":
        (example.output / "dataset/candidate_pool_sample_manifest.json").write_text("changed pool")
    else:
        example.cells[0] = replace(example.cell, split_role="reporting")
    with pytest.raises(ValueError):
        promote(example)
    assert not (example.runtime / "measured-results.json").exists()


def test_all_declared_arms_are_required_before_any_import_or_selection(example):
    example.cells.append(replace(example.cell, arm_id="missing"))
    with pytest.raises(ValueError, match="every declared cell"):
        promote(example)
    assert not (example.runtime / "artifacts").exists()


def test_resume_seals_training_without_claiming_scored_pairs(tmp_path):
    from tests.experiment_recovery_test import identity

    key = identity("extraction")
    output = tmp_path / "run"
    write(output / "recovery-runtime.json", {"identity": key})
    write(output / "fitting/identity/shard.json", {"source_ids": ["s"], "rows": [{"score": 0.5}]})
    write(
        output / "dataset/candidate_pool_sample_manifest.json",
        {"gold_free_summary": {"candidate_pairs": 2}},
    )
    checkpoint = preserve_training_progress(tmp_path, key)
    assert checkpoint["cursor"] == {"next_pair": 0, "dataset_rows": 2}
    assert checkpoint["completed_ids"] == []
    destination = tmp_path / "restored"
    ArtifactStore(tmp_path).restore_checkpoint(checkpoint, destination)
    assert (destination / "fitting/identity/shard.json").read_bytes() == (
        output / "fitting/identity/shard.json"
    ).read_bytes()
    assert preserve_training_progress(tmp_path, key) == checkpoint


def test_stopped_training_identity_change_is_rejected(tmp_path):
    from tests.experiment_recovery_test import identity

    write(tmp_path / "run/recovery-runtime.json", {"identity": identity("extraction", seed=42)})
    with pytest.raises(ValueError, match="identity changed"):
        preserve_training_progress(tmp_path, identity("extraction", seed=43))


def test_completed_replay_has_no_second_compute_forecast():
    proof = {
        "estimate": {
            "cold_seconds": 1000,
            "fitting_seconds": 2000,
            "units": 3,
            "seconds_per_unit": 10,
            "peak_ram_gb": 8,
        },
        "comparison_seconds": 4545,
        "inventory_seconds": 60,
        "whole_family_seconds": 4605,
        "hours": 4605 / 3600,
    }
    replay = once.reused_comparison_forecast(proof)
    assert replay["comparison_seconds"] == replay["estimate"]["fitting_seconds"] == 0
    assert replay["reused_execution_forecast_seconds"] == proof["comparison_seconds"]
    assert replay["whole_family_seconds"] == 60
    assert proof["estimate"]["fitting_seconds"] == 2000
