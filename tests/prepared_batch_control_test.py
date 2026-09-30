"""Prepared paired baselines reuse verified results without another numerical run."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.experiments import campaign, harness
from exact.experiments.recovery import ArtifactStore
from tests.measured_once_test import example  # noqa: F401: shared completed-artifact fixture
from tools import measured_once, prepared_batch as batch


@pytest.fixture
def control(example, monkeypatch):
    source_campaign = example.origin / "campaign.yaml"
    batch.write(
        source_campaign, {"steps": [{"id": "E04", "arms": [{"id": "fixed", "role": "baseline"}]}]}
    )
    selection = example.origin / "screen/selection.json"
    batch.write(selection, {"experiments": {"E04": {"status": "selected"}}})
    manifest = example.output / "experiment_manifest.json"
    report = batch.read(manifest)
    report.update(
        experiment_id="E04",
        arm_id="fixed",
        task_id=example.cell.task_id,
        seed=17,
        stage="screen",
        return_code=0,
        extraction_complete=True,
        generate_rationales=False,
        arm_role="baseline",
        split_role="development",
        reference_role="valid",
        source_cap=300,
        execution_mode="global_alignment",
    )
    batch.write(manifest, report)
    completion = example.origin / "completion.json"
    batch.write(
        completion,
        {
            "status": "complete",
            "exit_code": 0,
            "campaign": batch.binding(source_campaign),
            "selection": batch.binding(selection),
            "manifests": [batch.binding(manifest)],
        },
    )
    registry = {"runs": [{"id": "main", "completion_path": str(completion)}]}
    recipe = {
        "scientific_step": "E19",
        "depends_on": ["main"],
        "reuse_controls": [{"run_id": "main", "scientific_step": "E04", "arm_id": "fixed"}],
    }
    monkeypatch.setattr(campaign, "load_campaign", lambda _: (SimpleNamespace(steps=[]), {}))
    return SimpleNamespace(
        **vars(example),
        recipe=recipe,
        registry=registry,
        completion=completion,
        source_campaign=source_campaign,
        selection=selection,
        manifest=manifest
    )


def import_control(control):
    return batch.import_controls(
        control.recipe, control.registry, Path("target.yaml"), control.runtime, control.workdir
    )


def test_absent_reuse_controls_is_a_side_effect_free_noop(tmp_path):
    assert batch.import_controls({}, {}, None, tmp_path / "missing", None) == []
    assert not list(tmp_path.iterdir())


def test_imported_baseline_runs_without_numerics_or_duplicate_charges(control, monkeypatch):
    before = (control.runtime / "budget.json").read_bytes()
    source_manifest = control.manifest.read_bytes()
    records = import_control(control)
    assert records[0]["source_run"] == "main"
    assert records[0]["source_manifest"] == batch.binding(control.manifest)
    assert not (control.runtime / "recovery/cells").exists()
    assert not control.cell.manifest_path.exists()
    assert not (control.runtime / "screen/selection.json").exists()
    assert import_control(control) == records
    assert (control.runtime / "budget.json").read_bytes() == before
    assert control.manifest.read_bytes() == source_manifest
    monkeypatch.setattr(
        harness,
        "_post_run_provenance",
        lambda *a, **kw: {
            "candidate_pool_fingerprint": "pool",
            "candidate_pool_manifest_provenance": {"sha256": "hash"},
        },
    )
    import exact.runs.manifest
    from exact.llm.routing import OpenRouterClient

    monkeypatch.setattr(exact.runs.manifest, "refresh_manifest", lambda *_: None)
    monkeypatch.setattr(
        OpenRouterClient, "_generation", lambda *a, **kw: pytest.fail("unexpected hosted call")
    )
    cell = replace(control.cell, recovery={"root": str(control.runtime)})
    with measured_once.require_reuse_only():
        result = harness.execute_cell(cell, control.suite, workdir=control.workdir, resume=True)
    assert result["status"] == "complete", result.get("failure")
    assert set(result["recovery"]["reused_stages"]) == {"inputs", "extraction", "evaluation"}
    assert result["execution_measurement"] == control.measurement
    assert (cell.output_dir / "alignment/paper.maps_global.tsv").read_bytes() == (
        control.output / "alignment/paper.maps_global.tsv"
    ).read_bytes()
    assert batch.read(cell.output_dir / "evaluation/metrics.json")["f1"] == 0.85
    assert (control.runtime / "budget.json").read_bytes() == before


@pytest.mark.parametrize(
    "damage",
    [
        "missing_run",
        "incomplete",
        "dependency",
        "manifest_changed",
        "missing_cell",
        "source_role",
        "target_role",
        "manifest_role",
        "seed",
        "population",
        "config",
        "source_code",
        "blob",
        "decision",
        "duplicate",
        "missing_target",
    ],
)
def test_control_import_fails_closed_before_importing_any_artifacts(control, damage):
    completion = batch.read(control.completion)
    if damage == "missing_run":
        control.registry["runs"] = []
    elif damage == "incomplete":
        completion["status"] = "failed"
    elif damage == "dependency":
        control.recipe["depends_on"] = []
    elif damage == "manifest_changed":
        control.manifest.write_text("changed source")
    elif damage == "missing_cell":
        completion["manifests"] = []
    elif damage == "source_role":
        batch.write(
            control.source_campaign,
            {"steps": [{"id": "E04", "arms": [{"id": "fixed", "role": "candidate"}]}]},
        )
        completion["campaign"] = batch.binding(control.source_campaign)
    elif damage == "target_role":
        control.cells[0] = replace(control.cell, arm_role="candidate")
    elif damage == "manifest_role":
        batch.write(control.manifest, {**batch.read(control.manifest), "arm_role": "candidate"})
        completion["manifests"] = [batch.binding(control.manifest)]
    elif damage == "seed":
        control.cells[0] = replace(control.cell, seed=29)
    elif damage == "population":
        control.cells[0] = replace(control.cell, source_cap=301)
    elif damage == "config":
        control.cells[0] = replace(
            control.cell,
            resolved_config={**control.cell.resolved_config, "matching": {"different": True}},
        )
    elif damage == "source_code":
        (control.workdir / "exact/score.py").write_text("SCORER = 2\n")
    elif damage == "blob":
        store = ArtifactStore(control.origin)
        saved = store.verify(control.saved["artifacts"]["extraction"])
        store._blob(saved["outputs"]["fitting/fusion.json"]["sha256"]).write_bytes(b"changed")
    elif damage == "decision":
        batch.write(control.selection, {"experiments": {"E04": {"status": "blocked"}}})
        completion["selection"] = batch.binding(control.selection)
    elif damage == "duplicate":
        control.recipe["reuse_controls"] *= 2
    elif damage == "missing_target":
        control.cells[0] = replace(control.cell, arm_id="another")
    batch.write(control.completion, completion)
    with pytest.raises(ValueError):
        import_control(control)
    assert not (control.runtime / "artifacts").exists()
    assert not (control.runtime / "control-imports.json").exists()


def test_control_reconstruction_uses_bound_historical_inheritance(control, monkeypatch):
    producer = SimpleNamespace(config=SimpleNamespace(experiment_id="E05_initial"))
    control.suite.sources.append(producer)
    control.source.config.depends_on = ["E05_initial"]
    step = SimpleNamespace(id="E05_initial", external_selection={"bound": True})
    monkeypatch.setattr(campaign, "load_campaign", lambda _: (SimpleNamespace(steps=[step]), {}))
    historical = {"status": "selected", "published_policy_overlay": {"fusion": "saved"}}
    monkeypatch.setattr(campaign, "external_selection_result", lambda *a: historical)
    monkeypatch.setattr(harness, "_bind_external_selection", lambda *a: historical)
    seen = []
    monkeypatch.setattr(
        harness,
        "inherited_selection_overlay",
        lambda selection, dependencies: seen.append((selection, dependencies)) or {},
    )
    import_control(control)
    assert seen == [({"experiments": {"E05_initial": historical}}, ["E05_initial"])]
    step.external_selection = None
    with pytest.raises(ValueError, match="bound historical selection"):
        import_control(control)
