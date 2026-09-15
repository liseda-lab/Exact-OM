"""Scientific boundaries and executable materialization of the bounded campaign."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from exact.core.entities.configs.config import ConfigModel
from exact.experiments.campaign import (
    CampaignLock,
    CampaignStep,
    CaseBinding,
    campaign_plan,
    dependency_order,
    load_campaign,
    materialize_campaign,
    openrouter_only,
)


def _binding(path: Path, text: str = "fixture\n") -> dict:
    path.write_text(text)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _step(**updates) -> dict:
    result = {
        "id": "E00",
        "family": "E00",
        "case": "D0",
        "budget_group": "foundation",
        "arms": [{"id": "baseline", "role": "baseline"}, {"id": "replay", "role": "candidate"}],
        "readiness": {
            arm: {"screen": {"status": "implementing", "reason": "fixture validation pending"}}
            for arm in ("baseline", "replay")
        },
        "selection": {
            "decisions": [
                {"id": "replay", "baseline": "baseline", "candidates": ["replay"], "metric": "F1"}
            ]
        },
        "design": {
            "primary_comparison": "replay equality",
            "primary_endpoint": "F1",
            "independent_unit": "source_group",
            "power_status": "descriptive",
            "assumptions": ["No scientific gain claimed from replay"],
        },
    }
    result.update(updates)
    return result


def _lock(tmp_path: Path) -> Path:
    blueprint = {
        "design_schema_version": 2,
        "kind": "exact_om_campaign_design",
        "experiments": [{"id": "E00", "initial_treatment_cap": 2}],
        "profiles": {
            "core_14d": {
                "node_hours_cap": 336,
                "recovery_reserve_hours": 60,
                "final_hours_reserved": 54,
                "llm_request_planning_cap": 20000,
                "llm_token_planning_cap": 32000000,
                "final_arm_task_design_cap": 24,
            }
        },
        "budget_envelopes_core_hours": {
            "foundation": 12,
            "retrieval": 48,
            "channels": 48,
            "decisions": 36,
            "llm": 42,
            "extensions": 18,
            "sentinels": 18,
            "final": 54,
            "reserve": 60,
        },
    }
    data = {
        "campaign_id": "fixture-campaign",
        "blueprint": _binding(tmp_path / "blueprint.yaml", yaml.safe_dump(blueprint)),
        "base_config": str(Path(__file__).resolve().parents[1] / "exact/default_config.yaml"),
        "cases": {
            "D0": {
                "task": "fixture",
                "role": "development",
                "selection_reason": "capability before outcomes",
                "source": _binding(tmp_path / "source.owl"),
                "target": _binding(tmp_path / "target.owl"),
                "source_universe": _binding(tmp_path / "sources.txt"),
                "references": {"valid": _binding(tmp_path / "valid.tsv")},
            }
        },
        "steps": [_step()],
    }
    path = tmp_path / "campaign.lock.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_screen_plan_is_read_only_and_retains_unimplemented_arms(tmp_path):
    path = _lock(tmp_path)
    before = set(tmp_path.iterdir())
    plan = campaign_plan(path, stage="screen")
    assert set(tmp_path.iterdir()) == before
    assert len(plan["rows"]) == 2
    assert all(row["status"] == "implementing" for row in plan["rows"])
    assert all(
        "measured cold/warm resource forecast missing" in row["issues"] for row in plan["rows"]
    )
    assert plan["final_hours_reserved"] == 54
    assert plan["recovery_reserve_hours"] == 60


def test_case_roles_reject_final_reference_in_screen():
    with pytest.raises(ValueError, match="role violates"):
        CaseBinding(
            task="leaking",
            role="development",
            selection_reason="test",
            references={"test": {"path": "secret.tsv", "sha256": "a" * 64}},
        )
    with pytest.raises(ValueError, match="ordinary negatives"):
        CaseBinding(
            task="incomplete",
            role="development",
            selection_reason="test",
            negative_policy="complete_reference",
        )


def test_dependency_ports_resolve_and_cycles_unknown_ports_fail():
    first = CampaignStep.model_validate(_step(produces=["baseline_evidence"]))
    second = CampaignStep.model_validate(_step(id="E26", requires=["baseline_evidence"]))
    assert dependency_order([second, first]) == ["E00", "E26"]
    with pytest.raises(ValueError, match="unknown dependency"):
        dependency_order([second])
    first.requires = ["E26"]
    with pytest.raises(ValueError, match="cycle"):
        dependency_order([first, second])


def test_initial_screens_and_final_populations_remain_bounded():
    with pytest.raises(ValueError, match="300"):
        CampaignStep.model_validate(_step(source_cap=1000))
    with pytest.raises(ValueError, match="cannot cap"):
        CampaignStep.model_validate(_step(phase="final"))
    with pytest.raises(ValueError, match="2000"):
        CampaignStep.model_validate(_step(training_source_cap=2001))


def test_blueprint_tampering_and_excess_treatments_fail(tmp_path):
    path = _lock(tmp_path)
    lock, _ = load_campaign(path)
    Path(lock.blueprint.path).write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        load_campaign(path)
    path = _lock(tmp_path)
    data = yaml.safe_load(path.read_text())
    extra = _step(id="E00-extra")
    data["steps"].append(extra)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="exceed"):
        load_campaign(path)


def test_all_generative_roles_bind_hosted_and_no_fallback():
    config = ConfigModel.from_mapping(openrouter_only({}, "openrouter_gpt4o_mini"), warn_v1=False)
    for key, value in config.llm.routing.model_dump().items():
        assert value == (None if "fallback" in key else "openrouter_gpt4o_mini")
    assert config.llm.profiles["openrouter_gpt4o_mini"].provider["allow_fallbacks"] is False
    with pytest.raises(ValueError, match="OpenRouter"):
        openrouter_only({}, "local_llm_default")


def test_materialization_uses_strict_runner_declarations_without_final_paths(tmp_path):
    path = _lock(tmp_path)
    suite = materialize_campaign(path, tmp_path / "declarations", stage="screen")
    assert suite.sources[0].config.schema_version == 2
    assert suite.sources[0].config.implementation.status == "blocked"
    assert (
        suite.sources[0].config.screen.tasks[0].overlay["data"]["execution_mode"]
        == "global_alignment"
    )
    assert suite.sources[0].config.confirm.tasks[0].overlay == {"data": {}}
    assert suite.baseline_id == "R_v2"
    # Same lock is idempotent; previous declarations are never overwritten.
    materialize_campaign(path, tmp_path / "declarations", stage="screen")


def test_confirm_case_cannot_be_bound_to_development_step(tmp_path):
    path = _lock(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["cases"]["D0"]["role"] = "reporting"
    data["cases"]["D0"]["references"] = {}
    with pytest.raises(ValueError, match="matching case role"):
        CampaignLock.model_validate(data)


def test_global_retrieval_does_not_consume_local_pool(tmp_path):
    path = _lock(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["cases"]["D0"]["candidates"] = {"valid": _binding(tmp_path / "local.tsv")}
    raw["steps"][0]["execution_modes"] = ["global_alignment", "local_ranking"]
    path.write_text(yaml.safe_dump(raw))
    suite = materialize_campaign(path, tmp_path / "runs", stage="screen")
    tasks = suite.sources[0].config.screen.tasks
    assert tasks[0].overlay["data"]["candidates"] is None
    assert tasks[0].overlay["data"]["candidate_source"] == "generated"
    assert tasks[1].overlay["data"]["candidates"] == str(tmp_path / "local.tsv")


def test_dependency_null_publishes_baseline_but_pending_does_not():
    from types import SimpleNamespace

    from exact.experiments.campaign import dependency_blockers

    source = SimpleNamespace(
        config=SimpleNamespace(
            frozen_constants={"campaign_v2": {"requires": ["retrieval", "heads"]}}
        )
    )
    assert (
        dependency_blockers(
            source, {"retrieval": {"status": "screened_out"}, "heads": {"status": "selected"}}
        )
        == []
    )
    assert dependency_blockers(
        source, {"retrieval": {"status": "blocked"}, "heads": {"status": "selected"}}
    ) == ["retrieval"]


def test_directory_binding_survives_relocation_and_detects_changed_fact(tmp_path):
    import shutil

    from exact.experiments.campaign import InputBinding
    from exact.utils.provenance import sha256_path

    original = tmp_path / "original"
    original.mkdir()
    (original / "facts.csv").write_text("a,b,c\n")
    digest = sha256_path(original)
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    InputBinding(path=relocated, sha256=digest).verify(tmp_path)
    (relocated / "facts.csv").write_text("a,b,changed\n")
    with pytest.raises(ValueError, match="changed"):
        InputBinding(path=relocated, sha256=digest).verify(tmp_path)


def test_cooperative_stop_requires_explicit_resume_and_restores_handlers(tmp_path):
    import os
    import signal

    from exact.experiments.campaign import cooperative_signals

    prior = signal.getsignal(signal.SIGTERM)
    with cooperative_signals(tmp_path):
        os.kill(os.getpid(), signal.SIGTERM)
        assert (tmp_path / "STOP").is_file()
    assert signal.getsignal(signal.SIGTERM) == prior
    with pytest.raises(ValueError, match="--resume"):
        with cooperative_signals(tmp_path):
            pass
    with cooperative_signals(tmp_path, resume=True):
        assert not (tmp_path / "STOP").exists()
    assert len(list(tmp_path.glob("STOP.resumed-*"))) == 1


def test_g4_freezes_full_panel_without_opening_final_references(tmp_path):
    import json

    from exact.experiments.campaign import (
        campaign_identity,
        digest,
        freeze_final_selection,
        validate_final_selection,
    )

    path = _lock(tmp_path)
    raw = yaml.safe_load(path.read_text())
    blueprint_path = Path(raw["blueprint"]["path"])
    blueprint = yaml.safe_load(blueprint_path.read_text())
    blueprint["experiments"].append({"id": "E17", "initial_treatment_cap": "final_matrix_contract"})
    raw["blueprint"] = _binding(blueprint_path, yaml.safe_dump(blueprint))
    raw["cases"]["D1"] = {**raw["cases"]["D0"], "task": "sentinel"}
    for case in ("H0", "H1", "H2"):
        raw["cases"][case] = {
            **raw["cases"]["D0"],
            "role": "reporting",
            "references": {
                "test": {"path": str(tmp_path / f"{case}-UNOPENED.tsv"), "sha256": "f" * 64}
            },
        }
    gate = _step(id="G4", family="E17", phase="freeze", additional_cases=["D1"], requires=["E00"])
    final = _step(
        id="E17",
        family="E17",
        case="H0",
        additional_cases=["H1", "H2"],
        phase="final",
        budget_group="final",
        source_cap=None,
        seeds=[17, 29, 43],
    )
    final["arms"] = [
        {"id": "baseline", "role": "baseline"},
        {"id": "stack_all", "role": "candidate"},
    ]
    final["readiness"] = {
        arm: {
            "confirm": {
                "status": "confirm_ready",
                "reason": "fixture",
                "implemented_paths": ["exact/experiments/campaign.py"],
                "tests": ["campaign_v2_test.py"],
                "inspected_commit": "fixture",
            }
        }
        for arm in ("baseline", "stack_all")
    }
    final["selection"]["decisions"][0].update(baseline="baseline", candidates=["stack_all"])
    raw["steps"].extend([gate, final])
    raw.update(freeze_step="G4", composition_sources=["E00"])
    path.write_text(yaml.safe_dump(raw))
    lock, _ = load_campaign(path)
    evidence = {
        "stage": "screen",
        "suite_hash": campaign_identity(lock, tmp_path),
        "experiments": {"E00": {"status": "screened_out"}, "G4": {"status": "screened_out"}},
    }
    evidence["selection_hash"] = digest(evidence)
    selection_path = tmp_path / "development.json"
    selection_path.write_text(json.dumps(evidence))
    frozen = freeze_final_selection(path, selection_path, tmp_path / "G4.json")
    assert frozen["final_arm_task_count"] == 6
    assert frozen["experiments"]["E17"]["arms"]["stack_all"] == {}
    raw["final_selection"] = {
        "path": str(tmp_path / "G4.json"),
        "sha256": hashlib.sha256((tmp_path / "G4.json").read_bytes()).hexdigest(),
    }
    path.write_text(yaml.safe_dump(raw))
    suite = materialize_campaign(path, tmp_path / "final", stage="confirm")
    assert validate_final_selection(frozen, suite)["selection_hash"] == frozen["selection_hash"]
    from dataclasses import replace

    source = suite.sources[0]
    changed = replace(
        source,
        config=source.config.model_copy(
            update={"confirm": source.config.confirm.model_copy(update={"seeds": [17]})}
        ),
    )
    with pytest.raises(ValueError, match="seeds"):
        validate_final_selection(frozen, replace(suite, sources=(changed,)))
    task = source.config.confirm.tasks[0]
    tasks = [task.model_copy(update={"source_cap": 1}), *source.config.confirm.tasks[1:]]
    changed = replace(
        source,
        config=source.config.model_copy(
            update={"confirm": source.config.confirm.model_copy(update={"tasks": tasks})}
        ),
    )
    with pytest.raises(ValueError, match="population"):
        validate_final_selection(frozen, replace(suite, sources=(changed,)))
    frozen["experiments"]["E17"]["arms"]["stack_all"] = {"seed": 999}
    with pytest.raises(ValueError, match="identity"):
        validate_final_selection(frozen, suite)


def _ready_suite(tmp_path, *, composition=False, plan_only=False):
    from dataclasses import replace

    path = _lock(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["cases"]["D0"]["references"]["valid"] = _binding(
        tmp_path / "valid.tsv", "SrcEntity\tTgtEntity\nsource:1\ttarget:1\nsource:2\ttarget:2\n"
    )
    raw["cases"]["D0"]["source_universe"] = _binding(
        tmp_path / "sources.txt", "source:1\nsource:2\n"
    )
    step = raw["steps"][0]
    step["id"] = "E06"
    step["arms"][1]["overlay"] = {"matching": {"threshold": 0.5}}
    if composition:
        raw["cases"]["D1"] = {**raw["cases"]["D0"], "task": "sentinel"}
        gate = _step(
            id="G4", family="E00", phase="freeze", additional_cases=["D1"], requires=["E06"]
        )
        gate["arms"][1]["id"] = "core"
        gate["readiness"]["core"] = gate["readiness"].pop("replay")
        gate["selection"]["decisions"][0]["candidates"] = ["core"]
        raw["steps"].append(gate)
        raw.update(freeze_step="G4", composition_sources=["E06"])
    base = Path(raw["base_config"])
    baseline = {
        "baseline_id": "R_v2",
        "parent": "R_0",
        "exact_om_version": "fixture",
        "pyowlcore_version": "fixture",
        "config": str(base),
        "config_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "source_commit": "1" * 40,
        "source_tree_sha256": "2" * 64,
        "source_tree_files": 0,
        "status": "frozen_configuration",
        "experiment_flags": "disabled",
        "note": "fixture baseline identity",
    }
    raw["baseline_manifest"] = _binding(tmp_path / "baseline.yaml", yaml.safe_dump(baseline))
    measurement = _binding(tmp_path / "measurement.json", "{}")
    for item in raw["steps"]:
        item["estimate"] = {
            "cold_seconds": 0,
            "units": 1,
            "seconds_per_unit": 1,
            "peak_ram_gb": 1,
            "measurement_artifact": measurement,
        }
        for readiness in item["readiness"].values():
            readiness["screen"] = {
                "status": "screen_ready",
                "reason": "fake-runner control-flow check",
                "implemented_paths": ["exact/experiments/harness.py"],
                "tests": ["campaign_v2_test.py"],
                "inspected_commit": "fixture",
            }
    path.write_text(yaml.safe_dump(raw))
    suite = materialize_campaign(path, tmp_path / "declarations", stage="screen")
    root = tmp_path / "results" / suite.suite_id
    metadata = {
        "lock_path": str(path),
        "root": str(root),
        "stage": "screen",
        "reuse_plan_only": plan_only,
        "allowed_steps": [item["id"] for item in raw["steps"]],
        "budget_limits": {
            "envelopes_hours": {"foundation": 12},
            "requests_cap": 100,
            "tokens_cap": 1000,
        },
    }
    return replace(suite, campaign=metadata, model_lock_payload={"status": "unresolved"})


def _fake_model_run(command, **kwargs):
    import json

    from exact.core.actions.evaluation import run_evaluation

    wrapper = yaml.safe_load(Path(command[-1]).read_text())["job"]
    config = yaml.safe_load(Path(wrapper["config_file"]).read_text())
    output = Path(wrapper["output_dir"])
    (output / "alignment").mkdir(exist_ok=True)
    (output / "dataset").mkdir(exist_ok=True)
    (output / "dataset/candidate_pool_manifest.json").write_text(
        json.dumps({"fingerprint": "frozen-pool"})
    )
    second = "target:2" if config["matching"]["threshold"] == 0.5 else "wrong"
    alignment = output / "alignment/maps_global.tsv"
    alignment.write_text(
        f"SrcEntity\tTgtEntity\tScore\nsource:1\ttarget:1\t0.9\nsource:2\t{second}\t0.8\n"
    )
    run_evaluation(
        alignment,
        output / "evaluation",
        full_reference_file_path=Path(config["data"]["refs"]["valid"]),
        run_stats_path=output / "stats/run_stats.json",
        error_on_fail=True,
    )
    return 0, 0.01, 100


def test_shared_stage_plan_only_never_executes_selects_or_aggregates(tmp_path, monkeypatch):
    from exact.experiments import harness

    suite = _ready_suite(tmp_path, plan_only=True)

    def forbidden(*args, **kwargs):
        pytest.fail("plan-only attempted execution or outcome aggregation")

    monkeypatch.setattr(harness, "_run_subprocess", forbidden)
    monkeypatch.setattr(harness, "aggregate_stage", forbidden)
    monkeypatch.setattr(harness, "select_experiment", forbidden)
    assert (
        harness.run_stage(
            suite,
            stage="screen",
            output_root=tmp_path / "results",
            jobs=1,
            resume=False,
            workdir=tmp_path,
        )
        is None
    )
    root = Path(suite.campaign["root"])
    assert len(list((root / "recovery/plans").glob("*.json"))) == 2
    assert not (root / "attempts").exists()
    assert not (root / "screen/selection.json").exists()
    assert not (root / "budget.json").exists()


def test_shared_stage_retains_partial_selection_and_resumes_current_artifacts(
    tmp_path, monkeypatch
):
    import json

    from exact.experiments import harness

    suite = _ready_suite(tmp_path)
    calls = []

    def run(command, **kwargs):
        wrapper = yaml.safe_load(Path(command[-1]).read_text())["job"]
        output = Path(wrapper["output_dir"])
        arm = output.parts[-3]
        calls.append(arm)
        if arm == "replay" and calls.count("replay") == 1:
            (output / "interrupted.json").write_text("{}")
            return 130, 0.01, 100
        return _fake_model_run(command, **kwargs)

    monkeypatch.setattr(harness, "_run_subprocess", run)
    monkeypatch.setattr(harness, "build_dataset_inventory", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="incomplete cells"):
        harness.run_stage(
            suite,
            stage="screen",
            output_root=tmp_path / "results",
            jobs=1,
            resume=False,
            workdir=tmp_path,
        )
    root = Path(suite.campaign["root"])
    partial = json.loads((root / "screen/progress.json").read_text())
    assert partial["status"] == "interrupted"
    assert partial["experiments"]["E06"]["status"] == "deferred_runtime"
    selected = harness.run_stage(
        suite,
        stage="screen",
        output_root=tmp_path / "results",
        jobs=1,
        resume=True,
        workdir=tmp_path,
    )
    assert selected.is_file()
    assert calls == ["baseline", "replay", "replay"]
    current = json.loads((root / "screen/current-result-set.json").read_text())
    assert len(current["cells"]) == 2
    assert len(list((root / "screen/progress").glob("*.json"))) >= 2


def test_shared_stage_composition_uses_resolved_source_and_exact_expected_cells(
    tmp_path, monkeypatch
):
    import json

    from exact.experiments import harness

    suite = _ready_suite(tmp_path, composition=True)
    monkeypatch.setattr(harness, "_run_subprocess", _fake_model_run)
    monkeypatch.setattr(harness, "build_dataset_inventory", lambda *args, **kwargs: None)
    selected = harness.run_stage(
        suite,
        stage="screen",
        output_root=tmp_path / "results",
        jobs=1,
        resume=False,
        workdir=tmp_path,
    )
    record = json.loads(selected.read_text())
    assert record["experiments"]["G4"]["status"] == "selected"
    root = Path(suite.campaign["root"])
    progress = json.loads((root / "screen/progress.json").read_text())
    resolved = yaml.safe_load(Path(progress["declarations"]["G4"]["path"]).read_text())
    arms = {arm["id"]: arm["overlay"] for arm in resolved["arms"]}
    assert arms["baseline"] == {}
    assert arms["core"]["matching"]["threshold"] == 0.5
    assert len(json.loads((root / "screen/current-result-set.json").read_text())["cells"]) == 6


def test_ready_declarations_are_versioned_when_readiness_changes(tmp_path):
    path = _lock(tmp_path)
    original = materialize_campaign(path, tmp_path / "declarations", stage="screen")
    before = original.sources[0].path.read_bytes()
    raw = yaml.safe_load(path.read_text())
    raw["steps"][0]["readiness"]["replay"]["screen"][
        "reason"
    ] = "additional implementation evidence"
    path.write_text(yaml.safe_dump(raw))
    updated = materialize_campaign(path, tmp_path / "declarations", stage="screen")
    assert updated.suite_hash == original.suite_hash
    assert updated.sources[0].path != original.sources[0].path
    assert original.sources[0].path.read_bytes() == before


def test_baseline_guard_keeps_historical_parent_and_frozen_bytes(tmp_path):
    from exact.experiments.campaign import validate_baseline

    suite = _ready_suite(tmp_path)
    path = Path(suite.campaign["lock_path"])
    lock, _ = load_campaign(path)
    assert validate_baseline(lock, tmp_path).parent == "R_0"
    baseline = yaml.safe_load(Path(lock.baseline_manifest.path).read_text())
    baseline["parent"] = "changed_parent"
    lock.baseline_manifest = type(lock.baseline_manifest).model_validate(
        _binding(tmp_path / "changed-baseline.yaml", yaml.safe_dump(baseline))
    )
    with pytest.raises(ValueError, match="historical parent"):
        validate_baseline(lock, tmp_path)


def test_training_encoder_locks_cover_only_active_training_models():
    from exact.experiments.harness import (
        _bind_model_lock_revisions,
        _configured_model_ids,
    )

    config = {
        "candidates": {
            "encoder_finetune": {"mode": "contrastive", "training": {"base_model": "encoder"}},
            "cross_encoder": {"mode": "off", "training": {"base_model": "inactive"}},
        }
    }
    assert _configured_model_ids(config, hosted_only=True) == {"encoder"}
    lock = {
        "status": "complete",
        "models": {"encoder": {"requested_id": "encoder", "resolved_revision": "a" * 40}},
    }
    bound = _bind_model_lock_revisions(config, lock, require_complete=True, hosted_only=True)
    assert bound["candidates"]["encoder_finetune"]["training"]["revision"] == "a" * 40


def test_diagnostic_only_comparison_completes_without_promoting_controls(tmp_path, monkeypatch):
    import json
    from dataclasses import replace

    from exact.experiments import harness

    suite = _ready_suite(tmp_path)
    lock_path = Path(suite.campaign["lock_path"])
    raw = yaml.safe_load(lock_path.read_text())
    raw["steps"][0]["selection"]["decisions"] = []
    raw["steps"][0]["arms"][1].update(role="diagnostic", deployable=False)
    lock_path.write_text(yaml.safe_dump(raw))
    updated = materialize_campaign(lock_path, tmp_path / "diagnostic-declarations", stage="screen")
    suite = replace(updated, campaign=suite.campaign, model_lock_payload=suite.model_lock_payload)
    monkeypatch.setattr(harness, "_run_subprocess", _fake_model_run)
    monkeypatch.setattr(harness, "build_dataset_inventory", lambda *args, **kwargs: None)
    path = harness.run_stage(
        suite,
        stage="screen",
        output_root=tmp_path / "results",
        jobs=1,
        resume=False,
        workdir=tmp_path,
    )
    result = json.loads(path.read_text())["experiments"]["E06"]
    assert result["status"] == "complete"
    assert result["decision_mode"] == "diagnostic"
    assert result["combined_selected_overlay"] == {}
    assert result["decisions"] == []
    from exact.experiments.campaign import dependency_blockers

    dependent = replace(
        suite.sources[0],
        config=suite.sources[0].config.model_copy(
            update={"frozen_constants": {"campaign_v2": {"requires": ["E06"]}}}
        ),
    )
    assert dependency_blockers(dependent, {"E06": result}) == []
    with pytest.raises(ValueError, match="at least one decision"):
        suite.sources[0].config.__class__.model_validate(
            {**suite.sources[0].config.model_dump(mode="json"), "schema_version": 1}
        )


def test_component_exports_retain_fusion_ranker_and_judge_without_shared_controls(tmp_path):
    from dataclasses import replace

    from exact.experiments import harness
    from exact.experiments.schema import ArmConfig

    source = _ready_suite(tmp_path).sources[0]
    label_free = {
        "supervision": {
            "mode": "label_free",
            "components": {"fusion": "label_free", "rerank": "label_free", "accept": "label_free"},
        }
    }
    controls = harness.deep_merge(label_free, {"llm": {"experiment": {"gate": {"mode": "off"}}}})
    specifications = {
        "E19": (
            {"matching": {"fusion": {"enabled": True, "mode": "analytic_shipped"}}},
            {
                "matching": {
                    "fusion": {"enabled": True, "mode": "supervised", "artifact": "fusion.json"}
                },
                "supervision": {"components": {"fusion": "supervised"}},
            },
            ["matching.fusion", "supervision.components.fusion"],
        ),
        "E18": (
            {
                "selector": {"enabled": True, "rank_objective": "analytic"},
                "supervision": {"components": {"rerank": "supervised", "accept": "supervised"}},
            },
            {
                "selector": {
                    "enabled": True,
                    "rank_objective": "pairwise",
                    "artifact": "ranker.json",
                }
            },
            ["selector", "supervision.components.rerank", "supervision.components.accept"],
        ),
        "E25": (
            {},
            {
                "llm": {
                    "experiment": {
                        "gate": {"mode": "source_top_fraction", "quantile_fraction": 0.5}
                    }
                }
            },
            ["llm.experiment.gate"],
        ),
    }
    selections = {}
    for name, (baseline, candidate, paths) in specifications.items():
        baseline = harness.deep_merge(controls, baseline)
        candidate = harness.deep_merge(baseline, candidate)
        arms = [
            ArmConfig(id="baseline", role="baseline", overlay=baseline),
            ArmConfig(id="replay", role="candidate", overlay=candidate),
        ]
        config = source.config.model_copy(
            update={
                "experiment_id": name,
                "arms": arms,
                "frozen_constants": {"campaign_v2": {"policy_paths": paths}},
            }
        )
        chosen_source = replace(source, config=config)
        records = [
            {
                "experiment_id": name,
                "arm_id": arm,
                "task_id": "D0",
                "seed": 17,
                "status": "complete",
                "metrics": {"F1": score},
            }
            for arm, score in (("baseline", 0.5), ("replay", 0.7))
        ]
        selections[name] = harness.select_experiment(chosen_source, records)
        if name == "E19":
            records[1]["metrics"]["F1"] = 0.4
            rejected = harness.select_experiment(chosen_source, records)
            assert rejected["status"] == "screened_out"
            assert (
                harness.selected_experiment_overlays({"experiments": {name: rejected}}, [name])[
                    name
                ]["matching"]["fusion"]["mode"]
                == "analytic_shipped"
            )
    composed = {"matching": {"fusion": {"sigma_mode": "energy"}}}
    for overlay in harness.selected_experiment_overlays(
        {"experiments": selections}, list(specifications)
    ).values():
        composed = harness.deep_merge(composed, overlay)
    assert composed["matching"]["fusion"] == {
        "sigma_mode": "energy",
        "enabled": True,
        "mode": "supervised",
        "artifact": "fusion.json",
    }
    assert composed["selector"]["enabled"] is True
    assert composed["selector"]["rank_objective"] == "pairwise"
    assert composed["supervision"]["components"] == {
        "fusion": "supervised",
        "rerank": "supervised",
        "accept": "supervised",
    }
    assert composed["llm"]["experiment"]["gate"]["mode"] == "source_top_fraction"


def test_local_reference_binding_preserves_shared_training_gold(tmp_path):
    from exact.experiments.campaign import _case_task

    case = CaseBinding.model_validate(
        {
            "task": "fixture",
            "role": "development",
            "selection_reason": "official split roles",
            "references": {
                "train": _binding(tmp_path / "train.tsv"),
                "valid": _binding(tmp_path / "global.tsv"),
            },
            "local_references": {"valid": _binding(tmp_path / "local.tsv")},
        }
    )
    local = _case_task(case, "D0", "local_ranking", "valid", tmp_path)["overlay"]["data"]
    global_ = _case_task(case, "D0", "global_alignment", "valid", tmp_path)["overlay"]["data"]
    assert local["refs"]["valid"] == str(tmp_path / "local.tsv")
    assert global_["refs"]["valid"] == str(tmp_path / "global.tsv")
    assert local["refs"]["train"] == global_["refs"]["train"] == str(tmp_path / "train.tsv")


def test_e13_bound_views_change_graphs_without_changing_sources_or_gold(tmp_path):
    from exact.experiments.harness import build_cells

    path = _lock(tmp_path)
    raw = yaml.safe_load(path.read_text())
    bp_path = Path(raw["blueprint"]["path"])
    blueprint = yaml.safe_load(bp_path.read_text())
    blueprint["experiments"][0]["id"] = "E13"
    raw["blueprint"] = _binding(bp_path, yaml.safe_dump(blueprint))
    raw["cases"]["D0"]["capabilities"] = ["normalized_evidence"]
    step = raw["steps"][0]
    step.update(
        id="E13",
        family="E13",
        arm_inputs={
            "replay": {
                "source": _binding(
                    tmp_path / "source.csv", "SrcEntity,Relation,TgtEntity\nsource:1,label,one\n"
                ),
                "target": _binding(
                    tmp_path / "target.csv", "SrcEntity,Relation,TgtEntity\ntarget:1,label,one\n"
                ),
            }
        },
    )
    path.write_text(yaml.safe_dump(raw))
    suite = materialize_campaign(path, tmp_path / "views", stage="screen")
    source = suite.sources[0]
    source.config.implementation.status = "ready"
    cells = build_cells(suite, source, stage="screen", output_root=tmp_path / "results")
    assert len(cells) == 2
    first, second = [cell.resolved_config["data"] for cell in cells]
    assert {first["source"], second["source"]} == {
        str(tmp_path / "source.owl"),
        str(tmp_path / "source.csv"),
    }
    for field in ("source_universe", "refs", "reference_role", "execution_mode"):
        assert first[field] == second[field]
    Path(step["arm_inputs"]["replay"]["source"]["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        materialize_campaign(path, tmp_path / "tampered", stage="screen")
    step["arms"][1]["overlay"] = {"data": {"refs": {"valid": "different-gold.tsv"}}}
    with pytest.raises(ValueError, match="arm data overrides"):
        CampaignStep.model_validate(step)


@pytest.mark.parametrize("generate_rationales", [False, True])
def test_campaign_materializes_explicit_rationale_policy(tmp_path, generate_rationales):
    path = _lock(tmp_path)
    if generate_rationales:
        payload = yaml.safe_load(path.read_text())
        payload["generate_rationales"] = True
        path.write_text(yaml.safe_dump(payload))
    original = path.read_bytes()
    lock, _ = load_campaign(path)
    assert lock.generate_rationales is generate_rationales
    suite = materialize_campaign(path, tmp_path / "materialized", stage="screen")
    assert suite.sources
    assert all(source.config.generate_rationales is generate_rationales for source in suite.sources)
    assert path.read_bytes() == original
