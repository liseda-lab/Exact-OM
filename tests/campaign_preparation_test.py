from pathlib import Path

import pytest

from exact.experiments.campaign import InputBinding, dependency_order, load_campaign
from exact.experiments.preparation import prepare_campaign


def _cases():
    cases = {}
    for name in ["D0", "D1", "N0", "P0", "K0", "R0_case", "T0", "H0", "H1", "H2"]:
        cases[name] = {
            "task": name,
            "role": "reporting" if name.startswith("H") else "development",
            "selection_reason": "capability selected before outcomes",
        }
    return cases


def test_preparation_covers_every_family_without_executing_or_inventing_readiness(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cases = _cases()
    path = prepare_campaign(
        root / "specs/experiments/campaign-v2.yaml",
        root / "exact/default_config.yaml",
        {"cases": cases},
        tmp_path / "prepared",
    )
    lock, _ = load_campaign(path)
    assert {step.family for step in lock.steps} == {f"E{i:02}" for i in range(27)}
    order = dependency_order(lock.steps)
    for before, after in [
        ("E05", "E20"),
        ("E20", "E01"),
        ("E20", "E26"),
        ("E26", "E19"),
        ("E12-retrieval", "E12"),
        ("E13", "E13-enrichment"),
        ("E25", "E07"),
        ("E07", "E25-trust"),
        ("E10-analytic", "E10"),
        ("E10", "E19"),
        ("E19", "E18"),
        ("E22", "E22-policy"),
        ("G4", "E17"),
    ]:
        assert order.index(before) < order.index(after)
    by_id = {step.id: step for step in lock.steps}
    e01 = by_id["E01"]
    assert len(e01.arms) == 6
    assert e01.selection.decisions[0].baseline == "threshold_unrestricted"
    assert set(e01.selection.decisions[0].candidates) == {
        "greedy",
        "mutual_best",
        "stable_marriage",
        "assignment_accepted_utility",
    }
    assert e01.selection.decisions[0].required_controls == ["threshold_unrestricted", "greedy"]
    assert e01.policy_paths == [
        "matching.extraction",
        "matching.cardinality",
        "matching.target_cardinality",
    ]
    for arm in e01.arms:
        unrestricted = arm.id == "threshold_unrestricted"
        matching = arm.overlay["matching"]
        assert matching["threshold"] == 0.7
        assert matching["cardinality"] == (None if unrestricted else 1)
        assert matching["target_cardinality"] == (None if unrestricted else 1)
        assert matching["extraction"] == {
            "mode": "threshold" if unrestricted else arm.id,
            "anchor_conflict_policy": "compete",
        }
        if arm.id == "assignment_legacy":
            assert arm.role == "diagnostic" and not arm.deployable
        assert arm.overlay["llm"]["experiment"] == {"enabled": True, "gate": {"mode": "off"}}
        assert arm.overlay["supervision"]["mode"] == "label_free"
        assert set(arm.overlay["supervision"]["components"].values()) == {"label_free"}
        assert arm.overlay["selector"]["runtime_enabled"] is False
    extraction_only = [
        {key: value for key, value in arm.overlay.items() if key != "matching"}
        for arm in by_id["E01"].arms
    ]
    assert all(item == extraction_only[0] for item in extraction_only)
    assert len(by_id["E10-analytic"].arms) == 6
    assert len(by_id["E10"].arms) == 2
    assert by_id["E10"].selection.decisions[0].baseline == "winner_only"
    assert "selected_E10_analytic_setting" in by_id["E10"].inherits
    assert "selected_E10_analytic_setting" in by_id["E19"].inherits
    assert by_id["E04-listwise"].selection.decisions[0].baseline == "nil_heuristic"
    assert by_id["E07"].source_cap == by_id["E25"].source_cap == 200
    assert "E13-enrichment" not in by_id["G4"].requires
    assert "E22-policy" not in by_id["G4"].requires
    assert set(by_id["G4"].requires) == {"E00", *lock.composition_sources}
    assert by_id["E22-policy"].case == "D1"
    assert "selected_heads" in by_id["E22"].inherits
    assert by_id["E22-policy"].inherits == ["selected_heads", "E05_initial"]
    assert by_id["E04"].selection.decisions[0].metric == "nil.nil_aware.F1"
    assert any(
        guard.metric == "nil.non_nil_MRR" for guard in by_id["E04"].selection.decisions[0].guards
    )
    pool_miss = by_id["E04-pool-miss"]
    assert pool_miss.case == "N0"
    assert pool_miss.source_cap == by_id["E04"].source_cap == 300
    assert pool_miss.seeds == by_id["E04"].seeds == [17]
    assert pool_miss.execution_modes == by_id["E04"].execution_modes == ["global_alignment"]
    assert pool_miss.requires == ["E00", "pool_freeze", "E05_initial", "E04"]
    assert pool_miss.inherits == by_id["E04"].inherits == ["E05_initial"]
    assert all(
        record.status == "blocked_input_resolution"
        for roles in pool_miss.readiness.values()
        for record in roles.values()
    )
    assert by_id["E17"].source_cap is None
    assert by_id["E17"].seeds == [17, 29, 43]
    for identifier in ["E00", "E13", "E25-trust", "E04-pool-miss"]:
        assert by_id[identifier].selection.decisions == []
    assert not any(
        record.status.endswith("_ready")
        for step in lock.steps
        for roles in step.readiness.values()
        for record in roles.values()
    )
    assert set((tmp_path / "prepared").iterdir()) == {
        path,
        path.parent / "arm-prerequisites.yaml",
        path.parent / "blueprint.yaml",
    }
    with pytest.raises(FileExistsError):
        prepare_campaign(
            root / "specs/experiments/campaign-v2.yaml",
            root / "exact/default_config.yaml",
            {"cases": cases},
            path.parent,
        )


def test_instance_pool_is_selected_before_evidence_and_class_policies_stay_separate(tmp_path):
    from exact.experiments import harness
    from exact.experiments.campaign import materialize_campaign

    root = Path(__file__).resolve().parents[1]
    path = prepare_campaign(
        root / "specs/experiments/campaign-v2.yaml",
        root / "exact/default_config.yaml",
        {"cases": _cases()},
        tmp_path / "prepared",
    )
    lock, _ = load_campaign(path)
    steps = {step.id: step for step in lock.steps}
    retrieval = steps["E12-retrieval"]
    assert {arm.id for arm in retrieval.arms} == {"labels", "multi_view"}
    left, right = retrieval.arms
    assert left.overlay["matching"] == right.overlay["matching"]
    assert retrieval.selection.decisions[0].metric == "candidate_recall"
    assert steps["E12"].inherits == ["instance_pool_freeze"]
    assert "multi_view" not in {arm.id for arm in steps["E12"].arms}
    assert steps["E23"].inherits == ["instance_pool_freeze"]
    assert steps["E26"].inherits == ["E05_initial"]
    assert steps["E20"].inherits == ["E05_initial"]
    assert "pool_freeze" in steps["E26"].requires
    suite = materialize_campaign(path, tmp_path / "materialized", stage="screen")
    for identifier, overlay in (
        ("E12", {"candidates": {"multi_view": {"mode": "labels_relations"}}}),
        ("E26", {"candidates": {"encoder": "fixture/frozen-label-free"}}),
    ):
        source = suite.by_id[identifier]
        source.config.implementation.status = "ready"
        cells = harness.build_cells(
            suite, source, stage="screen", output_root=tmp_path / "runs", inherited_overlay=overlay
        )
        assert cells
        for cell in cells:
            if identifier == "E12":
                assert (
                    cell.resolved_config["candidates"]["multi_view"]["mode"] == "labels_relations"
                )
            else:
                assert cell.resolved_config["candidates"]["encoder"] == "fixture/frozen-label-free"
                assert (
                    cell.resolved_config["supervision"]["components"]["retrieval"] == "label_free"
                )
    records = {
        arm: {("K0", 17): {"candidate_pool_fingerprint": arm}} for arm in ("labels", "multi_view")
    }
    status, _, error = harness._candidate_pool_guard("E12-retrieval", records)
    assert status == "allowed_retrieval_treatment" and error is None
    _, _, error = harness._candidate_pool_guard("E12", records)
    assert error is not None


def test_enrichment_and_progressive_combination_cannot_block_or_fake_initial_results(tmp_path):
    from types import SimpleNamespace

    from exact.core.entities.configs.yaml_io import load_yaml_mapping
    from exact.experiments.campaign import dependency_blockers

    root = Path(__file__).resolve().parents[1]
    path = prepare_campaign(
        root / "specs/experiments/campaign-v2.yaml",
        root / "exact/default_config.yaml",
        {"cases": _cases()},
        tmp_path / "prepared",
    )
    lock, _ = load_campaign(path)
    steps = {step.id: step for step in lock.steps}
    assert {arm.id for arm in steps["E13"].arms} == {"owl_parity", "csv_parity"}
    assert steps["E13"].produces == ["typed_pool_freeze"]
    enrichment = steps["E13-enrichment"]
    assert enrichment.case == "R0_enrichment"
    assert lock.cases[enrichment.case].heldout_case == "R1_case"
    assert enrichment.phase == "late"
    assert all(
        record.status == "blocked_input_resolution"
        for roles in enrichment.readiness.values()
        for record in roles.values()
    )
    assert "E13-enrichment" not in steps["E14"].requires
    source = SimpleNamespace(
        config=SimpleNamespace(frozen_constants={"campaign_v2": {"requires": ["E00", "E13"]}})
    )
    assert (
        dependency_blockers(
            source,
            {
                "E00": {"status": "complete"},
                "E13": {"status": "complete"},
                "E13-enrichment": {"status": "blocked_input_resolution"},
            },
        )
        == []
    )
    inventory = load_yaml_mapping(path.parent / "arm-prerequisites.yaml")
    for family in ("E05", "E20"):
        assert "combined" not in {arm.id for arm in steps[family].arms}
        combined = next(arm for arm in inventory[family] if arm["arm"] == "combined")
        assert combined["status"] == "conditional_unadmitted"
        assert combined["comparisons"] == []
        assert "supported component results" in combined["reason"]


def test_g4_enforces_scoped_local_quality_and_original_inference_cost(tmp_path):
    from exact.experiments import harness
    from exact.experiments.campaign import materialize_campaign

    root = Path(__file__).resolve().parents[1]
    path = prepare_campaign(
        root / "specs/experiments/campaign-v2.yaml",
        root / "exact/default_config.yaml",
        {"cases": _cases()},
        tmp_path / "prepared",
    )
    suite = materialize_campaign(path, tmp_path / "materialized", stage="screen")
    source = suite.by_id["G4"]
    assert {task.overlay["data"]["execution_mode"] for task in source.config.screen.tasks} == {
        "global_alignment",
        "local_ranking",
    }
    assert source.config.design.cost_bound == 1.2
    records = []
    for task in source.config.screen.tasks:
        mode = task.overlay["data"]["execution_mode"]
        for arm in ("baseline", "core"):
            metric = (
                {"F1": 0.5 if arm == "baseline" else 0.6}
                if mode == "global_alignment"
                else {"local.MRR": 0.7 if arm == "baseline" else 0.697}
            )
            records.append(
                {
                    "experiment_id": "G4",
                    "arm_id": arm,
                    "task_id": task.id,
                    "execution_mode": mode,
                    "seed": 17,
                    "status": "complete",
                    "metrics": metric,
                    "candidate_recall": 0.9,
                    "inference_seconds": 10 if arm == "baseline" else 12,
                    "wall_seconds": 0.001,
                }
            )
    assert harness.select_experiment(source, records)["status"] == "selected"
    records[1]["inference_seconds"] = 12.01
    result = harness.select_experiment(source, records)
    assert result["status"] == "screened_out"
    assert (
        "guard_failed:default_inference_ceiling"
        in result["decisions"][0]["candidate_evaluations"]["core"]["rejection_reasons"]
    )
    records[1]["inference_seconds"] = 12
    local = next(
        row
        for row in records
        if row["arm_id"] == "core" and row["execution_mode"] == "local_ranking"
    )
    local["metrics"]["local.MRR"] = 0.69
    assert harness.select_experiment(source, records)["status"] == "screened_out"
    records.remove(local)
    with pytest.raises(ValueError, match="unequal"):
        harness.select_experiment(source, records)


def test_published_comparator_is_frozen_and_counts_three_additional_final_designs(tmp_path):
    import json
    from dataclasses import replace

    from exact.core.entities.configs.yaml_io import dump_yaml_document
    from exact.experiments.campaign import (
        campaign_identity,
        digest,
        freeze_final_selection,
        materialize_campaign,
        validate_final_selection,
    )
    from exact.utils.provenance import sha256_file

    root = Path(__file__).resolve().parents[1]
    cases = _cases()
    for name in ("H0", "H1", "H2"):
        universe = tmp_path / f"{name}.sources"
        universe.write_text("source:one\nsource:two\n")
        cases[name].update(
            source_universe={
                "path": str(universe),
                "sha256": sha256_file(universe),
            },
            references={
                "test": {"path": str(tmp_path / f"{name}.unopened.gold"), "sha256": "b" * 64}
            },
        )
    binding = {
        "matcher": "logmap",
        "bundle": {"path": str(tmp_path / "pinned-bundle"), "sha256": "c" * 64},
        "jar": "logmap.jar",
        "timeout_seconds": 30,
        "java_heap_gb": 1,
        "java_threads": 1,
    }
    path = prepare_campaign(
        root / "specs/experiments/campaign-v2.yaml",
        root / "exact/default_config.yaml",
        {"cases": cases, "published_matcher": binding},
        tmp_path / "prepared",
    )
    lock, _ = load_campaign(path)
    evidence = {
        "stage": "screen",
        "suite_hash": campaign_identity(lock, path.parent),
        "experiments": {
            step.id: {"status": "screened_out"} for step in lock.steps if step.phase != "final"
        },
    }
    evidence["experiments"]["E13-enrichment"] = {"status": "implementation_blocked"}
    evidence["selection_hash"] = digest(evidence)
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(evidence))
    final = tmp_path / "final.json"
    frozen = freeze_final_selection(path, selected, final)
    assert frozen["optional_branch_dispositions"]["E13-enrichment"]["empirical_result"] is False
    assert (
        frozen["optional_branch_dispositions"]["E13-enrichment"]["status"]
        == "blocked_input_resolution"
    )
    assert frozen["final_arm_task_count"] == 21
    assert frozen["experiments"]["E17"]["independent_group_counts"] == {"H0": 2, "H1": 2, "H2": 2}
    assert frozen["experiments"]["E17"]["design"]["practical_effect"] == 0.003
    assert frozen["experiments"]["E17"]["design"]["multiplicity"] == "holm"
    assert not any(tmp_path.glob("*.unopened.gold"))
    assert frozen["experiments"]["E17-published"]["arms"] == {"logmap": {}}
    assert frozen["experiments"]["E17-published"]["published_matchers"] == {"logmap": binding}
    lock.final_selection = InputBinding(path=final, sha256=sha256_file(final))
    path.write_text(dump_yaml_document(lock.model_dump(mode="json")))
    suite = materialize_campaign(path, tmp_path / "materialized", stage="confirm")
    validate_final_selection(frozen, suite)
    source = suite.by_id["E17-published"]
    altered = source.config.arms[0].model_copy(
        update={"published_matcher": {**binding, "java_heap_gb": 2}}
    )
    changed = replace(source, config=source.config.model_copy(update={"arms": [altered]}))
    suite = replace(
        suite,
        sources=tuple(
            changed if item.config.experiment_id == source.config.experiment_id else item
            for item in suite.sources
        ),
    )
    with pytest.raises(ValueError, match="published matcher changed"):
        validate_final_selection(frozen, suite)


def test_source_label_binding_never_enters_scoring_or_planning_reads(tmp_path):
    from exact.experiments import harness
    from exact.experiments.campaign import materialize_campaign

    root = Path(__file__).resolve().parents[1]
    cases = _cases()
    unseen = tmp_path / "unopened.source-labels.json"
    cases["N0"]["evaluation_source_labels"] = {"path": str(unseen), "sha256": "a" * 64}
    path = prepare_campaign(
        root / "specs/experiments/campaign-v2.yaml",
        root / "exact/default_config.yaml",
        {"cases": cases},
        tmp_path / "prepared",
    )
    suite = materialize_campaign(path, tmp_path / "materialized", stage="screen")
    source = suite.by_id["E04"]
    source.config.implementation.status = "ready"
    source.config.arms = [next(arm for arm in source.config.arms if arm.role == "baseline")]
    cells = harness.build_cells(suite, source, stage="screen", output_root=tmp_path / "runs")
    assert cells
    for cell in cells:
        assert cell.diagnostics == {
            "role": "development",
            "reference_role": "valid",
            "evaluation_source_labels": {"path": str(unseen), "sha256": "a" * 64},
        }
        assert str(unseen) not in harness.canonical_json(cell.resolved_config)
    assert not unseen.exists()


def test_prepared_design_survives_workspace_blueprint_revision(tmp_path):
    root = Path(__file__).resolve().parents[1]
    blueprint = tmp_path / "working-design.yaml"
    original = (root / "specs/experiments/campaign-v2.yaml").read_bytes()
    blueprint.write_bytes(original)
    path = prepare_campaign(
        blueprint, root / "exact/default_config.yaml", {"cases": _cases()}, tmp_path / "frozen"
    )
    blueprint.write_text("kind: subsequent-scientific-revision\n")
    lock, design = load_campaign(path)
    assert design["kind"] == "exact_om_campaign_design"
    assert lock.blueprint.verify(path.parent).read_bytes() == original
    lock.blueprint.verify(path.parent).write_text("changed frozen design\n")
    with pytest.raises(ValueError, match="materialized input changed"):
        load_campaign(path)


def _public_n0_cases():
    # Planning binds metadata only: deliberately absent files must never be opened.
    cases = _cases()

    def binding(name):
        return {"path": "public-N0/" + name, "sha256": "a" * 64}

    cases["N0"].update(
        source_universe=binding("valid.sources.txt"),
        references={"valid": binding("valid.reference.tsv")},
        frozen_global_candidates={"valid": binding("valid.candidates.tsv")},
        evaluation_source_labels=binding("valid.source_labels.tsv"),
        evaluation_candidate_labels=binding("valid.confirmed_candidates.tsv"),
        evaluation_label_semantics="benchmark_pool",
        reference_completeness="known_incomplete",
        negative_policy="confirmed_only",
        overlay={"matching": {"nil": {"label_semantics": "benchmark_pool"}}},
    )
    return cases


def test_pool_miss_binds_public_n0_gold_removal_without_changing_main_control(tmp_path):
    from copy import deepcopy

    from exact.core.entities.configs.config import ConfigModel
    from exact.experiments.fitting_recipes import fitting_requirements
    from exact.experiments.harness import deep_merge

    root = Path(__file__).resolve().parents[1]
    cases = _public_n0_cases()
    original = deepcopy(cases)
    path = prepare_campaign(
        root / "specs/experiments/campaign-v2.yaml",
        root / "exact/default_config.yaml",
        {"cases": cases},
        tmp_path / "prepared",
    )
    lock, _ = load_campaign(path)
    steps = {step.id: step for step in lock.steps}
    main, diagnostic = steps["E04"], steps["E04-pool-miss"]
    assert [arm.id for arm in main.arms] == ["nil_off", "nil_heuristic", "nil_fitted"]
    assert diagnostic.arms[0] == main.arms[0]
    assert diagnostic.selection.decisions == []
    treatment = diagnostic.arms[1]
    assert treatment.role == "diagnostic" and not treatment.deployable
    expected = deepcopy(main.arms[0].overlay)
    expected["matching"]["nil"].update(
        pool_miss_development_reference=str(path.parent / "public-N0/valid.reference.tsv"),
        pool_miss_candidate_labels={
            "path": str(path.parent / "public-N0/valid.confirmed_candidates.tsv"),
            "sha256": "a" * 64,
        },
    )
    assert treatment.overlay == expected
    resolved = ConfigModel.from_mapping(
        deep_merge({"config_version": 2, **cases["N0"]["overlay"]}, treatment.overlay)
    )
    assert resolved.matching.nil.label_semantics == "benchmark_pool"
    assert resolved.matching.nil.mode == "off"
    assert resolved.matching.nil.pool_miss_candidate_labels is not None
    assert diagnostic.design.primary_endpoint == "nil.nil_aware.F1"
    assert any("identities match exactly" in text for text in diagnostic.design.assumptions)
    assert all(
        record.status == "implementing"
        for roles in diagnostic.readiness.values()
        for record in roles.values()
    )
    requirements = fitting_requirements()["E04/pool_miss"]
    assert "complete_N0_benchmark_candidate_labels" in requirements
    assert "paired_unchanged_N0_development_pool" in requirements
    assert "complete_development_reference" not in requirements
    assert cases == original
    assert lock.cases["N0"].negative_policy == "confirmed_only"
    assert lock.cases["N0"].reference_completeness == "known_incomplete"
    assert not (path.parent / "public-N0").exists()


@pytest.mark.parametrize("override", [{"case": "D0"}, {"source_cap": 200}, {"inherits": ["E03"]}])
def test_pool_miss_rejects_old_or_unpaired_overrides(tmp_path, override):
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="paired N0 control contract"):
        prepare_campaign(
            root / "specs/experiments/campaign-v2.yaml",
            root / "exact/default_config.yaml",
            {"cases": _public_n0_cases(), "steps": {"E04-pool-miss": override}},
            tmp_path / "prepared",
        )


def test_pool_miss_rejects_inherited_fitted_heads(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cases = _public_n0_cases()
    cases["N0"]["overlay"]["matching"]["fusion"] = {"artifact": "D0-trained.json"}
    with pytest.raises(ValueError, match="cannot import fitted heads"):
        prepare_campaign(
            root / "specs/experiments/campaign-v2.yaml",
            root / "exact/default_config.yaml",
            {"cases": cases},
            tmp_path / "prepared",
        )


def test_pool_miss_rejects_drift_from_main_control_settings(tmp_path):
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="paired N0 control contract"):
        prepare_campaign(
            root / "specs/experiments/campaign-v2.yaml",
            root / "exact/default_config.yaml",
            {"cases": _public_n0_cases(), "steps": {"E04": {"source_cap": 200}}},
            tmp_path / "prepared",
        )
