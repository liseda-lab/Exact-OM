import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exact.experiments import mixed_scale as mixed
from exact.experiments.inputs import prepare_pool
from exact.experiments.public_inference import prepare_population, prepare_public_inference
from exact.utils.fitted_artifacts import freeze_json


def saved(path, payload):
    freeze_json(path, payload)
    return mixed.binding(path)


def test_mixed_design_retains_primary_and_bounded_roles(tmp_path):
    design = mixed.compile_design()
    assert design["logical_cells"] == 71
    assert design["primary_cells"] == 12
    assert design["full_exact_cells"] == 14
    assert design["published_unique_runs"] == 3
    bounded = [row for row in design["cells"] if row["section"] == "bounded"]
    assert len(bounded) == 54
    assert {row["seed"] for row in bounded} == {17, 29, 43}
    assert all(row["report_only"] and not row["official_submission"] for row in bounded)
    with pytest.raises(ValueError, match="equivalence"):
        mixed.compile_design(target_supervised=False)
    proof = saved(tmp_path / "equivalence.json", {
        "target_label_free": True, "equivalent_fitted_dependencies": True,
    })
    reduced = mixed.compile_design(target_supervised=False, equivalence_receipt=proof)
    assert reduced["logical_cells"] == 69 and reduced["bounded_cells"] == 54


def test_component_design_requires_complete_corners_and_caps(tmp_path):
    recipe = lambda i: {"overlay": {"component": i}, "descendant_refits": []}
    interaction = {"kind": "interaction", "objective": "check combined interaction",
                   "disposition": "new", "corners": {key: recipe(i) for i, key in enumerate(("00", "01", "10", "11"))}}
    assert mixed.compile_design(diagnostics=[interaction])["additional_component_cells"] == 8
    broken = copy.deepcopy(interaction)
    del broken["corners"]["11"]
    with pytest.raises(ValueError, match="complete corners"):
        mixed.compile_design(diagnostics=[broken])
    removals = [{"kind": "removal", "objective": str(i), "disposition": "new",
                 "corners": {"removed": recipe(i + 10)}} for i in range(5)]
    with pytest.raises(ValueError, match="ceiling"):
        mixed.compile_design(diagnostics=removals)


def cost_inputs(tmp_path, *, task="D0-global", seed=17, selected_seconds=12):
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol = saved(tmp_path / "protocol.json", {
        "kind": "prospective_deployment_cost_protocol", "max_ratio": 1.2,
        "required_phases": ["preprocessing", "numerical", "hosted", "durable_outputs", "recovery"],
        "diagnostic_replays_included": False,
    })
    parity = saved(tmp_path / "parity.json", {"natural_decisions_equal": True, "query_boundaries_equal": True})
    common = {"protocol": protocol, "task_id": task, "seed": seed,
              "hardware": "fixture", "resources": {"gpu": 1}, "cache_regime": "cold",
              "population_sha256": "public-population", "hosted_cost_disposition": "measured_original"}
    baseline = saved(tmp_path / "baseline.json", dict(common, arm_id="baseline", phase_seconds={
        "preprocessing": 0, "numerical": 8, "hosted": 1, "durable_outputs": 1, "recovery": 0,
    }))
    selected = saved(tmp_path / "selected.json", dict(common, arm_id="selected", phase_seconds={
        "preprocessing": 0, "numerical": selected_seconds, "hosted": 1, "durable_outputs": 1, "recovery": 0,
    }))
    return protocol, baseline, selected, parity


def test_corrected_cost_guard_is_pending_for_unmeasured_hosted_and_compares_task_seed(tmp_path):
    args = cost_inputs(tmp_path)
    assert mixed.corrected_cost_guard(*args)["status"] == "failed"
    selected = mixed.read_binding(args[2])
    selected["phase_seconds"]["hosted"] = None
    unknown = saved(tmp_path / "unknown.json", selected)
    assert mixed.corrected_cost_guard(args[0], args[1], unknown, args[3])["status"] == "pending"
    selected["phase_seconds"]["hosted"] = 1
    for field, value in (("seed", 29), ("task_id", "D1-local"), ("cache_regime", "warm")):
        changed = dict(selected, **{field: value})
        changed_binding = saved(tmp_path / (field + ".json"), changed)
        with pytest.raises(ValueError, match="same task, seed"):
            mixed.corrected_cost_guard(args[0], args[1], changed_binding, args[3])
    narrow = mixed.read_binding(args[0])
    narrow["required_phases"] = ["numerical"]
    with pytest.raises(ValueError, match="natural phases"):
        mixed.corrected_cost_guard(saved(tmp_path / "narrow.json", narrow), *args[1:])
    selected["hosted_cost_disposition"] = "cache_replay"
    with pytest.raises(ValueError, match="original service"):
        mixed.corrected_cost_guard(args[0], args[1], saved(tmp_path / "replay.json", selected), args[3])


def selection_fixture(tmp_path, monkeypatch):
    from exact.experiments.campaign import digest
    historical = {"no_final_outcomes_consumed": True, "kind": "fixture_historical_selection"}
    historical["selection_hash"] = digest(historical)
    historical = saved(tmp_path / "historical.json", historical)
    source = SimpleNamespace(config=SimpleNamespace(
        experiment_id="G4", arms=[SimpleNamespace(id=arm, stages=["screen"]) for arm in ("baseline", "selected")],
        screen=SimpleNamespace(tasks=[SimpleNamespace(id="D0-global", availability=SimpleNamespace(status="ready"))], seeds=[17]),
    ), raw_hash=lambda: "immutable-config-hash")
    records = [{"experiment_id": "G4", "arm_id": arm, "task_id": "D0-global", "seed": 17,
                "inference_seconds": 999, "metrics": {"F1": score, "inference_seconds": 999}}
               for arm, score in (("baseline", 0.7), ("selected", 0.8))]
    captured = []
    def select(source, corrected):
        captured.extend(copy.deepcopy(corrected))
        return {"status": "selected", "combined_selected_overlay": {"matching": {"threshold": 0.8}}}
    monkeypatch.setattr("exact.experiments.harness.select_experiment", select)
    cost_args = cost_inputs(tmp_path / "cost", selected_seconds=9)
    guard = saved(tmp_path / "guard.json", mixed.corrected_cost_guard(*cost_args))
    return historical, source, records, guard, captured


def test_reconcile_requires_complete_declared_matrix_and_preserves_quality(tmp_path, monkeypatch):
    historical, source, records, guard, captured = selection_fixture(tmp_path, monkeypatch)
    original = copy.deepcopy(records)
    result = mixed.reconcile_g4_selection(source, records, [guard], historical_selection=historical)
    assert records == original
    assert [row["metrics"]["F1"] for row in captured] == [0.7, 0.8]
    assert [row["inference_seconds"] for row in captured] == [10, 11]
    assert len(result["declared_cost_cells"]) == 2
    for incomplete in (records[:1], records + [records[0]]):
        with pytest.raises(ValueError, match="complete declared screen matrix"):
            mixed.reconcile_g4_selection(source, incomplete, [guard], historical_selection=historical)
    source.config.screen.tasks.append(SimpleNamespace(id="D1-local", availability=SimpleNamespace(status="ready")))
    with pytest.raises(ValueError, match="complete declared screen matrix"):
        mixed.reconcile_g4_selection(source, records, [guard], historical_selection=historical)


def selection_freeze(tmp_path, monkeypatch):
    historical, source, records, guard, _ = selection_fixture(tmp_path, monkeypatch)
    mechanical = saved(tmp_path / "mechanical.json", mixed.reconcile_g4_selection(
        source, records, [guard], historical_selection=historical))
    recipe = saved(tmp_path / "recipe.json", {"seed": 17, "fitting_inputs": "fixture-only"})
    output = tmp_path / "freeze.json"
    mixed.freeze_corrected_selection(output, historical_selection=historical, cost_guard=guard,
        fitting_recipes={"H0/17": recipe}, target_supervised=True, mechanical_selection=mechanical)
    return mixed.binding(output)


def test_freeze_recursively_checks_measurements_without_reading_training_labels(tmp_path, monkeypatch):
    frozen = selection_freeze(tmp_path, monkeypatch)
    assert mixed.validate_selection_freeze(frozen)["selection_status"] == "frozen"
    measured = tmp_path / "cost" / "selected.json"
    payload = json.loads(measured.read_text())
    payload["phase_seconds"]["hosted"] = 0
    measured.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Changed final-study input"):
        mixed.validate_selection_freeze(frozen)


def test_freeze_rejects_self_consistent_but_reduced_design(tmp_path, monkeypatch):
    frozen = selection_freeze(tmp_path, monkeypatch)
    payload = mixed.read_binding(frozen)
    payload["design"]["cells"].pop()
    payload.pop("identity")
    payload["identity"] = mixed.fingerprint(payload)
    tampered = saved(tmp_path / "reduced-freeze.json", payload)
    with pytest.raises(ValueError, match="corrected mixed-scale matrix"):
        mixed.validate_selection_freeze(tampered)


def native_case(tmp_path, name="H0"):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    source, target = root / "source.ofn", root / "target.ofn"
    source.write_text("Ontology(Declaration(Class(<urn:s>)) Declaration(Class(<urn:without-query>)))")
    target.write_text("Ontology(Declaration(Class(<urn:t1>)) Declaration(Class(<urn:t2>)))")
    populations = {}
    for side, ontology in (("source", source), ("target", target)):
        path = root / (side + ".population.txt")
        prepare_population(ontology, path, entity_kinds=["class"])
        populations[side + "_population"] = mixed.binding(path.with_suffix(".txt.manifest.json"))
    public = root / "public.tsv"
    public.write_text("SrcEntity\tTgtCandidates\nurn:s\t['urn:t1']\nurn:s\t['urn:t2', 'urn:t1']\n")
    prepared = prepare_pool(public, root / "local", role="test", expose_labels=False)
    populations["local_queries"] = mixed.binding(root / "local" / "test.inputs.json")
    populations["query_count"] = 2
    return populations, source, target, Path(prepared["outputs"]["public_queries"]["path"])


def test_successor_preparation_preserves_registry_and_remains_offline(tmp_path):
    case, _, _, _ = native_case(tmp_path)
    inputs = saved(tmp_path / "inputs.json", {name: case for name in ("H0", "H1", "H2")})
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"pending": [{"id": "E17-run-once-followup"}, {"id": "unrelated"}], "history": ["retained"]}))
    before = registry.read_bytes()
    bundle = mixed.prepare_successor_bundle(tmp_path / "successor", registry=registry, public_inputs=inputs["path"])
    assert registry.read_bytes() == before
    assert len(bundle["logical_to_physical"]) == 71
    assert not bundle["launchable"] and not bundle["live_queue_changed"]
    assert bundle["pending_rows_to_replace"] == [{"id": "E17-run-once-followup"}]
    assert "corrected_G4_quality_cost_selection" in bundle["blockers"]


def test_deployment_retains_case_full_population_and_original_query_rows(tmp_path):
    from exact.core.entities.configs.config import ConfigModel
    case, source, target, queries = native_case(tmp_path)
    config = ConfigModel.from_mapping({"config_version": 2, "supervision": {"mode": "label_free"},
                                      "dataset": {"filter_ignored_alignment_classes": False}})
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(config.model_dump(mode="json")))
    global_manifest = json.loads(prepare_public_inference(selected, tmp_path / "global", source=source,
        target=target, track="bioml-global").read_text())
    mixed._validate_case_deployment(global_manifest, case, {"mode": "global_alignment"})
    local_manifest = json.loads(prepare_public_inference(selected, tmp_path / "local", source=source,
        target=target, track="bioml-local", public_candidates=queries).read_text())
    mixed._validate_case_deployment(local_manifest, case, {"mode": "local_ranking"})
    invalid_index = copy.deepcopy(local_manifest)
    invalid_index["runs"][1]["query_indices"] = [True]
    with pytest.raises(ValueError, match="every original nonempty query"):
        mixed._validate_case_deployment(invalid_index, case, {"mode": "local_ranking"})
    wrong = copy.deepcopy(case)
    wrong["source_population"] = case["target_population"]
    with pytest.raises(ValueError, match="different ontology case"):
        mixed._validate_case_deployment(global_manifest, wrong, {"mode": "global_alignment"})
    local_manifest["query_count"] = 1
    with pytest.raises(ValueError, match="membership or denominator"):
        mixed._validate_case_deployment(local_manifest, case, {"mode": "local_ranking"})


def test_h2_validation_gate_precedes_any_validation_input_read(tmp_path, monkeypatch):
    missing = {"path": str(tmp_path / "must-not-be-read"), "sha256": "missing"}
    with pytest.raises(ValueError, match="freeze first"):
        mixed.bind_diagnostic_cohort(tmp_path, case="D_H2_valid", source_universe=missing,
                                    public_queries=missing, target_population=missing)
    frozen = selection_freeze(tmp_path / "freeze", monkeypatch)
    with pytest.raises(ValueError, match="explicit immutable public validation descriptor"):
        mixed.bind_diagnostic_cohort(tmp_path, case="D_H2_valid", source_universe=missing,
            public_queries=missing, target_population=missing, selection_freeze=frozen)


def test_control_rejects_supervised_artifact_despite_relabelled_component_modes(tmp_path):
    from exact.core.entities.configs.config import ConfigModel
    case, source, target, queries = native_case(tmp_path)
    head = saved(tmp_path / "fitted-fusion.json", {"kind": "fitted_fusion", "fit_provenance": {
        "training_reference_sha256": "target-training-labels",
    }})
    config = ConfigModel.from_mapping({"config_version": 2, "run": {"seed": 17},
        "supervision": {"mode": "label_free"}, "dataset": {"filter_ignored_alignment_classes": False},
        "matching": {"fusion": {"mode": "analytic_fitted", "artifact": head["path"]}}})
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(config.model_dump(mode="json")))
    manifest = prepare_public_inference(selected, tmp_path / "inference", source=source,
        target=target, track="bioml-local", public_candidates=queries)
    registry = tmp_path / "registry.json"
    registry.write_text('{"pending": []}')
    inputs = saved(tmp_path / "inputs.json", {name: case for name in ("H0", "H1", "H2")})
    with pytest.raises(ValueError, match="target-label-independent lineage"):
        mixed.prepare_successor_bundle(tmp_path / "successor", registry=registry, public_inputs=inputs["path"],
            deployments={"supervision/H0/label_free/local_ranking/seed-17": mixed.binding(manifest)})


def test_h2_cohort_is_fixed_report_only_and_retains_duplicate_source_queries(tmp_path, monkeypatch):
    frozen = selection_freeze(tmp_path / "freeze", monkeypatch)
    case, source, _, queries = native_case(tmp_path)
    source_record = mixed.read_binding(case["source_population"])
    universe = source_record["population"]
    public = mixed.binding(queries)
    descriptor = saved(tmp_path / "validation.json", {
        "kind": "public_validation_descriptor", "case": "D_H2_valid", "role": "valid",
        "reference_access": "public_development", "dataset_revision": "a" * 40,
        "source_population_basis": "eligible_ontology_entities_independent_of_reference_positives",
        "source_universe": universe, "public_queries": public,
        "target_population": case["target_population"], "source_ontology": mixed.binding(source),
    })
    results = [mixed.bind_diagnostic_cohort(tmp_path / name, case="D_H2_valid", source_universe=universe,
        public_queries=public, target_population=case["target_population"], selection_freeze=frozen,
        public_validation_descriptor=descriptor) for name in ("first", "second")]
    assert results[0] == results[1]
    assert results[0]["source_count"] == 2 and results[0]["local_query_count"] == 2
    assert results[0]["sources_without_local_queries"] == ["urn:without-query"]
    assert [query["candidates"] for query in results[0]["original_queries"]] == [["urn:t1"], ["urn:t2", "urn:t1"]]
    assert not results[0]["selection_eligible"] and not results[0]["official_submission"]
    assert results[0]["full_target_count"] == 2
