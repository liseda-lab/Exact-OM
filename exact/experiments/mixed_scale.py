"""Corrected final-study preparation, separate from immutable historical locks."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from exact.utils.fitted_artifacts import fingerprint, freeze_json
from exact.utils.provenance import sha256_file

DESIGN = "exact-om-final-mixed-scale-20261009-v1"
MODES = ("global_alignment", "local_ranking")
SEEDS = (17, 29, 43)


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def verified(value):
    path = Path(value["path"])
    if binding(path) != value:
        raise ValueError("Changed final-study input binding")
    return path


def read_binding(value):
    return json.loads(verified(value).read_text())


def compile_design(*, target_supervised=None, equivalence_receipt=None, diagnostics=()):
    if target_supervised is False and not equivalence_receipt:
        raise ValueError("Omitting H0 controls requires inspected supervision-equivalence evidence")
    if equivalence_receipt:
        proof = read_binding(equivalence_receipt)
        if proof.get("target_label_free") is not True or proof.get("equivalent_fitted_dependencies") is not True:
            raise ValueError("Invalid supervision-equivalence evidence")
    cells = []
    def add(section, case, arm, mode, seed, **extra):
        cells.append(dict(id=f"{section}/{case}/{arm}/{mode}/seed-{seed}", section=section,
                          case=case, arm=arm, mode=mode, seed=seed, **extra))
    for case in ("H0", "H1", "H2"):
        for arm in ("baseline", "stack_all"):
            for mode in MODES:
                add("primary", case, arm, mode, 17, population="full", report_only=True)
    if target_supervised is not False:
        for mode in MODES:
            add("supervision", "H0", "label_free", mode, 17, population="full",
                required_when="selected_H0_uses_target_training_labels", condition_bound=target_supervised is True)
    for case in ("H0", "H1", "H2"):
        add("published", case, "pinned_LogMap", "global_alignment", 17,
            population="full", deterministic=True, independent_seed_replication=False)
    bounded_arms = ("baseline", "stack_all", "label_free")
    for case in ("D0_E03", "D1", "D_H2_valid"):
        for arm in bounded_arms:
            for mode in MODES:
                for seed in SEEDS:
                    add("bounded", case, arm, mode, seed, population="fixed_public_development_cohort",
                        source_cap=619 if case == "D0_E03" else 1000,
                        report_only=True, official_submission=False)
    if sum(d.get("kind") == "removal" for d in diagnostics) > 4 or sum(d.get("kind") == "interaction" for d in diagnostics) > 2:
        raise ValueError("Combined-system diagnostic ceiling exceeded")
    unique = {}
    for diagnostic in diagnostics:
        if diagnostic.get("kind") not in {"removal", "interaction"}:
            raise ValueError("Unknown combined-system diagnostic kind")
        if not diagnostic.get("objective") or diagnostic.get("disposition") not in {"new", "reuse", "inapplicable"}:
            raise ValueError("Every diagnostic needs an objective and explicit disposition")
        if diagnostic["disposition"] != "new":
            if not diagnostic.get("evidence"):
                raise ValueError("Reused/inapplicable objective needs its evidence binding")
            verified(diagnostic["evidence"])
            continue
        expected = {"removed"} if diagnostic["kind"] == "removal" else {"00", "01", "10", "11"}
        corners = diagnostic.get("corners", {})
        if set(corners) != expected:
            raise ValueError("Every admitted removal/interaction needs exact complete corners")
        for recipe in corners.values():
            if not isinstance(recipe.get("overlay"), dict) or "descendant_refits" not in recipe:
                raise ValueError("Exact overlays and descendant-refit policies are required")
            if recipe.get("shared_stack"):
                shared = read_binding(recipe["shared_stack"])
                if shared.get("overlay") != recipe["overlay"] or shared.get("descendant_refits") != recipe["descendant_refits"]:
                    raise ValueError("Shared stack corner differs from its frozen recipe")
                continue
            unique.setdefault(fingerprint(recipe), recipe)
    if len(unique) > 10:
        raise ValueError("At most ten additional combined-system recipes are permitted")
    for key, recipe in unique.items():
        for mode in MODES:
            add("components", "D0_E03", key, mode, 17, recipe=recipe, report_only=True,
                population="fixed_public_development_cohort", official_submission=False)
    return dict(schema_version=1, design_id=DESIGN, kind="mixed_scale_final_design",
                cells=cells, logical_cells=len(cells), primary_cells=12,
                full_exact_cells=12 + (0 if target_supervised is False else 2), published_unique_runs=3,
                bounded_cells=18 * len(bounded_arms), additional_component_cells=2 * len(unique),
                supervision_equivalence=equivalence_receipt, diagnostics=list(diagnostics),
                generate_rationales=False, primary_endpoint="paired_task_macro_global_F1_from_organizer",
                secondary_endpoint="local_MRR", historical_locks_rewritten=False,
                deployment_status="not_applied", target_supervised=target_supervised)


def corrected_cost_guard(protocol_binding, baseline_binding, selected_binding, parity_binding):
    """Evaluate the prospective 1.2x guard; missing hosted costs remain unknown."""
    protocol = read_binding(protocol_binding)
    parity = read_binding(parity_binding)
    if protocol.get("kind") != "prospective_deployment_cost_protocol" or protocol.get("max_ratio") != 1.2:
        raise ValueError("Expected the prospectively frozen G4 cost boundary and 1.2x guard")
    phases = protocol.get("required_phases", [])
    if (not {"preprocessing", "numerical", "hosted", "durable_outputs", "recovery"} <= set(phases)
            or len(phases) != len(set(phases))
            or protocol.get("diagnostic_replays_included") is not False):
        raise ValueError("Cost boundary must declare natural phases and exclude synthetic diagnostics")
    if (parity.get("kind") != "complete_G4_natural_output_parity" or parity.get("status") != "passed"
            or parity.get("namespace") == "qualification_only"
            or parity.get("fixture_outputs_promotable") is False
            or any(parity.get(field) is not True for field in (
                "natural_decisions_equal", "query_boundaries_equal", "candidate_coverage_equal",
                "rankings_equal", "mappings_equal", "fitted_artifacts_equal", "hosted_decisions_compatible"))):
        raise ValueError("G4 quality reuse requires decision and query-boundary parity")
    rows = [read_binding(b) for b in (baseline_binding, selected_binding)]
    keys = ("task_id", "seed", "hardware", "resources", "cache_regime", "population_sha256")
    if any(rows[0].get(k) is None or rows[0].get(k) != rows[1].get(k) for k in keys):
        raise ValueError("Costs must use the same task, seed, population, hardware, resources and cache regime")
    if rows[0].get("arm_id") == rows[1].get("arm_id") or any(not row.get("arm_id") for row in rows):
        raise ValueError("Costs require distinct declared baseline and selected arms")
    missing, totals = [], []
    for row in rows:
        if row.get("protocol") != protocol_binding or row.get("fixture_service_time_as_paid_cost", False):
            raise ValueError("Cost measurement does not bind the prospective natural boundary")
        service = row.get("phase_seconds", {}).get("hosted")
        disposition = row.get("hosted_cost_disposition")
        if service is not None and disposition not in {"measured_original", "verified_dependency_free"}:
            raise ValueError("Hosted cost must be original service evidence, never cached or fixture replay")
        if disposition == "verified_dependency_free":
            proof = read_binding(row["hosted_dependency_free_receipt"])
            if service != 0 or proof.get("hosted_enabled") is not False or proof.get("requests") != 0:
                raise ValueError("Zero hosted cost requires verified absence of a hosted dependency")
        total = 0.0
        for phase in protocol["required_phases"]:
            value = row.get("phase_seconds", {}).get(phase)
            if value is None:
                missing.append(phase)
            elif not isinstance(value, (int, float)) or value < 0 or not math.isfinite(value):
                raise ValueError("Invalid measured duration")
            else:
                total += value
        totals.append(total)
    ratio = totals[1] / totals[0] if not missing and totals[0] > 0 else None
    return dict(kind="corrected_G4_cost_guard", protocol=protocol_binding,
                baseline=baseline_binding, selected=selected_binding, parity=parity_binding,
                missing_phases=sorted(set(missing)), ratio=ratio,
                status="pending" if ratio is None else ("passed" if ratio <= 1.2 else "failed"),
                original_wall_time_and_spend_retained=True, final_outcomes_consumed=False)


def reconcile_g4_selection(source, quality_records, cost_guards, *, historical_selection,
                           final_outcomes_consumed=False):
    from exact.experiments.harness import select_experiment
    if final_outcomes_consumed or source.config.experiment_id != "G4":
        raise ValueError("G4 reconciliation is development-only and must precede final exposure")
    read_binding(historical_selection)
    corrected, measurements = copy.deepcopy(list(quality_records)), {}
    expected = {(arm.id, task.id, seed) for arm in source.config.arms if "screen" in arm.stages
                for task in source.config.screen.tasks if task.availability.status == "ready"
                for seed in source.config.screen.seeds}
    observed = [(r["arm_id"], r["task_id"], r["seed"]) for r in corrected
                if r.get("experiment_id") == "G4"]
    if not expected or len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Immutable G4 quality cells must cover the complete declared screen matrix")
    for guard_binding in cost_guards:
        guard = _validate_cost_guard(guard_binding)
        if guard.get("kind") != "corrected_G4_cost_guard" or guard.get("status") not in {"passed", "failed"}:
            raise ValueError("Every task requires a complete prospective cost guard")
        protocol = read_binding(guard["protocol"])
        for arm in ("baseline", "selected"):
            measured = read_binding(guard[arm])
            key = (measured["arm_id"], measured["task_id"], measured["seed"])
            if key in measurements:
                raise ValueError("Duplicate corrected cost cell")
            measurements[key] = sum(measured["phase_seconds"][p] for p in protocol["required_phases"])
    if set(measurements) != expected:
        raise ValueError("Corrected cost coverage differs from immutable G4 quality cells")
    for row in corrected:
        if row.get("experiment_id") == "G4":
            value = measurements[(row["arm_id"], row["task_id"], row["seed"])]
            row["inference_seconds"] = value
            row.setdefault("metrics", {})["inference_seconds"] = value
    return dict(kind="mechanically_reconciled_G4_selection", historical_selection=historical_selection,
                cost_guards=list(cost_guards), quality_records_sha256=fingerprint(list(quality_records)),
                declared_cost_cells=[list(key) for key in sorted(expected)],
                experiment_config_hash=source.raw_hash(), result=select_experiment(source, corrected),
                final_outcomes_consumed=False, historical_quality_and_costs_rewritten=False)


def _validate_cost_guard(value):
    guard = read_binding(value)
    if guard != corrected_cost_guard(guard["protocol"], guard["baseline"], guard["selected"], guard["parity"]):
        raise ValueError("Corrected cost guard differs from its recursively verified measurements")
    return guard


def _validate_mechanical(value, historical_selection):
    mechanical = read_binding(value)
    if (mechanical.get("kind") != "mechanically_reconciled_G4_selection"
            or mechanical.get("historical_selection") != historical_selection
            or mechanical.get("final_outcomes_consumed") is not False
            or mechanical.get("result", {}).get("status") not in {"selected", "screened_out"}):
        raise ValueError("Corrected G4 selection is not mechanically reconciled")
    declared = [tuple(key) for key in mechanical.get("declared_cost_cells", [])]
    observed = []
    for value in mechanical.get("cost_guards", []):
        guard = _validate_cost_guard(value)
        if guard["status"] not in {"passed", "failed"}:
            raise ValueError("Corrected cost coverage remains pending")
        for arm in ("baseline", "selected"):
            row = read_binding(guard[arm])
            observed.append((row["arm_id"], row["task_id"], row["seed"]))
    if (not declared or len(declared) != len(set(declared)) or len(observed) != len(set(observed))
            or set(observed) != set(declared)):
        raise ValueError("Corrected cost coverage differs from the declared G4 matrix")
    return mechanical


def freeze_corrected_selection(destination, *, historical_selection, cost_guard, fitting_recipes,
                               target_supervised, mechanical_selection, equivalence_receipt=None):
    from exact.experiments.campaign import digest
    prior = read_binding(historical_selection)
    claimed = prior.pop("selection_hash", None)
    if claimed != digest(prior) or prior.get("no_final_outcomes_consumed") is not True:
        raise ValueError("Historical final selection must be verified and outcome-free")
    costs = _validate_cost_guard(cost_guard)
    mechanical = _validate_mechanical(mechanical_selection, historical_selection)
    if (type(target_supervised) is not bool or not fitting_recipes
            or costs.get("status") not in {"passed", "failed"}
            or mechanical.get("kind") != "mechanically_reconciled_G4_selection"
            or mechanical.get("historical_selection") != historical_selection
            or mechanical.get("final_outcomes_consumed") is not False
            or mechanical.get("result", {}).get("status") not in {"selected", "screened_out"}
            or cost_guard not in mechanical.get("cost_guards", [])):
        raise ValueError("Corrected selection, supervision identity or fitting recipes are incomplete")
    for recipe in fitting_recipes.values():
        verified(recipe)
    record = dict(schema_version=1, kind="corrected_selection_and_fitting_freeze", design_id=DESIGN,
                  historical_selection=historical_selection, corrected_cost_guard=cost_guard,
                  mechanical_selection=mechanical_selection, fitting_recipes=fitting_recipes,
                  selected_overlay=mechanical["result"]["combined_selected_overlay"],
                  design=compile_design(target_supervised=target_supervised, equivalence_receipt=equivalence_receipt),
                  final_outcomes_consumed=False, selection_status="frozen", execution_ready=False)
    record["identity"] = fingerprint(record)
    return freeze_json(destination, record)


def validate_selection_freeze(value):
    frozen = read_binding(value)
    unhashed = dict(frozen)
    claimed = unhashed.pop("identity", None)
    if (claimed != fingerprint(unhashed) or frozen.get("kind") != "corrected_selection_and_fitting_freeze"
            or frozen.get("design_id") != DESIGN or frozen.get("selection_status") != "frozen"
            or not frozen.get("fitting_recipes") or frozen.get("final_outcomes_consumed") is not False):
        raise ValueError("Corrected G4 selection and fitting recipes are not frozen")
    for key in ("historical_selection", "corrected_cost_guard", "mechanical_selection"):
        verified(frozen[key])
    from exact.experiments.campaign import digest
    historical = read_binding(frozen["historical_selection"])
    claimed_history = historical.pop("selection_hash", None)
    if claimed_history != digest(historical) or historical.get("no_final_outcomes_consumed") is not True:
        raise ValueError("Historical final selection must be verified and outcome-free")
    _validate_cost_guard(frozen["corrected_cost_guard"])
    mechanical = _validate_mechanical(frozen["mechanical_selection"], frozen["historical_selection"])
    if (frozen["corrected_cost_guard"] not in mechanical["cost_guards"]
            or frozen.get("selected_overlay") != mechanical["result"]["combined_selected_overlay"]):
        raise ValueError("Frozen selection differs from mechanical G4 reconciliation")
    for recipe in frozen["fitting_recipes"].values():
        verified(recipe)
    design = frozen["design"]
    if design != compile_design(target_supervised=design["target_supervised"],
                                equivalence_receipt=design["supervision_equivalence"],
                                diagnostics=design.get("diagnostics", [])):
        raise ValueError("Frozen design differs from the corrected mixed-scale matrix")
    return frozen


def bind_diagnostic_cohort(destination, *, case, source_universe, public_queries,
                           target_population, selection_freeze=None, frozen_membership=None,
                           public_validation_descriptor=None):
    if case not in {"D0_E03", "D1", "D_H2_valid"}:
        raise ValueError("Unknown bounded diagnostic role")
    if case == "D_H2_valid":
        if selection_freeze is None:
            raise ValueError("H2 public validation requires corrected selection/fitting freeze first")
        validate_selection_freeze(selection_freeze)
        if public_validation_descriptor is None:
            raise ValueError("H2 requires an explicit immutable public validation descriptor")
        descriptor = read_binding(public_validation_descriptor)
        if (descriptor.get("kind") != "public_validation_descriptor"
                or descriptor.get("case") != "D_H2_valid" or descriptor.get("role") != "valid"
                or descriptor.get("reference_access") != "public_development"
                or descriptor.get("source_population_basis") != "eligible_ontology_entities_independent_of_reference_positives"
                or descriptor.get("source_universe") != source_universe
                or descriptor.get("public_queries") != public_queries
                or descriptor.get("target_population") != target_population
                or len(descriptor.get("dataset_revision", "")) != 40
                or any(c not in "0123456789abcdef" for c in descriptor.get("dataset_revision", ""))):
            raise ValueError("H2 descriptor must bind public validation inputs independently of reference positives")
        verified(descriptor["source_ontology"])
    # No H2 validation input is opened before the corrected selection/fitting gate.
    universe = verified(source_universe).read_text().splitlines()
    if len(set(universe)) != len(universe) or not universe:
        raise ValueError("Eligible development source universe must be nonempty and unique")
    if case == "D_H2_valid":
        prefix = DESIGN + "/H2-public-valid/"
        selected = sorted(universe, key=lambda iri: (hashlib.sha256((prefix + iri).encode()).hexdigest(), iri))[:1000]
    else:
        if frozen_membership is None:
            raise ValueError("D0/D1 require their existing frozen G4 membership")
        selected = verified(frozen_membership).read_text().splitlines()
        expected = 619 if case == "D0_E03" else 1000
        if len(selected) != expected or len(set(selected)) != expected or not set(selected) <= set(universe):
            raise ValueError("Existing G4 source membership count/identity is inconsistent")
    from exact.experiments.inputs import prepare_pool
    from exact.experiments.public_inference import validate_population
    target_record, _ = validate_population(verified(target_population))
    destination = Path(destination)
    prepared = prepare_pool(verified(public_queries), destination / "public", role="valid", expose_labels=False)
    original = [json.loads(line) for line in verified(prepared["outputs"]["queries"]).read_text().splitlines() if line]
    members = set(selected)
    queries = [query for query in original if query["source"] in members]
    record = dict(schema_version=1, kind="fixed_public_diagnostic_cohort", case=case,
                  role="report_only_development", selection_eligible=False, official_submission=False,
                  selection_freeze=selection_freeze, source_universe=source_universe,
                  public_validation_descriptor=public_validation_descriptor,
                  frozen_membership=frozen_membership, public_queries=public_queries,
                  target_population=target_population, full_target_count=target_record["count"],
                  universe_count=len(universe), source_ids=selected, source_count=len(selected),
                  source_ids_sha256=hashlib.sha256(('\n'.join(selected) + '\n').encode()).hexdigest(),
                  original_queries=queries, local_query_count=len(queries),
                  sources_without_local_queries=sorted(members - {q["source"] for q in queries}),
                  cohort_independent_of_execution_seed=True, reference_labels_used=False,
                  global_candidates_from_local_queries=False)
    return freeze_json(destination / "cohort.json", record)


def _population_identity(record):
    core = record["ontology_core"]
    return {"ontology_sha256": record["ontology"]["sha256"],
            "population_sha256": record["population"]["sha256"], "count": record["count"],
            "counts_by_kind": record["counts_by_kind"], "entity_kinds": sorted(record["entity_kinds"]),
            "filter_ignored_alignment_classes": record["filter_ignored_alignment_classes"],
            "fingerprints": core["fingerprints"],
            "source_documents": sorted(row["source_sha256"] for row in core["closure"]["source_documents"])}


def _validate_case_deployment(manifest, case, cell):
    from exact.experiments.public_inference import validate_population
    from exact.experiments.submission import _pools
    populations = {}
    for side in ("source", "target"):
        expected, _ = validate_population(verified(case[side + "_population"]))
        populations[side] = expected
        if binding(verified(manifest[side]))["sha256"] != expected["ontology"]["sha256"]:
            raise ValueError("Deployment belongs to a different ontology case")
        if cell["mode"] == "global_alignment":
            actual, _ = validate_population(verified(manifest[side + "_population_manifest"]))
            if _population_identity(actual) != _population_identity(expected):
                raise ValueError("Global deployment differs from its complete native population")
    if cell["mode"] == "global_alignment":
        if len(manifest["runs"]) != 1 or "public_candidates" in manifest:
            raise ValueError("Global deployment must retain a single complete global assignment")
        return populations
    receipt = read_binding(case["local_queries"])
    public = receipt["outputs"]["public_queries"]
    verified(public)
    verified(manifest["public_candidates"])
    queries = _pools(Path(public["path"]), manifest["track"])
    if (receipt.get("labels_exposed") is not False
            or receipt.get("original_query_rows") != len(queries)
            or case.get("query_count", len(queries)) != len(queries)
            or manifest.get("query_count") != len(queries)
            or manifest.get("queries") != queries
            or manifest["public_candidates"]["sha256"] != public["sha256"]
            or manifest.get("score_scope") != "original_query"):
        raise ValueError("Local deployment changed original query membership or denominator")
    empty = [index for index, query in enumerate(queries) if not query["candidates"]]
    indices = [index for run in manifest["runs"] for index in run["query_indices"]]
    if (any(type(index) is not int or not 0 <= index < len(queries) for index in indices)
            or manifest.get("empty_query_indices", []) != empty
            or len(indices) != len(set(indices))
            or set(indices) != set(range(len(queries))) - set(empty)):
        raise ValueError("Local deployment must cover every original nonempty query exactly once")
    for run in manifest["runs"]:
        sources = [queries[index]["source"] for index in run["query_indices"]]
        if len(sources) != len(set(sources)):
            raise ValueError("A local shard cannot union different original pools of one source")
    return populations


def prepare_successor_bundle(destination, *, registry, public_inputs, selection_freeze=None,
                             deployments=None, source_revision=None):
    """Prepare a review artifact without publishing, launching or resetting accounting."""
    registry = Path(registry)
    before = binding(registry)
    live = json.loads(registry.read_text())
    inputs = json.loads(Path(public_inputs).read_text())
    cases = inputs.get("cases", inputs)
    if set(cases) != {"H0", "H1", "H2"}:
        raise ValueError("All three existing full public populations must be bound")
    for case in cases.values():
        for field in ("source_population", "target_population", "local_queries"):
            verified(case[field])
    frozen = validate_selection_freeze(selection_freeze) if selection_freeze else None
    design = frozen["design"] if frozen else compile_design()
    blockers = ["explicit_rollout_authorization", "reviewed_source_and_spending_admission",
                "measured_complete_program_forecast", "verified_runtime_fitted_artifacts",
                "strict_output_coverage_validator_fixture_checks"]
    if frozen is None:
        blockers += ["corrected_G4_quality_cost_selection", "frozen_seed_specific_fitting_recipes",
                     "postfreeze_D_H2_valid_binding", "H0_supervision_identity"]
    dependencies = deployments or {}
    if set(dependencies) - {cell["id"] for cell in design["cells"]}:
        raise ValueError("Deployment binding names undeclared corrected cells")
    logical = []
    from exact.core.entities.configs.config import ConfigModel
    from exact.utils.frozen_inference import validate_inference_config
    for cell in design["cells"]:
        row = dict(cell, status="pending_binding", physical_execution=None)
        if cell["id"] in dependencies:
            if cell["section"] not in {"primary", "supervision"}:
                raise ValueError("Bounded and published cells require their own execution descriptors")
            manifest = read_binding(dependencies[cell["id"]])
            if manifest.get("kind") != "reference_free_inference" or manifest.get("run_eval") is not False:
                raise ValueError("Full inference requires the reference-free deployment manifest")
            if frozen is None or cell["id"] not in frozen["fitting_recipes"]:
                raise ValueError("Deployment requires its exact corrected selection/fitting recipe binding")
            recipe_binding = frozen["fitting_recipes"][cell["id"]]
            recipe = read_binding(recipe_binding)
            if (recipe.get("kind") != "frozen_final_fitting_recipe" or recipe.get("cell_id") != cell["id"]
                    or recipe.get("selected_config") != manifest.get("selected_config")
                    or recipe.get("runtime_fitted_artifacts") != manifest.get("runtime_fitted_artifacts")
                    or not isinstance(recipe.get("artifacts"), dict)):
                raise ValueError("Deployment differs from its corrected frozen fitting recipe")
            verified(recipe["selected_config"])
            if recipe.get("runtime_fitted_artifacts"):
                verified(recipe["runtime_fitted_artifacts"])
            populations = _validate_case_deployment(manifest, cases[cell["case"]], cell)
            for run in manifest["runs"]:
                config = ConfigModel.load_config(verified(run["config"]))
                from exact.experiments.rationale_policy import require_rationale_policy
                from exact.experiments.evidence_diagnostics import ordinary_inference_config
                mapping = config.model_dump(mode="json", by_alias=True)
                require_rationale_policy(mapping)
                if ordinary_inference_config(mapping)[1]["changed_paths"]:
                    raise ValueError("Ordinary deployment cannot contain synthetic diagnostic replays")
                fitted = validate_inference_config(config)
                if fitted is None:
                    raise ValueError("Deployment must load immutable fitted artifacts")
                if fitted["artifacts"] != recipe["artifacts"]:
                    raise ValueError("Deployment fitted artifact lineage differs from the frozen recipe")
                if config.seed != cell["seed"] or config.data.execution_mode != cell["mode"]:
                    raise ValueError("Deployment seed/mode differs from corrected logical cell")
                for side in ("source", "target"):
                    if binding(getattr(config.data, side))["sha256"] != manifest[side]["sha256"]:
                        raise ValueError("Deployment config ontology differs from its declared case")
                    if (sorted(config.matching.entity_kinds) != sorted(populations[side]["entity_kinds"])
                            or config.dataset.filter_ignored_alignment_classes
                            != populations[side]["filter_ignored_alignment_classes"]):
                        raise ValueError("Deployment config changed its native population policy")
                if cell["mode"] == "global_alignment":
                    if binding(config.data.source_universe)["sha256"] != populations["source"]["population"]["sha256"]:
                        raise ValueError("Global deployment config changed its full source universe")
                else:
                    from exact.experiments.submission import _pools, NIL_IRI
                    expected_queries = [manifest["queries"][index] for index in run["query_indices"]]
                    actual_queries = _pools(config.data.candidates, "bioml-local")
                    expected_pairs = [(query["source"], [t for t in query["candidates"] if t != NIL_IRI])
                                      for query in expected_queries]
                    if ([(query["source"], query["candidates"]) for query in actual_queries] != expected_pairs
                            or Path(config.data.source_universe).read_text().splitlines()
                            != sorted(query["source"] for query in expected_queries)):
                        raise ValueError("Local deployment config changed its original query shards")
                if cell["arm"] == "label_free":
                    components = ("retrieval", "fusion", "rerank", "llm", "accept", "calibration", "structure", "relation")
                    if (any(config.supervision.components.get(name, config.supervision.mode) != "label_free"
                            for name in components) or fitted["artifacts"]):
                        raise ValueError("Label-free control cannot reuse fitted heads without verified target-label-independent lineage")
            row.update(status="bound_not_admitted", inference=dependencies[cell["id"]],
                       fitting_recipe=recipe_binding,
                       physical_execution=dependencies[cell["id"]]["sha256"])
        logical.append(row)
    pending = live.get("pending", live.get("pending_batches", []))
    candidates = [row for row in pending if row.get("id") in {"E17-run-once-followup", "E17-published-run-once-followup"}]
    record = dict(schema_version=1, kind="offline_corrected_successor_bundle", design_id=DESIGN,
                  source_revision=source_revision, registry_snapshot=before,
                  public_inputs=binding(public_inputs), selection_freeze=selection_freeze,
                  design=design, logical_to_physical=logical, pending_rows_to_replace=candidates,
                  queue_replacement=dict(atomic_under_registry_lock=True, expected_registry_sha256=before["sha256"],
                                         preserve_all_other_rows=True, fresh_dispatch_nonces_required=True),
                  launchable=False, blockers=blockers, live_queue_changed=False,
                  historical_accounting_retained=True, generate_rationales=False,
                  spending_limits_changed=False, heavy_gpu_workers=1,
                  future_scientific_acceptance=["actual_output_coverage_and_submission_validation", "organizer_results"],
                  next_action="Bind missing receipts, prepare guarded descriptors, obtain rollout authorization, revalidate registry then atomically replace only unstarted E17 rows")
    if binding(registry) != before:
        raise ValueError("Registry changed while preparing offline successor proposal")
    return freeze_json(Path(destination) / "successor-bundle.json", record)
