"""Protocol-driven repair training with bounded teachers and decoded development selection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import tempfile
import time
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import pyowl_core as owl

from exact.repair.graph import FEATURE_SCHEMA_V3, EffectivePreparation
from exact.repair.learning import (
    SUPPORT_READOUT_IDENTITY,
    OwlTeacherOracle,
    RepairLabel,
    SemanticTargetSpec,
    TeacherCache,
    benefit_losses,
    collect_sampled_repairs,
    conditional_proposal_loss,
    covered_proposal_loss,
    enumerate_teacher,
    evaluate_teacher,
    fidelity_comparison_losses,
    generated_checkpoint_criterion,
    interaction_loss,
    risk_loss,
    sample_conditioned_marginals,
    support_loss,
    support_targets,
    teacher_marginals,
)
from exact.repair.records import (
    candidate_cost,
    canonical_hash,
    canonical_json,
    make_objective,
)
from exact.repair.workers import bounded_call
from tools.repair.corpus import GeneratedCase
from tools.repair.training_report import publish_report, read_report

DEFAULT_PROFILE = (
    ("delete", 0.1),
    ("edit", 0.05),
    ("ontology_edit", 0.02),
    ("human_authored_ontology_edit", 0.03),
)


# The pre-fix trainer stopped at a bounded case deadline before any optimization.
# These pins admit only that implementation and unchanged graph/model/teacher code.
_ACQUISITION_DEADLINE_PREDECESSOR = (
    "e71392f8cbc31a0932cde613f8357e20b0f6cd5efe824a737d8561ce9544da24"
)
_ACQUISITION_UNCHANGED_DEPENDENCIES = (
    "a820a6a062a810a6756fe379d8e69ee3709027b0659a85424cb1075d226af99d"
)
# Same dependencies with the coherent-endpoint fallback. Retained acquisitions
# must prove they never needed that fallback before this pin permits reuse.
_ACQUISITION_ENDPOINT_DEPENDENCIES = (
    "56ef92798d0abc0b9405cacce5a647d76575fe3761aee163d523b7600843a347"
)


def _check_acquisition_endpoints(saved, identity_options):
    from exact.repair.candidates import _original_mapping_relation

    completed = saved.get("acquisition_completed", [])
    retained = {case_id for epoch, case_id in completed if epoch == 0}
    if len(retained) != len(completed):
        raise ValueError("Acquisition recovery requires epoch-zero acquisitions")
    retained.add(saved["pending_acquisition"].get("case_id"))
    cases = {case.case_id: case for case, _ in identity_options["training"]}
    if not retained <= cases.keys() or not set(saved.get("sampled_training", {})) <= retained:
        raise ValueError("Retained acquisitions are not in the frozen training split")
    for case_id in retained:
        for obj in cases[case_id].problem.objects:
            if (
                obj.kind == "mapping"
                and _original_mapping_relation(obj, obj.source_entity, obj.target_entity, "class")
                is None
            ):
                raise ValueError("Retained acquisition depends on changed endpoint generation")
    return sorted(retained)


def _acquisition_deadline_recovery(saved, identity_options, warm_start_hash, dependencies):
    """Admit a diagnosed pre-optimization deadline without editing checkpoint identity.

    All input/configuration dependencies must reproduce the predecessor identity.
    The pending case is finalized as bounded/unvisited, never granted more time.
    """
    pending = saved.get("pending_acquisition", {})
    collection = pending.get("collection_state", {})
    expected = canonical_hash(
        (identity_options, warm_start_hash, _ACQUISITION_DEADLINE_PREDECESSOR)
    )
    if not (
        dependencies in {_ACQUISITION_UNCHANGED_DEPENDENCIES, _ACQUISITION_ENDPOINT_DEPENDENCIES}
        and saved.get("schema") == "exact-repair/training-state/v3"
        and saved.get("recovery_revision") == "exact-phase-resume/v3.1"
        and saved.get("identity") == expected
        and saved.get("phase") == "acquisition"
        and saved.get("next_epoch") == saved.get("next_offset") == saved.get("optimized") == 0
        and not saved.get("history")
        and not saved.get("optimizer", {}).get("state")
        and saved.get("best_state") is None
        and pending.get("epoch") == 0
        and "proposed" in pending
        and collection.get("schema") == "exact-repair/collection-state/v3.2"
        and collection.get("collection_identity")
        == canonical_hash(collection.get("collection_dependencies"))
    ):
        raise ValueError("Acquisition deadline recovery dependencies or phase are incompatible")
    endpoint_check = {}
    if dependencies == _ACQUISITION_ENDPOINT_DEPENDENCIES:
        endpoint_check = dict(
            retained_acquisition_cases=_check_acquisition_endpoints(saved, identity_options),
            endpoint_reuse="original anchored bundles; corrected fallback not consulted",
        )
    return dict(
        migration="bounded-acquisition-deadline/v1",
        source_identity=expected,
        source_implementation=_ACQUISITION_DEADLINE_PREDECESSOR,
        reused="model, optimizer, RNG, completed acquisition and partial labels",
        invalidated="pending case continuation after its exhausted deadline",
        budgets_reset=False,
        **endpoint_check,
    )


def save_training_state(path: Path, state: dict) -> None:
    """Atomically publish a tensors-and-primitives checkpoint, including optimizer/RNG."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".training-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _publish_training_checkpoint(state: dict, directory: Path) -> dict:
    """Publish immutable weights through the existing atomic checkpoint writer."""
    directory.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".selected-", dir=directory)
    os.close(fd)
    temporary = Path(name)
    try:
        save_training_state(temporary, state)
        with temporary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            size = os.fstat(stream.fileno()).st_size
        path = directory / (digest + ".pt")
        os.replace(temporary, path)
        return {
            "schema": "exact-repair/checkpoint-artifact/v3",
            "path": str(path.resolve()),
            "sha256": digest,
            "size_bytes": size,
        }
    finally:
        temporary.unlink(missing_ok=True)


def _read_training_checkpoint(artifact: dict, directory: Path) -> dict:
    """Validate descriptor and bytes before any CPU weights-only deserialization."""
    import torch

    if (
        not isinstance(artifact, dict)
        or set(artifact) != {"schema", "path", "sha256", "size_bytes"}
        or artifact["schema"] != "exact-repair/checkpoint-artifact/v3"
    ):
        raise ValueError("Expected a v3 checkpoint artifact descriptor")
    digest, size = artifact["sha256"], artifact["size_bytes"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(size) is not int
        or size < 1
        or not isinstance(artifact["path"], str)
    ):
        raise ValueError("Invalid checkpoint artifact digest, size or path")
    path = Path(artifact["path"])
    if (
        not path.is_absolute()
        or path != directory.resolve() / (digest + ".pt")
        or path.is_symlink()
    ):
        raise ValueError("Checkpoint artifact is outside its declared output directory")
    try:
        with path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != size:
                raise ValueError("Checkpoint artifact size mismatch")
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("Checkpoint artifact integrity mismatch")
            stream.seek(0)
            checkpoint = torch.load(stream, weights_only=True, map_location="cpu")
    except FileNotFoundError as error:
        raise ValueError("Checkpoint artifact is missing") from error
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("model_schema") != "exact-repair/model/v3"
    ):
        raise ValueError("Checkpoint artifact requires model schema v3")
    return checkpoint


def _freeze_training_model(problem, model, *, seconds: float, **options):
    """Keep CUDA objects inside the supervised trainer; transport immutable CPU bytes."""
    from exact.repair.pipeline import (
        bounded_freeze_checkpoint,
        bounded_freeze_neural_round,
    )

    if model.empty_bundle.device.type != "cuda":
        return bounded_freeze_neural_round(problem, model, seconds=seconds, **options)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="repair-frozen-model-") as temporary:
        path = Path(temporary) / "model.pt"
        save_training_state(
            path,
            {
                "model_schema": f"exact-repair/model/{model.revision}",
                "config": model.config,
                "metadata": model.metadata,
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
            },
        )
        return bounded_freeze_checkpoint(
            problem,
            str(path),
            seconds=max(0.001, seconds - (time.monotonic() - started)),
            **options,
        )


def _assignment_label(
    case: GeneratedCase,
    assignment: tuple[int, ...],
    profile: tuple,
    desired_family_weight: float = 1.0,
    false_positive_weight: float = 1.0,
    semantic_target: SemanticTargetSpec | None = None,
) -> RepairLabel:
    from exact.repair.kernel import materialize
    from exact.repair.owl import OwlVerifier, snapshot_from_axioms

    axioms, active = materialize(case.problem, assignment)
    verifier = OwlVerifier("hermit", backend="python")
    snapshot = snapshot_from_axioms(axioms)
    report = verifier.check_theory(
        snapshot,
        case.problem.policy.monitored_classes,
        required=case.problem.policy.required,
        prohibited=case.problem.policy.prohibited,
        activated=active,
    )
    cost = sum(
        candidate_cost(obj, obj.candidates[choice], profile)
        for obj, choice in zip(case.problem.objects, assignment)
    )
    auxiliary = support_targets(case.problem, assignment, report, axioms, active)
    if report.logical_status != "VERIFIED_FEASIBLE":
        feasible = False if report.logical_status == "VERIFIED_INFEASIBLE" else None
        return RepairLabel(assignment, feasible, None, cost, support_targets=auxiliary)
    target = semantic_target or SemanticTargetSpec(
        canonical_hash(case.probes), desired_family_weight, false_positive_weight
    )
    semantic = target.evaluate(OwlTeacherOracle(verifier, snapshot, case.probes), case.probes)
    return RepairLabel(assignment, True, semantic.benefit, cost, semantic.outcomes, auxiliary)


def _verify_intended(case: GeneratedCase) -> bool:
    """Verify the evaluator-only intended parent even when its candidate is withheld."""
    from exact.repair.owl import OwlVerifier, snapshot_from_axioms

    verifier = OwlVerifier("hermit", backend="python")
    snapshot = snapshot_from_axioms(case.intended_theory)
    report = verifier.check_theory(
        snapshot,
        case.problem.policy.monitored_classes,
        activated=case.intended_active,
        required=case.problem.policy.required,
        prohibited=case.problem.policy.prohibited,
    )
    return (
        report.logical_status == "VERIFIED_FEASIBLE"
        and evaluate_teacher(
            OwlTeacherOracle(verifier, snapshot, case.probes), case.probes
        ).complete
    )


def label_case(
    case: GeneratedCase,
    *,
    max_assignments: int = 256,
    deadline_seconds: float = 30.0,
    call_seconds: float = 5.0,
    profile: tuple = DEFAULT_PROFILE,
    desired_family_weight: float = 1.0,
    false_positive_weight: float = 1.0,
) -> TeacherCache:
    """Verify the intended clean parent, then enumerate whole-case labels under deadlines."""
    from importlib.metadata import version

    if case.control == "ambiguous":
        raise ValueError("Ambiguous observations have no single supervised target")
    if max_assignments < 1 or deadline_seconds <= 0 or call_seconds <= 0:
        raise ValueError("Teacher caps and deadlines must be positive")
    semantic_target = SemanticTargetSpec(
        canonical_hash(case.probes), desired_family_weight, false_positive_weight
    )
    started = time.monotonic()
    intended = bounded_call(_verify_intended, case, timeout=min(call_seconds, deadline_seconds))
    if intended.status != "complete" or intended.value is not True:
        raise ValueError(
            "Generated intended parent has not been verified feasible and query-complete"
        )

    def label(assignment: tuple[int, ...]) -> RepairLabel:
        remaining = deadline_seconds - (time.monotonic() - started)
        result = bounded_call(
            _assignment_label,
            case,
            assignment,
            profile,
            desired_family_weight,
            false_positive_weight,
            semantic_target,
            timeout=min(call_seconds, max(0.0, remaining)),
        )
        if result.status == "complete":
            return cast(RepairLabel, result.value)
        return RepairLabel(assignment, None, None, 0.0)

    cache = enumerate_teacher(
        tuple(len(obj.candidates) for obj in case.problem.objects),
        label,
        hashes={
            "input": case.problem.content_hash,
            "patch": canonical_hash(case.problem.objects),
            "policy": case.problem.policy.content_hash,
            "query": canonical_hash(case.probes),
            "inventory": canonical_hash(tuple(obj.candidates for obj in case.problem.objects)),
            "backend": canonical_hash(("pyhermit", version("pyhermit"), "python")),
            "profile": canonical_hash(profile),
            "teacher_weights": canonical_hash((desired_family_weight, false_positive_weight)),
            "semantic_target": semantic_target.content_hash,
        },
        max_assignments=max_assignments,
        deadline_seconds=max(0.001, deadline_seconds - (time.monotonic() - started)),
    )
    if case.schema_revision == "v3":
        from dataclasses import replace

        cache = replace(cache, schema="exact-repair/teacher-cache/v3")
    return cache


def _label_payload(case: GeneratedCase, directory: Path, **options) -> dict:
    from tools.repair.prepare import publish_label_cache

    return publish_label_cache(label_case(case, **options), directory)


def _retryable_label_transport(row: Mapping[str, Any]) -> bool:
    """Only retry lost result transport, never a logical unknown or partial cache."""
    return (
        row.get("status") == "unverified_parent"
        and row.get("detail")
        == "Label worker error: ValueError: worker result exceeds the transport frame limit"
    )


def decoded_development(
    objective, cache: TeacherCache, *, max_checks: int = 256, deadline_seconds: float = 10.0
) -> dict[str, Any]:
    """Decode the actual frozen integer model objective using cached policy decisions.

    Each solve is the production RC2 master. Only proved infeasibility adds a
    logical exclusion. An unvisited/unknown selection ends this validation run;
    it is never silently replaced by the best labelled feasible assignment.
    """
    from exact.repair.maxsat import solve_master

    if max_checks < 1 or deadline_seconds <= 0:
        raise ValueError("Decoded development budgets must be positive")
    labels = {label.assignment: label for label in cache.labels}
    started = time.monotonic()
    excluded: list[tuple[int, ...]] = []
    status, selected = "budget_exhausted", None
    checks = 0
    for _ in range(max_checks):
        remaining = deadline_seconds - (time.monotonic() - started)
        if remaining <= 0:
            break
        result = bounded_call(solve_master, objective, tuple(excluded), timeout=remaining)
        if result.status != "complete":
            status = "solver_" + result.status
            break
        assignment = result.value.assignment
        if assignment is None:
            status = "no_feasible_assignment"
            break
        checks += 1
        label = labels.get(assignment)
        if label is None or label.feasible is None:
            status = "unknown_selected_assignment"
            break
        if label.feasible is False:
            excluded.append(assignment)
            continue
        selected = label
        status = "verified" if label.usable else "unknown_selected_queries"
        break
    complete_utilities = [
        cast(float, label.benefit) - label.cost for label in cache.labels if label.usable
    ]
    optimum = max(complete_utilities) if cache.complete and complete_utilities else None
    utility = (
        cast(float, selected.benefit) - selected.cost
        if selected is not None and selected.usable
        else None
    )
    return {
        "status": status,
        "assignment": selected.assignment if selected else None,
        "checks": checks,
        "infeasible_checks": len(excluded),
        "teacher_complete": cache.complete,
        "selected_utility": utility,
        "exact_teacher_optimum": optimum,
        "exact_regret": (
            max(0.0, optimum - utility) if optimum is not None and utility is not None else None
        ),
        "label_coverage": cache.coverage,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "frozen candidate universe; production MaxSAT with cached policy decisions",
    }


def generated_development(
    model,
    case,
    cache,
    *,
    seconds: float,
    temperature: float = 1.0,
    selection_options: Mapping[str, Any] | None = None,
    fidelity_evaluator_labels: Sequence[tuple[Any, Any]] = (),
    case_cpu_seconds: float | None = None,
    semantic_target: SemanticTargetSpec | None = None,
    target_basis: str = "symbolic",
    mixture_symbolic_weight: float = 0.5,
    **options,
):
    """Evaluate sampled grammar availability and expose uncached policy choices.

    Regret remains scoped to the common complete teacher inventory. Additional
    generated bundles receive no fabricated teacher score or feasibility label.
    """
    from dataclasses import replace

    if seconds <= 0:
        return {"status": "generation_deadline", "useful_candidate_coverage": None}
    semantic_target = semantic_target or SemanticTargetSpec(canonical_hash(case.probes))
    started = time.monotonic()
    frozen = _freeze_training_model(
        case.problem,
        model,
        seconds=seconds * 0.5,
        final_candidate_removals=dict(case.final_candidate_removals),
        cpu_seconds=case_cpu_seconds,
        **options,
    )
    if frozen.status != "complete":
        return {
            "status": "generation_" + frozen.status,
            "detail": frozen.detail,
            "useful_candidate_coverage": None,
        }
    generated = frozen.value
    inventories = [
        {candidate.candidate_id: index for index, candidate in enumerate(obj.candidates)}
        for obj in generated.problem.objects
    ]
    maps = [
        {
            index: generated_index[candidate.candidate_id]
            for index, candidate in enumerate(obj.candidates)
            if candidate.candidate_id in generated_index
        }
        for obj, generated_index in zip(case.problem.objects, inventories)
    ]
    remaining = seconds - (time.monotonic() - started)
    # Development executes the same baseline accounting, proof cuts and optional
    # risk shortlist as deployment. Semantic query labels remain external.
    from exact.repair.pipeline import repair_neural_round

    settings = dict(selection_options or {})
    generated_case = replace(case, problem=generated.problem)
    decoded = {"status": "generation_deadline", "checks": 0, "selected_utility": None}
    if remaining > 0:
        budget = replace(
            generated.problem.budgets,
            total_seconds=max(0.001, remaining * 0.75),
            max_checks=settings.get("max_candidate_checks", 100),
            max_solves=settings.get("max_master_solves", 100),
            retries=settings.get("retry_budget", 0),
        )
        active = replace(generated, problem=replace(generated.problem, budgets=budget))
        if not settings.get("risk_ordering", True):
            active = replace(active, risk_scorer=None)
        solved = bounded_call(
            repair_neural_round,
            active,
            shortlist_size=settings.get("shortlist_size", 1),
            utility_window=settings.get("utility_window", 0),
            shortlist_seconds=settings.get("construction_seconds"),
            diagnose=True,
            preserve_verified_input=False,
            timeout=max(0.001, remaining * 0.8),
            cpu_seconds=case_cpu_seconds,
        )
        if solved.status != "complete":
            decoded["status"] = "verification_" + solved.status
            decoded["detail"] = solved.detail
        else:
            result = solved.value
            decoded.update(
                status="unresolved",
                checks=result.checks,
                master_solves=result.solves,
                assignment=result.assignment,
                certification=result.search_status,
                lower_bound=result.lower_bound,
                upper_bound=result.upper_bound,
                first_verified_seconds=result.first_verified_seconds,
                resource_counters=result.resource_counters,
                stage_seconds=result.stage_seconds,
                failures=result.failures,
            )
            remaining = seconds - (time.monotonic() - started)
            if (
                result.assignment is not None
                and result.logical_status == "VERIFIED_FEASIBLE"
                and remaining > 0
            ):
                labeled = bounded_call(
                    _assignment_label,
                    generated_case,
                    result.assignment,
                    options.get("profile", DEFAULT_PROFILE),
                    semantic_target=semantic_target
                    or SemanticTargetSpec(canonical_hash(case.probes)),
                    timeout=remaining,
                )
                if labeled.status == "complete":
                    label = labeled.value
                    if target_basis != "symbolic":
                        from exact.repair.semantic_fidelity import offline_plan_label

                        weak = offline_plan_label(
                            generated_case,
                            result.assignment,
                            options.get("profile", DEFAULT_PROFILE),
                            fidelity_evaluator_labels,
                            role="evaluator",
                        )
                        score = weak.benefit if weak is not None else None
                        if target_basis == "mixed":
                            score = (
                                mixture_symbolic_weight * label.benefit
                                + (1 - mixture_symbolic_weight) * score
                                if label.benefit is not None and score is not None
                                else None
                            )
                        label = replace(label, benefit=score)
                    decoded.update(
                        status="verified" if label.usable else "unknown_selected_labels",
                        selected_utility=label.benefit - label.cost if label.usable else None,
                        semantic_benefit=label.benefit,
                        cost=label.cost,
                    )
                else:
                    decoded["status"] = "semantic_labels_" + labeled.status
    decoded.update(
        elapsed_seconds=time.monotonic() - started,
        inventory_hash=canonical_hash(tuple(obj.candidates for obj in generated.problem.objects)),
        policy_hash=generated.problem.policy.content_hash,
    )
    useful, optimum_present = None, None
    if cache.complete and any(label.usable for label in cache.labels):
        marginals = teacher_marginals(cache, temperature)
        useful = sum(
            sum(weights[index] for index in menu) for weights, menu in zip(marginals, maps)
        ) / len(maps)
        optimum = max(
            cast(float, label.benefit) - label.cost for label in cache.labels if label.usable
        )
        optimum_present = any(
            label.usable
            and cast(float, label.benefit) - label.cost == optimum
            and all(choice in menu for menu, choice in zip(maps, label.assignment))
            for label in cache.labels
        )
    return {
        "status": "generated",
        "parent_group_id": case.structural_parent,
        "semantic_target_hash": semantic_target.content_hash,
        "fidelity_aggregate_hash": canonical_hash(tuple(fidelity_evaluator_labels)),
        "useful_candidate_coverage": useful,
        "teacher_optimal_repair_available": optimum_present,
        "candidate_counts": [len(obj.candidates) for obj in generated.problem.objects],
        "mapped_teacher_candidates": [len(menu) for menu in maps],
        "decoded": decoded,
        "proposal_reports": json.loads(
            canonical_json(tuple(dict(row) for row in generated.proposal_reports))
        ),
        "elapsed_seconds": time.monotonic() - started,
        "scope": "generated-pool symbolic verification and independent declared semantic queries",
    }


def train_cases(
    training: Sequence[tuple[GeneratedCase, TeacherCache]],
    development: Sequence[tuple[GeneratedCase, TeacherCache]],
    *,
    encoder: str = "hgt",
    epochs: int = 2,
    seed: int = 13,
    hidden_dim: int = 32,
    pairwise: bool = False,
    learning_rate: float = 0.001,
    profile: tuple = DEFAULT_PROFILE,
    arm: str = "generated_only",
    warm_start_weights: Mapping[str, Any] | None = None,
    layers: int = 1,
    heads: int = 4,
    dropout: float = 0.1,
    weight_decay: float = 0.0001,
    batch_cases: int = 1,
    clip_gradient_norm: float = 1.0,
    smooth_l1_beta: float = 1.0,
    ranking_temperature: float = 1.0,
    max_repair_pairs_per_case: int = 64,
    loss_weights: tuple[float, float, float] = (1.0, 0.2, 1.0),
    proposal_temperature: float = 1.0,
    mixtures: int = 4,
    development_decode_every_epochs: int = 1,
    decode_seconds: float = 10.0,
    decode_max_checks: int = 256,
    patience: int | None = None,
    min_dev_improvement: float = 0.0,
    deadline_seconds: float = 300.0,
    device: str = "cpu",
    threads: int = 1,
    quantization_scale: int = 1000,
    warm_start_metadata: Any | None = None,
    proposal_arm: str = "grammar_mixture",
    compile_seconds: float = 20.0,
    max_circuit_nodes: int = 100000,
    max_depth: int = 2,
    max_constructors: int = 2,
    retrieval_config: Any | None = None,
    max_graph_nodes: int = 4096,
    max_graph_edges: int = 32768,
    max_explanations: int = 64,
    max_text_tokens: int = 128,
    pair_factor_limit_per_object: int = 16,
    pair_max_pairs: int | None = 128,
    pair_max_factors: int | None = 65536,
    development_draws_per_object: int = 32,
    candidate_cap: int = 64,
    checkpoint_path: Path | None = None,
    revision: str = "v3",
    interaction_loss_weight: float = 0.2,
    risk_loss_weight: float = 0.2,
    sampled_proposal_loss_weight: float = 1.0,
    minimum_generated_coverage: float = 1.0,
    quality_uncertainty_z: float = 1.96,
    missing_label_fallback: str = "stop",
    sampled_assignments: int = 32,
    active_round_every_epochs: int = 1,
    fidelity_labels: Mapping[str, Sequence[tuple[Any, Any]]] | None = None,
    fidelity_loss_weight: float = 1.0,
    max_collection_rounds: int = 3,
    collection_cases_per_round: int | None = None,
    collection_options: Mapping[str, Any] | None = None,
    plan_risk: bool = True,
    support_enabled: bool = False,
    support_readout_identity: str = SUPPORT_READOUT_IDENTITY,
    support_loss_weight: float = 0.2,
    support_target: str = "qualified_witness_violation/v1",
    selection_options: Mapping[str, Any] | None = None,
    circuit_limits: Mapping[str, Any] | None = None,
    compiler_cache_directory: str | None = None,
    vtree_type: str = "balanced",
    proposal_context: str = "independent",
    pair_factor_bound: float = 1.0,
    desired_family_weight: float = 1.0,
    false_positive_weight: float = 1.0,
    target_basis: str = "symbolic",
    fidelity_development_labels: Mapping[str, Sequence[tuple[Any, Any]]] | None = None,
    mixture_symbolic_weight: float = 0.5,
    case_cpu_seconds: float | None = None,
    total_training_seconds: float | None = None,
    resume_acquisition_deadline: bool = False,
    resume_endpoint_retrieval: Path | None = None,
    resume_report_transport: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Train masked full-plan tasks and select v3 checkpoints on generated repair quality.

    Cases without complete finite distributions still contribute masked benefit
    regression/ranking. No test case can enter optimization or checkpoint selection.
    """
    # An invocation may use a smaller slice of the predeclared total. The CLI
    # also enforces its durable wall/CPU ledger independently of this checkpoint.
    total_training_seconds = (
        deadline_seconds if total_training_seconds is None else total_training_seconds
    )
    if not math.isfinite(total_training_seconds) or total_training_seconds <= 0:
        raise ValueError("Cumulative training allowance must be finite and positive")
    if resume_report_transport and (
        checkpoint_path is None
        or not checkpoint_path.is_file()
        or resume_acquisition_deadline
        or resume_endpoint_retrieval is not None
    ):
        raise ValueError("Report recovery requires an existing checkpoint and no other migration")
    identity_options = dict(locals())
    for key in (
        "checkpoint_path",
        "deadline_seconds",
        "warm_start_weights",
        "resume_acquisition_deadline",
        "resume_endpoint_retrieval",
        "resume_report_transport",
    ):
        identity_options.pop(key)
    import torch

    from exact.repair.grammar import mapping_grammar, with_immutable_context
    from exact.repair.model import RepairModel, repair_benefits
    from exact.repair.pipeline import model_digest, proposal_distribution
    from exact.repair.retrieval import retrieve_vocabulary

    started = time.monotonic()
    if revision not in {"v2", "v3"} or sampled_assignments < 0 or active_round_every_epochs < 1:
        raise ValueError("Invalid training revision or acquisition schedule")
    if (
        target_basis not in {"symbolic", "ai_weak", "mixed"}
        or not 0 <= mixture_symbolic_weight <= 1
    ):
        raise ValueError("Declare a supported semantic target and frozen mixture weight")
    if target_basis == "symbolic" and (fidelity_labels or fidelity_development_labels):
        raise ValueError("Symbolic-only training cannot silently ingest weak labels")
    if target_basis != "symbolic" and (not fidelity_labels or not fidelity_development_labels):
        raise ValueError(
            "Weak target training requires teacher and independent development label records"
        )
    if proposal_context not in {"independent", "selected_other_actions"}:
        raise ValueError("Unknown proposal context convention")
    if (
        support_target != "qualified_witness_violation/v1"
        or support_loss_weight < 0
        or not math.isfinite(support_loss_weight)
    ):
        raise ValueError("Invalid qualified support auxiliary declaration")
    if target_basis != "symbolic":
        from exact.repair.semantic_fidelity import (
            ValidatedFidelityAggregateV3,
            validate_fidelity_training_records,
        )

        fidelity_labels = validate_fidelity_training_records(fidelity_labels or {}, "train")
        fidelity_development_labels = validate_fidelity_training_records(
            fidelity_development_labels or {}, "development"
        )
        if any(
            not isinstance(comparison, ValidatedFidelityAggregateV3)
            for source in (fidelity_labels, fidelity_development_labels)
            for rows in (source or {}).values()
            for _, comparison in rows
        ):
            raise ValueError("Weak supervision requires unique-observation validated aggregates")
    if max_collection_rounds < 0 or (
        collection_cases_per_round is not None and collection_cases_per_round < 1
    ):
        raise ValueError("Invalid collection round/case budget")
    if any(
        not math.isfinite(w) or w < 0
        for w in (
            interaction_loss_weight,
            risk_loss_weight,
            sampled_proposal_loss_weight,
            fidelity_loss_weight,
        )
    ):
        raise ValueError("Invalid v3 loss weights")
    if proposal_arm not in {"grammar_mixture", "bounded_enumeration"}:
        raise ValueError(
            "Training proposal arm must declare grammar_mixture or bounded_enumeration"
        )
    if any(case.control == "ambiguous" for case, _ in [*training, *development]):
        raise ValueError("Ambiguous observations must remain an unresolved evaluation stratum")
    if (
        batch_cases < 1
        or development_decode_every_epochs < 1
        or threads < 1
        or deadline_seconds <= 0
        or clip_gradient_norm <= 0
        or weight_decay < 0
        or learning_rate <= 0
        or len(loss_weights) != 3
        or any(not math.isfinite(value) or value < 0 for value in loss_weights)
    ):
        raise ValueError("Invalid training settings")
    patience = epochs if patience is None else patience
    if patience < 1 or min_dev_improvement < 0:
        raise ValueError("Invalid checkpoint patience/improvement")
    torch.set_num_threads(threads)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("Supported training devices are auto, cpu and cuda")
    if not training or not development or epochs < 1:
        raise ValueError("Training requires train/development cases and a positive epoch count")
    if any(case.split != "train" for case, _ in training) or any(
        case.split != "development" for case, _ in development
    ):
        raise ValueError("Training/checkpoint selection must respect saved clean-parent splits")
    if {case.structural_parent for case, _ in training} & {
        case.structural_parent for case, _ in development
    }:
        raise ValueError("A structural parent cannot cross training and development")
    if arm not in {"generated_only", "training_side_adaptation"}:
        raise ValueError("Unknown training arm")
    expected_origin = "generated" if arm == "generated_only" else "training_side_real"
    if any(case.origin != expected_origin for case, _ in [*training, *development]):
        raise ValueError("Training provenance must match the separately declared arm")
    if arm == "training_side_adaptation" and warm_start_weights is None:
        raise ValueError("Adaptation requires pretrained model weights only")
    semantic_targets = {
        case.case_id: SemanticTargetSpec(
            canonical_hash(case.probes), desired_family_weight, false_positive_weight
        )
        for case, _ in [*training, *development]
    }
    for case, cache in [*training, *development]:
        expected = {
            "input": case.problem.content_hash,
            "policy": case.problem.policy.content_hash,
            "query": canonical_hash(case.probes),
            "inventory": canonical_hash(tuple(obj.candidates for obj in case.problem.objects)),
            "profile": canonical_hash(profile),
        }
        if revision == "v3":
            expected["semantic_target"] = semantic_targets[case.case_id].content_hash
        if any(dict(cache.hashes).get(key) != digest for key, digest in expected.items()):
            raise ValueError("Teacher cache dependencies/profile do not match this training case")
        if cache.candidate_counts != tuple(len(obj.candidates) for obj in case.problem.objects):
            raise ValueError("Teacher cache candidate counts do not match")
    from exact.repair.retrieval import RetrievalConfig

    preparation = EffectivePreparation(
        max_graph_nodes,
        max_graph_edges,
        max_explanations,
        max_text_tokens,
        pair_factor_limit_per_object,
        pair_max_pairs,
        pair_max_factors,
        retrieval_config or RetrievalConfig(),
        revision,
    )
    retrieval_config = preparation.retrieval_config
    torch.manual_seed(seed)
    all_cases = [*training, *development]
    retrievals = {
        case.case_id: retrieve_vocabulary(case.problem, config=retrieval_config)
        for case, _ in all_cases
    }
    graphs = {
        case.case_id: preparation.graph(case.problem, retrievals[case.case_id])
        for case, _ in all_cases
    }
    grammars = {}
    if proposal_arm == "grammar_mixture":
        for case, _ in all_cases:
            for obj in case.problem.objects:
                menu = retrievals[case.case_id].for_object(obj.object_id)
                grammars[case.case_id, obj.object_id] = mapping_grammar(
                    obj,
                    menu.classes,
                    menu.properties,
                    max_depth=max_depth,
                    max_constructors=max_constructors,
                    fixed_axioms=case.problem.fixed_axioms,
                    source_classes=menu.source_classes,
                    target_classes=menu.target_classes,
                    source_properties=menu.source_properties,
                    target_properties=menu.target_properties,
                    constraint_identity=canonical_hash((case.problem.policy, menu)),
                )
                if revision == "v3":
                    grammars[case.case_id, obj.object_id] = with_immutable_context(
                        grammars[case.case_id, obj.object_id],
                        case.problem.fixed_axioms,
                        case.problem.policy,
                    )
    pair_reports = {
        case.case_id: preparation.pairs(case.problem, graphs[case.case_id], enabled=pairwise)
        for case, _ in all_cases
    }
    interaction_pairs = {key: report.pairs for key, report in pair_reports.items()}
    proposal_coverage: dict[str, dict[str, Any]] = {}
    loss_eligibility: dict[str, dict[str, Any]] = {}
    acquisition_reports: list[dict] = []
    sampled_training: dict[str, tuple[GeneratedCase, TeacherCache]] = {}
    acquisition_epoch = -1
    acquisition_completed: set[tuple[int, str]] = set()
    failed_compilations: set[str] = set()
    node_types = {kind for graph in graphs.values() for kind in graph.metadata[0]}
    edge_types = {edge for graph in graphs.values() for edge in graph.metadata[1]}
    if warm_start_metadata is not None:
        node_types.update(warm_start_metadata[0])
        edge_types.update(tuple(edge) for edge in warm_start_metadata[1])
    model = RepairModel(
        (node_types, edge_types),
        hidden_dim=hidden_dim,
        heads=heads,
        layers=layers,
        dropout=dropout,
        encoder=encoder,
        pairwise=pairwise,
        revision=revision,
        plan_risk=plan_risk,
        support_enabled=support_enabled,
        support_readout_identity=support_readout_identity,
        pair_factor_bound=pair_factor_bound,
    ).to(device)
    adaptation_new_parameters = []
    if warm_start_weights is not None:
        restored_weights = dict(warm_start_weights)
        if encoder == "rgcn" and warm_start_metadata is not None:
            old_edges = [tuple(edge) for edge in warm_start_metadata[1]]
            new_edges = {edge: index for index, edge in enumerate(model.metadata[1])}
            current_weights = model.state_dict()
            for key, value in tuple(restored_weights.items()):
                if key.startswith("graph_layers.") and key.endswith(".weight") and value.ndim == 3:
                    expanded = current_weights[key].clone()
                    if value.shape[0] != len(old_edges) or value.shape[1:] != expanded.shape[1:]:
                        raise ValueError("Incompatible R-GCN warm-start relation weights")
                    for old_index, edge in enumerate(old_edges):
                        expanded[new_edges[edge]] = value[old_index]
                    restored_weights[key] = expanded
        loaded = model.load_state_dict(restored_weights, strict=False)
        if loaded.unexpected_keys:
            raise ValueError("Warm-start weights have incompatible model keys")
        adaptation_new_parameters = list(loaded.missing_keys)
    warm_start_hash = model_digest(model) if warm_start_weights is not None else None
    implementation = canonical_hash(
        [
            (path.name, path.read_bytes().hex())
            for path in sorted(
                (Path(__file__).resolve().parents[2] / "exact" / "repair").glob("*.py")
            )
        ]
        + [("train.py", Path(__file__).read_bytes().hex())]
    )
    resume_identity = canonical_hash((identity_options, warm_start_hash, implementation))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    best_criterion, best_state, best_epoch = None, None, None
    stale_evaluations = 0
    next_epoch, next_offset, saved_loss, saved_optimized = 0, 0, 0.0, 0
    stopped_early = False
    current_phase = "acquisition"
    epoch_order: tuple[str, ...] = ()
    development_progress: dict[str, Any] = {}
    interrupted = False
    pending_acquisition: dict[str, Any] = {}
    previous_elapsed = 0.0
    execution_count = 1
    recovery_lineage: list[dict[str, Any]] = []
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, weights_only=True, map_location=device)
        if (
            saved.get("schema") != f"exact-repair/training-state/{revision}"
            or saved.get("identity") != resume_identity
            or (revision == "v3" and saved.get("recovery_revision") != "exact-phase-resume/v3.1")
        ):
            if resume_report_transport and revision == "v3":
                from tools.repair.report_recovery import completed_report_recovery

                dependencies = canonical_hash(
                    [
                        (path.name, path.read_bytes().hex())
                        for path in sorted(
                            (Path(__file__).resolve().parents[2] / "exact" / "repair").glob("*.py")
                        )
                    ]
                )
                recovery_lineage.append(
                    completed_report_recovery(
                        saved, identity_options, warm_start_hash, dependencies
                    )
                )
            elif resume_endpoint_retrieval is not None and revision == "v3":
                from tools.repair.endpoint_recovery import recover_endpoint_state

                dependencies = canonical_hash(
                    [
                        (path.name, path.read_bytes().hex())
                        for path in sorted(
                            (Path(__file__).resolve().parents[2] / "exact" / "repair").glob("*.py")
                        )
                        if path.name != "candidates.py"
                    ]
                )
                saved = recover_endpoint_state(
                    saved,
                    torch.load(resume_endpoint_retrieval, weights_only=True, map_location=device),
                    identity_options,
                    warm_start_hash,
                    dependencies,
                )
            else:
                if not resume_acquisition_deadline or revision != "v3":
                    raise ValueError(
                        "Training checkpoint is incompatible with settings, inputs or splits"
                    )
                dependencies = canonical_hash(
                    [
                        (path.name, path.read_bytes().hex())
                        for path in sorted(
                            (Path(__file__).resolve().parents[2] / "exact" / "repair").glob("*.py")
                        )
                        if path.name != "learning.py"
                    ]
                )
                migration = _acquisition_deadline_recovery(
                    saved, identity_options, warm_start_hash, dependencies
                )
                saved["pending_acquisition"]["case_deadline_exhausted"] = True
                recovery_lineage.append(migration)
        recovery_lineage = [*saved.get("recovery_lineage", []), *recovery_lineage]
        previous_elapsed = float(saved.get("elapsed_seconds", 0.0))
        execution_count = int(saved.get("execution_count", 0)) + 1
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        random.setstate(saved["python_rng"])
        next_epoch, next_offset = saved["next_epoch"], saved["next_offset"]
        current_phase = saved.get("phase", "acquisition")
        epoch_order = tuple(saved.get("epoch_order", ()))
        development_progress = saved.get("development_progress", {})
        pending_acquisition = saved.get("pending_acquisition", {})
        saved_loss, saved_optimized = saved["train_loss"], saved["optimized"]
        history = saved["history"]
        best_criterion, best_state, best_epoch = (
            saved["best_criterion"],
            saved["best_state"],
            saved["best_epoch"],
        )
        stale_evaluations, stopped_early = saved["stale_evaluations"], saved["stopped_early"]
        proposal_coverage = saved["proposal_coverage"]
        loss_eligibility = saved.get("loss_eligibility", {})
        failed_compilations = set(saved["failed_compilations"])
        acquisition_epoch = saved.get("acquisition_epoch", -1)
        acquisition_completed = {tuple(row) for row in saved.get("acquisition_completed", [])}
        acquisition_reports = saved.get("acquisition_reports", [])
        if revision == "v3":
            from tools.repair.prepare import cache_from_dict, case_from_dict

            sampled_training = {
                key: (case_from_dict(value[0]), cache_from_dict(value[1]))
                for key, value in saved.get("sampled_training", {}).items()
            }
            for _, (case, cache) in sampled_training.items():
                key = case.case_id
                retrieval = retrieve_vocabulary(case.problem, config=retrieval_config)
                retrievals[key] = retrieval
                graphs[key] = preparation.graph(case.problem, retrieval)
                pair_reports[key] = preparation.pairs(case.problem, graphs[key], enabled=pairwise)
                interaction_pairs[key] = pair_reports[key].pairs
                for obj in case.problem.objects:
                    menu = retrieval.for_object(obj.object_id)
                    grammars[key, obj.object_id] = with_immutable_context(
                        mapping_grammar(
                            obj,
                            menu.classes,
                            menu.properties,
                            max_depth=max_depth,
                            max_constructors=max_constructors,
                            fixed_axioms=case.problem.fixed_axioms,
                            source_classes=menu.source_classes,
                            target_classes=menu.target_classes,
                            source_properties=menu.source_properties,
                            target_properties=menu.target_properties,
                            constraint_identity=canonical_hash((case.problem.policy, menu)),
                        ),
                        case.problem.fixed_axioms,
                        case.problem.policy,
                    )

    def remaining_training_seconds() -> float:
        elapsed = time.monotonic() - started
        return max(
            0.0,
            min(deadline_seconds - elapsed, total_training_seconds - previous_elapsed - elapsed),
        )

    def checkpoint(epoch, offset=0, loss=0.0, optimized=0):
        if checkpoint_path is not None:
            from dataclasses import asdict

            from tools.repair.prepare import case_to_dict

            save_training_state(
                checkpoint_path,
                dict(
                    schema=f"exact-repair/training-state/{revision}",
                    identity=resume_identity,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else [],
                    python_rng=random.getstate(),
                    next_epoch=epoch,
                    next_offset=offset,
                    recovery_revision="exact-phase-resume/v3.1",
                    elapsed_seconds=previous_elapsed + time.monotonic() - started,
                    total_training_seconds=total_training_seconds,
                    execution_count=execution_count,
                    recovery_lineage=recovery_lineage,
                    phase=current_phase,
                    epoch_order=epoch_order,
                    development_progress=development_progress,
                    pending_acquisition=pending_acquisition,
                    train_loss=loss,
                    optimized=optimized,
                    history=history,
                    best_criterion=best_criterion,
                    best_state=best_state,
                    best_epoch=best_epoch,
                    stale_evaluations=stale_evaluations,
                    stopped_early=stopped_early,
                    proposal_coverage=proposal_coverage,
                    loss_eligibility=loss_eligibility,
                    failed_compilations=sorted(failed_compilations),
                    acquisition_epoch=acquisition_epoch,
                    acquisition_completed=sorted(acquisition_completed),
                    acquisition_reports=acquisition_reports,
                    sampled_training={
                        key: (case_to_dict(case), asdict(cache))
                        for key, (case, cache) in sampled_training.items()
                    },
                ),
            )

    if revision == "v2" and not any(
        cache.complete and any(label.usable for label in cache.labels) for _, cache in development
    ):
        raise ValueError(
            "Decoded checkpoint selection requires a complete usable development cache"
        )

    def case_loss(case: GeneratedCase, cache: TeacherCache) -> Any:
        if target_basis != "symbolic":
            from dataclasses import replace

            from exact.repair.semantic_fidelity import offline_plan_label

            labels = []
            source = fidelity_labels if case.split == "train" else fidelity_development_labels
            role = "teacher" if case.split == "train" else "evaluator"
            base_id = case.case_id.split(":active-", 1)[0]
            for label in cache.labels:
                weak = offline_plan_label(
                    replace(case, case_id=base_id),
                    label.assignment,
                    profile,
                    (source or {}).get(base_id, ()),
                    role=role,
                )
                benefit = weak.benefit if weak is not None else None
                if target_basis == "mixed":
                    benefit = (
                        mixture_symbolic_weight * label.benefit
                        + (1 - mixture_symbolic_weight) * benefit
                        if label.benefit is not None and benefit is not None
                        else None
                    )
                if label.feasible is not True:
                    benefit = None
                labels.append(replace(label, benefit=benefit))
            cache = replace(
                cache,
                labels=tuple(labels),
                complete=False,
                stop_reason="offline_weak_sample_conditioned",
            )
        if not cache.labels:
            return model.empty_bundle.sum() * 0.0
        graph = graphs[case.case_id]
        memory = model.encode(graph)
        unary, pairs = model.score_inventory(
            case.problem.objects, memory, interaction_pairs=interaction_pairs[case.case_id]
        )
        predictions = repair_benefits((row.assignment for row in cache.labels), unary, pairs)
        losses = benefit_losses(
            predictions,
            cache.labels,
            seed=seed,
            beta=smooth_l1_beta,
            rank_temperature=ranking_temperature,
            max_pairs=max_repair_pairs_per_case,
        )
        loss = loss_weights[0] * losses["value"] + loss_weights[1] * losses["rank"]
        eligibility = {"value": losses["usable"], "rank": losses["pairs"], "quartet": 0, "risk": 0}
        if revision == "v3":
            quartets = interaction_loss(
                predictions, cache.labels, max_quartets=max_repair_pairs_per_case
            )
            risk: dict[str, Any] = {"loss": predictions.sum() * 0, "eligible": 0, "unknown": 0}
            if plan_risk:
                risks = torch.stack(
                    [
                        model.plan_risk_logit(
                            case.problem.objects,
                            memory,
                            label.assignment,
                            supports=graph.admitted_supports,
                        )
                        for label in cache.labels
                    ]
                )
                risk = risk_loss(risks, cache.labels)
            loss = (
                loss + interaction_loss_weight * quartets["loss"] + risk_loss_weight * risk["loss"]
            )
            eligibility.update(
                quartet=quartets["eligible"],
                risk=risk["eligible"],
                risk_unknown=risk["unknown"],
                risk_loss=float(risk["loss"].detach()),
            )
        auxiliary = [target for label in cache.labels for target in label.support_targets]
        eligible_auxiliary = [target for target in auxiliary if target.eligible]
        eligibility.update(
            support_available=len(auxiliary),
            support_eligible=len(eligible_auxiliary),
            support_optimized=0,
            support_loss=0.0,
            support_unavailable_inputs=0,
        )
        if support_enabled and eligible_auxiliary:
            from exact.repair.graph import structural_id
            from exact.repair.kernel import materialize

            selected_targets, support_logits = [], []
            qualified_assignments = {}
            for label in cache.labels:
                axioms, active = materialize(case.problem, label.assignment)
                qualified_assignments[label.assignment] = (
                    canonical_hash(
                        (
                            tuple(sorted(set(axioms), key=canonical_hash)),
                            tuple(sorted(set(active), key=canonical_hash)),
                        )
                    ),
                    tuple(
                        (obj.object_id, obj.candidates[choice].candidate_id)
                        for obj, choice in zip(case.problem.objects, label.assignment)
                    ),
                    tuple(value.canonical_bytes().hex() for value in active),
                )
            for target in eligible_auxiliary:
                if (
                    target.policy_hash != case.problem.policy.content_hash
                    or qualified_assignments.get(target.assignment)
                    != (target.theory_hash, target.occurrence_ids, target.activation)
                ):
                    raise ValueError("Support label dependencies changed")
                witness = owl.decode_canonical(bytes.fromhex(target.witness))
                # Missing query vocabulary is an explicit unavailable auxiliary,
                # never an unbudgeted insertion of post-decision proof contents.
                if any(
                    structural_id(symbol) not in memory.rows for symbol in owl.signature(witness)
                ):
                    continue
                selected_targets.append(target)
                support_logits.append(
                    model.support_violation_logit(
                        case.problem.objects,
                        memory,
                        target.assignment,
                        witness,
                        obligation_kind=target.obligation_kind,
                        expected_truth=target.expected_truth,
                    )
                )
            if support_logits:
                terms = support_loss(torch.stack(support_logits), selected_targets)
                loss = loss + support_loss_weight * terms["loss"]
                eligibility["support_optimized"] = terms["eligible"]
                eligibility["support_loss"] = float(terms["loss"].detach())
            eligibility["support_unavailable_inputs"] = len(eligible_auxiliary) - len(
                selected_targets
            )
        fidelity_terms = []
        for packet, comparison in (
            (fidelity_labels or {}).get(case.case_id, ()) if target_basis == "ai_weak" else ()
        ):
            if (
                packet.case_id != case.case_id
                or packet.parent_group_id != case.structural_parent
                or packet.split != case.split
                or comparison.split != case.split
                or comparison.packet_hash != packet.content_hash
                or comparison.policy_hash != case.problem.policy.content_hash
            ):
                raise ValueError("Fidelity label dependency or grouped split mismatch")
            by_id = {packet.plan_a.plan_id: packet.plan_a, packet.plan_b.plan_id: packet.plan_b}
            assignments = []
            for plan_id in (comparison.plan_a_id, comparison.plan_b_id):
                plan = by_id[plan_id]
                mapping = dict(plan.complete_assignment)
                if set(mapping) != {obj.object_id for obj in case.problem.objects}:
                    raise ValueError("Fidelity plan must contain every object")
                choices = []
                for obj in case.problem.objects:
                    menu = {c.candidate_id: i for i, c in enumerate(obj.candidates)}
                    if mapping[obj.object_id] not in menu:
                        raise ValueError("Fidelity assignment is unavailable in this inventory")
                    choices.append(menu[mapping[obj.object_id]])
                assignments.append(tuple(choices))
            values = repair_benefits(assignments, unary, pairs)
            fidelity = fidelity_comparison_losses(
                values[0],
                values[1],
                comparison,
                temperature=ranking_temperature,
                beta=smooth_l1_beta,
            )
            if fidelity["eligible"]:
                fidelity_terms.append(fidelity["value"] + fidelity["rank"] + fidelity["tie"])
        eligibility["llm_weak_comparisons"] = len(fidelity_terms)
        if fidelity_terms:
            loss = loss + fidelity_loss_weight * torch.stack(fidelity_terms).mean()
        loss_eligibility[case.case_id] = eligibility
        if proposal_context == "selected_other_actions" and any(
            label.usable for label in cache.labels
        ):
            distributions = {}

            def conditional_log_probability(index, prefix, choice):
                key = (index, prefix)
                obj = case.problem.objects[index]
                if key not in distributions:
                    remaining = remaining_training_seconds()
                    if remaining <= 0:
                        raise TimeoutError("conditional proposal training deadline")
                    partial = (
                        *prefix,
                        *(None for _ in range(len(case.problem.objects) - len(prefix))),
                    )
                    context = model.plan_context(
                        case.problem.objects, memory, partial, target_object_id=obj.object_id
                    )
                    distribution = proposal_distribution(
                        model,
                        memory,
                        obj,
                        profile=profile,
                        mixtures=mixtures,
                        encoding=grammars.get((case.case_id, obj.object_id)),
                        compile_seconds=min(compile_seconds, remaining),
                        max_circuit_nodes=max_circuit_nodes,
                        max_depth=max_depth,
                        max_constructors=max_constructors,
                        selected_context=context,
                        circuit_limits=dict(circuit_limits) if circuit_limits is not None else None,
                        compiler_cache_directory=compiler_cache_directory,
                        vtree_type=vtree_type,
                    )
                    if not getattr(distribution.circuit, "complete", True):
                        raise RuntimeError("partial circuit conditional loss unavailable")
                    distributions[key] = distribution
                return distributions[key].candidate_log_probability(obj.candidates[choice])

            try:
                conditional = conditional_proposal_loss(
                    cache, conditional_log_probability, temperature=proposal_temperature
                )
                loss = (
                    loss
                    + (loss_weights[2] if cache.complete else sampled_proposal_loss_weight)
                    * conditional["loss"]
                )
                proposal_coverage[case.case_id + ":conditional"] = {
                    k: v for k, v in conditional.items() if k != "loss"
                }
            except (RuntimeError, TimeoutError) as error:
                proposal_coverage[case.case_id + ":conditional"] = {
                    "status": "conditional_loss_unavailable",
                    "detail": str(error),
                }
        elif (cache.complete or revision == "v3") and any(label.usable for label in cache.labels):
            marginals = (
                teacher_marginals(cache, proposal_temperature)
                if cache.complete
                else sample_conditioned_marginals(cache, proposal_temperature)
            )
            for obj, marginal_target in zip(case.problem.objects, marginals):
                key = case.case_id + ":" + obj.object_id
                if key in failed_compilations:
                    continue
                remaining = remaining_training_seconds()
                if remaining <= 0:
                    proposal_coverage[key] = {"status": "training_deadline"}
                    continue
                try:
                    distribution = proposal_distribution(
                        model,
                        memory,
                        obj,
                        profile=profile,
                        mixtures=mixtures,
                        encoding=grammars.get((case.case_id, obj.object_id)),
                        compile_seconds=min(compile_seconds, remaining),
                        max_circuit_nodes=max_circuit_nodes,
                        max_depth=max_depth,
                        max_constructors=max_constructors,
                        circuit_limits=dict(circuit_limits) if circuit_limits is not None else None,
                        compiler_cache_directory=compiler_cache_directory,
                        vtree_type=vtree_type,
                    )
                except (TimeoutError, RuntimeError) as error:
                    failed_compilations.add(key)
                    proposal_coverage[key] = {
                        "status": "compilation_unavailable",
                        "detail": str(error),
                    }
                    continue
                if not getattr(distribution.circuit, "complete", True):
                    proposal_coverage[key] = {
                        "status": "partial_circuit_loss_excluded",
                        "missing_target_mass": None,
                        "target_kind": "exact" if cache.complete else "sample_conditioned",
                    }
                    continue
                probabilities = torch.stack(
                    [
                        distribution.candidate_log_probability(candidate)
                        for candidate in obj.candidates
                    ]
                )
                coverage = covered_proposal_loss(
                    probabilities,
                    marginal_target,
                    target_kind="exact" if cache.complete else "sample_conditioned",
                    elementary=tuple(
                        all(
                            isinstance(node, (owl.Entity, owl.Axiom))
                            for axiom in candidate.axioms
                            for node in owl.walk(axiom)
                        )
                        for candidate in obj.candidates
                    ),
                )
                proposal_coverage[key] = {k: v for k, v in coverage.items() if k != "loss"}
                weight = loss_weights[2] if cache.complete else sampled_proposal_loss_weight
                loss = loss + weight * coverage["loss"] / max(1, len(case.problem.objects))
        return loss

    checkpoint(next_epoch, next_offset, saved_loss, saved_optimized)
    for epoch in range(next_epoch, epochs):
        if stopped_early:
            break
        if remaining_training_seconds() <= 0:
            interrupted = True
            checkpoint(
                epoch,
                next_offset if epoch == next_epoch else 0,
                saved_loss if epoch == next_epoch else 0.0,
                saved_optimized if epoch == next_epoch else 0,
            )
            break
        if (
            revision == "v3"
            and current_phase == "acquisition"
            and sampled_assignments
            and epoch % active_round_every_epochs == 0
            and acquisition_epoch != epoch
            and epoch // active_round_every_epochs < max_collection_rounds
        ):
            from dataclasses import replace

            model.eval()
            round_hash = model_digest(model)
            scheduled_cases = list(training)
            random.Random(seed + epoch).shuffle(scheduled_cases)
            if collection_cases_per_round is not None:
                scheduled_cases = scheduled_cases[:collection_cases_per_round]
            for case, _ in scheduled_cases:
                if (epoch, case.case_id) in acquisition_completed:
                    continue
                remaining = remaining_training_seconds()
                if remaining <= 0:
                    break
                continuing = (
                    pending_acquisition.get("case_id") == case.case_id
                    and pending_acquisition.get("epoch") == epoch
                )
                if continuing:
                    from types import SimpleNamespace

                    from exact.repair.records import read_record
                    from exact.repair.workers import CallResult
                    from tools.repair.prepare import case_from_dict

                    if pending_acquisition["model_hash"] != round_hash:
                        raise ValueError("Partial acquisition model changed")
                    restored_case = case_from_dict(pending_acquisition["case"])
                    frozen = CallResult(
                        "complete",
                        SimpleNamespace(
                            problem=restored_case.problem,
                            objective=read_record(pending_acquisition["objective"]),
                            proposal_reports=pending_acquisition["reports"],
                        ),
                    )
                else:
                    with torch.no_grad():
                        frozen = _freeze_training_model(
                            case.problem,
                            model,
                            seconds=min(decode_seconds, remaining),
                            cpu_seconds=case_cpu_seconds,
                            max_graph_nodes=max_graph_nodes,
                            max_graph_edges=max_graph_edges,
                            max_explanations=max_explanations,
                            max_text_tokens=max_text_tokens,
                            draws_per_object=development_draws_per_object,
                            candidate_cap=candidate_cap,
                            mixtures=mixtures,
                            seed=seed + epoch,
                            profile=profile,
                            max_depth=max_depth,
                            max_constructors=max_constructors,
                            compile_seconds=compile_seconds,
                            max_circuit_nodes=max_circuit_nodes,
                            retrieval_config=retrieval_config,
                            proposal_arm=proposal_arm,
                            pair_factor_limit_per_object=pair_factor_limit_per_object,
                            pair_max_pairs=pair_max_pairs,
                            pair_max_factors=pair_max_factors,
                            final_candidate_removals=dict(case.final_candidate_removals),
                            selected_other_assignment=(
                                tuple(0 for _ in case.problem.objects)
                                if proposal_context == "selected_other_actions"
                                else None
                            ),
                            circuit_limits=(
                                dict(circuit_limits) if circuit_limits is not None else None
                            ),
                            compiler_cache_directory=compiler_cache_directory,
                            vtree_type=vtree_type,
                        )
                if frozen.status != "complete":
                    acquisition_reports.append(
                        {
                            "round": epoch,
                            "case_id": case.case_id,
                            "status": frozen.status,
                            "requested": sampled_assignments,
                        }
                    )
                    acquisition_completed.add((epoch, case.case_id))
                    checkpoint(epoch)
                    continue
                generated = replace(
                    case, case_id=case.case_id + f":active-{epoch}", problem=frozen.value.problem
                )
                if not continuing:
                    from tools.repair.prepare import case_to_dict

                    pending_acquisition = dict(
                        case_id=case.case_id,
                        epoch=epoch,
                        model_hash=round_hash,
                        case=case_to_dict(generated),
                        objective=frozen.value.objective.to_dict(),
                        reports=json.loads(
                            canonical_json(
                                tuple(dict(row) for row in frozen.value.proposal_reports)
                            )
                        ),
                    )
                    checkpoint(epoch)
                acquire_started = time.monotonic()
                previous_acquisition_seconds = pending_acquisition.get("elapsed_seconds", 0.0)
                case_remaining = max(0.0, decode_seconds - previous_acquisition_seconds)
                if pending_acquisition.get("case_deadline_exhausted"):
                    case_remaining = 0.0
                stage_limited = remaining_training_seconds() < case_remaining
                acquire_budget = min(case_remaining, remaining_training_seconds())
                if acquire_budget <= 0 and stage_limited:
                    break

                def acquire_label(assignment):
                    left = acquire_budget - (time.monotonic() - acquire_started)
                    if left <= 0:
                        return RepairLabel(assignment, None, None, 0.0)
                    result = bounded_call(
                        _assignment_label,
                        generated,
                        assignment,
                        profile,
                        semantic_target=semantic_targets[case.case_id],
                        timeout=min(left, decode_seconds),
                    )
                    return (
                        result.value
                        if result.status == "complete"
                        else RepairLabel(assignment, None, None, 0.0)
                    )

                if "proposed" in pending_acquisition:
                    proposed = pending_acquisition["proposed"]
                else:
                    from exact.repair.maxsat import solve_master
                    from exact.repair.workers import CallResult

                    utility_cap = (
                        (collection_options or {})
                        .get("plan_quotas", {})
                        .get("utility", (collection_options or {}).get("utility_attempts", 1))
                    )
                    proposal = (
                        bounded_call(
                            solve_master,
                            frozen.value.objective,
                            (),
                            timeout=max(0.001, acquire_budget / 4),
                        )
                        if utility_cap
                        else CallResult("unavailable", detail="No utility quota")
                    )
                    proposed = []
                    if proposal.status == "complete" and proposal.value.assignment is not None:
                        proposed.append((proposal.value.assignment, "maxsat_initial"))
                    for attempt in range(1, utility_cap):
                        left = acquire_budget - (time.monotonic() - acquire_started)
                        if left <= 0:
                            break
                        alternate = bounded_call(
                            solve_master,
                            frozen.value.objective,
                            tuple(row[0] for row in proposed),
                            timeout=max(0.001, left / 2),
                        )
                        if alternate.status != "complete" or alternate.value.assignment is None:
                            break
                        proposed.append((alternate.value.assignment, "maxsat_diverse"))
                    # Product of the frozen per-object proposal draw streams. Missing,
                    # filtered or exhausted draws leave unavailable schedule slots.
                    proposal_quota = (
                        (collection_options or {}).get("plan_quotas", {}).get("proposal", 0)
                    )
                    proposal_reports = frozen.value.proposal_reports
                    for draw in range(proposal_quota):
                        choices, probability = [], 1.0
                        for obj, report in zip(generated.problem.objects, proposal_reports):
                            samples = report.get("samples", ())
                            if draw >= len(samples):
                                break
                            sample = samples[draw]
                            ids = [candidate.candidate_id for candidate in obj.candidates]
                            if sample["candidate_id"] not in ids:
                                break
                            choices.append(ids.index(sample["candidate_id"]))
                            probability *= math.exp(sample["log_probability"])
                        if len(choices) == len(generated.problem.objects):
                            proposed.append((tuple(choices), "proposal", probability))
                    pending_acquisition["proposed"] = proposed
                    checkpoint(epoch)

                def save_collection(state):
                    pending_acquisition["collection_state"] = copy.deepcopy(state)
                    pending_acquisition["elapsed_seconds"] = (
                        previous_acquisition_seconds + time.monotonic() - acquire_started
                    )
                    checkpoint(epoch)

                acquired = collect_sampled_repairs(
                    tuple(len(obj.candidates) for obj in generated.problem.objects),
                    acquire_label,
                    case_id=case.case_id,
                    parent_group_id=case.structural_parent,
                    split=case.split,
                    hashes={
                        "inventory": canonical_hash(
                            tuple(obj.candidates for obj in generated.problem.objects)
                        ),
                        "input": generated.problem.content_hash,
                        "patch": canonical_hash(generated.problem.objects),
                        "backend": canonical_hash(
                            ("pyhermit", distribution_version("pyhermit"), "python")
                        ),
                        "policy": generated.problem.policy.content_hash,
                        "query": canonical_hash(case.probes),
                        "profile": canonical_hash(profile),
                        "semantic_target": semantic_targets[case.case_id].content_hash,
                    },
                    model_hash=round_hash,
                    round_id=str(epoch),
                    max_assignments=(collection_options or {}).get(
                        "plan_attempts_per_case", sampled_assignments
                    ),
                    plan_quotas=(collection_options or {}).get("plan_quotas"),
                    resume_state=pending_acquisition.get("collection_state"),
                    progress=save_collection,
                    deadline_seconds=max(
                        0.0, acquire_budget - (time.monotonic() - acquire_started)
                    ),
                    seed=seed + epoch,
                    proposed=proposed,
                    object_candidate_ids=tuple(
                        (obj.object_id, tuple(c.candidate_id for c in obj.candidates))
                        for obj in generated.problem.objects
                    ),
                    exploration_fraction=(collection_options or {}).get("exploration_fraction"),
                    counterfactual_attempts=(collection_options or {}).get("diversity_attempts"),
                    quartet_attempts=(collection_options or {}).get("quartet_attempts", 0),
                )
                if acquired.stop_reason == "deadline" and stage_limited:
                    # Resume only an interrupted whole-stage slice. A completed
                    # bounded case keeps its partial labels and unvisited slots.
                    checkpoint(epoch)
                    break
                pending_acquisition = {}
                acquisition_reports.append(json.loads(canonical_json(acquired)))
                if not acquired.cache.labels:
                    # Unavailable slots remain in the acquisition denominator;
                    # no assignment exists to add to the optimizer's inputs.
                    acquisition_completed.add((epoch, case.case_id))
                    checkpoint(epoch)
                    continue
                sampled_training[case.case_id] = (generated, acquired.cache)
                retrievals[generated.case_id] = retrieve_vocabulary(
                    generated.problem, config=retrieval_config
                )
                retrieval = retrievals[generated.case_id]
                graphs[generated.case_id] = preparation.graph(generated.problem, retrieval)
                pair_reports[generated.case_id] = preparation.pairs(
                    generated.problem, graphs[generated.case_id], enabled=pairwise
                )
                interaction_pairs[generated.case_id] = pair_reports[generated.case_id].pairs
                for obj in generated.problem.objects:
                    menu = retrieval.for_object(obj.object_id)
                    grammars[generated.case_id, obj.object_id] = with_immutable_context(
                        mapping_grammar(
                            obj,
                            menu.classes,
                            menu.properties,
                            max_depth=max_depth,
                            max_constructors=max_constructors,
                            fixed_axioms=generated.problem.fixed_axioms,
                            source_classes=menu.source_classes,
                            target_classes=menu.target_classes,
                            source_properties=menu.source_properties,
                            target_properties=menu.target_properties,
                            constraint_identity=canonical_hash((generated.problem.policy, menu)),
                        ),
                        generated.problem.fixed_axioms,
                        generated.problem.policy,
                    )
                acquisition_completed.add((epoch, case.case_id))
                checkpoint(epoch)
            if any(
                (epoch, case.case_id) not in acquisition_completed for case, _ in scheduled_cases
            ):
                interrupted = True
                checkpoint(epoch)
                break
            acquisition_epoch = epoch
            checkpoint(
                epoch,
                next_offset if epoch == next_epoch else 0,
                saved_loss if epoch == next_epoch else 0.0,
                saved_optimized if epoch == next_epoch else 0,
            )
        model.train()
        train_loss, optimized = (saved_loss, saved_optimized) if epoch == next_epoch else (0.0, 0)
        order = [*training, *sampled_training.values()]
        random.Random(seed + epoch).shuffle(order)
        requested_order = tuple(case.case_id for case, _ in order)
        if epoch_order and epoch_order != requested_order:
            raise ValueError("Partial epoch order changed; exact resume is incompatible")
        epoch_order = requested_order
        if current_phase == "acquisition":
            current_phase = "train"
        offset_start = (
            (next_offset if epoch == next_epoch else 0) if current_phase == "train" else len(order)
        )
        processed = offset_start
        for offset in range(offset_start, len(order), batch_cases):
            if remaining_training_seconds() <= 0:
                break
            batch = order[offset : offset + batch_cases]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack([case_loss(case, cache) for case, cache in batch]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gradient_norm)
            optimizer.step()
            train_loss += float(loss.detach()) * len(batch)
            optimized += len(batch)
            processed = offset + len(batch)
            checkpoint(epoch, processed, train_loss, optimized)
        if processed < len(order):
            interrupted = True
            checkpoint(epoch, processed, train_loss, optimized)
            break
        model.eval()
        if current_phase == "train":
            current_phase = "development_loss"
        if development_progress.get("epoch") != epoch:
            development_progress = dict(epoch=epoch, losses={}, decoded={}, generated={})
        checkpoint(epoch, len(order), train_loss, optimized)
        with torch.no_grad():
            for case, cache in development:
                if case.case_id in development_progress["losses"]:
                    continue
                if remaining_training_seconds() <= 0:
                    interrupted = True
                    break
                development_progress["losses"][case.case_id] = float(case_loss(case, cache))
                checkpoint(epoch, len(order), train_loss, optimized)
        if interrupted:
            break
        dev_loss = sum(development_progress["losses"].values()) / len(development)
        current_phase = "development_selection"
        checkpoint(epoch, len(order), train_loss, optimized)
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(1, optimized),
            "optimized_cases": optimized,
            "development_loss": dev_loss,
            "loss_eligibility": copy.deepcopy(loss_eligibility),
        }
        # Evaluate on the first epoch and periodically, so a short bounded run can
        # always produce a reviewable decoded checkpoint.
        if epoch == 0 or (epoch + 1) % development_decode_every_epochs == 0 or epoch + 1 == epochs:
            decoded: dict[str, dict[str, Any]] = development_progress["decoded"]
            generated_reports: dict[str, dict[str, Any]] = development_progress["generated"]
            with torch.no_grad():
                for case, cache in development:
                    if case.case_id in generated_reports:
                        continue
                    remaining = remaining_training_seconds()
                    if remaining <= 0:
                        interrupted = True
                        break
                    memory = model.encode(graphs[case.case_id])
                    unary, pairs = model.score_inventory(
                        case.problem.objects,
                        memory,
                        interaction_pairs=interaction_pairs[case.case_id],
                    )
                    objective = make_objective(
                        case.problem.objects,
                        tuple(tuple(float(v) for v in values) for values in unary),
                        pairs=tuple((*indices, float(value)) for indices, value in pairs.items()),
                        profile=profile,
                        scale=quantization_scale,
                    )
                    if case.case_id not in decoded:
                        decoded[case.case_id] = decoded_development(
                            objective,
                            cache,
                            max_checks=decode_max_checks,
                            deadline_seconds=min(decode_seconds, remaining),
                        )
                        checkpoint(epoch, len(order), train_loss, optimized)
                    remaining = remaining_training_seconds()
                    if remaining <= 0:
                        interrupted = True
                        break
                    generated_reports[case.case_id] = generated_development(
                        model,
                        case,
                        cache,
                        seconds=min(decode_seconds, remaining),
                        temperature=proposal_temperature,
                        selection_options=selection_options,
                        case_cpu_seconds=case_cpu_seconds,
                        semantic_target=semantic_targets[case.case_id],
                        target_basis=target_basis,
                        mixture_symbolic_weight=mixture_symbolic_weight,
                        fidelity_evaluator_labels=(fidelity_development_labels or {}).get(
                            case.case_id, ()
                        ),
                        selected_other_assignment=(
                            tuple(0 for _ in case.problem.objects)
                            if proposal_context == "selected_other_actions"
                            else None
                        ),
                        circuit_limits=dict(circuit_limits) if circuit_limits is not None else None,
                        compiler_cache_directory=compiler_cache_directory,
                        vtree_type=vtree_type,
                        graph=graphs[case.case_id],
                        draws_per_object=development_draws_per_object,
                        candidate_cap=candidate_cap,
                        mixtures=mixtures,
                        seed=seed,
                        profile=profile,
                        pair_factor_limit_per_object=pair_factor_limit_per_object,
                        pair_max_pairs=pair_max_pairs,
                        pair_max_factors=pair_max_factors,
                        max_depth=max_depth,
                        max_constructors=max_constructors,
                        max_circuit_nodes=max_circuit_nodes,
                        compile_seconds=compile_seconds,
                        proposal_arm=proposal_arm,
                        max_graph_nodes=max_graph_nodes,
                        max_graph_edges=max_graph_edges,
                        max_explanations=max_explanations,
                        max_text_tokens=max_text_tokens,
                        quantization_scale=quantization_scale,
                        retrieval_config=retrieval_config,
                    )

                    checkpoint(epoch, len(order), train_loss, optimized)
            if interrupted:
                checkpoint(epoch, len(order), train_loss, optimized)
                break

            complete_ids = [
                case.case_id
                for case, cache in development
                if cache.complete and any(label.usable for label in cache.labels)
            ]
            regrets = [
                decoded[key]["exact_regret"]
                for key in complete_ids
                if decoded[key]["exact_regret"] is not None
            ]
            coverage = len(regrets) / max(1, len(complete_ids))
            useful_coverage = sum(
                generated_reports.get(key, {}).get("useful_candidate_coverage") or 0.0
                for key in complete_ids
            ) / max(1, len(complete_ids))
            legacy_criterion = (
                -coverage,
                sum(regrets) / len(regrets) if regrets else float("inf"),
                -useful_coverage,
                dev_loss,
            )
            criterion = (
                generated_checkpoint_criterion(
                    [generated_reports.get(case.case_id, {}) for case, _ in development],
                    minimum_coverage=minimum_generated_coverage,
                    uncertainty_z=quality_uncertainty_z,
                    fallback=missing_label_fallback,
                )
                if revision == "v3"
                else legacy_criterion
            )
            row.update(
                decoded=decoded,
                generated=generated_reports,
                useful_candidate_coverage=useful_coverage,
                decoded_complete_coverage=coverage,
                decoded_mean_regret=sum(regrets) / len(regrets) if regrets else None,
                selection_criterion=criterion,
                selection_status=(
                    "eligible" if criterion is not None else "coverage_or_labels_unavailable"
                ),
            )
            improved = criterion is not None and (
                best_criterion is None
                or any(
                    a < b - min_dev_improvement and criterion[:i] == best_criterion[:i]
                    for i, (a, b) in enumerate(zip(criterion, best_criterion))
                )
            )
            if improved:
                best_criterion, best_state, best_epoch = (
                    criterion,
                    copy.deepcopy(model.state_dict()),
                    epoch + 1,
                )
                stale_evaluations = 0
            else:
                stale_evaluations += 1
        history.append(row)
        stopped_early = stale_evaluations >= patience
        current_phase = "acquisition"
        epoch_order = ()
        development_progress = {}
        checkpoint(epoch + 1)
        if stale_evaluations >= patience:
            break
    if best_state is None:
        if interrupted:
            raise TimeoutError(
                "Training interrupted; exact phase checkpoint retained for compatible resume"
            )
        last = history[-1].get("generated", {}) if history else {}
        raise ValueError(
            "No eligible development checkpoint; generated validation: "
            + str({key: value.get("decoded", value) for key, value in last.items()})
        )

    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "schema": f"exact-repair/training/{revision}",
        "status": "interrupted" if interrupted else "complete",
        "resumable": interrupted,
        "recovery_revision": "exact-phase-resume/v3.1",
        "encoder": encoder,
        "feature_schema": (
            FEATURE_SCHEMA_V3 if revision == "v3" else "exact-repair/observable-features/v2"
        ),
        "factor_gauge": "keep_reference_zero/v3" if revision == "v3" else "unrestricted/v2",
        "support_enabled": support_enabled,
        "support_target": support_target,
        "support_loss_weight": support_loss_weight,
        "risk_calibration": "uncalibrated" if revision == "v3" else "absent",
        "semantic_target": target_basis,
        "semantic_target_specs": {
            key: json.loads(canonical_json(value)) for key, value in semantic_targets.items()
        },
        "mixture_symbolic_weight": mixture_symbolic_weight if target_basis == "mixed" else None,
        "seed": seed,
        "epochs": epochs,
        "history": history,
        "checkpoint_criterion": (
            "generated_pool_verified_quality_effort"
            if revision == "v3"
            else "development_decoded_teacher_regret_on_complete_subset"
        ),
        "checkpoint_tiebreaks": (
            [
                "scheduled_generated_coverage",
                "external_quality_parent_mean_lower_bound",
                "verifier_checks",
            ]
            if revision == "v3"
            else [
                "complete_case_coverage",
                "cached_regret",
                "generated_useful_candidate_coverage",
                "joint_development_loss",
            ]
        ),
        "regret_scope": "common frozen teacher inventory diagnostic only for v3",
        "selection_fallback": missing_label_fallback,
        "loss_eligibility": loss_eligibility,
        "acquisition_rounds": acquisition_reports,
        "effective_preparation": json.loads(canonical_json(preparation)),
        "preparation_identity": preparation.content_hash,
        "pair_selection": {
            key: json.loads(canonical_json(value)) for key, value in pair_reports.items()
        },
        "selected_epoch": best_epoch,
        "elapsed_seconds": previous_elapsed + time.monotonic() - started,
        "execution_count": execution_count,
        "recovery_lineage": recovery_lineage,
        "total_training_seconds": total_training_seconds,
        "adaptation_new_parameters": adaptation_new_parameters,
        "proposal_coverage": proposal_coverage,
        "proposal_arm": proposal_arm,
        "retrieval": {
            key: json.loads(canonical_json(value.provenance)) for key, value in retrievals.items()
        },
        "model_configuration": dict(model.config),
        "settings": {
            "layers": layers,
            "heads": heads,
            "dropout": dropout,
            "hidden_dim": hidden_dim,
            "batch_cases": batch_cases,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "clip_gradient_norm": clip_gradient_norm,
            "loss_weights": loss_weights,
            "interaction_loss_weight": interaction_loss_weight,
            "risk_loss_weight": risk_loss_weight,
            "sampled_proposal_loss_weight": sampled_proposal_loss_weight,
            "fidelity_loss_weight": fidelity_loss_weight,
            "minimum_generated_coverage": minimum_generated_coverage,
            "quality_uncertainty_z": quality_uncertainty_z,
            "missing_label_fallback": missing_label_fallback,
            "active_round_every_epochs": active_round_every_epochs,
            "sampled_assignments": sampled_assignments,
            "smooth_l1_beta": smooth_l1_beta,
            "ranking_temperature": ranking_temperature,
            "max_pairs": max_repair_pairs_per_case,
            "proposal_temperature": proposal_temperature,
            "mixtures": mixtures,
            "device": device,
            "threads": threads,
            "quantization_scale": quantization_scale,
            "decode_every": development_decode_every_epochs,
            "decode_seconds": decode_seconds,
            "decode_max_checks": decode_max_checks,
            "deadline_seconds": deadline_seconds,
            "patience": patience,
            "min_dev_improvement": min_dev_improvement,
            "compile_seconds": compile_seconds,
            "max_circuit_nodes": max_circuit_nodes,
            "max_depth": max_depth,
            "max_constructors": max_constructors,
            "max_graph_nodes": max_graph_nodes,
            "max_graph_edges": max_graph_edges,
            "max_explanations": max_explanations,
            "max_text_tokens": max_text_tokens,
            "pair_factor_limit_per_object": pair_factor_limit_per_object,
            "pair_max_pairs": pair_max_pairs,
            "pair_max_factors": pair_max_factors,
            "development_draws_per_object": development_draws_per_object,
            "candidate_cap": candidate_cap,
        },
        "model_hash": model_digest(model),
        "splits": {
            case.case_id: {"parent": case.structural_parent, "split": case.split}
            for case, _ in all_cases
        },
        "coverage": {case.case_id: cache.coverage for case, cache in all_cases},
        "arm": arm,
        "warm_start_hash": warm_start_hash,
        "scope": "conformance; no model-quality or transfer claim",
    }


def _protocol_arguments(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Translate each optimizer/decoder setting, rejecting unsupported declarations."""
    from exact.repair.retrieval import RetrievalConfig

    training, graph = protocol["training"], protocol["graph"]
    revision = "v3" if protocol.get("schema", "").endswith("/v3") else "v2"
    supported = {
        "optimizer": "adamw",
        "dtype": "float32",
        "benefit_loss": "anchored_full_repair_huber_plus_ranking",
        "proposal_loss": "complete_cache_marginal_cross_entropy",
        "checkpoint_selection": "development_decoded_teacher_regret_on_complete_subset",
        "held_out_supervision": False,
    }
    if revision == "v3":
        if protocol["model"]["unary_benefit"] is not True:
            raise ValueError("Disabled unary_benefit is not supported by this trainer")
        supported.update(
            benefit_loss="anchored_value_rank_quartet_risk",
            proposal_loss="exact_and_sample_conditioned",
            checkpoint_selection="generated_pool_verified_quality_effort",
        )
    if any(training[key] != value for key, value in supported.items()):
        raise ValueError("Unsupported optimizer, loss, checkpoint or supervision protocol")
    return {
        "revision": revision,
        "encoder": graph["encoder"],
        "pairwise": protocol["model"]["pair_benefit"] if revision == "v3" else False,
        **(
            {
                "case_cpu_seconds": protocol["resources"]["case_cpu_seconds"],
                "target_basis": protocol["losses"]["target_basis"],
                "desired_family_weight": protocol["teacher"]["desired_family_weight"],
                "false_positive_weight": protocol["teacher"]["false_positive_weight"],
                "mixture_symbolic_weight": protocol["losses"].get("mixture_symbolic_weight", 0.5),
                "max_collection_rounds": protocol["collection"]["rounds"],
                "collection_cases_per_round": protocol["collection"]["cases_per_round"],
                "collection_options": protocol["collection"],
                "plan_risk": protocol["model"]["plan_risk"],
                "support_enabled": protocol["model"].get("support_enabled", False),
                "support_readout_identity": SUPPORT_READOUT_IDENTITY,
                "support_target": protocol["model"].get(
                    "support_target", "qualified_witness_violation/v1"
                ),
                "support_loss_weight": protocol["losses"].get("support_loss_weight", 0.2),
                "selection_options": protocol["selection"],
                "proposal_context": protocol["model"]["proposal_context"],
                "pair_factor_bound": protocol["objective"]["pair_factor_bound"],
                "circuit_limits": {
                    key: protocol["circuit"][key]
                    for key in (
                        "allocated_node_limit",
                        "live_node_limit",
                        "reachable_node_limit",
                        "element_limit",
                        "rss_mb",
                        "call_seconds",
                        "aggregate_seconds",
                    )
                },
                "compiler_cache_directory": protocol["circuit"]["cache_directory"],
                "vtree_type": (
                    "right"
                    if protocol["circuit"]["vtree"] == "right_linear"
                    else protocol["circuit"]["vtree"]
                ),
            }
            if revision == "v3"
            else {}
        ),
        **(
            {
                key: training[key]
                for key in (
                    "interaction_loss_weight",
                    "risk_loss_weight",
                    "sampled_proposal_loss_weight",
                    "minimum_generated_coverage",
                    "quality_uncertainty_z",
                    "missing_label_fallback",
                    "sampled_assignments",
                    "active_round_every_epochs",
                )
            }
            if revision == "v3"
            else {}
        ),
        "epochs": training["max_epochs"],
        "hidden_dim": graph["hidden_width"],
        "layers": graph["layers"],
        "heads": graph["attention_heads"],
        "dropout": graph["dropout"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "batch_cases": training["batch_cases"],
        "clip_gradient_norm": training["clip_gradient_norm"],
        "smooth_l1_beta": training["smooth_l1_beta"],
        "ranking_temperature": training["ranking_temperature"],
        "max_repair_pairs_per_case": training["max_repair_pairs_per_case"],
        "loss_weights": (
            training["benefit_loss_weight"],
            training["ranking_loss_weight"],
            training["proposal_loss_weight"],
        ),
        "proposal_temperature": protocol["teacher"]["temperature"],
        "mixtures": protocol["circuit"]["components"],
        "development_decode_every_epochs": training["development_decode_every_epochs"],
        "decode_seconds": protocol["resources"]["run_deadline_seconds"],
        "decode_max_checks": protocol["teacher"]["max_assignments"],
        "patience": training["patience"],
        "min_dev_improvement": training["min_dev_improvement"],
        "deadline_seconds": protocol["resources"]["stage_deadline_seconds"]["train"],
        "total_training_seconds": protocol["resources"]["stage_deadline_seconds"]["train"],
        "device": training["device"],
        "threads": training["threads"],
        "quantization_scale": protocol["solver"]["quantization_scale"],
        "development_draws_per_object": protocol["circuit"]["max_draws_per_object"],
        "candidate_cap": protocol["actions"]["max_states"]["generated"],
        "compile_seconds": protocol["circuit"]["compile_seconds"],
        "max_circuit_nodes": protocol["circuit"]["max_nodes"],
        "max_depth": protocol["grammar"]["max_depth"],
        "max_constructors": protocol["grammar"]["max_constructors"],
        "max_graph_nodes": graph["max_nodes"],
        "max_graph_edges": graph["max_edges"],
        "max_explanations": graph["max_explanations"],
        "max_text_tokens": graph["max_text_tokens"],
        "pair_factor_limit_per_object": graph["pair_factor_limit_per_object"],
        "pair_max_pairs": graph.get("pair_max_pairs", 128),
        "pair_max_factors": graph.get("pair_max_factors", 65536),
        "retrieval_config": RetrievalConfig(
            classes_per_side=protocol["grammar"]["classes_per_side"],
            properties_per_side=protocol["grammar"]["properties_per_side"],
            endpoints_per_side=protocol["grammar"]["endpoints_per_side"],
            neighborhood_hops=protocol["grammar"]["retrieval_hops"],
        ),
    }


def _resolve_architecture(
    config: Mapping[str, Any],
    *,
    encoder: str | None,
    pairwise: bool | None,
    warm: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """V3 architecture is frozen by its protocol; overrides require a new protocol."""
    resolved = dict(config)
    if warm is not None and (
        warm.get("model_schema") not in {"exact-repair/model/v2", "exact-repair/model/v3"}
        or warm.get("training_provenance", {}).get("heldout_supervision") is not False
    ):
        raise ValueError(
            "Adaptation requires a versioned checkpoint with declared held-out restrictions"
        )
    if config["revision"] != "v3":
        # Preserve the historical v2 CLI contract; v3 amendments are explicit files.
        architecture = warm["config"] if warm else {}
        for key in ("hidden_dim", "layers", "heads", "dropout"):
            if key in architecture:
                resolved[key] = architecture[key]
        resolved["encoder"] = encoder or architecture.get("encoder", config["encoder"])
        resolved["pairwise"] = architecture.get("pairwise", bool(pairwise))
        return resolved
    for key, override in (("encoder", encoder), ("pairwise", pairwise)):
        if override is not None and override != config[key]:
            raise ValueError(
                f"CLI {key} conflicts with the protocol architecture; amend the protocol explicitly"
            )
    if warm is not None:
        expected = {
            key: config[key]
            for key in (
                "revision",
                "hidden_dim",
                "layers",
                "heads",
                "dropout",
                "encoder",
                "pairwise",
                "plan_risk",
                "support_enabled",
                "support_readout_identity",
                "pair_factor_bound",
            )
        }
        expected["feature_dim"] = 128
        actual = warm.get("config", {})
        conflicts = [
            key for key, value in expected.items() if key not in actual or actual[key] != value
        ]
        if warm.get("model_schema") != "exact-repair/model/v3":
            conflicts.append("model_schema")
        if conflicts:
            raise ValueError(
                "Warm-start architecture conflicts with the protocol: "
                + ", ".join(conflicts)
                + "; use compatible weights or an explicit protocol amendment"
            )
    return resolved


def _protocol_profile(protocol: Mapping[str, Any]) -> tuple:
    return tuple(sorted(protocol["preferences"]["cost_weights"].items()))


def _train_payload(training, development, options):
    """Return a small immutable artifact reference for v3; preserve legacy v2 bytes."""
    import io

    import torch

    revision = options.get("revision", "v3")
    if revision == "v3" and options.get("checkpoint_path") is None:
        raise ValueError("V3 training requires an output checkpoint directory")
    model, report = train_cases(training, development, **options)
    state = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": model.metadata,
        "config": model.config,
    }
    if revision == "v3":
        state["model_schema"] = "exact-repair/model/v3"
        return (
            _publish_training_checkpoint(
                state, Path(options["checkpoint_path"]).parent / "checkpoints"
            ),
            publish_report(report, Path(options["checkpoint_path"]).parent / "reports"),
        )
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue(), report


def main() -> int:
    """Prepare or train one declared configuration; never fan out into a campaign."""
    from tools.repair.prepare import (
        generated_from_protocol,
        load_preparation,
        load_protocol,
        prepare_real_manifest,
        read_label_cache,
        save_preparation,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("specs/exact-repair/protocol/smoke.json")
    )
    parser.add_argument("--encoder", choices=("hgt", "rgcn", "none"))
    parser.add_argument("--seed", type=int)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--prepared", type=Path, help="resume an integrity-protected training manifest"
    )
    source.add_argument(
        "--real-manifest", type=Path, help="explicit local training-side ontology pairs"
    )
    parser.add_argument(
        "--warm-start", type=Path, help="weights-only checkpoint for a separate adaptation run"
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="save cases and bounded labels without training"
    )
    parser.add_argument(
        "--retry-label-transport-errors",
        action="store_true",
        help="retry recorded oversized result errors within the original cumulative budgets",
    )
    parser.add_argument(
        "--case-limit", type=int, help="explicit per-split conformance cap, recorded in coverage"
    )
    parser.add_argument(
        "--resume-endpoint-retrieval",
        type=Path,
        help="dependency-checked rollback to an archived pre-optimization checkpoint",
    )
    parser.add_argument(
        "--resume-acquisition-deadline",
        action="store_true",
        help="recover the pinned pre-optimization case-deadline defect without renewing its cap",
    )
    parser.add_argument(
        "--resume-report-transport",
        action="store_true",
        help="finalize a dependency-checked completed run after its report transfer failed",
    )
    parser.add_argument("--pairwise", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--fidelity-labels",
        type=Path,
        help="Offline validated v3 packet/comparison envelopes; never invokes a judge",
    )
    parser.add_argument(
        "--fidelity-development-labels",
        type=Path,
        help="Independent offline v3 development evaluator comparisons",
    )
    args = parser.parse_args()
    from exact.repair.checkpointing import CumulativeBudget

    protocol = load_protocol(args.protocol)
    if protocol.get("schema") == "exact-repair/protocol/v3":
        from exact.repair.protocol import load_protocol_v3, training_projection_v3

        protocol = training_projection_v3(load_protocol_v3(args.protocol, for_execution=True))
    seed = args.seed if args.seed is not None else protocol["training"]["seeds"][0]
    if seed not in protocol["training"]["seeds"]:
        parser.error("--seed must be a declared protocol seed")
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("--case-limit must be positive")
    config = _protocol_arguments(protocol)
    warm = None
    if args.warm_start:
        import torch

        warm = torch.load(args.warm_start, weights_only=True, map_location="cpu")
    config = _resolve_architecture(config, encoder=args.encoder, pairwise=args.pairwise, warm=warm)
    profile = _protocol_profile(protocol)
    from contextlib import nullcontext

    campaign_context: Any = nullcontext(None)
    if config["revision"] == "v3":
        import torch

        allocation = protocol["resources"]["allocated_gpus"]
        slurm_devices = os.environ.get("SLURM_STEP_GPUS", os.environ.get("SLURM_JOB_GPUS", ""))
        if (config["device"] in {"cuda", "auto"} and torch.cuda.device_count() > allocation) or (
            slurm_devices and len(slurm_devices.split(",")) > allocation
        ):
            raise ValueError("Allocated CUDA devices exceed the frozen allocated_gpus declaration")
        campaign_context = CumulativeBudget(
            args.output / "campaign-budget.json",
            canonical_hash(
                (protocol, seed, config["encoder"], config["pairwise"], args.case_limit)
            ),
            protocol["resources"]["campaign_wall_seconds"],
            gpu_hours=protocol["resources"]["campaign_gpu_hours"],
            allocated_gpus=allocation,
        )
    with campaign_context as campaign_budget:
        if campaign_budget is not None:
            campaign_budget.begin()
        caches: dict[str, TeacherCache]
        resume_preparation = args.output / "preparation.json"
        if resume_preparation.exists():
            cases, caches, preparation = load_preparation(resume_preparation)
            if (
                preparation.get("protocol_hash") != canonical_hash(protocol)
                or preparation.get("case_limit") != args.case_limit
            ):
                raise ValueError("Preparation checkpoint protocol or case selection changed")
        elif args.real_manifest:
            with CumulativeBudget(
                args.output / "corpus-budget.json",
                canonical_hash(protocol),
                protocol["resources"]["stage_deadline_seconds"]["corpus"],
                cpu_seconds=protocol["resources"].get("stage_cpu_seconds", {}).get("corpus"),
            ) as corpus_budget:
                allowed = corpus_budget.begin(
                    seconds=campaign_budget.remaining if campaign_budget is not None else None
                )
                prepared = bounded_call(
                    prepare_real_manifest,
                    args.real_manifest,
                    deadline_seconds=allowed,
                    call_seconds=protocol["resources"]["verification_call_seconds"],
                    timeout=allowed,
                    cpu_seconds=corpus_budget.remaining_cpu,
                    memory_mb=protocol["resources"]
                    .get("stage_rss_mb", {})
                    .get("corpus", protocol["resources"]["memory_mb"]),
                )
                corpus_budget.finish(
                    prepared.status, cpu_seconds=dict(prepared.resource_usage).get("cpu_seconds")
                )
                if prepared.status != "complete":
                    raise RuntimeError(
                        "Real corpus preparation " + prepared.status + ": " + prepared.detail
                    )
                cases, preparation = prepared.value
            caches = {}
        elif args.prepared:
            cases, caches, preparation = load_preparation(args.prepared)
            if preparation.get("protocol_hash") != canonical_hash(protocol):
                raise ValueError("Prepared teacher data belongs to a different protocol")
        else:
            with CumulativeBudget(
                args.output / "corpus-budget.json",
                canonical_hash(protocol),
                protocol["resources"]["stage_deadline_seconds"]["corpus"],
                cpu_seconds=protocol["resources"].get("stage_cpu_seconds", {}).get("corpus"),
            ) as corpus_budget:
                generated = bounded_call(
                    generated_from_protocol,
                    protocol,
                    timeout=corpus_budget.begin(
                        seconds=campaign_budget.remaining if campaign_budget is not None else None
                    ),
                    cpu_seconds=corpus_budget.remaining_cpu,
                    memory_mb=protocol["resources"]
                    .get("stage_rss_mb", {})
                    .get("corpus", protocol["resources"]["memory_mb"]),
                )
                corpus_budget.finish(
                    generated.status, cpu_seconds=dict(generated.resource_usage).get("cpu_seconds")
                )
                if generated.status != "complete":
                    raise RuntimeError(
                        "Corpus preparation " + generated.status + ": " + generated.detail
                    )
                cases = generated.value
            caches = {}
            preparation = {"requested": len(cases), "produced": len(cases), "origin": "generated"}
        # Keep every omitted/test/unknown case in the denominator and evaluator store.
        selected, counts = [], {"train": 0, "development": 0}
        for case in cases:
            if case.split not in counts or case.control == "ambiguous":
                continue
            if args.case_limit is None or counts[case.split] < args.case_limit:
                selected.append(case)
                counts[case.split] += 1
        preparation.update(
            protocol_hash=canonical_hash(protocol),
            protocol=protocol,
            seed=seed,
            case_limit=args.case_limit,
            selected_counts=counts,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        save_preparation(resume_preparation, cases, preparation, caches)
        rows = {row["case_id"]: row for row in preparation.get("label_rows", [])}
        identity = canonical_hash(
            (protocol, [(c.case_id, c.problem.content_hash) for c in cases], args.case_limit)
        )
        with CumulativeBudget(
            args.output / "label-budget.json",
            identity,
            protocol["resources"]["stage_deadline_seconds"]["label"],
            cpu_seconds=protocol["resources"].get("stage_cpu_seconds", {}).get("label"),
        ) as label_budget:
            # Imported preparation has already paid for its labels. Keep that cost
            # when moving compatible work into a replacement or training directory.
            label_budget.state["spent_seconds"] = max(
                label_budget.state["spent_seconds"], preparation.get("label_seconds", 0.0)
            )
            if label_budget.remaining_cpu is not None:
                label_budget.state["spent_cpu_seconds"] = max(
                    label_budget.state["spent_cpu_seconds"],
                    preparation.get("label_cpu_seconds", 0.0),
                )
            label_budget._save()
            for case in selected:
                if case.case_id in rows:
                    previous = rows[case.case_id]
                    if not (
                        args.retry_label_transport_errors
                        and case.case_id not in caches
                        and _retryable_label_transport(previous)
                    ):
                        continue
                    history = preparation.setdefault("label_retry_history", [])
                    if sum(row["case_id"] == case.case_id for row in history) >= 2:
                        continue
                    history.append(dict(previous))
                    save_preparation(resume_preparation, cases, preparation, caches)
                if case.case_id in caches:
                    row = dict(
                        case_id=case.case_id,
                        status="cached",
                        coverage=caches[case.case_id].coverage,
                    )
                elif label_budget.remaining <= 0:
                    row = dict(case_id=case.case_id, status="label_stage_deadline")
                elif len(case.probes) > protocol["teacher"]["max_queries"]:
                    row = dict(case_id=case.case_id, status="query_cap")
                else:
                    cpu_cap = (
                        min(label_budget.remaining_cpu, protocol["resources"]["case_cpu_seconds"])
                        if label_budget.remaining_cpu is not None
                        else None
                    )
                    reserved = label_budget.begin(
                        (
                            min(
                                protocol["teacher"]["case_deadline_seconds"],
                                campaign_budget.remaining,
                            )
                            if campaign_budget is not None
                            else protocol["teacher"]["case_deadline_seconds"]
                        ),
                        cpu_seconds=cpu_cap,
                    )
                    labeled = None
                    try:
                        labeled = bounded_call(
                            _label_payload,
                            case,
                            args.output / "label-caches",
                            timeout=reserved,
                            cpu_seconds=cpu_cap,
                            memory_mb=protocol["resources"]
                            .get("stage_rss_mb", {})
                            .get("label", protocol["resources"]["memory_mb"]),
                            max_assignments=protocol["teacher"]["max_assignments"],
                            deadline_seconds=reserved,
                            call_seconds=protocol["resources"]["verification_call_seconds"],
                            profile=profile,
                            desired_family_weight=protocol["teacher"]["desired_family_weight"],
                            false_positive_weight=protocol["teacher"]["false_positive_weight"],
                        )
                        if labeled.status != "complete":
                            raise ValueError(f"Label worker {labeled.status}: {labeled.detail}")
                        cache = read_label_cache(labeled.value, args.output / "label-caches", case)
                        caches[case.case_id] = cache
                        row = dict(
                            case_id=case.case_id,
                            status="complete" if cache.complete else "partial",
                            coverage=cache.coverage,
                            artifact=labeled.value,
                        )
                    except ValueError as error:
                        row = dict(
                            case_id=case.case_id, status="unverified_parent", detail=str(error)
                        )
                    finally:
                        label_budget.finish(
                            cpu_seconds=(
                                dict(labeled.resource_usage).get("cpu_seconds")
                                if labeled is not None
                                else None
                            )
                        )
                rows[case.case_id] = row
                preparation.update(
                    label_rows=list(rows.values()),
                    labelled_cases=len(caches),
                    label_seconds=label_budget.state["spent_seconds"],
                    label_cpu_seconds=label_budget.state.get("spent_cpu_seconds"),
                )
                save_preparation(resume_preparation, cases, preparation, caches)
            preparation.update(
                label_rows=list(rows.values()),
                labelled_cases=len(caches),
                label_seconds=label_budget.state["spent_seconds"],
                label_cpu_seconds=label_budget.state.get("spent_cpu_seconds"),
            )
            save_preparation(resume_preparation, cases, preparation, caches)
        if args.prepare_only:
            print(
                json.dumps({key: value for key, value in preparation.items() if key != "protocol"})
            )
            return 0
        labelled_train = [
            (case, caches[case.case_id])
            for case in selected
            if case.split == "train" and case.case_id in caches
        ]
        labelled_dev = [
            (case, caches[case.case_id])
            for case in selected
            if case.split == "development" and case.case_id in caches
        ]
        origins = {case.origin for case in selected}
        if len(origins) != 1:
            raise ValueError("Generated-only and real-adaptation runs must be reported separately")
        arm = "training_side_adaptation" if origins == {"training_side_real"} else "generated_only"
        import torch

        from exact.repair.semantic_fidelity import read_fidelity_training_artifact

        def read_fidelity_file(path, expected_split):
            return {} if path is None else read_fidelity_training_artifact(path, expected_split)

        fidelity_labels = read_fidelity_file(args.fidelity_labels, "train")
        fidelity_development_labels = read_fidelity_file(
            args.fidelity_development_labels, "development"
        )
        options = dict(
            config,
            fidelity_labels=fidelity_labels,
            fidelity_development_labels=fidelity_development_labels,
            seed=seed,
            profile=profile,
            arm=arm,
            warm_start_weights=warm["state_dict"] if warm else None,
            warm_start_metadata=warm["metadata"] if warm else None,
            checkpoint_path=args.output / "training-state.pt",
            resume_acquisition_deadline=args.resume_acquisition_deadline,
            resume_endpoint_retrieval=args.resume_endpoint_retrieval,
            resume_report_transport=args.resume_report_transport,
        )
        with CumulativeBudget(
            args.output / "training-budget.json",
            canonical_hash((identity, seed, options["encoder"], options["pairwise"], arm)),
            config["deadline_seconds"],
            cpu_seconds=protocol["resources"].get("stage_cpu_seconds", {}).get("train"),
        ) as training_budget:
            remaining = training_budget.begin(
                seconds=campaign_budget.remaining if campaign_budget is not None else None
            )
            # Leave time for portable checkpoint transport/cleanup inside the stage cap.
            options["deadline_seconds"] = max(0.001, remaining - min(5.0, remaining / 10))
            outcome = bounded_call(
                _train_payload,
                labelled_train,
                labelled_dev,
                options,
                timeout=remaining,
                memory_mb=protocol["resources"]
                .get("stage_rss_mb", {})
                .get("train", protocol["resources"]["memory_mb"]),
                cpu_seconds=training_budget.remaining_cpu,
            )
            training_budget.finish(
                outcome.status, cpu_seconds=dict(outcome.resource_usage).get("cpu_seconds")
            )
        if outcome.status != "complete":
            report = {
                "schema": f"exact-repair/training/{options.get('revision', 'v2')}",
                "status": outcome.status,
                "detail": outcome.detail,
                "protocol_hash": canonical_hash(protocol),
                "scope": "no completed model checkpoint",
            }
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report))
            return 2
        checkpoint_transfer, report = outcome.value
        if options.get("revision") == "v3":
            checkpoint = _read_training_checkpoint(checkpoint_transfer, args.output / "checkpoints")
            report_transfer = report
            report = read_report(report_transfer, args.output / "reports")
            report["training_report_artifact"] = report_transfer
            report["selected_checkpoint_artifact"] = checkpoint_transfer
        else:
            import io

            checkpoint = torch.load(
                io.BytesIO(checkpoint_transfer), weights_only=True, map_location="cpu"
            )
        report.update(
            protocol_hash=canonical_hash(protocol),
            preparation_hash=canonical_hash(preparation),
            measured_training_resources=dict(outcome.resource_usage),
            model_configuration=dict(checkpoint["config"]),
        )
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        training_provenance = {
            "arm": arm,
            "model_schema": f"exact-repair/model/{options.get('revision', 'v2')}",
            "grammar_schema_hash": canonical_hash(protocol["grammar"]),
            "action_schema_hash": canonical_hash(protocol["actions"]),
            "retrieval_schema_hash": canonical_hash(
                ("observable-retrieval/v1", protocol["grammar"])
            ),
            "feature_schema_hash": canonical_hash(
                (
                    f"observable-graph/{options.get('revision', 'v2')}",
                    protocol["graph"],
                    checkpoint["config"]["feature_dim"],
                )
            ),
            "protocol_hash": canonical_hash(protocol),
            "training_groups": sorted({case.structural_parent for case, _ in labelled_train}),
            "development_groups": sorted({case.structural_parent for case, _ in labelled_dev}),
            "excluded_test_groups": sorted(
                {case.structural_parent for case in cases if case.split == "test"}
            ),
            "case_manifest": [
                {
                    "case_id": case.case_id,
                    "parent": case.structural_parent,
                    "split": case.split,
                    "input_hash": case.problem.content_hash,
                    "origin": case.origin,
                }
                for case, _ in [*labelled_train, *labelled_dev]
            ],
            "heldout_supervision": False,
            "warm_start_weights_only": warm is not None,
            "model_configuration": dict(checkpoint["config"]),
            "supervision_manifest_hash": preparation.get("manifest_hash"),
        }
        checkpoint.update(
            model_schema=f"exact-repair/model/{options.get('revision', 'v2')}",
            training_provenance=training_provenance,
            report_hash=canonical_hash(report),
        )
        save_training_state(args.output / "model.pt", checkpoint)
        print(json.dumps(report))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
