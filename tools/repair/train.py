"""Protocol-driven repair training with bounded teachers and decoded development selection."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from exact.repair.graph import build_observable_graph, observable_interaction_pairs
from exact.repair.learning import (
    OwlTeacherOracle,
    RepairLabel,
    TeacherCache,
    benefit_losses,
    enumerate_teacher,
    evaluate_teacher,
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

DEFAULT_PROFILE = (
    ("delete", 0.1),
    ("edit", 0.05),
    ("ontology_edit", 0.02),
    ("human_authored_ontology_edit", 0.03),
)


def _assignment_label(
    case: GeneratedCase,
    assignment: tuple[int, ...],
    profile: tuple,
    desired_family_weight: float = 1.0,
    false_positive_weight: float = 1.0,
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
    if report.logical_status != "VERIFIED_FEASIBLE":
        feasible = False if report.logical_status == "VERIFIED_INFEASIBLE" else None
        return RepairLabel(assignment, feasible, None, cost)
    semantic = evaluate_teacher(
        OwlTeacherOracle(verifier, snapshot, case.probes),
        case.probes,
        desired_weights={probe.family: desired_family_weight for probe in case.probes},
        unwanted_weights={probe.family: false_positive_weight for probe in case.probes},
    )
    return RepairLabel(assignment, True, semantic.benefit, cost, semantic.outcomes)


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
            timeout=min(call_seconds, max(0.0, remaining)),
        )
        if result.status == "complete":
            return cast(RepairLabel, result.value)
        return RepairLabel(assignment, None, None, 0.0)

    return enumerate_teacher(
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
        },
        max_assignments=max_assignments,
        deadline_seconds=max(0.001, deadline_seconds - (time.monotonic() - started)),
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
    model, case, cache, *, seconds: float, temperature: float = 1.0, **options
):
    """Evaluate sampled grammar availability and expose uncached policy choices.

    Regret remains scoped to the common complete teacher inventory. Additional
    generated bundles receive no fabricated teacher score or feasibility label.
    """
    from dataclasses import replace

    from exact.repair.pipeline import bounded_freeze_neural_round

    if seconds <= 0:
        return {"status": "generation_deadline", "useful_candidate_coverage": None}
    started = time.monotonic()
    frozen = bounded_freeze_neural_round(case.problem, model, seconds=seconds * 0.8, **options)
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
    mapped_labels = tuple(
        replace(
            label, assignment=tuple(menu[choice] for menu, choice in zip(maps, label.assignment))
        )
        for label in cache.labels
        if all(choice in menu for menu, choice in zip(maps, label.assignment))
    )
    # Even a fully decided subset is not the original teacher search universe.
    mapped_cache = TeacherCache(
        tuple(len(obj.candidates) for obj in generated.problem.objects),
        mapped_labels,
        False,
        "generated_inventory_policy_cache",
        cache.hashes,
        cache.elapsed_seconds,
    )
    remaining = seconds - (time.monotonic() - started)
    decoded = (
        decoded_development(generated.objective, mapped_cache, deadline_seconds=remaining)
        if remaining > 0
        else {"status": "generation_deadline"}
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
        "useful_candidate_coverage": useful,
        "teacher_optimal_repair_available": optimum_present,
        "candidate_counts": [len(obj.candidates) for obj in generated.problem.objects],
        "mapped_teacher_candidates": [len(menu) for menu in maps],
        "decoded": decoded,
        "proposal_reports": json.loads(canonical_json(generated.proposal_reports)),
        "elapsed_seconds": time.monotonic() - started,
        "scope": "generated-pool availability; unmapped bundles have unknown cached policy",
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
    development_draws_per_object: int = 32,
    candidate_cap: int = 64,
) -> tuple[Any, dict[str, Any]]:
    """Train joint benefits/proposals and select on periodic decoded development regret.

    Cases without complete finite distributions still contribute masked benefit
    regression/ranking. No test case can enter optimization or checkpoint selection.
    """
    import torch

    from exact.repair.grammar import mapping_grammar
    from exact.repair.model import RepairModel, repair_benefits
    from exact.repair.pipeline import model_digest, proposal_distribution
    from exact.repair.retrieval import retrieve_vocabulary

    started = time.monotonic()
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
    for case, cache in [*training, *development]:
        expected = {
            "input": case.problem.content_hash,
            "policy": case.problem.policy.content_hash,
            "query": canonical_hash(case.probes),
            "inventory": canonical_hash(tuple(obj.candidates for obj in case.problem.objects)),
            "profile": canonical_hash(profile),
        }
        if any(dict(cache.hashes).get(key) != digest for key, digest in expected.items()):
            raise ValueError("Teacher cache dependencies/profile do not match this training case")
        if cache.candidate_counts != tuple(len(obj.candidates) for obj in case.problem.objects):
            raise ValueError("Teacher cache candidate counts do not match")
    torch.manual_seed(seed)
    all_cases = [*training, *development]
    retrievals = {
        case.case_id: retrieve_vocabulary(case.problem, config=retrieval_config)
        for case, _ in all_cases
    }
    graphs = {
        case.case_id: build_observable_graph(
            case.problem.objects,
            fixed_axioms=case.problem.fixed_axioms,
            source_axioms=case.problem.source_axioms,
            target_axioms=case.problem.target_axioms,
            evidence=retrievals[case.case_id].graph_evidence(case.problem.evidence),
            retrieved_symbols=retrievals[case.case_id].symbols,
            explanations=retrievals[case.case_id].explanations,
            max_nodes=max_graph_nodes,
            max_edges=max_graph_edges,
            max_explanations=max_explanations,
            max_text_tokens=max_text_tokens,
        )
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
    interaction_pairs = {
        case.case_id: (
            observable_interaction_pairs(
                case.problem, per_object_limit=pair_factor_limit_per_object
            )
            if pairwise
            else ()
        )
        for case, _ in all_cases
    }
    proposal_coverage: dict[str, dict[str, Any]] = {}
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    best_criterion, best_state, best_epoch = None, None, None
    stale_evaluations = 0
    if not any(
        cache.complete and any(label.usable for label in cache.labels) for _, cache in development
    ):
        raise ValueError(
            "Decoded checkpoint selection requires a complete usable development cache"
        )

    def case_loss(case: GeneratedCase, cache: TeacherCache) -> Any:
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
        if cache.complete and any(label.usable for label in cache.labels):
            marginals = teacher_marginals(cache, proposal_temperature)
            for obj, target in zip(case.problem.objects, marginals):
                key = case.case_id + ":" + obj.object_id
                if key in failed_compilations:
                    continue
                remaining = deadline_seconds - (time.monotonic() - started)
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
                    )
                except (TimeoutError, RuntimeError) as error:
                    failed_compilations.add(key)
                    proposal_coverage[key] = {
                        "status": "compilation_unavailable",
                        "detail": str(error),
                    }
                    continue
                probabilities = torch.stack(
                    [
                        distribution.candidate_log_probability(candidate)
                        for candidate in obj.candidates
                    ]
                )
                weights = probabilities.new_tensor(target)
                positive = weights > 0
                missing = positive & ~torch.isfinite(probabilities)
                if missing.any():
                    proposal_coverage[key] = {
                        "status": "retrieval_miss",
                        "missing_target_mass": float(weights[missing].sum()),
                    }
                    # No reweighting onto reachable labels: retain the exact teacher
                    # distribution and explicitly omit this object's impossible loss.
                    continue
                proposal_coverage[key] = {
                    "status": "supervised",
                    "teacher_inventory_mass": float(probabilities.detach().exp().sum()),
                }
                loss = loss + loss_weights[2] * -(weights[positive] * probabilities[positive]).sum()
        return loss

    for epoch in range(epochs):
        if time.monotonic() - started >= deadline_seconds:
            break
        model.train()
        train_loss, optimized = 0.0, 0
        order = list(training)
        random.Random(seed + epoch).shuffle(order)
        for offset in range(0, len(order), batch_cases):
            if time.monotonic() - started >= deadline_seconds:
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
        model.eval()
        with torch.no_grad():
            dev_loss = sum(float(case_loss(case, cache)) for case, cache in development) / len(
                development
            )
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(1, optimized),
            "optimized_cases": optimized,
            "development_loss": dev_loss,
        }
        # Evaluate on the first epoch and periodically, so a short bounded run can
        # always produce a reviewable decoded checkpoint.
        if epoch == 0 or (epoch + 1) % development_decode_every_epochs == 0 or epoch + 1 == epochs:
            decoded: dict[str, dict[str, Any]] = {}
            generated: dict[str, dict[str, Any]] = {}
            with torch.no_grad():
                for case, cache in development:
                    remaining = deadline_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        decoded[case.case_id] = {
                            "status": "training_deadline",
                            "exact_regret": None,
                        }
                        continue
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
                    decoded[case.case_id] = decoded_development(
                        objective,
                        cache,
                        max_checks=decode_max_checks,
                        deadline_seconds=min(decode_seconds, remaining),
                    )
                    remaining = deadline_seconds - (time.monotonic() - started)
                    generated[case.case_id] = generated_development(
                        model,
                        case,
                        cache,
                        seconds=min(decode_seconds, remaining),
                        temperature=proposal_temperature,
                        graph=graphs[case.case_id],
                        draws_per_object=development_draws_per_object,
                        candidate_cap=candidate_cap,
                        mixtures=mixtures,
                        seed=seed,
                        profile=profile,
                        interaction_pairs=interaction_pairs[case.case_id],
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
            coverage = len(regrets) / len(complete_ids)
            useful_coverage = sum(
                generated.get(key, {}).get("useful_candidate_coverage") or 0.0
                for key in complete_ids
            ) / len(complete_ids)
            criterion = (
                -coverage,
                sum(regrets) / len(regrets) if regrets else float("inf"),
                -useful_coverage,
                dev_loss,
            )
            row.update(
                decoded=decoded,
                generated=generated,
                useful_candidate_coverage=useful_coverage,
                decoded_complete_coverage=coverage,
                decoded_mean_regret=criterion[1] if regrets else None,
            )
            improved = (
                best_criterion is None
                or criterion[0] < best_criterion[0]
                or (
                    criterion[0] == best_criterion[0]
                    and criterion[1] < best_criterion[1] - min_dev_improvement
                )
                or (
                    criterion[:2] == best_criterion[:2]
                    and (
                        criterion[2] < best_criterion[2] - min_dev_improvement
                        or (
                            criterion[2] == best_criterion[2]
                            and criterion[3] < best_criterion[3] - min_dev_improvement
                        )
                    )
                )
            )
            if regrets and improved:
                best_criterion, best_state, best_epoch = (
                    criterion,
                    copy.deepcopy(model.state_dict()),
                    epoch + 1,
                )
                stale_evaluations = 0
            else:
                stale_evaluations += 1
        history.append(row)
        if stale_evaluations >= patience:
            break
    if best_state is None:
        raise ValueError("No decoded development checkpoint completed within the training budget")

    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "schema": "exact-repair/training/v2",
        "encoder": encoder,
        "seed": seed,
        "epochs": epochs,
        "history": history,
        "checkpoint_criterion": "development_decoded_teacher_regret_on_complete_subset",
        "checkpoint_tiebreaks": [
            "complete_case_coverage",
            "generated_useful_candidate_coverage",
            "joint_development_loss",
        ],
        "regret_scope": "common frozen teacher inventory; generated-pool availability is a separate tiebreak",
        "selected_epoch": best_epoch,
        "elapsed_seconds": time.monotonic() - started,
        "adaptation_new_parameters": adaptation_new_parameters,
        "proposal_coverage": proposal_coverage,
        "proposal_arm": proposal_arm,
        "retrieval": {
            key: json.loads(canonical_json(value.provenance)) for key, value in retrievals.items()
        },
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
    supported = {
        "optimizer": "adamw",
        "dtype": "float32",
        "benefit_loss": "anchored_full_repair_huber_plus_ranking",
        "proposal_loss": "complete_cache_marginal_cross_entropy",
        "checkpoint_selection": "development_decoded_teacher_regret_on_complete_subset",
        "held_out_supervision": False,
    }
    if any(training[key] != value for key, value in supported.items()):
        raise ValueError("Unsupported optimizer, loss, checkpoint or supervision protocol")
    return {
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
        "retrieval_config": RetrievalConfig(
            classes_per_side=protocol["grammar"]["classes_per_side"],
            properties_per_side=protocol["grammar"]["properties_per_side"],
            endpoints_per_side=protocol["grammar"]["endpoints_per_side"],
            neighborhood_hops=protocol["grammar"]["retrieval_hops"],
        ),
    }


def _protocol_profile(protocol: Mapping[str, Any]) -> tuple:
    return tuple(sorted(protocol["preferences"]["cost_weights"].items()))


def _train_payload(training, development, options):
    """Return a portable checkpoint, never live tensor storage from an exited worker."""
    import io

    import torch

    model, report = train_cases(training, development, **options)
    buffer = io.BytesIO()
    torch.save(
        {"state_dict": model.state_dict(), "metadata": model.metadata, "config": model.config},
        buffer,
    )
    return buffer.getvalue(), report


def main() -> int:
    """Prepare or train one declared configuration; never fan out into a campaign."""
    from tools.repair.prepare import (
        generated_from_protocol,
        load_preparation,
        load_protocol,
        prepare_real_manifest,
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
        "--case-limit", type=int, help="explicit per-split conformance cap, recorded in coverage"
    )
    parser.add_argument("--pairwise", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    seed = args.seed if args.seed is not None else protocol["training"]["seeds"][0]
    if seed not in protocol["training"]["seeds"]:
        parser.error("--seed must be a declared protocol seed")
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("--case-limit must be positive")
    config = _protocol_arguments(protocol)
    profile = _protocol_profile(protocol)
    caches: dict[str, TeacherCache]
    if args.real_manifest:
        cases, preparation = prepare_real_manifest(
            args.real_manifest,
            deadline_seconds=protocol["resources"]["stage_deadline_seconds"]["corpus"],
            call_seconds=protocol["resources"]["verification_call_seconds"],
        )
        caches = {}
    elif args.prepared:
        cases, caches, preparation = load_preparation(args.prepared)
    else:
        cases = generated_from_protocol(protocol)
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
    label_started = time.monotonic()
    label_rows = []
    for case in selected:
        if case.case_id in caches:
            label_rows.append(
                {
                    "case_id": case.case_id,
                    "status": "cached",
                    "coverage": caches[case.case_id].coverage,
                }
            )
            continue
        remaining = protocol["resources"]["stage_deadline_seconds"]["label"] - (
            time.monotonic() - label_started
        )
        if remaining <= 0:
            label_rows.append({"case_id": case.case_id, "status": "label_stage_deadline"})
            continue
        if len(case.probes) > protocol["teacher"]["max_queries"]:
            label_rows.append({"case_id": case.case_id, "status": "query_cap"})
            continue
        try:
            cache = label_case(
                case,
                max_assignments=protocol["teacher"]["max_assignments"],
                deadline_seconds=min(protocol["teacher"]["case_deadline_seconds"], remaining),
                call_seconds=protocol["resources"]["verification_call_seconds"],
                profile=profile,
                desired_family_weight=protocol["teacher"]["desired_family_weight"],
                false_positive_weight=protocol["teacher"]["false_positive_weight"],
            )
        except ValueError as error:
            label_rows.append(
                {"case_id": case.case_id, "status": "unverified_parent", "detail": str(error)}
            )
            continue
        caches[case.case_id] = cache
        label_rows.append(
            {
                "case_id": case.case_id,
                "status": "complete" if cache.complete else "partial",
                "coverage": cache.coverage,
            }
        )
    preparation.update(
        protocol_hash=canonical_hash(protocol),
        protocol=protocol,
        seed=seed,
        case_limit=args.case_limit,
        selected_counts=counts,
        labelled_cases=len(caches),
        label_rows=label_rows,
        label_seconds=time.monotonic() - label_started,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    save_preparation(args.output / "preparation.json", cases, preparation, caches)
    if args.prepare_only:
        print(json.dumps({key: value for key, value in preparation.items() if key != "protocol"}))
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

    warm = (
        torch.load(args.warm_start, weights_only=True, map_location="cpu")
        if args.warm_start
        else None
    )
    if warm is not None:
        if (
            warm.get("model_schema") != "exact-repair/model/v2"
            or warm.get("training_provenance", {}).get("heldout_supervision") is not False
        ):
            raise ValueError(
                "Adaptation requires a versioned checkpoint with declared held-out restrictions"
            )
        # Architecture is inherited from pretraining; optimizer and development
        # selection remain fresh and depend exclusively on the adaptation split.
        architecture = warm["config"]
        for key, config_key in (
            ("hidden_dim", "hidden_dim"),
            ("layers", "layers"),
            ("heads", "heads"),
            ("dropout", "dropout"),
        ):
            config[config_key] = architecture[key]
    options = dict(
        config,
        encoder=args.encoder
        or (warm["config"]["encoder"] if warm else protocol["graph"]["encoder"]),
        seed=seed,
        profile=profile,
        pairwise=warm["config"]["pairwise"] if warm else args.pairwise,
        arm=arm,
        warm_start_weights=warm["state_dict"] if warm else None,
        warm_start_metadata=warm["metadata"] if warm else None,
    )
    outcome = bounded_call(
        _train_payload,
        labelled_train,
        labelled_dev,
        options,
        timeout=config["deadline_seconds"],
        memory_mb=protocol["resources"]["memory_mb"],
    )
    if outcome.status != "complete":
        report = {
            "schema": "exact-repair/training/v2",
            "status": outcome.status,
            "detail": outcome.detail,
            "protocol_hash": canonical_hash(protocol),
            "scope": "no completed model checkpoint",
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        return 2
    checkpoint_bytes, report = outcome.value
    import io

    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), weights_only=True, map_location="cpu")
    report.update(
        protocol_hash=canonical_hash(protocol), preparation_hash=canonical_hash(preparation)
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    training_provenance = {
        "arm": arm,
        "model_schema": "exact-repair/model/v2",
        "grammar_schema_hash": canonical_hash(protocol["grammar"]),
        "action_schema_hash": canonical_hash(protocol["actions"]),
        "retrieval_schema_hash": canonical_hash(("observable-retrieval/v1", protocol["grammar"])),
        "feature_schema_hash": canonical_hash(
            ("observable-graph/v2", protocol["graph"], checkpoint["config"]["feature_dim"])
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
        "supervision_manifest_hash": preparation.get("manifest_hash"),
    }
    checkpoint.update(
        model_schema="exact-repair/model/v2",
        training_provenance=training_provenance,
        report_hash=canonical_hash(report),
    )
    torch.save(checkpoint, args.output / "model.pt")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
