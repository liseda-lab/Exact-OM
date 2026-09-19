"""Matched research-arm preparation over one captured candidate inventory."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .records import ObjectiveV2, canonical_hash, make_objective
from .study import DEFAULT_ARMS, StudyArmV2, StudyCaseV2, _captured_scores


def research_arms(
    *,
    model_keys: Sequence[str] = ("hgt", "rgcn", "none"),
    profile_keys: Sequence[str] = (),
    omitted_families: Sequence[str] = (),
) -> tuple[StudyArmV2, ...]:
    """Declare all matched selector/model/profile arms; absent artifacts stay unavailable."""
    arms = list(DEFAULT_ARMS)
    for key in ("uniform", "confidence", "symbolic", *model_keys):
        for pairs in (False, True):
            arms.append(
                StudyArmV2(
                    f'{key}_{"pairwise" if pairs else "unary"}', objective_key=key, pairwise=pairs
                )
            )
    arms.extend(
        StudyArmV2(f"profile_{key}", objective_key=key, pairwise=True) for key in profile_keys
    )
    arms.extend(
        StudyArmV2(f"without_{family}", pairwise=True, omitted_actions=(family,))
        for family in omitted_families
    )
    if len({a.arm_id for a in arms}) != len(arms):
        raise ValueError("research arm identifiers overlap")
    return tuple(arms)


def freeze_controls(
    case: StudyCaseV2, *, profile: tuple[tuple[str, float], ...] = ()
) -> StudyCaseV2:
    """Freeze declared uniform/confidence/symbolic controls with identical edit costs.

    Confidence credits only unchanged original mapping directions. Symbolic score
    credits the fraction of retained original axioms. Neither consults a teacher.
    """
    if case.problem is None or case.objective is None:
        return case
    objects = case.problem.objects
    profile = profile or case.objective.profile
    scores = _captured_scores(case)
    variants = dict(case.objective_variants)
    benefit: dict[str, tuple[tuple[float, ...], ...]] = {
        "uniform": tuple(tuple(0.0 for _ in obj.candidates) for obj in objects),
        "symbolic": tuple(
            tuple(
                len(set(c.axioms) & set(obj.original_axioms))
                / max(1, len(set(obj.original_axioms)))
                for c in obj.candidates
            )
            for obj in objects
        ),
    }
    if all(obj.kind != "mapping" or obj.object_id in scores for obj in objects):
        benefit["confidence"] = tuple(
            tuple(
                (scores.get(obj.object_id, 0.0) if obj.kind == "mapping" else 0.0) * retained
                for retained in row
            )
            for obj, row in zip(objects, benefit["symbolic"])
        )
    for key, values in benefit.items():
        objective = make_objective(objects, values, profile=profile, scale=case.objective.scale)
        if key in variants and variants[key] != objective:
            raise ValueError(f"control {key} already has a different frozen objective")
        variants[key] = objective
    return replace(case, objective_variants=tuple(sorted(variants.items())))


def freeze_model_objectives(
    case: StudyCaseV2,
    models: Mapping[str, Any],
    *,
    profile: tuple[tuple[str, float], ...] = (),
    interaction_pairs: Sequence[tuple[int, int]] = (),
    max_graph_nodes: int = 4096,
) -> StudyCaseV2:
    """Score one common inventory using pretrained readouts without regeneration or fitting.

    Use inside a supervised campaign stage; graph schemas must already be included
    in each frozen checkpoint. A schema mismatch is a visible unavailable arm.
    """
    import torch

    from .graph import build_observable_graph
    from .pipeline import model_digest
    from .retrieval import retrieve_vocabulary

    if case.problem is None or case.objective is None:
        raise ValueError("model scoring requires an available frozen case")
    retrieval = retrieve_vocabulary(case.problem)
    problem = case.problem
    symbols = set(retrieval.symbols)
    import pyowl_core as owl

    for obj in problem.objects:
        for candidate in obj.candidates:
            for axiom in (*candidate.axioms, *candidate.active_expressions):
                symbols.update(owl.signature(axiom))
    graph = build_observable_graph(
        problem.objects,
        fixed_axioms=problem.fixed_axioms,
        source_axioms=problem.source_axioms,
        target_axioms=problem.target_axioms,
        evidence=retrieval.graph_evidence(problem.evidence),
        explanations=retrieval.explanations,
        retrieved_symbols=symbols,
        max_nodes=max_graph_nodes,
    )
    variants: dict[str, ObjectiveV2] = dict(case.objective_variants)
    artifacts = dict(case.artifact_hashes)
    for key, model in models.items():
        previous = model.training
        model.eval()
        try:
            with torch.no_grad():
                memory = model.encode(graph)
                unary, pairs = model.score_inventory(
                    problem.objects, memory, interaction_pairs=interaction_pairs
                )
                variants[key] = make_objective(
                    problem.objects,
                    tuple(tuple(float(v) for v in row) for row in unary),
                    profile=profile or case.objective.profile,
                    scale=case.objective.scale,
                    pairs=tuple(
                        (*indices, float(value)) for indices, value in sorted(pairs.items())
                    ),
                )
                artifacts[f"model:{key}"] = model_digest(model)
        finally:
            model.train(previous)
    artifacts["common_scoring_graph"] = canonical_hash(graph)
    return replace(
        case,
        objective_variants=tuple(sorted(variants.items())),
        artifact_hashes=tuple(sorted(artifacts.items())),
    )
