"""Opt-in neural proposal/value handoff to the exact finite-pool repair kernel.

This is the bounded-enumeration circuit arm: grammar-valid candidate bundles are
supplied by the candidate builder. The learned round samples that finite universe,
retains deterministic controls, then freezes all benefit coefficients before solve.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Iterable

import pyowl_core as owl

from .graph import (
    GraphNode,
    ObservableGraph,
    build_observable_graph,
    feature_vector,
    structural_id,
)
from .records import (
    FrozenMapping,
    ObjectiveV2,
    RepairInputV2,
    canonical_hash,
    make_objective,
)


@dataclass(frozen=True)
class FrozenNeuralRound:
    """Reviewable immutable solver input and proposal/model coverage provenance."""

    problem: RepairInputV2
    objective: ObjectiveV2
    graph_hash: str
    model_hash: str
    proposal_reports: tuple[FrozenMapping, ...]


def proposal_distribution(
    model: Any,
    memory: Any,
    obj: Any,
    *,
    mixtures: int = 4,
    profile: tuple[tuple[str, float], ...] = (),
    max_depth: int = 2,
    max_constructors: int = 2,
    constraint_identity: str = "",
) -> Any:
    """Connect shared slot/menu embeddings and precomputed logits to exact SDD WMC."""
    import torch

    from .candidates import deduplicate_candidates
    from .circuit import ConditionedMixture, compile_encoding, encode_candidates

    obj = replace(obj, candidates=deduplicate_candidates(obj.candidates))
    encoding = encode_candidates(
        obj.candidates,
        max_depth=max_depth,
        max_constructors=max_constructors,
        constraint_identity=constraint_identity,
    )
    compiled = compile_encoding(encoding)
    bundles = {candidate.candidate_id: candidate for candidate in obj.candidates}
    retained = {
        owl.structural_hexdigest(axiom): axiom
        for candidate in obj.candidates
        for axiom in candidate.axioms
    }
    context = model.object_context(obj.object_id, memory)
    choices = []
    for field, categories in encoding.fields:
        for category in categories:
            if field == "bundle":
                embedding = model.candidate_embedding(bundles[category], memory)
            elif field.startswith("retained:"):
                embedding = model.encode_structure(retained[field.split(":", 1)[1]], memory)
                embedding = embedding + model._role(f"retained:{category}")
            elif field.endswith(":class") and category != "unused":
                embedding = memory.rows[structural_id(owl.Class(owl.IRI(category)))]
            elif field.endswith(":property") and category != "unused":
                embedding = memory.rows[structural_id(owl.ObjectProperty(owl.IRI(category)))]
            else:
                embedding = model._role(f"choice:{category}")
            role = field.split(":", 1)[0] if field.startswith("retained:") else field
            choices.append(model.argument_head(torch.cat((model._role(role), embedding))))
    profile_vector = (
        context.new_tensor(
            feature_vector(GraphNode("profile", "cost_profile", profile), model.feature_dim)
        )
        if profile
        else None
    )
    weights, logits = model.proposal_logits(
        context, torch.stack(choices), mixtures=mixtures, profile_features=profile_vector
    )
    return ConditionedMixture(compiled, logits, weights)


def model_digest(model: Any) -> str:
    """Hash architecture identity and exact parameter bytes before an evaluated solve."""
    import torch

    digest = hashlib.sha256(canonical_hash((model.metadata, model.config)).encode())
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(canonical_hash((name, str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def freeze_neural_round(
    problem: RepairInputV2,
    model: Any,
    *,
    graph: ObservableGraph | None = None,
    retrieved_symbols: Iterable[Any] = (),
    draws_per_object: int = 32,
    candidate_cap: int = 64,
    mixtures: int = 4,
    seed: int = 13,
    profile: tuple[tuple[str, float], ...] = (),
    interaction_pairs: Iterable[tuple[int, int]] = (),
    max_depth: int = 2,
    max_constructors: int = 2,
    max_circuit_nodes: int = 100000,
) -> FrozenNeuralRound:
    """Freeze constrained samples, complete candidate values and costs for exact solve.

    No model state changes during inference. Compile and graph execution should be
    supervised with ``workers.bounded_call`` by a campaign that sets wall limits.
    An exceeded circuit-node budget raises instead of silently relaxing constraints.
    """
    import torch

    from .candidates import budget_candidates, deduplicate_candidates

    if draws_per_object < 0 or candidate_cap < 1 or max_circuit_nodes < 1:
        raise ValueError("Invalid finite neural round budget")
    problem = replace(
        problem,
        objects=tuple(
            replace(obj, candidates=deduplicate_candidates(obj.candidates))
            for obj in problem.objects
        ),
    )
    graph = graph or build_observable_graph(
        problem.objects,
        fixed_axioms=problem.fixed_axioms,
        source_axioms=problem.source_axioms,
        target_axioms=problem.target_axioms,
        evidence=dict(problem.evidence),
        retrieved_symbols=retrieved_symbols,
    )
    prior_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            memory = model.encode(graph)
            objects, reports = [], []
            for index, obj in enumerate(problem.objects):
                distribution = proposal_distribution(
                    model,
                    memory,
                    obj,
                    mixtures=mixtures,
                    profile=profile,
                    max_depth=max_depth,
                    max_constructors=max_constructors,
                    constraint_identity=canonical_hash((problem.policy, obj.eligible, obj.locked)),
                )
                if distribution.circuit.node_count > max_circuit_nodes:
                    raise ValueError("Compiled circuit exceeds the declared node budget")
                samples = distribution.sample(draws_per_object, seed=seed + index)
                sampled = {sample.candidate_id for sample in samples}
                # Mandatory controls and one member per available action family are
                # selected independently of stochastic proposal availability.
                ranked = {
                    candidate_id: float(distribution.log_probability(bits))
                    for candidate_id, bits in zip(
                        distribution.circuit.encoding.candidate_ids,
                        distribution.circuit.encoding.assignments,
                    )
                }
                all_controls = budget_candidates(obj.candidates, len(obj.candidates), scores=ranked)
                elementary = {"keep", "delete", "retain_subsumption"}
                mandatory_ids = {
                    c.candidate_id for c in all_controls if elementary & set(c.action_tags)
                }
                for family in sorted({tag for c in obj.candidates for tag in c.action_tags}):
                    members = [c for c in obj.candidates if family in c.action_tags]
                    if not any(c.candidate_id in mandatory_ids for c in members):
                        mandatory_ids.add(
                            max(
                                members, key=lambda c: (ranked[c.candidate_id], c.candidate_id)
                            ).candidate_id
                        )
                offered = tuple(
                    c for c in obj.candidates if c.candidate_id in sampled | mandatory_ids
                )
                selected = budget_candidates(offered, candidate_cap, scores=ranked)
                objects.append(replace(obj, candidates=selected))
                reports.append(
                    {
                        "object_id": obj.object_id,
                        "arm": "bounded_enumeration_circuit",
                        "seed": seed + index,
                        "mixtures": mixtures,
                        "max_depth": max_depth,
                        "max_constructors": max_constructors,
                        "samples": tuple(
                            {
                                "candidate_id": sample.candidate_id,
                                "assignment": sample.assignment,
                                "component": sample.component,
                                "log_probability": sample.log_probability,
                            }
                            for sample in samples
                        ),
                        "attempted_draws": len(samples),
                        "unique_draws": len(sampled),
                        "duplicate_draws": len(samples) - len(sampled),
                        "available": len(obj.candidates),
                        "retained": len(selected),
                        "circuit_hash": distribution.circuit.cache_key,
                        "circuit_nodes": distribution.circuit.node_count,
                        "compilation_seconds": distribution.circuit.compilation_seconds,
                    }
                )
            graph_hash, model_hash = canonical_hash(graph), model_digest(model)
            proposal_reports = tuple(FrozenMapping(report) for report in reports)
            frozen_problem = replace(
                problem,
                objects=tuple(objects),
                candidate_coverage="bounded_sampled",
                model_status=f"{model.encoder}:{model_hash}; uncertainty uncalibrated",
                graph_identity=graph_hash,
                proposal_provenance=proposal_reports,
            )
            unary, pairs = model.score_inventory(
                objects, memory, interaction_pairs=interaction_pairs
            )
            objective = make_objective(
                tuple(objects),
                tuple(tuple(float(value) for value in row) for row in unary),
                profile=profile,
                pairs=tuple((*key, float(value)) for key, value in sorted(pairs.items())),
            )
            return FrozenNeuralRound(
                frozen_problem,
                objective,
                graph_hash,
                model_hash,
                proposal_reports,
            )
    finally:
        model.train(prior_training)
